use half::f16;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
fn manifest(path: &Path, expected: &str) -> Value {
    let bytes = fs::read(path.join("manifest.json")).unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), expected);
    serde_json::from_slice(&bytes).unwrap()
}
fn array(path: &Path, item: &Value) -> Vec<f32> {
    let data = fs::read(path.join(item["file"].as_str().unwrap())).unwrap();
    assert_eq!(data.len() as u64, item["bytes"].as_u64().unwrap());
    assert_eq!(
        format!("{:x}", Sha256::digest(&data)),
        item["sha256"].as_str().unwrap()
    );
    match item.get("dtype").and_then(Value::as_str).unwrap_or("f32le") {
        "f16le" => {
            assert!(data.len().is_multiple_of(2));
            data.chunks_exact(2)
                .map(|v| f16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).to_f32())
                .collect()
        }
        "f32le" => {
            assert!(data.len().is_multiple_of(4));
            data.chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect()
        }
        _ => panic!("Frozen numerical dtype"),
    }
}
fn probe(kind: &str) -> Value {
    manifest(
        &root().join(format!("tests/fixtures/qwen-larger-{kind}-v1")),
        if kind == "half" {
            "6ba8fa4c41f98f13730662d85d06b9fb03b223b60a081e8f80662e417c20a71f"
        } else {
            "83140123f91296412002aa5432b120401477be2213a243c48ad3ee8a43dd1508"
        },
    )
}

#[test]
fn larger_storage_probes_bind_complete_arrays_shapes_gates_and_argmax() {
    for kind in ["half", "q4"] {
        let m = probe(kind);
        assert_eq!(m["layers"], json!([0, 1, 27]));
        assert_eq!(
            m["model"]["weights"]["sha256"],
            "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
        );
        assert_eq!(m["arrays"].as_object().unwrap().len(), 228);
        assert_eq!(m["cases"].as_object().unwrap().len(), 4);
        assert_eq!(m["policy"]["atol_floor"], 1e-4);
        assert_eq!(m["policy"]["envelope_multiplier"], 2.0);
        assert_eq!(
            m["policy"]["rtol"],
            if kind == "half" { 1e-4 } else { 1e-3 }
        );
        let path = root().join(format!("tests/fixtures/qwen-larger-{kind}-v1"));
        for (name, item) in m["arrays"].as_object().unwrap() {
            let data = array(&path, item);
            let elements: usize = item["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .product();
            let bytes = if item["dtype"] == "f16le" { 2 } else { 4 };
            assert_eq!(
                item["full_array_bytes"].as_u64().unwrap() as usize,
                elements * bytes
            );
            let indices: Vec<_> = item["indices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(data.len(), indices.len());
            assert!(data.iter().all(|v| v.is_finite()));
            assert!(indices.iter().all(|&v| v < elements));
            assert!(indices.windows(2).all(|v| v[0] < v[1]));
            let gate = &item["gate"];
            let (pooled, prefix) = if kind == "half" {
                (
                    gate["pooled_fp16_vs_source_max_abs_error"]
                        .as_f64()
                        .unwrap(),
                    gate["equivalent_prefix_max_abs_error"].as_f64().unwrap(),
                )
            } else {
                (
                    gate["pooled_fp32_fp64_max_abs"].as_f64().unwrap(),
                    gate["equivalent_prefix_max_abs"].as_f64().unwrap(),
                )
            };
            assert_eq!(
                gate["atol"].as_f64().unwrap(),
                1e-4f64.max(2.0 * pooled).max(2.0 * prefix)
            );
            assert_eq!(gate["rtol"], m["policy"]["rtol"]);
            if name.ends_with(".logits") {
                let case = name.split('.').nth(1).unwrap();
                let columns = item["shape"][1].as_u64().unwrap() as usize;
                let families = m["argmax"][case].as_object().unwrap();
                assert_eq!(
                    families.keys().cloned().collect::<BTreeSet<_>>(),
                    BTreeSet::from([
                        "source_fp32".to_owned(),
                        "stored_fp32".to_owned(),
                        "stored_fp64".to_owned()
                    ])
                );
                for ids in families.values() {
                    for (position, id) in ids.as_array().unwrap().iter().enumerate() {
                        assert!(
                            indices.contains(&(position * columns + id.as_u64().unwrap() as usize))
                        );
                    }
                }
            }
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod native {
    use super::*;
    use std::io::{Read, Seek, SeekFrom, Write};
    use toklet::{metal, session, weights};

    fn file_sha(path: &Path) -> String {
        let mut file = fs::File::open(path).unwrap();
        let mut digest = Sha256::new();
        let mut bytes = [0u8; 65536];
        loop {
            let n = file.read(&mut bytes).unwrap();
            if n == 0 {
                break;
            }
            digest.update(&bytes[..n]);
        }
        format!("{:x}", digest.finalize())
    }
    fn physical_peak() -> u64 {
        #[repr(C)]
        struct Usage {
            uuid: [u8; 16],
            fields: [u64; 35],
        }
        #[link(name = "proc")]
        unsafe extern "C" {
            fn proc_pid_rusage(pid: i32, flavor: i32, usage: *mut Usage) -> i32;
        }
        assert_eq!(std::mem::size_of::<Usage>(), 296);
        let mut usage = Usage {
            uuid: [0; 16],
            fields: [0; 35],
        };
        // SAFETY: initialized aligned exclusive296-byte SDK rusage_info_v4
        // storage; synchronous own-process query writes only this stack object.
        assert_eq!(
            unsafe { proc_pid_rusage(std::process::id() as i32, 4, &mut usage) },
            0
        );
        let peak = usage.fields[28];
        assert!(peak > 0 && peak <= weights::INITIAL_MEMORY_LIMIT_BYTES as u64);
        peak
    }
    fn name(case: &str, d: &metal::Diagnostic) -> String {
        let stage = match d.stage {
            "q" => "q_projection",
            "k" => "k_projection",
            "v" => "v_projection",
            "gate" => "gate_projection",
            "up" => "up_projection",
            "silu" => "gate_silu",
            "down" => "down_projection",
            other => other,
        };
        if let Some(layer) = d.layer {
            format!("cases.{case}.layers.{layer}.{stage}")
        } else {
            format!("cases.{case}.{stage}")
        }
    }
    fn run(
        kind: &str,
        projection: metal::ProjectionMode,
        normalization: metal::NormalizationMode,
        version: &str,
    ) {
        let r = root();
        let probes = probe(kind);
        let full = r.join(format!("artifacts/m5-larger-{kind}-oracle-v3"));
        let m = manifest(&full, probes["full_manifest_sha256"].as_str().unwrap());
        let source_path = r.join("artifacts/m5-larger-oracle-v1");
        let source = manifest(
            &source_path,
            "676cf5392c0dcb8ceff8d098fae21378b0dfdca92b69f9de7aac48bc23e234fe",
        );
        assert_eq!(m["cases"], source["cases"]);
        assert_eq!(m["layers"], source["layers"]);
        for (file, sha) in m["identities"]["oracle_sources"].as_object().unwrap() {
            let path = r
                .join(if file == "native_usage.py" {
                    "tools/experiments"
                } else {
                    "tools/reference"
                })
                .join(file);
            assert_eq!(
                format!("{:x}", Sha256::digest(fs::read(path).unwrap())),
                sha.as_str().unwrap()
            );
        }
        // Retained reports are immutable. A caller may choose a fresh evidence
        // path; ordinary reruns use an exclusive, uniquely named local artifact.
        let output = std::env::var_os("TOKLET_LARGER_PARITY_REPORT")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                r.join(format!(
                    "artifacts/larger-{kind}-parity-{version}-{}-{}.json",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                ))
            });
        assert!(!output.exists(), "Native evidence already exists");
        let model_path = r.join("models/Qwen2.5-1.5B-Instruct");
        let packed_path = r.join("artifacts/m5-larger-q4-v1");
        let source_sha = file_sha(&model_path.join("model.safetensors"));
        assert_eq!(
            source_sha,
            m["model"]["weights"]["sha256"].as_str().unwrap()
        );
        let config_sha = file_sha(&model_path.join("config.json"));
        assert_eq!(
            config_sha,
            m["identities"]["config_sha256"].as_str().unwrap()
        );
        let packed_sha = if kind == "q4" {
            let value = file_sha(&packed_path.join("manifest.json"));
            assert_eq!(
                value,
                m["identities"]["packed_manifest_sha256"].as_str().unwrap()
            );
            Some(value)
        } else {
            None
        };

        let options = weights::LoadOptions {
            context_limit: 32,
            ..Default::default()
        };
        let model = if kind == "q4" {
            metal::Model::load_q4(
                &r.join("models/Qwen2.5-1.5B-Instruct"),
                &r.join("artifacts/m5-larger-q4-v1"),
                options,
            )
        } else {
            metal::Model::load(&r.join("models/Qwen2.5-1.5B-Instruct"), options)
        }
        .unwrap();
        assert_eq!(
            (
                model.config().hidden_size,
                model.config().intermediate_size,
                model.config().query_heads,
                model.config().kv_heads,
                model.config().head_dim,
                model.config().num_layers
            ),
            (1536, 8960, 12, 2, 128, 28)
        );
        let initial_peak = physical_peak();
        let mut metrics = Vec::new();
        let mut failures = Vec::new();
        let mut compared = 0;
        let mut profile_events = 0;
        for (case, spec) in m["cases"].as_object().unwrap() {
            let tokens: Vec<u32> = spec["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let offset = spec["offset"].as_u64().unwrap() as usize;
            let expected_names: BTreeSet<_> = m["arrays"]
                .as_object()
                .unwrap()
                .keys()
                .filter(|k| {
                    k.starts_with(&format!("cases.{case}."))
                        && !k.ends_with(".attention_scores")
                        && !k.ends_with(".attention_probabilities")
                })
                .cloned()
                .collect();
            assert_eq!(expected_names.len(), 51);
            let mut chunks = vec![tokens.len(), 3, 1];
            chunks.iter_mut().for_each(|n| *n = (*n).min(tokens.len()));
            chunks.sort_unstable();
            chunks.dedup();
            for chunk in chunks {
                let mut s = metal::Session::with_execution_modes(
                    &model,
                    session::Options {
                        capacity: 27,
                        prefill_chunk_size: chunk,
                        position_offset: offset,
                        ..Default::default()
                    },
                    projection,
                    normalization,
                )
                .unwrap();
                let budget = s.memory().admitted_bytes;
                assert!(budget + 128 * 1024 * 1024 <= weights::INITIAL_MEMORY_LIMIT_BYTES);
                if profile_events == 0 && case == "future_changed" && chunk == 1 {
                    if kind == "q4" || projection == metal::ProjectionMode::HalfCompensated {
                        s.enable_profile().unwrap();
                    } else {
                        assert!(matches!(
                            s.enable_profile(),
                            Err(metal::Error::Unsupported(
                                "model profiling exceeds admitted counter capacity"
                            ))
                        ));
                        assert_eq!(s.memory().admitted_bytes, budget);
                    }
                }
                s.enable_diagnostics(&[0, 1, 27]).unwrap();
                assert!(
                    s.memory().admitted_bytes + 128 * 1024 * 1024
                        <= weights::INITIAL_MEMORY_LIMIT_BYTES
                );
                s.set_tokens(&tokens).unwrap();
                while s.pending_len() > 0 {
                    let start = s.evaluated_len();
                    let rows = s.evaluate_next_chunk().unwrap();
                    assert_eq!(rows, chunk.min(tokens.len() - start));
                    let mut seen = BTreeSet::new();
                    for d in s.diagnostics().unwrap() {
                        let key = name(case, &d);
                        assert!(seen.insert(key.clone()) && expected_names.contains(&key));
                        let frozen = &m["arrays"][&key];
                        assert_eq!(
                            frozen["sha256"],
                            probes["arrays"][&key]["full_array_sha256"]
                        );
                        assert_eq!(frozen["gate"], probes["arrays"][&key]["gate"]);
                        let shape: Vec<usize> = frozen["shape"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_u64().unwrap() as usize)
                            .collect();
                        assert_eq!(shape[0], tokens.len());
                        assert_eq!(
                            (d.rows, d.columns, d.values.len()),
                            (
                                rows,
                                shape[1..].iter().product::<usize>(),
                                rows * shape[1..].iter().product::<usize>()
                            )
                        );
                        let values = if kind == "half" {
                            array(&source_path, &source["arrays"][&key])
                        } else {
                            array(&full, frozen)
                        };
                        assert_eq!(values.len(), shape.iter().product::<usize>());
                        let expected = &values[start * d.columns..(start + rows) * d.columns];
                        let atol = frozen["gate"]["atol"].as_f64().unwrap();
                        let rtol = frozen["gate"]["rtol"].as_f64().unwrap();
                        let mut delta = 0f64;
                        let mut fraction = 0f64;
                        for (i, (&actual, &want)) in d.values.iter().zip(expected).enumerate() {
                            assert!(actual.is_finite());
                            let error = (f64::from(actual) - f64::from(want)).abs();
                            let ratio = error / (atol + rtol * f64::from(want).abs());
                            delta = delta.max(error);
                            fraction = fraction.max(ratio);
                            if ratio > 1.0 && failures.len() < 32 {
                                failures.push(format!("{key} chunk{chunk} start{start} index{i}: {actual}/{want} fraction{ratio}"));
                            }
                        }
                        if d.stage == "logits" {
                            let ids = &m["argmax"][case][if kind == "half" {
                                "source_fp32"
                            } else {
                                "stored_fp32"
                            }];
                            for row in 0..rows {
                                let id = toklet::sampling::greedy(
                                    &d.values[row * d.columns..(row + 1) * d.columns],
                                )
                                .unwrap();
                                if u64::from(id) != ids[start + row].as_u64().unwrap() {
                                    failures.push(format!(
                                        "greedy {case} chunk{chunk} row{}: {id}/{}",
                                        start + row,
                                        ids[start + row]
                                    ));
                                }
                            }
                        }
                        metrics.push(json!({"array":key,"chunk":chunk,"start":start,"rows":rows,"columns":d.columns,"max_abs":delta,"max_gate_fraction":fraction}));
                        compared += 1;
                    }
                    assert_eq!(seen, expected_names);
                    if (kind == "q4" || projection == metal::ProjectionMode::HalfCompensated)
                        && case == "future_changed"
                        && chunk == 1
                    {
                        let profile = s.profile().unwrap();
                        assert_eq!(
                            profile.events.len(),
                            if projection == metal::ProjectionMode::HalfCompensated {
                                480
                            } else {
                                482
                            }
                        );
                        profile_events = profile.events.len();
                    }
                }
                assert_eq!((s.evaluated_len(), s.pending_len()), (tokens.len(), 0));
                println!(
                    "Larger {kind} {case} chunk{chunk} completed; admitted {}",
                    s.memory().admitted_bytes
                );
            }
        }
        assert_eq!(compared, 969);
        assert_eq!(file_sha(&model_path.join("model.safetensors")), source_sha);
        assert_eq!(file_sha(&model_path.join("config.json")), config_sha);
        if let Some(sha) = &packed_sha {
            assert_eq!(&file_sha(&packed_path.join("manifest.json")), sha);
        }

        let peak = physical_peak();
        let mut report = json!({"format":"toklet-larger-metal-parity-v1","kind":kind,"passed":failures.is_empty(),"scope":"Short larger GPU geometry/storage numerical parity only; no lossy quality or performance acceptance","device":model.device_name(),"weights_bytes":model.weights_bytes(),"load_timing":{"load_seconds":model.load_timing.load_seconds,"conversion_seconds":model.load_timing.conversion_seconds,"compilation_seconds":model.load_timing.compilation_seconds},"q4_execution_widths":model.q4_execution_widths(),"half_compensated_execution_widths":model.half_compensated_execution_widths(),"compensated_execution_width":model.q4_compensated_execution_width(),"projection_mode":projection,"full_arrays_compared":compared,"initial_native_lifetime_peak_bytes":initial_peak,"final_native_lifetime_peak_bytes":peak,"native_scope":"Own-process v4 lifetime charge includes fixture/hash/diagnostics; no concurrent model worker; complements allocation admission","profile_events":profile_events,"metrics":metrics,"failures":failures});
        report["identities"] = json!({"source_sha256":source_sha,"config_sha256":config_sha,"packed_manifest_sha256":packed_sha});
        report["normalization_mode"] = json!(normalization);
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)
            .unwrap()
            .write_all(&serde_json::to_vec_pretty(&report).unwrap())
            .unwrap();
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
    #[test]
    #[ignore = "Explicit preselected larger Half geometry/parity on Apple GPU"]
    fn larger_half_matches_source_with_frozen_storage_envelope() {
        run(
            "half",
            metal::ProjectionMode::Original,
            metal::NormalizationMode::Original,
            "v3",
        );
    }
    #[test]
    #[ignore = "Explicit preselected larger Q4 reconstructed oracle/profile on Apple GPU"]
    fn larger_q4_matches_frozen_reconstructed_oracle() {
        run(
            "q4",
            metal::ProjectionMode::Q4Compensated,
            metal::NormalizationMode::Original,
            "v3",
        );
    }

    #[test]
    #[ignore = "Explicit larger Q4 compensated projection and RMS qualification under unchanged gates"]
    fn larger_q4_compensated_rms_matches_frozen_reconstructed_oracle() {
        run(
            "q4",
            metal::ProjectionMode::Q4Compensated,
            metal::NormalizationMode::Compensated,
            "v4",
        );
    }

    #[test]
    #[ignore = "Explicit larger compensated Half projection under unchanged independent storage gates"]
    fn larger_half_compensated_matches_frozen_storage_envelope() {
        run(
            "half",
            metal::ProjectionMode::HalfCompensated,
            metal::NormalizationMode::Original,
            "compensated-v1",
        );
    }

    #[test]
    #[ignore = "Bounded identical-input larger RMS/rotary diagnostic capture, no numerical qualification"]
    fn capture_larger_first_layer_for_numerical_localization() {
        let r = root();
        let probes = probe("q4");
        let model_path = r.join("models/Qwen2.5-1.5B-Instruct");
        let source = model_path.join("model.safetensors");
        let source_sha = file_sha(&source);
        assert_eq!(source_sha, probes["model"]["weights"]["sha256"]);
        let config_sha = file_sha(&model_path.join("config.json"));
        assert_eq!(
            config_sha,
            "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670"
        );
        // Read only the pinned 1536-element norm vector. Source identity and
        // exact header/shape/range checks precede this bounded diagnostic copy.
        let mut file = fs::File::open(&source).unwrap();
        assert_eq!(file.metadata().unwrap().len(), 3_087_467_144);
        let mut length = [0; 8];
        file.read_exact(&mut length).unwrap();
        assert_eq!(u64::from_le_bytes(length), 38528);
        let mut header = vec![0; 38528];
        file.read_exact(&mut header).unwrap();
        let header: Value = serde_json::from_slice(&header).unwrap();
        let norm = &header["model.layers.0.input_layernorm.weight"];
        assert_eq!(norm["dtype"], "BF16");
        assert_eq!(norm["shape"], json!([1536]));
        let start = norm["data_offsets"][0].as_u64().unwrap();
        let end = norm["data_offsets"][1].as_u64().unwrap();
        assert_eq!(end.checked_sub(start), Some(3072));
        assert!(end <= 3_087_428_608);
        file.seek(SeekFrom::Start(38536 + start)).unwrap();
        let mut payload = vec![0; 3072];
        file.read_exact(&mut payload).unwrap();
        let norm_bits: Vec<u16> = payload
            .chunks_exact(2)
            .map(|v| {
                f16::from_f32(f32::from_bits(
                    u32::from(u16::from_le_bytes(v.try_into().unwrap())) << 16,
                ))
                .to_bits()
            })
            .collect();
        assert!(norm_bits.iter().all(|&v| f16::from_bits(v).is_finite()));
        let packed = r.join("artifacts/m5-larger-q4-v1");
        let packed_sha = file_sha(&packed.join("manifest.json"));
        assert_eq!(packed_sha, probes["identities"]["packed_manifest_sha256"]);
        let model = metal::Model::load_q4(
            &model_path,
            &packed,
            weights::LoadOptions {
                context_limit: 32,
                ..Default::default()
            },
        )
        .unwrap();
        let mut cases = Vec::new();
        for (name, spec) in probes["cases"].as_object().unwrap() {
            let tokens: Vec<u32> = spec["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let mut s = metal::Session::with_projection_mode(
                &model,
                session::Options {
                    capacity: 27,
                    prefill_chunk_size: tokens.len(),
                    position_offset: spec["offset"].as_u64().unwrap() as usize,
                    ..Default::default()
                },
                metal::ProjectionMode::Q4Compensated,
            )
            .unwrap();
            s.enable_diagnostics(&[0]).unwrap();
            assert!(
                s.memory().admitted_bytes + 128 * 1024 * 1024
                    <= weights::INITIAL_MEMORY_LIMIT_BYTES
            );
            s.set_tokens(&tokens).unwrap();
            assert_eq!(s.evaluate_next_chunk().unwrap(), tokens.len());
            let snapshots: Vec<_> = s.diagnostics().unwrap().into_iter().filter(|v| {
                (v.layer.is_none() && v.stage == "embedding") || (v.layer == Some(0) && ["input_norm", "q", "k", "q_rotary", "k_rotary"].contains(&v.stage))
            }).map(|v| json!({"layer":v.layer,"stage":v.stage,"rows":v.rows,"columns":v.columns,"values_bits":v.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>() })).collect();
            assert_eq!(snapshots.len(), 6);
            cases.push(json!({"case":name,"spec":spec,"snapshots":snapshots}));
        }
        assert_eq!(file_sha(&source), source_sha);
        assert_eq!(file_sha(&model_path.join("config.json")), config_sha);
        assert_eq!(file_sha(&packed.join("manifest.json")), packed_sha);
        let report = json!({"scope":"Identical-input RMS/rotary diagnostic only; no qualification, gate or quality changes", "source_sha256":source_sha,"config_sha256":config_sha,"packed_manifest_sha256":packed_sha,"norm_bf16_payload_sha256":format!("{:x}",Sha256::digest(payload)),"norm_half_bits":norm_bits,"cases":cases,"native_lifetime_peak_bytes":physical_peak()});
        let bytes = serde_json::to_vec(&report).unwrap();
        assert!(bytes.len() <= 4 * 1024 * 1024);
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(r.join("artifacts/m5-larger-first-layer-capture-v1.json"))
            .unwrap()
            .write_all(&bytes)
            .unwrap();
    }
}
