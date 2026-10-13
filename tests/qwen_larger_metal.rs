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
    use std::io::{Read, Write};
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
    fn run(kind: &str) {
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
        let output = r.join(format!(
            "benchmarks/results/2026-10-13-larger-{kind}-gpu-parity-v3.json"
        ));
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
                let mut s = metal::Session::with_projection_mode(
                    &model,
                    session::Options {
                        capacity: 27,
                        prefill_chunk_size: chunk,
                        position_offset: offset,
                        ..Default::default()
                    },
                    if kind == "q4" {
                        metal::ProjectionMode::Q4Compensated
                    } else {
                        metal::ProjectionMode::Original
                    },
                )
                .unwrap();
                let budget = s.memory().admitted_bytes;
                assert!(budget + 128 * 1024 * 1024 <= weights::INITIAL_MEMORY_LIMIT_BYTES);
                if profile_events == 0 && case == "future_changed" && chunk == 1 {
                    if kind == "q4" {
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
                    if kind == "q4" && case == "future_changed" && chunk == 1 {
                        let profile = s.profile().unwrap();
                        assert_eq!(profile.events.len(), 482);
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
        let mut report = json!({"format":"toklet-larger-metal-parity-v1","kind":kind,"passed":failures.is_empty(),"scope":"Short larger GPU geometry/storage numerical parity only; no lossy quality or performance acceptance","device":model.device_name(),"weights_bytes":model.weights_bytes(),"load_timing":{"load_seconds":model.load_timing.load_seconds,"conversion_seconds":model.load_timing.conversion_seconds,"compilation_seconds":model.load_timing.compilation_seconds},"q4_execution_widths":model.q4_execution_widths(),"compensated_execution_width":model.q4_compensated_execution_width(),"projection_mode":if kind=="q4" {"Q4Compensated"}else{"Original"},"full_arrays_compared":compared,"initial_native_lifetime_peak_bytes":initial_peak,"final_native_lifetime_peak_bytes":peak,"native_scope":"Own-process v4 lifetime charge includes fixture/hash/diagnostics; no concurrent model worker; complements allocation admission","profile_events":profile_events,"metrics":metrics,"failures":failures});
        report["identities"] = json!({"source_sha256":source_sha,"config_sha256":config_sha,"packed_manifest_sha256":packed_sha});
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
        run("half");
    }
    #[test]
    #[ignore = "Explicit preselected larger Q4 reconstructed oracle/profile on Apple GPU"]
    fn larger_q4_matches_frozen_reconstructed_oracle() {
        run("q4");
    }
}
