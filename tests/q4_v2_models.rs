#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
//! Explicit full-element Q4 v2 real-geometry checks; no held-out quality claims.
use half::f16;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};
use toklet::{metal, session, weights};

fn sha(path: &Path) -> String {
    let mut stream = fs::File::open(path).unwrap();
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let n = stream.read(&mut buffer).unwrap();
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    format!("{:x}", hash.finalize())
}
fn manifest(path: &Path, pin: &str) -> Value {
    let bytes = fs::read(path.join("manifest.json")).unwrap();
    assert!(bytes.len() <= 1024 * 1024);
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), pin);
    serde_json::from_slice(&bytes).unwrap()
}
fn values(root: &Path, item: &Value) -> Vec<f32> {
    let path = root.join(item["file"].as_str().unwrap());
    assert!(fs::metadata(&path).unwrap().len() <= 64 * 1024 * 1024);
    let bytes = fs::read(path).unwrap();
    assert!(bytes.len() <= 64 * 1024 * 1024);
    assert_eq!(bytes.len() as u64, item["bytes"].as_u64().unwrap());
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        item["sha256"].as_str().unwrap()
    );
    match item["dtype"].as_str().unwrap() {
        "f16le" => {
            assert_eq!(bytes.len() % 2, 0);
            bytes
                .chunks_exact(2)
                .map(|x| f16::from_bits(u16::from_le_bytes(x.try_into().unwrap())).to_f32())
                .collect()
        }
        "f32le" => {
            assert_eq!(bytes.len() % 4, 0);
            bytes
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect()
        }
        _ => panic!("frozen dtype"),
    }
}
fn native_peak() -> u64 {
    #[repr(C)]
    struct Usage {
        uuid: [u8; 16],
        fields: [u64; 35],
    }
    #[link(name = "proc")]
    unsafe extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, data: *mut Usage) -> i32;
    }
    let mut usage = Usage {
        uuid: [0; 16],
        fields: [0; 35],
    };
    assert_eq!(std::mem::size_of::<Usage>(), 296);
    // SAFETY: initialized aligned exclusive296-byte SDK rusage_info_v4 stack
    // storage, synchronous own-process call; no alias, callback or retained pointer.
    assert_eq!(
        unsafe { proc_pid_rusage(std::process::id() as i32, 4, &mut usage) },
        0
    );
    let peak = usage.fields[28];
    assert!(peak > 0 && peak <= weights::INITIAL_MEMORY_LIMIT_BYTES as u64);
    peak
}
fn stage(case: &str, d: &metal::Diagnostic) -> String {
    let name = match d.stage {
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
        format!("cases.{case}.layers.{layer}.{name}")
    } else {
        format!("cases.{case}.{name}")
    }
}
fn run(larger: bool) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let (label, name, full_pin, probe_pin, source_pin, weight_bytes, geometry) = if larger {
        (
            "larger",
            "Qwen2.5-1.5B-Instruct",
            "79ce188cbc130dd2c97f64df4a83f27b5e4055ad4b93dd94ee5ddf08c3ae3d1c",
            "960e24d1dacc37ed7decde6c038f1cbae200a1ca05151861b5142e6d981d22bc",
            "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
            1777247428usize,
            (1536, 8960, 12, 2, 128, 28),
        )
    } else {
        (
            "small",
            "Qwen2.5-0.5B-Instruct",
            "6413875965b4bf364a376988f66bf114519e80ef7e33fcd515bfb79cfbaf8527",
            "6275f1dbdeca79210352463341d0e5416c7ffcd3e708c1ff6303367119fd3a29",
            "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
            630251436usize,
            (896, 4864, 14, 2, 64, 24),
        )
    };
    let output = std::env::var_os("TOKLET_Q4_V2_GEOMETRY_REPORT")
        .map(PathBuf::from)
        .expect("exclusive report destination required");
    assert!(!output.exists());
    let full = root.join(format!("artifacts/m5-{label}-q4-v2-oracle-v1"));
    let m = manifest(&full, full_pin);
    let probe = root.join(format!("tests/fixtures/qwen-{label}-q4-v2-v1"));
    let p = manifest(&probe, probe_pin);
    assert_eq!(m["format"], "toklet-streamed-qwen-storage-oracle-v2");
    assert_eq!(m["q4_version"], 2);
    assert_eq!(m["arrays"].as_object().unwrap().len(), 342);
    assert_eq!(p["full_manifest_sha256"], full_pin);
    assert_eq!(m["cases"], p["cases"]);
    for (file, pin) in m["identities"]["oracle_sources"].as_object().unwrap() {
        let path = root
            .join(if file == "native_usage.py" {
                "tools/experiments"
            } else {
                "tools/reference"
            })
            .join(file);
        assert_eq!(sha(&path), pin.as_str().unwrap());
    }
    let source = root.join("models").join(name);
    let packed = root.join(format!("artifacts/m5-{label}-q4-v2"));
    assert_eq!(sha(&source.join("model.safetensors")), source_pin);
    let packed_pin = m["identities"]["packed_manifest_sha256"].as_str().unwrap();
    assert_eq!(sha(&packed.join("manifest.json")), packed_pin);
    let config_pin = m["identities"]["config_sha256"].as_str().unwrap();
    assert_eq!(sha(&source.join("config.json")), config_pin);
    let app = 128 * 1024 * 1024;
    let limit = weights::INITIAL_MEMORY_LIMIT_BYTES - app;
    let model = metal::Model::load_q4(
        &source,
        &packed,
        weights::LoadOptions {
            context_limit: 32,
            memory_limit_bytes: limit,
        },
    )
    .unwrap();
    let c = model.config();
    assert_eq!(
        (
            c.hidden_size,
            c.intermediate_size,
            c.query_heads,
            c.kv_heads,
            c.head_dim,
            c.num_layers
        ),
        geometry
    );
    assert_eq!(model.q4_format(), Some(toklet::q4::Format::V2));
    assert_eq!(model.weights_bytes(), weight_bytes);
    let initial_peak = native_peak();
    let mut metrics = Vec::new();
    let mut failures = Vec::new();
    let mut maximum_admission = 0;
    let mut expected_comparisons = 0;
    for mode in [
        metal::ProjectionMode::Q4Compensated,
        metal::ProjectionMode::Q4CompensatedTiled,
    ] {
        for (case, spec) in m["cases"].as_object().unwrap() {
            let tokens = spec["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect::<Vec<_>>();
            let offset = spec["offset"].as_u64().unwrap() as usize;
            let names = m["arrays"]
                .as_object()
                .unwrap()
                .keys()
                .filter(|name| {
                    name.starts_with(&format!("cases.{case}."))
                        && !name.ends_with(".attention_scores")
                        && !name.ends_with(".attention_probabilities")
                })
                .cloned()
                .collect::<BTreeSet<_>>();
            assert_eq!(names.len(), 51);
            let expected = names
                .iter()
                .map(|name| (name.as_str(), values(&full, &m["arrays"][name])))
                .collect::<BTreeMap<_, _>>();
            let mut chunks = vec![tokens.len(), 8, 3, 1];
            chunks.iter_mut().for_each(|n| *n = (*n).min(tokens.len()));
            chunks.sort_unstable();
            chunks.dedup();
            for chunk in chunks {
                expected_comparisons += 51 * tokens.len().div_ceil(chunk);
                let options = session::Options {
                    capacity: 27,
                    prefill_chunk_size: chunk,
                    position_offset: offset,
                    memory_limit_bytes: limit,
                };
                let mut s = metal::Session::with_execution_modes(
                    &model,
                    options,
                    mode,
                    metal::NormalizationMode::Compensated,
                )
                .unwrap();
                s.enable_diagnostics(&[0, 1, c.num_layers - 1]).unwrap();
                let admitted = s.memory().admitted_bytes.checked_add(app).unwrap();
                assert!(admitted <= weights::INITIAL_MEMORY_LIMIT_BYTES);
                maximum_admission = maximum_admission.max(admitted);
                s.set_tokens(&tokens).unwrap();
                while s.pending_len() > 0 {
                    let start = s.evaluated_len();
                    let rows = s.evaluate_next_chunk().unwrap();
                    assert_eq!(rows, chunk.min(tokens.len() - start));
                    let mut seen = BTreeSet::new();
                    for d in s.diagnostics().unwrap() {
                        let key = stage(case, &d);
                        assert!(names.contains(&key) && seen.insert(key.clone()));
                        let item = &m["arrays"][&key];
                        assert_eq!(item["sha256"], p["arrays"][&key]["full_array_sha256"]);
                        assert_eq!(item["gate"], p["arrays"][&key]["gate"]);
                        let shape = item["shape"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_u64().unwrap() as usize)
                            .collect::<Vec<_>>();
                        assert_eq!(shape[0], tokens.len());
                        assert_eq!(
                            (d.rows, d.columns, d.values.len()),
                            (
                                rows,
                                shape[1..].iter().product(),
                                rows * shape[1..].iter().product::<usize>()
                            )
                        );
                        let whole = &expected[key.as_str()];
                        assert_eq!(whole.len(), shape.iter().product::<usize>());
                        let want = &whole[start * d.columns..(start + rows) * d.columns];
                        let atol = item["gate"]["atol"].as_f64().unwrap();
                        let rtol = item["gate"]["rtol"].as_f64().unwrap();
                        let mut max_fraction = 0f64;
                        for (i, (&actual, &value)) in d.values.iter().zip(want).enumerate() {
                            assert!(actual.is_finite());
                            let fraction = (f64::from(actual) - f64::from(value)).abs()
                                / (atol + rtol * f64::from(value).abs());
                            max_fraction = max_fraction.max(fraction);
                            if fraction > 1. && failures.len() < 32 {
                                failures.push(format!("{mode:?}/{key} chunk{chunk} start{start} index{i}: {actual}/{value} fraction{fraction}"));
                            }
                        }
                        if d.stage == "logits" {
                            for row in 0..rows {
                                let id = toklet::sampling::greedy(
                                    &d.values[row * d.columns..(row + 1) * d.columns],
                                )
                                .unwrap();
                                let wanted = m["argmax"][case]["stored_fp32"][start + row]
                                    .as_u64()
                                    .unwrap();
                                if u64::from(id) != wanted {
                                    failures.push(format!(
                                        "greedy {mode:?}/{case} chunk{chunk} row{} {id}/{wanted}",
                                        start + row
                                    ));
                                }
                            }
                        }
                        metrics.push(json!({"mode":mode,"array":key,"chunk":chunk,"start":start,"rows":rows,"max_gate_fraction":max_fraction}));
                    }
                    assert_eq!(seen, names);
                }
                assert_eq!((s.evaluated_len(), s.pending_len()), (tokens.len(), 0));
                native_peak();
                println!(
                    "V2 {label}/{mode:?}/{case} chunk{chunk} full-array/state/memory check completed"
                );
            }
        }
    }
    assert_eq!(metrics.len(), expected_comparisons);
    assert_eq!(sha(&source.join("model.safetensors")), source_pin);
    assert_eq!(sha(&source.join("config.json")), config_pin);
    assert_eq!(sha(&packed.join("manifest.json")), packed_pin);
    assert_eq!(manifest(&full, full_pin), m);
    assert_eq!(manifest(&probe, probe_pin), p);
    for (file, pin) in m["identities"]["oracle_sources"].as_object().unwrap() {
        let path = root
            .join(if file == "native_usage.py" {
                "tools/experiments"
            } else {
                "tools/reference"
            })
            .join(file);
        assert_eq!(sha(&path), pin.as_str().unwrap());
    }
    let final_peak = native_peak();
    let report = json!({"format":"toklet-q4-v2-real-geometry-v1","passed":failures.is_empty(),"scope":"Independent real GPU geometry/full-element/chunk/offset/greedy/memory only; no heldout quality or performance","model":label,"device":model.device_name(),"weights_bytes":model.weights_bytes(),"q4_format":"V2","compensated_simd_widths":model.q4_compensated_execution_widths(),"source_sha256":source_pin,"config_sha256":config_pin,"packed_manifest_sha256":packed_pin,"full_reference_sha256":full_pin,"probe_reference_sha256":probe_pin,"initial_native_lifetime_charge":initial_peak,"final_native_lifetime_charge":final_peak,"maximum_admitted_numerical_plus_validation_bytes":maximum_admission,"validation_allowance_bytes":app,"full_stage_arrays_compared":metrics.len(),"metrics":metrics,"failures":failures});
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    serde_json::to_writer_pretty(&mut file, &report).unwrap();
    writeln!(file).unwrap();
    file.flush().unwrap();
    assert!(failures.is_empty(), "{failures:?}");
}
#[test]
#[ignore = "Explicit small Q4 v2 full GPU graph/memory qualification; pinned local artifacts required"]
fn small_q4_v2_full_graph_and_memory() {
    run(false);
}
#[test]
#[ignore = "Explicit larger Q4 v2 full GPU graph/memory qualification; pinned local artifacts required"]
fn larger_q4_v2_full_graph_and_memory() {
    run(true);
}
