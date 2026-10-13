#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
use half::f16;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};
use toklet::{metal, session, weights};

mod allocations {
    use std::{
        alloc::{GlobalAlloc, Layout, System},
        cell::Cell,
    };
    thread_local! { static COUNT:Cell<Option<usize>>=const {Cell::new(None)}; }
    fn record() {
        COUNT.with(|n| {
            if let Some(value) = n.get() {
                n.set(Some(value + 1));
            }
        });
    }
    pub struct Counting;
    // SAFETY: counting leaves System's allocation/alignment/lifetime contracts
    // unchanged. Const TLS records only the measured test thread and allocates
    // no payload. Native framework allocations are outside this Rust count.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            record();
            unsafe { System.alloc(l) }
        }
        unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
            record();
            unsafe { System.alloc_zeroed(l) }
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
            record();
            unsafe { System.realloc(p, l, n) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            unsafe { System.dealloc(p, l) }
        }
    }
    pub fn start() {
        COUNT.with(|n| n.set(Some(0)));
    }
    pub fn finish() -> usize {
        COUNT.with(|n| n.replace(None).unwrap())
    }
}
#[global_allocator]
static ALLOCATOR: allocations::Counting = allocations::Counting;

#[test]
#[ignore = "explicit Apple GPU Q4 generation/state/Rust-allocation suite; local tiny artifact required"]
fn q4_sessions_generate_continue_cancel_and_allocate_no_steady_state_rust_buffers() {
    use std::sync::atomic::AtomicBool;
    use toklet::generation::{self, Input, Request, StopReason};
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model = metal::Model::load_q4(
        &root.join("artifacts/m5-tiny-source-v1"),
        &root.join("artifacts/m5-tiny-q4-v1"),
        weights::LoadOptions {
            context_limit: 32,
            ..Default::default()
        },
    )
    .unwrap();
    let options = session::Options {
        capacity: 32,
        prefill_chunk_size: 3,
        ..Default::default()
    };
    let mut s = metal::Session::new(&model, options).unwrap();
    let exact = s.memory().admitted_bytes;
    assert!(
        metal::Session::new(
            &model,
            session::Options {
                memory_limit_bytes: exact,
                ..options
            }
        )
        .is_ok()
    );
    assert!(
        metal::Session::new(
            &model,
            session::Options {
                memory_limit_bytes: exact - 1,
                ..options
            }
        )
        .is_err()
    );
    let cancel = AtomicBool::new(false);
    let mut events = Vec::new();
    let first = generation::generate(
        &mut s,
        Request {
            input: Input::Prompt(&[7, 13, 3, 2]),
            max_new_tokens: 4,
            stop_tokens: &[],
        },
        &cancel,
        |e| {
            events.push(e.token_id);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(
        (first.sampled_tokens, s.evaluated_len(), s.pending_len()),
        (4, 7, 1)
    );
    let history = s.history().to_vec();
    let mut replay = metal::Session::new(&model, options).unwrap();
    replay.set_tokens(&history).unwrap();
    replay.evaluate().unwrap();
    s.evaluate().unwrap();
    assert_eq!(s.logits().unwrap(), replay.logits().unwrap());
    let cached = s.history().to_vec();
    s.set_tokens(&cached).unwrap();
    assert_eq!(s.pending_len(), 0);
    let prefix = s.evaluated_len();
    assert!(s.set_tokens(&[64, 2]).is_err());
    assert_eq!(s.evaluated_len(), prefix);
    let next = toklet::sampling::greedy(s.logits().unwrap()).unwrap();
    let stop = generation::generate(
        &mut s,
        Request {
            input: Input::Continue,
            max_new_tokens: 1,
            stop_tokens: &[next],
        },
        &cancel,
        |_| panic!("stop ID is hidden"),
    )
    .unwrap();
    assert_eq!(stop.reason, StopReason::StopToken { token_id: next });
    assert_eq!(s.pending_len(), 1);
    cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    let cancelled = generation::generate(
        &mut s,
        Request {
            input: Input::Continue,
            max_new_tokens: 1,
            stop_tokens: &[],
        },
        &cancel,
        |_| panic!("cancelled"),
    )
    .unwrap();
    assert_eq!(cancelled.reason, StopReason::Cancelled);
    assert_eq!(s.pending_len(), 1);
    s.discard_pending();
    assert_eq!(s.pending_len(), 0);
    assert_eq!(s.evaluated_len(), prefix);
    cancel.store(false, std::sync::atomic::Ordering::Relaxed);
    assert!(
        generation::generate(
            &mut s,
            Request {
                input: Input::Continue,
                max_new_tokens: 1,
                stop_tokens: &[]
            },
            &cancel,
            |_| Err(std::io::Error::other("sink failed"))
        )
        .is_err()
    );
    assert_eq!(s.pending_len(), 1);
    s.discard_pending();
    s.set_tokens(&[1, 2]).unwrap();
    assert_eq!((s.evaluated_len(), s.pending_len()), (0, 2));
    s.evaluate().unwrap();
    replay.reset();
    replay.set_tokens(&[1, 2]).unwrap();
    replay.evaluate().unwrap();
    assert_eq!(s.logits().unwrap(), replay.logits().unwrap());
    for _ in 0..3 {
        s.reset();
        s.set_tokens(&[7]).unwrap();
        s.evaluate().unwrap();
    }
    s.reset();
    s.set_tokens(&[7]).unwrap();
    s.evaluate().unwrap();
    allocations::start();
    for _ in 0..20 {
        s.append(&[13]).unwrap();
        s.evaluate().unwrap();
    }
    let count = allocations::finish();
    assert_eq!(count, 0);
    println!(
        "Q4 session/state suite: 20 warmed steps, {count} Rust allocations, {} admitted bytes",
        s.memory().admitted_bytes
    );
}

#[test]
#[ignore = "explicit complete pinned Qwen Q4 generation on Apple GPU; local packed artifact required"]
fn q4_qwen_runs_complete_resident_generation() {
    use std::sync::atomic::AtomicBool;
    use toklet::generation::{self, Input, Request};
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model = metal::Model::load_q4(
        &root.join("models/Qwen2.5-0.5B-Instruct"),
        &root.join("artifacts/m5-qwen-q4-v1"),
        weights::LoadOptions::default(),
    )
    .unwrap();
    let mut s = metal::Session::new(
        &model,
        session::Options {
            capacity: 32,
            prefill_chunk_size: 4,
            ..Default::default()
        },
    )
    .unwrap();
    let mut ids = Vec::new();
    let report = generation::generate(
        &mut s,
        Request {
            input: Input::Prompt(&[9707, 11, 358, 2776]),
            max_new_tokens: 8,
            stop_tokens: &[],
        },
        &AtomicBool::new(false),
        |e| {
            ids.push(e.token_id);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!((ids.len(), s.evaluated_len(), s.pending_len()), (8, 11, 1));
    assert_eq!(model.weights_bytes(), 476307372);
    println!(
        "{}",
        serde_json::json!({"scope":"execution smoke, no quality/performance acceptance","ids":ids,"report":report,"memory":s.memory(),"last_command_timing":s.timing()})
    );
}

fn manifest(root: &Path) -> Value {
    let data = fs::read(root.join("manifest.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&data)),
        "e4928e4c132c3971798ab67cfd9eec73dc1dfaeae0107797738fc45eb73c8a0c"
    );
    let value: Value = serde_json::from_slice(&data).unwrap();
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(
                fs::read(workspace.join("tests/fixtures/tiny-v1/manifest.json")).unwrap()
            )
        ),
        value["source_manifest_sha256"].as_str().unwrap()
    );
    for (file, sha) in value["oracle_sources"].as_object().unwrap() {
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(fs::read(workspace.join("tools/reference").join(file)).unwrap())
            ),
            sha.as_str().unwrap()
        );
    }
    value
}

fn read(root: &Path, entry: &Value) -> Vec<f32> {
    let data = fs::read(root.join(entry["file"].as_str().unwrap())).unwrap();
    assert_eq!(data.len() as u64, entry["bytes"].as_u64().unwrap());
    assert_eq!(
        format!("{:x}", Sha256::digest(&data)),
        entry["sha256"].as_str().unwrap()
    );
    match entry["dtype"].as_str().unwrap() {
        "f16le" => data
            .chunks_exact(2)
            .map(|v| f16::from_le_bytes([v[0], v[1]]).to_f32())
            .collect(),
        "f32le" => data
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect(),
        _ => panic!("frozen numerical dtype"),
    }
}

#[test]
#[ignore = "explicit Apple GPU Q4 tiny parity; requires artifacts/m5-tiny-{source,q4}-v1"]
fn resident_q4_tiny_matches_independent_oracle_at_all_stages_and_chunks() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = root.join("tests/fixtures/q4-tiny-v1");
    let manifest = manifest(&reference);
    let source = root.join("artifacts/m5-tiny-source-v1");
    let packed = root.join("artifacts/m5-tiny-q4-v1");
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(fs::read(packed.join("manifest.json")).unwrap())
        ),
        manifest["packed_manifest_sha256"].as_str().unwrap()
    );
    let model = metal::Model::load_q4(
        &source,
        &packed,
        weights::LoadOptions {
            context_limit: 32,
            ..Default::default()
        },
    )
    .unwrap();
    for (case, spec) in manifest["cases"].as_object().unwrap() {
        let tokens: Vec<u32> = spec["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        for chunk in [tokens.len(), 3, 1] {
            let expected_names: BTreeSet<_> = manifest["arrays"]
                .as_object()
                .unwrap()
                .keys()
                .filter(|name| name.starts_with(&format!("cases.{case}.")))
                .cloned()
                .collect();
            assert_eq!(expected_names.len(), 35);
            let mut session = metal::Session::new(
                &model,
                session::Options {
                    capacity: 32 - spec["position_offset"].as_u64().unwrap() as usize,
                    prefill_chunk_size: chunk,
                    position_offset: spec["position_offset"].as_u64().unwrap() as usize,
                    ..Default::default()
                },
            )
            .unwrap();
            session.enable_diagnostics(&[0, 1]).unwrap();
            session.set_tokens(&tokens).unwrap();
            let mut compared = 0;
            while session.pending_len() > 0 {
                let start = session.evaluated_len();
                let rows = session.evaluate_next_chunk().unwrap();
                let diagnostics = session.diagnostics().unwrap();
                assert_eq!(diagnostics.len(), 35);
                let mut seen = BTreeSet::new();
                for diagnostic in diagnostics {
                    let stage = match diagnostic.stage {
                        "q" => "q_projection",
                        "k" => "k_projection",
                        "v" => "v_projection",
                        "gate" => "gate_projection",
                        "up" => "up_projection",
                        "silu" => "gate_silu",
                        "down" => "down_projection",
                        other => other,
                    };
                    let name = if let Some(layer) = diagnostic.layer {
                        format!("cases.{case}.layers.{layer}.{stage}")
                    } else {
                        format!("cases.{case}.{stage}")
                    };
                    let entry = &manifest["arrays"][&name];
                    assert!(seen.insert(name.clone()));
                    assert!(expected_names.contains(&name));
                    let shape: Vec<_> = entry["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .collect();
                    assert_eq!(shape[0], tokens.len());
                    assert_eq!(diagnostic.rows, rows);
                    assert_eq!(diagnostic.columns, shape[1..].iter().product::<usize>());
                    assert_eq!(diagnostic.values.len(), rows * diagnostic.columns);
                    let expected = read(&reference, entry);
                    assert_eq!(expected.len(), shape.iter().product::<usize>());
                    let expected =
                        &expected[start * diagnostic.columns..(start + rows) * diagnostic.columns];
                    let atol = entry["gate"]["atol"].as_f64().unwrap();
                    let rtol = entry["gate"]["rtol"].as_f64().unwrap();
                    for (i, (&actual, &want)) in diagnostic.values.iter().zip(expected).enumerate()
                    {
                        assert!(
                            actual.is_finite()
                                && (f64::from(actual) - f64::from(want)).abs()
                                    <= atol + rtol * f64::from(want).abs(),
                            "{name} chunk{chunk} start{start} index{i}: {actual} vs {want}; gate{atol}/{rtol}"
                        );
                    }
                    compared += 1;
                }
                assert_eq!(seen, expected_names);
            }
            assert_eq!(compared, 35 * tokens.len().div_ceil(chunk));
            assert_eq!(session.evaluated_len(), tokens.len());
            assert_eq!(
                toklet::sampling::greedy(session.logits().unwrap()).unwrap(),
                spec["greedy"]
                    .as_array()
                    .unwrap()
                    .last()
                    .unwrap()
                    .as_u64()
                    .unwrap() as u32
            );
            println!(
                "Q4 {case} chunk{chunk}: {compared} stage comparisons, weights {}",
                model.weights_bytes()
            );
        }
    }
}

#[test]
fn frozen_q4_numerical_fixture_is_bounded_and_hash_checked_without_gpu() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/q4-tiny-v1");
    let manifest = manifest(&root);
    assert_eq!(manifest["arrays"].as_object().unwrap().len(), 175);
    assert_eq!(manifest["policy"]["rtol"].as_f64(), Some(1e-3));
    for entry in manifest["arrays"].as_object().unwrap().values() {
        assert!(read(&root, entry).iter().all(|v| v.is_finite()));
    }
}
