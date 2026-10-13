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
#[ignore = "explicit Apple GPU required-position versus all-row FP16/Q4 logits"]
fn ordinary_prefill_last_logits_match_all_required_diagnostic_rows() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // Freeze the existing session envelope before native execution. Changing
    // the MPS row count may change its internal reduction; no bit equality claim.
    for real in [false, true] {
        let source = root.join(if real {
            "models/Qwen2.5-0.5B-Instruct"
        } else {
            "artifacts/m5-tiny-source-v1"
        });
        let packed = root.join(if real {
            "artifacts/m5-qwen-q4-v1"
        } else {
            "artifacts/m5-tiny-q4-v1"
        });
        let tokens: &[u32] = if real {
            &[9707, 11, 358, 2776]
        } else {
            &[1, 7, 13, 3, 47, 11, 2]
        };
        for quantized in [false, true] {
            let options = weights::LoadOptions {
                context_limit: 32,
                ..Default::default()
            };
            let model = if quantized {
                metal::Model::load_q4(&source, &packed, options)
            } else {
                metal::Model::load(&source, options)
            }
            .unwrap();
            for offset in [0, 5] {
                for chunk in [4, 3, 1] {
                    let options = session::Options {
                        capacity: 27,
                        prefill_chunk_size: chunk,
                        position_offset: offset,
                        ..Default::default()
                    };
                    let mut ordinary = metal::Session::new(&model, options).unwrap();
                    let mut diagnostic = metal::Session::new(&model, options).unwrap();
                    diagnostic.enable_diagnostics(&[]).unwrap();
                    for part in tokens.chunks(chunk) {
                        ordinary.append(part).unwrap();
                        diagnostic.append(part).unwrap();
                        let rows = ordinary.evaluate_next_chunk().unwrap();
                        assert_eq!(rows, diagnostic.evaluate_next_chunk().unwrap());
                        assert_eq!(rows, part.len());
                        let snapshots = diagnostic.diagnostics().unwrap();
                        let all = snapshots.iter().find(|s| s.stage == "logits").unwrap();
                        assert_eq!(all.rows, rows);
                        assert_eq!(all.values.len(), rows * model.config().vocab_size);
                        let last = &all.values[(rows - 1) * all.columns..];
                        assert_eq!(last, diagnostic.logits().unwrap());
                        for (&actual, &want) in ordinary.logits().unwrap().iter().zip(last) {
                            assert!(
                                (actual - want).abs() <= 1e-4 + 1e-3 * want.abs(),
                                "last-row mismatch real{real} q4{quantized} offset{offset} chunk{chunk}: {actual}/{want}"
                            );
                        }
                        assert_eq!(
                            toklet::sampling::greedy(ordinary.logits().unwrap()).unwrap(),
                            toklet::sampling::greedy(last).unwrap()
                        );
                    }
                    assert_eq!(ordinary.evaluate_next_chunk().unwrap(), 0);
                    assert_eq!(diagnostic.evaluate_next_chunk().unwrap(), 0);
                    assert_eq!(ordinary.evaluated_len(), tokens.len());
                    println!(
                        "required/all-row distribution parity real{real} q4{quantized} offset{offset} chunk{chunk} passed"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "explicit Apple GPU FP16/Q4 longest-prefix and repeated-cleanup suite"]
fn metal_prefix_edits_shortening_and_recovery_match_reset_replay() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = root.join("artifacts/m5-tiny-source-v1");
    let packed = root.join("artifacts/m5-tiny-q4-v1");
    for quantized in [false, true] {
        let load = weights::LoadOptions {
            context_limit: 32,
            ..Default::default()
        };
        let model = if quantized {
            metal::Model::load_q4(&source, &packed, load)
        } else {
            metal::Model::load(&source, load)
        }
        .unwrap();
        for offset in [0, 5] {
            let options = session::Options {
                capacity: 27,
                prefill_chunk_size: 3,
                position_offset: offset,
                ..Default::default()
            };
            let mut reused = metal::Session::new(&model, options).unwrap();
            let mut reference = metal::Session::new(&model, options).unwrap();
            let base = [1, 7, 13, 3, 47, 11];
            let same = |left: &metal::Session<'_>, right: &metal::Session<'_>| {
                let a = left.logits().unwrap();
                let b = right.logits().unwrap();
                assert_eq!(a.len(), b.len());
                for (&actual, &want) in a.iter().zip(b) {
                    assert!(
                        (actual - want).abs() <= 1e-4 + 1e-3 * want.abs(),
                        "{actual}/{want}"
                    );
                }
                assert_eq!(
                    toklet::sampling::greedy(a).unwrap(),
                    toklet::sampling::greedy(b).unwrap()
                );
            };
            for (tokens, keep) in [
                (&base[..], 6),
                (&[1, 7, 13, 3, 47, 12][..], 5),
                (&[1, 8, 13, 3][..], 1),
                (&[2, 7, 13][..], 0),
                (&base[..4], 3),
                (&base[..1], 0),
            ] {
                reused.reset();
                reused.set_tokens(&base).unwrap();
                reused.evaluate().unwrap();
                reused.set_tokens(tokens).unwrap();
                assert_eq!(reused.evaluated_len(), keep);
                reused.evaluate().unwrap();
                reference.reset();
                reference.set_tokens(tokens).unwrap();
                reference.evaluate().unwrap();
                same(&reused, &reference);
            }
            reused.reset();
            reused.set_tokens(&base).unwrap();
            reused.evaluate().unwrap();
            reused.append(&[47]).unwrap();
            reused.set_tokens(&base).unwrap();
            assert_eq!((reused.evaluated_len(), reused.pending_len()), (6, 0));
            reused.set_tokens(&[1, 7, 13, 3, 47, 12]).unwrap();
            assert!(reused.logits().is_none());
            reused.discard_pending();
            reused.discard_pending();
            reused.append(&[12, 13]).unwrap();
            reused.discard_pending();
            reused.discard_pending();
            assert_eq!(reused.history(), &base[..5]);
            assert_eq!((reused.evaluated_len(), reused.pending_len()), (4, 1));
            assert!(reused.logits().is_none());
            let same_history = reused.history().to_vec();
            reused.set_tokens(&same_history).unwrap();
            reused.discard_pending();
            assert_eq!(reused.history(), &base[..5]);
            reused.evaluate().unwrap();
            reference.reset();
            reference.set_tokens(&base[..5]).unwrap();
            reference.evaluate().unwrap();
            same(&reused, &reference);
            reused.set_tokens(&[1, 7, 13, 3, 48]).unwrap();
            reused.set_tokens(&base[..4]).unwrap();
            reused.discard_pending();
            reused.discard_pending();
            assert_eq!(reused.history(), &base[..4]);
            assert_eq!((reused.evaluated_len(), reused.pending_len()), (3, 1));
            reused.evaluate().unwrap();
            reference.reset();
            reference.set_tokens(&base[..4]).unwrap();
            reference.evaluate().unwrap();
            same(&reused, &reference);
            reused.set_tokens(&[]).unwrap();
            assert!(reused.logits().is_none());
            assert_eq!(reused.evaluated_len(), 0);
            println!(
                "longest-prefix/repeated-recovery suite quantized{quantized} offset{offset} passed"
            );
        }
    }
}

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
        if root.ends_with("q4-tiny-prefill-v1") {
            "af489fadf32b1a2a46a04098fabc315152cf63f507ee2887bc6a4f13de2545c9"
        } else {
            "e4928e4c132c3971798ab67cfd9eec73dc1dfaeae0107797738fc45eb73c8a0c"
        }
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
    check_tiny_oracle("q4-tiny-v1", metal::ProjectionMode::Original);
}
#[test]
#[ignore = "explicit Apple GPU eight-row Q4 tile independent stage/tail/offset qualification"]
fn tiled_q4_prefill_matches_independent_oracle_at_all_stages_and_chunks() {
    check_tiny_oracle("q4-tiny-prefill-v1", metal::ProjectionMode::Original);
    check_tiny_oracle("q4-tiny-prefill-v1", metal::ProjectionMode::Q4Tiled);
}

#[test]
#[ignore = "explicit Apple GPU tile refinement; expanded independent oracle remains a separate gate"]
fn tiled_q4_prefill_preserves_every_original_stage_bit() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = manifest(&root.join("tests/fixtures/q4-tiny-prefill-v1"));
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(fs::read(root.join("artifacts/m5-tiny-q4-v1/manifest.json")).unwrap())
        ),
        reference["packed_manifest_sha256"].as_str().unwrap()
    );
    let model = metal::Model::load_q4(
        &root.join("artifacts/m5-tiny-source-v1"),
        &root.join("artifacts/m5-tiny-q4-v1"),
        weights::LoadOptions {
            context_limit: 32,
            ..Default::default()
        },
    )
    .unwrap();
    // Bit preservation is qualified on the measured device's equal queried
    // widths. This is not a full-graph independent numerical acceptance.
    let widths = model.q4_execution_widths();
    assert_eq!(widths[0], widths[1]);
    let mut compared = 0;
    for (case, spec) in reference["cases"].as_object().unwrap() {
        let expected_names: BTreeSet<_> = reference["arrays"]
            .as_object()
            .unwrap()
            .keys()
            .filter(|name| name.starts_with(&format!("cases.{case}.")))
            .cloned()
            .collect();
        assert_eq!(expected_names.len(), 35);
        let tokens: Vec<u32> = spec["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let offset = spec["position_offset"].as_u64().unwrap() as usize;
        for chunk in [tokens.len(), 8, 3, 1] {
            let options = session::Options {
                capacity: 32 - offset,
                prefill_chunk_size: chunk,
                position_offset: offset,
                ..Default::default()
            };
            let mut original = metal::Session::new(&model, options).unwrap();
            let mut tiled = metal::Session::with_projection_mode(
                &model,
                options,
                metal::ProjectionMode::Q4Tiled,
            )
            .unwrap();
            for session in [&mut original, &mut tiled] {
                session.enable_diagnostics(&[0, 1]).unwrap();
                session.set_tokens(&tokens).unwrap();
            }
            while original.pending_len() > 0 {
                let start = original.evaluated_len();
                let rows = original.evaluate_next_chunk().unwrap();
                assert_eq!(rows, tiled.evaluate_next_chunk().unwrap());
                assert_eq!(rows, chunk.min(tokens.len() - start));
                let a = original.diagnostics().unwrap();
                let b = tiled.diagnostics().unwrap();
                assert_eq!(a.len(), 35);
                assert_eq!(b.len(), 35);
                let mut seen = BTreeSet::new();
                for (a, b) in a.iter().zip(&b) {
                    assert_eq!(
                        (a.layer, a.stage, a.rows, a.columns),
                        (b.layer, b.stage, b.rows, b.columns)
                    );
                    assert_eq!(a.values.len(), b.values.len());
                    let stage = match a.stage {
                        "q" => "q_projection",
                        "k" => "k_projection",
                        "v" => "v_projection",
                        "gate" => "gate_projection",
                        "up" => "up_projection",
                        "silu" => "gate_silu",
                        "down" => "down_projection",
                        other => other,
                    };
                    let name = if let Some(layer) = a.layer {
                        format!("cases.{case}.layers.{layer}.{stage}")
                    } else {
                        format!("cases.{case}.{stage}")
                    };
                    assert!(expected_names.contains(&name));
                    assert!(seen.insert(name.clone()));
                    let shape: Vec<_> = reference["arrays"][name]["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .collect();
                    assert_eq!(shape[0], tokens.len());
                    for actual in [a, b] {
                        assert_eq!(actual.rows, rows);
                        assert_eq!(actual.columns, shape[1..].iter().product::<usize>());
                        assert_eq!(actual.values.len(), rows * actual.columns);
                    }
                    for (i, (a, b)) in a.values.iter().zip(&b.values).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{case} chunk{chunk} start{start} index{i}"
                        );
                    }
                    compared += 1;
                }
                assert_eq!(seen, expected_names);
            }
            assert_eq!(tiled.pending_len(), 0);
            assert_eq!(original.evaluated_len(), tokens.len());
            assert_eq!(tiled.evaluated_len(), tokens.len());
            assert_eq!(
                toklet::sampling::greedy(original.logits().unwrap()).unwrap(),
                toklet::sampling::greedy(tiled.logits().unwrap()).unwrap()
            );
        }
    }
    assert_eq!(compared, 4550);
    println!(
        "Experimental tile refinement: {compared} complete stage arrays preserve bits; queried widths {widths:?}. Expanded independent decoder gate remains unqualified."
    );
}
fn check_tiny_oracle(fixture: &str, mode: metal::ProjectionMode) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = root.join("tests/fixtures").join(fixture);
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
    println!(
        "Q4 SIMD widths {:?}, mode {:?}",
        model.q4_execution_widths(),
        mode
    );
    for (case, spec) in manifest["cases"].as_object().unwrap() {
        let tokens: Vec<u32> = spec["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let chunks = if mode == metal::ProjectionMode::Original {
            vec![tokens.len(), 3, 1]
        } else {
            vec![tokens.len(), 8, 3, 1]
        };
        for chunk in chunks {
            let expected_names: BTreeSet<_> = manifest["arrays"]
                .as_object()
                .unwrap()
                .keys()
                .filter(|name| name.starts_with(&format!("cases.{case}.")))
                .cloned()
                .collect();
            assert_eq!(expected_names.len(), 35);
            let mut session = metal::Session::with_projection_mode(
                &model,
                session::Options {
                    capacity: 32 - spec["position_offset"].as_u64().unwrap() as usize,
                    prefill_chunk_size: chunk,
                    position_offset: spec["position_offset"].as_u64().unwrap() as usize,
                    ..Default::default()
                },
                mode,
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
fn frozen_q4_prefill_fixture_is_bounded_and_hash_checked_without_gpu() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/q4-tiny-prefill-v1");
    let manifest = manifest(&root);
    assert_eq!(manifest["arrays"].as_object().unwrap().len(), 210);
    assert_eq!(manifest["policy"]["rtol"].as_f64(), Some(1e-3));
    assert_eq!(manifest["cases"].as_object().unwrap().len(), 6);
    for entry in manifest["arrays"].as_object().unwrap().values() {
        assert!(read(&root, entry).iter().all(|v| v.is_finite()));
        assert_eq!(entry["gate"]["rtol"].as_f64(), Some(1e-3));
        assert_eq!(
            entry["gate"]["atol"].as_f64().unwrap(),
            (2. * entry["independent_precision_envelope"].as_f64().unwrap()).max(1e-4)
        );
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
