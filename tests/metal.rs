//! Explicit Apple GPU suite; ordinary `cargo test` does not execute hardware work.
#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod support;
use safetensors::{Dtype, tensor::TensorView};
use serde_json::{Value, json};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    collections::BTreeMap,
    fs,
    path::PathBuf,
};
struct CountAllocations;
thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn count_allocation() {
    let _ = TRACK.try_with(|track| {
        if track.get() {
            ALLOCATIONS.with(|count| count.set(count.get() + 1));
        }
    });
}

// SAFETY: This test-only shim delegates all allocation/deallocation and layout
// contracts unchanged to System. TLS counting touches no allocated memory and
// records only the measured test thread, excluding concurrent harness activity.
unsafe impl GlobalAlloc for CountAllocations {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count_allocation();
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountAllocations = CountAllocations;

use support::Fixture;
use toklet::{metal, session, weights};

struct Artifact(PathBuf);
impl Artifact {
    fn tiny() -> Self {
        Self::configured(32, false)
    }
    fn configured(max_positions: usize, zero: bool) -> Self {
        let f = Fixture::tiny();
        let path = std::env::temp_dir().join(format!("toklet-metal-{}", std::process::id()));
        fs::create_dir(&path).unwrap();
        let config = json!({"model_type":"qwen2", "architectures":["Qwen2ForCausalLM"], "hidden_act":"silu", "hidden_size":32, "intermediate_size":64, "num_hidden_layers":2, "num_attention_heads":4, "num_key_value_heads":2, "vocab_size":64, "max_position_embeddings":max_positions, "rms_norm_eps":1e-6, "rope_theta":1e6, "tie_word_embeddings":true, "torch_dtype":"float32"});
        fs::write(
            path.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let mut data = BTreeMap::new();
        for (name, _) in f.manifest["arrays"].as_object().unwrap() {
            if !name.starts_with("weights.") {
                continue;
            }
            let suffix = name.strip_prefix("weights.").unwrap();
            let target =
                if suffix == "embed_tokens" || suffix == "norm" || suffix.ends_with("layernorm") {
                    format!("model.{suffix}.weight")
                } else {
                    let parts: Vec<_> = suffix.split('.').collect();
                    let block = if ["q_proj", "k_proj", "v_proj", "o_proj"].contains(&parts[2]) {
                        "self_attn"
                    } else {
                        "mlp"
                    };
                    format!(
                        "model.layers.{}.{}.{}.{}",
                        parts[1], block, parts[2], parts[3]
                    )
                };
            let mut bytes = f.bytes(name);
            if zero && !name.ends_with("layernorm") && name != "weights.norm" {
                bytes.fill(0);
            }
            data.insert(target, (f.shape(name), bytes));
        }
        let views: Vec<_> = data
            .iter()
            .map(|(name, (shape, bytes))| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        safetensors::serialize_to_file(views, None, &path.join("model.safetensors")).unwrap();
        Self(path)
    }
}
impl Drop for Artifact {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn options(chunk: usize, offset: usize) -> session::Options {
    session::Options {
        capacity: 32 - offset,
        prefill_chunk_size: chunk,
        position_offset: offset,
        ..Default::default()
    }
}
fn qualify(actual: &[f32], source: &[f32], gate: &Value) {
    let atol = gate["atol"].as_f64().unwrap();
    let rtol = gate["rtol"].as_f64().unwrap();
    assert_eq!(actual.len(), source.len());
    for (i, (&a, &b)) in actual.iter().zip(source).enumerate() {
        assert!(
            a.is_finite()
                && (f64::from(a) - f64::from(b)).abs() <= atol + rtol * f64::from(b).abs(),
            "frozen source gate at {i}: {a} vs {b}"
        );
    }
}
#[test]
#[ignore = "requires Apple Silicon Metal; run explicitly with --ignored --test-threads=1"]
fn resident_tiny_decoder_and_session_contract() {
    let artifact = Artifact::tiny();
    let model = metal::Model::load(
        &artifact.0,
        weights::LoadOptions {
            context_limit: 32,
            ..Default::default()
        },
    )
    .unwrap();
    let capture_path = std::env::var_os("TOKLET_TEST_CAPTURE").map(PathBuf::from);
    let capture = capture_path
        .as_ref()
        .map(|path| model.capture(path).unwrap());
    let cpu = weights::load_qwen(
        &artifact.0,
        weights::LoadOptions {
            context_limit: 32,
            ..Default::default()
        },
    )
    .unwrap();
    let source = Fixture::tiny();
    let half: Value = serde_json::from_slice(
        &fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fp16-v1/manifest.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for case in ["capacity", "future_changed", "offset", "prompt", "single"] {
        let tokens = source.u32(&format!("cases.{case}.tokens"));
        let offset = half["cases"][case]["position_offset"].as_u64().unwrap() as usize;
        let reference = source.f32(&format!("cases.{case}.logits"));
        let gate = &half["arrays"][format!("cases.{case}.logits")]["gate_against_source_fp32"];
        for chunk in [tokens.len(), 3, 1] {
            let mut s = metal::Session::new(&model, options(chunk, offset)).unwrap();
            if std::env::var_os("TOKLET_TEST_PROFILE").is_some() {
                s.enable_profile().unwrap();
            }
            s.enable_diagnostics(&[0, 1]).unwrap();
            s.set_tokens(&tokens).unwrap();
            s.evaluate().unwrap();
            qualify(
                s.logits().unwrap(),
                &reference[reference.len() - 64..],
                gate,
            );
            if std::env::var_os("TOKLET_TEST_PROFILE").is_some() {
                assert_eq!(
                    s.profile().unwrap().events.len(),
                    38 * model.config().num_layers + 6
                );
            }
            for d in s.diagnostics().unwrap() {
                let name = diagnostic_name(case, &d);
                let expected = source.f32(&name);
                let start = (tokens.len() - d.rows) * d.columns;
                qualify(
                    &d.values,
                    &expected[start..],
                    &half["arrays"][&name]["gate_against_source_fp32"],
                );
            }
            assert_eq!(s.evaluated_len(), tokens.len());
            assert_eq!(s.timing().unwrap().command_count, 1);
            s.set_tokens(&tokens).unwrap();
            assert_eq!(s.evaluate_next_chunk().unwrap(), 0);
        }
    }
    let mut gpu = metal::Session::new(&model, options(3, 0)).unwrap();
    let mut scalar = session::Session::new(&cpu.model, options(3, 0)).unwrap();
    for tokens in [
        vec![1, 7, 13, 3],
        vec![1, 7, 13, 3, 47],
        vec![1, 7, 13, 3, 11],
        vec![1, 7, 13, 3],
        vec![1, 7],
        vec![],
        vec![11],
    ] {
        gpu.set_tokens(&tokens).unwrap();
        scalar.set_tokens(&tokens).unwrap();
        assert_eq!(gpu.evaluated_len(), scalar.evaluated_len());
        assert_eq!(gpu.logits().is_some(), scalar.logits().is_some());
        // Keep sampled pending tokens through the next set_tokens call.
        if tokens.len() != 5 {
            gpu.evaluate().unwrap();
            scalar.evaluate().unwrap();
        }
        assert_eq!(gpu.evaluated_len(), scalar.evaluated_len());
    }
    generation_contract(&model);
    gpu.reset();
    gpu.append(&[1, 7, 13, 3]).unwrap();
    gpu.evaluate().unwrap();
    let memory = gpu.memory();
    for step in 0..4 {
        let reference = source.f32(&format!("greedy.step_{step}.logits"));
        qualify(
            gpu.logits().unwrap(),
            &reference,
            &half["arrays"][format!("greedy.step_{step}.logits")]["gate_against_source_fp32"],
        );
        assert_eq!(toklet::sampling::greedy(gpu.logits().unwrap()).unwrap(), 47);
        gpu.append(&[47]).unwrap();
        assert!(gpu.logits().is_none());
        gpu.evaluate().unwrap();
        assert_eq!(gpu.memory(), memory);
    }
    drop(capture);
    gpu.reset();
    gpu.append(&[1]).unwrap();
    gpu.evaluate().unwrap();
    ALLOCATIONS.with(|count| count.set(0));
    TRACK.with(|track| track.set(true));
    for _ in 0..20 {
        gpu.append(&[47]).unwrap();
        gpu.evaluate().unwrap();
    }
    TRACK.with(|track| track.set(false));
    assert_eq!(
        ALLOCATIONS.with(Cell::get),
        0,
        "steady-state numerical Rust allocations"
    );
}

fn diagnostic_name(case: &str, d: &metal::Diagnostic) -> String {
    let stage = match d.stage {
        "q" => "q_projection",
        "k" => "k_projection",
        "v" => "v_projection",
        "gate" => "gate_projection",
        "silu" => "gate_silu",
        "up" => "up_projection",
        "down" => "down_projection",
        other => other,
    };
    match d.layer {
        Some(layer) => format!("cases.{case}.layers.{layer}.{stage}"),
        None => format!("cases.{case}.{stage}"),
    }
}
#[test]
#[ignore = "Explicit compensated Half full tiny independent gates, offsets, publication and warmed allocation checks"]
fn compensated_half_tiny_independent_graph_and_publication() {
    use std::collections::BTreeSet;
    let artifact = Artifact::tiny();
    let model = metal::Model::load(
        &artifact.0,
        weights::LoadOptions {
            context_limit: 32,
            ..Default::default()
        },
    )
    .unwrap();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let half = manifest(
        &root.join("tests/fixtures/fp16-v1"),
        "4a416b21f52ecc89c6bbafd4c948b173cd9709c3f5b1a60cd6ce037a63964e6b",
    );
    let source = Fixture::tiny();
    let mode = metal::ProjectionMode::HalfCompensated;
    let mut compared = 0;
    for (case, spec) in half["cases"].as_object().unwrap() {
        let tokens = source.u32(&format!("cases.{case}.tokens"));
        let offset = spec["position_offset"].as_u64().unwrap() as usize;
        let mut expected_names = BTreeSet::from([
            format!("cases.{case}.embedding"),
            format!("cases.{case}.final_norm"),
            format!("cases.{case}.logits"),
        ]);
        for layer in 0..2 {
            for stage in [
                "input_norm",
                "q_projection",
                "k_projection",
                "v_projection",
                "q_rotary",
                "k_rotary",
                "attention_context",
                "attention_projection",
                "attention_residual",
                "post_attention_norm",
                "gate_projection",
                "gate_silu",
                "up_projection",
                "gated",
                "down_projection",
                "output",
            ] {
                let name = format!("cases.{case}.layers.{layer}.{stage}");
                assert!(half["arrays"].get(&name).is_some());
                assert!(expected_names.insert(name));
            }
        }
        assert_eq!(expected_names.len(), 35);
        let mut chunks = vec![tokens.len(), 8, 3, 1];
        chunks.sort_unstable();
        chunks.dedup();
        for chunk in chunks {
            let mut diagnostic =
                metal::Session::with_projection_mode(&model, options(chunk, offset), mode).unwrap();
            let mut ordinary =
                metal::Session::with_projection_mode(&model, options(chunk, offset), mode).unwrap();
            diagnostic.enable_diagnostics(&[0, 1]).unwrap();
            diagnostic.enable_profile().unwrap();
            for part in tokens.chunks(chunk) {
                let start = diagnostic.evaluated_len();
                diagnostic.append(part).unwrap();
                ordinary.append(part).unwrap();
                let rows = diagnostic.evaluate_next_chunk().unwrap();
                assert_eq!(ordinary.evaluate_next_chunk().unwrap(), rows);
                assert_eq!(
                    diagnostic
                        .logits()
                        .unwrap()
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>(),
                    ordinary
                        .logits()
                        .unwrap()
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>(),
                    "ordinary final-row offset publication {case}/{chunk}/{start}"
                );
                let diagnostics = diagnostic.diagnostics().unwrap();
                assert_eq!(diagnostics.len(), 35);
                let mut seen = BTreeSet::new();
                for d in diagnostics {
                    let name = diagnostic_name(case, &d);
                    assert!(expected_names.contains(&name) && seen.insert(name.clone()));
                    let expected = source.f32(&name);
                    assert_eq!(d.rows, rows);
                    assert_eq!(expected.len(), tokens.len() * d.columns);
                    qualify(
                        &d.values,
                        &expected[start * d.columns..(start + rows) * d.columns],
                        &half["arrays"][&name]["gate_against_source_fp32"],
                    );
                    if d.stage == "logits" {
                        for row in 0..rows {
                            assert_eq!(
                                toklet::sampling::greedy(
                                    &d.values[row * d.columns..(row + 1) * d.columns]
                                )
                                .unwrap(),
                                toklet::sampling::greedy(
                                    &expected
                                        [(start + row) * d.columns..(start + row + 1) * d.columns]
                                )
                                .unwrap(),
                                "frozen source argmax {case}/{chunk}/{start}/{row}"
                            );
                        }
                    }
                    compared += 1;
                }
                assert_eq!(seen, expected_names);
                assert_eq!(diagnostic.profile().unwrap().events.len(), 38);
            }
            assert_eq!(
                (diagnostic.evaluated_len(), ordinary.evaluated_len()),
                (tokens.len(), tokens.len())
            );
            // Exact admission boundaries apply to the explicit mode as well.
            let budget = ordinary.memory().admitted_bytes;
            assert!(
                metal::Session::with_projection_mode(
                    &model,
                    session::Options {
                        memory_limit_bytes: budget,
                        ..options(chunk, offset)
                    },
                    mode
                )
                .is_ok()
            );
            assert!(
                metal::Session::with_projection_mode(
                    &model,
                    session::Options {
                        memory_limit_bytes: budget - 1,
                        ..options(chunk, offset)
                    },
                    mode
                )
                .is_err()
            );
        }
    }
    let mut s = metal::Session::with_projection_mode(&model, options(3, 0), mode).unwrap();
    let mut fresh = metal::Session::with_projection_mode(&model, options(3, 0), mode).unwrap();
    for tokens in [
        vec![1, 7, 13, 3],
        vec![1, 7, 13, 3, 47],
        vec![1, 7, 13, 3, 11],
        vec![1, 7],
        vec![11],
    ] {
        s.set_tokens(&tokens).unwrap();
        s.evaluate().unwrap();
        fresh.reset();
        fresh.set_tokens(&tokens).unwrap();
        fresh.evaluate().unwrap();
        assert_eq!(s.logits().unwrap(), fresh.logits().unwrap());
        assert_eq!((s.evaluated_len(), s.pending_len()), (tokens.len(), 0));
    }
    for _ in 0..3 {
        s.reset();
        s.append(&[1]).unwrap();
        s.evaluate().unwrap();
    }
    let budget = s.memory();
    TRACK.with(|v| v.set(true));
    ALLOCATIONS.with(|v| v.set(0));
    for _ in 0..20 {
        s.append(&[7]).unwrap();
        s.evaluate().unwrap();
    }
    TRACK.with(|v| v.set(false));
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
    assert_eq!(s.memory(), budget);
    println!(
        "Compensated Half: {compared} complete independent tiny stage arrays; ordinary/diagnostic logits bit exact; exact budgets, append/edit/reset and20 allocation-free warmed steps pass"
    );
}
fn manifest(root: &std::path::Path, digest: &str) -> Value {
    use sha2::{Digest, Sha256};
    let bytes = fs::read(root.join("manifest.json")).unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), digest);
    serde_json::from_slice(&bytes).unwrap()
}
fn payload(root: &std::path::Path, m: &Value, name: &str) -> Vec<f32> {
    use sha2::{Digest, Sha256};
    let item = &m["arrays"][name];
    assert!(item["file"].as_str().unwrap().ends_with(".f32le"));
    assert_eq!(
        item["elements"].as_u64().unwrap() as usize * 4,
        item["bytes"].as_u64().unwrap() as usize
    );
    let bytes = fs::read(root.join(item["file"].as_str().unwrap())).unwrap();
    assert_eq!(bytes.len(), item["bytes"].as_u64().unwrap() as usize);
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        item["sha256"].as_str().unwrap()
    );
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
#[test]
#[ignore = "requires installed Qwen and frozen full references, Apple GPU; run explicitly"]
fn resident_qwen_full_forward() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let half_root = root.join("artifacts/qwen-fp16-full-v2");
    let source_root = root.join("artifacts/qwen-forward-v2");
    let half = manifest(
        &half_root,
        "49206b6f5bed4f605671bf45474f25c81cb0c48ee3c33a9f741c513eb9ecbb72",
    );
    let source = manifest(
        &source_root,
        "bdd368d21dfae574197e3b64423b75d4f1b2b37c083c6172c429e36bb4dd69c7",
    );
    let model = metal::Model::load(
        &root.join("models/Qwen2.5-0.5B-Instruct"),
        weights::LoadOptions::default(),
    )
    .unwrap();
    println!(
        "Qwen device={} weights_bytes={} load={:?}",
        model.device_name(),
        model.weights_bytes(),
        model.load_timing
    );
    for case in ["future_changed", "offset", "short", "single"] {
        let tokens: Vec<u32> = half["cases"][case]["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_u64().unwrap() as u32)
            .collect();
        let offset = half["cases"][case]["offset"].as_u64().unwrap() as usize;
        for chunk in [tokens.len(), 3, 1] {
            let mut s = metal::Session::new(
                &model,
                session::Options {
                    capacity: 32,
                    prefill_chunk_size: chunk,
                    position_offset: offset,
                    ..Default::default()
                },
            )
            .unwrap();
            if std::env::var_os("TOKLET_TEST_PROFILE").is_some() {
                s.enable_profile().unwrap();
            }
            s.enable_diagnostics(&[0, 1, 23]).unwrap();
            s.append(&tokens).unwrap();
            s.evaluate().unwrap();
            let mut arrays = 0;
            if std::env::var_os("TOKLET_TEST_PROFILE").is_some() {
                assert_eq!(
                    s.profile().unwrap().events.len(),
                    38 * model.config().num_layers + 6
                );
            }
            for d in s.diagnostics().unwrap() {
                let name = diagnostic_name(case, &d);
                let expected = payload(&source_root, &source, &name);
                let start = (tokens.len() - d.rows) * d.columns;
                qualify(
                    &d.values,
                    &expected[start..],
                    &half["arrays"][&name]["gate_against_source_fp32"],
                );
                arrays += 1;
            }
            let token = toklet::sampling::greedy(s.logits().unwrap()).unwrap();
            assert_eq!(
                u64::from(token),
                source["argmax"][case][tokens.len() - 1].as_u64().unwrap()
            );
            println!(
                "Qwen case={case} chunk={chunk} arrays={arrays} token={token} timing={:?} admitted={}",
                s.timing(),
                s.memory().admitted_bytes
            );
        }
    }
}

fn generation_contract(model: &metal::Model) {
    let s = metal::Session::new(model, options(2, 0)).unwrap();
    generation_contract_session(
        model,
        s,
        metal::ProjectionMode::Original,
        metal::AttentionMode::Reference,
    );
}
fn generation_contract_session(
    model: &metal::Model,
    mut s: metal::Session<'_>,
    projection: metal::ProjectionMode,
    attention: metal::AttentionMode,
) {
    let make = |options| {
        metal::Session::with_attention_mode(
            model,
            options,
            projection,
            metal::NormalizationMode::Original,
            metal::ReadbackMode::FullLogits,
            attention,
        )
    };
    use std::sync::atomic::{AtomicBool, Ordering};
    use toklet::generation::{self, Input, Request, StopReason};
    let cancel = AtomicBool::new(true);
    let request = || Request {
        input: Input::Prompt(&[1, 7, 13, 3]),
        max_new_tokens: 4,
        stop_tokens: &[],
    };
    let r = generation::generate(&mut s, request(), &cancel, |_| panic!()).unwrap();
    assert_eq!(r.reason, StopReason::Cancelled);
    assert!(s.history().is_empty());
    cancel.store(false, Ordering::Relaxed);
    let r = generation::generate(&mut s, request(), &cancel, |e| {
        assert_eq!(e.token_id, 47);
        cancel.store(true, Ordering::Relaxed);
        Ok(())
    })
    .unwrap();
    assert_eq!(r.reason, StopReason::Cancelled);
    assert_eq!((s.evaluated_len(), s.pending_len()), (4, 1));
    cancel.store(false, Ordering::Relaxed);
    let r = generation::generate(
        &mut s,
        Request {
            input: Input::Continue,
            max_new_tokens: 1,
            stop_tokens: &[],
        },
        &cancel,
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(
        (
            r.reused_tokens,
            r.prefill_tokens,
            s.evaluated_len(),
            s.pending_len()
        ),
        (4, 1, 5, 1)
    );
    let r = generation::generate(
        &mut s,
        Request {
            input: Input::Reset(&[1, 7, 13, 3]),
            max_new_tokens: 4,
            stop_tokens: &[47],
        },
        &cancel,
        |_| panic!(),
    )
    .unwrap();
    assert_eq!(r.reason, StopReason::StopToken { token_id: 47 });
    assert_eq!(
        (r.emitted_tokens, s.evaluated_len(), s.pending_len()),
        (0, 4, 1)
    );
    let error = generation::generate(
        &mut s,
        Request {
            input: Input::Reset(&[1, 7, 13, 3]),
            max_new_tokens: 4,
            stop_tokens: &[],
        },
        &cancel,
        |_| Err(std::io::Error::other("test output failure")),
    )
    .unwrap_err();
    assert!(matches!(error, generation::Error::Output(_)));
    assert_eq!((s.evaluated_len(), s.pending_len()), (4, 1));
    s.discard_pending();
    assert!(s.logits().is_some());
    let logits = s.logits().unwrap().to_vec();
    for request in [
        Request {
            input: Input::Reset(&[]),
            max_new_tokens: 1,
            stop_tokens: &[],
        },
        Request {
            input: Input::Reset(&[64]),
            max_new_tokens: 1,
            stop_tokens: &[],
        },
        Request {
            input: Input::Continue,
            max_new_tokens: usize::MAX,
            stop_tokens: &[],
        },
        Request {
            input: Input::Continue,
            max_new_tokens: 1,
            stop_tokens: &[64],
        },
    ] {
        assert!(generation::generate(&mut s, request, &cancel, |_| panic!()).is_err());
        assert_eq!(s.history(), [1, 7, 13, 3]);
        assert_eq!(s.logits().unwrap(), logits);
    }
    let m = s.memory();
    assert_eq!(
        m.admitted_bytes,
        m.weights_bytes
            + m.kv_bytes
            + m.scratch_bytes
            + m.history_bytes
            + m.logits_bytes
            + m.metadata_bytes
            + m.metadata_allowance_bytes
    );
    assert!(
        make(session::Options {
            memory_limit_bytes: m.admitted_bytes - 1,
            ..options(2, 0)
        })
        .is_err()
    );
    assert!(
        make(session::Options {
            memory_limit_bytes: m.admitted_bytes,
            ..options(2, 0)
        })
        .is_ok()
    );
}
#[test]
#[ignore = "Apple GPU full-capacity structural check, exact zero-weight oracle"]
fn resident_capacity_boundary() {
    let artifact = Artifact::configured(2048, true);
    let model = metal::Model::load(&artifact.0, weights::LoadOptions::default()).unwrap();
    for (capacity, offset) in [(2048, 0), (2047, 1)] {
        for chunk in [128, 256, 512, 1024] {
            let mut s = metal::Session::new(
                &model,
                session::Options {
                    capacity,
                    prefill_chunk_size: chunk,
                    position_offset: offset,
                    ..Default::default()
                },
            )
            .unwrap();
            let tokens = vec![1; capacity];
            s.append(&tokens).unwrap();
            s.evaluate().unwrap();
            assert_eq!(s.evaluated_len(), capacity);
            assert!(s.logits().unwrap().iter().all(|x| x.to_bits() == 0));
            assert!(s.append(&[1]).is_err());
            s.set_tokens(&tokens).unwrap();
            assert_eq!(s.evaluate_next_chunk().unwrap(), 0);
            let memory = s.memory();
            assert!(matches!(
                metal::Session::new(
                    &model,
                    session::Options {
                        capacity,
                        prefill_chunk_size: chunk,
                        position_offset: offset,
                        memory_limit_bytes: memory.admitted_bytes - 1,
                    }
                ),
                Err(metal::Error::Budget { .. })
            ));
            drop(s);
            assert!(
                metal::Session::new(
                    &model,
                    session::Options {
                        capacity,
                        prefill_chunk_size: chunk,
                        position_offset: offset,
                        memory_limit_bytes: memory.admitted_bytes,
                    }
                )
                .is_ok()
            );
        }
    }
    for (capacity, chunk, offset) in [(2048, 1025, 0), (256, 512, 0), (2048, 1024, 1)] {
        assert!(matches!(
            metal::Session::new(
                &model,
                session::Options {
                    capacity,
                    prefill_chunk_size: chunk,
                    position_offset: offset,
                    ..Default::default()
                }
            ),
            Err(metal::Error::Options(_))
        ));
    }
    println!(
        "Eight complete exact-zero capacity checks: chunks128/256/512/1024, offsets0/1; exact budget and rejected oversized bounds"
    );
}

#[test]
#[ignore = "Explicit online attention full tiny independent stages and session state"]
fn online_attention_tiny_graph_and_session_contract() {
    let artifact = Artifact::tiny();
    let model = metal::Model::load(
        &artifact.0,
        weights::LoadOptions {
            context_limit: 32,
            ..Default::default()
        },
    )
    .unwrap();
    let source = Fixture::tiny();
    let half: Value = serde_json::from_slice(
        &fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fp16-v1/manifest.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let make = |chunk, offset| {
        metal::Session::with_attention_mode(
            &model,
            options(chunk, offset),
            metal::ProjectionMode::HalfCompensated,
            metal::NormalizationMode::Original,
            metal::ReadbackMode::FullLogits,
            metal::AttentionMode::Online,
        )
        .unwrap()
    };
    let mut compared = 0;
    for case in ["capacity", "future_changed", "offset", "prompt", "single"] {
        let tokens = source.u32(&format!("cases.{case}.tokens"));
        let offset = half["cases"][case]["position_offset"].as_u64().unwrap() as usize;
        let mut expected_names = std::collections::BTreeSet::from([
            format!("cases.{case}.embedding"),
            format!("cases.{case}.final_norm"),
            format!("cases.{case}.logits"),
        ]);
        for layer in 0..2 {
            for stage in [
                "input_norm",
                "q_projection",
                "k_projection",
                "v_projection",
                "q_rotary",
                "k_rotary",
                "attention_context",
                "attention_projection",
                "attention_residual",
                "post_attention_norm",
                "gate_projection",
                "gate_silu",
                "up_projection",
                "gated",
                "down_projection",
                "output",
            ] {
                assert!(expected_names.insert(format!("cases.{case}.layers.{layer}.{stage}")));
            }
        }
        assert_eq!(expected_names.len(), 35);
        for chunk in [tokens.len(), 3.min(tokens.len()), 1] {
            let mut s = make(chunk, offset);
            let reference = metal::Session::new(&model, options(chunk, offset)).unwrap();
            assert!(s.memory().scratch_bytes < reference.memory().scratch_bytes);
            drop(reference);
            s.enable_diagnostics(&[0, 1]).unwrap();
            s.set_tokens(&tokens).unwrap();
            let mut start = 0;
            loop {
                let rows = s.evaluate_next_chunk().unwrap();
                if rows == 0 {
                    break;
                }
                assert_eq!(rows, chunk.min(tokens.len() - start));
                let diagnostics = s.diagnostics().unwrap();
                assert_eq!(diagnostics.len(), 35);
                let mut seen = std::collections::BTreeSet::new();
                for d in diagnostics {
                    let name = diagnostic_name(case, &d);
                    assert!(expected_names.contains(&name) && seen.insert(name.clone()));
                    let expected = source.f32(&name);
                    let columns = expected.len() / tokens.len();
                    assert_eq!(expected.len(), tokens.len() * columns);
                    assert_eq!(
                        (d.rows, d.columns, d.values.len()),
                        (rows, columns, rows * columns)
                    );
                    qualify(
                        &d.values,
                        &expected[start * columns..(start + rows) * columns],
                        &half["arrays"][&name]["gate_against_source_fp32"],
                    );
                    compared += d.values.len();
                }
                assert_eq!(seen, expected_names);
                start += rows;
            }
            assert_eq!(start, tokens.len());
            assert_eq!(s.evaluated_len(), tokens.len());
            let saved = s.logits().unwrap().to_vec();
            s.set_tokens(&tokens).unwrap();
            assert_eq!(s.evaluate_next_chunk().unwrap(), 0);
            assert_eq!(s.logits().unwrap(), saved);
        }
    }
    let mut s = make(3, 0);
    for tokens in [
        vec![1, 7, 13, 3],
        vec![1, 7, 13, 3, 47],
        vec![1, 7, 13, 3, 11],
        vec![1, 7],
        vec![],
        vec![11],
    ] {
        s.set_tokens(&tokens).unwrap();
        s.evaluate().unwrap();
        let mut fresh = make(3, 0);
        fresh.set_tokens(&tokens).unwrap();
        fresh.evaluate().unwrap();
        assert_eq!(s.evaluated_len(), fresh.evaluated_len());
        assert_eq!(s.logits(), fresh.logits());
    }
    s.reset();
    s.append(&[1, 7, 13, 3]).unwrap();
    s.evaluate().unwrap();
    let memory = s.memory();
    ALLOCATIONS.with(|v| v.set(0));
    TRACK.with(|v| v.set(true));
    for _ in 0..20 {
        s.append(&[47]).unwrap();
        s.evaluate().unwrap();
    }
    TRACK.with(|v| v.set(false));
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
    assert_eq!(s.memory(), memory);
    let before = s.history().to_vec();
    assert!(s.append(&[0; 32]).is_err());
    assert_eq!(s.history(), before);
    generation_contract_session(
        &model,
        make(2, 0),
        metal::ProjectionMode::HalfCompensated,
        metal::AttentionMode::Online,
    );
    for readback in [
        metal::ReadbackMode::FullLogits,
        metal::ReadbackMode::GreedyToken,
    ] {
        let mut session = metal::Session::with_attention_mode(
            &model,
            options(3, 0),
            metal::ProjectionMode::HalfCompensated,
            metal::NormalizationMode::Original,
            readback,
            metal::AttentionMode::Online,
        )
        .unwrap();
        session.set_tokens(&[1, 7, 13, 3]).unwrap();
        session.evaluate().unwrap();
        assert_eq!(session.greedy_token().unwrap().unwrap(), 47);
        session.append(&[47]).unwrap();
        assert!(session.greedy_token().is_none());
        session.evaluate().unwrap();
        assert_eq!(session.greedy_token().unwrap().unwrap(), 47);
    }
    let mut low_budget = options(3, 0);
    low_budget.memory_limit_bytes = make(3, 0).memory().admitted_bytes - 1;
    assert!(matches!(
        metal::Session::with_attention_mode(
            &model,
            low_budget,
            metal::ProjectionMode::HalfCompensated,
            metal::NormalizationMode::Original,
            metal::ReadbackMode::FullLogits,
            metal::AttentionMode::Online
        ),
        Err(metal::Error::Budget { .. })
    ));
    println!("online tiny: {compared} independent stage values; warmed allocations=0");
}
