use serde_json::{Value, json};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
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

use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use toklet::{
    mlx::{dense::Architecture, qwen::Source, reference},
    model::{LayerTrace, Trace},
    session::{self, Session},
    weights::{Error, LoadOptions},
};
const SHA: &str = "5dc5b7ec0fccd8ac6876814b88a917107993aab5f5e07dcb222f57ec377a7a4a";
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mlx-qwen3-v1")
}
fn manifest() -> Value {
    let raw = fs::read(root().join("manifest.json")).unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(&raw)), SHA);
    let m: Value = serde_json::from_slice(&raw).unwrap();
    for (name, item) in m["files"].as_object().unwrap() {
        let bytes = fs::read(root().join(name)).unwrap();
        assert_eq!(bytes.len() as u64, item["bytes"]);
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), item["sha256"]);
    }
    m
}
fn array(m: &Value, name: &str) -> Vec<f32> {
    let item = &m["arrays"][name];
    let raw = fs::read(root().join("arrays.bin")).unwrap();
    let start = item["offset"].as_u64().unwrap() as usize;
    let bytes = &raw[start..start + item["bytes"].as_u64().unwrap() as usize];
    assert_eq!(format!("{:x}", Sha256::digest(bytes)), item["sha256"]);
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite()
                && b.is_finite()
                && (f64::from(a) - f64::from(b)).abs() <= 1e-5 + 1e-4 * f64::from(b).abs(),
            "index{i}: {a} vs {b}"
        );
    }
}
fn options() -> LoadOptions {
    LoadOptions {
        context_limit: 32,
        memory_limit_bytes: 128 * 1024 * 1024,
    }
}
fn tokens(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_u64().unwrap() as u32)
        .collect()
}
fn values(trace: &Trace) -> BTreeMap<String, &[f32]> {
    let mut result = BTreeMap::from([
        ("embedding".into(), trace.embedding.as_slice()),
        ("final_norm".into(), trace.final_norm.as_slice()),
        ("logits".into(), trace.logits.as_slice()),
    ]);
    for (i, l) in trace.layers.iter().enumerate() {
        for (name, data) in layer(l) {
            result.insert(format!("layers.{i}.{name}"), data);
        }
    }
    result
}
fn layer(l: &LayerTrace) -> [(&str, &[f32]); 20] {
    [
        ("input_norm", &l.input_norm),
        ("q_projection", &l.q_projection),
        ("k_projection", &l.k_projection),
        ("v_projection", &l.v_projection),
        ("q_norm", &l.q_norm),
        ("k_norm", &l.k_norm),
        ("q_rotary", &l.q_rotary),
        ("k_rotary", &l.k_rotary),
        ("attention_scores", &l.attention_scores),
        ("attention_probabilities", &l.attention_probabilities),
        ("attention_context", &l.attention_context),
        ("attention_projection", &l.attention_projection),
        ("attention_residual", &l.attention_residual),
        ("post_attention_norm", &l.post_attention_norm),
        ("gate_projection", &l.gate_projection),
        ("up_projection", &l.up_projection),
        ("gate_silu", &l.gate_silu),
        ("gated", &l.gated),
        ("down_projection", &l.down_projection),
        ("output", &l.output),
    ]
}

#[test]
fn tied_and_untied_scalar_qwen3_match_every_frozen_stage_and_actual_upstream_logits() {
    let m = manifest();
    let mut arrays = 0;
    for variant in ["tied", "untied"] {
        let source = Source::open_dense(&root().join(variant), options()).unwrap();
        assert!(Source::open(&root().join(variant), options()).is_err());
        assert_eq!(source.architecture(), Architecture::Qwen3);
        assert_eq!(source.tied_output(), variant == "tied");
        assert_eq!(
            source.tensors().contains_key("lm_head.weight"),
            variant == "untied"
        );
        assert!(source.tensors().keys().all(|k| !k.ends_with(".bias")));
        assert!(reference::load_qwen(&root().join(variant), options()).is_err());
        let loaded = reference::load_dense(&root().join(variant), options()).unwrap();
        assert_eq!(
            loaded.model.allocated_weights_bytes().unwrap(),
            loaded.fp32_weights_bytes
        );
        for (case, spec) in m["variants"][variant]["cases"].as_object().unwrap() {
            let ts = tokens(&spec["tokens"]);
            let offset = spec["position_offset"].as_u64().unwrap() as usize;
            let mut scratch = loaded.model.scratch(ts.len()).unwrap();
            let trace = loaded
                .model
                .forward_diagnostic(&ts, offset, &mut scratch)
                .unwrap();
            for (stage, actual) in values(&trace) {
                close(
                    actual,
                    &array(&m, &format!("{variant}.fp32.cases.{case}.{stage}")),
                );
                arrays += 1;
            }
            if offset == 0 {
                for backend in ["CPU", "GPU"] {
                    close(
                        &trace.logits,
                        &array(&m, &format!("{variant}.native.{backend}.{case}.logits")),
                    );
                }
            }
        }
    }
    assert_eq!(arrays, 430);
}
#[test]
fn cached_chunks_offsets_exact_reuse_reset_and_overflow_preserve_qwen3_state() {
    let m = manifest();
    for variant in ["tied", "untied"] {
        let loaded = reference::load_dense(&root().join(variant), options()).unwrap();
        for (case, spec) in m["variants"][variant]["cases"].as_object().unwrap() {
            let ts = tokens(&spec["tokens"]);
            let offset = spec["position_offset"].as_u64().unwrap() as usize;
            for chunk in [1, 3, ts.len()] {
                let mut s = Session::new(
                    &loaded.model,
                    session::Options {
                        capacity: 32 - offset,
                        prefill_chunk_size: chunk,
                        position_offset: offset,
                        memory_limit_bytes: 128 * 1024 * 1024,
                    },
                )
                .unwrap();
                s.set_tokens(&ts).unwrap();
                s.evaluate().unwrap();
                let expected = array(&m, &format!("{variant}.fp32.cases.{case}.logits"));
                close(s.logits().unwrap(), &expected[expected.len() - 64..]);
                let before = s.logits().unwrap().to_vec();
                s.set_tokens(&ts).unwrap();
                s.evaluate().unwrap();
                assert_eq!(s.logits().unwrap(), before);
                assert_eq!(s.evaluated_len(), ts.len());
                assert!(s.set_tokens(&vec![1; 33 - offset]).is_err());
                assert_eq!(s.evaluated_len(), ts.len());
                assert_eq!(s.logits().unwrap(), before);
                s.reset();
                s.set_tokens(&ts).unwrap();
                s.evaluate().unwrap();
                assert_eq!(s.logits().unwrap(), before);
            }
        }
    }
}
#[test]
fn independent_output_is_counted_and_admitted_before_expansion() {
    manifest();
    let tied = reference::load_dense(&root().join("tied"), options()).unwrap();
    let untied = reference::load_dense(&root().join("untied"), options()).unwrap();
    assert_eq!(
        untied.fp32_weights_bytes - tied.fp32_weights_bytes,
        64 * 32 * 4
    );
    for variant in ["tied", "untied"] {
        let source = Source::open_dense(&root().join(variant), options()).unwrap();
        let exact = source.admitted_load_bytes();
        assert!(
            Source::open_dense(
                &root().join(variant),
                LoadOptions {
                    memory_limit_bytes: exact - 1,
                    ..options()
                }
            )
            .is_err()
        );
        assert!(
            Source::open_dense(
                &root().join(variant),
                LoadOptions {
                    memory_limit_bytes: exact,
                    ..options()
                }
            )
            .is_ok()
        );
        let loaded = reference::load_dense(&root().join(variant), options()).unwrap();
        let limit = loaded.admitted_load_bytes - 1;
        assert!(matches!(
            reference::load_dense(
                &root().join(variant),
                LoadOptions {
                    memory_limit_bytes: limit,
                    ..options()
                }
            ),
            Err(Error::Budget { .. })
        ));
    }
}
static NEXT: AtomicU64 = AtomicU64::new(0);
struct CopyArtifact(PathBuf);
impl CopyArtifact {
    fn new(variant: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "toklet-qwen3-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        for name in ["config.json", "model.safetensors"] {
            fs::copy(root().join(variant).join(name), path.join(name)).unwrap();
        }
        Self(path)
    }
    fn config(&self, key: &str, value: Value) {
        let file = self.0.join("config.json");
        let mut c: Value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
        c[key] = value;
        fs::write(file, serde_json::to_vec(&c).unwrap()).unwrap();
    }
}
impl Drop for CopyArtifact {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn qwen3_config_is_strict_and_ordinary_export_metadata_is_allowed() {
    let a = CopyArtifact::new("tied");
    for (key, value) in [
        ("max_window_layers", json!(2)),
        ("sliding_window", Value::Null),
        ("use_sliding_window", json!(false)),
        ("eos_token_id", json!(63)),
    ] {
        a.config(key, value);
    }
    assert!(Source::open_dense(&a.0, options()).is_ok());
    for (key, value) in [
        ("model_type", json!("qwen3_moe")),
        ("architectures", json!(["Qwen2ForCausalLM"])),
        ("hidden_act", json!("gelu")),
        ("attention_bias", json!(true)),
        ("mlp_bias", json!(true)),
        ("attention_dropout", json!(0.1)),
        ("rope_scaling", json!({"factor":2})),
        ("rope_parameters", json!({"rope_type":"default"})),
        ("rope_traditional", json!(false)),
        ("use_sliding_window", json!(true)),
        ("sliding_window", json!(16)),
        ("max_window_layers", json!(3)),
        ("head_dim", json!(0)),
        ("head_dim", json!(15)),
        ("head_dim", json!(8)),
        ("hidden_size", json!(0)),
        ("num_attention_heads", json!(0)),
        ("num_key_value_heads", json!(3)),
        ("num_attention_heads", json!(u64::MAX)),
        ("rms_norm_eps", json!(0)),
        ("rope_theta", json!(0)),
        ("max_position_embeddings", json!(31)),
        ("torch_dtype", json!("int8")),
        ("tie_word_embeddings", json!(false)),
        ("layer_types", json!(["full_attention"])),
        ("partial_rotary_factor", json!(0.5)),
        ("num_experts", json!(4)),
    ] {
        let a = CopyArtifact::new("tied");
        a.config(key, value);
        assert!(Source::open_dense(&a.0, options()).is_err(), "{key}");
    }
}

#[test]
fn missing_malformed_nonfinite_or_extra_normalization_tensors_reject() {
    use safetensors::{Dtype, tensor::TensorView};
    for mode in ["missing", "wrong_shape", "nonfinite", "extra", "bias"] {
        let a = CopyArtifact::new("tied");
        let file = a.0.join("model.safetensors");
        let raw = fs::read(&file).unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&raw).unwrap();
        let mut views = tensors.tensors();
        let target = "model.layers.0.self_attn.q_norm.weight";
        let shape = if mode == "wrong_shape" {
            vec![4, 4]
        } else {
            vec![16]
        };
        let payload: Vec<_> = (0..16)
            .flat_map(|_| if mode == "nonfinite" { f32::NAN } else { 1.0 }.to_le_bytes())
            .collect();
        match mode {
            "missing" => views.retain(|(n, _)| n != target),
            "wrong_shape" | "nonfinite" => {
                views.retain(|(n, _)| n != target);
                views.push((
                    target.into(),
                    TensorView::new(Dtype::F32, shape, &payload).unwrap(),
                ));
            }
            "extra" | "bias" => views.push((
                format!(
                    "model.layers.0.self_attn.{}",
                    if mode == "extra" {
                        "v_norm.weight"
                    } else {
                        "q_norm.bias"
                    }
                ),
                TensorView::new(Dtype::F32, shape, &payload).unwrap(),
            )),
            _ => unreachable!(),
        }
        safetensors::serialize_to_file(views, None, &file).unwrap();
        assert!(reference::load_dense(&a.0, options()).is_err(), "{mode}");
    }
}

#[test]
fn qk_norm_constructor_checks_ownership_shapes_finiteness_and_independent_widths() {
    use toklet::model::{Config, LayerWeights, Model, Projection, QkNormWeights, Weights};
    for query_heads in [1, 4] {
        let c = Config {
            vocab_size: 2,
            hidden_size: 4,
            intermediate_size: 4,
            num_layers: 1,
            query_heads,
            kv_heads: 1,
            head_dim: 2,
            max_positions: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
        };
        let q = query_heads * 2;
        let weights = || Weights {
            embedding: vec![0.0; 8],
            final_norm: vec![1.0; 4],
            layers: vec![LayerWeights {
                input_norm: vec![1.0; 4],
                post_attention_norm: vec![1.0; 4],
                q: Projection {
                    weight: vec![0.0; q * 4],
                    bias: vec![],
                },
                k: Projection {
                    weight: vec![0.0; 8],
                    bias: vec![],
                },
                v: Projection {
                    weight: vec![0.0; 8],
                    bias: vec![],
                },
                o: vec![0.0; 4 * q],
                gate: vec![0.0; 16],
                up: vec![0.0; 16],
                down: vec![0.0; 16],
            }],
        };
        for norms in [
            vec![],
            vec![QkNormWeights {
                q: vec![1.0],
                k: vec![1.0; 2],
            }],
            vec![QkNormWeights {
                q: vec![1.0; 2],
                k: vec![f32::NAN; 2],
            }],
        ] {
            assert!(Model::new_qk_normalized(c.clone(), weights(), None, norms).is_err());
        }
        let plain = Model::new_bias_free(c.clone(), weights(), None).unwrap();
        let normalized = Model::new_qk_normalized(
            c,
            weights(),
            None,
            vec![QkNormWeights {
                q: vec![1.0; 2],
                k: vec![1.0; 2],
            }],
        )
        .unwrap();
        assert_eq!(
            normalized.allocated_weights_bytes().unwrap()
                - plain.allocated_weights_bytes().unwrap(),
            16
        );
        let mut scratch = normalized.scratch(2).unwrap();
        let mut logits = [f32::NAN; 2];
        normalized
            .forward(&[0, 1], 0, &mut scratch, &mut logits)
            .unwrap();
        assert_eq!(logits, [0.0; 2]);

        let mut q_norm = Vec::with_capacity(4);
        q_norm.resize(2, 1.0);
        let mut k_norm = Vec::with_capacity(8);
        k_norm.resize(2, 1.0);
        let extra_weights = (q_norm.capacity() + k_norm.capacity()) * 4;
        let mut norms = Vec::with_capacity(5);
        norms.push(QkNormWeights {
            q: q_norm,
            k: k_norm,
        });
        let extra_metadata = norms.capacity() * std::mem::size_of::<QkNormWeights>();
        let over_reserved =
            Model::new_qk_normalized(normalized.config().clone(), weights(), None, norms).unwrap();
        let settings = session::Options {
            capacity: 4,
            prefill_chunk_size: 1,
            position_offset: 0,
            memory_limit_bytes: 128 * 1024 * 1024,
        };
        let plain_session = Session::new(&plain, settings).unwrap();
        let over_session = Session::new(&over_reserved, settings).unwrap();
        assert_eq!(
            over_session.memory().weights_bytes - plain_session.memory().weights_bytes,
            extra_weights
        );
        assert_eq!(
            over_session.memory().metadata_bytes - plain_session.memory().metadata_bytes,
            extra_metadata
        );
        let exact = over_session.memory().admitted_bytes;
        assert!(
            Session::new(
                &over_reserved,
                session::Options {
                    memory_limit_bytes: exact - 1,
                    ..settings
                }
            )
            .is_err()
        );
        assert!(
            Session::new(
                &over_reserved,
                session::Options {
                    memory_limit_bytes: exact,
                    ..settings
                }
            )
            .is_ok()
        );
    }
}

#[cfg(feature = "metal")]
#[test]
fn unqualified_qwen3_geometry_rejects_before_device_creation() {
    let a = CopyArtifact::new("tied");
    a.config("rope_theta", json!(1e6));
    let result = toklet::metal::Model::load_mlx(&a.0, options());
    assert!(
        matches!(result,Err(toklet::metal::Error::Unsupported(reason)) if reason.contains("Qwen3"))
    );
}

#[test]
fn normalized_decode_is_allocation_free_and_admission_accounts_for_norms() {
    for variant in ["tied", "untied"] {
        let loaded = reference::load_dense(&root().join(variant), options()).unwrap();
        let options = session::Options {
            capacity: 32,
            prefill_chunk_size: 3,
            position_offset: 0,
            memory_limit_bytes: 128 * 1024 * 1024,
        };
        let mut s = Session::new(&loaded.model, options).unwrap();
        let memory = s.memory();
        assert_eq!(memory.weights_bytes, loaded.fp32_weights_bytes);
        assert!(
            Session::new(
                &loaded.model,
                session::Options {
                    memory_limit_bytes: memory.admitted_bytes - 1,
                    ..options
                }
            )
            .is_err()
        );
        assert!(
            Session::new(
                &loaded.model,
                session::Options {
                    memory_limit_bytes: memory.admitted_bytes,
                    ..options
                }
            )
            .is_ok()
        );
        s.set_tokens(&[1, 7, 13, 3]).unwrap();
        s.evaluate().unwrap();
        s.append(&[11]).unwrap();
        ALLOCATIONS.with(|c| c.set(0));
        TRACK.with(|c| c.set(true));
        let result = s.evaluate_next_chunk();
        TRACK.with(|c| c.set(false));
        let allocations = ALLOCATIONS.with(Cell::get);
        assert_eq!(result.unwrap(), 1);
        assert_eq!(allocations, 0);
        assert_eq!(s.memory(), memory);
    }
}

#[test]
fn cancellation_pending_append_and_edited_replay_preserve_a_valid_qwen3_prefix() {
    use std::sync::atomic::AtomicBool;
    use toklet::generation::{self, Input, Request, StopReason};
    for variant in ["tied", "untied"] {
        let loaded = reference::load_dense(&root().join(variant), options()).unwrap();
        let mut s = Session::new(
            &loaded.model,
            session::Options {
                capacity: 32,
                prefill_chunk_size: 2,
                position_offset: 0,
                memory_limit_bytes: 128 * 1024 * 1024,
            },
        )
        .unwrap();
        let cancel = AtomicBool::new(true);
        let request = Request {
            input: Input::Prompt(&[1, 7, 13, 3]),
            max_new_tokens: 4,
            stop_tokens: &[],
        };
        let report = generation::generate(&mut s, request, &cancel, |_| panic!()).unwrap();
        assert_eq!(report.reason, StopReason::Cancelled);
        assert!(s.history().is_empty());
        cancel.store(false, Ordering::Relaxed);
        let report = generation::generate(
            &mut s,
            Request {
                input: Input::Prompt(&[1, 7, 13, 3]),
                max_new_tokens: 4,
                stop_tokens: &[],
            },
            &cancel,
            |_| {
                cancel.store(true, Ordering::Relaxed);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(report.reason, StopReason::Cancelled);
        assert_eq!(report.emitted_tokens, 1);
        assert_eq!(s.evaluated_len(), 4);
        assert_eq!(s.pending_len(), 1);
        let mut scratch = loaded.model.scratch(32).unwrap();
        let mut expected = vec![0.0; 64];
        loaded
            .model
            .forward(s.history(), 0, &mut scratch, &mut expected)
            .unwrap();
        let next = toklet::sampling::greedy(&expected).unwrap();
        cancel.store(false, Ordering::Relaxed);
        let report = generation::generate(
            &mut s,
            Request {
                input: Input::Continue,
                max_new_tokens: 1,
                stop_tokens: &[],
            },
            &cancel,
            |event| {
                assert_eq!(event.token_id, next);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(report.prefill_tokens, 1);
        assert_eq!(report.reused_tokens, 4);
        assert_eq!(s.evaluated_len(), 5);
        assert_eq!(s.pending_len(), 1);
        s.discard_pending();
        let mut edited = s.history().to_vec();
        edited[1] = 2;
        s.set_tokens(&edited).unwrap();
        assert_eq!(s.evaluated_len(), 1);
        s.evaluate().unwrap();
        loaded
            .model
            .forward(&edited, 0, &mut scratch, &mut expected)
            .unwrap();
        close(s.logits().unwrap(), &expected);
        s.append(&[17]).unwrap();
        s.evaluate().unwrap();
        loaded
            .model
            .forward(s.history(), 0, &mut scratch, &mut expected)
            .unwrap();
        close(s.logits().unwrap(), &expected);
    }
}
