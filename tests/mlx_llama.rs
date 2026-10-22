use serde_json::{Value, json};
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
const SHA: &str = "824d2ee2ae9dd5b623c5aa9db10c13b2e4dee32647ea5fa9d78274fb3529a1f5";
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mlx-llama-v1")
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
fn layer(l: &LayerTrace) -> [(&str, &[f32]); 18] {
    [
        ("input_norm", &l.input_norm),
        ("q_projection", &l.q_projection),
        ("k_projection", &l.k_projection),
        ("v_projection", &l.v_projection),
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
fn tied_and_untied_scalar_llama_match_every_frozen_stage_and_actual_upstream_logits() {
    let m = manifest();
    let mut arrays = 0;
    for variant in ["tied", "untied"] {
        let source = Source::open_dense(&root().join(variant), options()).unwrap();
        assert!(Source::open(&root().join(variant), options()).is_err());
        assert_eq!(source.architecture(), Architecture::Llama);
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
    assert_eq!(arrays, 390);
}
#[test]
fn cached_chunks_offsets_exact_reuse_reset_and_overflow_preserve_llama_state() {
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
            "toklet-llama-{}-{}",
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
fn ordinary_smollm_export_metadata_preserves_bias_free_graph() {
    let a = CopyArtifact::new("tied");
    for (key, value) in [
        ("rope_interleaved", json!(false)),
        ("pretraining_tp", json!(1)),
        ("is_llama_config", json!(true)),
        (
            "transformers.js_config",
            json!({"kv_cache_dtype":{"q4f16":"float16","fp16":"float16"}}),
        ),
    ] {
        a.config(key, value);
    }
    let source = Source::open_dense(&a.0, options()).unwrap();
    assert_eq!(source.architecture(), Architecture::Llama);
    assert!(source.tied_output());
    assert!(source.tensors().keys().all(|k| !k.ends_with(".bias")));
    // Even an additional otherwise plausible export field must be inspected first.
    a.config(
        "transformers.js_config",
        json!({"kv_cache_dtype":{"q4f16":"float16","fp16":"float16"},"unknown":true}),
    );
    assert!(Source::open_dense(&a.0, options()).is_err());
}
#[test]
fn unsupported_llama_graph_variants_geometry_and_head_mismatches_reject() {
    manifest();
    for (key, value) in [
        ("model_type", json!("mistral")),
        ("architectures", json!(["Other"])),
        ("hidden_act", json!("gelu")),
        ("attention_bias", json!(true)),
        ("mlp_bias", json!(true)),
        ("attention_dropout", json!(0.1)),
        ("rope_traditional", json!(true)),
        ("rope_interleaved", json!(true)),
        ("pretraining_tp", json!(2)),
        ("is_llama_config", json!(false)),
        (
            "transformers.js_config",
            json!({"kv_cache_dtype":{"fp16":"int8"}}),
        ),
        ("rope_scaling", json!({"factor":2})),
        ("use_sliding_window", json!(true)),
        ("sliding_window", json!(16)),
        ("partial_rotary_factor", json!(0.5)),
        ("head_dim", json!(16)),
        ("num_attention_heads", json!(0)),
        ("num_key_value_heads", json!(3)),
        ("rope_theta", json!(0)),
        ("rms_norm_eps", json!(0)),
        ("max_position_embeddings", json!(31)),
        ("tie_word_embeddings", json!(false)),
        ("layer_types", json!(["full_attention"])),
    ] {
        let a = CopyArtifact::new("tied");
        a.config(key, value);
        assert!(Source::open_dense(&a.0, options()).is_err(), "{key}");
    }
    let a = CopyArtifact::new("untied");
    a.config("tie_word_embeddings", json!(true));
    assert!(Source::open_dense(&a.0, options()).is_err());
}
#[test]
fn explicit_projection_biases_reject_even_if_zero() {
    let a = CopyArtifact::new("tied");
    let raw = fs::read(a.0.join("model.safetensors")).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&raw).unwrap();
    let zero = vec![0u8; 32 * 4];
    let mut views = tensors.tensors();
    views.push((
        "model.layers.0.self_attn.q_proj.bias".into(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![32], &zero).unwrap(),
    ));
    safetensors::serialize_to_file(views, None, &a.0.join("model.safetensors")).unwrap();
    assert!(Source::open_dense(&a.0, options()).is_err());
}

#[test]
fn public_dense_constructors_validate_bias_and_independent_head_boundaries() {
    use toklet::model::{Config, LayerWeights, Model, Projection, Weights};
    let config = Config {
        vocab_size: 2,
        hidden_size: 4,
        intermediate_size: 4,
        num_layers: 1,
        query_heads: 2,
        kv_heads: 1,
        head_dim: 2,
        max_positions: 4,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
    };
    let weights = |with_bias: bool| Weights {
        embedding: vec![0.0; 8],
        final_norm: vec![1.0; 4],
        layers: vec![LayerWeights {
            input_norm: vec![1.0; 4],
            post_attention_norm: vec![1.0; 4],
            q: Projection {
                weight: vec![0.0; 16],
                bias: if with_bias { vec![0.0; 4] } else { vec![] },
            },
            k: Projection {
                weight: vec![0.0; 8],
                bias: if with_bias { vec![0.0; 2] } else { vec![] },
            },
            v: Projection {
                weight: vec![0.0; 8],
                bias: if with_bias { vec![0.0; 2] } else { vec![] },
            },
            o: vec![0.0; 16],
            gate: vec![0.0; 16],
            up: vec![0.0; 16],
            down: vec![0.0; 16],
        }],
    };
    assert!(Model::new(config.clone(), weights(false)).is_err());
    assert!(Model::new(config.clone(), weights(true)).is_ok());
    assert!(Model::new_bias_free(config.clone(), weights(true), None).is_err());
    assert!(Model::new_bias_free(config.clone(), weights(false), Some(vec![0.0; 7])).is_err());
    assert!(Model::new_bias_free(config.clone(), weights(false), Some(vec![f32::NAN; 8])).is_err());
    let tied = Model::new_bias_free(config.clone(), weights(false), None).unwrap();
    let mut head = Vec::with_capacity(16);
    head.resize(8, 0.0);
    let bytes = head.capacity() * 4;
    let untied = Model::new_bias_free(config, weights(false), Some(head)).unwrap();
    assert_eq!(
        untied.allocated_weights_bytes().unwrap() - tied.allocated_weights_bytes().unwrap(),
        bytes
    );
}
