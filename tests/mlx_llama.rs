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
    model::{LayerTrace, PositionPlan, Trace},
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
fn llama3_scaling() -> Value {
    json!({"rope_type":"llama3", "factor":8.0, "low_freq_factor":1.0,
        "high_freq_factor":4.0, "original_max_position_embeddings":64})
}

#[test]
fn scaled_llama_source_model_and_cached_graph_match_independent_fixture() {
    manifest();
    let raw = include_bytes!("fixtures/llama3-graph-v1/graph.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "9e4d80dcec02b25f2f94f34225171dd30c279fdf6b7347f72ad75a4988c61402"
    );
    let fixture: Value = serde_json::from_slice(raw).unwrap();
    assert_eq!(
        fixture["producer"]["sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../tools/reference/llama3_graph.py"))
        )
    );
    let plan = PositionPlan::Llama3(toklet::cpu::Llama3Scaling::new(8., 1., 4., 64).unwrap());
    for variant in ["tied", "untied"] {
        let a = CopyArtifact::new(variant);
        a.config("rope_scaling", llama3_scaling());
        let source = Source::open_dense(&a.0, options()).unwrap();
        assert_eq!(source.position_plan(), plan);
        assert!(Source::open(&a.0, options()).is_err());
        let loaded = reference::load_dense(&a.0, options()).unwrap();
        assert_eq!(loaded.model.position_plan(), plan);
        let ordinary = reference::load_dense(&root().join(variant), options()).unwrap();
        assert_eq!(ordinary.model.position_plan(), PositionPlan::Ordinary);
        for case in fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["variant"] == variant)
        {
            let ts = tokens(&case["tokens"]);
            let offset = case["offset"].as_u64().unwrap() as usize;
            let expected = |name: &str| {
                case["arrays"][name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>()
            };
            let mut scratch = loaded.model.scratch(ts.len()).unwrap();
            let trace = loaded
                .model
                .forward_diagnostic(&ts, offset, &mut scratch)
                .unwrap();
            for (i, layer) in trace.layers.iter().enumerate() {
                close(&layer.q_rotary, &expected(&format!("layer{i}.q_rotary")));
                close(&layer.k_rotary, &expected(&format!("layer{i}.k_rotary")));
            }
            let logits = expected("logits");
            close(&trace.logits, &logits);
            let mut old_scratch = ordinary.model.scratch(ts.len()).unwrap();
            let old = ordinary
                .model
                .forward_diagnostic(&ts, offset, &mut old_scratch)
                .unwrap();
            assert!(
                trace
                    .logits
                    .iter()
                    .zip(&old.logits)
                    .any(|(a, b)| (a - b).abs() > 1e-5)
            );
            for chunk in [1, 3] {
                let mut session = Session::new(
                    &loaded.model,
                    session::Options {
                        capacity: 32 - offset,
                        prefill_chunk_size: chunk,
                        position_offset: offset,
                        memory_limit_bytes: 128 * 1024 * 1024,
                    },
                )
                .unwrap();
                session.set_tokens(&ts).unwrap();
                let mut cached = Vec::new();
                while session.pending_len() > 0 {
                    let count = session.pending_len().min(chunk);
                    let mut rows = vec![0.; count * 64];
                    assert_eq!(
                        session.evaluate_next_chunk_with_logits(&mut rows).unwrap(),
                        count
                    );
                    cached.extend(rows);
                }
                close(&cached, &logits);
                let before = session.logits().unwrap().to_vec();
                session.set_tokens(&ts).unwrap();
                session.evaluate().unwrap();
                assert_eq!(session.logits().unwrap(), before);
                assert!(session.append(&[64]).is_err());
                assert_eq!(session.logits().unwrap(), before);
                session.reset();
                session.set_tokens(&ts).unwrap();
                session.evaluate().unwrap();
                assert_eq!(session.logits().unwrap(), before);
            }
        }
    }
}

#[test]
fn scaled_llama_parser_aliases_and_backend_capability_are_explicit() {
    use toklet::mlx::compatibility::{self, Backend, Reason, Status};
    for aliases in [
        json!({"rope_type":"llama3"}),
        json!({"type":"llama3"}),
        json!({"type":"llama3","rope_type":"llama3"}),
    ] {
        let a = CopyArtifact::new("tied");
        let mut scale = llama3_scaling();
        scale.as_object_mut().unwrap().remove("rope_type");
        scale
            .as_object_mut()
            .unwrap()
            .extend(aliases.as_object().unwrap().clone());
        a.config("rope_scaling", scale);
        let source = Source::open_dense(&a.0, options()).unwrap();
        assert!(matches!(source.position_plan(), PositionPlan::Llama3(_)));
        let cpu = compatibility::inspect(&a.0, Backend::ScalarReference, options());
        assert_eq!(cpu.status, Status::CompatibleUntested);
        let metal = compatibility::inspect(&a.0, Backend::Metal, options());
        assert_eq!(metal.status, Status::CompatibleUntested);
        assert!(matches!(
            metal.reason,
            Reason::MetadataAccepted | Reason::BackendUnavailable
        ));
        assert!(metal.resources.unwrap().metal_runtime_planning_required);
    }
    for (key, value) in [
        ("rope_type", json!("dynamic")),
        ("type", json!("linear")),
        ("rope_type", Value::Null),
        ("factor", json!(0.5)),
        ("low_freq_factor", json!(0)),
        ("high_freq_factor", json!(1)),
        ("original_max_position_embeddings", json!(0)),
        ("attention_factor", json!(2)),
        ("factor", json!("8")),
    ] {
        let a = CopyArtifact::new("tied");
        let mut scale = llama3_scaling();
        scale[key] = value;
        a.config("rope_scaling", scale);
        assert!(Source::open_dense(&a.0, options()).is_err(), "{key}");
    }
    let a = CopyArtifact::new("tied");
    let mut scale = llama3_scaling();
    scale.as_object_mut().unwrap().remove("rope_type");
    a.config("rope_scaling", scale);
    assert!(Source::open_dense(&a.0, options()).is_err());
    a.config("model_type", json!("mistral"));
    a.config("architectures", json!(["MistralForCausalLM"]));
    a.config("sliding_window", Value::Null);
    a.config("rope_scaling", llama3_scaling());
    assert!(Source::open_dense(&a.0, options()).is_err());
}

#[cfg(all(
    feature = "metal",
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "ios")
))]
#[test]
#[ignore = "explicit Apple GPU scaled positional resource accounting and factory boundaries"]
fn scaled_llama_runtime_and_loading_count_one_shared_positional_owner() {
    use toklet::metal::{self, AttentionMode, NormalizationMode, ProjectionMode, ReadbackMode};
    let a = CopyArtifact::new("tied");
    a.config("rope_scaling", llama3_scaling());
    let opts = session::Options {
        capacity: 32,
        prefill_chunk_size: 4,
        position_offset: 0,
        memory_limit_bytes: 128 * 1024 * 1024,
    };
    let resource = metal::planning::inspect_mlx_capacity(
        &a.0,
        options(),
        opts,
        NormalizationMode::Compensated,
        ReadbackMode::FullLogits,
        AttentionMode::Reference,
    )
    .unwrap();
    assert!(!resource.resources.numerical_buffers_allocated);
    assert!(resource.resources.positional_resident_bytes > 0);
    assert_eq!(
        resource.resources.positional_resident_bytes,
        resource.resources.session_memory.positional_bytes
    );
    assert_eq!(
        resource
            .capacity
            .as_ref()
            .unwrap()
            .selected
            .unwrap()
            .capacity,
        32
    );
    let model = metal::Model::load_mlx(&a.0, options()).unwrap();
    assert_eq!(
        model.weights_bytes(),
        resource.resources.resident_weights_bytes
    );
    assert_eq!(
        model.positional_resident_bytes(),
        resource.resources.positional_resident_bytes
    );
    assert_eq!(
        model.model_resident_bytes().unwrap(),
        model.weights_bytes() + model.positional_resident_bytes()
    );
    let ordinary = metal::Model::load_mlx(&root().join("tied"), options()).unwrap();
    assert_eq!(ordinary.weights_bytes(), model.weights_bytes());
    assert_eq!(ordinary.positional_resident_bytes(), 0);
    let create = |o| {
        metal::Session::with_attention_mode(
            &model,
            o,
            ProjectionMode::MlxAffine,
            NormalizationMode::Compensated,
            ReadbackMode::FullLogits,
            AttentionMode::Reference,
        )
    };
    let first = create(opts).unwrap();
    assert_eq!(first.memory(), resource.resources.session_memory);
    let exact = first.memory().admitted_bytes;
    assert!(matches!(
        create(session::Options {
            memory_limit_bytes: exact - 1,
            ..opts
        }),
        Err(metal::Error::Budget { .. })
    ));
    let second = create(session::Options {
        memory_limit_bytes: exact,
        ..opts
    })
    .unwrap();
    assert_eq!(second.memory(), first.memory());
    assert_eq!(
        first.memory().admitted_bytes + second.memory().admitted_bytes
            - model.model_resident_bytes().unwrap(),
        model.model_resident_bytes().unwrap() + 2 * (exact - model.model_resident_bytes().unwrap())
    );
    let load = resource
        .resources
        .resident_admitted_load_bytes
        .max(resource.resources.source_admitted_load_bytes);
    assert!(
        metal::Model::load_mlx(
            &a.0,
            LoadOptions {
                memory_limit_bytes: load - 1,
                ..options()
            }
        )
        .is_err()
    );
    let precise = metal::Model::load_mlx(
        &a.0,
        LoadOptions {
            memory_limit_bytes: load,
            ..options()
        },
    )
    .unwrap();
    assert_eq!(
        precise.model_resident_bytes().unwrap(),
        model.model_resident_bytes().unwrap()
    );
    let mut small = opts;
    small.memory_limit_bytes = exact - 1;
    let denied = metal::planning::inspect_mlx_capacity(
        &a.0,
        options(),
        small,
        NormalizationMode::Compensated,
        ReadbackMode::FullLogits,
        AttentionMode::Reference,
    )
    .unwrap();
    assert!(!denied.resources.fits_session_budget);
    assert_eq!(
        denied.capacity.as_ref().unwrap().selected.unwrap().capacity,
        31
    );
    assert_eq!(
        denied
            .capacity
            .as_ref()
            .unwrap()
            .first_excluded
            .unwrap()
            .capacity,
        32
    );
    assert!(matches!(
        metal::Session::with_attention_mode(
            &model,
            opts,
            ProjectionMode::MlxAffine,
            NormalizationMode::Compensated,
            ReadbackMode::FullLogits,
            AttentionMode::Online
        ),
        Err(metal::Error::Unsupported(_))
    ));
    println!(
        "LLAMA3_POSITIONAL_RESOURCES {}",
        serde_json::json!({"report":resource,"weights":model.weights_bytes(),"positional":model.positional_resident_bytes(),"device":model.device_name(),"model_qualified":false})
    );
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

#[test]
fn smollm3_mask_selects_identity_and_rotary_before_cached_attention() {
    let raw = include_bytes!("fixtures/smollm3-half-graph-v1/fixture.json");
    let fixture: Value = serde_json::from_slice(raw).unwrap();
    for variant in ["tied", "untied"] {
        let a = CopyArtifact::new(variant);
        fs::write(
            a.0.join("config.json"),
            serde_json::to_vec(&fixture["config_templates"][variant]).unwrap(),
        )
        .unwrap();
        let source = Source::open_dense(&a.0, options()).unwrap();
        assert_eq!(source.architecture(), Architecture::SmolLm3);
        assert_eq!(source.position_plan(), PositionPlan::Ordinary);
        let mask = if variant == "tied" {
            [true, false]
        } else {
            [false, true]
        };
        assert_eq!(source.rotary_layers(), Some(mask.as_slice()));
        assert!(Source::open(&a.0, options()).is_err());
        let loaded = reference::load_dense(&a.0, options()).unwrap();
        let tokens = [1, 7, 13, 3];
        let mut scratch = loaded.model.scratch(tokens.len()).unwrap();
        let trace = loaded
            .model
            .forward_diagnostic(&tokens, 5, &mut scratch)
            .unwrap();
        for (layer, enabled) in trace.layers.iter().zip(mask) {
            for (before, after) in [
                (&layer.q_projection, &layer.q_rotary),
                (&layer.k_projection, &layer.k_rotary),
            ] {
                if enabled {
                    let (sin, cos) = 5_f64.sin_cos();
                    close(
                        &after[..1],
                        &[(f64::from(before[0]) * cos - f64::from(before[4]) * sin) as f32],
                    );
                    assert_ne!(before, after);
                } else {
                    assert!(
                        before
                            .iter()
                            .zip(after)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    );
                }
            }
        }
        for chunk in [1, 3] {
            let mut s = Session::new(
                &loaded.model,
                session::Options {
                    capacity: 16,
                    prefill_chunk_size: chunk,
                    position_offset: 5,
                    memory_limit_bytes: 128 * 1024 * 1024,
                },
            )
            .unwrap();
            s.append(&tokens[..2]).unwrap();
            s.evaluate().unwrap();
            s.append(&tokens[2..]).unwrap();
            s.evaluate().unwrap();
            close(
                s.logits().unwrap(),
                &trace.logits[trace.logits.len() - 64..],
            );
            s.reset();
            s.append(&tokens).unwrap();
            s.evaluate().unwrap();
            close(
                s.logits().unwrap(),
                &trace.logits[trace.logits.len() - 64..],
            );
        }
    }
}

#[test]
fn smollm3_rejects_ambiguous_masks_and_attention_semantics() {
    let fixture: Value = serde_json::from_slice(include_bytes!(
        "fixtures/smollm3-half-graph-v1/fixture.json"
    ))
    .unwrap();
    for (name, bad) in [
        ("no_rope_layers", json!([1])),
        ("no_rope_layers", json!([true, false])),
        ("no_rope_layers", json!([1, 2])),
        ("no_rope_layers", json!(null)),
        ("no_rope_layer_interval", json!(0)),
        (
            "layer_types",
            json!(["full_attention", "sliding_attention"]),
        ),
        ("max_window_layers", json!(3)),
        ("pretraining_tp", json!(3)),
        ("rope_scaling", llama3_scaling()),
        ("attention_bias", json!(true)),
    ] {
        let a = CopyArtifact::new("tied");
        fs::write(
            a.0.join("config.json"),
            serde_json::to_vec(&fixture["config_templates"]["tied"]).unwrap(),
        )
        .unwrap();
        a.config(name, bad);
        assert!(Source::open_dense(&a.0, options()).is_err(), "{name}");
    }
}

#[test]
fn smollm3_independent_half_fixture_preserves_all_stage_gates() {
    let raw = include_bytes!("fixtures/smollm3-half-graph-v1/fixture.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "a6cf4fbc824e8a0af25721181be477015b907dd3f6d3c093501b4eba88813466"
    );
    let fixture: Value = serde_json::from_slice(raw).unwrap();
    for category in ["inputs", "producers"] {
        for (name, pin) in fixture[category].as_object().unwrap() {
            let bytes = fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(name)).unwrap();
            assert_eq!(bytes.len() as u64, pin["bytes"]);
            assert_eq!(format!("{:x}", Sha256::digest(bytes)), pin["sha256"]);
        }
    }
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 8);
    for case in fixture["cases"].as_array().unwrap() {
        assert_eq!(case["arrays"].as_object().unwrap().len(), 39);
        for spec in case["arrays"].as_object().unwrap().values() {
            assert_eq!(
                spec["main_bits"].as_array().unwrap().len(),
                spec["wide_bits"].as_array().unwrap().len()
            );
            assert_eq!(spec["gate"]["rtol"], 1e-3);
            assert_eq!(
                spec["gate"]["atol"].as_f64().unwrap(),
                1e-4_f64.max(2.0 * spec["precision_envelope"].as_f64().unwrap())
            );
        }
    }
}

#[test]
fn smollm3_implicit_tying_requires_one_table_and_explicit_layer_semantics() {
    let fixture: Value = serde_json::from_slice(include_bytes!(
        "fixtures/smollm3-half-graph-v1/fixture.json"
    ))
    .unwrap();
    for (variant, remove) in [
        ("untied", "tie_word_embeddings"),
        ("tied", "no_rope_layers"),
        ("tied", "layer_types"),
    ] {
        let a = CopyArtifact::new(variant);
        let mut config = fixture["config_templates"][variant].clone();
        config.as_object_mut().unwrap().remove(remove);
        fs::write(
            a.0.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        assert!(Source::open_dense(&a.0, options()).is_err());
    }
}

#[test]
fn granite_scalar_graph_and_cache_match_independent_source() {
    let raw = include_bytes!("fixtures/granite-half-graph-v1/fixture.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "852711a8bfffd68f389f5db8a987ff6e4f56dd5e5d747bcbe89cccb089b58dd9"
    );
    let f: Value = serde_json::from_slice(raw).unwrap();
    for variant in ["tied", "untied"] {
        let a = CopyArtifact::new(variant);
        let mut config = f["config_templates"][variant].clone();
        config["max_position_embeddings"] = json!(512);
        fs::write(
            a.0.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let source = Source::open_dense(
            &a.0,
            LoadOptions {
                context_limit: 512,
                ..options()
            },
        )
        .unwrap();
        assert_eq!(source.architecture(), Architecture::Granite);
        assert_eq!(source.dense_scales().attention, Some(0.015625));
        assert_eq!(source.dense_scales().embedding, 12.);
        let loaded = reference::load_dense(
            &a.0,
            LoadOptions {
                context_limit: 512,
                ..options()
            },
        )
        .unwrap();
        for case in f["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["variant"] == variant)
        {
            let ids = tokens(&case["tokens"]);
            let offset = case["position_offset"].as_u64().unwrap() as usize;
            let expected = |name: &str| {
                case["arrays"][name]["cpu_bits"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| f32::from_bits(v.as_u64().unwrap() as u32))
                    .collect::<Vec<_>>()
            };
            let mut scratch = loaded.model.scratch(ids.len()).unwrap();
            let trace = loaded
                .model
                .forward_diagnostic(&ids, offset, &mut scratch)
                .unwrap();
            assert_eq!(values(&trace).len(), 39);
            for (name, actual) in values(&trace) {
                close(actual, &expected(&name));
            }
            let logits = expected("logits");
            for chunk in [1, 3] {
                let mut s = Session::new(
                    &loaded.model,
                    session::Options {
                        capacity: 512 - offset,
                        prefill_chunk_size: chunk,
                        position_offset: offset,
                        memory_limit_bytes: 128 * 1024 * 1024,
                    },
                )
                .unwrap();
                s.append(&ids[..2]).unwrap();
                s.evaluate().unwrap();
                s.append(&ids[2..]).unwrap();
                s.evaluate().unwrap();
                close(s.logits().unwrap(), &logits[logits.len() - 64..]);
                s.reset();
                s.set_tokens(&ids).unwrap();
                s.evaluate().unwrap();
                close(s.logits().unwrap(), &logits[logits.len() - 64..]);
            }
        }
    }
}

#[test]
fn granite_requires_valid_explicit_scales() {
    let f: Value = serde_json::from_slice(include_bytes!(
        "fixtures/granite-half-graph-v1/fixture.json"
    ))
    .unwrap();
    for name in [
        "embedding_multiplier",
        "attention_multiplier",
        "residual_multiplier",
        "logits_scaling",
    ] {
        for value in [
            json!(null),
            json!(0),
            json!(-1),
            json!(true),
            json!(1e99),
            json!(1e-99),
        ] {
            let a = CopyArtifact::new("tied");
            fs::write(
                a.0.join("config.json"),
                serde_json::to_vec(&f["config_templates"]["tied"]).unwrap(),
            )
            .unwrap();
            a.config(name, value);
            assert!(
                Source::open_dense(
                    &a.0,
                    LoadOptions {
                        context_limit: 512,
                        ..options()
                    }
                )
                .is_err(),
                "{name}"
            );
        }
    }
    assert_eq!(f["scale_omissions_detected"].as_array().unwrap().len(), 5);
    assert_ne!(
        f["residual_witness"]["expected"],
        f["residual_witness"]["wrong_fused"]
    );
    assert_ne!(
        f["coefficient_witness"]["expected"],
        f["coefficient_witness"]["wrong_fp32_coefficient"]
    );
}
