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

fn artifact() -> PathBuf {
    root().parent().unwrap().join("mlx-mistral-v1/untied")
}
#[test]
fn mistral_annotation_matches_every_frozen_untied_scalar_stage() {
    let m = manifest();
    let source = Source::open_dense(&artifact(), options()).unwrap();
    assert_eq!(source.architecture(), Architecture::Mistral);
    assert!(!source.tied_output());
    assert!(source.tensors().keys().all(|name| !name.ends_with(".bias")));
    assert!(Source::open(&artifact(), options()).is_err());
    assert!(reference::load_qwen(&artifact(), options()).is_err());
    let loaded = reference::load_dense(&artifact(), options()).unwrap();
    let mut annotation: Value =
        serde_json::from_slice(&fs::read(artifact().join("config.json")).unwrap()).unwrap();
    assert_eq!(
        annotation.as_object_mut().unwrap().remove("sliding_window"),
        Some(Value::Null)
    );
    annotation["model_type"] = json!("llama");
    annotation["architectures"] = json!(["LlamaForCausalLM"]);
    let original: Value =
        serde_json::from_slice(&fs::read(root().join("untied/config.json")).unwrap()).unwrap();
    assert_eq!(annotation, original);
    let mut compared = 0;
    for (case, spec) in m["variants"]["untied"]["cases"].as_object().unwrap() {
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
                &array(&m, &format!("untied.fp32.cases.{case}.{stage}")),
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 195);
    // This fixture changes annotations only; its packed numerical source is
    // exactly the already frozen independent untied graph.
    assert_eq!(
        fs::read(artifact().join("model.safetensors")).unwrap(),
        fs::read(root().join("untied/model.safetensors")).unwrap()
    );
}
struct CopyArtifact(PathBuf);
impl CopyArtifact {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let folder = std::env::temp_dir().join(format!(
            "toklet-mistral-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&folder).unwrap();
        for name in ["config.json", "model.safetensors"] {
            fs::copy(artifact().join(name), folder.join(name)).unwrap();
        }
        Self(folder)
    }
    fn config(&self, key: &str, value: Value) {
        let path = self.0.join("config.json");
        let mut c: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        c[key] = value;
        fs::write(path, serde_json::to_vec(&c).unwrap()).unwrap();
    }
}
impl Drop for CopyArtifact {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
#[test]
fn mistral_rejects_unqualified_semantics_and_llama_only_metadata() {
    for (key, value) in [
        ("model_type", json!("llama")),
        ("architectures", json!(["LlamaForCausalLM"])),
        ("attention_bias", json!(true)),
        ("mlp_bias", json!(true)),
        ("hidden_act", json!("gelu")),
        ("rope_scaling", json!({"factor":2})),
        ("rope_interleaved", json!(true)),
        ("sliding_window", json!(4096)),
        ("use_sliding_window", json!(true)),
        ("is_llama_config", json!(true)),
        ("pretraining_tp", json!(1)),
        (
            "transformers.js_config",
            json!({"kv_cache_dtype":{"q4f16":"float16","fp16":"float16"}}),
        ),
        ("head_dim", json!(16)),
        ("num_key_value_heads", json!(3)),
        ("num_attention_heads", json!(0)),
        ("num_local_experts", json!(8)),
        ("attention_dropout", json!(0.1)),
        ("rope_theta", json!(0)),
    ] {
        let a = CopyArtifact::new();
        a.config(key, value);
        assert!(Source::open_dense(&a.0, options()).is_err(), "{key}");
    }
    for key in [
        "is_llama_config",
        "transformers.js_config",
        "pretraining_tp",
    ] {
        let a = CopyArtifact::new();
        a.config(key, Value::Null);
        assert!(Source::open_dense(&a.0, options()).is_err(), "null {key}");
    }
    let absent = CopyArtifact::new();
    let path = absent.0.join("config.json");
    let mut c: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    c.as_object_mut().unwrap().remove("sliding_window");
    fs::write(path, serde_json::to_vec(&c).unwrap()).unwrap();
    assert!(Source::open_dense(&absent.0, options()).is_err());
    for value in [json!(false), json!(0), json!([]), json!({})] {
        let a = CopyArtifact::new();
        a.config("sliding_window", value);
        assert!(Source::open_dense(&a.0, options()).is_err());
    }
    let a = CopyArtifact::new();
    a.config("sliding_window", Value::Null);
    assert!(Source::open_dense(&a.0, options()).is_ok());
}
#[test]
fn explicit_mistral_projection_bias_rejects_even_when_zero() {
    let a = CopyArtifact::new();
    let bytes = fs::read(a.0.join("model.safetensors")).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
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
#[ignore = "Explicit immutable real Mistral metadata; reject full FP32 expansion before allocation"]
fn real_mistral_diagnostic_expansion_rejects_the_six_gib_ceiling() {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/MLX-Mistral-7B-Instruct-v0.3-4bit-a4b8f87");
    let options = LoadOptions {
        context_limit: 512,
        memory_limit_bytes: 6 * 1024 * 1024 * 1024,
    };
    let source = Source::open_dense(&directory, options).unwrap();
    assert_eq!(source.architecture(), Architecture::Mistral);
    assert!(
        matches!(reference::load_dense(&directory,options),Err(Error::Budget{required,limit}) if required>limit && limit==6*1024*1024*1024)
    );
}
