mod support;

use safetensors::{Dtype, tensor::TensorView};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use support::Fixture;
use toklet::weights::{self, Error, LoadOptions};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Artifact {
    root: PathBuf,
    config: Value,
    tensors: BTreeMap<String, Tensor>,
}
struct Tensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}
impl Artifact {
    fn tiny() -> Self {
        let f = Fixture::tiny();
        let root = std::env::temp_dir().join(format!(
            "toklet-weights-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let config = json!({"model_type":"qwen2", "architectures":["Qwen2ForCausalLM"], "hidden_act":"silu", "hidden_size":32, "intermediate_size":64, "num_hidden_layers":2,
            "num_attention_heads":4, "num_key_value_heads":2, "vocab_size":64, "max_position_embeddings":32, "rms_norm_eps":1e-6, "rope_theta":1e6,
            "tie_word_embeddings":true, "torch_dtype":"float32"});
        let mut tensors = BTreeMap::new();
        let mut insert = |name: String, fixture_name: String| {
            tensors.insert(
                name,
                Tensor {
                    dtype: Dtype::F32,
                    shape: f.shape(&fixture_name),
                    data: f.bytes(&fixture_name),
                },
            );
        };
        insert(
            "model.embed_tokens.weight".into(),
            "weights.embed_tokens".into(),
        );
        insert("model.norm.weight".into(), "weights.norm".into());
        for index in 0..2 {
            for name in ["input_layernorm", "post_attention_layernorm"] {
                insert(
                    format!("model.layers.{index}.{name}.weight"),
                    format!("weights.layers.{index}.{name}"),
                );
            }
            for name in ["q", "k", "v", "o", "gate", "up", "down"] {
                let block = if ["q", "k", "v", "o"].contains(&name) {
                    "self_attn"
                } else {
                    "mlp"
                };
                insert(
                    format!("model.layers.{index}.{block}.{name}_proj.weight"),
                    format!("weights.layers.{index}.{name}_proj.weight"),
                );
                if ["q", "k", "v"].contains(&name) {
                    insert(
                        format!("model.layers.{index}.{block}.{name}_proj.bias"),
                        format!("weights.layers.{index}.{name}_proj.bias"),
                    );
                }
            }
        }
        Self {
            root,
            config,
            tensors,
        }
    }
    fn write(&self, sharded: bool) {
        fs::write(
            self.root.join("config.json"),
            serde_json::to_vec(&self.config).unwrap(),
        )
        .unwrap();
        let mut index = BTreeMap::new();
        let groups = if sharded { 2 } else { 1 };
        for group in 0..groups {
            let file = if sharded {
                format!("model-0000{}.safetensors", group + 1)
            } else {
                "model.safetensors".into()
            };
            let views: Vec<_> = self
                .tensors
                .iter()
                .enumerate()
                .filter(|(ordinal, _)| ordinal % groups == group)
                .map(|(_, (name, tensor))| {
                    index.insert(name.clone(), file.clone());
                    (
                        name.as_str(),
                        TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.data).unwrap(),
                    )
                })
                .collect();
            safetensors::serialize_to_file(views, None, &self.root.join(file)).unwrap();
        }
        if sharded {
            let total: usize = self.tensors.values().map(|tensor| tensor.data.len()).sum();
            fs::write(
                self.root.join("model.safetensors.index.json"),
                serde_json::to_vec(&json!({"metadata":{"total_size":total},"weight_map":index}))
                    .unwrap(),
            )
            .unwrap();
        }
    }
    fn load(&self) -> Result<weights::LoadedModel, Error> {
        weights::load_qwen(
            &self.root,
            LoadOptions {
                context_limit: 32,
                ..Default::default()
            },
        )
    }
}
impl Drop for Artifact {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).unwrap();
    }
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs(),
            "{actual} != {expected}"
        );
    }
}

#[test]
fn single_and_indexed_shards_load_the_same_frozen_tiny_decoder() {
    let f = Fixture::tiny();
    for sharded in [false, true] {
        let artifact = Artifact::tiny();
        artifact.write(sharded);
        let loaded = artifact.load().unwrap();
        assert_eq!(loaded.report.tensor_count, 26);
        assert_eq!(loaded.report.source_dtypes, ["F32"]);
        assert_eq!(
            loaded.report.source_tensor_bytes,
            loaded.report.fp32_weights_bytes
        );
        assert_eq!(loaded.report.conversion_buffer_bytes, 65536);
        let mut scratch = loaded.model.scratch(4).unwrap();
        let trace = loaded
            .model
            .forward_diagnostic(&f.u32("cases.prompt.tokens"), 0, &mut scratch)
            .unwrap();
        close(&trace.logits, &f.f32("cases.prompt.logits"));
    }
}

#[test]
fn explicit_fp16_and_bf16_little_endian_values_convert_exactly() {
    for (dtype, bits) in [
        (Dtype::F16, [0x3c00_u16, 0xc000, 0x3800, 0x8000]),
        (Dtype::BF16, [0x3f80, 0xc000, 0x3f00, 0x8000]),
    ] {
        let mut artifact = Artifact::tiny();
        artifact.config["torch_dtype"] = json!(if dtype == Dtype::F16 {
            "float16"
        } else {
            "bfloat16"
        });
        for tensor in artifact.tensors.values_mut() {
            let elements = tensor.data.len() / 4;
            tensor.dtype = dtype;
            tensor.data = vec![0; elements * 2];
        }
        let embedding = artifact
            .tensors
            .get_mut("model.embed_tokens.weight")
            .unwrap();
        for (destination, bits) in embedding.data[..8].chunks_exact_mut(2).zip(bits) {
            destination.copy_from_slice(&bits.to_le_bytes());
        }
        artifact.write(false);
        let loaded = artifact.load().unwrap();
        assert_eq!(loaded.report.source_dtypes, [format!("{dtype:?}")]);
        assert_eq!(
            loaded.report.fp32_weights_bytes,
            2 * loaded.report.source_tensor_bytes
        );
        let mut scratch = loaded.model.scratch(1).unwrap();
        let trace = loaded
            .model
            .forward_diagnostic(&[0], 0, &mut scratch)
            .unwrap();
        assert_eq!(&trace.embedding[..4], &[1.0, -2.0, 0.5, -0.0]);
        assert!(trace.embedding[3].is_sign_negative());
    }
}

#[test]
fn conversion_crosses_chunk_boundaries_and_preserves_the_short_tail() {
    for dtype in [Dtype::F32, Dtype::F16, Dtype::BF16] {
        let mut artifact = Artifact::tiny();
        artifact.config["vocab_size"] = json!(1025);
        for tensor in artifact.tensors.values_mut() {
            tensor.data.fill(0);
        }
        let embedding = artifact
            .tensors
            .get_mut("model.embed_tokens.weight")
            .unwrap();
        embedding.shape = vec![1025, 32];
        embedding.dtype = dtype;
        let width = if dtype == Dtype::F32 { 4 } else { 2 };
        embedding.data = vec![0; 1025 * 32 * width];
        let values: Vec<u8> = match dtype {
            Dtype::F32 => [1.0_f32, -2.0, 0.5, 4.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
            Dtype::F16 => [0x3c00_u16, 0xc000, 0x3800, 0x4400]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
            _ => [0x3f80_u16, 0xc000, 0x3f00, 0x4080]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        };
        let start = 1024 * 32 * width;
        embedding.data[start..start + values.len()].copy_from_slice(&values);
        artifact.write(true);
        let loaded = artifact.load().unwrap();
        let mut scratch = loaded.model.scratch(1).unwrap();
        let trace = loaded
            .model
            .forward_diagnostic(&[1024], 0, &mut scratch)
            .unwrap();
        assert_eq!(&trace.embedding[..4], &[1.0, -2.0, 0.5, 4.0]);
    }
}

#[test]
fn tied_output_is_checked_and_does_not_add_persistent_weights() {
    let mut artifact = Artifact::tiny();
    artifact.write(false);
    let original = artifact.load().unwrap().report.fp32_weights_bytes;
    let embedding = &artifact.tensors["model.embed_tokens.weight"];
    artifact.tensors.insert(
        "lm_head.weight".into(),
        Tensor {
            dtype: embedding.dtype,
            shape: embedding.shape.clone(),
            data: embedding.data.clone(),
        },
    );
    artifact.write(false);
    let loaded = artifact.load().unwrap();
    assert_eq!(loaded.report.fp32_weights_bytes, original);
    assert_eq!(loaded.report.tensor_count, 27);
    artifact.tensors.get_mut("lm_head.weight").unwrap().data[..4]
        .copy_from_slice(&42.0_f32.to_le_bytes());
    artifact.write(false);
    assert!(
        artifact
            .load()
            .unwrap_err_message()
            .contains("does not equal tied")
    );
}

// Preserve useful error text without requiring Debug on the owned numerical model.
trait ErrorMessage {
    fn unwrap_err_message(self) -> String;
}
impl ErrorMessage for Result<weights::LoadedModel, Error> {
    fn unwrap_err_message(self) -> String {
        match self {
            Err(error) => error.to_string(),
            Ok(_) => panic!("expected import failure"),
        }
    }
}

#[test]
fn unsupported_configuration_and_insufficient_budget_fail_before_loading() {
    for (key, value) in [
        ("tie_word_embeddings", json!(false)),
        ("rope_scaling", json!({"type":"linear","factor":2})),
        ("use_sliding_window", json!(true)),
        ("hidden_act", json!("gelu")),
        ("quantization_config", json!({"bits":4})),
        ("attention_bias", json!(false)),
        ("head_dim", json!(9)),
        ("hidden_size", json!(31)),
        ("new_rope_mechanism", json!(true)),
    ] {
        let mut artifact = Artifact::tiny();
        artifact.config[key] = value;
        artifact.write(false);
        assert!(artifact.load().is_err(), "accepted {key}");
    }
    let artifact = Artifact::tiny();
    artifact.write(false);
    assert!(matches!(
        weights::load_qwen(
            &artifact.root,
            LoadOptions {
                context_limit: 32,
                memory_limit_bytes: 1
            }
        ),
        Err(Error::Budget { .. })
    ));
    assert!(weights::load_qwen(&artifact.root, LoadOptions::default()).is_err());
}

#[test]
fn tensor_shape_dtype_names_nonfinite_and_shard_mapping_are_validated() {
    for variant in 0..5 {
        let mut artifact = Artifact::tiny();
        match variant {
            0 => {
                artifact
                    .tensors
                    .remove("model.layers.1.self_attn.q_proj.bias");
            }
            1 => {
                artifact
                    .tensors
                    .get_mut("model.embed_tokens.weight")
                    .unwrap()
                    .shape = vec![32, 64]
            }
            2 => {
                artifact
                    .tensors
                    .get_mut("model.embed_tokens.weight")
                    .unwrap()
                    .dtype = Dtype::I32
            }
            3 => artifact
                .tensors
                .get_mut("model.layers.1.mlp.down_proj.weight")
                .unwrap()
                .data[..4]
                .copy_from_slice(&f32::NAN.to_le_bytes()),
            _ => {
                artifact.tensors.insert(
                    "extra.weight".into(),
                    Tensor {
                        dtype: Dtype::F32,
                        shape: vec![1],
                        data: 1.0_f32.to_le_bytes().to_vec(),
                    },
                );
            }
        }
        artifact.write(false);
        assert!(artifact.load().is_err(), "accepted variant {variant}");
    }
    let artifact = Artifact::tiny();
    artifact.write(true);
    let path = artifact.root.join("model.safetensors.index.json");
    let mut index: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    index["weight_map"]["model.embed_tokens.weight"] = json!("../outside.safetensors");
    fs::write(path, serde_json::to_vec(&index).unwrap()).unwrap();
    assert!(artifact.load().unwrap_err_message().contains("plain local"));
}

#[test]
fn malformed_and_duplicate_metadata_are_rejected() {
    let artifact = Artifact::tiny();
    artifact.write(false);
    let path = artifact.root.join("model.safetensors");
    let original = fs::read(&path).unwrap();
    fs::write(&path, &original[..original.len() - 1]).unwrap();
    assert!(artifact.load().is_err());
    let header = br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut duplicate = (header.len() as u64).to_le_bytes().to_vec();
    duplicate.extend_from_slice(header);
    duplicate.extend_from_slice(&[0; 4]);
    fs::write(path, duplicate).unwrap();
    assert!(
        artifact
            .load()
            .unwrap_err_message()
            .contains("duplicate tensor name")
    );
    artifact.write(true);
    fs::write(
        artifact.root.join("model.safetensors.index.json"),
        br#"{"weight_map":{"a":"x.safetensors","a":"y.safetensors"}}"#,
    )
    .unwrap();
    assert!(
        artifact
            .load()
            .unwrap_err_message()
            .contains("duplicate index tensor")
    );
}

#[test]
#[ignore = "requires the explicitly downloaded pinned Qwen checkpoint"]
fn pinned_qwen_loads_within_the_initial_budget() {
    use std::io::Read;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/Qwen2.5-0.5B-Instruct");
    let mut file = fs::File::open(root.join("model.safetensors")).unwrap();
    let mut digest = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).unwrap();
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    assert_eq!(
        format!("{:x}", digest.finalize()),
        "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
    );
    drop(buffer);
    let loaded = weights::load_qwen(&root, LoadOptions::default()).unwrap();
    assert_eq!(loaded.model.config().num_layers, 24);
    assert_eq!(loaded.model.config().max_positions, 2048);
    assert_eq!(loaded.report.tensor_count, 290);
    assert_eq!(loaded.report.source_dtypes, ["BF16"]);
    assert_eq!(loaded.report.source_tensor_bytes, 988065536);
    assert_eq!(loaded.report.fp32_weights_bytes, 1976131072);
    assert!(loaded.report.admitted_load_bytes <= weights::INITIAL_MEMORY_LIMIT_BYTES);
    eprintln!(
        "Pinned Qwen: {} tensors; source {} bytes; FP32 {} bytes; conversion {} bytes; admitted estimate {} bytes; load {:.6}s",
        loaded.report.tensor_count,
        loaded.report.source_tensor_bytes,
        loaded.report.fp32_weights_bytes,
        loaded.report.conversion_buffer_bytes,
        loaded.report.admitted_load_bytes,
        loaded.report.load_seconds
    );
    #[cfg(target_os = "macos")]
    {
        let peak_path =
            std::env::temp_dir().join(format!("toklet-load-footprint-{}.json", std::process::id()));
        let output = std::process::Command::new("/usr/bin/footprint")
            .args([
                "-p",
                &std::process::id().to_string(),
                "-f",
                "bytes",
                "--noCategories",
                "-j",
            ])
            .arg(&peak_path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let record: Value = serde_json::from_slice(&fs::read(&peak_path).unwrap()).unwrap();
        fs::remove_file(peak_path).unwrap();
        assert_eq!(record["unit"], "byte");
        let peak = record["processes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["pid"] == std::process::id())
            .unwrap()["auxiliary"]["phys_footprint_peak"]
            .as_u64()
            .unwrap();
        assert!(peak <= weights::INITIAL_MEMORY_LIMIT_BYTES as u64);
        eprintln!(
            "Own-process physical load peak: {peak} bytes (includes test/hash setup; source hash warms file cache)"
        );
    }
}
