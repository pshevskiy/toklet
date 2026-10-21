use half::{bf16, f16};
use safetensors::{Dtype, tensor::TensorView};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use toklet::{
    mlx::{
        affine::{Matrix, MetadataDtype},
        qwen::{Source, Storage},
    },
    weights::{Error, LoadOptions},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
#[derive(Clone)]
struct Tensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}
struct Artifact {
    root: PathBuf,
    config: Value,
    tensors: BTreeMap<String, Tensor>,
}
impl Artifact {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "toklet-mlx-qwen-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let config = json!({"model_type":"qwen2", "architectures":["Qwen2ForCausalLM"], "hidden_act":"silu", "hidden_size":32,
            "intermediate_size":64,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"vocab_size":64,
            "max_position_embeddings":32,"rms_norm_eps":1e-6,"rope_theta":1e6,"tie_word_embeddings":true,"torch_dtype":"bfloat16",
            "quantization":{"bits":4,"group_size":32}});
        let mut tensors = BTreeMap::new();
        let mut float = |name: String, shape: Vec<usize>| {
            let data = (0..shape.iter().product::<usize>())
                .flat_map(|i| ((i % 31) as f32 / 32.0).to_le_bytes())
                .collect();
            tensors.insert(
                name,
                Tensor {
                    dtype: Dtype::F32,
                    shape,
                    data,
                },
            );
        };
        float("model.embed_tokens.weight".into(), vec![64, 32]);
        float("model.norm.weight".into(), vec![32]);
        for l in 0..2 {
            for norm in ["input_layernorm", "post_attention_layernorm"] {
                float(format!("model.layers.{l}.{norm}.weight"), vec![32]);
            }
            for (name, rows, columns) in [
                ("q", 32, 32),
                ("k", 16, 32),
                ("v", 16, 32),
                ("o", 32, 32),
                ("gate", 64, 32),
                ("up", 64, 32),
                ("down", 32, 64),
            ] {
                let block = if ["q", "k", "v", "o"].contains(&name) {
                    "self_attn"
                } else {
                    "mlp"
                };
                float(
                    format!("model.layers.{l}.{block}.{name}_proj.weight"),
                    vec![rows, columns],
                );
                if ["q", "k", "v"].contains(&name) {
                    float(
                        format!("model.layers.{l}.{block}.{name}_proj.bias"),
                        vec![rows],
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
    fn pack(&mut self, module: &str, bits: usize, group: usize, dtype: Dtype) {
        let name = format!("{module}.weight");
        let shape = self.tensors[&name].shape.clone();
        let (rows, columns) = (shape[0], shape[1]);
        // Literal low-nibble/high-nibble sequence 0,1,2,...,7 repeated for Q4.
        // Other widths exercise importer layouts, not a second quantizer oracle.
        let data = if bits == 4 {
            [0x10, 0x32, 0x54, 0x76]
                .into_iter()
                .cycle()
                .take(rows * columns / 2)
                .collect()
        } else {
            vec![0xee; rows * columns * bits / 8]
        };
        self.tensors.insert(
            name,
            Tensor {
                dtype: Dtype::U32,
                shape: vec![rows, columns * bits / 32],
                data,
            },
        );
        for (suffix, value) in [("scales", 0.125f32), ("biases", -1.0f32)] {
            let raw = match dtype {
                Dtype::F16 => f16::from_f32(value).to_le_bytes().to_vec(),
                Dtype::BF16 => bf16::from_f32(value).to_le_bytes().to_vec(),
                Dtype::F32 => value.to_le_bytes().to_vec(),
                _ => unreachable!(),
            };
            let data = raw
                .iter()
                .copied()
                .cycle()
                .take(rows * columns / group * raw.len())
                .collect();
            self.tensors.insert(
                format!("{module}.{suffix}"),
                Tensor {
                    dtype,
                    shape: vec![rows, columns / group],
                    data,
                },
            );
        }
    }
    fn pack_all(&mut self) {
        let modules: Vec<_> = self
            .tensors
            .iter()
            .filter(|(_, t)| t.shape.len() == 2)
            .map(|(n, _)| n.strip_suffix(".weight").unwrap().to_owned())
            .collect();
        for module in modules {
            self.pack(&module, 4, 32, Dtype::F16);
        }
    }
    fn write(&self, sharded: bool) {
        fs::write(
            self.root.join("config.json"),
            serde_json::to_vec(&self.config).unwrap(),
        )
        .unwrap();
        let groups = if sharded { 2 } else { 1 };
        let mut index = BTreeMap::new();
        for g in 0..groups {
            let file = if sharded {
                format!("model-{g}.safetensors")
            } else {
                "model.safetensors".into()
            };
            let views: Vec<_> = self
                .tensors
                .iter()
                .enumerate()
                .filter(|(i, _)| i % groups == g)
                .map(|(_, (n, t))| {
                    index.insert(n.clone(), file.clone());
                    (
                        n.as_str(),
                        TensorView::new(t.dtype, t.shape.clone(), &t.data).unwrap(),
                    )
                })
                .collect();
            safetensors::serialize_to_file(views, None, &self.root.join(file)).unwrap();
        }
        if sharded {
            fs::write(self.root.join("model.safetensors.index.json"),serde_json::to_vec(&json!({"metadata":{"total_size":self.tensors.values().map(|t|t.data.len()).sum::<usize>()},"weight_map":index})).unwrap()).unwrap();
        }
    }
    fn open(&self) -> Result<Source, Error> {
        Source::open(
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
fn failure(a: &Artifact, expected: &str) {
    let error = match a.open() {
        Ok(_) => panic!("accepted invalid artifact"),
        Err(e) => e.to_string(),
    };
    assert!(
        error.contains(expected),
        "{error:?} did not contain {expected:?}"
    );
}

#[test]
fn mixed_single_and_sharded_payloads_stream_without_expansion() {
    for sharded in [false, true] {
        let mut a = Artifact::new();
        a.pack_all();
        // Retain one floating projection and override one packed projection.
        let excluded = "model.layers.0.self_attn.o_proj";
        let original = Artifact::new();
        for suffix in ["scales", "biases"] {
            a.tensors.remove(&format!("{excluded}.{suffix}"));
        }
        a.tensors.insert(
            format!("{excluded}.weight"),
            original.tensors[&format!("{excluded}.weight")].clone(),
        );
        a.config["quantization"][excluded] = json!(false);
        let overridden = "model.layers.1.mlp.down_proj";
        a.tensors.insert(
            format!("{overridden}.weight"),
            original.tensors[&format!("{overridden}.weight")].clone(),
        );
        a.pack(overridden, 8, 64, Dtype::BF16);
        a.config["quantization"][overridden] = json!({"bits":8,"group_size":64,"mode":"affine"});
        a.write(sharded);
        let mut source = a.open().unwrap();
        assert_eq!(source.config().num_layers, 2);
        assert_eq!(source.tensors().len(), 26);
        let payload: usize = a.tensors.values().map(|t| t.data.len()).sum();
        assert_eq!(source.tensor_payload_bytes(), payload);
        assert_eq!(
            source.admitted_load_bytes(),
            payload + 64 * 1024 * 1024 + 64 * 1024
        );
        let specs = source.tensors().clone();
        for (name, spec) in specs {
            match spec.storage() {
                Storage::Float(_) => {
                    let mut actual = Vec::new();
                    source
                        .visit_float(&name, |v| {
                            actual.extend_from_slice(&v.to_le_bytes());
                            Ok(())
                        })
                        .unwrap();
                    assert_eq!(actual, a.tensors[&name].data);
                }
                Storage::Affine(layout) => {
                    let mut w = vec![0; layout.weight_bytes()];
                    let mut s = vec![0; layout.metadata_bytes()];
                    let mut b = s.clone();
                    assert_eq!(
                        source.read_affine(&name, &mut w, &mut s, &mut b).unwrap(),
                        layout
                    );
                    let module = name.strip_suffix(".weight").unwrap();
                    assert_eq!(w, a.tensors[&name].data);
                    assert_eq!(s, a.tensors[&format!("{module}.scales")].data);
                    assert_eq!(b, a.tensors[&format!("{module}.biases")].data);
                    if layout.bits() == 4 {
                        let matrix = Matrix::from_le_bytes(layout, &w, &s, &b).unwrap();
                        for r in 0..layout.rows() {
                            for c in 0..layout.columns() {
                                assert_eq!(matrix.value(r, c).unwrap(), (c % 8) as f32 / 8.0 - 1.0);
                            }
                        }
                    } else {
                        assert_eq!(layout.metadata_dtype(), MetadataDtype::BF16);
                    }
                }
            }
        }
        // No accidental reinterpretation of the existing unquantized loader.
        assert!(
            toklet::weights::load_qwen(
                &a.root,
                LoadOptions {
                    context_limit: 32,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
}

#[test]
fn all_widths_and_metadata_dtypes_resolve_at_loading() {
    for bits in [2, 3, 4, 5, 6, 8] {
        for (dtype, expected) in [
            (Dtype::F16, MetadataDtype::F16),
            (Dtype::BF16, MetadataDtype::BF16),
            (Dtype::F32, MetadataDtype::F32),
        ] {
            let mut a = Artifact::new();
            a.config["quantization"]["bits"] = json!(bits);
            a.pack("model.embed_tokens", bits, 32, dtype);
            a.write(false);
            let source = a.open().unwrap();
            let Storage::Affine(l) = source.tensors()["model.embed_tokens.weight"].storage() else {
                panic!()
            };
            assert_eq!(l.bits() as usize, bits);
            assert_eq!(l.metadata_dtype(), expected);
            // Global defaults do not silently quantize tensors without .scales.
            assert!(matches!(
                source.tensors()["model.layers.0.self_attn.q_proj.weight"].storage(),
                Storage::Float(_)
            ));
        }
    }
}

#[test]
fn quantization_aliases_and_exact_tied_head_ownership() {
    for packed in [false, true] {
        let mut a = Artifact::new();
        if packed {
            a.pack_all();
        }
        for suffix in if packed {
            vec!["weight", "scales", "biases"]
        } else {
            vec!["weight"]
        } {
            a.tensors.insert(
                format!("lm_head.{suffix}"),
                a.tensors[&format!("model.embed_tokens.{suffix}")].clone(),
            );
        }
        a.config["quantization_config"] = a.config["quantization"].clone();
        a.write(true);
        let source = a.open().unwrap();
        assert!(!source.tensors().contains_key("lm_head.weight"));
        let expected: usize = a
            .tensors
            .iter()
            .filter(|(n, _)| !n.starts_with("lm_head."))
            .map(|(_, t)| t.data.len())
            .sum();
        assert_eq!(source.tensor_payload_bytes(), expected);
        drop(source);
        a.config.as_object_mut().unwrap().remove("quantization");
        a.write(true);
        assert!(a.open().is_ok());
        a.tensors.get_mut("lm_head.weight").unwrap().data[0] ^= 1;
        a.write(true);
        failure(&a, "differs from embedding bytes");
    }
}

#[test]
fn rejects_ambiguous_or_unsupported_config_semantics() {
    for (field, value, expected) in [
        ("model_type", json!("llama"), "only tied"),
        ("tie_word_embeddings", json!(false), "only tied"),
        (
            "rope_scaling",
            json!({"type":"linear","factor":2}),
            "only tied",
        ),
        ("use_sliding_window", json!(true), "only tied"),
        ("num_hidden_layers", json!(100000), "8192 logical"),
        (
            "unrecognized_graph_option",
            json!(true),
            "unsupported config field",
        ),
        (
            "quantization_config",
            json!({"bits":8,"group_size":32}),
            "conflicting",
        ),
        (
            "quantization",
            json!({"bits":4,"group_size":32,"mode":"mxfp4"}),
            "only MLX affine",
        ),
        (
            "quantization",
            json!({"bits":7,"group_size":32}),
            "bit width",
        ),
        (
            "quantization",
            json!({"bits":4,"group_size":16}),
            "group size",
        ),
        (
            "quantization",
            json!({"bits":4,"group_size":32,"model.layers.9.mlp.up_proj":false}),
            "unknown MLX",
        ),
        (
            "quantization",
            json!({"bits":4,"group_size":32,"model.embed_tokens":true}),
            "invalid MLX layer",
        ),
        (
            "quantization",
            json!({"bits":4,"group_size":32,"model.embed_tokens":{"bits":8}}),
            "group_size",
        ),
        ("quantization", Value::Null, "must be an object"),
    ] {
        let mut a = Artifact::new();
        a.config[field] = value;
        a.write(false);
        failure(&a, expected);
    }
    let a = Artifact::new();
    a.write(false);
    fs::write(
        a.root.join("config.json"),
        "{\"quantization\":{\"bits\":4,\"bits\":8},\"quantization\":{}}",
    )
    .unwrap();
    failure(&a, "duplicate MLX config key");
    fs::write(
        a.root.join("config.json"),
        "{\"quantization\":{\"bits\":4,\"bits\":8}}",
    )
    .unwrap();
    failure(&a, "duplicate MLX config key");
    fs::write(a.root.join("config.json"), vec![b' '; 1024 * 1024 + 1]).unwrap();
    failure(&a, "exceeds 1 MiB");
}

#[test]
fn rejects_inconsistent_packed_tensors_and_exclusions() {
    for case in 0..7 {
        let mut a = Artifact::new();
        a.pack_all();
        match case {
            0 => {
                a.tensors.remove("model.embed_tokens.biases");
            }
            1 => {
                a.tensors
                    .get_mut("model.embed_tokens.scales")
                    .unwrap()
                    .dtype = Dtype::BF16;
            }
            2 => {
                a.tensors
                    .get_mut("model.embed_tokens.weight")
                    .unwrap()
                    .dtype = Dtype::I32;
            }
            3 => {
                a.tensors
                    .get_mut("model.embed_tokens.weight")
                    .unwrap()
                    .shape = vec![32, 8];
            }
            4 => {
                a.config["quantization"]["model.embed_tokens"] = json!(false);
            }
            5 => {
                a.tensors.remove("model.embed_tokens.scales");
            }
            6 => {
                a.tensors.insert(
                    "model.embed_tokens.bias".into(),
                    Tensor {
                        dtype: Dtype::F32,
                        shape: vec![64],
                        data: vec![0; 256],
                    },
                );
            }
            _ => unreachable!(),
        }
        a.write(false);
        assert!(a.open().is_err(), "case {case}");
    }
}

#[test]
fn payload_validation_rejects_nonfinite_and_partial_or_invalid_destinations() {
    let mut a = Artifact::new();
    a.pack_all();
    a.tensors.get_mut("model.embed_tokens.scales").unwrap().data[..2]
        .copy_from_slice(&f16::NAN.to_le_bytes());
    a.tensors.get_mut("model.norm.weight").unwrap().data[..4]
        .copy_from_slice(&f32::INFINITY.to_le_bytes());
    a.write(false);
    let mut source = a.open().unwrap();
    let Storage::Affine(l) = source.tensors()["model.embed_tokens.weight"].storage() else {
        panic!()
    };
    let mut w = vec![0x55; l.weight_bytes()];
    let mut s = vec![0x55; l.metadata_bytes()];
    let mut b = s.clone();
    assert!(
        source
            .read_affine("model.embed_tokens.weight", &mut w[..1], &mut s, &mut b)
            .is_err()
    );
    assert!(w.iter().all(|&v| v == 0x55));
    assert!(s.iter().all(|&v| v == 0x55));
    assert!(
        source
            .read_affine("model.embed_tokens.weight", &mut w, &mut s, &mut b)
            .unwrap_err()
            .to_string()
            .contains("finite")
    );
    assert!(
        source
            .visit_float("model.norm.weight", |_| Ok(()))
            .unwrap_err()
            .to_string()
            .contains("nonfinite")
    );
    assert!(
        source
            .visit_float("model.embed_tokens.weight", |_| Ok(()))
            .is_err()
    );
    assert!(
        source
            .read_affine("model.norm.weight", &mut w, &mut s, &mut b)
            .is_err()
    );
    assert!(source.visit_float("absent", |_| Ok(())).is_err());
    // The same open handle observes a truncation, rather than fabricating bytes.
    fs::OpenOptions::new()
        .write(true)
        .open(a.root.join("model.safetensors"))
        .unwrap()
        .set_len(0)
        .unwrap();
    assert!(
        source
            .read_affine("model.embed_tokens.weight", &mut w, &mut s, &mut b)
            .is_err()
    );
}

#[test]
fn validates_index_mapping_and_budget_before_payload_ownership() {
    let mut a = Artifact::new();
    a.pack_all();
    a.write(true);
    let source = a.open().unwrap();
    let required = source.admitted_load_bytes();
    drop(source);
    assert!(
        matches!(Source::open(&a.root,LoadOptions{context_limit:32,memory_limit_bytes:required-1}),Err(Error::Budget{required:r,..}) if r==required)
    );
    assert!(
        Source::open(
            &a.root,
            LoadOptions {
                context_limit: 32,
                memory_limit_bytes: required
            }
        )
        .is_ok()
    );
    let path = a.root.join("model.safetensors.index.json");
    let original: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    for change in 0..4 {
        let mut index = original.clone();
        match change {
            0 => {
                index["weight_map"]["model.embed_tokens.weight"] = json!("../outside.safetensors");
            }
            1 => {
                index["metadata"]["total_size"] = json!(0);
            }
            2 => {
                index["weight_map"]
                    .as_object_mut()
                    .unwrap()
                    .remove("model.norm.weight");
            }
            3 => {
                index["weight_map"]["absent"] = json!("model-0.safetensors");
            }
            _ => unreachable!(),
        }
        fs::write(&path, serde_json::to_vec(&index).unwrap()).unwrap();
        assert!(a.open().is_err());
    }
}

#[test]
fn tied_raw_streaming_checks_bytes_beyond_the_shared_chunk() {
    let mut a = Artifact::new();
    a.config["vocab_size"] = json!(5000);
    a.tensors.insert(
        "model.embed_tokens.weight".into(),
        Tensor {
            dtype: Dtype::F32,
            shape: vec![5000, 32],
            data: vec![0; 5000 * 32 * 4],
        },
    );
    a.pack("model.embed_tokens", 4, 32, Dtype::F32);
    for suffix in ["weight", "scales", "biases"] {
        a.tensors.insert(
            format!("lm_head.{suffix}"),
            a.tensors[&format!("model.embed_tokens.{suffix}")].clone(),
        );
    }
    a.write(true);
    let mut source = a.open().unwrap();
    let Storage::Affine(l) = source.tensors()["model.embed_tokens.weight"].storage() else {
        panic!()
    };
    assert!(l.weight_bytes() > 64 * 1024);
    let mut w = vec![0; l.weight_bytes()];
    let mut s = vec![0; l.metadata_bytes()];
    let mut b = s.clone();
    source
        .read_affine("model.embed_tokens.weight", &mut w, &mut s, &mut b)
        .unwrap();
    assert_eq!(w, a.tensors["model.embed_tokens.weight"].data);
    drop(source);
    a.tensors.get_mut("lm_head.weight").unwrap().data[70_000] ^= 1;
    a.write(true);
    failure(&a, "differs from embedding bytes");
}
