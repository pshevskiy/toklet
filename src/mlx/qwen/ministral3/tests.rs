use super::*;
use crate::weights::range_tests::Fixture;
use serde_json::json;

fn options() -> LoadOptions {
    LoadOptions {
        context_limit: 512,
        memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES,
    }
}

fn document() -> Value {
    let raw =
        include_bytes!("../../../../tests/fixtures/ministral3-config-v1/ministral_modified.json");
    let mut document: Value = serde_json::from_slice(raw).unwrap();
    let mut recipe = json!({"bits":4,"group_size":64,"mode":"affine"});
    recipe["language_model.model.embed_tokens"] = json!({"bits":8,"group_size":64,"mode":"affine"});
    for layer in 0..26 {
        recipe[format!("language_model.model.layers.{layer}.self_attn.o_proj")] =
            json!({"bits":8,"group_size":64,"mode":"affine"});
        recipe[format!("language_model.model.layers.{layer}.mlp.down_proj")] =
            json!({"bits":8,"group_size":64,"mode":"affine"});
    }
    document["quantization"] = recipe;
    document
}

#[test]
fn fixed_plan_keeps_distinct_widths_exact_storage_and_single_embedding() {
    let planned = plan(document(), options()).unwrap();
    let c = &planned.config;
    assert_eq!(
        (
            c.hidden_size,
            c.query_heads * c.head_dim,
            c.kv_heads * c.head_dim
        ),
        (3072, 4096, 1024)
    );
    assert_eq!(
        (
            c.num_layers,
            c.intermediate_size,
            c.vocab_size,
            c.max_positions
        ),
        (26, 9216, 131072, 512)
    );
    assert_eq!((c.rms_norm_eps, c.rope_theta), (1e-5, 1e6));
    assert_eq!(planned.profile.geometry().source_max_positions, 262144);
    assert_eq!(planned.physical.len(), 602);
    assert_eq!(planned.tensors.len(), 236);
    assert_eq!(
        planned.admitted_load_bytes,
        PAYLOAD_BYTES + METADATA_ALLOWANCE + STREAM_BYTES
    );
    let mut bits4 = 0;
    let mut bits8 = 0;
    let mut norms = 0;
    for (name, tensor) in &planned.tensors {
        assert!(name.starts_with("language_model.model."));
        match tensor.storage() {
            Storage::Affine(packed) => {
                assert_eq!(packed.group_size(), 64);
                assert_eq!(packed.metadata_dtype(), MetadataDtype::BF16);
                if packed.bits() == 8 {
                    assert!(
                        name == "language_model.model.embed_tokens.weight"
                            || name.ends_with(".self_attn.o_proj.weight")
                            || name.ends_with(".mlp.down_proj.weight")
                    );
                    bits8 += 1;
                } else {
                    assert_eq!(packed.bits(), 4);
                    bits4 += 1;
                }
            }
            Storage::Float(dtype) => {
                assert_eq!(dtype, MetadataDtype::BF16);
                assert_eq!(tensor.shape(), [3072]);
                norms += 1;
            }
        }
    }
    assert_eq!((bits4, bits8, norms), (130, 53, 53));
    assert_eq!(
        planned
            .tensors
            .values()
            .map(TensorSpec::payload_bytes)
            .sum::<usize>(),
        PAYLOAD_BYTES
    );
    assert!(
        !planned
            .physical
            .keys()
            .any(|name| name.contains("lm_head") || name.starts_with("vision_"))
    );
    assert_eq!(
        planned.physical["language_model.model.layers.0.self_attn.q_proj.weight"].shape,
        [4096, 384]
    );
    assert_eq!(
        planned.physical["language_model.model.layers.0.self_attn.o_proj.weight"].shape,
        [3072, 1024]
    );
}

#[test]
fn exact_recipe_rejects_aliases_omissions_duplicates_and_semantic_equivalents() {
    for change in 0..9 {
        let mut value = document();
        match change {
            0 => value["quantization"]["bits"] = json!(8),
            1 => value["quantization"]["group_size"] = json!(32),
            2 => value["quantization"]["mode"] = json!("mxfp4"),
            3 => value["quantization"]["language_model.model.embed_tokens"] = json!(false),
            4 => {
                value["quantization"]
                    .as_object_mut()
                    .unwrap()
                    .remove("language_model.model.embed_tokens");
            }
            5 => {
                value["quantization"]["language_model.model.layers.0.self_attn.q_proj"] =
                    json!({"bits":4,"group_size":64,"mode":"affine"});
            }
            6 => value["quantization"]["bits"] = json!(4.0),
            7 => value["quantization_config"] = value["quantization"].clone(),
            _ => {
                let recipe = value
                    .as_object_mut()
                    .unwrap()
                    .remove("quantization")
                    .unwrap();
                value["quantization_config"] = recipe;
            }
        }
        assert!(plan(value, options()).is_err(), "change {change}");
    }
    let mut value = document();
    value.as_object_mut().unwrap().remove("quantization");
    assert!(plan(value, options()).is_err());
    let duplicate = br#"{"quantization":{},"quantization":{}}"#;
    assert!(parse_json(duplicate).is_err());
}

#[test]
fn strict_wrapper_geometry_positions_and_budget_reject_without_weight_allocation() {
    for (name, changed) in [
        ("hidden_size", json!(4096)),
        ("head_dim", json!(96)),
        ("num_attention_heads", json!(24)),
        ("dtype", json!("float16")),
        ("sliding_window", json!(4096)),
        ("tie_word_embeddings", json!(false)),
    ] {
        let mut value = document();
        value["text_config"][name] = changed;
        assert!(plan(value, options()).is_err(), "{name}");
    }
    for context_limit in [0, 513, usize::MAX] {
        let options = LoadOptions {
            context_limit,
            ..options()
        };
        assert!(plan(document(), options).is_err());
        assert!(matches!(
            Source::open_ministral3_diagnostic(Path::new("/nonexistent-ministral-child"), options),
            Err(Error::Invalid(ref message)) if message == "Ministral diagnostic requires1..=512 positions"
        ));
    }
    for memory_limit_bytes in [0, weights::INITIAL_MEMORY_LIMIT_BYTES + 1, usize::MAX] {
        let options = LoadOptions {
            memory_limit_bytes,
            ..options()
        };
        assert!(matches!(plan(document(), options), Err(Error::Invalid(_))));
    }
    let required = PAYLOAD_BYTES + METADATA_ALLOWANCE + STREAM_BYTES;
    assert!(
        matches!(plan(document(), LoadOptions { memory_limit_bytes: required - 1, ..options() }),
                    Err(Error::Budget { required: actual, .. }) if actual == required)
    );
    assert!(
        plan(
            document(),
            LoadOptions {
                memory_limit_bytes: required,
                ..options()
            }
        )
        .is_ok()
    );
}

#[test]
fn all602_virtual_physical_descriptors_are_exact_and_reject_dtype_shape_and_extent_changes() {
    let planned = plan(document(), options()).unwrap();
    validate_names(
        &planned.physical,
        planned.physical.keys().map(String::as_str),
    )
    .unwrap();
    let mut cursor = 0;
    for (name, item) in &planned.physical {
        let mut info = TensorInfo {
            dtype: item.dtype,
            shape: item.shape.clone(),
            data_offsets: (cursor, cursor + item.bytes),
        };
        validate_tensor(name, item, &info).unwrap();
        cursor += item.bytes;
        info.data_offsets.1 += 1;
        assert!(validate_tensor(name, item, &info).is_err());
        info.data_offsets.1 -= 1;
        info.dtype = if item.dtype == Dtype::BF16 {
            Dtype::F16
        } else {
            Dtype::I32
        };
        assert!(validate_tensor(name, item, &info).is_err());
        info.dtype = item.dtype;
        info.shape[0] += 1;
        assert!(validate_tensor(name, item, &info).is_err());
    }
    assert_eq!(cursor, PAYLOAD_BYTES);
    let names: Vec<_> = planned.physical.keys().map(String::as_str).collect();
    assert!(validate_names(&planned.physical, names[..names.len() - 1].iter().copied()).is_err());
    for extra in [
        "lm_head.weight",
        "language_model.lm_head.weight",
        "vision_tower.ln_pre.weight",
    ] {
        assert!(validate_names(&planned.physical, names.iter().copied().chain([extra])).is_err());
    }
    assert!(validate_names(&planned.physical, names.iter().copied().chain([names[0]])).is_err());
}

#[test]
fn whole_aliases_preserve_metadata_and_unique_physical_payload() {
    let planned = plan(document(), options()).unwrap();
    let aliases = logical::ministral3_aliases(&planned.tensors).unwrap();
    assert_eq!(aliases.len(), 236);
    assert_eq!(
        aliases
            .values()
            .map(|item| item.tensor().payload_bytes())
            .sum::<usize>(),
        PAYLOAD_BYTES
    );
    for (name, item) in &aliases {
        assert!(name.starts_with("model."));
        assert_eq!(
            item.tensor(),
            &planned.tensors[&format!("language_model.{name}")]
        );
    }
    assert!(logical::ministral3_aliases(&BTreeMap::new()).is_err());
    let mut wrong = planned.tensors.clone();
    let item = wrong.remove("language_model.model.norm.weight").unwrap();
    wrong.insert("model.norm.weight".into(), item);
    assert!(logical::ministral3_aliases(&wrong).is_err());
    let mut broken = planned.tensors;
    broken
        .get_mut("language_model.model.norm.weight")
        .unwrap()
        .payload_bytes += 1;
    assert!(logical::ministral3_aliases(&broken).is_err());
}

#[test]
fn aliased_whole_reads_consume_original_unaligned_affine_and_norm_bytes() {
    let packed = Layout::new(2, 64, 4, 64, MetadataDtype::BF16).unwrap();
    let matrix_name = "language_model.model.embed_tokens.weight";
    let norm_name = "language_model.model.norm.weight";
    let codes = vec![0x10; packed.weight_bytes()];
    let scales = [half::bf16::ONE.to_le_bytes(); 2].concat();
    let biases = vec![0; packed.metadata_bytes()];
    let norm: Vec<_> = [1_f32, -0.5]
        .into_iter()
        .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
        .collect();
    let fixture = Fixture::new(vec![
        (matrix_name, Dtype::U32, vec![2, 8], codes.clone()),
        (
            "language_model.model.embed_tokens.scales",
            Dtype::BF16,
            vec![2, 1],
            scales.clone(),
        ),
        (
            "language_model.model.embed_tokens.biases",
            Dtype::BF16,
            vec![2, 1],
            biases.clone(),
        ),
        (norm_name, Dtype::BF16, vec![2], norm),
    ]);
    assert_eq!(fixture.data_start % 4, 1);
    let tensors = BTreeMap::from([
        (
            matrix_name.into(),
            TensorSpec {
                shape: vec![2, 64],
                storage: Storage::Affine(packed),
                payload_bytes: packed.total_bytes(),
            },
        ),
        (
            norm_name.into(),
            TensorSpec {
                shape: vec![2],
                storage: Storage::Float(MetadataDtype::BF16),
                payload_bytes: 4,
            },
        ),
    ]);
    let mut source = Source {
        config: Config {
            vocab_size: 2,
            hidden_size: 64,
            intermediate_size: 64,
            num_layers: 1,
            query_heads: 1,
            kv_heads: 1,
            head_dim: 64,
            max_positions: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 1e6,
        },
        declared_positions: 8,
        declared_sliding_window: None,
        architecture: Architecture::Mistral,
        tied_output: true,
        position_plan: crate::model::PositionPlan::Ordinary,
        ministral3_profile: None,
        rotary_layers: None,
        scales: Default::default(),
        logical: logical::ministral3_aliases(&tensors).unwrap(),
        tensors,
        source: fixture.source(),
        tensor_payload_bytes: packed.total_bytes() + 4,
        admitted_load_bytes: METADATA_ALLOWANCE + STREAM_BYTES + packed.total_bytes() + 4,
    };
    let mut w = vec![0; packed.weight_bytes()];
    let mut s = vec![0; packed.metadata_bytes()];
    let mut b = s.clone();
    assert_eq!(
        source
            .read_logical_affine("model.embed_tokens.weight", &mut w, &mut s, &mut b)
            .unwrap(),
        packed
    );
    assert_eq!((w, s, b), (codes, scales, biases));
    let mut values = Vec::new();
    source
        .visit_logical_float("model.norm.weight", |v| {
            values.push(v);
            Ok(())
        })
        .unwrap();
    assert_eq!(values, [1., -0.5]);
    assert!(source.tensors().contains_key(norm_name));
    assert!(!source.tensors().contains_key("model.norm.weight"));
    assert!(source.ministral3_profile().is_none());
}

#[test]
fn existing_ordinary_parser_and_inspection_still_reject_selected_wrapper() {
    assert!(dense::config(document(), options()).is_err());
    assert!(plan(json!({"model_type":"llama"}), options()).is_err());
    let fixture = Fixture::new(vec![("placeholder", Dtype::BF16, vec![1], vec![0, 0])]);
    std::fs::write(
        fixture.path.join("config.json"),
        serde_json::to_vec(&document()).unwrap(),
    )
    .unwrap();
    assert!(Source::open(&fixture.path, options()).is_err());
    assert!(Source::open_dense(&fixture.path, options()).is_err());
    let report = crate::mlx::compatibility::inspect(
        &fixture.path,
        crate::mlx::compatibility::Backend::Metal,
        options(),
    );
    assert_eq!(
        report.reason,
        crate::mlx::compatibility::Reason::UnknownArchitecture
    );
    // The selected opener rejects this tiny wrong inventory without reading its
    // tensor payload or accepting a small fixture as the real source profile.
    assert!(Source::open_ministral3_diagnostic(&fixture.path, options()).is_err());
}

#[test]
#[ignore = "Explicit selected converted-child cold metadata probe; no GPU or weight expansion"]
fn actual_converted_child_metadata_matches_profile_inventory_and_logical_aliases() {
    let path = std::path::PathBuf::from(
        std::env::var_os("TOKLET_MINISTRAL3_CHILD").expect("explicit child path"),
    );
    let source = Source::open_ministral3_diagnostic(&path, options()).unwrap();
    assert_eq!(source.architecture(), Architecture::Mistral);
    assert!(source.tied_output());
    assert!(source.ministral3_profile().is_some());
    assert_eq!(source.declared_positions(), 262144);
    assert_eq!(source.config().max_positions, 512);
    assert_eq!(source.tensor_payload_bytes(), PAYLOAD_BYTES);
    assert_eq!(source.tensors().len(), 236);
    assert_eq!(source.logical_tensors().len(), 236);
    assert_eq!(
        source.logical_tensors()["model.layers.0.self_attn.q_proj.weight"]
            .tensor()
            .shape(),
        [4096, 3072]
    );
    assert!(source.content_identity().is_none());
    assert!(source.dense_content_identity().is_none());
}
