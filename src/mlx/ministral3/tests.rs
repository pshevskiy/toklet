use super::*;
use crate::{mlx::qwen::parse_json, weights::LoadOptions};
use serde_json::json;
use sha2::{Digest, Sha256};

const ORIGINAL: &[u8] =
    include_bytes!("../../../tests/fixtures/ministral3-config-v1/ministral_original.json");
const MODIFIED: &[u8] =
    include_bytes!("../../../tests/fixtures/ministral3-config-v1/ministral_modified.json");

fn documents() -> [Value; 2] {
    [parse_json(ORIGINAL).unwrap(), parse_json(MODIFIED).unwrap()]
}

fn reject_replacement(pointer: &str, raw: &str) {
    for mut document in documents() {
        *document.pointer_mut(pointer).unwrap() = parse_json(raw.as_bytes()).unwrap();
        assert!(validate(&document).is_err(), "accepted {pointer}: {raw}");
    }
}

fn reject_insertion(pointer: &str, name: &str, raw: &str) {
    for mut document in documents() {
        document
            .pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert(name.into(), parse_json(raw.as_bytes()).unwrap());
        assert!(
            validate(&document).is_err(),
            "accepted {pointer}/{name}: {raw}"
        );
    }
}

#[test]
fn pinned_configs_bind_selected_metadata_without_mutation_or_width_coercion() {
    let manifest = parse_json(include_bytes!(
        "../../../tests/fixtures/ministral3-config-v1/manifest.json"
    ))
    .unwrap();
    assert_eq!(manifest["format"], "toklet-ministral3-config-v1");
    assert_eq!(manifest["payload_authenticated"], false);
    assert_eq!(manifest["graph_admitted"], false);
    for (name, raw, length, sha) in [
        (
            "ministral_original",
            ORIGINAL,
            1579,
            "c89d1a0b4f237d2892ce911d1fe03e9e5a4834579f7149ebc715a4c3fa564214",
        ),
        (
            "ministral_modified",
            MODIFIED,
            1788,
            "457b89531a7a992698b81025bf0071c89a91ad3b20423915627e55c59fb81b50",
        ),
    ] {
        let pin = &manifest["sources"][name];
        assert_eq!(raw.len(), length);
        assert_eq!(format!("{:x}", Sha256::digest(raw)), sha);
        assert_eq!(pin["bytes"], raw.len() as u64);
        assert_eq!(pin["sha256"], sha);
        assert_eq!(
            pin["path"],
            format!("tests/fixtures/ministral3-config-v1/{name}.json")
        );
        let document = parse_json(raw).unwrap();
        let before = document.clone();
        let selected = validate(&document).unwrap();
        assert_eq!(document, before);
        assert_eq!(
            selected.geometry(),
            Geometry {
                hidden_size: 3072,
                intermediate_size: 9216,
                layers: 26,
                query_heads: 32,
                kv_heads: 8,
                head_dim: 128,
                query_width: 4096,
                kv_width: 1024,
                vocab_size: 131072,
                original_max_positions: 16384,
                source_max_positions: 262144,
            }
        );
        assert_ne!(
            selected.geometry().hidden_size,
            selected.geometry().query_width
        );
    }
    let [original, modified] = documents();
    assert_eq!(validate(&original).unwrap(), validate(&modified).unwrap());
}

#[test]
fn dtype_inheritance_and_matching_aliases_remain_bf16() {
    for mut document in documents() {
        for pointer in ["", "/text_config", "/vision_config"] {
            let object = document
                .pointer_mut(pointer)
                .unwrap()
                .as_object_mut()
                .unwrap();
            object.insert("dtype".into(), json!("bfloat16"));
            object.insert("torch_dtype".into(), json!("bfloat16"));
        }
        validate(&document).unwrap();
        document.as_object_mut().unwrap().remove("dtype");
        for pointer in ["/text_config", "/vision_config"] {
            let object = document
                .pointer_mut(pointer)
                .unwrap()
                .as_object_mut()
                .unwrap();
            object.remove("dtype");
            object.remove("torch_dtype");
        }
        validate(&document).unwrap();
        document.as_object_mut().unwrap().remove("torch_dtype");
        assert!(validate(&document).is_err());
    }
    for pointer in ["", "/text_config", "/vision_config"] {
        for value in [
            r#""float16""#,
            r#""float32""#,
            r#""bf16""#,
            "null",
            "true",
            "16",
        ] {
            reject_insertion(pointer, "dtype", value);
            reject_insertion(pointer, "torch_dtype", value);
        }
    }
}

#[test]
fn wrapper_and_every_selected_geometry_field_reject_variants() {
    for (pointer, bad) in [
        ("/model_type", r#""ministral3""#),
        ("/architectures", r#"["MistralForCausalLM"]"#),
        (
            "/architectures",
            r#"["Mistral3ForConditionalGeneration","Other"]"#,
        ),
        ("/text_config/model_type", r#""mistral""#),
        ("/text_config/hidden_size", "4096"),
        ("/text_config/intermediate_size", "8192"),
        ("/text_config/num_hidden_layers", "25"),
        ("/text_config/num_attention_heads", "24"),
        ("/text_config/num_key_value_heads", "4"),
        ("/text_config/head_dim", "96"),
        ("/text_config/vocab_size", "131073"),
        ("/text_config/max_position_embeddings", "16384"),
        ("/text_config/hidden_size", "3072.0"),
        ("/text_config/head_dim", "true"),
        (
            "/text_config/rms_norm_eps",
            "0.000010000000000000000000000001",
        ),
        ("/text_config/hidden_act", r#""gelu""#),
        ("/text_config/tie_word_embeddings", "false"),
        ("/text_config/use_cache", "false"),
        ("/text_config/attention_dropout", "0.1"),
    ] {
        reject_replacement(pointer, bad);
    }
    for pointer in ["", "/text_config", "/vision_config"] {
        for raw in ["null", "[]", "1", r#""object""#] {
            reject_replacement(pointer, raw);
        }
    }
}

#[test]
fn yarn_parameters_aliases_and_precision_conflicts_reject() {
    for (name, bad) in [
        ("rope_type", r#""default""#),
        ("type", r#""llama3""#),
        ("type", "null"),
        ("factor", "8"),
        ("beta_fast", "16"),
        ("beta_slow", "2"),
        ("mscale", "0"),
        ("mscale_all_dim", "0"),
        ("original_max_position_embeddings", "16384.0"),
        ("original_max_position_embeddings", "8192"),
        ("llama_4_scaling_beta", "0.1000000000000000000000000001"),
        ("rope_theta", "1000000.00000000000000000001"),
    ] {
        reject_replacement(&format!("/text_config/rope_parameters/{name}"), bad);
    }
    reject_insertion("/text_config/rope_parameters", "truncate", "false");
    reject_insertion("/text_config/rope_parameters", "unknown_scale", "1");
    reject_insertion("/text_config", "rope_theta", "1000000.00000000001");
    for mut document in documents() {
        let rope = document["text_config"]["rope_parameters"]
            .as_object_mut()
            .unwrap();
        rope.insert("truncate".into(), json!(true));
        rope.remove("type");
        validate(&document).unwrap();
        let rope = document["text_config"]["rope_parameters"]
            .as_object_mut()
            .unwrap();
        rope.insert("type".into(), json!("yarn"));
        rope.remove("rope_type");
        validate(&document).unwrap();
        document["text_config"]["rope_parameters"]
            .as_object_mut()
            .unwrap()
            .remove("type");
        assert!(validate(&document).is_err());
    }
}

#[test]
fn exact_decimal_comparison_accepts_equivalent_notation_without_float_coercion() {
    for mut document in documents() {
        for (pointer, raw) in [
            ("/text_config/rms_norm_eps", "1.000e-5"),
            ("/text_config/rope_parameters/rope_theta", "1e6"),
            ("/text_config/rope_parameters/llama_4_scaling_beta", "0.10"),
            ("/text_config/rope_parameters/factor", "160e-1"),
        ] {
            *document.pointer_mut(pointer).unwrap() = parse_json(raw.as_bytes()).unwrap();
        }
        validate(&document).unwrap();
    }
    for raw in ["1e99999", "1e-99999", "true", r#""0.1""#, "null"] {
        reject_replacement("/text_config/rope_parameters/llama_4_scaling_beta", raw);
    }
}

#[test]
fn optional_semantic_fields_accept_only_selected_defaults() {
    for mut document in documents() {
        let text = document["text_config"].as_object_mut().unwrap();
        for name in [
            "attention_bias",
            "mlp_bias",
            "rope_traditional",
            "rope_interleaved",
            "use_sliding_window",
        ] {
            text.insert(name.into(), json!(false));
        }
        text.insert("rope_theta".into(), json!(1000000));
        text.insert("rope_scaling".into(), Value::Null);
        text.insert("layer_types".into(), json!(vec!["full_attention"; 26]));
        validate(&document).unwrap();
    }
    for name in [
        "attention_bias",
        "mlp_bias",
        "rope_traditional",
        "rope_interleaved",
        "use_sliding_window",
    ] {
        reject_insertion("/text_config", name, "true");
        reject_insertion("/text_config", name, "0");
    }
    for (name, value) in [
        ("rope_scaling", r#"{"rope_type":"yarn","factor":16}"#),
        ("sliding_window", "512"),
        ("partial_rotary_factor", "0.5"),
        ("partial_rotary_factor", "1"),
        ("query_pre_attn_scalar", "128"),
        ("embedding_multiplier", "1"),
        ("logits_scaling", "1"),
        ("layer_types", r#"["full_attention"]"#),
        ("layer_types", "null"),
        ("quantization", r#"{"bits":4,"group_size":64}"#),
    ] {
        reject_insertion("/text_config", name, value);
    }
    reject_insertion("", "tie_word_embeddings", "false");
    reject_insertion("", "attention_bias", "false");
}

#[test]
fn optional_token_ids_are_canonical_in_both_scopes_and_never_codec_admission() {
    for mut document in documents() {
        for pointer in ["", "/text_config"] {
            let object = document
                .pointer_mut(pointer)
                .unwrap()
                .as_object_mut()
                .unwrap();
            for (name, value) in [
                ("bos_token_id", 1),
                ("eos_token_id", 2),
                ("pad_token_id", 11),
            ] {
                object.insert(name.into(), json!(value));
            }
        }
        validate(&document).unwrap();
        for pointer in ["", "/text_config"] {
            let object = document
                .pointer_mut(pointer)
                .unwrap()
                .as_object_mut()
                .unwrap();
            for name in ["bos_token_id", "eos_token_id", "pad_token_id"] {
                object.remove(name);
            }
        }
        validate(&document).unwrap();
    }
    for pointer in ["", "/text_config"] {
        for (name, values) in [
            ("bos_token_id", ["2", "1.0", "true", "null"]),
            ("eos_token_id", ["1", "2.0", "[2]", "null"]),
            ("pad_token_id", ["0", "11.0", "-1", "null"]),
        ] {
            for value in values {
                reject_insertion(pointer, name, value);
            }
        }
    }
    reject_replacement("/image_token_index", "11");
}

#[test]
fn unknown_fields_and_vision_variants_are_not_blanket_ignored() {
    for pointer in [
        "",
        "/text_config",
        "/vision_config",
        "/text_config/rope_parameters",
        "/vision_config/rope_parameters",
    ] {
        reject_insertion(pointer, "future_semantic_field", "1");
    }
    for (pointer, raw) in [
        ("/vision_config/model_type", r#""other""#),
        ("/vision_config/hidden_size", "2048"),
        ("/vision_config/attention_dropout", "0.1"),
        (
            "/vision_config/rope_parameters/rope_theta",
            "10000.0000000000000000001",
        ),
        ("/multimodal_projector_bias", "true"),
        ("/projector_hidden_act", r#""silu""#),
        ("/spatial_merge_size", "1"),
        ("/vision_feature_layer", "-2"),
    ] {
        reject_replacement(pointer, raw);
    }
    reject_insertion("/vision_config", "rope_theta", "9999");
}

#[test]
fn harmless_export_metadata_is_explicit_and_typed() {
    for mut document in documents() {
        for pointer in ["", "/text_config", "/vision_config"] {
            let object = document
                .pointer_mut(pointer)
                .unwrap()
                .as_object_mut()
                .unwrap();
            object.insert("transformers_version".into(), json!("metadata-only-export"));
            object.insert("_name_or_path".into(), json!("local/export"));
        }
        validate(&document).unwrap();
    }
    for pointer in ["", "/text_config", "/vision_config"] {
        reject_insertion(pointer, "transformers_version", "1");
        reject_insertion(pointer, "_name_or_path", "null");
        reject_insertion(pointer, "initializer_range", "0.03");
    }
    reject_insertion("", "unsloth_fixed", r#""true""#);
}

#[test]
fn existing_source_json_boundary_rejects_duplicates_and_nonfinite_values() {
    for raw in [
        br#"{"model_type":"mistral3","model_type":"mistral3"}"#.as_slice(),
        br#"{"text_config":{"dtype":"bfloat16","dtype":"float16"}}"#,
        br#"{"text_config":{"rope_parameters":{"factor":16,"factor":8}}}"#,
        br#"{"rms_norm_eps":NaN}"#,
        br#"{"rms_norm_eps":Infinity}"#,
    ] {
        assert!(parse_json(raw).is_err());
    }
    let raw = " ".repeat(1024 * 1024 + 1);
    assert!(parse_json(raw.as_bytes()).is_err());
}

#[test]
fn ordinary_dense_architecture_and_public_source_admission_still_reject() {
    for document in documents() {
        for value in [&document, &document["text_config"]] {
            assert!(matches!(
                crate::mlx::dense::config(value.clone(), LoadOptions::default()),
                Err(Error::Invalid(reason)) if reason == "unsupported native MLX decoder architecture"
            ));
        }
    }
    // The ordinary public affine loader rejects these genuine BF16 wrappers at
    // its existing storage boundary, before looking for any tensor/index file.
    // Its separate architecture boundary is checked above for wrapper and text.
    for raw in [ORIGINAL, MODIFIED] {
        let directory = std::env::temp_dir().join(format!(
            "toklet-ministral3-metadata-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(directory.join("config.json"), raw).unwrap();
        let result = crate::mlx::qwen::Source::open_dense(&directory, LoadOptions::default());
        std::fs::remove_dir_all(&directory).unwrap();
        assert!(matches!(
            result,
            Err(Error::Invalid(reason)) if reason == "MLX affine artifact requires quantization settings"
        ));
    }
}
