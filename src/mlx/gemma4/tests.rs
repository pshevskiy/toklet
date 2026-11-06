use super::*;
use sha2::{Digest, Sha256};

const CONFIG: &[u8] = include_bytes!("../../../tests/fixtures/gemma4-e2b-metadata-v1/config.json");
const INDEX: &[u8] = include_bytes!("../../../tests/fixtures/gemma4-e2b-metadata-v1/index.json");
const HEADER: &[u8] = include_bytes!("../../../tests/fixtures/gemma4-e2b-metadata-v1/header.json");
const CONTAINER: u64 = 3550670554;

fn plan() -> E2bPlan {
    inspect(CONFIG, INDEX, HEADER, CONTAINER, ModalityPolicy::TextOnly).unwrap()
}
fn encoded(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).unwrap()
}
fn edited_header(header: Value, index: Value) -> Result<E2bPlan, Error> {
    let bytes = encoded(&header);
    // Re-encoding changes only the header length. Preserve advertised payload
    // length so a negative semantic/layout test does not fail on that boundary.
    inspect(
        CONFIG,
        &encoded(&index),
        &bytes,
        8 + bytes.len() as u64 + 3550352326,
        ModalityPolicy::TextOnly,
    )
}

#[test]
fn mixed_policy_is_exact_and_keeps_the_affine4_inspector_separate() {
    let mut doc: Value = serde_json::from_slice(CONFIG).unwrap();
    let quant = Format::QatMixed.quantization();
    assert_eq!(quant.as_object().unwrap().len(), 108);
    doc["quantization"] = quant.clone();
    doc["quantization_config"] = quant;
    let text = doc["text_config"].as_object_mut().unwrap();
    text.remove("expert_intermediate_size");
    text.insert("moe_intermediate_size".into(), Value::Null);
    assert!(config(&doc, ModalityPolicy::TextOnly, Format::QatMixed).is_ok());
    assert!(config(&doc, ModalityPolicy::TextOnly, Format::Affine4).is_err());
    assert!(inspect_qat(CONFIG, INDEX, HEADER, ModalityPolicy::TextOnly).is_err());
    let tensors = logical(&layer_plans(), Format::QatMixed).unwrap();
    let mut widths = [0; 2];
    for tensor in tensors {
        if let Storage::Affine(layout) = tensor.storage {
            match layout.bits() {
                4 => widths[0] += 1,
                8 => {
                    widths[1] += 1;
                    assert!(tensor.name.contains(".mlp."));
                }
                _ => panic!("unexpected Gemma QAT bit width"),
            }
        }
    }
    assert_eq!(widths, [173, 105]);
    let role = "language_model.model.layers.0.mlp.gate_proj";
    doc["quantization"][role]["bits"] = json!(4);
    doc["quantization_config"][role]["bits"] = json!(4);
    assert!(config(&doc, ModalityPolicy::TextOnly, Format::QatMixed).is_err());
}

#[test]
#[ignore = "explicit pinned QAT metadata buffers, no model payload"]
fn pinned_qat_metadata_and_mutations() {
    let root = std::path::PathBuf::from(std::env::var_os("TOKLET_GEMMA_QAT_METADATA").unwrap());
    let names = [
        "config.json",
        "model.safetensors.index.json",
        "weight-header.json",
    ];
    let mut buffers = names.map(|name| std::fs::read(root.join(name)).unwrap());
    let inspect =
        |bytes: &[Vec<u8>; 3], policy| inspect_qat(&bytes[0], &bytes[1], &bytes[2], policy);
    let mixed = inspect(&buffers, ModalityPolicy::TextOnly).unwrap();
    assert_eq!(mixed.layers(), plan().layers());
    assert_eq!(mixed.text_tensors().len(), 1096);
    assert_eq!(mixed.excluded_tensors().len(), 1415);
    assert_eq!(mixed.text_payload_bytes(), 3_382_549_062);
    assert_eq!(mixed.payload_bytes(), 4_328_920_006);
    assert!(inspect(&buffers, ModalityPolicy::Reject).is_err());
    for index in 0..buffers.len() {
        buffers[index][0] ^= 1;
        assert!(inspect(&buffers, ModalityPolicy::TextOnly).is_err());
        buffers[index][0] ^= 1;
    }
}

#[test]
fn frozen_candidate_metadata_identity_layout_counts_bytes_and_layer_routes() {
    let raw = include_bytes!("../../../tests/fixtures/gemma4-e2b-metadata-v1/manifest.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "e2acede6a0c9eb42360eddf8f403fc2214fac01a01a79ecd10230a61ad9acdab"
    );
    let manifest: Value = serde_json::from_slice(raw).unwrap();
    for (name, bytes) in [
        ("config.json", CONFIG),
        ("index.json", INDEX),
        ("header.json", HEADER),
    ] {
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            manifest["files"][name]["sha256"].as_str().unwrap()
        );
        assert_eq!(
            bytes.len() as u64,
            manifest["files"][name]["bytes"].as_u64().unwrap()
        );
    }
    assert_eq!(manifest["payload_downloaded"], false);
    assert_eq!(manifest["payload_integrity_verified"], false);
    let p = plan();
    assert_eq!(p.logical_tensors().len(), 540);
    assert_eq!(p.text_tensors().len(), 1096);
    assert_eq!(p.excluded_tensors().len(), 1415);
    assert_eq!(p.text_payload_bytes(), 2603981382);
    assert_eq!(p.excluded_payload_bytes(), 946370944);
    assert_eq!(p.payload_bytes(), 3550352326);
    assert_eq!(p.declared_total_positions(), 131072);
    assert_eq!(
        p.logical_tensors()
            .iter()
            .filter(|t| matches!(t.storage, Storage::Affine(_)))
            .count(),
        278
    );
    assert_eq!(
        p.logical_tensors()
            .iter()
            .filter(|t| matches!(t.storage, Storage::Bf16Vector))
            .count(),
        262
    );
    let bytes = p
        .logical_tensors()
        .iter()
        .map(|tensor| match (tensor.shape, tensor.storage) {
            (Shape::Vector(n), Storage::Bf16Vector) => n * 2,
            (Shape::Matrix { .. }, Storage::Affine(layout)) => layout.total_bytes(),
            _ => unreachable!(),
        })
        .sum::<usize>();
    assert_eq!(bytes, p.text_payload_bytes());
    for (i, layer) in p.layers().iter().enumerate() {
        let full = i % 5 == 4;
        assert_eq!(
            layer.kind,
            if full { KvKind::Full } else { KvKind::Sliding }
        );
        assert_eq!(
            (layer.head_dim, layer.rotated_dims),
            if full { (512, 128) } else { (256, 256) }
        );
        assert_eq!(
            layer.producer,
            if i < 15 {
                i
            } else if full {
                14
            } else {
                13
            }
        );
        assert_eq!(layer.intermediate, if i < 15 { 6144 } else { 12288 });
        if i >= 15 {
            assert!(!p.text_tensors().contains_key(&format!(
                "language_model.model.layers.{i}.self_attn.k_proj.weight"
            )));
        }
    }
    let ple = p
        .logical_tensors()
        .iter()
        .find(|t| t.name == "language_model.model.embed_tokens_per_layer.weight")
        .unwrap();
    assert_eq!(
        ple.shape,
        Shape::Matrix {
            rows: 262144,
            columns: 8960
        }
    );
    assert!(
        matches!(ple.storage,Storage::Affine(layout) if layout.total_bytes()==1321205760 && layout.metadata_dtype()==MetadataDtype::BF16)
    );
}

#[test]
fn every_changed_or_missing_text_option_and_unknown_field_rejects() {
    let base: Value = serde_json::from_slice(CONFIG).unwrap();
    let cases = [
        ("attention_bias", json!(true)),
        ("attention_dropout", json!(0.1)),
        ("attention_k_eq_v", json!(true)),
        ("enable_moe_block", json!(true)),
        ("global_head_dim", json!(256)),
        ("head_dim", json!(128)),
        ("hidden_activation", json!("silu")),
        ("hidden_size", json!(2048)),
        ("hidden_size_per_layer_input", json!(0)),
        ("intermediate_size", json!(8192)),
        ("max_position_embeddings", json!(262144)),
        ("num_attention_heads", json!(16)),
        ("num_experts", json!(8)),
        ("num_global_key_value_heads", json!(8)),
        ("num_hidden_layers", json!(34)),
        ("num_key_value_heads", json!(2)),
        ("num_kv_shared_layers", json!(0)),
        ("rms_norm_eps", json!(1e-5)),
        ("sliding_window", json!(1024)),
        ("tie_word_embeddings", json!(false)),
        ("use_bidirectional_attention", json!(true)),
        ("use_cache", json!(false)),
        ("use_double_wide_mlp", json!(false)),
        ("vocab_size", json!(262145)),
        ("rope_parameters", json!({})),
        ("layer_types", json!(["full_attention"])),
        ("final_logit_softcapping", json!(null)),
    ];
    for (name, value) in cases {
        let mut changed = base.clone();
        changed["text_config"][name] = value;
        assert!(
            inspect(
                &encoded(&changed),
                INDEX,
                HEADER,
                CONTAINER,
                ModalityPolicy::TextOnly
            )
            .is_err(),
            "{name}"
        );
    }
    for name in base["text_config"].as_object().unwrap().keys() {
        let mut changed = base.clone();
        changed["text_config"].as_object_mut().unwrap().remove(name);
        assert!(
            inspect(
                &encoded(&changed),
                INDEX,
                HEADER,
                CONTAINER,
                ModalityPolicy::TextOnly
            )
            .is_err(),
            "missing {name}"
        );
    }
    for (group, name) in [
        ("text_config", "rope_traditional"),
        ("", "new_decoder_graph_option"),
    ] {
        let mut changed = base.clone();
        let object = if group.is_empty() {
            &mut changed
        } else {
            &mut changed[group]
        };
        object[name] = json!(false);
        assert!(
            inspect(
                &encoded(&changed),
                INDEX,
                HEADER,
                CONTAINER,
                ModalityPolicy::TextOnly
            )
            .is_err()
        );
    }
}

#[test]
fn explicit_modality_and_quantized_representation_policy_is_required() {
    assert!(inspect(CONFIG, INDEX, HEADER, CONTAINER, ModalityPolicy::Reject).is_err());
    let base: Value = serde_json::from_slice(CONFIG).unwrap();
    for (name, value) in [
        ("model_type", json!("gemma4_text")),
        ("architectures", json!(["Gemma4ForCausalLM"])),
        ("tie_word_embeddings", json!(false)),
        ("dtype", json!("float16")),
        ("audio_config", json!({"model_type":"unknown_audio"})),
        ("vision_config", Value::Null),
        (
            "quantization",
            json!({"group_size":64,"bits":8,"mode":"affine"}),
        ),
        ("quantization_config", Value::Null),
    ] {
        let mut changed = base.clone();
        changed[name] = value;
        assert!(
            inspect(
                &encoded(&changed),
                INDEX,
                HEADER,
                CONTAINER,
                ModalityPolicy::TextOnly
            )
            .is_err(),
            "{name}"
        );
    }
}

#[test]
fn tensor_names_dtypes_shapes_layout_and_index_bijection_are_exact() {
    let header: Value = serde_json::from_slice(HEADER).unwrap();
    let index: Value = serde_json::from_slice(INDEX).unwrap();
    let name = "language_model.model.embed_tokens.weight";
    for (key, value) in [
        ("dtype", json!("BF16")),
        ("shape", json!([262144, 1536])),
        ("extra", json!(true)),
    ] {
        let mut h = header.clone();
        h[name][key] = value;
        assert!(edited_header(h, index.clone()).is_err());
    }
    for name in [
        "language_model.model.embed_tokens.scales",
        "language_model.model.layers.0.self_attn.k_norm.weight",
    ] {
        let mut h = header.clone();
        h.as_object_mut().unwrap().remove(name);
        let mut i = index.clone();
        i["weight_map"].as_object_mut().unwrap().remove(name);
        assert!(edited_header(h, i).is_err());
    }
    let mut i = index.clone();
    i["weight_map"][name] = json!("../model.safetensors");
    assert!(edited_header(header.clone(), i).is_err());
    let mut i = index.clone();
    i["weight_map"]["not_in_header"] = json!("model.safetensors");
    assert!(edited_header(header.clone(), i).is_err());
    let mut h = header.clone();
    let mut i = index.clone();
    let old = "language_model.model.layers.0.self_attn.k_norm.weight";
    let new = "language_model.model.layers.15.self_attn.k_norm.weight";
    let value = h.as_object_mut().unwrap().remove(old).unwrap();
    h[new] = value;
    i["weight_map"].as_object_mut().unwrap().remove(old);
    i["weight_map"][new] = json!("model.safetensors");
    assert!(edited_header(h, i).is_err());
    let mut h = header.clone();
    let mut i = index.clone();
    let old = "embed_audio.embedding_projection.weight";
    let value = h.as_object_mut().unwrap().remove(old).unwrap();
    h["unknown_component.weight"] = value;
    i["weight_map"].as_object_mut().unwrap().remove(old);
    i["weight_map"]["unknown_component.weight"] = json!("model.safetensors");
    assert!(edited_header(h, i).is_err());
}

#[test]
fn shape_span_arithmetic_gap_overlap_and_advertised_container_bounds_reject() {
    let header: Value = serde_json::from_slice(HEADER).unwrap();
    let index: Value = serde_json::from_slice(INDEX).unwrap();
    let name = "language_model.model.norm.weight";
    for offsets in [
        json!([0, u64::MAX]),
        json!([10, 1]),
        json!([0, 3071]),
        json!([3550352326_u64, 3550355398_u64]),
    ] {
        let mut h = header.clone();
        h[name]["data_offsets"] = offsets;
        assert!(edited_header(h, index.clone()).is_err());
    }
    for shape in [
        json!([1_u64 << 31]),
        json!([1_u64 << 30, 1_u64 << 30, 1_u64 << 30]),
        json!([true]),
        json!([1, 1, 1, 1, 1, 1, 1, 1, 1]),
    ] {
        let mut h = header.clone();
        h[name]["shape"] = shape;
        assert!(edited_header(h, index.clone()).is_err());
    }
    let mut h = header.clone();
    let span = h[name]["data_offsets"].as_array().unwrap();
    let start = span[0].as_u64().unwrap();
    let end = span[1].as_u64().unwrap();
    h[name]["data_offsets"] = json!([start + 2, end + 2]);
    assert!(edited_header(h, index.clone()).is_err());
    for size in [0, 7, CONTAINER - 1, CONTAINER + 1, u64::MAX] {
        assert!(inspect(CONFIG, INDEX, HEADER, size, ModalityPolicy::TextOnly).is_err());
    }
    let mut i = index;
    i["metadata"]["total_size"] = json!(1);
    assert!(edited_header(header, i).is_err());
}

#[test]
fn bounded_duplicate_free_json_has_no_payload_or_execution_side_effects() {
    let oversized = vec![b' '; 1024 * 1024 + 1];
    for (c, i, h) in [
        (&oversized[..], INDEX, HEADER),
        (CONFIG, &oversized[..], HEADER),
        (CONFIG, INDEX, &oversized[..]),
    ] {
        assert!(inspect(c, i, h, CONTAINER, ModalityPolicy::TextOnly).is_err());
    }
    for raw in [
        b"{\"model_type\":\"gemma4\",\"model_type\":\"gemma4\"}".as_slice(),
        b"{\"nested\":{\"x\":1,\"x\":2}}",
        b"{\"x\":NaN}",
        b"[]",
    ] {
        assert!(inspect(raw, INDEX, HEADER, CONTAINER, ModalityPolicy::TextOnly).is_err());
    }
    let duplicate_index = b"{\"metadata\":{},\"weight_map\":{\"x\":\"a\",\"x\":\"a\"}}";
    assert!(
        inspect(
            CONFIG,
            duplicate_index,
            HEADER,
            CONTAINER,
            ModalityPolicy::TextOnly
        )
        .is_err()
    );
    let duplicate_header = b"{\"__metadata__\":{},\"x\":{},\"x\":{}}";
    assert!(
        inspect(
            CONFIG,
            INDEX,
            duplicate_header,
            CONTAINER,
            ModalityPolicy::TextOnly
        )
        .is_err()
    );
}

#[test]
fn finite_float_options_ignore_lexical_exponents_but_counts_stay_integers() {
    let mut value: Value = serde_json::from_slice(CONFIG).unwrap();
    value["text_config"]["rms_norm_eps"] = serde_json::from_str("1e-6").unwrap();
    value["text_config"]["attention_dropout"] = json!(0);
    value["text_config"]["rope_parameters"]["full_attention"]["rope_theta"] = json!(1000000);
    assert!(
        inspect(
            &encoded(&value),
            INDEX,
            HEADER,
            CONTAINER,
            ModalityPolicy::TextOnly
        )
        .is_ok()
    );
    for bad in [
        json!(true),
        json!(0),
        serde_json::from_str("1e400").unwrap(),
        json!(1.0000000001e-6),
    ] {
        let mut changed = value.clone();
        changed["text_config"]["rms_norm_eps"] = bad;
        assert!(
            inspect(
                &encoded(&changed),
                INDEX,
                HEADER,
                CONTAINER,
                ModalityPolicy::TextOnly
            )
            .is_err()
        );
    }
    for bad in [json!(true), json!(1536.0), json!("1536")] {
        let mut changed = value.clone();
        changed["text_config"]["hidden_size"] = bad;
        assert!(
            inspect(
                &encoded(&changed),
                INDEX,
                HEADER,
                CONTAINER,
                ModalityPolicy::TextOnly
            )
            .is_err()
        );
    }
}
