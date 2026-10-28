use super::*;
use sha2::{Digest, Sha256};
const CONFIG: &[u8] = include_bytes!("../../../tests/fixtures/gemma3-270m-metadata-v1/config.json");
const INDEX: &[u8] = include_bytes!("../../../tests/fixtures/gemma3-270m-metadata-v1/index.json");
const HEADER: &[u8] = include_bytes!("../../../tests/fixtures/gemma3-270m-metadata-v1/header.json");
const CONTAINER: u64 = 150939130;
const PAYLOAD: usize = 150885632;
fn encoded(v: &Value) -> Vec<u8> {
    serde_json::to_vec(v).unwrap()
}
fn plan() -> Plan {
    inspect(CONFIG, INDEX, HEADER, CONTAINER).unwrap()
}
fn changed(header: &Value, index: &Value) -> Result<Plan, Error> {
    let header = encoded(header);
    inspect(
        CONFIG,
        &encoded(index),
        &header,
        8 + header.len() as u64 + PAYLOAD as u64,
    )
}

#[test]
fn frozen_inventory_and_separate_residual_attention_widths() {
    let raw = include_bytes!("../../../tests/fixtures/gemma3-270m-metadata-v1/manifest.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "f0ddee0a975ec206fc0031d6fcb818b31dbe407c0bbc9a21a3b7e9537b627f29"
    );
    let metadata: Value = serde_json::from_slice(raw).unwrap();
    for key in [
        "payload_downloaded",
        "payload_integrity_verified",
        "numerical_execution_qualified",
        "text_qualified",
    ] {
        assert_eq!(metadata[key], false);
    }
    for (bytes, sha) in [
        (
            CONFIG,
            "8fa17cb1689800679eeb5b19ab2d2ef5d425e16cf9f36f59925f130ed976ea38",
        ),
        (
            INDEX,
            "3ef9f6408eebccbe14976dd8db24d75d848f01c4430aa8b5924d97782a0930fb",
        ),
        (
            HEADER,
            "994efdf1aa3b4f6bc39d7935b1286380c6f9f49e926e02405fed6ed72f6e578f",
        ),
    ] {
        assert_eq!(format!("{:x}", Sha256::digest(bytes)), sha);
    }
    let p = plan();
    assert_eq!(p.logical_tensors().len(), 236);
    assert_eq!(p.physical_tensors().len(), 490);
    assert_eq!(p.payload_bytes(), PAYLOAD);
    assert_eq!(p.declared_total_positions(), 32768);
    assert_eq!(p.declared_sliding_window(), 512);
    assert_eq!(p.embedding_scale(), 25.25);
    assert_eq!(p.norm_gamma_offset(), 1.0);
    assert!(p.tied_output());
    assert_eq!(
        p.logical_tensors()
            .iter()
            .filter(|t| matches!(t.storage, Storage::Bf16Vector))
            .count(),
        109
    );
    let sum = p
        .logical_tensors()
        .iter()
        .map(|t| match t.storage {
            Storage::Bf16Vector => {
                let Shape::Vector(n) = t.shape else {
                    panic!("vector shape")
                };
                n * 2
            }
            Storage::Affine(layout) => {
                assert_eq!(layout.bits(), 4);
                assert_eq!(layout.group_size(), 64);
                assert_eq!(layout.metadata_dtype(), MetadataDtype::BF16);
                layout.total_bytes()
            }
        })
        .sum::<usize>();
    assert_eq!(sum, PAYLOAD);
    for (i, layer) in p.layers().iter().enumerate() {
        let global = i % 6 == 5;
        assert_eq!(
            layer.kind,
            if global {
                KvKind::Full
            } else {
                KvKind::Sliding
            }
        );
        assert_eq!(
            layer.rope_theta,
            if global { 1_000_000.0 } else { 10_000.0 }
        );
        assert_eq!(
            (layer.query_width, layer.kv_width, layer.head_dim),
            (1024, 256, 256)
        );
        assert_eq!(layer.attention_scale, 1.0 / 16.0);
    }
    assert!(
        !p.physical_tensors()
            .keys()
            .any(|n| n.starts_with("lm_head") || n.contains("v_norm"))
    );
    let q = p
        .logical_tensors()
        .iter()
        .find(|t| t.name == "model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(
        q.shape,
        Shape::Matrix {
            rows: 1024,
            columns: 640
        }
    );
    // Preserve the existing shared Config capability; no equality of residual
    // hidden width and concatenated query heads is required by its math checks.
    crate::model::Config {
        vocab_size: 262144,
        hidden_size: 640,
        intermediate_size: 2048,
        num_layers: 18,
        query_heads: 4,
        kv_heads: 1,
        head_dim: 256,
        max_positions: 32768,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
    }
    .validate()
    .unwrap();
}

#[test]
fn every_missing_changed_and_unknown_config_field_rejects() {
    let base: Value = serde_json::from_slice(CONFIG).unwrap();
    for key in base.as_object().unwrap().keys() {
        let mut value = base.clone();
        value.as_object_mut().unwrap().remove(key);
        assert!(
            inspect(&encoded(&value), INDEX, HEADER, CONTAINER).is_err(),
            "missing {key}"
        );
        let mut value = base.clone();
        value[key] = if base[key].is_null() {
            json!(0)
        } else {
            Value::Null
        };
        assert!(
            inspect(&encoded(&value), INDEX, HEADER, CONTAINER).is_err(),
            "changed {key}"
        );
    }
    for key in [
        "tie_word_embeddings",
        "vision_config",
        "rope_traditional",
        "attention_k_eq_v",
        "unknown",
    ] {
        let mut value = base.clone();
        value[key] = json!(true);
        assert!(
            inspect(&encoded(&value), INDEX, HEADER, CONTAINER).is_err(),
            "unknown {key}"
        );
    }
    let mut conflict = base;
    conflict["quantization_config"]["bits"] = json!(8);
    assert!(inspect(&encoded(&conflict), INDEX, HEADER, CONTAINER).is_err());
}

#[test]
fn all_physical_entries_and_both_inventories_are_required() {
    let header: Value = serde_json::from_slice(HEADER).unwrap();
    let index: Value = serde_json::from_slice(INDEX).unwrap();
    for name in index["weight_map"].as_object().unwrap().keys() {
        let mut missing = header.clone();
        missing.as_object_mut().unwrap().remove(name);
        assert!(changed(&missing, &index).is_err(), "missing {name}");
    }
    for name in [
        "model.norm.weight",
        "model.embed_tokens.weight",
        "model.layers.17.self_attn.k_norm.weight",
    ] {
        let mut missing = index.clone();
        missing["weight_map"].as_object_mut().unwrap().remove(name);
        assert!(changed(&header, &missing).is_err());
        let mut shard = index.clone();
        shard["weight_map"][name] = json!("other.safetensors");
        assert!(changed(&header, &shard).is_err());
    }
    let mut redundant = header.clone();
    let mut index2 = index.clone();
    redundant["lm_head.weight"] = header["model.embed_tokens.weight"].clone();
    index2["weight_map"]["lm_head.weight"] = json!("model.safetensors");
    assert!(changed(&redundant, &index2).is_err());
}

#[test]
fn malformed_representation_extent_and_container_reject() {
    let header: Value = serde_json::from_slice(HEADER).unwrap();
    let index: Value = serde_json::from_slice(INDEX).unwrap();
    for (field, value) in [
        ("dtype", json!("F16")),
        ("shape", json!([640, 1])),
        ("shape", json!([0])),
        ("shape", json!([2147483648u64])),
        ("data_offsets", json!([1, 1281])),
        ("data_offsets", json!([-1, 1279])),
        ("data_offsets", json!([false, 1280])),
        ("data_offsets", json!([0, 150885634])),
        ("unknown", json!(0)),
    ] {
        let mut bad = header.clone();
        bad["model.norm.weight"][field] = value;
        assert!(changed(&bad, &index).is_err(), "{field}");
    }
    let mut bad = index.clone();
    bad["metadata"]["total_size"] = json!(PAYLOAD + 1);
    assert!(changed(&header, &bad).is_err());
    let mut bad = index.clone();
    bad["metadata"]["total_parameters"] = json!(268098175);
    assert!(changed(&header, &bad).is_err());
    let mut bad = index.clone();
    bad["metadata"]
        .as_object_mut()
        .unwrap()
        .remove("total_parameters");
    assert!(changed(&header, &bad).is_err());
    let mut bad = header.clone();
    bad["__metadata__"] = json!({"format":"pt"});
    assert!(changed(&bad, &index).is_err());
    for size in [0, 7, CONTAINER - 1, CONTAINER + 1, u64::MAX] {
        assert!(inspect(CONFIG, INDEX, HEADER, size).is_err());
    }
}

#[test]
fn bounded_unique_json_and_numeric_field_types_reject() {
    assert!(
        inspect(
            br#"{"hidden_size":640,"hidden_size":640}"#,
            INDEX,
            HEADER,
            CONTAINER
        )
        .is_err()
    );
    assert!(
        inspect(
            CONFIG,
            br#"{"metadata":{},"metadata":{},"weight_map":{}}"#,
            HEADER,
            CONTAINER
        )
        .is_err()
    );
    let oversized = vec![b' '; 1024 * 1024 + 1];
    assert!(inspect(&oversized, INDEX, HEADER, CONTAINER).is_err());
    assert!(inspect(CONFIG, &oversized, HEADER, CONTAINER).is_err());
    assert!(inspect(CONFIG, INDEX, &oversized, CONTAINER).is_err());
    let mut config: Value = serde_json::from_slice(CONFIG).unwrap();
    config["hidden_size"] = json!(640.0);
    assert!(inspect(&encoded(&config), INDEX, HEADER, CONTAINER).is_err());
}
