use super::*;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
const CONFIG: &[u8] = include_bytes!("../../../tests/fixtures/lfm2-350m-metadata-v1/config.json");
const INDEX: &[u8] = include_bytes!("../../../tests/fixtures/lfm2-350m-metadata-v1/index.json");
const HEADER: &[u8] = include_bytes!("../../../tests/fixtures/lfm2-350m-metadata-v1/header.json");
const CONTAINER: u64 = 199527820;
const PAYLOAD: usize = 199491072;
fn encoded(v: &Value) -> Vec<u8> {
    serde_json::to_vec(v).unwrap()
}
fn plan() -> Plan {
    inspect(CONFIG, INDEX, HEADER, CONTAINER).unwrap()
}
fn changed(config: &Value, index: &Value, header: &Value) -> Result<Plan, Error> {
    let header = encoded(header);
    inspect(
        &encoded(config),
        &encoded(index),
        &header,
        8 + header.len() as u64 + PAYLOAD as u64,
    )
}
fn values() -> (Value, Value, Value) {
    (
        serde_json::from_slice(CONFIG).unwrap(),
        serde_json::from_slice(INDEX).unwrap(),
        serde_json::from_slice(HEADER).unwrap(),
    )
}

#[test]
fn frozen_original_inventory_and_explicit_no_admission() {
    for (bytes, sha) in [
        (
            CONFIG,
            "95e6906d77ac647e8bf93c2d2bbd2f37eb9bf3fac8732a563653236ad81072af",
        ),
        (
            INDEX,
            "9921c33945951c541dfb7de7a11c6ff3b0bb8fb58009be615f851a7de89bcc38",
        ),
        (
            HEADER,
            "7c5e241d377d693fac694426ed10fb79df470864a9ec31d0ca82f3044171506d",
        ),
    ] {
        assert_eq!(format!("{:x}", Sha256::digest(bytes)), sha);
    }
    let p = plan();
    assert_eq!(
        (
            p.logical_tensors().len(),
            p.convolutions().len(),
            p.logical_owner_count(),
            p.physical_tensors().len()
        ),
        (138, 10, 148, 334)
    );
    assert_eq!(p.payload_bytes(), PAYLOAD);
    assert_eq!(
        (p.hidden_size(), p.intermediate_size(), p.vocab_size()),
        (1024, 4608, 65536)
    );
    assert_eq!(
        p.intermediate_size(),
        (2 * 6656 / 3_usize).div_ceil(256) * 256
    );
    assert_eq!(p.declared_total_positions(), 128000);
    assert_eq!(p.implemented_total_positions(), None);
    assert_eq!(p.qualified_total_positions(), None);
    assert!(p.norm_weights_are_direct());
    assert_eq!(p.norm_epsilon(), 1e-5_f32);
    assert!(p.tied_output());
    assert!(
        !p.physical_tensors()
            .keys()
            .any(|k| k.starts_with("lm_head") || k.ends_with(".bias"))
    );
    let mut affine_count = 0;
    let mut norm_count = 0;
    let mut bytes = 0;
    let mut matrix_elements = 0;
    let mut norm_elements = 0;
    for t in p.logical_tensors() {
        match t.storage {
            Storage::Affine(layout) => {
                affine_count += 1;
                matrix_elements += layout.rows() * layout.columns();
                bytes += layout.total_bytes();
                assert_eq!(
                    (layout.bits(), layout.group_size(), layout.metadata_dtype()),
                    (4, 64, MetadataDtype::BF16)
                );
            }
            Storage::Bf16Vector => {
                norm_count += 1;
                let Shape::Vector(n) = t.shape else {
                    panic!("norm must remain a vector")
                };
                norm_elements += n;
                bytes += n * 2;
            }
        }
    }
    assert_eq!(
        (affine_count, norm_count, matrix_elements, norm_elements),
        (93, 45, 354418688, 34560)
    );
    assert_eq!(
        bytes
            + p.convolutions()
                .iter()
                .map(|c| c.channels * c.taps * c.input_group_channels * 2)
                .sum::<usize>(),
        PAYLOAD
    );
    let table = p
        .logical_tensors()
        .iter()
        .filter(|t| t.name == "model.embed_tokens.weight")
        .count();
    assert_eq!(table, 1);
    // Each logical physical owner is unique, including rank-three exclusions.
    let names: BTreeSet<_> = p
        .logical_tensors()
        .iter()
        .map(|t| t.name.as_str())
        .chain(p.convolutions().iter().map(|c| c.coefficient_name.as_str()))
        .collect();
    assert_eq!(names.len(), 148);
}

#[test]
fn separate_state_routes_and_original_bcx_partitions() {
    let p = plan();
    let mut attention = Vec::new();
    let mut history = Vec::new();
    for (layer, recipe) in p.layers().iter().enumerate() {
        match recipe {
            LayerRecipe::FullAttention(a) => {
                assert!(ATTENTION_LAYERS.contains(&layer));
                assert_eq!(a.kv_owner, attention.len());
                assert_eq!((a.query_heads, a.kv_heads, a.head_dim), (16, 8, 64));
                assert_eq!((a.rope_theta, a.attention_scale), (1e6, 0.125));
                attention.push(a.kv_owner);
            }
            LayerRecipe::ShortConv3 { history_owner } => {
                assert!(!ATTENTION_LAYERS.contains(&layer));
                assert_eq!(*history_owner, history.len());
                let c = &p.convolutions()[*history_owner];
                assert_eq!(
                    (
                        c.layer,
                        c.history_owner,
                        c.channels,
                        c.taps,
                        c.input_group_channels
                    ),
                    (layer, *history_owner, 1024, 3, 1)
                );
                assert_eq!(
                    p.physical_tensors()[&c.coefficient_name].shape,
                    [1024, 3, 1]
                );
                history.push(*history_owner);
                let parent = p
                    .logical_tensors()
                    .iter()
                    .find(|t| t.name == c.fused_input_name)
                    .unwrap();
                let Storage::Affine(layout) = parent.storage else {
                    panic!()
                };
                let mut code = 0;
                let mut meta = 0;
                let mut selected_total = 0;
                for (branch, span) in c.bcx.iter().enumerate() {
                    assert_eq!(span.layout().rows(), 1024);
                    assert_eq!(span.layout().columns(), 1024);
                    assert_eq!(span.weight_range(), branch * 524288..(branch + 1) * 524288);
                    assert_eq!(span.metadata_range(), branch * 32768..(branch + 1) * 32768);
                    assert_eq!(span.weight_range().start, code);
                    assert_eq!(span.metadata_range().start, meta);
                    code = span.weight_range().end;
                    meta = span.metadata_range().end;
                    selected_total += span.layout().total_bytes();
                    for (suffix, relative) in [
                        ("weight", span.weight_range()),
                        ("scales", span.metadata_range()),
                        ("biases", span.metadata_range()),
                    ] {
                        let name = format!(
                            "{}.{}",
                            c.fused_input_name.strip_suffix(".weight").unwrap(),
                            suffix
                        );
                        let physical = &p.physical_tensors()[&name].range;
                        assert!(physical.start + relative.end <= physical.end);
                        if branch == 2 {
                            assert_eq!(physical.start + relative.end, physical.end);
                        }
                    }
                }
                assert_eq!(
                    (code, meta, selected_total),
                    (
                        layout.weight_bytes(),
                        layout.metadata_bytes(),
                        layout.total_bytes()
                    )
                );
            }
        }
    }
    assert_eq!(attention, (0..6).collect::<Vec<_>>());
    assert_eq!(history, (0..10).collect::<Vec<_>>());
}

#[test]
fn pure_payload_geometry_reuses_unique_kv_owners_without_admission() {
    use crate::model::{
        shared_kv::{KvKind, LayerKvGeometry, SharedKvPlan},
        window_cache::WindowKvCache,
    };
    let p = plan();
    let half = p.state_payload_bytes(512, KvStorage::F16).unwrap();
    assert_eq!(
        half,
        StatePayloadBytes {
            attention_kv: 6291456,
            convolution_committed: 40960,
            convolution_candidate: 40960,
            total: 6373376
        }
    );
    let float = p.state_payload_bytes(512, KvStorage::F32).unwrap();
    assert_eq!(float.total, half.total * 2);
    assert_eq!(float.attention_kv, half.attention_kv * 2);
    assert_eq!(float.convolution_committed, 81920);
    let geometries = [LayerKvGeometry {
        kind: KvKind::Full,
        heads: 8,
        head_dim: 64,
    }; 6];
    let shared = SharedKvPlan::new(&geometries, 0, 1, 512, 128000).unwrap();
    assert_eq!(
        shared.planned_cache_bytes().unwrap(),
        float.attention_kv
            + 6 * std::mem::size_of::<WindowKvCache>()
            + std::mem::size_of::<Vec<WindowKvCache>>()
    );
    for capacity in [1, 2, 31, 32, 511, 512, 128000] {
        let a = p.state_payload_bytes(capacity, KvStorage::F16).unwrap();
        let b = p.state_payload_bytes(capacity + 1, KvStorage::F16).unwrap();
        assert_eq!(b.attention_kv - a.attention_kv, 12288);
        assert_eq!(a.convolution_committed, b.convolution_committed);
    }
    assert!(p.state_payload_bytes(0, KvStorage::F16).is_err());
    assert!(p.state_payload_bytes(usize::MAX, KvStorage::F32).is_err());
    assert_eq!(p.implemented_total_positions(), None); // accounting is not admission
}

#[test]
fn strict_alias_math_and_unsupported_config_fields() {
    let (config, index, header) = values();
    for (field, value) in [
        ("hidden_size", json!(1025)),
        ("block_dim", json!(1025)),
        ("conv_dim", json!(1023)),
        ("conv_dim_out", json!(512)),
        ("block_ff_dim", json!(4608)),
        ("block_multiple_of", json!(128)),
        ("block_auto_adjust_ff_dim", json!(false)),
        ("block_ffn_dim_multiplier", json!(0.9)),
        ("num_heads", json!(8)),
        ("num_key_value_heads", json!(16)),
        ("norm_eps", json!(1e-6)),
        ("block_norm_eps", json!(1e-6)),
        ("conv_bias", json!(true)),
        ("conv_L_cache", json!(4)),
        ("full_attn_idxs", json!([2, 5, 8, 10, 14, 12])),
        ("block_use_swiglu", json!(false)),
        ("use_pos_enc", json!(false)),
        ("max_position_embeddings", json!(512)),
        ("rope_theta", json!(10000.0)),
        ("eos_token_id", json!(2)),
        ("quantization", json!({"group_size":32,"bits":4})),
        ("quantization_config", json!({"group_size":64,"bits":8})),
    ] {
        let mut bad = config.clone();
        bad[field] = value;
        assert!(
            changed(&bad, &index, &header).is_err(),
            "accepted changed {field}"
        );
    }
    for (field, value) in [
        ("rope_parameters", json!({"rope_theta":1e6})),
        ("intermediate_size", json!(4608)),
        ("layer_types", json!([])),
        ("model_file", json!("custom.py")),
        ("unknown_math", json!(true)),
        ("tie_word_embeddings", json!(false)),
    ] {
        let mut bad = config.clone();
        bad[field] = value;
        assert!(
            changed(&bad, &index, &header).is_err(),
            "accepted unsupported {field}"
        );
    }
    for field in [
        "num_heads",
        "block_norm_eps",
        "conv_dim",
        "quantization_config",
    ] {
        let mut bad = config.clone();
        bad.as_object_mut().unwrap().remove(field);
        assert!(changed(&bad, &index, &header).is_err());
    }
    // Numeric spelling follows the unchanged common semantic helper.
    let alternate = String::from_utf8(CONFIG.to_vec())
        .unwrap()
        .replace("1e-05", "0.00001");
    assert!(inspect(alternate.as_bytes(), INDEX, HEADER, CONTAINER).is_ok());
    let mut bad = config.clone();
    bad["hidden_size"] = json!(1024.0);
    assert!(changed(&bad, &index, &header).is_err());
}

#[test]
fn original_inventory_dtype_shape_and_tied_head_rejections() {
    let (config, index, header) = values();
    for name in [
        "model.embed_tokens.weight",
        "model.layers.0.conv.conv.weight",
        "model.layers.0.conv.in_proj.scales",
        "model.layers.2.self_attn.q_layernorm.weight",
    ] {
        let mut bad = header.clone();
        bad.as_object_mut().unwrap().remove(name);
        assert!(changed(&config, &index, &bad).is_err());
    }
    let mut i = index.clone();
    let mut h = header.clone();
    i["weight_map"]["lm_head.weight"] = json!("model.safetensors");
    h["lm_head.weight"] = h["model.embed_tokens.weight"].clone();
    assert!(changed(&config, &i, &h).is_err());
    for (name, field, value) in [
        (
            "model.layers.0.conv.conv.weight",
            "shape",
            json!([1024, 1, 3]),
        ),
        ("model.layers.0.conv.conv.weight", "dtype", json!("F16")),
        ("model.layers.0.ffn_norm.weight", "dtype", json!("F32")),
        ("model.layers.0.conv.in_proj.scales", "dtype", json!("F16")),
        (
            "model.layers.2.self_attn.k_proj.weight",
            "shape",
            json!([1024, 64]),
        ),
        (
            "model.layers.0.feed_forward.w1.weight",
            "shape",
            json!([6656, 128]),
        ),
        ("model.layers.0.conv.in_proj.weight", "extra", json!(0)),
    ] {
        let mut bad = header.clone();
        bad[name][field] = value;
        assert!(
            changed(&config, &index, &bad).is_err(),
            "accepted {name}.{field}"
        );
    }
    let mut bad = index.clone();
    bad["weight_map"]["model.layers.0.conv.conv.weight"] = json!("other.safetensors");
    assert!(changed(&config, &bad, &header).is_err());
    for field in ["total_size", "total_parameters"] {
        let mut bad = index.clone();
        bad["metadata"].as_object_mut().unwrap().remove(field);
        assert!(changed(&config, &bad, &header).is_err());
        bad["metadata"][field] = json!(1);
        assert!(changed(&config, &bad, &header).is_err());
    }
}

#[test]
fn checked_extents_overlap_overflow_and_bounded_json() {
    let (config, index, header) = values();
    let name = "model.layers.0.ffn_norm.weight";
    let original = &header[name]["data_offsets"];
    let start = original[0].as_u64().unwrap();
    let end = original[1].as_u64().unwrap();
    for range in [
        json!([end, start]),
        json!([start, end + 1]),
        json!([PAYLOAD, PAYLOAD + 2048]),
        json!([u64::MAX - 2048, u64::MAX]),
        json!([-1, 2047]),
    ] {
        let mut bad = header.clone();
        bad[name]["data_offsets"] = range;
        assert!(changed(&config, &index, &bad).is_err());
    }
    // Equal-sized norm aliases are invalid even though each individual span fits.
    let mut bad = header.clone();
    bad[name]["data_offsets"] =
        header["model.layers.0.operator_norm.weight"]["data_offsets"].clone();
    assert!(changed(&config, &index, &bad).is_err());
    let mut bad = header.clone();
    bad[name]["shape"] = json!(vec![1073741824_u64; 8]);
    assert!(changed(&config, &index, &bad).is_err());
    for size in [0, 7, 8, CONTAINER - 1, CONTAINER + 1, u64::MAX] {
        assert!(inspect(CONFIG, INDEX, HEADER, size).is_err());
    }
    assert!(
        inspect(
            br#"{"model_type":"lfm2","model_type":"lfm2"}"#,
            INDEX,
            HEADER,
            CONTAINER
        )
        .is_err()
    );
    assert!(inspect(CONFIG, INDEX, &vec![b' '; 1024 * 1024 + 1], CONTAINER).is_err());
    assert!(inspect(CONFIG, b"[]", HEADER, CONTAINER).is_err());
    assert!(inspect(CONFIG, INDEX, b"null", CONTAINER).is_err());
}
