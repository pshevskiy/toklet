use super::*;
const CONFIG: &[u8] =
    include_bytes!("../../../tests/fixtures/qwen35-4b-avan-metadata-v1/config.json");
const HEADER: &[u8] =
    include_bytes!("../../../tests/fixtures/qwen35-4b-avan-metadata-v1/header.json");

fn plan() -> Plan {
    inspect(CONFIG, HEADER, CONTAINER_BYTES, ModalityPolicy::TextOnly).unwrap()
}
fn changed(config: &Value, header: &Value) -> Result<Plan, Error> {
    layout(
        &serde_json::to_vec(config).unwrap(),
        &serde_json::to_vec(header).unwrap(),
        CONTAINER_BYTES,
    )
}
fn documents() -> (Value, Value) {
    (
        serde_json::from_slice(CONFIG).unwrap(),
        serde_json::from_slice(HEADER).unwrap(),
    )
}

#[test]
fn pinned_inventory_matches_independent_fixture_and_grants_no_execution() {
    let p = plan();
    let fixture: Value = serde_json::from_slice(include_bytes!(
        "../../../tests/fixtures/qwen35-4b-avan-metadata-v1/inventory.json"
    ))
    .unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(CONFIG)),
        fixture["config_sha256"]
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(HEADER)),
        fixture["header_sha256"]
    );
    assert_eq!(p.physical_tensors().len(), 924);
    assert_eq!(p.logical_tensors().len(), 402);
    assert_eq!(p.convolutions().len(), 24);
    assert_eq!(p.logical_owner_count(), 426);
    let mut matrices = 0;
    let mut vectors = 0;
    let mut parameters = 0u64;
    for tensor in p.logical_tensors() {
        match (tensor.shape, tensor.storage) {
            (Shape::Matrix { rows, columns }, Storage::Affine(layout)) => {
                assert_eq!(
                    (
                        layout.rows(),
                        layout.columns(),
                        layout.bits(),
                        layout.group_size(),
                        layout.metadata_dtype()
                    ),
                    (rows, columns, 4, 64, MetadataDtype::BF16)
                );
                matrices += 1;
                parameters += (rows as u64) * (columns as u64);
            }
            (Shape::Vector(size), Storage::Bf16Vector) => {
                vectors += 1;
                parameters += size as u64;
            }
            _ => panic!("unexpected Qwen3.5 source representation"),
        }
    }
    parameters += p
        .convolutions()
        .iter()
        .map(|c| (c.channels * c.taps) as u64)
        .sum::<u64>();
    assert_eq!((matrices, vectors, parameters), (249, 153, 4_205_751_296));
    assert_eq!(parameters, fixture["logical_parameters"].as_u64().unwrap());
    assert_eq!(
        p.payload_bytes(),
        PayloadBytes {
            affine_codes: 2_102_394_880,
            affine_metadata: 262_799_360,
            bf16_vectors: 350_208,
            bf16_convolutions: 1_572_864,
            total: 2_367_117_312
        }
    );
    assert_eq!(
        (p.hidden_size(), p.intermediate_size(), p.vocab_size()),
        (2560, 9216, 248320)
    );
    assert_eq!(p.declared_total_positions(), 262144);
    assert_eq!(
        (
            p.implemented_total_positions(),
            p.qualified_total_positions()
        ),
        (None, None)
    );
    assert_eq!(p.declared_recurrent_state_dtype(), "float32");
    assert!(std::ptr::eq(p.embedding(), p.tied_head()));
    assert_eq!(
        p.embedding().name,
        "language_model.model.embed_tokens.weight"
    );
    assert!(p.physical_tensors().keys().all(|n| n.starts_with(PREFIX)));
    assert!(
        !p.physical_tensors()
            .keys()
            .any(|n| n.contains("mtp") || n.contains("vision") || n.contains("lm_head"))
    );
}

#[test]
fn recurrent_convolution_and_attention_owners_are_separate_and_indexed() {
    let p = plan();
    let mut recurrent = 0;
    let mut attention = 0;
    for (i, layer) in p.layers().iter().enumerate() {
        if i % 4 == 3 {
            assert_eq!(
                *layer,
                LayerRecipe::GatedFullAttention(AttentionPlan {
                    kv_owner: attention,
                    query_heads: 16,
                    kv_heads: 4,
                    head_dim: 256,
                    rotated_dims: 64,
                    output_gate: true
                })
            );
            let q = &p.physical_tensors()[&format!("{PREFIX}layers.{i}.self_attn.q_proj.weight")];
            assert_eq!(q.shape, [8192, 320]);
            attention += 1;
        } else {
            assert_eq!(
                *layer,
                LayerRecipe::GatedDelta(DeltaPlan {
                    recurrent_owner: recurrent,
                    convolution_owner: recurrent,
                    key_heads: 16,
                    value_heads: 32,
                    key_dim: 128,
                    value_dim: 128
                })
            );
            let conv = &p.convolutions()[recurrent];
            assert_eq!(
                (conv.layer, conv.history_owner, conv.channels, conv.taps),
                (i, recurrent, 8192, 4)
            );
            let physical = &p.physical_tensors()[&conv.name];
            assert_eq!(
                (
                    physical.dtype,
                    physical.shape.as_slice(),
                    physical.range.len()
                ),
                (Dtype::Bf16, &[8192, 4, 1][..], 65536)
            );
            recurrent += 1;
        }
    }
    assert_eq!((recurrent, attention), (24, 8));
    assert_eq!(p.rope_sections(), [11, 11, 10]);
    assert_eq!(p.rope_theta(), 10_000_000.0);
}

#[test]
fn byte_ranges_preserve_the_unaligned_source_origin_without_typed_casts() {
    let p = plan();
    assert_eq!(p.payload_offset(), 119837);
    assert_eq!(p.payload_offset() % 4, 1);
    let mut spans: Vec<_> = p
        .physical_tensors()
        .values()
        .map(|v| v.range.clone())
        .collect();
    spans.sort_unstable_by_key(|r| r.start);
    let mut cursor = 0;
    for span in spans {
        assert_eq!(span.start, cursor);
        cursor = span.end;
    }
    assert_eq!(cursor, p.payload_bytes().total);
    assert_eq!((cursor + p.payload_offset()) as u64, CONTAINER_BYTES);
}

#[test]
fn public_pins_reject_mutations_lengths_and_implicit_modality() {
    let mut config = CONFIG.to_vec();
    config[0] ^= 1;
    let mut header = HEADER.to_vec();
    header[100] ^= 1;
    for (c, h, n, policy) in [
        (
            config.as_slice(),
            HEADER,
            CONTAINER_BYTES,
            ModalityPolicy::TextOnly,
        ),
        (
            CONFIG,
            header.as_slice(),
            CONTAINER_BYTES,
            ModalityPolicy::TextOnly,
        ),
        (
            &CONFIG[..CONFIG.len() - 1],
            HEADER,
            CONTAINER_BYTES,
            ModalityPolicy::TextOnly,
        ),
        (
            CONFIG,
            &HEADER[..HEADER.len() - 1],
            CONTAINER_BYTES,
            ModalityPolicy::TextOnly,
        ),
        (
            CONFIG,
            HEADER,
            CONTAINER_BYTES - 1,
            ModalityPolicy::TextOnly,
        ),
        (
            CONFIG,
            HEADER,
            CONTAINER_BYTES + 1,
            ModalityPolicy::TextOnly,
        ),
        (CONFIG, HEADER, CONTAINER_BYTES, ModalityPolicy::Reject),
    ] {
        assert!(inspect(c, h, n, policy).is_err());
    }
}

#[test]
fn graph_option_mutations_do_not_bypass_the_fixed_hybrid_recipe() {
    let (original, h) = documents();
    for (path, value) in [
        ("/text_config/layer_types/0", json!("full_attention")),
        ("/text_config/full_attention_interval", json!(3)),
        ("/text_config/linear_conv_kernel_dim", json!(3)),
        ("/text_config/linear_key_head_dim", json!(64)),
        ("/text_config/linear_num_value_heads", json!(16)),
        ("/text_config/head_dim", json!(128)),
        ("/text_config/attn_output_gate", json!(false)),
        ("/text_config/partial_rotary_factor", json!(1.0)),
        (
            "/text_config/rope_parameters/mrope_interleaved",
            json!(false),
        ),
        (
            "/text_config/rope_parameters/mrope_section",
            json!([16, 16]),
        ),
        ("/text_config/mamba_ssm_dtype", json!("bfloat16")),
        ("/text_config/tie_word_embeddings", json!(false)),
        ("/tie_word_embeddings", json!(false)),
        ("/text_config/mtp_num_hidden_layers", json!(2)),
        ("/text_config/mtp_use_dedicated_embeddings", json!(true)),
        ("/text_config/num_hidden_layers", json!(31)),
        ("/text_config/hidden_size", json!(2560.0)),
        ("/text_config/hidden_act", json!("gelu")),
        ("/text_config/attention_bias", json!(true)),
    ] {
        let mut c = original.clone();
        *c.pointer_mut(path).unwrap() = value;
        assert!(changed(&c, &h).is_err(), "{path}");
    }
    let mut c = original;
    c["text_config"]["unreviewed_option"] = json!(true);
    assert!(changed(&c, &h).is_err());
}

#[test]
fn tensor_encoding_shape_and_descriptor_mutations_are_rejected() {
    let (c, original) = documents();
    for (name, field, value) in [
        (
            format!("{PREFIX}layers.0.linear_attn.conv1d.weight"),
            "shape",
            json!([8192, 3, 1]),
        ),
        (
            format!("{PREFIX}layers.0.linear_attn.A_log"),
            "dtype",
            json!("F32"),
        ),
        (
            format!("{PREFIX}layers.0.linear_attn.norm.weight"),
            "shape",
            json!([32, 128]),
        ),
        (
            format!("{PREFIX}layers.3.self_attn.q_proj.weight"),
            "shape",
            json!([4096, 320]),
        ),
        (
            format!("{PREFIX}embed_tokens.weight"),
            "shape",
            json!([248320, 640]),
        ),
        (
            format!("{PREFIX}embed_tokens.scales"),
            "dtype",
            json!("U32"),
        ),
        (
            format!("{PREFIX}embed_tokens.biases"),
            "shape",
            json!([248320, 39]),
        ),
        (
            format!("{PREFIX}norm.weight"),
            "data_offsets",
            json!([false, 5120]),
        ),
        (format!("{PREFIX}norm.weight"), "extra_field", json!(0)),
    ] {
        let mut h = original.clone();
        h[&name][field] = value;
        assert!(changed(&c, &h).is_err(), "{name}/{field}");
    }
}

#[test]
fn unknown_missing_duplicate_and_overlapping_owners_are_rejected() {
    let (c, original) = documents();
    let root = format!("{PREFIX}norm.weight");
    let mut h = original.clone();
    h.as_object_mut().unwrap().remove(&root);
    assert!(changed(&c, &h).is_err());
    let tensor = original[&root].clone();
    for extra in [
        "language_model.model.layers.32.extra.weight",
        "vision_tower.weight",
    ] {
        let mut h = original.clone();
        h.as_object_mut().unwrap().remove(&root);
        h[extra] = tensor.clone();
        assert!(changed(&c, &h).is_err());
    }
    let mut h = original.clone();
    let span = h[&root]["data_offsets"].as_array().unwrap();
    h[&root]["data_offsets"] =
        json!([span[0].as_u64().unwrap() + 2, span[1].as_u64().unwrap() + 2]);
    assert!(changed(&c, &h).is_err());
    let mut h = original.clone();
    h["__metadata__"] = json!({"format":"pt"});
    assert!(changed(&c, &h).is_err());
    let mut raw = serde_json::to_vec(&original).unwrap();
    raw.pop();
    raw.extend(format!(",{root:?}:{} }}", serde_json::to_string(&tensor).unwrap()).bytes());
    assert!(layout(CONFIG, &raw, CONTAINER_BYTES).is_err());
    assert!(layout(CONFIG, HEADER, (HEADER_BYTES + 7) as u64).is_err());
    assert!(layout(&vec![b' '; 64 * 1024 + 1], HEADER, CONTAINER_BYTES).is_err());
    assert!(layout(CONFIG, &vec![b' '; 512 * 1024 + 1], CONTAINER_BYTES).is_err());
}

#[test]
fn existing_dense_qwen3_admission_does_not_accept_this_source() {
    assert!(
        crate::mlx::qwen3::config(
            super::super::qwen::parse_json(CONFIG).unwrap(),
            crate::weights::LoadOptions::default()
        )
        .is_err()
    );
}

#[test]
#[ignore = "explicit two-process pinned Qwen3.5 metadata timing; no payload or GPU"]
fn actual_metadata_probe() {
    use std::{io::Read, path::PathBuf, time::Instant};
    let folder = PathBuf::from(std::env::var_os("TOKLET_QWEN35_METADATA_DIRECTORY").unwrap());
    fn read(path: &std::path::Path, expected: usize) -> Vec<u8> {
        let file = std::fs::File::open(path).unwrap();
        let metadata = file.metadata().unwrap();
        assert!(metadata.is_file());
        assert_eq!(metadata.len(), expected as u64);
        let mut bytes = Vec::with_capacity(expected + 1);
        file.take((expected + 1) as u64)
            .read_to_end(&mut bytes)
            .unwrap();
        assert_eq!(bytes.len(), expected);
        bytes
    }
    println!();
    for i in 0..13 {
        let start = Instant::now();
        let c = read(&folder.join("config.json"), CONFIG_BYTES);
        let h = read(&folder.join("model.safetensors.header.json"), HEADER_BYTES);
        let p = inspect(&c, &h, CONTAINER_BYTES, ModalityPolicy::TextOnly).unwrap();
        assert_eq!(
            (
                p.logical_tensors().len(),
                p.convolutions().len(),
                p.physical_tensors().len()
            ),
            (402, 24, 924)
        );
        drop(p);
        let ns = start.elapsed().as_nanos();
        println!(
            "QWEN35_METADATA {}",
            json!({"ordinal":i,"warmup":i<3,"elapsed_ns":ns,
            "metadata_bytes_read":CONFIG_BYTES+HEADER_BYTES,"payload_bytes_read":0,"execution_qualified":false})
        );
    }
}
