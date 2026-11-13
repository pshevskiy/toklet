use super::*;

pub(super) fn documents() -> (Vec<u8>, Vec<u8>, ArtifactPins) {
    // Synthetic metadata only: no large payload is allocated or authenticated.
    let config = serde_json::to_vec(&configuration::expected()).unwrap();
    let (_, common, convolutions) = recipe::build().unwrap();
    let mut expected = expected_physical_with_f32(&common);
    for c in convolutions {
        expected.insert(
            c.name,
            (
                Dtype::Bf16,
                vec![c.channels, c.taps, 1],
                c.channels * c.taps * 2,
            ),
        );
    }
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), json!({"format":"mlx"}));
    let mut offset = 0;
    for (name, (dtype, shape, bytes)) in expected {
        let dtype = match dtype {
            Dtype::U32 => "U32",
            Dtype::Bf16 => "BF16",
            Dtype::F32 => "F32",
        };
        header.insert(
            name,
            json!({"dtype":dtype,"shape":shape,"data_offsets":[offset,offset+bytes]}),
        );
        offset += bytes;
    }
    assert_eq!(offset, BODY_BYTES);
    let header = serde_json::to_vec(&header).unwrap();
    let pins = ArtifactPins {
        config_bytes: config.len(),
        config_sha256: Sha256::digest(&config).into(),
        header_bytes: header.len(),
        header_sha256: Sha256::digest(&header).into(),
        container_bytes: (8 + header.len() + BODY_BYTES) as u64,
        container_sha256: Sha256::digest(b"unread synthetic metadata container").into(),
    };
    (config, header, pins)
}
fn plan() -> Plan {
    let (config, header, pins) = documents();
    inspect(
        &config,
        &header,
        pins.container_bytes,
        ModalityPolicy::TextOnly,
        &pins,
    )
    .unwrap()
}
fn mutated(config: Value, header: Value) -> Result<Plan, Error> {
    let config = serde_json::to_vec(&config).unwrap();
    let header = serde_json::to_vec(&header).unwrap();
    layout(&config, &header, documents().2)
}

#[test]
fn distinct_inventory_typed_bytes_and_tied_owner_are_exact() {
    let p = plan();
    assert_eq!(
        (
            p.logical_tensors().len(),
            p.convolutions().len(),
            p.logical_owner_count(),
            p.physical_tensors().len()
        ),
        (302, 18, 320, 694)
    );
    let (mut matrices, mut bf16, mut f32, mut parameters) = (0, 0, 0, 0_u64);
    for t in p.logical_tensors() {
        match (t.shape, t.storage) {
            (Shape::Matrix { rows, columns }, Storage::Affine(l)) => {
                assert_eq!(
                    (
                        l.rows(),
                        l.columns(),
                        l.bits(),
                        l.group_size(),
                        l.metadata_dtype()
                    ),
                    (rows, columns, 4, 64, MetadataDtype::BF16)
                );
                matrices += 1;
                parameters += (rows * columns) as u64;
            }
            (Shape::Vector(n), Storage::Bf16Vector) => {
                bf16 += 1;
                parameters += n as u64;
            }
            (Shape::Vector(n), Storage::F32Vector) => {
                assert!(t.name.ends_with(".linear_attn.A_log"));
                assert_eq!(n, 16);
                f32 += 1;
                parameters += n as u64;
            }
            _ => panic!("unexpected2B logical storage"),
        }
    }
    parameters += p
        .convolutions()
        .iter()
        .map(|c| (c.channels * c.taps) as u64)
        .sum::<u64>();
    assert_eq!(
        (matrices, bf16, f32, parameters),
        (187, 97, 18, 1_881_825_088)
    );
    assert_eq!(
        p.payload_bytes(),
        PayloadBytes {
            affine_codes: 940_638_208,
            affine_metadata: 117_579_776,
            bf16_vectors: 212_032,
            f32_vectors: 1152,
            bf16_convolutions: 884_736,
            total: BODY_BYTES
        }
    );
    assert_eq!(p.source_f32_vector_bytes(), 1152);
    assert_eq!(p.runtime_learned_bytes().unwrap(), 1_060_412_672);
    assert!(std::ptr::eq(p.embedding(), p.tied_head()));
    assert_eq!(
        (p.hidden_size(), p.intermediate_size(), p.vocab_size()),
        (2048, 6144, 248320)
    );
    assert_eq!(
        (
            p.declared_total_positions(),
            p.implemented_total_positions(),
            p.qualified_total_positions()
        ),
        (262144, None, None)
    );
    assert_eq!(p.declared_recurrent_state_dtype(), "float32");
}

#[test]
fn ratio_one_delta_and_six_gated_full_attention_state_owners_are_distinct() {
    let p = plan();
    let (mut delta, mut full) = (0, 0);
    for (i, layer) in p.layers().iter().enumerate() {
        if i % 4 == 3 {
            assert_eq!(
                *layer,
                LayerRecipe::GatedFullAttention(AttentionPlan {
                    kv_owner: full,
                    query_heads: 8,
                    kv_heads: 2,
                    head_dim: 256,
                    rotated_dims: 64,
                    output_gate: true
                })
            );
            let q = &p.physical_tensors()[&format!("{PREFIX}layers.{i}.self_attn.q_proj.weight")];
            assert_eq!(q.shape, [4096, 256]);
            full += 1;
        } else {
            assert_eq!(
                *layer,
                LayerRecipe::GatedDelta(DeltaPlan {
                    recurrent_owner: delta,
                    convolution_owner: delta,
                    key_heads: 16,
                    value_heads: 16,
                    key_dim: 128,
                    value_dim: 128
                })
            );
            let c = &p.convolutions()[delta];
            assert_eq!(
                (c.layer, c.history_owner, c.channels, c.taps),
                (i, delta, 6144, 4)
            );
            assert_eq!(p.physical_tensors()[&c.name].shape, [6144, 4, 1]);
            delta += 1;
        }
    }
    assert_eq!((delta, full), (18, 6));
    assert_eq!(p.rope_sections(), [11, 11, 10]);
    assert_eq!(p.rope_theta(), 10_000_000.0);
}

#[test]
fn all_external_pins_modality_and_bounds_are_required() {
    let (config, header, pins) = documents();
    let mut changed = pins;
    changed.container_bytes += 1;
    assert!(
        inspect(
            &config,
            &header,
            pins.container_bytes,
            ModalityPolicy::TextOnly,
            &changed
        )
        .is_err()
    );
    for field in 0..6 {
        let mut changed = pins;
        match field {
            0 => changed.config_bytes += 1,
            1 => changed.config_sha256[0] ^= 1,
            2 => changed.header_bytes += 1,
            3 => changed.header_sha256[0] ^= 1,
            4 => changed.header_bytes = 512 * 1024 + 1,
            _ => changed.container_sha256 = [0; 32],
        }
        assert!(
            inspect(
                &config,
                &header,
                pins.container_bytes,
                ModalityPolicy::TextOnly,
                &changed
            )
            .is_err()
        );
    }
    assert!(
        inspect(
            &config,
            &header,
            pins.container_bytes,
            ModalityPolicy::Reject,
            &pins
        )
        .is_err()
    );
    assert!(
        inspect(
            &config,
            &header,
            pins.container_bytes + 1,
            ModalityPolicy::TextOnly,
            &pins
        )
        .is_err()
    );
}

#[test]
fn config_geometry_quantization_and_foreign_source_dialects_reject() {
    let (c, h, _) = documents();
    let c: Value = serde_json::from_slice(&c).unwrap();
    let h: Value = serde_json::from_slice(&h).unwrap();
    for (path, value) in [
        ("/text_config/num_hidden_layers", json!(32)),
        ("/text_config/linear_num_value_heads", json!(32)),
        ("/text_config/num_attention_heads", json!(16)),
        ("/text_config/hidden_size", json!(2048.0)),
        ("/text_config/tie_word_embeddings", json!(false)),
        ("/text_config/mamba_ssm_dtype", json!("bfloat16")),
        (
            "/text_config/rope_parameters/partial_rotary_factor",
            json!(0.5),
        ),
        ("/quantization/bits", json!(8)),
        ("/quantization/group_size", json!(32)),
    ] {
        let mut wrong = c.clone();
        *wrong.pointer_mut(path).unwrap() = value;
        assert!(mutated(wrong, h.clone()).is_err(), "{path}");
    }
    let mut wrong = c.clone();
    wrong["quantization_config"] = wrong["quantization"].clone();
    assert!(mutated(wrong, h.clone()).is_err());
    for old in [
        include_bytes!("../../../../tests/fixtures/qwen35-4b-avan-metadata-v1/config.json")
            .as_slice(),
        include_bytes!("../../../../tests/fixtures/qwen35-9b-lukey-metadata-v1/config.json")
            .as_slice(),
    ] {
        assert!(configuration::validate(old).is_err());
    }
}

#[test]
fn f32_opt_in_is_limited_to_the_exact_eighteen_alog_vectors() {
    let (c, h, _) = documents();
    let c: Value = serde_json::from_slice(&c).unwrap();
    let h: Value = serde_json::from_slice(&h).unwrap();
    let alog = format!("{PREFIX}layers.0.linear_attn.A_log");
    let norm = format!("{PREFIX}norm.weight");
    for (name, field, value) in [
        (alog.as_str(), "dtype", json!("BF16")),
        (alog.as_str(), "shape", json!([32])),
        (norm.as_str(), "dtype", json!("F32")),
        (
            "language_model.model.layers.0.linear_attn.conv1d.weight",
            "shape",
            json!([6144, 1, 4]),
        ),
        (
            "language_model.model.embed_tokens.scales",
            "dtype",
            json!("F32"),
        ),
    ] {
        let mut wrong = h.clone();
        wrong[name][field] = value;
        assert!(mutated(c.clone(), wrong).is_err(), "{name}/{field}");
    }
    let value = json!({"dtype":"F32","shape":[16],"data_offsets":[0,64]});
    assert!(crate::mlx::tensor_metadata::descriptor(&value, 64).is_err());
    assert_eq!(descriptor_with_f32(&value, 64).unwrap().dtype, Dtype::F32);
    assert!(
        descriptor_with_f32(
            &json!({"dtype":"F32","shape":[4,4],"data_offsets":[0,64]}),
            64
        )
        .is_err()
    );
}

#[test]
fn missing_extra_alias_gap_duplicate_and_overflow_controls_reject() {
    let (c, h, _) = documents();
    let c: Value = serde_json::from_slice(&c).unwrap();
    let h: Value = serde_json::from_slice(&h).unwrap();
    let name = format!("{PREFIX}layers.0.linear_attn.A_log");
    let mut missing = h.clone();
    missing.as_object_mut().unwrap().remove(&name);
    assert!(mutated(c.clone(), missing).is_err());
    for extra in [
        "language_model.lm_head.weight",
        "vision_tower.weight",
        "mtp.weight",
    ] {
        let mut wrong = h.clone();
        wrong[extra] = h[&name].clone();
        assert!(mutated(c.clone(), wrong).is_err());
    }
    let mut alias = h.clone();
    alias[&name]["data_offsets"] =
        h["language_model.model.layers.1.linear_attn.A_log"]["data_offsets"].clone();
    assert!(mutated(c.clone(), alias).is_err());
    let mut gap = h.clone();
    let offset = gap[&name]["data_offsets"][0].as_u64().unwrap();
    gap[&name]["data_offsets"] = json!([offset + 4, offset + 68]);
    assert!(mutated(c.clone(), gap).is_err());
    let mut duplicate = serde_json::to_vec(&h).unwrap();
    duplicate.pop();
    duplicate.extend(format!(",{name:?}:{} }}", h[&name]).bytes());
    assert!(layout(&serde_json::to_vec(&c).unwrap(), &duplicate, documents().2).is_err());
    let mut p = plan();
    p.bytes.affine_codes = usize::MAX;
    assert!(p.runtime_learned_bytes().is_err());
}
