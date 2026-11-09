use super::*;
use crate::mlx::{
    affine::MetadataDtype,
    qwen35::{AttentionPlan, DeltaPlan, Shape, Storage},
};
const CONFIG: &[u8] =
    include_bytes!("../../../../tests/fixtures/qwen35-9b-lukey-metadata-v1/config.json");
const HEADER: &[u8] =
    include_bytes!("../../../../tests/fixtures/qwen35-9b-lukey-metadata-v1/header.json");
fn plan() -> Plan {
    inspect(CONFIG, HEADER, CONTAINER_BYTES, ModalityPolicy::TextOnly).unwrap()
}
fn documents() -> (Value, Value) {
    (
        serde_json::from_slice(CONFIG).unwrap(),
        serde_json::from_slice(HEADER).unwrap(),
    )
}
fn changed(c: &Value, h: &Value) -> Result<Plan, Error> {
    layout(
        &serde_json::to_vec(c).unwrap(),
        &serde_json::to_vec(h).unwrap(),
        CONTAINER_BYTES,
    )
}

#[test]
fn independent_inventory_accounts_for_both_original_output_owners() {
    let p = plan();
    let fixture: Value = serde_json::from_slice(include_bytes!(
        "../../../../tests/fixtures/qwen35-9b-lukey-metadata-v1/inventory.json"
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
    assert_eq!(
        (
            p.physical_tensors().len(),
            p.logical_tensors().len(),
            p.convolutions().len(),
            p.logical_owner_count()
        ),
        (927, 403, 24, 427)
    );
    let (mut matrices, mut vectors, mut parameters) = (0, 0, 0u64);
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
                parameters += rows as u64 * columns as u64;
            }
            (Shape::Vector(n), Storage::Bf16Vector) => {
                vectors += 1;
                parameters += n as u64;
            }
            _ => panic!("unexpected source storage"),
        }
    }
    parameters += p
        .convolutions()
        .iter()
        .map(|c| (c.channels * c.taps) as u64)
        .sum::<u64>();
    assert_eq!((matrices, vectors, parameters), (250, 153, 8_953_803_264));
    assert_eq!(parameters, fixture["logical_parameters"].as_u64().unwrap());
    assert_eq!(
        p.payload_bytes(),
        PayloadBytes {
            affine_codes: 4_476_370_944,
            affine_metadata: 559_546_368,
            bf16_vectors: 549_888,
            bf16_convolutions: 1_572_864,
            total: 5_038_040_064
        }
    );
    assert_eq!(p.embedding().name, fixture["embedding_name"]);
    assert_eq!(p.output_head().name, fixture["head_name"]);
    assert!(!std::ptr::eq(p.embedding(), p.output_head()));
    for t in [p.embedding(), p.output_head()] {
        assert_eq!(
            t.shape,
            Shape::Matrix {
                rows: 248320,
                columns: 4096
            }
        );
    }
    for suffix in ["weight", "scales", "biases"] {
        let a = &p.physical_tensors()[&format!("language_model.model.embed_tokens.{suffix}")].range;
        let b = &p.physical_tensors()[&format!("language_model.lm_head.{suffix}")].range;
        assert!(a.end <= b.start || b.end <= a.start);
    }
    assert_eq!(
        (p.hidden_size(), p.intermediate_size(), p.vocab_size()),
        (4096, 12288, 248320)
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
fn hybrid_state_owners_and_rotary_declarations_remain_separate() {
    let p = plan();
    let (mut delta, mut attention) = (0, 0);
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
            assert_eq!(
                p.physical_tensors()
                    [&format!("language_model.model.layers.{i}.self_attn.q_proj.weight")]
                    .shape,
                [8192, 512]
            );
            attention += 1;
        } else {
            assert_eq!(
                *layer,
                LayerRecipe::GatedDelta(DeltaPlan {
                    recurrent_owner: delta,
                    convolution_owner: delta,
                    key_heads: 16,
                    value_heads: 32,
                    key_dim: 128,
                    value_dim: 128
                })
            );
            let c = &p.convolutions()[delta];
            assert_eq!(
                (c.layer, c.history_owner, c.channels, c.taps),
                (i, delta, 8192, 4)
            );
            assert_eq!(p.physical_tensors()[&c.name].shape, [8192, 4, 1]);
            delta += 1;
        }
    }
    assert_eq!((delta, attention), (24, 8));
    assert_eq!(p.rope_sections(), [11, 11, 10]);
    assert_eq!(p.rope_theta(), 10_000_000.);
}

#[test]
fn full_unaligned_payload_ranges_are_metadata_without_typed_casts() {
    let p = plan();
    assert_eq!(p.payload_offset(), 120889);
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
    assert!(cursor > u32::MAX as usize);
}

#[test]
fn public_identity_and_explicit_modality_are_required() {
    let mut c = CONFIG.to_vec();
    c[0] ^= 1;
    let mut h = HEADER.to_vec();
    h[100] ^= 1;
    for (c, h, n, mode) in [
        (
            c.as_slice(),
            HEADER,
            CONTAINER_BYTES,
            ModalityPolicy::TextOnly,
        ),
        (
            CONFIG,
            h.as_slice(),
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
        assert!(inspect(c, h, n, mode).is_err());
    }
}

#[test]
fn source_graph_mutations_do_not_bypass_the_untied_recipe() {
    let (c, h) = documents();
    for (path, value) in [
        ("/hidden_size", json!(2560)),
        ("/hidden_size", json!(4096.0)),
        ("/intermediate_size", json!(9216)),
        ("/tie_word_embeddings", json!(true)),
        ("/layer_types/0", json!("full_attention")),
        ("/num_hidden_layers", json!(31)),
        ("/linear_num_value_heads", json!(16)),
        ("/linear_conv_kernel_dim", json!(3)),
        ("/mamba_ssm_dtype", json!("bfloat16")),
        ("/attn_output_gate", json!(false)),
        ("/rope_parameters/mrope_interleaved", json!(false)),
        ("/partial_rotary_factor", json!(0.5)),
        ("/quantization/bits", json!(8)),
        ("/quantization_config/group_size", json!(32)),
        ("/mtp_num_hidden_layers", json!(2)),
        ("/architectures/0", json!("Qwen3ForCausalLM")),
        ("/eos_token_id", json!([248044])),
        ("/rms_norm_eps", json!(1e-5)),
    ] {
        let mut changed_c = c.clone();
        *changed_c.pointer_mut(path).unwrap() = value;
        assert!(changed(&changed_c, &h).is_err(), "{path}");
    }
    let mut c = c;
    c["vision_config"] = json!({});
    assert!(changed(&c, &h).is_err());
}

#[test]
fn changed_head_storage_aliases_and_owner_gaps_are_rejected() {
    let (c, h) = documents();
    for (name, field, value) in [
        (HEAD, "shape", json!([248320, 320])),
        (HEAD, "dtype", json!("BF16")),
        ("language_model.lm_head.scales", "dtype", json!("F32")),
        (
            "language_model.lm_head.biases",
            "shape",
            json!([248320, 128]),
        ),
        (
            "language_model.model.layers.0.linear_attn.conv1d.weight",
            "shape",
            json!([8192, 1, 4]),
        ),
        (
            "language_model.model.layers.0.linear_attn.A_log",
            "dtype",
            json!("F32"),
        ),
        (HEAD, "data_offsets", json!([false, 1])),
        (HEAD, "extra", json!(true)),
    ] {
        let mut mutated = h.clone();
        mutated[name][field] = value;
        assert!(changed(&c, &mutated).is_err(), "{name}/{field}");
    }
    let mut alias = h.clone();
    alias[HEAD]["data_offsets"] =
        h["language_model.model.embed_tokens.weight"]["data_offsets"].clone();
    assert!(changed(&c, &alias).is_err());
    for name in [
        HEAD,
        "language_model.lm_head.scales",
        "language_model.lm_head.biases",
    ] {
        let mut missing = h.clone();
        missing.as_object_mut().unwrap().remove(name);
        assert!(changed(&c, &missing).is_err());
    }
    let mut gap = h.clone();
    let a = h[HEAD]["data_offsets"][0].as_u64().unwrap();
    let b = h[HEAD]["data_offsets"][1].as_u64().unwrap();
    gap[HEAD]["data_offsets"] = json!([a + 4, b + 4]);
    assert!(changed(&c, &gap).is_err());
}

#[test]
fn duplicate_unknown_and_unbounded_metadata_never_publishes_a_plan() {
    let (c, h) = documents();
    let mut raw = serde_json::to_vec(&h).unwrap();
    raw.pop();
    raw.extend(format!(",{HEAD:?}:{} }}", h[HEAD]).bytes());
    assert!(layout(CONFIG, &raw, CONTAINER_BYTES).is_err());
    let mut raw = serde_json::to_vec(&c).unwrap();
    raw.pop();
    raw.extend(b",\"tie_word_embeddings\":false}");
    assert!(layout(&raw, HEADER, CONTAINER_BYTES).is_err());
    for name in [
        "language_model.model.layers.32.extra.weight",
        "vision_tower.weight",
        "mtp.weight",
    ] {
        let mut wrong = h.clone();
        let row = wrong.as_object_mut().unwrap().remove(HEAD).unwrap();
        wrong[name] = row;
        assert!(changed(&c, &wrong).is_err());
    }
    let mut wrong = h;
    wrong["__metadata__"] = json!({"format":"pt"});
    assert!(changed(&c, &wrong).is_err());
    assert!(layout(CONFIG, HEADER, (HEADER_BYTES + 7) as u64).is_err());
    assert!(layout(CONFIG, HEADER, u64::MAX).is_err());
    assert!(layout(&vec![b' '; 64 * 1024 + 1], HEADER, CONTAINER_BYTES).is_err());
    assert!(layout(CONFIG, &vec![b' '; 512 * 1024 + 1], CONTAINER_BYTES).is_err());
}

#[test]
fn four_b_source_and_payload_admission_do_not_expand_by_family_name() {
    let c = include_bytes!("../../../../tests/fixtures/qwen35-4b-avan-metadata-v1/config.json");
    let h = include_bytes!("../../../../tests/fixtures/qwen35-4b-avan-metadata-v1/header.json");
    let p = crate::mlx::qwen35::inspect(
        c,
        h,
        crate::mlx::qwen35::CONTAINER_BYTES,
        ModalityPolicy::TextOnly,
    )
    .unwrap();
    assert!(std::ptr::eq(p.embedding(), p.tied_head()));
    assert_eq!(p.hidden_size(), 2560);
    assert_eq!(p.physical_tensors().len(), 924);
    assert!(
        inspect(
            c,
            h,
            crate::mlx::qwen35::CONTAINER_BYTES,
            ModalityPolicy::TextOnly
        )
        .is_err()
    );
    assert!(
        crate::mlx::qwen35::inspect(CONFIG, HEADER, CONTAINER_BYTES, ModalityPolicy::TextOnly)
            .is_err()
    );
    assert!(
        crate::mlx::qwen3::config(
            crate::mlx::qwen::parse_json(CONFIG).unwrap(),
            crate::weights::LoadOptions::default()
        )
        .is_err()
    );
}

#[test]
#[ignore = "explicit pinned9B metadata-only release measurement; no tensor payload or GPU"]
fn actual_metadata_probe() {
    use std::{
        fs::{File, OpenOptions},
        io::Read,
        path::{Path, PathBuf},
        time::Instant,
    };
    fn open(path: &Path) -> File {
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        }
        let f = options.open(path).unwrap();
        assert!(f.metadata().unwrap().is_file());
        f
    }
    let folder = PathBuf::from(std::env::var_os("TOKLET_QWEN35_9B_METADATA_DIRECTORY").unwrap());
    println!();
    for ordinal in 0..13 {
        let start = Instant::now();
        let mut config = vec![0u8; CONFIG_BYTES];
        let mut f = open(&folder.join("config.json"));
        assert_eq!(f.metadata().unwrap().len(), CONFIG_BYTES as u64);
        f.read_exact(&mut config).unwrap();
        let mut f = open(&folder.join("model.safetensors"));
        assert_eq!(f.metadata().unwrap().len(), CONTAINER_BYTES);
        let mut prefix = [0u8; 8];
        f.read_exact(&mut prefix).unwrap();
        assert_eq!(u64::from_le_bytes(prefix), HEADER_BYTES as u64);
        let mut header = vec![0u8; HEADER_BYTES];
        f.read_exact(&mut header).unwrap();
        let p = inspect(&config, &header, CONTAINER_BYTES, ModalityPolicy::TextOnly).unwrap();
        assert_eq!(
            (
                p.logical_tensors().len(),
                p.convolutions().len(),
                p.physical_tensors().len()
            ),
            (403, 24, 927)
        );
        assert!(!std::ptr::eq(p.embedding(), p.output_head()));
        drop(p);
        let ns = start.elapsed().as_nanos();
        println!(
            "QWEN35_9B_METADATA {}",
            json!({"ordinal":ordinal,"warmup":ordinal<3,"elapsed_ns":ns,"metadata_bytes_read":CONFIG_BYTES+HEADER_BYTES+8,"payload_bytes_read":0,"execution_qualified":false,"memory_fit_qualified":false,"managed_adoption_qualified":false})
        );
    }
}
