use super::*;
use crate::weights::content::SourceFixture;

fn wrapper(source: weights::Source) -> Source {
    let tensors = BTreeMap::from([
        (
            "a".into(),
            TensorSpec {
                shape: vec![1, 2],
                storage: Storage::Float(MetadataDtype::F32),
                payload_bytes: 8,
            },
        ),
        (
            "b".into(),
            TensorSpec {
                shape: vec![1, 1],
                storage: Storage::Float(MetadataDtype::F32),
                payload_bytes: 4,
            },
        ),
    ]);
    Source {
        declared_positions: 8,
        declared_sliding_window: None,
        config: Config {
            vocab_size: 1,
            hidden_size: 2,
            intermediate_size: 2,
            num_layers: 1,
            query_heads: 1,
            kv_heads: 1,
            head_dim: 2,
            max_positions: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.,
        },
        architecture: Architecture::Qwen3,
        tied_output: true,
        position_plan: crate::model::PositionPlan::Ordinary,
        logical: logical::plan(&tensors, &[]).unwrap(),
        tensors,
        source,
        tensor_payload_bytes: 12,
        admitted_load_bytes: 12 + METADATA_ALLOWANCE + STREAM_BYTES,
    }
}

#[test]
fn audited_qwen_public_wrappers_poison_on_invalid_type_name_and_range() {
    let f = SourceFixture::new();
    for case in 0..4 {
        let mut source = wrapper(f.source());
        source.visit_logical_float("a", |_| Ok(())).unwrap();
        source.visit_logical_float("b", |_| Ok(())).unwrap();
        assert!(source.qwen_content_identity().is_some());
        let rejected = match case {
            0 => source.visit_float("absent", |_| Ok(())).is_err(),
            1 => source.read_affine("a", &mut [], &mut [], &mut []).is_err(),
            2 => source.visit_float_rows("a", 0..1, |_| Ok(())).is_err(),
            _ => source
                .read_affine_rows("a", 0..1, &mut [], &mut [], &mut [])
                .is_err(),
        };
        assert!(rejected);
        assert!(source.qwen_content_identity().is_none());
        assert!(source.visit_float("a", |_| Ok(())).is_err());
    }
}

#[test]
fn pinned_qwen_open_rejects_unadmitted_budget_before_source_open() {
    let path = Path::new("/toklet-nonexistent-qwen-source");
    for memory_limit_bytes in [0, weights::INITIAL_MEMORY_LIMIT_BYTES + 1] {
        assert!(matches!(
            Source::open_qwen3_audited(
                path,
                LoadOptions {
                    memory_limit_bytes,
                    ..LoadOptions::default()
                }
            ),
            Err(Error::Invalid(_))
        ));
    }
    assert!(matches!(
        Source::open_qwen3_audited(
            path,
            LoadOptions {
                memory_limit_bytes: METADATA_ALLOWANCE + STREAM_BYTES - 1,
                ..LoadOptions::default()
            }
        ),
        Err(Error::Budget { .. })
    ));
}

#[test]
#[ignore = "Explicit cached pinned Qwen3 source-only full stream; no model assembly, codec, GPU or generation"]
fn pinned_qwen3_audited_source_streams_complete_original_payload() {
    let mut source = Source::open_qwen3_audited(
        Path::new("models/MLX-Qwen3-0.6B-8bit-11de968"),
        LoadOptions {
            context_limit: 512,
            memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES,
        },
    )
    .unwrap();
    assert_eq!(source.architecture(), Architecture::Qwen3);
    assert_eq!(source.config().max_positions, 512);
    assert_eq!(source.declared_positions(), 40960);
    assert!(source.qwen3_content_identity().is_none());
    let names: Vec<_> = source.source.names().map(str::to_owned).collect();
    assert!(names.len() + 2 <= crate::weights::content::MAX_RANGES);
    let mut bytes = 0_u64;
    for name in &names {
        source
            .source
            .visit_raw(name, |chunk| {
                bytes = bytes.checked_add(chunk.len() as u64).unwrap();
                Ok(())
            })
            .unwrap();
    }
    let receipt = source.qwen3_content_identity().unwrap();
    assert_eq!(receipt.container_bytes(), 633442994);
    assert_eq!(bytes, 633364480);
    assert_eq!(receipt.container_physical_ranges(), names.len() + 2);
    assert!(receipt.verified_container_reads() >= names.len() + 2);
    assert!(source.content_identity().is_none());
    let before = receipt;
    assert!(source.visit_float("absent", |_| Ok(())).is_err());
    assert!(source.qwen3_content_identity().is_none());
    println!(
        "TOKLET_QWEN_CONSUMED_SOURCE {}",
        serde_json::json!({
            "passed":true,"source_only":true,"model_assembled":false,"codec_paired":false,
            "expected_manifest_sha256":before.expected_manifest_sha256(),"config_sha256":before.config_sha256(),
            "index_sha256":before.index_sha256(),"container_sha256":before.container_sha256(),
            "container_bytes":before.container_bytes(),"payload_bytes_delivered":bytes,"tensor_ranges":names.len(),
            "physical_ranges":before.container_physical_ranges(),"verified_reads":before.verified_container_reads(),
            "retained_audit_bytes":source.content_audit_owned_bytes(),"after_invalid_public_read_identity_absent":true
        })
    );
}
