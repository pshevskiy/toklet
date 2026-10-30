use super::*;
use crate::weights::content::{QwenProfile, SourceFixture};

#[test]
fn qwen2_public_open_rejects_budget_before_files() {
    let path = Path::new("/toklet-absent-qwen2-source");
    for memory_limit_bytes in [0, weights::INITIAL_MEMORY_LIMIT_BYTES + 1] {
        assert!(matches!(
            Source::open_qwen2_audited(
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
        Source::open_qwen2_audited(
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
fn ordinary_qwen_geometry_never_becomes_audited_content_authority() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mlx-qwen-v1/model");
    let options = LoadOptions {
        context_limit: 16,
        memory_limit_bytes: 256 * 1024 * 1024,
    };
    let source = Source::open_dense(&path, options).unwrap();
    assert_eq!(source.architecture(), Architecture::Qwen2);
    assert!(source.qwen_content_identity().is_none());
    assert!(source.qwen2_content_identity().is_none());
    assert!(source.qwen3_content_identity().is_none());
    // Same valid architecture does not replace the exact pinned config bytes.
    assert!(Source::open_qwen2_audited(&path, options).is_err());
    assert!(Source::open_qwen3_audited(&path, options).is_err());
}

#[test]
fn selected_qwen_audit_rejects_wrong_architecture_before_source_metadata_open() {
    let f = SourceFixture::new();
    let options = LoadOptions {
        context_limit: 16,
        memory_limit_bytes: 256 * 1024 * 1024,
    };
    let absent = Path::new("/toklet-absent-qwen-audit-index");
    for (profile, config) in [
        (
            QwenProfile::Qwen3,
            include_str!("../../../tests/fixtures/mlx-qwen-v1/model/config.json"),
        ),
        (
            QwenProfile::Qwen2,
            include_str!("../../../tests/fixtures/mlx-qwen3-v1/tied/config.json"),
        ),
    ] {
        let value = serde_json::from_str(config).unwrap();
        assert!(
            matches!(Source::open_document(absent,options,true,value,None,Some((weights::content::DenseProfile::Qwen(profile),f.config_receipt()))),
            Err(Error::Invalid(message)) if message=="Dense source audit profile mismatch")
        );
    }
}

#[test]
#[ignore = "Explicit cached pinned Qwen2.5 source-only stream; no model assembly, codec, GPU or generation"]
fn pinned_qwen2_audited_source_streams_complete_original_payload() {
    let mut source = Source::open_qwen2_audited(
        Path::new("models/MLX-Qwen2.5-0.5B-Instruct-8bit-6474d4c"),
        LoadOptions {
            context_limit: 512,
            memory_limit_bytes: 1024 * 1024 * 1024,
        },
    )
    .unwrap();
    assert_eq!(source.architecture(), Architecture::Qwen2);
    assert_eq!(source.config().max_positions, 512);
    assert_eq!(source.declared_positions(), 32768);
    assert!(source.tied_output());
    assert!(source.qwen_content_identity().is_none());
    assert!(source.qwen2_content_identity().is_none());
    assert!(source.qwen3_content_identity().is_none());
    let names: Vec<_> = source.source.names().map(str::to_owned).collect();
    assert_eq!(names.len(), 628);
    assert!(
        names
            .iter()
            .any(|name| name == "model.layers.0.self_attn.q_proj.bias")
    );
    assert!(
        !names
            .iter()
            .any(|name| name.contains("q_norm") || name.contains("k_norm"))
    );
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
    let receipt = source.qwen2_content_identity().unwrap();
    assert_eq!(source.qwen_content_identity(), Some(receipt));
    assert!(source.qwen3_content_identity().is_none());
    assert!(source.content_identity().is_none());
    assert_eq!(receipt.container_bytes(), 525045902);
    assert_eq!(bytes, 524976896);
    assert_eq!(receipt.container_physical_ranges(), 630);
    assert!(receipt.verified_container_reads() >= 630);
    assert!(source.content_audit_owned_bytes().unwrap() < METADATA_ALLOWANCE);
    assert!(source.visit_float("absent", |_| Ok(())).is_err());
    assert!(source.qwen2_content_identity().is_none());
    assert!(source.qwen_content_identity().is_none());
    println!(
        "TOKLET_QWEN2_CONSUMED_SOURCE {}",
        serde_json::json!({
            "passed":true,"source_only":true,"model_assembled":false,"codec_paired":false,
            "expected_manifest_sha256":receipt.expected_manifest_sha256(),"config_sha256":receipt.config_sha256(),
            "index_sha256":receipt.index_sha256(),"container_sha256":receipt.container_sha256(),
            "container_bytes":receipt.container_bytes(),"payload_bytes_delivered":bytes,"tensor_ranges":names.len(),
            "physical_ranges":receipt.container_physical_ranges(),"verified_reads":receipt.verified_container_reads(),
            "retained_audit_bytes":source.content_audit_owned_bytes(),"qwen3_identity_absent":true,
            "after_invalid_public_read_identity_absent":true
        })
    );
}
