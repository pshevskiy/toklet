use super::*;
use crate::weights::content::{DenseProfile, SourceFixture};

#[test]
fn mistral_public_open_rejects_budget_before_files() {
    let absent = Path::new("/toklet-absent-mistral-source");
    for memory_limit_bytes in [0, weights::INITIAL_MEMORY_LIMIT_BYTES + 1] {
        assert!(matches!(
            Source::open_mistral_audited(
                absent,
                LoadOptions {
                    memory_limit_bytes,
                    ..LoadOptions::default()
                }
            ),
            Err(Error::Invalid(_))
        ));
    }
    assert!(
        matches!(Source::open_mistral_audited(absent, LoadOptions {memory_limit_bytes: METADATA_ALLOWANCE+STREAM_BYTES-1, ..LoadOptions::default()}), Err(Error::Budget {required, ..}) if required == METADATA_ALLOWANCE+STREAM_BYTES)
    );
}

#[test]
fn ordinary_mistral_layout_is_not_consumed_identity() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mlx-mistral-v1/untied");
    let options = LoadOptions {
        context_limit: 16,
        memory_limit_bytes: 256 * 1024 * 1024,
    };
    let source = Source::open_dense(&path, options).unwrap();
    assert_eq!(source.architecture(), Architecture::Mistral);
    assert!(!source.tied_output());
    assert!(source.source.info("lm_head.weight").is_some());
    assert!(
        !source.source.names().any(|name| name.ends_with(".bias")
            || name.contains("q_norm")
            || name.contains("k_norm"))
    );
    assert!(source.dense_content_identity().is_none());
    assert!(source.mistral_content_identity().is_none());
    assert!(source.qwen_content_identity().is_none());
    assert!(Source::open_mistral_audited(&path, options).is_err());
}

#[test]
fn selected_dense_audit_rejects_cross_architecture_before_index() {
    let f = SourceFixture::new();
    let absent = Path::new("/toklet-absent-mistral-index");
    let options = LoadOptions {
        context_limit: 16,
        memory_limit_bytes: 256 * 1024 * 1024,
    };
    let mistral = include_str!("../../../tests/fixtures/mlx-mistral-v1/untied/config.json");
    let qwen = include_str!("../../../tests/fixtures/mlx-qwen-v1/model/config.json");
    for (profile, document) in [
        (DenseProfile::Mistral, qwen),
        (
            DenseProfile::Qwen(weights::content::QwenProfile::Qwen2),
            mistral,
        ),
        (
            DenseProfile::Qwen(weights::content::QwenProfile::Qwen3),
            mistral,
        ),
    ] {
        assert!(
            matches!(Source::open_document(absent,options,true,serde_json::from_str(document).unwrap(),None,Some((profile,f.config_receipt()))),Err(Error::Invalid(message)) if message=="Dense source audit profile mismatch")
        );
    }
}

#[test]
fn mistral_requires_explicit_full_attention_and_bias_free_layout() {
    let mut original: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/mlx-mistral-v1/untied/config.json"
    ))
    .unwrap();
    let options = LoadOptions {
        context_limit: 16,
        memory_limit_bytes: 256 * 1024 * 1024,
    };
    original.as_object_mut().unwrap().remove("quantization");
    assert!(dense::config(original.clone(), options).is_ok());
    let mut missing = original.clone();
    missing.as_object_mut().unwrap().remove("sliding_window");
    assert!(dense::config(missing, options).is_err());
    for (key, value) in [
        ("sliding_window", serde_json::json!(4096)),
        ("attention_bias", serde_json::json!(true)),
        ("mlp_bias", serde_json::json!(true)),
        ("use_sliding_window", serde_json::json!(true)),
    ] {
        let mut bad = original.clone();
        bad[key] = value;
        assert!(dense::config(bad, options).is_err(), "{key}");
    }
}

#[test]
#[ignore = "Explicit cached pinned Mistral source-only stream; no model assembly, codec, GPU or generation"]
fn pinned_mistral_audited_source_streams_complete_original_payload() {
    let mut source = Source::open_mistral_audited(
        Path::new("models/MLX-Mistral-7B-Instruct-v0.3-4bit-a4b8f87"),
        LoadOptions {
            context_limit: 512,
            memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES,
        },
    )
    .unwrap();
    assert_eq!(source.architecture(), Architecture::Mistral);
    assert_eq!(source.config().max_positions, 512);
    assert_eq!(source.declared_positions(), 32768);
    assert!(!source.tied_output());
    assert!(source.mistral_content_identity().is_none());
    let names: Vec<_> = source.source.names().map(str::to_owned).collect();
    assert_eq!(names.len(), 743);
    assert!(names.iter().any(|name| name == "lm_head.weight"));
    assert!(
        !names.iter().any(|name| name.ends_with(".bias")
            || name.contains("q_norm")
            || name.contains("k_norm"))
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
    let receipt = source.mistral_content_identity().unwrap();
    assert_eq!(source.dense_content_identity(), Some(receipt));
    assert!(source.qwen_content_identity().is_none());
    assert!(source.qwen2_content_identity().is_none());
    assert!(source.qwen3_content_identity().is_none());
    assert!(source.content_identity().is_none());
    assert_eq!(receipt.container_bytes(), 4_077_480_215);
    assert_eq!(bytes, 4_077_395_968);
    assert_eq!(receipt.container_physical_ranges(), 745);
    assert_eq!(receipt.verified_container_reads(), 745);
    let owned = source.content_audit_owned_bytes().unwrap();
    assert!(owned < METADATA_ALLOWANCE);
    assert!(source.visit_float("absent", |_| Ok(())).is_err());
    assert!(source.mistral_content_identity().is_none());
    assert!(source.dense_content_identity().is_none());
    println!(
        "TOKLET_MISTRAL_CONSUMED_SOURCE {}",
        serde_json::json!({"passed":true,"source_only":true,"model_assembled":false,"codec_paired":false,"expected_manifest_sha256":receipt.expected_manifest_sha256(),"config_sha256":receipt.config_sha256(),"index_sha256":receipt.index_sha256(),"container_sha256":receipt.container_sha256(),"container_bytes":receipt.container_bytes(),"payload_bytes_delivered":bytes,"tensor_ranges":names.len(),"physical_ranges":receipt.container_physical_ranges(),"verified_reads":receipt.verified_container_reads(),"retained_audit_bytes":owned,"qwen_identities_absent":true,"after_invalid_public_read_identity_absent":true})
    );
}
