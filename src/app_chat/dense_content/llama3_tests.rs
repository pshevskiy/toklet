use super::tests::synthetic_source;
use super::*;
#[test]
fn llama3_admission_rejects_cross_profile_before_files() {
    let source = synthetic_source();
    let missing = Path::new("/toklet-absent-llama3-codec");
    for allowance in [
        0,
        Profile::Llama3.allowance() - 1,
        crate::weights::INITIAL_MEMORY_LIMIT_BYTES + 1,
    ] {
        assert!(
            Codec::load_llama3_audited(missing, &source, allowance)
                .err()
                .unwrap()
                .to_string()
                .contains("allowance")
        );
    }
    assert!(
        Codec::load_llama3_audited(missing, &source, Profile::Llama3.allowance())
            .err()
            .unwrap()
            .to_string()
            .contains("Unsupported authenticated source")
    );
    assert_eq!(text_profile(DenseProfile::Llama3), Profile::Llama3);
    assert_eq!(Profile::Llama3.stop_ids(), [128001, 128008, 128009]);
}
#[test]
fn llama3_manifest_matches_independent_source_and_text_pins() {
    let source: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/mlx-llama3-real-v1/manifest.json"
    ))
    .unwrap();
    let text: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/llama3-plain-text-v1/fixture.json"
    ))
    .unwrap();
    let manifest = DenseProfile::Llama3.manifest().unwrap();
    let files = manifest.files();
    assert_eq!(
        files.iter().map(|f| f.name).collect::<Vec<_>>(),
        [
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json"
        ]
    );
    for file in files {
        let expected = source["model"]["file_pins"][file.name]
            .as_str()
            .or_else(|| text["file_pins"][file.name].as_str())
            .unwrap();
        let actual: String = file.sha256.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(actual, expected);
    }
    assert_eq!(files[1].bytes, 1313157436);
}
#[test]
fn llama3_source_upper_rejects_before_open_and_legacy_allowlist_stays_closed() {
    let missing = Path::new("/toklet-absent-llama3-source");
    let error = crate::mlx::qwen::Source::open_llama3_audited(
        missing,
        crate::weights::LoadOptions {
            context_limit: 513,
            memory_limit_bytes: 6 * 1024 * 1024 * 1024,
        },
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("context <=512"));
    // The original profile detector still accepts only its pre-existing config hashes.
    let d = super::tests::Directory::new();
    std::fs::write(d.0.join("config.json"), b"unsupported").unwrap();
    assert!(Profile::load(&d.0).is_err());
}
