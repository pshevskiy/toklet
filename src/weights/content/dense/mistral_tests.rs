use super::tests::SourceFixture;
use super::*;

#[test]
fn mistral_manifest_and_qwen_compatibility_keep_exact_scopes() {
    let m = DenseProfile::Mistral.manifest().unwrap();
    assert_eq!(
        m.identity()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>(),
        "0f543bc8679995fe028857674e6e327d5baa6832453587adda92b0c45e4f4a26"
    );
    assert_eq!(m.files().len(), 6);
    assert_eq!(m.files()[1].bytes, 4_077_480_215);
    assert_eq!(m.files()[4].bytes, 138_039);
    assert_ne!(
        m.identity(),
        QwenProfile::Qwen2.manifest().unwrap().identity()
    );
    assert_ne!(
        m.identity(),
        QwenProfile::Qwen3.manifest().unwrap().identity()
    );
    // Compile-time compatibility of every old const method, with no model authority.
    const fn legacy(value: &super::super::QwenSourceContentIdentity) -> u64 {
        let _ = value.expected_manifest_sha256();
        let _ = value.config_sha256();
        let _ = value.index_sha256();
        let _ = value.container_sha256();
        let _ = value.container_physical_ranges();
        let _ = value.verified_container_reads();
        value.container_bytes()
    }
    let f = SourceFixture::new();
    let mut source = f.source();
    source.visit_raw("a", |_| Ok(())).unwrap();
    source.visit_raw("b", |_| Ok(())).unwrap();
    let value: super::super::QwenSourceContentIdentity = source.dense_content_identity().unwrap();
    assert_eq!(legacy(&value), value.container_bytes());
}

#[test]
fn dense_scope_filters_never_broaden_qwen_and_poison_stays_sticky() {
    for profile in [
        DenseProfile::Qwen(QwenProfile::Qwen2),
        DenseProfile::Qwen(QwenProfile::Qwen3),
        DenseProfile::Mistral,
    ] {
        let f = SourceFixture::new();
        let mut source = f.source();
        source.visit_raw("a", |_| Ok(())).unwrap();
        source.visit_raw("b", |_| Ok(())).unwrap();
        // Synthetic scope-selector control only: no assertion that fixture bytes
        // authenticate a real distribution. Production has no receipt setter.
        let anchor = profile.manifest().unwrap().identity();
        let audit = source.dense_audit.as_mut().unwrap();
        audit.config.manifest = anchor;
        audit.index.manifest = anchor;
        audit.container.receipt.manifest = anchor;
        assert!(source.dense_content_identity().is_some());
        for candidate in [
            DenseProfile::Qwen(QwenProfile::Qwen2),
            DenseProfile::Qwen(QwenProfile::Qwen3),
            DenseProfile::Mistral,
        ] {
            assert_eq!(
                source.dense_profile_content_identity(candidate).is_some(),
                candidate == profile
            );
        }
        assert_eq!(
            source.qwen_content_identity().is_some(),
            matches!(profile, DenseProfile::Qwen(_))
        );
        assert_eq!(
            source
                .qwen_profile_content_identity(QwenProfile::Qwen2)
                .is_some(),
            profile == DenseProfile::Qwen(QwenProfile::Qwen2)
        );
        assert_eq!(
            source
                .qwen_profile_content_identity(QwenProfile::Qwen3)
                .is_some(),
            profile == DenseProfile::Qwen(QwenProfile::Qwen3)
        );
        assert!(source.visit_raw_range("a", 0..8, |_| Ok(())).is_err());
        assert!(source.dense_content_identity().is_none());
        assert!(source.qwen_content_identity().is_none());
        assert!(source.dense_profile_content_identity(profile).is_none());
        assert!(source.visit_raw("a", |_| Ok(())).is_err());
    }
}

#[test]
fn mistral_cross_manifest_config_rejected_before_file_open() {
    let absent = Path::new("/toklet-absent-dense-audit");
    for from in [
        DenseProfile::Qwen(QwenProfile::Qwen2),
        DenseProfile::Qwen(QwenProfile::Qwen3),
        DenseProfile::Mistral,
    ] {
        let m = from.manifest().unwrap();
        let receipt = FileDelivery {
            manifest: m.identity(),
            file: m.files()[0].sha256,
            name: sha2::Sha256::digest(b"config.json").into(),
            bytes: m.files()[0].bytes,
            ranges: 1,
            reads: 1,
        };
        for to in [
            DenseProfile::Qwen(QwenProfile::Qwen2),
            DenseProfile::Qwen(QwenProfile::Qwen3),
            DenseProfile::Mistral,
        ] {
            if from == to {
                continue;
            }
            assert!(
                matches!(DenseAudit::open_source(to, absent, receipt, 6*1024*1024*1024), Err(Error::Invalid(message)) if message == "Dense source config receipt mismatch")
            );
        }
    }
}
