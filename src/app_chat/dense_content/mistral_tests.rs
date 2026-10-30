use super::tests::{Directory, fixture, synthetic_source};
use super::*;
use sha2::{Digest, Sha256};
use std::fs;

#[test]
fn mistral_codec_memory_and_policy_caps_are_profile_specific() {
    assert_eq!(text_profile(DenseProfile::Mistral), Profile::Mistral);
    assert_eq!(Profile::Mistral.allowance(), 64 * 1024 * 1024);
    assert_eq!(tokenizer_cap(DenseProfile::Mistral), 4 * 1024 * 1024);
    assert_eq!(policy_cap(DenseProfile::Mistral), 256 * 1024);
    for qwen in [QwenProfile::Qwen2, QwenProfile::Qwen3] {
        assert_eq!(policy_cap(DenseProfile::Qwen(qwen)), 16 * 1024);
    }
    let source = synthetic_source();
    let missing = Path::new("/toklet-absent-mistral-codec");
    for allowance in [
        0,
        Profile::Mistral.allowance() - 1,
        crate::weights::INITIAL_MEMORY_LIMIT_BYTES + 1,
    ] {
        assert!(
            Codec::load_mistral_audited(missing, &source, allowance)
                .err()
                .unwrap()
                .to_string()
                .contains("allowance")
        );
    }
    assert!(
        Codec::load_mistral_audited(missing, &source, Profile::Mistral.allowance())
            .err()
            .unwrap()
            .to_string()
            .contains("Unsupported authenticated source")
    );
    let d = Directory::new();
    let mut files = fixture(&d.0, b"tokenizer");
    let policy = vec![b'x'; 138_039];
    fs::write(d.0.join(files[4].name), &policy).unwrap();
    files[4].bytes = policy.len() as u64;
    files[4].sha256 = Sha256::digest(&policy).into();
    assert!(assets(DenseProfile::Mistral, &d.0, files[0].sha256, &files).is_ok());
    for qwen in [QwenProfile::Qwen2, QwenProfile::Qwen3] {
        assert!(assets(DenseProfile::Qwen(qwen), &d.0, files[0].sha256, &files).is_err());
    }
    files[4].bytes = 256 * 1024 + 1;
    assert!(assets(DenseProfile::Mistral, &d.0, files[0].sha256, &files).is_err());
    files[4].bytes = policy.len() as u64;
    files[3].bytes = 4 * 1024 * 1024 + 1;
    assert!(assets(DenseProfile::Mistral, &d.0, files[0].sha256, &files).is_err());
}

#[test]
fn mistral_same_buffer_parser_has_no_qwen_receipt_and_preserves_const_aliases() {
    // Synthetic parser/role control, not actual artifact authentication.
    let d = Directory::new();
    let json=br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"<s>":1,"</s>":2,"[INST]":3,"[/INST]":4,"hello":5},"unk_token":"<unk>"}}"#;
    let files = fixture(&d.0, json);
    let source = synthetic_source();
    let codec = build(
        DenseProfile::Mistral,
        assets(DenseProfile::Mistral, &d.0, files[0].sha256, &files).unwrap(),
        &source,
    )
    .unwrap();
    const fn old_qwen(
        codec: &Codec,
    ) -> Option<&crate::app_chat::qwen3_content::QwenCodecContentIdentity> {
        codec.qwen3_content_identity()
    }
    assert!(old_qwen(&codec).is_none());
    assert!(codec.qwen2_content_identity().is_none());
    assert!(codec.content_identity().is_none());
    let receipt = codec.mistral_content_identity().unwrap();
    assert_eq!(codec.dense_content_identity(), Some(receipt));
    assert_eq!(receipt.source(), &source);
    assert_eq!(receipt.text_policy(), "mistral-instruct-v1");
    assert_eq!(receipt.tokenizer_sha256(), files[3].sha256);
    assert_eq!(receipt.tokenizer_config_policy_sha256(), files[4].sha256);
    assert_eq!(receipt.ancillary_special_map_sha256(), files[5].sha256);
    assert_eq!(codec.stop_ids(), &[2]);
    assert!(Codec::load_mistral_audited(&d.0, &source, Profile::Mistral.allowance()).is_err());
    let bad = String::from_utf8(json.to_vec())
        .unwrap()
        .replace("\"</s>\":2", "\"</s>\":9");
    let files = fixture(&d.0, bad.as_bytes());
    assert!(
        build(
            DenseProfile::Mistral,
            assets(DenseProfile::Mistral, &d.0, files[0].sha256, &files).unwrap(),
            &source
        )
        .is_err()
    );
}

#[test]
fn mistral_each_codec_role_and_failed_parser_prevent_publication() {
    let d = Directory::new();
    let files = fixture(&d.0, b"invalid tokenizer JSON");
    let source = synthetic_source();
    assert!(
        build(
            DenseProfile::Mistral,
            assets(DenseProfile::Mistral, &d.0, files[0].sha256, &files).unwrap(),
            &source
        )
        .is_err()
    );
    for i in [0, 3, 4, 5] {
        let path = d.0.join(files[i].name);
        let original = fs::read(&path).unwrap();
        let mut corrupt = original.clone();
        corrupt[0] ^= 1;
        fs::write(&path, corrupt).unwrap();
        assert!(assets(DenseProfile::Mistral, &d.0, files[0].sha256, &files).is_err());
        fs::write(&path, &original).unwrap();
        fs::rename(&path, d.0.join("missing")).unwrap();
        assert!(assets(DenseProfile::Mistral, &d.0, files[0].sha256, &files).is_err());
        fs::rename(d.0.join("missing"), &path).unwrap();
    }
    fs::remove_file(d.0.join(files[1].name)).unwrap();
    fs::remove_file(d.0.join(files[2].name)).unwrap();
    assert!(assets(DenseProfile::Mistral, &d.0, files[0].sha256, &files).is_ok());
}
