use super::tests::{Directory, fixture, synthetic_source};
use super::*;
use std::fs;

#[test]
fn qwen_profile_memory_and_asset_caps_remain_distinct() {
    assert_eq!(text_profile(QwenProfile::Qwen2), Profile::Qwen);
    assert_eq!(text_profile(QwenProfile::Qwen3), Profile::Qwen3);
    assert_eq!(Profile::Qwen.allowance(), 128 * 1024 * 1024);
    assert_eq!(Profile::Qwen3.allowance(), 256 * 1024 * 1024);
    assert_eq!(tokenizer_cap(QwenProfile::Qwen2), 8 * 1024 * 1024);
    assert_eq!(tokenizer_cap(QwenProfile::Qwen3), 16 * 1024 * 1024);
    let source = synthetic_source();
    let missing = Path::new("/toklet-absent-audited-qwen2-codec");
    for allowance in [
        0,
        Profile::Qwen.allowance() - 1,
        crate::weights::INITIAL_MEMORY_LIMIT_BYTES + 1,
    ] {
        assert!(
            Codec::load_qwen2_audited(missing, &source, allowance)
                .err()
                .unwrap()
                .to_string()
                .contains("allowance")
        );
    }
    for profile in [QwenProfile::Qwen2, QwenProfile::Qwen3] {
        let e =
            Codec::load_qwen_audited(profile, missing, &source, text_profile(profile).allowance())
                .err()
                .unwrap();
        assert!(e.to_string().contains("Unsupported authenticated source"));
    }
    let d = Directory::new();
    let mut files = fixture(&d.0, b"tokenizer");
    files[4].bytes = 16 * 1024 + 1;
    assert!(assets(QwenProfile::Qwen2, &d.0, files[0].sha256, &files).is_err());
    files[4].bytes = b"policy".len() as u64;
    files[3].bytes = tokenizer_cap(QwenProfile::Qwen2) as u64 + 1;
    assert!(assets(QwenProfile::Qwen2, &d.0, files[0].sha256, &files).is_err());
}

#[test]
fn qwen2_parser_publishes_only_its_profile_and_old_const_api_remains() {
    // Private format fixture only: public source admission rejects this identity.
    let d = Directory::new();
    let json=br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"hello":0,"<|endoftext|>":151643,"<|im_start|>":151644,"<|im_end|>":151645},"unk_token":"hello"}}"#;
    let files = fixture(&d.0, json);
    let source = synthetic_source();
    let codec = build(
        QwenProfile::Qwen2,
        assets(QwenProfile::Qwen2, &d.0, files[0].sha256, &files).unwrap(),
        &source,
    )
    .unwrap();
    const fn old_getter(
        codec: &Codec,
    ) -> Option<&crate::app_chat::qwen3_content::QwenCodecContentIdentity> {
        codec.qwen3_content_identity()
    }
    assert!(old_getter(&codec).is_none());
    assert!(codec.content_identity().is_none());
    let receipt = codec.qwen2_content_identity().unwrap();
    assert_eq!(receipt.source(), &source);
    assert_eq!(receipt.text_policy(), "qwen-chatml-v1");
    assert_eq!(receipt.tokenizer_sha256(), files[3].sha256);
    assert_eq!(receipt.tokenizer_config_policy_sha256(), files[4].sha256);
    assert_eq!(receipt.ancillary_special_map_sha256(), files[5].sha256);
    assert_eq!(codec.stop_ids(), &[151643, 151645]);
    assert!(Codec::load_qwen2_audited(&d.0, &source, Profile::Qwen.allowance()).is_err());
    let qwen3 = build(
        QwenProfile::Qwen3,
        assets(QwenProfile::Qwen3, &d.0, files[0].sha256, &files).unwrap(),
        &source,
    )
    .unwrap();
    assert!(qwen3.qwen2_content_identity().is_none());
    assert_eq!(
        old_getter(&qwen3).unwrap().text_policy(),
        "qwen3-plain-nonthinking-v1"
    );
    let bad = String::from_utf8(json.to_vec())
        .unwrap()
        .replace("151643", "151642");
    let files = fixture(&d.0, bad.as_bytes());
    assert!(
        build(
            QwenProfile::Qwen2,
            assets(QwenProfile::Qwen2, &d.0, files[0].sha256, &files).unwrap(),
            &source
        )
        .is_err()
    );
}

#[test]
fn qwen2_each_verified_role_and_failed_parser_prevents_publication() {
    let d = Directory::new();
    let files = fixture(&d.0, b"invalid tokenizer JSON");
    let source = synthetic_source();
    assert!(
        build(
            QwenProfile::Qwen2,
            assets(QwenProfile::Qwen2, &d.0, files[0].sha256, &files).unwrap(),
            &source
        )
        .is_err()
    );
    for i in [0, 3, 4, 5] {
        let path = d.0.join(files[i].name);
        let original = fs::read(&path).unwrap();
        let mut corrupt = original.clone();
        corrupt[0] ^= 1;
        fs::write(&path, &corrupt).unwrap();
        assert!(assets(QwenProfile::Qwen2, &d.0, files[0].sha256, &files).is_err());
        fs::write(&path, &original).unwrap();
        fs::rename(&path, d.0.join("missing")).unwrap();
        assert!(assets(QwenProfile::Qwen2, &d.0, files[0].sha256, &files).is_err());
        fs::rename(d.0.join("missing"), &path).unwrap();
    }
    fs::remove_file(d.0.join(files[1].name)).unwrap();
    fs::remove_file(d.0.join(files[2].name)).unwrap();
    assert!(assets(QwenProfile::Qwen2, &d.0, files[0].sha256, &files).is_ok());
}
