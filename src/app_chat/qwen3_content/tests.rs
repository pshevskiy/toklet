use super::*;
use crate::weights::content::SourceFixture;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
static SERIAL: AtomicUsize = AtomicUsize::new(0);
pub(super) struct Directory(pub(super) PathBuf);
impl Directory {
    pub(super) fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "toklet-qwen-codec-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
pub(super) fn synthetic_source() -> DenseSourceContentIdentity {
    let f = SourceFixture::new();
    let mut source = f.source();
    source.visit_raw("a", |_| Ok(())).unwrap();
    source.visit_raw("b", |_| Ok(())).unwrap();
    source.dense_content_identity().unwrap()
}
pub(super) fn fixture(directory: &Path, tokenizer: &[u8]) -> [ExpectedFile<'static>; 6] {
    let data: [(&str, &[u8]); 6] = [
        ("config.json", b"{}"),
        ("model.safetensors", b"unused"),
        ("model.safetensors.index.json", b"unused"),
        ("tokenizer.json", tokenizer),
        ("tokenizer_config.json", b"policy"),
        ("special_tokens_map.json", b"ancillary"),
    ];
    data.map(|(name, bytes)| {
        fs::write(directory.join(name), bytes).unwrap();
        ExpectedFile {
            name,
            bytes: bytes.len() as u64,
            sha256: Sha256::digest(bytes).into(),
        }
    })
}
#[test]
fn source_and_allowance_admission_precedes_file_access() {
    let source = synthetic_source();
    let missing = Path::new("/toklet-absent-audited-qwen-codec");
    for n in [
        0,
        Profile::Qwen3.allowance() - 1,
        crate::weights::INITIAL_MEMORY_LIMIT_BYTES + 1,
    ] {
        let e = Codec::load_qwen3_audited(missing, &source, n)
            .err()
            .unwrap();
        assert!(e.to_string().contains("allowance"));
    }
    let e = Codec::load_qwen3_audited(missing, &source, Profile::Qwen3.allowance())
        .err()
        .unwrap();
    assert!(e.to_string().contains("Unsupported authenticated source"));
    assert!(
        crate::app_chat::decode_fixture()
            .qwen3_content_identity()
            .is_none()
    );
}
#[test]
fn each_consumed_or_verified_role_is_required_and_same_length_changes_fail() {
    let d = Directory::new();
    let expected = fixture(&d.0, b"tokenizer");
    let a = assets(
        DenseProfile::Qwen(QwenProfile::Qwen3),
        &d.0,
        expected[0].sha256,
        &expected,
    )
    .unwrap();
    assert_eq!(a.tokenizer, b"tokenizer");
    assert_eq!(a.tokenizer_sha, expected[3].sha256);
    assert_eq!(a.policy_sha, expected[4].sha256);
    assert_eq!(a.ancillary_sha, expected[5].sha256);
    for index in [0, 3, 4, 5] {
        let p = d.0.join(expected[index].name);
        let original = fs::read(&p).unwrap();
        let mut changed = original.clone();
        changed[0] ^= 1;
        fs::write(&p, &changed).unwrap();
        assert!(
            assets(
                DenseProfile::Qwen(QwenProfile::Qwen3),
                &d.0,
                expected[0].sha256,
                &expected
            )
            .is_err()
        );
        fs::write(&p, &original).unwrap();
        fs::rename(&p, d.0.join("missing")).unwrap();
        assert!(
            assets(
                DenseProfile::Qwen(QwenProfile::Qwen3),
                &d.0,
                expected[0].sha256,
                &expected
            )
            .is_err()
        );
        fs::rename(d.0.join("missing"), p).unwrap();
    }
    assert!(
        assets(
            DenseProfile::Qwen(QwenProfile::Qwen3),
            &d.0,
            [0; 32],
            &expected
        )
        .is_err()
    );
    // Model/index are covered by source evidence, not reopened by text loading.
    fs::remove_file(d.0.join(expected[1].name)).unwrap();
    fs::remove_file(d.0.join(expected[2].name)).unwrap();
    assert!(
        assets(
            DenseProfile::Qwen(QwenProfile::Qwen3),
            &d.0,
            expected[0].sha256,
            &expected
        )
        .is_ok()
    );
}
#[test]
fn verified_buffer_limits_and_parser_failure_do_not_publish_codec() {
    let d = Directory::new();
    let expected = fixture(&d.0, b"invalid tokenizer JSON");
    assert!(verified(&d.0, expected[3], expected[3].bytes as usize - 1).is_err());
    let a = assets(
        DenseProfile::Qwen(QwenProfile::Qwen3),
        &d.0,
        expected[0].sha256,
        &expected,
    )
    .unwrap();
    assert_eq!(
        a.tokenizer_sha,
        <[u8; 32]>::from(Sha256::digest(&a.tokenizer))
    );
    assert!(
        build(
            DenseProfile::Qwen(QwenProfile::Qwen3),
            a,
            &synthetic_source()
        )
        .is_err()
    );
    fs::write(d.0.join(expected[3].name), b"longer invalid tokenizer JSON").unwrap();
    assert!(verified(&d.0, expected[3], 16 * 1024 * 1024).is_err());
}
#[test]
fn private_construction_attaches_only_after_same_buffer_parser_policy_passes() {
    // Synthetic format control only; public admission rejects this synthetic
    // source and these assets. It is not a forged production Qwen identity.
    let d = Directory::new();
    let json=br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"hello":0,"<|endoftext|>":151643,"<|im_start|>":151644,"<|im_end|>":151645},"unk_token":"hello"}}"#;
    let expected = fixture(&d.0, json);
    let source = synthetic_source();
    let codec = build(
        DenseProfile::Qwen(QwenProfile::Qwen3),
        assets(
            DenseProfile::Qwen(QwenProfile::Qwen3),
            &d.0,
            expected[0].sha256,
            &expected,
        )
        .unwrap(),
        &source,
    )
    .unwrap();
    let id = codec.qwen3_content_identity().unwrap();
    assert_eq!(id.source(), &source);
    assert_eq!(id.tokenizer_sha256(), expected[3].sha256);
    assert_eq!(id.tokenizer_config_policy_sha256(), expected[4].sha256);
    assert_eq!(id.ancillary_special_map_sha256(), expected[5].sha256);
    assert_eq!(id.text_policy(), "qwen3-plain-nonthinking-v1");
    assert_eq!(codec.stop_ids(), &[151645]);
    assert!(codec.content_identity().is_none());
    assert!(Codec::load_qwen3_audited(&d.0, &source, Profile::Qwen3.allowance()).is_err());
    let bad = String::from_utf8(json.to_vec())
        .unwrap()
        .replace("151645", "151646");
    let expected = fixture(&d.0, bad.as_bytes());
    assert!(
        build(
            DenseProfile::Qwen(QwenProfile::Qwen3),
            assets(
                DenseProfile::Qwen(QwenProfile::Qwen3),
                &d.0,
                expected[0].sha256,
                &expected
            )
            .unwrap(),
            &source
        )
        .is_err()
    );
}
#[cfg(unix)]
#[test]
fn codec_final_component_symlink_is_not_followed() {
    let d = Directory::new();
    let expected = fixture(&d.0, b"tokenizer");
    let path = d.0.join("tokenizer.json");
    fs::rename(&path, d.0.join("real-tokenizer")).unwrap();
    std::os::unix::fs::symlink("real-tokenizer", &path).unwrap();
    assert!(
        assets(
            DenseProfile::Qwen(QwenProfile::Qwen3),
            &d.0,
            expected[0].sha256,
            &expected
        )
        .is_err()
    );
}
