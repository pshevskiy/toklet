use super::*;
use crate::weights::audit::tests::{invalid_codec_sources, synthetic_codec_source};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
static SERIAL: AtomicUsize = AtomicUsize::new(0);
struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "toklet-codec-content-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
#[test]
fn admission_precedes_open_and_preserves_repeated_reads() {
    let directory = Directory::new();
    let missing = directory.0.join("missing");
    let identity = synthetic_codec_source(694);
    let error = Codec::load_smol_audited(&missing, &identity, TOKENIZER_ALLOWANCE - 1)
        .err()
        .unwrap();
    assert!(error.to_string().contains("allowance"));
    for invalid in invalid_codec_sources() {
        let error = Codec::load_smol_audited(&missing, &invalid, TOKENIZER_ALLOWANCE)
            .err()
            .unwrap();
        assert!(
            error
                .to_string()
                .contains("Unsupported authenticated source")
        );
    }
    for reads in [694, 700, usize::MAX] {
        let identity = synthetic_codec_source(reads);
        admit(&identity, TOKENIZER_ALLOWANCE).unwrap();
        let error = Codec::load_smol_audited(&missing, &identity, TOKENIZER_ALLOWANCE)
            .err()
            .unwrap();
        assert!(
            !error
                .to_string()
                .contains("Unsupported authenticated source")
        );
        assert!(error.to_string().contains("missing"));
    }
}
#[test]
fn bounded_regular_assets_and_same_saved_buffer() {
    let directory = Directory::new();
    let path = directory.0.join("asset");
    fs::write(&path, b"ok").unwrap();
    let bytes = bounded_regular(&path, 2).unwrap();
    assert_eq!(bytes, b"ok");
    assert!(bounded_regular(&path, 1).is_err());
    assert!(bounded_regular(&directory.0, 2).is_err());
    let expected = format!("{:x}", Sha256::digest(&bytes));
    fs::write(&path, b"NO").unwrap();
    // Authentication uses the consumed buffer, not a reopened path. This
    // synthetic digest is local to the helper test, never production loading.
    assert_eq!(
        authenticate(&bytes, &expected).unwrap(),
        <[u8; 32]>::from(Sha256::digest(b"ok"))
    );
    assert!(authenticate(&bounded_regular(&path, 2).unwrap(), &expected).is_err());
}
#[cfg(unix)]
#[test]
fn symlink_and_fifo_reject_without_following_or_blocking() {
    use std::{
        ffi::CString,
        os::unix::{ffi::OsStrExt, fs::symlink},
    };
    let directory = Directory::new();
    fs::write(directory.0.join("regular"), b"ok").unwrap();
    let link = directory.0.join("link");
    symlink(directory.0.join("regular"), &link).unwrap();
    assert!(bounded_regular(&link, 2).is_err());
    let fifo = directory.0.join("fifo");
    let name = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    // SAFETY: the owned, NUL-terminated path remains live for mkfifo; mode is
    // valid and the test creates no shared-memory/pointer aliasing contract.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    assert!(bounded_regular(&fifo, 2).is_err());
}
#[test]
fn wrong_assets_reject_and_source_evidence_remains_unchanged() {
    let directory = Directory::new();
    let identity = synthetic_codec_source(700);
    let saved = identity;
    fs::write(directory.0.join("config.json"), b"{}").unwrap();
    let error = Codec::load_smol_audited(&directory.0, &identity, TOKENIZER_ALLOWANCE)
        .err()
        .unwrap();
    assert!(error.to_string().contains("differs"));
    let config = include_bytes!("../../../tests/fixtures/smol-load-identity-v1/config.json");
    fs::write(directory.0.join("config.json"), config).unwrap();
    fs::write(directory.0.join("tokenizer_config.json"), b"{}").unwrap();
    let error = Codec::load_smol_audited(&directory.0, &identity, TOKENIZER_ALLOWANCE)
        .err()
        .unwrap();
    assert!(error.to_string().contains("differs"));
    for expected in [MODEL_CONFIG_SHA, CONFIG_SHA, TOKENIZER_SHA] {
        assert!(authenticate(b"{}", expected).is_err());
    }
    assert_eq!(identity, saved);
}
#[test]
fn parser_policy_failures_publish_no_codec_and_legacy_stays_unbound() {
    assert!(Codec::from_tokenizer_bytes(b"{}", Profile::Smol).is_err());
    let raw = include_bytes!("../../../tests/fixtures/chat-stream-v1/smol-tokenizer.json");
    let codec = Codec::from_tokenizer_bytes(raw, Profile::Smol).unwrap();
    assert!(codec.content_identity().is_none());
    assert!(
        crate::app_chat::decode_fixture()
            .content_identity()
            .is_none()
    );
    let original: serde_json::Value = serde_json::from_slice(raw).unwrap();
    let mut changed = original.clone();
    changed["padding"] = serde_json::json!({"strategy":"BatchLongest", "direction":"Right", "pad_to_multiple_of":null,"pad_id":0,"pad_type_id":0,"pad_token":"<|endoftext|>"});
    assert!(
        Codec::from_tokenizer_bytes(&serde_json::to_vec(&changed).unwrap(), Profile::Smol).is_err()
    );
    let mut changed = original;
    // Added-token IDs can be remapped back to existing model-vocabulary IDs by
    // tokenizers. Mutate the actual model vocabulary and verify the fault is
    // present after parsing before exercising the shared admission policy.
    changed["model"]["vocab"]["!"] = serde_json::json!(49152);
    let changed = serde_json::to_vec(&changed).unwrap();
    assert_eq!(
        tokenizers::Tokenizer::from_bytes(&changed)
            .unwrap()
            .token_to_id("!"),
        Some(49152)
    );
    assert!(Codec::from_tokenizer_bytes(&changed, Profile::Smol).is_err());
}

pub(crate) mod capture;

// Explicit hardware/model diagnostic, excluded from ordinary offline tests.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod real {
    use super::*;
    use crate::{metal, weights::LoadOptions};
    use serde_json::{Value, json};
    use std::{error::Error, mem::size_of};

    const PROCESS_POLICY: usize = 1536 * 1024 * 1024;
    const CAPTURE_LIMIT: usize = 16 * 1024 * 1024;
    // Two codec allowances, two capture bounds and scratch/JSON/file overhead
    // are reserved BEFORE any model or tokenizer allocation. The capture bounds
    // include complete files even though only one small frame is held at once.
    const APPLICATION_RESERVE: usize =
        2 * TOKENIZER_ALLOWANCE + 2 * CAPTURE_LIMIT + 16 * 1024 * 1024;
    use super::capture::{Capture, capture_pair};
    fn decode(
        files: &mut [Capture; 2],
        codecs: &[Codec; 2],
        group: &str,
        ordinal: usize,
        ids: &[u32],
    ) -> Result<String, Box<dyn Error>> {
        let values = [codecs[0].decode(ids), codecs[1].decode(ids)];
        capture_pair(files, "decode", group, ordinal, Some(ids), &values)?;
        let [left, right] = values;
        let (left, right) = (left?, right?);
        if left != right {
            return Err("legacy/audited decoded bytes differ".into());
        }
        Ok(left)
    }
    fn hex(bytes: [u8; 32]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
    fn run() -> Result<(), Box<dyn Error>> {
        let directory = PathBuf::from(
            std::env::var_os("TOKLET_SMOL_CODEC_MODEL_DIRECTORY")
                .ok_or("missing model directory")?,
        );
        let capture = PathBuf::from(
            std::env::var_os("TOKLET_SMOL_CODEC_CAPTURE_DIRECTORY")
                .ok_or("missing capture directory")?,
        );
        if !directory.is_absolute() || !capture.is_absolute() {
            return Err("diagnostic paths must be absolute".into());
        }
        let mut files = [
            Capture::new(&capture.join("legacy.jsonl"), CAPTURE_LIMIT)?,
            Capture::new(&capture.join("audited.jsonl"), CAPTURE_LIMIT)?,
        ];
        let model = metal::Model::load_mlx_smol_audited(
            &directory,
            LoadOptions {
                context_limit: 8,
                memory_limit_bytes: PROCESS_POLICY - APPLICATION_RESERVE,
            },
        )?;
        let source = *model
            .mlx_source_content_identity()
            .ok_or("missing model evidence")?;
        let device = model.device_name();
        let weights_bytes = model.weights_bytes();
        let codecs = [
            Codec::load(&directory)?,
            Codec::load_smol_audited(&directory, &source, TOKENIZER_ALLOWANCE)?,
        ];
        if codecs[0].content_identity().is_some()
            || codecs[1]
                .content_identity()
                .is_none_or(|id| *id.source() != source)
        {
            return Err("incorrect codec pairing".into());
        }
        let identity = *codecs[1].content_identity().unwrap();
        drop(model);
        if codecs[1].content_identity() != Some(&identity) {
            return Err("pairing did not survive model drop".into());
        }
        for id in 0..49152 {
            decode(&mut files, &codecs, "singleton", id as usize, &[id])?;
        }
        // Historical independent reference, unchanged from the accepted chat
        // qualification. This literal is not a newly generated target oracle.
        let fixture_bytes = include_bytes!("../../../tests/fixtures/app-chat-v1/reference.json");
        authenticate(
            fixture_bytes,
            "3b962a75fb9e84fc0d098ef4cf350520ebb620e0d1d68f22a4462a9375b83577",
        )?;
        let fixture: Value = serde_json::from_slice(fixture_bytes)?;
        if fixture["format"] != "toklet-smollm-chat-reference-v1"
            || fixture["repo"] != source.repository()
            || fixture["revision"] != source.revision()
            || fixture["tokenizer_sha256"] != TOKENIZER_SHA
            || fixture["config_sha256"] != CONFIG_SHA
        {
            return Err("independent reference metadata differs".into());
        }
        let mut prefixes = 0;
        let cases = fixture["cases"].as_array().ok_or("missing fixture cases")?;
        if cases.len() != 2 {
            return Err("independent reference must contain two cases".into());
        }
        for (ordinal, case) in cases.iter().enumerate() {
            let messages = serde_json::to_vec(&case["messages"])?;
            let values = [codecs[0].encode(&messages), codecs[1].encode(&messages)];
            capture_pair(&mut files, "encode", "chat", ordinal, None, &values)?;
            let [left, right] = values;
            let (left, right) = (left?, right?);
            if left != right || serde_json::to_value(&left)? != case["ids"] {
                return Err("chat IDs differ from frozen reference".into());
            }
            let text = decode(&mut files, &codecs, "full-chat", ordinal, &left)?;
            if text != case["decoded"].as_str().ok_or("missing decoded witness")? {
                return Err("chat bytes differ from frozen reference".into());
            }
            for end in 0..=left.len() {
                decode(&mut files, &codecs, "chat-prefix", prefixes, &left[..end])?;
                prefixes += 1;
            }
        }
        for file in &mut files {
            file.flush()?;
        }
        println!(
            "SMOL_CODEC_CONTENT {}",
            json!({"format":"toklet-smol-codec-content-v1","device":device,"source_repository":source.repository(),"source_revision":source.revision(),"config_sha256":hex(source.config_sha256()),"index_sha256":hex(source.index_sha256()),"header_sha256":hex(source.header_sha256()),"authenticated_reference_container_sha256":hex(source.reference_container_sha256()),"physical_ranges":source.physical_ranges(),"verified_reads":source.verified_reads(),"payload_bytes":source.payload_bytes(),"container_bytes":source.container_bytes(),"tokenizer_config_sha256":hex(identity.tokenizer_config_sha256()),"tokenizer_sha256":hex(identity.tokenizer_sha256()),"codec_structure_bytes":size_of::<Codec>(),"identity_bytes":size_of::<CodecContentIdentity>(),"identity_option_bytes":size_of::<Option<CodecContentIdentity>>(),"weights_bytes":weights_bytes,"process_policy_bytes":PROCESS_POLICY,"application_reserve_bytes":APPLICATION_RESERVE,"model_load_policy_bytes":PROCESS_POLICY-APPLICATION_RESERVE,"capture_limit_each_bytes":CAPTURE_LIMIT,"capture_bytes":[files[0].bytes(),files[1].bytes()],"singleton_comparisons":49152,"independent_chat_cases":cases.len(),"same_implementation_prefix_comparisons":prefixes,"legacy_identity_absent":true,"historical_pairing_survives_model_drop":true,"live_model_ownership_qualified":false,"inference_executed":false,"app_eligibility_changed":false,"phone_fit_qualified":false,"performance_qualified":false})
        );
        Ok(())
    }
    #[test]
    #[ignore = "requires explicit local Smol model, Metal device and empty capture directory"]
    fn audited_codec_matches_legacy_and_frozen_chat_cases() {
        run().unwrap();
    }
}

/// Synthetic historical fields for native report-builder controls only.
/// No model, file consumption, or successful complete factory is attested.
pub(crate) fn synthetic_complete_identity(reads: usize, complete: bool) -> CodecContentIdentity {
    let source = synthetic_codec_source(reads);
    let m = crate::weights::content::smol_manifest().unwrap();
    let f = m.files();
    CodecContentIdentity {
        source,
        tokenizer_config: f[4].sha256,
        tokenizer: f[3].sha256,
        complete: complete.then_some(CompleteAssets {
            manifest: m.identity(),
            ancillary: f[5].sha256,
        }),
    }
}
#[test]
fn legacy_and_complete_receipts_are_distinct_historical_values() {
    let old = synthetic_complete_identity(694, false);
    let full = synthetic_complete_identity(694, true);
    assert_eq!(old.source(), full.source());
    assert_ne!(old, full);
    assert_eq!(old.complete_manifest_sha256(), None);
    assert_eq!(old.ancillary_special_map_sha256(), None);
    assert_eq!(
        full.complete_manifest_sha256(),
        Some(crate::weights::content::smol_manifest().unwrap().identity())
    );
}
