use super::*;
use crate::weights::audit::tests::{invalid_codec_sources, synthetic_codec_source};
use std::{
    fs,
    sync::atomic::{AtomicUsize, Ordering},
};
static SERIAL: AtomicUsize = AtomicUsize::new(0);
struct Directory(std::path::PathBuf);
impl Directory {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "toklet-smol-complete-{}-{}",
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
#[test]
fn complete_admission_precedes_files_and_has_profile_bounds() {
    let source = synthetic_codec_source(694);
    let missing = Path::new("/toklet-no-complete-smol-assets");
    for allowance in [
        0,
        TOKENIZER_ALLOWANCE - 1,
        crate::weights::INITIAL_MEMORY_LIMIT_BYTES + 1,
    ] {
        let error = Codec::load_smol_complete_audited(missing, &source, allowance)
            .err()
            .unwrap();
        assert!(error.to_string().contains("allowance"));
    }
    for invalid in invalid_codec_sources() {
        let error = Codec::load_smol_complete_audited(missing, &invalid, TOKENIZER_ALLOWANCE)
            .err()
            .unwrap();
        assert!(
            error
                .to_string()
                .contains("Unsupported authenticated source")
        );
    }
    complete_admit(&source, TOKENIZER_ALLOWANCE).unwrap();
    let error = Codec::load_smol_complete_audited(missing, &source, TOKENIZER_ALLOWANCE)
        .err()
        .unwrap();
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert!(error.to_string().contains("special_tokens_map.json"));
}
#[test]
fn missing_mutated_truncated_and_oversized_ancillary_fail_before_codec_parse() {
    let d = Directory::new();
    let source = synthetic_codec_source(694);
    let file = d.0.join("special_tokens_map.json");
    let error = Codec::load_smol_complete_audited(&d.0, &source, TOKENIZER_ALLOWANCE)
        .err()
        .unwrap();
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert!(error.to_string().contains("special_tokens_map.json"));
    for size in [654, 655, 656, 16 * 1024 + 1] {
        fs::write(&file, vec![b'x'; size]).unwrap();
        let e = Codec::load_smol_complete_audited(&d.0, &source, TOKENIZER_ALLOWANCE)
            .err()
            .unwrap();
        assert_eq!(e.kind(), io::ErrorKind::InvalidInput);
        // Missing-file errors are also wrapped as InvalidInput. The exact
        // ancillary diagnostic proves rejection before absent config/tokenizer.
        assert_eq!(
            e.to_string(),
            if size == 655 {
                "Audited codec consumed bytes differ"
            } else {
                "Audited codec asset extent differs"
            }
        );
    }
}
#[cfg(unix)]
#[test]
fn complete_ancillary_rejects_symlink_without_following() {
    let d = Directory::new();
    fs::write(d.0.join("target"), vec![0; 655]).unwrap();
    std::os::unix::fs::symlink(d.0.join("target"), d.0.join("special_tokens_map.json")).unwrap();
    let source = synthetic_codec_source(694);
    assert!(Codec::load_smol_complete_audited(&d.0, &source, TOKENIZER_ALLOWANCE).is_err());
}
#[test]
fn complete_manifest_is_canonical_and_preserves_reference_witness_anchor() {
    let m = smol_manifest().unwrap();
    let f = m.files();
    let source = synthetic_codec_source(694);
    let mut reverse = f.to_vec();
    reverse.reverse();
    let same =
        crate::weights::content::Manifest::new(source.repository(), source.revision(), &reverse)
            .unwrap();
    assert_eq!(same.identity(), m.identity());
    assert_eq!(f.len(), 6);
    assert_eq!(f[1].bytes, 143030401);
    assert_eq!(f[1].sha256, source.reference_container_sha256());
    assert_eq!(f[0].sha256, source.config_sha256());
    assert_eq!(f[2].sha256, source.index_sha256());
    assert_eq!(source.physical_ranges(), 694);
    assert!(std::mem::size_of::<CodecContentIdentity>() <= 512);
}
