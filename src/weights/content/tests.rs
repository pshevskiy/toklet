use super::*;
use std::{
    fs,
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

const REPO: &str = "test/artifact";
pub(super) const REV: &str = "0123456789abcdef0123456789abcdef01234567";
static SERIAL: AtomicUsize = AtomicUsize::new(0);
pub(super) struct Fixture {
    pub(super) path: PathBuf,
    bytes: Vec<u8>,
    entries: [ExpectedFile<'static>; 1],
}
impl Fixture {
    pub(super) fn new(bytes: Vec<u8>) -> Self {
        let path = std::env::temp_dir().join(format!(
            "toklet-content-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        fs::write(path.join("data.bin"), &bytes).unwrap();
        let entries = [ExpectedFile {
            name: "data.bin",
            bytes: bytes.len() as u64,
            sha256: Sha256::digest(&bytes).into(),
        }];
        Self {
            path,
            bytes,
            entries,
        }
    }
    fn manifest(&self) -> Manifest<'_> {
        Manifest::new(REPO, REV, &self.entries).unwrap()
    }
    fn open(&self, ends: &[u64]) -> FileAudit {
        FileAudit::open(&self.path, &self.manifest(), 0, ends, 1024 * 1024).unwrap()
    }
    fn overwrite(&self, offset: u64, bytes: &[u8]) {
        let mut file = File::options()
            .write(true)
            .open(self.path.join("data.bin"))
            .unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(bytes).unwrap();
        file.flush().unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[test]
fn manifest_is_canonical_and_every_content_field_matters() {
    let a = ExpectedFile {
        name: "a",
        bytes: 1,
        sha256: [1; 32],
    };
    let b = ExpectedFile {
        name: "b",
        bytes: 2,
        sha256: [2; 32],
    };
    let files = [a, b];
    let expected = Manifest::new(REPO, REV, &files).unwrap().identity();
    assert_eq!(
        expected,
        Manifest::new(REPO, REV, &[b, a]).unwrap().identity()
    );
    for altered in [
        ExpectedFile { name: "c", ..b },
        ExpectedFile { bytes: 3, ..b },
        ExpectedFile {
            sha256: [3; 32],
            ..b
        },
    ] {
        assert_ne!(
            expected,
            Manifest::new(REPO, REV, &[a, altered]).unwrap().identity()
        );
    }
    assert_ne!(
        expected,
        Manifest::new("test/other", REV, &files).unwrap().identity()
    );
    assert_ne!(
        expected,
        Manifest::new(REPO, "1123456789abcdef0123456789abcdef01234567", &files)
            .unwrap()
            .identity()
    );
    // Independent explicit canonical byte sequence for this small two-file case.
    let mut bytes = b"toklet-full-manifest-v1\0".to_vec();
    bytes.extend_from_slice(&13_u64.to_le_bytes());
    bytes.extend_from_slice(b"test/artifact");
    bytes.extend_from_slice(&40_u64.to_le_bytes());
    bytes.extend_from_slice(REV.as_bytes());
    bytes.extend_from_slice(&2_u64.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(b"a");
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&[1; 32]);
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(b"b");
    bytes.extend_from_slice(&2_u64.to_le_bytes());
    bytes.extend_from_slice(&[2; 32]);
    assert_eq!(expected, <[u8; 32]>::from(Sha256::digest(bytes)));
}

#[test]
fn manifest_rejects_ambiguous_and_unbounded_inputs() {
    let base = ExpectedFile {
        name: "a",
        bytes: 1,
        sha256: [0; 32],
    };
    for name in [
        "", ".", "..", "../a", "/a", "a/b", "a\\b", "a\0b", "a b", "é",
    ] {
        assert!(Manifest::new(REPO, REV, &[ExpectedFile { name, ..base }]).is_err());
    }
    let long = "a".repeat(129);
    assert!(
        Manifest::new(
            REPO,
            REV,
            &[ExpectedFile {
                name: &long,
                ..base
            }]
        )
        .is_err()
    );
    for repository in ["a", "a/b/c", "/a", "a/..", "a/b\0"] {
        assert!(Manifest::new(repository, REV, &[base]).is_err());
    }
    for revision in ["", "main", "0123456789ABCDEF0123456789abcdef01234567"] {
        assert!(Manifest::new(REPO, revision, &[base]).is_err());
    }
    assert!(Manifest::new(REPO, REV, &[]).is_err());
    assert!(Manifest::new(REPO, REV, &[base; MAX_FILES + 1]).is_err());
    assert!(Manifest::new(REPO, REV, &[base, base]).is_err());
    for bytes in [0, MAX_FILE_BYTES + 1, u64::MAX] {
        assert!(Manifest::new(REPO, REV, &[ExpectedFile { bytes, ..base }]).is_err());
    }
    assert!(
        Manifest::new(
            REPO,
            REV,
            &[
                ExpectedFile {
                    bytes: MAX_FILE_BYTES,
                    ..base
                },
                ExpectedFile {
                    name: "b",
                    bytes: MAX_FILE_BYTES,
                    ..base
                },
            ]
        )
        .is_err()
    );
}

#[test]
fn pinned_qwen_manifest_matches_compiled_catalog_metadata_without_model_reads() {
    let m = qwen3_manifest().unwrap();
    assert_eq!(m.files().len(), 6);
    assert_eq!(m.files().iter().map(|f| f.bytes).sum::<u64>(), 644927148);
    assert_eq!(m.files()[1].name, "model.safetensors");
    assert_eq!(m.files()[1].bytes, 633442994);
    assert_eq!(m.files()[5].name, "special_tokens_map.json");
    let reverse: Vec<_> = m.files().iter().copied().rev().collect();
    assert_eq!(
        m.identity(),
        Manifest::new(
            "mlx-community/Qwen3-0.6B-8bit",
            "11de96878523501bcaa86104e3c186de07ff9068",
            &reverse
        )
        .unwrap()
        .identity()
    );
}

#[test]
fn admission_precedes_open_and_checks_actual_owned_storage() {
    let f = Fixture::new(vec![1, 2, 3]);
    let required = FileAudit::required_bytes(2).unwrap();
    let missing = f.path.join("missing");
    assert!(matches!(
        FileAudit::open(&missing, &f.manifest(), 0, &[1, 3], required - 1),
        Err(Error::Budget { .. })
    ));
    let audit = FileAudit::open(&f.path, &f.manifest(), 0, &[1, 3], required).unwrap();
    assert_eq!(audit.owned_bytes().unwrap(), required);
    assert!(FileAudit::storage_bytes(usize::MAX, 1).is_err());
    assert!(FileAudit::storage_bytes(1, usize::MAX).is_err());
    assert!(FileAudit::required_bytes(0).is_err());
    assert!(FileAudit::required_bytes(MAX_RANGES + 1).is_err());
    assert!(FileAudit::required_bytes(usize::MAX).is_err());
    for ends in [
        vec![],
        vec![0, 3],
        vec![2, 2, 3],
        vec![2, 1, 3],
        vec![2],
        vec![4],
        vec![u64::MAX],
    ] {
        assert!(FileAudit::open(&missing, &f.manifest(), 0, &ends, usize::MAX).is_err());
    }
    assert!(FileAudit::open(&missing, &f.manifest(), 1, &[3], usize::MAX).is_err());
}

#[test]
fn receipt_requires_actual_complete_delivery_and_checks_repeated_ranges() {
    let f = Fixture::new(vec![10, 11, 12, 13]);
    assert!(f.open(&[1, 4]).finish().is_err());
    let mut a = f.open(&[1, 4]);
    let mut got = Vec::new();
    a.visit(1, |b| {
        got.extend_from_slice(b);
        Ok(())
    })
    .unwrap();
    assert_eq!(got, [11, 12, 13]);
    a.visit(0, |_| Ok(())).unwrap();
    a.visit(0, |_| Ok(())).unwrap();
    let receipt = a.finish().unwrap();
    assert_eq!(receipt.manifest_identity(), f.manifest().identity());
    assert_eq!(
        receipt.file_name_sha256(),
        <[u8; 32]>::from(Sha256::digest(b"data.bin"))
    );
    assert_eq!(receipt.file_sha256(), f.entries[0].sha256);
    assert_eq!(receipt.bytes(), 4);
    assert_eq!(receipt.physical_ranges(), 2);
    assert_eq!(receipt.verified_reads(), 3);
    let mut incomplete = f.open(&[1, 4]);
    incomplete.visit(0, |_| Ok(())).unwrap();
    assert!(incomplete.finish().is_err());
}

#[test]
fn prepass_rejects_wrong_digest_truncation_and_growth() {
    let f = Fixture::new(vec![1, 2, 3]);
    f.overwrite(1, &[9]);
    assert!(FileAudit::open(&f.path, &f.manifest(), 0, &[3], 1024 * 1024).is_err());
    fs::write(f.path.join("data.bin"), [1, 2]).unwrap();
    assert!(FileAudit::open(&f.path, &f.manifest(), 0, &[3], 1024 * 1024).is_err());
    fs::write(f.path.join("data.bin"), [1, 2, 3, 4]).unwrap();
    assert!(FileAudit::open(&f.path, &f.manifest(), 0, &[3], 1024 * 1024).is_err());
}

#[test]
fn same_size_mutation_after_prepass_poisoned_even_after_partial_delivery() {
    let f = Fixture::new(vec![1, 2, 3]);
    let mut a = f.open(&[1, 3]);
    a.visit(0, |_| Ok(())).unwrap();
    f.overwrite(1, &[9]);
    let mut partial = Vec::new();
    assert!(
        a.visit(1, |b| {
            partial.extend_from_slice(b);
            Ok(())
        })
        .is_err()
    );
    assert_eq!(partial, [9, 3]); // Consumers must discard these partial outputs.
    f.overwrite(1, &[2]);
    assert!(a.visit(1, |_| Ok(())).is_err());
    assert!(a.finish().is_err());
}

#[test]
fn retained_descriptor_survives_path_replacement_and_receipt_is_historical() {
    let f = Fixture::new(vec![1, 2, 3]);
    let mut a = f.open(&[3]);
    fs::rename(f.path.join("data.bin"), f.path.join("old.bin")).unwrap();
    fs::write(f.path.join("data.bin"), [9, 9, 9]).unwrap();
    let mut actual = Vec::new();
    a.visit(0, |b| {
        actual.extend_from_slice(b);
        Ok(())
    })
    .unwrap();
    let receipt = a.finish().unwrap();
    assert_eq!(actual, f.bytes);
    assert_eq!(receipt.file_sha256(), f.entries[0].sha256);
    assert!(FileAudit::open(&f.path, &f.manifest(), 0, &[3], 1024 * 1024).is_err());
}

#[test]
fn errors_unwinds_and_invalid_indices_are_sticky_after_complete_coverage() {
    let f = Fixture::new(vec![1]);
    let mut a = f.open(&[1]);
    a.visit(0, |_| Ok(())).unwrap();
    assert!(
        a.visit(0, |_| Err(invalid("synthetic consumer failure")))
            .is_err()
    );
    assert!(a.finish().is_err());
    let mut a = f.open(&[1]);
    a.visit(0, |_| Ok(())).unwrap();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = a.visit(0, |_| panic!("synthetic consumer panic"));
    }));
    assert!(panic.is_err());
    assert!(a.visit(0, |_| Ok(())).is_err());
    assert!(a.finish().is_err());
    let mut a = f.open(&[1]);
    assert!(a.visit(usize::MAX, |_| Ok(())).is_err());
    assert!(a.visit(0, |_| Ok(())).is_err());
    assert!(a.finish().is_err());
}

#[test]
fn streaming_mutation_and_extent_changes_fail_closed() {
    let f = Fixture::new(vec![7; BUFFER_BYTES + 2]);
    let mut a = f.open(&[f.bytes.len() as u64]);
    let mut calls = 0;
    assert!(
        a.visit(0, |b| {
            assert!(b.len() <= BUFFER_BYTES);
            calls += 1;
            if calls == 1 {
                f.overwrite(BUFFER_BYTES as u64, &[8]);
            }
            Ok(())
        })
        .is_err()
    );
    assert_eq!(calls, 2);
    assert!(a.finish().is_err());
    fs::write(f.path.join("data.bin"), &f.bytes).unwrap();
    let mut a = f.open(&[f.bytes.len() as u64]);
    File::options()
        .write(true)
        .open(f.path.join("data.bin"))
        .unwrap()
        .set_len(1)
        .unwrap();
    let mut delivered = false;
    assert!(
        a.visit(0, |_| {
            delivered = true;
            Ok(())
        })
        .is_err()
    );
    assert!(!delivered);
    assert!(a.finish().is_err());
}

#[cfg(unix)]
#[test]
fn symlink_directory_and_fifo_are_rejected_without_blocking() {
    use std::{ffi::CString, os::unix::ffi::OsStrExt, os::unix::fs::symlink};
    let f = Fixture::new(vec![1]);
    let path = f.path.join("data.bin");
    fs::rename(&path, f.path.join("target")).unwrap();
    symlink("target", &path).unwrap();
    assert!(FileAudit::open(&f.path, &f.manifest(), 0, &[1], 1024 * 1024).is_err());
    fs::remove_file(&path).unwrap();
    fs::create_dir(&path).unwrap();
    assert!(FileAudit::open(&f.path, &f.manifest(), 0, &[1], 1024 * 1024).is_err());
    fs::remove_dir(&path).unwrap();
    let c = CString::new(path.as_os_str().as_bytes()).unwrap();
    // SAFETY: c is a live NUL-terminated path; mkfifo reads it without retaining
    // the pointer. The isolated fixture owns this new path and removes it.
    assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
    assert!(FileAudit::open(&f.path, &f.manifest(), 0, &[1], 1024 * 1024).is_err());
}
