use super::*;
use crate::weights::Source;
use serde_json::json;
use std::{
    fs,
    io::{Seek, SeekFrom, Write},
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
static SERIAL: AtomicUsize = AtomicUsize::new(0);
pub(crate) struct Fixture {
    pub path: PathBuf,
    pub data_start: u64,
    pub original: Vec<u8>,
    pub source: Source,
}
impl Fixture {
    pub(crate) fn new(records: Vec<(&str, Dtype, Vec<usize>, Vec<u8>)>) -> Self {
        let path = std::env::temp_dir().join(format!(
            "toklet-audit-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        let mut header = serde_json::Map::new();
        let mut offset = 0;
        let mut payload = Vec::new();
        let mut ranges = Vec::new();
        for (name, dtype, shape, data) in records {
            let end = offset + data.len();
            let dtype_text = match dtype {
                Dtype::F32 => "F32",
                Dtype::F16 => "F16",
                Dtype::U32 => "U32",
                _ => panic!(),
            };
            header.insert(
                name.into(),
                json!({"dtype":dtype_text,"shape":shape,"data_offsets":[offset,end]}),
            );
            ranges.push(Range {
                name: name.into(),
                dtype,
                shape,
                begin: offset,
                end,
                digest: Sha256::digest(&data).into(),
            });
            payload.extend_from_slice(&data);
            offset = end;
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !(header.len() + 8).is_multiple_of(8) {
            header.push(b' ');
        }
        let data_start = (header.len() + 8) as u64;
        let mut original = (header.len() as u64).to_le_bytes().to_vec();
        original.extend_from_slice(&header);
        original.extend_from_slice(&payload);
        fs::write(path.join("model.safetensors"), &original).unwrap();
        let mut source = Source::open(&path).unwrap();
        ranges.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        let count = ranges.len();
        source.audit = Some(Audit {
            ranges,
            seen: vec![false; count],
            unique: 0,
            reads: 0,
            state: State::Clean,
            metadata: 15,
            identity: SourceContentIdentity {
                config: [0; 32],
                index: [0; 32],
                header: [0; 32],
                reference_container: [0; 32],
                verified_reads: 0,
                physical_ranges: count,
                payload_bytes: payload.len(),
                container_bytes: original.len() as u64,
            },
        });
        Self {
            path,
            data_start,
            original,
            source,
        }
    }
    fn tiny() -> Self {
        Self::new(vec![
            ("a", Dtype::F32, vec![2], vec![0, 0, 128, 63, 0, 0, 0, 64]),
            ("b", Dtype::U32, vec![1], vec![1, 2, 3, 4]),
        ])
    }
    fn overwrite(&self, offset: u64, bytes: &[u8]) {
        let mut f = File::options()
            .write(true)
            .open(self.path.join("model.safetensors"))
            .unwrap();
        f.seek(SeekFrom::Start(offset)).unwrap();
        f.write_all(bytes).unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
fn consume(source: &mut Source, name: &str) -> Result<(), Error> {
    source.visit_raw(name, |_| Ok(()))
}
fn assert_sticky(source: &mut Source) {
    assert!(source.content_identity().is_none());
    let mut called = false;
    assert!(
        source
            .visit_raw("a", |_| {
                called = true;
                Ok(())
            })
            .is_err()
    );
    assert!(!called);
    assert!(source.content_identity().is_none());
}
#[test]
fn frozen_profile_and_same_metadata_buffers_are_authenticated_without_payload() {
    assert!(matches!(Audit::smol(1), Err(Error::Budget { .. })));
    let mut a = Audit::smol(METADATA_ALLOWANCE_BYTES + CONVERSION_BYTES).unwrap();
    assert_eq!(a.ranges.len(), 694);
    assert!(a.owned_bytes() < 1024 * 1024);
    for (kind, raw) in [
        (
            MetadataKind::Config,
            &include_bytes!("../../../tests/fixtures/smol-load-identity-v1/config.json")[..],
        ),
        (
            MetadataKind::Index,
            &include_bytes!("../../../tests/fixtures/smol-load-identity-v1/index.json")[..],
        ),
        (
            MetadataKind::Prefix,
            &include_bytes!("../../../tests/fixtures/smol-load-identity-v1/prefix.bin")[..],
        ),
        (
            MetadataKind::Header,
            &include_bytes!("../../../tests/fixtures/smol-load-identity-v1/header.bin")[..],
        ),
    ] {
        a.verify_metadata(kind, raw).unwrap();
    }
    assert_eq!(a.metadata, 15);
    assert!(a.identity().is_none());
    assert_eq!(
        a.ranges.iter().map(|r| r.end - r.begin).sum::<usize>(),
        PAYLOAD_BYTES
    );
    assert!(a.verify_metadata(MetadataKind::Config, b"{}").is_err());
    assert!(a.begin("model.norm.weight").is_err());
    assert!(a.identity().is_none());
}
#[test]
fn complete_delivery_repeats_and_copied_identity_have_bounded_ownership() {
    let mut f = Fixture::tiny();
    let bytes = f.source.audit_owned_bytes().unwrap();
    assert!(f.source.content_identity().is_none());
    let mut values = Vec::new();
    f.source
        .visit("a", |x| {
            values.push(x);
            Ok(())
        })
        .unwrap();
    assert_eq!(values, [1., 2.]);
    assert!(f.source.content_identity().is_none());
    let mut delivered = Vec::new();
    f.source
        .visit_raw("b", |b| {
            delivered.extend_from_slice(b);
            Ok(())
        })
        .unwrap();
    assert_eq!(delivered, [1, 2, 3, 4]);
    let copied = f.source.content_identity().unwrap();
    assert_eq!(copied.verified_reads(), 2);
    consume(&mut f.source, "a").unwrap();
    assert_eq!(f.source.content_identity().unwrap().verified_reads(), 3);
    assert_eq!(f.source.audit_owned_bytes(), Some(bytes));
    fs::remove_file(f.path.join("model.safetensors")).unwrap();
    fs::write(f.path.join("model.safetensors"), b"replacement path").unwrap();
    consume(&mut f.source, "b").unwrap();
    assert_eq!(copied.verified_reads(), 2);
    assert_eq!(f.source.content_identity().unwrap().verified_reads(), 4);
}
#[test]
fn mutation_restore_defeats_endpoint_only_hashing_and_cannot_resurrect() {
    let mut f = Fixture::tiny();
    let before = Sha256::digest(fs::read(f.path.join("model.safetensors")).unwrap());
    f.overwrite(f.data_start, &[0, 0, 64, 64]);
    let path = f.path.join("model.safetensors");
    let original = f.original.clone();
    let mut delivered = Vec::new();
    assert!(
        f.source
            .visit_raw("a", |raw| {
                delivered.extend_from_slice(raw);
                fs::write(&path, &original).unwrap();
                Ok(())
            })
            .is_err()
    );
    assert_eq!(&delivered[..4], &[0, 0, 64, 64]);
    assert_eq!(before, Sha256::digest(fs::read(path).unwrap()));
    assert_sticky(&mut f.source);
}
#[test]
fn repeated_read_mutation_poison_does_not_revoke_historical_copy() {
    let mut f = Fixture::tiny();
    consume(&mut f.source, "a").unwrap();
    consume(&mut f.source, "b").unwrap();
    let copied = f.source.content_identity().unwrap();
    f.overwrite(f.data_start, &[0, 0, 64, 64]);
    assert!(consume(&mut f.source, "a").is_err());
    assert_sticky(&mut f.source);
    assert_eq!(copied.verified_reads(), 2);
}
#[test]
fn consumer_error_and_unwind_poison_even_after_all_ranges_were_seen() {
    let mut f = Fixture::tiny();
    consume(&mut f.source, "a").unwrap();
    consume(&mut f.source, "b").unwrap();
    assert!(
        f.source
            .visit_raw("a", |_| Err(invalid("injected consumer")))
            .is_err()
    );
    assert_sticky(&mut f.source);
    let mut f = Fixture::tiny();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        f.source
            .visit_raw("a", |_| panic!("injected consumer panic"))
    }));
    assert!(result.is_err());
    assert_sticky(&mut f.source);
}
#[test]
fn nonfinite_float_delivery_invalidates_proof() {
    let mut f = Fixture::new(vec![(
        "a",
        Dtype::F32,
        vec![1],
        f32::NAN.to_le_bytes().to_vec(),
    )]);
    assert!(f.source.visit("a", |_| Ok(())).is_err());
    assert_sticky(&mut f.source);
}
#[test]
fn persistent_append_and_truncation_reject_same_handle() {
    for appended in [false, true] {
        let mut f = Fixture::tiny();
        let file = File::options()
            .write(true)
            .open(f.path.join("model.safetensors"))
            .unwrap();
        file.set_len(f.original.len() as u64 + u64::from(appended))
            .unwrap();
        if !appended {
            file.set_len(f.data_start + 1).unwrap();
        }
        assert!(consume(&mut f.source, "a").is_err());
        assert_sticky(&mut f.source);
    }
}
#[test]
fn callback_extent_change_and_invalid_handle_are_sticky() {
    let mut f = Fixture::tiny();
    let path = f.path.join("model.safetensors");
    assert!(
        f.source
            .visit_raw("a", |_| {
                File::options()
                    .append(true)
                    .open(&path)
                    .unwrap()
                    .write_all(&[0])
                    .unwrap();
                Ok(())
            })
            .is_err()
    );
    assert_sticky(&mut f.source);
    let mut f = Fixture::tiny();
    f.source.shards[0].file = File::open(&f.path).unwrap();
    assert!(consume(&mut f.source, "a").is_err());
    assert_sticky(&mut f.source);
}
#[test]
fn offset_dtype_shape_and_missing_name_cannot_publish() {
    for variant in 0..4 {
        let mut f = Fixture::tiny();
        let a = f.source.audit.as_mut().unwrap();
        match variant {
            0 => a.ranges[0].begin = 1,
            1 => a.ranges[0].dtype = Dtype::F16,
            2 => a.ranges[0].shape = vec![1],
            _ => {}
        }
        assert!(consume(&mut f.source, if variant == 3 { "unknown" } else { "a" }).is_err());
        assert_sticky(&mut f.source);
    }
}
#[test]
fn tickets_check_counts_ordinal_and_checked_counters() {
    let mut f = Fixture::tiny();
    let a = f.source.audit.as_mut().unwrap();
    let mut t = a.begin("a").unwrap();
    assert!(t.update(&[0; 9]).is_err());
    assert!(a.finish(t).is_err());
    assert!(a.begin("a").is_err());
    let mut f = Fixture::tiny();
    let a = f.source.audit.as_mut().unwrap();
    let mut t = a.begin("a").unwrap();
    t.update(&[0; 4]).unwrap();
    assert!(a.finish(t).is_err());
    assert!(a.identity().is_none());
    let mut f = Fixture::tiny();
    let a = f.source.audit.as_mut().unwrap();
    a.reads = usize::MAX;
    assert!(a.begin("a").is_err());
    assert!(a.identity().is_none());
    let mut f = Fixture::tiny();
    let a = f.source.audit.as_mut().unwrap();
    let mut t = a.begin("a").unwrap();
    t.ordinal += 1;
    t.update(&[0, 0, 128, 63, 0, 0, 0, 64]).unwrap();
    assert!(a.finish(t).is_err());
    assert!(a.begin("a").is_err());
}
#[test]
fn legacy_sources_never_acquire_identity() {
    let f = Fixture::tiny();
    let mut source = Source::open(&f.path).unwrap();
    consume(&mut source, "a").unwrap();
    consume(&mut source, "b").unwrap();
    assert!(source.content_identity().is_none());
    assert_eq!(source.audit_owned_bytes(), None);
}
#[cfg(unix)]
#[test]
fn regular_open_rejects_final_symlink_directory_and_fifo_without_blocking() {
    use std::{
        ffi::CString,
        os::unix::{ffi::OsStrExt, fs::symlink},
    };
    let f = Fixture::tiny();
    assert!(open_regular(&f.path.join("model.safetensors")).is_ok());
    assert!(open_regular(&f.path).is_err());
    symlink(f.path.join("model.safetensors"), f.path.join("link")).unwrap();
    assert!(open_regular(&f.path.join("link")).is_err());
    let name = f.path.join("fifo");
    let c = CString::new(name.as_os_str().as_bytes()).unwrap();
    // SAFETY: owned NUL-terminated path remains alive for the synchronous POSIX
    // creation call; the new FIFO is owned exclusively by this test directory.
    assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
    assert!(open_regular(&name).is_err());
}

#[test]
fn mid_read_truncation_reaches_read_exact_failure_and_remains_sticky() {
    let mut f = Fixture::new(vec![(
        "a",
        Dtype::F32,
        vec![CONVERSION_BYTES / 4 + 1],
        vec![0; CONVERSION_BYTES + 4],
    )]);
    let path = f.path.join("model.safetensors");
    let end = f.data_start + CONVERSION_BYTES as u64;
    let mut callbacks = 0;
    let failure = f.source.visit_raw("a", |bytes| {
        callbacks += 1;
        assert_eq!(bytes.len(), CONVERSION_BYTES);
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(end)
            .unwrap();
        Ok(())
    });
    assert!(
        matches!(failure,Err(Error::Io {source,..}) if source.kind()==std::io::ErrorKind::UnexpectedEof)
    );
    assert_eq!(callbacks, 1);
    fs::write(&path, &f.original).unwrap();
    assert_sticky(&mut f.source);
}
