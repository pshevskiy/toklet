use super::*;
use serde_json::json;
use std::{
    fs,
    sync::atomic::{AtomicUsize, Ordering},
};

static SERIAL: AtomicUsize = AtomicUsize::new(0);
/// Test-only unaudited fixture. Its absolute payload is deliberately unaligned.
pub(crate) struct Fixture {
    pub path: PathBuf,
    pub data_start: u64,
    pub original: Vec<u8>,
}
impl Fixture {
    pub(crate) fn new(records: Vec<(&str, Dtype, Vec<usize>, Vec<u8>)>) -> Self {
        let path = std::env::temp_dir().join(format!(
            "toklet-range-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, dtype, shape, bytes) in records {
            let start = payload.len();
            payload.extend_from_slice(&bytes);
            let dtype = match dtype {
                Dtype::F16 => "F16",
                Dtype::BF16 => "BF16",
                Dtype::F32 => "F32",
                Dtype::U32 => "U32",
                _ => panic!("unsupported test dtype"),
            };
            header.insert(
                name.into(),
                json!({"dtype":dtype,"shape":shape,"data_offsets":[start,payload.len()]}),
            );
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while (header.len() + 8) % 4 != 1 {
            header.push(b' ');
        }
        let data_start = (header.len() + 8) as u64;
        let mut original = (header.len() as u64).to_le_bytes().to_vec();
        original.extend(header);
        original.extend(payload);
        fs::write(path.join("model.safetensors"), &original).unwrap();
        Self {
            path,
            data_start,
            original,
        }
    }
    pub(crate) fn source(&self) -> Source {
        Source::open(&self.path).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.path).unwrap();
    }
}

#[test]
fn raw_ranges_are_exact_nonsuffix_bytes_and_reuse_buffer() {
    let data: Vec<_> = (0..70_000).map(|n| (n % 251) as u8).collect();
    let f = Fixture::new(vec![("x", Dtype::U32, vec![17_500], data.clone())]);
    assert_eq!(f.data_start % 4, 1);
    let mut source = f.source();
    let ptr = source.buffer.as_ptr();
    let mut result = Vec::new();
    let mut sizes = Vec::new();
    source
        .visit_raw_range("x", 3..65_557, |bytes| {
            assert_eq!(bytes.as_ptr(), ptr);
            sizes.push(bytes.len());
            result.extend_from_slice(bytes);
            Ok(())
        })
        .unwrap();
    assert_eq!(sizes, [65_536, 18]);
    assert_eq!(result, data[3..65_557]);
    for range in [0..1, 40..51, 69_999..70_000, 0..70_000] {
        result.clear();
        source
            .visit_raw_range("x", range.clone(), |bytes| {
                result.extend_from_slice(bytes);
                Ok(())
            })
            .unwrap();
        assert_eq!(result, data[range]);
    }
    let before = source.shards[0].file.stream_position().unwrap();
    let mut called = false;
    for range in [
        0..0,
        std::ops::Range { start: 10, end: 9 },
        0..70_001,
        usize::MAX..usize::MAX,
    ] {
        assert!(
            source
                .visit_raw_range("x", range, |_| {
                    called = true;
                    Ok(())
                })
                .is_err()
        );
    }
    assert!(
        source
            .visit_raw_range("missing", 0..1, |_| {
                called = true;
                Ok(())
            })
            .is_err()
    );
    assert!(!called);
    assert_eq!(source.shards[0].file.stream_position().unwrap(), before);
    assert_eq!(source.buffer.as_ptr(), ptr);
}

#[test]
fn range_absolute_offset_overflow_precedes_seek_or_callback() {
    let f = Fixture::new(vec![("x", Dtype::U32, vec![2], vec![0; 8])]);
    let mut source = f.source();
    let before = source.shards[0].file.stream_position().unwrap();
    let start = source.shards[0].data_start;
    source.shards[0].data_start = u64::MAX - 1;
    let mut called = false;
    assert!(
        source
            .visit_raw_range("x", 0..2, |_| {
                called = true;
                Ok(())
            })
            .is_err()
    );
    assert!(!called);
    assert_eq!(source.shards[0].file.stream_position().unwrap(), before);
    source.shards[0].data_start = start;
    source.visit_raw_range("x", 0..8, |_| Ok(())).unwrap();
}

#[test]
fn range_consumer_failure_and_truncation_allow_valid_reread() {
    let data: Vec<_> = (0..70_000).map(|n| (n % 239) as u8).collect();
    let f = Fixture::new(vec![("x", Dtype::U32, vec![17_500], data.clone())]);
    let mut source = f.source();
    let mut calls = 0;
    assert!(
        source
            .visit_raw_range("x", 1..69_999, |_| {
                calls += 1;
                Err(invalid("injected consumer failure"))
            })
            .is_err()
    );
    assert_eq!(calls, 1);
    let file = fs::OpenOptions::new()
        .write(true)
        .open(f.path.join("model.safetensors"))
        .unwrap();
    file.set_len(f.data_start + 65_540).unwrap();
    calls = 0;
    assert!(
        source
            .visit_raw_range("x", 1..69_999, |_| {
                calls += 1;
                Ok(())
            })
            .is_err()
    );
    assert_eq!(calls, 1);
    fs::write(f.path.join("model.safetensors"), &f.original).unwrap();
    let mut actual = Vec::new();
    source
        .visit_raw_range("x", 1..69_999, |bytes| {
            actual.extend_from_slice(bytes);
            Ok(())
        })
        .unwrap();
    assert_eq!(actual, data[1..69_999]);
}

#[test]
fn every_audited_range_attempt_invalidates_complete_identity_before_io() {
    for (name, range) in [("x", 0..8), ("x", 1..2), ("missing", 0..1), ("x", 0..0)] {
        let mut f = audit::tests::Fixture::new(vec![("x", Dtype::F32, vec![2], vec![0; 8])]);
        f.source.visit_raw("x", |_| Ok(())).unwrap();
        let copied = f.source.content_identity().unwrap();
        let before = f.source.shards[0].file.stream_position().unwrap();
        let mut called = false;
        assert!(
            f.source
                .visit_raw_range(name, range, |_| {
                    called = true;
                    Ok(())
                })
                .is_err()
        );
        assert!(!called);
        assert!(f.source.content_identity().is_none());
        assert_eq!(f.source.shards[0].file.stream_position().unwrap(), before);
        assert!(
            f.source
                .visit_raw("x", |_| {
                    called = true;
                    Ok(())
                })
                .is_err()
        );
        assert!(!called);
        assert_eq!(copied.verified_reads(), 1);
    }
    // Control: repeated whole reads still complete and retain the original audit.
    let mut f = audit::tests::Fixture::new(vec![("x", Dtype::F32, vec![2], vec![0; 8])]);
    f.source.visit_raw("x", |_| Ok(())).unwrap();
    f.source.visit("x", |_| Ok(())).unwrap();
    assert_eq!(f.source.content_identity().unwrap().verified_reads(), 2);
}
