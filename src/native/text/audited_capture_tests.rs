//! Shared offline observation/fault tests; absent from product builds.
use crate::app_chat::content::tests::capture::{Capture, capture_pair};
use serde::Serialize;
use std::{error::Error, fmt::Debug};
#[derive(Debug, PartialEq, Serialize)]
pub(super) struct Observation<T> {
    status: u32,
    reported_count: usize,
    capacity: usize,
    prefix: Vec<T>,
}
pub(super) fn observe<T: Copy>(status: u32, reported_count: usize, buffer: &[T]) -> Observation<T> {
    Observation {
        status,
        reported_count,
        capacity: buffer.len(),
        prefix: buffer[..reported_count.min(buffer.len())].to_vec(),
    }
}
pub(super) fn compare<T: Serialize + Debug + PartialEq>(
    files: &mut [Capture; 2],
    kind: &str,
    ordinal: usize,
    ids: Option<&[u32]>,
    actual: T,
    expected: T,
) -> Result<(), Box<dyn Error>> {
    let values = [Ok(actual), Ok(expected)];
    capture_pair(files, kind, "native-chat", ordinal, ids, &values)?;
    if values[0].as_ref().unwrap() != values[1].as_ref().unwrap() {
        return Err("native value differs from expected result".into());
    }
    Ok(())
}
struct Directory(std::path::PathBuf);
impl Directory {
    fn new() -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static SERIAL: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "toklet-native-observation-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn files(&self) -> [Capture; 2] {
        [
            Capture::new(&self.0.join("actual"), 4096).unwrap(),
            Capture::new(&self.0.join("expected"), 4096).unwrap(),
        ]
    }
    fn actual(&self) -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(self.0.join("actual")).unwrap()).unwrap()
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
#[test]
fn wrong_status_and_oversized_count_retain_bounded_actual_prefix() {
    let directory = Directory::new();
    let mut files = directory.files();
    assert!(
        compare(
            &mut files,
            "bad-status-count",
            0,
            None,
            observe(crate::native::ENGINE_ERROR, usize::MAX, &[11u32, 22, 33]),
            observe(crate::native::OK, 3, &[11u32, 22, 33])
        )
        .is_err()
    );
    let actual = directory.actual();
    let value = &actual["value"];
    assert_eq!(value["status"], crate::native::ENGINE_ERROR);
    assert_eq!(value["reported_count"].as_u64(), Some(usize::MAX as u64));
    assert_eq!(value["capacity"], 3);
    assert_eq!(value["prefix"], serde_json::json!([11, 22, 33]));
}
#[test]
fn invalid_utf8_raw_bytes_survive_before_decode_or_comparison() {
    let directory = Directory::new();
    let mut files = directory.files();
    let bytes = [0xffu8, 0x80];
    assert!(
        compare(
            &mut files,
            "invalid-utf8",
            0,
            None,
            observe(crate::native::OK, 2, &bytes),
            observe(crate::native::OK, 2, b"ok")
        )
        .is_err()
    );
    assert_eq!(
        directory.actual()["value"]["prefix"],
        serde_json::json!([255, 128])
    );
    let captured = directory.actual()["value"]["prefix"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u8::try_from(v.as_u64().unwrap()).unwrap())
        .collect::<Vec<_>>();
    assert!(std::str::from_utf8(&captured).is_err());
}
