//! Fault-tested bounded diagnostic capture; never part of the product library.
use super::*;
use serde::Serialize;
use std::{
    fmt::{self, Debug, Write as _},
    fs::File,
    io::Write,
};
const FRAME_LIMIT: usize = 1024 * 1024;
struct Frame(Vec<u8>);
impl Write for Frame {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self
            .0
            .len()
            .checked_add(bytes.len())
            .is_none_or(|n| n > FRAME_LIMIT)
        {
            return Err(invalid("capture frame exceeds admitted bound"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
struct Active {
    text: String,
    truncated: bool,
}
impl fmt::Write for Active {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        let available = FRAME_LIMIT - self.text.len();
        if value.len() <= available {
            self.text.push_str(value);
            return Ok(());
        }
        let mut end = available;
        while !value.is_char_boundary(end) {
            end -= 1;
        }
        self.text.push_str(&value[..end]);
        self.truncated = true;
        Err(fmt::Error)
    }
}
#[derive(Debug)]
pub(super) struct CaptureFailure {
    codec_errors: Vec<String>,
    capture_errors: Vec<String>,
    active: String,
    active_truncated: bool,
}
impl fmt::Display for CaptureFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "codec_errors={:?}; capture_errors={:?}; active_pair_truncated={}; active_pair={}",
            self.codec_errors, self.capture_errors, self.active_truncated, self.active
        )
    }
}
impl std::error::Error for CaptureFailure {}
pub(super) struct Capture {
    file: File,
    bytes: usize,
    limit: usize,
}
impl Capture {
    pub(super) fn new(path: &Path, limit: usize) -> io::Result<Self> {
        Ok(Self {
            file: File::options().write(true).create_new(true).open(path)?,
            bytes: 0,
            limit,
        })
    }
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn bytes(&self) -> usize {
        self.bytes
    }
    fn record<T: Serialize>(
        &mut self,
        kind: &str,
        group: &str,
        ordinal: usize,
        input: Option<&[u32]>,
        result: &io::Result<T>,
    ) -> io::Result<()> {
        #[derive(Serialize)]
        struct Record<'a, T> {
            kind: &'a str,
            group: &'a str,
            ordinal: usize,
            input: Option<&'a [u32]>,
            value: Option<&'a T>,
            error: Option<String>,
        }
        let record = Record {
            kind,
            group,
            ordinal,
            input,
            value: result.as_ref().ok(),
            error: result.as_ref().err().map(ToString::to_string),
        };
        let mut frame = Frame(Vec::new());
        serde_json::to_writer(&mut frame, &record)?;
        frame.write_all(b"\n")?;
        let next = self
            .bytes
            .checked_add(frame.0.len())
            .filter(|&n| n <= self.limit)
            .ok_or_else(|| invalid("capture exceeds admitted bound"))?;
        self.file.write_all(&frame.0)?;
        self.bytes = next;
        Ok(())
    }
}
pub(super) fn capture_pair<T: Serialize + Debug>(
    files: &mut [Capture; 2],
    kind: &str,
    group: &str,
    ordinal: usize,
    input: Option<&[u32]>,
    values: &[io::Result<T>; 2],
) -> Result<(), CaptureFailure> {
    // Retain the active pair BEFORE serialization/bounds/writes can fail. For
    // actual bounded codec outputs (2 x 64 KiB text or 2 x 512 IDs), 1 MiB also
    // covers debug escaping. Arbitrary fault-test values may truncate explicitly.
    let mut active = Active {
        text: String::new(),
        truncated: false,
    };
    let _ = write!(
        &mut active,
        "{kind}/{group}/{ordinal} input={input:?} values={values:?}"
    );
    let codec_errors = values
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.as_ref().err().map(|e| format!("side {i}: {e}")))
        .collect::<Vec<_>>();
    // Attempt BOTH captures even after a first-side error. Old complete frames
    // and any partial late write remain on disk; the owned failure retains the
    // complete active pair and original codec error alongside capture errors.
    let first = files[0].record(kind, group, ordinal, input, &values[0]);
    let second = files[1].record(kind, group, ordinal, input, &values[1]);
    let capture_errors = [first, second]
        .into_iter()
        .enumerate()
        .filter_map(|(i, v)| v.err().map(|e| format!("side {i}: {e}")))
        .collect::<Vec<_>>();
    if !codec_errors.is_empty() || !capture_errors.is_empty() {
        return Err(CaptureFailure {
            codec_errors,
            capture_errors,
            active: active.text,
            active_truncated: active.truncated,
        });
    }
    Ok(())
}
#[test]
fn first_write_failure_preserves_second_active_pair_and_original_codec_error() {
    let directory = Directory::new();
    let left = directory.0.join("left");
    fs::write(&left, b"prior complete evidence\n").unwrap();
    let mut files = [
        Capture {
            file: File::open(&left).unwrap(),
            bytes: 24,
            limit: 4096,
        },
        Capture::new(&directory.0.join("right"), 4096).unwrap(),
    ];
    let values = [
        Err(io::Error::other("original codec failure")),
        Ok("second complete actual"),
    ];
    let error = capture_pair(&mut files, "decode", "fault", 0, Some(&[9]), &values).unwrap_err();
    assert!(error.codec_errors[0].contains("original codec failure"));
    assert_eq!(error.capture_errors.len(), 1);
    assert!(error.capture_errors[0].contains("side 0"));
    assert!(
        error.active.contains("original codec failure")
            && error.active.contains("second complete actual")
    );
    assert!(!error.active_truncated);
    assert_eq!(fs::read(left).unwrap(), b"prior complete evidence\n");
    let right: serde_json::Value =
        serde_json::from_slice(&fs::read(directory.0.join("right")).unwrap()).unwrap();
    assert_eq!(right["value"], "second complete actual");
}
#[test]
fn bound_and_serializer_failure_keep_both_active_results() {
    let directory = Directory::new();
    let mut files = [
        Capture::new(&directory.0.join("bound-left"), 1).unwrap(),
        Capture::new(&directory.0.join("bound-right"), 1).unwrap(),
    ];
    let error = capture_pair(
        &mut files,
        "decode",
        "bound",
        0,
        None,
        &[Ok("first actual"), Ok("second actual")],
    )
    .unwrap_err();
    assert_eq!(error.capture_errors.len(), 2);
    assert!(error.active.contains("first actual") && error.active.contains("second actual"));
    #[derive(Debug)]
    struct Fails(&'static str);
    impl Serialize for Fails {
        fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom(format!(
                "serialization failed for {}",
                self.0
            )))
        }
    }
    let mut files = [
        Capture::new(&directory.0.join("serialize-left"), 4096).unwrap(),
        Capture::new(&directory.0.join("serialize-right"), 4096).unwrap(),
    ];
    let error = capture_pair(
        &mut files,
        "decode",
        "serialize",
        0,
        None,
        &[Ok(Fails("first actual")), Ok(Fails("second actual"))],
    )
    .unwrap_err();
    assert_eq!(error.capture_errors.len(), 2);
    assert!(error.active.contains("first actual") && error.active.contains("second actual"));
    let mut frame = Frame(Vec::new());
    assert!(frame.write_all(&vec![0; FRAME_LIMIT + 1]).is_err());
    assert!(frame.0.is_empty());
}
#[test]
fn later_comparison_failure_retains_both_complete_records() {
    let directory = Directory::new();
    let mut files = [
        Capture::new(&directory.0.join("left"), 4096).unwrap(),
        Capture::new(&directory.0.join("right"), 4096).unwrap(),
    ];
    capture_pair(
        &mut files,
        "decode",
        "comparison",
        0,
        None,
        &[Ok("prior"), Ok("prior")],
    )
    .unwrap();
    let values = [Ok("actual left"), Ok("actual right")];
    capture_pair(&mut files, "decode", "comparison", 1, None, &values).unwrap();
    assert_ne!(values[0].as_ref().unwrap(), values[1].as_ref().unwrap());
    for (path, expected) in [("left", "actual left"), ("right", "actual right")] {
        let body = fs::read_to_string(directory.0.join(path)).unwrap();
        let rows = body
            .lines()
            .map(|s| serde_json::from_str::<serde_json::Value>(s).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["value"], "prior");
        assert_eq!(rows[1]["value"], expected);
    }
}
