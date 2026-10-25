//! Exact pinned text-stream diagnostic; synthetic long IDs are not LLM output.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::{File, OpenOptions},
    io::{self, BufWriter, Write},
    path::PathBuf,
};
use toklet::app_chat::{
    gemma4::{self, Codec},
    stream::ResponseLimits,
};

const FIXTURE: &[u8] = include_bytes!("../tests/fixtures/gemma4-stream-v1/reference.json");
const DIAGNOSTIC_ALLOWANCE: usize = 16 * 1024 * 1024;
const BINARY_LIMIT: usize = 16 * 1024 * 1024;
struct Capture {
    output: BufWriter<File>,
    bytes: usize,
}
impl Capture {
    fn write(&mut self, bytes: &[u8]) -> io::Result<()> {
        let end = self
            .bytes
            .checked_add(bytes.len())
            .filter(|n| *n <= BINARY_LIMIT)
            .ok_or_else(|| io::Error::other("fixed stream diagnostic byte limit exceeded"))?;
        self.output.write_all(bytes)?;
        self.bytes = end;
        Ok(())
    }
}
fn ids(value: &Value) -> Result<Vec<u32>, Box<dyn Error>> {
    Ok(serde_json::from_value(value.clone())?)
}
fn publish(output: &mut File, report: &mut Value, actual: &Value) -> Result<(), Box<dyn Error>> {
    // Keep the exact record through any serialization/write/flush failure.
    report["active_observation"] = actual.clone();
    serde_json::to_writer(&mut *output, actual)?;
    output.write_all(b"\n")?;
    output.flush()?;
    report["completed_records"] = (report["completed_records"]
        .as_u64()
        .ok_or("record count absent")?
        + 1)
    .into();
    report["active_case"] = Value::Null;
    report["active_observation"] = Value::Null;
    Ok(())
}
fn small_chunk(
    observed: &mut Value,
    through: usize,
    text: &str,
    tail: bool,
    prefix: Option<&str>,
    expected: &str,
) -> Result<(), Box<dyn Error>> {
    // Retain successful decoder output before interpreting/comparing it.
    let mut visible = observed["text"]
        .as_str()
        .ok_or("partial text absent")?
        .to_owned();
    visible.push_str(text);
    observed["text"] = visible.into();
    observed["tokens"] = through.into();
    if tail {
        observed["tail"] = text.into();
    } else {
        observed["chunks"]
            .as_array_mut()
            .ok_or("partial chunks absent")?
            .push(json!({"through_tokens":through,"text":text}));
    }
    observed["stage"] = if tail {
        "final_compare"
    } else {
        "prefix_compare"
    }
    .into();
    let visible = observed["text"].as_str().ok_or("partial text absent")?;
    if !prefix.ok_or("prefix absent")?.starts_with(visible)
        || !expected.starts_with(visible)
        || (tail && visible != expected)
    {
        return Err("stream text/prefix differs; actual output retained".into());
    }
    Ok(())
}
fn small(
    codec: &Codec,
    case: &Value,
    batch: usize,
    observed: &mut Value,
) -> Result<Value, Box<dyn Error>> {
    *observed = json!({"kind":"case","name":case["name"],"batch":batch,"chunks":[],"tail":null,"text":"","tokens":0,"stage":"create"});
    let input = ids(&case["ids"])?;
    let expected = case["expected"].as_str().ok_or("expected text absent")?;
    let mut stream = codec.response_stream(
        ResponseLimits::new(input.len().max(1), 64 * 1024)?,
        gemma4::STREAM_ALLOWANCE_BYTES,
    )?;
    let allocated = stream.allocated_bytes()?;
    observed["allocated_bytes"] = allocated.into();
    for input in input.chunks(batch) {
        observed["stage"] = "push".into();
        let text = stream.push(input)?.to_owned();
        let through = stream.consumed_tokens();
        small_chunk(
            observed,
            through,
            &text,
            false,
            case["prefixes"][through].as_str(),
            expected,
        )?;
        if stream.allocated_bytes()? != allocated {
            return Err("stream storage grew".into());
        }
    }
    observed["stage"] = "finish".into();
    let tail = stream.finish()?.to_owned();
    small_chunk(
        observed,
        stream.consumed_tokens(),
        &tail,
        true,
        Some(expected),
        expected,
    )?;
    observed["utf8_bytes"] = stream.emitted_bytes().into();
    if stream.allocated_bytes()? != allocated
        || !stream.is_finished()
        || stream.is_failed()
        || stream.consumed_tokens() != input.len()
        || stream.emitted_bytes()
            != observed["text"]
                .as_str()
                .ok_or("partial text absent")?
                .len()
    {
        return Err("completed stream state differs".into());
    }
    observed["stage"] = "complete".into();
    Ok(std::mem::take(observed))
}

fn long(
    codec: &Codec,
    case: &Value,
    batch: usize,
    capture: &mut Capture,
) -> Result<Value, Box<dyn Error>> {
    let pattern = ids(&case["pattern_ids"])?;
    let suffix = ids(&case["suffix_ids"])?;
    let count = case["tokens"].as_u64().ok_or("long token count absent")? as usize;
    if count != 131072 || pattern.is_empty() || suffix.len() > count {
        return Err("fixed long stream plan differs".into());
    }
    let mut stream = codec.response_stream(
        ResponseLimits::new(count, 8 * 1024 * 1024)?,
        gemma4::STREAM_ALLOWANCE_BYTES,
    )?;
    let allocated = stream.allocated_bytes()?;
    let start_offset = capture.bytes;
    let mut hash = Sha256::new();
    let mut input = Vec::with_capacity(128);
    for start in (0..count).step_by(batch) {
        input.clear();
        for position in start..(start + batch).min(count) {
            input.push(if position < count - suffix.len() {
                pattern[position % pattern.len()]
            } else {
                suffix[position - (count - suffix.len())]
            });
        }
        let chunk = stream.push(&input)?;
        capture.write(chunk.as_bytes())?;
        hash.update(chunk.as_bytes());
        if case["name"]
            .as_str()
            .ok_or("long case name absent")?
            .contains("fallback")
            && capture.bytes != start_offset
        {
            return Err("whole fallback run was published before finish".into());
        }
        if stream.allocated_bytes()? != allocated {
            return Err("stream storage grew".into());
        }
    }
    let tail = stream.finish()?;
    capture.write(tail.as_bytes())?;
    hash.update(tail.as_bytes());
    capture.output.flush()?;
    let digest = format!("{:x}", hash.finalize());
    let bytes = capture.bytes - start_offset;
    if digest != case["expected_sha256"].as_str().ok_or("long hash absent")?
        || bytes as u64
            != case["expected_utf8_bytes"]
                .as_u64()
                .ok_or("long bytes absent")?
        || stream.allocated_bytes()? != allocated
        || stream.emitted_bytes() != bytes
        || stream.consumed_tokens() != count
    {
        return Err("long stream hash/storage/count differs".into());
    }
    Ok(
        json!({"kind":"long","name":case["name"],"batch":batch,"offset":start_offset,"bytes":bytes,"sha256":digest,
        "tokens":count,"allocated_bytes":allocated,"completed":true}),
    )
}
fn vocabulary_record(
    id: u32,
    actual: &str,
    capture: &mut Capture,
    observed: &mut Value,
    expected: impl FnOnce() -> Result<String, Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    observed["active_id"] = id.into();
    observed["record_offset"] = capture.bytes.into();
    observed["actual"] = actual.into();
    observed["stage"] = "write_id".into();
    capture.write(&id.to_le_bytes())?;
    observed["stage"] = "write_length".into();
    capture.write(&u32::try_from(actual.len())?.to_le_bytes())?;
    observed["stage"] = "write_actual_utf8".into();
    capture.write(actual.as_bytes())?;
    observed["stage"] = "flush_actual".into();
    capture.output.flush()?;
    observed["stage"] = "expected_decode".into();
    let expected = expected()?;
    observed["expected"] = expected.into();
    observed["stage"] = "compare".into();
    if observed["actual"] != observed["expected"] {
        return Err(format!("vocabulary stream differs at ID{id}; actual output retained").into());
    }
    observed["stage"] = "complete".into();
    Ok(())
}
fn vocabulary(
    codec: &Codec,
    capture: &mut Capture,
    observed: &mut Value,
) -> Result<Value, Box<dyn Error>> {
    let start = capture.bytes;
    let mut maximum = 0;
    *observed = json!({"kind":"vocabulary","offset":start,"completed_ids":0});
    for id in 0..gemma4::VOCAB as u32 {
        observed["active_id"] = id.into();
        observed["record_offset"] = capture.bytes.into();
        observed["stage"] = "stream_create".into();
        observed["actual"] = Value::Null;
        observed["expected"] = Value::Null;
        // Single-ID limits avoid maximum-response allocation per vocabulary ID.
        let mut stream = codec.response_stream(
            ResponseLimits::new(1, 4096)?,
            gemma4::STREAM_ALLOWANCE_BYTES,
        )?;
        let allocated = stream.allocated_bytes()?;
        maximum = maximum.max(allocated);
        let mut actual = String::with_capacity(4096);
        observed["stage"] = "stream_push".into();
        actual.push_str(stream.push(&[id])?);
        observed["actual"] = actual.clone().into();
        observed["stage"] = "stream_finish".into();
        actual.push_str(stream.finish()?);
        observed["actual"] = actual.clone().into();
        vocabulary_record(id, &actual, capture, observed, || {
            Ok(codec.decode(&[id], true)?)
        })?;
        if stream.consumed_tokens() != 1
            || stream.emitted_bytes() != actual.len()
            || stream.allocated_bytes()? != allocated
        {
            return Err("vocabulary stream state/storage differs".into());
        }
        observed["completed_ids"] = (id + 1).into();
    }
    observed["stage"] = "flush".into();
    capture.output.flush()?;
    Ok(
        json!({"kind":"vocabulary","offset":start,"bytes":capture.bytes-start,"ids":gemma4::VOCAB,
        "framing":"little-endian u32 ID/u32 UTF-8 length followed by exact UTF-8 bytes","maximum_allocated_bytes":maximum,"completed":true}),
    )
}

fn run(root: PathBuf, directory: PathBuf) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir(&directory)?;
    let mut capture = Capture {
        output: BufWriter::with_capacity(
            64 * 1024,
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(directory.join("stream.bin"))?,
        ),
        bytes: 0,
    };
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("actual.jsonl"))?;
    let allowance = gemma4::ALLOWANCE_BYTES
        .checked_add(gemma4::STREAM_ALLOWANCE_BYTES)
        .and_then(|n| n.checked_add(DIAGNOSTIC_ALLOWANCE))
        .ok_or("stream admission overflow")?;
    let mut report = json!({"format":"toklet-gemma4-native-stream-v1","passed":false,"primary_error":null,"endpoint_errors":[],
        "completed_records":0,"active_case":null,"active_observation":null,"stream_comparison_complete":false,"codec_allowance_bytes":gemma4::ALLOWANCE_BYTES,
        "stream_allowance_bytes":gemma4::STREAM_ALLOWANCE_BYTES,"diagnostic_allowance_bytes":DIAGNOSTIC_ALLOWANCE,
        "combined_admitted_bytes":allowance,"binary_limit_bytes":BINARY_LIMIT,"native_model_executed":false,
        "native_app_profile_admitted":false,"numerical_output_capacity_qualified":false,"app_eligibility_changed":false});
    let result = (|| -> Result<(), Box<dyn Error>> {
        if format!("{:x}", Sha256::digest(FIXTURE))
            != "620b1cb817a7e3976eaa7bdc910545133a3fa645661e4b25f39c2d59720c8158"
        {
            return Err("frozen stream fixture differs".into());
        }
        let fixture: Value = serde_json::from_slice(FIXTURE)?;
        let codec = Codec::load(&root, gemma4::ALLOWANCE_BYTES)?;
        let cases = fixture["cases"].as_array().ok_or("stream cases absent")?;
        let long_cases = fixture["long_cases"]
            .as_array()
            .ok_or("long cases absent")?;
        if cases.len() != 67 || long_cases.len() != 5 {
            return Err("fixed stream inventory differs".into());
        }
        for case in cases {
            for batch in [1, 3, 7, 128] {
                report["active_case"] = json!({"kind":"case","name":case["name"],"batch":batch});
                let actual = small(&codec, case, batch, &mut report["active_observation"])?;
                publish(&mut output, &mut report, &actual)?;
            }
        }
        for case in long_cases {
            for batch in [1, 3, 128] {
                report["active_case"] =
                    json!({"kind":"long","name":case["name"],"batch":batch,"offset":capture.bytes});
                let actual = long(&codec, case, batch, &mut capture)?;
                publish(&mut output, &mut report, &actual)?;
            }
        }
        report["active_case"] = json!({"kind":"vocabulary","offset":capture.bytes});
        let actual = vocabulary(&codec, &mut capture, &mut report["active_observation"])?;
        publish(&mut output, &mut report, &actual)?;
        if report["completed_records"] != 284 {
            return Err("incomplete stream record inventory".into());
        }
        report["stream_comparison_complete"] = true.into();
        Ok(())
    })();
    report["primary_error"] = result.as_ref().err().map(ToString::to_string).into();
    if let Err(error) = capture.output.flush() {
        report["endpoint_errors"]
            .as_array_mut()
            .unwrap()
            .push(error.to_string().into());
    }
    report["completed_binary_bytes"] = capture.bytes.into();
    report["passed"] =
        (result.is_ok() && report["endpoint_errors"].as_array().unwrap().is_empty()).into();
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("observation.json"))?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.flush()?;
    println!(
        "{}",
        json!({"passed":report["passed"],"records":report["completed_records"],"primary_error":report["primary_error"]})
    );
    result?;
    if report["passed"] != true {
        return Err("stream endpoint failed; observations retained".into());
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let usage = "usage: gemma4_stream CHECKPOINT_DIRECTORY NEW_OUTPUT_DIRECTORY";
    let root = PathBuf::from(args.next().ok_or(usage)?);
    let output = PathBuf::from(args.next().ok_or(usage)?);
    if args.next().is_some() {
        return Err(usage.into());
    }
    run(root, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Temporary(PathBuf);
    impl Temporary {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "toklet-gemma-stream-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn capture(&self) -> Capture {
            Capture {
                output: BufWriter::with_capacity(
                    64,
                    File::create_new(self.0.join("stream.bin")).unwrap(),
                ),
                bytes: 0,
            }
        }
    }
    impl Drop for Temporary {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
    #[test]
    fn later_prefix_mismatch_retains_every_successful_chunk() {
        let mut observed = json!({"text":"","chunks":[],"tail":null});
        small_chunk(&mut observed, 1, "x", false, Some("x"), "xZ").unwrap();
        let error = small_chunk(&mut observed, 2, "Y", false, Some("xZ"), "xZ").unwrap_err();
        assert!(error.to_string().contains("actual output retained"));
        assert_eq!(observed["text"], "xY");
        assert_eq!(observed["tokens"], 2);
        assert_eq!(observed["stage"], "prefix_compare");
        assert_eq!(
            observed["chunks"],
            json!([{"through_tokens":1,"text":"x"},{"through_tokens":2,"text":"Y"}])
        );
    }
    #[test]
    fn expected_decode_failure_and_mismatch_preserve_framed_actual_before_compare() {
        let temp = Temporary::new();
        let mut capture = temp.capture();
        let mut observed = json!({});
        let error = vocabulary_record(17, "actual", &mut capture, &mut observed, || {
            Err(io::Error::other("synthetic expected-decode failure").into())
        })
        .unwrap_err();
        assert_eq!(error.to_string(), "synthetic expected-decode failure");
        let mut expected = Vec::new();
        expected.extend(17_u32.to_le_bytes());
        expected.extend(6_u32.to_le_bytes());
        expected.extend(b"actual");
        assert_eq!(std::fs::read(temp.0.join("stream.bin")).unwrap(), expected);
        assert_eq!(observed["active_id"], 17);
        assert_eq!(observed["actual"], "actual");
        assert_eq!(observed["record_offset"], 0);
        assert_eq!(observed["stage"], "expected_decode");
        let error = vocabulary_record(18, "other", &mut capture, &mut observed, || {
            Ok("different".into())
        })
        .unwrap_err();
        assert!(error.to_string().contains("ID18"));
        assert_eq!(observed["actual"], "other");
        assert_eq!(observed["expected"], "different");
        assert_eq!(observed["stage"], "compare");
        expected.extend(18_u32.to_le_bytes());
        expected.extend(5_u32.to_le_bytes());
        expected.extend(b"other");
        assert_eq!(std::fs::read(temp.0.join("stream.bin")).unwrap(), expected);
    }
    #[test]
    fn partial_vocabulary_record_identifies_failed_stage_without_losing_first_header() {
        let temp = Temporary::new();
        let mut capture = temp.capture();
        capture.bytes = BINARY_LIMIT - 4;
        let mut observed = json!({});
        let mut called = false;
        let error = vocabulary_record(17, "actual", &mut capture, &mut observed, || {
            called = true;
            Ok("actual".into())
        })
        .unwrap_err();
        assert!(error.to_string().contains("byte limit"));
        assert!(!called);
        capture.output.flush().unwrap();
        assert_eq!(
            std::fs::read(temp.0.join("stream.bin")).unwrap(),
            17_u32.to_le_bytes()
        );
        assert_eq!(observed["active_id"], 17);
        assert_eq!(observed["record_offset"], BINARY_LIMIT - 4);
        assert_eq!(observed["stage"], "write_length");
        assert_eq!(observed["actual"], "actual");
        assert_eq!(capture.bytes, BINARY_LIMIT);
    }
    #[test]
    fn failed_publication_retains_exact_completed_record_and_original_io_error() {
        let temp = Temporary::new();
        let path = temp.0.join("read-only.jsonl");
        File::create_new(&path).unwrap();
        let mut output = File::open(&path).unwrap();
        let actual = json!({"kind":"case","text":"retained","chunks":[{"through_tokens":1,"text":"retained"}]});
        let active_case = json!({"kind":"case","name":"synthetic","batch":1});
        let mut report =
            json!({"completed_records":0,"active_case":active_case,"active_observation":null});
        let error = publish(&mut output, &mut report, &actual).unwrap_err();
        assert!(
            error
                .downcast_ref::<serde_json::Error>()
                .unwrap()
                .io_error_kind()
                .is_some()
        );
        assert_eq!(report["completed_records"], 0);
        assert_eq!(report["active_case"], active_case);
        assert_eq!(report["active_observation"], actual);
        assert_eq!(std::fs::metadata(path).unwrap().len(), 0);
    }
}
