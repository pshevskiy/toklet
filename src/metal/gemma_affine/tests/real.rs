//! Explicit real-model diagnostic. Ordinary tests never open model/artifact files.
use super::*;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

const MODEL_BYTES: u64 = 3550670554;
const RAW_LIMIT: usize = 128 * 1024 * 1024;

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn file_hash(path: &Path, limit: u64) -> String {
    let mut file = File::open(path).unwrap();
    assert!(file.metadata().unwrap().is_file());
    assert!(file.metadata().unwrap().len() <= limit);
    let mut digest = Sha256::new();
    let mut total = 0;
    let mut buffer = [0; 65536];
    loop {
        let n = file.read(&mut buffer).unwrap();
        if n == 0 {
            break;
        }
        total += n as u64;
        assert!(total <= limit);
        digest.update(&buffer[..n]);
    }
    format!("{:x}", digest.finalize())
}
fn bounded(path: &Path, limit: usize, expected: &str) -> Vec<u8> {
    let mut result = Vec::new();
    File::open(path)
        .unwrap()
        .take(limit as u64 + 1)
        .read_to_end(&mut result)
        .unwrap();
    assert!(result.len() <= limit);
    assert_eq!(hash(&result), expected);
    result
}
fn array(file: &mut File, entry: &Value, width: usize) -> Vec<f32> {
    assert_eq!(entry["dtype"], "F32");
    assert_eq!(entry["shape"], json!([4, width]));
    assert_eq!(entry["completed"], true);
    let count = width.checked_mul(16).unwrap();
    assert!(count <= 196608);
    assert_eq!(entry["bytes"].as_u64().unwrap(), count as u64);
    let start = entry["offset"].as_u64().unwrap();
    assert!(start.checked_add(count as u64).unwrap() <= file.metadata().unwrap().len());
    let mut raw = vec![0; count];
    file.seek(SeekFrom::Start(start)).unwrap();
    file.read_exact(&mut raw).unwrap();
    assert_eq!(hash(&raw), entry["sha256"].as_str().unwrap());
    let values: Vec<_> = raw
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert!(values.iter().all(|v| v.is_finite()));
    values
}
fn tensor(
    file: &mut File,
    header: &Value,
    base: u64,
    name: &str,
    shape: Value,
    dtype: &str,
    count: usize,
) -> (Vec<u8>, Value) {
    assert!(count <= 128 * 1024 * 1024);
    let spec = &header[name];
    assert_eq!(spec["shape"], shape);
    assert_eq!(spec["dtype"], dtype);
    let start = spec["data_offsets"][0].as_u64().unwrap();
    let end = spec["data_offsets"][1].as_u64().unwrap();
    assert_eq!(end.checked_sub(start).unwrap(), count as u64);
    let absolute = base.checked_add(start).unwrap();
    assert!(base.checked_add(end).unwrap() <= MODEL_BYTES);
    let mut raw = vec![0; count];
    file.seek(SeekFrom::Start(absolute)).unwrap();
    file.read_exact(&mut raw).unwrap();
    let record = json!({"name":name,"shape":shape,"dtype":dtype,"offset":absolute,"bytes":count,"sha256":hash(&raw)});
    (raw, record)
}
struct Capture {
    file: File,
    bytes: usize,
}
impl Capture {
    fn append(&mut self, key: &str, role: &str, raw: &[u8], tokens: usize, rows: usize) -> Value {
        assert_eq!(
            raw.len(),
            tokens.checked_mul(rows).unwrap().checked_mul(4).unwrap()
        );
        assert!(raw.len() <= 196608);
        let next = self.bytes.checked_add(raw.len()).unwrap();
        assert!(next <= RAW_LIMIT);
        let item = json!({"key":key,"role":role,"dtype":"F32","shape":[tokens,rows],"offset":self.bytes,"bytes":raw.len(),"sha256":hash(raw)});
        report("GEMMA_REAL_ARRAY_ATTEMPT", item.clone());
        self.file.write_all(raw).unwrap();
        self.file.flush().unwrap();
        self.bytes = next;
        report("GEMMA_REAL_ARRAY", item.clone());
        item
    }
}
fn raw_values(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn digest_words(words: &[u32]) -> Value {
    assert!(words.len() >= GUARD);
    let mut digest = Sha256::new();
    for word in words {
        digest.update(word.to_le_bytes());
    }
    json!({"words":words.len(),"bytes":words.len()*4,"sha256":format!("{:x}",digest.finalize()),"guards":words[words.len()-GUARD..].to_vec()})
}
// Capture actual device output independently of the transactional caller view.
// On an apply error, only an explicitly Completed retained command permits a
// shared-buffer read; a preflight/unfinished/error command supplies no result.
fn completed_device(
    owner: &Affine<'_>,
    succeeded: bool,
    active: usize,
) -> Option<(Value, Vec<u8>)> {
    if !succeeded
        && owner
            .pending
            .as_ref()
            .is_none_or(|command| command.status() != MTLCommandBufferStatus::Completed)
    {
        return None;
    }
    let metadata = json!({"codes":owner.codes.with_slice(digest_words),"scales":owner.scales.with_slice(digest_words),"biases":owner.biases.with_slice(digest_words),
        "input":owner.input.with_slice(digest_words),"output":owner.output.with_slice(digest_words),"status":owner.status.with_slice(digest_words)});
    let raw = owner.output.with_slice(|words| {
        words[..active]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    });
    Some((metadata, raw))
}
// The original apply cause and completed-buffer metadata must survive even if
// persisting the raw result fails. This coarse diagnostic writer is test-only.
fn publish_completed<W: Write>(
    log: &mut W,
    outcome: &Value,
    capture: &mut Capture,
    key: &str,
    raw: Option<&[u8]>,
    tokens: usize,
    rows: usize,
) -> Option<Value> {
    writeln!(log, "GEMMA_REAL_STEP_OUTCOME {outcome}").unwrap();
    log.flush().unwrap();
    raw.map(|bytes| capture.append(key, "device-output", bytes, tokens, rows))
}
#[test]
fn completed_error_survives_failed_raw_writer_without_gpu() {
    // A read-only descriptor deterministically rejects writes, including when
    // tests run as a privileged user. No fixture or source is modified.
    let mut capture = Capture {
        file: File::open(file!()).unwrap(),
        bytes: 0,
    };
    let outcome = json!({"completed":false,"error":"original completed apply error","device":{"status":{"words":8,"guards":vec![SENTINEL;7]}}});
    let mut log = Vec::new();
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        publish_completed(
            &mut log,
            &outcome,
            &mut capture,
            "injected",
            Some(&[0; 4]),
            1,
            1,
        )
    }));
    assert!(failed.is_err());
    assert_eq!(capture.bytes, 0);
    let line = String::from_utf8(log).unwrap();
    let body = line.strip_prefix("GEMMA_REAL_STEP_OUTCOME ").unwrap();
    assert_eq!(serde_json::from_str::<Value>(body).unwrap(), outcome);
}
fn step(
    owner: &mut Affine<'_>,
    capture: &mut Capture,
    name: &str,
    phase: &str,
    index: usize,
    input: &[f32],
    tokens: usize,
) -> Vec<f32> {
    let key = format!("{name}/{phase}/{index}");
    let mut caller = vec![42.; tokens * owner.layout.rows()];
    report(
        "GEMMA_REAL_STEP_ATTEMPT",
        json!({"key":key,"name":name,"phase":phase,"index":index,"tokens":tokens,"input_sha256":hash(&raw_values(input))}),
    );
    let result = owner.apply(input, &mut caller, tokens);
    let observation = completed_device(owner, result.is_ok(), caller.len());
    let outcome = json!({"key":key,"name":name,"phase":phase,"index":index,"tokens":tokens,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),
        "poisoned":owner.is_poisoned(),"pending":owner.pending.is_some(),"owned_bytes":owner.owned_bytes(),"caller_sha256":hash(&raw_values(&caller)),"device":observation.as_ref().map(|v|&v.0)});
    let output = publish_completed(
        &mut std::io::stdout(),
        &outcome,
        capture,
        &key,
        observation.as_ref().map(|v| v.1.as_slice()),
        tokens,
        owner.layout.rows(),
    );
    report(
        "GEMMA_REAL_STEP",
        json!({"key":key,"name":name,"phase":phase,"index":index,"tokens":tokens,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),
        "poisoned":owner.is_poisoned(),"pending":owner.pending.is_some(),"owned_bytes":owner.owned_bytes(),"caller_sha256":hash(&raw_values(&caller)),"device":observation.as_ref().map(|v|&v.0),"output":output}),
    );
    result.unwrap();
    assert!(!owner.is_poisoned() && owner.pending.is_none());
    let (_, raw) = observation.unwrap();
    assert_eq!(raw, raw_values(&caller));
    caller
}

#[test]
#[ignore = "explicit bounded real Gemma checkpoint projection qualification"]
fn real_checkpoint_projections() {
    autoreleasepool(|_| {
        let model_path =
            std::env::var_os("TOKLET_GEMMA_PROJECTION_MODEL").expect("explicit model path");
        let frozen_path = std::env::var_os("TOKLET_GEMMA_PROJECTION_REFERENCE")
            .expect("explicit frozen reference path");
        let output_path = std::env::var_os("TOKLET_GEMMA_PROJECTION_OUTPUT")
            .expect("exclusive result binary path");
        let frozen = Path::new(&frozen_path);
        let metadata = bounded(
            &frozen.join("observation.json"),
            2 * 1024 * 1024,
            "6ac7c122fb89c3afadb3bef39b4a15118100cff990130851129b3771beff87e4",
        );
        bounded(
            &frozen.join("manifest.json"),
            65536,
            "89827c6b24c6e626bc96d95e63cae787a8b5fecd4754880d8176d191cdd2113c",
        );
        assert_eq!(
            file_hash(&frozen.join("arrays.bin"), 32 * 1024 * 1024),
            "037ca2048d2ee9f1fea58bd3ea1ba6008543dc8a13e7a064e2d755817c7f021e"
        );
        let reference: Value = serde_json::from_slice(&metadata).unwrap();
        assert_eq!(reference["passed"], true);
        assert_eq!(reference["records"].as_array().unwrap().len(), 277);
        let mut reference_data = File::open(frozen.join("arrays.bin")).unwrap();
        assert_eq!(reference_data.metadata().unwrap().len(), 26333184);
        let mut model = File::open(model_path).unwrap();
        assert_eq!(model.metadata().unwrap().len(), MODEL_BYTES);
        let mut prefix = [0; 8];
        model.read_exact(&mut prefix).unwrap();
        let length = u64::from_le_bytes(prefix);
        assert_eq!(length, 318220);
        let mut header = vec![0; usize::try_from(length).unwrap()];
        model.read_exact(&mut header).unwrap();
        assert_eq!(
            hash(&header),
            "4d1819ededded19812059691c0f77cfec0fe10c9c89c720950f919bfc26544a7"
        );
        let header: Value = serde_json::from_slice(&header).unwrap();
        let data_start = 8 + length;
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let mut capture = Capture {
            file: File::options()
                .write(true)
                .create_new(true)
                .open(output_path)
                .unwrap(),
            bytes: 0,
        };
        report(
            "GEMMA_REAL_META",
            json!({"device":context.device_name(),"simd_width":context.width,"owner_struct_bytes":size_of::<Affine<'_>>(),"context_allowance_bytes":CONTEXT_ALLOWANCE_BYTES,
            "binary_cap_bytes":RAW_LIMIT,"reference_contract":reference["expected_contract"],"whole_model_resident":false,"native_activation_parity":false}),
        );
        let mut cases = 0;
        let mut evidence = [0; 2];
        for case in reference["records"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let rows = case["rows"].as_u64().unwrap() as usize;
            let columns = case["columns"].as_u64().unwrap() as usize;
            let layout = Layout::new(rows, columns, 4, 64, MetadataDtype::BF16).unwrap();
            if case["status"] == "geometry-rejected" {
                assert_eq!(name, "embed_tokens.weight");
                assert_eq!(rows, 262144);
                let result = context.affine(layout, &[], &[], &[], 4, 1024 * 1024 * 1024);
                report(
                    "GEMMA_REAL_HEAD_REJECTION",
                    json!({"name":name,"rejected":result.is_err(),"error":result.as_ref().err().map(ToString::to_string),"large_spans_read":false}),
                );
                assert!(matches!(
                    result,
                    Err(Error::Options(
                        "Gemma affine fixed geometry/format outside bounds"
                    ))
                ));
                continue;
            }
            report(
                "GEMMA_REAL_CASE_ATTEMPT",
                json!({"name":name,"rows":rows,"columns":columns,"evidence":case["input_evidence"]}),
            );
            let physical = format!("language_model.model.{name}");
            let base = physical.strip_suffix(".weight").unwrap();
            let (codes, code_record) = tensor(
                &mut model,
                &header,
                data_start,
                &physical,
                json!([rows, columns / 8]),
                "U32",
                layout.weight_bytes(),
            );
            let (scales, scale_record) = tensor(
                &mut model,
                &header,
                data_start,
                &format!("{base}.scales"),
                json!([rows, columns / 64]),
                "BF16",
                layout.metadata_bytes(),
            );
            let (biases, bias_record) = tensor(
                &mut model,
                &header,
                data_start,
                &format!("{base}.biases"),
                json!([rows, columns / 64]),
                "BF16",
                layout.metadata_bytes(),
            );
            let input = array(&mut reference_data, &case["arrays"]["input"], columns);
            let expected = array(
                &mut reference_data,
                &case["arrays"]["expected_staged_fp32"],
                rows,
            );
            let sizes = Sizes::new(layout, 4).unwrap();
            let mut owner = context
                .affine(layout, &codes, &scales, &biases, 4, sizes.owned)
                .unwrap();
            report(
                "GEMMA_REAL_CASE",
                json!({"name":name,"rows":rows,"columns":columns,"batch":4,"evidence":case["input_evidence"],"reference_admission":case["status"],"owned_bytes":owner.owned_bytes(),
                "loaded_spans":[code_record,scale_record,bias_record],"host_tensor_bytes":codes.capacity()+scales.capacity()+biases.capacity()}),
            );
            let matrix = Matrix::from_le_bytes(layout, &codes, &scales, &biases).unwrap();
            let mut ordered = vec![42.; 4 * rows];
            report(
                "GEMMA_REAL_ORDERED_ATTEMPT",
                json!({"name":name,"contract":"native ordered FP64 dot, output rounded FP32"}),
            );
            let mut ordered_parts = Vec::with_capacity(4);
            for (token, (row, out)) in input
                .chunks_exact(columns)
                .zip(ordered.chunks_exact_mut(rows))
                .enumerate()
            {
                report(
                    "GEMMA_REAL_ORDERED_ROW_ATTEMPT",
                    json!({"name":name,"token":token}),
                );
                let result = matrix.matvec(row, out);
                let item = result.is_ok().then(|| {
                    capture.append(
                        &format!("{name}/ordered/{token}"),
                        "native-ordered-fp32",
                        &raw_values(out),
                        1,
                        rows,
                    )
                });
                report(
                    "GEMMA_REAL_ORDERED_ROW",
                    json!({"name":name,"token":token,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"array":item}),
                );
                result.unwrap();
                ordered_parts.push(item.unwrap());
            }
            report(
                "GEMMA_REAL_ORDERED",
                json!({"name":name,"arrays":ordered_parts,"contract":"native ordered FP64 dot, output rounded FP32"}),
            );
            let mut baseline = Vec::new();
            for repeat in 0..2 {
                let actual = step(&mut owner, &mut capture, name, "repeat", repeat, &input, 4);
                for (contract, reference) in [
                    ("staged-rounded-math", &expected),
                    ("native-ordered-fp32", &ordered),
                ] {
                    for (&actual, &expected) in actual.iter().zip(reference.iter()) {
                        assert!(
                            actual.is_finite()
                                && (f64::from(actual) - f64::from(expected)).abs()
                                    <= 1e-5 + 1e-4 * f64::from(expected).abs(),
                            "{name} {contract}: {actual} vs {expected}"
                        );
                    }
                }
                if repeat == 0 {
                    baseline = bits(&actual);
                } else {
                    assert_eq!(bits(&actual), baseline);
                }
            }
            for token in 0..4 {
                let actual = step(
                    &mut owner,
                    &mut capture,
                    name,
                    "part",
                    token,
                    &input[token * columns..(token + 1) * columns],
                    1,
                );
                assert_eq!(bits(&actual), baseline[token * rows..(token + 1) * rows]);
            }
            let resumed = step(&mut owner, &mut capture, name, "resume", 0, &input, 4);
            assert_eq!(bits(&resumed), baseline);
            assert_eq!(owner.owned_bytes(), sizes.owned);
            evidence[usize::from(case["input_evidence"] == "derived-mathematical-gelu-product")] +=
                1;
            cases += 1;
        }
        report(
            "GEMMA_REAL_FINAL",
            json!({"cases":cases,"captured_state_inputs":evidence[0],"derived_recipe_inputs":evidence[1],"binary_bytes":capture.bytes,"head_geometry_rejected":true,"whole_graph_qualified":false}),
        );
        assert_eq!((cases, evidence, capture.bytes), (276, [206, 70], 76021760));
    });
}
