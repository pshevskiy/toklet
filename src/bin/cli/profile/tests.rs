use super::{Witness, Workload, load_workload, parse};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    ffi::OsString,
    fs::{self, OpenOptions},
    io::{self, Write},
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};
use toklet::session::{Error, InferenceSession, Memory};

fn arguments() -> Vec<OsString> {
    [
        "--model",
        "model directory",
        "--workload",
        "workload.json",
        "--evaluation",
        "3",
        "--output",
        "fresh directory",
    ]
    .into_iter()
    .map(OsString::from)
    .collect()
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn descriptor(input: usize, output: usize) -> Value {
    let conversation = r#"[{"role":"user","content":"Public café fixture."}]"#;
    json!({
        "schema_version": 1,
        "workload_id": "portable_fixture",
        "conversation_json": conversation,
        "conversation_sha256": digest(conversation.as_bytes()),
        "input_token_count": input,
        "input_tokens_sha256": digest(&[1, 0, 0, 0]),
        "maximum_output_tokens": output,
        "capacity": 512,
        "chunk": 32,
        "cache_policy": "reset_chat_each_request",
        "expected_output": {
            "tokens": vec![7; output],
            "history": input + output,
            "committed": input + output - 1,
            "pending": 1,
            "reason": 1,
            "stop_token": u32::MAX,
        },
    })
}

// Each test owns an exclusively created file. No environment variables or
// process-wide working directories change, so tests may run concurrently.
struct WorkloadFile(PathBuf);

impl WorkloadFile {
    fn new(raw: &[u8]) -> io::Result<Self> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        for attempt in 0..100 {
            let path = std::env::temp_dir().join(format!(
                "toklet-profile-test-{}-{timestamp}-{attempt}.json",
                std::process::id()
            ));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    let owned = Self(path);
                    file.write_all(raw)?;
                    return Ok(owned);
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not create unique workload fixture",
        ))
    }
}

impl Drop for WorkloadFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn load_raw(raw: &[u8]) -> io::Result<Workload> {
    let file = WorkloadFile::new(raw)?;
    load_workload(&file.0)
}

fn load(value: &Value) -> io::Result<Workload> {
    load_raw(&serde_json::to_vec(value).unwrap())
}

#[test]
fn arguments_require_the_four_explicit_options_and_preserve_paths() {
    let parsed = parse(arguments()).unwrap();
    assert_eq!(parsed.model, PathBuf::from("model directory"));
    assert_eq!(parsed.workload, PathBuf::from("workload.json"));
    assert_eq!(parsed.evaluation, 3);
    assert_eq!(parsed.output, PathBuf::from("fresh directory"));

    let mut reordered = arguments();
    reordered.rotate_left(4);
    assert_eq!(parse(reordered).unwrap().evaluation, 3);
    for ordinal in ["1", "13", "139", "512"] {
        let mut values = arguments();
        values[5] = ordinal.into();
        assert_eq!(parse(values).unwrap().evaluation.to_string(), ordinal);
    }
}

#[test]
fn arguments_reject_missing_duplicate_unknown_and_parse_level_help() {
    for index in [0, 2, 4, 6] {
        let mut values = arguments();
        values.drain(index..index + 2);
        assert!(parse(values).is_err(), "missing option at {index}");

        let mut values = arguments();
        values.extend_from_within(index..index + 2);
        assert!(parse(values).is_err(), "duplicate option at {index}");
    }
    for flag in ["--help", "-h", "--unknown", "profile", "--model=x"] {
        assert!(parse(vec![flag.into()]).is_err(), "flag {flag}");
        let mut values = arguments();
        values.push(flag.into());
        assert!(parse(values).is_err(), "extra flag {flag}");
    }
    assert!(parse(Vec::new()).is_err());
    let mut missing_value = arguments();
    missing_value.pop();
    assert!(parse(missing_value).is_err());
}

#[test]
fn evaluation_ordinal_is_bounded_unsigned_ascii_decimal() {
    for ordinal in [
        "",
        "0",
        "513",
        "+1",
        "-1",
        "1.0",
        "1e1",
        " 1",
        "1 ",
        "١",
        "１",
        "184467440737095516160000000000000000000",
    ] {
        let mut values = arguments();
        values[5] = ordinal.into();
        assert!(parse(values).is_err(), "ordinal {ordinal:?}");
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        for index in [0, 5] {
            let mut values = arguments();
            values[index] = OsString::from_vec(vec![0xff]);
            assert!(parse(values).is_err(), "non-UTF-8 argument at {index}");
        }
    }
}

#[test]
fn full_wire_document_and_exact_raw_file_hash_are_retained() {
    let value = descriptor(33, 2);
    assert_eq!(value.as_object().unwrap().len(), 11);
    let mut raw = serde_json::to_vec(&value).unwrap();
    raw.extend_from_slice(b"\n \t");
    let loaded = load_raw(&raw).unwrap();
    assert_eq!(loaded.sha256, digest(&raw));
    let doc = loaded.document;
    assert_eq!(doc.schema_version, 1);
    assert_eq!(doc.workload_id, "portable_fixture");
    assert_eq!(
        doc.conversation_json,
        value["conversation_json"].as_str().unwrap()
    );
    assert_eq!(
        doc.conversation_sha256,
        value["conversation_sha256"].as_str().unwrap()
    );
    assert_eq!(
        doc.input_tokens_sha256,
        value["input_tokens_sha256"].as_str().unwrap()
    );
    assert_eq!(doc.input_token_count, 33);
    assert_eq!(doc.maximum_output_tokens, 2);
    assert_eq!((doc.capacity, doc.chunk), (512, 32));
    assert_eq!(doc.cache_policy, "reset_chat_each_request");
    assert_eq!(doc.expected_output.tokens, [7, 7]);
    assert_eq!(doc.expected_output.history, 35);
    assert_eq!(doc.expected_output.committed, 34);
    assert_eq!(doc.expected_output.pending, 1);
    assert_eq!(doc.expected_output.reason, 1);
    assert_eq!(doc.expected_output.stop_token, u32::MAX);
}

#[test]
fn raw_16k_and_utf8_conversation_8k_limits_are_inclusive() {
    let mut value = descriptor(1, 1);
    let prefix = r#"[{"role":"user","content":""#;
    let suffix = r#""}]"#;
    let content_bytes = 8192 - prefix.len() - suffix.len();
    let content = format!(
        "{}{}",
        "é".repeat(content_bytes / 2),
        "a".repeat(content_bytes % 2)
    );
    let conversation = format!("{prefix}{content}{suffix}");
    assert_eq!(conversation.len(), 8192);
    value["conversation_json"] = json!(conversation);
    value["conversation_sha256"] = json!(digest(conversation.as_bytes()));
    let mut raw = serde_json::to_vec(&value).unwrap();
    assert!(raw.len() < 16384);
    raw.resize(16384, b' ');
    assert_eq!(load_raw(&raw).unwrap().sha256, digest(&raw));
    raw.push(b' ');
    assert!(load_raw(&raw).is_err());

    let too_long = format!("{conversation} ");
    value["conversation_json"] = json!(too_long);
    value["conversation_sha256"] = json!(digest(too_long.as_bytes()));
    assert!(load(&value).is_err());
}

#[test]
fn wire_rejects_duplicate_unknown_missing_and_nonfinite_fields() {
    let value = descriptor(1, 1);
    let raw = serde_json::to_string(&value).unwrap();
    for bytes in [
        &b""[..],
        &b"\xff"[..],
        &b"[]"[..],
        &b"null"[..],
        &b"{"[..],
        &b"{} trailing"[..],
    ] {
        assert!(load_raw(bytes).is_err(), "wire {bytes:?}");
    }
    for insertion in [
        ",\"schema_version\":1}",
        ",\"extra\":0}",
        ",\"_sha256\":\"private\"}",
    ] {
        let duplicate = format!("{}{insertion}", &raw[..raw.len() - 1]);
        assert!(load_raw(duplicate.as_bytes()).is_err());
    }
    for nonfinite in ["NaN", "Infinity", "-Infinity", "1e9999"] {
        let invalid = raw.replace(
            "\"schema_version\":1",
            &format!("\"schema_version\":{nonfinite}"),
        );
        assert!(load_raw(invalid.as_bytes()).is_err(), "{nonfinite}");
    }
    for field in value.as_object().unwrap().keys() {
        let mut changed = value.clone();
        changed.as_object_mut().unwrap().remove(field);
        assert!(load(&changed).is_err(), "missing {field}");
    }
    let nested = raw.replace("\"pending\":1", "\"pending\":1,\"pending\":1");
    assert!(load_raw(nested.as_bytes()).is_err());
}

#[test]
fn workload_metadata_uses_exact_types_and_bounded_identifier() {
    let changes = [
        ("schema_version", json!(true)),
        ("schema_version", json!(1.0)),
        ("schema_version", json!(2)),
        ("workload_id", json!("")),
        ("workload_id", json!("_bad")),
        ("workload_id", json!("UPPER")),
        ("workload_id", json!("a/b")),
        ("workload_id", json!("é")),
        ("workload_id", json!("x".repeat(65))),
        ("workload_id", json!(true)),
        ("conversation_json", json!("")),
        ("conversation_json", json!([])),
        ("conversation_json", json!(true)),
    ];
    for (field, bad) in changes {
        let mut value = descriptor(1, 1);
        value[field] = bad;
        assert!(load(&value).is_err(), "{field} = {}", value[field]);
    }
    let mut value = descriptor(1, 1);
    value["workload_id"] = json!(format!("a{}", "_-9".repeat(21)));
    assert_eq!(value["workload_id"].as_str().unwrap().len(), 64);
    assert!(load(&value).is_ok());
}

#[test]
fn conversation_identity_is_exact_and_hashes_are_lower_hex_64() {
    for field in ["conversation_sha256", "input_tokens_sha256"] {
        for bad in [
            json!(""),
            json!("a".repeat(63)),
            json!("A".repeat(64)),
            json!("g".repeat(64)),
            json!(true),
        ] {
            let mut value = descriptor(1, 1);
            value[field] = bad;
            assert!(load(&value).is_err(), "invalid {field}");
        }
    }
    let mut value = descriptor(1, 1);
    let original = value["conversation_json"].as_str().unwrap().to_owned();
    value["conversation_json"] = json!(format!("{original} "));
    assert!(
        load(&value).is_err(),
        "trailing conversation byte changes identity"
    );
    value["conversation_sha256"] = json!(digest(
        value["conversation_json"].as_str().unwrap().as_bytes()
    ));
    assert!(load(&value).is_ok());

    let conversation = r#"[{"role":"user","content":"One"},{"role":"assistant","content":"Two"},{"role":"user","content":"Three"}]"#;
    value["conversation_json"] = json!(conversation);
    value["conversation_sha256"] = json!(digest(conversation.as_bytes()));
    assert!(load(&value).is_ok(), "ordinary alternating basic-text chat");

    // The route validates ordinary Gemma basic-text chat before loading a model.
    // Arrays admitted by a syntax-only JSON parser are insufficient here.
    let unsupported_arrays = [
        "[]".to_owned(),
        "[true,false,null,1,\"finite\",{\"metadata\":{\"value\":2}}]".to_owned(),
        format!("{}0{}", "[".repeat(64), "]".repeat(64)),
    ];
    for conversation in unsupported_arrays {
        let mut value = descriptor(1, 1);
        value["conversation_json"] = json!(conversation);
        value["conversation_sha256"] = json!(digest(conversation.as_bytes()));
        assert!(
            load(&value).is_err(),
            "unsupported chat array {conversation}"
        );
    }

    for conversation in [
        "{}",
        "null",
        "[NaN]",
        "[{\"role\":\"user\",\"role\":\"assistant\"}]",
        "[1e9999]",
    ] {
        let mut value = descriptor(1, 1);
        value["conversation_json"] = json!(conversation);
        value["conversation_sha256"] = json!(digest(conversation.as_bytes()));
        assert!(load(&value).is_err(), "conversation {conversation}");
    }
}

#[test]
fn admitted_geometry_is_general_and_counts_real_chunk_evaluations() {
    for (input, output, evaluations) in [
        (1, 1, 1),
        (31, 2, 2),
        (32, 1, 1),
        (33, 2, 3),
        (256, 256, 263),
        (384, 128, 139),
        (511, 1, 16),
    ] {
        let doc = load(&descriptor(input, output)).unwrap().document;
        assert_eq!(doc.sampled_tokens(), output);
        assert_eq!(
            doc.evaluations(),
            evaluations,
            "input {input}, output {output}"
        );
    }
}

#[test]
fn admission_rejects_invalid_counts_window_capacity_chunk_and_policy() {
    let changes = [
        ("input_token_count", json!(0)),
        ("input_token_count", json!(513)),
        ("input_token_count", json!(true)),
        ("input_token_count", json!(1.0)),
        ("input_token_count", json!("1")),
        ("input_token_count", json!(-1)),
        ("input_token_count", json!(u64::MAX)),
        ("maximum_output_tokens", json!(0)),
        ("maximum_output_tokens", json!(257)),
        ("maximum_output_tokens", json!(true)),
        ("maximum_output_tokens", json!(1.0)),
        ("capacity", json!(511)),
        ("capacity", json!(513)),
        ("capacity", json!(true)),
        ("capacity", json!(512.0)),
        ("chunk", json!(31)),
        ("chunk", json!(33)),
        ("chunk", json!(true)),
        ("chunk", json!(32.0)),
        ("cache_policy", json!("reuse")),
        ("cache_policy", Value::Null),
    ];
    for (field, bad) in changes {
        let mut value = descriptor(1, 1);
        value[field] = bad;
        assert!(load(&value).is_err(), "{field} = {}", value[field]);
    }
    assert!(load(&descriptor(512, 1)).is_err());
    assert!(load(&descriptor(257, 256)).is_err());
}

#[test]
fn expected_output_is_nonnull_exact_typed_positive_and_state_checked() {
    let changes = [
        ("tokens", json!([])),
        ("tokens", json!([true, 7])),
        ("tokens", json!([7.0, 7])),
        ("tokens", json!([-1, 7])),
        ("tokens", json!([4294967296_u64, 7])),
        ("tokens", json!([7, 7, 7])),
        ("history", json!(true)),
        ("history", json!(35.0)),
        ("history", json!(36)),
        ("committed", json!(33)),
        ("pending", json!(0)),
        ("pending", json!(true)),
        ("reason", json!(true)),
        ("reason", json!(3)),
        ("stop_token", json!(106)),
        ("stop_token", json!(4294967296_u64)),
    ];
    for (field, bad) in changes {
        let mut value = descriptor(33, 2);
        value["expected_output"][field] = bad;
        assert!(
            load(&value).is_err(),
            "expected {field} = {}",
            value["expected_output"][field]
        );
    }
    let original = descriptor(33, 2);
    for field in original["expected_output"].as_object().unwrap().keys() {
        let mut value = original.clone();
        value["expected_output"]
            .as_object_mut()
            .unwrap()
            .remove(field);
        assert!(load(&value).is_err(), "missing expected {field}");
    }
    let mut value = original;
    value["expected_output"]["extra"] = json!(0);
    assert!(load(&value).is_err());
    value["expected_output"] = Value::Null;
    assert!(load(&value).is_err());
}

#[test]
fn hidden_stop_adds_one_sample_and_requires_qualified_reason_and_state() {
    for stop in [1, 106, 50] {
        let mut value = descriptor(33, 4);
        value["expected_output"] = json!({
            "tokens": [7, 8], "history": 36, "committed": 35,
            "pending": 1, "reason": 2, "stop_token": stop,
        });
        let doc = load(&value).unwrap().document;
        assert_eq!(doc.sampled_tokens(), 3);
        assert_eq!(doc.evaluations(), 4);
        for (field, bad) in [
            ("stop_token", json!(2)),
            ("stop_token", json!(u32::MAX)),
            ("history", json!(35)),
            ("committed", json!(34)),
            ("tokens", json!([])),
            ("tokens", json!([7, 8, 9, 10])),
        ] {
            let mut changed = value.clone();
            changed["expected_output"][field] = bad;
            assert!(load(&changed).is_err(), "hidden stop {field}");
        }
    }
    let mut short_limit = descriptor(33, 4);
    short_limit["expected_output"]["tokens"] = json!([7, 8]);
    short_limit["expected_output"]["history"] = json!(35);
    short_limit["expected_output"]["committed"] = json!(34);
    assert!(load(&short_limit).is_err());
}

// This fixture exercises the coarse delegate contract, not a second sampler or
// numerical implementation. Each successful evaluation commits up to two rows.
struct FixtureSession {
    history: Vec<u32>,
    committed: usize,
    successful: usize,
    logits: Vec<f32>,
    visible: bool,
    fail_next: bool,
    fail_reset: bool,
    nonfinite: bool,
    calls: Vec<&'static str>,
}

impl FixtureSession {
    fn new() -> Self {
        Self {
            history: vec![1, 2, 3, 0, 1, 2],
            committed: 0,
            successful: 0,
            logits: vec![0.0; 4],
            visible: true,
            fail_next: false,
            fail_reset: false,
            nonfinite: false,
            calls: Vec::new(),
        }
    }

    fn row(step: usize) -> [f32; 4] {
        [f32::from_bits(1), -0.0, step as f32 + 0.25, -17.5]
    }
}

impl InferenceSession for FixtureSession {
    fn history(&self) -> &[u32] {
        &self.history
    }
    fn evaluated_len(&self) -> usize {
        self.committed
    }
    fn pending_len(&self) -> usize {
        self.history.len() - self.committed
    }
    fn capacity(&self) -> usize {
        64
    }
    fn vocab_size(&self) -> usize {
        4
    }
    fn memory(&self) -> Memory {
        Memory {
            weights_bytes: 1,
            positional_bytes: 2,
            kv_bytes: 3,
            scratch_bytes: 4,
            history_bytes: 5,
            logits_bytes: 6,
            metadata_bytes: 7,
            metadata_allowance_bytes: 8,
            admitted_bytes: 36,
        }
    }
    fn logits(&self) -> Option<&[f32]> {
        self.visible.then_some(self.logits.as_slice())
    }
    fn greedy_token(&self) -> Option<Result<u32, toklet::cpu::Error>> {
        // Deliberately differs from row argmax: the wrapper must forward it.
        Some(Ok(3))
    }
    fn append(&mut self, tokens: &[u32]) -> Result<(), Error> {
        self.calls.push("append");
        self.history.extend_from_slice(tokens);
        Ok(())
    }
    fn set_tokens(&mut self, tokens: &[u32]) -> Result<(), Error> {
        self.calls.push("set_tokens");
        self.history.clear();
        self.history.extend_from_slice(tokens);
        self.committed = 0;
        Ok(())
    }
    fn evaluate_next_chunk(&mut self) -> Result<usize, Error> {
        self.calls.push("evaluate");
        if self.fail_next {
            self.fail_next = false;
            return Err(Error::Backend("fixture evaluation failure".into()));
        }
        let rows = self.pending_len().min(2);
        self.committed += rows;
        self.successful += 1;
        if self.logits.len() == 4 {
            self.logits.copy_from_slice(&Self::row(self.successful));
            if self.nonfinite {
                self.logits[0] = f32::NAN;
            }
        }
        Ok(rows)
    }
    fn discard_pending(&mut self) {
        self.calls.push("discard_pending");
        self.history.truncate(self.committed);
    }
    fn reset(&mut self) {
        self.calls.push("reset");
        self.history.clear();
        self.committed = 0;
    }
    fn try_reset(&mut self) -> Result<(), Error> {
        self.calls.push("try_reset");
        if self.fail_reset {
            return Err(Error::Backend("fixture reset failure".into()));
        }
        self.history.clear();
        self.committed = 0;
        Ok(())
    }
}

#[test]
fn witness_captures_the_full_bit_pattern_once_at_exact_successful_ordinal() {
    let mut inner = FixtureSession::new();
    let mut selected = [91.0; 4];
    let mut witness = Witness::new(&mut inner, 2, &mut selected).unwrap();
    assert_eq!(witness.evaluate_next_chunk().unwrap(), 2);
    assert_eq!(witness.evaluations, 1);
    assert!(!witness.captured);
    assert_eq!(&*witness.selected, &[91.0; 4]);
    assert_eq!(witness.evaluate_next_chunk().unwrap(), 2);
    assert!(witness.captured);
    let expected = FixtureSession::row(2).map(f32::to_bits);
    assert_eq!(bits(witness.selected), expected);
    assert_eq!(witness.evaluate_next_chunk().unwrap(), 2);
    assert_eq!(witness.evaluations, 3);
    assert_eq!(bits(witness.selected), expected);
}

#[test]
fn witness_propagates_failed_evaluation_without_counting_or_capturing() {
    let mut inner = FixtureSession::new();
    inner.fail_next = true;
    let mut selected = [91.0; 4];
    let mut witness = Witness::new(&mut inner, 1, &mut selected).unwrap();
    assert_eq!(
        witness.evaluate_next_chunk().unwrap_err(),
        Error::Backend("fixture evaluation failure".into())
    );
    assert_eq!(witness.evaluations, 0);
    assert_eq!(witness.evaluated_len(), 0);
    assert!(!witness.captured);
    assert_eq!(&*witness.selected, &[91.0; 4]);
    assert_eq!(witness.evaluate_next_chunk().unwrap(), 2);
    assert_eq!(witness.evaluations, 1);
    assert!(witness.captured);
    assert_eq!(
        bits(witness.selected),
        FixtureSession::row(1).map(f32::to_bits)
    );
}

#[test]
fn witness_forwards_state_mutation_greedy_and_fallible_reset() {
    let mut inner = FixtureSession::new();
    let expected_memory = inner.memory();
    inner.fail_reset = true;
    let mut selected = [0.0; 4];
    let mut witness = Witness::new(&mut inner, 2, &mut selected).unwrap();
    assert_eq!(witness.history(), [1, 2, 3, 0, 1, 2]);
    assert_eq!((witness.evaluated_len(), witness.pending_len()), (0, 6));
    assert_eq!((witness.capacity(), witness.vocab_size()), (64, 4));
    assert_eq!(witness.memory(), expected_memory);
    assert_eq!(witness.logits().unwrap(), [0.0; 4]);
    assert_eq!(witness.greedy_token().unwrap().unwrap(), 3);
    assert_eq!(
        witness.try_reset().unwrap_err(),
        Error::Backend("fixture reset failure".into())
    );
    assert_eq!(witness.history().len(), 6);
    witness.inner.fail_reset = false;
    witness.try_reset().unwrap();
    assert!(witness.history().is_empty());
    witness.append(&[2, 1, 0]).unwrap();
    witness.evaluate_next_chunk().unwrap();
    assert_eq!((witness.evaluated_len(), witness.pending_len()), (2, 1));
    assert_eq!(witness.logits().unwrap(), FixtureSession::row(1));
    witness.discard_pending();
    assert_eq!(witness.history(), [2, 1]);
    witness.set_tokens(&[3, 2]).unwrap();
    assert_eq!((witness.evaluated_len(), witness.pending_len()), (0, 2));
    witness.reset();
    assert!(witness.history().is_empty());
    assert_eq!(
        witness.inner.calls,
        [
            "try_reset",
            "try_reset",
            "append",
            "evaluate",
            "discard_pending",
            "set_tokens",
            "reset",
        ]
    );
}

#[test]
fn witness_rejects_invalid_capture_buffers_and_unavailable_or_bad_rows() {
    for ordinal in [0, 513] {
        let mut inner = FixtureSession::new();
        let mut selected = [91.0; 4];
        assert!(Witness::new(&mut inner, ordinal, &mut selected).is_err());
        assert!(inner.calls.is_empty());
        assert_eq!(selected, [91.0; 4]);
    }
    for length in [0, 3, 5] {
        let mut inner = FixtureSession::new();
        let mut selected = vec![91.0; length];
        assert!(Witness::new(&mut inner, 1, &mut selected).is_err());
        assert!(inner.calls.is_empty());
        assert_eq!(selected, vec![91.0; length]);
    }
    for failure in ["missing", "short", "nonfinite"] {
        let mut inner = FixtureSession::new();
        match failure {
            "missing" => inner.visible = false,
            "short" => inner.logits.truncate(3),
            "nonfinite" => inner.nonfinite = true,
            _ => unreachable!(),
        }
        let mut selected = [91.0; 4];
        let mut witness = Witness::new(&mut inner, 1, &mut selected).unwrap();
        assert!(
            matches!(witness.evaluate_next_chunk(), Err(Error::Backend(_))),
            "{failure}"
        );
        assert!(!witness.captured, "{failure}");
        assert_eq!(&*witness.selected, &[91.0; 4], "{failure}");
    }
}
