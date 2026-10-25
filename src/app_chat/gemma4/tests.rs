use super::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn fixture() -> Vec<Value> {
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../../../tests/fixtures/gemma4-text-codec-v1/manifest.json"
            ))
        ),
        "4760a6d6cc8f4a63cc07f4da5f2f51c6fa3d1a3bd179b8a56d49481bb25f56ac"
    );
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../../../tests/fixtures/gemma4-text-codec-v1/witness.jsonl"
            ))
        ),
        "391077f928d1e288c227bc981f42ae505f7b407f9ee854332f9d2d9c3a410c76"
    );
    include_str!("../../../tests/fixtures/gemma4-text-codec-v1/witness.jsonl")
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}
fn basic() -> RenderOptions {
    RenderOptions {
        enable_thinking: false,
        add_generation_prompt: true,
    }
}

#[test]
fn frozen_basic_prompts_and_unsupported_requests_match_reference() {
    let mut prompts = 0;
    let mut rejected = 0;
    for row in fixture() {
        if row["kind"] == "chat" || row["kind"] == "rejection" {
            let options = serde_json::from_value::<RenderOptions>(row["options"].clone());
            let result = options
                .map_err(invalid)
                .and_then(|v| render(&serde_json::to_vec(&row["messages"]).unwrap(), v));
            if row["kind"] == "chat" {
                assert_eq!(result.unwrap(), row["rendered"].as_str().unwrap());
                prompts += 1;
            } else {
                assert!(result.is_err(), "{}", row["name"]);
                rejected += 1;
            }
        }
    }
    assert_eq!((prompts, rejected), (6, 33));
}

#[test]
fn duplicate_unknown_and_escaped_controls_reject_at_typed_boundary() {
    for options in [
        r#"{"enable_thinking":false,"enable_thinking":true,"add_generation_prompt":true}"#,
        r#"{"enable_thinking":false,"add_generation_prompt":true,"tools":[]}"#,
        r#"{"enable_thinking":0,"add_generation_prompt":true}"#,
        r#"{"enable_thinking":false}"#,
    ] {
        assert!(serde_json::from_str::<RenderOptions>(options).is_err());
    }
    for messages in [
        r#"[{"role":"user","role":"assistant","content":"Hi"}]"#,
        r#"[{"role":"user","content":"Hi","content":"Bye"}]"#,
        r#"[{"role":"user","content":"Hi","reasoning":"secret"}]"#,
        r#"[{"role":"user","content":"\u003cbos\u003e"}]"#,
        r#"[{"role":"user","content":"\u001cHi"}]"#,
    ] {
        assert!(render(messages.as_bytes(), basic()).is_err());
    }
    // DEL is outside the C0 rejection set; it is not Python's C0 trim mismatch.
    assert!(
        render(
            &serde_json::to_vec(&json!([{"role":"user","content":"a\u{7f}b"}])).unwrap(),
            basic()
        )
        .is_ok()
    );
}

#[test]
fn utf8_message_count_and_canonical_byte_admission_precede_output() {
    let messages: Vec<_> = (0..63)
        .map(|i| json!({"role":if i%2==0 {"user"}else{"assistant"},"content":"\u{2003}é\u{3000}"}))
        .collect();
    let bytes = serde_json::to_vec(&messages).unwrap();
    let text = render(&bytes, basic()).unwrap();
    assert_eq!(text.matches("é<turn|>\n").count(), 63);
    assert!(text.ends_with("<|turn>model\n"));
    for count in [0, 2, 64, 65] {
        let values: Vec<_> = (0..count)
            .map(|i| json!({"role":if i%2==0 {"user"}else{"assistant"},"content":"x"}))
            .collect();
        assert!(render(&serde_json::to_vec(&values).unwrap(), basic()).is_err());
    }
    let too_large =
        serde_json::to_vec(&json!([{"role":"user","content":"é".repeat(MAX_TEXT_BYTES/2)}]))
            .unwrap();
    assert!(render(&too_large, basic()).is_err());
    // Compact wire JSON fits; the reference's spaced canonical form does not.
    let almost =
        serde_json::to_vec(&json!([{"role":"user","content":"x".repeat(MAX_TEXT_BYTES-31)}]))
            .unwrap();
    assert!(almost.len() <= MAX_TEXT_BYTES);
    assert!(render(&almost, basic()).is_err());
}

#[test]
fn codec_allowance_rejects_before_opening_assets() {
    let missing = Path::new("/nonexistent-toklet-gemma-codec-admission-test");
    let error = Codec::load(missing, ALLOWANCE_BYTES - 1).err().unwrap();
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert!(error.to_string().contains("allowance"));
    assert_eq!(
        Codec::load(missing, ALLOWANCE_BYTES).err().unwrap().kind(),
        io::ErrorKind::NotFound
    );
}

#[test]
fn decode_preflight_bounds_ids_bytes_and_skipped_control_tokens() {
    assert_eq!(decode_bound(&[5, 5, 5, 5], &[0, 1, 2, 3], true).unwrap(), 0);
    assert_eq!(
        decode_bound(&[5, 5, 5, 5], &[0, 1, 2, 3], false).unwrap(),
        20
    );
    assert!(decode_bound(&[5; 4], &[4], true).is_err());
    assert!(decode_bound(&[4096], &[0; 17], false).is_err());
    assert_eq!(
        decode_bound(&[4096, 1], &[0; 16], false).unwrap(),
        MAX_TEXT_BYTES
    );
    let mut over = vec![0; 16];
    over.push(1);
    assert!(decode_bound(&[4096, 1], &over, false).is_err());
    let mut skipped = vec![0; MAX_CODEC_TOKENS];
    assert_eq!(decode_bound(&[5], &skipped, true).unwrap(), 0);
    skipped.push(0);
    assert!(decode_bound(&[5], &skipped, true).is_err());
}

fn stream_fixture() -> (Codec, Value) {
    let manifest = include_bytes!("../../../tests/fixtures/gemma4-stream-v1/manifest.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(manifest)),
        "1605844dcb15d3c2a69d3e2df02099750153cef6ec3d754a93ab563646f03b54"
    );
    let manifest: Value = serde_json::from_slice(manifest).unwrap();
    let tokenizer = include_bytes!("../../../tests/fixtures/gemma4-stream-v1/tokenizer.json");
    let reference = include_bytes!("../../../tests/fixtures/gemma4-stream-v1/reference.json");
    for (name, raw) in [
        ("tokenizer.json", tokenizer.as_slice()),
        ("reference.json", reference.as_slice()),
    ] {
        assert_eq!(
            format!("{:x}", Sha256::digest(raw)),
            manifest["files"][name]["sha256"]
        );
        assert_eq!(
            raw.len() as u64,
            manifest["files"][name]["bytes"].as_u64().unwrap()
        );
    }
    let tokenizer = Tokenizer::from_bytes(tokenizer).unwrap();
    let mut token_bytes = vec![0; VOCAB];
    for (token, id) in tokenizer.get_vocab(true) {
        token_bytes[id as usize] = token.len() as u16;
    }
    (
        Codec {
            tokenizer,
            token_bytes,
        },
        serde_json::from_slice(reference).unwrap(),
    )
}
fn stream_ids(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|id| u32::try_from(id.as_u64().unwrap()).unwrap())
        .collect()
}

#[test]
fn stream_matches_frozen_prefixes_without_strip_for_all_partitions() {
    let (codec, fixture) = stream_fixture();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 67);
    for case in fixture["cases"].as_array().unwrap() {
        let ids = stream_ids(&case["ids"]);
        let expected = case["expected"].as_str().unwrap();
        for (count, prefix) in case["prefixes"].as_array().unwrap().iter().enumerate() {
            assert_eq!(
                codec.tokenizer.decode(&ids[..count], true).unwrap(),
                prefix.as_str().unwrap(),
                "{} prefix{count}",
                case["name"]
            );
        }
        for batch in [1, 3, 7, 128] {
            let mut stream = codec
                .response_stream(
                    ResponseLimits::new(ids.len().max(1), MAX_TEXT_BYTES).unwrap(),
                    STREAM_ALLOWANCE_BYTES,
                )
                .unwrap();
            let allocated = stream.allocated_bytes().unwrap();
            assert!(allocated <= STREAM_ALLOWANCE_BYTES);
            let mut visible = String::new();
            for chunk in ids.chunks(batch) {
                visible.push_str(stream.push(chunk).unwrap());
                let prefix = case["prefixes"][stream.consumed_tokens()].as_str().unwrap();
                assert!(
                    prefix.starts_with(&visible),
                    "{} current prefix",
                    case["name"]
                );
                assert!(
                    expected.starts_with(&visible),
                    "{} later rewrite",
                    case["name"]
                );
                assert_eq!(stream.allocated_bytes().unwrap(), allocated);
            }
            visible.push_str(stream.finish().unwrap());
            assert_eq!(visible, expected, "{} batch{batch}", case["name"]);
            assert_eq!(stream.consumed_tokens(), ids.len());
            assert_eq!(stream.emitted_bytes(), visible.len());
            assert!(stream.is_finished() && !stream.is_failed());
            assert_eq!(stream.allocated_bytes().unwrap(), allocated);
        }
    }
}

#[test]
fn long_text_stream_hashes_preserve_whole_run_delay_and_bounded_storage() {
    let (codec, fixture) = stream_fixture();
    assert_eq!(fixture["long_cases"].as_array().unwrap().len(), 5);
    for case in fixture["long_cases"].as_array().unwrap() {
        let pattern = stream_ids(&case["pattern_ids"]);
        let suffix = stream_ids(&case["suffix_ids"]);
        let count = case["tokens"].as_u64().unwrap() as usize;
        assert_eq!(count, 131072);
        for batch in [1, 3, 128] {
            let mut stream = codec
                .response_stream(
                    ResponseLimits::new(count, 8 * 1024 * 1024).unwrap(),
                    STREAM_ALLOWANCE_BYTES,
                )
                .unwrap();
            let allocated = stream.allocated_bytes().unwrap();
            let mut hash = Sha256::new();
            let mut bytes = 0;
            let mut ids = Vec::with_capacity(128);
            for start in (0..count).step_by(batch) {
                ids.clear();
                for position in start..(start + batch).min(count) {
                    ids.push(if position < count - suffix.len() {
                        pattern[position % pattern.len()]
                    } else {
                        suffix[position - (count - suffix.len())]
                    });
                }
                let chunk = stream.push(&ids).unwrap();
                hash.update(chunk.as_bytes());
                bytes += chunk.len();
                if case["name"].as_str().unwrap().contains("fallback") {
                    assert_eq!(bytes, 0, "whole fallback run must wait for finish");
                }
                assert_eq!(stream.allocated_bytes().unwrap(), allocated);
            }
            let tail = stream.finish().unwrap();
            hash.update(tail.as_bytes());
            bytes += tail.len();
            assert_eq!(
                format!("{:x}", hash.finalize()),
                case["expected_sha256"].as_str().unwrap()
            );
            assert_eq!(bytes as u64, case["expected_utf8_bytes"].as_u64().unwrap());
            assert_eq!(stream.consumed_tokens(), count);
            assert_eq!(stream.emitted_bytes(), bytes);
            assert_eq!(stream.allocated_bytes().unwrap(), allocated);
        }
    }
}

#[test]
fn stream_admission_terminal_flush_and_poisoned_counters_remain_bounded() {
    let (codec, _) = stream_fixture();
    let limits = ResponseLimits::new(4, 32).unwrap();
    assert!(
        codec
            .response_stream(limits, STREAM_ALLOWANCE_BYTES - 1)
            .err()
            .unwrap()
            .to_string()
            .contains("allowance")
    );
    let x = codec.tokenizer.token_to_id("x").unwrap();
    for ids in [vec![VOCAB as u32], vec![x; 129], vec![x; 5]] {
        let mut stream = codec
            .response_stream(limits, STREAM_ALLOWANCE_BYTES)
            .unwrap();
        assert!(stream.push(&ids).is_err());
        assert_eq!((stream.consumed_tokens(), stream.emitted_bytes()), (0, 0));
        assert!(stream.is_failed());
        assert!(stream.finish().is_err());
    }
    let byte = |value: &str| codec.tokenizer.token_to_id(value).unwrap();
    let mut stream = codec
        .response_stream(ResponseLimits::new(3, 1).unwrap(), STREAM_ALLOWANCE_BYTES)
        .unwrap();
    assert_eq!(stream.push(&[x]).unwrap(), "x");
    assert!(stream.push(&[byte("�")]).is_err());
    assert_eq!((stream.consumed_tokens(), stream.emitted_bytes()), (1, 1));
    assert!(stream.is_failed());
    assert!(stream.finish().is_err());
    let mut cancelled = codec
        .response_stream(limits, STREAM_ALLOWANCE_BYTES)
        .unwrap();
    assert_eq!(
        cancelled.push(&[byte("<0xE2>"), byte("<0x82>")]).unwrap(),
        ""
    );
    // Stop/cancellation flushes the admitted visible prefix; no hidden stop ID
    // is delivered to this text owner.
    assert_eq!(cancelled.finish().unwrap(), "��");
    assert_eq!(
        (cancelled.consumed_tokens(), cancelled.emitted_bytes()),
        (2, 6)
    );
    assert!(cancelled.push(&[]).is_err());
    let mut terminal = codec
        .response_stream(ResponseLimits::new(1, 2).unwrap(), STREAM_ALLOWANCE_BYTES)
        .unwrap();
    assert_eq!(terminal.push(&[byte("<0xFF>")]).unwrap(), "");
    assert!(terminal.finish().is_err());
    assert_eq!(
        (terminal.consumed_tokens(), terminal.emitted_bytes()),
        (1, 0)
    );
    assert!(terminal.is_failed());
}
