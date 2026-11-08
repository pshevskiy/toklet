use super::*;
use serde_json::{Value, json};
use std::time::Instant;

const FIXTURE_SHA: &str = "ffc3ca64bf65f337d58bfef0f565d6f5c993eb480d349ebd400bdfb2ce8011db";
fn fixture() -> Vec<Value> {
    let manifest = include_bytes!("../../../tests/fixtures/qwen35-text-codec-v1/manifest.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(manifest)),
        "48bd6ece40eb6617a9c3c5d1d817513ab0feffd6e6508817285cd75a964e441c"
    );
    let raw = include_bytes!("../../../tests/fixtures/qwen35-text-codec-v1/witness.jsonl");
    assert_eq!(format!("{:x}", Sha256::digest(raw)), FIXTURE_SHA);
    raw.split(|&b| b == b'\n')
        .filter(|s| !s.is_empty())
        .map(|s| serde_json::from_slice(s).unwrap())
        .collect()
}
fn options() -> RenderOptions {
    RenderOptions {
        enable_thinking: false,
        add_generation_prompt: true,
    }
}
fn ids(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

#[test]
fn frozen_original_whitespace_and_rejections() {
    let mut counts = (0, 0);
    for row in fixture() {
        if row["kind"] != "chat" && row["kind"] != "rejection" {
            continue;
        }
        let settings = serde_json::from_value::<RenderOptions>(row["options"].clone());
        let result = settings
            .map_err(invalid)
            .and_then(|o| render(&serde_json::to_vec(&row["messages"]).unwrap(), o));
        if row["kind"] == "chat" {
            assert_eq!(result.unwrap(), row["rendered"].as_str().unwrap());
            counts.0 += 1;
        } else {
            assert!(result.is_err(), "{}", row["name"]);
            counts.1 += 1;
        }
    }
    assert_eq!(counts, (6, 43));
    assert_eq!(STOP_IDS, [248044, 248046]);
    assert_eq!(VOCAB, 248077);
    assert_eq!(MODEL_PADDED_VOCAB, 248320);
}
#[test]
fn duplicate_unknown_typed_and_escaped_reserved_fields() {
    for raw in [
        r#"{"enable_thinking":false,"enable_thinking":true,"add_generation_prompt":true}"#,
        r#"{"enable_thinking":false,"add_generation_prompt":1}"#,
        r#"{"enable_thinking":false,"add_generation_prompt":true,"tools":[]}"#,
    ] {
        assert!(serde_json::from_str::<RenderOptions>(raw).is_err());
    }
    for raw in [
        r#"[{"role":"user","role":"assistant","content":"Hi"}]"#,
        r#"[{"role":"user","content":"Hi","content":"Bye"}]"#,
        r#"[{"role":"user","content":"Hi","reasoning_content":"secret"}]"#,
        r#"[{"role":"user","content":"\u003cthink\u003e"}]"#,
        r#"[{"role":"user","content":"\u001cHi"}]"#,
    ] {
        assert!(render(raw.as_bytes(), options()).is_err());
    }
}
#[test]
fn utf8_canonical_rendered_and_message_bounds() {
    let many: Vec<_> = (0..63)
        .map(|i| json!({"role":if i%2==0 {"user"}else{"assistant"},"content":"\u{a0}é\u{3000}"}))
        .collect();
    assert!(render(&serde_json::to_vec(&many).unwrap(), options()).is_ok());
    for count in [0, 2, 64, 65] {
        let values: Vec<_> = (0..count)
            .map(|i| json!({"role":if i%2==0 {"user"}else{"assistant"},"content":"x"}))
            .collect();
        assert!(render(&serde_json::to_vec(&values).unwrap(), options()).is_err());
    }
    for content in [
        "é".repeat(MAX_TEXT_BYTES / 2),
        "x".repeat(MAX_TEXT_BYTES - 31),
        "x".repeat(MAX_TEXT_BYTES - 50),
    ] {
        assert!(
            render(
                &serde_json::to_vec(&json!([{"role":"user","content":content}])).unwrap(),
                options()
            )
            .is_err()
        );
    }
}
#[test]
fn codec_allowance_precedes_all_asset_opens() {
    let path = Path::new("/nonexistent-toklet-qwen35-text-codec");
    let error = Codec::load(path, ALLOWANCE_BYTES - 1).err().unwrap();
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert!(error.to_string().contains("allowance"));
    assert_eq!(
        Codec::load(path, ALLOWANCE_BYTES).err().unwrap().kind(),
        io::ErrorKind::NotFound
    );
}
#[test]
fn regular_captured_asset_bytes_and_identity() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "toklet-qwen35-text-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(b"test").unwrap();
    }
    let digest = format!("{:x}", Sha256::digest(b"test"));
    assert_eq!(captured(&path, 4, &digest).unwrap(), b"test");
    assert!(captured(&path, 3, &digest).is_err());
    assert!(captured(&path, 4, &"0".repeat(64)).is_err());
    #[cfg(unix)]
    {
        let link = path.with_extension("link");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(captured(&link, 4, &digest).is_err());
        std::fs::remove_file(link).unwrap();
    }
    std::fs::remove_file(&path).unwrap();
    assert!(captured(&std::env::temp_dir(), 4, &digest).is_err());
}
#[test]
fn decode_bound_counts_special_flags_and_rejects_padded_ids() {
    let bounds = vec![3; VOCAB];
    assert_eq!(decode_bound(&bounds, &STOP_IDS, true).unwrap(), 0);
    assert_eq!(decode_bound(&bounds, &[248068, 248069], true).unwrap(), 6);
    assert_eq!(decode_bound(&bounds, &STOP_IDS, false).unwrap(), 6);
    for id in [VOCAB as u32, MODEL_PADDED_VOCAB as u32 - 1, u32::MAX] {
        assert!(decode_bound(&bounds, &[id], true).is_err());
    }
    assert!(decode_bound(&[4096], &[0; 17], false).is_err());
    assert_eq!(
        decode_bound(&[4096], &[0; 16], false).unwrap(),
        MAX_TEXT_BYTES
    );
    assert_eq!(
        decode_bound(&bounds, &vec![STOP_IDS[0]; MAX_CODEC_TOKENS], true).unwrap(),
        0
    );
    assert!(decode_bound(&bounds, &vec![STOP_IDS[0]; MAX_CODEC_TOKENS + 1], true).is_err());
}
#[test]
fn bounded_bytelevel_stream_constructor_and_atomic_padding_rejection() {
    // Existing independent tiny ByteLevel oracle exercises the new cold
    // constructor without a real tokenizer download. Actual source is explicit.
    let tokenizer = Tokenizer::from_bytes(include_bytes!(
        "../../../tests/fixtures/chat-stream-v1/qwen-tokenizer.json"
    ))
    .unwrap();
    let codec = Codec {
        tokenizer,
        token_bytes: Vec::new(),
    };
    let limits = ResponseLimits::new(128, 1024).unwrap();
    assert!(
        codec
            .response_stream(limits, STREAM_ALLOWANCE_BYTES - 1)
            .is_err()
    );
    let mut stream = codec
        .response_stream(limits, STREAM_ALLOWANCE_BYTES)
        .unwrap();
    assert!(stream.allocated_bytes().unwrap() <= STREAM_ALLOWANCE_BYTES);
    assert!(stream.push(&[0, VOCAB as u32]).is_err());
    assert_eq!(stream.consumed_tokens(), 0);
    assert_eq!(stream.emitted_bytes(), 0);
    assert!(stream.is_failed());
    assert!(stream.finish().is_err());
}

fn allocator_environment() -> Value {
    let value = |name| {
        std::env::var_os(name).map(|v| {
            let text = v.into_string().expect("Unicode allocator setting required");
            assert!(text.len() <= 16, "Bounded allocator setting required");
            text
        })
    };
    json!({"MallocLargeCache":value("MallocLargeCache"),"MallocAggressiveMadvise":value("MallocAggressiveMadvise"),"MallocNanoZone":value("MallocNanoZone")})
}
fn directory() -> std::path::PathBuf {
    std::env::var_os("TOKLET_QWEN35_TEXT_MODEL").unwrap().into()
}
fn verify(codec: &Codec, rows: &[Value]) -> usize {
    let mut streams = 0;
    for row in rows {
        match row["kind"].as_str().unwrap() {
            "chat" => {
                let raw = serde_json::to_vec(&row["messages"]).unwrap();
                assert_eq!(
                    codec.encode_chat(&raw, options()).unwrap(),
                    ids(&row["ids"])
                );
                assert_eq!(
                    codec
                        .encode_raw(row["rendered"].as_str().unwrap(), true)
                        .unwrap(),
                    ids(&row["add_special_tokens_ids"])
                );
            }
            "text" => {
                let text = row["input"].as_str().unwrap();
                assert_eq!(codec.encode_raw(text, false).unwrap(), ids(&row["ids"]));
                assert_eq!(
                    codec.encode_raw(text, true).unwrap(),
                    ids(&row["add_special_tokens_ids"])
                );
            }
            "control" => {
                let literal = row["metadata"]["content"].as_str().unwrap();
                assert_eq!(
                    codec.tokenizer.token_to_id(literal),
                    Some(row["resolved_id"].as_u64().unwrap() as u32)
                );
                assert_eq!(codec.encode_raw(literal, false).unwrap(), ids(&row["ids"]));
            }
            "structure" => {
                assert_eq!(ids(&row["stop_ids"]), STOP_IDS);
                assert_eq!(codec.stop_ids(), STOP_IDS);
            }
            "padding" => {
                for id in ids(&row["ids"]) {
                    assert!(codec.decode(&[id], true).is_err());
                }
            }
            _ => {}
        }
        if row.get("decode_skip").is_none() {
            continue;
        }
        let input = ids(&row["ids"]);
        assert_eq!(
            codec.decode(&input, true).unwrap(),
            row["decode_skip"].as_str().unwrap()
        );
        assert_eq!(
            codec.decode(&input, false).unwrap(),
            row["decode_keep"].as_str().unwrap()
        );
        if let Some(prefixes) = row.get("prefix_decodes") {
            for list in prefixes.as_object().unwrap().values() {
                for prefix in list.as_array().unwrap() {
                    let count = prefix["tokens"].as_u64().unwrap() as usize;
                    assert_eq!(
                        codec.decode(&input[..count], true).unwrap(),
                        prefix["text"].as_str().unwrap()
                    );
                }
            }
        }
        let expected = row["decode_skip"].as_str().unwrap();
        for chunk in [1, 3, 128] {
            let mut stream = codec
                .response_stream(
                    ResponseLimits::new(input.len().max(1), MAX_TEXT_BYTES).unwrap(),
                    STREAM_ALLOWANCE_BYTES,
                )
                .unwrap();
            let mut actual = String::new();
            for part in input.chunks(chunk) {
                actual.push_str(stream.push(part).unwrap());
                assert!(
                    expected.starts_with(&actual),
                    "append-only prefix: {}",
                    row["kind"]
                );
            }
            actual.push_str(stream.finish().unwrap());
            assert_eq!(actual, expected);
            assert_eq!(stream.emitted_bytes(), expected.len());
            assert_eq!(stream.consumed_tokens(), input.len());
            assert!(stream.is_finished());
            streams += 1;
        }
    }
    assert_eq!(streams, 165);
    streams
}
#[test]
#[ignore = "explicit pinned tokenizer only; no model inference or ordinary route admission"]
fn actual_source_codec_and_streams() {
    println!();
    let start = Instant::now();
    let codec = Codec::load(&directory(), ALLOWANCE_BYTES).unwrap();
    let load_ns = start.elapsed().as_nanos();
    let rows = fixture();
    assert_eq!(rows.len(), 100);
    let streams = verify(&codec, &rows);
    for id in [248077, 248319] {
        let mut stream = codec
            .response_stream(
                ResponseLimits::new(3, 1024).unwrap(),
                STREAM_ALLOWANCE_BYTES,
            )
            .unwrap();
        assert!(stream.push(&[1, id]).is_err());
        assert_eq!(stream.consumed_tokens(), 0);
        assert!(stream.is_failed());
    }
    println!(
        "QWEN35_TEXT_REPORT {}",
        json!({"records":100,"stream_schedules":streams,"stop_ids":STOP_IDS,"text_vocab":VOCAB,"padded_model_vocab":MODEL_PADDED_VOCAB,
        "codec_allowance_bytes":ALLOWANCE_BYTES,"stream_allowance_bytes":STREAM_ALLOWANCE_BYTES,"load_ns":load_ns,"fixture_sha256":FIXTURE_SHA,
        "allocator_environment":allocator_environment(),"all_text_ids_decode_streams_exact":true,"padded_ids_rejected":true,"model_inference_executed":false,"controller_qualified":false,"managed_adoption_qualified":false})
    );
}
#[test]
#[ignore = "explicit release text-only baseline,3warm+10measure per fresh process; no model inference"]
fn matched_codec_phase_baseline() {
    println!();
    let rows = fixture();
    let chat = rows
        .iter()
        .find(|r| r["kind"] == "chat" && r["name"] == "code")
        .unwrap();
    let text = rows
        .iter()
        .find(|r| r["kind"] == "text" && r["input"] == "Привет 世界 🙂")
        .unwrap();
    let input = serde_json::to_vec(&chat["messages"]).unwrap();
    let decode_ids = ids(&text["ids"]);
    for iteration in 0..13 {
        let start = Instant::now();
        let codec = Codec::load(&directory(), ALLOWANCE_BYTES).unwrap();
        let load_ns = start.elapsed().as_nanos();
        let start = Instant::now();
        let encoded = codec.encode_chat(&input, options()).unwrap();
        let encode_ns = start.elapsed().as_nanos();
        let start = Instant::now();
        let decoded = codec.decode(&decode_ids, true).unwrap();
        let decode_ns = start.elapsed().as_nanos();
        let start = Instant::now();
        let mut stream = codec
            .response_stream(
                ResponseLimits::new(128, 1024).unwrap(),
                STREAM_ALLOWANCE_BYTES,
            )
            .unwrap();
        let mut streamed = String::new();
        for id in &decode_ids {
            streamed.push_str(stream.push(&[*id]).unwrap());
        }
        streamed.push_str(stream.finish().unwrap());
        let stream_ns = start.elapsed().as_nanos();
        assert_eq!(encoded, ids(&chat["ids"]));
        assert_eq!(decoded, text["decode_skip"].as_str().unwrap());
        assert_eq!(streamed, decoded);
        println!(
            "QWEN35_TEXT_SAMPLE {}",
            json!({"iteration":iteration,"warmup":iteration<3,"load_ns":load_ns,"encode_ns":encode_ns,"decode_ns":decode_ns,"stream_ns":stream_ns,"prompt_ids":encoded.len(),"decode_ids":decode_ids.len()})
        );
    }
    println!(
        "QWEN35_TEXT_BASELINE_REPORT {}",
        json!({"allocator_environment":allocator_environment(),"warmups":3,"measurements":10,"fixture_sha256":FIXTURE_SHA,"codec_allowance_bytes":ALLOWANCE_BYTES,"stream_allowance_bytes":STREAM_ALLOWANCE_BYTES,"scope":"Repeated authenticated construction with warm filesystem cache; encode three-turn code chat, decode Unicode, one-ID stream including construction/finalization; no model/TTFT/overhead claim","model_inference_executed":false,"managed_adoption_qualified":false})
    );
}
