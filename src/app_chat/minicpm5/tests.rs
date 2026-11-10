use super::*;
use serde_json::{Value, json};
use std::time::Instant;

fn fixture() -> Vec<Value> {
    let manifest = include_bytes!("../../../tests/fixtures/minicpm5-text-codec-v1/manifest.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(manifest)),
        "db01118bd97184fc8e681fc2d10d05e3536c6f354e7d7585d3fcf1e696ecc83c"
    );
    let raw = include_bytes!("../../../tests/fixtures/minicpm5-text-codec-v1/witness.jsonl");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "57dd38b7360a9bb713d2dadcd81e6f42ab1a64aef0e0aad96f9c8f2f028eca67"
    );
    raw.split(|&b| b == b'\n')
        .filter(|row| !row.is_empty())
        .map(|row| serde_json::from_slice(row).unwrap())
        .collect()
}
fn settings() -> RenderOptions {
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
fn directory() -> std::path::PathBuf {
    std::env::var_os("TOKLET_MINICPM5_TEXT_MODEL")
        .unwrap()
        .into()
}

#[test]
fn original_template_flags_system_empty_whitespace_and_rejections() {
    let mut counts = (0, 0);
    for row in fixture() {
        if row["kind"] != "chat" && row["kind"] != "rejection" {
            continue;
        }
        let options = serde_json::from_value::<RenderOptions>(row["options"].clone());
        let actual = options
            .map_err(invalid)
            .and_then(|options| render(&serde_json::to_vec(&row["messages"]).unwrap(), options));
        if row["kind"] == "chat" {
            assert_eq!(
                actual.unwrap(),
                row["rendered"].as_str().unwrap(),
                "{}",
                row["name"]
            );
            counts.0 += 1;
        } else {
            assert!(actual.is_err(), "{}", row["name"]);
            counts.1 += 1;
        }
    }
    assert_eq!(counts, (46, 28));
}
#[test]
fn typed_duplicate_unknown_and_option_fields_reject() {
    for raw in [
        r#"[{"role":"user","role":"user","content":"x"}]"#,
        r#"[{"role":"user","content":"x","content":"y"}]"#,
        r#"[{"role":"user","content":"x","extra":0}]"#,
    ] {
        assert!(render(raw.as_bytes(), settings()).is_err());
    }
    for raw in [
        r#"{"enable_thinking":false,"enable_thinking":true,"add_generation_prompt":true}"#,
        r#"{"enable_thinking":0,"add_generation_prompt":true}"#,
        r#"{"enable_thinking":false,"add_generation_prompt":true,"extra":false}"#,
    ] {
        assert!(serde_json::from_str::<RenderOptions>(raw).is_err());
    }
}
#[test]
fn reserved_inventory_is_exact_for_all_510_tokens_and_unused_boundaries() {
    let rows = fixture();
    let added = rows[0]["added_tokens"].as_array().unwrap();
    assert_eq!(added.len(), 510);
    for item in added {
        assert!(reserved(item["content"].as_str().unwrap()));
    }
    for text in [
        "<unused_token_478>",
        "<unused_token_0477>",
        "<unused_token_>",
        "<unused_token_999999>",
        "<unused_token_🙂>",
        "ordinary text",
    ] {
        assert!(!reserved(text));
    }
    assert_eq!(STOP_IDS, [1, 130073]);
    assert!(!CONTROLS.iter().find(|v| v.0 == 8).unwrap().2);
}
#[test]
fn text_and_render_bounds_precede_output_growth() {
    let raw = json!([{"role":"user","content":"x".repeat(MAX_TEXT_BYTES)}]);
    assert!(render(&serde_json::to_vec(&raw).unwrap(), settings()).is_err());
    let messages: Vec<_> = (0..63)
        .map(|i| json!({"role":if i%2==0{"user"}else{"assistant"},"content":""}))
        .collect();
    assert!(render(&serde_json::to_vec(&messages).unwrap(), settings()).is_ok());
    let mut too_many = messages;
    too_many.push(json!({"role":"assistant","content":""}));
    assert!(render(&serde_json::to_vec(&too_many).unwrap(), settings()).is_err());
    assert!(render(&[b' '; MAX_TEXT_BYTES + 1], settings()).is_err());
}
#[test]
fn admission_precedes_io_and_captured_bytes_require_regular_exact_identity() {
    let missing = std::env::temp_dir().join(format!("toklet-mini-missing-{}", std::process::id()));
    assert_eq!(
        Codec::load(&missing, ALLOWANCE_BYTES - 1)
            .err()
            .unwrap()
            .kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(
        Codec::load(&missing, ALLOWANCE_BYTES).err().unwrap().kind(),
        io::ErrorKind::NotFound
    );
    let path = std::env::temp_dir().join(format!("toklet-mini-assets-{}", std::process::id()));
    std::fs::write(&path, b"mini").unwrap();
    let pin = format!("{:x}", Sha256::digest(b"mini"));
    assert_eq!(captured(&path, 4, &pin).unwrap(), b"mini");
    assert!(captured(&path, 3, &pin).is_err());
    assert!(captured(&path, 4, &"0".repeat(64)).is_err());
    #[cfg(unix)]
    {
        let link = path.with_extension("link");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(captured(&link, 4, &pin).is_err());
        std::fs::remove_file(link).unwrap();
    }
    assert!(captured(&std::env::temp_dir(), 4, &pin).is_err());
    std::fs::remove_file(path).unwrap();
}
#[test]
fn decode_bounds_preserve_thinking_and_exclude_only_special_flags() {
    let bounds = vec![3; VOCAB];
    assert_eq!(decode_bound(&bounds, &STOP_IDS, true).unwrap(), 0);
    assert_eq!(decode_bound(&bounds, &[8, 9], true).unwrap(), 6);
    assert_eq!(decode_bound(&bounds, &[130082, 130559], true).unwrap(), 6);
    assert_eq!(decode_bound(&bounds, &STOP_IDS, false).unwrap(), 6);
    for id in [VOCAB as u32, u32::MAX] {
        assert!(decode_bound(&bounds, &[id], true).is_err());
    }
    assert!(decode_bound(&bounds, &vec![0; MAX_CODEC_TOKENS + 1], true).is_err());
    assert_eq!(
        decode_bound(&bounds, &vec![0; MAX_CODEC_TOKENS], true).unwrap(),
        0
    );
    assert!(decode_bound(&[u16::MAX], &[0, 0], false).is_err());
}
#[test]
fn ascii_decode_admission_and_lossy_upper_bounds_match_the_bytelevel_decoder() {
    use tokenizers::{Decoder, decoders::byte_level::ByteLevel};
    let decoder = ByteLevel::default();
    let mut tokens: Vec<_> = ByteLevel::alphabet()
        .into_iter()
        .map(|c| c.to_string())
        .collect();
    tokens.sort();
    tokens.extend(["ASCII text", "🙂", "é", "Ġhello", "<think>", "é🙂ASCII"].map(str::to_owned));
    for token in &tokens {
        let decoded = decoder.decode(vec![token.clone()]).unwrap();
        assert!(usize::from(token_decode_bound(token).unwrap()) >= decoded.len());
    }
    for pair in tokens.windows(2) {
        let decoded = decoder.decode(pair.to_vec()).unwrap();
        let bound: usize = pair
            .iter()
            .map(|t| usize::from(token_decode_bound(t).unwrap()))
            .sum();
        assert!(bound >= decoded.len());
    }
    assert_eq!(token_decode_bound(&"x".repeat(250)).unwrap(), 250);
    assert_eq!(
        decode_bound(&[1], &vec![0; MAX_TEXT_BYTES], false).unwrap(),
        MAX_TEXT_BYTES
    );
    assert!(decode_bound(&[3], &vec![0; 22000], false).is_err());
    assert!(token_decode_bound(&"x".repeat(MAX_TOKEN_BYTES + 1)).is_err());
}
#[test]
fn only_exact_source_single_sequence_padding_is_admitted() {
    let original = PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        direction: PaddingDirection::Left,
        pad_to_multiple_of: None,
        pad_id: 1,
        pad_type_id: 0,
        pad_token: "</s>".into(),
    };
    assert!(source_padding(Some(&original)));
    assert!(!source_padding(None));
    let mut changed = original.clone();
    changed.strategy = PaddingStrategy::Fixed(0);
    assert!(!source_padding(Some(&changed)));
    let mut changed = original.clone();
    changed.direction = PaddingDirection::Right;
    assert!(!source_padding(Some(&changed)));
    let mut changed = original.clone();
    changed.pad_to_multiple_of = Some(8);
    assert!(!source_padding(Some(&changed)));
    let mut changed = original.clone();
    changed.pad_id = 0;
    assert!(!source_padding(Some(&changed)));
    let mut changed = original.clone();
    changed.pad_type_id = 1;
    assert!(!source_padding(Some(&changed)));
    let mut changed = original;
    changed.pad_token = "<s>".into();
    assert!(!source_padding(Some(&changed)));
}
#[test]
fn stream_limits_owner_and_atomic_unknown_id_rejection() {
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
    assert!(
        codec
            .response_stream(
                ResponseLimits::new(MAX_CODEC_TOKENS + 1, 1024).unwrap(),
                STREAM_ALLOWANCE_BYTES
            )
            .is_err()
    );
    assert!(
        codec
            .response_stream(
                ResponseLimits::new(128, MAX_TEXT_BYTES + 1).unwrap(),
                STREAM_ALLOWANCE_BYTES
            )
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

fn verify(codec: &Codec, rows: &[Value]) -> usize {
    let mut streams = 0;
    for row in rows {
        match row["kind"].as_str().unwrap() {
            "chat" => {
                let options = serde_json::from_value(row["options"].clone()).unwrap();
                assert_eq!(
                    codec
                        .encode_chat(&serde_json::to_vec(&row["messages"]).unwrap(), options)
                        .unwrap(),
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
                assert_eq!(
                    codec
                        .encode_raw(row["input"].as_str().unwrap(), false)
                        .unwrap(),
                    ids(&row["ids"])
                );
                assert_eq!(
                    codec
                        .encode_raw(row["input"].as_str().unwrap(), true)
                        .unwrap(),
                    ids(&row["add_special_tokens_ids"])
                );
            }
            "decode" => {
                assert_eq!(
                    codec.decode(&ids(&row["ids"]), false).unwrap(),
                    row["raw_decoded"].as_str().unwrap()
                );
            }
            "padding" => {
                assert!(source_padding(codec.tokenizer.get_padding()));
                assert!(codec.tokenizer.get_truncation().is_none());
                for row in row["single_sequence_controls"].as_array().unwrap() {
                    let text = row["text"].as_str().unwrap();
                    let special = row["add_special_tokens"].as_bool().unwrap();
                    assert_eq!(codec.encode_raw(text, special).unwrap(), ids(&row["ids"]));
                    assert_eq!(
                        codec.tokenizer.encode_batch(vec![text], special).unwrap()[0].get_ids(),
                        ids(&row["ids"])
                    );
                }
            }
            _ => continue,
        }
        if row["kind"] == "text" || row["kind"] == "decode" {
            let sequence = ids(&row["ids"]);
            assert_eq!(
                codec.decode(&sequence, true).unwrap(),
                row["decoded"].as_str().unwrap()
            );
            for chunk in [1, 3, 128] {
                for prefix in row["prefixes"][chunk.to_string()].as_array().unwrap() {
                    let count = prefix["tokens"].as_u64().unwrap() as usize;
                    let mut stream = codec
                        .response_stream(
                            ResponseLimits::new(MAX_CODEC_TOKENS, MAX_TEXT_BYTES).unwrap(),
                            STREAM_ALLOWANCE_BYTES,
                        )
                        .unwrap();
                    let mut text = String::new();
                    for batch in sequence[..count].chunks(chunk) {
                        text.push_str(stream.push(batch).unwrap());
                    }
                    text.push_str(stream.finish().unwrap());
                    assert_eq!(
                        text,
                        prefix["text"].as_str().unwrap(),
                        "{} / {chunk} / {count}",
                        row["name"]
                    );
                }
                streams += 1;
            }
        }
    }
    assert!(
        codec
            .encode_raw(&"x".repeat(MAX_TEXT_BYTES + 1), false)
            .is_err()
    );
    assert!(codec.decode(&[VOCAB as u32], true).is_err());
    assert_eq!(codec.stop_ids(), &STOP_IDS);
    assert_eq!(codec.allowance_bytes(), ALLOWANCE_BYTES);
    streams
}
fn allocator() -> Value {
    let get = |name| std::env::var_os(name).map(|value| value.into_string().unwrap());
    json!({"MallocLargeCache":get("MallocLargeCache"),"MallocAggressiveMadvise":get("MallocAggressiveMadvise"),"MallocNanoZone":get("MallocNanoZone")})
}
#[test]
#[ignore = "Explicit real pinned MiniCPM5 tokenizer, no numerical model/GPU"]
fn real_codec_fidelity() {
    let codec = Codec::load(&directory(), ALLOWANCE_BYTES).unwrap();
    let rows = fixture();
    let streams = verify(&codec, &rows);
    println!(
        "{}",
        json!({"event":"minicpm5_codec_fidelity","records":rows.len(),"stream_schedules":streams,"stop_ids":codec.stop_ids(),"allocator":allocator(),"quality_adoption_qualified":false})
    );
}
#[test]
#[ignore = "Explicit release MiniCPM5 codec13fresh-construction timing/resource campaign"]
fn real_codec_benchmark() {
    let rows = fixture();
    let prompt = rows
        .iter()
        .find(|r| {
            r["kind"] == "chat"
                && r["name"] == "multiturn"
                && r["options"]["enable_thinking"] == false
                && r["options"]["add_generation_prompt"] == true
        })
        .unwrap();
    let text_row = rows
        .iter()
        .find(|r| r["kind"] == "text" && r["input"] == "Привет 世界 🙂")
        .unwrap();
    let sequence = ids(&text_row["ids"]);
    let messages = serde_json::to_vec(&prompt["messages"]).unwrap();
    for sample in 0..13 {
        let clock = Instant::now();
        let codec = Codec::load(&directory(), ALLOWANCE_BYTES).unwrap();
        let construction_ns = clock.elapsed().as_nanos();
        let clock = Instant::now();
        let rendered = render(&messages, settings()).unwrap();
        let render_ns = clock.elapsed().as_nanos();
        let clock = Instant::now();
        let encoded = codec.encode_raw(&rendered, false).unwrap();
        let encode_ns = clock.elapsed().as_nanos();
        assert_eq!(encoded, ids(&prompt["ids"]));
        let clock = Instant::now();
        let text = codec.decode(&sequence, true).unwrap();
        let decode_ns = clock.elapsed().as_nanos();
        assert_eq!(text, text_row["decoded"].as_str().unwrap());
        let clock = Instant::now();
        let mut stream = codec
            .response_stream(
                ResponseLimits::new(MAX_CODEC_TOKENS, MAX_TEXT_BYTES).unwrap(),
                STREAM_ALLOWANCE_BYTES,
            )
            .unwrap();
        let mut streamed = String::new();
        for &id in &sequence {
            streamed.push_str(stream.push(&[id]).unwrap());
        }
        streamed.push_str(stream.finish().unwrap());
        let stream_ns = clock.elapsed().as_nanos();
        assert_eq!(streamed, text);
        println!(
            "{}",
            json!({"event":"minicpm5_codec_sample","sample":sample,"warmup":sample<3,"construction_ns":construction_ns,"render_ns":render_ns,"encode_ns":encode_ns,"decode_ns":decode_ns,"stream_ns":stream_ns,"input_tokens":encoded.len(),"output_tokens":sequence.len(),"allocator":allocator(),"model_executed":false})
        );
    }
}
#[test]
#[ignore = "Explicit MiniCPM5 singleton BPE cache22000distinct250-byte words/660phases"]
fn real_codec_singleton_cache_stress() {
    let codec = Codec::load(&directory(), ALLOWANCE_BYTES).unwrap();
    let started = Instant::now();
    let mut phases = 0;
    for round in 0..3 {
        for batch in 0..220 {
            let mut text = String::new();
            for item in 0..100 {
                if item > 0 {
                    text.push(' ');
                }
                let mut n = batch * 100 + item;
                let mut prefix = [b'a'; 6];
                for value in &mut prefix {
                    *value += (n % 26) as u8;
                    n /= 26;
                }
                text.push_str(std::str::from_utf8(&prefix).unwrap());
                text.push_str(&"x".repeat(244));
            }
            let tokens = codec.encode_raw(&text, false).unwrap();
            assert_eq!(codec.decode(&tokens, true).unwrap(), text);
            phases += 1;
        }
        println!(
            "{}",
            json!({"event":"minicpm5_codec_cache_round","round":round,"phases":phases,"allocator":allocator()})
        );
    }
    assert_eq!(phases, 660);
    println!(
        "{}",
        json!({"event":"minicpm5_codec_cache_completed","phases":phases,"distinct_words":22000,"word_bytes":250,"elapsed_seconds":started.elapsed().as_secs_f64(),"quality_adoption_qualified":false})
    );
}
