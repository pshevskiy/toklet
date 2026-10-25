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
