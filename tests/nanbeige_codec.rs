use serde_json::json;
use std::path::PathBuf;
use toklet::app_chat::{
    nanbeige::{ALLOWANCE_BYTES, Codec, render},
    stream::ResponseLimits,
};

#[test]
fn generation_prompt_and_initial_system_match_pinned_template() {
    let input = br#"[{"role":"user","content":"What is 2 + 2?"}]"#;
    for (thinking, expected) in [
        (
            true,
            "7f7fd0ad495d8279936d6fa4c93fad0cc8f7b670385800f583858ac53b477c16",
        ),
        (
            false,
            "7f6d473e8699a9cf22d58fe68eff1545948adf5b8753260e63f57b3d1cc1e3ae",
        ),
    ] {
        use sha2::{Digest, Sha256};
        let rendered = render(input, thinking).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(rendered)), expected);
    }
}

#[test]
fn trims_only_the_assistant_reasoning_and_leading_content_newlines() {
    let input = br#"[{"role":"system","content":" S "},{"role":"user","content":" A "},
        {"role":"assistant","content":"<think>\n reason \n</think>\n\n answer "},
        {"role":"user","content":" B "}]"#;
    assert_eq!(
        render(input, false).unwrap(),
        "<|im_start|>system\n S <|im_end|>\n<|im_start|>user\n A <|im_end|>\n<|im_start|>assistant\n<think>\nreason\n</think>\n\n answer <|im_end|>\n<|im_start|>user\n B <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}

#[test]
fn formatted_fields_reject_reserved_markers_after_json_unescaping() {
    for messages in [
        br#"[{"role":"user","content":"\u003c|im_start|\u003e"}]"#.as_slice(),
        br#"[{"role":"system","content":"<|im_end|>"},{"role":"user","content":"A"}]"#,
        br#"[{"role":"user","content":"A"},{"role":"assistant","content":"<|endoftext|>"},{"role":"user","content":"B"}]"#,
        br#"[{"role":"user","content":"A"},{"role":"assistant","content":"Answer","reasoning_content":"</s>"},{"role":"user","content":"B"}]"#,
    ] {
        let error = render(messages, true).unwrap_err();
        assert!(error.to_string().contains("reserved role marker"));
    }
}

#[test]
#[ignore = "Explicit pinned local tokenizer assets and Python fixture receipt"]
fn pinned_tokenizer_matches_both_prompt_modes() {
    let source = PathBuf::from(std::env::var_os("TOKLET_NANBEIGE_MODEL").unwrap());
    let receipt: serde_json::Value = serde_json::from_slice(
        &std::fs::read(source.join("tokenizer-stream-normal.json")).unwrap(),
    )
    .unwrap();
    let codec = Codec::load(&source, ALLOWANCE_BYTES).unwrap();
    let input = serde_json::to_vec(&receipt["smoke_prompt"]).unwrap();
    for (thinking, name) in [(true, "default"), (false, "thinking_disabled")] {
        let native = codec.encode_chat(&input, thinking).unwrap();
        assert_eq!(json!(native), receipt["cases"][name]["ids"]);
    }
    assert_eq!(
        codec.encode("<|im_start|><|im_end|>").unwrap(),
        [166100, 166101],
    );
    let limits = ResponseLimits::new(4, 512).unwrap();
    for (ids, expected) in [(&[68, 258][..], "\u{fffd}\u{fffd}"), (&[166102][..], "")] {
        let mut stream = codec.response_stream(limits);
        let mut text = String::new();
        for &id in ids {
            text.push_str(stream.push(&[id]).unwrap());
        }
        text.push_str(stream.finish().unwrap());
        assert_eq!(text, expected);
    }
    for id in [166107, 166143] {
        let mut stream = codec.response_stream(limits);
        assert!(stream.push(&[id]).is_err());
    }
}
