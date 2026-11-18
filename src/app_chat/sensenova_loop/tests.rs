use super::*;

const FIXTURE: &[u8] = include_bytes!("../../../tests/fixtures/sensenova_loop/text-codec.json");

#[derive(Deserialize)]
struct Fixture {
    eos_ids: [u32; 2],
    cases: Vec<Case>,
    response: Response,
}
#[derive(Deserialize)]
struct Case {
    label: String,
    messages: serde_json::Value,
    rendered: String,
    ids: Vec<u32>,
}
#[derive(Deserialize)]
struct Response {
    text: String,
    ids: Vec<u32>,
}

#[test]
fn sense_nova_text_renderer_matches_checked_jinja_and_rejects_other_modes() {
    let fixture: Fixture = serde_json::from_slice(FIXTURE).unwrap();
    for case in fixture.cases {
        assert_eq!(
            render(&serde_json::to_vec(&case.messages).unwrap()).unwrap(),
            case.rendered,
            "{}",
            case.label
        );
    }
    for unsupported in [
        r#"[]"#,
        r#"[{"role":"system","content":"only"}]"#,
        r#"[{"role":"assistant","content":"unexpected"}]"#,
        r#"[{"role":"user","content":"first"},{"role":"assistant","content":"answer"}]"#,
        r#"[{"role":"user","content":"a"},{"role":"system","content":"b"},{"role":"user","content":"c"}]"#,
        r#"[{"role":"user","content":"<tool_response>payload</tool_response>"}]"#,
        r#"[{"role":"user","content":"<think>reason</think>"}]"#,
        r#"[{"role":"user","content":[{"type":"image"}]}]"#,
        r#"[{"role":"user","content":"hello","tool_calls":[]}]"#,
    ] {
        assert!(render(unsupported.as_bytes()).is_err(), "{unsupported}");
    }
    assert!(render(&vec![b' '; MAX_TEXT_BYTES + 1]).is_err());
    let error = Codec::load(Path::new("missing-text-assets"), ALLOWANCE_BYTES - 1)
        .err()
        .unwrap();
    assert_eq!(
        error.to_string(),
        "SenseNova Loop codec exceeds available bytes"
    );
}

#[test]
#[ignore = "explicit checked official text assets without model weights"]
fn sense_nova_token_ids_and_bytelevel_stream_match_checked_oracle() {
    let fixture: Fixture = serde_json::from_slice(FIXTURE).unwrap();
    let path = std::path::PathBuf::from(std::env::var_os("TOKLET_SENSENOVA_TEXT_ASSETS").unwrap());
    let codec = Codec::load(&path, ALLOWANCE_BYTES).unwrap();
    assert_eq!(codec.stop_ids(), fixture.eos_ids);
    for case in fixture.cases {
        assert_eq!(
            codec
                .encode_chat(&serde_json::to_vec(&case.messages).unwrap())
                .unwrap(),
            case.ids,
            "{}",
            case.label
        );
    }
    let limits = ResponseLimits::new(64, 256).unwrap();
    assert!(
        codec
            .response_stream(limits, STREAM_ALLOWANCE_BYTES - 1)
            .is_err()
    );
    let mut stream = codec
        .response_stream(limits, STREAM_ALLOWANCE_BYTES)
        .unwrap();
    let mut text = String::new();
    for id in fixture.response.ids {
        text.push_str(stream.push(&[id]).unwrap());
    }
    text.push_str(stream.finish().unwrap());
    assert_eq!(text, fixture.response.text);
    let mut stream = codec
        .response_stream(limits, STREAM_ALLOWANCE_BYTES)
        .unwrap();
    assert!(stream.push(&[VOCAB]).is_err());
    println!(
        "SENSENOVA_TEXT_CODEC_ORACLE prompts=4 exact_ids=true response=true eos=248044,248046"
    );
}
