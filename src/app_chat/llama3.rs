//! Pinned Llama 3.2 1B plain-text codec. No remote template execution.
//!
//! This cold text boundary does not activate a native model or qualify context.
//! The frozen date is part of the independently checked prompt policy.
use super::{MAX_TEXT, Message, invalid, pinned};
use std::{io, path::Path};

pub(super) const VOCAB: u32 = 128256;
pub(super) const SPECIAL: &[(&str, u32)] = &[
    ("<|begin_of_text|>", 128000),
    ("<|end_of_text|>", 128001),
    ("<|start_header_id|>", 128006),
    ("<|end_header_id|>", 128007),
    ("<|eom_id|>", 128008),
    ("<|eot_id|>", 128009),
];
pub(super) const STOPS: &[u32] = &[128001, 128008, 128009];
const ASSETS: &[(&str, usize, &str)] = &[
    (
        "tokenizer.json",
        16 * 1024 * 1024,
        "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
    ),
    (
        "tokenizer_config.json",
        64 * 1024,
        "9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e",
    ),
    (
        "special_tokens_map.json",
        16 * 1024,
        "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec",
    ),
];

// Python str.strip additionally treats these four C0 separators as whitespace.
fn python_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

pub(super) fn render(json: &[u8], reserved: &[String]) -> io::Result<String> {
    if json.len() > MAX_TEXT {
        return Err(invalid("Conversation exceeds 64 KiB"));
    }
    let messages: Vec<Message> = serde_json::from_slice(json).map_err(invalid)?;
    if messages.is_empty() || messages.len() > 63 || messages.len() % 2 != 1 {
        return Err(invalid("Conversation must end in a user message"));
    }
    let mut text = String::from(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
         Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>",
    );
    for (index, message) in messages.iter().enumerate() {
        let role = if index % 2 == 0 { "user" } else { "assistant" };
        if message.role != role || reserved.iter().any(|s| message.content.contains(s)) {
            return Err(invalid(
                "Expected alternating user/assistant text without chat control tokens",
            ));
        }
        text.push_str("<|start_header_id|>");
        text.push_str(role);
        text.push_str("<|end_header_id|>\n\n");
        text.push_str(message.content.trim_matches(python_space));
        text.push_str("<|eot_id|>");
        if text.len() > MAX_TEXT {
            return Err(invalid("Rendered conversation exceeds 64 KiB"));
        }
    }
    text.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    if text.len() > MAX_TEXT {
        return Err(invalid("Rendered conversation exceeds 64 KiB"));
    }
    Ok(text)
}

/// Direct pinned entry point sharing the ordinary bounded text implementation.
pub struct Codec(super::Codec);
impl Codec {
    /// Exact text assets only; model/session pairing is a separate gate.
    pub fn load(directory: &Path) -> io::Result<Self> {
        let mut tokenizer = None;
        for &(name, limit, digest) in ASSETS {
            let bytes = pinned(&directory.join(name), limit, digest)?;
            if name == "tokenizer.json" {
                tokenizer = Some(bytes);
            }
        }
        let bytes = tokenizer.ok_or_else(|| invalid("Missing tokenizer"))?;
        Ok(Self(super::Codec::from_tokenizer_bytes(
            &bytes,
            super::Profile::Llama3,
        )?))
    }
    pub fn stop_ids(&self) -> &'static [u32] {
        self.0.stop_ids()
    }
    pub fn encode(&self, json: &[u8]) -> io::Result<Vec<u32>> {
        self.0.encode(json)
    }
    pub fn decode(&self, ids: &[u32]) -> io::Result<String> {
        self.0.decode(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> serde_json::Value {
        serde_json::from_str(include_str!(
            "../../tests/fixtures/llama3-plain-text-v1/fixture.json"
        ))
        .unwrap()
    }
    #[test]
    fn independent_rendered_bytes_and_python_trim() {
        let f = fixture();
        let reserved: Vec<String> = SPECIAL.iter().map(|(s, _)| s.to_string()).collect();
        for case in f["cases"].as_array().unwrap() {
            let input = serde_json::to_vec(&case["messages"]).unwrap();
            let text = render(&input, &reserved).unwrap();
            assert_eq!(text, case["rendered"].as_str().unwrap(), "{}", case["name"]);
            assert_eq!(text.matches("<|begin_of_text|>").count(), 1);
        }
        assert_eq!(STOPS, [128001, 128008, 128009]);
    }
    #[test]
    fn rejects_unsupported_conversations_and_control_literals() {
        let reserved: Vec<String> = SPECIAL.iter().map(|(s, _)| s.to_string()).collect();
        for json in [
            "[]",
            "{}",
            "[{\"role\":\"system\",\"content\":\"x\"}]",
            "[{\"role\":\"user\",\"content\":\"x\",\"tools\":[]}]",
            "[{\"role\":\"user\",\"content\":3}]",
        ] {
            assert!(render(json.as_bytes(), &reserved).is_err());
        }
        for literal in &reserved {
            let input = serde_json::to_vec(&serde_json::json!([{"role":"user","content":literal}]))
                .unwrap();
            assert!(render(&input, &reserved).is_err());
        }
        assert!(render(&vec![b' '; MAX_TEXT + 1], &reserved).is_err());
        let messages: Vec<_> = (0..65)
            .map(|i| serde_json::json!({"role":if i%2==0 {"user"} else {"assistant"},"content":""}))
            .collect();
        assert!(render(&serde_json::to_vec(&messages).unwrap(), &reserved).is_err());
    }
    #[test]
    #[ignore = "explicit pinned cached Llama tokenizer; no download or model load"]
    fn cached_tokenizer_matches_independent_fixture() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models/MLX-Llama-3.2-1B-Instruct-8bit-d48cdf0");
        let codec = Codec::load(&directory).unwrap();
        let f = fixture();
        for case in f["cases"].as_array().unwrap() {
            let input = serde_json::to_vec(&case["messages"]).unwrap();
            let expected: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
            assert_eq!(codec.encode(&input).unwrap(), expected);
            assert_eq!(expected.iter().filter(|&&id| id == 128000).count(), 1);
        }
        for case in f["raw_roundtrips"].as_array().unwrap() {
            let ids: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
            assert_eq!(
                codec.decode(&ids).unwrap(),
                case["decoded"].as_str().unwrap()
            );
        }
        for &stop in codec.stop_ids() {
            assert_eq!(codec.decode(&[stop]).unwrap(), "");
        }
        for literal in &codec.0.special_literals {
            let input = serde_json::to_vec(&serde_json::json!([{"role":"user","content":literal}]))
                .unwrap();
            assert!(codec.encode(&input).is_err());
        }
        assert!(codec.decode(&[VOCAB]).is_err());
        assert!(
            codec
                .decode(&vec![0; super::super::MAX_RESPONSE_TOKENS + 1])
                .is_err()
        );
    }
}
