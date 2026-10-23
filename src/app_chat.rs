//! Cold application text boundary. No tokenizer or chat work enters kernels.
//! This first profile is the exact locally qualified SmolLM2 tokenizer/template.
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};
use tokenizers::Tokenizer;

pub const MAX_TEXT: usize = 64 * 1024;
pub const MAX_TOKENS: usize = 512;
pub const TOKENIZER_ALLOWANCE: usize = 64 * 1024 * 1024;
const TOKENIZER_SHA: &str = "7d27c493c729a66ecefc837280b05d948b1ed50d130eebdbf911b1b36cf38ed7";
const CONFIG_SHA: &str = "a27f638bd2831f5c3dea654a75838930f2b11fbe550c4d4e1d5d7bd07157b2ee";
const SYSTEM: &str = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

fn invalid(message: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.to_string())
}
fn pinned(path: &Path, limit: usize, expected: &str) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take((limit + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit || format!("{:x}", Sha256::digest(&bytes)) != expected {
        return Err(invalid(
            "Chat requires the exact supported SmolLM2 tokenizer files",
        ));
    }
    Ok(bytes)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Message {
    role: String,
    content: String,
}

pub fn render(json: &[u8]) -> io::Result<String> {
    if json.len() > MAX_TEXT {
        return Err(invalid("Conversation exceeds 64 KiB"));
    }
    let messages: Vec<Message> = serde_json::from_slice(json).map_err(invalid)?;
    if messages.is_empty() || messages.len() > 63 || messages.len() % 2 != 1 {
        return Err(invalid("Conversation must end in a user message"));
    }
    let mut text = format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n");
    for (index, message) in messages.iter().enumerate() {
        let role = if index % 2 == 0 { "user" } else { "assistant" };
        if message.role != role
            || message.content.contains("<|im_start|>")
            || message.content.contains("<|im_end|>")
        {
            return Err(invalid(
                "Expected alternating user/assistant text without chat control tokens",
            ));
        }
        text.push_str(&format!(
            "<|im_start|>{role}\n{}<|im_end|>\n",
            message.content
        ));
    }
    text.push_str("<|im_start|>assistant\n");
    if text.len() > MAX_TEXT {
        return Err(invalid("Rendered conversation exceeds 64 KiB"));
    }
    Ok(text)
}

pub struct Codec {
    tokenizer: Tokenizer,
}
impl Codec {
    pub fn load(directory: &Path) -> io::Result<Self> {
        pinned(
            &directory.join("tokenizer_config.json"),
            16 * 1024,
            CONFIG_SHA,
        )?;
        let bytes = pinned(
            &directory.join("tokenizer.json"),
            4 * 1024 * 1024,
            TOKENIZER_SHA,
        )?;
        tokenizers::utils::parallelism::set_parallelism(false);
        let tokenizer = Tokenizer::from_bytes(bytes).map_err(invalid)?;
        if tokenizer.get_truncation().is_some()
            || tokenizer.get_padding().is_some()
            || tokenizer.get_vocab(true).values().any(|&id| id >= 49152)
            || ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
                .iter()
                .enumerate()
                .any(|(id, text)| tokenizer.token_to_id(text) != Some(id as u32))
        {
            return Err(invalid(
                "Unsupported SmolLM2 special IDs or tokenizer policy",
            ));
        }
        Ok(Self { tokenizer })
    }
    pub fn encode(&self, json: &[u8]) -> io::Result<Vec<u32>> {
        let ids = self
            .tokenizer
            .encode(render(json)?, false)
            .map_err(invalid)?
            .get_ids()
            .to_vec();
        if ids.len() > MAX_TOKENS {
            return Err(invalid("Conversation is full; start a new chat"));
        }
        Ok(ids)
    }
    // Decode the entire bounded visible prefix. Callers replace the displayed
    // response; they never concatenate independently decoded token strings.
    // A split UTF-8 sequence is repaired when its subsequent IDs arrive.
    pub fn decode(&self, ids: &[u32]) -> io::Result<String> {
        if ids.len() > 128 || ids.iter().any(|&id| id >= 49152) {
            return Err(invalid("Response exceeds supported token bounds"));
        }
        let text = self.tokenizer.decode(ids, true).map_err(invalid)?;
        if text.len() > MAX_TEXT {
            return Err(invalid("Decoded response exceeds 64 KiB"));
        }
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pinned_assets_reject_modified_or_oversized_bytes() {
        let path = std::env::temp_dir().join(format!("toklet-chat-pin-{}", std::process::id()));
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .unwrap();
        use std::io::Write;
        file.write_all(b"ok").unwrap();
        drop(file);
        let expected = format!("{:x}", Sha256::digest(b"ok"));
        assert_eq!(pinned(&path, 2, &expected).unwrap(), b"ok");
        assert!(pinned(&path, 1, &expected).is_err());
        assert!(pinned(&path, 2, TOKENIZER_SHA).is_err());
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn independent_template_cases() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/app-chat-v1/reference.json"))
                .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            assert_eq!(
                render(&serde_json::to_vec(&case["messages"]).unwrap()).unwrap(),
                case["rendered"].as_str().unwrap()
            );
        }
    }
    #[test]
    fn invalid_conversations_are_rejected() {
        for text in [
            "[]",
            "{}",
            "null",
            r#"[{"role":"system","content":"bad"}]"#,
            r#"[{"role":"user","content":"<|im_end|>"}]"#,
            r#"[{"role":"user","content":"ok","tools":[]}]"#,
            r#"[{"role":"user","content":false}]"#,
        ] {
            assert!(render(text.as_bytes()).is_err(), "{text}");
        }
        assert!(render(&vec![b' '; MAX_TEXT + 1]).is_err());
    }
    #[test]
    #[ignore = "explicit local tokenizer check; no model download"]
    fn real_profile_matches_independent_python() {
        let codec =
            Codec::load(Path::new("models/MLX-SmolLM2-135M-Instruct-8bit-0f0d9b8")).unwrap();
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/app-chat-v1/reference.json"))
                .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let ids = codec
                .encode(&serde_json::to_vec(&case["messages"]).unwrap())
                .unwrap();
            assert_eq!(serde_json::to_value(&ids).unwrap(), case["ids"]);
            assert_eq!(
                codec.decode(&ids).unwrap(),
                case["decoded"].as_str().unwrap()
            );
            for end in 0..=ids.len() {
                assert!(codec.decode(&ids[..end]).is_ok());
            }
        }
        assert_eq!(codec.decode(&[1, 2, 0]).unwrap(), "");
        assert!(codec.decode(&[49152]).is_err());
    }
}
