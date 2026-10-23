//! Cold application text boundary. No tokenizer or chat work enters kernels.
//! Exact, independently rendered SmolLM2 and Qwen2.5 profiles; no remote Jinja execution.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Smol,
    Qwen,
}
impl Profile {
    pub fn load(directory: &Path) -> io::Result<Self> {
        let bytes = bounded(&directory.join("config.json"), 16 * 1024)?;
        match format!("{:x}", Sha256::digest(bytes)).as_str() {
            "5da91905efdfde905f566d76d4031e6ca7c4cb2a01dc4e3812a70b855188fc73" => Ok(Self::Smol),
            "4dae7c9622e41e50752ba422149df0694044266c06cd851aff630799701c83c6" => Ok(Self::Qwen),
            _ => Err(invalid(
                "Chat requires a supported pinned model configuration",
            )),
        }
    }
    pub fn supports(self, config: &crate::model::Config) -> bool {
        let (shape, theta, eps) = match self {
            Self::Smol => ((49152, 576, 1536, 30, 9, 3, 64), 100000.0, 1e-5),
            Self::Qwen => ((151936, 896, 4864, 24, 14, 2, 64), 1000000.0, 1e-6),
        };
        (
            config.vocab_size,
            config.hidden_size,
            config.intermediate_size,
            config.num_layers,
            config.query_heads,
            config.kv_heads,
            config.head_dim,
        ) == shape
            && config.rope_theta == theta
            && config.rms_norm_eps == eps
    }
    pub fn allowance(self) -> usize {
        match self {
            Self::Smol => TOKENIZER_ALLOWANCE,
            Self::Qwen => 128 * 1024 * 1024,
        }
    }
    pub fn stop_ids(self) -> &'static [u32] {
        match self {
            Self::Smol => &[0, 2],
            Self::Qwen => &[151643, 151645],
        }
    }
    fn special_ids(self) -> [u32; 3] {
        match self {
            Self::Smol => [0, 1, 2],
            Self::Qwen => [151643, 151644, 151645],
        }
    }
    fn vocab(self) -> u32 {
        match self {
            Self::Smol => 49152,
            Self::Qwen => 151936,
        }
    }
    fn render(self, json: &[u8]) -> io::Result<String> {
        render_profile(self, json)
    }
}

fn invalid(message: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.to_string())
}
fn bounded(path: &Path, limit: usize) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take((limit + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(invalid("Chat asset exceeds its size bound"));
    }
    Ok(bytes)
}
fn pinned(path: &Path, limit: usize, expected: &str) -> io::Result<Vec<u8>> {
    let bytes = bounded(path, limit)?;
    if format!("{:x}", Sha256::digest(&bytes)) != expected {
        return Err(invalid("Chat requires the exact supported tokenizer files"));
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
    Profile::Smol.render(json)
}
fn render_profile(profile: Profile, json: &[u8]) -> io::Result<String> {
    if json.len() > MAX_TEXT {
        return Err(invalid("Conversation exceeds 64 KiB"));
    }
    let messages: Vec<Message> = serde_json::from_slice(json).map_err(invalid)?;
    if messages.is_empty() || messages.len() > 63 || messages.len() % 2 != 1 {
        return Err(invalid("Conversation must end in a user message"));
    }
    let system = match profile {
        Profile::Smol => "You are a helpful AI assistant named SmolLM, trained by Hugging Face",
        Profile::Qwen => "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    };
    let mut text = format!("<|im_start|>system\n{system}<|im_end|>\n");
    for (index, message) in messages.iter().enumerate() {
        let role = if index % 2 == 0 { "user" } else { "assistant" };
        if message.role != role
            || message.content.contains("<|im_start|>")
            || message.content.contains("<|im_end|>")
            || message.content.contains("<|endoftext|>")
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
    profile: Profile,
}
impl Codec {
    pub fn load(directory: &Path) -> io::Result<Self> {
        Self::load_profile(directory, Profile::load(directory)?)
    }
    pub fn load_profile(directory: &Path, profile: Profile) -> io::Result<Self> {
        if Profile::load(directory)? != profile {
            return Err(invalid("Chat profile changed while opening"));
        }
        let (config_sha, tokenizer_sha, limit) = match profile {
            Profile::Smol => (CONFIG_SHA, TOKENIZER_SHA, 4 * 1024 * 1024),
            Profile::Qwen => (
                "f7c61e32b7a17d19bf8e7037dcb74079a833e53ea9801f24008cac68458f03b7",
                "a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf",
                8 * 1024 * 1024,
            ),
        };
        pinned(
            &directory.join("tokenizer_config.json"),
            16 * 1024,
            config_sha,
        )?;
        let bytes = pinned(&directory.join("tokenizer.json"), limit, tokenizer_sha)?;
        tokenizers::utils::parallelism::set_parallelism(false);
        let tokenizer = Tokenizer::from_bytes(bytes).map_err(invalid)?;
        if tokenizer.get_truncation().is_some()
            || tokenizer.get_padding().is_some()
            || tokenizer
                .get_vocab(true)
                .values()
                .any(|&id| id >= profile.vocab())
            || ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
                .iter()
                .enumerate()
                .any(|(id, text)| tokenizer.token_to_id(text) != Some(profile.special_ids()[id]))
        {
            return Err(invalid("Unsupported special IDs or tokenizer policy"));
        }
        Ok(Self { tokenizer, profile })
    }
    pub fn stop_ids(&self) -> &'static [u32] {
        self.profile.stop_ids()
    }
    pub fn encode(&self, json: &[u8]) -> io::Result<Vec<u32>> {
        let ids = self
            .tokenizer
            .encode(self.profile.render(json)?, false)
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
        if ids.len() > 128 || ids.iter().any(|&id| id >= self.profile.vocab()) {
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
    fn independent_qwen_template_cases() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/app-chat-qwen-v1/reference.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            assert_eq!(
                Profile::Qwen
                    .render(&serde_json::to_vec(&case["messages"]).unwrap())
                    .unwrap(),
                case["rendered"].as_str().unwrap()
            );
        }
        assert_eq!(Profile::Qwen.stop_ids(), [151643, 151645]);
        assert_eq!(Profile::Smol.stop_ids(), [0, 2]);
    }
    #[test]
    #[ignore = "explicit local Qwen tokenizer check; no model download"]
    fn real_qwen_profile_matches_independent_python() {
        let codec =
            Codec::load(Path::new("models/MLX-Qwen2.5-0.5B-Instruct-8bit-6474d4c")).unwrap();
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/app-chat-qwen-v1/reference.json"
        ))
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
        }
        assert_eq!(codec.decode(&[151643, 151644, 151645]).unwrap(), "");
        assert!(codec.decode(&[151936]).is_err());
        assert!(
            codec
                .encode(br#"[{"role":"user","content":"<|endoftext|>"}]"#)
                .is_err()
        );
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
