//! Cold application text boundary. No tokenizer or chat work enters kernels.
//! Independently rendered SmolLM2, Qwen2.5, plain Qwen3 and Mistral; no remote Jinja execution.
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};
use tokenizers::Tokenizer;

pub mod content;
pub mod dense_content;
pub mod gemma4;
pub mod llama3;
pub mod qwen35;
pub mod qwen3_content;
mod qwen_content;
pub mod stream;

pub const MAX_TEXT: usize = 64 * 1024;
/// Rendered-token safety ceiling, not model/session admission or output capacity.
pub const MAX_TOKENS: usize = crate::weights::MAX_CONTEXT_LIMIT;
/// Maximum visible output prefix, separate from the whole conversation limit.
pub const MAX_RESPONSE_TOKENS: usize = 256;
pub const TOKENIZER_ALLOWANCE: usize = 64 * 1024 * 1024;
const TOKENIZER_SHA: &str = "7d27c493c729a66ecefc837280b05d948b1ed50d130eebdbf911b1b36cf38ed7";
const CONFIG_SHA: &str = "a27f638bd2831f5c3dea654a75838930f2b11fbe550c4d4e1d5d7bd07157b2ee";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Smol,
    Qwen,
    Qwen15,
    Qwen3,
    Mistral,
    Llama3,
}
impl Profile {
    pub fn load(directory: &Path) -> io::Result<Self> {
        let bytes = bounded(&directory.join("config.json"), 16 * 1024)?;
        match format!("{:x}", Sha256::digest(bytes)).as_str() {
            "5da91905efdfde905f566d76d4031e6ca7c4cb2a01dc4e3812a70b855188fc73" => Ok(Self::Smol),
            "4dae7c9622e41e50752ba422149df0694044266c06cd851aff630799701c83c6" => Ok(Self::Qwen),
            "9f387dcb1bf045cadb34a36e67569abd73c00e5322603df2b19d0c0f8cc9d240" => Ok(Self::Qwen3),
            "efc145d2f53d6d6ef2639be0556b4ecf6faac6cf8a7199ad2b7bb5ebcf443bdd" => Ok(Self::Mistral),
            _ => Err(invalid(
                "Chat requires a supported pinned model configuration",
            )),
        }
    }
    pub fn supports(self, config: &crate::model::Config) -> bool {
        let (shape, theta, eps) = match self {
            Self::Smol => ((49152, 576, 1536, 30, 9, 3, 64), 100000.0, 1e-5),
            Self::Qwen => ((151936, 896, 4864, 24, 14, 2, 64), 1000000.0, 1e-6),
            Self::Qwen15 => ((151936, 1536, 8960, 28, 12, 2, 128), 1000000.0, 1e-6),
            Self::Qwen3 => ((151936, 1024, 3072, 28, 16, 8, 128), 1000000.0, 1e-6),
            Self::Mistral => ((32768, 4096, 14336, 32, 32, 8, 128), 1000000.0, 1e-5),
            Self::Llama3 => ((128256, 2048, 8192, 16, 32, 8, 64), 500000.0, 1e-5),
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
            Self::Smol | Self::Mistral => TOKENIZER_ALLOWANCE,
            Self::Qwen | Self::Qwen15 => 128 * 1024 * 1024,
            Self::Qwen3 | Self::Llama3 => 256 * 1024 * 1024,
        }
    }
    pub fn stop_ids(self) -> &'static [u32] {
        match self {
            Self::Smol => &[0, 2],
            Self::Qwen | Self::Qwen15 => &[151643, 151645],
            Self::Qwen3 => &[151645],
            Self::Mistral => &[2],
            Self::Llama3 => llama3::STOPS,
        }
    }
    fn special_ids(self) -> &'static [(&'static str, u32)] {
        match self {
            Self::Smol => &[("<|endoftext|>", 0), ("<|im_start|>", 1), ("<|im_end|>", 2)],
            Self::Qwen | Self::Qwen15 | Self::Qwen3 => &[
                ("<|endoftext|>", 151643),
                ("<|im_start|>", 151644),
                ("<|im_end|>", 151645),
            ],
            Self::Llama3 => llama3::SPECIAL,
            Self::Mistral => &[
                ("<unk>", 0),
                ("<s>", 1),
                ("</s>", 2),
                ("[INST]", 3),
                ("[/INST]", 4),
            ],
        }
    }
    fn vocab(self) -> u32 {
        match self {
            Self::Smol => 49152,
            Self::Qwen | Self::Qwen15 | Self::Qwen3 => 151936,
            Self::Mistral => 32768,
            Self::Llama3 => llama3::VOCAB,
        }
    }
    fn render(self, json: &[u8]) -> io::Result<String> {
        render_profile(self, json, &[])
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
fn render_profile(
    profile: Profile,
    json: &[u8],
    special_literals: &[String],
) -> io::Result<String> {
    if profile == Profile::Llama3 {
        return llama3::render(json, special_literals);
    }
    if json.len() > MAX_TEXT {
        return Err(invalid("Conversation exceeds 64 KiB"));
    }
    let messages: Vec<Message> = serde_json::from_slice(json).map_err(invalid)?;
    if messages.is_empty() || messages.len() > 63 || messages.len() % 2 != 1 {
        return Err(invalid("Conversation must end in a user message"));
    }
    let system = match profile {
        Profile::Smol => "You are a helpful AI assistant named SmolLM, trained by Hugging Face",
        Profile::Qwen | Profile::Qwen15 => {
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        }
        Profile::Mistral | Profile::Qwen3 | Profile::Llama3 => "",
    };
    let mut text = if profile == Profile::Mistral {
        "<s>".to_owned()
    } else if profile == Profile::Qwen3 {
        // Original plain Qwen3 branch does not invent a system message.
        String::new()
    } else {
        format!("<|im_start|>system\n{system}<|im_end|>\n")
    };
    for (index, message) in messages.iter().enumerate() {
        let role = if index % 2 == 0 { "user" } else { "assistant" };
        if message.role != role
            || profile
                .special_ids()
                .iter()
                .any(|(literal, _)| message.content.contains(literal))
            || special_literals
                .iter()
                .any(|literal| message.content.contains(literal))
        {
            return Err(invalid(
                "Expected alternating user/assistant text without chat control tokens",
            ));
        }
        if profile == Profile::Mistral {
            // Original pinned default template: one BOS, user instruction blocks,
            // assistant text followed by EOS, and no system/tool template.
            if index % 2 == 0 {
                text.push_str(&format!(" [INST] {} [/INST]", message.content));
            } else {
                text.push_str(&format!(" {} </s>", message.content));
            }
        } else {
            text.push_str(&format!(
                "<|im_start|>{role}\n{}<|im_end|>\n",
                message.content
            ));
        }
    }
    if profile != Profile::Mistral {
        text.push_str("<|im_start|>assistant\n");
    }
    if profile == Profile::Qwen3 {
        // Exact pinned enable_thinking=false branch. Reasoning/tool messages
        // remain unsupported; ordinary decoded token flags are unchanged.
        text.push_str("<think>\n\n</think>\n\n");
    }
    if text.len() > MAX_TEXT {
        return Err(invalid("Rendered conversation exceeds 64 KiB"));
    }
    Ok(text)
}

pub struct Codec {
    tokenizer: Tokenizer,
    profile: Profile,
    // Cold, bounded, exact-tokenizer control inventory. Kernels never see it.
    special_literals: Vec<String>,
    // Historical consumed asset pairing; legacy constructors remain unbound.
    content_identity: Option<content::CodecContentIdentity>,
    dense_content_identity: Option<dense_content::DenseCodecContentIdentity>,
}
impl Codec {
    pub fn load(directory: &Path) -> io::Result<Self> {
        Self::load_profile(directory, Profile::load(directory)?)
    }
    pub fn load_profile(directory: &Path, profile: Profile) -> io::Result<Self> {
        if Profile::load(directory)? != profile {
            return Err(invalid("Chat profile changed while opening"));
        }
        let (config_sha, tokenizer_sha, limit, config_limit) = match profile {
            Profile::Smol => (CONFIG_SHA, TOKENIZER_SHA, 4 * 1024 * 1024, 16 * 1024),
            Profile::Qwen => (
                "f7c61e32b7a17d19bf8e7037dcb74079a833e53ea9801f24008cac68458f03b7",
                "a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf",
                8 * 1024 * 1024,
                16 * 1024,
            ),
            Profile::Qwen3 => (
                "8e7dd4250db58b2a29792562d727b2715dc0fe79a94b446caa4c07db39eceba6",
                "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
                16 * 1024 * 1024,
                16 * 1024,
            ),
            Profile::Qwen15 => {
                return Err(invalid("Qwen1.5B requires the audited constructor"));
            }
            Profile::Llama3 => {
                return Err(invalid(
                    "Llama3 requires the direct pinned or audited constructor",
                ));
            }
            Profile::Mistral => (
                "e8fbcfef34d855dba2c7f1d91a2c5470cf493d00ecfd06488ce21e45325bc8ae",
                "e553af6fff7d7ad76e830608b218c5c0b0822998d5a1a96099a74cd3c1cb1a49",
                4 * 1024 * 1024,
                256 * 1024,
            ),
        };
        pinned(
            &directory.join("tokenizer_config.json"),
            config_limit,
            config_sha,
        )?;
        let bytes = pinned(&directory.join("tokenizer.json"), limit, tokenizer_sha)?;
        Self::from_tokenizer_bytes(&bytes, profile)
    }
    fn from_tokenizer_bytes(bytes: &[u8], profile: Profile) -> io::Result<Self> {
        tokenizers::utils::parallelism::set_parallelism(false);
        let tokenizer = Tokenizer::from_bytes(bytes).map_err(invalid)?;
        if tokenizer.get_truncation().is_some()
            || tokenizer.get_padding().is_some()
            || tokenizer
                .get_vocab(true)
                .values()
                .any(|&id| id >= profile.vocab())
            || profile
                .special_ids()
                .iter()
                .any(|(text, id)| tokenizer.token_to_id(text) != Some(*id))
        {
            return Err(invalid("Unsupported special IDs or tokenizer policy"));
        }
        let special_literals =
            if matches!(profile, Profile::Mistral | Profile::Qwen3 | Profile::Llama3) {
                tokenizer
                    .get_added_tokens_decoder()
                    .into_values()
                    // Qwen3's 26 reserved input literals include 12 ordinary added
                    // tokens. Reject all on input; preserve original decode flags.
                    .filter(|token| {
                        matches!(profile, Profile::Qwen3 | Profile::Llama3) || token.special
                    })
                    .map(|token| token.content)
                    .collect()
            } else {
                Vec::new()
            };
        Ok(Self {
            tokenizer,
            profile,
            special_literals,
            content_identity: None,
            dense_content_identity: None,
        })
    }
    pub fn stop_ids(&self) -> &'static [u32] {
        self.profile.stop_ids()
    }
    pub fn encode(&self, json: &[u8]) -> io::Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(
                render_profile(self.profile, json, &self.special_literals)?,
                false,
            )
            .map_err(invalid)?;
        let ids = encoding.get_ids();
        if ids.len() > MAX_TOKENS {
            return Err(invalid("Conversation is full; start a new chat"));
        }
        Ok(ids.to_vec())
    }
    // Decode the entire bounded visible prefix. Callers replace the displayed
    // response; they never concatenate independently decoded token strings.
    // A split UTF-8 sequence is repaired when its subsequent IDs arrive.
    pub fn decode(&self, ids: &[u32]) -> io::Result<String> {
        if ids.len() > MAX_RESPONSE_TOKENS || ids.iter().any(|&id| id >= self.profile.vocab()) {
            return Err(invalid("Response exceeds supported token bounds"));
        }
        let text = self.tokenizer.decode(ids, true).map_err(invalid)?;
        if text.len() > MAX_TEXT {
            return Err(invalid("Decoded response exceeds 64 KiB"));
        }
        Ok(text)
    }
}

// Decode-only portable fixture; production construction always validates pinned
// profile assets. Shared by codec and foreign-buffer tests, never model loading.
#[cfg(test)]
pub(crate) fn decode_fixture() -> Codec {
    Codec {
        tokenizer: Tokenizer::from_bytes(include_bytes!(
            "../tests/fixtures/chat-v1/tokenizer.json"
        ))
        .unwrap(),
        profile: Profile::Qwen,
        special_literals: Vec::new(),
        content_identity: None,
        dense_content_identity: None,
    }
}

// No merges: each ASCII payload byte has a known ByteLevel vocabulary ID.
// Production assets and the existing decode fixture remain unchanged.
#[cfg(test)]
pub(crate) fn encode_boundary_fixture() -> Codec {
    let mut codec = decode_fixture();
    let mut definition: serde_json::Value =
        serde_json::from_slice(include_bytes!("../tests/fixtures/chat-v1/tokenizer.json")).unwrap();
    definition["model"]["merges"] = serde_json::json!([]);
    codec.tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&definition).unwrap()).unwrap();
    codec
}

// Independent fixed Qwen template oracle. ASCII !..~ maps to byte-33;
// newline=198, space=220, start/end=151644/151645. Fixed overhead is97 IDs.
#[cfg(test)]
pub(crate) fn encode_boundary_case(total: usize) -> (Vec<u8>, Vec<u32>) {
    fn ascii(text: &str, ids: &mut Vec<u32>) {
        ids.extend(text.bytes().map(|byte| match byte {
            b'\n' => 198,
            b' ' => 220,
            33..=126 => u32::from(byte - 33),
            _ => panic!("non-ASCII boundary oracle"),
        }));
    }
    let payload = total.checked_sub(97).unwrap();
    let json =
        serde_json::to_vec(&serde_json::json!([{"role":"user","content":"1".repeat(payload)}]))
            .unwrap();
    let mut ids = vec![151644];
    ascii(
        "system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        &mut ids,
    );
    ids.extend([151645, 198, 151644]);
    ascii("user\n", &mut ids);
    ids.extend(std::iter::repeat_n(16, payload));
    ids.extend([151645, 198, 151644]);
    ascii("assistant\n", &mut ids);
    assert_eq!(ids.len(), total);
    (json, ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rendered_input_uses_engine_ceiling_without_relaxing_output_or_byte_limits() {
        let codec = encode_boundary_fixture();
        for count in [513, 1024, 32768] {
            let (json, expected) = encode_boundary_case(count);
            assert_eq!(codec.encode(&json).unwrap(), expected);
        }
        assert!(codec.encode(&encode_boundary_case(32769).0).is_err());
        assert!(codec.encode(&vec![b' '; MAX_TEXT + 1]).is_err());
        assert_eq!(MAX_RESPONSE_TOKENS, 256);
    }
    #[test]
    fn visible_response_bounds_preserve_exact_output_and_reject_overflow() {
        let codec = decode_fixture();
        for count in [0, 128, 129, 256] {
            assert_eq!(codec.decode(&vec![32; count]).unwrap(), "A".repeat(count));
        }
        assert!(codec.decode(&vec![32; 257]).is_err());
        assert!(codec.decode(&[151936]).is_err());
        assert_eq!(codec.decode(&[151645; 256]).unwrap(), "");
    }
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
    fn qwen3_reference() -> serde_json::Value {
        serde_json::from_str(include_str!(
            "../tests/fixtures/app-chat-qwen3-v1/reference.json"
        ))
        .unwrap()
    }
    #[test]
    fn independent_qwen3_plain_template_cases() {
        let fixture = qwen3_reference();
        for case in fixture["cases"].as_array().unwrap() {
            let rendered = Profile::Qwen3
                .render(&serde_json::to_vec(&case["messages"]).unwrap())
                .unwrap();
            assert_eq!(rendered, case["rendered"].as_str().unwrap());
            assert!(!rendered.starts_with("<|im_start|>system"));
        }
        assert_eq!(Profile::Qwen3.stop_ids(), [151645]);
        assert_eq!(Profile::Qwen3.allowance(), 256 * 1024 * 1024);
    }
    #[test]
    fn qwen3_denies_all_reserved_input_literals_without_reasoning_or_tools() {
        let fixture = qwen3_reference();
        let literals: Vec<_> = fixture["special_literals"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["text"].as_str().unwrap().to_owned())
            .collect();
        assert_eq!(literals.len(), 26);
        for literal in &literals {
            let input =
                serde_json::json!([{"role":"user","content":format!("before{literal}after")}]);
            assert!(
                render_profile(
                    Profile::Qwen3,
                    &serde_json::to_vec(&input).unwrap(),
                    &literals
                )
                .is_err()
            );
        }
        for input in [
            br#"[{"role":"user","content":"\u003cthink\u003e"}]"#.as_slice(),
            br#"[{"role":"system","content":"system"}]"#.as_slice(),
            br#"[{"role":"user","content":"hello","tool_calls":[]}]"#.as_slice(),
        ] {
            assert!(render_profile(Profile::Qwen3, input, &literals).is_err());
        }
    }
    #[test]
    #[ignore = "explicit pinned local Qwen3 tokenizer; no model download"]
    fn real_qwen3_profile_matches_independent_python() {
        let path = Path::new("models/MLX-Qwen3-0.6B-8bit-11de968");
        let codec = Codec::load(path).unwrap();
        assert_eq!(codec.profile, Profile::Qwen3);
        assert!(Codec::load_profile(path, Profile::Qwen).is_err());
        let fixture = qwen3_reference();
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
        for case in fixture["decode_cases"].as_array().unwrap() {
            let ids: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
            for end in 0..=ids.len() {
                assert_eq!(
                    codec.decode(&ids[..end]).unwrap(),
                    case["prefixes"][end].as_str().unwrap()
                );
                let mut stream =
                    codec.response_stream(stream::ResponseLimits::new(end.max(1), 4096).unwrap());
                let mut text = stream.push(&ids[..end]).unwrap().to_owned();
                text.push_str(stream.finish().unwrap());
                assert_eq!(text, case["prefixes"][end].as_str().unwrap());
            }
            for chunk in [1, 3, 128] {
                let mut stream = codec
                    .response_stream(stream::ResponseLimits::new(ids.len().max(1), 4096).unwrap());
                let mut text = String::new();
                for part in ids.chunks(chunk) {
                    text.push_str(stream.push(part).unwrap());
                }
                text.push_str(stream.finish().unwrap());
                assert_eq!(text, case["decoded"].as_str().unwrap());
            }
        }
        for item in fixture["special_literals"].as_array().unwrap() {
            let literal = item["text"].as_str().unwrap();
            let id = item["id"].as_u64().unwrap() as u32;
            let special = item["special"].as_bool().unwrap();
            assert_eq!(
                codec.decode(&[id]).unwrap(),
                if special { "" } else { literal }
            );
            let input = serde_json::json!([{"role":"user","content":literal}]);
            assert!(codec.encode(&serde_json::to_vec(&input).unwrap()).is_err());
        }
        assert!(codec.decode(&[151936]).is_err());
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
    fn independent_mistral_template_cases() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/app-chat-mistral-v1/reference.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            assert_eq!(
                Profile::Mistral
                    .render(&serde_json::to_vec(&case["messages"]).unwrap())
                    .unwrap(),
                case["rendered"].as_str().unwrap()
            );
        }
        assert_eq!(Profile::Mistral.stop_ids(), [2]);
        for marker in ["<unk>", "<s>", "</s>", "[INST]", "[/INST]"] {
            assert!(
                Profile::Mistral
                    .render(
                        &serde_json::to_vec(&serde_json::json!([
                {"role":"user", "content":marker}]))
                        .unwrap()
                    )
                    .is_err()
            );
        }
        assert!(
            Profile::Mistral
                .render(br#"[{"role":"system","content":"hello"}]"#)
                .is_err()
        );
    }
    #[test]
    #[ignore = "explicit local Mistral tokenizer check; no model download"]
    fn real_mistral_profile_matches_independent_python() {
        let codec = Codec::load(Path::new(
            "models/MLX-Mistral-7B-Instruct-v0.3-4bit-a4b8f87",
        ))
        .unwrap();
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/app-chat-mistral-v1/reference.json"
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
            for end in 0..=ids.len() {
                assert!(codec.decode(&ids[..end]).is_ok());
            }
        }
        assert_eq!(codec.stop_ids(), [2]);
        assert_eq!(codec.special_literals.len(), 771);
        assert_eq!(codec.decode(&[0, 1, 2, 3, 4, 10]).unwrap(), "");
        assert!(codec.decode(&[32768]).is_err());
        assert!(codec.decode(&[23325; MAX_RESPONSE_TOKENS]).is_ok());
        assert!(codec.decode(&[23325; MAX_RESPONSE_TOKENS + 1]).is_err());
        for literal in &codec.special_literals {
            for messages in [
                serde_json::json!([{"role":"user", "content":format!("x{literal}y")}]),
                serde_json::json!([{"role":"user","content":"hi"}, {"role":"assistant","content":literal},
                    {"role":"user","content":"more"}]),
            ] {
                assert!(
                    codec
                        .encode(&serde_json::to_vec(&messages).unwrap())
                        .is_err(),
                    "{literal}"
                );
            }
        }
        assert!(
            codec
                .encode(br#"[{"role":"user","content":"\u005bcontrol_8\u005d"}]"#)
                .is_err()
        );
        let long = serde_json::json!([{"role":"user","content":" hello".repeat(600)}]);
        assert!(codec.encode(&serde_json::to_vec(&long).unwrap()).is_ok());
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
