//! Exact pinned MiniCPM5 basic text; outside kernels and managed model routes.
use super::{
    invalid,
    stream::{ResponseLimits, ResponseStream},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    io::{self, Read},
    path::Path,
};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer};

pub const MAX_TEXT_BYTES: usize = 64 * 1024;
pub const MAX_CODEC_TOKENS: usize = 65536;
pub const VOCAB: usize = 130560;
/// One codec: captured JSON/construction, tables, metadata and bounded calls.
pub const ALLOWANCE_BYTES: usize = 512 * 1024 * 1024;
/// Separate stream owner; caller-retained output/history has its own admission.
pub const STREAM_ALLOWANCE_BYTES: usize = 4 * 1024 * 1024;
pub const STOP_IDS: [u32; 2] = [1, 130073];
const MAX_TOKEN_BYTES: usize = 1024;
const EMPTY_THINK: &str = "<think>\n\n</think>\n\n";
const ASSETS: [(&str, usize, &str); 5] = [
    (
        "config.json",
        1056,
        "ba1fda219aca07e7809d454db02512fe3fb29ff49e1eb2364c8e9487e35a6fe3",
    ),
    (
        "tokenizer.json",
        9894424,
        "ad92aaa8d3032c98a9158b8c5e8682bed10027ed6463e4fb1320fe5384210873",
    ),
    (
        "tokenizer_config.json",
        539,
        "4d3748f79d8f82b68edf38b2a082b2df1f598e8e5a318dcb1f5c507d2f08b31b",
    ),
    (
        "generation_config.json",
        227,
        "90a38054c1abd33cb068dd17286e29eb3827f8619eab86bfea50e1fdb1eb1beb",
    ),
    (
        "chat_template.jinja",
        9060,
        "cc945752db555d60949b16989df4ccfeb52a313d6b4b5c5229dd786e2e9fcf1c",
    ),
];
const CONTROLS: [(u32, &str, bool); 32] = [
    (0, "<s>", true),
    (1, "</s>", true),
    (2, "<tool_call>", true),
    (3, "</tool_call>", true),
    (4, "<|im_sep|>", true),
    (5, "<|fim_prefix|>", true),
    (6, "<|fim_middle|>", true),
    (7, "<|fim_suffix|>", true),
    (8, "<think>", false),
    (9, "</think>", false),
    (10, "<tool_response>", true),
    (11, "</tool_response>", true),
    (12, "<tools>", true),
    (13, "</tools>", true),
    (14, "<arguments>", true),
    (15, "</arguments>", true),
    (16, "<parameters>", true),
    (17, "</parameters>", true),
    (18, "<function", true),
    (19, "</function>", true),
    (20, "<param", true),
    (21, "</param>", true),
    (130072, "<|im_start|>", true),
    (130073, "<|im_end|>", true),
    (130074, "<unk>", true),
    (130075, "<|thought_begin|>", true),
    (130076, "<|thought_end|>", true),
    (130077, "<|tool_call|>", true),
    (130078, "<|execute_start|>", true),
    (130079, "<|execute_end|>", true),
    (130080, "/think", true),
    (130081, "/no_think", true),
];

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderOptions {
    pub enable_thinking: bool,
    pub add_generation_prompt: bool,
}
#[derive(Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Role {
    System,
    User,
    Assistant,
}
impl Role {
    fn label(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Message {
    role: Role,
    content: String,
}

fn reserved(text: &str) -> bool {
    CONTROLS
        .iter()
        .any(|(_, literal, _)| text.contains(literal))
        || text.match_indices("<unused_token_").any(|(start, prefix)| {
            let rest = &text[start + prefix.len()..];
            let digits = rest.bytes().take(4).take_while(u8::is_ascii_digit).count();
            (1..=3).contains(&digits)
                && rest.as_bytes().get(digits) == Some(&b'>')
                && (digits == 1 || !rest.starts_with('0'))
                && rest[..digits].parse::<u16>().is_ok_and(|n| n <= 477)
        })
}

/// Original no-tools string subset, including optional first system and both
/// explicit thinking flags. System/user text is verbatim; plain assistant turns
/// use the source's empty-think wrapper and leading-LF stripping.
pub fn render(messages_json: &[u8], options: RenderOptions) -> io::Result<String> {
    if messages_json.len() > MAX_TEXT_BYTES {
        return Err(invalid("MiniCPM5 wire text bound exceeded"));
    }
    let messages: Vec<Message> = serde_json::from_slice(messages_json).map_err(invalid)?;
    if messages.is_empty() || messages.len() > 63 {
        return Err(invalid("MiniCPM5 message count outside bounds"));
    }
    let canonical = serde_json::to_vec(&messages)
        .map_err(invalid)?
        .len()
        .checked_add(4 * messages.len() - 1)
        .ok_or_else(|| invalid("MiniCPM5 canonical size overflow"))?;
    if canonical > MAX_TEXT_BYTES {
        return Err(invalid("MiniCPM5 canonical text bound exceeded"));
    }
    let first_user = usize::from(messages[0].role == Role::System);
    if first_user == messages.len() {
        return Err(invalid("MiniCPM5 requires at least one user"));
    }
    for (index, message) in messages.iter().enumerate() {
        let expected = if index < first_user {
            Role::System
        } else if (index - first_user).is_multiple_of(2) {
            Role::User
        } else {
            Role::Assistant
        };
        if message.role != expected
            || reserved(&message.content)
            || message
                .content
                .chars()
                .any(|c| c < '\u{20}' && !matches!(c, '\t' | '\n' | '\r'))
        {
            return Err(invalid(
                "Unsupported MiniCPM5 role, field or control literal",
            ));
        }
    }
    if options.add_generation_prompt && messages.last().unwrap().role != Role::User {
        return Err(invalid("MiniCPM5 generation prompt requires final user"));
    }
    let suffix = if !options.add_generation_prompt {
        ""
    } else if options.enable_thinking {
        "<|im_start|>assistant\n<think>\n"
    } else {
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    };
    let mut bytes = "<s>".len() + suffix.len();
    for message in &messages {
        let wrapper = if message.role == Role::Assistant {
            EMPTY_THINK.len()
        } else {
            0
        };
        bytes = bytes
            .checked_add(
                "<|im_start|>".len()
                    + message.role.label().len()
                    + 1
                    + wrapper
                    + if message.role == Role::Assistant {
                        message.content.trim_start_matches('\n').len()
                    } else {
                        message.content.len()
                    }
                    + "<|im_end|>\n".len(),
            )
            .ok_or_else(|| invalid("MiniCPM5 rendered size overflow"))?;
    }
    if bytes > MAX_TEXT_BYTES {
        return Err(invalid("MiniCPM5 rendered text bound exceeded"));
    }
    let mut text = String::new();
    text.try_reserve_exact(bytes).map_err(invalid)?;
    text.push_str("<s>");
    for message in &messages {
        text.push_str("<|im_start|>");
        text.push_str(message.role.label());
        text.push('\n');
        if message.role == Role::Assistant {
            text.push_str(EMPTY_THINK);
        }
        text.push_str(if message.role == Role::Assistant {
            message.content.trim_start_matches('\n')
        } else {
            &message.content
        });
        text.push_str("<|im_end|>\n");
    }
    text.push_str(suffix);
    Ok(text)
}

fn captured(path: &Path, bytes: usize, expected: &str) -> io::Result<Vec<u8>> {
    let file = crate::weights::audit::open_regular(path).map_err(|error| match error {
        crate::weights::Error::Io { source, .. } => source,
        other => invalid(other),
    })?;
    if file.metadata()?.len() != bytes as u64 {
        return Err(invalid("MiniCPM5 codec asset extent differs"));
    }
    let mut raw = Vec::new();
    raw.try_reserve_exact(bytes).map_err(invalid)?;
    file.take(bytes as u64 + 1).read_to_end(&mut raw)?;
    if raw.len() != bytes || format!("{:x}", Sha256::digest(&raw)) != expected {
        return Err(invalid("MiniCPM5 codec asset identity differs"));
    }
    Ok(raw)
}

pub struct Codec {
    tokenizer: Tokenizer,
    token_bytes: Vec<u16>,
}
fn source_padding(padding: Option<&PaddingParams>) -> bool {
    padding.is_some_and(|p| {
        matches!(p.strategy, PaddingStrategy::BatchLongest)
            && matches!(p.direction, PaddingDirection::Left)
            && p.pad_to_multiple_of.is_none()
            && p.pad_id == 1
            && p.pad_type_id == 0
            && p.pad_token == "</s>"
    })
}
fn token_decode_bound(text: &str) -> io::Result<u16> {
    if text.len() > MAX_TOKEN_BYTES {
        return Err(invalid("MiniCPM5 token byte extent differs"));
    }
    // ByteLevel maps ASCII characters to their ASCII bytes (one output byte).
    // Other mapped bytes need at most three bytes for a lossy replacement.
    // A token outside that alphabet falls back to its original UTF-8 bytes.
    let lossy: usize = text.chars().map(|c| if c.is_ascii() { 1 } else { 3 }).sum();
    u16::try_from(text.len().max(lossy)).map_err(invalid)
}
fn decode_bound(bounds: &[u16], ids: &[u32], skip_special: bool) -> io::Result<usize> {
    if ids.len() > MAX_CODEC_TOKENS {
        return Err(invalid("MiniCPM5 decode ID bound exceeded"));
    }
    let mut bytes = 0usize;
    for &id in ids {
        let upper = bounds
            .get(id as usize)
            .ok_or_else(|| invalid("MiniCPM5 ID outside text vocabulary"))?;
        if skip_special
            && CONTROLS
                .binary_search_by_key(&id, |(id, _, _)| *id)
                .is_ok_and(|index| CONTROLS[index].2)
        {
            continue;
        }
        bytes = bytes
            .checked_add(usize::from(*upper))
            .ok_or_else(|| invalid("MiniCPM5 decode size overflow"))?;
    }
    if bytes > MAX_TEXT_BYTES {
        return Err(invalid("MiniCPM5 conservative decode byte bound exceeded"));
    }
    Ok(bytes)
}
impl Codec {
    /// Admission precedes file access; parsing consumes the authenticated bytes.
    pub fn load(directory: &Path, available_bytes: usize) -> io::Result<Self> {
        if available_bytes < ALLOWANCE_BYTES {
            return Err(invalid("MiniCPM5 requires separately admitted512MiB codec"));
        }
        for (name, bytes, pin) in ASSETS
            .iter()
            .filter(|(name, _, _)| *name != "tokenizer.json")
        {
            captured(&directory.join(name), *bytes, pin)?;
        }
        let (_, bytes, pin) = ASSETS
            .iter()
            .find(|(name, _, _)| *name == "tokenizer.json")
            .unwrap();
        let raw = captured(&directory.join("tokenizer.json"), *bytes, pin)?;
        tokenizers::utils::parallelism::set_parallelism(false);
        let tokenizer = Tokenizer::from_bytes(&raw).map_err(invalid)?;
        if tokenizer.get_truncation().is_some()
            || !source_padding(tokenizer.get_padding())
            || tokenizer.get_vocab_size(false) != 130072
            || tokenizer.get_vocab_size(true) != VOCAB
        {
            return Err(invalid("MiniCPM5 tokenizer vocabulary/truncation differs"));
        }
        let added = tokenizer.get_added_tokens_decoder();
        if added.len() != 510
            || CONTROLS.iter().any(|(id, literal, special)| {
                tokenizer.token_to_id(literal) != Some(*id)
                    || !added.get(id).is_some_and(|token| {
                        token.content == *literal
                            && token.special == *special
                            && !token.normalized
                            && !token.single_word
                            && !token.lstrip
                            && !token.rstrip
                    })
            })
        {
            return Err(invalid("MiniCPM5 control token inventory differs"));
        }
        for index in 0..478u32 {
            let id = 130082 + index;
            let literal = format!("<unused_token_{index}>");
            if tokenizer.token_to_id(&literal) != Some(id)
                || !added.get(&id).is_some_and(|token| {
                    token.content == literal
                        && !token.special
                        && token.normalized
                        && !token.single_word
                        && !token.lstrip
                        && !token.rstrip
                })
            {
                return Err(invalid("MiniCPM5 unused token inventory differs"));
            }
        }
        let mut token_bytes = Vec::new();
        token_bytes.try_reserve_exact(VOCAB).map_err(invalid)?;
        for id in 0..VOCAB {
            let text = tokenizer
                .id_to_token(id as u32)
                .ok_or_else(|| invalid("MiniCPM5 tokenizer has ID holes"))?;
            token_bytes.push(token_decode_bound(&text)?);
        }
        Ok(Self {
            tokenizer,
            token_bytes,
        })
    }
    pub fn allowance_bytes(&self) -> usize {
        ALLOWANCE_BYTES
    }
    pub fn stop_ids(&self) -> &'static [u32] {
        &STOP_IDS
    }
    pub fn encode_raw(&self, text: &str, add_special_tokens: bool) -> io::Result<Vec<u32>> {
        if text.len() > MAX_TEXT_BYTES {
            return Err(invalid("MiniCPM5 raw text bound exceeded"));
        }
        let encoded = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(invalid)?;
        if encoded.len() > MAX_CODEC_TOKENS
            || encoded.get_ids().iter().any(|&id| id as usize >= VOCAB)
        {
            return Err(invalid("MiniCPM5 encoded ID bound exceeded"));
        }
        Ok(encoded.get_ids().to_vec())
    }
    pub fn encode_chat(&self, json: &[u8], options: RenderOptions) -> io::Result<Vec<u32>> {
        self.encode_raw(&render(json, options)?, false)
    }
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> io::Result<String> {
        let upper = decode_bound(&self.token_bytes, ids, skip_special_tokens)?;
        let text = self
            .tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(invalid)?;
        if text.len() > upper {
            return Err(invalid("MiniCPM5 decode exceeded conservative bound"));
        }
        Ok(text)
    }
    pub fn response_stream(
        &self,
        limits: ResponseLimits,
        available_bytes: usize,
    ) -> io::Result<ResponseStream<'_>> {
        if available_bytes < STREAM_ALLOWANCE_BYTES
            || !limits.fits(MAX_CODEC_TOKENS, MAX_TEXT_BYTES)
        {
            return Err(invalid("MiniCPM5 stream admission/text limits differ"));
        }
        let stream = ResponseStream::byte_level(&self.tokenizer, VOCAB as u32, limits);
        if stream.allocated_bytes()? > STREAM_ALLOWANCE_BYTES {
            return Err(invalid("MiniCPM5 stream owner exceeds allowance"));
        }
        Ok(stream)
    }
}
#[cfg(test)]
mod tests;
