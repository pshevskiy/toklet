//! Exact pinned basic-text codec, separate from app profiles and model kernels.
//! These text bounds grant no numerical context/output or app capability.
use super::{
    invalid, pinned,
    stream::{ResponseLimits, ResponseStream},
};
use serde::{Deserialize, Serialize};
use std::{io, path::Path};
use tokenizers::Tokenizer;

pub const MAX_TEXT_BYTES: usize = 64 * 1024;
pub const MAX_CODEC_TOKENS: usize = 262144;
pub const VOCAB: usize = 262144;
/// Conservative single-codec allowance includes captured tokenizer JSON,
/// construction overlap, owned tables, metadata and bounded operation buffers.
/// Caller-owned retained responses/history need separate application admission.
pub const ALLOWANCE_BYTES: usize = 1024 * 1024 * 1024;
/// Separate text-stream owner allowance, including bounded temporary token text.
/// This grants no numerical context/output or app capability.
pub const STREAM_ALLOWANCE_BYTES: usize = 4 * 1024 * 1024;
pub const STOP_IDS: [u32; 3] = [1, 106, 50];
const MAX_TOKEN_BYTES: usize = 4096;
const CONTROLS: [(u32, &str); 24] = [
    (0, "<pad>"),
    (1, "<eos>"),
    (2, "<bos>"),
    (3, "<unk>"),
    (4, "<mask>"),
    (46, "<|tool>"),
    (47, "<tool|>"),
    (48, "<|tool_call>"),
    (49, "<tool_call|>"),
    (50, "<|tool_response>"),
    (51, "<tool_response|>"),
    (52, "<|\"|>"),
    (98, "<|think|>"),
    (100, "<|channel>"),
    (101, "<channel|>"),
    (105, "<|turn>"),
    (106, "<turn|>"),
    (255999, "<|image>"),
    (256000, "<|audio>"),
    (258880, "<|image|>"),
    (258881, "<|audio|>"),
    (258882, "<image|>"),
    (258883, "<audio|>"),
    (258884, "<|video|>"),
];

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderOptions {
    pub enable_thinking: bool,
    pub add_generation_prompt: bool,
}

#[derive(Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Role {
    User,
    Assistant,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Message {
    role: Role,
    content: String,
}

/// Parse typed JSON directly: serde rejects unknown and duplicate message fields.
/// Options are likewise typed/duplicate-safe when deserialized from JSON.
pub fn render(messages_json: &[u8], options: RenderOptions) -> io::Result<String> {
    if messages_json.len() > MAX_TEXT_BYTES
        || options.enable_thinking
        || !options.add_generation_prompt
    {
        return Err(invalid(
            "Unsupported Gemma basic-text request or byte bound",
        ));
    }
    let messages: Vec<Message> = serde_json::from_slice(messages_json).map_err(invalid)?;
    if messages.is_empty() || messages.len() > 63 || messages.len().is_multiple_of(2) {
        return Err(invalid("Gemma conversation must alternate and end in user"));
    }
    // Match the witness's UTF-8 messages JSON bound, including Python's default
    // separator spaces. The compact typed form differs by3 spaces per object
    // and one space between adjacent objects. Reject before output allocation.
    let canonical_bytes = serde_json::to_vec(&messages)
        .map_err(invalid)?
        .len()
        .checked_add(4 * messages.len() - 1)
        .ok_or_else(|| invalid("Gemma text size overflow"))?;
    if canonical_bytes > MAX_TEXT_BYTES {
        return Err(invalid("Gemma conversation exceeds text bound"));
    }
    let mut bytes = "<bos>".len() + "<|turn>model\n".len();
    for (index, message) in messages.iter().enumerate() {
        if message.role
            != if index % 2 == 0 {
                Role::User
            } else {
                Role::Assistant
            }
            || message.content.trim().is_empty()
            || message
                .content
                .chars()
                .any(|c| c < '\u{20}' && !matches!(c, '\t' | '\n' | '\r'))
            || CONTROLS
                .iter()
                .any(|(_, literal)| message.content.contains(literal))
        {
            return Err(invalid(
                "Unsupported Gemma role, content or control literal",
            ));
        }
        let role = if message.role == Role::User {
            "user"
        } else {
            "model"
        };
        bytes = bytes
            .checked_add(
                "<|turn>".len() + role.len() + 1 + message.content.trim().len() + "<turn|>\n".len(),
            )
            .ok_or_else(|| invalid("Gemma rendered size overflow"))?;
    }
    if bytes > MAX_TEXT_BYTES {
        return Err(invalid("Gemma rendered prompt exceeds byte bound"));
    }
    let mut text = String::new();
    text.try_reserve_exact(bytes).map_err(invalid)?;
    text.push_str("<bos>");
    for message in messages {
        text.push_str("<|turn>");
        text.push_str(if message.role == Role::User {
            "user\n"
        } else {
            "model\n"
        });
        text.push_str(message.content.trim());
        text.push_str("<turn|>\n");
    }
    text.push_str("<|turn>model\n");
    Ok(text)
}

pub struct Codec {
    tokenizer: Tokenizer,
    /// Cold precomputed upper bounds avoid vocabulary lookups/clones while
    /// admitting complete decode output. Replace never expands a normal token;
    /// a6-byte fallback spelling produces at most3 UTF-8 replacement bytes.
    token_bytes: Vec<u16>,
}

fn decode_bound(token_bytes: &[u16], ids: &[u32], skip_special_tokens: bool) -> io::Result<usize> {
    if ids.len() > MAX_CODEC_TOKENS {
        return Err(invalid("Gemma decode exceeds codec ID bound"));
    }
    let mut upper = 0_usize;
    for &id in ids {
        let size = token_bytes
            .get(id as usize)
            .ok_or_else(|| invalid("Gemma decode ID outside vocabulary"))?;
        if skip_special_tokens && CONTROLS.binary_search_by_key(&id, |(v, _)| *v).is_ok() {
            continue;
        }
        upper = upper
            .checked_add(usize::from(*size))
            .ok_or_else(|| invalid("Gemma decode byte count overflow"))?;
    }
    if upper > MAX_TEXT_BYTES {
        return Err(invalid(
            "Gemma decoded text exceeds conservative byte bound",
        ));
    }
    Ok(upper)
}

impl Codec {
    /// Admit the allowance before opening any asset. Hash and parse the same
    /// captured tokenizer bytes; earlier small assets select this exact profile.
    pub fn load(directory: &Path, available_bytes: usize) -> io::Result<Self> {
        if available_bytes < ALLOWANCE_BYTES {
            return Err(invalid("Gemma text allowance exceeds available bytes"));
        }
        for (name, bytes, sha) in [
            (
                "config.json",
                6395,
                "6397cb6eca41b911d1dcab74e17941351057bd759284052a2331918ff6f9246c",
            ),
            (
                "tokenizer_config.json",
                2740,
                "080d9e1aff284e2f6043889cd05367966f7c7b80e025fbc0b06745e218158656",
            ),
            (
                "generation_config.json",
                208,
                "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
            ),
            (
                "chat_template.jinja",
                17336,
                "2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082",
            ),
        ] {
            pinned(&directory.join(name), bytes, sha)?;
        }
        let bytes = pinned(
            &directory.join("tokenizer.json"),
            32169626,
            "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
        )?;
        tokenizers::utils::parallelism::set_parallelism(false);
        let tokenizer = Tokenizer::from_bytes(&bytes).map_err(invalid)?;
        if tokenizer.get_truncation().is_some()
            || tokenizer.get_padding().is_some()
            || tokenizer.get_vocab_size(true) != VOCAB
        {
            return Err(invalid("Unsupported Gemma tokenizer geometry or policy"));
        }
        let added = tokenizer.get_added_tokens_decoder();
        if added.len() != CONTROLS.len()
            || CONTROLS.iter().any(|(id, text)| {
                tokenizer.token_to_id(text) != Some(*id)
                    || !added
                        .get(id)
                        .is_some_and(|t| t.special && t.content == *text)
            })
        {
            return Err(invalid("Unsupported Gemma control-token inventory"));
        }
        let mut token_bytes = Vec::new();
        token_bytes.try_reserve_exact(VOCAB).map_err(invalid)?;
        for id in 0..VOCAB {
            let token = tokenizer
                .id_to_token(id as u32)
                .ok_or_else(|| invalid("Missing Gemma vocabulary ID"))?;
            if token.len() > MAX_TOKEN_BYTES {
                return Err(invalid("Gemma token exceeds text admission bound"));
            }
            token_bytes.push(token.len() as u16);
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
    /// No-Strip Replace/ByteFallback/Fuse delivery from this exact pinned codec.
    /// Whole fallback runs may be delayed until a normal token or finish; a
    /// later invalid byte can rewrite the entire run. Always finish on stop or
    /// cancellation. Caller-owned copies need separate memory admission.
    pub fn response_stream(
        &self,
        limits: ResponseLimits,
        available_bytes: usize,
    ) -> io::Result<ResponseStream<'_>> {
        if available_bytes < STREAM_ALLOWANCE_BYTES {
            return Err(invalid(
                "Gemma stream owner allowance exceeds available bytes",
            ));
        }
        let stream = ResponseStream::byte_fallback(&self.tokenizer, VOCAB as u32, false, limits);
        if stream.allocated_bytes()? > STREAM_ALLOWANCE_BYTES {
            return Err(invalid("Gemma stream storage exceeds admitted allowance"));
        }
        Ok(stream)
    }
    /// Raw text may contain control literals intentionally; only render rejects
    /// them. Identity postprocessing is witnessed with either explicit setting.
    pub fn encode_raw(&self, text: &str, add_special_tokens: bool) -> io::Result<Vec<u32>> {
        if text.len() > MAX_TEXT_BYTES {
            return Err(invalid("Gemma raw input exceeds byte bound"));
        }
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(invalid)?;
        if encoding.len() > MAX_CODEC_TOKENS {
            return Err(invalid("Gemma encoded IDs exceed codec bound"));
        }
        Ok(encoding.get_ids().to_vec())
    }
    pub fn encode_chat(
        &self,
        messages_json: &[u8],
        options: RenderOptions,
    ) -> io::Result<Vec<u32>> {
        self.encode_raw(&render(messages_json, options)?, false)
    }
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> io::Result<String> {
        let upper = decode_bound(&self.token_bytes, ids, skip_special_tokens)?;
        let text = self
            .tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(invalid)?;
        if text.len() > upper {
            return Err(invalid("Gemma tokenizer exceeded admitted decode bytes"));
        }
        Ok(text)
    }
}

#[cfg(test)]
mod tests;
