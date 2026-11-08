//! Pinned bounded Qwen3.5 basic text, outside numerical kernels and app routes.
//! Original template whitespace and special flags are intentional. Both declared
//! EOS IDs follow the selected MLX-LM tokenizer wrapper's union convention.
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
use tokenizers::Tokenizer;

pub const MAX_TEXT_BYTES: usize = 64 * 1024;
pub const MAX_CODEC_TOKENS: usize = 65536;
pub const VOCAB: usize = 248077;
pub const MODEL_PADDED_VOCAB: usize = 248320;
/// Policy for one codec, captured JSON/construction overlap, metadata and bounded
/// encode/decode operations; not a measured footprint or numerical admission.
pub const ALLOWANCE_BYTES: usize = 512 * 1024 * 1024;
pub const STREAM_ALLOWANCE_BYTES: usize = 4 * 1024 * 1024;
pub const STOP_IDS: [u32; 2] = [248044, 248046];
const MAX_TOKEN_BYTES: usize = 4096;
const SUFFIX: &str = "\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const ASSETS: [(&str, usize, &str); 5] = [
    (
        "tokenizer_config.json",
        9432,
        "8fd4316758163961892a98841fe13c6354373a9080e13eaa86b1bff257f6ff9d",
    ),
    (
        "config.json",
        3024,
        "ac6f8df4f4c1fd9d2583996bdd7070ea801b703e8a2b36f6d4b673a94ea419b5",
    ),
    (
        "tokenizer.json",
        19989343,
        "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
    ),
    (
        "generation_config.json",
        115,
        "0c35bb39fbaed1ac0656baabc4f4e9bda20214e12336d0e4e8755aac1f487c2e",
    ),
    (
        "chat_template.jinja",
        7967,
        "19b43ed4a381c5ff680a33611685df18c9af5f64c381878f0614b7f8f2cc5755",
    ),
];
const CONTROLS: [(u32, &str, bool); 33] = [
    (248044, "<|endoftext|>", true),
    (248045, "<|im_start|>", true),
    (248046, "<|im_end|>", true),
    (248047, "<|object_ref_start|>", true),
    (248048, "<|object_ref_end|>", true),
    (248049, "<|box_start|>", true),
    (248050, "<|box_end|>", true),
    (248051, "<|quad_start|>", true),
    (248052, "<|quad_end|>", true),
    (248053, "<|vision_start|>", true),
    (248054, "<|vision_end|>", true),
    (248055, "<|vision_pad|>", true),
    (248056, "<|image_pad|>", true),
    (248057, "<|video_pad|>", true),
    (248058, "<tool_call>", false),
    (248059, "</tool_call>", false),
    (248060, "<|fim_prefix|>", false),
    (248061, "<|fim_middle|>", false),
    (248062, "<|fim_suffix|>", false),
    (248063, "<|fim_pad|>", false),
    (248064, "<|repo_name|>", false),
    (248065, "<|file_sep|>", false),
    (248066, "<tool_response>", false),
    (248067, "</tool_response>", false),
    (248068, "<think>", false),
    (248069, "</think>", false),
    (248070, "<|audio_start|>", true),
    (248071, "<|audio_end|>", true),
    (248072, "<tts_pad>", true),
    (248073, "<tts_text_bos>", true),
    (248074, "<tts_text_eod>", true),
    (248075, "<tts_text_bos_single>", true),
    (248076, "<|audio_pad|>", true),
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

/// Exact alternating text-only subset of the frozen source template. No system,
/// tool, media, reasoning fields or implicit default thinking branch are admitted.
pub fn render(messages_json: &[u8], options: RenderOptions) -> io::Result<String> {
    if messages_json.len() > MAX_TEXT_BYTES
        || options.enable_thinking
        || !options.add_generation_prompt
    {
        return Err(invalid("Unsupported Qwen3.5 text options or byte bound"));
    }
    let messages: Vec<Message> = serde_json::from_slice(messages_json).map_err(invalid)?;
    if messages.is_empty() || messages.len() > 63 || messages.len().is_multiple_of(2) {
        return Err(invalid(
            "Qwen3.5 conversation must alternate and end in user",
        ));
    }
    // Independent Python witness uses ensure_ascii=False with spaced separators.
    let canonical = serde_json::to_vec(&messages)
        .map_err(invalid)?
        .len()
        .checked_add(4 * messages.len() - 1)
        .ok_or_else(|| invalid("Qwen3.5 text size overflow"))?;
    if canonical > MAX_TEXT_BYTES {
        return Err(invalid("Qwen3.5 canonical text bound exceeded"));
    }
    let mut bytes = 3 + SUFFIX.len();
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
                .any(|(_, literal, _)| message.content.contains(literal))
        {
            return Err(invalid(
                "Unsupported Qwen3.5 role, text or reserved literal",
            ));
        }
        let role = if message.role == Role::User {
            "user"
        } else {
            "assistant"
        };
        bytes = bytes
            .checked_add(
                "<|im_start|>".len()
                    + role.len()
                    + 1
                    + message.content.trim().len()
                    + if message.role == Role::Assistant {
                        2
                    } else {
                        0
                    }
                    + "<|im_end|>\n".len(),
            )
            .ok_or_else(|| invalid("Qwen3.5 rendered size overflow"))?;
    }
    if bytes > MAX_TEXT_BYTES {
        return Err(invalid("Qwen3.5 rendered text bound exceeded"));
    }
    let mut text = String::new();
    text.try_reserve_exact(bytes).map_err(invalid)?;
    text.push_str("\n\n\n");
    for message in messages {
        text.push_str("<|im_start|>");
        text.push_str(if message.role == Role::User {
            "user\n"
        } else {
            "assistant\n"
        });
        text.push_str(message.content.trim());
        if message.role == Role::Assistant {
            text.push_str("\n\n");
        }
        text.push_str("<|im_end|>\n");
    }
    text.push_str(SUFFIX);
    Ok(text)
}

fn captured(path: &Path, bytes: usize, digest: &str) -> io::Result<Vec<u8>> {
    let file = crate::weights::audit::open_regular(path).map_err(|error| match error {
        crate::weights::Error::Io { source, .. } => source,
        other => invalid(other),
    })?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() != bytes as u64 {
        return Err(invalid("Qwen3.5 codec requires exact regular assets"));
    }
    let mut raw = Vec::new();
    raw.try_reserve_exact(bytes).map_err(invalid)?;
    file.take(bytes as u64 + 1).read_to_end(&mut raw)?;
    if raw.len() != bytes || format!("{:x}", Sha256::digest(&raw)) != digest {
        return Err(invalid("Qwen3.5 codec asset identity differs"));
    }
    Ok(raw)
}

pub struct Codec {
    tokenizer: Tokenizer,
    token_bytes: Vec<u16>,
}
fn decode_bound(token_bytes: &[u16], ids: &[u32], skip_special_tokens: bool) -> io::Result<usize> {
    if ids.len() > MAX_CODEC_TOKENS {
        return Err(invalid("Qwen3.5 decode ID bound exceeded"));
    }
    let mut bytes = 0usize;
    for &id in ids {
        let upper = token_bytes
            .get(id as usize)
            .ok_or_else(|| invalid("Qwen3.5 decode ID outside text vocabulary"))?;
        if skip_special_tokens
            && CONTROLS
                .binary_search_by_key(&id, |(id, _, _)| *id)
                .is_ok_and(|index| CONTROLS[index].2)
        {
            continue;
        }
        bytes = bytes
            .checked_add(usize::from(*upper))
            .ok_or_else(|| invalid("Qwen3.5 decode size overflow"))?;
    }
    if bytes > MAX_TEXT_BYTES {
        return Err(invalid("Qwen3.5 conservative decoded byte bound exceeded"));
    }
    Ok(bytes)
}
impl Codec {
    /// Admit before opening assets. Authentication and parsing use the same
    /// captured tokenizer bytes; no repository code or Jinja executes here.
    pub fn load(directory: &Path, available_bytes: usize) -> io::Result<Self> {
        if available_bytes < ALLOWANCE_BYTES {
            return Err(invalid("Qwen3.5 codec allowance exceeds available bytes"));
        }
        for (name, bytes, digest) in ASSETS
            .iter()
            .filter(|(name, _, _)| *name != "tokenizer.json")
        {
            captured(&directory.join(name), *bytes, digest)?;
        }
        let (_, bytes, digest) = ASSETS
            .iter()
            .find(|(name, _, _)| *name == "tokenizer.json")
            .unwrap();
        let raw = captured(&directory.join("tokenizer.json"), *bytes, digest)?;
        tokenizers::utils::parallelism::set_parallelism(false);
        let tokenizer = Tokenizer::from_bytes(&raw).map_err(invalid)?;
        if tokenizer.get_truncation().is_some()
            || tokenizer.get_padding().is_some()
            || tokenizer.get_vocab_size(false) != 248044
            || tokenizer.get_vocab_size(true) != VOCAB
        {
            return Err(invalid(
                "Unsupported Qwen3.5 tokenizer geometry or truncation",
            ));
        }
        let added = tokenizer.get_added_tokens_decoder();
        if added.len() != CONTROLS.len()
            || CONTROLS.iter().any(|(id, literal, special)| {
                tokenizer.token_to_id(literal) != Some(*id)
                    || !added
                        .get(id)
                        .is_some_and(|token| token.content == *literal && token.special == *special)
            })
        {
            return Err(invalid("Qwen3.5 added token inventory differs"));
        }
        let mut token_bytes = Vec::new();
        token_bytes.try_reserve_exact(VOCAB).map_err(invalid)?;
        for id in 0..VOCAB {
            let token = tokenizer
                .id_to_token(id as u32)
                .ok_or_else(|| invalid("Qwen3.5 text vocabulary has holes"))?;
            if token.len() > MAX_TOKEN_BYTES {
                return Err(invalid("Qwen3.5 token exceeds bounded text size"));
            }
            // ByteLevel maps one alphabet character to one byte. Lossy invalid
            // UTF-8 can expand each byte to a3-byte replacement character. Raw
            // fallback UTF-8 is also covered because token.len() is at most4
            // bytes per character, so include that separate conservative bound.
            let upper = token.len().max(3 * token.chars().count());
            token_bytes.push(u16::try_from(upper).map_err(invalid)?);
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
    /// This uses actual text vocabulary248077, so padded numerical IDs reject.
    /// Caller-retained output is separate from this stream owner allowance.
    pub fn response_stream(
        &self,
        limits: ResponseLimits,
        available_bytes: usize,
    ) -> io::Result<ResponseStream<'_>> {
        if available_bytes < STREAM_ALLOWANCE_BYTES {
            return Err(invalid("Qwen3.5 stream allowance exceeds available bytes"));
        }
        let stream = ResponseStream::byte_level(&self.tokenizer, VOCAB as u32, limits);
        if stream.allocated_bytes()? > STREAM_ALLOWANCE_BYTES {
            return Err(invalid("Qwen3.5 stream storage exceeds allowance"));
        }
        Ok(stream)
    }
    pub fn encode_raw(&self, text: &str, add_special_tokens: bool) -> io::Result<Vec<u32>> {
        if text.len() > MAX_TEXT_BYTES {
            return Err(invalid("Qwen3.5 raw text byte bound exceeded"));
        }
        let encoded = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(invalid)?;
        if encoded.len() > MAX_CODEC_TOKENS
            || encoded.get_ids().iter().any(|&id| id as usize >= VOCAB)
        {
            return Err(invalid("Qwen3.5 encoded ID bound exceeded"));
        }
        Ok(encoded.get_ids().to_vec())
    }
    pub fn encode_chat(
        &self,
        messages_json: &[u8],
        options: RenderOptions,
    ) -> io::Result<Vec<u32>> {
        self.encode_raw(&render(messages_json, options)?, false)
    }
    /// Padded/unknown IDs reject explicitly. No masking or numerical sampling
    /// policy follows from this cold text boundary.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> io::Result<String> {
        let upper = decode_bound(&self.token_bytes, ids, skip_special_tokens)?;
        let text = self
            .tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(invalid)?;
        if text.len() > upper {
            return Err(invalid("Qwen3.5 tokenizer exceeded admitted decode size"));
        }
        Ok(text)
    }
}
#[cfg(test)]
mod tests;
