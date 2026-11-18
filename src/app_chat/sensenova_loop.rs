//! Pinned plain-text SenseNova Loop-9B codec for the published MLX assets.
use super::{
    invalid,
    stream::{ResponseLimits, ResponseStream},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    io::{self, Read},
    path::Path,
};
use tokenizers::Tokenizer;

pub const MAX_TEXT_BYTES: usize = 64 * 1024;
pub const MAX_CODEC_TOKENS: usize = 65536;
pub const VOCAB: u32 = 248077;
pub const STOP_IDS: [u32; 2] = [248044, 248046];
/// Cold allowance for the bounded capture and tokenizer construction overlap.
pub const ALLOWANCE_BYTES: usize = 512 * 1024 * 1024;
pub const STREAM_ALLOWANCE_BYTES: usize = 4 * 1024 * 1024;
const BASE_VOCAB: u32 = 248044;
const START: &str = "<|im_start|>";
const END: &str = "<|im_end|>\n";
const GENERATION: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
const MAX_TURNS: usize = 64;
const ASSETS: [(&str, usize, &str); 4] = [
    (
        "chat_template.jinja",
        7756,
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    ),
    (
        "tokenizer_config.json",
        1165,
        "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b",
    ),
    (
        "generation_config.json",
        198,
        "f288bb4281580138395c6acc7b3adb62f45c21530bc0708cb2cf9edf093f6a4f",
    ),
    (
        "tokenizer.json",
        19989325,
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    ),
];
const CONTROLS: [&str; 33] = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<tool_call>",
    "</tool_call>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<tts_pad>",
    "<tts_text_bos>",
    "<tts_text_eod>",
    "<tts_text_bos_single>",
    "<|audio_pad|>",
];

#[derive(Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn text(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Message {
    role: Role,
    content: String,
}

/// Exact plain alternating text branch with a generation prompt and thinking off.
/// An optional initial system turn precedes the user/assistant sequence.
pub fn render(messages_json: &[u8]) -> io::Result<String> {
    if messages_json.len() > MAX_TEXT_BYTES {
        return Err(invalid("SenseNova Loop chat JSON byte bound"));
    }
    let messages: Vec<Message> = serde_json::from_slice(messages_json).map_err(invalid)?;
    let initial_system = usize::from(
        messages
            .first()
            .is_some_and(|message| message.role == Role::System),
    );
    let turns = messages.len() - initial_system;
    if messages.len() > MAX_TURNS || turns.is_multiple_of(2) {
        return Err(invalid(
            "SenseNova Loop chat must alternate and end with a user turn",
        ));
    }
    let mut bytes = GENERATION.len();
    for (i, message) in messages.iter().enumerate() {
        let role = if i < initial_system {
            Role::System
        } else if (i - initial_system).is_multiple_of(2) {
            Role::User
        } else {
            Role::Assistant
        };
        let content = message.content.trim();
        if role != message.role
            || content.is_empty()
            || message
                .content
                .chars()
                .any(|c| c < '\u{20}' && !matches!(c, '\t' | '\n' | '\r'))
            || CONTROLS
                .iter()
                .any(|literal| message.content.contains(literal))
        {
            return Err(invalid(
                "Unsupported SenseNova Loop role, content or reserved literal",
            ));
        }
        bytes = bytes
            .checked_add(START.len() + role.text().len() + 1 + content.len() + END.len())
            .ok_or_else(|| invalid("SenseNova Loop rendered size overflow"))?;
    }
    if bytes > MAX_TEXT_BYTES {
        return Err(invalid("SenseNova Loop rendered text byte bound"));
    }
    let mut text = String::new();
    text.try_reserve_exact(bytes).map_err(invalid)?;
    for message in messages {
        text.push_str(START);
        text.push_str(message.role.text());
        text.push('\n');
        text.push_str(message.content.trim());
        text.push_str(END);
    }
    text.push_str(GENERATION);
    Ok(text)
}

fn captured(directory: &Path, (name, bytes, digest): (&str, usize, &str)) -> io::Result<Vec<u8>> {
    let file =
        crate::weights::audit::open_regular(&directory.join(name)).map_err(
            |error| match error {
                crate::weights::Error::Io { source, .. } => source,
                other => invalid(other),
            },
        )?;
    if file.metadata()?.len() != bytes as u64 {
        return Err(invalid("SenseNova Loop codec asset size differs"));
    }
    let mut raw = Vec::new();
    raw.try_reserve_exact(bytes).map_err(invalid)?;
    file.take(bytes as u64 + 1).read_to_end(&mut raw)?;
    if raw.len() != bytes || format!("{:x}", Sha256::digest(&raw)) != digest {
        return Err(invalid("SenseNova Loop codec asset identity differs"));
    }
    Ok(raw)
}

pub struct Codec {
    tokenizer: Tokenizer,
}

impl Codec {
    /// Admit a cold bounded codec and authenticate all four assets before parsing.
    pub fn load(directory: &Path, available_bytes: usize) -> io::Result<Self> {
        if available_bytes < ALLOWANCE_BYTES {
            return Err(invalid("SenseNova Loop codec exceeds available bytes"));
        }
        let [template, tokenizer_metadata, generation, tokenizer] = ASSETS;
        captured(directory, template)?;
        captured(directory, tokenizer_metadata)?;
        captured(directory, generation)?;
        let raw = captured(directory, tokenizer)?;
        tokenizers::utils::parallelism::set_parallelism(false);
        let tokenizer = Tokenizer::from_bytes(&raw).map_err(invalid)?;
        if tokenizer.get_padding().is_some()
            || tokenizer.get_truncation().is_some()
            || tokenizer.get_vocab_size(false) != BASE_VOCAB as usize
            || tokenizer.get_vocab_size(true) != VOCAB as usize
        {
            return Err(invalid("Unsupported SenseNova Loop tokenizer geometry"));
        }
        let added = tokenizer.get_added_tokens_decoder();
        if added.len() != CONTROLS.len()
            || CONTROLS.iter().enumerate().any(|(i, literal)| {
                let id = BASE_VOCAB + i as u32;
                let special = !(14..26).contains(&i);
                tokenizer.token_to_id(literal) != Some(id)
                    || !added
                        .get(&id)
                        .is_some_and(|token| token.content == *literal && token.special == special)
            })
        {
            return Err(invalid(
                "SenseNova Loop tokenizer added token inventory differs",
            ));
        }
        Ok(Self { tokenizer })
    }

    pub fn stop_ids(&self) -> &'static [u32] {
        &STOP_IDS
    }

    pub fn encode_chat(&self, messages_json: &[u8]) -> io::Result<Vec<u32>> {
        let text = render(messages_json)?;
        let encoded = self.tokenizer.encode(text, false).map_err(invalid)?;
        if encoded.len() > MAX_CODEC_TOKENS || encoded.get_ids().iter().any(|&id| id >= VOCAB) {
            return Err(invalid("SenseNova Loop encoded ID bound"));
        }
        Ok(encoded.get_ids().to_vec())
    }

    /// A ByteLevel text response stream uses the unpadded vocabulary.
    pub fn response_stream(
        &self,
        limits: ResponseLimits,
        available_bytes: usize,
    ) -> io::Result<ResponseStream<'_>> {
        if available_bytes < STREAM_ALLOWANCE_BYTES {
            return Err(invalid("SenseNova Loop stream exceeds available bytes"));
        }
        let stream = ResponseStream::byte_level(&self.tokenizer, VOCAB, limits);
        if stream.allocated_bytes()? > STREAM_ALLOWANCE_BYTES {
            return Err(invalid("SenseNova Loop stream storage exceeds allowance"));
        }
        Ok(stream)
    }
}

#[cfg(test)]
mod tests;
