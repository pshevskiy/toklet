//! Pinned Nanbeige plain-text chat framing and tokenizer boundary.

use super::{
    MAX_TEXT, invalid,
    stream::{ResponseLimits, ResponseStream},
};
use crate::mlx::nanbeige::{Asset, EOS, TOKENIZER_VOCAB, pinned_asset};
use serde::Deserialize;
use std::{io, path::Path};
use tokenizers::Tokenizer;

pub const ALLOWANCE_BYTES: usize = 512 * 1024 * 1024;
pub const STREAM_ALLOWANCE_BYTES: usize = 4 * 1024 * 1024;
const BOS: u32 = 166100;
const MAX_TURNS: usize = 64;
const DEFAULT_SYSTEM: &str = "你是南北阁，一款由BOSS直聘自主研发并训练的专业大语言模型。";
const START: &str = "<|im_start|>";
const END: &str = "<|im_end|>\n";
const ROLE_MARKERS: [&str; 5] = [START, "<|im_end|>", "<|endoftext|>", "<s>", "</s>"];
const ASSETS: [Asset; 4] = [
    (
        "chat_template.jinja",
        8362,
        "cb1699773afe04d17eacf8d0802feabac9ec10c3061e280f3121754aa749d87d",
    ),
    (
        "tokenizer_config.json",
        492,
        "b94dfdd9f7b774c3770a7722dea064a11207d8e8812563968161c51151832f25",
    ),
    (
        "generation_config.json",
        187,
        "68c690ce23efb6caae30c006ff3c1efd826297ff1df4338c04f7ac6f685d8746",
    ),
    (
        "tokenizer.json",
        18450976,
        "a002676e923776f5a59d21320d667920e732000f7762fb72e6be6e911790e060",
    ),
];

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Role {
    System,
    User,
    Assistant,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Message {
    role: Role,
    content: String,
    reasoning_content: Option<String>,
}

/// Plain-text portion of the pinned template. Supports an optional initial
/// system message and alternating user/assistant turns ending with a user.
pub fn render(messages_json: &[u8], thinking: bool) -> io::Result<String> {
    if messages_json.len() > MAX_TEXT {
        return Err(invalid("Nanbeige chat JSON byte bound"));
    }
    let messages: Vec<Message> = serde_json::from_slice(messages_json).map_err(invalid)?;
    let explicit_system = messages.first().is_some_and(|m| m.role == Role::System);
    let initial = usize::from(explicit_system);
    if messages.len() > MAX_TURNS || (messages.len() - initial).is_multiple_of(2) {
        return Err(invalid(
            "Nanbeige text chat must alternate and end with a user turn",
        ));
    }
    let system = if explicit_system {
        &messages[0].content
    } else {
        DEFAULT_SYSTEM
    };
    let mut text = format!("{START}system\n{system}{END}");
    for (i, message) in messages.iter().enumerate() {
        let role = if i < initial {
            Role::System
        } else if (i - initial).is_multiple_of(2) {
            Role::User
        } else {
            Role::Assistant
        };
        if role != message.role || (role != Role::Assistant && message.reasoning_content.is_some())
        {
            return Err(invalid("Nanbeige chat role or reasoning field differs"));
        }
        if [
            Some(message.content.as_str()),
            message.reasoning_content.as_deref(),
        ]
        .into_iter()
        .flatten()
        .any(|field| ROLE_MARKERS.iter().any(|marker| field.contains(marker)))
        {
            return Err(invalid(
                "Nanbeige formatted chat contains a reserved role marker",
            ));
        }
        match role {
            Role::System => {}
            Role::User => {
                text.push_str(START);
                text.push_str("user\n");
                text.push_str(&message.content);
                text.push_str(END);
            }
            Role::Assistant => {
                let (reasoning, visible) = match &message.reasoning_content {
                    Some(reasoning) => (reasoning.as_str(), message.content.as_str()),
                    None if message.content.contains("</think>") => {
                        let first = message.content.split_once("</think>").unwrap().0;
                        let reasoning = first
                            .trim_end_matches('\n')
                            .rsplit("<think>")
                            .next()
                            .unwrap();
                        let visible = message.content.rsplit("</think>").next().unwrap();
                        (
                            reasoning.trim_start_matches('\n'),
                            visible.trim_start_matches('\n'),
                        )
                    }
                    None => ("", message.content.as_str()),
                };
                text.push_str(START);
                text.push_str("assistant\n<think>\n");
                text.push_str(reasoning.trim());
                text.push_str("\n</think>\n\n");
                text.push_str(visible);
                text.push_str(END);
            }
        }
    }
    text.push_str(START);
    text.push_str("assistant\n<think>\n");
    if !thinking {
        text.push_str("\n</think>\n\n");
    }
    if text.len() > MAX_TEXT {
        return Err(invalid("Nanbeige rendered chat byte bound"));
    }
    Ok(text)
}

pub struct Codec {
    tokenizer: Tokenizer,
}

impl Codec {
    pub fn load(directory: &Path, available_bytes: usize) -> io::Result<Self> {
        if available_bytes < ALLOWANCE_BYTES {
            return Err(invalid("Nanbeige codec exceeds available bytes"));
        }
        let [template, config, generation, tokenizer] = ASSETS;
        for asset in [template, config, generation] {
            pinned_asset(directory, asset).map_err(invalid)?;
        }
        let raw = pinned_asset(directory, tokenizer).map_err(invalid)?;
        tokenizers::utils::parallelism::set_parallelism(false);
        let tokenizer = Tokenizer::from_bytes(&raw).map_err(invalid)?;
        if tokenizer.get_vocab_size(true) != TOKENIZER_VOCAB as usize
            || tokenizer.get_padding().is_some()
            || tokenizer.get_truncation().is_some()
            || tokenizer.token_to_id(START) != Some(BOS)
            || tokenizer.token_to_id(END.trim_end()) != Some(EOS)
            || tokenizer.token_to_id("<think>") != Some(166103)
            || tokenizer.token_to_id("</think>") != Some(166104)
        {
            return Err(invalid("Nanbeige tokenizer inventory differs"));
        }
        Ok(Self { tokenizer })
    }

    pub fn encode_chat(&self, messages_json: &[u8], thinking: bool) -> io::Result<Vec<u32>> {
        self.encode(&render(messages_json, thinking)?)
    }

    pub fn encode(&self, text: &str) -> io::Result<Vec<u32>> {
        if text.len() > MAX_TEXT {
            return Err(invalid("Nanbeige tokenizer input byte bound"));
        }
        let encoding = self.tokenizer.encode(text, false).map_err(invalid)?;
        let ids = encoding.get_ids();
        if ids.len() > super::MAX_TOKENS || ids.iter().any(|&id| id >= TOKENIZER_VOCAB) {
            return Err(invalid("Nanbeige encoded ID bound"));
        }
        Ok(ids.to_vec())
    }

    /// Hugging Face byte-fallback text decoding with skip_special_tokens=true.
    pub fn response_stream(&self, limits: ResponseLimits) -> ResponseStream<'_> {
        ResponseStream::byte_fallback(&self.tokenizer, TOKENIZER_VOCAB, true, limits)
    }
}
