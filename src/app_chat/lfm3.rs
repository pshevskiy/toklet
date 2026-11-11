//! Plain-text rendering and tokenizer admission for the pinned LFM3 checkpoint.
use std::{
    io::{self, Read},
    path::Path,
};

use serde::{Deserialize, Deserializer};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokenizers::{PaddingDirection, PaddingStrategy, Tokenizer};

use super::invalid;

pub const ALLOWANCE_BYTES: usize = 512 * 1024 * 1024;
pub const EOS: u32 = 124_900;
pub const MAX_TEXT_BYTES: usize = 256 * 1024;
const ASSETS: [(&str, usize, &str); 5] = [
    (
        "config.json",
        1334,
        "097cebcaefcd43e50f570dfdbf2cfeeb6ad2db004a2f24b51a301bc5fda9e156",
    ),
    (
        "generation_config.json",
        328,
        "1135b5376a736315f05f4cc16f9995f2256bb4035bf17fdcf8869b139930919a",
    ),
    (
        "tokenizer_config.json",
        1860,
        "86f27a9379ad5932315ed8dd9c2b54ba83b942e4e9f6dcac19e5d75664d29637",
    ),
    (
        "chat_template.jinja",
        12129,
        "f45005c50cd2a97329722bf3a2dee9f16c04cdf825a589b280fb7e99e6827818",
    ),
    (
        "tokenizer.json",
        17908795,
        "8f7ed4182b60c4ade9645e2f98b8f2141961bfd71c783e15daaa89e95ae66338",
    ),
];

fn captured(directory: &Path, (name, bytes, digest): (&str, usize, &str)) -> io::Result<Vec<u8>> {
    let path = directory.join(name);
    let file = crate::weights::audit::open_regular(&path).map_err(invalid)?;
    if file.metadata()?.len() != bytes as u64 {
        return Err(invalid(format!("LFM3 asset extent differs: {name}")));
    }
    let mut data = Vec::new();
    data.try_reserve_exact(bytes + 1).map_err(invalid)?;
    file.take(bytes as u64 + 1).read_to_end(&mut data)?;
    if data.len() != bytes || format!("{:x}", Sha256::digest(&data)) != digest {
        return Err(invalid(format!("LFM3 asset identity differs: {name}")));
    }
    Ok(data)
}

pub fn load_tokenizer(directory: &Path, available_bytes: usize) -> io::Result<Tokenizer> {
    if available_bytes < ALLOWANCE_BYTES {
        return Err(invalid("LFM3 codec exceeds admitted bytes"));
    }
    for entry in &ASSETS[..4] {
        captured(directory, *entry)?;
    }
    let bytes = captured(directory, ASSETS[4])?;
    tokenizers::utils::parallelism::set_parallelism(false);
    let mut tokenizer = Tokenizer::from_bytes(&bytes).map_err(invalid)?;
    let source_padding = tokenizer.get_padding().is_some_and(|p| {
        matches!(p.strategy, PaddingStrategy::BatchLongest)
            && matches!(p.direction, PaddingDirection::Left)
            && p.pad_to_multiple_of.is_none()
            && p.pad_id == 124_893
            && p.pad_type_id == 0
            && p.pad_token == "<|pad|>"
    });
    if !source_padding || tokenizer.get_truncation().is_some() {
        return Err(invalid(
            "LFM3 tokenizer padding or truncation profile differs",
        ));
    }
    tokenizer.with_padding(None);
    for (text, id) in [
        ("<|pad|>", 124_893),
        ("<|startoftext|>", 124_894),
        ("<|im_start|>", 124_899),
        ("<|im_end|>", EOS),
    ] {
        if tokenizer.token_to_id(text) != Some(id) {
            return Err(invalid("LFM3 tokenizer control ID differs"));
        }
    }
    if tokenizer
        .get_vocab(true)
        .iter()
        .any(|(text, id)| text.len() > 1024 || *id as usize >= crate::model::lfm3::VOCAB)
    {
        return Err(invalid("LFM3 tokenizer vocabulary or token extent differs"));
    }
    Ok(tokenizer)
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
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
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Message<'a> {
    role: Role,
    content: &'a str,
    #[serde(
        default,
        deserialize_with = "thinking_field",
        alias = "reasoning",
        alias = "reasoning_content"
    )]
    thinking: Option<&'a str>,
}

fn thinking_field<'de, D: Deserializer<'de>>(input: D) -> Result<Option<&'de str>, D::Error> {
    Ok(Some(Option::<&str>::deserialize(input)?.unwrap_or("")))
}

/// Apply the pinned plain-string branch with the source defaults for thinking.
/// Reject structured content and tool calls at this entry point.
pub fn render(messages: &[Value]) -> io::Result<String> {
    if messages.is_empty() {
        return Err(invalid("LFM3 text chat requires messages"));
    }
    let messages: Vec<Message<'_>> = messages
        .iter()
        .map(|message| Message::deserialize(message).map_err(invalid))
        .collect::<Result<_, _>>()?;
    let last_user = messages
        .iter()
        .rposition(|message| message.role == Role::User);
    let mut output = String::from("<|startoftext|>");
    for (index, message) in messages.iter().enumerate() {
        if message.role == Role::System && index != 0 {
            return Err(invalid("LFM3 system message must come first"));
        }
        let thinking = message.thinking;
        if message.role != Role::Assistant && thinking.is_some() {
            return Err(invalid("LFM3 thinking belongs to assistant messages"));
        }
        if message.role == Role::System && message.content.is_empty() {
            continue;
        }
        if message
            .content
            .len()
            .saturating_add(thinking.map_or(0, str::len))
            .saturating_add(output.len())
            > MAX_TEXT_BYTES
        {
            return Err(invalid("rendered LFM3 text exceeds 256 KiB"));
        }
        output.push_str("<|im_start|>");
        output.push_str(message.role.label());
        output.push('\n');
        let mut content = message.content;
        if message.role == Role::Assistant {
            if last_user.is_some_and(|last| index < last) {
                output.push_str("<think></think>");
                if let Some((_, tail)) = content.rsplit_once("</think>") {
                    content = tail.trim_matches(|c: char| {
                        c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
                    });
                }
            } else if let Some(thinking) = thinking {
                output.push_str("<think>");
                output.push_str(thinking);
                output.push_str("</think>");
            } else if !content.contains("</think>") {
                output.push_str("<think></think>");
            }
        }
        output.push_str(content);
        output.push_str("<|im_end|>\n");
    }
    output.push_str("<|im_start|>assistant\n<think>");
    if output.len() > MAX_TEXT_BYTES {
        return Err(invalid("rendered LFM3 text exceeds 256 KiB"));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn source_defaults_and_unsupported_chat_shapes() {
        let messages = [
            json!({"role":"user","content":"A"}),
            json!({"role":"assistant","content":"<think>x</think>  B "}),
            json!({"role":"user","content":"C"}),
        ];
        assert_eq!(
            render(&messages).unwrap(),
            "<|startoftext|><|im_start|>user\nA<|im_end|>\n<|im_start|>assistant\n<think></think>B<|im_end|>\n<|im_start|>user\nC<|im_end|>\n<|im_start|>assistant\n<think>"
        );
        for invalid in [
            json!({"role":"tool","content":"A"}),
            json!({"role":"user","content":[{"type":"text","text":"A"}]}),
            json!({"role":"assistant","content":"A","tool_calls":[]}),
            json!({"role":"assistant","content":"A","thinking":42}),
        ] {
            assert!(render(&[invalid]).is_err());
        }
        assert!(render(&[]).is_err());
        assert!(
            load_tokenizer(Path::new("missing"), ALLOWANCE_BYTES - 1)
                .err()
                .unwrap()
                .to_string()
                .contains("admitted bytes")
        );
    }
}
