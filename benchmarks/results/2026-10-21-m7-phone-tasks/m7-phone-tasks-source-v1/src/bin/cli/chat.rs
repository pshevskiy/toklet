use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};
use tokenizers::Tokenizer;

pub const MAX_INPUT_BYTES: usize = 1024 * 1024;
pub const MAX_TEXT_BYTES: usize = 256 * 1024;
const TEMPLATE_SHA: &str = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f";
const DEFAULT_SYSTEM: &str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

pub fn invalid(message: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.to_string())
}

pub fn read_bounded(path: &Path, limit: usize) -> io::Result<Vec<u8>> {
    let mut data = Vec::new();
    File::open(path)?
        .take((limit + 1) as u64)
        .read_to_end(&mut data)?;
    if data.len() > limit {
        return Err(invalid(format!("{} exceeds {limit} bytes", path.display())));
    }
    Ok(data)
}

pub fn json_file(path: &Path, limit: usize) -> io::Result<Value> {
    serde_json::from_slice(&read_bounded(path, limit)?).map_err(invalid)
}

pub struct Assets {
    pub tokenizer: Tokenizer,
    /// Conservative parser/container allowance, not a physical measurement.
    pub admitted_bytes: usize,
}

/// Owned state for the tokenizers streaming decoder. Keep unfinished bytes
/// across resumable request boundaries; flush only when the response ends.
#[derive(Default)]
pub struct OutputDecoder {
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}
impl OutputDecoder {
    pub fn step(&mut self, assets: &Assets, id: u32) -> io::Result<String> {
        tokenizers::tokenizer::step_decode_stream(
            &assets.tokenizer,
            vec![id],
            true,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
        .map(|text| text.unwrap_or_default())
        .map_err(|e| invalid(e.to_string()))
    }

    pub fn finish(&mut self, assets: &Assets) -> io::Result<String> {
        let full = assets
            .tokenizer
            .decode(&self.ids, true)
            .map_err(|e| invalid(e.to_string()))?;
        let tail = full
            .strip_prefix(&self.prefix)
            .ok_or_else(|| invalid("streaming decode prefix mismatch"))?
            .to_owned();
        self.ids.clear();
        self.prefix.clear();
        self.prefix_index = 0;
        Ok(tail)
    }
}

impl Assets {
    pub fn load(directory: &Path, vocab_size: usize) -> io::Result<Self> {
        let config = json_file(&directory.join("tokenizer_config.json"), MAX_INPUT_BYTES)?;
        let template = config["chat_template"]
            .as_str()
            .ok_or_else(|| invalid("missing Qwen chat template"))?;
        if format!("{:x}", Sha256::digest(template.as_bytes())) != TEMPLATE_SHA {
            return Err(invalid(
                "unsupported chat template; expected the qualified Qwen2.5 template",
            ));
        }
        let bytes = read_bounded(&directory.join("tokenizer.json"), 32 * 1024 * 1024)?;
        let admitted_bytes = bytes
            .len()
            .checked_mul(16)
            .and_then(|n| n.checked_add(MAX_INPUT_BYTES))
            .ok_or_else(|| invalid("tokenizer admission overflow"))?;
        tokenizers::utils::parallelism::set_parallelism(false);
        let tokenizer = Tokenizer::from_bytes(&bytes).map_err(|e| invalid(e.to_string()))?;
        if tokenizer.get_truncation().is_some() || tokenizer.get_padding().is_some() {
            return Err(invalid("tokenizer truncation/padding must be disabled"));
        }
        for (text, id) in [
            ("<|endoftext|>", 151643),
            ("<|im_start|>", 151644),
            ("<|im_end|>", 151645),
        ] {
            if tokenizer.token_to_id(text) != Some(id) || id as usize >= vocab_size {
                return Err(invalid("tokenizer special IDs do not match the model"));
            }
        }
        // Bound token strings used by streaming decode, including padded model IDs.
        if tokenizer
            .get_vocab(true)
            .iter()
            .any(|(text, &id)| id as usize >= vocab_size || text.len() > 1024)
        {
            return Err(invalid(
                "tokenizer vocabulary exceeds the model or supported token length",
            ));
        }
        Ok(Self {
            tokenizer,
            admitted_bytes,
        })
    }

    pub fn encode(&self, text: &str) -> io::Result<Vec<u32>> {
        if text.len() > MAX_TEXT_BYTES {
            return Err(invalid("rendered text exceeds 256 KiB"));
        }
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| invalid(e.to_string()))?
            .get_ids()
            .to_vec())
    }
}

pub fn stop_ids(directory: &Path, vocab_size: usize) -> io::Result<Vec<u32>> {
    let path = directory.join("generation_config.json");
    let config = match json_file(&path, MAX_INPUT_BYTES) {
        Ok(config) => config,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            json_file(&directory.join("config.json"), MAX_INPUT_BYTES)?
        }
        Err(e) => return Err(e),
    };
    let values: Vec<&Value> = match &config["eos_token_id"] {
        Value::Null => Vec::new(),
        Value::Array(values) => values.iter().collect(),
        value => vec![value],
    };
    values
        .into_iter()
        .map(|value| {
            let id = value
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| invalid("invalid EOS token ID"))?;
            if id as usize >= vocab_size {
                return Err(invalid("EOS token ID outside model vocabulary"));
            }
            Ok(id)
        })
        .collect()
}

fn content(message: &Value, optional: bool) -> io::Result<&str> {
    match &message["content"] {
        Value::String(s) => Ok(s),
        Value::Null if optional => Ok(""),
        _ => Err(invalid("chat message content must be text")),
    }
}

/// Qualified rendering of the frozen Qwen template, including its tool branches.
/// Keep it in the CLI; no formatting or tokenizer work enters inference kernels.
pub fn render(messages: &[Value], tools: &[Value], generation_prompt: bool) -> io::Result<String> {
    if messages.is_empty() {
        return Err(invalid("chat messages must be nonempty"));
    }
    for value in messages.iter().chain(tools) {
        validate_json(value)?;
    }
    for message in messages {
        if message
            .get("tool_calls")
            .is_some_and(|v| !v.is_null() && !v.is_array())
        {
            return Err(invalid("tool_calls must be an array"));
        }
        if !matches!(
            message["role"].as_str(),
            Some("system" | "user" | "assistant" | "tool")
        ) {
            return Err(invalid("unsupported chat role"));
        }
        content(
            message,
            message["role"] == "assistant"
                && message["tool_calls"]
                    .as_array()
                    .is_some_and(|a| !a.is_empty()),
        )?;
    }
    let mut text = String::from("<|im_start|>system\n");
    if messages[0]["role"] == "system" {
        text.push_str(content(&messages[0], false)?);
    } else {
        text.push_str(DEFAULT_SYSTEM);
    }
    if !tools.is_empty() {
        text.push_str("\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
        for tool in tools {
            text.push('\n');
            text.push_str(&python_json(tool));
        }
        text.push_str("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n");
    } else {
        text.push_str("<|im_end|>\n");
    }
    for (index, message) in messages.iter().enumerate() {
        let role = message["role"].as_str().unwrap();
        if role == "system" && index == 0 {
            continue;
        }
        let calls = message.get("tool_calls").and_then(Value::as_array);
        if role == "tool" {
            if index == 0 || messages[index - 1]["role"] != "tool" {
                text.push_str("<|im_start|>user");
            }
            text.push_str("\n<tool_response>\n");
            text.push_str(content(message, false)?);
            text.push_str("\n</tool_response>");
            if index + 1 == messages.len() || messages[index + 1]["role"] != "tool" {
                text.push_str("<|im_end|>\n");
            }
        } else if role == "assistant" && calls.is_some_and(|a| !a.is_empty()) {
            text.push_str("<|im_start|>assistant");
            let c = content(message, true)?;
            if !c.is_empty() {
                text.push('\n');
                text.push_str(c);
            }
            for call in calls.unwrap() {
                let function = call.get("function").unwrap_or(call);
                let name = function["name"]
                    .as_str()
                    .ok_or_else(|| invalid("tool call name must be text"))?;
                let arguments = function
                    .get("arguments")
                    .ok_or_else(|| invalid("missing tool call arguments"))?;
                text.push_str("\n<tool_call>\n{\"name\": \"");
                text.push_str(name);
                text.push_str("\", \"arguments\": ");
                text.push_str(&python_json(arguments));
                text.push_str("}\n</tool_call>");
            }
            text.push_str("<|im_end|>\n");
        } else {
            text.push_str("<|im_start|>");
            text.push_str(role);
            text.push('\n');
            text.push_str(content(message, false)?);
            text.push_str("<|im_end|>\n");
        }
    }
    if generation_prompt {
        text.push_str("<|im_start|>assistant\n");
    }
    if text.len() > MAX_TEXT_BYTES {
        return Err(invalid("rendered chat exceeds 256 KiB"));
    }
    Ok(text)
}

// Match Transformers' json.dumps(ensure_ascii=False), insertion order and default
// separator spacing. Python float repr uses scientific notation below 1e-4 or
// at/above 1e16 and a signed, at-least-two-digit exponent.
fn validate_json(value: &Value) -> io::Result<()> {
    match value {
        Value::Number(n) if n.to_string().contains(['.', 'e', 'E']) => {
            if !n.as_f64().is_some_and(f64::is_finite) {
                return Err(invalid("tool JSON floats must be finite"));
            }
        }
        Value::Array(values) => {
            for value in values {
                validate_json(value)?;
            }
        }
        Value::Object(values) => {
            for value in values.values() {
                validate_json(value)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn python_float(value: f64) -> String {
    let encoded = serde_json::to_string(&value).unwrap();
    let (negative, body) = encoded
        .strip_prefix('-')
        .map_or((false, encoded.as_str()), |s| (true, s));
    let (mantissa, exponent) = body
        .split_once('e')
        .map_or((body, 0), |(m, e)| (m, e.parse::<i32>().unwrap()));
    let dot = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
    let digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    let first = digits.find(|c: char| c != '0');
    let Some(first) = first else {
        return if negative { "-0.0" } else { "0.0" }.into();
    };
    let order = dot + exponent - first as i32 - 1;
    let significant = digits[first..].trim_end_matches('0');
    let mut output = if negative { "-" } else { "" }.to_string();
    if !(-4..16).contains(&order) {
        output.push_str(&significant[..1]);
        if significant.len() > 1 {
            output.push('.');
            output.push_str(&significant[1..]);
        }
        output.push('e');
        output.push(if order < 0 { '-' } else { '+' });
        output.push_str(&format!("{:02}", order.abs()));
    } else {
        let point = order + 1;
        if point <= 0 {
            output.push_str("0.");
            output.push_str(&"0".repeat((-point) as usize));
            output.push_str(significant);
        } else if point as usize >= significant.len() {
            output.push_str(significant);
            output.push_str(&"0".repeat(point as usize - significant.len()));
            output.push_str(".0");
        } else {
            output.push_str(&significant[..point as usize]);
            output.push('.');
            output.push_str(&significant[point as usize..]);
        }
    }
    output
}

fn python_json(value: &Value) -> String {
    match value {
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_json)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(k, v)| format!("{}: {}", serde_json::to_string(k).unwrap(), python_json(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Number(n) if n.to_string().contains(['.', 'e', 'E']) => {
            python_float(n.as_f64().unwrap())
        }
        _ => serde_json::to_string(value).unwrap(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/chat-v1")
    }
    fn manifest() -> Value {
        json_file(&root().join("manifest.json"), MAX_INPUT_BYTES).unwrap()
    }
    #[test]
    fn frozen_transformers_chat_and_raw_ids_match_without_model_downloads() {
        let m = manifest();
        for (name, item) in m["files"].as_object().unwrap() {
            let bytes = std::fs::read(root().join(name)).unwrap();
            assert_eq!(bytes.len() as u64, item["bytes"].as_u64().unwrap());
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                item["sha256"].as_str().unwrap()
            );
        }
        let assets = Assets::load(&root(), 151936).unwrap();
        assert!(assets.admitted_bytes > 0);
        assert_eq!(stop_ids(&root(), 151936).unwrap(), [151645, 151643]);
        for case in m["chat_cases"].as_array().unwrap() {
            let text = render(
                case["messages"].as_array().unwrap(),
                case["tools"].as_array().unwrap(),
                case["add_generation_prompt"].as_bool().unwrap(),
            )
            .unwrap();
            assert_eq!(text, case["rendered"].as_str().unwrap());
            assert_eq!(
                serde_json::to_value(assets.encode(&text).unwrap()).unwrap(),
                case["ids"]
            );
        }
        for case in m["raw_cases"].as_array().unwrap() {
            let ids = assets.encode(case["text"].as_str().unwrap()).unwrap();
            assert_eq!(serde_json::to_value(&ids).unwrap(), case["ids"]);
            let mut stream = assets.tokenizer.decode_stream(true);
            let mut sent = String::new();
            for &id in &ids {
                if let Some(text) = stream.step(id).unwrap() {
                    sent.push_str(&text);
                }
            }
            let full = assets.tokenizer.decode(&ids, true).unwrap();
            assert!(full.starts_with(&sent));
            sent.push_str(&full[sent.len()..]);
            assert_eq!(sent, case["decoded"].as_str().unwrap());
            // Every output boundary, including incomplete UTF-8, flushes to
            // the exact batch decode rather than emitting premature replacement.
            for end in 0..=ids.len() {
                let mut stream = assets.tokenizer.decode_stream(true);
                let mut sent = String::new();
                for &id in &ids[..end] {
                    if let Some(text) = stream.step(id).unwrap() {
                        sent.push_str(&text);
                    }
                }
                assert!(
                    assets
                        .tokenizer
                        .decode(&ids[..end], true)
                        .unwrap()
                        .starts_with(&sent)
                );
            }
        }
    }
    #[test]
    fn continued_output_preserves_split_unicode_and_finalizes_once() {
        let assets = Assets::load(&root(), 151936).unwrap();
        let mut decoder = OutputDecoder::default();
        assert_eq!(decoder.step(&assets, 9284).unwrap(), "");
        // Output-limit/cancellation boundaries deliberately do not finish.
        assert_eq!(decoder.step(&assets, 104).unwrap(), "");
        assert_eq!(decoder.step(&assets, 101).unwrap(), "🫨");
        assert_eq!(decoder.finish(&assets).unwrap(), "");
        assert_eq!(decoder.finish(&assets).unwrap(), "");
        assert_eq!(decoder.step(&assets, 9284).unwrap(), "");
        assert_eq!(decoder.finish(&assets).unwrap(), "�");
        for case in manifest()["raw_cases"].as_array().unwrap() {
            let ids: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
            for boundary in 0..=ids.len() {
                let mut decoder = OutputDecoder::default();
                let mut sent = String::new();
                for &id in &ids[..boundary] {
                    sent.push_str(&decoder.step(&assets, id).unwrap());
                }
                for &id in &ids[boundary..] {
                    sent.push_str(&decoder.step(&assets, id).unwrap());
                }
                sent.push_str(&decoder.finish(&assets).unwrap());
                assert_eq!(sent, case["decoded"].as_str().unwrap());
            }
        }
    }
    #[test]
    fn missing_generation_config_uses_model_eos() {
        let directory = std::env::temp_dir().join(format!("toklet-eos-{}", std::process::id()));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("config.json");
        std::fs::write(&path, br#"{"eos_token_id":151645}"#).unwrap();
        assert_eq!(stop_ids(&directory, 151936).unwrap(), [151645]);
        std::fs::write(&path, br#"{"eos_token_id":[151645,151643]}"#).unwrap();
        assert_eq!(stop_ids(&directory, 151936).unwrap(), [151645, 151643]);
        std::fs::write(&path, br#"{"eos_token_id":151936}"#).unwrap();
        assert!(stop_ids(&directory, 151936).is_err());
        std::fs::write(
            directory.join("generation_config.json"),
            br#"{"eos_token_id":4}"#,
        )
        .unwrap();
        assert_eq!(stop_ids(&directory, 151936).unwrap(), [4]);
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn python_tool_json_float_spelling_matches_independent_probes() {
        for case in manifest()["python_float_cases"].as_array().unwrap() {
            let value: Value = serde_json::from_str(case["input"].as_str().unwrap()).unwrap();
            assert_eq!(
                python_json(&value),
                case["expected"].as_str().unwrap(),
                "{}",
                case["input"]
            );
        }
    }
    #[test]
    fn invalid_roles_multimodal_content_and_nonfinite_tools_are_rejected() {
        for messages in [
            vec![],
            vec![serde_json::json!({"role":"other","content":"x"})],
            vec![serde_json::json!({"role":"user","content":[]})],
            vec![serde_json::json!({"role":"assistant","content":"x","tool_calls":"bad"})],
        ] {
            assert!(render(&messages, &[], true).is_err());
        }
        let overflow: Value = serde_json::from_str("1e9999").unwrap();
        assert!(
            render(
                &[serde_json::json!({"role":"user","content":"x"})],
                &[overflow],
                true
            )
            .is_err()
        );
        let assets = Assets::load(&root(), 151936).unwrap();
        assert!(assets.encode(&"x".repeat(MAX_TEXT_BYTES + 1)).is_err());
        assert!(Assets::load(&root(), 64).is_err());
    }
}

#[cfg(test)]
mod real_tests {
    use super::*;
    #[test]
    #[ignore = "requires explicitly downloaded pinned local Qwen tokenizer assets"]
    fn full_pinned_tokenizer_matches_frozen_ids() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let sources = json_file(&root.join("benchmarks/sources.json"), MAX_INPUT_BYTES).unwrap();
        for item in sources["assets"].as_array().unwrap().iter().filter(|v| {
            matches!(
                v["id"].as_str(),
                Some(
                    "qwen.tokenizer.json"
                        | "qwen.tokenizer_config.json"
                        | "qwen.generation_config.json"
                )
            )
        }) {
            let bytes =
                read_bounded(&root.join(item["path"].as_str().unwrap()), 32 * 1024 * 1024).unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                item["sha256"].as_str().unwrap()
            );
        }
        let assets = Assets::load(&root.join("models/Qwen2.5-0.5B-Instruct"), 151936).unwrap();
        let m = json_file(
            &root.join("tests/fixtures/chat-v1/manifest.json"),
            MAX_INPUT_BYTES,
        )
        .unwrap();
        for case in m["raw_cases"].as_array().unwrap() {
            assert_eq!(
                serde_json::to_value(assets.encode(case["text"].as_str().unwrap()).unwrap())
                    .unwrap(),
                case["ids"]
            );
        }
        for case in m["chat_cases"].as_array().unwrap() {
            assert_eq!(
                serde_json::to_value(assets.encode(case["rendered"].as_str().unwrap()).unwrap())
                    .unwrap(),
                case["ids"]
            );
        }
    }
}
