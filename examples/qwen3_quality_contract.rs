//! Independent Rust rendering/tokenizer replay of frozen Qwen3 quality inputs.
//! Loads no model weights and executes no inference or generated code.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::OpenOptions,
    io::{self, Read},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
};
use toklet::app_chat::{self, Codec, Profile};
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn read(path: &Path, limit: usize, pin: Option<&str>) -> io::Result<Vec<u8>> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let before = file.metadata()?;
    let key = |m: &std::fs::Metadata| {
        (
            m.dev(),
            m.ino(),
            m.mode(),
            m.len(),
            m.mtime(),
            m.mtime_nsec(),
            m.ctime(),
            m.ctime_nsec(),
        )
    };
    if !before.is_file()
        || before.len() > limit as u64
        || key(&before) != key(&std::fs::symlink_metadata(path)?)
    {
        return Err(io::Error::other("bounded regular quality input required"));
    }
    let mut body = Vec::new();
    (&file).take(limit as u64 + 1).read_to_end(&mut body)?;
    if body.len() as u64 != before.len()
        || key(&before) != key(&file.metadata()?)
        || key(&before) != key(&std::fs::symlink_metadata(path)?)
        || pin.is_some_and(|p| sha(&body) != p)
    {
        return Err(io::Error::other("quality input identity changed"));
    }
    Ok(body)
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    tokenizers::utils::parallelism::set_parallelism(false);
    let mut args = std::env::args_os().skip(1);
    let directory = PathBuf::from(args.next().ok_or("missing frozen input directory")?);
    let pin = args
        .next()
        .ok_or("missing external manifest SHA")?
        .into_string()
        .map_err(|_| "manifest SHA UTF8")?;
    let model = PathBuf::from(args.next().ok_or("missing pinned model directory")?);
    if args.next().is_some() {
        return Err("extra codec replay argument".into());
    }
    let m: Value = serde_json::from_slice(&read(
        &directory.join("manifest.json"),
        2 * 1024 * 1024,
        Some(&pin),
    )?)?;
    if m["format"] != "toklet-qwen3-quality-contract-v1"
        || m["loss_targets"] != 10000
        || m["full_suite_capacity"] != 1025
    {
        return Err("complete Qwen3 contract required".into());
    }
    let asset = |name: &str| -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let item = &m["files"][name];
        let pin = item["sha256"].as_str().ok_or("asset SHA absent")?;
        let body = read(&directory.join(name), 4 * 1024 * 1024, Some(pin))?;
        if Some(body.len() as u64) != item["bytes"].as_u64() {
            return Err("asset extent differs".into());
        }
        Ok(body)
    };
    let tokenizer = tokenizers::Tokenizer::from_bytes(read(
        &model.join("tokenizer.json"),
        16 * 1024 * 1024,
        Some("aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    )?)
    .map_err(io::Error::other)?;
    if tokenizer.get_truncation().is_some() || tokenizer.get_padding().is_some() {
        return Err("unclipped tokenizer required".into());
    }
    let codec = Codec::load_profile(&model, Profile::Qwen3)?;
    if codec.stop_ids() != [151645] {
        return Err("Qwen3 EOS differs".into());
    }
    let loss_bytes = asset("held_out_loss.u32le")?;
    if loss_bytes.len() != 40004 {
        return Err("all10000 loss targets required".into());
    }
    let ids: Vec<u32> = loss_bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let prefix = String::from_utf8(asset("loss_verification_prefix.txt")?)?;
    let encoded = tokenizer.encode(prefix, false).map_err(io::Error::other)?;
    if encoded.get_ids().get(..10001) != Some(ids.as_slice()) {
        return Err("raw loss tokenizer differs".into());
    }
    let tasks = asset("tasks.jsonl")?;
    let mut checked = 0;
    let mut app_checked = 0;
    for line in tasks.split(|b| *b == b'\n').filter(|b| !b.is_empty()) {
        let task: Value = serde_json::from_slice(line)?;
        let tokens: Vec<u32> = serde_json::from_value(task["prompt_tokens"].clone())?;
        let prompt = task["prompt_text"].as_str().ok_or("prompt text absent")?;
        if !task["messages"].is_null() {
            let messages = task["messages"].as_array().ok_or("messages absent")?;
            if messages.len() != 1 || messages[0]["role"] != "user" {
                return Err("plain single-user quality task required".into());
            }
            let content = messages[0]["content"]
                .as_str()
                .ok_or("user content absent")?;
            let manual = format!(
                "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            );
            if manual != prompt {
                return Err("independent Rust Qwen3 rendering differs".into());
            }
            let encoded_messages = serde_json::to_vec(messages)?;
            if tokens.len() <= app_chat::MAX_TOKENS {
                if codec.encode(&encoded_messages)? != tokens {
                    return Err("app codec IDs differ".into());
                }
                app_checked += 1;
            } else if codec.encode(&encoded_messages).is_ok() {
                return Err("app codec unexpectedly enlarged its window".into());
            }
        }
        if tokenizer
            .encode(prompt, false)
            .map_err(io::Error::other)?
            .get_ids()
            != tokens
        {
            return Err("complete prompt tokenization differs".into());
        }
        let cap = task["max_output_tokens"]
            .as_u64()
            .ok_or("output cap absent")? as usize;
        if tokens.is_empty() || tokens.len().checked_add(cap).is_none_or(|n| n > 1025) {
            return Err("complete shared quality budget differs".into());
        }
        checked += 1;
    }
    if checked != 100 {
        return Err("all100 complete tasks required".into());
    }
    println!(
        "{}",
        json!({"passed":true,"tasks":checked,"loss_targets":10000,"app_chat_codec_cases":app_checked,
        "raw_and_rendered_tokenizer_replayed":true,"quality_scored":false,"generated_code_executed":false})
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
