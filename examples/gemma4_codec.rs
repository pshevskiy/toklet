//! Explicit pinned text-only parity probe; no model inference or app capability.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{error::Error, fs::OpenOptions, io::Write, path::PathBuf};
use toklet::app_chat::gemma4::{self, Codec, RenderOptions};

const WITNESS: &str = include_str!("../tests/fixtures/gemma4-text-codec-v1/witness.jsonl");
fn ids(value: &Value) -> Result<Vec<u32>, Box<dyn Error>> {
    Ok(serde_json::from_value(value.clone())?)
}
fn prefixes(codec: &Codec, row: &Value) -> Result<Value, Box<dyn Error>> {
    let tokens = ids(&row["ids"])?;
    let mut result = serde_json::Map::new();
    for (chunk, observations) in row["prefix_decodes"]
        .as_object()
        .ok_or("prefix map absent")?
    {
        let mut values = Vec::new();
        for observation in observations.as_array().ok_or("prefix array absent")? {
            let count = observation["tokens"]
                .as_u64()
                .ok_or("prefix count absent")? as usize;
            values.push(json!({"tokens":count,"text":codec.decode(tokens.get(..count).ok_or("prefix outside IDs")?,true)?}));
        }
        result.insert(chunk.clone(), values.into());
    }
    Ok(result.into())
}
fn observe(codec: &Codec, row: &Value) -> Result<Value, Box<dyn Error>> {
    let kind = row["kind"].as_str().ok_or("record kind absent")?;
    Ok(match kind {
        "structure" => {
            json!({"kind":kind,"vocab":gemma4::VOCAB,"allowance":codec.allowance_bytes()})
        }
        "chat" => {
            let messages = serde_json::to_vec(&row["messages"])?;
            let options: RenderOptions = serde_json::from_value(row["options"].clone())?;
            let text = gemma4::render(&messages, options)?;
            let tokens = codec.encode_chat(&messages, options)?;
            json!({"kind":kind,"rendered":text,"ids":tokens,
                "add_special_tokens_ids":codec.encode_raw(&text,true)?,
                "decode_keep_special":codec.decode(&tokens,false)?,"decode_skip_special":codec.decode(&tokens,true)?,
                "prefix_decodes":prefixes(codec,row)?})
        }
        "text" => {
            let input = row["input"].as_str().ok_or("text input absent")?;
            let tokens = codec.encode_raw(input, false)?;
            json!({"kind":kind,"ids":tokens,"add_special_tokens_ids":codec.encode_raw(input,true)?,
                "normalized_ids":codec.encode_raw(row["normalized"].as_str().ok_or("normalized text absent")?,false)?,
                "decode":codec.decode(&tokens,true)?,"prefix_decodes":prefixes(codec,row)?})
        }
        "control" => {
            let tokens = codec.encode_raw(
                row["metadata"]["content"]
                    .as_str()
                    .ok_or("control absent")?,
                false,
            )?;
            json!({"kind":kind,"ids":tokens,"decode_keep":codec.decode(&tokens,false)?,"decode_skip":codec.decode(&tokens,true)?})
        }
        "fallback" => {
            json!({"kind":kind,"decode":codec.decode(&ids(&row["ids"])?,true)?,"prefix_decodes":prefixes(codec,row)?})
        }
        "stop_policy" => json!({"kind":kind,"selected_artifact_stops":codec.stop_ids()}),
        "rejection" => {
            let options = serde_json::from_value::<RenderOptions>(row["options"].clone());
            let result = options
                .map_err(std::io::Error::other)
                .and_then(|v| gemma4::render(&serde_json::to_vec(&row["messages"]).unwrap(), v));
            json!({"kind":kind,"rejected":result.is_err(),"error":result.err().map(|e|e.to_string())})
        }
        "upstream_trim_diagnostic" => {
            // Exclude unsupported thinking/control diagnostics; no production
            // thinking-channel stripper is implemented for this basic profile.
            let result = gemma4::render(
                &serde_json::to_vec(&json!([{"role":"user","content":row["input"]}]))?,
                RenderOptions {
                    enable_thinking: false,
                    add_generation_prompt: true,
                },
            );
            json!({"kind":kind,"admitted_basic_subset":result.is_ok(),"rendered":result.ok()})
        }
        _ => return Err("unexpected fixed witness kind".into()),
    })
}
fn check(expected: &Value, actual: &Value) -> Result<(), Box<dyn Error>> {
    let kind = expected["kind"].as_str().ok_or("record kind absent")?;
    let fields: &[&str] = match kind {
        "structure" => &["vocab"],
        "chat" => &[
            "rendered",
            "ids",
            "add_special_tokens_ids",
            "decode_keep_special",
            "decode_skip_special",
            "prefix_decodes",
        ],
        "text" => &["ids", "add_special_tokens_ids", "decode", "prefix_decodes"],
        "control" => &["ids", "decode_keep", "decode_skip"],
        "fallback" => &["decode", "prefix_decodes"],
        "stop_policy" => &["selected_artifact_stops"],
        "rejection" => &["rejected"],
        "upstream_trim_diagnostic" => &["admitted_basic_subset"],
        _ => return Err("unexpected fixed witness kind".into()),
    };
    for field in fields {
        if expected[field] != actual[field] {
            return Err(format!("{kind} differs at {field}").into());
        }
    }
    if kind == "text" && actual["normalized_ids"] != expected["ids"] {
        return Err("normalized text encoding differs".into());
    }
    if kind == "upstream_trim_diagnostic" && actual["admitted_basic_subset"] == true {
        let text = actual["rendered"]
            .as_str()
            .ok_or("admitted diagnostic has no prompt")?;
        let expected_text = format!(
            "<bos><|turn>user\n{}<turn|>\n<|turn>model\n",
            expected["strip_thinking"]
                .as_str()
                .ok_or("plain trim absent")?
        );
        if text != expected_text {
            return Err("plain trimming differs".into());
        }
    }
    Ok(())
}
fn run(root: PathBuf, output: PathBuf) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir(&output)?;
    let mut observations = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join("actual.jsonl"))?;
    let mut report = json!({"format":"toklet-gemma4-native-codec-v1","passed":false,"primary_error":null,
        "completed_records":0,"native_model_executed":false,"native_app_profile_admitted":false,
        "append_only_streaming_qualified":false,"app_eligibility_changed":false,
        "codec_allowance_bytes":gemma4::ALLOWANCE_BYTES,"text_bytes":gemma4::MAX_TEXT_BYTES,
        "codec_token_bound":gemma4::MAX_CODEC_TOKENS,"numerical_positions_unchanged":32});
    let result = (|| -> Result<(), Box<dyn Error>> {
        if format!("{:x}", Sha256::digest(WITNESS.as_bytes()))
            != "391077f928d1e288c227bc981f42ae505f7b407f9ee854332f9d2d9c3a410c76"
        {
            return Err("frozen text witness identity differs".into());
        }
        let codec = Codec::load(&root, gemma4::ALLOWANCE_BYTES)?;
        for (index, line) in WITNESS.lines().enumerate() {
            let expected: Value = serde_json::from_str(line)?;
            let observation = observe(&codec, &expected);
            let actual = match &observation {
                Ok(value) => value.clone(),
                Err(e) => json!({"error":e.to_string()}),
            };
            serde_json::to_writer(&mut observations, &json!({"index":index,"actual":actual}))?;
            observations.write_all(b"\n")?;
            observations.flush()?;
            observation?;
            check(&expected, &actual)?;
            report["completed_records"] = (index + 1).into();
        }
        if report["completed_records"] != 97 {
            return Err("incomplete codec record inventory".into());
        }
        Ok(())
    })();
    report["primary_error"] = result.as_ref().err().map(ToString::to_string).into();
    report["passed"] = result.is_ok().into();
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join("observation.json"))?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.flush()?;
    println!(
        "{}",
        json!({"passed":report["passed"],"records":report["completed_records"],"primary_error":report["primary_error"]})
    );
    result
}
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let usage = "usage: gemma4_codec CHECKPOINT_DIRECTORY NEW_OUTPUT_DIRECTORY";
    let root = PathBuf::from(args.next().ok_or(usage)?);
    let output = PathBuf::from(args.next().ok_or(usage)?);
    if args.next().is_some() {
        return Err(usage.into());
    }
    run(root, output)
}
