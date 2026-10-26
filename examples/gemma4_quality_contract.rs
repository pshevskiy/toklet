//! Replay frozen Gemma quality inputs through the Rust codec; no inference/grading.
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    error::Error,
    fs::{Metadata, OpenOptions},
    io::{self, Read},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
};
use toklet::app_chat::gemma4::{self, Codec};

const DATA_ALLOWANCE: usize = 32 * 1024 * 1024;
fn invalid(message: &str) -> io::Error {
    io::Error::other(message)
}
fn sha(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}
fn identity(m: &Metadata) -> (u64, u64, u64, i64, i64, i64, i64) {
    (
        m.dev(),
        m.ino(),
        m.len(),
        m.mtime(),
        m.mtime_nsec(),
        m.ctime(),
        m.ctime_nsec(),
    )
}
fn read(path: &Path, maximum: usize, expected: Option<&str>) -> io::Result<Vec<u8>> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let before = file.metadata()?;
    if !before.is_file()
        || before.len() > maximum as u64
        || identity(&before) != identity(&std::fs::symlink_metadata(path)?)
    {
        return Err(invalid("regular bounded contract input required"));
    }
    let mut raw = Vec::new();
    (&file).take(maximum as u64 + 1).read_to_end(&mut raw)?;
    if raw.len() as u64 != before.len()
        || identity(&before) != identity(&file.metadata()?)
        || identity(&before) != identity(&std::fs::symlink_metadata(path)?)
        || expected.is_some_and(|value| sha(&raw) != value)
    {
        return Err(invalid("contract identity changed"));
    }
    Ok(raw)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Asset {
    file: String,
    bytes: usize,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Summary {
    id: String,
    category: String,
    prompt_tokens: usize,
    max_output_tokens: usize,
    required_capacity: usize,
    payload_sha256: String,
}
#[derive(Deserialize, PartialEq, Debug)]
#[serde(deny_unknown_fields)]
struct Block {
    block: usize,
    start: usize,
    end: usize,
    targets: usize,
    input_start: usize,
    input_end: usize,
    reset_position: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Capacity {
    loss_context_required: usize,
    maximum_task_required: usize,
    full_suite_required: usize,
    current_engine_implemented_positions: usize,
    current_engine_satisfies_contract: bool,
    no_workload_truncation: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Policy {
    scored_loss_tokens: usize,
    loss_block_targets: usize,
    loss_context_policy: String,
    mean_nll_increase_limit_nats_per_token: f64,
    task_pass_rate_loss_limit_percentage_points: f64,
    case_policy: String,
    categories: BTreeMap<String, usize>,
    task_ids: Vec<String>,
    aggregation: String,
    calibration: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format: String,
    assets: BTreeMap<String, Asset>,
    policy: Policy,
    blocks: Vec<Block>,
    cases: Vec<Summary>,
    capacity: Capacity,
    stop_ids: Vec<u32>,
    quality_scored: bool,
    generated_code_executed: bool,
    app_eligibility_changed: bool,
    full_model_window_qualified: bool,
    // Provenance is authenticated by the external manifest digest and the
    // separate source-data replay; this diagnostic checks codec/position inputs.
    #[serde(rename = "candidate")]
    _candidate: Value,
    #[serde(rename = "unquantized_baseline")]
    _unquantized_baseline: Value,
    #[serde(rename = "source_parity_reference")]
    _source_parity_reference: Value,
    #[serde(rename = "dataset_sources")]
    _dataset_sources: Value,
    #[serde(rename = "model_assets")]
    _model_assets: Value,
    #[serde(rename = "tokenization")]
    _tokenization: Value,
    #[serde(rename = "scoring")]
    _scoring: Value,
    #[serde(rename = "preparation_versions")]
    _preparation_versions: Value,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Message {
    role: String,
    content: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Task {
    id: String,
    category: String,
    source_row: usize,
    max_output_tokens: usize,
    messages: Option<Vec<Message>>,
    prompt_text: String,
    prompt_tokens: Vec<u32>,
    required_capacity: usize,
    stop_ids: Vec<u32>,
    code_prefix: Option<String>,
    tests: Option<String>,
    entry_point: Option<String>,
    expected: Option<String>,
    option_rotation: Option<usize>,
    options: Option<Vec<String>>,
}
fn expected_ids() -> Vec<String> {
    (0..40)
        .map(|i| format!("HumanEval/{i}"))
        .chain((0..30).map(|i| format!("truthfulqa/row-{i}")))
        .chain((0..30).map(|i| format!("lambada/row-{i}")))
        .collect()
}
fn validate_blocks(blocks: &[Block]) -> io::Result<()> {
    let expected: Vec<_> = (0..10000)
        .step_by(1024)
        .enumerate()
        .map(|(block, start)| {
            let end = (start + 1024).min(10000);
            Block {
                block,
                start,
                end,
                targets: end - start,
                input_start: start,
                input_end: end + 1,
                reset_position: 0,
            }
        })
        .collect();
    if blocks != expected {
        return Err(invalid("complete ten-block loss coverage required"));
    }
    Ok(())
}
fn validate_policy(m: &Manifest) -> io::Result<()> {
    let p = &m.policy;
    if m.format != "toklet-gemma-quality-contract-v1"
        || m.quality_scored
        || m.generated_code_executed
        || m.app_eligibility_changed
        || m.full_model_window_qualified
        || p.scored_loss_tokens != 10000
        || p.loss_block_targets != 1024
        || p.loss_context_policy
            != "independent_blocks_of_up_to_1025_tokens_one_token_overlap_score_every_target_once"
        || p.mean_nll_increase_limit_nats_per_token != 0.05
        || p.task_pass_rate_loss_limit_percentage_points != 2.0
        || p.case_policy != "first_source_rows_in_order_no_model_based_selection"
        || p.aggregation != "100_equal_weight_binary_cases"
        || p.calibration != "held_out_sources_and_all_selected_cases_forbidden_for_quantizer_tuning"
        || p.task_ids != expected_ids()
        || p.categories
            != BTreeMap::from([
                ("code".into(), 40),
                ("prose".into(), 30),
                ("continuation".into(), 30),
            ])
        || m.stop_ids != gemma4::STOP_IDS
        || m.cases.len() != 100
    {
        return Err(invalid("frozen quality policy differs"));
    }
    validate_blocks(&m.blocks)
}
fn asset(root: &Path, m: &Manifest, name: &str, maximum: usize) -> io::Result<Vec<u8>> {
    let a = m
        .assets
        .get(name)
        .ok_or_else(|| invalid("contract asset missing"))?;
    if a.file != name || a.bytes > maximum {
        return Err(invalid("contract asset bound/path differs"));
    }
    let raw = read(&root.join(name), maximum, Some(&a.sha256))?;
    if raw.len() != a.bytes {
        return Err(invalid("contract asset size differs"));
    }
    Ok(raw)
}
fn validate_task(
    t: &Task,
    s: &Summary,
    index: usize,
    expected_id: &str,
    raw: &[u8],
) -> io::Result<()> {
    let (category, local, cap) = if index < 40 {
        ("code", index, 256)
    } else if index < 70 {
        ("prose", index - 40, 32)
    } else {
        ("continuation", index - 70, 16)
    };
    let required = t
        .prompt_tokens
        .len()
        .checked_add(cap)
        .ok_or_else(|| invalid("capacity overflow"))?;
    if t.id != expected_id
        || t.category != category
        || t.source_row != local
        || t.max_output_tokens != cap
        || t.stop_ids != gemma4::STOP_IDS
        || t.prompt_tokens.is_empty()
        || t.prompt_tokens.iter().any(|&i| i as usize >= gemma4::VOCAB)
        || t.required_capacity != required
        || required > 131072
        || s.id != t.id
        || s.category != category
        || s.prompt_tokens != t.prompt_tokens.len()
        || s.max_output_tokens != cap
        || s.required_capacity != required
        || s.payload_sha256 != sha(raw)
    {
        return Err(invalid(
            "task identity, requested capacity or output cap differs",
        ));
    }
    let shape = match category {
        "code" => {
            t.code_prefix.is_some()
                && t.tests.is_some()
                && t.entry_point.is_some()
                && t.expected.is_none()
                && t.options.is_none()
                && t.option_rotation.is_none()
        }
        "prose" => {
            t.expected.is_some()
                && t.options.is_some()
                && t.option_rotation.is_some()
                && t.code_prefix.is_none()
                && t.tests.is_none()
                && t.entry_point.is_none()
        }
        _ => {
            t.expected.is_some()
                && t.code_prefix.is_none()
                && t.tests.is_none()
                && t.entry_point.is_none()
                && t.options.is_none()
                && t.option_rotation.is_none()
        }
    };
    if !shape {
        return Err(invalid("task scorer inputs differ"));
    }
    Ok(())
}
fn execute(model: &Path, root: &Path, expected: &str) -> Result<Value, Box<dyn Error>> {
    let raw = read(&root.join("manifest.json"), 256 * 1024, Some(expected))?;
    let m: Manifest = serde_json::from_slice(&raw)?;
    validate_policy(&m)?;
    let loss = asset(root, &m, "held_out_loss.u32le", 40004)?;
    if loss.len() != 40004 {
        return Err(invalid("all10001 loss IDs required").into());
    }
    let loss: Vec<u32> = loss
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    if loss.iter().any(|&v| v as usize >= gemma4::VOCAB) {
        return Err(invalid("loss ID outside vocabulary").into());
    }
    let prefix = asset(
        root,
        &m,
        "loss_verification_prefix.txt",
        gemma4::MAX_TEXT_BYTES,
    )?;
    let tasks = asset(root, &m, "tasks.jsonl", 4 * 1024 * 1024)?;
    if m.assets.len() != 3 || !tasks.ends_with(b"\n") {
        return Err(invalid("complete exact contract assets required").into());
    }
    let codec = Codec::load(model, gemma4::ALLOWANCE_BYTES)?;
    if codec.stop_ids() != gemma4::STOP_IDS {
        return Err(invalid("model stop IDs differ").into());
    }
    let encoded = codec.encode_raw(std::str::from_utf8(&prefix)?, false)?;
    if encoded.get(..10001) != Some(loss.as_slice()) {
        return Err(invalid("Rust loss tokenization differs").into());
    }
    let mut count = 0;
    let mut maximum = 0;
    for line in tasks.split_inclusive(|&v| v == b'\n') {
        if count >= 100 || line.len() > 64 * 1024 {
            return Err(invalid("task count/record bound differs").into());
        }
        let t: Task = serde_json::from_slice(line)?;
        validate_task(&t, &m.cases[count], count, &m.policy.task_ids[count], line)?;
        let rendered = if let Some(messages) = &t.messages {
            if t.category == "continuation" || messages.len() != 1 || messages[0].role != "user" {
                return Err(invalid("quality conversation scope differs").into());
            }
            gemma4::render(
                &serde_json::to_vec(messages)?,
                gemma4::RenderOptions {
                    enable_thinking: false,
                    add_generation_prompt: true,
                },
            )?
        } else {
            if t.category != "continuation" {
                return Err(invalid("chat messages required").into());
            }
            t.prompt_text.clone()
        };
        if rendered != t.prompt_text || codec.encode_raw(&rendered, false)? != t.prompt_tokens {
            return Err(invalid("Rust prompt rendering/tokenization differs").into());
        }
        maximum = maximum.max(t.required_capacity);
        count += 1;
    }
    let c = &m.capacity;
    if count != 100
        || c.loss_context_required != 1025
        || c.maximum_task_required != maximum
        || c.full_suite_required != maximum.max(1025)
        || c.current_engine_implemented_positions != 32
        || c.current_engine_satisfies_contract
        || !c.no_workload_truncation
    {
        return Err(invalid("complete workload capacity summary differs").into());
    }
    Ok(
        json!({"passed":true,"manifest_sha256":expected,"tasks_compared":count,"loss_tokens_compared":10001,"loss_context_required":1025,"maximum_task_required":maximum,
        "full_suite_required":maximum.max(1025),"codec_policy_bytes":gemma4::ALLOWANCE_BYTES,"diagnostic_policy_bytes":DATA_ALLOWANCE,
        "inference_executed":false,"quality_scored":false,"generated_code_executed":false,"app_eligibility_changed":false,"full_model_window_qualified":false}),
    )
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: gemma4_quality_contract MODEL_DIRECTORY CONTRACT_DIRECTORY EXPECTED_MANIFEST_SHA256".into());
    }
    let expected = args[2].to_str().ok_or("UTF8 manifest digest required")?;
    let result = execute(&PathBuf::from(&args[0]), &PathBuf::from(&args[1]), expected)?;
    println!("{result}");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn case() -> (Task, Summary) {
        let t = Task {
            id: "HumanEval/0".into(),
            category: "code".into(),
            source_row: 0,
            max_output_tokens: 256,
            messages: Some(vec![Message {
                role: "user".into(),
                content: "Synthetic only".into(),
            }]),
            prompt_text: "synthetic".into(),
            prompt_tokens: vec![2, 105, 107],
            required_capacity: 259,
            stop_ids: vec![1, 106, 50],
            code_prefix: Some("def f():".into()),
            tests: Some("assert f()".into()),
            entry_point: Some("f".into()),
            expected: None,
            option_rotation: None,
            options: None,
        };
        let s = Summary {
            id: t.id.clone(),
            category: t.category.clone(),
            prompt_tokens: 3,
            max_output_tokens: 256,
            required_capacity: 259,
            payload_sha256: sha(b"synthetic\n"),
        };
        (t, s)
    }
    #[test]
    fn full_caps_and_requested_capacity_must_match() {
        let (mut t, s) = case();
        validate_task(&t, &s, 0, "HumanEval/0", b"synthetic\n").unwrap();
        t.max_output_tokens = 255;
        assert!(validate_task(&t, &s, 0, "HumanEval/0", b"synthetic\n").is_err());
        t.max_output_tokens = 256;
        t.required_capacity = 258;
        assert!(validate_task(&t, &s, 0, "HumanEval/0", b"synthetic\n").is_err());
    }
    #[test]
    fn original_case_identity_and_vocabulary_are_checked() {
        let (mut t, s) = case();
        t.id = "HumanEval/1".into();
        assert!(validate_task(&t, &s, 0, "HumanEval/0", b"synthetic\n").is_err());
        t.id = "HumanEval/0".into();
        t.prompt_tokens[1] = 262144;
        assert!(validate_task(&t, &s, 0, "HumanEval/0", b"synthetic\n").is_err());
    }
    #[test]
    fn all_loss_targets_once_with_one_token_overlap() {
        let blocks: Vec<_> = (0..10000)
            .step_by(1024)
            .enumerate()
            .map(|(block, start)| {
                let end = (start + 1024).min(10000);
                Block {
                    block,
                    start,
                    end,
                    targets: end - start,
                    input_start: start,
                    input_end: end + 1,
                    reset_position: 0,
                }
            })
            .collect();
        validate_blocks(&blocks).unwrap();
        assert_eq!(blocks.iter().map(|v| v.targets).sum::<usize>(), 10000);
        let mut bad = blocks;
        bad[1].input_start += 1;
        assert!(validate_blocks(&bad).is_err());
    }
    #[test]
    fn missing_tail_or_reset_position_rejects() {
        assert!(validate_blocks(&[]).is_err());
        let block = Block {
            block: 0,
            start: 0,
            end: 10000,
            targets: 10000,
            input_start: 0,
            input_end: 10001,
            reset_position: 1,
        };
        assert!(validate_blocks(&[block]).is_err());
    }
    #[test]
    fn typed_case_rejects_duplicate_and_unknown_fields() {
        assert!(
            serde_json::from_str::<Task>(r#"{"id":"a","id":"b"}"#)
                .err()
                .unwrap()
                .to_string()
                .contains("duplicate field")
        );
        assert!(
            serde_json::from_str::<Task>(r#"{"id":"a","extra":1}"#)
                .err()
                .unwrap()
                .to_string()
                .contains("unknown field")
        );
        assert!(serde_json::from_str::<Block>(r#"{"block":true,"start":0,"end":1,"targets":1,"input_start":0,"input_end":2,"reset_position":0}"#).is_err());
    }
}
