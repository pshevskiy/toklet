//! Frozen raw task inputs; formatting and grading stay outside inference.
use super::artifact::{Result, VOCAB};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};
const MANIFEST: &[u8] = include_bytes!("../../benchmarks/llama3-quality-tasks-v1/manifest.json");
pub const STOPS: [u32; 3] = [128001, 128008, 128009];
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Task {
    pub id: String,
    pub category: String,
    pub prompt_tokens: Vec<u32>,
    pub max_output_tokens: usize,
}
pub fn validate(cases: &[Task]) -> Result<()> {
    if cases.len() != 100 {
        return Err("all100 frozen tasks required".into());
    }
    for (index, case) in cases.iter().enumerate() {
        let (id, category, cap) = if index < 40 {
            (format!("HumanEval/{index}"), "code", 256)
        } else if index < 70 {
            (format!("truthfulqa/row-{}", index - 40), "prose", 32)
        } else {
            (format!("lambada/row-{}", index - 70), "continuation", 16)
        };
        if case.id != id
            || case.category != category
            || case.max_output_tokens != cap
            || case.prompt_tokens.is_empty()
            || case
                .prompt_tokens
                .len()
                .checked_add(cap)
                .is_none_or(|n| n > 566)
            || case.prompt_tokens.iter().any(|&id| id as usize >= VOCAB)
        {
            return Err("frozen per-ID category, reserve or token extent differs".into());
        }
    }
    Ok(())
}
pub fn load(folder: &Path, expected_sha: &str) -> Result<Vec<Task>> {
    if expected_sha != format!("{:x}", Sha256::digest(MANIFEST)) {
        return Err("external task manifest differs".into());
    }
    let mut manifest = Vec::new();
    File::open(folder.join("manifest.json"))?
        .take(MANIFEST.len() as u64 + 1)
        .read_to_end(&mut manifest)?;
    if manifest != MANIFEST {
        return Err("consumed task manifest differs".into());
    }
    let value: serde_json::Value = serde_json::from_slice(&manifest)?;
    if value["sampling"] != "greedy_lowest_token_id_on_ties"
        || value["capacity"] != 566
        || value["stop_ids"] != serde_json::json!(STOPS)
    {
        return Err("task sampling, stops or capacity differs".into());
    }
    let mut body = Vec::new();
    File::open(folder.join("execution.jsonl"))?
        .take(4 * 1024 * 1024 + 1)
        .read_to_end(&mut body)?;
    if body.len() > 4 * 1024 * 1024
        || value["tasks"]["bytes"] != body.len()
        || value["tasks"]["sha256"] != format!("{:x}", Sha256::digest(&body))
    {
        return Err("task execution identity differs".into());
    }
    let cases: Vec<Task> = body
        .split(|&b| b == b'\n')
        .filter(|s| !s.is_empty())
        .map(serde_json::from_slice)
        .collect::<std::result::Result<_, _>>()?;
    validate(&cases)?;
    Ok(cases)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn cases() -> Vec<Task> {
        include_bytes!("../../benchmarks/llama3-quality-tasks-v1/execution.jsonl")
            .split(|&b| b == b'\n')
            .filter(|s| !s.is_empty())
            .map(|s| serde_json::from_slice(s).unwrap())
            .collect()
    }
    #[test]
    fn full_input_keeps_all_reserves_and_566_boundary() {
        let cases = cases();
        validate(&cases).unwrap();
        assert_eq!(
            cases[32].prompt_tokens.len() + cases[32].max_output_tokens,
            566
        );
    }
    #[test]
    fn swapped_categories_invalid_tokens_and_missing_cases_reject() {
        let mut input = cases();
        input.swap(0, 40);
        assert!(validate(&input).is_err());
        let mut input = cases();
        input[0].prompt_tokens[0] = VOCAB as u32;
        assert!(validate(&input).is_err());
        assert!(validate(&cases()[..99]).is_err());
    }
    #[test]
    fn manifest_rejects_before_open() {
        assert!(load(Path::new("/nonexistent-tasks"), "wrong").is_err());
    }
}
