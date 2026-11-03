//! Frozen corpus/block contract shared by the explicit full-loss worker.
use super::artifact::{Result, VOCAB};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};
const MANIFEST: &[u8] = include_bytes!("../../benchmarks/llama3-original-quality-v1/manifest.json");
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Block {
    pub block: usize,
    pub input_start: usize,
    pub input_end: usize,
    pub score_rows: [usize; 2],
    pub global_target_indices: [usize; 2],
    pub reset_position: usize,
}
#[derive(Debug)]
pub struct Inputs {
    pub ids: Vec<u32>,
    pub blocks: Vec<Block>,
}
fn parse(manifest: &[u8], raw: &[u8]) -> Result<Inputs> {
    let value: serde_json::Value = serde_json::from_slice(manifest)?;
    if raw.len() != 40004
        || value["format"] != "toklet-llama3-original-quality-inputs-v1"
        || value["tokens"]["sha256"] != format!("{:x}", Sha256::digest(raw))
        || value["policy"]["targets"] != 10000
        || value["policy"]["maximum_input_positions"] != 1025
        || value["policy"]["mean_nll_increase_limit"] != 0.05
    {
        return Err("frozen quality inputs differ".into());
    }
    let ids: Vec<u32> = raw
        .chunks_exact(4)
        .map(|p| u32::from_le_bytes(p.try_into().unwrap()))
        .collect();
    if ids.iter().any(|&id| id as usize >= VOCAB) {
        return Err("quality token outside vocabulary".into());
    }
    let blocks: Vec<Block> = serde_json::from_value(value["blocks"].clone())?;
    let mut next = 1;
    for (index, block) in blocks.iter().enumerate() {
        let count = block
            .input_end
            .checked_sub(block.input_start)
            .filter(|n| (2..=1025).contains(n))
            .ok_or("block extent differs")?;
        if block.block != index
            || block.reset_position != 0
            || block.input_end > ids.len()
            || block.score_rows != [0, count - 1]
            || block.global_target_indices != [block.input_start + 1, block.input_end]
            || block.global_target_indices[0] != next
        {
            return Err("block target coverage differs".into());
        }
        next = block.global_target_indices[1];
    }
    if blocks.len() != 10 || next != 10001 {
        return Err("loss target coverage is incomplete".into());
    }
    Ok(Inputs { ids, blocks })
}
pub fn load(folder: &Path, expected_sha: &str) -> Result<Inputs> {
    if expected_sha != format!("{:x}", Sha256::digest(MANIFEST)) {
        return Err("external frozen manifest differs".into());
    }
    let mut manifest = Vec::new();
    File::open(folder.join("manifest.json"))?
        .take(MANIFEST.len() as u64 + 1)
        .read_to_end(&mut manifest)?;
    if manifest != MANIFEST {
        return Err("consumed frozen manifest differs".into());
    }
    let mut raw = Vec::new();
    File::open(folder.join("tokens.u32le"))?
        .take(40005)
        .read_to_end(&mut raw)?;
    parse(&manifest, &raw)
}
#[cfg(test)]
mod tests {
    use super::*;
    const TOKENS: &[u8] =
        include_bytes!("../../benchmarks/llama3-original-quality-v1/tokens.u32le");
    #[test]
    fn frozen_blocks_cover_every_target_once_and_fit() {
        let input = parse(MANIFEST, TOKENS).unwrap();
        let indices: Vec<_> = input
            .blocks
            .iter()
            .flat_map(|b| b.global_target_indices[0]..b.global_target_indices[1])
            .collect();
        assert_eq!(indices, (1..=10000).collect::<Vec<_>>());
        assert_eq!(input.ids.len(), 10001);
    }
    #[test]
    fn rejects_corruption_out_of_vocabulary_and_broken_coverage() {
        let mut bytes = TOKENS.to_vec();
        bytes[0] ^= 1;
        assert!(parse(MANIFEST, &bytes).is_err());
        bytes = TOKENS.to_vec();
        bytes[..4].copy_from_slice(&(VOCAB as u32).to_le_bytes());
        let mut value: serde_json::Value = serde_json::from_slice(MANIFEST).unwrap();
        value["tokens"]["sha256"] = format!("{:x}", Sha256::digest(&bytes)).into();
        assert!(parse(&serde_json::to_vec(&value).unwrap(), &bytes).is_err());
        for block in [0, 9] {
            let mut value: serde_json::Value = serde_json::from_slice(MANIFEST).unwrap();
            value["blocks"][block]["global_target_indices"][1] = 0.into();
            assert!(parse(&serde_json::to_vec(&value).unwrap(), TOKENS).is_err());
        }
        assert!(parse(MANIFEST, &TOKENS[..40000]).is_err());
    }
    #[test]
    fn manifest_pin_rejects_before_opening() {
        assert!(
            load(Path::new("/nonexistent-quality-input"), "wrong")
                .unwrap_err()
                .to_string()
                .contains("external frozen manifest differs")
        );
    }
}
