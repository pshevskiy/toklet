//! Collect the frozen LFM3 held-out loss rows with the authenticated BF16 graph.
#[path = "support/nll.rs"]
mod nll;

use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};

use half::bf16;
use serde_json::json;
use sha2::{Digest, Sha256};
use toklet::{
    cpu::lfm3_linear_bf16,
    model::{
        Error,
        lfm3::{HIDDEN as WIDTH, VOCAB as VOCABULARY},
    },
    weights::{INITIAL_MEMORY_LIMIT_BYTES, load_lfm3},
};

const CHUNK: usize = 128;
const CONTEXT: usize = 2048;
const HEAD_OFFSET: u64 = 32376;
const HEAD_SHA: &str = "5f1df3bbc564554f48a7fb542f96e745c92e2315c92279944d9e65d234bec4bc";
const LOSS_SHA: &str = "6c74034756552082f1ab629d2a720170a667c7aa08eb1815a3b4008b03be339b";
const TARGETS: usize = 10_000;
const BLOCK: usize = 1024;
const READ_CHUNK: usize = 1024 * 1024;

fn read_ids(path: &Path) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut raw = Vec::with_capacity((TARGETS + 1) * 4);
    File::open(path)?
        .take(((TARGETS + 1) * 4 + 1) as u64)
        .read_to_end(&mut raw)?;
    if raw.len() != (TARGETS + 1) * 4 || format!("{:x}", Sha256::digest(&raw)) != LOSS_SHA {
        return Err("held-out token payload differs".into());
    }
    let ids: Vec<_> = raw
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte token")))
        .collect();
    if ids.iter().any(|&id| id as usize >= VOCABULARY) {
        return Err("held-out token is outside the model vocabulary".into());
    }
    Ok(ids)
}

fn read_head(model: &Path) -> Result<Vec<bf16>, Box<dyn std::error::Error>> {
    let mut file = File::open(model.join("model.safetensors"))?;
    file.seek(SeekFrom::Start(HEAD_OFFSET))?;
    let mut bank = Vec::new();
    bank.try_reserve_exact(VOCABULARY * WIDTH)?;
    let mut digest = Sha256::new();
    let mut bytes = vec![0; READ_CHUNK];
    while bank.len() < VOCABULARY * WIDTH {
        let count = ((VOCABULARY * WIDTH - bank.len()) * 2).min(bytes.len());
        file.read_exact(&mut bytes[..count])?;
        digest.update(&bytes[..count]);
        bank.extend(
            bytes[..count]
                .chunks_exact(2)
                .map(|word| bf16::from_bits(u16::from_le_bytes([word[0], word[1]]))),
        );
    }
    if format!("{:x}", digest.finalize()) != HEAD_SHA {
        return Err("tied head differs from authenticated tensor".into());
    }
    Ok(bank)
}

fn score(logits: &[bf16], target: u32, widened: &mut [f32]) -> Result<(usize, f64), &'static str> {
    if widened.len() != logits.len() {
        return Err("loss scratch extent differs");
    }
    for (value, logit) in widened.iter_mut().zip(logits) {
        *value = logit.to_f32();
    }
    let (maximum, _, loss) = nll::statistics(widened, target as usize)?;
    let greedy = widened
        .iter()
        .position(|&value| f64::from(value) == maximum)
        .ok_or("finite loss row has no maximum")?;
    Ok((greedy, loss))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let [model, ids, output, first_block, last_block] = args.as_slice() else {
        return Err("usage: MODEL HELD_OUT_IDS NEW_JSONL FIRST_BLOCK EXCLUSIVE_LAST_BLOCK".into());
    };
    let first: usize = first_block.parse()?;
    let last: usize = last_block.parse()?;
    if first >= last || last > TARGETS.div_ceil(BLOCK) {
        return Err("invalid loss block selection".into());
    }
    let model = Path::new(model);
    let ids = read_ids(Path::new(ids))?;
    let diagnostic_bytes = (VOCABULARY * (WIDTH + CHUNK)) * size_of::<bf16>() + 4 * READ_CHUNK;
    let loaded = load_lfm3(
        model,
        CONTEXT,
        CHUNK,
        INITIAL_MEMORY_LIMIT_BYTES - diagnostic_bytes,
    )?;
    let source_sha256 = loaded.report.source_sha256;
    let load_allowance_bytes = loaded.report.temporary_allowance_bytes;
    let mut model_owner = loaded.model;
    let model_bytes = model_owner.memory().total_bytes;
    let bank = read_head(model)?;
    let mut scores = vec![bf16::ZERO; CHUNK * VOCABULARY];
    let mut widened = vec![0.0; VOCABULARY];
    let mut session = model_owner.session()?;
    let mut out = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?,
    );
    println!(
        "{}",
        json!({"event":"ready","source_sha256":source_sha256,"first_block":first,
            "end_block":last,"model_bytes":model_bytes,"load_allowance_bytes":load_allowance_bytes,
            "diagnostic_bytes":diagnostic_bytes,"context_capacity":CONTEXT,"chunk":CHUNK})
    );
    for block in first..last {
        let start = block * BLOCK;
        let end = (start + BLOCK).min(TARGETS);
        session.reset()?;
        let block_ids = &ids[start..=end];
        let mut seen = 0;
        for chunk in block_ids.chunks(CHUNK) {
            let rows = chunk.len().min(end - start - seen);
            let mut head_seen = false;
            let mut end_verified = false;
            let output = &mut scores[..rows * VOCABULARY];
            let distribution = session.forward_observed(chunk, |event| {
                if event.stage == "final_norm" && rows != 0 {
                    if head_seen || event.values.len() != chunk.len() * WIDTH {
                        return Err(Error::InvalidConfig(
                            "loss final norm extent or count differs",
                        ));
                    }
                    lfm3_linear_bf16(
                        &event.values[..rows * WIDTH],
                        &bank,
                        output,
                        rows,
                        WIDTH,
                        VOCABULARY,
                    )?;
                    head_seen = true;
                }
                if event.stage == "logits" && rows == chunk.len() {
                    end_verified = head_seen
                        && !end_verified
                        && event.values.len() == VOCABULARY
                        && event
                            .values
                            .iter()
                            .zip(&output[(rows - 1) * VOCABULARY..])
                            .all(|(a, b)| a.to_bits() == b.to_bits());
                    if !end_verified {
                        return Err(Error::InvalidConfig("loss head endpoint differs"));
                    }
                }
                Ok(())
            })?;
            if distribution.len() != VOCABULARY
                || (rows != 0 && !head_seen)
                || (rows == chunk.len() && !end_verified)
            {
                return Err("loss graph observations incomplete".into());
            }
            for (index, logits) in output.chunks_exact(VOCABULARY).enumerate() {
                let target_index = start + seen + index;
                let target = ids[target_index + 1];
                let (greedy, nll) = score(logits, target, &mut widened)?;
                serde_json::to_writer(
                    &mut out,
                    &json!({"target_index":target_index,"input_id":ids[target_index],
                        "target_id":target,"greedy_id":greedy,"nll":nll}),
                )?;
                out.write_all(b"\n")?;
            }
            seen += rows;
            out.flush()?;
        }
        assert_eq!(seen, end - start);
        assert_eq!(session.history(), block_ids);
        println!(
            "{}",
            json!({"event":"block","block":block,"targets":seen,"end":end})
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_loss_and_lowest_id_tie() {
        let same = [bf16::MAX; 3];
        let (greedy, loss) = score(&same, 2, &mut [0.0; 3]).unwrap();
        assert_eq!(greedy, 0);
        assert_eq!(loss, 3.0_f64.ln());
        let values = [bf16::from_f32(-2.0), bf16::ONE, bf16::ONE];
        let (greedy, loss) = score(&values, 0, &mut [0.0; 3]).unwrap();
        assert_eq!(greedy, 1);
        assert_eq!(loss, 3.0 + (2.0 + (-3.0_f64).exp()).ln());
    }

    #[test]
    fn malformed_loss_rows_are_rejected() {
        assert!(score(&[bf16::ZERO], 1, &mut [0.0]).is_err());
        assert!(score(&[bf16::NAN], 0, &mut [0.0]).is_err());
        assert!(score(&[bf16::ZERO], 0, &mut []).is_err());
    }
}
