//! Bounded held-out teacher-forced loss worker. This is not a latency benchmark.
//! Independent blocks reset rotary/KV position to zero and overlap one token;
//! every target is scored once from its preceding committed distribution.
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    io::{self, Read, Write},
    ops::Range,
    path::{Path, PathBuf},
};
use toklet::{
    cpu::ProjectionKernel,
    session::{Options, Session},
    weights::{self, LoadOptions},
};

const TARGETS: usize = 10_000;
const BLOCK: usize = 1024;
const CAPACITY: usize = BLOCK + 1;
const APP_BYTES: usize = 64 * 1024 * 1024;
const POLICY: &str =
    "independent_blocks_of_up_to_1025_tokens_one_token_overlap_score_every_target_once";

#[derive(Default)]
struct Sum {
    value: f64,
    correction: f64,
}
impl Sum {
    fn add(&mut self, value: f64) {
        let adjusted = value - self.correction;
        let next = self.value + adjusted;
        self.correction = (next - self.value) - adjusted;
        self.value = next;
    }
}

fn loss(logits: &[f32], target: usize) -> Result<f64, &'static str> {
    if logits.is_empty() || target >= logits.len() {
        return Err("loss target or vocabulary outside bounds");
    }
    let mut maximum = f64::NEG_INFINITY;
    for &value in logits {
        if !value.is_finite() {
            return Err("loss logits must be finite");
        }
        maximum = maximum.max(f64::from(value));
    }
    let mut exponential = Sum::default();
    for &value in logits {
        exponential.add((f64::from(value) - maximum).exp());
    }
    // Subtract before adding log(sum), retaining equal-logit loss even at MAX.
    let result = (maximum - f64::from(logits[target])) + exponential.value.ln();
    if !result.is_finite() || result < 0.0 {
        return Err("loss reduction produced an invalid result");
    }
    Ok(result)
}

fn spans(targets: usize) -> impl Iterator<Item = Range<usize>> {
    (0..targets)
        .step_by(BLOCK)
        .map(move |start| start..start.saturating_add(BLOCK).min(targets))
}

fn tokens(path: &Path) -> Result<(Vec<u32>, String), Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let size = file.metadata()?.len();
    let limit = (TARGETS + 1) * size_of::<u32>();
    if !(8..=limit as u64).contains(&size) || !size.is_multiple_of(4) {
        return Err("loss input must contain 2..10001 little-endian UInt32 tokens".into());
    }
    let mut data = Vec::new();
    file.take((limit + 1) as u64).read_to_end(&mut data)?;
    if data.len() as u64 != size {
        return Err("loss input changed size during bounded read".into());
    }
    let sha = format!("{:x}", Sha256::digest(&data));
    let ids = data
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect();
    Ok((ids, sha))
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let model_path = PathBuf::from(
        args.next()
            .ok_or("usage: quality_bench MODEL TOKENS [scalar|tiled]")?,
    );
    let token_path = PathBuf::from(args.next().ok_or("missing loss token file")?);
    let kernel = match args.next().as_deref().map(std::ffi::OsStr::to_str) {
        None | Some(Some("scalar")) => ProjectionKernel::Scalar,
        Some(Some("tiled")) => ProjectionKernel::Tiled,
        _ => return Err("loss CPU projection must be scalar or tiled".into()),
    };
    if args.next().is_some() {
        return Err("too many loss worker arguments".into());
    }
    let (ids, corpus_sha) = tokens(&token_path)?;
    let targets = ids.len() - 1;
    let budget = weights::INITIAL_MEMORY_LIMIT_BYTES - APP_BYTES;
    let loaded = weights::load_qwen(
        &model_path,
        LoadOptions {
            memory_limit_bytes: budget,
            ..LoadOptions::default()
        },
    )?;
    let model = loaded.model.with_projection_kernel(kernel);
    if ids
        .iter()
        .any(|&token| token as usize >= model.config().vocab_size)
    {
        return Err("loss input token outside vocabulary".into());
    }
    let mut session = Session::new(
        &model,
        Options {
            capacity: CAPACITY,
            prefill_chunk_size: 1,
            memory_limit_bytes: budget,
            ..Options::default()
        },
    )?;
    let memory = session.memory();
    let logit_bytes = model
        .config()
        .vocab_size
        .checked_mul(size_of::<f32>())
        .ok_or("loss logit size overflow")?;
    if logit_bytes > APP_BYTES / 2 {
        return Err("loss diagnostic storage exceeds application allowance".into());
    }
    let mut serialized = Vec::new();
    serialized.try_reserve_exact(logit_bytes)?;
    serialized.resize(logit_bytes, 0);
    let admitted = memory
        .admitted_bytes
        .checked_add(APP_BYTES)
        .ok_or("loss admission overflow")?;
    let mut out = io::stdout().lock();
    writeln!(
        out,
        "{}",
        json!({"event":"ready","format":"toklet-held-out-loss-events-v1",
        "cpu_projection":model.projection_kernel().name(),"target_tokens":targets,"corpus_sha256":corpus_sha,
        "context_policy":POLICY,"block_targets":BLOCK,"context_capacity":CAPACITY,"prefill_chunk_size":1,
        "weight_dtype":"FP32","kv_dtype":"FP32","load_seconds":loaded.report.load_seconds,
        "conversion_seconds":loaded.report.conversion_seconds,"memory_admitted_bytes":admitted,
        "logical_weights_bytes":memory.weights_bytes,"allocated_kv_bytes":memory.kv_bytes,
        "scratch_bytes":memory.scratch_bytes,"history_bytes":memory.history_bytes,"logits_bytes":memory.logits_bytes,
        "metadata_bytes":memory.metadata_bytes,"metadata_allowance_bytes":memory.metadata_allowance_bytes,
        "application_allowance_bytes":APP_BYTES,"diagnostic_logit_bytes":serialized.len()})
    )?;
    out.flush()?;
    let mut total = Sum::default();
    for (block, range) in spans(targets).enumerate() {
        session.reset();
        session.append(&ids[range.start..range.start + 1])?;
        session.evaluate()?;
        let mut block_loss = Sum::default();
        for (position, target_index) in range.clone().enumerate() {
            let target = ids[target_index + 1];
            if session.evaluated_len() != position + 1 || session.pending_len() != 0 {
                return Err("loss prefix is not fully committed".into());
            }
            let logits = session.logits().ok_or("loss distribution unavailable")?;
            let nll = loss(logits, target as usize)?;
            for (bytes, &value) in serialized.chunks_exact_mut(4).zip(logits) {
                bytes.copy_from_slice(&value.to_le_bytes());
            }
            let sha = format!("{:x}", Sha256::digest(&serialized));
            total.add(nll);
            block_loss.add(nll);
            writeln!(
                out,
                "{}",
                json!({"event":"loss","target_index":target_index,"block_index":block,
                "position":position,"prefix_tokens":session.evaluated_len(),"target_token":target,
                "negative_log_likelihood":nll,"target_logit_f32_bits":logits[target as usize].to_bits(),
                "logits_sha256":sha})
            )?;
            // Diagnostic output/hashing is outside inference and not timed as
            // performance. Every scored target then commits before the next.
            out.flush()?;
            session.append(&[target])?;
            session.evaluate()?;
        }
        writeln!(
            out,
            "{}",
            json!({"event":"block_done","block_index":block,"target_start":range.start,
            "target_end":range.end,"scored_targets":range.len(),"total_negative_log_likelihood":block_loss.value,
            "evaluated_tokens":session.evaluated_len(),"pending_tokens":session.pending_len()})
        )?;
        out.flush()?;
    }
    writeln!(
        out,
        "{}",
        json!({"event":"done","scored_targets":targets,
        "total_negative_log_likelihood":total.value,"mean_negative_log_likelihood":total.value / targets as f64,
        "evaluated_tokens":session.evaluated_len(),"pending_tokens":session.pending_len()})
    )?;
    out.flush()?;
    // Keep the completed child alive for one own-process footprint query. No
    // CPU model work resumes; only the owning driver can close the handshake.
    let mut reply = String::new();
    io::stdin().take(6).read_to_string(&mut reply)?;
    if reply != "exit\n" {
        return Err("loss worker requires completed exit handshake".into());
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_loss_matches_manual_distributions_and_extremes() {
        assert_eq!(loss(&[42.0], 0).unwrap(), 0.0);
        assert!((loss(&[0.0, 0.0], 1).unwrap() - 2.0_f64.ln()).abs() < 1e-15);
        assert!((loss(&[f32::MAX; 3], 2).unwrap() - 3.0_f64.ln()).abs() < 1e-15);
        assert_eq!(loss(&[1000.0, -1000.0], 1).unwrap(), 2000.0);
        assert_eq!(
            loss(&[f32::MAX, -f32::MAX], 1).unwrap(),
            2.0 * f64::from(f32::MAX)
        );
        let expected = 1.0 + (-1.0_f64).exp().ln_1p();
        assert!((loss(&[-1000.0, -999.0], 0).unwrap() - expected).abs() < 1e-15);
        assert_eq!(loss(&[-1000.0, -999.0], 0), loss(&[1000.0, 1001.0], 0));
    }

    #[test]
    fn invalid_distributions_and_targets_fail_closed() {
        assert!(loss(&[], 0).is_err());
        assert!(loss(&[1.0], 1).is_err());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(loss(&[1.0, value], 0).is_err());
        }
    }

    #[test]
    fn frozen_blocks_score_each_target_once_with_one_token_overlap() {
        let blocks: Vec<_> = spans(TARGETS).collect();
        assert_eq!(blocks.len(), 10);
        assert_eq!(blocks[0], 0..1024);
        assert_eq!(blocks[1], 1024..2048);
        assert_eq!(blocks[9], 9216..10000);
        assert_eq!(
            blocks.iter().flat_map(Clone::clone).collect::<Vec<_>>(),
            (0..TARGETS).collect::<Vec<_>>()
        );
        for range in blocks {
            // Token slice is inclusive of both the initial prefix and target:
            // [start, end] has <=1025 IDs. Adjacent blocks share only token end.
            assert!(range.len() < CAPACITY);
        }
    }
}
