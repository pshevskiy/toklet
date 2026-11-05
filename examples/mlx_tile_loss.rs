//! Mac P2 loss qualification for the pinned audited Qwen2.5 MLX8 checkpoint.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "support/loss.rs"]
mod loss;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use super::loss::{APP_BYTES, BLOCK, CAPACITY, POLICY, Sum, TARGETS, loss, spans, tokens};
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{
        io::{self, Write},
        path::Path,
    };
    use toklet::{metal, sampling, session, weights};

    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    const WIDTH: usize = 151936;
    const BUDGET: usize = 1536 * 1024 * 1024 - APP_BYTES;
    const CORPUS: &str = "f53c5b0c446959164c4d2265166040123f95be5b6d9eec784895fb45a8215beb";
    const BASELINE: f64 = 2.883725149929484;

    fn open(model: &metal::Model, chunk: usize, limit: usize) -> Result<metal::Session<'_>> {
        Ok(metal::Session::with_attention_mode(
            model,
            session::Options {
                capacity: CAPACITY,
                prefill_chunk_size: chunk,
                position_offset: 0,
                memory_limit_bytes: limit,
            },
            metal::ProjectionMode::MlxAffine,
            metal::NormalizationMode::Compensated,
            metal::ReadbackMode::FullLogits,
            metal::AttentionMode::Online,
        )?)
    }

    fn append_step(s: &mut metal::Session<'_>, part: &[u32], evaluated: usize) -> Result<()> {
        s.append(part)?;
        if s.evaluate_next_chunk()? != part.len()
            || s.evaluated_len() != evaluated
            || s.pending_len() != 0
        {
            return Err("loss committed state differs".into());
        }
        Ok(())
    }

    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args().collect();
        if cfg!(debug_assertions)
            || args.len() != 4
            || !matches!(args[1].as_str(), "legacy" | "paired")
        {
            return Err("usage: release mlx_tile_loss legacy|paired MODEL CORPUS".into());
        }
        let paired = args[1] == "paired";
        let chunk = if paired { 128 } else { 1 };
        let (ids, sha) = tokens(Path::new(&args[3]))?;
        if sha != CORPUS || ids.len() != TARGETS + 1 || ids.iter().any(|&t| t as usize >= WIDTH) {
            return Err("frozen Qwen loss corpus differs".into());
        }
        let model = metal::Model::load_mlx_qwen2_audited(
            Path::new(&args[2]),
            weights::LoadOptions {
                context_limit: CAPACITY,
                memory_limit_bytes: BUDGET,
            },
        )?;
        if model.config().vocab_size != WIDTH || model.mlx_qwen2_source_content_identity().is_none()
        {
            return Err("pinned audited Qwen source required".into());
        }
        if paired {
            let mut probe = open(&model, chunk, BUDGET)?;
            probe.enable_affine_prefill_tile4()?;
            probe.enable_profile()?;
            append_step(&mut probe, &ids[..chunk], chunk)?;
            let stages = probe
                .profile()?
                .events
                .into_iter()
                .filter(|event| event.name == "mlx_product_tile4")
                .count();
            if stages != 24 * 7 {
                return Err("eligible loss prefill did not use tile4".into());
            }
            println!(
                "{}",
                json!({"event":"dispatch","rows":chunk,"tile4_stages":stages})
            );
        }
        let mut baseline = open(&model, chunk, BUDGET)?;
        let mut candidate = if paired {
            baseline.enable_diagnostics(&[])?;
            let other = baseline
                .memory()
                .admitted_bytes
                .checked_sub(model.model_resident_bytes()?)
                .ok_or("invalid baseline owner bytes")?;
            let mut s = open(
                &model,
                chunk,
                BUDGET
                    .checked_sub(other)
                    .ok_or("paired admission exceeds budget")?,
            )?;
            s.enable_affine_prefill_tile4()?;
            s.enable_diagnostics(&[])?;
            Some(s)
        } else {
            None
        };
        let admitted = baseline.memory().admitted_bytes
            + if let Some(s) = candidate.as_ref() {
                s.memory().admitted_bytes - model.model_resident_bytes()?
            } else {
                0
            }
            + APP_BYTES;
        let mut stream = io::BufWriter::new(io::stdout().lock());
        writeln!(
            stream,
            "{}",
            json!({"event":"ready","mode":args[1],"device":model.device_name(),
            "chunk":chunk,"capacity":CAPACITY,"block_targets":BLOCK,"policy":POLICY,
            "corpus_sha256":sha,"combined_admitted_bytes":admitted})
        )?;
        let mut total = Sum::default();
        let mut serialized = vec![0; WIDTH * 4];
        for (block, range) in spans(TARGETS).enumerate() {
            baseline.reset();
            if let Some(s) = candidate.as_mut() {
                s.reset();
            }
            let mut position = 0;
            if !paired {
                baseline.append(&ids[range.start..range.start + 1])?;
            }
            for part in ids[range.clone()].chunks(chunk) {
                let rows = part.len();
                if paired {
                    append_step(&mut baseline, part, position + rows)?;
                } else if baseline.evaluate_next_chunk()? != 1
                    || baseline.evaluated_len() != position + 1
                    || baseline.pending_len() != 0
                {
                    return Err("legacy loss committed state differs".into());
                }
                let diagnostics;
                let values: &[f32] = if let Some(s) = candidate.as_mut() {
                    append_step(s, part, position + rows)?;
                    if s.history() != baseline.history() {
                        return Err("paired loss history differs".into());
                    }
                    diagnostics = baseline.diagnostics()?;
                    let other = s.diagnostics()?;
                    let a = diagnostics
                        .iter()
                        .find(|x| x.stage == "logits")
                        .ok_or("baseline all-row loss missing")?;
                    let b = other
                        .iter()
                        .find(|x| x.stage == "logits")
                        .ok_or("candidate all-row loss missing")?;
                    if a.rows != rows
                        || b.rows != rows
                        || a.columns != WIDTH
                        || b.columns != WIDTH
                        || a.values.len() != rows * WIDTH
                        || b.values.len() != rows * WIDTH
                        || a.values
                            .iter()
                            .zip(&b.values)
                            .any(|(a, b)| a.to_bits() != b.to_bits())
                    {
                        return Err(format!(
                            "loss row mismatch at block {block}, input {position}"
                        )
                        .into());
                    }
                    &a.values
                } else {
                    baseline
                        .logits()
                        .ok_or("legacy loss distribution missing")?
                };
                for row in 0..rows {
                    let index = range.start + position + row;
                    let target = ids[index + 1] as usize;
                    let logits = &values[row * WIDTH..(row + 1) * WIDTH];
                    let nll = loss(logits, target)?;
                    let greedy = sampling::greedy(logits)?;
                    total.add(nll);
                    for (bytes, &value) in serialized.chunks_exact_mut(4).zip(logits) {
                        bytes.copy_from_slice(&value.to_le_bytes());
                    }
                    writeln!(
                        stream,
                        "{}",
                        json!({"event":"target","index":index,"block":block,
                        "local_position":position+row,"target":target,"greedy":greedy,
                        "nll":nll,"target_bits":logits[target].to_bits(),
                        "row_sha256":format!("{:x}",Sha256::digest(&serialized))})
                    )?;
                }
                position += rows;
                if !paired {
                    baseline.append(&ids[range.start + position..range.start + position + 1])?;
                    if baseline.pending_len() != 1 || baseline.evaluated_len() != position {
                        return Err("legacy loss pending state differs".into());
                    }
                }
            }
            stream.flush()?;
        }
        let mean = total.value / TARGETS as f64;
        let delta = mean - BASELINE;
        let passed = delta <= 0.05;
        writeln!(
            stream,
            "{}",
            json!({"event":"done","targets":TARGETS,"mean_nll":mean,
            "fp32_baseline_mean":BASELINE,"delta_nll":delta,"passed":passed,
            "paired_complete_rows":if paired { TARGETS } else { 0 }})
        )?;
        stream.flush()?;
        if !passed {
            return Err("unchanged loss threshold exceeded".into());
        }
        Ok(())
    }
}
fn main() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Err(error) = apple::run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        eprintln!("Loss qualification requires Apple Silicon macOS");
        std::process::exit(1);
    }
}
