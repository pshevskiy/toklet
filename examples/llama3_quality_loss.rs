//! Full frozen 10000-target native loss collection; quality decided separately.
#[cfg(any(test, all(target_os = "macos", target_arch = "aarch64")))]
#[path = "support/llama3_artifact.rs"]
mod artifact;
#[cfg(any(test, all(target_os = "macos", target_arch = "aarch64")))]
#[path = "support/llama3_quality_inputs.rs"]
mod inputs;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "support/nll.rs"]
mod nll;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use super::{artifact::*, inputs, nll};
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{
        fs::{self, OpenOptions},
        io::{BufWriter, Write},
        path::PathBuf,
    };
    use toklet::{
        metal::{self, AttentionMode, NormalizationMode, ProjectionMode, ReadbackMode},
        session,
        weights::LoadOptions,
    };
    const TOTAL: usize = 6 * 1024 * 1024 * 1024;
    const APP: usize = 64 * 1024 * 1024;
    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args_os().skip(1).collect();
        if args.len() != 4 {
            return Err("usage: llama3_quality_loss MODEL INPUTS MANIFEST-SHA NEW-OUTPUT".into());
        }
        let model_path = PathBuf::from(&args[0]);
        let folder = PathBuf::from(&args[1]);
        let manifest_sha = args[2].to_str().ok_or("manifest SHA must be UTF8")?;
        let data = inputs::load(&folder, manifest_sha)?;
        for &(name, size, sha) in FILES {
            hash_file(&model_path.join(name), size, sha)?;
        }
        let output = PathBuf::from(&args[3]);
        fs::create_dir(&output)?;
        let mut journal = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(output.join("targets.jsonl"))?,
        );
        let model = metal::Model::load_mlx_llama3_quality_diagnostic(
            &model_path,
            LoadOptions {
                context_limit: 1025,
                memory_limit_bytes: TOTAL - APP,
            },
        )?;
        let mut session = metal::Session::with_attention_mode(
            &model,
            session::Options {
                capacity: 1025,
                prefill_chunk_size: 32,
                position_offset: 0,
                memory_limit_bytes: TOTAL - APP,
            },
            ProjectionMode::MlxAffine,
            NormalizationMode::Compensated,
            ReadbackMode::FullLogits,
            AttentionMode::Reference,
        )?;
        session.enable_diagnostics(&[])?;
        let admitted = session.memory().admitted_bytes + APP;
        if admitted > TOTAL {
            return Err("logical process budget exceeded".into());
        }
        let mut total = nll::Sum::default();
        let mut count = 0;
        for block in &data.blocks {
            println!("native block {}", block.block);
            session.reset();
            let tokens = &data.ids[block.input_start..block.input_end];
            session.set_tokens(tokens)?;
            let mut position = 0;
            loop {
                let rows = session.evaluate_next_chunk()?;
                if rows == 0 {
                    break;
                }
                let snapshots = session.diagnostics()?;
                let logits = snapshots
                    .iter()
                    .find(|d| d.stage == "logits")
                    .ok_or("missing loss logits")?;
                if logits.rows != rows
                    || logits.columns != VOCAB
                    || logits.values.len() != rows * VOCAB
                {
                    return Err("loss shape differs".into());
                }
                for row in 0..rows {
                    let local = position + row;
                    let values = &logits.values[row * VOCAB..(row + 1) * VOCAB];
                    if local == tokens.len() - 1 {
                        if values.iter().any(|v| !v.is_finite()) {
                            return Err("nonfinite unscored final row".into());
                        }
                        continue;
                    }
                    let target = tokens[local + 1] as usize;
                    let (maximum, exp_mass, nll) = nll::statistics(values, target)?;
                    let greedy = values
                        .iter()
                        .position(|&v| f64::from(v) == maximum)
                        .ok_or("missing maximum")?;
                    let mut digest = Sha256::new();
                    let mut numeric = Sha256::new();
                    for value in values {
                        digest.update(value.to_le_bytes());
                        numeric.update(if *value == 0. { 0f32 } else { *value }.to_le_bytes());
                    }
                    serde_json::to_writer(
                        &mut journal,
                        &json!({"block":block.block,"local_row":local,"global_target_index":block.input_start+local+1,"target":target,"vocab":VOCAB,"target_fp32_bits":values[target].to_bits(),"maximum":maximum,"exp_mass":exp_mass,"nll":nll,"greedy_lowest_id":greedy,"row_sha256":format!("{:x}",digest.finalize()),"numeric_row_sha256":format!("{:x}",numeric.finalize())}),
                    )?;
                    journal.write_all(b"\n")?;
                    total.add(nll);
                    count += 1;
                }
                position += rows;
                journal.flush()?;
            }
            if position != tokens.len()
                || session.evaluated_len() != tokens.len()
                || session.pending_len() != 0
                || session.history() != tokens
            {
                return Err("loss block state differs".into());
            }
        }
        if count != 10000 {
            return Err("target count differs".into());
        }
        journal.flush()?;
        for &(name, size, sha) in FILES {
            hash_file(&model_path.join(name), size, sha)?;
        }
        fs::write(
            output.join("result.json"),
            serde_json::to_vec_pretty(
                &json!({"format":"toklet-llama3-quality-loss-native-v1","completed":true,"full_quality_qualified":false,"manifest_sha256":manifest_sha,"repository":REPOSITORY,"revision":REVISION,"targets":count,"mean_nll":total.value/count as f64,"admitted_bytes_including_app":admitted,"profile":{"projection":"MlxAffine","normalization":"Compensated","attention":"Reference","chunk":32,"capacity":1025},"targets_sha256":format!("{:x}",Sha256::digest(fs::read(output.join("targets.jsonl"))?))}),
            )?,
        )?;
        Ok(())
    }
}
fn main() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Err(error) = apple::run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        eprintln!("requires Apple Silicon Mac");
        std::process::exit(1);
    }
}
