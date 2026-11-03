//! Explicit pinned 1025-position consistency probe; no adoption quality claim.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "support/llama3_artifact.rs"]
mod artifact;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use super::artifact::*;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{collections::BTreeMap, fs, path::PathBuf};
    use toklet::{
        metal::{self, AttentionMode, NormalizationMode, ProjectionMode, ReadbackMode},
        session,
        weights::LoadOptions,
    };
    const TOTAL: usize = 6 * 1024 * 1024 * 1024;
    const APP: usize = 64 * 1024 * 1024;
    const PREFIXES: [usize; 5] = [17, 512, 513, 1024, 1025];
    fn make(model: &metal::Model, capacity: usize, chunk: usize) -> Result<metal::Session<'_>> {
        Ok(metal::Session::with_attention_mode(
            model,
            session::Options {
                capacity,
                prefill_chunk_size: chunk,
                position_offset: 0,
                memory_limit_bytes: TOTAL - APP,
            },
            ProjectionMode::MlxAffine,
            NormalizationMode::Compensated,
            ReadbackMode::FullLogits,
            AttentionMode::Reference,
        )?)
    }
    // Signed zero has identical numerical meaning; every other FP32 bit must agree.
    fn hash(values: &[f32]) -> Result<String> {
        let mut sha = Sha256::new();
        if values.len() != VOCAB || values.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite or incorrectly shaped row".into());
        }
        for &value in values {
            sha.update(if value == 0. { 0f32 } else { value }.to_le_bytes());
        }
        Ok(format!("{:x}", sha.finalize()))
    }
    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args_os().skip(1).collect();
        if args.len() != 2 {
            return Err("usage: llama3_quality_context MODEL NEW-OUTPUT".into());
        }
        let model_path = PathBuf::from(&args[0]);
        let output = PathBuf::from(&args[1]);
        fs::create_dir(&output)?;
        // Frozen corpus bytes, unchanged from the existing 1024-target screen.
        let bytes = include_bytes!("../benchmarks/llama3-loss1024-v1/tokens.u32le");
        if bytes.len() != 4100
            || format!("{:x}", Sha256::digest(bytes))
                != "e4086f7621d9d781f5487db05f17df63e9693d0d2788d43a69c8108fc28aa493"
        {
            return Err("frozen token identity differs".into());
        }
        let ids: Vec<_> = bytes
            .chunks_exact(4)
            .map(|p| u32::from_le_bytes(p.try_into().unwrap()))
            .collect();
        if ids.iter().any(|&v| v as usize >= VOCAB) {
            return Err("token outside vocabulary".into());
        }
        for &(name, size, sha) in FILES {
            hash_file(&model_path.join(name), size, sha)?;
        }
        let options = |context_limit| LoadOptions {
            context_limit,
            memory_limit_bytes: TOTAL - APP,
        };
        // All three reject before payload upload/device creation.
        if metal::Model::load_mlx(&model_path, options(513)).is_ok()
            || metal::Model::load_mlx_llama3_audited(&model_path, options(513)).is_ok()
            || metal::Model::load_mlx_llama3_quality_diagnostic(&model_path, options(1026)).is_ok()
        {
            return Err("loader boundary opened unexpectedly".into());
        }
        let mut expected = BTreeMap::new();
        let mut admitted = 0;
        // Drop the entire ordinary owner before diagnostic upload; never retain two models.
        {
            let model = metal::Model::load_mlx(&model_path, options(512))?;
            let mut session = make(&model, 512, 32)?;
            admitted = admitted.max(session.memory().admitted_bytes + APP);
            for count in [17, 512] {
                session.reset();
                session.set_tokens(&ids[..count])?;
                session.evaluate()?;
                expected.insert(
                    count,
                    hash(session.logits().ok_or("missing ordinary logits")?)?,
                );
            }
        }
        let model = metal::Model::load_mlx_llama3_quality_diagnostic(&model_path, options(1025))?;
        if model.config().max_positions != 1025
            || model.mlx_declared_positions() != Some(131072)
            || model.mlx_dense_source_content_identity().is_some()
        {
            return Err("diagnostic capacity or app identity differs".into());
        }
        let mut observations = Vec::new();
        for chunk in [32, 128] {
            let mut session = make(&model, 1025, chunk)?;
            admitted = admitted.max(session.memory().admitted_bytes + APP);
            session.enable_diagnostics(&[])?;
            admitted = admitted.max(session.memory().admitted_bytes + APP);
            session.set_tokens(&ids)?;
            let mut position = 0;
            loop {
                let rows = session.evaluate_next_chunk()?;
                if rows == 0 {
                    break;
                }
                let diagnostics = session.diagnostics()?;
                let logits = diagnostics
                    .iter()
                    .find(|d| d.stage == "logits")
                    .ok_or("missing diagnostic logits")?;
                if logits.rows != rows
                    || logits.columns != VOCAB
                    || logits.values.len() != rows * VOCAB
                {
                    return Err("diagnostic shape differs".into());
                }
                for &count in &PREFIXES {
                    if count > position && count <= position + rows {
                        let row = count - position - 1;
                        let values = &logits.values[row * VOCAB..(row + 1) * VOCAB];
                        let actual = hash(values)?;
                        if let Some(reference) = expected.get(&count) {
                            if reference != &actual {
                                return Err(format!("prefix{count} chunk{chunk} differs").into());
                            }
                        } else {
                            expected.insert(count, actual.clone());
                        }
                        observations.push(json!({"kind":"chunk","chunk":chunk,"prefix":count,"numeric_sha256":actual}));
                    }
                }
                position += rows;
            }
            let before = session.logits().ok_or("missing boundary logits")?.to_vec();
            if position != 1025
                || session.evaluated_len() != 1025
                || session.pending_len() != 0
                || session.history() != ids
                || session.append(&[1]).is_ok()
                || session.set_tokens(&[1; 1026]).is_ok()
                || session.history() != ids
                || session.evaluated_len() != 1025
                || session.pending_len() != 0
                || session.logits() != Some(before.as_slice())
            {
                return Err("overflow changed valid prefix".into());
            }
        }
        // Independently reset prefixes exercise both long boundaries without future rows.
        let mut session = make(&model, 1025, 32)?;
        for &count in &PREFIXES {
            session.reset();
            session.set_tokens(&ids[..count])?;
            session.evaluate()?;
            let row = session.logits().ok_or("missing prefix logits")?;
            let actual = hash(row)?;
            if expected.get(&count) != Some(&actual) {
                return Err(format!("independent prefix{count} differs").into());
            }
            let mut bytes = Vec::with_capacity(VOCAB * 4);
            for &value in row {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            fs::write(output.join(format!("prefix-{count}.f32le")), &bytes)?;
            observations.push(json!({"kind":"independent","prefix":count,"numeric_sha256":actual,"raw_sha256":format!("{:x}",Sha256::digest(&bytes))}));
        }
        if admitted > TOTAL {
            return Err("logical process budget exceeded".into());
        }
        for &(name, size, sha) in FILES {
            hash_file(&model_path.join(name), size, sha)?;
        }
        fs::write(
            output.join("result.json"),
            serde_json::to_vec_pretty(&json!({
                "format":"toklet-llama3-quality-context-v1","ok":true,"capacity":1025,
                "ordinary_capacity":512,"repository":REPOSITORY,"revision":REVISION,
                "full_quality_qualified":false,"admitted_bytes_including_app":admitted,
                "loader_rejections":["ordinary513","audited513","quality1026"],
                "overflow_preserves_prefix":true,"observations":observations
            }))?,
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
