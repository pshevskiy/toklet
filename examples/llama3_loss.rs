//! Fixed same-artifact 1,024-target diagnostic screen, not adoption quality.
#[cfg(any(test, all(target_os = "macos", target_arch = "aarch64")))]
#[path = "support/llama3_artifact.rs"]
mod artifact;
#[cfg(any(test, all(target_os = "macos", target_arch = "aarch64")))]
#[path = "support/nll.rs"]
mod nll;
#[cfg(any(test, all(target_os = "macos", target_arch = "aarch64")))]
mod protocol {
    use super::artifact::*;
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::{fs::File, io::Read, path::Path};
    pub fn blocks() -> Value {
        Value::Array((0..4).map(|block| {
        let start=block*256;
        json!({"block":block,"input_start":start,"input_end":start+257,"score_rows":[0,256],
            "global_target_indices":[start+1,start+257],"targets":256,"reset_position":0})
    }).collect())
    }
    pub fn policy() -> Value {
        json!({"targets":1024,"capacity":512,"block_tokens":257,"reset_each_block":true,
        "add_special_tokens":false,"chat_template":false,"stop_on_eos":false,"mean_nll_increase_limit":0.05,
        "reference":"same packed artifact canonical FP64 arithmetic/Half storage","cross_profile_greedy_gate":false,
        "unquantized_quality_qualified":false,"full_task_quality_qualified":false})
    }
    pub fn validate(value: &Value) -> Result<()> {
        if value["format"] != "toklet-llama3-loss1024-inputs-v1"
            || value["repository"] != REPOSITORY
            || value["revision"] != REVISION
            || value["blocks"] != blocks()
            || value["policy"] != policy()
            || value["tokens"]["file"] != "tokens.u32le"
            || value["tokens"]["dtype"] != "u32le"
            || value["tokens"]["count"] != 1025
            || value["tokens"]["bytes"] != 4100
            || value["model_executed"] != false
        {
            return Err("fixed loss input contract differs".into());
        }
        let digest = value["tokens"]["sha256"]
            .as_str()
            .ok_or("token digest missing")?;
        if digest.len() != 64
            || !digest
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            return Err("token digest invalid".into());
        }
        Ok(())
    }
    pub fn load(folder: &Path, sha: &str) -> Result<Vec<u32>> {
        if sha.len() != 64 || !sha.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err("external input SHA required".into());
        }
        let mut raw = Vec::new();
        File::open(folder.join("manifest.json"))?
            .take(65537)
            .read_to_end(&mut raw)?;
        if raw.len() > 65536 || format!("{:x}", Sha256::digest(&raw)) != sha {
            return Err("loss manifest identity differs".into());
        }
        let manifest: Value = serde_json::from_slice(&raw)?;
        validate(&manifest)?;
        let digest = manifest["tokens"]["sha256"]
            .as_str()
            .ok_or("token digest missing")?;
        let mut raw = Vec::new();
        File::open(folder.join("tokens.u32le"))?
            .take(4101)
            .read_to_end(&mut raw)?;
        if raw.len() != 4100 || format!("{:x}", Sha256::digest(&raw)) != digest {
            return Err("consumed loss token identity differs".into());
        }
        let ids: Vec<_> = raw
            .chunks_exact(4)
            .map(|p| u32::from_le_bytes(p.try_into().unwrap()))
            .collect();
        if ids.iter().any(|&id| id as usize >= VOCAB) {
            return Err("loss target outside vocabulary".into());
        }
        Ok(ids)
    }
}
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use super::{artifact::*, nll, protocol};
    use serde_json::{Value, json};
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
    fn hashes(row: &[f32]) -> (String, String) {
        let mut raw = Sha256::new();
        let mut numeric = Sha256::new();
        for value in row {
            raw.update(value.to_le_bytes());
            numeric.update(if *value == 0. {
                0f32.to_le_bytes()
            } else {
                value.to_le_bytes()
            });
        }
        (
            format!("{:x}", raw.finalize()),
            format!("{:x}", numeric.finalize()),
        )
    }
    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args_os().skip(1).collect();
        if args.len() != 4 {
            return Err("usage: llama3_loss MODEL INPUT-FOLDER INPUT-SHA NEW-OUTPUT".into());
        }
        let model_path = PathBuf::from(&args[0]);
        let folder = PathBuf::from(&args[1]);
        let input_sha = args[2].to_str().ok_or("SHA must be UTF-8")?;
        let ids = protocol::load(&folder, input_sha)?;
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
        let model = metal::Model::load_mlx(
            &model_path,
            LoadOptions {
                context_limit: 512,
                memory_limit_bytes: TOTAL - APP,
            },
        )?;
        if model.mlx_declared_positions() != Some(131072) || model.positional_resident_bytes() == 0
        {
            return Err("loaded model position identity differs".into());
        }
        let mut maximum_admitted = 0;
        let mut summaries = Vec::<Value>::new();
        let mut reference_hashes = Vec::new();
        let mut final_hashes = Vec::new();
        for chunk in [32, 128] {
            let mut session = metal::Session::with_attention_mode(
                &model,
                session::Options {
                    capacity: 512,
                    prefill_chunk_size: chunk,
                    position_offset: 0,
                    memory_limit_bytes: TOTAL - APP,
                },
                ProjectionMode::MlxAffine,
                NormalizationMode::Compensated,
                ReadbackMode::FullLogits,
                AttentionMode::Reference,
            )?;
            session.enable_diagnostics(&[])?;
            maximum_admitted = maximum_admitted.max(session.memory().admitted_bytes + APP);
            let mut total = nll::Sum::default();
            let mut targets = 0;
            for block in 0..4 {
                session.reset();
                let start = block * 256;
                let tokens = &ids[start..start + 257];
                session.set_tokens(tokens)?;
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
                        .ok_or("loss diagnostic missing")?;
                    if logits.rows != rows
                        || logits.columns != VOCAB
                        || logits.values.len() != rows * VOCAB
                    {
                        return Err("loss readback shape differs".into());
                    }
                    for row in 0..rows {
                        let local = position + row;
                        let values = &logits.values[row * VOCAB..(row + 1) * VOCAB];
                        if local == 256 {
                            if values.iter().any(|v| !v.is_finite()) {
                                return Err("nonfinite unscored final row".into());
                            }
                            let (raw, numeric) = hashes(values);
                            let matches = if chunk == 32 {
                                final_hashes.push(numeric.clone());
                                true
                            } else {
                                final_hashes[block] == numeric
                            };
                            serde_json::to_writer(
                                &mut journal,
                                &json!({"kind":"unscored","chunk":chunk,"block":block,"local_row":256,
                                "row_sha256":raw,"numeric_row_sha256":numeric}),
                            )?;
                            journal.write_all(b"\n")?;
                            journal.flush()?;
                            if !matches {
                                return Err(format!(
                                    "unscored native schedules differ: block{block} chunk{chunk}"
                                )
                                .into());
                            }
                            continue;
                        }
                        if local > 256 {
                            return Err("unexpected loss row".into());
                        }
                        let target = tokens[local + 1] as usize;
                        let (maximum, exp_mass, nll) = nll::statistics(values, target)?;
                        let greedy = values
                            .iter()
                            .position(|&v| f64::from(v) == maximum)
                            .ok_or("finite maximum missing")?;
                        let (raw, numeric) = hashes(values);
                        let matches = if chunk == 32 {
                            reference_hashes.push(numeric.clone());
                            true
                        } else {
                            reference_hashes[targets] == numeric
                        };
                        serde_json::to_writer(
                            &mut journal,
                            &json!({"kind":"target","chunk":chunk,"block":block,"local_row":local,
                            "global_target_index":start+local+1,"target":target,"target_fp32_bits":values[target].to_bits(),
                            "vocab":VOCAB,"maximum":maximum,"exp_mass":exp_mass,"nll":nll,"greedy_lowest_id":greedy,
                            "row_sha256":raw,"numeric_row_sha256":numeric}),
                        )?;
                        journal.write_all(b"\n")?;
                        journal.flush()?;
                        if !matches {
                            return Err(format!("native numeric logit schedules differ: block{block} row{local} chunk{chunk}").into());
                        }
                        total.add(nll);
                        targets += 1;
                    }
                    position += rows;
                    if session.evaluated_len() != position {
                        return Err("loss committed prefix differs".into());
                    }
                }
                if position != 257 || session.history() != tokens || session.pending_len() != 0 {
                    return Err("loss block incomplete".into());
                }
            }
            if targets != 1024 {
                return Err("loss target coverage differs".into());
            }
            summaries.push(json!({"chunk":chunk,"targets":targets,"mean_nll":total.value/1024.}));
        }
        if maximum_admitted > TOTAL {
            return Err("loss diagnostic admission exceeds6GiB".into());
        }
        journal.flush()?;
        let report = json!({"format":"toklet-llama3-loss1024-native-v1","repository":REPOSITORY,"revision":REVISION,
            "input_manifest_sha256":input_sha,"policy":protocol::policy(),"blocks":protocol::blocks(),"device":model.device_name(),
            "profile":"MlxAffine/Compensated/Reference/Half","readback":"FullLogits","summaries":summaries,
            "maximum_admitted_bytes":maximum_admitted,"application_allowance_bytes":APP,
            "weights_bytes":model.weights_bytes(),"positional_bytes":model.positional_resident_bytes(),
            "exact_schedule_control":"finite FP32 row hashes with signed zero canonicalized; raw hashes retained separately",
            "source_comparison_pending":true,"unquantized_quality_qualified":false,"full_task_quality_qualified":false,
            "app_qualified":false,"iphone_qualified":false});
        let mut out = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.join("native.json"))?;
        serde_json::to_writer_pretty(&mut out, &report)?;
        out.write_all(b"\n")?;
        println!(
            "{}",
            json!({"capture_completed":true,"summaries":summaries,"maximum_admitted_bytes":maximum_admitted,"qualification_pending":true})
        );
        Ok(())
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn numeric_hash_preserves_numeric_equality_without_erasing_one_ulp() {
            let positive = hashes(&[0., 1.]);
            let negative = hashes(&[-0., 1.]);
            assert_ne!(positive.0, negative.0);
            assert_eq!(positive.1, negative.1);
            let adjacent = hashes(&[0., f32::from_bits(1f32.to_bits() + 1)]);
            assert_ne!(positive.1, adjacent.1);
        }
    }
}
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() -> artifact::Result<()> {
    apple::run()
}
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("llama3_loss requires an Apple Silicon Mac with Metal");
    std::process::exit(1);
}
#[cfg(test)]
mod tests {
    use super::{artifact::*, nll, protocol};
    use serde_json::json;
    use sha2::{Digest, Sha256};
    #[test]
    fn loss_loader_hashes_consumed_bytes_and_rejects_size_and_vocabulary_changes() {
        let folder =
            std::env::temp_dir().join(format!("toklet-llama3-loss-input-{}", std::process::id()));
        std::fs::create_dir(&folder).unwrap();
        let mut raw = vec![0u8; 4100];
        let mut manifest = json!({"format":"toklet-llama3-loss1024-inputs-v1","repository":REPOSITORY,"revision":REVISION,
            "blocks":protocol::blocks(),"policy":protocol::policy(),"tokens":{"file":"tokens.u32le","dtype":"u32le","count":1025,"bytes":4100,"sha256":format!("{:x}",Sha256::digest(&raw))},"model_executed":false});
        std::fs::write(folder.join("tokens.u32le"), &raw).unwrap();
        let document = serde_json::to_vec(&manifest).unwrap();
        std::fs::write(folder.join("manifest.json"), &document).unwrap();
        let sha = format!("{:x}", Sha256::digest(&document));
        assert_eq!(protocol::load(&folder, &sha).unwrap(), vec![0; 1025]);
        raw[0] = 1;
        std::fs::write(folder.join("tokens.u32le"), &raw).unwrap();
        assert!(protocol::load(&folder, &sha).is_err());
        raw.push(0);
        std::fs::write(folder.join("tokens.u32le"), &raw).unwrap();
        assert!(protocol::load(&folder, &sha).is_err());
        raw.pop();
        raw[..4].copy_from_slice(&(VOCAB as u32).to_le_bytes());
        manifest["tokens"]["sha256"] = json!(format!("{:x}", Sha256::digest(&raw)));
        let document = serde_json::to_vec(&manifest).unwrap();
        std::fs::write(folder.join("tokens.u32le"), &raw).unwrap();
        std::fs::write(folder.join("manifest.json"), &document).unwrap();
        assert!(protocol::load(&folder, &format!("{:x}", Sha256::digest(&document))).is_err());
        std::fs::remove_dir_all(folder).unwrap();
    }
    #[test]
    fn fixed_blocks_cover_every_target_once_and_reject_modified_contracts() {
        let manifest = json!({"format":"toklet-llama3-loss1024-inputs-v1","repository":REPOSITORY,"revision":REVISION,
            "blocks":protocol::blocks(),"policy":protocol::policy(),"tokens":{"file":"tokens.u32le","dtype":"u32le","count":1025,"bytes":4100,"sha256":"0".repeat(64)},"model_executed":false});
        protocol::validate(&manifest).unwrap();
        let mut targets = Vec::new();
        for block in protocol::blocks().as_array().unwrap() {
            let start = block["input_start"].as_u64().unwrap();
            let end = block["input_end"].as_u64().unwrap();
            assert_eq!(end - start, 257);
            targets.extend(start + 1..end);
        }
        assert_eq!(targets, (1..=1024).collect::<Vec<_>>());
        for (key, value) in [
            ("policy", json!({})),
            ("blocks", json!([])),
            ("revision", json!("main")),
            ("model_executed", json!(true)),
        ] {
            let mut changed = manifest.clone();
            changed[key] = value;
            assert!(protocol::validate(&changed).is_err());
        }
    }
    #[test]
    fn stable_native_statistics_match_manual_distributions_and_extremes() {
        let (maximum, mass, value) = nll::statistics(&[0., 0.], 1).unwrap();
        assert_eq!((maximum, mass), (0., 2.));
        assert_eq!(value, 2f64.ln());
        let (maximum, mass, value) = nll::statistics(&[f32::MAX; 3], 1).unwrap();
        assert_eq!(maximum, f64::from(f32::MAX));
        assert_eq!(mass, 3.);
        assert_eq!(value, 3f64.ln());
        let (_, _, value) = nll::statistics(&[1000., -1000.], 1).unwrap();
        assert_eq!(value, 2000.);
        for values in [&[][..], &[f32::NAN][..], &[f32::INFINITY][..]] {
            assert!(nll::statistics(values, 0).is_err());
        }
    }
}
