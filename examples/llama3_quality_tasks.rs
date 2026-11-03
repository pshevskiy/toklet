//! Raw-ID Llama task collection; decoding and isolated grading are separate.
#[cfg(any(test, all(target_os = "macos", target_arch = "aarch64")))]
#[path = "support/llama3_artifact.rs"]
mod artifact;
#[cfg(any(test, all(target_os = "macos", target_arch = "aarch64")))]
#[path = "support/llama3_task_inputs.rs"]
mod inputs;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use super::{artifact::*, inputs};
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{
        fs::{self, OpenOptions},
        io::{self, BufWriter, Write},
        path::PathBuf,
        sync::atomic::AtomicBool,
    };
    use toklet::{
        generation::{self, Input, Request, StopReason},
        metal::{self, AttentionMode, NormalizationMode, ProjectionMode, ReadbackMode},
        session,
        weights::LoadOptions,
    };
    const TOTAL: usize = 6 * 1024 * 1024 * 1024;
    const APP: usize = 64 * 1024 * 1024;
    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args_os().skip(1).collect();
        if !(4..=5).contains(&args.len()) {
            return Err(
                "usage: llama3_quality_tasks MODEL INPUTS MANIFEST-SHA NEW-OUTPUT [--probe]".into(),
            );
        }
        let probe = if args.len() == 5 {
            if args[4] != "--probe" {
                return Err("unknown task option".into());
            }
            true
        } else {
            false
        };
        let model_path = PathBuf::from(&args[0]);
        let folder = PathBuf::from(&args[1]);
        let input_sha = args[2].to_str().ok_or("SHA must be UTF8")?;
        let frozen = inputs::load(&folder, input_sha)?;
        let cases = if probe {
            vec![inputs::Task {
                id: "authored-probe".into(),
                category: "probe".into(),
                prompt_tokens: vec![128000, 1, 7, 13],
                max_output_tokens: 4,
            }]
        } else {
            frozen
        };
        for &(name, size, sha) in FILES {
            hash_file(&model_path.join(name), size, sha)?;
        }
        let output = PathBuf::from(&args[3]);
        fs::create_dir(&output)?;
        let mut journal = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(output.join("tasks.jsonl"))?,
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
                capacity: 566,
                prefill_chunk_size: 32,
                position_offset: 0,
                memory_limit_bytes: TOTAL - APP,
            },
            ProjectionMode::MlxAffine,
            NormalizationMode::Compensated,
            ReadbackMode::FullLogits,
            AttentionMode::Reference,
        )?;
        let admitted = session.memory().admitted_bytes + APP;
        if admitted > TOTAL {
            return Err("native task budget exceeded".into());
        }
        let cancelled = AtomicBool::new(false);
        let mut count = 0;
        for (index, case) in cases.iter().enumerate() {
            println!("native case {index} {}", case.id);
            let mut visible = Vec::with_capacity(case.max_output_tokens);
            let report = generation::generate(
                &mut session,
                Request {
                    input: Input::Reset(&case.prompt_tokens),
                    max_new_tokens: case.max_output_tokens,
                    stop_tokens: &inputs::STOPS,
                },
                &cancelled,
                |event| {
                    if event.index != visible.len() || inputs::STOPS.contains(&event.token_id) {
                        return Err(io::Error::other("task emitted index/hidden stop differs"));
                    }
                    visible.push(event.token_id);
                    Ok(())
                },
            )?;
            let mut history = case.prompt_tokens.clone();
            history.extend_from_slice(&visible);
            match report.reason {
                StopReason::StopToken { token_id } => {
                    if !inputs::STOPS.contains(&token_id) {
                        return Err("unrecognized task stop".into());
                    }
                    history.push(token_id);
                }
                StopReason::OutputLimit => {
                    if visible.len() != case.max_output_tokens {
                        return Err("task output reserve truncated".into());
                    }
                }
                StopReason::Cancelled => return Err("unexpected task cancellation".into()),
            }
            if session.history() != history
                || report.reused_tokens != 0
                || report.prefill_tokens != case.prompt_tokens.len()
                || report.emitted_tokens != visible.len()
                || report.sampled_tokens
                    != visible.len()
                        + usize::from(matches!(report.reason, StopReason::StopToken { .. }))
                || report.history_tokens != history.len()
                || report.pending_tokens != 1
                || report.evaluated_tokens + 1 != history.len()
            {
                return Err("native sampled/committed task state differs".into());
            }
            serde_json::to_writer(
                &mut journal,
                &json!({"id":case.id,"category":case.category,"prompt_tokens":case.prompt_tokens.len(),"output_reserve":case.max_output_tokens,"visible_token_ids":visible,"stop_reason":report.reason,"generation":report}),
            )?;
            journal.write_all(b"\n")?;
            journal.flush()?;
            session.reset();
            if !session.history().is_empty()
                || session.evaluated_len() != 0
                || session.pending_len() != 0
                || session.logits().is_some()
            {
                return Err("native task reset failed".into());
            }
            count += 1;
        }
        for &(name, size, sha) in FILES {
            hash_file(&model_path.join(name), size, sha)?;
        }
        fs::write(
            output.join("result.json"),
            serde_json::to_vec_pretty(
                &json!({"format":"toklet-llama3-quality-tasks-native-v1","completed":true,"mode":if probe{"authored-probe"}else{"full-tasks"},"manifest_sha256":input_sha,"repository":REPOSITORY,"revision":REVISION,"cases":count,"full_quality_qualified":false,"admitted_bytes_including_app":admitted,"profile":{"projection":"MlxAffine","normalization":"Compensated","attention":"Reference","readback":"FullLogits","chunk":32,"capacity":566},"tasks_sha256":format!("{:x}",Sha256::digest(fs::read(output.join("tasks.jsonl"))?))}),
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
