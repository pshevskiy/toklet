//! Explicit experimental Qwen3 quality execution. No app admission or gate waiver.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[path = "support/loss.rs"]
mod loss;

#[cfg(any(
    test,
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
mod protocol {
    use serde::Deserialize;
    pub(super) fn validate_loss(ids: &[u32], probe: bool) -> Result<(), &'static str> {
        if (probe && ids != [9707, 11, 1879, 0])
            || (!probe && ids.len() != 10001)
            || ids.iter().any(|&id| id >= 151936)
        {
            return Err(
                "loss requires all10001 IDs; probe requires the exact frozen four-token input",
            );
        }
        Ok(())
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(super) struct Task {
        pub id: String,
        pub category: String,
        pub prompt_tokens: Vec<u32>,
        pub max_output_tokens: usize,
    }
    impl Task {
        pub fn validate(&self) -> Result<(), &'static str> {
            let cap = match self.category.as_str() {
                "code" => 256,
                "prose" => 32,
                "continuation" => 16,
                _ => return Err("unknown quality task category"),
            };
            if self.id.is_empty()
                || self.id.len() > 128
                || self.max_output_tokens != cap
                || self.prompt_tokens.is_empty()
                || self
                    .prompt_tokens
                    .len()
                    .checked_add(cap)
                    .is_none_or(|n| n > 1025)
                || self.prompt_tokens.iter().any(|&id| id >= 151936)
            {
                return Err("quality task identity, tokens, output or shared capacity differs");
            }
            Ok(())
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use super::{
        loss::*,
        protocol::{Task, validate_loss},
    };
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{
        collections::BTreeSet,
        io::{self, Read, Write},
        path::{Path, PathBuf},
        sync::atomic::AtomicBool,
    };
    use toklet::{
        generation::{self, Input, Request},
        metal::{self, AttentionMode, NormalizationMode, ProjectionMode, ReadbackMode},
        session, weights,
    };
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    const FORMAT: &str = "toklet-qwen3-quality-diagnostic-events-v1";
    enum Work {
        Loss { ids: Vec<u32>, sha: String },
        Tasks { cases: Vec<Task>, sha: String },
    }
    fn tasks(path: &Path) -> Result<(Vec<Task>, String)> {
        let file = std::fs::File::open(path)?;
        let size = file.metadata()?.len();
        const LIMIT: u64 = 4 * 1024 * 1024;
        if size == 0 || size > LIMIT {
            return Err("bounded complete task file required".into());
        }
        let mut body = Vec::new();
        file.take(LIMIT + 1).read_to_end(&mut body)?;
        if body.len() as u64 != size {
            return Err("task input changed size".into());
        }
        let mut cases = Vec::new();
        let mut ids = BTreeSet::new();
        let mut categories = [0; 3];
        for line in body.split(|b| *b == b'\n').filter(|s| !s.is_empty()) {
            let task: Task = serde_json::from_slice(line)?;
            task.validate()?;
            if !ids.insert(task.id.clone()) {
                return Err("duplicate quality task ID".into());
            }
            let i = match task.category.as_str() {
                "code" => 0,
                "prose" => 1,
                _ => 2,
            };
            categories[i] += 1;
            cases.push(task);
        }
        if categories != [40, 30, 30] || cases.len() != 100 {
            return Err("all100 quality tasks required".into());
        }
        Ok((cases, format!("{:x}", Sha256::digest(&body))))
    }
    pub fn run() -> Result<()> {
        let mut args = std::env::args_os().skip(1);
        let directory = PathBuf::from(args.next().ok_or("missing model directory")?);
        let mode = args.next().ok_or("quality mode must be loss or tasks")?;
        let input = PathBuf::from(args.next().ok_or("missing frozen quality input")?);
        let parallel = match args.next() {
            None => false,
            Some(value) if value == "--parallel-reference-experimental" => true,
            Some(_) => return Err("unknown quality schedule argument".into()),
        };
        if args.next().is_some() {
            return Err("extra quality argument".into());
        }
        let work = if mode == "loss" || mode == "probe" {
            let (ids, sha) = tokens(&input)?;
            validate_loss(&ids, mode == "probe")?;
            Work::Loss { ids, sha }
        } else if mode == "tasks" {
            let (cases, sha) = tasks(&input)?;
            Work::Tasks { cases, sha }
        } else {
            return Err("quality mode must be loss, tasks or explicit frozen probe".into());
        };
        let budget = weights::INITIAL_MEMORY_LIMIT_BYTES - APP_BYTES;
        let load_options = weights::LoadOptions {
            context_limit: CAPACITY,
            memory_limit_bytes: budget,
        };
        let model = if parallel {
            metal::Model::load_mlx_qwen3_context_diagnostic(&directory, load_options)?
        } else {
            metal::Model::load_mlx_qwen3_quality_diagnostic(&directory, load_options)?
        };
        let options = session::Options {
            capacity: CAPACITY,
            prefill_chunk_size: 128,
            position_offset: 0,
            memory_limit_bytes: budget,
        };
        let mut session = if parallel {
            metal::Session::with_qwen3_reference_schedule_diagnostic(
                &model,
                options,
                ReadbackMode::FullLogits,
                metal::ReferenceSchedule::ParallelOutputsExperimental,
            )?
        } else {
            metal::Session::with_attention_mode(
                &model,
                options,
                ProjectionMode::MlxAffine,
                NormalizationMode::Compensated,
                ReadbackMode::FullLogits,
                AttentionMode::Reference,
            )?
        };
        if matches!(work, Work::Loss { .. }) {
            session.enable_diagnostics(&[])?;
        }
        let admitted = session
            .memory()
            .admitted_bytes
            .checked_add(APP_BYTES)
            .ok_or("quality memory overflow")?;
        if admitted > weights::INITIAL_MEMORY_LIMIT_BYTES {
            return Err("quality process allowance exceeds6GiB".into());
        }
        let (input_sha, targets, cases) = match &work {
            Work::Loss { ids, sha } => (sha, ids.len() - 1, 0),
            Work::Tasks { cases, sha } => (sha, 0, cases.len()),
        };
        let mut out = io::stdout().lock();
        let mut readiness = json!({"event":"ready","format":FORMAT,"mode":mode.to_string_lossy(),
            "diagnostic_only":true,"full_quality_workload":mode!="probe","c512_numerical_gate_status":"rejected_unchanged","quality_qualified":false,
            "device":model.device_name(),"debug_assertions":cfg!(debug_assertions),"context_capacity":CAPACITY,
            "prefill_chunk_size":128,"input_sha256":input_sha,"targets":targets,"tasks":cases,
            "projection":"mlx_affine","normalization":"compensated","attention":"reference",
            "activation_dtype":"FP16","kv_dtype":"FP16","memory":session.memory(),
            "application_allowance_bytes":APP_BYTES,"admitted_bytes":admitted,
            "external_source_identity_and_endpoint_required":true});
        if parallel {
            if session.reference_schedule() != metal::ReferenceSchedule::ParallelOutputsExperimental
            {
                return Err("explicit quality reference schedule differs".into());
            }
            readiness.as_object_mut().unwrap().insert(
                "reference_schedule".into(),
                json!("parallel_outputs_experimental"),
            );
        }
        writeln!(out, "{readiness}")?;
        out.flush()?;
        match work {
            Work::Loss { ids, .. } => {
                let mut total = Sum::default();
                let mut serialized = vec![0; 151936 * 4];
                for (block, range) in spans(targets).enumerate() {
                    session.reset();
                    let mut block_loss = Sum::default();
                    let mut position = 0;
                    for part in ids[range.start..=range.end].chunks(128) {
                        session.append(part)?;
                        let rows = session.evaluate_next_chunk()?;
                        if rows != part.len()
                            || session.pending_len() != 0
                            || session.evaluated_len() != position + rows
                        {
                            return Err("quality chunk commitment differs".into());
                        }
                        let snapshots = session.diagnostics()?;
                        let logits = snapshots
                            .iter()
                            .find(|s| s.stage == "logits")
                            .ok_or("all-row logits absent")?;
                        if logits.rows != rows
                            || logits.columns != 151936
                            || logits.values.len() != rows * 151936
                        {
                            return Err("complete logit extent differs".into());
                        }
                        for row in 0..rows {
                            if position + row == range.len() {
                                break;
                            }
                            let index = range.start + position + row;
                            let target = ids[index + 1];
                            let values = &logits.values[row * 151936..(row + 1) * 151936];
                            let nll = loss(values, target as usize)?;
                            total.add(nll);
                            block_loss.add(nll);
                            for (bytes, value) in serialized.chunks_exact_mut(4).zip(values) {
                                bytes.copy_from_slice(&value.to_le_bytes());
                            }
                            writeln!(
                                out,
                                "{}",
                                json!({"event":"loss","target_index":index,"block":block,
                                "position":position+row,"target_token":target,"negative_log_likelihood":nll,
                                "target_logit_f32_bits":values[target as usize].to_bits(),
                                "logits_sha256":format!("{:x}",Sha256::digest(&serialized))})
                            )?;
                        }
                        position += rows;
                        drop(snapshots);
                        writeln!(
                            out,
                            "{}",
                            json!({"event":"chunk_done","block":block,"rows":rows,
                            "evaluated_tokens":position,"pending_tokens":session.pending_len()})
                        )?;
                        out.flush()?;
                    }
                    writeln!(
                        out,
                        "{}",
                        json!({"event":"block_done","block":block,"target_start":range.start,
                        "target_end":range.end,"targets":range.len(),"evaluated_tokens":session.evaluated_len(),
                        "pending_tokens":session.pending_len(),"total_negative_log_likelihood":block_loss.value})
                    )?;
                    out.flush()?;
                }
                writeln!(
                    out,
                    "{}",
                    json!({"event":"done","mode":mode.to_string_lossy(),"full_quality_workload":mode!="probe","targets":targets,"tasks":0,"context_policy":POLICY,
                    "total_negative_log_likelihood":total.value,"mean_negative_log_likelihood":total.value/targets as f64})
                )?;
            }
            Work::Tasks { cases, .. } => {
                let cancel = AtomicBool::new(false);
                for task in cases {
                    let mut visible = Vec::with_capacity(task.max_output_tokens);
                    let report = generation::generate(
                        &mut session,
                        Request {
                            input: Input::Reset(&task.prompt_tokens),
                            max_new_tokens: task.max_output_tokens,
                            stop_tokens: &[151645],
                        },
                        &cancel,
                        |event| {
                            visible.push(event.token_id);
                            writeln!(
                                out,
                                "{}",
                                json!({"event":"token","id":task.id,"index":event.index,"token_id":event.token_id})
                            )?;
                            out.flush()
                        },
                    )?;
                    if report.emitted_tokens != visible.len()
                        || report.sampled_tokens > task.max_output_tokens
                    {
                        return Err("task output extent differs".into());
                    }
                    writeln!(
                        out,
                        "{}",
                        json!({"event":"task_done","id":task.id,"category":task.category,
                        "prompt_tokens":task.prompt_tokens.len(),"requested_output_tokens":task.max_output_tokens,
                        "visible_token_ids":visible,"report":report})
                    )?;
                    out.flush()?;
                }
                writeln!(out, "{}", json!({"event":"done","targets":0,"tasks":100}))?;
            }
        }
        out.flush()?;
        Ok(())
    }
}
fn main() {
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    if let Err(error) = apple::run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
    #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
    {
        eprintln!(
            "Qwen3 diagnostic quality execution requires Apple Silicon macOS and --features metal"
        );
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::protocol::{Task, validate_loss};
    #[test]
    fn loss_cannot_report_a_clipped_workload_as_complete() {
        validate_loss(&vec![11; 10001], false).unwrap();
        for count in [0, 4, 512, 1025, 10000, 10002] {
            assert!(validate_loss(&vec![11; count], false).is_err());
        }
        validate_loss(&[9707, 11, 1879, 0], true).unwrap();
        assert!(validate_loss(&[9707, 11, 1879, 17], true).is_err());
        let mut invalid = vec![11; 10001];
        invalid[10000] = 151936;
        assert!(validate_loss(&invalid, false).is_err());
    }
    #[test]
    fn task_admission_preserves_complete_output_and_shared_budget() {
        for (category, cap) in [("code", 256), ("prose", 32), ("continuation", 16)] {
            let mut task = Task {
                id: "case/0".into(),
                category: category.into(),
                prompt_tokens: vec![11; 1025 - cap],
                max_output_tokens: cap,
            };
            task.validate().unwrap();
            task.prompt_tokens.push(11);
            assert!(task.validate().is_err());
            task.prompt_tokens.pop();
            task.max_output_tokens = cap - 1;
            assert!(task.validate().is_err());
        }
        let mut task = Task {
            id: "case/0".into(),
            category: "code".into(),
            prompt_tokens: vec![151936],
            max_output_tokens: 256,
        };
        assert!(task.validate().is_err());
        task.prompt_tokens.clear();
        assert!(task.validate().is_err());
        assert!(serde_json::from_str::<Task>(r#"{"id":"0","category":"code","prompt_tokens":[1],"max_output_tokens":256,"truncate":true}"#).is_err());
    }
}
