//! Explicit pinned real-Phi diagnostic. No application admission or kernel changes.
#[cfg(any(test, all(target_os = "macos", target_arch = "aarch64")))]
#[path = "support/phi3_artifact.rs"]
mod artifact;
#[cfg(any(test, all(target_os = "macos", target_arch = "aarch64")))]
mod contract {
    pub use super::artifact::{FILES, REPOSITORY, REVISION, Result, VOCAB, hash_file};
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::{fs::File, io::Read, path::Path};
    pub fn cases() -> Value {
        json!({"single":{"tokens":[1],"offset":0},
            "short":{"tokens":[1,7,13,3],"offset":0},
            "future_changed":{"tokens":[1,7,13,17],"offset":0},
            "offset5":{"tokens":[7,13,3],"offset":5},
            "tile17":{"tokens":[1,7,13,3,1,7,13,3,1,7,13,3,1,7,13,3,1],"offset":0},
            "offset509":{"tokens":[7,13,3],"offset":509}})
    }
    pub fn tokens(value: &Value) -> Result<Vec<u32>> {
        let input = value.as_array().ok_or("token array required")?;
        if input.is_empty() || input.len() > 512 {
            return Err("token count outside diagnostic scope".into());
        }
        input
            .iter()
            .map(|v| {
                v.as_u64()
                    .filter(|&v| v < VOCAB as u64)
                    .map(|v| v as u32)
                    .ok_or_else(|| "token ID outside pinned vocabulary".into())
            })
            .collect()
    }
    pub fn reference(path: &Path, expected: &str) -> Result<Value> {
        if expected.len() != 64 || !expected.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err("external manifest SHA required".into());
        }
        let mut bytes = Vec::new();
        File::open(path)?
            .take(1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 1024 * 1024 || format!("{:x}", Sha256::digest(&bytes)) != expected {
            return Err("frozen reference identity differs".into());
        }
        let value: Value = serde_json::from_slice(&bytes)?;
        validate_reference(&value)?;
        Ok(value)
    }
    pub fn validate_reference(value: &Value) -> Result<()> {
        if value["format"] != "toklet-mlx-phi3-real-reference-v1"
            || value["model"]["repository"] != REPOSITORY
            || value["model"]["revision"] != REVISION
            || value["cases"] != cases()
            || value["layers"] != json!([0, 1, 31])
            || value["design_protocol_sha256"]
                != "1d8bfa6eb27fee9a35c5ea4bbe3bccdf6957008576aa97ec05cf3a6e1f22ce5e"
            || value["source_cpu_graph_executed"] != true
            || value["target_gpu_executed"] != false
        {
            return Err("fixed independent reference contract differs".into());
        }
        let selected = tokens(&value["greedy"]["fp64_tokens"])?;
        if selected.len() != 4 || value["greedy"]["prompt"] != cases()["short"]["tokens"] {
            return Err("independent teacher-forcing history differs".into());
        }
        Ok(())
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use super::contract::*;
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::sync::atomic::AtomicBool;
    use std::{
        fs::{self, File, OpenOptions},
        io::{BufWriter, Write},
        path::PathBuf,
    };
    use toklet::{
        generation::{self, Input, Request, StopReason},
        metal::{self, AttentionMode, NormalizationMode, ProjectionMode, ReadbackMode},
        session,
        weights::LoadOptions,
    };
    const TOTAL: usize = 6 * 1024 * 1024 * 1024;
    const APP: usize = 32 * 1024 * 1024;
    const OUTPUT: usize = 256 * 1024 * 1024;
    struct Capture {
        output: BufWriter<File>,
        journal: BufWriter<File>,
        arrays: Vec<Value>,
        bytes: usize,
    }
    impl Capture {
        fn new(path: &std::path::Path) -> Result<Self> {
            Ok(Self {
                output: BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?),
                journal: BufWriter::new(
                    OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(path.with_extension("jsonl"))?,
                ),
                arrays: Vec::new(),
                bytes: 0,
            })
        }
        fn save(
            &mut self,
            name: &str,
            start: usize,
            rows: usize,
            columns: usize,
            values: &[f32],
        ) -> Result<()> {
            let bytes = values.len().checked_mul(4).ok_or("capture size overflow")?;
            if values.len() != rows.checked_mul(columns).ok_or("capture shape overflow")?
                || self.bytes.checked_add(bytes).is_none_or(|n| n > OUTPUT)
                || self.arrays.len() >= 1288
            {
                return Err("bounded capture shape/count differs".into());
            }
            let started = json!({"event":"started","name":name,"row_start":start,
                "shape":[rows,columns],"offset":self.bytes,"expected_bytes":bytes});
            serde_json::to_writer(&mut self.journal, &started)?;
            self.journal.write_all(b"\n")?;
            self.journal.flush()?;
            let mut hash = Sha256::new();
            for (index, value) in values.iter().enumerate() {
                if !value.is_finite() {
                    serde_json::to_writer(
                        &mut self.journal,
                        &json!({"event":"nonfinite","name":name,
                        "row_start":start,"local_index":index,"raw_u32_bits":value.to_bits()}),
                    )?;
                    self.journal.write_all(b"\n")?;
                    self.journal.flush()?;
                    self.output.flush()?;
                    return Err("nonfinite diagnostic output".into());
                }
                let word = value.to_le_bytes();
                hash.update(word);
                self.output.write_all(&word)?;
            }
            self.output.flush()?;
            let descriptor = json!({"event":"completed","name":name,"row_start":start,"shape":[rows,columns],"dtype":"<f4",
                "offset":self.bytes,"bytes":bytes,"sha256":format!("{:x}",hash.finalize())});
            serde_json::to_writer(&mut self.journal, &descriptor)?;
            self.journal.write_all(b"\n")?;
            self.journal.flush()?;
            self.arrays.push(descriptor);
            self.bytes += bytes;
            Ok(())
        }
    }
    fn make_session(
        model: &metal::Model,
        chunk: usize,
        offset: usize,
        readback: ReadbackMode,
    ) -> Result<metal::Session<'_>> {
        Ok(metal::Session::with_attention_mode(
            model,
            session::Options {
                capacity: 512 - offset,
                prefill_chunk_size: chunk,
                position_offset: offset,
                memory_limit_bytes: TOTAL - APP,
            },
            ProjectionMode::MlxAffine,
            NormalizationMode::Compensated,
            readback,
            AttentionMode::Reference,
        )?)
    }
    fn stage(d: &metal::Diagnostic) -> String {
        let name = match d.stage {
            "q" => "q_projection",
            "k" => "k_projection",
            "v" => "v_projection",
            "gate" => "gate_projection",
            "silu" => "gate_silu",
            "up" => "up_projection",
            "down" => "down_projection",
            other => other,
        };
        d.layer
            .map_or_else(|| name.to_owned(), |layer| format!("layers.{layer}.{name}"))
    }
    fn extra_owner_ledger() -> Result<Value> {
        // enable_diagnostics already admits GPU snapshots AND every F32 CPU
        // readback. Count only additional bounded owners here. save() streams
        // four-byte words; there is no complete serialization tensor copy.
        let input = 1024 * 1024;
        let parsed_reference = 8 * 1024 * 1024;
        let descriptor_tables = 8 * 1024 * 1024; // two conservative 4MiB tables
        let numeric_witnesses = 8 * VOCAB * 4;
        let histories = 64 * 1024;
        let io_buffers = 3 * 8192;
        let total = input
            + parsed_reference
            + descriptor_tables
            + numeric_witnesses
            + histories
            + io_buffers;
        if total > APP {
            return Err("extra capture owners exceed reserved allowance".into());
        }
        Ok(
            json!({"source_manifest_input_bytes":input,"parsed_reference_reserve_bytes":parsed_reference,
            "descriptor_tables_reserve_bytes":descriptor_tables,"maximum_descriptors":1288,
            "numeric_witnesses_bytes":numeric_witnesses,"history_reserve_bytes":histories,
            "io_buffers_bytes":io_buffers,"full_tensor_serialization_bytes":0,
            "snapshot_and_cpu_readback_already_in_session":true,"total_extra_owner_reserve_bytes":total,
            "reserved_bytes":APP,"allocator_and_process_peak_measured_separately":true}),
        )
    }

    fn generation_controls(
        model: &metal::Model,
        prompt: &[u32],
        expected: &[u32],
    ) -> Result<Value> {
        if expected.len() != 4 {
            return Err("four native own-history IDs required".into());
        }
        let mut records = Vec::new();
        for readback in [ReadbackMode::FullLogits, ReadbackMode::GreedyToken] {
            let mut session = make_session(model, 32, 0, readback)?;
            let flag = AtomicBool::new(false);
            let mut emitted = Vec::new();
            let report = generation::generate(
                &mut session,
                Request {
                    input: Input::Reset(prompt),
                    max_new_tokens: 4,
                    stop_tokens: &[],
                },
                &flag,
                |event| {
                    if event.index != emitted.len() {
                        return Err(std::io::Error::other("raw event order differs"));
                    }
                    emitted.push(event.token_id);
                    Ok(())
                },
            )?;
            if emitted != expected
                || report.reason != StopReason::OutputLimit
                || (
                    report.sampled_tokens,
                    report.emitted_tokens,
                    report.evaluated_tokens,
                    report.pending_tokens,
                ) != (4, 4, prompt.len() + 3, 1)
            {
                return Err("raw generation helper differs from manual native selection".into());
            }
            let before = session.history().to_vec();
            let position = (session.evaluated_len(), session.pending_len());
            let cancelled = generation::generate(
                &mut session,
                Request {
                    input: Input::Reset(&[7, 13, 3]),
                    max_new_tokens: 1,
                    stop_tokens: &[],
                },
                &AtomicBool::new(true),
                |_| Err(std::io::Error::other("cancelled output emitted")),
            )?;
            if cancelled.reason != StopReason::Cancelled
                || cancelled.sampled_tokens != 0
                || cancelled.emitted_tokens != 0
                || session.history() != before
                || (session.evaluated_len(), session.pending_len()) != position
            {
                return Err("pre-cancelled raw reset changed a valid state".into());
            }
            let failure = generation::generate(
                &mut session,
                Request {
                    input: Input::Reset(prompt),
                    max_new_tokens: 1,
                    stop_tokens: &[],
                },
                &flag,
                |_| Err(std::io::Error::other("injected callback failure")),
            );
            if !matches!(failure, Err(generation::Error::Output(_)))
                || session.evaluated_len() != prompt.len()
                || session.pending_len() != 1
                || session.history().last() != Some(&expected[0])
            {
                return Err("callback failure did not preserve valid committed prefix".into());
            }
            session.discard_pending();
            if session.history() != prompt
                || session
                    .greedy_token()
                    .ok_or("missing restored selection")??
                    != expected[0]
            {
                return Err("callback failure recovery changed the raw prefix".into());
            }
            let stopped = generation::generate(
                &mut session,
                Request {
                    input: Input::Reset(prompt),
                    max_new_tokens: 4,
                    stop_tokens: &expected[..1],
                },
                &flag,
                |_| Err(std::io::Error::other("forced stop emitted")),
            )?;
            if stopped.reason
                != (StopReason::StopToken {
                    token_id: expected[0],
                })
                || (
                    stopped.sampled_tokens,
                    stopped.emitted_tokens,
                    stopped.evaluated_tokens,
                    stopped.pending_tokens,
                ) != (1, 0, prompt.len(), 1)
            {
                return Err("explicit raw stop control differs".into());
            }
            records.push(json!({"readback":format!("{readback:?}"),"generated_ids":emitted,
                "output_limit":true,"pre_cancel_preserved_state":true,"callback_failure_valid_prefix":true,
                "forced_stop_id":expected[0],"natural_model_eos_tested":false,"text_chat_qualified":false}));
        }
        Ok(Value::Array(records))
    }

    fn resource_plans(path: &std::path::Path) -> Result<(Vec<metal::planning::MlxReport>, Value)> {
        use objc2_metal_performance_shaders::{MPSDataType, MPSMatrixDescriptor};
        // SAFETY: fixed nonzero3072 columns and concrete Float16 dtype; only
        // queries the framework layout recommendation, no tensor pointer access.
        let row_bytes =
            unsafe { MPSMatrixDescriptor::rowBytesForColumns_dataType(3072, MPSDataType::Float16) };
        if row_bytes < 6144 || !row_bytes.is_multiple_of(2) {
            return Err("invalid queried norm stride".into());
        }
        let resident = 2149244928usize
            .checked_add(
                65usize
                    .checked_mul(row_bytes.checked_add(14).ok_or("guard overflow")?)
                    .ok_or("norm size overflow")?,
            )
            .ok_or("resident size overflow")?;
        let mut plans = Vec::new();
        for chunk in [32, 128] {
            for readback in [ReadbackMode::FullLogits, ReadbackMode::GreedyToken] {
                let report = metal::planning::inspect_mlx(
                    path,
                    LoadOptions {
                        context_limit: 512,
                        memory_limit_bytes: TOTAL - APP,
                    },
                    session::Options {
                        capacity: 512,
                        prefill_chunk_size: chunk,
                        position_offset: 0,
                        memory_limit_bytes: TOTAL - APP,
                    },
                    NormalizationMode::Compensated,
                    readback,
                    AttentionMode::Reference,
                )?;
                if report.source_payload_bytes != 2149644288
                    || report.source_admitted_load_bytes != 2216818688
                    || report.resident_weights_bytes != resident
                    || report.positional_resident_bytes != 0
                    || report.resident_admitted_load_bytes
                        != resident + 64 * 1024 * 1024 + 64 * 1024
                    || !report.fits_requested_budgets
                    || report.numerical_buffers_allocated
                    || !report.payload_validation_required
                    || report.qualification_evaluated
                {
                    return Err("real cold source/resident/session plan differs".into());
                }
                println!("RESOURCE_PLAN {}", serde_json::to_string(&report)?);
                std::io::stdout().flush()?;
                plans.push(report);
            }
        }
        Ok((
            plans,
            json!({"queried_half_norm_row_bytes":row_bytes,"packed_resident_bytes":2149244928usize,
            "norm_count":65,"guard_bytes_per_norm":14,"resident_weights_bytes":resident}),
        ))
    }
    fn session_budget_controls(
        model: &metal::Model,
        plans: &[metal::planning::MlxReport],
    ) -> Result<Value> {
        let mut records = Vec::new();
        for plan in plans {
            if model.weights_bytes() != plan.resident_weights_bytes {
                return Err("actual resident weight accounting differs".into());
            }
            let required = plan.session_memory.admitted_bytes;
            let options = session::Options {
                capacity: 512,
                prefill_chunk_size: plan.requested_chunk_size,
                position_offset: 0,
                memory_limit_bytes: required,
            };
            {
                let session = metal::Session::with_attention_mode(
                    model,
                    options,
                    ProjectionMode::MlxAffine,
                    NormalizationMode::Compensated,
                    plan.readback,
                    AttentionMode::Reference,
                )?;
                if session.memory() != plan.session_memory {
                    return Err("exact-budget session memory differs".into());
                }
            }
            let rejected = metal::Session::with_attention_mode(
                model,
                session::Options {
                    memory_limit_bytes: required - 1,
                    ..options
                },
                ProjectionMode::MlxAffine,
                NormalizationMode::Compensated,
                plan.readback,
                AttentionMode::Reference,
            );
            if !matches!(rejected,Err(metal::Error::Budget{required:r,limit}) if r==required && limit==required-1)
            {
                return Err("one-byte-under session did not return exact budget error".into());
            }
            let record = json!({"chunk":plan.requested_chunk_size,"readback":format!("{:?}",plan.readback),
                "required":required,"exact_budget_constructed":true,"one_byte_under_exact_budget_error":true});
            println!("BUDGET {record}");
            std::io::stdout().flush()?;
            records.push(record);
        }
        Ok(Value::Array(records))
    }

    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args_os().skip(1).collect();
        if args.len() != 4 {
            return Err(
                "usage: phi3_checkpoint MODEL REFERENCE-MANIFEST MANIFEST-SHA NEW-OUTPUT".into(),
            );
        }
        let extra_owners = extra_owner_ledger()?;
        let model_path = PathBuf::from(&args[0]);
        let source = PathBuf::from(&args[1]);
        let source_sha = args[2].to_str().ok_or("manifest SHA must be UTF-8")?;
        let reference = reference(&source, source_sha)?;
        for &(name, size, sha) in FILES {
            hash_file(&model_path.join(name), size, sha)?;
        }
        // Source window facts are read and validated before numerical loading;
        // this short-lived source retains no expanded tensors or numerical KV.
        {
            let metadata = toklet::mlx::qwen::Source::open_dense(
                &model_path,
                LoadOptions {
                    context_limit: 512,
                    memory_limit_bytes: TOTAL - APP,
                },
            )?;
            if metadata.declared_positions() != 4096
                || metadata.declared_sliding_window() != Some(2047)
                || metadata.architecture() != toklet::mlx::dense::Architecture::Phi3
                || metadata.tied_output()
                || metadata.config().max_positions != 512
            {
                return Err("pinned Phi source semantics differ".into());
            }
        }
        let out = PathBuf::from(&args[3]);
        fs::create_dir(&out)?;
        let mut capture = Capture::new(&out.join("arrays.bin"))?;
        let (resource_reports, resident_ledger) = resource_plans(&model_path)?;
        let exact_load_budget = resource_reports
            .iter()
            .map(|r| {
                r.source_admitted_load_bytes
                    .max(r.resident_admitted_load_bytes)
            })
            .max()
            .ok_or("missing resource plans")?;
        let under = metal::Model::load_mlx(
            &model_path,
            LoadOptions {
                context_limit: 512,
                memory_limit_bytes: exact_load_budget - 1,
            },
        );
        if !matches!(under,Err(metal::Error::Budget{required,limit}) if required==exact_load_budget && limit==exact_load_budget-1)
        {
            return Err("one-byte-under model did not return exact budget error".into());
        }
        println!(
            "LOAD_BUDGET {}",
            json!({"required":exact_load_budget,"one_byte_under_exact_budget_error":true})
        );
        std::io::stdout().flush()?;
        let model = metal::Model::load_mlx(
            &model_path,
            LoadOptions {
                context_limit: 512,
                memory_limit_bytes: exact_load_budget,
            },
        )?;
        let budget_records = session_budget_controls(&model, &resource_reports)?;
        if model.mlx_declared_positions() != Some(4096)
            || model.positional_resident_bytes() != 0
            || model.mlx_architecture() != Some(toklet::mlx::dense::Architecture::Phi3)
        {
            return Err("loaded positional identity differs".into());
        }
        let mut admitted = 0;
        for (name, case) in cases().as_object().unwrap() {
            let ids = tokens(&case["tokens"])?;
            let offset = case["offset"].as_u64().unwrap() as usize;
            for chunk in [32, 128, 3] {
                let mut session = make_session(
                    &model,
                    chunk.min(512 - offset),
                    offset,
                    ReadbackMode::FullLogits,
                )?;
                session.enable_diagnostics(&[0, 1, 31])?;
                admitted = admitted.max(session.memory().admitted_bytes + APP);
                session.set_tokens(&ids)?;
                let mut start = 0;
                loop {
                    let rows = session.evaluate_next_chunk()?;
                    if rows == 0 {
                        break;
                    }
                    let diagnostics = session.diagnostics()?;
                    if diagnostics.len() != 51 {
                        return Err("selected graph stage count differs".into());
                    }
                    for d in diagnostics {
                        if d.rows != rows {
                            return Err("diagnostic chunk rows differ".into());
                        }
                        capture.save(
                            &format!("chunk{chunk}.cases.{name}.{}", stage(&d)),
                            start,
                            rows,
                            d.columns,
                            &d.values,
                        )?;
                    }
                    start += rows;
                }
                if start != ids.len() {
                    return Err("incomplete prompt commitment".into());
                }
                let before = session.logits().ok_or("distribution missing")?.to_vec();
                session.set_tokens(&ids)?;
                if session.evaluate_next_chunk()? != 0
                    || session.logits() != Some(before.as_slice())
                {
                    return Err("cached prompt changed distribution".into());
                }
            }
        }
        let prompt = tokens(&reference["greedy"]["prompt"])?;
        let teacher = tokens(&reference["greedy"]["fp64_tokens"])?;
        let mut native_selected = Vec::new();
        for readback in [ReadbackMode::FullLogits, ReadbackMode::GreedyToken] {
            let mut session = make_session(&model, 32, 0, readback)?;
            session.set_tokens(&prompt)?;
            for (step, &token) in teacher.iter().enumerate() {
                session.evaluate()?;
                let selected = session.greedy_token().ok_or("missing selection")??;
                match readback {
                    ReadbackMode::FullLogits => {
                        native_selected.push(selected);
                        capture.save(
                            &format!("greedy.step_{step}.logits"),
                            0,
                            1,
                            VOCAB,
                            session.logits().ok_or("full logits missing")?,
                        )?;
                    }
                    ReadbackMode::GreedyToken => {
                        if session.logits().is_some() || selected != native_selected[step] {
                            return Err("native readback selection differs".into());
                        }
                    }
                }
                session.append(&[token])?;
                if session.pending_len() != 1 || session.greedy_token().is_some() {
                    return Err("pending selection exposed".into());
                }
                session.discard_pending();
                if session.pending_len() != 0 {
                    return Err("pending discard failed".into());
                }
                session.append(&[token])?;
            }
            admitted = admitted.max(session.memory().admitted_bytes + APP);
        }
        for count in 1..4 {
            let mut session = make_session(&model, 32, 0, ReadbackMode::FullLogits)?;
            session.set_tokens(&prompt[..count])?;
            session.evaluate()?;
            capture.save(
                &format!("scoring.prefix_{count}.logits"),
                0,
                1,
                VOCAB,
                session.logits().ok_or("scoring distribution missing")?,
            )?;
        }
        // A separate four-step RAW native journey selects its own history.
        // It has no text/template/stop or source-profile quality claim.
        let mut generated = Vec::new();
        for readback in [ReadbackMode::FullLogits, ReadbackMode::GreedyToken] {
            let mut session = make_session(&model, 32, 0, readback)?;
            session.set_tokens(&prompt)?;
            for step in 0..4 {
                session.evaluate()?;
                let selected = session
                    .greedy_token()
                    .ok_or("native journey selection missing")??;
                match readback {
                    ReadbackMode::FullLogits => {
                        generated.push(selected);
                        capture.save(
                            &format!("native_generated.step_{step}.logits"),
                            0,
                            1,
                            VOCAB,
                            session.logits().ok_or("native journey logits missing")?,
                        )?;
                    }
                    ReadbackMode::GreedyToken => {
                        if session.logits().is_some() || selected != generated[step] {
                            return Err("native generated readbacks diverged".into());
                        }
                    }
                }
                session.append(&[selected])?;
                if session.evaluated_len() != prompt.len() + step || session.pending_len() != 1 {
                    return Err("native sampled/committed history differs".into());
                }
            }
            admitted = admitted.max(session.memory().admitted_bytes + APP);
        }
        let generation_records = generation_controls(&model, &prompt, &generated)?;
        let boundary: Vec<u32> = (0..512).map(|i| prompt[i % prompt.len()]).collect();
        let mut boundary_distribution = None;
        let mut boundary_records = Vec::new();
        for chunk in [32, 128] {
            let mut session = make_session(&model, chunk, 0, ReadbackMode::FullLogits)?;
            session.set_tokens(&boundary)?;
            let committed = session.evaluate_next_chunk()?;
            session.discard_pending();
            if session.evaluated_len() != committed || session.history() != &boundary[..committed] {
                return Err("partial prefill discard changed committed prefix".into());
            }
            session.set_tokens(&boundary)?;
            session.evaluate()?;
            let before = session
                .logits()
                .ok_or("boundary distribution missing")?
                .to_vec();
            let before_history = session.history().to_vec();
            let before_position = (session.evaluated_len(), session.pending_len());
            let append_result = session.append(&[1]);
            let append_state_equal = session.history() == before_history
                && (session.evaluated_len(), session.pending_len()) == before_position
                && session.logits() == Some(before.as_slice());
            let set_result = session.set_tokens(&[1; 513]);
            let set_state_equal = session.history() == before_history
                && (session.evaluated_len(), session.pending_len()) == before_position
                && session.logits() == Some(before.as_slice());
            // Persist both actual rows before a possible cross-schedule failure.
            capture.save(
                &format!("boundary.chunk{chunk}.logits"),
                511,
                1,
                VOCAB,
                &before,
            )?;
            let boundary_record = json!({"chunk":chunk,"append_error":format!("{append_result:?}"),
                "set_error":format!("{set_result:?}"),"append_state_equal":append_state_equal,
                "set_state_equal":set_state_equal,"evaluated":before_position.0,"pending":before_position.1});
            println!("STATE {boundary_record}");
            std::io::stdout().flush()?;
            boundary_records.push(boundary_record);
            if !matches!(
                append_result,
                Err(metal::Error::Options(
                    "token ID/count outside session capacity"
                ))
            ) || !matches!(
                set_result,
                Err(metal::Error::Options(
                    "token ID/count outside session capacity"
                ))
            ) || !append_state_equal
                || !set_state_equal
                || before_history != boundary
                || before_position != (512, 0)
            {
                return Err("boundary capacity error or state differs".into());
            }
            if let Some(expected) = &boundary_distribution {
                if expected != &before {
                    return Err("boundary schedules differ".into());
                }
            } else {
                boundary_distribution = Some(before.clone());
            }
            session.reset();
            if session.evaluated_len() != 0 || !session.history().is_empty() {
                return Err("reset retained state".into());
            }
            admitted = admitted.max(session.memory().admitted_bytes + APP);
        }
        if admitted > TOTAL {
            return Err("combined diagnostic admission exceeds budget".into());
        }
        if capture.arrays.len() != 1288 {
            return Err("fixed native descriptor inventory differs".into());
        }
        capture.output.flush()?;
        // Capture is observational. A separate source-bound checker must qualify
        // every value; successful execution alone is not a numerical pass.
        let report = json!({"format":"toklet-phi3-checkpoint-capture-v1","repository":REPOSITORY,"revision":REVISION,
            "source_manifest_sha256":source_sha,"device":model.device_name(),"declared_positions":model.mlx_declared_positions(),
            "implemented_positions":512,"declared_sliding_window":2047,"profile":"MlxAffine/Compensated/Reference/Half","requested_chunks":[32,128,3],"effective_chunk":"min(requested,512-position_offset)",
            "resource_plans":resource_reports,"resident_owner_ledger":resident_ledger,"exact_load_budget":exact_load_budget,"one_byte_under_load_exact_budget_error":true,"session_budget_controls":budget_records,"weights_bytes":model.weights_bytes(),"positional_bytes":model.positional_resident_bytes(),
            "load_timing":{"load_seconds":model.load_timing.load_seconds,"conversion_seconds":model.load_timing.conversion_seconds,"compilation_seconds":model.load_timing.compilation_seconds},"maximum_admitted_bytes":admitted,"application_allowance_bytes":APP,
            "native_selected_on_teacher_forced_prefixes":native_selected,"array_bytes":capture.bytes,"arrays":capture.arrays,
            "native_generated_ids":generated,"boundary_controls":boundary_records,"generation_controls":generation_records,"extra_owner_ledger":extra_owners,"native_generation_scope":"four raw IDs; no text/stop/quality qualification",
            "held_out_quality_qualified":false,"app_qualified":false,"iphone_qualified":false,"numerical_comparison_pending":true});
        let mut manifest = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join("capture.json"))?;
        serde_json::to_writer_pretty(&mut manifest, &report)?;
        manifest.write_all(b"\n")?;
        println!(
            "{}",
            json!({"capture_completed":true,"array_bytes":capture.bytes,"maximum_admitted_bytes":admitted,"qualification_pending":true})
        );
        Ok(())
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn interrupted_array_keeps_flushed_index_of_completed_and_started_data() {
            let path =
                std::env::temp_dir().join(format!("toklet-phi3-capture-{}", std::process::id()));
            let mut capture = Capture::new(&path).unwrap();
            capture.save("complete", 0, 1, 1, &[1.]).unwrap();
            assert!(
                capture
                    .save("interrupted", 0, 1, 2, &[2., f32::NAN])
                    .is_err()
            );
            drop(capture);
            let journal = fs::read_to_string(path.with_extension("jsonl")).unwrap();
            let records: Vec<Value> = journal
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            assert_eq!(records.len(), 4);
            assert_eq!(records[0]["event"], "started");
            assert_eq!(records[1]["event"], "completed");
            assert_eq!(records[1]["offset"], 0);
            assert_eq!(records[1]["bytes"], 4);
            assert_eq!(records[2]["event"], "started");
            assert_eq!(records[2]["offset"], 4);
            assert_eq!(records[2]["expected_bytes"], 8);
            assert_eq!(records[3]["event"], "nonfinite");
            assert_eq!(records[3]["local_index"], 1);
            assert_eq!(records[3]["raw_u32_bits"], f32::NAN.to_bits());
            let raw = fs::read(&path).unwrap();
            assert_eq!(&raw[..4], &1f32.to_le_bytes());
            assert_eq!(raw.len(), 8);
            fs::remove_file(&path).unwrap();
            fs::remove_file(path.with_extension("jsonl")).unwrap();
        }
    }
}
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() -> contract::Result<()> {
    apple::run()
}
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("phi3_checkpoint requires an Apple Silicon Mac with Metal");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::contract::*;
    use serde_json::json;
    #[test]
    fn source_contract_and_token_boundaries_reject_before_loading() {
        let valid = json!({"format":"toklet-mlx-phi3-real-reference-v1","model":{"repository":REPOSITORY,"revision":REVISION},
            "design_protocol_sha256":"1d8bfa6eb27fee9a35c5ea4bbe3bccdf6957008576aa97ec05cf3a6e1f22ce5e","source_cpu_graph_executed":true,"target_gpu_executed":false,"cases":cases(),"layers":[0,1,31],"greedy":{"prompt":[1,7,13,3],"fp64_tokens":[1,2,3,4]}});
        validate_reference(&valid).unwrap();
        for (key, value) in [
            ("cases", json!({})),
            ("layers", json!([0, 1, 14])),
            ("format", json!("other")),
            ("model", json!({"repository":REPOSITORY,"revision":"main"})),
            (
                "greedy",
                json!({"prompt":[1,7,13,3],"fp64_tokens":[1,2,3,32064]}),
            ),
        ] {
            let mut changed = valid.clone();
            changed[key] = value;
            assert!(validate_reference(&changed).is_err());
        }
        for ids in [
            json!([]),
            json!([1.5]),
            json!([-1]),
            json!([32064]),
            json!(vec![1; 513]),
        ] {
            assert!(tokens(&ids).is_err());
        }
        assert_eq!(tokens(&json!([0, 32063])).unwrap(), [0, 32063]);
    }
    #[test]
    fn file_and_manifest_integrity_fail_closed_without_model_download() {
        let path =
            std::env::temp_dir().join(format!("toklet-phi3-contract-{}", std::process::id()));
        let raw = b"abc";
        {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            file.write_all(raw).unwrap();
        }
        let sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        hash_file(&path, 3, sha).unwrap();
        assert!(hash_file(&path, 2, sha).is_err());
        assert!(hash_file(&path, 3, &"0".repeat(64)).is_err());
        assert!(reference(&path, sha).is_err());
        assert!(reference(&path, "main").is_err());
        std::fs::remove_file(path).unwrap();
    }
}
