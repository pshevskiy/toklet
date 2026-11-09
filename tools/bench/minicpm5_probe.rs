//! Explicit pinned MiniCPM5 diagnostic. No application admission or kernel changes.
mod artifact {
    use sha2::{Digest, Sha256};
    use std::{fs::File, io::Read, path::Path};
    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    pub const REPOSITORY: &str = "bumblebuttpow/MiniCPM5-2B-heretic-abliterated-MLX-8bit";
    pub const REVISION: &str = "50ab1b226834c290cdf68e460eb899e5cff7c61d";
    pub const VOCAB: usize = 130560;
    pub const FILES: &[(&str, u64, &str)] = &[
        (
            "config.json",
            1056,
            "ba1fda219aca07e7809d454db02512fe3fb29ff49e1eb2364c8e9487e35a6fe3",
        ),
        (
            "model.safetensors.index.json",
            68721,
            "50747e37090508f851cdea2f4d5a71c59c2a26a09c8e95c505371ac669e2ac36",
        ),
        (
            "model.safetensors",
            2674327346,
            "6fadca0fcdefe8b77fc144851ee2c2fd3c8949ccc0ec756b8ada27e3156c8910",
        ),
    ];
    pub fn hash_file(path: &Path, size: u64, expected: &str) -> Result<()> {
        let metadata = path.symlink_metadata()?;
        if !metadata.is_file() || metadata.len() != size {
            return Err("pinned regular file size differs".into());
        }
        let mut f = File::open(path)?;
        let mut hasher = Sha256::new();
        let mut bytes = [0; 64 * 1024];
        let mut count = 0_u64;
        loop {
            let n = f.read(&mut bytes)?;
            if n == 0 {
                break;
            }
            count += n as u64;
            if count > size {
                return Err("pinned file grew".into());
            }
            hasher.update(&bytes[..n]);
        }
        if count != size || format!("{:x}", hasher.finalize()) != expected {
            return Err("pinned file digest differs".into());
        }
        Ok(())
    }
}
#[cfg(any(test, all(target_os = "macos", target_arch = "aarch64")))]
mod contract {
    pub use super::artifact::{FILES, REPOSITORY, REVISION, Result, VOCAB, hash_file};
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::{fs::File, io::Read, path::Path};
    pub fn cases() -> Value {
        json!({"single":{"tokens":[0],"position_offset":0},"short":{"tokens":[0,7,13,3],"position_offset":0},"future_changed":{"tokens":[0,7,13,17],"position_offset":0},"offset5":{"tokens":[7,13,3],"position_offset":5},"tile17":{"tokens":[0,7,13,3,0,7,13,3,0,7,13,3,0,7,13,3,0],"position_offset":0},"offset509":{"tokens":[7,13,3],"position_offset":509}})
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
    pub fn teacher(value: &Value) -> Result<Vec<u32>> {
        let rows = value["greedy"].as_array().ok_or("teacher rows required")?;
        if rows.len() != 4 {
            return Err("four teacher rows required".into());
        }
        let mut prefix = tokens(&cases()["short"]["tokens"])?;
        let mut selected = Vec::new();
        for row in rows {
            if row.as_object().map_or(0, |o| o.len()) != 3 || tokens(&row["prefix"])? != prefix {
                return Err("teacher prefix differs".into());
            }
            let id = tokens(&json!([row["fp64_id"]]))?[0];
            tokens(&json!([row["fp32_id"]]))?;
            selected.push(id);
            prefix.push(id);
        }
        Ok(selected)
    }
    pub fn validate_reference(value: &Value) -> Result<()> {
        if value["format"] != "toklet-minicpm5-independent-short-v1"
            || value["model"]["repository"] != REPOSITORY
            || value["model"]["revision"] != REVISION
            || value["cases"] != cases()
            || value["layers"] != json!([0, 1, 41])
            || value["arrays"].as_object().map_or(0, |o| o.len()) != 698
            || value["native_outcomes_seen"].as_bool() != Some(false)
            || value["model_quality_qualified"].as_bool() != Some(false)
            || value["model_adoption_qualified"].as_bool() != Some(false)
        {
            return Err("fixed independent reference contract differs".into());
        }
        let geometry = &value["geometry"];
        for (key, expected) in [
            ("vocab_size", 130560),
            ("hidden_size", 2048),
            ("num_layers", 42),
            ("num_query_heads", 16),
            ("num_kv_heads", 2),
            ("head_dim", 128),
            ("intermediate_size", 6144),
        ] {
            if geometry[key].as_u64() != Some(expected) {
                return Err("reference geometry differs".into());
            }
        }
        if geometry["rms_norm_eps"].as_f64() != Some(1e-6)
            || geometry["rope_theta"].as_f64() != Some(5000000.)
            || value["policy"]["atol_floor"].as_f64() != Some(1e-4)
            || value["policy"]["envelope_multiplier"].as_u64() != Some(2)
            || value["policy"]["rtol"].as_f64() != Some(1e-3)
        {
            return Err("reference geometry/policy differs".into());
        }
        if value["model"]["files"].as_object().map_or(0, |o| o.len()) != FILES.len() {
            return Err("reference file inventory differs".into());
        }
        for &(name, _, pin) in FILES {
            if value["model"]["files"][name] != pin {
                return Err("reference source pin differs".into());
            }
        }
        teacher(value)?;
        Ok(())
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use super::contract::*;
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::{
        fs::{self, File, OpenOptions},
        io::{BufWriter, Write},
        path::PathBuf,
    };
    use toklet::{
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
                || self.arrays.len() >= 10000
            {
                return Err("bounded capture shape/count differs".into());
            }
            let started = json!({"event":"started","name":name,"row_start":start,
                "shape":[rows,columns],"offset":self.bytes,"expected_bytes":bytes});
            serde_json::to_writer(&mut self.journal, &started)?;
            self.journal.write_all(b"\n")?;
            self.journal.flush()?;
            let mut hash = Sha256::new();
            for value in values {
                if !value.is_finite() {
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
    #[derive(Clone, Copy)]
    struct Step {
        rows: usize,
        committed_after: usize,
        pending_after: usize,
        wall_nanoseconds: u64,
        sampling_nanoseconds: u64,
        decision_nanoseconds: u64,
        selected: u32,
        full_logits_available: bool,
        timing: metal::Timing,
    }

    /// Loaded-model, empty-cache raw requests. JSON and reset are outside each
    /// request clock; completed evaluation and token selection are inside it.
    pub fn bench() -> Result<()> {
        let args: Vec<_> = std::env::args_os().skip(2).collect();
        if args.len() != 5 {
            return Err("usage: minicpm5_probe bench MODEL REFERENCE SHA NEW-OUTPUT full-logits|greedy-token".into());
        }
        let model_path = PathBuf::from(&args[0]);
        let source = PathBuf::from(&args[1]);
        let source_sha = args[2].to_str().ok_or("manifest SHA must be UTF-8")?;
        let reference = reference(&source, source_sha)?;
        let (mode_name, readback) = match args[4].to_str() {
            Some("full-logits") => ("full-logits", ReadbackMode::FullLogits),
            Some("greedy-token") => ("greedy-token", ReadbackMode::GreedyToken),
            _ => return Err("benchmark readback must be full-logits or greedy-token".into()),
        };
        for &(name, size, pin) in FILES {
            hash_file(&model_path.join(name), size, pin)?;
        }
        let out = PathBuf::from(&args[3]);
        fs::create_dir(&out)?;
        let model = metal::Model::load_mlx(
            &model_path,
            LoadOptions {
                context_limit: 512,
                memory_limit_bytes: TOTAL - APP,
            },
        )?;
        if model.weights_bytes() != 2674218150 || model.positional_resident_bytes() != 0 {
            return Err("benchmark resident identity differs".into());
        }
        let mut session = make_session(&model, 32, 0, readback)?;
        if session.memory().admitted_bytes + APP > TOTAL {
            return Err("benchmark admission exceeds6GiB".into());
        }
        let prompt = tokens(&cases()["short"]["tokens"])?;
        let teacher = teacher(&reference)?;
        let mut samples = Vec::with_capacity(13);
        for index in 0..13 {
            session.reset();
            if session.evaluated_len() != 0
                || session.pending_len() != 0
                || !session.history().is_empty()
            {
                return Err("benchmark reset retained valid history".into());
            }
            let mut steps: [Option<Step>; 4] = [None; 4];
            let request = std::time::Instant::now();
            session.set_tokens(&prompt)?;
            for (step, &token) in teacher.iter().enumerate() {
                let phase = std::time::Instant::now();
                let rows = session.evaluate_next_chunk()?;
                let wall_nanoseconds = u64::try_from(phase.elapsed().as_nanos())?;
                if rows != if step == 0 { 4 } else { 1 } || session.pending_len() != 0 {
                    return Err("benchmark completed work differs".into());
                }
                let timing = session
                    .timing()
                    .ok_or("benchmark completed timing missing")?;
                let sampling = std::time::Instant::now();
                let selected = session
                    .greedy_token()
                    .ok_or("benchmark selection missing")??;
                let sampling_nanoseconds = u64::try_from(sampling.elapsed().as_nanos())?;
                let decision_nanoseconds = u64::try_from(request.elapsed().as_nanos())?;
                steps[step] = Some(Step {
                    rows,
                    committed_after: session.evaluated_len(),
                    pending_after: session.pending_len(),
                    wall_nanoseconds,
                    sampling_nanoseconds,
                    decision_nanoseconds,
                    selected,
                    full_logits_available: session.logits().is_some(),
                    timing,
                });
                // Fixed teacher forcing includes the final stop ID. It measures
                // identical work, rather than different self-selected histories.
                session.append(&[token])?;
            }
            let request_nanoseconds = u64::try_from(request.elapsed().as_nanos())?;
            if session.evaluated_len() != 7
                || session.pending_len() != 1
                || session.history().len() != 8
            {
                return Err("benchmark sampled/committed history differs".into());
            }
            let mut records = Vec::with_capacity(4);
            for (step, entry) in steps.into_iter().enumerate() {
                let entry = entry.ok_or("benchmark phase omitted")?;
                records.push(json!({"step":step,"rows":entry.rows,
                    "committed_after":entry.committed_after,"pending_after":entry.pending_after,
                    "wall_nanoseconds":entry.wall_nanoseconds,"sampling_nanoseconds":entry.sampling_nanoseconds,
                    "decision_nanoseconds":entry.decision_nanoseconds,"selected":entry.selected,
                    "full_logits_available":entry.full_logits_available,"completed_timing":entry.timing}));
            }
            samples.push(json!({"sample":index,"warmup":index<3,
                "request_nanoseconds":request_nanoseconds,"steps":records,
                "committed_after":session.evaluated_len(),"pending_after":session.pending_len(),
                "history_after":session.history().len(),"outcome":"fixed_teacher_continuation_completed"}));
        }
        for &(name, size, pin) in FILES {
            hash_file(&model_path.join(name), size, pin)?;
        }
        let report = json!({"format":"toklet-minicpm5-raw-request-bench-v1",
            "repository":REPOSITORY,"revision":REVISION,"source_manifest_sha256":source_sha,
            "device":model.device_name(),"profile":"MlxAffine/Compensated/Reference/Half",
            "readback":mode_name,"context_capacity":512,"prefill_chunk":32,"position_offset":0,
            "prompt":prompt,"teacher_tokens":teacher,"warmups":3,"measured_samples":10,
            "cache_state":"loaded-model;reset-empty-history-per-request;warm-filesystem",
            "stop_policy":"fixed-four-teacher-tokens-including-stop-id;last-token-pending",
            "load_timing":{"load_seconds":model.load_timing.load_seconds,
                "conversion_seconds":model.load_timing.conversion_seconds,
                "compilation_seconds":model.load_timing.compilation_seconds},
            "session_memory":session.memory(),"application_allowance_bytes":APP,
            "request_clock_scope":"set_tokens through final pending append; completed evaluation and selection; excludes reset and JSON",
            "decision_ttft_scope":"internal raw token decision; not streamed text or visible UI",
            "diagnostics_enabled":false,"operator_counts_available":false,
            "allocation_count":null,"user_visible_ttft_nanoseconds":null,
            "samples":samples,"model_quality_qualified":false,"managed_adoption_qualified":false,
            "ordinary_telemetry_overhead_qualified":false});
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join("bench.json"))?;
        serde_json::to_writer_pretty(&mut output, &report)?;
        output.write_all(b"\n")?;
        println!(
            "{}",
            json!({"benchmark_completed":true,"readback":mode_name,"samples":13,"quality_qualified":false})
        );
        Ok(())
    }

    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args_os().skip(1).collect();
        if args.len() != 4 {
            return Err(
                "usage: minicpm5_probe MODEL REFERENCE-MANIFEST MANIFEST-SHA NEW-OUTPUT".into(),
            );
        }
        let model_path = PathBuf::from(&args[0]);
        let source = PathBuf::from(&args[1]);
        let source_sha = args[2].to_str().ok_or("manifest SHA must be UTF-8")?;
        let reference = reference(&source, source_sha)?;
        for &(name, size, sha) in FILES {
            hash_file(&model_path.join(name), size, sha)?;
        }
        let out = PathBuf::from(&args[3]);
        fs::create_dir(&out)?;
        let mut capture = Capture::new(&out.join("arrays.bin"))?;
        let model = metal::Model::load_mlx(
            &model_path,
            LoadOptions {
                context_limit: 512,
                memory_limit_bytes: TOTAL - APP,
            },
        )?;
        if model.mlx_declared_positions() != Some(131072) || model.positional_resident_bytes() != 0
        {
            return Err("loaded positional identity differs".into());
        }
        let mut admitted = 0;
        let mut chunk_metrics = Vec::new();
        for (name, case) in cases().as_object().unwrap() {
            let ids = tokens(&case["tokens"])?;
            let offset = case["position_offset"].as_u64().unwrap() as usize;
            for chunk in [1, 3, 32] {
                let mut session = make_session(
                    &model,
                    chunk.min(512 - offset),
                    offset,
                    ReadbackMode::FullLogits,
                )?;
                session.enable_diagnostics(&[0, 1, 41])?;
                admitted = admitted.max(session.memory().admitted_bytes + APP);
                session.set_tokens(&ids)?;
                let mut start = 0;
                loop {
                    let clock = std::time::Instant::now();
                    let rows = session.evaluate_next_chunk()?;
                    let wall_nanoseconds = clock.elapsed().as_nanos();
                    if rows == 0 {
                        break;
                    }
                    let timing = session.timing().ok_or("completed chunk timing missing")?;
                    chunk_metrics.push(json!({"case":name,"requested_chunk":chunk,
                        "rows":rows,"position_offset":offset,"committed_before":start,
                        "committed_after":session.evaluated_len(),"pending_after":session.pending_len(),
                        "wall_nanoseconds":wall_nanoseconds,"completed_timing":timing,
                        "session_memory":session.memory()}));
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
        let prompt = tokens(&cases()["short"]["tokens"])?;
        let teacher = teacher(&reference)?;
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
        let boundary: Vec<u32> = (0..512).map(|i| prompt[i % prompt.len()]).collect();
        let mut boundary_distribution = None;
        for chunk in [3, 32] {
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
            if session.append(&[1]).is_ok()
                || session.set_tokens(&[1; 513]).is_ok()
                || session.history() != boundary
                || session.evaluated_len() != 512
                || session.logits() != Some(before.as_slice())
            {
                return Err("boundary rejection changed state".into());
            }
            if let Some(expected) = &boundary_distribution {
                if expected != &before {
                    return Err("boundary schedules differ".into());
                }
            } else {
                boundary_distribution = Some(before.clone());
            }
            capture.save(
                &format!("boundary.chunk{chunk}.logits"),
                511,
                1,
                VOCAB,
                &before,
            )?;
            session.reset();
            if session.evaluated_len() != 0 || !session.history().is_empty() {
                return Err("reset retained state".into());
            }
            admitted = admitted.max(session.memory().admitted_bytes + APP);
        }
        if admitted > TOTAL {
            return Err("combined diagnostic admission exceeds budget".into());
        }
        capture.output.flush()?;
        // Capture is observational. A separate source-bound checker must qualify
        // every value; successful execution alone is not a numerical pass.
        let report = json!({"format":"toklet-minicpm5-checkpoint-capture-v1","repository":REPOSITORY,"revision":REVISION,
            "source_manifest_sha256":source_sha,"device":model.device_name(),"declared_positions":model.mlx_declared_positions(),
            "implemented_positions":512,"profile":"MlxAffine/Compensated/Reference/Half","requested_chunks":[1,3,32],"effective_chunk":"min(requested,512-position_offset)",
            "weights_bytes":model.weights_bytes(),"positional_bytes":model.positional_resident_bytes(),
            "load_timing":{"load_seconds":model.load_timing.load_seconds,"conversion_seconds":model.load_timing.conversion_seconds,"compilation_seconds":model.load_timing.compilation_seconds},"maximum_admitted_bytes":admitted,"application_allowance_bytes":APP,
            "native_selected_on_teacher_forced_prefixes":native_selected,"array_bytes":capture.bytes,"arrays":capture.arrays,
            "native_generated_ids":generated,"native_generation_scope":"four raw IDs; no text/stop/quality qualification",
            "chunk_metrics":chunk_metrics,"chunk_metric_scope":"completed evaluate_next_chunk only; excludes diagnostic array copy/serialization; instrumented numerical diagnostic, not ordinary request/telemetry overhead",
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
            let path = std::env::temp_dir()
                .join(format!("toklet-minicpm5-capture-{}", std::process::id()));
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
            assert_eq!(records.len(), 3);
            assert_eq!(records[0]["event"], "started");
            assert_eq!(records[1]["event"], "completed");
            assert_eq!(records[1]["offset"], 0);
            assert_eq!(records[1]["bytes"], 4);
            assert_eq!(records[2]["event"], "started");
            assert_eq!(records[2]["offset"], 4);
            assert_eq!(records[2]["expected_bytes"], 8);
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
    if std::env::args_os().nth(1).is_some_and(|v| v == "bench") {
        apple::bench()
    } else {
        apple::run()
    }
}
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("minicpm5_probe requires an Apple Silicon Mac with Metal");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::contract::*;
    use serde_json::json;
    #[test]
    fn token_boundaries_reject_without_model_or_gpu() {
        for ids in [
            json!([]),
            json!([true]),
            json!([1.5]),
            json!([-1]),
            json!([130560]),
            json!(vec![1; 513]),
        ] {
            assert!(tokens(&ids).is_err());
        }
        assert_eq!(tokens(&json!([0, 130559])).unwrap(), [0, 130559]);
    }
    #[test]
    fn teacher_history_requires_exact_valid_prefix() {
        let mut prefix = vec![0, 7, 13, 3];
        let mut rows = Vec::new();
        for id in [1, 2, 3, 4] {
            rows.push(json!({"prefix":prefix,"fp32_id":id,"fp64_id":id}));
            prefix.push(id);
        }
        let value = json!({"greedy":rows});
        assert_eq!(teacher(&value).unwrap(), [1, 2, 3, 4]);
        for field in ["prefix", "fp32_id", "fp64_id"] {
            let mut damaged = value.clone();
            damaged["greedy"][2][field] = json!(-1);
            assert!(teacher(&damaged).is_err());
        }
    }
    #[test]
    fn file_and_manifest_identity_fail_closed() {
        let path =
            std::env::temp_dir().join(format!("toklet-minicpm5-contract-{}", std::process::id()));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        use std::io::Write;
        file.write_all(b"abc").unwrap();
        drop(file);
        let sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        hash_file(&path, 3, sha).unwrap();
        assert!(hash_file(&path, 2, sha).is_err());
        assert!(hash_file(&path, 3, &"0".repeat(64)).is_err());
        assert!(reference(&path, sha).is_err());
        assert!(reference(&path, "main").is_err());
        std::fs::remove_file(path).unwrap();
    }
}
