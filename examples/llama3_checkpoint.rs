//! Explicit pinned real-Llama diagnostic. No application admission or kernel changes.
#[cfg(any(test, all(target_os = "macos", target_arch = "aarch64")))]
mod contract {
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::{fs::File, io::Read, path::Path};
    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    pub const REPOSITORY: &str = "mlx-community/Llama-3.2-1B-Instruct-8bit";
    pub const REVISION: &str = "d48cdf0a4ea22d893b7c63a99d6a693e24822795";
    pub const VOCAB: usize = 128256;
    pub const FILES: &[(&str, u64, &str)] = &[
        (
            "config.json",
            1041,
            "847794a29e25927b4b98f35f5e66829e1b8bfd78f7743a9582780b4fd4a8ef9d",
        ),
        (
            "model.safetensors.index.json",
            26160,
            "c444df144fe9d86b42f8b1d797432c1f8379225316e41b1b9551e90d63136155",
        ),
        (
            "tokenizer.json",
            9085657,
            "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
        ),
        (
            "tokenizer_config.json",
            54528,
            "9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e",
        ),
        (
            "special_tokens_map.json",
            296,
            "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec",
        ),
        (
            "model.safetensors",
            1313157436,
            "cd3582a78766654e3e2107e3b4d51d9cb18e6754b659bfb44a34328ef9d7fa52",
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
    pub fn cases() -> Value {
        json!({"single":{"tokens":[128000],"offset":0},
            "short":{"tokens":[128000,1,7,13],"offset":0},
            "future_changed":{"tokens":[128000,1,7,17],"offset":0},
            "offset5":{"tokens":[7,13,3],"offset":5},
            "tile17":{"tokens":[128000,1,7,13,128000,1,7,13,128000,1,7,13,128000,1,7,13,128000],"offset":0},
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
        if value["format"] != "toklet-mlx-llama3-real-reference-v1"
            || value["model"]["repository"] != REPOSITORY
            || value["model"]["revision"] != REVISION
            || value["cases"] != cases()
            || value["layers"] != json!([0, 1, 15])
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
    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args_os().skip(1).collect();
        if args.len() != 4 {
            return Err(
                "usage: llama3_checkpoint MODEL REFERENCE-MANIFEST MANIFEST-SHA NEW-OUTPUT".into(),
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
        if model.mlx_declared_positions() != Some(131072) || model.positional_resident_bytes() == 0
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
                session.enable_diagnostics(&[0, 1, 15])?;
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
        let boundary: Vec<u32> = (0..512).map(|i| prompt[i % prompt.len()]).collect();
        let mut boundary_distribution = None;
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
        let report = json!({"format":"toklet-llama3-checkpoint-capture-v1","repository":REPOSITORY,"revision":REVISION,
            "source_manifest_sha256":source_sha,"device":model.device_name(),"declared_positions":model.mlx_declared_positions(),
            "implemented_positions":512,"profile":"MlxAffine/Compensated/Reference/Half","requested_chunks":[32,128,3],"effective_chunk":"min(requested,512-position_offset)",
            "weights_bytes":model.weights_bytes(),"positional_bytes":model.positional_resident_bytes(),
            "load_timing":{"load_seconds":model.load_timing.load_seconds,"conversion_seconds":model.load_timing.conversion_seconds,"compilation_seconds":model.load_timing.compilation_seconds},"maximum_admitted_bytes":admitted,"application_allowance_bytes":APP,
            "native_selected_on_teacher_forced_prefixes":native_selected,"array_bytes":capture.bytes,"arrays":capture.arrays,
            "native_generated_ids":generated,"native_generation_scope":"four raw IDs; no text/stop/quality qualification",
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
                std::env::temp_dir().join(format!("toklet-llama3-capture-{}", std::process::id()));
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
    apple::run()
}
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("llama3_checkpoint requires an Apple Silicon Mac with Metal");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::contract::*;
    use serde_json::json;
    #[test]
    fn source_contract_and_token_boundaries_reject_before_loading() {
        let valid = json!({"format":"toklet-mlx-llama3-real-reference-v1","model":{"repository":REPOSITORY,"revision":REVISION},
            "cases":cases(),"layers":[0,1,15],"greedy":{"prompt":[128000,1,7,13],"fp64_tokens":[1,2,3,4]}});
        validate_reference(&valid).unwrap();
        for (key, value) in [
            ("cases", json!({})),
            ("layers", json!([0, 1, 14])),
            ("format", json!("other")),
            ("model", json!({"repository":REPOSITORY,"revision":"main"})),
            (
                "greedy",
                json!({"prompt":[128000,1,7,13],"fp64_tokens":[1,2,3,128256]}),
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
            json!([128256]),
            json!(vec![1; 513]),
        ] {
            assert!(tokens(&ids).is_err());
        }
        assert_eq!(tokens(&json!([0, 128255])).unwrap(), [0, 128255]);
    }
    #[test]
    fn file_and_manifest_integrity_fail_closed_without_model_download() {
        let path =
            std::env::temp_dir().join(format!("toklet-llama3-contract-{}", std::process::id()));
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
