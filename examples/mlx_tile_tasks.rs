//! Mac P2 task collection using the frozen Qwen2.5 MLX8 prompts and stops.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use serde::Deserialize;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{
        fs::File,
        io::{Read, Write},
        path::Path,
        sync::atomic::AtomicBool,
    };
    use toklet::{
        generation, metal,
        session::{self, InferenceSession},
        weights,
    };

    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    const BUDGET: usize = 1536 * 1024 * 1024;
    const CAPACITY: usize = 562;
    const CHUNK: usize = 128;
    const VOCAB: usize = 151936;
    const TASK_SHA: &str = "ccb24a6f0b6e4798cabc393c475e377d3b941a640e5b6414cc9359bf5ee5659b";

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields, rename_all = "camelCase")]
    struct Case {
        id: String,
        prompt_tokens: Vec<u32>,
        maximum: usize,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Tasks {
        version: u32,
        vocabulary: usize,
        stops: Vec<u32>,
        cases: Vec<Case>,
    }

    fn open(model: &metal::Model, limit: usize) -> Result<metal::Session<'_>> {
        Ok(metal::Session::with_attention_mode(
            model,
            session::Options {
                capacity: CAPACITY,
                prefill_chunk_size: CHUNK,
                position_offset: 0,
                memory_limit_bytes: limit,
            },
            metal::ProjectionMode::MlxAffine,
            metal::NormalizationMode::Compensated,
            metal::ReadbackMode::FullLogits,
            metal::AttentionMode::Online,
        )?)
    }

    // A test-only pair delegates generation to the normal transaction and stop
    // handling. It compares each available distribution before sampling.
    struct Pair<'m> {
        baseline: metal::Session<'m>,
        tile: metal::Session<'m>,
    }
    impl Pair<'_> {
        fn verify(&self) -> std::result::Result<(), session::Error> {
            if self.baseline.history() != self.tile.history()
                || self.baseline.evaluated_len() != self.tile.evaluated_len()
                || self.baseline.pending_len() != self.tile.pending_len()
            {
                return Err(session::Error::InvalidOptions("paired task state differs"));
            }
            match (self.baseline.logits(), self.tile.logits()) {
                (None, None) => Ok(()),
                (Some(a), Some(b))
                    if a.len() == b.len()
                        && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) =>
                {
                    Ok(())
                }
                _ => Err(session::Error::InvalidOptions(
                    "paired task distribution differs",
                )),
            }
        }
    }
    impl InferenceSession for Pair<'_> {
        fn history(&self) -> &[u32] {
            self.baseline.history()
        }
        fn evaluated_len(&self) -> usize {
            self.baseline.evaluated_len()
        }
        fn pending_len(&self) -> usize {
            self.baseline.pending_len()
        }
        fn capacity(&self) -> usize {
            self.baseline.capacity()
        }
        fn vocab_size(&self) -> usize {
            self.baseline.vocab_size()
        }
        fn memory(&self) -> session::Memory {
            self.baseline.memory()
        }
        fn logits(&self) -> Option<&[f32]> {
            self.baseline.logits()
        }
        fn append(&mut self, tokens: &[u32]) -> std::result::Result<(), session::Error> {
            InferenceSession::append(&mut self.baseline, tokens)?;
            InferenceSession::append(&mut self.tile, tokens)?;
            self.verify()
        }
        fn set_tokens(&mut self, tokens: &[u32]) -> std::result::Result<(), session::Error> {
            InferenceSession::set_tokens(&mut self.baseline, tokens)?;
            InferenceSession::set_tokens(&mut self.tile, tokens)?;
            self.verify()
        }
        fn evaluate_next_chunk(&mut self) -> std::result::Result<usize, session::Error> {
            let rows = InferenceSession::evaluate_next_chunk(&mut self.baseline)?;
            if rows != InferenceSession::evaluate_next_chunk(&mut self.tile)? {
                return Err(session::Error::InvalidOptions("paired task chunk differs"));
            }
            self.verify()?;
            Ok(rows)
        }
        fn discard_pending(&mut self) {
            self.baseline.discard_pending();
            self.tile.discard_pending();
        }
        fn reset(&mut self) {
            self.baseline.reset();
            self.tile.reset();
        }
    }

    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args().collect();
        if cfg!(debug_assertions) || args.len() != 3 {
            return Err("usage: release mlx_tile_tasks MODEL TASKS_JSON".into());
        }
        let mut bytes = Vec::new();
        File::open(&args[2])?
            .take(1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        if format!("{:x}", Sha256::digest(&bytes)) != TASK_SHA {
            return Err("frozen Qwen task payload differs".into());
        }
        let tasks: Tasks = serde_json::from_slice(&bytes)?;
        if tasks.version != 1
            || tasks.vocabulary != VOCAB
            || tasks.stops != [151645, 151643]
            || tasks.cases.len() != 100
            || tasks.cases.iter().any(|case| {
                case.id.is_empty()
                    || case.prompt_tokens.is_empty()
                    || case.maximum == 0
                    || case
                        .prompt_tokens
                        .len()
                        .checked_add(case.maximum)
                        .is_none_or(|n| n > CAPACITY)
                    || case.prompt_tokens.iter().any(|&id| id as usize >= VOCAB)
            })
        {
            return Err("frozen task schema differs".into());
        }

        let model = metal::Model::load_mlx_qwen2_audited(
            Path::new(&args[1]),
            weights::LoadOptions {
                context_limit: CAPACITY,
                memory_limit_bytes: BUDGET,
            },
        )?;
        if model.config().vocab_size != VOCAB || model.mlx_qwen2_source_content_identity().is_none()
        {
            return Err("pinned audited Qwen source required".into());
        }
        {
            let mut probe = open(&model, BUDGET)?;
            probe.enable_affine_prefill_tile4()?;
            probe.enable_profile()?;
            let case = tasks
                .cases
                .iter()
                .find(|case| case.prompt_tokens.len() >= CHUNK)
                .ok_or("no eligible frozen task prefill")?;
            probe.append(&case.prompt_tokens[..CHUNK])?;
            probe.evaluate()?;
            let dispatches = probe
                .profile()?
                .events
                .iter()
                .filter(|e| e.name == "mlx_product_tile4")
                .count();
            if dispatches != 24 * 7 {
                return Err("frozen task prefill did not dispatch tile4".into());
            }
            println!(
                "{}",
                json!({"event":"dispatch","case_id":case.id,"rows":CHUNK,"tile4_stages":dispatches})
            );
        }
        let baseline = open(&model, BUDGET)?;
        let other = baseline
            .memory()
            .admitted_bytes
            .checked_sub(model.model_resident_bytes()?)
            .ok_or("invalid shared model admission")?;
        let mut tile = open(
            &model,
            BUDGET.checked_sub(other).ok_or("paired task budget")?,
        )?;
        tile.enable_affine_prefill_tile4()?;
        let admitted = other + tile.memory().admitted_bytes;
        let mut pair = Pair { baseline, tile };
        println!(
            "{}",
            json!({"event":"ready","device":model.device_name(),"chunk":CHUNK,
            "capacity":CAPACITY,"task_sha256":TASK_SHA,"combined_admitted_bytes":admitted})
        );
        let cancel = AtomicBool::new(false);
        for (index, case) in tasks.cases.iter().enumerate() {
            let mut ids = Vec::with_capacity(case.maximum);
            let report = generation::generate(
                &mut pair,
                generation::Request {
                    input: generation::Input::Reset(&case.prompt_tokens),
                    max_new_tokens: case.maximum,
                    stop_tokens: &tasks.stops,
                },
                &cancel,
                |event| {
                    ids.push(event.token_id);
                    Ok(())
                },
            )?;
            pair.verify()?;
            println!(
                "{}",
                json!({"event":"task","index":index,"id":case.id,"ids":ids,
                "sampled":report.sampled_tokens,"reason":report.reason,
                "history":report.history_tokens,"evaluated":report.evaluated_tokens,"pending":report.pending_tokens})
            );
            std::io::stdout().flush()?;
            pair.reset();
            pair.verify()?;
            if !pair.history().is_empty() || pair.evaluated_len() != 0 {
                return Err("task reset state differs".into());
            }
        }
        println!(
            "{}",
            json!({"event":"done","tasks":tasks.cases.len(),"bitwise_decision_rows":true})
        );
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
        eprintln!("Task qualification requires Apple Silicon macOS");
        std::process::exit(1);
    }
}
