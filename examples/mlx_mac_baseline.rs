//! Explicit Mac P0 worker: completed uninstrumented timing or per-chunk counters.
//! Fixed teacher-forced IDs; no text, sampling, downloads or kernel changes.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{fs, hint::black_box, io::Write, path::Path, time::Instant};
    use toklet::{benchmark::Sample, metal, session, weights};

    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    const BUDGET: usize = 1536 * 1024 * 1024;
    const CAPACITY: usize = 1152;
    const CHUNK: usize = 128;
    const ROLES: [&str; 17] = [
        "input_norm",
        "q",
        "k",
        "v",
        "q_rotary",
        "k_rotary",
        "cache",
        "attention",
        "attention_projection",
        "attention_residual",
        "post_attention_norm",
        "gate",
        "silu",
        "up",
        "gated",
        "down",
        "output_residual",
    ];

    fn tokens(
        root: &Path,
        manifest: &serde_json::Value,
        key: &str,
        count: usize,
    ) -> Result<Vec<u32>> {
        let entry = &manifest["arrays"][key];
        let file = entry["file"].as_str().ok_or("missing token file")?;
        if Path::new(file).components().count() != 1 || file.starts_with('.') {
            return Err("plain token filename required".into());
        }
        let raw = fs::read(root.join(file))?;
        if raw.len() != count * 4
            || entry["count"] != count
            || entry["dtype"] != "u32le"
            || entry["sha256"] != format!("{:x}", Sha256::digest(&raw))
        {
            return Err("frozen token inventory differs".into());
        }
        Ok(raw
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }

    fn session(model: &metal::Model, budget: usize) -> Result<metal::Session<'_>> {
        Ok(metal::Session::with_attention_mode(
            model,
            session::Options {
                capacity: CAPACITY,
                prefill_chunk_size: CHUNK,
                position_offset: 0,
                memory_limit_bytes: budget,
            },
            metal::ProjectionMode::MlxAffine,
            metal::NormalizationMode::Compensated,
            metal::ReadbackMode::FullLogits,
            metal::AttentionMode::Online,
        )?)
    }

    fn hash(s: &metal::Session<'_>) -> Result<String> {
        let mut h = Sha256::new();
        for value in s.logits().ok_or("missing logits")? {
            if !value.is_finite() {
                return Err("nonfinite logits".into());
            }
            h.update(value.to_le_bytes());
        }
        Ok(format!("{:x}", h.finalize()))
    }

    fn measure(s: &mut metal::Session<'_>, prompt: &[u32], output: &[u32]) -> Result<Sample> {
        s.reset();
        let mut events = Vec::with_capacity(output.len());
        let started = Instant::now();
        s.append(prompt)?;
        let prefill = Instant::now();
        s.evaluate()?;
        let prefill_seconds = prefill.elapsed().as_secs_f64();
        black_box(s.logits().ok_or("missing prefill distribution")?);
        events.push(started.elapsed().as_secs_f64());
        s.append(&output[..1])?;
        for id in &output[1..] {
            if s.evaluate_next_chunk()? != 1 {
                return Err("decode must commit one input".into());
            }
            black_box(s.logits().ok_or("missing decode distribution")?);
            events.push(started.elapsed().as_secs_f64());
            s.append(&[*id])?;
        }
        let total_seconds = started.elapsed().as_secs_f64();
        if s.evaluated_len() != prompt.len() + output.len() - 1 || s.pending_len() != 1 {
            return Err("teacher-forced state differs".into());
        }
        Ok(Sample {
            prefill_seconds,
            ttft_seconds: events[0],
            decode_seconds: events[events.len() - 1] - events[0],
            total_seconds,
            token_event_seconds: events,
            emitted_tokens: output.len(),
            reused_tokens: 0,
            prefill_tokens: prompt.len(),
            evaluated_tokens: s.evaluated_len(),
            pending_tokens: 1,
        })
    }

    fn check_parity(base: &metal::Session<'_>, profiled: &metal::Session<'_>) -> Result<()> {
        let a = base.logits().ok_or("missing baseline logits")?;
        let b = profiled.logits().ok_or("missing profiled logits")?;
        if base.history() != profiled.history()
            || base.evaluated_len() != profiled.evaluated_len()
            || a.len() != b.len()
            || a.iter().zip(b).any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err("profiling changed complete logits".into());
        }
        Ok(())
    }

    fn export(
        profiled: &metal::Session<'_>,
        trial: usize,
        prompt: usize,
        phase: &str,
        offset: usize,
        rows: usize,
    ) -> Result<()> {
        let p = profiled.profile()?;
        let layers = 24;
        if p.events.len() != layers * ROLES.len() + 4 {
            return Err("unexpected stage inventory".into());
        }
        let mut stages = Vec::with_capacity(p.events.len());
        for (index, sample) in p.events.iter().enumerate() {
            let (layer, role, expected) = if index == 0 {
                (None, "embedding", "mlx_gather")
            } else if index <= layers * ROLES.len() {
                let i = (index - 1) % ROLES.len();
                let name = match i {
                    0 | 10 => "rms",
                    1 | 2 | 3 | 8 | 11 | 13 | 15 => "mlx_product",
                    4 | 5 => "rotary",
                    6 => "cache",
                    7 => {
                        if rows == 1 {
                            "attention_decode_online"
                        } else {
                            "attention_prefill_online"
                        }
                    }
                    9 | 16 => "residual",
                    12 => "silu",
                    14 => "gating",
                    _ => unreachable!(),
                };
                (Some((index - 1) / ROLES.len()), ROLES[i], name)
            } else {
                match index - layers * ROLES.len() - 1 {
                    0 => (None, "final_norm", "rms"),
                    1 => (None, "head", "mlx_product"),
                    2 => (None, "validate", "validate_float"),
                    _ => unreachable!(),
                }
            };
            if sample.name != expected {
                return Err(format!("stage{index}: {} != {expected}", sample.name).into());
            }
            stages.push(json!({"layer":layer,"role":role,"sample":sample}));
        }
        println!(
            "{}",
            json!({"event":"profile","trial":trial,"prompt_tokens":prompt,
            "phase":phase,"offset":offset,"rows":rows,"logits_sha256":hash(profiled)?,
            "start":p.start,"end":p.end,"host_clock_sample_bracket_seconds":p.host_clock_sample_bracket_seconds,
            "stages":stages})
        );
        Ok(())
    }

    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args().collect();
        if args.len() != 4
            || !matches!(
                args[1].as_str(),
                "time" | "profile" | "time-tile4" | "check-tile4"
            )
        {
            return Err(
                "usage: mlx_mac_baseline time|profile|time-tile4|check-tile4 MODEL CONTRACT".into(),
            );
        }
        let mode = args[1].as_str();
        let compare_tile = mode == "check-tile4";
        let root = Path::new(&args[3]);
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("manifest.json"))?)?;
        let output = tokens(root, &manifest, "continuation_128", 128)?;
        let prompts = [128, 512, 1024]
            .into_iter()
            .map(|n| tokens(root, &manifest, &format!("prompt_{n}"), n))
            .collect::<Result<Vec<_>>>()?;
        let model = metal::Model::load_mlx_qwen2_audited(
            Path::new(&args[2]),
            weights::LoadOptions {
                context_limit: CAPACITY,
                memory_limit_bytes: BUDGET,
            },
        )?;
        if model.mlx_qwen2_source_content_identity().is_none() || model.config().num_layers != 24 {
            return Err("pinned audited Qwen2 model required".into());
        }
        if prompts
            .iter()
            .flatten()
            .chain(&output)
            .any(|id| *id as usize >= model.config().vocab_size)
        {
            return Err("token outside model vocabulary".into());
        }
        let mut base = session(&model, BUDGET)?;
        if mode == "time-tile4" {
            base.enable_affine_prefill_tile4()?;
        }
        let mut profiled = if mode == "profile" || compare_tile {
            let remaining = BUDGET
                .checked_sub(base.memory().admitted_bytes - model.model_resident_bytes()?)
                .ok_or("combined budget")?;
            let mut s = session(&model, remaining)?;
            if compare_tile {
                s.enable_affine_prefill_tile4()?;
            }
            s.enable_profile()?;
            Some(s)
        } else {
            None
        };
        println!(
            "{}",
            json!({"event":"ready","mode":args[1],"device":model.device_name(),
            "debug_assertions":cfg!(debug_assertions),"capacity":CAPACITY,"chunk":CHUNK,
            "projection":"MlxAffine","normalization":"Compensated","attention":"Online",
            "readback":"FullLogits","kv":"F16","memory":base.memory(),
            "combined_admitted_bytes":base.memory().admitted_bytes+profiled.as_ref().map_or(0,|s|s.memory().admitted_bytes-model.model_resident_bytes().unwrap()),
            "profile_memory":profiled.as_ref().map(|s|s.memory()),"sampled_decode_indices":[0,31,63,126]})
        );
        for prompt in &prompts {
            if profiled.is_none() {
                for trial in 0..13 {
                    let sample = measure(&mut base, prompt, &output)?;
                    base.discard_pending();
                    println!(
                        "{}",
                        json!({"event":"timing","instrumented_worker":profiled.is_some(),
                        "prompt_tokens":prompt.len(),"trial":trial,"warmup":trial<3,
                        "sample":sample,"logits_sha256":hash(&base)?})
                    );
                }
            }
            if let Some(p) = &mut profiled {
                let trials = if compare_tile { 1 } else { 3 };
                for trial in 0..trials {
                    base.reset();
                    p.reset();
                    for tokens in prompt.chunks(CHUNK) {
                        let offset = base.evaluated_len();
                        base.append(tokens)?;
                        p.append(tokens)?;
                        let rows = base.evaluate_next_chunk()?;
                        if rows != tokens.len() || p.evaluate_next_chunk()? != rows {
                            return Err("profile row count differs".into());
                        }
                        check_parity(&base, p)?;
                        if compare_tile {
                            let count = p
                                .profile()?
                                .events
                                .iter()
                                .filter(|e| e.name == "mlx_product_tile4")
                                .count();
                            if count != 24 * 7 {
                                return Err(
                                    "tile4 did not dispatch for all audited projections".into()
                                );
                            }
                            println!(
                                "{}",
                                json!({"event":"tile4-prefill","prompt_tokens":prompt.len(),
                                                    "offset":offset,"rows":rows,"tile_dispatches":count,
                                                    "logits_sha256":hash(p)?})
                            );
                        } else {
                            export(p, trial, prompt.len(), "prefill", offset, rows)?;
                        }
                    }
                    for (index, id) in output[..127].iter().enumerate() {
                        base.append(&[*id])?;
                        p.append(&[*id])?;
                        if base.evaluate_next_chunk()? != 1 || p.evaluate_next_chunk()? != 1 {
                            return Err("profile decode differs".into());
                        }
                        check_parity(&base, p)?;
                        // Trace all completed operations, export only declared samples.
                        if compare_tile
                            && index == 0
                            && p.profile()?
                                .events
                                .iter()
                                .any(|e| e.name == "mlx_product_tile4")
                        {
                            return Err("tile4 used on decode".into());
                        }
                        if !compare_tile && [0, 31, 63, 126].contains(&index) {
                            export(p, trial, prompt.len(), "decode", prompt.len() + index, 1)?;
                        }
                    }
                    if compare_tile {
                        println!(
                            "{}",
                            json!({"event":"tile4-decode","prompt_tokens":prompt.len(),
                                                "positions":127,"committed":p.evaluated_len(),
                                                "logits_sha256":hash(p)?})
                        );
                    }
                }
            }
        }
        println!("{}", json!({"event":"done"}));
        std::io::stdout().flush()?;
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
        eprintln!("Mac baseline requires Apple Silicon macOS");
        std::process::exit(1);
    }
}
