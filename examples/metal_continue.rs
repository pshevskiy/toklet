//! Paired continuation versus explicit reset/replay, with an idle handshake.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{
        io::{self, BufRead, Read, Write},
        path::{Path, PathBuf},
        sync::atomic::AtomicBool,
    };
    use toklet::{
        generation::{self, Input, Request},
        metal, session, weights,
    };
    const APP: usize = 64 * 1024 * 1024;
    fn handshake() -> Result<(), Box<dyn std::error::Error>> {
        let mut line = String::new();
        if io::stdin().lock().take(16).read_line(&mut line)? == 0 || line != "continue\n" {
            return Err("bounded idle continuation handshake required".into());
        }
        Ok(())
    }
    fn ids(path: &Path, identity: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let mut data = Vec::new();
        std::fs::File::open(path)?
            .take(2049)
            .read_to_end(&mut data)?;
        if data.len() != 2048 || format!("{:x}", Sha256::digest(&data)) != identity {
            return Err("frozen continuation tokens changed".into());
        }
        Ok(data
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
    fn prepare(s: &mut metal::Session<'_>, base: &[u32]) -> Result<(), Box<dyn std::error::Error>> {
        s.reset();
        s.set_tokens(base)?;
        s.evaluate()?;
        Ok(())
    }
    fn request(
        s: &mut metal::Session<'_>,
        input: Input<'_>,
    ) -> Result<(toklet::generation::Report, u32), Box<dyn std::error::Error>> {
        let mut id = None;
        let report = generation::generate(
            s,
            Request {
                input,
                max_new_tokens: 1,
                stop_tokens: &[],
            },
            &AtomicBool::new(false),
            |e| {
                id = Some(e.token_id);
                Ok(())
            },
        )?;
        s.discard_pending();
        Ok((report, id.ok_or("token event missing")?))
    }
    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let mut args = std::env::args_os().skip(1);
        let source = PathBuf::from(args.next().ok_or("missing model path")?);
        let packed = args.next().ok_or("missing packed path or -")?;
        let contract = PathBuf::from(args.next().ok_or("missing contract path")?);
        if args.next().is_some() {
            return Err("too many continuation arguments".into());
        }
        let quantized = packed != "-";
        let model = if quantized {
            metal::Model::load_q4(&source, Path::new(&packed), weights::LoadOptions::default())
        } else {
            metal::Model::load(&source, weights::LoadOptions::default())
        }?;
        if model.config().hidden_size != 896 || model.config().num_layers != 24 {
            return Err("named Qwen0.5B continuation geometry required".into());
        }
        let base = ids(
            &contract.join("prompt_512.u32le"),
            "4ff74dae2ee5b778f9313f4c707f9ccb6415609a9c0653684cce7be63d457dd6",
        )?;
        let changed = ids(
            &contract.join("changed_prompt_512.u32le"),
            "e3b16c222cf07b09d6ded35fcdda26b540e6127fe37ee3a97fa929656d306162",
        )?;
        let mut tail = base.clone();
        tail[511] = (tail[511] + 1) % model.config().vocab_size as u32;
        let cases = [("changed128", changed, 128usize), ("tail_edit", tail, 511)];
        let budget = weights::INITIAL_MEMORY_LIMIT_BYTES - APP;
        let options = session::Options {
            capacity: 1024,
            prefill_chunk_size: 128,
            memory_limit_bytes: budget,
            ..Default::default()
        };
        let mut baseline = metal::Session::new(&model, options)?;
        let remaining = budget
            .checked_sub(baseline.memory().admitted_bytes - model.weights_bytes())
            .ok_or("shared model admission overflow")?;
        let mut reused = metal::Session::new(
            &model,
            session::Options {
                memory_limit_bytes: remaining,
                ..options
            },
        )?;
        let combined = baseline
            .memory()
            .admitted_bytes
            .checked_add(reused.memory().admitted_bytes)
            .and_then(|b| b.checked_sub(model.weights_bytes()))
            .and_then(|b| b.checked_add(APP))
            .ok_or("combined admission overflow")?;
        if combined > weights::INITIAL_MEMORY_LIMIT_BYTES {
            return Err("combined continuation admission exceeds6GiB".into());
        }
        let mut out = io::stdout().lock();
        // Q4 layer reductions are row-invariant; only Float32 tied projection
        // may change reduction order. FP16 mode is diagnostic until qualified.
        let atol = 1e-4;
        let rtol = 1e-4;
        writeln!(
            out,
            "{}",
            json!({"event":"ready","format":"toklet-metal-continuation-v1","quantized":quantized,
            "device":model.device_name(),"debug_assertions":cfg!(debug_assertions),"warmups":3,"samples":10,"cases":["changed128","tail_edit"],
            "capacity":1024,"chunk":128,"combined_admitted_bytes":combined,"baseline_memory":baseline.memory(),"reuse_memory":reused.memory(),
            "application_bytes":APP,"load_seconds":model.load_timing.load_seconds,"numerical_gate":{"atol":atol,"rtol":rtol}})
        )?;
        out.flush()?;
        handshake()?;
        for run in 0..13 {
            for (case_index, (name, target, keep)) in cases.iter().enumerate() {
                prepare(&mut baseline, &base)?;
                prepare(&mut reused, &base)?;
                let (a, b) = if (run + case_index) % 2 == 0 {
                    (
                        request(&mut baseline, Input::Reset(target))?,
                        request(&mut reused, Input::Prompt(target))?,
                    )
                } else {
                    let b = request(&mut reused, Input::Prompt(target))?;
                    let a = request(&mut baseline, Input::Reset(target))?;
                    (a, b)
                };
                if a.0.reused_tokens != 0
                    || a.0.prefill_tokens != 512
                    || b.0.reused_tokens != *keep
                    || b.0.prefill_tokens != 512 - keep
                    || a.1 != b.1
                    || baseline.evaluated_len() != 512
                    || reused.evaluated_len() != 512
                    || baseline.history() != target
                    || reused.history() != target
                {
                    return Err("continuation cache/greedy semantics differ".into());
                }
                let left = baseline.logits().ok_or("baseline logits unavailable")?;
                let right = reused.logits().ok_or("reused logits unavailable")?;
                if left.len() != right.len() {
                    return Err("continuation vocabulary mismatch".into());
                }
                let mut maximum = 0f64;
                for (&want, &actual) in left.iter().zip(right) {
                    let delta = (f64::from(want) - f64::from(actual)).abs();
                    maximum = maximum.max(delta);
                    if !actual.is_finite()
                        || !want.is_finite()
                        || delta > atol + rtol * f64::from(want).abs()
                    {
                        return Err(format!(
                            "continuation logit gate failed delta{delta} want{want} actual{actual}"
                        )
                        .into());
                    }
                }
                writeln!(
                    out,
                    "{}",
                    json!({"event":"sample","run":run,"case":name,"order":if (run+case_index)%2==0 {"reset_first"} else {"reuse_first"},
                    "reset_report":a.0,"reuse_report":b.0,"greedy_id":a.1,"max_logit_delta":maximum,"reset_timing":baseline.timing(),"reuse_timing":reused.timing(),
                    "reset_memory":baseline.memory(),"reuse_memory":reused.memory()})
                )?;
                out.flush()?;
                handshake()?;
            }
        }
        writeln!(
            out,
            "{}",
            json!({"event":"done","pairs":26,"combined_admitted_bytes":combined})
        )?;
        out.flush()?;
        handshake()?;
        Ok(())
    }
}
fn main() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Err(e) = apple::run() {
        eprintln!("metal continuation failed: {e}");
        std::process::exit(1);
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        eprintln!("Apple GPU required");
        std::process::exit(1);
    }
}
