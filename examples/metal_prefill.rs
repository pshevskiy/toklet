//! Opt-in Q4 tile versus original warm512-token prefill; first-token engine timing.
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
            return Err("bounded idle prefill handshake required".into());
        }
        Ok(())
    }
    fn ids(path: &Path, identity: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let mut data = Vec::new();
        std::fs::File::open(path)?
            .take(2049)
            .read_to_end(&mut data)?;
        if data.len() != 2048 || format!("{:x}", Sha256::digest(&data)) != identity {
            return Err("frozen prefill tokens changed".into());
        }
        Ok(data
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect())
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
        let accurate = match args.next() {
            None => false,
            Some(v) if v == "accurate" => true,
            _ => return Err("optional prefill mode must be accurate".into()),
        };
        let larger = match args.next() {
            None => false,
            Some(v) if v == "larger" => true,
            _ => return Err("optional prefill geometry must be larger".into()),
        };
        if larger && !accurate {
            return Err("larger Q4 prefill requires accurate mode".into());
        }
        if args.next().is_some() {
            return Err("too many prefill arguments".into());
        }
        let model =
            metal::Model::load_q4(&source, Path::new(&packed), weights::LoadOptions::default())?;
        if model.config().hidden_size != if larger { 1536 } else { 896 }
            || model.config().num_layers != if larger { 28 } else { 24 }
        {
            return Err("selected frozen Qwen prefill geometry required".into());
        }
        let base = ids(
            &contract.join("prompt_512.u32le"),
            "4ff74dae2ee5b778f9313f4c707f9ccb6415609a9c0653684cce7be63d457dd6",
        )?;
        let budget = weights::INITIAL_MEMORY_LIMIT_BYTES - APP;
        let options = session::Options {
            capacity: 1024,
            prefill_chunk_size: 128,
            memory_limit_bytes: budget,
            ..Default::default()
        };
        let projection_modes = if accurate {
            [
                metal::ProjectionMode::Q4Compensated,
                metal::ProjectionMode::Q4CompensatedTiled,
            ]
        } else {
            [
                metal::ProjectionMode::Original,
                metal::ProjectionMode::Q4Tiled,
            ]
        };
        let normalization = if accurate {
            metal::NormalizationMode::Compensated
        } else {
            metal::NormalizationMode::Original
        };
        let mut baseline = metal::Session::with_execution_modes(
            &model,
            options,
            projection_modes[0],
            normalization,
        )?;
        let remaining = budget
            .checked_sub(baseline.memory().admitted_bytes - model.weights_bytes())
            .ok_or("shared model admission overflow")?;
        let mut reused = metal::Session::with_execution_modes(
            &model,
            session::Options {
                memory_limit_bytes: remaining,
                ..options
            },
            projection_modes[1],
            normalization,
        )?;
        let combined = baseline
            .memory()
            .admitted_bytes
            .checked_add(reused.memory().admitted_bytes)
            .and_then(|b| b.checked_sub(model.weights_bytes()))
            .and_then(|b| b.checked_add(APP))
            .ok_or("combined admission overflow")?;
        if combined > weights::INITIAL_MEMORY_LIMIT_BYTES {
            return Err("combined prefill admission exceeds6GiB".into());
        }
        let mut out = io::stdout().lock();
        // Freeze the existing same-Q4 envelope before native tile qualification.
        // Ordinary final-row vocabulary execution is the same in both paths.
        let atol = 1e-4;
        let rtol = 1e-4;
        writeln!(
            out,
            "{}",
            json!({"event":"ready","format":"toklet-metal-q4-prefill-v1","projection_modes":projection_modes,"weight_dtype":match model.q4_format() {Some(toklet::q4::Format::V2) => "Q4 v2 + FP16 exclusions",_ => "Q4 v1 + FP16 exclusions"},"packed_format_version":model.q4_format().map(|f| match f {toklet::q4::Format::V1 => 1,toklet::q4::Format::V2 => 2}),"group_size":model.q4_format().map(|f| f.group()),"quantizer":model.q4_format().map(|f| f.quantizer()),
            "normalization_mode":normalization,"requires_logit_bits":accurate,"hidden_size":model.config().hidden_size,"num_layers":model.config().num_layers,
            "device":model.device_name(),"q4_execution_widths":if accurate {model.q4_compensated_execution_widths()}else{model.q4_execution_widths()},"debug_assertions":cfg!(debug_assertions),"warmups":3,"samples":10,"prompt_tokens":512,"output_tokens":1,
            "capacity":1024,"chunk":128,"combined_admitted_bytes":combined,"original_memory":baseline.memory(),"tiled_memory":reused.memory(),
            "application_bytes":APP,"load_seconds":model.load_timing.load_seconds,"numerical_gate":{"atol":atol,"rtol":rtol}})
        )?;
        out.flush()?;
        handshake()?;
        for run in 0..13 {
            let (a, b) = if run % 2 == 0 {
                (
                    request(&mut baseline, Input::Reset(&base))?,
                    request(&mut reused, Input::Reset(&base))?,
                )
            } else {
                let b = request(&mut reused, Input::Reset(&base))?;
                (request(&mut baseline, Input::Reset(&base))?, b)
            };
            if a.0.reused_tokens != 0
                || b.0.reused_tokens != 0
                || a.0.prefill_tokens != 512
                || b.0.prefill_tokens != 512
                || a.1 != b.1
                || baseline.evaluated_len() != 512
                || reused.evaluated_len() != 512
                || baseline.history() != base
                || reused.history() != base
            {
                return Err("paired prefill request/cache/greedy semantics differ".into());
            }
            let left = baseline.logits().ok_or("original logits unavailable")?;
            let right = reused.logits().ok_or("tiled logits unavailable")?;
            if left.len() != right.len() {
                return Err("paired prefill vocabulary differs".into());
            }
            let mut maximum = 0f64;
            let mut bits_equal = true;
            for (&want, &actual) in left.iter().zip(right) {
                bits_equal &= want.to_bits() == actual.to_bits();
                let delta = (f64::from(want) - f64::from(actual)).abs();
                maximum = maximum.max(delta);
                if !actual.is_finite()
                    || !want.is_finite()
                    || delta > atol + rtol * f64::from(want).abs()
                {
                    return Err(format!(
                        "paired prefill logit gate failed delta{delta} want{want} actual{actual}"
                    )
                    .into());
                }
            }
            if accurate && !bits_equal {
                return Err("accurate tiled prefill changes logit bits".into());
            }
            writeln!(
                out,
                "{}",
                json!({"event":"sample","run":run,"order":if run%2==0 {"original_first"}else{"tiled_first"},
                "original_report":a.0,"tiled_report":b.0,"greedy_id":a.1,"max_logit_delta":maximum,"logit_bits_equal":bits_equal,
                "original_timing":baseline.timing(),"tiled_timing":reused.timing(),"original_memory":baseline.memory(),"tiled_memory":reused.memory()})
            )?;
            out.flush()?;
            handshake()?;
        }

        writeln!(
            out,
            "{}",
            json!({"event":"done","pairs":13,"combined_admitted_bytes":combined})
        )?;
        out.flush()?;
        handshake()?;
        Ok(())
    }
}
fn main() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Err(e) = apple::run() {
        eprintln!("metal prefill failed: {e}");
        std::process::exit(1);
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        eprintln!("Apple GPU required");
        std::process::exit(1);
    }
}
