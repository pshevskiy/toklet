//! Explicit Q4 chunk sweep; completed first-token timing, not release acceptance.
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
    const CHUNKS: [usize; 4] = [128, 256, 512, 1024];
    fn handshake() -> Result<(), Box<dyn std::error::Error>> {
        let mut line = String::new();
        if io::stdin().lock().take(16).read_line(&mut line)? == 0 || line != "continue\n" {
            return Err("bounded chunk handshake required".into());
        }
        Ok(())
    }
    fn request(
        s: &mut metal::Session<'_>,
        input: Input<'_>,
    ) -> Result<(generation::Report, u32), Box<dyn std::error::Error>> {
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
        Ok((report, id.ok_or("chunk token event missing")?))
    }
    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let mut args = std::env::args_os().skip(1);
        let source = PathBuf::from(args.next().ok_or("missing source")?);
        let packed = PathBuf::from(args.next().ok_or("missing Q4 artifact")?);
        let contract = PathBuf::from(args.next().ok_or("missing contract")?);
        let check = match args.next() {
            None => false,
            Some(v) if v == "check" => true,
            _ => return Err("optional mode must be check".into()),
        };
        if args.next().is_some() {
            return Err("too many chunk arguments".into());
        }
        let mut bytes = Vec::new();
        std::fs::File::open(contract.join("prompt_1024.u32le"))?
            .take(4097)
            .read_to_end(&mut bytes)?;
        if bytes.len() != 4096
            || format!("{:x}", Sha256::digest(&bytes))
                != "8acd07e9a1a11ae7e6642247f9e7621da9668863b58060620629bbad366a1ba9"
        {
            return Err("frozen chunk prompt changed".into());
        }
        let base: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let model =
            metal::Model::load_q4(&source, Path::new(&packed), weights::LoadOptions::default())?;
        if model.config().hidden_size != 1536
            || model.config().num_layers != 28
            || model.config().vocab_size != 151936
        {
            return Err("fixed larger chunk geometry required".into());
        }
        let projection = metal::ProjectionMode::Q4CompensatedTiled;
        let normalization = metal::NormalizationMode::Compensated;
        let (capacity, offset) = if check { (2031, 17) } else { (2048, 0) };
        let budget = weights::INITIAL_MEMORY_LIMIT_BYTES - APP;
        let mut allocated = model.weights_bytes();
        let mut sessions = Vec::with_capacity(4);
        for chunk in CHUNKS {
            let remaining = budget
                .checked_sub(allocated - model.weights_bytes())
                .ok_or("shared chunk budget overflow")?;
            let session = metal::Session::with_execution_modes(
                &model,
                session::Options {
                    capacity,
                    prefill_chunk_size: chunk,
                    position_offset: offset,
                    memory_limit_bytes: remaining,
                },
                projection,
                normalization,
            )?;
            allocated = allocated
                .checked_add(session.memory().admitted_bytes - model.weights_bytes())
                .ok_or("combined chunk admission overflow")?;
            sessions.push(session);
        }
        let combined = allocated
            .checked_add(APP)
            .ok_or("application admission overflow")?;
        if combined > weights::INITIAL_MEMORY_LIMIT_BYTES {
            return Err("combined chunk admission exceeds6GiB".into());
        }
        let memories: Vec<_> = sessions.iter().map(metal::Session::memory).collect();
        let mut out = io::stdout().lock();
        writeln!(
            out,
            "{}",
            json!({"event":"ready","format":"toklet-metal-chunks-v1","check_only":check,"debug_assertions":cfg!(debug_assertions),"chunks":CHUNKS,"projection_mode":projection,"normalization_mode":normalization,"q4_execution_widths":model.q4_compensated_execution_widths(),"device":model.device_name(),"hidden_size":1536,"num_layers":28,"vocab_size":151936,"capacity":capacity,"position_offset":offset,"prompt_tokens":1024,"output_tokens":1,"warmups":if check {0}else{3},"samples":if check {3}else{10},"memories":memories,"combined_admitted_bytes":combined,"application_bytes":APP,"load_seconds":model.load_timing.load_seconds,"requires_logit_bits":true})
        )?;
        out.flush()?;
        handshake()?;
        let mut appended = base.clone();
        appended.extend_from_slice(&base[..7]);
        let mut changed = appended.clone();
        changed[1000] = (changed[1000] + 1) % 151936;
        let iterations = if check { 3 } else { 13 };
        for run in 0..iterations {
            // Rotate four candidates through execution positions. Comparison and
            // output happen after all completed request timers in this round.
            let order: [usize; 4] = std::array::from_fn(|i| (run + i) % 4);
            let mut reports: [Option<(generation::Report, u32)>; 4] = std::array::from_fn(|_| None);
            let (name, input, history, prefill, reused) = if !check || run == 0 {
                ("reset", Input::Reset(&base), base.as_slice(), 1024, 0)
            } else if run == 1 {
                (
                    "append7",
                    Input::Append(&base[..7]),
                    appended.as_slice(),
                    7,
                    1024,
                )
            } else {
                (
                    "edit1000",
                    Input::Prompt(&changed),
                    changed.as_slice(),
                    31,
                    1000,
                )
            };
            for index in order {
                reports[index] = Some(request(&mut sessions[index], input)?);
            }
            let ids: [u32; 4] = std::array::from_fn(|i| reports[i].as_ref().unwrap().1);
            let left = sessions[0]
                .logits()
                .ok_or("chunk baseline logits unavailable")?;
            for (i, s) in sessions.iter().enumerate() {
                let report = &reports[i].as_ref().unwrap().0;
                let expected_history = history.len();
                if ids[i] != ids[0]
                    || report.prefill_tokens != prefill
                    || report.reused_tokens != reused
                    || s.evaluated_len() != expected_history
                    || s.history().len() != expected_history
                    || s.pending_len() != 0
                {
                    return Err("chunk request/state semantics differ".into());
                }
                if s.history() != history {
                    return Err("chunk committed history differs".into());
                }
                let actual = s.logits().ok_or("chunk candidate logits unavailable")?;
                if left.len() != 151936
                    || actual.len() != left.len()
                    || left.iter().zip(actual).any(|(&a, &b)| {
                        !a.is_finite() || !b.is_finite() || a.to_bits() != b.to_bits()
                    })
                {
                    return Err("chunk final-logit bit refinement failed".into());
                }
            }
            let reports: Vec<_> = reports.into_iter().map(|r| r.unwrap().0).collect();
            writeln!(
                out,
                "{}",
                json!({"event":"sample","run":run,"case":name,"order":order,"reports":reports,"greedy_ids":ids,"logit_bits_equal":true,"memories":sessions.iter().map(metal::Session::memory).collect::<Vec<_>>()})
            )?;
            out.flush()?;
            handshake()?;
        }
        writeln!(
            out,
            "{}",
            json!({"event":"done","rounds":iterations,"combined_admitted_bytes":combined})
        )?;
        out.flush()?;
        handshake()?;
        Ok(())
    }
}
fn main() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Err(e) = apple::run() {
        eprintln!("metal chunks failed: {e}");
        std::process::exit(1);
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        eprintln!("Apple GPU required");
        std::process::exit(1);
    }
}
