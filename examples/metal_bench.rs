//! Explicit resident Half controlled worker. Frozen CPU timing code is unchanged.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use serde::Deserialize;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{
        hint::black_box,
        io::{self, BufRead, Read, Write},
        path::PathBuf,
        time::Instant,
    };
    use toklet::{
        benchmark::Sample,
        generation::{self, Input, Request as GenerationRequest},
        metal::{self, Session},
        session, weights,
    };
    const APP: usize = 64 * 1024 * 1024;
    #[derive(Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum Action {
        Reset,
        Append,
        Continue,
        Prompt,
        Prepare,
        Stage,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Command {
        action: Action,
        #[serde(default)]
        tokens: Vec<u32>,
        #[serde(default)]
        continuation: Vec<u32>,
    }
    fn history_sha(s: &Session<'_>) -> String {
        let mut h = Sha256::new();
        for id in s.history() {
            h.update(id.to_le_bytes());
        }
        format!("{:x}", h.finalize())
    }
    /// Admission and fixed IDs are checked before state mutation. Callers prepare
    /// cache state outside this function and record which prefix policy they use.
    /// Synchronous evaluation completes every layer before each token timestamp.
    fn measure_teacher_forced(
        session: &mut Session<'_>,
        input: Input<'_>,
        continuation: &[u32],
    ) -> Result<Sample, Box<dyn std::error::Error>> {
        if continuation.is_empty() {
            return Err(generation::Error::InvalidRequest("benchmark needs an output step").into());
        }
        generation::validate_request(
            session,
            &GenerationRequest {
                input,
                max_new_tokens: continuation.len(),
                stop_tokens: &[],
            },
        )?;
        if continuation
            .iter()
            .any(|&id| id as usize >= session.vocab_size())
        {
            return Err(
                generation::Error::InvalidRequest("fixed output ID outside vocabulary").into(),
            );
        }
        let mut events = Vec::with_capacity(continuation.len());
        let started = Instant::now();
        match input {
            Input::Prompt(tokens) => session.set_tokens(tokens)?,
            Input::Append(tokens) => session.append(tokens)?,
            Input::Continue => {}
            Input::Reset(tokens) => {
                session.reset();
                session.append(tokens)?;
            }
        }
        let reused_tokens = session.evaluated_len();
        let pending = session.pending_len();
        let prefill_start = Instant::now();
        session.evaluate()?;
        let prefill_seconds = prefill_start.elapsed().as_secs_f64();
        black_box(session.logits().ok_or(generation::Error::InvalidRequest(
            "no benchmark distribution",
        ))?);
        events.push(started.elapsed().as_secs_f64());
        session.append(&continuation[..1])?;
        for &id in &continuation[1..] {
            session.evaluate_next_chunk()?;
            black_box(session.logits().ok_or(generation::Error::InvalidRequest(
                "no benchmark distribution",
            ))?);
            events.push(started.elapsed().as_secs_f64());
            session.append(&[id])?;
        }
        Ok(Sample {
            prefill_seconds,
            ttft_seconds: events[0],
            decode_seconds: events[events.len() - 1] - events[0],
            total_seconds: started.elapsed().as_secs_f64(),
            emitted_tokens: continuation.len(),
            reused_tokens,
            prefill_tokens: pending,
            evaluated_tokens: session.evaluated_len(),
            pending_tokens: session.pending_len(),
            token_event_seconds: events,
        })
    }

    fn state(s: &Session<'_>) -> serde_json::Value {
        json!({"history_tokens":s.history().len(),"evaluated_tokens":s.evaluated_len(),"pending_tokens":s.pending_len(),"history_sha256":history_sha(s)})
    }
    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let mut args = std::env::args_os().skip(1);
        let source = PathBuf::from(args.next().ok_or("usage: metal_bench MODEL")?);
        if args.next().is_some() {
            return Err("too many arguments".into());
        }
        let budget = weights::INITIAL_MEMORY_LIMIT_BYTES - APP;
        let model = metal::Model::load(
            &source,
            weights::LoadOptions {
                memory_limit_bytes: budget,
                ..Default::default()
            },
        )?;
        let options = session::Options {
            capacity: 2048,
            prefill_chunk_size: 128,
            memory_limit_bytes: budget,
            ..Default::default()
        };
        let mut s =
            Session::with_projection_mode(&model, options, metal::ProjectionMode::HalfCompensated)?;
        let widths = model.half_compensated_execution_widths();
        let c = model.config();
        let mut out = io::stdout().lock();
        writeln!(
            out,
            "{}",
            json!({"event":"ready","format":"toklet-metal-controlled-v1","debug_assertions":cfg!(debug_assertions),"backend":"metal","projection_mode":"HalfCompensated","normalization_mode":"Original","readback_mode":"FullLogits","weight_dtype":"FP16","kv_dtype":"FP16","half_compensated_execution_widths":widths,"device":model.device_name(),"hidden_size":c.hidden_size,"num_layers":c.num_layers,"vocab_size":c.vocab_size,"context_capacity":s.capacity(),"chunk_size":128,"memory":s.memory(),"application_bytes":APP,"memory_admitted_bytes":s.memory().admitted_bytes+APP,"load_seconds":model.load_timing.load_seconds,"conversion_seconds":model.load_timing.conversion_seconds})
        )?;
        out.flush()?;
        let mut input = io::BufReader::new(io::stdin());
        let mut count = 0;
        loop {
            let mut line = Vec::new();
            let n = input.by_ref().take(65537).read_until(b'\n', &mut line)?;
            if n == 0 {
                break;
            }
            if n > 65536 {
                return Err("benchmark line exceeds64KiB".into());
            }
            count += 1;
            if count > 256 {
                return Err("benchmark request count exceeds256".into());
            }
            let result = (|| -> Result<serde_json::Value, Box<dyn std::error::Error>> {
                let cmd: Command = serde_json::from_slice(&line)?;
                if matches!(cmd.action, Action::Continue) && !cmd.tokens.is_empty() {
                    return Err("continue accepts no tokens".into());
                }
                if matches!(cmd.action, Action::Prepare | Action::Stage) {
                    if !cmd.continuation.is_empty() {
                        return Err("preparation accepts no continuation".into());
                    }
                    if matches!(cmd.action, Action::Prepare) {
                        // Validate all IDs and capacity before resetting a valid prefix.
                        generation::validate_request(
                            &s,
                            &GenerationRequest {
                                input: Input::Reset(&cmd.tokens),
                                max_new_tokens: 0,
                                stop_tokens: &[],
                            },
                        )?;
                        s.reset();
                        s.append(&cmd.tokens)?;
                        s.evaluate()?;
                    } else {
                        s.append(&cmd.tokens)?;
                    }
                    return Ok(json!({"event":"prepared","state":state(&s),"memory":s.memory()}));
                }
                let action = match cmd.action {
                    Action::Reset => Input::Reset(&cmd.tokens),
                    Action::Append => Input::Append(&cmd.tokens),
                    Action::Continue => Input::Continue,
                    Action::Prompt => Input::Prompt(&cmd.tokens),
                    Action::Prepare | Action::Stage => unreachable!(),
                };
                let sample = measure_teacher_forced(&mut s, action, &cmd.continuation)?;
                Ok(
                    json!({"event":"sample","sample":sample,"state":state(&s),"memory":s.memory(),"last_completed_timing":s.timing()}),
                )
            })();
            match result {
                Ok(value) => writeln!(out, "{value}")?,
                Err(error) => writeln!(
                    out,
                    "{}",
                    json!({"event":"error","message":error.to_string(),"state":state(&s),"memory":s.memory()})
                )?,
            }
            out.flush()?;
        }
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
        eprintln!("error: Metal requires Apple Silicon macOS");
        std::process::exit(1);
    }
}
