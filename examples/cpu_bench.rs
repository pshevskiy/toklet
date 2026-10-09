//! Explicit local-model benchmark worker. JSON transport is outside timers.
use serde::Deserialize;
use serde_json::json;
use std::{
    io::{self, BufRead, Read, Write},
    path::PathBuf,
};
use toklet::{
    benchmark,
    generation::Input,
    session::{Options, Session},
    weights::{self, LoadOptions},
};

const APP_BYTES: usize = 64 * 1024 * 1024;
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
struct Request {
    action: Action,
    #[serde(default)]
    tokens: Vec<u32>,
    #[serde(default)]
    continuation: Vec<u32>,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = PathBuf::from(args.next().ok_or("usage: cpu_bench MODEL [CHUNK]")?);
    let chunk = args
        .next()
        .map(|v| {
            v.into_string()
                .map_err(|_| "invalid chunk")?
                .parse::<usize>()
                .map_err(|_| "invalid chunk")
        })
        .transpose()?
        .unwrap_or(128);
    if args.next().is_some() {
        return Err("too many arguments".into());
    }
    let budget = weights::INITIAL_MEMORY_LIMIT_BYTES - APP_BYTES;
    let loaded = weights::load_qwen(
        &path,
        LoadOptions {
            memory_limit_bytes: budget,
            ..LoadOptions::default()
        },
    )?;
    let mut session = Session::new(
        &loaded.model,
        Options {
            prefill_chunk_size: chunk,
            memory_limit_bytes: budget,
            ..Options::default()
        },
    )?;
    let m = session.memory();
    let mut out = io::stdout().lock();
    writeln!(
        out,
        "{}",
        json!({"event":"ready","load_seconds":loaded.report.load_seconds,
        "conversion_seconds":loaded.report.conversion_seconds,"memory_admitted_bytes":m.admitted_bytes+APP_BYTES,
        "logical_weights_bytes":m.weights_bytes,"allocated_kv_bytes":m.kv_bytes,"scratch_bytes":m.scratch_bytes,
        "chunk_size":chunk,"context_capacity":session.capacity()})
    )?;
    out.flush()?;
    let mut input = io::BufReader::new(io::stdin());
    loop {
        let mut line = Vec::new();
        let n = input.by_ref().take(65537).read_until(b'\n', &mut line)?;
        if n == 0 {
            break;
        }
        if n > 65536 {
            return Err("benchmark line exceeds 64 KiB".into());
        }
        let result = (|| -> Result<serde_json::Value, Box<dyn std::error::Error>> {
            let request: Request = serde_json::from_slice(&line)?;
            if matches!(request.action, Action::Continue) && !request.tokens.is_empty() {
                return Err("continue accepts no tokens".into());
            }
            if matches!(request.action, Action::Prepare | Action::Stage) {
                if !request.continuation.is_empty() {
                    return Err("prepare accepts no continuation".into());
                }
                if matches!(request.action, Action::Prepare) {
                    session.set_tokens(&request.tokens)?;
                    session.evaluate()?;
                } else {
                    session.append(&request.tokens)?;
                }
                return Ok(
                    json!({"event":"prepared","evaluated_tokens":session.evaluated_len(), "pending_tokens":session.pending_len()}),
                );
            }
            let action = match request.action {
                Action::Reset => Input::Reset(&request.tokens),
                Action::Append => Input::Append(&request.tokens),
                Action::Continue => Input::Continue,
                Action::Prompt => Input::Prompt(&request.tokens),
                Action::Prepare | Action::Stage => unreachable!(),
            };
            let sample =
                benchmark::measure_teacher_forced(&mut session, action, &request.continuation)?;
            Ok(json!({"event":"sample","sample":sample}))
        })();
        match result {
            Ok(record) => writeln!(out, "{record}")?,
            Err(error) => writeln!(
                out,
                "{}",
                json!({"event":"error","message":error.to_string()})
            )?,
        }
        out.flush()?;
    }
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
