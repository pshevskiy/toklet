//! Bounded raw-ID larger FP32 task reference. No tokenizer is instantiated.
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    io::{self, BufRead, Read, Write},
    path::PathBuf,
    sync::atomic::AtomicBool,
};
use toklet::{
    cpu::ProjectionKernel,
    generation::{self, Input, Request},
    session::{Options, Session},
    weights,
};
const APP: usize = 64 * 1024 * 1024;
const CAPACITY: usize = 562;
const CHUNK: usize = 8;
const EOS: [u32; 2] = [151643, 151645];
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Action {
    Generate,
    Reset,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Command {
    action: Action,
    #[serde(default)]
    tokens: Vec<u32>,
    max_new_tokens: Option<usize>,
}
fn state(s: &Session<'_>) -> serde_json::Value {
    let mut h = Sha256::new();
    for id in s.history() {
        h.update(id.to_le_bytes());
    }
    json!({"history_tokens":s.history().len(),"evaluated_tokens":s.evaluated_len(),"pending_tokens":s.pending_len(),"history_sha256":format!("{:x}",h.finalize())})
}
fn handle(
    cmd: Command,
    s: &mut Session<'_>,
    out: &mut impl Write,
    eos: &[u32],
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    match cmd.action {
        Action::Reset => {
            if !cmd.tokens.is_empty() || cmd.max_new_tokens.is_some() {
                return Err("reset accepts no fields".into());
            }
            s.reset();
            Ok(json!({"event":"reset"}))
        }
        Action::Generate => {
            let cap = cmd.max_new_tokens.ok_or("missing output cap")?;
            if cap == 0 || cap > 256 {
                return Err("raw task output cap must be1..256".into());
            }
            let report = generation::generate(
                s,
                Request {
                    input: Input::Reset(&cmd.tokens),
                    max_new_tokens: cap,
                    stop_tokens: eos,
                },
                &AtomicBool::new(false),
                |e| {
                    writeln!(
                        out,
                        "{}",
                        json!({"event":"token","token_id":e.token_id,"index":e.index,"elapsed_seconds":e.elapsed_seconds})
                    )?;
                    out.flush()
                },
            )?;
            Ok(
                json!({"event":"done","statistics":{"generation":report,"backend":"cpu","cpu_projection":"tiled","memory":s.memory(),"memory_admitted_bytes":s.memory().admitted_bytes+APP,"tokenizer_resident":false,"state":state(s)}}),
            )
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let source = PathBuf::from(args.next().ok_or("usage: cpu_task_reference MODEL")?);
    if args.next().is_some() {
        return Err("too many arguments".into());
    }
    let budget = weights::INITIAL_MEMORY_LIMIT_BYTES - APP;
    let loaded = weights::load_qwen(
        &source,
        weights::LoadOptions {
            memory_limit_bytes: budget,
            ..Default::default()
        },
    )?;
    let model = loaded.model.with_projection_kernel(ProjectionKernel::Tiled);
    let c = model.config();
    if c.hidden_size != 1536 || c.num_layers != 28 || c.vocab_size != 151936 {
        return Err("fixed larger task reference geometry required".into());
    }
    let mut s = Session::new(
        &model,
        Options {
            capacity: CAPACITY,
            prefill_chunk_size: CHUNK,
            memory_limit_bytes: budget,
            ..Default::default()
        },
    )?;
    let mut out = io::stdout().lock();
    writeln!(
        out,
        "{}",
        json!({"event":"ready","format":"toklet-cpu-task-reference-v1","debug_assertions":cfg!(debug_assertions),"hidden_size":c.hidden_size,"num_layers":c.num_layers,"vocab_size":c.vocab_size,"backend":"cpu","cpu_projection":"tiled","weight_dtype":"FP32","kv_dtype":"FP32","normalization_mode":"OrderedFP64","readback_mode":"FullLogits","context_capacity":CAPACITY,"chunk_size":CHUNK,"tokenizer_resident":false,"eos_ids":EOS,"memory":s.memory(),"application_bytes":APP,"memory_admitted_bytes":s.memory().admitted_bytes+APP,"load_seconds":loaded.report.load_seconds,"conversion_seconds":loaded.report.conversion_seconds})
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
            return Err("raw task line exceeds64KiB".into());
        }
        count += 1;
        if count > 256 {
            return Err("raw task request count exceeds256".into());
        }
        let result = serde_json::from_slice::<Command>(&line)
            .map_err(|error| -> Box<dyn std::error::Error> { error.into() })
            .and_then(|cmd| handle(cmd, &mut s, &mut out, &EOS));
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
fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use toklet::model::{Config, LayerWeights, Model, Projection, Weights};
    fn model() -> Model {
        let c = Config {
            vocab_size: 2,
            hidden_size: 2,
            intermediate_size: 2,
            num_layers: 1,
            query_heads: 1,
            kv_heads: 1,
            head_dim: 2,
            max_positions: CAPACITY,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.,
        };
        let projection = || Projection {
            weight: vec![0.; 4],
            bias: vec![0.; 2],
        };
        Model::new(
            c,
            Weights {
                embedding: vec![0.; 4],
                layers: vec![LayerWeights {
                    input_norm: vec![1.; 2],
                    post_attention_norm: vec![1.; 2],
                    q: projection(),
                    k: projection(),
                    v: projection(),
                    o: vec![0.; 4],
                    gate: vec![0.; 4],
                    up: vec![0.; 4],
                    down: vec![0.; 4],
                }],
                final_norm: vec![1.; 2],
            },
        )
        .unwrap()
        .with_projection_kernel(ProjectionKernel::Tiled)
    }
    #[test]
    fn raw_generation_preserves_hidden_eos_caps_invalid_prefix_and_reset() {
        let m = model();
        let mut s = Session::new(
            &m,
            Options {
                capacity: CAPACITY,
                prefill_chunk_size: CHUNK,
                ..Default::default()
            },
        )
        .unwrap();
        let mut out = Vec::new();
        let value = handle(
            Command {
                action: Action::Generate,
                tokens: vec![1; 306],
                max_new_tokens: Some(256),
            },
            &mut s,
            &mut out,
            &[0],
        )
        .unwrap();
        let g = &value["statistics"]["generation"];
        assert_eq!(g["sampled_tokens"], 1);
        assert_eq!(g["emitted_tokens"], 0);
        assert_eq!(g["evaluated_tokens"], 306);
        assert_eq!(g["pending_tokens"], 1);
        assert!(out.is_empty());
        assert_eq!(g["reason"]["token_id"], 0);
        let before = state(&s);
        for cmd in [
            Command {
                action: Action::Generate,
                tokens: vec![1; 307],
                max_new_tokens: Some(256),
            },
            Command {
                action: Action::Generate,
                tokens: vec![2],
                max_new_tokens: Some(1),
            },
            Command {
                action: Action::Reset,
                tokens: vec![1],
                max_new_tokens: None,
            },
        ] {
            assert!(handle(cmd, &mut s, &mut out, &[0]).is_err());
            assert_eq!(state(&s), before);
        }
        assert_eq!(
            handle(
                Command {
                    action: Action::Reset,
                    tokens: vec![],
                    max_new_tokens: None
                },
                &mut s,
                &mut out,
                &[1]
            )
            .unwrap()["event"],
            "reset"
        );
        assert_eq!(s.evaluated_len(), 0);
        assert!(s.history().is_empty());
        let value = handle(
            Command {
                action: Action::Generate,
                tokens: vec![1; 306],
                max_new_tokens: Some(256),
            },
            &mut s,
            &mut out,
            &[1],
        )
        .unwrap();
        let g = &value["statistics"]["generation"];
        assert_eq!(g["reason"]["kind"], "output_limit");
        assert_eq!(g["sampled_tokens"], 256);
        assert_eq!(g["emitted_tokens"], 256);
        assert_eq!(g["evaluated_tokens"], 561);
        assert_eq!(g["pending_tokens"], 1);
        let events: Vec<serde_json::Value> = out
            .split(|&b| b == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect();
        assert_eq!(events.len(), 256);
        for (i, e) in events.iter().enumerate() {
            assert_eq!(e["index"], i);
            assert_eq!(e["token_id"], 0);
            assert!(e.get("text").is_none());
        }
        s.evaluate().unwrap();
        assert_eq!(s.evaluated_len(), 562);
        let before = state(&s);
        assert!(s.append(&[0]).is_err());
        assert_eq!(state(&s), before);
    }
}
