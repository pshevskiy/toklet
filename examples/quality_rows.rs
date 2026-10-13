//! Bounded CPU all-row reference scoring. This is not a latency benchmark.
#[path = "support/loss.rs"]
mod loss;
use loss::{APP_BYTES, BLOCK, CAPACITY, POLICY, Sum, loss, spans, tokens};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    io::{self, BufRead, Read, Write},
    path::PathBuf,
};
use toklet::{cpu::ProjectionKernel, session, weights};
fn handshake() -> Result<(), Box<dyn std::error::Error>> {
    let mut line = String::new();
    if io::stdin().lock().take(16).read_line(&mut line)? == 0 || line != "continue\n" {
        return Err("bounded idle quality handshake required".into());
    }
    Ok(())
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let source = PathBuf::from(args.next().ok_or("missing model directory")?);
    let token_path = PathBuf::from(args.next().ok_or("missing frozen token file")?);
    let chunk = args
        .next()
        .map(|v| v.to_string_lossy().parse::<usize>())
        .transpose()?
        .unwrap_or(8);
    if !matches!(chunk, 1 | 8) {
        return Err("reference chunk must be 1 or 8".into());
    }
    if args.next().is_some() {
        return Err("extra quality argument".into());
    }
    let (ids, corpus_sha) = tokens(&token_path)?;
    let targets = ids.len() - 1;
    let budget = weights::INITIAL_MEMORY_LIMIT_BYTES - APP_BYTES;
    let load = weights::LoadOptions {
        memory_limit_bytes: budget,
        ..Default::default()
    };
    let loaded = weights::load_qwen(&source, load)?;
    let model = loaded.model.with_projection_kernel(ProjectionKernel::Tiled);
    if !matches!(
        (
            model.config().hidden_size,
            model.config().num_layers,
            model.config().vocab_size
        ),
        (896, 24, 151936) | (1536, 28, 151936)
    ) {
        return Err("pinned Qwen 0.5B/1.5B reference geometry required".into());
    }
    if ids
        .iter()
        .any(|&id| id as usize >= model.config().vocab_size)
    {
        return Err("invalid loss token".into());
    }
    let mut session = session::Session::new(
        &model,
        session::Options {
            capacity: CAPACITY,
            prefill_chunk_size: chunk,
            memory_limit_bytes: budget,
            ..Default::default()
        },
    )?;
    let memory = session.memory();
    let width = model.config().vocab_size;
    let output_elements = chunk
        .checked_mul(width)
        .ok_or("reference output overflow")?;
    let output_bytes = output_elements
        .checked_mul(size_of::<f32>())
        .ok_or("reference output overflow")?;
    let serialized_bytes = width
        .checked_mul(size_of::<f32>())
        .ok_or("reference serialization overflow")?;
    if output_bytes
        .checked_add(serialized_bytes)
        .ok_or("reference storage overflow")?
        > APP_BYTES / 2
    {
        return Err("reference diagnostic buffers exceed application allowance".into());
    }
    let mut logits = Vec::new();
    logits.try_reserve_exact(output_elements)?;
    logits.resize(output_elements, 0.);
    let mut serialized = Vec::new();
    serialized.try_reserve_exact(serialized_bytes)?;
    serialized.resize(serialized_bytes, 0);
    let admitted = memory
        .admitted_bytes
        .checked_add(APP_BYTES)
        .ok_or("reference budget overflow")?;
    if admitted > weights::INITIAL_MEMORY_LIMIT_BYTES {
        return Err("reference admission exceeds 6 GiB".into());
    }
    let mut out = io::stdout().lock();
    writeln!(
        out,
        "{}",
        json!({"event":"ready","format":"toklet-cpu-row-loss-events-v1","backend":"cpu","cpu_projection":model.projection_kernel().name(),"debug_assertions":cfg!(debug_assertions),"target_tokens":targets,"corpus_sha256":corpus_sha,"context_policy":POLICY,"block_targets":BLOCK,"context_capacity":CAPACITY,"prefill_chunk_size":chunk,"logit_positions":"all_causal_rows","weight_dtype":"FP32","kv_dtype":"FP32","memory":memory,"application_allowance_bytes":APP_BYTES,"memory_admitted_bytes":admitted,"external_logits_bytes":logits.capacity()*size_of::<f32>(),"serialization_bytes":serialized.capacity(),"hidden_size":model.config().hidden_size,"num_layers":model.config().num_layers,"vocab_size":width,"load_seconds":loaded.report.load_seconds})
    )?;
    out.flush()?;
    handshake()?;
    let mut total = Sum::default();
    for (block, range) in spans(targets).enumerate() {
        session.reset();
        let mut block_loss = Sum::default();
        let mut position = 0;
        for part in ids[range.start..=range.end].chunks(chunk) {
            session.append(part)?;
            let rows =
                session.evaluate_next_chunk_with_logits(&mut logits[..part.len() * width])?;
            if rows != part.len()
                || session.pending_len() != 0
                || session.evaluated_len() != position + rows
            {
                return Err("loss chunk commitment differs".into());
            }
            for row in 0..rows {
                if position + row == range.len() {
                    break;
                } // Final target commits; its next distribution is not scored.
                let target_index = range.start + position + row;
                let target = ids[target_index + 1];
                let values = &logits[row * width..(row + 1) * width];
                let nll = loss(values, target as usize)?;
                for (bytes, &value) in serialized.chunks_exact_mut(4).zip(values) {
                    bytes.copy_from_slice(&value.to_le_bytes());
                }
                total.add(nll);
                block_loss.add(nll);
                writeln!(
                    out,
                    "{}",
                    json!({"event":"loss","target_index":target_index,"block_index":block,"position":position+row,"prefix_tokens":position+row+1,"target_token":target,"negative_log_likelihood":nll,"target_logit_f32_bits":values[target as usize].to_bits(),"logits_sha256":format!("{:x}",Sha256::digest(&serialized))})
                )?;
            }
            position += rows;
            writeln!(
                out,
                "{}",
                json!({"event":"chunk_done","block_index":block,"rows":rows,"evaluated_tokens":position,"scored_targets":position.min(range.len()),"pending_tokens":0})
            )?;
            out.flush()?;
            handshake()?;
        }
        writeln!(
            out,
            "{}",
            json!({"event":"block_done","block_index":block,"target_start":range.start,"target_end":range.end,"scored_targets":range.len(),"total_negative_log_likelihood":block_loss.value,"evaluated_tokens":session.evaluated_len(),"pending_tokens":session.pending_len()})
        )?;
        out.flush()?;
    }
    writeln!(
        out,
        "{}",
        json!({"event":"done","scored_targets":targets,"total_negative_log_likelihood":total.value,"mean_negative_log_likelihood":total.value/targets as f64,"evaluated_tokens":session.evaluated_len(),"pending_tokens":session.pending_len()})
    )?;
    out.flush()?;
    handshake()?;
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
