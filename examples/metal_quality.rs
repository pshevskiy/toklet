//! All-required-position Metal scoring with bounded idle handshakes.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[path = "support/loss.rs"]
mod loss;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use super::loss::{APP_BYTES, BLOCK, CAPACITY, POLICY, Sum, loss, spans, tokens};
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{
        io::{self, BufRead, Read, Write},
        path::PathBuf,
    };
    use toklet::{metal, session, weights};
    fn handshake() -> Result<(), Box<dyn std::error::Error>> {
        let mut line = String::new();
        if io::stdin().lock().take(16).read_line(&mut line)? == 0 || line != "continue\n" {
            return Err("bounded idle quality handshake required".into());
        }
        Ok(())
    }
    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let mut args = std::env::args_os().skip(1);
        let source = PathBuf::from(args.next().ok_or("missing model directory")?);
        let packed = args.next().ok_or("missing packed directory or -")?;
        let token_path = PathBuf::from(args.next().ok_or("missing frozen token file")?);
        let chunk = args
            .next()
            .map(|v| v.to_string_lossy().parse::<usize>())
            .transpose()?
            .unwrap_or(128);
        if !matches!(chunk, 1 | 128) {
            return Err("quality chunk must be 1 or 128".into());
        }
        let larger = match args.next() {
            None => false,
            Some(v) if v == "small" => false,
            Some(v) if v == "larger" => true,
            _ => return Err("optional quality geometry must be small or larger".into()),
        };
        let selection = match args.next() {
            None => "original",
            Some(v) if v == "half-compensated" => "half-compensated",
            Some(v) if v == "q4-v2-compensated-tiled" => "q4-v2-compensated-tiled",
            _ => return Err("unknown optional quality projection".into()),
        };
        let half_compensated = selection == "half-compensated";
        let q4_v2 = selection == "q4-v2-compensated-tiled";
        let attention = match args.next() {
            None => metal::AttentionMode::Reference,
            Some(v) if v == "online" => metal::AttentionMode::Online,
            _ => return Err("optional quality attention must be online".into()),
        };
        if args.next().is_some() {
            return Err("extra quality argument".into());
        }
        if half_compensated && packed != "-" {
            return Err("compensated Half scoring requires unquantized Half weights".into());
        }
        if q4_v2 && packed == "-" {
            return Err("Q4 v2 scoring requires packed weights".into());
        }
        let (ids, corpus_sha) = tokens(&token_path)?;
        let targets = ids.len() - 1;
        let budget = weights::INITIAL_MEMORY_LIMIT_BYTES - APP_BYTES;
        let load = weights::LoadOptions {
            memory_limit_bytes: budget,
            ..Default::default()
        };
        let quantized = packed != "-";
        let model = if quantized {
            metal::Model::load_q4(&source, &PathBuf::from(packed), load)
        } else {
            metal::Model::load(&source, load)
        }?;
        if q4_v2 && model.q4_format() != Some(toklet::q4::Format::V2) {
            return Err("explicit Q4 v2 scoring requires a v2 artifact".into());
        }
        let weight_dtype = match model.q4_format() {
            Some(toklet::q4::Format::V1) => "Q4 v1 + FP16 exclusions",
            Some(toklet::q4::Format::V2) => "Q4 v2 + FP16 exclusions",
            None => "FP16",
        };
        if model.config().hidden_size != if larger { 1536 } else { 896 }
            || model.config().num_layers != if larger { 28 } else { 24 }
            || model.config().vocab_size != 151936
        {
            return Err("selected frozen Qwen quality geometry required".into());
        }
        if ids
            .iter()
            .any(|&id| id as usize >= model.config().vocab_size)
        {
            return Err("invalid loss token".into());
        }
        let projection = if q4_v2 {
            metal::ProjectionMode::Q4CompensatedTiled
        } else if half_compensated {
            metal::ProjectionMode::HalfCompensated
        } else if larger && quantized {
            metal::ProjectionMode::Q4Compensated
        } else {
            metal::ProjectionMode::Original
        };
        let normalization = if q4_v2 || larger && quantized {
            metal::NormalizationMode::Compensated
        } else {
            metal::NormalizationMode::Original
        };
        let mut session = metal::Session::with_attention_mode(
            &model,
            session::Options {
                capacity: CAPACITY,
                prefill_chunk_size: chunk,
                memory_limit_bytes: budget,
                ..Default::default()
            },
            projection,
            normalization,
            metal::ReadbackMode::FullLogits,
            attention,
        )?;
        // Every causal row is required for teacher forcing. The diagnostic
        // admission includes both GPU snapshots and their bounded CPU copies.
        session.enable_diagnostics(&[])?;
        let memory = session.memory();
        let width = model.config().vocab_size;
        let mut serialized = vec![0; width * 4];
        if serialized.len() > APP_BYTES / 2 {
            return Err("logit serialization budget exceeded".into());
        }
        let admitted = memory
            .admitted_bytes
            .checked_add(APP_BYTES)
            .ok_or("quality budget overflow")?;
        let mut out = io::stdout().lock();
        let mut readiness = json!({"event":"ready","format":"toklet-metal-held-out-loss-events-v1","backend":"metal","quantized":quantized,"device":model.device_name(),"debug_assertions":cfg!(debug_assertions),"target_tokens":targets,"corpus_sha256":corpus_sha,"context_policy":POLICY,"block_targets":BLOCK,"context_capacity":CAPACITY,"prefill_chunk_size":chunk,"logit_positions":"all_causal_rows","weight_dtype":weight_dtype,"kv_dtype":"FP16","memory":memory,"application_allowance_bytes":APP_BYTES,"memory_admitted_bytes":admitted,"load_seconds":model.load_timing.load_seconds,"hidden_size":model.config().hidden_size,"num_layers":model.config().num_layers,"vocab_size":model.config().vocab_size,"projection_mode":projection,"normalization_mode":normalization,"half_compensated_execution_widths":model.half_compensated_execution_widths(),"q4_format":model.q4_format().map(|f| match f {toklet::q4::Format::V1 => "V1",toklet::q4::Format::V2 => "V2"}),"q4_quantizer":model.q4_format().map(|f| f.quantizer()),"q4_compensated_execution_widths":model.q4_compensated_execution_widths()});
        if attention == metal::AttentionMode::Online {
            readiness["attention_mode"] = json!(attention);
        }
        writeln!(out, "{readiness}")?;
        out.flush()?;
        handshake()?;
        let mut total = Sum::default();
        for (block, range) in spans(targets).enumerate() {
            session.reset();
            let mut block_loss = Sum::default();
            let mut position = 0;
            for part in ids[range.start..=range.end].chunks(chunk) {
                session.append(part)?;
                let rows = session.evaluate_next_chunk()?;
                if rows != part.len()
                    || session.pending_len() != 0
                    || session.evaluated_len() != position + rows
                {
                    return Err("loss chunk commitment differs".into());
                }
                let diagnostics = session.diagnostics()?;
                let logits = diagnostics
                    .iter()
                    .find(|s| s.stage == "logits")
                    .ok_or("all-row logits unavailable")?;
                if logits.rows != rows
                    || logits.columns != width
                    || logits.values.len() != rows * width
                {
                    return Err("loss shape differs".into());
                }
                for row in 0..rows {
                    if position + row == range.len() {
                        break;
                    } // Final target commits; its next distribution is not scored.
                    let target_index = range.start + position + row;
                    let target = ids[target_index + 1];
                    let values = &logits.values[row * width..(row + 1) * width];
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
                // Drop diagnostic host copies before the idle charge read.
                drop(diagnostics);
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
}
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
fn main() {
    if let Err(error) = apple::run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
#[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("Metal quality requires the metal feature on Apple Silicon macOS");
    std::process::exit(1);
}
