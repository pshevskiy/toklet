use super::{
    APPLICATION_ALLOWANCE, Args, CANCELLED, chat, invalid, logical_memory, publish_terminal,
    token_list,
};
use serde_json::json;
use std::{
    io::{self, Write},
    sync::atomic::Ordering,
    time::Instant,
};
use toklet::{
    app_chat::{self, nanbeige as codec, stream::ResponseLimits},
    generation::{self, Input, Request, StopReason},
    metal::nanbeige::{Model, Session},
    mlx::nanbeige::{EOS, LOGIT_ROWS, PROFILE, REVISION, TOKENIZER_VOCAB},
    session::{InferenceSession, Options},
    weights::INITIAL_MEMORY_LIMIT_BYTES,
};

pub(super) fn run(a: Args) -> Result<u8, Box<dyn std::error::Error>> {
    let started = Instant::now();
    ctrlc::try_set_handler(|| CANCELLED.store(true, Ordering::Relaxed))?;
    let needs_codec = a.prompt.is_some() || a.messages.is_some() || !a.json;
    let external = APPLICATION_ALLOWANCE
        + usize::from(needs_codec) * codec::ALLOWANCE_BYTES
        + usize::from(!a.json) * codec::STREAM_ALLOWANCE_BYTES;
    let options = Options {
        capacity: a.capacity,
        prefill_chunk_size: a.chunk,
        position_offset: 0,
        memory_limit_bytes: INITIAL_MEMORY_LIMIT_BYTES - external,
    };
    let clock = Instant::now();
    let codec = needs_codec
        .then(|| codec::Codec::load(&a.model, codec::ALLOWANCE_BYTES))
        .transpose()?;
    let tokenizer_load_seconds = clock.elapsed().as_secs_f64();
    let clock = Instant::now();
    let tokens = if let Some(ids) = a.tokens.as_deref() {
        token_list(ids)?
    } else if let Some(path) = &a.token_file {
        let raw = chat::read_bounded(path, a.capacity * 4)?;
        if !raw.len().is_multiple_of(4) {
            return Err(invalid("token file must contain complete u32 values").into());
        }
        raw.chunks_exact(4)
            .map(|row| u32::from_le_bytes(row.try_into().unwrap()))
            .collect()
    } else {
        let codec = codec
            .as_ref()
            .ok_or_else(|| invalid("missing Nanbeige text codec"))?;
        if let Some(prompt) = &a.prompt {
            if a.raw {
                codec.encode(prompt)?
            } else {
                let messages = serde_json::to_vec(&json!([{"role": "user", "content": prompt}]))?;
                codec.encode_chat(&messages, !a.no_thinking)?
            }
        } else {
            let path = a
                .messages
                .as_ref()
                .ok_or_else(|| invalid("missing Nanbeige messages"))?;
            let messages = chat::read_bounded(path, app_chat::MAX_TEXT)?;
            codec.encode_chat(&messages, !a.no_thinking)?
        }
    };
    let input_seconds = clock.elapsed().as_secs_f64();
    let mut stops = a.stop;
    if !a.no_eos && !stops.contains(&EOS) {
        stops.push(EOS);
    }
    if tokens.is_empty()
        || tokens.len() > a.capacity - a.max
        || tokens
            .iter()
            .chain(&stops)
            .any(|&id| id as usize >= LOGIT_ROWS)
    {
        return Err(invalid("Nanbeige input or output exceeds the admitted model/context").into());
    }

    let clock = Instant::now();
    let want_dense = a.dense_prefill && options.prefill_chunk_size == 128 && tokens.len() > 128;
    let model = if want_dense {
        Model::load_dense(&a.model, options, false)?
    } else {
        Model::load(&a.model, options, false)?
    };
    let load_seconds = clock.elapsed().as_secs_f64();
    let mut session = Session::new(&model, options, false, &CANCELLED)?;
    let mut stream = if a.json {
        None
    } else {
        let codec = codec
            .as_ref()
            .ok_or_else(|| invalid("missing Nanbeige text codec"))?;
        let stream = codec.response_stream(ResponseLimits::new(a.max.max(1), app_chat::MAX_TEXT)?);
        if stream.allocated_bytes()? > codec::STREAM_ALLOWANCE_BYTES {
            return Err(invalid("Nanbeige response stream exceeds its allowance").into());
        }
        Some(stream)
    };

    let mut out = io::stdout().lock();
    let inference = Instant::now();
    let mut first_output = None;
    let mut output_seconds = 0.;
    let mut report = generation::generate(
        &mut session,
        Request {
            input: Input::Prompt(&tokens),
            max_new_tokens: a.max,
            stop_tokens: &stops,
        },
        &CANCELLED,
        |event| {
            let clock = Instant::now();
            let emitted = if let Some(stream) = &mut stream {
                if event.token_id >= TOKENIZER_VOCAB {
                    return Err(invalid(format!(
                        "Nanbeige selected padded output ID {} outside the tokenizer vocabulary",
                        event.token_id,
                    )));
                }
                let text = stream.push(&[event.token_id])?;
                out.write_all(text.as_bytes())?;
                !text.is_empty()
            } else {
                writeln!(
                    out,
                    "{}",
                    json!({"event":"token", "token_id":event.token_id, "index":event.index,
                        "elapsed_seconds":event.elapsed_seconds})
                )?;
                true
            };
            out.flush()?;
            let flushed = Instant::now();
            if emitted {
                first_output.get_or_insert(flushed);
            }
            output_seconds += flushed.duration_since(clock).as_secs_f64();
            Ok(())
        },
    )?;
    if let Some(stream) = &mut stream {
        let clock = Instant::now();
        let tail = stream.finish()?;
        out.write_all(tail.as_bytes())?;
        writeln!(out)?;
        out.flush()?;
        let flushed = Instant::now();
        if !tail.is_empty() {
            first_output.get_or_insert(flushed);
        }
        output_seconds += flushed.duration_since(clock).as_secs_f64();
    }
    publish_terminal(&CANCELLED, |cancelled| {
        if cancelled && report.reason == StopReason::OutputLimit {
            report.reason = StopReason::Cancelled;
        }
        let exit = if report.reason == StopReason::Cancelled {
            130
        } else {
            0
        };
        if a.json || a.stats {
            let memory = session.memory();
            let mut stats = json!({
                "event": "stats",
                "model_profile": PROFILE,
                "revision": REVISION,
                "backend": "metal",
                "device": model.device_name(),
                "projection": {"multirow": "tiled_bf16_weights_fp32_accumulate",
                               "single_row": "grouped_affine_vector_fp32_accumulate",
                               "ffn_m32_prefill_rows": model.ffn_m32_prefill_rows(),
                               "gate_up_compact_prefill_rows": model.gate_up_compact_prefill_rows(),
                               "down_k64_prefill_rows": session.down_k64_prefill_rows(),
                               "attention_compact_prefill_rows": session.attention_compact_prefill_rows()},
                "kv_dtype": "BF16",
                "sampler": "raw_logits_greedy",
                "numerical_qualification": false,
                "tokenizer_ids": TOKENIZER_VOCAB,
                "logit_rows": LOGIT_ROWS,
                "prompt_tokens": tokens.len(),
                "context_capacity": a.capacity,
                "prefill_chunk_size": a.chunk,
                "generation": report,
                "decode_evaluations": report.decode_evaluations,
                "input_seconds": input_seconds,
                "first_output_seconds": first_output.map(|t| t.duration_since(inference).as_secs_f64()),
                "start_to_first_output_seconds": first_output.map(|t| t.duration_since(started).as_secs_f64()),
                "tokenizer_load_seconds": tokenizer_load_seconds,
                "load_seconds": load_seconds,
                "output_seconds": output_seconds,
                "total_seconds": started.elapsed().as_secs_f64(),
                "logical_memory": logical_memory(memory, external),
                "memory_admitted_bytes": memory.admitted_bytes + external,
            });
            if a.dense_prefill {
                let available = session.dense_gate_up_prefill_rows() == [128];
                let committed = if available {
                    (report.prefill_tokens / 128).min((tokens.len() - 1) / 128)
                } else {
                    0
                };
                let allocation = model.dense_prefill_receipt().map(|r| json!({
                    "physical_pairs":r.physical_pairs,"allocated_bytes":r.allocated_bytes,
                    "payload_bytes":r.payload_bytes,"allocation_seconds":r.allocation_seconds,
                    "prepare_host_seconds":r.prepare_host_seconds,"prepare_gpu_seconds":r.prepare_gpu_seconds,
                    "startup_basis_seconds":r.startup_basis_seconds,
                    "device_admitted_limit_bytes":r.device_budget_bytes,
                }));
                let denial = model.dense_prefill_denial().or({
                    if !want_dense {
                        Some("dense prefill requires a T128 chunk with another prompt position pending")
                    } else if !available {
                        Some("dense prefill normalization scratch is unsupported")
                    } else {
                        None
                    }
                });
                stats["dense_prefill"] = json!({
                    "requested": true, "admitted": allocation.is_some(), "available":available,
                    "committed_dense_chunks":committed, "committed_dense_rows":committed * 128,
                    "allocation": allocation, "denial":denial,
                });
            }
            if a.json {
                writeln!(out, "{stats}")?;
                out.flush()?;
            } else {
                writeln!(io::stderr().lock(), "{stats}")?;
            }
        }
        Ok(exit)
    })
}
