use super::{Args, CANCELLED, chat, invalid, publish_terminal, token_list};
use serde_json::json;
use std::{
    io::{self, Write},
    sync::atomic::Ordering,
    time::Instant,
};
use toklet::{
    app_chat::{sensenova_loop as codec, stream::ResponseLimits},
    mlx::sensenova_loop::{self as mlx, payload, text_weights::SmallWeights},
    model::{
        qwen35::{Error, Options},
        sensenova_loop::{Plan, text::ScalarSession},
    },
    sampling::greedy,
    weights::INITIAL_MEMORY_LIMIT_BYTES,
};

const PROFILE: &str = "sensenova-loop-9b-mlx";
const PROMOTION_BYTES: usize = 16_829_952;

fn tokens_per_second(tokens: usize, seconds: f64) -> Option<f64> {
    (tokens != 0 && seconds > 0.0).then(|| tokens as f64 / seconds)
}

pub(super) fn run(a: Args) -> Result<u8, Box<dyn std::error::Error>> {
    let started = Instant::now();
    ctrlc::try_set_handler(|| CANCELLED.store(true, Ordering::Relaxed))?;
    let need_codec = a.prompt.is_some() || a.messages.is_some() || !a.json;
    let external = super::APPLICATION_ALLOWANCE
        + if need_codec {
            codec::ALLOWANCE_BYTES + codec::STREAM_ALLOWANCE_BYTES
        } else {
            0
        };
    let options = Options {
        capacity: a.capacity,
        chunk: a.chunk,
        position_offset: 0,
        memory_limit_bytes: INITIAL_MEMORY_LIMIT_BYTES - external,
    };
    let config = chat::read_bounded(&a.model.join("config.json"), 8192)?;
    let plan = Plan::from_mlx_config(&config)?;
    ScalarSession::planned(
        &plan,
        options,
        payload::ADMITTED_LOAD_BYTES + PROMOTION_BYTES,
    )?;
    let clock = Instant::now();
    let codec = need_codec
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
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    } else {
        let messages = if let Some(prompt) = a.prompt.as_deref() {
            serde_json::to_vec(&json!([{"role":"user","content":prompt}]))?
        } else {
            chat::read_bounded(
                a.messages
                    .as_deref()
                    .ok_or_else(|| invalid("missing messages"))?,
                codec::MAX_TEXT_BYTES,
            )?
        };
        codec
            .as_ref()
            .ok_or_else(|| invalid("missing text codec"))?
            .encode_chat(&messages)?
    };
    let input_seconds = clock.elapsed().as_secs_f64();
    let vocab = plan.text_shape().vocab_size;
    if tokens.is_empty()
        || tokens.len() + a.max > a.capacity
        || tokens.iter().chain(&a.stop).any(|&id| id as usize >= vocab)
    {
        return Err(invalid("SenseNova token IDs or total context exceed bounds").into());
    }
    let clock = Instant::now();
    let payload = payload::load(&a.model, payload::ADMITTED_LOAD_BYTES)?;
    let load_seconds = clock.elapsed().as_secs_f64();
    let clock = Instant::now();
    let small = SmallWeights::new(&payload, PROMOTION_BYTES)?;
    let weights = small.view()?;
    let binding_seconds = clock.elapsed().as_secs_f64();
    let model_bytes = payload::ADMITTED_LOAD_BYTES + small.promoted_bytes();
    let mut session = ScalarSession::new(&plan, &weights, options, model_bytes)?;

    let mut stream = if a.json {
        None
    } else {
        Some(
            codec
                .as_ref()
                .ok_or_else(|| invalid("missing output codec"))?
                .response_stream(
                    ResponseLimits::new(a.max.max(1), codec::MAX_TEXT_BYTES)?,
                    codec::STREAM_ALLOWANCE_BYTES,
                )?,
        )
    };
    let mut out = io::stdout().lock();
    let inference = Instant::now();
    let mut sampled = 0;
    let mut emitted = 0;
    let mut stop = None;
    let mut first_token_seconds = None;
    let mut first_output = None;
    let mut prefill_seconds = 0.0;
    let mut decode_seconds = 0.0;
    let mut sampling_seconds = 0.0;
    let mut output_seconds = 0.0;
    let evaluation = (|| -> Result<(), Box<dyn std::error::Error>> {
        for chunk in tokens.chunks(a.chunk) {
            let clock = Instant::now();
            session.forward(chunk, &CANCELLED)?;
            prefill_seconds += clock.elapsed().as_secs_f64();
        }
        let mut previous = None;
        for index in 0..a.max {
            if CANCELLED.load(Ordering::Relaxed) {
                break;
            }
            if let Some(token) = previous {
                let clock = Instant::now();
                session.forward(&[token], &CANCELLED)?;
                decode_seconds += clock.elapsed().as_secs_f64();
            }
            let clock = Instant::now();
            let token = greedy(
                session
                    .logits()
                    .ok_or_else(|| invalid("missing SenseNova logits"))?,
            )?;
            sampling_seconds += clock.elapsed().as_secs_f64();
            if CANCELLED.load(Ordering::Relaxed) {
                break;
            }
            sampled += 1;
            if a.stop.contains(&token) || (!a.no_eos && codec::STOP_IDS.contains(&token)) {
                stop = Some(token);
                break;
            }
            previous = Some(token);
            let clock = Instant::now();
            let elapsed_seconds = inference.elapsed().as_secs_f64();
            let wrote_output = if a.json {
                writeln!(
                    out,
                    "{}",
                    json!({"event":"token", "token_id":token, "index":index, "elapsed_seconds":elapsed_seconds})
                )?;
                true
            } else if let Some(stream) = &mut stream {
                let text = stream.push(&[token])?;
                out.write_all(text.as_bytes())?;
                !text.is_empty()
            } else {
                false
            };
            out.flush()?;
            let flushed = Instant::now();
            if wrote_output {
                first_output.get_or_insert(flushed);
            }
            output_seconds += flushed.duration_since(clock).as_secs_f64();
            emitted += 1;
            first_token_seconds.get_or_insert(elapsed_seconds);
        }
        Ok(())
    })();
    if let Err(error) = evaluation
        && !matches!(error.downcast_ref::<Error>(), Some(Error::Cancelled))
    {
        return Err(error);
    }
    if let Some(stream) = &mut stream {
        let clock = Instant::now();
        let text = stream.finish()?;
        out.write_all(text.as_bytes())?;
        writeln!(out)?;
        out.flush()?;
        let flushed = Instant::now();
        if !text.is_empty() {
            first_output.get_or_insert(flushed);
        }
        output_seconds += flushed.duration_since(clock).as_secs_f64();
    }
    publish_terminal(&CANCELLED, |cancelled| {
        let exit_code = if cancelled { 130 } else { 0 };
        if !a.json && !a.stats {
            return Ok(exit_code);
        }
        let reason = if cancelled {
            "cancelled"
        } else if stop.is_some() {
            "stop_token"
        } else {
            "output_limit"
        };
        let memory = session.memory();
        let committed_tokens = session.history().len();
        let prefill_tokens_processed = committed_tokens.min(tokens.len());
        let decode_tokens_processed = committed_tokens - prefill_tokens_processed;
        let first_output_seconds =
            first_output.map(|flushed| flushed.duration_since(inference).as_secs_f64());
        let start_to_first_output_seconds =
            first_output.map(|flushed| flushed.duration_since(started).as_secs_f64());
        let report = json!({
            "event":"stats", "model_profile":PROFILE, "revision":mlx::REVISION,
            "backend":"cpu", "projection_mode":"PackedQ4G64ScalarReference", "kv_dtype":"FP32",
            "numerical_qualification":false, "reason":reason, "stop_token":stop,
            "prompt_tokens":tokens.len(), "sampled_tokens":sampled, "emitted_tokens":emitted,
            "committed_tokens":committed_tokens, "context_capacity":a.capacity,
            "prefill_chunk_size":a.chunk, "first_token_seconds":first_token_seconds,
            "first_output_seconds":first_output_seconds,
            "start_to_first_output_seconds":start_to_first_output_seconds,
            "tokenizer_load_seconds":tokenizer_load_seconds, "input_seconds":input_seconds,
            "load_seconds":load_seconds, "binding_seconds":binding_seconds,
            "prefill_seconds":prefill_seconds, "decode_seconds":decode_seconds,
            "prefill_tokens_processed":prefill_tokens_processed,
            "decode_tokens_processed":decode_tokens_processed,
            "prefill_tokens_per_second":tokens_per_second(prefill_tokens_processed, prefill_seconds),
            "decode_tokens_per_second":tokens_per_second(decode_tokens_processed, decode_seconds),
            "sampling_seconds":sampling_seconds, "output_seconds":output_seconds,
            "total_seconds":started.elapsed().as_secs_f64(),
            "weights_bytes":mlx::TEXT_BYTES, "promoted_bytes":small.promoted_bytes(),
            "memory_admitted_bytes":memory.admitted_bytes + external,
            "kv_bytes":memory.kv_bytes, "recurrent_bank_bytes":memory.recurrent_bank_bytes,
            "convolution_bank_bytes":memory.convolution_bank_bytes, "scratch_bytes":memory.scratch_bytes
        });
        if a.json {
            writeln!(out, "{report}")?;
        } else {
            writeln!(io::stderr().lock(), "{report}")?;
        }
        out.flush()?;
        Ok(exit_code)
    })
}
