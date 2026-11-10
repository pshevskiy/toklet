//! Explicit bounded text journey; no managed/default model-route selection.
use super::contract::*;
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    time::Instant,
};
use toklet::{
    app_chat::{minicpm5 as text, stream::ResponseLimits},
    metal::{self, AttentionMode, NormalizationMode, ProjectionMode, ReadbackMode},
    session,
    weights::LoadOptions,
};
const TOTAL: usize = 6 * 1024 * 1024 * 1024;
const APP: usize = 32 * 1024 * 1024;
const MAX_NEW: usize = 256;

fn nanos(clock: &Instant) -> Result<u64> {
    Ok(clock.elapsed().as_nanos().try_into()?)
}
fn input(path: &Path) -> Result<Vec<u8>> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)?;
    if !file.metadata()?.is_file() || file.metadata()?.len() > text::MAX_TEXT_BYTES as u64 {
        return Err("bounded regular text input required".into());
    }
    let mut raw = Vec::new();
    file.take(text::MAX_TEXT_BYTES as u64 + 1)
        .read_to_end(&mut raw)?;
    if raw.len() > text::MAX_TEXT_BYTES {
        return Err("text input grew beyond bound".into());
    }
    Ok(raw)
}
fn output(path: &Path, value: &serde_json::Value) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    Ok(())
}
fn max_new(raw: Option<&std::ffi::OsString>) -> Result<usize> {
    let value = match raw {
        None => 16,
        Some(raw) => raw.to_str().ok_or("non-UTF8 output limit")?.parse()?,
    };
    if !(1..=MAX_NEW).contains(&value) {
        return Err("text output limit must be1..256".into());
    }
    Ok(value)
}
fn open(model: &metal::Model, limit: usize) -> Result<metal::Session<'_>> {
    Ok(metal::Session::with_attention_mode(
        model,
        session::Options {
            capacity: 512,
            prefill_chunk_size: 32,
            position_offset: 0,
            memory_limit_bytes: limit,
        },
        ProjectionMode::MlxAffine,
        NormalizationMode::Compensated,
        ReadbackMode::FullLogits,
        AttentionMode::Reference,
    )?)
}
#[derive(Serialize)]
struct Chunk {
    decision: usize,
    rows: usize,
    committed_after: usize,
    pending_after: usize,
    wall_ns: u64,
    completed_timing: metal::Timing,
}
#[derive(Serialize)]
struct Decision {
    step: usize,
    selected: u32,
    decision_ns: u64,
    selection_ns: u64,
    decode_ns: u64,
    delta_bytes: usize,
    text_bytes: usize,
    first_chunk: usize,
    chunks: usize,
    committed_after: usize,
    pending_after: usize,
}

pub fn run() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(2).collect();
    if !(5..=6).contains(&args.len()) {
        return Err("usage: minicpm5_probe text MODEL MESSAGES OPTIONS NEW-OUTPUT serial|parallel-store|parity [MAX-NEW=16]".into());
    }
    let mode = args[4].to_str().ok_or("non-UTF8 text mode")?;
    if !["serial", "parallel-store", "parity"].contains(&mode) {
        return Err("unknown text mode".into());
    }
    let maximum = max_new(args.get(5))?;
    let model_path = PathBuf::from(&args[0]);
    let out = PathBuf::from(&args[3]);
    let messages = input(Path::new(&args[1]))?;
    let options_raw = input(Path::new(&args[2]))?;
    let options: text::RenderOptions = serde_json::from_slice(&options_raw)?;
    // Typed validation before large owners. The codec reserve also covers its
    // lifetime during model loading; no second full numerical copy is needed.
    text::render(&messages, options)?;
    for &(name, size, pin) in FILES {
        hash_file(&model_path.join(name), size, pin)?;
    }
    fs::create_dir(&out)?;
    let external = APP
        .checked_add(text::ALLOWANCE_BYTES)
        .and_then(|n| n.checked_add(text::STREAM_ALLOWANCE_BYTES))
        .ok_or("text external admission overflow")?;
    let limit = TOTAL
        .checked_sub(external)
        .ok_or("text external owners exceed6GiB")?;
    let clock = Instant::now();
    let codec = text::Codec::load(&model_path, text::ALLOWANCE_BYTES)?;
    let codec_load_ns = nanos(&clock)?;
    let clock = Instant::now();
    let prompt = codec.encode_chat(&messages, options)?;
    let preflight_text_ns = nanos(&clock)?;
    if prompt.is_empty() || prompt.len().checked_add(maximum).is_none_or(|n| n > 512) {
        return Err("text prompt plus output exceeds512tokens".into());
    }
    let model = metal::Model::load_mlx(
        &model_path,
        LoadOptions {
            context_limit: 512,
            memory_limit_bytes: limit,
        },
    )?;
    if model.weights_bytes() != 2674218150 || model.positional_resident_bytes() != 0 {
        return Err("text model resident identity differs".into());
    }
    if mode == "parity" {
        return parity(&model, &codec, &prompt, maximum, limit, external, &out);
    }
    let clock = Instant::now();
    let mut session = open(&model, limit)?;
    if mode == "parallel-store" {
        session.enable_rms_parallel_store()?;
    }
    let session_setup_ns = nanos(&clock)?;
    if session.rms_parallel_store_enabled() != (mode == "parallel-store")
        || session
            .memory()
            .admitted_bytes
            .checked_add(external)
            .is_none_or(|n| n > TOTAL)
    {
        return Err("text selection/admission differs".into());
    }
    // Fixed bounded telemetry storage is reserved once and reused. Reset and
    // JSON/full-decode validation stay outside the declared request clock.
    let mut chunks = Vec::with_capacity(512);
    let mut decisions = Vec::with_capacity(MAX_NEW);
    let mut samples = Vec::with_capacity(13);
    for sample in 0..13 {
        session.reset();
        chunks.clear();
        decisions.clear();
        if session.evaluated_len() != 0
            || session.pending_len() != 0
            || !session.history().is_empty()
        {
            return Err("text reset retained valid state".into());
        }
        let mut answer = String::with_capacity(text::MAX_TEXT_BYTES);
        let mut first_text = None;
        let request = Instant::now();
        let clock = Instant::now();
        let rendered = text::render(&messages, options)?;
        let render_ns = nanos(&clock)?;
        let clock = Instant::now();
        let encoded = codec.encode_raw(&rendered, false)?;
        let encode_ns = nanos(&clock)?;
        if encoded != prompt {
            return Err("text tokenization changed across requests".into());
        }
        let clock = Instant::now();
        let mut stream = codec.response_stream(
            ResponseLimits::new(maximum, text::MAX_TEXT_BYTES)?,
            text::STREAM_ALLOWANCE_BYTES,
        )?;
        let stream_setup_ns = nanos(&clock)?;
        session.set_tokens(&encoded)?;
        let mut stopped = false;
        for step in 0..maximum {
            let first_chunk = chunks.len();
            while session.pending_len() > 0 {
                if chunks.len() == 512 {
                    return Err("text completed-chunk telemetry overflow".into());
                }
                let clock = Instant::now();
                let rows = session.evaluate_next_chunk()?;
                let wall_ns = nanos(&clock)?;
                if rows == 0 {
                    return Err("text evaluation failed to progress".into());
                }
                chunks.push(Chunk {
                    decision: step,
                    rows,
                    committed_after: session.evaluated_len(),
                    pending_after: session.pending_len(),
                    wall_ns,
                    completed_timing: session.timing().ok_or("text completed timing absent")?,
                });
            }
            let clock = Instant::now();
            let selected = session
                .greedy_token()
                .ok_or("text greedy selection absent")??;
            let selection_ns = nanos(&clock)?;
            let decision_ns = nanos(&request)?;
            if session.logits().is_none() || selected as usize >= text::VOCAB {
                return Err("text logits/vocabulary differ".into());
            }
            session.append(&[selected])?;
            let clock = Instant::now();
            let delta = stream.push(&[selected])?;
            if !delta.is_empty() && first_text.is_none() {
                first_text = Some(nanos(&request)?);
            }
            let delta_bytes = delta.len();
            answer.push_str(delta);
            let decode_ns = nanos(&clock)?;
            decisions.push(Decision {
                step,
                selected,
                decision_ns,
                selection_ns,
                decode_ns,
                delta_bytes,
                text_bytes: answer.len(),
                first_chunk,
                chunks: chunks.len() - first_chunk,
                committed_after: session.evaluated_len(),
                pending_after: session.pending_len(),
            });
            if codec.stop_ids().contains(&selected) {
                stopped = true;
                break;
            }
        }
        let clock = Instant::now();
        let tail = stream.finish()?;
        if !tail.is_empty() && first_text.is_none() {
            first_text = Some(nanos(&request)?);
        }
        let finish_bytes = tail.len();
        answer.push_str(tail);
        let stream_finish_ns = nanos(&clock)?;
        let request_ns = nanos(&request)?;
        let sampled: Vec<_> = decisions.iter().map(|row| row.selected).collect();
        if stream.consumed_tokens() != sampled.len()
            || stream.emitted_bytes() != answer.len()
            || session.pending_len() != 1
            || session.evaluated_len() != prompt.len() + sampled.len() - 1
            || session.history().len() != prompt.len() + sampled.len()
        {
            return Err("text decoded/sampled/committed state differs".into());
        }
        let clock = Instant::now();
        if codec.decode(&sampled, true)? != answer {
            return Err("text incremental/full decode differ".into());
        }
        let validation_ns = nanos(&clock)?;
        samples.push(json!({"sample":sample,"warmup":sample<3,"request_ns":request_ns,"render_ns":render_ns,"encode_ns":encode_ns,"stream_setup_ns":stream_setup_ns,"stream_finish_ns":stream_finish_ns,"finish_bytes":finish_bytes,"first_decoded_text_ns":first_text,"full_decode_validation_ns":validation_ns,"input_tokens":encoded.len(),"sampled_ids":sampled,"text":answer,"chunks":chunks,"decisions":decisions,"committed_after":session.evaluated_len(),"pending_after":session.pending_len(),"history_after":session.history().len(),"stop_reason":if stopped{"eos"}else{"limit"}}));
    }
    for &(name, size, pin) in FILES {
        hash_file(&model_path.join(name), size, pin)?;
    }
    let report = json!({"format":"toklet-minicpm5-text-request-v1","repository":REPOSITORY,"revision":REVISION,"device":model.device_name(),"profile":"MlxAffine/Compensated/Reference/Half","normalization_implementation":if mode=="serial"{"serial"}else{"serial-reduction-parallel-store"},"tokenizer_sha256":"ad92aaa8d3032c98a9158b8c5e8682bed10027ed6463e4fb1320fe5384210873","options":{"enable_thinking":options.enable_thinking,"add_generation_prompt":options.add_generation_prompt},"messages_sha256":format!("{:x}",Sha256::digest(&messages)),"options_sha256":format!("{:x}",Sha256::digest(&options_raw)),"prompt_ids":prompt,"stop_ids":codec.stop_ids(),"maximum_new_tokens":maximum,"capacity":512,"prefill_chunk":32,"position_offset":0,"warmups":3,"measured_samples":10,"cache_state":"one loaded model/codec/session;reset-empty-history-per-request;warm-filesystem","load_timing":{"load_seconds":model.load_timing.load_seconds,"conversion_seconds":model.load_timing.conversion_seconds,"compilation_seconds":model.load_timing.compilation_seconds},"codec_load_ns":codec_load_ns,"preflight_text_ns":preflight_text_ns,"session_setup_ns":session_setup_ns,"session_memory":session.memory(),"codec_allowance_bytes":text::ALLOWANCE_BYTES,"stream_allowance_bytes":text::STREAM_ALLOWANCE_BYTES,"application_allowance_bytes":APP,"maximum_admitted_bytes":session.memory().admitted_bytes+external,"samples":samples,"request_clock_scope":"render+encode+stream construction+set_tokens through final stream finish; excludes reset, telemetry capacity reservation, JSON and full-decode validation","first_text_scope":"first nonempty incremental decoder chunk; no console publication/paint claim","diagnostics_enabled":false,"operator_counts_available":false,"allocation_count":null,"console_publication_ns":null,"user_visible_ttft_ns":null,"model_quality_qualified":false,"managed_adoption_qualified":false,"ordinary_telemetry_overhead_qualified":false});
    output(&out.join("text.json"), &report)?;
    println!(
        "{}",
        json!({"text_requests_completed":13,"implementation":mode,"model_adoption_qualified":false})
    );
    Ok(())
}

fn parity(
    model: &metal::Model,
    codec: &text::Codec,
    prompt: &[u32],
    maximum: usize,
    limit: usize,
    external: usize,
    out: &Path,
) -> Result<()> {
    let mut serial = open(model, limit)?;
    let shared = model.model_resident_bytes()?;
    let combined = serial
        .memory()
        .admitted_bytes
        .checked_mul(2)
        .and_then(|n| n.checked_add(64 * 1024 * 1024))
        .and_then(|n| n.checked_sub(shared))
        .and_then(|n| n.checked_add(external))
        .ok_or("text parity admission overflow")?;
    if combined > TOTAL {
        return Err("text parity combined owners exceed6GiB".into());
    }
    let mut parallel = open(model, limit)?;
    parallel.enable_rms_parallel_store()?;
    if serial.memory().admitted_bytes + parallel.memory().admitted_bytes - shared + external
        != combined
    {
        return Err("text parity owner ledger differs".into());
    }
    serial.set_tokens(prompt)?;
    parallel.set_tokens(prompt)?;
    let mut records = Vec::new();
    let mut sampled = Vec::new();
    for step in 0..maximum {
        while serial.pending_len() > 0 {
            let rows = serial.evaluate_next_chunk()?;
            if rows == 0
                || parallel.evaluate_next_chunk()? != rows
                || serial.evaluated_len() != parallel.evaluated_len()
                || serial.pending_len() != parallel.pending_len()
            {
                return Err("text parity work/state differs".into());
            }
        }
        let a = serial.logits().ok_or("serial text logits absent")?;
        let b = parallel.logits().ok_or("parallel text logits absent")?;
        if a.len() != text::VOCAB
            || a.len() != b.len()
            || !a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        {
            return Err("text parity complete logits differ".into());
        }
        let values = a.len();
        let mut hash = Sha256::new();
        for value in a {
            hash.update(value.to_le_bytes());
        }
        let selected = serial
            .greedy_token()
            .ok_or("serial text selection absent")??;
        if parallel
            .greedy_token()
            .ok_or("parallel text selection absent")??
            != selected
        {
            return Err("text parity selection differs".into());
        }
        sampled.push(selected);
        serial.append(&[selected])?;
        parallel.append(&[selected])?;
        if serial.history() != parallel.history() {
            return Err("text parity history differs".into());
        }
        records.push(json!({"step":step,"logits_values":values,"logits_sha256":format!("{:x}",hash.finalize()),"all_logits_exact":true,"selected":selected,"committed_after":serial.evaluated_len(),"pending_after":serial.pending_len()}));
        if codec.stop_ids().contains(&selected) {
            break;
        }
    }
    let text = codec.decode(&sampled, true)?;
    output(
        &out.join("parity.json"),
        &json!({"format":"toklet-minicpm5-text-parity-v1","profile":"MlxAffine/Compensated/Reference/Half","capacity":512,"prefill_chunk":32,"position_offset":0,"stop_ids":codec.stop_ids(),"repository":REPOSITORY,"revision":REVISION,"prompt_ids":prompt,"sampled_ids":sampled,"text":text,"maximum_new_tokens":maximum,"combined_admitted_bytes":combined,"shared_model_bytes":shared,"serial_memory":serial.memory(),"parallel_memory":parallel.memory(),"codec_allowance_bytes":text::ALLOWANCE_BYTES,"stream_allowance_bytes":text::STREAM_ALLOWANCE_BYTES,"application_allowance_bytes":APP,"steps":records,"pending_after":serial.pending_len(),"history_after":serial.history().len(),"committed_after":serial.evaluated_len(),"model_quality_adoption_qualified":false}),
    )?;
    println!(
        "{}",
        json!({"text_parity_completed":true,"sampled_tokens":sampled.len(),"model_adoption_qualified":false})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_output_argument_preserves_default_and_rejects_invalid_values() {
        assert_eq!(max_new(None).unwrap(), 16);
        for raw in ["1", "16", "256"] {
            assert_eq!(
                max_new(Some(&raw.into())).unwrap(),
                raw.parse::<usize>().unwrap()
            );
        }
        for raw in ["0", "257", "-1", "1.5", "false", ""] {
            assert!(max_new(Some(&raw.into())).is_err());
        }
    }
}
