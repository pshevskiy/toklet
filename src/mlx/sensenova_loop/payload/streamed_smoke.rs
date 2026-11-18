//! Opt-in Apple Silicon smoke over the pinned public artifact bytes.

use super::*;
use crate::{
    mlx::sensenova_loop::{REVISION, text_weights::SmallWeights},
    model::{qwen35::Options, sensenova_loop::text::ScalarSession},
    sampling::greedy,
    weights::INITIAL_MEMORY_LIMIT_BYTES,
};
use serde_json::{Value, json};
use std::{
    io::{BufWriter, Write},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::atomic::AtomicBool,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const REPOSITORY: &str = "nativ-community/SenseNova-Loop-9B-4bit";
const PROMOTION_BYTES: usize = 16_829_952;
const TEST_ALLOWANCE: usize = 64 << 20;
const SAVED_ROWS_BYTES: u64 = 2_979_840;
// "Hello" and newline from the checked text-codec fixture. The smoke starts
// at position zero and supplies IDs directly.
const PROMPT: [u32; 2] = [9419, 198];

struct Transfer(Child);

impl Transfer {
    fn start(pin: &ShardPin) -> std::io::Result<Self> {
        let request = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(std::io::Error::other)?
            .as_nanos();
        let url = format!(
            "https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{}?download=true&toklet_smoke={request}",
            pin.name
        );
        Command::new("/usr/bin/curl")
            .args([
                "-q",
                "--fail",
                "--silent",
                "--show-error",
                "--location",
                "--max-redirs",
                "5",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--connect-timeout",
                "30",
                "--max-time",
                "3600",
                "--max-filesize",
                &pin.file_bytes.to_string(),
                "--output",
                "-",
                &url,
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map(Self)
    }
}

impl Drop for Transfer {
    fn drop(&mut self) {
        if !matches!(self.0.try_wait(), Ok(Some(_))) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

fn stream_payload(
    plan: Plan,
    headers: [&[u8]; 2],
) -> Result<TextPayload, Box<dyn std::error::Error>> {
    let mut transfer = buffer(TRANSFER_BYTES)?;
    let mut tensors = BTreeMap::new();
    for (slot, pin) in SHARDS.iter().enumerate() {
        let start = Instant::now();
        let mut child = Transfer::start(pin)?;
        let mut stdout = child
            .0
            .stdout
            .take()
            .ok_or_else(|| invalid("missing curl stdout"))?;
        let (header, sha) = read_shard_header(&mut stdout, pin)?;
        if header != headers[slot] {
            return Err(invalid("streamed SenseNova header differs from preflight").into());
        }
        // Reads the declared extent, probes EOF on the underlying pipe and
        // authenticates the complete original shard, including skipped vision.
        consume_shard(&mut stdout, slot, &plan, sha, &mut transfer, &mut tensors)?;
        drop(stdout);
        let status = child.0.wait()?;
        if !status.success() {
            return Err(
                std::io::Error::other(format!("curl failed for {}: {status}", pin.name)).into(),
            );
        }
        println!(
            "{}",
            json!({
                "event": "authenticated_shard", "file": pin.name,
                "bytes": pin.file_bytes, "sha256": pin.file_sha256,
                "seconds": start.elapsed().as_secs_f64()
            })
        );
    }
    Ok(finish(plan, tensors)?)
}

fn observe(logits: &[f32], vocab: usize) -> Result<(u32, Value), crate::cpu::Error> {
    assert_eq!(logits.len(), vocab);
    let token = greedy(logits)?;
    let runner_up = logits
        .iter()
        .enumerate()
        .filter(|&(id, _)| id != token as usize)
        .map(|(_, &value)| value)
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(logits[token as usize] > logits.iter().copied().fold(f32::INFINITY, f32::min));
    let mut hash = Sha256::new();
    for value in logits {
        hash.update(value.to_le_bytes());
    }
    Ok((
        token,
        json!({
            "vocab": vocab, "finite": true,
            "greedy_id": token, "greedy_logit": logits[token as usize],
            "runner_up_logit": runner_up,
            "logits_f32_le_sha256": format!("{:x}", hash.finalize())
        }),
    ))
}

fn save_logits(output: &mut BufWriter<File>, logits: &[f32]) -> std::io::Result<()> {
    for value in logits {
        output.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

#[test]
#[ignore = "requires pinned release metadata, guarded network transfer and ~5 GiB of memory"]
fn real_weight_scalar_session() -> Result<(), Box<dyn std::error::Error>> {
    let metadata = PathBuf::from(
        std::env::var_os("TOKLET_SENSENOVA_SMOKE_METADATA").ok_or_else(|| {
            invalid("set TOKLET_SENSENOVA_SMOKE_METADATA to the retained metadata directory")
        })?,
    );
    let config = read_metadata(&metadata.join("mlx-config.json"), 8192)?;
    let index = read_metadata(&metadata.join("mlx-index.json"), 256 << 10)?;
    let first = read_metadata(
        &metadata.join("mlx-shard-1-header.json"),
        SHARDS[0].header_bytes,
    )?;
    let second = read_metadata(
        &metadata.join("mlx-shard-2-header.json"),
        SHARDS[1].header_bytes,
    )?;
    let headers = [first.as_slice(), second.as_slice()];
    let load_plan = inspect(&config, &index, headers, SHARDS.map(|pin| pin.file_bytes))?;
    let plan = crate::model::sensenova_loop::Plan::from_mlx_config(&config)?;
    let opts = Options {
        capacity: 4,
        chunk: 2,
        position_offset: 0,
        memory_limit_bytes: INITIAL_MEMORY_LIMIT_BYTES - TEST_ALLOWANCE,
    };
    let admitted = ScalarSession::planned(&plan, opts, ADMITTED_LOAD_BYTES + PROMOTION_BYTES)?;
    println!(
        "{}",
        json!({
            "event": "admission", "repository": REPOSITORY, "revision": REVISION,
            "precision": crate::model::sensenova_loop::text::PRECISION_PROFILE,
            "capacity": opts.capacity, "chunk": opts.chunk, "session": admitted,
            "smoke_allowance_bytes": TEST_ALLOWANCE,
            "total_admitted_bytes": admitted.admitted_bytes + TEST_ALLOWANCE
        })
    );
    let mut saved_rows = std::env::var_os("TOKLET_SENSENOVA_SMOKE_ROWS")
        .map(File::create_new)
        .transpose()?
        .map(|file| BufWriter::with_capacity(TRANSFER_BYTES, file));
    let start = Instant::now();
    let payload = stream_payload(load_plan, headers)?;
    let load_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let small = SmallWeights::new(&payload, PROMOTION_BYTES)?;
    let weights = small.view()?;
    let mut session = ScalarSession::new(
        &plan,
        &weights,
        opts,
        ADMITTED_LOAD_BYTES + small.promoted_bytes(),
    )?;
    let binding_seconds = start.elapsed().as_secs_f64();

    let cancel = AtomicBool::new(false);
    let vocab = plan.text_shape().vocab_size;
    let start = Instant::now();
    let rows = session.forward(&PROMPT, &cancel)?;
    let prefill_seconds = start.elapsed().as_secs_f64();
    assert_eq!(rows.len(), PROMPT.len() * vocab);
    let (_, first_row) = observe(&rows[..vocab], vocab)?;
    let start = Instant::now();
    let (picked, prefill) = observe(&rows[vocab..], vocab)?;
    let greedy_and_observe_seconds = start.elapsed().as_secs_f64();
    if let Some(output) = &mut saved_rows {
        // The two observed prefill rows are consecutive in the borrowed buffer.
        save_logits(output, rows)?;
    }
    assert_eq!(session.history(), PROMPT);
    let continued = [PROMPT[0], PROMPT[1], picked];
    assert_eq!(session.reuse_prefix(&continued)?, 2);
    let start = Instant::now();
    let row = session.forward(&[picked], &cancel)?;
    let decode_seconds = start.elapsed().as_secs_f64();
    let (_, decode) = observe(row, vocab)?;
    if let Some(output) = &mut saved_rows {
        save_logits(output, row)?;
    }
    assert_eq!(session.history(), continued);
    assert!(matches!(
        session.forward(&[picked], &AtomicBool::new(true)),
        Err(crate::model::qwen35::Error::Cancelled)
    ));
    assert_eq!(session.history(), continued);
    assert_eq!(observe(session.logits().unwrap(), vocab)?.1, decode);

    session.reset();
    assert!(session.history().is_empty());
    assert!(session.logits().is_none());
    let (_, replay_first) = observe(session.forward(&PROMPT[..1], &cancel)?, vocab)?;
    let (_, replay_prefill) = observe(session.forward(&PROMPT[1..], &cancel)?, vocab)?;
    assert_eq!(replay_first, first_row);
    assert_eq!(replay_prefill, prefill);
    assert_eq!(session.history(), PROMPT);
    let (_, replay_decode) = observe(session.forward(&[picked], &cancel)?, vocab)?;
    assert_eq!(replay_decode, decode);
    assert_eq!(session.history(), continued);
    let saved_logit_rows_bytes = if let Some(mut output) = saved_rows {
        output.flush()?;
        let file = output.get_ref();
        assert_eq!(file.metadata()?.len(), SAVED_ROWS_BYTES);
        file.sync_all()?;
        Some(SAVED_ROWS_BYTES)
    } else {
        None
    };
    println!(
        "{}",
        json!({
            "event": "real_weight_scalar_smoke", "prompt_ids": PROMPT,
            "committed_ids": session.history(), "first_row": first_row,
            "prefill": prefill, "decode": decode,
            "saved_logit_rows_bytes": saved_logit_rows_bytes,
            "cancel_preserved_history_and_logits": true, "reset_single_token_replay_exact": true,
            "memory": session.memory(), "promoted_bytes": small.promoted_bytes(),
            "stream_and_validate_seconds": load_seconds, "bind_and_admit_seconds": binding_seconds,
            "two_token_prefill_seconds": prefill_seconds, "cached_decode_seconds": decode_seconds,
            "greedy_and_prefill_row_observe_seconds": greedy_and_observe_seconds
        })
    );
    Ok(())
}
