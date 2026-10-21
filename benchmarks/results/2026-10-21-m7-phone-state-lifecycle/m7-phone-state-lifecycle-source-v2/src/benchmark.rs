//! Completed scalar CPU work on fixed teacher-forced output IDs.
//! Tokenization, sampling, detokenization and output I/O are excluded. This
//! diagnostic allocates its bounded event record before starting the timer;
//! session numerical buffers are reused. The last fixed token remains pending.

use crate::{
    generation::{self, Input, Request},
    session::Session,
};
use serde::Serialize;
use std::{hint::black_box, time::Instant};

#[derive(Debug, Serialize)]
pub struct Sample {
    pub prefill_seconds: f64,
    pub ttft_seconds: f64,
    pub decode_seconds: f64,
    pub total_seconds: f64,
    pub token_event_seconds: Vec<f64>,
    /// Fixed teacher-forced events, not greedily selected output.
    pub emitted_tokens: usize,
    pub reused_tokens: usize,
    pub prefill_tokens: usize,
    pub evaluated_tokens: usize,
    pub pending_tokens: usize,
}

/// Admission and fixed IDs are checked before state mutation. Callers prepare
/// cache state outside this function and record which prefix policy they use.
/// Synchronous evaluation completes every layer before each token timestamp.
pub fn measure_teacher_forced(
    session: &mut Session<'_>,
    input: Input<'_>,
    continuation: &[u32],
) -> Result<Sample, generation::Error> {
    if continuation.is_empty() {
        return Err(generation::Error::InvalidRequest(
            "benchmark needs an output step",
        ));
    }
    generation::validate_request(
        session,
        &Request {
            input,
            max_new_tokens: continuation.len(),
            stop_tokens: &[],
        },
    )?;
    if continuation
        .iter()
        .any(|&id| id as usize >= session.vocab_size())
    {
        return Err(generation::Error::InvalidRequest(
            "fixed output ID outside vocabulary",
        ));
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
