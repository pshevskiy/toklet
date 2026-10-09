//! Bounded greedy generation outside numerical kernels. All requested output
//! capacity is reserved before changing session state. Stop IDs enter history
//! as pending sampled tokens but are not emitted. The final sampled token also
//! remains pending at an output limit; continuation evaluates it exactly once.
//!
//! Cancellation is checked before session mutation, between prefill chunks and
//! between sampling/decode steps. A synchronous chunk completes before stopping.
//! Callback failure preserves a valid KV prefix and the pending sampled token;
//! callers may continue or discard pending input explicitly.

use crate::{
    cpu, sampling,
    session::{self, Session},
};
use serde::Serialize;
use std::{
    fmt, io,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

pub enum Input<'a> {
    Prompt(&'a [u32]),
    Append(&'a [u32]),
    Continue,
    Reset(&'a [u32]),
}

pub struct Request<'a> {
    pub input: Input<'a>,
    pub max_new_tokens: usize,
    /// Already resolved model EOS IDs and caller stop IDs, all in vocabulary.
    pub stop_tokens: &'a [u32],
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct TokenEvent {
    pub token_id: u32,
    pub index: usize,
    pub elapsed_seconds: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StopReason {
    OutputLimit,
    StopToken { token_id: u32 },
    Cancelled,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub reason: StopReason,
    pub reused_tokens: usize,
    pub prefill_tokens: usize,
    pub sampled_tokens: usize,
    pub emitted_tokens: usize,
    pub history_tokens: usize,
    pub evaluated_tokens: usize,
    pub pending_tokens: usize,
    pub first_token_seconds: Option<f64>,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub sampling_seconds: f64,
    /// Includes callback/output time; component timers exclude callback time.
    pub total_seconds: f64,
}

#[derive(Debug)]
pub enum Error {
    InvalidRequest(&'static str),
    Session(session::Error),
    Sampling(cpu::Error),
    Output(io::Error),
}
impl From<session::Error> for Error {
    fn from(e: session::Error) -> Self {
        Self::Session(e)
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(reason) => write!(f, "invalid generation request: {reason}"),
            Self::Session(e) => e.fmt(f),
            Self::Sampling(e) => e.fmt(f),
            Self::Output(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for Error {}

/// Validate admission before changing session or application output state.
pub fn validate_request(session: &Session<'_>, request: &Request<'_>) -> Result<(), Error> {
    let (tokens, base) = match &request.input {
        Input::Prompt(t) | Input::Reset(t) => (*t, 0),
        Input::Append(t) => (*t, session.history().len()),
        Input::Continue => (&[][..], session.history().len()),
    };
    let input_len = base
        .checked_add(tokens.len())
        .ok_or(Error::InvalidRequest("token count overflow"))?;
    let end = input_len
        .checked_add(request.max_new_tokens)
        .ok_or(Error::InvalidRequest("output count overflow"))?;
    if input_len == 0 {
        return Err(Error::InvalidRequest("a nonempty prompt is required"));
    }
    if end > session.capacity() {
        return Err(Error::InvalidRequest(
            "prompt plus requested output exceeds session capacity",
        ));
    }
    for &id in tokens.iter().chain(request.stop_tokens) {
        if id as usize >= session.vocab_size() {
            return Err(Error::InvalidRequest(
                "token or stop ID is outside the model vocabulary",
            ));
        }
    }
    Ok(())
}

/// Emit visible token IDs through a coarse statically dispatched callback.
/// Prompt IDs, stop IDs and prompt+output capacity are validated before mutation.
/// No event collection is allocated here; sinks own token/text/UI storage.
pub fn generate(
    session: &mut Session<'_>,
    request: Request<'_>,
    cancelled: &AtomicBool,
    mut emit: impl FnMut(TokenEvent) -> io::Result<()>,
) -> Result<Report, Error> {
    let started = Instant::now();
    validate_request(session, &request)?;
    let mut report = Report {
        reason: StopReason::OutputLimit,
        reused_tokens: 0,
        prefill_tokens: 0,
        sampled_tokens: 0,
        emitted_tokens: 0,
        history_tokens: 0,
        evaluated_tokens: 0,
        pending_tokens: 0,
        first_token_seconds: None,
        prefill_seconds: 0.0,
        decode_seconds: 0.0,
        sampling_seconds: 0.0,
        total_seconds: 0.0,
    };
    if cancelled.load(Ordering::Relaxed) {
        report.reason = StopReason::Cancelled;
        return Ok(finish(report, session, started));
    }
    match request.input {
        Input::Prompt(tokens) => session.set_tokens(tokens)?,
        Input::Append(tokens) => session.append(tokens)?,
        Input::Continue => {}
        Input::Reset(tokens) => {
            session.reset();
            session.append(tokens)?;
        }
    }
    report.reused_tokens = session.evaluated_len();
    while session.pending_len() > 0 {
        if cancelled.load(Ordering::Relaxed) {
            report.reason = StopReason::Cancelled;
            return Ok(finish(report, session, started));
        }
        let clock = Instant::now();
        let evaluated = session.evaluate_next_chunk()?;
        report.prefill_seconds += clock.elapsed().as_secs_f64();
        report.prefill_tokens += evaluated;
    }
    for index in 0..request.max_new_tokens {
        if cancelled.load(Ordering::Relaxed) {
            report.reason = StopReason::Cancelled;
            break;
        }
        if index > 0 {
            let clock = Instant::now();
            session.evaluate_next_chunk()?;
            report.decode_seconds += clock.elapsed().as_secs_f64();
            if cancelled.load(Ordering::Relaxed) {
                report.reason = StopReason::Cancelled;
                break;
            }
        }
        let clock = Instant::now();
        let token_id = sampling::greedy(
            session
                .logits()
                .ok_or(Error::InvalidRequest("no evaluated distribution"))?,
        )
        .map_err(Error::Sampling)?;
        report.sampling_seconds += clock.elapsed().as_secs_f64();
        if cancelled.load(Ordering::Relaxed) {
            report.reason = StopReason::Cancelled;
            break;
        }
        session.append(&[token_id])?;
        report.sampled_tokens += 1;
        if request.stop_tokens.contains(&token_id) {
            report.reason = StopReason::StopToken { token_id };
            break;
        }
        let elapsed_seconds = started.elapsed().as_secs_f64();
        emit(TokenEvent {
            token_id,
            index,
            elapsed_seconds,
        })
        .map_err(Error::Output)?;
        report.emitted_tokens += 1;
        report.first_token_seconds.get_or_insert(elapsed_seconds);
    }
    // A cancellation raised by the final callback still wins over output limit.
    if report.reason == StopReason::OutputLimit && cancelled.load(Ordering::Relaxed) {
        report.reason = StopReason::Cancelled;
    }
    Ok(finish(report, session, started))
}

fn finish(mut report: Report, session: &Session<'_>, started: Instant) -> Report {
    report.history_tokens = session.history().len();
    report.evaluated_tokens = session.evaluated_len();
    report.pending_tokens = session.pending_len();
    report.total_seconds = started.elapsed().as_secs_f64();
    report
}
