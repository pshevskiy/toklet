//! Explicit, nonstreaming CPU generation over the bounded lookup transaction.
use super::{Error, Input, Request, StopReason, VerifiedNext, lookup, validate_request};
use crate::{sampling, session::Session};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuLookupReport {
    pub reason: StopReason,
    pub emitted_tokens: usize,
    pub proposed_tokens: usize,
    pub accepted_tokens: usize,
}

/// Continue from a fully evaluated CPU prefix using caller-owned output and
/// target-row storage. The output needs spare capacity for all requested
/// tokens. Target rows need (1 + min(chunk, 8, max_new_tokens - 1)) times the
/// vocabulary when lookup is possible. No target rows are needed for zero or
/// one requested token. The last sampled ID remains pending, including EOS.
/// Output already appended to the vector survives a later error. Cancellation
/// after evaluating a pending token may leave that token committed. Use only
/// with qualified batched target arithmetic. This API does not stream.
pub fn generate_cpu_lookup(
    session: &mut Session<'_>,
    max_new_tokens: usize,
    stop_tokens: &[u32],
    output: &mut Vec<u32>,
    target_rows: &mut [f32],
    cancelled: &AtomicBool,
) -> Result<CpuLookupReport, Error> {
    validate_request(
        session,
        &Request {
            input: Input::Continue,
            max_new_tokens,
            stop_tokens,
        },
    )?;
    if session.logits().is_none() {
        return Err(Error::InvalidRequest(
            "lookup requires a committed distribution",
        ));
    }
    if max_new_tokens > output.capacity() - output.len() {
        return Err(Error::InvalidRequest("lookup output capacity is too small"));
    }
    let mut proposal = [0; lookup::MAX_PROPOSAL];
    let limit = proposal
        .len()
        .min(session.options().prefill_chunk_size)
        .min(max_new_tokens.saturating_sub(1));
    let width = session.vocab_size();
    if limit != 0
        && (limit + 1)
            .checked_mul(width)
            .is_none_or(|need| need > target_rows.len())
    {
        return Err(Error::InvalidRequest("lookup target storage is too small"));
    }

    let initial_output = output.len();
    let cancel = || cancelled.load(Ordering::Relaxed);
    let mut report = CpuLookupReport {
        reason: StopReason::OutputLimit,
        emitted_tokens: 0,
        proposed_tokens: 0,
        accepted_tokens: 0,
    };
    while output.len() - initial_output < max_new_tokens {
        if cancel() {
            report.reason = StopReason::Cancelled;
            break;
        }
        if session.pending_len() != 0 {
            session.evaluate_next_chunk()?;
            if cancel() {
                report.reason = StopReason::Cancelled;
                break;
            }
        }
        let remaining = max_new_tokens - (output.len() - initial_output);
        let bound = limit.min(remaining - 1);
        let count =
            lookup::prompt_lookup(session.history(), lookup::MAX_MATCH, &mut proposal[..bound])?;
        let next = if count == 0 {
            let id = sampling::greedy(session.logits().ok_or(Error::InvalidRequest(
                "lookup requires a committed distribution",
            ))?)
            .map_err(Error::Sampling)?;
            if cancel() {
                report.reason = StopReason::Cancelled;
                break;
            }
            session.append(&[id])?;
            if stop_tokens.contains(&id) {
                VerifiedNext::Stop(id)
            } else {
                VerifiedNext::Token(id)
            }
        } else {
            report.proposed_tokens += count;
            let Some(verified) = session.verify_greedy_round(
                &proposal[..count],
                stop_tokens,
                &mut target_rows[..(count + 1) * width],
                cancel,
            )?
            else {
                report.reason = StopReason::Cancelled;
                break;
            };
            output.extend_from_slice(&proposal[..verified.accepted]);
            report.accepted_tokens += verified.accepted;
            verified.next
        };
        match next {
            VerifiedNext::Token(id) => output.push(id),
            VerifiedNext::Stop(token_id) => {
                report.reason = StopReason::StopToken { token_id };
                break;
            }
        }
    }
    if report.reason == StopReason::OutputLimit && cancel() {
        report.reason = StopReason::Cancelled;
    }
    report.emitted_tokens = output.len() - initial_output;
    Ok(report)
}
