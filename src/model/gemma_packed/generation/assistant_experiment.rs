//! Bounded explicit MTP experiment for the exact Mac BF16 owner.
use super::*;
use crate::{
    generation::{self, Error as GenError, Input, Request, StopReason, VerifiedNext},
    metal::gemma_affine::Bf16GenerationSession,
    model::gemma_assistant::{Assistant, Scratch as DraftScratch, TARGET_HIDDEN, TargetKv},
};
use std::sync::atomic::{AtomicBool, Ordering};

const PROPOSALS: usize = 4;

pub(super) struct Report {
    pub(super) reason: StopReason,
    pub(super) proposed: usize,
    pub(super) accepted: usize,
    pub(super) verification_rounds: usize,
}

/// Run a fresh explicit request with bounded output and strict target greedy.
/// The owned anchor is captured immediately after successful target steps.
pub(super) fn generate_assisted(
    target: &mut Bf16GenerationSession<'_, '_, '_>,
    assistant: &Assistant,
    scratch: &mut DraftScratch,
    request: Request<'_>,
    output: &mut Vec<u32>,
    cancelled: &AtomicBool,
) -> Result<Report, GenError> {
    let Input::Reset(prompt) = request.input else {
        return Err(GenError::InvalidRequest(
            "assistant requires an explicit fresh prompt",
        ));
    };
    let (maximum, stops) = (request.max_new_tokens, request.stop_tokens);
    generation::request_admission(target, &request)?;
    if maximum > output.capacity() - output.len() {
        return Err(GenError::InvalidRequest(
            "assistant requires admitted output capacity",
        ));
    }
    let output_start = output.len();
    let cancel = || cancelled.load(Ordering::Relaxed);
    let mut report = Report {
        reason: StopReason::OutputLimit,
        proposed: 0,
        accepted: 0,
        verification_rounds: 0,
    };
    let mut anchor = [0.; TARGET_HIDDEN];
    if cancel() {
        report.reason = StopReason::Cancelled;
        return Ok(report);
    }
    target.reset();
    target.append(prompt)?;
    while target.pending_len() != 0 {
        if cancel() {
            report.reason = StopReason::Cancelled;
            return Ok(report);
        }
        target.evaluate_next_chunk()?;
    }
    capture_anchor(target, &mut anchor);
    while output.len() - output_start < maximum {
        if cancel() {
            report.reason = StopReason::Cancelled;
            break;
        }
        if target.pending_len() != 0 {
            target.evaluate_next_chunk()?;
            capture_anchor(target, &mut anchor);
        }
        let pending = maximum - (output.len() - output_start);
        let bound = PROPOSALS.min(target.chunk).min(pending - 1);
        let mut candidates = [0; PROPOSALS];
        let mut count = 0;
        if bound > 0 && !cancel() {
            let winner = crate::sampling::greedy(target.logits().expect("evaluated target"))
                .map_err(GenError::Sampling)?;
            if !stops.contains(&winner) {
                count = draft(
                    target,
                    assistant,
                    scratch,
                    &anchor,
                    winner,
                    &mut candidates[..bound],
                    cancel,
                )?;
                report.proposed += count;
                if count != 0 && candidates[0] != winner {
                    count = 0;
                }
            }
        }
        let Some(result) =
            target.verify_greedy_round(&candidates[..count], stops, &mut anchor, cancel)?
        else {
            report.reason = StopReason::Cancelled;
            break;
        };
        report.verification_rounds += usize::from(count != 0);
        report.accepted += result.accepted;
        output.extend_from_slice(&candidates[..result.accepted]);
        match result.next {
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
    Ok(report)
}

fn capture_anchor(target: &Bf16GenerationSession<'_, '_, '_>, anchor: &mut [f32; TARGET_HIDDEN]) {
    let end = target.inner.published_rows * TARGET_HIDDEN;
    anchor.copy_from_slice(&target.inner.scratch.norm[end - TARGET_HIDDEN..end]);
}

fn draft(
    target: &mut Bf16GenerationSession<'_, '_, '_>,
    assistant: &Assistant,
    scratch: &mut DraftScratch,
    anchor: &[f32; TARGET_HIDDEN],
    target_winner: u32,
    output: &mut [u32],
    cancelled: impl Fn() -> bool,
) -> Result<usize, GenError> {
    let inner = &mut target.inner;
    let mut last = *inner.tokens().last().expect("evaluated target");
    let kv = TargetKv {
        position: inner.committed - 1,
        local: inner.caches[13].committed_segments(),
        full: inner.caches[14].committed_segments(),
    };
    let mut input = [0.; 2 * TARGET_HIDDEN];
    input[TARGET_HIDDEN..].copy_from_slice(anchor);
    let mut raw = [0.; TARGET_HIDDEN];
    for (i, slot) in output.iter_mut().enumerate() {
        if cancelled() {
            return Ok(i);
        }
        inner
            .operations
            .gather(&inner.model.weights.embedding, &[last], &mut raw)
            .map_err(|err| GenError::Session(err.into()))?;
        cpu::prepare_per_layer_projection_input(
            &raw,
            &mut input[..TARGET_HIDDEN],
            1,
            TARGET_HIDDEN,
        )
        .map_err(GenError::Sampling)?;
        let proposal = assistant
            .propose(&input, &kv, scratch)
            .map_err(|err| GenError::Session(session::Error::Backend(err.to_string())))?;
        *slot = proposal.winner;
        if i == 0 && *slot != target_winner {
            return Ok(1);
        }
        last = *slot;
        input[TARGET_HIDDEN..].copy_from_slice(proposal.hidden);
    }
    Ok(output.len())
}
