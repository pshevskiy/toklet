//! Explicit bounded Qwen2 Metal proposal transaction.
use super::{AttentionMode, Error, NormalizationMode, ProjectionMode, ReadbackMode, Session};
use crate::{generation, session};
use objc2::rc::autoreleasepool;

impl Session<'_> {
    pub(crate) fn validate_lookup_profile(&self) -> std::result::Result<(), generation::Error> {
        if self.model.mlx_qwen2_source_content_identity().is_none()
            || self.options.capacity != 256
            || self.options.prefill_chunk_size != 8
            || self.options.position_offset != 0
            || self.readback_mode != ReadbackMode::FullLogits
            || self.projection_mode != ProjectionMode::MlxAffine
            || self.normalization_mode != NormalizationMode::Compensated
            || self.attention_mode != AttentionMode::Online
            || self.affine_tile4.is_some()
            || self.parallel_reference.is_some()
            || self.trace.is_some()
            || !self.diagnostics.is_empty()
        {
            return Err(generation::Error::InvalidRequest(
                "lookup requires the pinned ordinary Qwen2 Metal256 chunk8 profile",
            ));
        }
        Ok(())
    }

    /// Verify one tentative chunk for the pinned Qwen2 MLX8 Metal profile.
    /// Caller rows are (proposals + 1) by vocabulary, with the committed
    /// distribution first. Only the accepted prefix and one pending target
    /// publish. Observed cancellation preserves the published prefix.
    /// A command completion failure still requires ordinary session replay.
    pub fn verify_greedy_round(
        &mut self,
        proposals: &[u32],
        stop_tokens: &[u32],
        rows: &mut [f32],
        cancelled: impl Fn() -> bool,
    ) -> std::result::Result<Option<generation::GreedyVerification>, generation::Error> {
        use generation::{Error as GenerationError, VerifiedNext};
        let backend =
            |error: Error| GenerationError::Session(session::Error::Backend(error.to_string()));
        self.validate_lookup_profile()?;
        let start = self.evaluated;
        let width = self.model.config.vocab_size;
        let current = self.logits().ok_or(GenerationError::InvalidRequest(
            "verification requires a committed distribution",
        ))?;
        let n = proposals.len();
        if n > self.options.prefill_chunk_size {
            return Err(GenerationError::InvalidRequest(
                "proposal exceeds admitted chunk",
            ));
        }
        self.validate(proposals, start).map_err(backend)?;
        if start + n == self.capacity() {
            return Err(GenerationError::InvalidRequest(
                "no capacity for pending verification token",
            ));
        }
        if (n + 1).checked_mul(width) != Some(rows.len()) {
            return Err(GenerationError::InvalidRequest(
                "invalid target logits shape",
            ));
        }
        if stop_tokens.iter().any(|&id| id as usize >= width) {
            return Err(GenerationError::InvalidRequest(
                "stop token out of vocabulary",
            ));
        }
        if cancelled() {
            return Ok(None);
        }
        let (initial, speculative) = rows.split_at_mut(width);
        initial.copy_from_slice(current);
        let timing = if n == 0 {
            None
        } else {
            self.scratch
                .tokens
                .with_mut(|words| words[..n].copy_from_slice(proposals));
            self.scratch.status.with_mut(|words| words.fill(0));
            match autoreleasepool(|_| self.run_chunk(n, start, Some(speculative))) {
                Ok(timing) => Some(timing),
                Err(Error::Completion) => {
                    self.require_replay();
                    return Err(backend(Error::Completion));
                }
                Err(error) => return Err(backend(error)),
            }
        };
        let verification = generation::verify_greedy(proposals, rows, width, stop_tokens)?;
        if cancelled() {
            return Ok(None);
        }
        let accepted = verification.accepted;
        let next = match verification.next {
            VerifiedNext::Token(id) | VerifiedNext::Stop(id) => id,
        };
        let published = accepted * width;
        self.logits
            .copy_from_slice(&rows[published..published + width]);
        self.history.extend_from_slice(&proposals[..accepted]);
        self.evaluated += accepted;
        self.history.push(next);
        if let Some(timing) = timing {
            self.last_timing = Some(timing);
        }
        Ok(Some(verification))
    }
}
