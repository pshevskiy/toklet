//! Cold ownership inspection of the ordinary native factory. No GPU readback.
use super::*;
use std::mem::size_of;

impl Model {
    pub(crate) fn native_session_modes(&self) -> (ProjectionMode, NormalizationMode) {
        if self.is_mlx() {
            (ProjectionMode::MlxAffine, NormalizationMode::Compensated)
        } else if self.q4_format().is_some() {
            (
                ProjectionMode::Q4CompensatedTiled,
                NormalizationMode::Compensated,
            )
        } else {
            (ProjectionMode::HalfCompensated, NormalizationMode::Original)
        }
    }

    pub(crate) fn allocated_layer_metadata_bytes(&self) -> Result<usize> {
        self.layers
            .capacity()
            .checked_mul(size_of::<Layer>())
            .ok_or(Error::Options("model layer metadata overflow"))
    }

    /// The returned memory is the constructor's numerical plan, not an allocation.
    /// Driver/model bookkeeping remains covered by the unchanged allowance.
    pub(crate) fn native_session_plan(
        &self,
        options: session::Options,
    ) -> Result<(session::Memory, usize, usize)> {
        if self.qwen3_context_diagnostic {
            return Err(Error::Unsupported(
                "transition planning excludes context-diagnostic models",
            ));
        }
        let (projection, normalization) = self.native_session_modes();
        let attention = self.native_attention_mode();
        self.validate_session_modes(projection, normalization, attention)?;
        let plan = planning::SessionPlan::new(
            &self.config,
            self.weight_bytes,
            options,
            ReadbackMode::FullLogits,
            attention,
        )?
        .with_positional_bytes(self.positional_resident_bytes())?;
        let maximum = self.gpu.device.maxBufferLength();
        plan.validate_buffers(maximum)?;
        Ok((plan.memory, plan.largest_buffer_bytes()?, maximum))
    }
}

impl Session<'_> {
    pub(crate) fn native_options(&self) -> session::Options {
        self.options
    }

    /// Exact current numerical buffer lengths and host vector capacities.
    /// This does not overwrite the historical single-session admission report.
    pub(crate) fn native_owned_memory(&self) -> Result<session::Memory> {
        let (projection, normalization) = self.model.native_session_modes();
        // diagnostic_rows is a last-chunk marker even in ordinary execution.
        // Retained diagnostic storage, not that marker, identifies extra owners.
        if self.model.qwen3_context_diagnostic
            || self.projection_mode != projection
            || self.normalization_mode != normalization
            || self.readback_mode != ReadbackMode::FullLogits
            || self.attention_mode != self.model.native_attention_mode()
            || self.parallel_reference.is_some()
            || self.trace.is_some()
            || self.diagnostics.capacity() != 0
        {
            return Err(Error::Unsupported(
                "transition planning requires an ordinary native session without diagnostics",
            ));
        }
        fn bytes<T: memory::Scalar>(buffer: &Buffer<T>) -> Result<usize> {
            let bytes = buffer
                .len
                .checked_mul(size_of::<T>())
                .ok_or(Error::Options("owned buffer size overflow"))?;
            if buffer.raw.length() != bytes {
                return Err(Error::Options("owned buffer length changed"));
            }
            Ok(bytes)
        }
        let mut kv_bytes = 0;
        for cache in &self.cache {
            kv_bytes = add(kv_bytes, add(bytes(&cache.k)?, bytes(&cache.v)?)?)?;
        }
        let s = &self.scratch;
        let mut scratch_bytes = 0;
        for buffer in [
            &s.x, &s.norm, &s.q, &s.qr, &s.k, &s.kr, &s.v, &s.context, &s.delta, &s.gate, &s.silu,
            &s.up, &s.gated, &s.zero,
        ] {
            scratch_bytes = add(scratch_bytes, bytes(buffer)?)?;
        }
        for buffer in [&s.floats, &s.scores, &s.probs, &s.tokens, &s.status] {
            scratch_bytes = add(scratch_bytes, bytes(buffer)?)?;
        }
        if let Some(selection) = &s.selection {
            scratch_bytes = add(scratch_bytes, bytes(selection)?)?;
        }
        let history_bytes = self
            .history
            .capacity()
            .checked_mul(size_of::<u32>())
            .ok_or(Error::Options("owned history size overflow"))?;
        let logits_bytes = add(self.logits.capacity(), self.staged.capacity())?
            .checked_mul(size_of::<f32>())
            .ok_or(Error::Options("owned logits size overflow"))?;
        let metadata_bytes = add(
            self.cache
                .capacity()
                .checked_mul(size_of::<Cache>())
                .ok_or(Error::Options("owned cache table overflow"))?,
            self.views
                .capacity()
                .checked_mul(size_of::<Views>())
                .ok_or(Error::Options("owned view table overflow"))?,
        )?;
        let mut memory = session::Memory {
            weights_bytes: self.model.weight_bytes,
            positional_bytes: self.model.positional_resident_bytes(),
            kv_bytes,
            scratch_bytes,
            history_bytes,
            logits_bytes,
            metadata_bytes,
            metadata_allowance_bytes: self.memory.metadata_allowance_bytes,
            admitted_bytes: 0,
        };
        memory.admitted_bytes = [
            memory.weights_bytes,
            memory.positional_bytes,
            memory.kv_bytes,
            memory.scratch_bytes,
            memory.history_bytes,
            memory.logits_bytes,
            memory.metadata_bytes,
            memory.metadata_allowance_bytes,
        ]
        .into_iter()
        .try_fold(0, add)?;
        Ok(memory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_candidate_largest_buffer_boundary_is_checked_without_allocation() {
        let config = model::Config {
            vocab_size: 64,
            hidden_size: 64,
            intermediate_size: 64,
            num_layers: 1,
            query_heads: 1,
            kv_heads: 1,
            head_dim: 64,
            max_positions: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let plan = planning::SessionPlan::new(
            &config,
            4096,
            session::Options {
                capacity: 16,
                prefill_chunk_size: 3,
                ..Default::default()
            },
            ReadbackMode::FullLogits,
            AttentionMode::Reference,
        )
        .unwrap();
        let largest = plan.largest_buffer_bytes().unwrap();
        plan.validate_buffers(largest).unwrap();
        assert!(plan.validate_buffers(largest - 1).is_err());
    }
}
