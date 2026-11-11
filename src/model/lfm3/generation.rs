use std::mem::size_of;

use crate::session::{self, InferenceSession};

use super::{Session, VOCAB, add, filled, reserve};

/// Generation history can end in a sampled token that has not reached KV.
/// The FP32 distribution widens the last successful BF16 head exactly once.
pub struct GenerationSession<'a> {
    inner: Session<'a>,
    history: Vec<u32>,
    logits: Vec<f32>,
    memory: session::Memory,
}

impl<'a> GenerationSession<'a> {
    /// Reserve this amount from the loader budget before reading the checkpoint.
    pub fn planned_extra_bytes(capacity: usize) -> Result<usize, session::Error> {
        let history =
            crate::cpu::product(capacity, size_of::<u32>()).map_err(super::Error::from)?;
        let metadata = size_of::<Self>() - size_of::<Session<'a>>();
        Ok(add(add(history, VOCAB * size_of::<f32>())?, metadata)?)
    }

    pub fn new(inner: Session<'a>, limit: usize) -> Result<Self, session::Error> {
        let numerical = inner.model.memory();
        let capacity = inner.model.capacity;
        let extra = Self::planned_extra_bytes(capacity)?;
        let required = add(numerical.total_bytes, extra)?;
        let limit = limit.min(crate::weights::INITIAL_MEMORY_LIMIT_BYTES);
        if required > limit {
            return Err(session::Error::MemoryBudget { required, limit });
        }
        let mut history = reserve(capacity)?;
        history.extend_from_slice(inner.history());
        let mut logits = filled(VOCAB, 0.0)?;
        if let Some(source) = inner.logits() {
            widen(source, &mut logits);
        }
        Ok(Self {
            inner,
            history,
            logits,
            memory: session::Memory {
                weights_bytes: numerical.weights_bytes,
                positional_bytes: numerical.positional_bytes,
                kv_bytes: numerical.state_bytes,
                scratch_bytes: numerical.scratch_bytes,
                history_bytes: numerical.history_bytes + capacity * size_of::<u32>(),
                logits_bytes: numerical.published_logits_bytes + VOCAB * size_of::<f32>(),
                metadata_bytes: numerical.owner_bytes + size_of::<Self>()
                    - size_of::<Session<'a>>(),
                metadata_allowance_bytes: 0,
                admitted_bytes: required,
            },
        })
    }

    fn validate(&self, tokens: &[u32], offset: usize) -> Result<(), session::Error> {
        if add(offset, tokens.len())? > self.capacity() {
            return Err(super::Error::Capacity {
                offset,
                tokens: tokens.len(),
                limit: self.capacity(),
            }
            .into());
        }
        for (index, &token) in tokens.iter().enumerate() {
            if token as usize >= VOCAB {
                return Err(super::Error::from(crate::cpu::Error::Token {
                    position: offset + index,
                    token,
                    vocab_size: VOCAB,
                })
                .into());
            }
        }
        Ok(())
    }
}

fn widen(source: &[half::bf16], target: &mut [f32]) {
    for (a, b) in source.iter().zip(target) {
        *b = a.to_f32();
    }
}

impl InferenceSession for GenerationSession<'_> {
    fn history(&self) -> &[u32] {
        &self.history
    }
    fn evaluated_len(&self) -> usize {
        self.inner.history().len()
    }
    fn pending_len(&self) -> usize {
        self.history.len() - self.evaluated_len()
    }
    fn capacity(&self) -> usize {
        self.inner.model.capacity
    }
    fn vocab_size(&self) -> usize {
        VOCAB
    }
    fn memory(&self) -> session::Memory {
        self.memory
    }

    fn logits(&self) -> Option<&[f32]> {
        (self.pending_len() == 0 && self.evaluated_len() != 0).then_some(&self.logits)
    }

    fn append(&mut self, tokens: &[u32]) -> Result<(), session::Error> {
        self.validate(tokens, self.history.len())?;
        self.history.extend_from_slice(tokens);
        Ok(())
    }

    fn set_tokens(&mut self, tokens: &[u32]) -> Result<(), session::Error> {
        self.validate(tokens, 0)?;
        if !tokens.starts_with(self.inner.history()) {
            self.inner.reset()?;
        }
        self.history.clear();
        self.history.extend_from_slice(tokens);
        Ok(())
    }

    fn evaluate_next_chunk(&mut self) -> Result<usize, session::Error> {
        let start = self.evaluated_len();
        let count = self.pending_len().min(self.inner.model.chunk);
        if count == 0 {
            return Ok(0);
        }
        let logits = self.inner.forward(&self.history[start..start + count])?;
        widen(logits, &mut self.logits);
        Ok(count)
    }

    fn discard_pending(&mut self) {
        self.history.truncate(self.evaluated_len());
    }

    fn reset(&mut self) {
        // The fallible request entry point propagates counter exhaustion.
        let _ = self.try_reset();
    }

    fn try_reset(&mut self) -> Result<(), session::Error> {
        self.inner.reset()?;
        self.history.clear();
        Ok(())
    }
}
