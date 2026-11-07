//! Pending-token history and generation integration outside numerical kernels.
//!
//! Reuse only an entire exact committed prefix. A changed or shortened prompt
//! conservatively replays from position0; no KV truncation/ring recovery is
//! assumed. Invalidation is immediate; fallible physical reset is deferred until
//! evaluation, and its failure leaves an explicit replay-required state.
use super::*;
use crate::session::{self, InferenceSession, Memory, Options};

pub struct GenerationSession<'model, 'weights, M = Matrix<'weights>, B = Scalar> {
    inner: Session<'model, 'weights, M, B>,
    history: Vec<u32>,
    len: usize,
    capacity: usize,
    chunk: usize,
    needs_replay: bool,
    output: forward::OutputRows,
    memory: Memory,
}

impl<'weights> Model<'weights> {
    /// Generation/state integration for the same short raw-logit graph. Exact
    /// committed-prefix reuse is supported; changed/shortened prompts replay.
    /// No tokenizer/chat profile or new position/architecture capability is granted.
    pub fn new_generation_session(
        &self,
        options: Options,
    ) -> Result<GenerationSession<'_, 'weights>, session::Error> {
        GenerationSession::with_backend(self, Scalar, options)
    }
}

pub(crate) fn validate_options<M: MatrixSource>(
    model: &Model<'_, M>,
    options: Options,
) -> Result<(), session::Error> {
    if options.capacity == 0 || options.capacity > model.layout.positions {
        return Err(session::Error::InvalidOptions(
            "Gemma capacity must fit executable model positions",
        ));
    }
    if options.prefill_chunk_size == 0
        || options.prefill_chunk_size > options.capacity
        || options.prefill_chunk_size > MAX_POSITIONS
    {
        return Err(session::Error::InvalidOptions(
            "Gemma chunk must fit capacity and the32-row call bound",
        ));
    }
    if options.position_offset != 0 {
        return Err(session::Error::InvalidOptions(
            "Gemma requires position offset0",
        ));
    }
    if options.memory_limit_bytes == 0
        || options.memory_limit_bytes > crate::weights::INITIAL_MEMORY_LIMIT_BYTES
    {
        return Err(session::Error::InvalidOptions(
            "Gemma budget must be within the initial6GiB limit",
        ));
    }
    Ok(())
}

impl<'model, 'weights, M: MatrixSource, B: Operations<M>>
    GenerationSession<'model, 'weights, M, B>
{
    pub(crate) fn planned_wrapper_bytes(capacity: usize) -> Result<usize, Error> {
        add_bytes(
            std::mem::size_of::<Self>() - std::mem::size_of::<Session<'_, '_, M, B>>(),
            allocation_bytes(capacity)?,
        )
    }
    pub(crate) fn with_backend(
        model: &'model Model<'weights, M>,
        operations: B,
        options: Options,
    ) -> Result<Self, session::Error> {
        validate_options(model, options)?;
        let extra = Self::planned_wrapper_bytes(options.capacity)?;
        let required = add_bytes(
            add_bytes(
                add_bytes(
                    model.model_bytes(),
                    model.planned_backend_session_bytes_for_capacity::<B>(
                        options.capacity,
                        options.prefill_chunk_size,
                    )?,
                )?,
                operations.memory()?.total()?,
            )?,
            extra,
        )?;
        if required > options.memory_limit_bytes {
            return Err(session::Error::MemoryBudget {
                required,
                limit: options.memory_limit_bytes,
            });
        }
        let inner = model
            .session_with_backend_for_capacity(
                operations,
                options.capacity,
                options.prefill_chunk_size,
                options.memory_limit_bytes - extra,
            )
            .map_err(Into::into)?;
        let backend = inner.operations.memory()?;
        let raw_bytes = inner.allocated_bytes()?;
        let history_storage = token_history(options.capacity)?;
        let mut kv = 0;
        for cache in &inner.caches {
            kv = add_bytes(
                kv,
                cache.allocated_bytes() - std::mem::size_of::<WindowKvCache>(),
            )?;
        }
        for stage in &inner.staging {
            kv = add_bytes(kv, total_bytes(&[stage.k.capacity(), stage.v.capacity()])?)?;
        }
        let logits = total_bytes(&[
            inner.scratch.logits.capacity(),
            inner.scratch.published.capacity(),
        ])?;
        let scratch = add_bytes(
            total_bytes(&inner.scratch.capacities())? - logits,
            backend.scratch_bytes,
        )?;
        let history = total_bytes(&[inner.history.capacity(), history_storage.capacity()])?;
        let wrapper = add_bytes(
            std::mem::size_of::<Self>() - std::mem::size_of::<Session<'_, '_, M, B>>(),
            allocation_bytes(history_storage.capacity())?,
        )?;
        let session_bytes = add_bytes(raw_bytes, wrapper)?;
        let numerical = add_bytes(add_bytes(kv, scratch)?, add_bytes(logits, history)?)?;
        let metadata = session_bytes
            .checked_sub(add_bytes(numerical, backend.metadata_allowance_bytes)?)
            .ok_or(cpu::Error::SizeOverflow)
            .map_err(Error::Kernel)?;
        let model_metadata = model
            .model_bytes()
            .checked_sub(add_bytes(
                model.tensor_bytes,
                model.metadata_allowance_bytes,
            )?)
            .ok_or(cpu::Error::SizeOverflow)
            .map_err(Error::Kernel)?;
        let memory = Memory {
            weights_bytes: model.tensor_bytes,
            positional_bytes: 0,
            kv_bytes: kv,
            scratch_bytes: scratch,
            history_bytes: history,
            logits_bytes: logits,
            metadata_bytes: add_bytes(metadata, model_metadata)?,
            metadata_allowance_bytes: add_bytes(
                model.metadata_allowance_bytes,
                backend.metadata_allowance_bytes,
            )?,
            admitted_bytes: add_bytes(model.model_bytes(), session_bytes)?,
        };
        if memory.admitted_bytes > options.memory_limit_bytes {
            return Err(session::Error::MemoryBudget {
                required: memory.admitted_bytes,
                limit: options.memory_limit_bytes,
            });
        }
        Ok(Self {
            inner,
            history: history_storage,
            len: 0,
            capacity: options.capacity,
            chunk: options.prefill_chunk_size,
            needs_replay: false,
            output: forward::OutputRows::All,
            memory,
        })
    }
    pub fn history(&self) -> &[u32] {
        &self.history[..self.len]
    }
    pub fn evaluated_len(&self) -> usize {
        if self.needs_replay {
            0
        } else {
            self.inner.tokens().len()
        }
    }
    pub fn pending_len(&self) -> usize {
        self.len - self.evaluated_len()
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn vocab_size(&self) -> usize {
        self.inner.model.layout.vocab
    }
    pub fn memory(&self) -> Memory {
        self.memory
    }
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn operations(&self) -> &B {
        &self.inner.operations
    }
    pub fn logits(&self) -> Option<&[f32]> {
        if self.needs_replay || self.pending_len() != 0 || self.len == 0 {
            return None;
        }
        let values = self.inner.logits();
        let start = values.len().checked_sub(self.vocab_size())?;
        Some(&values[start..])
    }
    fn validate(&self, tokens: &[u32], base: usize) -> Result<(), session::Error> {
        let end = base
            .checked_add(tokens.len())
            .ok_or(cpu::Error::SizeOverflow)
            .map_err(Error::Kernel)?;
        if end > self.capacity {
            return Err(Error::Capacity {
                offset: base,
                tokens: tokens.len(),
                limit: self.capacity,
            }
            .into());
        }
        for (i, &token) in tokens.iter().enumerate() {
            if token as usize >= self.vocab_size() {
                return Err(Error::Kernel(cpu::Error::Token {
                    position: base + i,
                    token,
                    vocab_size: self.vocab_size(),
                })
                .into());
            }
        }
        Ok(())
    }
    pub fn append(&mut self, tokens: &[u32]) -> Result<(), session::Error> {
        self.validate(tokens, self.len)?;
        self.history[self.len..self.len + tokens.len()].copy_from_slice(tokens);
        self.len += tokens.len();
        Ok(())
    }
    pub fn set_tokens(&mut self, tokens: &[u32]) -> Result<(), session::Error> {
        self.validate(tokens, 0)?;
        self.inner.operations.generation_history_replaced();
        if !self.needs_replay && !tokens.starts_with(self.inner.tokens()) {
            self.needs_replay = true;
        }
        self.history[..tokens.len()].copy_from_slice(tokens);
        self.len = tokens.len();
        Ok(())
    }
    /// Physical reset is attempted only when new numerical work is needed. A
    /// reset failure keeps needs_replay=true, evaluated_len0 and logits hidden.
    /// A later forward failure preserves the now-valid committed inner prefix.
    pub fn evaluate_next_chunk(&mut self) -> Result<usize, session::Error> {
        self.evaluate_next_chunk_rows(self.output)
    }

    /// The audited Mac BF16 native bridge exposes only the final distribution.
    /// Raw scoring and speculative verification keep the complete rows.
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn use_last_row_prefill(&mut self) {
        self.output = forward::OutputRows::Last;
    }
    #[cfg(test)]
    pub(crate) fn evaluate_next_chunk_last_row(&mut self) -> Result<usize, session::Error> {
        self.evaluate_next_chunk_rows(forward::OutputRows::Last)
    }
    fn evaluate_next_chunk_rows(
        &mut self,
        output: forward::OutputRows,
    ) -> Result<usize, session::Error> {
        let count = self.pending_len().min(self.chunk);
        if count == 0 {
            return Ok(0);
        }
        if self.needs_replay {
            self.inner.reset().map_err(Into::into)?;
            self.needs_replay = false;
        }
        let start = self.inner.tokens().len();
        let last_row_only = match output {
            forward::OutputRows::All => false,
            #[cfg(any(
                test,
                all(feature = "metal", target_os = "macos", target_arch = "aarch64")
            ))]
            forward::OutputRows::Last => true,
        };
        self.inner
            .operations
            .begin_generation_step(start, count, last_row_only);
        let result = forward::run(&mut self.inner, &self.history[start..start + count], output);
        let committed = result.as_ref().ok().map(|_| self.inner.tokens().len());
        self.inner.operations.finish_generation_step(committed);
        result.map_err(Into::into)?;
        Ok(count)
    }
    /// Experimental nonstreaming round. Proposals are read-only until the
    /// strict target verifier and last cancellation check succeed. A hidden
    /// stop or correction is pending and will be evaluated before another draft.
    #[cfg(test)]
    pub(crate) fn verify_greedy_round(
        &mut self,
        proposals: &[u32],
        stop_tokens: &[u32],
        committed_hidden: &mut [f32],
        cancelled: impl Fn() -> bool,
    ) -> Result<Option<crate::generation::GreedyVerification>, crate::generation::Error> {
        use crate::generation::{self, Error as GenerationError, VerifiedNext};
        let start = self.evaluated_len();
        let current = self.logits().ok_or(GenerationError::InvalidRequest(
            "Gemma verification requires a committed distribution",
        ))?;
        if proposals.len() > self.chunk || committed_hidden.len() != self.inner.model.layout.hidden
        {
            return Err(GenerationError::InvalidRequest(
                "Gemma verification chunk or hidden shape differs",
            ));
        }
        self.validate(proposals, start)
            .map_err(GenerationError::Session)?;
        if start + proposals.len() == self.capacity {
            return Err(GenerationError::InvalidRequest(
                "no room for the pending target token",
            ));
        }
        if stop_tokens
            .iter()
            .any(|&id| id as usize >= self.vocab_size())
        {
            return Err(GenerationError::InvalidRequest(
                "Gemma stop token outside vocabulary",
            ));
        }
        if cancelled() {
            return Ok(None);
        }
        let verified = if proposals.is_empty() {
            let result = generation::verify_greedy(proposals, current, current.len(), stop_tokens)?;
            if cancelled() {
                return Ok(None);
            }
            result
        } else {
            let offside = forward::prepare(&mut self.inner, proposals, forward::OutputRows::All)
                .map_err(|error| GenerationError::Session(error.into()))?;
            let current = offside.current().expect("committed distribution validated");
            let result = generation::verify_greedy_segments(
                proposals,
                current,
                offside.staged(),
                stop_tokens,
            )?;
            if cancelled() {
                return Ok(None);
            }
            offside
                .commit_with_hidden(result.accepted, Some(committed_hidden))
                .map_err(|error| GenerationError::Session(error.into()))?;
            result
        };
        let pending = match verified.next {
            VerifiedNext::Token(id) | VerifiedNext::Stop(id) => id,
        };
        let accepted_end = start + verified.accepted;
        self.history[start..accepted_end].copy_from_slice(&proposals[..verified.accepted]);
        self.history[accepted_end] = pending;
        self.len = accepted_end + 1;
        Ok(Some(verified))
    }

    pub fn discard_pending(&mut self) {
        self.len = self.evaluated_len();
    }
    /// Invalidate the current distribution/prefix immediately without allocating.
    pub fn reset(&mut self) {
        self.inner.operations.generation_history_replaced();
        self.len = 0;
        self.needs_replay = true;
    }
}

impl<M: MatrixSource, B: Operations<M>> InferenceSession for GenerationSession<'_, '_, M, B> {
    fn history(&self) -> &[u32] {
        Self::history(self)
    }
    fn evaluated_len(&self) -> usize {
        Self::evaluated_len(self)
    }
    fn pending_len(&self) -> usize {
        Self::pending_len(self)
    }
    fn capacity(&self) -> usize {
        Self::capacity(self)
    }
    fn vocab_size(&self) -> usize {
        Self::vocab_size(self)
    }
    fn memory(&self) -> Memory {
        Self::memory(self)
    }
    fn logits(&self) -> Option<&[f32]> {
        Self::logits(self)
    }
    fn append(&mut self, tokens: &[u32]) -> Result<(), session::Error> {
        Self::append(self, tokens)
    }
    fn set_tokens(&mut self, tokens: &[u32]) -> Result<(), session::Error> {
        Self::set_tokens(self, tokens)
    }
    fn evaluate_next_chunk(&mut self) -> Result<usize, session::Error> {
        Self::evaluate_next_chunk(self)
    }
    fn discard_pending(&mut self) {
        Self::discard_pending(self)
    }
    fn reset(&mut self) {
        Self::reset(self)
    }
}

#[cfg(all(test, feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod assistant_experiment;

#[cfg(test)]
pub(in crate::model::gemma_packed) mod tests;
