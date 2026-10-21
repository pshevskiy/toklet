//! Bounded CPU sessions with longest-prefix reuse and transactional chunk commits.
//!
//! Weights are borrowed immutably. A session owns separate per-layer FP32 K/V
//! `[kv_heads, capacity, head_dim]`, fixed chunk scratch, two logits buffers and
//! token history. Only history[..evaluated_len] is committed to KV. Appending a
//! sampled token changes history alone; evaluate it before sampling again.

use crate::{
    cpu,
    model::{self, KvCache, Model, Scratch, total_bytes, zeros},
    weights,
};
use std::fmt;

const METADATA_ALLOWANCE_BYTES: usize = 64 * 1024 * 1024;

/// Longest representable committed prefix. A shortened fully cached prompt
/// needs its final token replayed: only the current committed logits are held.
/// Pending sampled/input tokens never count as KV, even when they match.
pub(crate) fn reusable_prefix(
    history: &[u32],
    evaluated: usize,
    tokens: &[u32],
    distribution_valid: bool,
) -> (usize, bool) {
    let shared = history[..evaluated]
        .iter()
        .zip(tokens)
        .take_while(|(old, new)| old == new)
        .count();
    if shared == tokens.len() && (shared < evaluated || !distribution_valid) {
        (shared.saturating_sub(1), shared > 0)
    } else {
        (shared, false)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub capacity: usize,
    pub prefill_chunk_size: usize,
    /// Absolute rotary position of the first session token. KV indices remain
    /// relative to the session; capacity plus this offset must fit the model.
    pub position_offset: usize,
    /// Single-model/single-session numerical admission plus metadata allowance.
    /// Tokenizer/application buffers and other live sessions need separate budget.
    pub memory_limit_bytes: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            capacity: weights::INITIAL_CONTEXT_LIMIT,
            prefill_chunk_size: 128,
            position_offset: 0,
            memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Memory {
    pub weights_bytes: usize,
    pub kv_bytes: usize,
    pub scratch_bytes: usize,
    pub history_bytes: usize,
    /// Current/staged FP32 distributions, or two cached token IDs in an
    /// explicitly selected token-only backend mode.
    pub logits_bytes: usize,
    /// Allocated model-layer and cache-layer tables (Vec headers, not tensors).
    pub metadata_bytes: usize,
    pub metadata_allowance_bytes: usize,
    pub admitted_bytes: usize,
}

impl Memory {
    fn sum(&mut self) -> Result<(), Error> {
        self.admitted_bytes = [
            self.weights_bytes,
            self.kv_bytes,
            self.scratch_bytes,
            self.history_bytes,
            self.logits_bytes,
            self.metadata_bytes,
            self.metadata_allowance_bytes,
        ]
        .into_iter()
        .try_fold(0_usize, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or(model::Error::Kernel(cpu::Error::SizeOverflow))
        })?;
        Ok(())
    }
    fn admit(&self, limit: usize) -> Result<(), Error> {
        if self.admitted_bytes > limit {
            Err(Error::MemoryBudget {
                required: self.admitted_bytes,
                limit,
            })
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Model(model::Error),
    InvalidOptions(&'static str),
    Backend(String),
    MemoryBudget { required: usize, limit: usize },
}

impl From<model::Error> for Error {
    fn from(error: model::Error) -> Self {
        Self::Model(error)
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(error) => error.fmt(f),
            Self::Backend(reason) => f.write_str(reason),
            Self::InvalidOptions(reason) => write!(f, "invalid session options: {reason}"),
            Self::MemoryBudget { required, limit } => write!(
                f,
                "session admission requires {required} bytes, budget is {limit}"
            ),
        }
    }
}
impl std::error::Error for Error {}

pub struct Session<'model> {
    model: &'model Model,
    options: Options,
    memory: Memory,
    history: Vec<u32>,
    evaluated_len: usize,
    distribution_valid: bool,
    distribution_replay: bool,
    cache: KvCache,
    scratch: Scratch,
    logits: Vec<f32>,
    staged_logits: Vec<f32>,
}

impl<'model> Session<'model> {
    /// Check sizes and budget before allocating KV/scratch. Buffer ownership is
    /// fixed for the session lifetime; numerical execution performs no allocation.
    pub fn new(model: &'model Model, options: Options) -> Result<Self, Error> {
        let c = model.config();
        if options.capacity == 0 || options.capacity > weights::INITIAL_CONTEXT_LIMIT {
            return Err(Error::InvalidOptions("capacity must be within 1..=2048"));
        }
        if options.prefill_chunk_size == 0 || options.prefill_chunk_size > options.capacity {
            return Err(Error::InvalidOptions(
                "chunk size must be within 1..=capacity",
            ));
        }
        if options.memory_limit_bytes == 0
            || options.memory_limit_bytes > weights::INITIAL_MEMORY_LIMIT_BYTES
        {
            return Err(Error::InvalidOptions(
                "memory budget must be within the initial 6 GiB limit",
            ));
        }
        let end = options
            .position_offset
            .checked_add(options.capacity)
            .ok_or(model::Error::Kernel(cpu::Error::SizeOverflow))?;
        if end > c.max_positions {
            return Err(model::Error::Capacity {
                offset: options.position_offset,
                tokens: options.capacity,
                limit: c.max_positions,
            }
            .into());
        }
        // u32 token IDs and f32 storage both occupy four bytes.
        let mut memory = Memory {
            weights_bytes: model.allocated_weights_bytes()?,
            kv_bytes: KvCache::planned_bytes(c, options.capacity)?,
            scratch_bytes: Scratch::planned_bytes(c, options.prefill_chunk_size, options.capacity)?,
            history_bytes: total_bytes(&[options.capacity])?,
            logits_bytes: total_bytes(&[c.vocab_size, c.vocab_size])?,
            metadata_bytes: model
                .allocated_metadata_bytes()?
                .checked_add(KvCache::planned_metadata_bytes(c)?)
                .ok_or(model::Error::Kernel(cpu::Error::SizeOverflow))?,
            metadata_allowance_bytes: METADATA_ALLOWANCE_BYTES,
            admitted_bytes: 0,
        };
        memory.sum()?;
        memory.admit(options.memory_limit_bytes)?;
        let cache = KvCache::new(c, options.capacity)?;
        let scratch = Scratch::for_session(c, options.prefill_chunk_size, options.capacity)?;
        let mut history = Vec::new();
        history
            .try_reserve_exact(options.capacity)
            .map_err(|_| model::Error::Allocation)?;
        let logits = zeros(c.vocab_size)?;
        let staged_logits = zeros(c.vocab_size)?;
        // Account for actual Vec capacity, which is permitted to exceed requests.
        memory.kv_bytes = cache.allocated_bytes;
        memory.metadata_bytes = model
            .allocated_metadata_bytes()?
            .checked_add(cache.metadata_bytes)
            .ok_or(model::Error::Kernel(cpu::Error::SizeOverflow))?;
        memory.scratch_bytes = scratch.allocated_bytes();
        memory.history_bytes = total_bytes(&[history.capacity()])?;
        memory.logits_bytes = total_bytes(&[logits.capacity(), staged_logits.capacity()])?;
        memory.sum()?;
        memory.admit(options.memory_limit_bytes)?;
        Ok(Self {
            model,
            options,
            memory,
            history,
            evaluated_len: 0,
            distribution_valid: false,
            distribution_replay: false,
            cache,
            scratch,
            logits,
            staged_logits,
        })
    }

    pub fn history(&self) -> &[u32] {
        &self.history
    }
    pub fn evaluated_len(&self) -> usize {
        self.evaluated_len
    }
    pub fn pending_len(&self) -> usize {
        self.history.len() - self.evaluated_len
    }
    pub fn capacity(&self) -> usize {
        self.options.capacity
    }
    pub fn vocab_size(&self) -> usize {
        self.model.config().vocab_size
    }
    pub fn memory(&self) -> Memory {
        self.memory
    }

    /// A next-token distribution is available only for a nonempty fully
    /// evaluated history. Pending tokens must be evaluated or discarded first.
    pub fn logits(&self) -> Option<&[f32]> {
        (self.distribution_valid && self.evaluated_len > 0 && self.pending_len() == 0)
            .then_some(self.logits.as_slice())
    }

    fn validate(&self, tokens: &[u32], base: usize) -> Result<(), Error> {
        let end = base
            .checked_add(tokens.len())
            .ok_or(model::Error::Kernel(cpu::Error::SizeOverflow))?;
        if end > self.capacity() {
            return Err(model::Error::Capacity {
                offset: base,
                tokens: tokens.len(),
                limit: self.capacity(),
            }
            .into());
        }
        for (index, &token) in tokens.iter().enumerate() {
            if token as usize >= self.model.config().vocab_size {
                return Err(model::Error::Kernel(cpu::Error::Token {
                    position: base + index,
                    token,
                    vocab_size: self.model.config().vocab_size,
                })
                .into());
            }
        }
        Ok(())
    }

    /// Append input or sampled token IDs without evaluating them. Validate the
    /// entire append before mutating history; appends never grow its allocation.
    pub fn append(&mut self, tokens: &[u32]) -> Result<(), Error> {
        self.validate(tokens, self.history.len())?;
        self.history.extend_from_slice(tokens);
        Ok(())
    }

    /// Reuse the longest exact committed prefix and overwrite only its suffix.
    /// Shortened fully cached prompts replay one token to reconstruct logits.
    /// Validation precedes mutation; invalid replacements preserve all state.
    pub fn set_tokens(&mut self, tokens: &[u32]) -> Result<(), Error> {
        self.validate(tokens, 0)?;
        let (keep, recover) = reusable_prefix(
            &self.history,
            self.evaluated_len,
            tokens,
            self.distribution_valid,
        );
        let recover = recover
            || (self.distribution_replay
                && keep == self.evaluated_len
                && tokens.get(keep) == self.history.get(keep));
        if keep < self.evaluated_len {
            self.distribution_valid = false;
        }
        self.evaluated_len = keep;
        self.distribution_replay = recover;
        self.history.truncate(keep);
        self.history.extend_from_slice(&tokens[keep..]);
        Ok(())
    }

    /// Evaluate at most one admitted chunk. On success commit all its layers and
    /// final logits together. On failure the previously committed prefix remains
    /// valid and pending history remains for replay or discard. Staged KV tails
    /// and logits are overwritten on retry. An already evaluated prompt returns
    /// zero and retains its distribution. Cancellation can stop between calls;
    /// this synchronous method does not interrupt a chunk in flight.
    pub fn evaluate_next_chunk(&mut self) -> Result<usize, Error> {
        self.evaluate_chunk(None)
    }
    /// Explicit all-row diagnostic output `[committed_chunk_rows, vocabulary]`.
    /// The caller owns and separately admits this storage; session memory does
    /// not include it. A failed step may dirty output, but preserves committed
    /// session state. Shape is checked before any numerical/cache mutation.
    pub fn evaluate_next_chunk_with_logits(&mut self, output: &mut [f32]) -> Result<usize, Error> {
        self.evaluate_chunk(Some(output))
    }
    fn evaluate_chunk(&mut self, output: Option<&mut [f32]>) -> Result<usize, Error> {
        let count = self.pending_len().min(self.options.prefill_chunk_size);
        if let Some(output) = &output {
            cpu::length(
                "all-row logits",
                output.len(),
                cpu::product(count, self.vocab_size()).map_err(model::Error::Kernel)?,
            )
            .map_err(model::Error::Kernel)?;
        }
        if count == 0 {
            return Ok(0);
        }
        let start = self.evaluated_len;
        match output {
            Some(logits) => {
                self.model.forward_cached(
                    &self.history[start..start + count],
                    self.options.position_offset + start,
                    &mut self.scratch,
                    &mut self.cache,
                    start,
                    logits,
                )?;
                let width = self.model.config().vocab_size;
                self.staged_logits
                    .copy_from_slice(&logits[(count - 1) * width..]);
            }
            None => {
                self.model.forward_cached(
                    &self.history[start..start + count],
                    self.options.position_offset + start,
                    &mut self.scratch,
                    &mut self.cache,
                    start,
                    &mut self.staged_logits,
                )?;
            }
        }
        std::mem::swap(&mut self.logits, &mut self.staged_logits);
        self.evaluated_len += count;
        self.distribution_valid = true;
        self.distribution_replay = false;
        Ok(count)
    }

    pub fn evaluate(&mut self) -> Result<(), Error> {
        while self.evaluate_next_chunk()? != 0 {}
        Ok(())
    }

    /// Remove unevaluated input. If a reused/truncated KV prefix has no retained
    /// logits, stage its final retained token for replay before sampling again.
    /// Stale KV tails remain outside the committed length and need no clearing.
    pub fn discard_pending(&mut self) {
        self.history
            .truncate(self.evaluated_len + usize::from(self.distribution_replay));
        if self.distribution_replay {
            return;
        }
        if !self.distribution_valid && self.evaluated_len > 0 {
            self.evaluated_len -= 1;
            self.distribution_replay = true;
        }
    }

    /// Reset positions/history and distribution validity, retaining allocations.
    pub fn reset(&mut self) {
        self.history.clear();
        self.evaluated_len = 0;
        self.distribution_valid = false;
        self.distribution_replay = false;
    }
}

/// Coarse request/token boundary for concrete CPU and Metal sessions. Numerical
/// loops remain inside statically dispatched backend implementations.
pub trait InferenceSession {
    fn history(&self) -> &[u32];
    fn evaluated_len(&self) -> usize;
    fn pending_len(&self) -> usize;
    fn capacity(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn memory(&self) -> Memory;
    fn logits(&self) -> Option<&[f32]>;
    /// Coarse statically dispatched sampling boundary. None has the same
    /// unavailable-distribution meaning as logits(); CPU retains its oracle.
    fn greedy_token(&self) -> Option<Result<u32, crate::cpu::Error>> {
        self.logits().map(crate::sampling::greedy)
    }
    fn append(&mut self, tokens: &[u32]) -> Result<(), Error>;
    fn set_tokens(&mut self, tokens: &[u32]) -> Result<(), Error>;
    fn evaluate_next_chunk(&mut self) -> Result<usize, Error>;
    fn discard_pending(&mut self);
    fn reset(&mut self);
}
impl InferenceSession for Session<'_> {
    fn history(&self) -> &[u32] {
        Session::history(self)
    }
    fn evaluated_len(&self) -> usize {
        Session::evaluated_len(self)
    }
    fn pending_len(&self) -> usize {
        Session::pending_len(self)
    }
    fn capacity(&self) -> usize {
        Session::capacity(self)
    }
    fn vocab_size(&self) -> usize {
        Session::vocab_size(self)
    }
    fn memory(&self) -> Memory {
        Session::memory(self)
    }
    fn logits(&self) -> Option<&[f32]> {
        Session::logits(self)
    }
    fn append(&mut self, tokens: &[u32]) -> Result<(), Error> {
        Session::append(self, tokens)
    }
    fn set_tokens(&mut self, tokens: &[u32]) -> Result<(), Error> {
        Session::set_tokens(self, tokens)
    }
    fn evaluate_next_chunk(&mut self) -> Result<usize, Error> {
        Session::evaluate_next_chunk(self)
    }
    fn discard_pending(&mut self) {
        Session::discard_pending(self)
    }
    fn reset(&mut self) {
        Session::reset(self)
    }
}
