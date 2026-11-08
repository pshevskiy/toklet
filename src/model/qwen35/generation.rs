//! Pending sampled history around the existing transactional numerical session.
//! Recurrent state permits reuse of an entire committed prefix, not arbitrary
//! truncation. Changed/shortened prefixes reset and replay from position zero.
use super::*;
use crate::session::{self, InferenceSession};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum PrefixAction {
    Cold,
    ExactPrefix,
    FullyCached,
    ChangedReplay,
    ShortenedReplay,
    ExplicitReset,
}

/// Requested ownership, not physical footprint. Numerical owners retain their
/// distinct recurrent/history/KV labels. Tokenizer, output and telemetry owners
/// must be admitted separately by the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct GenerationMemory {
    pub numerical: Memory,
    pub pending_history_bytes: usize,
    pub wrapper_metadata_bytes: usize,
    pub admitted_bytes: usize,
}
impl GenerationMemory {
    fn compatibility(self) -> Result<session::Memory> {
        let n = self.numerical;
        // The generic view predates hybrid state. Keep the detailed owner
        // categories in this report, without counting any allocation twice.
        Ok(session::Memory {
            weights_bytes: n.weights_and_model_bytes,
            positional_bytes: 0,
            kv_bytes: n.kv_bytes,
            scratch_bytes: add(
                n.scratch_bytes,
                add(n.recurrent_bank_bytes, n.convolution_bank_bytes)?,
            )?,
            history_bytes: add(n.history_bytes, self.pending_history_bytes)?,
            logits_bytes: n.logits_bytes,
            metadata_bytes: add(n.metadata_bytes, self.wrapper_metadata_bytes)?,
            metadata_allowance_bytes: n.metadata_allowance_bytes,
            admitted_bytes: self.admitted_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct GenerationState {
    pub history_tokens: usize,
    pub committed_tokens: usize,
    pub pending_tokens: usize,
    /// Last set_tokens/reset decision; sampling appends do not overwrite it.
    pub prefix_action: PrefixAction,
    /// Last numerical call only, not cumulative request work or wall time.
    pub last_numerical_work: Work,
}

pub struct GenerationSession<
    'model,
    'weights,
    M: MatrixStorage = Matrix<'weights>,
    E: Execution<M> = Cpu,
> {
    inner: Session<'model, 'weights, M, E>,
    history: Vec<u32>,
    memory: GenerationMemory,
    compatibility_memory: session::Memory,
    prefix_action: PrefixAction,
}

fn session_error(error: Error) -> session::Error {
    match error {
        Error::Model(error) => session::Error::Model(error),
        Error::MemoryBudget { required, limit } => session::Error::MemoryBudget { required, limit },
        other => session::Error::Backend(other.to_string()),
    }
}

impl<'model, 'weights, M: MatrixStorage, E: Execution<M>>
    GenerationSession<'model, 'weights, M, E>
{
    /// Consume an existing valid numerical session. Admit all additional owners
    /// before allocating pending history; committed history/logits are preserved.
    pub fn new(mut inner: Session<'model, 'weights, M, E>, limit: usize) -> Result<Self> {
        if limit == 0 || limit > MEMORY_LIMIT {
            return Err(invalid("Qwen3.5 generation budget must fit6GiB"));
        }
        let metadata = std::mem::size_of::<Self>() - std::mem::size_of::<Session<'_, '_, M, E>>();
        let mut extra = add(metadata, mul(inner.options.capacity, 4)?)?;
        let admit = |extra| -> Result<usize> {
            let required = add(inner.memory().admitted_bytes, extra)?;
            if required > limit {
                return Err(Error::MemoryBudget { required, limit });
            }
            Ok(required)
        };
        admit(extra)?;
        let mut history = reserved(inner.options.capacity)?;
        history.extend_from_slice(inner.history());
        let history_bytes = mul(history.capacity(), 4)?;
        extra = add(metadata, history_bytes)?;
        let admitted_bytes = admit(extra)?;
        let numerical = inner.memory();
        let memory = GenerationMemory {
            numerical,
            pending_history_bytes: history_bytes,
            wrapper_metadata_bytes: metadata,
            admitted_bytes,
        };
        // The existing generic controller's coarse memory view has no hybrid
        // state fields. Its scratch total includes recurrent/convolution banks;
        // memory_report() retains the authoritative distinct owner categories.
        let compatibility_memory = memory.compatibility()?;
        // Execution repair must also leave room for this wrapper. Never widen
        // the numerical owner's originally requested limit.
        inner.options.memory_limit_bytes = inner.options.memory_limit_bytes.min(limit - extra);
        let prefix_action = if inner.history().is_empty() {
            PrefixAction::Cold
        } else {
            PrefixAction::FullyCached
        };
        Ok(Self {
            inner,
            history,
            memory,
            compatibility_memory,
            prefix_action,
        })
    }

    pub fn memory_report(&self) -> GenerationMemory {
        self.memory
    }
    pub fn state(&self) -> GenerationState {
        GenerationState {
            history_tokens: self.history.len(),
            committed_tokens: self.inner.history().len(),
            pending_tokens: self.history.len() - self.inner.history().len(),
            prefix_action: self.prefix_action,
            last_numerical_work: self.inner.work(),
        }
    }
    #[cfg(test)]
    pub(super) fn numerical_for_tests(&self) -> &Session<'model, 'weights, M, E> {
        &self.inner
    }
    /// Discard any pending sampled/input suffix, returning the committed session.
    pub fn into_numerical(self) -> Session<'model, 'weights, M, E> {
        self.inner
    }
    fn validate(&self, tokens: &[u32], base: usize) -> Result<()> {
        let end = add(base, tokens.len())?;
        if end > self.inner.options.capacity {
            return Err(super::super::Error::Capacity {
                offset: base,
                tokens: tokens.len(),
                limit: self.inner.options.capacity,
            }
            .into());
        }
        for (position, &token) in tokens.iter().enumerate() {
            if token as usize >= self.inner.model.config.vocab {
                return Err(cpu::Error::Token {
                    position: base + position,
                    token,
                    vocab_size: self.inner.model.config.vocab,
                }
                .into());
            }
        }
        Ok(())
    }
    fn append_tokens(&mut self, tokens: &[u32]) -> Result<()> {
        self.validate(tokens, self.history.len())?;
        self.history.extend_from_slice(tokens);
        Ok(())
    }
    fn replace_tokens(&mut self, tokens: &[u32]) -> Result<()> {
        self.validate(tokens, 0)?;
        let committed = self.inner.history();
        self.prefix_action = if !tokens.starts_with(committed) {
            let action = if committed.starts_with(tokens) {
                PrefixAction::ShortenedReplay
            } else {
                PrefixAction::ChangedReplay
            };
            self.inner.reset();
            action
        } else if committed.is_empty() {
            PrefixAction::Cold
        } else if tokens.len() == committed.len() {
            PrefixAction::FullyCached
        } else {
            PrefixAction::ExactPrefix
        };
        self.history.clear();
        self.history.extend_from_slice(tokens);
        Ok(())
    }
    fn evaluate_with(
        &mut self,
        run: impl FnOnce(&mut Session<'model, 'weights, M, E>, &[u32]) -> Result<()>,
    ) -> Result<usize> {
        let start = self.inner.history().len();
        let count = (self.history.len() - start).min(self.inner.options.chunk);
        if count == 0 {
            return Ok(0);
        }
        run(&mut self.inner, &self.history[start..start + count])?;
        Ok(count)
    }
}

impl<M: MatrixStorage, E: Execution<M>> InferenceSession for GenerationSession<'_, '_, M, E> {
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
        self.inner.options.capacity
    }
    fn vocab_size(&self) -> usize {
        self.inner.model.config.vocab
    }
    fn memory(&self) -> session::Memory {
        self.compatibility_memory
    }
    fn logits(&self) -> Option<&[f32]> {
        if self.pending_len() != 0 {
            None
        } else {
            self.inner.logits()
        }
    }
    fn append(&mut self, tokens: &[u32]) -> std::result::Result<(), session::Error> {
        self.append_tokens(tokens).map_err(session_error)
    }
    fn set_tokens(&mut self, tokens: &[u32]) -> std::result::Result<(), session::Error> {
        self.replace_tokens(tokens).map_err(session_error)
    }
    fn evaluate_next_chunk(&mut self) -> std::result::Result<usize, session::Error> {
        self.evaluate_with(|inner, tokens| inner.forward(tokens).map(|_| ()))
            .map_err(session_error)
    }
    fn discard_pending(&mut self) {
        self.history.truncate(self.inner.history().len());
    }
    fn reset(&mut self) {
        self.inner.reset();
        self.history.clear();
        self.prefix_action = PrefixAction::ExplicitReset;
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl<'model, 'weights, 'context>
    GenerationSession<'model, 'weights, MetalMatrix<'weights>, MetalExecution<'context>>
{
    pub fn requires_execution_repair(&self) -> bool {
        self.inner.requires_repair()
    }
    pub fn repair_execution(
        &mut self,
        context: &'context crate::metal::gemma_affine::Context,
    ) -> Result<()> {
        self.inner.repair_execution(context)?;
        let numerical = self.inner.memory();
        let memory = GenerationMemory {
            numerical,
            admitted_bytes: add(
                numerical.admitted_bytes,
                add(
                    self.memory.pending_history_bytes,
                    self.memory.wrapper_metadata_bytes,
                )?,
            )?,
            ..self.memory
        };
        let compatibility = memory.compatibility()?;
        self.memory = memory;
        self.compatibility_memory = compatibility;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
