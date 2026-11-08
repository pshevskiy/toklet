use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub capacity: usize,
    pub chunk: usize,
    pub position_offset: usize,
    pub memory_limit_bytes: usize,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub enum Outcome {
    #[default]
    Idle,
    Success,
    Error,
    Cancelled,
}
/// Completed numerical operations, reset per call; failures never count as
/// committed work. These counters contain no wall-clock or GPU estimates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct Work {
    pub outcome: Outcome,
    pub tokens_completed: usize,
    pub tokens_committed: usize,
    pub projections_completed: usize,
    pub gathers_completed: usize,
    pub convolution_steps_completed: usize,
    pub recurrence_steps_completed: usize,
    pub attention_heads_completed: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Memory {
    pub weights_and_model_bytes: usize,
    pub recurrent_bank_bytes: usize,
    pub convolution_bank_bytes: usize,
    pub kv_bytes: usize,
    pub scratch_bytes: usize,
    pub logits_bytes: usize,
    pub history_bytes: usize,
    pub metadata_bytes: usize,
    pub metadata_allowance_bytes: usize,
    pub admitted_bytes: usize,
}
impl Memory {
    pub(super) fn sum(&mut self) -> Result<()> {
        self.admitted_bytes = [
            self.weights_and_model_bytes,
            self.recurrent_bank_bytes,
            self.convolution_bank_bytes,
            self.kv_bytes,
            self.scratch_bytes,
            self.logits_bytes,
            self.history_bytes,
            self.metadata_bytes,
            self.metadata_allowance_bytes,
        ]
        .into_iter()
        .try_fold(0, add)?;
        Ok(())
    }
    pub(super) fn admit(self, limit: usize) -> Result<()> {
        let limit = limit.min(MEMORY_LIMIT);
        if self.admitted_bytes > limit {
            return Err(Error::MemoryBudget {
                required: self.admitted_bytes,
                limit,
            });
        }
        Ok(())
    }
}

pub(super) struct Banks {
    committed: Vec<f32>,
    staged: [Vec<f32>; 2],
}
impl Banks {
    #[cfg(test)]
    pub(super) fn committed(&self) -> &[f32] {
        &self.committed
    }
    fn new(count: usize) -> Result<Self> {
        Ok(Self {
            committed: zeros(count)?,
            staged: [zeros(count)?, zeros(count)?],
        })
    }
    pub(super) fn next(&mut self, local: usize) -> (&[f32], &mut [f32]) {
        if local == 0 {
            return (&self.committed, &mut self.staged[0]);
        }
        let (a, b) = self.staged.split_at_mut(1);
        if local % 2 == 1 {
            (&a[0], &mut b[0])
        } else {
            (&b[0], &mut a[0])
        }
    }
    fn commit(&mut self, last: usize) {
        std::mem::swap(&mut self.committed, &mut self.staged[last]);
    }
    fn bytes(&self) -> Result<usize> {
        mul(
            add(
                add(self.committed.capacity(), self.staged[0].capacity())?,
                self.staged[1].capacity(),
            )?,
            4,
        )
    }
    fn reset(&mut self) {
        self.committed.fill(0.);
    }
}
pub(super) struct DeltaState {
    pub rnn: Banks,
    pub history: Banks,
}
pub(super) struct Kv {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}
pub(super) struct DeltaScratch {
    pub qkv: Vec<f32>,
    pub conv: Vec<f32>,
    pub activated: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub z: Vec<f32>,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub decay: Vec<f32>,
    pub beta: Vec<f32>,
    pub output: Vec<f32>,
    pub norm: Vec<f32>,
    pub gated: Vec<f32>,
}
pub(super) struct AttentionScratch {
    pub q_gate: Vec<f32>,
    pub q: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub q_rot: Vec<f32>,
    pub k: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub k_rot: Vec<f32>,
    pub v: Vec<f32>,
    pub gate: Vec<f32>,
    pub context: Vec<f32>,
    pub gated: Vec<f32>,
    pub scores: Vec<f32>,
    pub probabilities: Vec<f32>,
}
pub(super) struct Scratch {
    pub x: Vec<f32>,
    pub input_norm: Vec<f32>,
    pub operator: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub activated: Vec<f32>,
    pub output: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub delta: DeltaScratch,
    pub attention: AttentionScratch,
}
impl Scratch {
    fn new(c: Config, capacity: usize) -> Result<Self> {
        let qw = c.query_width()?;
        let kw = c.kv_width()?;
        let dk = c.key_width()?;
        let dv = c.value_width()?;
        let cv = c.conv_width()?;
        Ok(Self {
            x: zeros(c.hidden)?,
            input_norm: zeros(c.hidden)?,
            operator: zeros(c.hidden)?,
            post_norm: zeros(c.hidden)?,
            gate: zeros(c.intermediate)?,
            up: zeros(c.intermediate)?,
            activated: zeros(c.intermediate)?,
            output: zeros(c.hidden)?,
            final_norm: zeros(c.hidden)?,
            delta: DeltaScratch {
                qkv: zeros(cv)?,
                conv: zeros(cv)?,
                activated: zeros(cv)?,
                q: zeros(dk)?,
                k: zeros(dk)?,
                z: zeros(dv)?,
                a: zeros(c.value_heads)?,
                b: zeros(c.value_heads)?,
                decay: zeros(c.value_heads)?,
                beta: zeros(c.value_heads)?,
                output: zeros(dv)?,
                norm: zeros(dv)?,
                gated: zeros(dv)?,
            },
            attention: AttentionScratch {
                q_gate: zeros(mul(qw, 2)?)?,
                q: zeros(qw)?,
                q_norm: zeros(qw)?,
                q_rot: zeros(qw)?,
                k: zeros(kw)?,
                k_norm: zeros(kw)?,
                k_rot: zeros(kw)?,
                v: zeros(kw)?,
                gate: zeros(qw)?,
                context: zeros(qw)?,
                gated: zeros(qw)?,
                scores: zeros(capacity)?,
                probabilities: zeros(capacity)?,
            },
        })
    }
    fn bytes(&self) -> Result<usize> {
        let d = &self.delta;
        let a = &self.attention;
        let all = [
            &self.x,
            &self.input_norm,
            &self.operator,
            &self.post_norm,
            &self.gate,
            &self.up,
            &self.activated,
            &self.output,
            &self.final_norm,
            &d.qkv,
            &d.conv,
            &d.activated,
            &d.q,
            &d.k,
            &d.z,
            &d.a,
            &d.b,
            &d.decay,
            &d.beta,
            &d.output,
            &d.norm,
            &d.gated,
            &a.q_gate,
            &a.q,
            &a.q_norm,
            &a.q_rot,
            &a.k,
            &a.k_norm,
            &a.k_rot,
            &a.v,
            &a.gate,
            &a.context,
            &a.gated,
            &a.scores,
            &a.probabilities,
        ];
        mul(all.into_iter().try_fold(0, |n, v| add(n, v.capacity()))?, 4)
    }
    fn planned(c: Config, capacity: usize) -> Result<usize> {
        let h = mul(c.hidden, 6)?;
        let mlp = mul(c.intermediate, 3)?;
        let d = [
            mul(c.conv_width()?, 3)?,
            mul(c.key_width()?, 2)?,
            mul(c.value_width()?, 4)?,
            mul(c.value_heads, 4)?,
        ]
        .into_iter()
        .try_fold(0, add)?;
        let a = [
            mul(c.query_width()?, 8)?,
            mul(c.kv_width()?, 4)?,
            mul(capacity, 2)?,
        ]
        .into_iter()
        .try_fold(0, add)?;
        mul(add(add(h, mlp)?, add(d, a)?)?, 4)
    }
}

pub struct Session<'model, 'weights, M: MatrixStorage = Matrix<'weights>, E: Execution<M> = Cpu> {
    pub(super) model: &'model Model<'weights, M>,
    pub(super) execution: E,
    pub(super) options: Options,
    pub(super) delta: Vec<DeltaState>,
    pub(super) kv: Vec<Kv>,
    pub(super) scratch: Scratch,
    history: Vec<u32>,
    published: Vec<f32>,
    staged: Vec<f32>,
    published_rows: usize,
    pub(super) memory: Memory,
    work: Work,
}
impl<'model, 'weights, M: MatrixStorage, E: Execution<M>> Session<'model, 'weights, M, E> {
    pub(super) fn planned_with_execution(
        model: &Model<'_, M>,
        options: Options,
        execution_bytes: usize,
    ) -> Result<Memory> {
        let c = model.config;
        let o = options;
        if o.capacity == 0
            || o.capacity > CAPACITY_LIMIT
            || o.chunk == 0
            || o.chunk > CHUNK_LIMIT
            || o.chunk > o.capacity
            || o.position_offset
                .checked_add(o.capacity)
                .is_none_or(|end| end > c.max_positions)
        {
            return Err(invalid("Qwen3.5 session capacity/chunk/absolute position"));
        }
        let mut memory = Memory {
            weights_and_model_bytes: model.model_bytes()?,
            recurrent_bank_bytes: mul(
                mul(mul(c.value_width()?, c.key_dim)?, c.delta_count())?,
                12,
            )?,
            convolution_bank_bytes: mul(mul(c.conv_width()?, c.delta_count())?, 36)?,
            kv_bytes: mul(
                mul(mul(c.kv_width()?, o.capacity)?, c.attention_count())?,
                8,
            )?,
            scratch_bytes: add(Scratch::planned(c, o.capacity)?, execution_bytes)?,
            logits_bytes: mul(mul(o.chunk, c.vocab)?, 8)?,
            history_bytes: mul(o.capacity, 4)?,
            metadata_bytes: add(
                std::mem::size_of::<Self>(),
                add(
                    mul(c.delta_count(), std::mem::size_of::<DeltaState>())?,
                    mul(c.attention_count(), std::mem::size_of::<Kv>())?,
                )?,
            )?,
            metadata_allowance_bytes: METADATA_ALLOWANCE,
            admitted_bytes: 0,
        };
        memory.sum()?;
        Ok(memory)
    }
    pub(super) fn with_execution(
        model: &'model Model<'weights, M>,
        options: Options,
        execution: E,
    ) -> Result<Self> {
        let mut memory = Self::planned_with_execution(model, options, execution.owned_bytes()?)?;
        memory.admit(options.memory_limit_bytes)?;
        let c = model.config;
        let mut delta = reserved(c.delta_count())?;
        for _ in 0..c.delta_count() {
            delta.push(DeltaState {
                rnn: Banks::new(mul(c.value_width()?, c.key_dim)?)?,
                history: Banks::new(mul(c.conv_width()?, 3)?)?,
            });
        }
        let mut kv = reserved(c.attention_count())?;
        let count = mul(c.kv_width()?, options.capacity)?;
        for _ in 0..c.attention_count() {
            kv.push(Kv {
                k: zeros(count)?,
                v: zeros(count)?,
            });
        }
        let scratch = Scratch::new(c, options.capacity)?;
        let history = reserved(options.capacity)?;
        let published = zeros(mul(options.chunk, c.vocab)?)?;
        let staged = zeros(mul(options.chunk, c.vocab)?)?;
        memory.recurrent_bank_bytes = delta.iter().try_fold(0, |n, d| add(n, d.rnn.bytes()?))?;
        memory.convolution_bank_bytes = delta
            .iter()
            .try_fold(0, |n, d| add(n, d.history.bytes()?))?;
        memory.kv_bytes = kv.iter().try_fold(0, |n, k| {
            add(n, mul(add(k.k.capacity(), k.v.capacity())?, 4)?)
        })?;
        memory.scratch_bytes = add(scratch.bytes()?, execution.owned_bytes()?)?;
        memory.logits_bytes = mul(add(published.capacity(), staged.capacity())?, 4)?;
        memory.history_bytes = mul(history.capacity(), 4)?;
        memory.metadata_bytes = add(
            std::mem::size_of::<Self>(),
            add(
                mul(delta.capacity(), std::mem::size_of::<DeltaState>())?,
                mul(kv.capacity(), std::mem::size_of::<Kv>())?,
            )?,
        )?;
        memory.sum()?;
        memory.admit(options.memory_limit_bytes)?;
        Ok(Self {
            model,
            execution,
            options,
            delta,
            kv,
            scratch,
            history,
            published,
            staged,
            published_rows: 0,
            memory,
            work: Work::default(),
        })
    }
    pub fn memory(&self) -> Memory {
        self.memory
    }
    pub fn work(&self) -> Work {
        self.work
    }
    pub fn history(&self) -> &[u32] {
        &self.history
    }
    pub fn logits(&self) -> Option<&[f32]> {
        self.published_rows.checked_sub(1).map(|row| {
            &self.published[row * self.model.config.vocab..(row + 1) * self.model.config.vocab]
        })
    }
    pub fn reset(&mut self) {
        for d in &mut self.delta {
            d.rnn.reset();
            d.history.reset();
        }
        self.history.clear();
        self.published_rows = 0;
        self.work = Work::default();
    }
    pub fn forward(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        self.forward_with_cancel(tokens, &AtomicBool::new(false))
    }
    pub fn forward_with_cancel(&mut self, tokens: &[u32], cancel: &AtomicBool) -> Result<&[f32]> {
        self.evaluate::<false, _>(tokens, cancel, &mut |_| Ok(()))
    }
    pub fn forward_observed<F: for<'b> FnMut(Observation<'b>) -> Result<()>>(
        &mut self,
        tokens: &[u32],
        cancel: &AtomicBool,
        observe: &mut F,
    ) -> Result<&[f32]> {
        self.evaluate::<true, F>(tokens, cancel, observe)
    }
    fn evaluate<const TRACE: bool, F: for<'b> FnMut(Observation<'b>) -> Result<()>>(
        &mut self,
        tokens: &[u32],
        cancel: &AtomicBool,
        observe: &mut F,
    ) -> Result<&[f32]> {
        self.work = Work::default();
        let result = self.stage::<TRACE, F>(tokens, cancel, observe);
        if let Err(error) = result {
            self.work.outcome = if matches!(error, Error::Cancelled) {
                Outcome::Cancelled
            } else {
                Outcome::Error
            };
            return Err(error);
        }
        // No validation, allocation, callback or cancellation after this point.
        let last = (tokens.len() - 1) % 2;
        for d in &mut self.delta {
            d.rnn.commit(last);
            d.history.commit(last);
        }
        self.history.extend_from_slice(tokens);
        std::mem::swap(&mut self.published, &mut self.staged);
        self.published_rows = tokens.len();
        self.work.tokens_committed = tokens.len();
        self.work.outcome = Outcome::Success;
        Ok(&self.published[..tokens.len() * self.model.config.vocab])
    }
    fn stage<const TRACE: bool, F: for<'b> FnMut(Observation<'b>) -> Result<()>>(
        &mut self,
        tokens: &[u32],
        cancel: &AtomicBool,
        observe: &mut F,
    ) -> Result<()> {
        let c = self.model.config;
        let offset = self.history.len();
        if tokens.is_empty() {
            return Err(invalid("Qwen3.5 numerical chunk must be nonempty"));
        }
        if tokens.len() > self.options.chunk {
            return Err(super::super::Error::ScratchCapacity {
                tokens: tokens.len(),
                limit: self.options.chunk,
            }
            .into());
        }
        if offset
            .checked_add(tokens.len())
            .is_none_or(|n| n > self.options.capacity)
        {
            return Err(super::super::Error::Capacity {
                offset,
                tokens: tokens.len(),
                limit: self.options.capacity,
            }
            .into());
        }
        for (position, &token) in tokens.iter().enumerate() {
            if token as usize >= c.vocab {
                return Err(cpu::Error::Token {
                    position,
                    token,
                    vocab_size: c.vocab,
                }
                .into());
            }
        }
        for (local, &token) in tokens.iter().enumerate() {
            check_cancel(cancel)?;
            let position = offset + local;
            let absolute = self.options.position_offset + position;
            let mut trace = super::forward::Trace {
                observe,
                position: absolute,
                token: position,
                layer: None,
                cancel,
            };
            super::forward::frame::<TRACE, F, M, E>(
                self.model,
                &mut self.execution,
                &mut self.delta,
                &mut self.kv,
                &mut self.scratch,
                token,
                local,
                position,
                absolute,
                self.options.capacity,
                &mut self.staged[local * c.vocab..(local + 1) * c.vocab],
                &mut self.work,
                &mut trace,
            )?;
            self.work.tokens_completed += 1;
        }
        if TRACE {
            observe(Observation {
                token: offset + tokens.len() - 1,
                position: self.options.position_offset + offset + tokens.len() - 1,
                layer: None,
                stage: "before_commit",
                head: None,
                values: &[],
            })?;
        }
        check_cancel(cancel)
    }
}
impl<'model, 'weights> Session<'model, 'weights> {
    pub fn planned_memory(model: &Model<'weights>, options: Options) -> Result<Memory> {
        Self::planned_with_execution(model, options, 0)
    }
    pub(super) fn new(model: &'model Model<'weights>, options: Options) -> Result<Self> {
        Self::with_execution(model, options, Cpu)
    }
}
pub(super) fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}
