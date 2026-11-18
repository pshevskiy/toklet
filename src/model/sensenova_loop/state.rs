//! Transactional state for the 56 SenseNova text visits.

use super::{Attention, EXECUTION_VISITS, Plan, Visit};
use crate::{
    cpu,
    model::{
        self,
        qwen35::{
            CAPACITY_LIMIT, CHUNK_LIMIT, Error, Memory, Options,
            state::{Banks, DeltaState, Kv, check_cancel},
        },
    },
};
use std::{mem::size_of, sync::atomic::AtomicBool};

const DELTA: usize = 42;
const FULL: usize = 14;
const ALLOWANCE: usize = 64 * 1024 * 1024;
type Result<T> = std::result::Result<T, Error>;

fn invalid(reason: &'static str) -> Error {
    model::Error::InvalidConfig(reason).into()
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or(cpu::Error::SizeOverflow.into())
}

fn mul(a: usize, b: usize) -> Result<usize> {
    Ok(cpu::product(a, b)?)
}

fn reserved<T>(count: usize) -> Result<Vec<T>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .map_err(|_| model::Error::Allocation)?;
    Ok(output)
}

#[derive(Clone, Copy)]
struct Layout {
    rnn: usize,
    conv: usize,
    kv_heads: usize,
    head_dim: usize,
    position_limit: usize,
    vocab: usize,
}

impl Layout {
    fn for_plan(plan: &Plan) -> Result<Self> {
        let shape = plan.text_shape();
        let key = mul(shape.linear_num_key_heads, shape.linear_key_head_dim)?;
        let value = mul(shape.linear_num_value_heads, shape.linear_value_head_dim)?;
        Ok(Self {
            rnn: mul(value, shape.linear_key_head_dim)?,
            conv: mul(add(mul(2, key)?, value)?, shape.linear_conv_kernel_dim - 1)?,
            kv_heads: shape.num_key_value_heads,
            head_dim: shape.head_dim,
            position_limit: shape.max_position_embeddings,
            vocab: shape.vocab_size,
        })
    }

    fn kv_words(self, capacity: usize) -> Result<usize> {
        mul(mul(self.kv_heads, self.head_dim)?, capacity)
    }
}

#[derive(Clone, Copy)]
enum Owner {
    Linear(usize),
    Full(usize),
}

pub struct State {
    visits: [Visit; EXECUTION_VISITS],
    owners: [Owner; EXECUTION_VISITS],
    options: Options,
    layout: Layout,
    delta: Vec<DeltaState>,
    kv: Vec<Kv>,
    history: Vec<u32>,
    staged: Vec<f32>,
    logits: Vec<f32>,
    published_rows: usize,
    memory: Memory,
}

#[derive(Clone, Copy, Debug)]
pub struct Token {
    pub id: u32,
    pub position: usize,
    pub absolute: usize,
}

pub struct Transition<'a> {
    pub prior: &'a [f32],
    pub next: &'a mut [f32],
}

pub enum VisitState<'a> {
    Linear {
        recurrent: Transition<'a>,
        convolution: Transition<'a>,
    },
    Full(KvRows<'a>),
}

pub struct KvRows<'a> {
    storage: &'a mut Kv,
    position: usize,
    stride: usize,
    head_dim: usize,
}

pub struct KvHead<'a> {
    pub past_key: &'a [f32],
    pub past_value: &'a [f32],
    pub key: &'a mut [f32],
    pub value: &'a mut [f32],
}

impl KvRows<'_> {
    pub(super) fn storage(&mut self) -> &mut Kv {
        self.storage
    }

    /// Head-major append storage. Only the current row is writable.
    pub fn head(&mut self, head: usize) -> Option<KvHead<'_>> {
        let start = head.checked_mul(self.stride)?;
        let end = start.checked_add(self.stride)?;
        let offset = self.position * self.head_dim;
        let (past_key, rest_k) = self.storage.k.get_mut(start..end)?.split_at_mut(offset);
        let (past_value, rest_v) = self.storage.v.get_mut(start..end)?.split_at_mut(offset);
        Some(KvHead {
            past_key,
            past_value,
            key: &mut rest_k[..self.head_dim],
            value: &mut rest_v[..self.head_dim],
        })
    }
}

pub struct Frame<'a> {
    visits: &'a [Visit; EXECUTION_VISITS],
    owners: &'a [Owner; EXECUTION_VISITS],
    delta: &'a mut [DeltaState],
    kv: &'a mut [Kv],
    cancel: &'a AtomicBool,
    next: usize,
    position: usize,
    local: usize,
    stride: usize,
    head_dim: usize,
}

impl Frame<'_> {
    /// Lend the next numbered visit's state. Complete the staged write before
    /// advancing. An error from the frame's caller discards the entire chunk.
    pub fn next_visit(&mut self) -> Result<Option<(Visit, VisitState<'_>)>> {
        check_cancel(self.cancel)?;
        let Some(&visit) = self.visits.get(self.next) else {
            return Ok(None);
        };
        let owner = self.owners[self.next];
        self.next += 1;
        let state = match owner {
            Owner::Linear(i) => {
                let entry = &mut self.delta[i];
                let (read_rnn, write_rnn) = entry.rnn.next(self.local);
                let (read_conv, write_conv) = entry.history.next(self.local);
                VisitState::Linear {
                    recurrent: Transition {
                        prior: read_rnn,
                        next: write_rnn,
                    },
                    convolution: Transition {
                        prior: read_conv,
                        next: write_conv,
                    },
                }
            }
            Owner::Full(i) => VisitState::Full(KvRows {
                storage: &mut self.kv[i],
                position: self.position,
                stride: self.stride,
                head_dim: self.head_dim,
            }),
        };
        Ok(Some((visit, state)))
    }
}

impl State {
    pub fn planned(
        plan: &Plan,
        options: Options,
        weights_and_model_bytes: usize,
        scratch_bytes: usize,
    ) -> Result<Memory> {
        Self::plan_layout(
            Layout::for_plan(plan)?,
            options,
            weights_and_model_bytes,
            scratch_bytes,
        )
    }

    pub fn new(
        plan: &Plan,
        options: Options,
        weights_and_model_bytes: usize,
        scratch_bytes: usize,
    ) -> Result<Self> {
        Self::with_layout(
            *plan.visits(),
            Layout::for_plan(plan)?,
            options,
            weights_and_model_bytes,
            scratch_bytes,
        )
    }

    fn plan_layout(
        layout: Layout,
        options: Options,
        model_bytes: usize,
        scratch_bytes: usize,
    ) -> Result<Memory> {
        if options.capacity == 0
            || options.capacity > CAPACITY_LIMIT
            || options.chunk == 0
            || options.chunk > options.capacity.min(CHUNK_LIMIT)
            || options
                .position_offset
                .checked_add(options.capacity)
                .is_none_or(|end| end > layout.position_limit)
        {
            return Err(invalid("SenseNova state capacity, chunk or position bound"));
        }
        let mut memory = Memory {
            weights_and_model_bytes: model_bytes,
            recurrent_bank_bytes: mul(mul(layout.rnn, DELTA)?, 12)?,
            convolution_bank_bytes: mul(mul(layout.conv, DELTA)?, 12)?,
            kv_bytes: mul(mul(layout.kv_words(options.capacity)?, FULL)?, 8)?,
            scratch_bytes,
            logits_bytes: mul(mul(options.chunk, layout.vocab)?, 8)?,
            history_bytes: mul(options.capacity, 4)?,
            metadata_bytes: Self::metadata_bytes(DELTA, FULL)?,
            metadata_allowance_bytes: ALLOWANCE,
            admitted_bytes: 0,
        };
        memory.sum()?;
        memory.admit(options.memory_limit_bytes)?;
        Ok(memory)
    }

    fn metadata_bytes(delta: usize, full: usize) -> Result<usize> {
        add(
            size_of::<Self>(),
            add(
                mul(delta, size_of::<DeltaState>())?,
                mul(full, size_of::<Kv>())?,
            )?,
        )
    }

    fn with_layout(
        visits: [Visit; EXECUTION_VISITS],
        layout: Layout,
        options: Options,
        model_bytes: usize,
        scratch_bytes: usize,
    ) -> Result<Self> {
        let mut memory = Self::plan_layout(layout, options, model_bytes, scratch_bytes)?;
        let mut counts = [0; 2];
        let owners = visits.map(|visit| match visit.attention {
            Attention::LinearAttention => {
                let owner = Owner::Linear(counts[0]);
                counts[0] += 1;
                owner
            }
            Attention::FullAttention => {
                let owner = Owner::Full(counts[1]);
                counts[1] += 1;
                owner
            }
        });
        if counts != [DELTA, FULL] {
            return Err(invalid("SenseNova state visit owner counts"));
        }
        let mut delta = reserved(DELTA)?;
        for _ in 0..DELTA {
            delta.push(DeltaState {
                rnn: Banks::new(layout.rnn)?,
                history: Banks::new(layout.conv)?,
            });
        }
        let mut kv = reserved(FULL)?;
        let words = layout.kv_words(options.capacity)?;
        for _ in 0..FULL {
            kv.push(Kv {
                k: model::zeros(words)?,
                v: model::zeros(words)?,
            });
        }
        let history = reserved(options.capacity)?;
        let staged = model::zeros(mul(options.chunk, layout.vocab)?)?;
        let logits = model::zeros(staged.len())?;
        memory.recurrent_bank_bytes = delta.iter().try_fold(0, |n, d| add(n, d.rnn.bytes()?))?;
        memory.convolution_bank_bytes = delta
            .iter()
            .try_fold(0, |n, d| add(n, d.history.bytes()?))?;
        memory.kv_bytes = kv.iter().try_fold(0, |n, kv| {
            add(n, mul(add(kv.k.capacity(), kv.v.capacity())?, 4)?)
        })?;
        memory.logits_bytes = mul(add(staged.capacity(), logits.capacity())?, 4)?;
        memory.history_bytes = mul(history.capacity(), 4)?;
        memory.metadata_bytes = Self::metadata_bytes(delta.capacity(), kv.capacity())?;
        memory.sum()?;
        memory.admit(options.memory_limit_bytes)?;
        Ok(Self {
            visits,
            owners,
            options,
            layout,
            delta,
            kv,
            history,
            staged,
            logits,
            published_rows: 0,
            memory,
        })
    }

    pub fn memory(&self) -> Memory {
        self.memory
    }

    pub fn history(&self) -> &[u32] {
        &self.history
    }

    pub fn logits(&self) -> Option<&[f32]> {
        let row = self.published_rows.checked_sub(1)?;
        let from = row * self.layout.vocab;
        Some(&self.logits[from..from + self.layout.vocab])
    }

    pub fn reset(&mut self) {
        for entry in &mut self.delta {
            entry.rnn.reset();
            entry.history.reset();
        }
        self.history.clear();
        self.published_rows = 0;
    }

    /// Return the reusable prefix length. A changed or shortened prompt resets
    /// the owner after admission, so the caller can replay from position zero.
    pub fn reuse_prefix(&mut self, tokens: &[u32]) -> Result<usize> {
        if tokens.len() > self.options.capacity {
            return Err(model::Error::Capacity {
                offset: 0,
                tokens: tokens.len(),
                limit: self.options.capacity,
            }
            .into());
        }
        self.validate_tokens(tokens)?;
        if !tokens.starts_with(&self.history) {
            self.reset();
        }
        Ok(self.history.len())
    }

    fn validate_tokens(&self, tokens: &[u32]) -> Result<()> {
        for (position, &token) in tokens.iter().enumerate() {
            if token as usize >= self.layout.vocab {
                return Err(cpu::Error::Token {
                    position,
                    token,
                    vocab_size: self.layout.vocab,
                }
                .into());
            }
        }
        Ok(())
    }

    /// Stage each token's complete fixed visit sequence and publish only after
    /// the whole chunk succeeds. The concrete text forward writes each loaned
    /// state and the complete logits row before returning success.
    pub fn evaluate(
        &mut self,
        tokens: &[u32],
        cancel: &AtomicBool,
        mut forward: impl FnMut(Token, &mut Frame<'_>, &mut [f32]) -> Result<()>,
    ) -> Result<&[f32]> {
        if tokens.is_empty() {
            return Err(invalid("SenseNova state chunk must be nonempty"));
        }
        if tokens.len() > self.options.chunk {
            return Err(model::Error::ScratchCapacity {
                tokens: tokens.len(),
                limit: self.options.chunk,
            }
            .into());
        }
        let offset = self.history.len();
        if tokens.len() > self.options.capacity - offset {
            return Err(model::Error::Capacity {
                offset,
                tokens: tokens.len(),
                limit: self.options.capacity,
            }
            .into());
        }
        self.validate_tokens(tokens)?;
        for (local, &id) in tokens.iter().enumerate() {
            check_cancel(cancel)?;
            let position = offset + local;
            let token = Token {
                id,
                position,
                absolute: self.options.position_offset + position,
            };
            let mut frame = Frame {
                visits: &self.visits,
                owners: &self.owners,
                delta: &mut self.delta,
                kv: &mut self.kv,
                cancel,
                next: 0,
                local,
                position,
                stride: self.options.capacity * self.layout.head_dim,
                head_dim: self.layout.head_dim,
            };
            let start = local * self.layout.vocab;
            let logits = &mut self.staged[start..start + self.layout.vocab];
            logits.fill(f32::NAN);
            forward(token, &mut frame, logits)?;
            if frame.next != EXECUTION_VISITS {
                return Err(invalid("SenseNova state incomplete visit sequence"));
            }
            cpu::finite("SenseNova staged logits", logits)?;
        }
        check_cancel(cancel)?;
        let last = (tokens.len() - 1) % 2;
        for entry in &mut self.delta {
            entry.rnn.commit(last);
            entry.history.commit(last);
        }
        self.history.extend_from_slice(tokens);
        std::mem::swap(&mut self.staged, &mut self.logits);
        self.published_rows = tokens.len();
        Ok(&self.logits[..tokens.len() * self.layout.vocab])
    }
}

#[cfg(test)]
mod tests;
