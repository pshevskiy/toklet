use super::{
    Buffer, Error, HIDDEN, INTERMEDIATE, KV_WIDTH, LOGIT_ROWS, Model, Options, QUERY_WIDTH, Result,
    Sizes,
    kernel::{Encoder, Op, RawBuffer, k64_input_aligned},
    plan::{self, Matrix, Projection},
};
use crate::{
    cpu,
    session::{Error as SessionError, InferenceSession, Memory},
};
use half::bf16;
use objc2::rc::autoreleasepool;
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLDevice};
use std::{
    cell::Cell,
    sync::atomic::{AtomicBool, Ordering},
};

struct Kv {
    k: Buffer<u16>,
    v: Buffer<u16>,
}

struct Scratch {
    tokens: Buffer<u32>,
    status: Buffer<u32>,
    selected: Buffer<u32>,
    residual: Buffer<u16>,
    normalized: Buffer<u16>,
    delta: Buffer<u16>,
    query: Buffer<u16>,
    key: Buffer<u16>,
    value: Buffer<u16>,
    attention: Buffer<u16>,
    gate: Buffer<u16>,
    up: Buffer<u16>,
    logits: Buffer<u16>,
}

struct Diagnostic {
    staged_dense: Cell<bool>,
    last_dense: bool,
    staged_passes: Buffer<u16>,
    published_passes: Buffer<u16>,
    logits: Vec<f32>,
    rows: usize,
}

pub struct Session<'a> {
    model: &'a Model,
    options: Options,
    memory: Memory,
    cancelled: &'a AtomicBool,
    history: Vec<u32>,
    evaluated: usize,
    selected: Option<u32>,
    replay_required: bool,
    kv: Kv,
    scratch: Scratch,
    down_k64_ready: bool,
    attention_compact_ready: bool,
    fused_gate_up_ready: bool,
    diagnostic: Option<Diagnostic>,
}

impl<'a> Session<'a> {
    /// Diagnostic sessions additionally retain a full BF16-rounded logit row
    /// and both normalized residual panels from the last successful command.
    pub fn new(
        model: &'a Model,
        options: Options,
        diagnostics: bool,
        cancelled: &'a AtomicBool,
    ) -> Result<Self> {
        let mut sizes = Sizes::new(options, diagnostics)?;
        if let Some(dense) = &model.dense {
            let extra = dense.receipt.allocated_bytes;
            sizes.memory.weights_bytes = super::add(sizes.memory.weights_bytes, extra)?;
            sizes.memory.admitted_bytes = super::add(sizes.memory.admitted_bytes, extra)?;
            if sizes.memory.admitted_bytes > options.memory_limit_bytes {
                return Err(Error::Budget {
                    required: sizes.memory.admitted_bytes,
                    limit: options.memory_limit_bytes,
                });
            }
            let device = &model.context.device;
            let recommended = usize::try_from(device.recommendedMaxWorkingSetSize())
                .unwrap_or(usize::MAX)
                .min(options.memory_limit_bytes);
            let session_bytes = sizes.memory.admitted_bytes - sizes.memory.weights_bytes;
            let required = super::add(device.currentAllocatedSize(), session_bytes)?;
            if required > recommended {
                return Err(Error::Budget {
                    required,
                    limit: recommended,
                });
            }
        }
        sizes.validate_device(&model.context, model.plan.embedding.layout.weight_bytes())?;
        let chunk = options.prefill_chunk_size;
        let device = &model.context.device;
        let mut history = Vec::new();
        history
            .try_reserve_exact(options.capacity)
            .map_err(|_| Error::Native("Nanbeige token history allocation failed".into()))?;
        let mut memory = sizes.memory;
        memory.history_bytes = history.capacity() * 4;
        memory.admitted_bytes = super::add(
            memory.admitted_bytes - sizes.memory.history_bytes,
            memory.history_bytes,
        )?;
        if memory.admitted_bytes > options.memory_limit_bytes {
            return Err(Error::Budget {
                required: memory.admitted_bytes,
                limit: options.memory_limit_bytes,
            });
        }
        autoreleasepool(|_| {
            let words = |n| Buffer::<u16>::on_device(device, n);
            let diagnostic = if diagnostics {
                let mut logits = Vec::new();
                logits.try_reserve_exact(LOGIT_ROWS).map_err(|_| {
                    Error::Native("Nanbeige diagnostic logits allocation failed".into())
                })?;
                logits.resize(LOGIT_ROWS, 0.0);
                memory.admitted_bytes =
                    super::add(memory.admitted_bytes, (logits.capacity() - LOGIT_ROWS) * 4)?;
                memory.logits_bytes = (logits.capacity() * 4) + (LOGIT_ROWS * 2);
                if memory.admitted_bytes > options.memory_limit_bytes {
                    return Err(Error::Budget {
                        required: memory.admitted_bytes,
                        limit: options.memory_limit_bytes,
                    });
                }
                Some(Diagnostic {
                    staged_dense: Cell::new(false),
                    last_dense: false,
                    staged_passes: words(sizes.pass_elements)?,
                    published_passes: words(sizes.pass_elements)?,
                    logits,
                    rows: 0,
                })
            } else {
                None
            };
            let kv = Kv {
                k: words(sizes.kv_elements)?,
                v: words(sizes.kv_elements)?,
            };
            let scratch = Scratch {
                tokens: Buffer::on_device(device, chunk)?,
                status: Buffer::on_device(device, 1)?,
                selected: Buffer::on_device(device, 1)?,
                residual: words(chunk * HIDDEN)?,
                normalized: words(chunk * HIDDEN)?,
                delta: words(chunk * HIDDEN)?,
                query: words(chunk * QUERY_WIDTH)?,
                key: words(chunk * KV_WIDTH)?,
                value: words(chunk * KV_WIDTH)?,
                attention: words(chunk * QUERY_WIDTH)?,
                gate: words(chunk * INTERMEDIATE)?,
                up: words(chunk * INTERMEDIATE)?,
                logits: words(LOGIT_ROWS)?,
            };
            let k64_ready = model.down_k64_prefill_rows().contains(&chunk);
            let down_k64_ready = k64_ready && k64_input_aligned(&scratch.gate.raw);
            let attention_compact_ready = k64_ready
                && model.gate_up_compact_prefill_rows().contains(&chunk)
                && k64_input_aligned(&scratch.attention.raw);
            let fused_gate_up_ready =
                chunk == 128 && model.dense.is_some() && k64_input_aligned(&scratch.normalized.raw);
            Ok(Self {
                model,
                options,
                memory,
                cancelled,
                history,
                evaluated: 0,
                selected: None,
                replay_required: false,
                kv,
                scratch,
                down_k64_ready,
                attention_compact_ready,
                fused_gate_up_ready,
                diagnostic,
            })
        })
    }

    /// Prepared K64 Down rows with this session's aligned SwiGLU scratch.
    pub fn down_k64_prefill_rows(&self) -> &'static [usize] {
        if self.down_k64_ready {
            self.model.down_k64_prefill_rows()
        } else {
            &[]
        }
    }

    /// Active prefill row counts eligible for the paired compact Q and O projections.
    pub fn attention_compact_prefill_rows(&self) -> &'static [usize] {
        if self.attention_compact_ready {
            &[128]
        } else {
            &[]
        }
    }

    /// Actual route of the last successfully published diagnostic chunk.
    #[doc(hidden)]
    pub fn last_chunk_used_dense(&self) -> Option<bool> {
        self.selected?;
        self.diagnostic
            .as_ref()
            .map(|diagnostic| diagnostic.last_dense)
    }

    /// Aligned T128 dense route availability. The final pending chunk stays packed.
    pub fn dense_gate_up_prefill_rows(&self) -> &'static [usize] {
        if self.fused_gate_up_ready {
            &[128]
        } else {
            &[]
        }
    }

    /// Copy one logical layer's committed, head-major BF16 K/V prefix into
    /// exact [8, evaluated_len, 128] output arrays. Diagnostics must be enabled.
    pub fn copy_kv_bf16(&self, slot: usize, k: &mut [u16], v: &mut [u16]) -> Result<()> {
        if self.diagnostic.is_none()
            || self.selected.is_none()
            || slot >= plan::VISITS
            || k.len() != self.evaluated * KV_WIDTH
            || v.len() != k.len()
        {
            return Err(Error::Options("Nanbeige diagnostic KV request differs"));
        }
        let stride = self.options.capacity * plan::HEAD_DIM;
        let active = self.evaluated * plan::HEAD_DIM;
        for (bank, output) in [(&self.kv.k, k), (&self.kv.v, v)] {
            bank.with_slice(|storage| {
                for (head, target) in output.chunks_exact_mut(active).enumerate() {
                    let from = (slot * plan::KV_HEADS + head) * stride;
                    target.copy_from_slice(&storage[from..from + active]);
                }
            });
        }
        Ok(())
    }

    /// Copy one pass's post-norm residual [last successful chunk rows, hidden].
    pub fn copy_pass_bf16(&self, pass: usize, output: &mut [u16]) -> Result<()> {
        let diagnostic = self
            .diagnostic
            .as_ref()
            .filter(|_| self.selected.is_some())
            .ok_or(Error::Options("Nanbeige diagnostic pass is unavailable"))?;
        if pass >= plan::PASSES || output.len() != diagnostic.rows * HIDDEN {
            return Err(Error::Options("Nanbeige diagnostic pass shape differs"));
        }
        let start = pass * self.options.prefill_chunk_size * HIDDEN;
        diagnostic.published_passes.with_slice(|storage| {
            output.copy_from_slice(&storage[start..start + output.len()]);
        });
        Ok(())
    }

    fn evaluate(&mut self, limit: usize) -> std::result::Result<usize, SessionError> {
        if self.replay_required {
            return Err(backend_error(Error::ReplayRequired));
        }
        let rows = self.pending_len().min(limit);
        if rows == 0 {
            return Ok(0);
        }
        if self.cancelled.load(Ordering::Relaxed) {
            return Err(crate::model::Error::Cancelled.into());
        }
        let start = self.evaluated;
        self.scratch.tokens.with_mut(|output| {
            output[..rows].copy_from_slice(&self.history[start..start + rows]);
        });
        self.scratch.status.with_mut(|status| status[0] = 0);
        let result = autoreleasepool(|_| self.submit(rows, start));
        self.finish_chunk(rows, result)
    }

    fn finish_chunk(
        &mut self,
        rows: usize,
        result: Result<()>,
    ) -> std::result::Result<usize, SessionError> {
        if matches!(&result, Err(Error::Completion)) {
            self.replay_required = true;
            self.evaluated = 0;
            self.selected = None;
        }
        result.map_err(backend_error)?;
        if self.cancelled.load(Ordering::Relaxed) {
            return Err(crate::model::Error::Cancelled.into());
        }
        if self.scratch.status.with_slice(|words| words[0] != 0) {
            return Err(backend_error(Error::Nonfinite));
        }
        let token = self.scratch.selected.with_slice(|words| words[0]);
        if token as usize >= LOGIT_ROWS {
            return Err(backend_error(Error::Options(
                "Nanbeige GPU selection outside vocabulary",
            )));
        }
        if let Some(diagnostic) = &mut self.diagnostic {
            self.scratch.logits.with_slice(|words| {
                for (&word, output) in words.iter().zip(&mut diagnostic.logits) {
                    *output = bf16::from_bits(word).to_f32();
                }
            });
            std::mem::swap(
                &mut diagnostic.staged_passes,
                &mut diagnostic.published_passes,
            );
            diagnostic.rows = rows;
            diagnostic.last_dense = rows == 128 && diagnostic.staged_dense.get();
        }
        // Before this point, speculative writes occupy only the private tail
        // [evaluated, evaluated+rows) of all 44 caches. Publishing the common
        // length makes every visit visible together.
        self.evaluated += rows;
        self.selected = Some(token);
        Ok(rows)
    }

    fn submit(&self, rows: usize, start: usize) -> Result<()> {
        let context = &self.model.context;
        let command = context.command()?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::Native("Nanbeige compute encoder absent".into()))?;
        self.encode(&encoder, rows, start);
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Completion);
        }
        Ok(())
    }

    fn encode(&self, encoder: &Encoder, rows: usize, start: usize) {
        if rows == 128 {
            let dense = self.fused_gate_up_ready && self.pending_len() > rows;
            if let Some(diagnostic) = &self.diagnostic {
                diagnostic.staged_dense.set(dense);
            }
            if dense {
                self.encode_graph::<true>(encoder, rows, start);
                return;
            }
        }
        self.encode_graph::<false>(encoder, rows, start);
    }

    fn encode_graph<const FUSED_GATE_UP: bool>(
        &self,
        encoder: &Encoder,
        rows: usize,
        start: usize,
    ) {
        let model = self.model;
        let context = &model.context;
        let dense = if FUSED_GATE_UP {
            Some(model.dense.as_ref().expect("admitted dense prefill pair"))
        } else {
            None
        };
        let s = &self.scratch;
        let dispatch = |op, buffers: &[&RawBuffer], shape: &[u32], groups| {
            context.encode(encoder, op, buffers, shape, groups);
        };
        let weights = |id: usize| &*model.weights[id].raw;
        let ordinary = match Projection::for_chunk(rows) {
            Projection::GroupedVector => (Op::Vector, 1, 4),
            Projection::TiledBf16 => (Op::Tiled, 16, 32),
        };
        let (query, attention_out) = if self.attention_compact_prefill_rows().contains(&rows) {
            ((Op::GateUpCompact, 32, 32), (Op::DownK64, 32, 32))
        } else {
            (ordinary, ordinary)
        };
        let ffn = if model.ffn_m32_prefill_rows().contains(&rows) {
            (Op::FfnM32, 32, 32)
        } else {
            ordinary
        };
        let gate_up = if model.gate_up_compact_prefill_rows().contains(&rows) {
            (Op::GateUpCompact, 32, 32)
        } else {
            ffn
        };
        let down = if self.down_k64_prefill_rows().contains(&rows) {
            (Op::DownK64, 32, 32)
        } else {
            ffn
        };
        let affine = |(op, m_tile, n_tile): (Op, usize, usize),
                      matrix: Matrix,
                      input: &RawBuffer,
                      output: &RawBuffer,
                      count: usize,
                      input_row| {
            let columns = matrix.layout.columns();
            let outputs = matrix.layout.rows();
            dispatch(
                op,
                &[
                    weights(matrix.weights),
                    weights(matrix.scales),
                    weights(matrix.biases),
                    input,
                    output,
                    &s.status.raw,
                ],
                &[count as u32, columns as u32, outputs as u32, input_row],
                [outputs / n_tile, count.div_ceil(m_tile)],
            );
        };
        let norm = |input: &RawBuffer, gain: usize, output: &RawBuffer| {
            dispatch(
                Op::Rms,
                &[input, weights(gain), output, &s.status.raw],
                &[rows as u32, HIDDEN as u32],
                [rows, 1],
            );
        };
        let residual_add = || {
            dispatch(
                Op::Add,
                &[&s.residual.raw, &s.delta.raw, &s.status.raw],
                &[rows as u32, HIDDEN as u32],
                [(rows * HIDDEN).div_ceil(32), 1],
            );
        };
        let embedding = model.plan.embedding;
        dispatch(
            Op::Embedding,
            &[
                weights(embedding.weights),
                weights(embedding.scales),
                weights(embedding.biases),
                &s.tokens.raw,
                &s.residual.raw,
                &s.status.raw,
            ],
            &[rows as u32, HIDDEN as u32, LOGIT_ROWS as u32, 0],
            [(rows * HIDDEN).div_ceil(32), 1],
        );

        for visit in model.visits {
            let block = &model.plan.blocks[visit.physical_layer];
            let cache_shape = [
                rows as u32,
                start as u32,
                self.options.capacity as u32,
                visit.kv_slot as u32,
                self.options.position_offset as u32,
            ];
            norm(&s.residual.raw, block.input_norm, &s.normalized.raw);
            for (projection, matrix, target) in [
                (query, block.query, &s.query),
                (ordinary, block.key, &s.key),
                (ordinary, block.value, &s.value),
            ] {
                affine(projection, matrix, &s.normalized.raw, &target.raw, rows, 0);
            }
            dispatch(
                Op::RotateCache,
                &[
                    &s.query.raw,
                    &s.key.raw,
                    &s.value.raw,
                    &self.kv.k.raw,
                    &self.kv.v.raw,
                    &s.status.raw,
                ],
                &cache_shape,
                [(rows * ((QUERY_WIDTH + KV_WIDTH) / 2)).div_ceil(32), 1],
            );
            dispatch(
                Op::Attention,
                &[
                    &s.query.raw,
                    &self.kv.k.raw,
                    &self.kv.v.raw,
                    &s.attention.raw,
                    &s.status.raw,
                ],
                &cache_shape,
                [rows * plan::QUERY_HEADS, 1],
            );
            affine(
                attention_out,
                block.attention_out,
                &s.attention.raw,
                &s.delta.raw,
                rows,
                0,
            );
            residual_add();
            norm(
                &s.residual.raw,
                block.post_attention_norm,
                &s.normalized.raw,
            );
            if let Some(dense) = dense {
                dense.encode(
                    encoder,
                    visit.physical_layer,
                    &s.normalized.raw,
                    &s.gate.raw,
                    &s.status.raw,
                );
            } else {
                affine(gate_up, block.gate, &s.normalized.raw, &s.gate.raw, rows, 0);
                affine(gate_up, block.up, &s.normalized.raw, &s.up.raw, rows, 0);
                dispatch(
                    Op::Swiglu,
                    &[&s.gate.raw, &s.up.raw, &s.status.raw],
                    &[rows as u32, INTERMEDIATE as u32],
                    [(rows * INTERMEDIATE).div_ceil(32), 1],
                );
            }
            affine(down, block.down, &s.gate.raw, &s.delta.raw, rows, 0);
            residual_add();

            if visit.normalize_after {
                norm(&s.residual.raw, model.plan.final_norm, &s.normalized.raw);
                // The first loop's normalized output feeds loop two. The
                // second loop's normalized output feeds the untied head.
                if visit.kv_slot + 1 < plan::VISITS {
                    dispatch(
                        Op::Copy,
                        &[&s.normalized.raw, &s.residual.raw],
                        &[0, (rows * HIDDEN) as u32],
                        [(rows * HIDDEN).div_ceil(32), 1],
                    );
                }
                if let Some(diagnostic) = &self.diagnostic {
                    let pass = visit.kv_slot / plan::PHYSICAL_LAYERS;
                    dispatch(
                        Op::Copy,
                        &[&s.normalized.raw, &diagnostic.staged_passes.raw],
                        &[
                            (pass * self.options.prefill_chunk_size * HIDDEN) as u32,
                            (rows * HIDDEN) as u32,
                        ],
                        [(rows * HIDDEN).div_ceil(32), 1],
                    );
                }
            }
        }
        affine(
            ordinary,
            model.plan.head,
            &s.normalized.raw,
            &s.logits.raw,
            1,
            (rows - 1) as u32,
        );
        dispatch(
            Op::Greedy,
            &[&s.logits.raw, &s.selected.raw],
            &[LOGIT_ROWS as u32],
            [1, 1],
        );
    }

    fn validate_tokens(
        &self,
        prefix: usize,
        tokens: &[u32],
    ) -> std::result::Result<(), SessionError> {
        if self.replay_required {
            return Err(backend_error(Error::ReplayRequired));
        }
        if tokens.len() > self.options.capacity - prefix
            || tokens.iter().any(|&token| token as usize >= LOGIT_ROWS)
        {
            return Err(SessionError::InvalidOptions(
                "Nanbeige token or capacity exceeds bounds",
            ));
        }
        Ok(())
    }
}

fn backend_error(error: Error) -> SessionError {
    SessionError::Backend(error.to_string())
}

impl InferenceSession for Session<'_> {
    fn history(&self) -> &[u32] {
        &self.history
    }

    fn evaluated_len(&self) -> usize {
        self.evaluated
    }

    fn pending_len(&self) -> usize {
        self.history.len() - self.evaluated
    }

    fn capacity(&self) -> usize {
        self.options.capacity
    }

    fn vocab_size(&self) -> usize {
        LOGIT_ROWS
    }

    fn memory(&self) -> Memory {
        self.memory
    }

    fn logits(&self) -> Option<&[f32]> {
        self.selected?;
        self.diagnostic
            .as_ref()
            .map(|diagnostic| diagnostic.logits.as_slice())
    }

    fn greedy_token(&self) -> Option<std::result::Result<u32, cpu::Error>> {
        self.selected.map(Ok)
    }

    fn append(&mut self, tokens: &[u32]) -> std::result::Result<(), SessionError> {
        self.validate_tokens(self.history.len(), tokens)?;
        self.history.extend_from_slice(tokens);
        Ok(())
    }

    fn set_tokens(&mut self, tokens: &[u32]) -> std::result::Result<(), SessionError> {
        self.validate_tokens(0, tokens)?;
        // A new prompt has its own chunk topology. Append and Continue
        // explicitly preserve an existing prefix and its execution history.
        self.reset();
        self.history.extend_from_slice(tokens);
        Ok(())
    }

    fn evaluate_next_chunk(&mut self) -> std::result::Result<usize, SessionError> {
        self.evaluate(self.options.prefill_chunk_size)
    }

    fn evaluate_next_decode_chunk(&mut self) -> std::result::Result<usize, SessionError> {
        self.evaluate(1)
    }

    fn discard_pending(&mut self) {
        if !self.replay_required {
            self.history.truncate(self.evaluated);
        }
    }

    fn reset(&mut self) {
        self.history.clear();
        self.evaluated = 0;
        self.selected = None;
        self.replay_required = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Explicit Apple GPU and pinned local Nanbeige weight shard"]
    fn published_state_survives_a_cancelled_gpu_tail_and_requires_reset_after_failed_completion() {
        let directory = std::env::var_os("TOKLET_NANBEIGE_MODEL").expect("TOKLET_NANBEIGE_MODEL");
        let options = Options {
            capacity: 4,
            prefill_chunk_size: 2,
            ..Options::default()
        };
        let model = Model::load(std::path::Path::new(&directory), options, true).unwrap();
        let cancelled = AtomicBool::new(false);
        let mut session = Session::new(&model, options, true, &cancelled).unwrap();
        session.append(&[375, 13]).unwrap();
        assert_eq!(session.evaluate_next_chunk().unwrap(), 2);
        let logits = session.logits().unwrap().to_vec();
        let mut prefix = vec![0; 2 * KV_WIDTH];
        let mut values = prefix.clone();
        session
            .copy_kv_bf16(plan::VISITS - 1, &mut prefix, &mut values)
            .unwrap();

        session.append(&[374]).unwrap();
        session.scratch.tokens.with_mut(|tokens| tokens[0] = 374);
        session.scratch.status.with_mut(|status| status[0] = 0);
        autoreleasepool(|_| session.submit(1, 2)).unwrap();
        cancelled.store(true, Ordering::Relaxed);
        assert!(matches!(
            session.finish_chunk(1, Ok(())),
            Err(SessionError::Model(crate::model::Error::Cancelled))
        ));
        assert_eq!(session.evaluated_len(), 2);
        assert_eq!(session.pending_len(), 1);
        assert_eq!(session.logits().unwrap(), logits);
        let mut k = vec![0; 2 * KV_WIDTH];
        let mut v = k.clone();
        session
            .copy_kv_bf16(plan::VISITS - 1, &mut k, &mut v)
            .unwrap();
        assert_eq!(k, prefix);
        assert_eq!(v, values);
        cancelled.store(false, Ordering::Relaxed);
        assert_eq!(session.evaluate_next_decode_chunk().unwrap(), 1);
        assert_eq!(session.evaluated_len(), 3);

        assert!(session.finish_chunk(1, Err(Error::Completion)).is_err());
        assert!(session.logits().is_none());
        assert_eq!(session.evaluated_len(), 0);
        assert!(session.append(&[13]).is_err());
        assert!(session.evaluate_next_decode_chunk().is_err());
        session.reset();
        session.append(&[375, 13]).unwrap();
        assert_eq!(session.evaluate_next_chunk().unwrap(), 2);
        assert_eq!(session.logits().unwrap(), logits);
    }
}

#[cfg(test)]
#[path = "dense_fault_tests.rs"]
mod dense_fault_tests;
