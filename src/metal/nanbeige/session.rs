use super::{
    Buffer, Error, HIDDEN, INTERMEDIATE, KV_WIDTH, LOGIT_ROWS, Model, Options, QUERY_WIDTH, Result,
    Sizes,
    kernel::{Encoder, Op, RawBuffer, k64_input_aligned},
    plan::{self, Matrix, Projection},
    qmm, split_k,
    steel::{OUTPUT as FIXED_STEEL_OUTPUT, PROMPT as FIXED_STEEL_PROMPT, Steel},
};
use crate::{
    cpu,
    generation::{self, Input, Report, Request, TokenEvent},
    session::{Error as SessionError, InferenceSession, Memory},
};
use half::bf16;
use objc2::rc::autoreleasepool;
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLDevice};
use std::{
    cell::Cell,
    io,
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
    split_partials: Option<Buffer<u32>>,
}

struct Diagnostic {
    staged_dense: Cell<bool>,
    last_dense: bool,
    staged_qmm: Cell<bool>,
    last_qmm: bool,
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
    fixed_steel_request: bool,
    kv: Kv,
    scratch: Scratch,
    down_k64_ready: bool,
    attention_compact_ready: bool,
    fused_gate_up_ready: bool,
    qmm_ready: bool,
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
        let steel = model.steel.as_ref();
        if steel.is_some() {
            Steel::validate_options(options)?;
            if diagnostics {
                return Err(Error::Unsupported(
                    "Nanbeige Steel requires a one-shot session",
                ));
            }
            sizes.memory.scratch_bytes =
                super::add(sizes.memory.scratch_bytes, split_k::PARTIAL_BYTES)?;
            sizes.memory.admitted_bytes =
                super::add(sizes.memory.admitted_bytes, split_k::PARTIAL_BYTES)?;
        }
        if let Some(dense) = &model.dense {
            let qmm = model.qmm_receipt();
            let extra = super::add(
                dense.receipt.allocated_bytes,
                super::add(
                    qmm.map_or(0, |q| q.reserved_bytes),
                    steel.map_or(0, |p| p.reserved_bytes),
                )?,
            )?;
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
            let unallocated = super::add(
                qmm.map_or(0, |q| q.reserved_bytes - q.physical_delta_bytes),
                steel.map_or(0, |p| p.reserved_bytes - p.physical_bytes),
            )?;
            let required = super::add(
                device.currentAllocatedSize(),
                super::add(session_bytes, unallocated)?,
            )?;
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
                    staged_qmm: Cell::new(false),
                    last_qmm: false,
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
                split_partials: steel
                    .map(|_| Buffer::on_device(device, split_k::PARTIAL_WORDS))
                    .transpose()?,
            };
            let k64_ready = model.down_k64_prefill_rows().contains(&chunk);
            let down_k64_ready = k64_ready && k64_input_aligned(&scratch.gate.raw);
            let attention_compact_ready = k64_ready
                && model.gate_up_compact_prefill_rows().contains(&chunk)
                && k64_input_aligned(&scratch.attention.raw);
            let fused_gate_up_ready =
                chunk == 128 && model.dense.is_some() && k64_input_aligned(&scratch.normalized.raw);
            let qmm_ready = chunk == 128
                && model.qmm.is_some()
                && fused_gate_up_ready
                && attention_compact_ready
                && down_k64_ready
                && qmm::disjoint_exact_ranges([
                    (&scratch.normalized.raw, 128 * HIDDEN * 2),
                    (&scratch.query.raw, 128 * QUERY_WIDTH * 2),
                    (&scratch.attention.raw, 128 * QUERY_WIDTH * 2),
                    (&scratch.delta.raw, 128 * HIDDEN * 2),
                    (&scratch.gate.raw, 128 * INTERMEDIATE * 2),
                    (&scratch.status.raw, 4),
                ]);
            if steel.is_some()
                && (!qmm_ready
                    || !qmm::disjoint_exact_ranges([
                        (&scratch.normalized.raw, 128 * HIDDEN * 2),
                        (&scratch.up.raw, 128 * INTERMEDIATE * 2),
                        (&scratch.gate.raw, 128 * INTERMEDIATE * 2),
                        (&scratch.status.raw, 4),
                    ]))
            {
                return Err(Error::Unsupported(
                    "Nanbeige Steel dense prepack scratch layout",
                ));
            }
            if let Some(partials) = &scratch.split_partials
                && (!qmm_ready
                    || !qmm::disjoint_exact_ranges([
                        (&scratch.normalized.raw, split_k::ROWS * HIDDEN * 2),
                        (&scratch.key.raw, split_k::ROWS * KV_WIDTH * 2),
                        (&scratch.value.raw, split_k::ROWS * KV_WIDTH * 2),
                        (&partials.raw, split_k::PARTIAL_BYTES),
                        (&scratch.status.raw, 4),
                    ]))
            {
                return Err(Error::Unsupported("Nanbeige Steel split-K scratch layout"));
            }
            if steel.is_some() {
                Steel::validate_banks([
                    &scratch.query.raw,
                    &scratch.key.raw,
                    &scratch.value.raw,
                    &kv.k.raw,
                    &kv.v.raw,
                    &scratch.attention.raw,
                    &scratch.status.raw,
                ])?;
                Steel::validate_rms(
                    model,
                    [
                        &scratch.residual.raw,
                        &scratch.normalized.raw,
                        &scratch.status.raw,
                    ],
                )?;
            }
            let session = Self {
                model,
                options,
                memory,
                cancelled,
                history,
                evaluated: 0,
                selected: None,
                replay_required: false,
                fixed_steel_request: false,
                kv,
                scratch,
                down_k64_ready,
                attention_compact_ready,
                fused_gate_up_ready,
                qmm_ready,
                diagnostic,
            };
            if steel.is_some() {
                session.validate_wheel_t1_affines()?;
                Steel::validate_head(
                    model,
                    [
                        &session.scratch.normalized.raw,
                        &session.scratch.logits.raw,
                        &session.scratch.status.raw,
                    ],
                )?;
            }
            Ok(session)
        })
    }

    fn validate_wheel_t1_affines(&self) -> Result<()> {
        let s = &self.scratch;
        let check = |matrix, input: &RawBuffer, output: &RawBuffer, rows| {
            Steel::validate_t1_affine(self.model, matrix, [input, output, &s.status.raw], rows)
        };
        for block in &self.model.plan.blocks {
            for (matrix, input, output) in [
                (block.query, &s.normalized.raw, &s.query.raw),
                (block.key, &s.normalized.raw, &s.key.raw),
                (block.value, &s.normalized.raw, &s.value.raw),
                (block.attention_out, &s.attention.raw, &s.delta.raw),
                (block.gate, &s.normalized.raw, &s.gate.raw),
                (block.up, &s.normalized.raw, &s.up.raw),
                (block.down, &s.gate.raw, &s.delta.raw),
            ] {
                check(matrix, input, output, split_k::ROWS)?;
            }
        }
        check(self.model.plan.head, &s.normalized.raw, &s.logits.raw, 1)
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

    /// Actual aligned T128 eligibility for this Session's own scratch.
    pub fn qmm_prefill_rows(&self) -> &'static [usize] {
        if self.qmm_ready { &[128] } else { &[] }
    }

    /// QMM route of the most recent successfully committed diagnostic chunk.
    pub fn last_chunk_used_qmm(&self) -> Option<bool> {
        self.selected?;
        self.diagnostic
            .as_ref()
            .map(|diagnostic| diagnostic.last_qmm)
    }

    /// Aligned T128 dense route availability. Only fixed Steel covers the final pending chunk.
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

    /// Consume a fresh session on the exact experimental Steel workload.
    pub fn generate_fixed_steel(
        mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        stop_tokens: &[u32],
        emit: impl FnMut(TokenEvent) -> io::Result<()>,
    ) -> std::result::Result<Report, generation::Error> {
        if self.model.steel.is_none()
            || self.replay_required
            || !self.history.is_empty()
            || prompt.len() != FIXED_STEEL_PROMPT
            || max_new_tokens != FIXED_STEEL_OUTPUT
        {
            return Err(generation::Error::InvalidRequest(
                "Nanbeige Steel requires a fresh 512-input/32-output one-shot session",
            ));
        }
        self.fixed_steel_request = true;
        let cancelled = self.cancelled;
        generation::generate(
            &mut self,
            Request {
                input: Input::Prompt(prompt),
                max_new_tokens,
                stop_tokens,
            },
            cancelled,
            emit,
        )
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
        let score = !self.fixed_steel_request || rows == self.pending_len();
        let result = autoreleasepool(|_| self.submit(rows, start, score));
        self.finish_chunk(rows, result, score)
    }

    fn finish_chunk(
        &mut self,
        rows: usize,
        result: Result<()>,
        score: bool,
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
        let token = score.then(|| self.scratch.selected.with_slice(|words| words[0]));
        if token.is_some_and(|id| id as usize >= LOGIT_ROWS) {
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
            diagnostic.last_qmm = rows == 128 && diagnostic.staged_qmm.get();
        }
        // Before this point, speculative writes occupy only the private tail
        // [evaluated, evaluated+rows) of all 44 caches. Publishing the common
        // length makes every visit visible together.
        self.evaluated += rows;
        self.selected = token;
        Ok(rows)
    }

    fn submit(&self, rows: usize, start: usize, score: bool) -> Result<()> {
        if self.fixed_steel_request
            && !((rows == split_k::ROWS && matches!(start, 0 | 128 | 256 | 384))
                || (rows == 1 && Steel::supports_t1_attention(start)))
        {
            return Err(Error::Unsupported("Nanbeige Steel committed cut"));
        }
        let context = &self.model.context;
        let command = context.command()?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::Native("Nanbeige compute encoder absent".into()))?;
        self.encode(&encoder, rows, start, score);
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Completion);
        }
        Ok(())
    }

    fn encode(&self, encoder: &Encoder, rows: usize, start: usize, score: bool) {
        if rows == 128 {
            let pending = self.pending_len();
            let dense_tail = self.fixed_steel_request
                && score
                && start == FIXED_STEEL_PROMPT - split_k::ROWS
                && pending == rows;
            let dense = self.fused_gate_up_ready && (pending > rows || dense_tail);
            if let Some(diagnostic) = &self.diagnostic {
                diagnostic.staged_dense.set(dense);
                diagnostic.staged_qmm.set(self.qmm_ready);
            }
            if self.qmm_ready {
                if dense {
                    self.encode_graph::<true, true>(encoder, rows, start, score);
                } else {
                    self.encode_graph::<false, true>(encoder, rows, start, score);
                }
                return;
            }
            if dense {
                self.encode_graph::<true, false>(encoder, rows, start, score);
                return;
            }
        }
        self.encode_graph::<false, false>(encoder, rows, start, score);
    }

    fn encode_graph<const FUSED_GATE_UP: bool, const USE_QMM: bool>(
        &self,
        encoder: &Encoder,
        rows: usize,
        start: usize,
        score: bool,
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
        let wheel = model.steel.as_ref().filter(|_| self.fixed_steel_request);
        let decode = wheel.filter(|_| rows == 1);
        let steel = wheel.filter(|_| rows == split_k::ROWS);
        let affine = |(op, m_tile, n_tile): (Op, usize, usize),
                      matrix: Matrix,
                      input: &RawBuffer,
                      output: &RawBuffer,
                      count: usize,
                      input_row| {
            let columns = matrix.layout.columns();
            let outputs = matrix.layout.rows();
            let banks = [
                weights(matrix.weights),
                weights(matrix.scales),
                weights(matrix.biases),
                input,
                output,
            ];
            if let Some(wheel) = decode {
                debug_assert_eq!((count, input_row), (1, 0));
                wheel.encode_affine(encoder, banks, columns, outputs, 0);
                model
                    .qmm
                    .as_ref()
                    .expect("Nanbeige Steel T1 affine finite scan")
                    .scan_output(encoder, &[output, &s.status.raw], outputs);
                return;
            }
            dispatch(
                op,
                &[banks[0], banks[1], banks[2], input, output, &s.status.raw],
                &[count as u32, columns as u32, outputs as u32, input_row],
                [outputs / n_tile, count.div_ceil(m_tile)],
            );
        };
        let project = |projection, matrix: Matrix, input: &RawBuffer, output: &RawBuffer| {
            if USE_QMM {
                let qmm = model.qmm.as_ref().expect("admitted QMM pipeline");
                let buffers = [
                    weights(matrix.weights),
                    weights(matrix.scales),
                    weights(matrix.biases),
                    input,
                    output,
                    &s.status.raw,
                ];
                let k = matrix.layout.columns();
                let n = matrix.layout.rows();
                if let Some(steel) = steel {
                    qmm.encode_selected(encoder, buffers, k, n, Some(&steel.qmm_bm64));
                } else {
                    qmm.encode(encoder, buffers, k, n);
                }
            } else {
                affine(projection, matrix, input, output, rows, 0);
            }
        };
        let norm = |input: &RawBuffer, gain: usize, output: &RawBuffer| {
            if let Some(wheel) = wheel {
                wheel.encode_rms(encoder, [input, weights(gain), output], rows);
                model
                    .qmm
                    .as_ref()
                    .expect("Nanbeige Steel RMSNorm finite scan")
                    .scan_output(encoder, &[output, &s.status.raw], rows * HIDDEN);
            } else {
                dispatch(
                    Op::Rms,
                    &[input, weights(gain), output, &s.status.raw],
                    &[rows as u32, HIDDEN as u32],
                    [rows, 1],
                );
            }
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
            project(query, block.query, &s.normalized.raw, &s.query.raw);
            for (matrix, target) in [(block.key, &s.key), (block.value, &s.value)] {
                if let Some(steel) = steel {
                    let partials = s.split_partials.as_ref().expect("admitted split-K scratch");
                    steel.split.encode(
                        encoder,
                        [
                            weights(matrix.weights),
                            weights(matrix.scales),
                            weights(matrix.biases),
                        ],
                        &s.normalized.raw,
                        partials,
                        &target.raw,
                    );
                    model
                        .qmm
                        .as_ref()
                        .expect("Nanbeige Steel split-K finite scan")
                        .scan_output(encoder, &[&target.raw, &s.status.raw], rows * KV_WIDTH);
                } else {
                    affine(ordinary, matrix, &s.normalized.raw, &target.raw, rows, 0);
                }
            }
            let rotary_banks = [
                &*s.query.raw,
                &*s.key.raw,
                &*s.value.raw,
                &*self.kv.k.raw,
                &*self.kv.v.raw,
                &*s.status.raw,
            ];
            if let Some(wheel) = wheel {
                wheel.encode_rotary(encoder, rotary_banks, rows, start, visit.kv_slot);
            } else {
                dispatch(
                    Op::RotateCache,
                    &rotary_banks,
                    &cache_shape,
                    [(rows * ((QUERY_WIDTH + KV_WIDTH) / 2)).div_ceil(32), 1],
                );
            }
            if let Some(wheel) = decode {
                wheel.encode_t1_attention(
                    encoder,
                    [
                        &s.query.raw,
                        &self.kv.k.raw,
                        &self.kv.v.raw,
                        &s.attention.raw,
                    ],
                    start,
                    visit.kv_slot,
                );
                model
                    .qmm
                    .as_ref()
                    .expect("Nanbeige Steel wheel vector attention finite scan")
                    .scan_output(encoder, &[&s.attention.raw, &s.status.raw], QUERY_WIDTH);
            } else if let Some(steel) = steel {
                steel.encode_attention(
                    encoder,
                    [
                        &s.query.raw,
                        &self.kv.k.raw,
                        &self.kv.v.raw,
                        &s.attention.raw,
                    ],
                    start,
                    visit.kv_slot,
                );
                model
                    .qmm
                    .as_ref()
                    .expect("Nanbeige Steel finite scan")
                    .scan_output(
                        encoder,
                        &[&s.attention.raw, &s.status.raw],
                        rows * QUERY_WIDTH,
                    );
            } else {
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
            }
            project(
                attention_out,
                block.attention_out,
                &s.attention.raw,
                &s.delta.raw,
            );
            residual_add();
            norm(
                &s.residual.raw,
                block.post_attention_norm,
                &s.normalized.raw,
            );
            if let Some(dense) = dense {
                if let Some(steel) = steel {
                    dense.encode_prepacked(
                        &steel.prepacked,
                        encoder,
                        visit.physical_layer,
                        [&s.normalized.raw, &s.up.raw, &s.gate.raw, &s.status.raw],
                    );
                } else {
                    dense.encode(
                        encoder,
                        visit.physical_layer,
                        &s.normalized.raw,
                        &s.gate.raw,
                        &s.status.raw,
                    );
                }
            } else {
                affine(gate_up, block.gate, &s.normalized.raw, &s.gate.raw, rows, 0);
                affine(gate_up, block.up, &s.normalized.raw, &s.up.raw, rows, 0);
                if let Some(wheel) = decode {
                    wheel.encode_decode_swiglu(encoder, [&s.gate.raw, &s.up.raw, &s.status.raw]);
                } else {
                    dispatch(
                        Op::Swiglu,
                        &[&s.gate.raw, &s.up.raw, &s.status.raw],
                        &[rows as u32, INTERMEDIATE as u32],
                        [(rows * INTERMEDIATE).div_ceil(32), 1],
                    );
                }
            }
            project(down, block.down, &s.gate.raw, &s.delta.raw);
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
        if !score {
            return;
        }
        if let Some(steel) = steel.filter(|_| start == FIXED_STEEL_PROMPT - split_k::ROWS) {
            let head = model.plan.head;
            steel.encode_prefill_head(
                encoder,
                [
                    weights(head.weights),
                    weights(head.scales),
                    weights(head.biases),
                    &s.normalized.raw,
                    &s.logits.raw,
                ],
            );
            model
                .qmm
                .as_ref()
                .expect("Nanbeige Steel head finite scan")
                .scan_output(encoder, &[&s.logits.raw, &s.status.raw], LOGIT_ROWS);
        } else {
            affine(
                ordinary,
                model.plan.head,
                &s.normalized.raw,
                &s.logits.raw,
                1,
                (rows - 1) as u32,
            );
        }
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
        autoreleasepool(|_| session.submit(1, 2, true)).unwrap();
        cancelled.store(true, Ordering::Relaxed);
        assert!(matches!(
            session.finish_chunk(1, Ok(()), true),
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

        assert!(
            session
                .finish_chunk(1, Err(Error::Completion), true)
                .is_err()
        );
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
