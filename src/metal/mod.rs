//! Opt-in Metal FP16 model and resident, transactional sessions on Apple Silicon.
//! Numerical kernels use concrete owned buffers; token/chat/application state is
//! outside GPU execution. A chunk has one completion boundary, not per-op waits.
mod capture;
mod diagnostic;
pub use capture::Capture;
mod memory;
use crate::{model, session, weights};
pub use diagnostic::Diagnostic;
use memory::{Buffer, Gpu, Kernel, Layout, matrix, multiply};
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus, MTLDevice};
use objc2_metal_performance_shaders::{MPSMatrix, MPSMatrixMultiplication};
use std::{collections::BTreeMap, path::Path, time::Instant};
const ALLOWANCE: usize = 64 * 1024 * 1024;
type Result<T> = std::result::Result<T, Error>;
#[derive(Debug)]
pub enum Error {
    Unsupported(&'static str),
    Options(&'static str),
    Native(String),
    Weights(weights::Error),
    Budget { required: usize, limit: usize },
    Nonfinite,
    Completion,
    ReplayRequired,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(s) | Self::Options(s) => f.write_str(s),
            Self::Native(s) => f.write_str(s),
            Self::Weights(e) => e.fmt(f),
            Self::Budget { required, limit } => write!(
                f,
                "Metal admission requires {required} bytes, budget {limit}"
            ),
            Self::Nonfinite => f.write_str("completed Metal chunk nonfinite; result discarded"),
            Self::Completion => f.write_str("Metal command failed; session requires reset/replay"),
            Self::ReplayRequired => f.write_str("Metal session requires reset/replay"),
        }
    }
}
impl std::error::Error for Error {}
impl From<weights::Error> for Error {
    fn from(e: weights::Error) -> Self {
        Self::Weights(e)
    }
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or(Error::Options("memory arithmetic overflow"))
}
fn geometry(c: &model::Config) -> Result<()> {
    c.validate()
        .map_err(|_| Error::Options("invalid model configuration"))?;
    if !matches!(
        (
            c.hidden_size,
            c.intermediate_size,
            c.query_heads,
            c.kv_heads,
            c.head_dim
        ),
        (32, 64, 4, 2, 8) | (896, 4864, 14, 2, 64)
    ) || c.num_layers > 24
        || c.vocab_size > 151936
        || c.rope_theta != 1e6
        || c.rms_norm_eps != 1e-6
    {
        return Err(Error::Unsupported(
            "Metal initially qualifies tiny/Qwen2-0.5B geometry and theta/epsilon",
        ));
    }
    Ok(())
}
struct Tensor {
    buffer: Buffer<u16>,
    layout: Layout,
    matrix: Retained<MPSMatrix>,
}
impl Tensor {
    fn allocate(g: &Gpu, l: Layout) -> Result<Self> {
        let buffer = Buffer::new(g, l.elements)?;
        let matrix = matrix(&buffer, l, true)?;
        Ok(Self {
            buffer,
            layout: l,
            matrix,
        })
    }
}
struct Layer {
    norm: Tensor,
    post: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    q_bias: Tensor,
    k_bias: Tensor,
    v_bias: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}
/// Immutable resident Half weights. The embedding allocation is tied to output.
#[derive(Clone, Copy, Debug)]
pub struct LoadTiming {
    pub load_seconds: f64,
    pub conversion_seconds: f64,
    pub compilation_seconds: f64,
}
pub struct Model {
    config: model::Config,
    gpu: Gpu,
    embedding: Tensor,
    norm: Tensor,
    layers: Vec<Layer>,
    weight_bytes: usize,
    pub load_timing: LoadTiming,
}
impl Model {
    pub fn config(&self) -> &model::Config {
        &self.config
    }
    pub fn weights_bytes(&self) -> usize {
        self.weight_bytes
    }
    pub fn device_name(&self) -> String {
        self.gpu.device.name().to_string()
    }
    pub fn load(directory: &Path, options: weights::LoadOptions) -> Result<Self> {
        autoreleasepool(|_| {
            let load_clock = Instant::now();
            let mut source = weights::HalfSource::open(directory, options)?;
            geometry(&source.config)?;
            let compile_clock = Instant::now();
            let gpu = Gpu::new()?;
            let compilation_seconds = compile_clock.elapsed().as_secs_f64();
            let mut layouts = BTreeMap::new();
            let mut weight_bytes = 0;
            for (name, shape) in &source.shapes {
                let (rows, cols) = match shape.as_slice() {
                    [n] => (1, *n),
                    [r, c] => (*r, *c),
                    _ => return Err(Error::Options("tensor rank differs")),
                };
                let l = Layout::new(rows, cols, true)?;
                weight_bytes = add(weight_bytes, l.bytes(2)?)?;
                layouts.insert(name.clone(), l);
            }
            let admitted = add(add(weight_bytes, ALLOWANCE)?, 64 * 1024)?;
            if admitted > options.memory_limit_bytes {
                return Err(Error::Budget {
                    required: admitted,
                    limit: options.memory_limit_bytes,
                });
            }
            let mut tensors = BTreeMap::new();
            // Embedding first permits exact source-level tied-output validation.
            let embedding_name = "model.embed_tokens.weight";
            for name in std::iter::once(embedding_name.to_string()).chain(
                layouts
                    .keys()
                    .filter(|n| n.as_str() != embedding_name)
                    .cloned(),
            ) {
                let l = layouts[&name];
                let mut t = Tensor::allocate(&gpu, l)?;
                let mut index = 0;
                t.buffer.with_mut(|data| {
                    source.visit_half(&name, |value| {
                        if index >= l.rows * l.columns {
                            return Err(weights::Error::Model(model::Error::InvalidConfig(
                                "Half tensor count differs",
                            )));
                        }
                        data[(index / l.columns) * l.stride + index % l.columns] = value;
                        index += 1;
                        Ok(())
                    })
                })?;
                if index != l.rows * l.columns {
                    return Err(Error::Options("Half tensor count differs"));
                }
                tensors.insert(name, t);
            }
            source.check_tied_output()?;
            let conversion_seconds = source.conversion_seconds();
            let mut model = Self::assemble(source.config, gpu, tensors, weight_bytes)?;
            model.load_timing = LoadTiming {
                load_seconds: load_clock.elapsed().as_secs_f64(),
                conversion_seconds,
                compilation_seconds,
            };
            Ok(model)
        })
    }
    fn assemble(
        config: model::Config,
        gpu: Gpu,
        mut tensors: BTreeMap<String, Tensor>,
        weight_bytes: usize,
    ) -> Result<Self> {
        let mut take = |name: &str| {
            tensors
                .remove(name)
                .ok_or(Error::Options("validated tensor absent"))
        };
        let embedding = take("model.embed_tokens.weight")?;
        let norm = take("model.norm.weight")?;
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                norm: take(&format!("{p}.input_layernorm.weight"))?,
                post: take(&format!("{p}.post_attention_layernorm.weight"))?,
                q: take(&format!("{p}.self_attn.q_proj.weight"))?,
                k: take(&format!("{p}.self_attn.k_proj.weight"))?,
                v: take(&format!("{p}.self_attn.v_proj.weight"))?,
                o: take(&format!("{p}.self_attn.o_proj.weight"))?,
                q_bias: take(&format!("{p}.self_attn.q_proj.bias"))?,
                k_bias: take(&format!("{p}.self_attn.k_proj.bias"))?,
                v_bias: take(&format!("{p}.self_attn.v_proj.bias"))?,
                gate: take(&format!("{p}.mlp.gate_proj.weight"))?,
                up: take(&format!("{p}.mlp.up_proj.weight"))?,
                down: take(&format!("{p}.mlp.down_proj.weight"))?,
            });
        }
        Ok(Self {
            config,
            gpu,
            embedding,
            norm,
            layers,
            weight_bytes,
            load_timing: LoadTiming {
                load_seconds: 0.,
                conversion_seconds: 0.,
                compilation_seconds: 0.,
            },
        })
    }
}
struct Cache {
    k: Buffer<u16>,
    v: Buffer<u16>,
}
struct Scratch {
    x: Buffer<u16>,
    norm: Buffer<u16>,
    q: Buffer<u16>,
    qr: Buffer<u16>,
    k: Buffer<u16>,
    kr: Buffer<u16>,
    v: Buffer<u16>,
    context: Buffer<u16>,
    delta: Buffer<u16>,
    gate: Buffer<u16>,
    silu: Buffer<u16>,
    up: Buffer<u16>,
    gated: Buffer<u16>,
    floats: Buffer<u32>,
    scores: Buffer<u32>,
    probs: Buffer<u32>,
    tokens: Buffer<u32>,
    status: Buffer<u32>,
    zero: Buffer<u16>,
}
struct Views {
    norm: Retained<MPSMatrix>,
    context: Retained<MPSMatrix>,
    gated: Retained<MPSMatrix>,
    result_h: Retained<MPSMatrix>,
    result_k: Retained<MPSMatrix>,
    result_m: Retained<MPSMatrix>,
    result_v: Retained<MPSMatrix>,
    hh: Retained<MPSMatrixMultiplication>,
    hk: Retained<MPSMatrixMultiplication>,
    hm: Retained<MPSMatrixMultiplication>,
    mh: Retained<MPSMatrixMultiplication>,
    hv: Retained<MPSMatrixMultiplication>,
}
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    pub host_encode_seconds: f64,
    pub wait_seconds: f64,
    pub gpu_start_seconds: f64,
    pub gpu_end_seconds: f64,
    pub readback_seconds: f64,
    pub command_count: usize,
}
/// Separate session KV/scratch. Only successfully completed chunks advance KV
/// validity/history position. Sampled tokens remain pending until evaluated.
pub struct Session<'m> {
    model: &'m Model,
    options: session::Options,
    memory: session::Memory,
    history: Vec<u32>,
    evaluated: usize,
    replay: bool,
    cache: Vec<Cache>,
    scratch: Scratch,
    views: Vec<Views>,
    logits: Vec<f32>,
    staged: Vec<f32>,
    h: Layout,
    k: Layout,
    m: Layout,
    f: Layout,
    last_timing: Option<Timing>,
    diagnostics: Vec<diagnostic::Snapshot>,
    diagnostic_rows: usize,
}
impl<'m> Session<'m> {
    pub fn new(model: &'m Model, options: session::Options) -> Result<Self> {
        autoreleasepool(|_| {
            let c = &model.config;
            let chunk = options.prefill_chunk_size;
            if options.capacity == 0
                || options.capacity > 2048
                || chunk == 0
                || chunk > 128
                || chunk > options.capacity
                || options
                    .position_offset
                    .checked_add(options.capacity)
                    .is_none_or(|n| n > c.max_positions.min(2048))
                || options.memory_limit_bytes > weights::INITIAL_MEMORY_LIMIT_BYTES
            {
                return Err(Error::Options(
                    "Metal session options outside admitted initial bounds",
                ));
            }
            let h = Layout::new(chunk, c.hidden_size, true)?;
            let k = Layout::new(chunk, c.kv_heads * c.head_dim, true)?;
            let m = Layout::new(chunk, c.intermediate_size, true)?;
            let f = Layout::new(chunk, c.vocab_size.max(c.intermediate_size), false)?;
            if h.stride != c.hidden_size
                || k.stride != c.kv_heads * c.head_dim
                || m.stride != c.intermediate_size
            {
                return Err(Error::Unsupported("requires contiguous Half activations"));
            }
            let cache_count = (c.kv_heads * options.capacity * c.head_dim)
                .checked_add(7)
                .ok_or(Error::Options("cache size overflow"))?;
            let score_count = c
                .query_heads
                .checked_mul(chunk)
                .and_then(|n| n.checked_mul(options.capacity))
                .and_then(|n| n.checked_add(7))
                .ok_or(Error::Options("score size overflow"))?;
            let zero_count = c
                .vocab_size
                .max(c.intermediate_size)
                .checked_add(7)
                .ok_or(Error::Options("zero bias overflow"))?;
            let scratch_bytes = add(
                add(
                    add(
                        h.bytes(2)?
                            .checked_mul(6)
                            .ok_or(Error::Options("scratch overflow"))?,
                        k.bytes(2)?
                            .checked_mul(3)
                            .ok_or(Error::Options("scratch overflow"))?,
                    )?,
                    m.bytes(2)?
                        .checked_mul(4)
                        .ok_or(Error::Options("scratch overflow"))?,
                )?,
                add(
                    add(
                        f.bytes(4)?,
                        score_count
                            .checked_mul(8)
                            .ok_or(Error::Options("scores overflow"))?,
                    )?,
                    add(add((chunk + 7) * 4, 32)?, zero_count * 2)?,
                )?,
            )?;
            let kv_bytes = cache_count
                .checked_mul(4)
                .and_then(|n| n.checked_mul(c.num_layers))
                .ok_or(Error::Options("KV bytes overflow"))?;
            let history_bytes = options.capacity * 4;
            let logits_bytes = c.vocab_size * 8;
            let metadata_bytes =
                chunk * std::mem::size_of::<Views>() + c.num_layers * std::mem::size_of::<Cache>();
            let admitted = [
                model.weight_bytes,
                scratch_bytes,
                kv_bytes,
                history_bytes,
                logits_bytes,
                metadata_bytes,
                ALLOWANCE,
            ]
            .into_iter()
            .try_fold(0, add)?;
            if admitted > options.memory_limit_bytes {
                return Err(Error::Budget {
                    required: admitted,
                    limit: options.memory_limit_bytes,
                });
            }
            let g = &model.gpu;
            let hb = || Buffer::new(g, h.elements);
            let kb = || Buffer::new(g, k.elements);
            let mb = || Buffer::new(g, m.elements);
            let scratch = Scratch {
                x: hb()?,
                norm: hb()?,
                q: hb()?,
                qr: hb()?,
                k: kb()?,
                kr: kb()?,
                v: kb()?,
                context: hb()?,
                delta: hb()?,
                gate: mb()?,
                silu: mb()?,
                up: mb()?,
                gated: mb()?,
                floats: Buffer::new(g, f.elements)?,
                scores: Buffer::new(g, score_count)?,
                probs: Buffer::new(g, score_count)?,
                tokens: Buffer::new(g, chunk + 7)?,
                status: Buffer::new(g, 8)?,
                zero: Buffer::new(g, zero_count)?,
            };
            let mut cache = Vec::with_capacity(c.num_layers);
            for _ in 0..c.num_layers {
                cache.push(Cache {
                    k: Buffer::new(g, cache_count)?,
                    v: Buffer::new(g, cache_count)?,
                });
            }
            let mut views = Vec::with_capacity(chunk);
            for n in 1..=chunk {
                views.push(Views {
                    norm: matrix(&scratch.norm, h.view(n, c.hidden_size)?, true)?,
                    context: matrix(&scratch.context, h.view(n, c.hidden_size)?, true)?,
                    gated: matrix(&scratch.gated, m.view(n, c.intermediate_size)?, true)?,
                    result_h: matrix(&scratch.floats, f.view(n, c.hidden_size)?, false)?,
                    result_k: matrix(&scratch.floats, f.view(n, c.kv_heads * c.head_dim)?, false)?,
                    result_m: matrix(&scratch.floats, f.view(n, c.intermediate_size)?, false)?,
                    result_v: matrix(&scratch.floats, f.view(n, c.vocab_size)?, false)?,
                    hh: multiply(g, n, c.hidden_size, c.hidden_size),
                    hk: multiply(g, n, c.hidden_size, c.kv_heads * c.head_dim),
                    hm: multiply(g, n, c.hidden_size, c.intermediate_size),
                    mh: multiply(g, n, c.intermediate_size, c.hidden_size),
                    hv: multiply(g, n, c.hidden_size, c.vocab_size),
                });
            }
            Ok(Self {
                model,
                options,
                memory: session::Memory {
                    weights_bytes: model.weight_bytes,
                    kv_bytes,
                    scratch_bytes,
                    history_bytes,
                    logits_bytes,
                    metadata_bytes,
                    metadata_allowance_bytes: ALLOWANCE,
                    admitted_bytes: admitted,
                },
                history: Vec::with_capacity(options.capacity),
                evaluated: 0,
                replay: false,
                cache,
                scratch,
                views,
                logits: vec![0.; c.vocab_size],
                staged: vec![0.; c.vocab_size],
                h,
                k,
                m,
                f,
                last_timing: None,
                diagnostics: Vec::new(),
                diagnostic_rows: 0,
            })
        })
    }
    pub fn history(&self) -> &[u32] {
        &self.history
    }
    pub fn evaluated_len(&self) -> usize {
        self.evaluated
    }
    pub fn pending_len(&self) -> usize {
        self.history.len() - self.evaluated
    }
    pub fn capacity(&self) -> usize {
        self.options.capacity
    }
    pub fn vocab_size(&self) -> usize {
        self.model.config.vocab_size
    }
    pub fn memory(&self) -> session::Memory {
        self.memory
    }
    pub fn logits(&self) -> Option<&[f32]> {
        (self.evaluated > 0 && self.pending_len() == 0 && !self.replay)
            .then_some(self.logits.as_slice())
    }
    pub fn timing(&self) -> Option<Timing> {
        self.last_timing
    }
    fn validate(&self, tokens: &[u32], base: usize) -> Result<()> {
        if base
            .checked_add(tokens.len())
            .is_none_or(|n| n > self.capacity())
            || tokens.iter().any(|t| *t as usize >= self.vocab_size())
        {
            return Err(Error::Options("token ID/count outside session capacity"));
        }
        Ok(())
    }
    pub fn append(&mut self, tokens: &[u32]) -> Result<()> {
        if self.replay {
            return Err(Error::ReplayRequired);
        }
        self.validate(tokens, self.history.len())?;
        self.history.extend_from_slice(tokens);
        Ok(())
    }
    pub fn set_tokens(&mut self, tokens: &[u32]) -> Result<()> {
        self.validate(tokens, 0)?;
        if !self.replay && tokens.starts_with(&self.history) {
            self.history
                .extend_from_slice(&tokens[self.history.len()..]);
        } else {
            self.reset();
            self.history.extend_from_slice(tokens);
        }
        Ok(())
    }
    pub fn discard_pending(&mut self) {
        self.history.truncate(self.evaluated);
    }
    pub fn reset(&mut self) {
        self.history.clear();
        self.evaluated = 0;
        self.replay = false;
        self.last_timing = None;
        self.diagnostic_rows = 0;
    }
    pub fn evaluate(&mut self) -> Result<()> {
        while self.evaluate_next_chunk()? != 0 {}
        Ok(())
    }
    pub fn evaluate_next_chunk(&mut self) -> Result<usize> {
        if self.replay {
            return Err(Error::ReplayRequired);
        }
        let n = self.pending_len().min(self.options.prefill_chunk_size);
        if n == 0 {
            return Ok(0);
        }
        let start = self.evaluated;
        self.scratch
            .tokens
            .with_mut(|words| words[..n].copy_from_slice(&self.history[start..start + n]));
        self.scratch.status.with_mut(|words| words.fill(0));
        self.diagnostic_rows = 0;
        let result = autoreleasepool(|_| self.run_chunk(n, start));
        self.finish_chunk(n, result)
    }
    fn finish_chunk(&mut self, n: usize, result: Result<Timing>) -> Result<usize> {
        match result {
            Ok(timing) => {
                std::mem::swap(&mut self.logits, &mut self.staged);
                self.evaluated += n;
                self.last_timing = Some(timing);
                self.diagnostic_rows = n;
                Ok(n)
            }
            Err(Error::Completion) => {
                self.evaluated = 0;
                self.replay = true;
                Err(Error::Completion)
            }
            Err(e) => Err(e),
        }
    }
    fn run_chunk(&mut self, n: usize, start: usize) -> Result<Timing> {
        let g = &self.model.gpu;
        let c = &self.model.config;
        let s = &self.scratch;
        let v = &self.views[n - 1];
        let command = g.command()?;
        let clock = Instant::now();
        let h = c.hidden_size;
        let kv = c.kv_heads * c.head_dim;
        let m = c.intermediate_size;
        g.encode(
            &command,
            Kernel::Embedding,
            &[&self.model.embedding.buffer.raw, &s.tokens.raw, &s.x.raw],
            &[n as u32, h as u32, c.vocab_size as u32],
            n * h,
            None,
        )?;
        self.snapshot(&command, None, "embedding", &s.x.raw)?;
        for (layer_index, (layer, cache)) in self.model.layers.iter().zip(&self.cache).enumerate() {
            rms(
                g,
                &command,
                &s.x,
                &layer.norm.buffer,
                &s.norm,
                n,
                h,
                c.rms_norm_eps,
                &s.status,
            )?;
            self.snapshot(&command, Some(layer_index), "input_norm", &s.norm.raw)?;
            project(
                g,
                &command,
                &layer.q,
                &layer.q_bias.buffer,
                &v.hh,
                &v.norm,
                &v.result_h,
                &s.floats,
                &s.q,
                self.f,
                self.h,
                n,
                &s.status,
            )?;
            self.snapshot(&command, Some(layer_index), "q", &s.q.raw)?;
            project(
                g,
                &command,
                &layer.k,
                &layer.k_bias.buffer,
                &v.hk,
                &v.norm,
                &v.result_k,
                &s.floats,
                &s.k,
                self.f,
                self.k,
                n,
                &s.status,
            )?;
            self.snapshot(&command, Some(layer_index), "k", &s.k.raw)?;
            project(
                g,
                &command,
                &layer.v,
                &layer.v_bias.buffer,
                &v.hk,
                &v.norm,
                &v.result_k,
                &s.floats,
                &s.v,
                self.f,
                self.k,
                n,
                &s.status,
            )?;
            self.snapshot(&command, Some(layer_index), "v", &s.v.raw)?;
            let position = self.options.position_offset + start;
            g.encode(
                &command,
                Kernel::Rotary,
                &[&s.q.raw, &s.qr.raw],
                &[
                    n as u32,
                    c.query_heads as u32,
                    c.head_dim as u32,
                    position as u32,
                ],
                n * h / 2,
                Some(&s.status),
            )?;
            self.snapshot(&command, Some(layer_index), "q_rotary", &s.qr.raw)?;
            g.encode(
                &command,
                Kernel::Rotary,
                &[&s.k.raw, &s.kr.raw],
                &[
                    n as u32,
                    c.kv_heads as u32,
                    c.head_dim as u32,
                    position as u32,
                ],
                n * kv / 2,
                Some(&s.status),
            )?;
            self.snapshot(&command, Some(layer_index), "k_rotary", &s.kr.raw)?;
            g.encode(
                &command,
                Kernel::Cache,
                &[&s.kr.raw, &s.v.raw, &cache.k.raw, &cache.v.raw],
                &[
                    n as u32,
                    c.kv_heads as u32,
                    c.head_dim as u32,
                    self.capacity() as u32,
                    start as u32,
                ],
                n * kv,
                None,
            )?;
            g.encode(
                &command,
                Kernel::Attention,
                &[
                    &s.qr.raw,
                    &cache.k.raw,
                    &cache.v.raw,
                    &s.scores.raw,
                    &s.probs.raw,
                    &s.context.raw,
                ],
                &[
                    n as u32,
                    c.query_heads as u32,
                    c.kv_heads as u32,
                    c.head_dim as u32,
                    self.capacity() as u32,
                    self.options.prefill_chunk_size as u32,
                    start as u32,
                    (start + n) as u32,
                ],
                n * c.query_heads,
                Some(&s.status),
            )?;
            self.snapshot(
                &command,
                Some(layer_index),
                "attention_context",
                &s.context.raw,
            )?;
            project(
                g,
                &command,
                &layer.o,
                &s.zero,
                &v.hh,
                &v.context,
                &v.result_h,
                &s.floats,
                &s.delta,
                self.f,
                self.h,
                n,
                &s.status,
            )?;
            self.snapshot(
                &command,
                Some(layer_index),
                "attention_projection",
                &s.delta.raw,
            )?;
            g.encode(
                &command,
                Kernel::Residual,
                &[&s.x.raw, &s.delta.raw],
                &[(n * h) as u32],
                n * h,
                Some(&s.status),
            )?;
            self.snapshot(&command, Some(layer_index), "attention_residual", &s.x.raw)?;
            rms(
                g,
                &command,
                &s.x,
                &layer.post.buffer,
                &s.norm,
                n,
                h,
                c.rms_norm_eps,
                &s.status,
            )?;
            self.snapshot(
                &command,
                Some(layer_index),
                "post_attention_norm",
                &s.norm.raw,
            )?;
            project(
                g,
                &command,
                &layer.gate,
                &s.zero,
                &v.hm,
                &v.norm,
                &v.result_m,
                &s.floats,
                &s.gate,
                self.f,
                self.m,
                n,
                &s.status,
            )?;
            self.snapshot(&command, Some(layer_index), "gate", &s.gate.raw)?;
            g.encode(
                &command,
                Kernel::Silu,
                &[&s.gate.raw, &s.silu.raw],
                &[(n * m) as u32],
                n * m,
                Some(&s.status),
            )?;
            self.snapshot(&command, Some(layer_index), "silu", &s.silu.raw)?;
            project(
                g,
                &command,
                &layer.up,
                &s.zero,
                &v.hm,
                &v.norm,
                &v.result_m,
                &s.floats,
                &s.up,
                self.f,
                self.m,
                n,
                &s.status,
            )?;
            self.snapshot(&command, Some(layer_index), "up", &s.up.raw)?;
            g.encode(
                &command,
                Kernel::Gating,
                &[&s.silu.raw, &s.up.raw, &s.gated.raw],
                &[(n * m) as u32],
                n * m,
                Some(&s.status),
            )?;
            self.snapshot(&command, Some(layer_index), "gated", &s.gated.raw)?;
            project(
                g,
                &command,
                &layer.down,
                &s.zero,
                &v.mh,
                &v.gated,
                &v.result_h,
                &s.floats,
                &s.delta,
                self.f,
                self.h,
                n,
                &s.status,
            )?;
            self.snapshot(&command, Some(layer_index), "down", &s.delta.raw)?;
            g.encode(
                &command,
                Kernel::Residual,
                &[&s.x.raw, &s.delta.raw],
                &[(n * h) as u32],
                n * h,
                Some(&s.status),
            )?;
            self.snapshot(&command, Some(layer_index), "output", &s.x.raw)?;
        }
        rms(
            g,
            &command,
            &s.x,
            &self.model.norm.buffer,
            &s.norm,
            n,
            h,
            c.rms_norm_eps,
            &s.status,
        )?;
        self.snapshot(&command, None, "final_norm", &s.norm.raw)?;
        encode_multiply(
            g,
            &command,
            &self.model.embedding,
            &s.zero,
            &v.hv,
            &v.norm,
            &v.result_v,
            &s.floats,
            self.f,
            n,
            c.vocab_size,
        )?;
        g.encode(
            &command,
            Kernel::Validate,
            &[&s.floats.raw],
            &[n as u32, c.vocab_size as u32, self.f.stride as u32, 0],
            n * c.vocab_size,
            Some(&s.status),
        )?;
        self.snapshot(&command, None, "logits", &s.floats.raw)?;
        let host_encode_seconds = clock.elapsed().as_secs_f64();
        command.commit();
        let wait = Instant::now();
        command.waitUntilCompleted();
        let wait_seconds = wait.elapsed().as_secs_f64();
        if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
            return Err(Error::Completion);
        }
        let readback = Instant::now();
        if s.status.with_slice(|words| words[0]) != 0 {
            return Err(Error::Nonfinite);
        }
        s.floats.with_slice(|words| {
            for (dst, bits) in self
                .staged
                .iter_mut()
                .zip(&words[(n - 1) * self.f.stride..(n - 1) * self.f.stride + c.vocab_size])
            {
                *dst = f32::from_bits(*bits);
            }
        });
        if self.staged.iter().any(|v| !v.is_finite()) {
            return Err(Error::Nonfinite);
        }
        Ok(Timing {
            host_encode_seconds,
            wait_seconds,
            gpu_start_seconds: command.GPUStartTime(),
            gpu_end_seconds: command.GPUEndTime(),
            readback_seconds: readback.elapsed().as_secs_f64(),
            command_count: 1,
        })
    }
}
#[allow(clippy::too_many_arguments)]
fn rms(
    g: &Gpu,
    c: &ProtocolObject<dyn MTLCommandBuffer>,
    x: &Buffer<u16>,
    w: &Buffer<u16>,
    out: &Buffer<u16>,
    n: usize,
    h: usize,
    eps: f32,
    status: &Buffer<u32>,
) -> Result<()> {
    g.encode(
        c,
        Kernel::Rms,
        &[&x.raw, &w.raw, &out.raw],
        &[n as u32, h as u32, eps.to_bits()],
        n,
        Some(status),
    )
}
#[allow(clippy::too_many_arguments)]
fn encode_multiply(
    g: &Gpu,
    c: &ProtocolObject<dyn MTLCommandBuffer>,
    weight: &Tensor,
    bias: &Buffer<u16>,
    kernel: &MPSMatrixMultiplication,
    left: &MPSMatrix,
    result: &MPSMatrix,
    floats: &Buffer<u32>,
    fl: Layout,
    n: usize,
    out: usize,
) -> Result<()> {
    g.encode(
        c,
        Kernel::Bias,
        &[&bias.raw, &floats.raw],
        &[n as u32, out as u32, fl.stride as u32, 0],
        n * out,
        None,
    )?;
    // SAFETY: exact same-device n-row views, validated input/output dimensions,
    // row-major weight transposed, distinct Half operands and FP32 result. Tracked
    // hazards order resident bias -> MPS -> consumers; owners survive checked wait.
    unsafe {
        kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
            c,
            left,
            &weight.matrix,
            result,
        );
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn project(
    g: &Gpu,
    c: &ProtocolObject<dyn MTLCommandBuffer>,
    weight: &Tensor,
    bias: &Buffer<u16>,
    kernel: &MPSMatrixMultiplication,
    left: &MPSMatrix,
    result: &MPSMatrix,
    floats: &Buffer<u32>,
    out: &Buffer<u16>,
    fl: Layout,
    hl: Layout,
    n: usize,
    status: &Buffer<u32>,
) -> Result<()> {
    encode_multiply(
        g,
        c,
        weight,
        bias,
        kernel,
        left,
        result,
        floats,
        fl,
        n,
        weight.layout.rows,
    )?;
    g.encode(
        c,
        Kernel::Store,
        &[&floats.raw, &out.raw],
        &[
            n as u32,
            weight.layout.rows as u32,
            fl.stride as u32,
            hl.stride as u32,
        ],
        n * weight.layout.rows,
        Some(status),
    )
}

impl session::InferenceSession for Session<'_> {
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
    fn memory(&self) -> session::Memory {
        Session::memory(self)
    }
    fn logits(&self) -> Option<&[f32]> {
        Session::logits(self)
    }
    fn append(&mut self, tokens: &[u32]) -> std::result::Result<(), session::Error> {
        Session::append(self, tokens).map_err(|e| session::Error::Backend(e.to_string()))
    }
    fn set_tokens(&mut self, tokens: &[u32]) -> std::result::Result<(), session::Error> {
        Session::set_tokens(self, tokens).map_err(|e| session::Error::Backend(e.to_string()))
    }
    fn evaluate_next_chunk(&mut self) -> std::result::Result<usize, session::Error> {
        Session::evaluate_next_chunk(self).map_err(|e| session::Error::Backend(e.to_string()))
    }
    fn discard_pending(&mut self) {
        Session::discard_pending(self)
    }
    fn reset(&mut self) {
        Session::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "explicit Apple GPU finite-overflow and injected completion-failure check"]
    fn failed_chunks_publish_no_state_and_completion_failure_requires_replay() {
        let artifact = weights::half_tests::artifact(safetensors::Dtype::F32, None, true);
        let model = Model::load(&artifact.0, weights::LoadOptions::default()).unwrap();
        let mut session = Session::new(
            &model,
            session::Options {
                capacity: 8,
                prefill_chunk_size: 1,
                ..Default::default()
            },
        )
        .unwrap();
        session.append(&[0]).unwrap();
        session.evaluate().unwrap();
        let logits = session.logits().unwrap().to_vec();
        session.append(&[1]).unwrap();
        assert!(matches!(
            session.evaluate_next_chunk(),
            Err(Error::Nonfinite)
        ));
        assert_eq!(session.evaluated_len(), 1);
        assert!(session.logits().is_none());
        session.discard_pending();
        assert_eq!(session.logits().unwrap(), logits);
        session.append(&[0]).unwrap();
        session.evaluate().unwrap();
        assert_eq!(session.evaluated_len(), 2);
        // Synthetic post-completion status failure exercises publication/replay
        // handling without deliberately provoking an unsafe GPU fault.
        assert!(matches!(
            session.finish_chunk(1, Err(Error::Completion)),
            Err(Error::Completion)
        ));
        assert_eq!(session.evaluated_len(), 0);
        assert!(session.logits().is_none());
        assert!(matches!(session.append(&[0]), Err(Error::ReplayRequired)));
        assert!(matches!(
            session.evaluate_next_chunk(),
            Err(Error::ReplayRequired)
        ));
        session.set_tokens(&[0, 0]).unwrap();
        session.evaluate().unwrap();
        assert_eq!(session.evaluated_len(), 2);
        assert_eq!(session.logits().unwrap(), logits);
    }
}
