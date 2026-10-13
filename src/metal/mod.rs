//! Opt-in Metal FP16 model and resident, transactional sessions on Apple Silicon.
//! Numerical kernels use concrete owned buffers; token/chat/application state is
//! outside GPU execution. A chunk has one completion boundary, not per-op waits.
mod capture;
mod diagnostic;
pub use capture::Capture;
mod memory;
mod profile;
use crate::{model, q4, session, weights};
pub use diagnostic::Diagnostic;
use memory::{Buffer, Gpu, Kernel, Layout, matrix, matrix_offset, multiply};
use objc2::rc::{Retained, autoreleasepool};
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus, MTLDevice};
use objc2_metal_performance_shaders::{MPSMatrix, MPSMatrixMultiplication};
use profile::Command;
pub use profile::{ClockPair, Profile, ProfileEvent};
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
    let supported = match (
        c.hidden_size,
        c.intermediate_size,
        c.query_heads,
        c.kv_heads,
        c.head_dim,
    ) {
        (32, 64, 4, 2, 8) | (896, 4864, 14, 2, 64) => c.num_layers <= 24,
        (1536, 8960, 12, 2, 128) => c.num_layers == 28,
        _ => false,
    };
    if !supported || c.vocab_size > 151936 || c.rope_theta != 1e6 || c.rms_norm_eps != 1e-6 {
        return Err(Error::Unsupported(
            "Metal requires supported tiny/Qwen geometry and theta/epsilon",
        ));
    }
    Ok(())
}
/// The planning source contributes shapes only. Reopen payload descriptors after
/// artifact identity is established, then check metadata and endpoint identities.
/// This prevents an atomic replacement before Artifact::open from leaving an
/// older open descriptor supplying excluded weights under a newer path hash.
fn bound_q4_source(
    directory: &Path,
    options: weights::LoadOptions,
    planning: weights::HalfSource,
    artifact: &q4::Artifact,
) -> Result<weights::HalfSource> {
    let config = planning.config.clone();
    let files = planning.source_files(directory)?;
    drop(planning);
    let source = weights::HalfSource::open(directory, options)?;
    geometry(&source.config)?;
    if source.config != config || source.source_files(directory)? != files {
        return Err(Error::Options("source metadata changed after Q4 planning"));
    }
    artifact.verify_sources()?;
    Ok(source)
}
struct Tensor {
    buffer: Buffer<u16>,
    layout: Layout,
    storage: TensorStorage,
}
enum TensorStorage {
    Half(Retained<MPSMatrix>),
    Q4 { stride: usize },
}
impl Tensor {
    fn allocate(g: &Gpu, l: Layout) -> Result<Self> {
        let buffer = Buffer::new(g, l.elements)?;
        let matrix = matrix(&buffer, l, true)?;
        Ok(Self {
            buffer,
            layout: l,
            storage: TensorStorage::Half(matrix),
        })
    }
    fn packed(g: &Gpu, layout: Layout, bytes: &[u8]) -> Result<Self> {
        let matrix = q4::Matrix::parse(bytes).map_err(|e| Error::Options(e.0))?;
        if matrix.rows() != layout.rows
            || matrix.columns() != layout.columns
            || bytes.len() > u32::MAX as usize
        {
            return Err(Error::Options("packed shader layout/index limits differ"));
        }
        let mut buffer = Buffer::new(g, bytes.len() / 2)?;
        buffer.with_mut(|out| {
            for (word, raw) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *word = u16::from_le_bytes([raw[0], raw[1]]);
            }
        });
        Ok(Self {
            buffer,
            layout,
            storage: TensorStorage::Q4 {
                stride: matrix.row_stride(),
            },
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
    /// Queried SIMD widths `[one-row decode, eight-row prefill]`. Numerical
    /// qualification applies independently even when the widths differ.
    pub fn q4_execution_widths(&self) -> [usize; 2] {
        [
            self.gpu.execution_width(Kernel::Q4Matvec),
            self.gpu.execution_width(Kernel::Q4Prefill),
        ]
    }
    pub fn q4_compensated_execution_width(&self) -> usize {
        self.gpu.execution_width(Kernel::Q4Compensated)
    }
    pub fn device_name(&self) -> String {
        self.gpu.device.name().to_string()
    }
    pub fn load(directory: &Path, options: weights::LoadOptions) -> Result<Self> {
        Self::load_impl(directory, None, options)
    }
    /// Opt-in Q4 layer matrices with resident Half exclusions; source stays on
    /// disk and is checked against packed provenance before and after loading.
    pub fn load_q4(directory: &Path, packed: &Path, options: weights::LoadOptions) -> Result<Self> {
        Self::load_impl(directory, Some(packed), options)
    }
    fn load_impl(
        directory: &Path,
        packed: Option<&Path>,
        options: weights::LoadOptions,
    ) -> Result<Self> {
        autoreleasepool(|_| {
            let load_clock = Instant::now();
            let mut source = weights::HalfSource::open(directory, options)?;
            geometry(&source.config)?;
            let quantized_shapes = q4::matrix_shapes(&source.config)
                .map_err(|_| Error::Options("Q4 inventory invalid"))?;
            let artifact = packed
                .map(|path| {
                    q4::Artifact::open(
                        path,
                        directory,
                        &quantized_shapes,
                        &source.source_files(directory)?,
                        options.memory_limit_bytes,
                    )
                })
                .transpose()?;
            if let Some(artifact) = &artifact {
                source = bound_q4_source(directory, options, source, artifact)?;
            }
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
                let size = if artifact.is_some() && quantized_shapes.contains_key(name) {
                    let groups = cols.div_ceil(q4::GROUP);
                    let stride = groups
                        .checked_mul(q4::RECORD_BYTES)
                        .and_then(|v| v.checked_add(63))
                        .ok_or(Error::Options("Q4 layout overflow"))?
                        / 64
                        * 64;
                    rows.checked_mul(stride)
                        .and_then(|v| v.checked_add(q4::HEADER_BYTES))
                        .ok_or(Error::Options("Q4 layout overflow"))?
                } else {
                    l.bytes(2)?
                };
                weight_bytes = add(weight_bytes, size)?;
                layouts.insert(name.clone(), l);
            }
            let admitted = add(
                add(add(weight_bytes, ALLOWANCE)?, 64 * 1024)?,
                artifact.as_ref().map_or(0, |a| a.largest_matrix_bytes),
            )?;
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
                if let Some(artifact) = &artifact
                    && quantized_shapes.contains_key(&name)
                {
                    let data = artifact.read_matrix(&name)?;
                    let tensor = Tensor::packed(&gpu, l, &data)?;
                    tensors.insert(name, tensor);
                    continue;
                }
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
            if let Some(artifact) = &artifact {
                artifact.verify_sources()?;
            }
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
    last_norm: Retained<MPSMatrix>,
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
/// Bounded operation-boundary selection. The original path remains the default.
/// The opt-in tile is experimental: equal-width device measurements establish
/// refinement of Original, while an expanded independent decoder gate remains
/// unresolved on both paths. This mode does not qualify Q4 quality or release.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub enum ProjectionMode {
    #[default]
    Original,
    Q4Tiled,
    /// Experimental FP32 two-component product/accumulation with midpoint-
    /// preserving Half rounding and unchanged Q4 reconstruction. Expanded tiny
    /// gates pass; larger Q4 and held-out quality remain unqualified.
    Q4Compensated,
}
/// Select the RMS reduction once per operation. Original remains the default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub enum NormalizationMode {
    #[default]
    Original,
    /// Experimental positive FP32 square-sum compensation. Remaining RMS
    /// arithmetic, epsilon and Half storage are unchanged.
    Compensated,
}
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Timing {
    pub command_creation_seconds: f64,
    pub clock_calibration_seconds: f64,
    pub commit_seconds: f64,
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
    projection_mode: ProjectionMode,
    normalization_mode: NormalizationMode,
    memory: session::Memory,
    history: Vec<u32>,
    evaluated: usize,
    distribution_valid: bool,
    distribution_replay: bool,
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
    trace: Option<profile::Trace>,
}
impl<'m> Session<'m> {
    pub fn new(model: &'m Model, options: session::Options) -> Result<Self> {
        Self::with_projection_mode(model, options, ProjectionMode::Original)
    }
    pub fn with_projection_mode(
        model: &'m Model,
        options: session::Options,
        projection_mode: ProjectionMode,
    ) -> Result<Self> {
        Self::with_execution_modes(model, options, projection_mode, NormalizationMode::Original)
    }
    /// Explicit experimental numerical choices; admission and resource ownership
    /// are the same as the original session and defaults remain unchanged.
    pub fn with_execution_modes(
        model: &'m Model,
        options: session::Options,
        projection_mode: ProjectionMode,
        normalization_mode: NormalizationMode,
    ) -> Result<Self> {
        autoreleasepool(|_| {
            if projection_mode != ProjectionMode::Original
                && !matches!(model.layers[0].q.storage, TensorStorage::Q4 { .. })
            {
                return Err(Error::Options(
                    "Q4 projection mode requires Q4 model weights",
                ));
            }
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
                    last_norm: matrix_offset(
                        &scratch.norm,
                        h.view(1, c.hidden_size)?,
                        true,
                        (n - 1) * h.stride,
                    )?,
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
                projection_mode,
                normalization_mode,
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
                distribution_valid: false,
                distribution_replay: false,
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
                trace: None,
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
        (self.distribution_valid && self.evaluated > 0 && self.pending_len() == 0 && !self.replay)
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
        if self.replay {
            self.reset();
            self.history.extend_from_slice(tokens);
            return Ok(());
        }
        let (keep, recover) = session::reusable_prefix(
            &self.history,
            self.evaluated,
            tokens,
            self.distribution_valid,
        );
        let recover = recover
            || (self.distribution_replay
                && keep == self.evaluated
                && tokens.get(keep) == self.history.get(keep));
        if keep < self.evaluated {
            self.distribution_valid = false;
            self.last_timing = None;
            self.diagnostic_rows = 0;
        }
        self.evaluated = keep;
        self.distribution_replay = recover;
        self.history.truncate(keep);
        self.history.extend_from_slice(&tokens[keep..]);
        Ok(())
    }
    pub fn discard_pending(&mut self) {
        self.history
            .truncate(self.evaluated + usize::from(self.distribution_replay));
        if self.distribution_replay {
            return;
        }
        if !self.distribution_valid && self.evaluated > 0 {
            self.evaluated -= 1;
            self.distribution_replay = true;
        }
    }
    pub fn reset(&mut self) {
        self.history.clear();
        self.evaluated = 0;
        self.distribution_valid = false;
        self.distribution_replay = false;
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
                self.distribution_valid = true;
                self.distribution_replay = false;
                self.last_timing = Some(timing);
                self.diagnostic_rows = n;
                Ok(n)
            }
            Err(Error::Completion) => {
                self.evaluated = 0;
                self.distribution_valid = false;
                self.distribution_replay = false;
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
        let creation_clock = Instant::now();
        let raw_command = g.command()?;
        let command_creation_seconds = creation_clock.elapsed().as_secs_f64();
        let mut clock_calibration_seconds = 0.;
        if let Some(trace) = &self.trace {
            clock_calibration_seconds += trace.reset(g);
        }
        let command = Command {
            raw: &raw_command,
            trace: self.trace.as_ref(),
        };
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
                self.normalization_mode,
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
                self.projection_mode,
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
                self.projection_mode,
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
                self.projection_mode,
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
                self.projection_mode,
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
                self.normalization_mode,
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
                self.projection_mode,
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
                self.projection_mode,
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
                self.projection_mode,
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
            self.normalization_mode,
            &s.status,
        )?;
        self.snapshot(&command, None, "final_norm", &s.norm.raw)?;
        // Ordinary prefill needs only the last causal distribution. Diagnostic
        // sessions explicitly require every row, retaining the all-row path.
        // All views/kernels are admitted and created before numerical execution.
        let all_logits = !self.diagnostics.is_empty();
        let (logit_rows, logit_input, logit_output, logit_kernel) = if all_logits {
            (n, &v.norm, &v.result_v, &v.hv)
        } else {
            (1, &v.last_norm, &self.views[0].result_v, &self.views[0].hv)
        };
        encode_multiply(
            g,
            &command,
            &self.model.embedding,
            &s.zero,
            logit_kernel,
            logit_input,
            logit_output,
            &s.floats,
            self.f,
            logit_rows,
            c.vocab_size,
        )?;
        g.encode(
            &command,
            Kernel::Validate,
            &[&s.floats.raw],
            &[
                logit_rows as u32,
                c.vocab_size as u32,
                self.f.stride as u32,
                0,
            ],
            logit_rows * c.vocab_size,
            Some(&s.status),
        )?;
        self.snapshot(&command, None, "logits", &s.floats.raw)?;
        let host_encode_seconds = clock.elapsed().as_secs_f64();
        let commit_clock = Instant::now();
        command.commit();
        let commit_seconds = commit_clock.elapsed().as_secs_f64();
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
            let base = if all_logits {
                (n - 1) * self.f.stride
            } else {
                0
            };
            for (dst, bits) in self
                .staged
                .iter_mut()
                .zip(&words[base..base + c.vocab_size])
            {
                *dst = f32::from_bits(*bits);
            }
        });
        if self.staged.iter().any(|v| !v.is_finite()) {
            return Err(Error::Nonfinite);
        }
        let readback_seconds = readback.elapsed().as_secs_f64();
        if let Some(trace) = &self.trace {
            clock_calibration_seconds += trace.complete(g);
        }
        Ok(Timing {
            command_creation_seconds,
            clock_calibration_seconds,
            commit_seconds,
            host_encode_seconds,
            wait_seconds,
            gpu_start_seconds: command.GPUStartTime(),
            gpu_end_seconds: command.GPUEndTime(),
            readback_seconds,
            command_count: 1,
        })
    }
}
#[allow(clippy::too_many_arguments)]
fn rms(
    g: &Gpu,
    c: &Command<'_>,
    x: &Buffer<u16>,
    w: &Buffer<u16>,
    out: &Buffer<u16>,
    n: usize,
    h: usize,
    eps: f32,
    mode: NormalizationMode,
    status: &Buffer<u32>,
) -> Result<()> {
    g.encode(
        c,
        Kernel::Rms,
        &[&x.raw, &w.raw, &out.raw],
        &[
            n as u32,
            h as u32,
            eps.to_bits(),
            u32::from(mode == NormalizationMode::Compensated),
        ],
        n,
        Some(status),
    )
}
#[allow(clippy::too_many_arguments)]
fn encode_multiply(
    g: &Gpu,
    c: &Command<'_>,
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
    g.marker(c, floats, "mps_before")?;
    // SAFETY: exact same-device n-row views, validated input/output dimensions,
    // row-major weight transposed, distinct Half operands and FP32 result. Tracked
    // hazards order resident bias -> MPS -> consumers; owners survive checked wait.
    unsafe {
        kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
            c.raw,
            left,
            match &weight.storage {
                TensorStorage::Half(matrix) => matrix,
                TensorStorage::Q4 { .. } => {
                    return Err(Error::Options("Q4 requires fused projection"));
                }
            },
            result,
        );
    }
    g.marker(c, floats, "mps_after")?;
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn project(
    g: &Gpu,
    c: &Command<'_>,
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
    mode: ProjectionMode,
    status: &Buffer<u32>,
) -> Result<()> {
    if let TensorStorage::Q4 { stride } = weight.storage {
        // SAFETY: these views were created from owned, admitted same-device Half
        // buffers. Immutable descriptor getters preserve their retained storage;
        // no host payload access or mutation is performed while commands encode.
        let (offset, row_bytes, input) = unsafe { (left.offset(), left.rowBytes(), left.data()) };
        if offset != 0 || !row_bytes.is_multiple_of(2) {
            return Err(Error::Options("Q4 input view offset/stride unsupported"));
        }
        let tiled = mode == ProjectionMode::Q4Tiled && n >= 8;
        return g.encode(
            c,
            if tiled {
                Kernel::Q4Prefill
            } else if mode == ProjectionMode::Q4Compensated {
                Kernel::Q4Compensated
            } else {
                Kernel::Q4Matvec
            },
            &[&weight.buffer.raw, &bias.raw, &input, &out.raw],
            &[
                n as u32,
                weight.layout.columns as u32,
                weight.layout.rows as u32,
                stride as u32,
                (row_bytes / 2) as u32,
                hl.stride as u32,
            ],
            if tiled { n.div_ceil(8) } else { n } * weight.layout.rows,
            Some(status),
        );
    }
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
    fn q4_binding_reopens_exclusions_after_atomic_source_replacement_without_gpu() {
        use serde_json::json;
        use sha2::{Digest, Sha256};
        let owned = weights::half_tests::artifact(safetensors::Dtype::F32, None, false);
        let directory = &owned.0;
        let options = weights::LoadOptions::default();
        let planning = weights::HalfSource::open(directory, options).unwrap();
        let shapes = q4::matrix_shapes(&planning.config).unwrap();
        let source_path = directory.join("model.safetensors");
        let mut bytes = std::fs::read(&source_path).unwrap();
        let (offset, length) = {
            let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            let tensor = tensors.tensor("model.embed_tokens.weight").unwrap();
            (
                tensor.data().as_ptr() as usize - bytes.as_ptr() as usize,
                tensor.data().len(),
            )
        };
        for word in bytes[offset..offset + length].chunks_exact_mut(4) {
            word.copy_from_slice(&2f32.to_le_bytes());
        }
        let replacement = directory.join("replacement");
        std::fs::write(&replacement, &bytes).unwrap();
        std::fs::rename(&replacement, &source_path).unwrap();
        // planning still owns the old descriptor; the new artifact correctly binds
        // the replaced source, whose excluded embedding is now 2 rather than 1.
        let packed = directory.join("packed");
        std::fs::create_dir(&packed).unwrap();
        let mut entries = Vec::new();
        let mut total = 0;
        for (name, shape) in &shapes {
            let (rows, cols) = (shape[0], shape[1]);
            let stride = cols.div_ceil(64) * 36_usize;
            let stride = stride.div_ceil(64) * 64;
            let mut data = vec![0u8; 64 + rows * stride];
            data[..8].copy_from_slice(q4::MAGIC);
            data[8..12].copy_from_slice(&1u32.to_le_bytes());
            data[12..16].copy_from_slice(&64u32.to_le_bytes());
            for (offset, value) in [(16, rows), (24, cols), (32, stride), (40, rows * stride)] {
                data[offset..offset + 8].copy_from_slice(&(value as u64).to_le_bytes());
            }
            for row in 0..rows {
                for group in 0..cols.div_ceil(64) {
                    let start = 64 + row * stride + group * 36;
                    data[start + 2..start + 4]
                        .copy_from_slice(&half::f16::ONE.to_bits().to_le_bytes());
                }
            }
            let filename = format!("{name}.q4");
            std::fs::write(packed.join(&filename), &data).unwrap();
            total += data.len();
            entries.push(json!({"name":name,"file":filename,"rows":rows,"columns":cols,"packed_bytes":data.len(),
                "packed_sha256":format!("{:x}",Sha256::digest(&data)),"source_f32_sha256":format!("{:x}",Sha256::digest(1f32.to_le_bytes().repeat(rows*cols)))}));
        }
        let manifest = json!({"format":"toklet-q4-model-v1","quantizer":q4::QUANTIZER,
            "config_sha256":format!("{:x}",Sha256::digest(std::fs::read(directory.join("config.json")).unwrap())),
            "sources":[{"file":"model.safetensors","sha256":format!("{:x}",Sha256::digest(&bytes))}],
            "matrices":entries,"packed_matrix_bytes":total,"exclusions":["norms","qkv_biases","tied_embedding_output"],"excluded_dtype":"F16"});
        std::fs::write(
            packed.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let artifact = q4::Artifact::open(
            &packed,
            directory,
            &shapes,
            &["model.safetensors".into()],
            options.memory_limit_bytes,
        )
        .unwrap();
        let mut source = bound_q4_source(directory, options, planning, &artifact).unwrap();
        let mut count = 0;
        source
            .visit_half("model.embed_tokens.weight", |value| {
                assert_eq!(value, half::f16::from_f32(2.).to_bits());
                count += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(count, 64 * 32);
        artifact.verify_sources().unwrap();
    }
    #[test]
    #[ignore = "explicit pinned Qwen Q4 projection oracle on Apple GPU; local artifacts required"]
    fn fused_q4_qwen_shapes_match_fp64_reconstruction_oracle() {
        autoreleasepool(|_| {
            let root = Path::new(env!("CARGO_MANIFEST_DIR"));
            let source = root.join("models/Qwen2.5-0.5B-Instruct");
            let packed = root.join("artifacts/m5-qwen-q4-v1");
            let model = Model::load_q4(&source, &packed, weights::LoadOptions::default()).unwrap();
            let artifact = q4::Artifact::open(
                &packed,
                &source,
                &q4::matrix_shapes(model.config()).unwrap(),
                &["model.safetensors".into()],
                weights::INITIAL_MEMORY_LIMIT_BYTES,
            )
            .unwrap();
            for index in [0, 1, 23] {
                let layer = &model.layers[index];
                let zero = Buffer::<u16>::new(&model.gpu, model.config.intermediate_size).unwrap();
                for (name, weight, bias) in [
                    ("self_attn.q_proj.weight", &layer.q, &layer.q_bias.buffer),
                    ("self_attn.k_proj.weight", &layer.k, &layer.k_bias.buffer),
                    ("self_attn.v_proj.weight", &layer.v, &layer.v_bias.buffer),
                    ("self_attn.o_proj.weight", &layer.o, &zero),
                    ("mlp.gate_proj.weight", &layer.gate, &zero),
                    ("mlp.up_proj.weight", &layer.up, &zero),
                    ("mlp.down_proj.weight", &layer.down, &zero),
                ] {
                    let data = artifact
                        .read_matrix(&format!("model.layers.{index}.{name}"))
                        .unwrap();
                    let oracle = q4::Matrix::parse(&data).unwrap();
                    let inputs: Vec<f32> = (0..oracle.columns())
                        .map(|i| (i as i32 % 23 - 11) as f32 * 0.125)
                        .collect();
                    let mut x = Buffer::<u16>::new(&model.gpu, inputs.len()).unwrap();
                    x.with_mut(|v| {
                        for (dst, &src) in v.iter_mut().zip(&inputs) {
                            *dst = half::f16::from_f32(src).to_bits();
                        }
                    });
                    let out = Buffer::<u16>::new(&model.gpu, oracle.rows()).unwrap();
                    let status = Buffer::<u32>::new(&model.gpu, 8).unwrap();
                    let mut expected = vec![0.0; oracle.rows()];
                    oracle.matvec(&inputs, &mut expected).unwrap();
                    let biases: Vec<f32> = bias.with_slice(|v| {
                        v[..oracle.rows()]
                            .iter()
                            .map(|&b| half::f16::from_bits(b).to_f32())
                            .collect()
                    });
                    for (value, &b) in expected.iter_mut().zip(&biases) {
                        *value += b;
                    }
                    let raw = model.gpu.command().unwrap();
                    let command = Command {
                        raw: &raw,
                        trace: None,
                    };
                    model
                        .gpu
                        .encode(
                            &command,
                            Kernel::Q4Matvec,
                            &[&weight.buffer.raw, &bias.raw, &x.raw, &out.raw],
                            &[
                                1,
                                oracle.columns() as u32,
                                oracle.rows() as u32,
                                oracle.row_stride() as u32,
                                oracle.columns() as u32,
                                oracle.rows() as u32,
                            ],
                            oracle.rows(),
                            Some(&status),
                        )
                        .unwrap();
                    raw.commit();
                    raw.waitUntilCompleted();
                    assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
                    assert_eq!(status.with_slice(|v| v[0]), 0);
                    let unit = 2f64.powi(-24);
                    let operations = (oracle.columns() + 2) as f64;
                    let gamma = operations * unit / (1.0 - operations * unit);
                    out.with_slice(|v| {
                        for (row, (&bits, &want)) in v.iter().zip(&expected).enumerate() {
                            let absolute: f64 = inputs
                                .iter()
                                .enumerate()
                                .map(|(col, &x)| {
                                    f64::from(oracle.value(row, col).unwrap()).abs()
                                        * f64::from(x).abs()
                                })
                                .sum::<f64>()
                                + f64::from(biases[row]).abs();
                            let bound = gamma * absolute
                                + 2f64.powi(-11) * f64::from(want).abs()
                                + 2f64.powi(-25);
                            let actual = half::f16::from_bits(bits).to_f32();
                            assert!(
                                actual.is_finite()
                                    && (f64::from(actual) - f64::from(want)).abs() <= bound,
                                "layer{index}/{name} row{row}: {actual}/{want} bound{bound}"
                            );
                        }
                    });
                    println!(
                        "Q4 Qwen layer{index}/{name} {}x{} passed mathematical bound",
                        oracle.rows(),
                        oracle.columns()
                    );
                }
            }
            println!("Q4 resident weight bytes {}", model.weight_bytes);
        });
    }
    #[test]
    #[ignore = "explicit Apple GPU Q4 affine/subnormal/padding/overflow oracle check"]
    fn fused_q4_matvec_matches_scalar_oracle_and_checks_storage_range() {
        autoreleasepool(|_| {
            let gpu = Gpu::new().unwrap();
            let mut packed = include_bytes!("../../tests/fixtures/q4/rows.q4").to_vec();
            let mut x = Buffer::<u16>::new(&gpu, 3 * 72).unwrap();
            let mut bias = Buffer::<u16>::new(&gpu, 2).unwrap();
            bias.with_mut(|v| {
                v[0] = half::f16::from_f32(0.25).to_bits();
                v[1] = half::f16::from_f32(-0.5).to_bits();
            });
            let mut output = Buffer::<u16>::new(&gpu, 3 * 8).unwrap();
            let mut status = Buffer::<u32>::new(&gpu, 8).unwrap();
            for case in 0..3 {
                if case == 1 {
                    packed[64..66].copy_from_slice(&1_u16.to_le_bytes());
                    packed[66..68].copy_from_slice(&0_u16.to_le_bytes());
                    bias.with_mut(|v| v.fill(0));
                }
                if case == 2 {
                    packed[64..66]
                        .copy_from_slice(&half::f16::from_f32(8736.).to_bits().to_le_bytes());
                    packed[66..68]
                        .copy_from_slice(&half::f16::from_f32(-65504.).to_bits().to_le_bytes());
                }
                let oracle = q4::Matrix::parse(&packed).unwrap();
                let mut weight = Buffer::<u16>::new(&gpu, packed.len() / 2).unwrap();
                weight.with_mut(|v| {
                    for (dst, src) in v.iter_mut().zip(packed.chunks_exact(2)) {
                        *dst = u16::from_le_bytes([src[0], src[1]]);
                    }
                });
                x.with_mut(|v| {
                    v.fill(0);
                    for row in 0..3 {
                        for col in 0..65 {
                            let value = if case == 0 {
                                (col as f32 % 7. - 3.) * 0.5
                            } else if col == 1 {
                                1.
                            } else {
                                0.
                            };
                            v[row * 72 + col] = half::f16::from_f32(value).to_bits();
                        }
                    }
                });
                output.with_mut(|v| v.fill(0x7bff));
                status.with_mut(|v| v.fill(0));
                let raw = gpu.command().unwrap();
                let command = Command {
                    raw: &raw,
                    trace: None,
                };
                gpu.encode(
                    &command,
                    Kernel::Q4Matvec,
                    &[&weight.raw, &bias.raw, &x.raw, &output.raw],
                    &[3, 65, 2, 128, 72, 8],
                    6,
                    Some(&status),
                )
                .unwrap();
                raw.commit();
                raw.waitUntilCompleted();
                assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
                assert_eq!(status.with_slice(|v| v[0]), 0);
                let mut inputs = [0.0; 65];
                x.with_slice(|v| {
                    for (dst, &bits) in inputs.iter_mut().zip(v) {
                        *dst = half::f16::from_bits(bits).to_f32();
                    }
                });
                let mut expected = [0.0; 2];
                oracle.matvec(&inputs, &mut expected).unwrap();
                bias.with_slice(|v| {
                    for (e, &bits) in expected.iter_mut().zip(v) {
                        *e += half::f16::from_bits(bits).to_f32();
                    }
                });
                output.with_slice(|v| {
                    for row in 0..3 {
                        for col in 0..8 {
                            if col < 2 {
                                assert_eq!(
                                    v[row * 8 + col],
                                    half::f16::from_f32(expected[col]).to_bits(),
                                    "case{case} row{row} col{col}"
                                );
                            } else {
                                assert_eq!(v[row * 8 + col], 0x7bff);
                            }
                        }
                    }
                });
                // Explicit finite FP32 -> nonfinite Half rejection.
                if case == 2 {
                    x.with_mut(|v| {
                        v.fill(0);
                        v[0] = half::f16::from_f32(2.).to_bits();
                    });
                    status.with_mut(|v| v.fill(0));
                    let raw = gpu.command().unwrap();
                    let command = Command {
                        raw: &raw,
                        trace: None,
                    };
                    gpu.encode(
                        &command,
                        Kernel::Q4Matvec,
                        &[&weight.raw, &bias.raw, &x.raw, &output.raw],
                        &[1, 65, 2, 128, 72, 8],
                        2,
                        Some(&status),
                    )
                    .unwrap();
                    raw.commit();
                    raw.waitUntilCompleted();
                    assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
                    assert_eq!(status.with_slice(|v| v[0]), 1);
                }
            }
        });
    }
    #[test]
    #[ignore = "explicit Apple GPU Q4 tile row tails, padded strides, subnormals and overflow"]
    fn q4_prefill_tile_matches_scalar_oracle_and_preserves_inactive_storage() {
        autoreleasepool(|_| {
            let gpu = Gpu::new().unwrap();
            println!(
                "Q4 queried SIMD widths decode{}, prefill{}",
                gpu.execution_width(Kernel::Q4Matvec),
                gpu.execution_width(Kernel::Q4Prefill)
            );
            let mut packed = include_bytes!("../../tests/fixtures/q4/rows.q4").to_vec();
            let mut input = Buffer::<u16>::new(&gpu, 17 * 72 + 7).unwrap();
            let mut output = Buffer::<u16>::new(&gpu, 17 * 8 + 7).unwrap();
            let mut bias = Buffer::<u16>::new(&gpu, 2).unwrap();
            let mut status = Buffer::<u32>::new(&gpu, 8).unwrap();
            for subnormal in [false, true] {
                if subnormal {
                    packed[64..66].copy_from_slice(&1u16.to_le_bytes());
                    packed[66..68].copy_from_slice(&0u16.to_le_bytes());
                }
                let oracle = q4::Matrix::parse(&packed).unwrap();
                let mut weight = Buffer::<u16>::new(&gpu, packed.len() / 2).unwrap();
                weight.with_mut(|values| {
                    for (dst, src) in values.iter_mut().zip(packed.chunks_exact(2)) {
                        *dst = u16::from_le_bytes([src[0], src[1]]);
                    }
                });
                let biases = if subnormal { [0., 0.] } else { [0.25, -0.5] };
                bias.with_mut(|values| {
                    for (dst, value) in values.iter_mut().zip(biases) {
                        *dst = half::f16::from_f32(value).to_bits();
                    }
                });
                for rows in [1usize, 7, 8, 9, 17] {
                    input.with_mut(|values| {
                        values.fill(0x7e00); // Poison padding and inactive rows.
                        for row in 0..rows {
                            for column in 0..65 {
                                let value =
                                    (column as f32 % 7. - 3.) * 0.5 + (row % 3) as f32 * 0.25;
                                values[row * 72 + column] = half::f16::from_f32(value).to_bits();
                            }
                        }
                    });
                    for kernel in [Kernel::Q4Matvec, Kernel::Q4Prefill, Kernel::Q4Compensated] {
                        output.with_mut(|values| values.fill(0x7bff));
                        status.with_mut(|values| values.fill(0));
                        let raw = gpu.command().unwrap();
                        let command = Command {
                            raw: &raw,
                            trace: None,
                        };
                        gpu.encode(
                            &command,
                            kernel,
                            &[&weight.raw, &bias.raw, &input.raw, &output.raw],
                            &[rows as u32, 65, 2, 128, 72, 8],
                            if matches!(kernel, Kernel::Q4Prefill) {
                                rows.div_ceil(8) * 2
                            } else {
                                rows * 2
                            },
                            Some(&status),
                        )
                        .unwrap();
                        raw.commit();
                        raw.waitUntilCompleted();
                        assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
                        assert_eq!(status.with_slice(|values| values[0]), 0);
                        output.with_slice(|values| {
                            input.with_slice(|inputs| {
                                for (index, &bits) in values.iter().enumerate() {
                                    let row=index/8;let column=index%8;
                                    if row>=rows || column>=2 { assert_eq!(bits,0x7bff);continue; }
                                    let mut expected=f64::from(biases[column]);let mut magnitude=expected.abs();
                                    for feature in 0..65 {
                                        let product=f64::from(oracle.value(column,feature).unwrap()) * f64::from(half::f16::from_bits(inputs[row*72+feature]).to_f32());
                                        expected+=product;magnitude+=product.abs();
                                    }
                                    // The existing reconstructed-Q4 projection bound:
                                    // FP32 products/reduction plus Half normal/subnormal rounding.
                                    let u=f64::from(f32::EPSILON);let steps=2.*65.+128.;
                                    let bound=(steps*u/(1.-steps*u))*magnitude + 2f64.powi(-11)*expected.abs()+2f64.powi(-25);
                                    let actual=half::f16::from_bits(bits).to_f32();
                                    assert!(actual.is_finite() && (f64::from(actual)-expected).abs()<=bound,"rows{rows} subnormal{subnormal} row{row}/{column}: {actual}/{expected} bound{bound}");
                                }
                            });
                        });
                    }
                }
            }
            packed[64..66].copy_from_slice(&half::f16::from_f32(8736.).to_bits().to_le_bytes());
            packed[66..68].copy_from_slice(&half::f16::from_f32(-65504.).to_bits().to_le_bytes());
            let mut weight = Buffer::<u16>::new(&gpu, packed.len() / 2).unwrap();
            weight.with_mut(|values| {
                for (dst, src) in values.iter_mut().zip(packed.chunks_exact(2)) {
                    *dst = u16::from_le_bytes([src[0], src[1]]);
                }
            });
            input.with_mut(|values| {
                values.fill(0);
                for row in 0..9 {
                    values[row * 72] = half::f16::from_f32(2.).to_bits();
                }
            });
            bias.with_mut(|values| values.fill(0));
            for kernel in [Kernel::Q4Prefill, Kernel::Q4Compensated] {
                status.with_mut(|values| values.fill(0));
                output.with_mut(|values| values.fill(0x7bff));
                let raw = gpu.command().unwrap();
                let command = Command {
                    raw: &raw,
                    trace: None,
                };
                gpu.encode(
                    &command,
                    kernel,
                    &[&weight.raw, &bias.raw, &input.raw, &output.raw],
                    &[9, 65, 2, 128, 72, 8],
                    if matches!(kernel, Kernel::Q4Prefill) {
                        4
                    } else {
                        18
                    },
                    Some(&status),
                )
                .unwrap();
                raw.commit();
                raw.waitUntilCompleted();
                assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
                assert_eq!(status.with_slice(|values| values[0]), 1);
                output.with_slice(|values| {
                    for (i, &v) in values.iter().enumerate() {
                        if i / 8 >= 9 || i % 8 >= 2 {
                            assert_eq!(v, 0x7bff);
                        }
                    }
                });
            }
        });
    }
    #[test]
    #[ignore = "explicit Apple GPU compensated product/add/reduction cancellation oracle"]
    fn q4_compensated_preserves_small_terms_across_large_cancellation() {
        autoreleasepool(|_| {
            let gpu = Gpu::new().unwrap();
            let mut packed = include_bytes!("../../tests/fixtures/q4/rows.q4").to_vec();
            packed[64..66].copy_from_slice(&half::f16::from_f32(0.10005).to_bits().to_le_bytes());
            packed[66..68].copy_from_slice(&half::f16::from_f32(-0.31).to_bits().to_le_bytes());
            let oracle = q4::Matrix::parse(&packed).unwrap();
            let value = oracle.value(0, 15).unwrap();
            let product = f64::from(value) * 65504.;
            assert_ne!(
                product,
                f64::from(value * 65504.),
                "Fixture must exercise explicit product residual"
            );
            let mut weight = Buffer::<u16>::new(&gpu, packed.len() / 2).unwrap();
            weight.with_mut(|values| {
                for (a, b) in values.iter_mut().zip(packed.chunks_exact(2)) {
                    *a = u16::from_le_bytes(b.try_into().unwrap());
                }
            });
            let mut input = Buffer::<u16>::new(&gpu, 72 + 7).unwrap();
            let mut output = Buffer::<u16>::new(&gpu, 8 + 7).unwrap();
            let mut bias = Buffer::<u16>::new(&gpu, 2).unwrap();
            bias.with_mut(|v| {
                v[0] = half::f16::from_f32(0.25).to_bits();
                v[1] = half::f16::from_f32(-0.5).to_bits();
            });
            let mut status = Buffer::<u32>::new(&gpu, 8).unwrap();
            for (large, sign) in [(false, 1.), (true, 1.), (true, -1.)] {
                input.with_mut(|v| {
                    v.fill(0x7e00);
                    v[..65].fill(0);
                    if large {
                        v[15] = half::f16::from_f32(sign * 65504.).to_bits();
                        v[31] = half::f16::from_f32(-sign * 65504.).to_bits();
                    }
                    v[47] = half::f16::from_f32(sign * 0.03125).to_bits();
                    v[64] = half::f16::from_f32(sign * 2f32.powi(-14)).to_bits();
                });
                output.with_mut(|v| v.fill(0x7bff));
                status.with_mut(|v| v.fill(0));
                let raw = gpu.command().unwrap();
                let command = Command {
                    raw: &raw,
                    trace: None,
                };
                gpu.encode(
                    &command,
                    Kernel::Q4Compensated,
                    &[&weight.raw, &bias.raw, &input.raw, &output.raw],
                    &[1, 65, 2, 128, 72, 8],
                    2,
                    Some(&status),
                )
                .unwrap();
                raw.commit();
                raw.waitUntilCompleted();
                assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
                assert_eq!(status.with_slice(|v| v[0]), 0);
                output.with_slice(|v| {
                    input.with_slice(|x| {
                        for (row, &actual) in v[..2].iter().enumerate() {
                            let mut expected = 0f64;
                            for (column, &input) in x[..65].iter().enumerate() {
                                expected += f64::from(oracle.value(row, column).unwrap())
                                    * f64::from(half::f16::from_bits(input).to_f32());
                            }
                            expected += if row == 0 { 0.25 } else { -0.5 };
                            assert_eq!(
                                actual,
                                half::f16::from_f64(expected).to_bits(),
                                "large{large} sign{sign} row{row}"
                            );
                        }
                        assert!(v[2..].iter().all(|&v| v == 0x7bff));
                    })
                });
            }
            println!(
                "Compensated Q4 cancellation oracle: exact FP64-rounded Half, queried SIMD width {}",
                gpu.execution_width(Kernel::Q4Compensated)
            );
        });
    }
    #[test]
    #[ignore = "Explicit Apple GPU compensated Half rounding at signed midpoint, subnormal and overflow boundaries"]
    fn q4_compensated_rounds_half_midpoints_without_double_rounding() {
        autoreleasepool(|_| {
            let gpu = Gpu::new().unwrap();
            let mut packed = include_bytes!("../../tests/fixtures/q4/rows.q4").to_vec();
            let mut weight = Buffer::<u16>::new(&gpu, packed.len() / 2).unwrap();
            let mut input = Buffer::<u16>::new(&gpu, 72 + 7).unwrap();
            let mut output = Buffer::<u16>::new(&gpu, 8 + 7).unwrap();
            let bias = Buffer::<u16>::new(&gpu, 2).unwrap();
            let mut status = Buffer::<u32>::new(&gpu, 8).unwrap();
            let mut double_rounding_cases = 0;
            let mut cases = 0;
            // Even/odd significands, exponent transitions, signed values,
            // subnormals/zero and the finite-to-infinity rounding threshold.
            for lower in [
                0u16, 1, 0x3ff, 0x400, 0x401, 0x3554, 0x3555, 0x3bff, 0x3c00, 0x3c01, 0x7bfe,
                0x7bff,
            ] {
                let overflow = lower == 0x7bff;
                let a = half::f16::from_bits(lower).to_f32();
                let b = if overflow {
                    65536.
                } else {
                    half::f16::from_bits(lower + 1).to_f32()
                };
                let midpoint = (a + b) * 0.5;
                let exponent = ((midpoint.to_bits() >> 23) & 255) as i32 - 127;
                let tiny = half::f16::from_f32(2f32.powi(exponent - 1).max(2f32.powi(-24)));
                for sign in [1., -1.] {
                    for side in [-1., 0., 1.] {
                        // Both rows have the same exactly representable weights.
                        for row in 0..2 {
                            let first = 64 + row * 128;
                            packed[first..first + 36].fill(0);
                            packed[first + 2..first + 4].copy_from_slice(
                                &half::f16::from_f32(if overflow { 1. } else { 0.5 })
                                    .to_bits()
                                    .to_le_bytes(),
                            );
                            packed[first + 36..first + 72].fill(0);
                            packed[first + 36..first + 38].copy_from_slice(&1u16.to_le_bytes());
                            packed[first + 40] = 1;
                        }
                        let oracle = q4::Matrix::parse(&packed).unwrap();
                        weight.with_mut(|v| {
                            for (dst, bytes) in v.iter_mut().zip(packed.chunks_exact(2)) {
                                *dst = u16::from_le_bytes(bytes.try_into().unwrap());
                            }
                        });
                        input.with_mut(|v| {
                            v.fill(0x7e00);
                            v[..65].fill(0);
                            v[0] = half::f16::from_f32(sign * a).to_bits();
                            v[1] = half::f16::from_f32(sign * if overflow { 16. } else { b })
                                .to_bits();
                            v[64] = half::f16::from_f32(sign * side * tiny.to_f32()).to_bits();
                        });
                        let expected = input.with_slice(|v| {
                            v[..65]
                                .iter()
                                .enumerate()
                                .map(|(i, &x)| {
                                    f64::from(oracle.value(0, i).unwrap())
                                        * f64::from(half::f16::from_bits(x).to_f32())
                                })
                                .sum::<f64>()
                        });
                        let want = half::f16::from_f64(expected);
                        if want != half::f16::from_f32(expected as f32) {
                            double_rounding_cases += 1;
                        }
                        output.with_mut(|v| v.fill(0x7bff));
                        status.with_mut(|v| v.fill(0));
                        let raw = gpu.command().unwrap();
                        let command = Command {
                            raw: &raw,
                            trace: None,
                        };
                        gpu.encode(
                            &command,
                            Kernel::Q4Compensated,
                            &[&weight.raw, &bias.raw, &input.raw, &output.raw],
                            &[1, 65, 2, 128, 72, 8],
                            2,
                            Some(&status),
                        )
                        .unwrap();
                        raw.commit();
                        raw.waitUntilCompleted();
                        assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
                        assert_eq!(status.with_slice(|v| v[0]), u32::from(!want.is_finite()));
                        output.with_slice(|v| {
                            for &actual in &v[..2] {
                                assert_eq!(
                                    actual,
                                    want.to_bits(),
                                    "lower{lower:x} sign{sign} side{side} exact{expected}"
                                );
                            }
                            assert!(v[2..].iter().all(|&v| v == 0x7bff));
                        });
                        cases += 1;
                    }
                }
            }
            // At the zero/subnormal midpoint the minimum representable product
            // is one Float32 ULP; the other eleven boundaries each contribute
            // one witness per sign. Exact ties are deliberately not witnesses.
            assert_eq!(double_rounding_cases, 22);
            assert_eq!(cases, 72);
            println!(
                "Compensated Half rounding: {cases} signed midpoint cases, {double_rounding_cases} distinguish direct FP64 from FP32-then-Half; unchanged overflow status and guards"
            );
        });
    }
    #[test]
    #[ignore = "Explicit identical-input RMS reduction oracle; requires frozen larger first-layer capture"]
    fn compensated_rms_preserves_fp32_arithmetic_after_exact_square_sum() {
        use sha2::Digest;
        autoreleasepool(|_| {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("artifacts/m5-larger-first-layer-capture-v1.json");
            assert!(std::fs::metadata(&path).unwrap().len() <= 4 * 1024 * 1024);
            let bytes = std::fs::read(path).unwrap();
            assert!(bytes.len() <= 4 * 1024 * 1024);
            assert_eq!(
                format!("{:x}", sha2::Sha256::digest(&bytes)),
                "9cab8309655256b0c117605816e36c7cb9b36783860fb6ef9f8fcf8ae3153e2a"
            );
            let capture: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            let gpu = Gpu::new().unwrap();
            let norm: Vec<u16> = capture["norm_half_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| u16::try_from(v.as_u64().unwrap()).unwrap())
                .collect();
            assert_eq!(norm.len(), 1536);
            let mut weight = Buffer::<u16>::new(&gpu, 1536).unwrap();
            weight.with_mut(|v| v.copy_from_slice(&norm));
            let mut cases: Vec<(String, Vec<u16>, usize)> = capture["cases"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| {
                    let e = c["snapshots"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|v| v["stage"] == "embedding")
                        .unwrap();
                    let rows = e["rows"].as_u64().unwrap() as usize;
                    let values: Vec<_> = e["values_bits"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| {
                            half::f16::from_f32(f32::from_bits(
                                u32::try_from(v.as_u64().unwrap()).unwrap(),
                            ))
                            .to_bits()
                        })
                        .collect();
                    assert_eq!(values.len(), rows * 1536);
                    (c["case"].as_str().unwrap().to_owned(), values, rows)
                })
                .collect();
            assert_eq!(cases.len(), 4);
            cases.push(("zero".into(), vec![0; 1536], 1));
            cases.push((
                "dynamic_range".into(),
                (0..1536)
                    .map(|i| {
                        half::f16::from_f32(if i == 0 {
                            65504.
                        } else if i % 2 == 0 {
                            0.25
                        } else {
                            2f32.powi(-24)
                        })
                        .to_bits()
                    })
                    .collect(),
                1,
            ));
            let mut compared = 0;
            for (name, x, rows) in cases {
                let mut input = Buffer::<u16>::new(&gpu, x.len() + 7).unwrap();
                input.with_mut(|v| {
                    v.fill(0x7e00);
                    v[..x.len()].copy_from_slice(&x);
                });
                let mut output = Buffer::<u16>::new(&gpu, x.len() + 7).unwrap();
                let mut status = Buffer::<u32>::new(&gpu, 8).unwrap();
                for mode in [NormalizationMode::Original, NormalizationMode::Compensated] {
                    output.with_mut(|v| v.fill(0x7bff));
                    status.with_mut(|v| v.fill(0));
                    let raw = gpu.command().unwrap();
                    let command = Command {
                        raw: &raw,
                        trace: None,
                    };
                    rms(
                        &gpu, &command, &input, &weight, &output, rows, 1536, 1e-6, mode, &status,
                    )
                    .unwrap();
                    raw.commit();
                    raw.waitUntilCompleted();
                    assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
                    assert_eq!(status.with_slice(|v| v[0]), 0);
                    output.with_slice(|actual| {
                        for row in 0..rows {
                            let values = &x[row * 1536..(row + 1) * 1536];
                            let total = if mode == NormalizationMode::Original {
                                values.iter().fold(0f32, |sum, &v| {
                                    let v = half::f16::from_bits(v).to_f32();
                                    sum + v * v
                                })
                            } else {
                                values
                                    .iter()
                                    .map(|&v| {
                                        let v = f64::from(half::f16::from_bits(v).to_f32());
                                        v * v
                                    })
                                    .sum::<f64>() as f32
                            };
                            let inverse = 1.0f32 / (total / 1536.0 + 1e-6).sqrt();
                            for (feature, (&x, &w)) in values.iter().zip(&norm).enumerate() {
                                let expected = half::f16::from_f32(
                                    (half::f16::from_bits(x).to_f32() * inverse)
                                        * half::f16::from_bits(w).to_f32(),
                                )
                                .to_bits();
                                assert_eq!(
                                    actual[row * 1536 + feature],
                                    expected,
                                    "{name} {mode:?} row{row} feature{feature}"
                                );
                                compared += 1;
                            }
                        }
                        assert!(actual[x.len()..].iter().all(|&v| v == 0x7bff));
                    });
                }
            }
            assert_eq!(compared, 2 * 14 * 1536);
            println!(
                "RMS identical-input oracle: {compared} exact Half values, original ordered FP32 or compensated square sum with FP32 remaining arithmetic; zero/dynamic-range/guards pass"
            );
        });
    }
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
        session.enable_profile().unwrap();
        session.append(&[0]).unwrap();
        session.evaluate().unwrap();
        assert!(session.profile().is_ok());
        // Simulate command creation failing before trace.reset(): the per-attempt
        // marker is cleared, but successful timing and counters remain stored.
        session.diagnostic_rows = 0;
        assert!(matches!(
            session.finish_chunk(1, Err(Error::Native("injected creation failure".into()))),
            Err(Error::Native(_))
        ));
        assert!(session.last_timing.is_some());
        assert!(session.profile().is_err());
        let logits = session.logits().unwrap().to_vec();
        session.append(&[1]).unwrap();
        assert!(matches!(
            session.evaluate_next_chunk(),
            Err(Error::Nonfinite)
        ));
        assert_eq!(session.evaluated_len(), 1);
        assert!(session.profile().is_err());
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
