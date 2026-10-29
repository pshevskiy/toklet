//! Concrete resident Gemma3 packed/Half graph, bounded to the qualified leaves.
//! Raw BF16 gamma is an explicit constructor contract, not artifact inference.
//! The caller authenticates borrowed original matrices before construction.
//! There is no checkpoint loader, tokenizer, sampling or Chat eligibility here.
use super::{
    ALLOWANCE, Error, Result, Tensor, TensorStorage, add, half_pointwise as pointwise,
    half_rms_f32 as rms,
    memory::{Buffer, Gpu, Kernel, Layout, matrix, matrix_offset, validate_buffer_bytes},
    mlx,
    profile::Command,
    staged_attention as attention, staged_cache as cache,
};
use crate::{
    mlx::affine::{Matrix, MetadataDtype},
    model::{
        Config,
        gemma3::{LayerSpec, Weights},
        shared_kv::KvKind,
    },
};
use half::bf16;
use objc2::rc::{Retained, autoreleasepool};
use objc2_metal::{MTLBuffer, MTLDevice};
use objc2_metal_performance_shaders::MPSMatrix;
use std::mem::size_of;

mod snapshots;
pub use snapshots::SnapshotInfo;
#[cfg(test)]
mod gpu_tests;
#[cfg(test)]
mod residual_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum TestFault {
    None,
    LateLayer,
    Head,
    FirstCopy,
}
const B: usize = 32;
const GUARD: usize = 7;
const MAX_LAYERS: usize = 32;
const EPSILON: f32 = 1e-6;

/// Last-row logits use one vocabulary row; diagnostics require all-row logits.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub memory_limit_bytes: usize,
    /// Known caller-owned source/application bytes still live during this session.
    pub external_resident_bytes: usize,
    pub all_logits: bool,
    pub diagnostics: bool,
}
/// Exact owned numerical allocations plus explicit nonnumerical estimates.
/// Unified shared buffers count once. Source overlap belongs only to cold_peak.
#[derive(Clone, Copy, Debug, Default)]
pub struct Memory {
    pub packed_weights: usize,
    pub effective_norms: usize,
    pub zero_bias: usize,
    pub cache: usize,
    pub scratch: usize,
    pub logits: usize,
    pub tokens_and_history: usize,
    pub diagnostic_gpu: usize,
    pub diagnostic_readback_reserved: usize,
    pub metadata: usize,
    pub shared_framework_allowance: usize,
    pub external_resident: usize,
    pub admitted: usize,
    /// Planned maximum of model-upload overlap and current-session admission.
    /// This accounted bound is not a measured process-memory peak.
    pub cold_peak: usize,
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or(Error::Options("Gemma3 memory overflow"))
}
fn extent(rows: usize, width: usize) -> Result<usize> {
    add(mul(rows, width)?, GUARD)
}
fn zero_bias_bytes(c: &Config) -> Result<usize> {
    let width = c
        .hidden_size
        .max(c.vocab_size)
        .max(c.intermediate_size)
        .max(mul(c.query_heads, c.head_dim)?);
    mul(add(width, GUARD)?, 2)
}
fn accounted_peak(model_upload: usize, session_admitted: usize) -> usize {
    model_upload.max(session_admitted)
}
fn admit(required: usize, limit: usize) -> Result<()> {
    if required > limit {
        Err(Error::Budget { required, limit })
    } else {
        Ok(())
    }
}
fn budget(limit: usize) -> Result<()> {
    if limit == 0 || limit > crate::weights::INITIAL_MEMORY_LIMIT_BYTES {
        return Err(Error::Options("Gemma3 initial memory budget"));
    }
    Ok(())
}
fn window(spec: LayerSpec, local: usize) -> Option<usize> {
    match spec.kind {
        KvKind::Sliding => Some(local),
        KvKind::Full => None,
    }
}
fn geometry(c: &Config, specs: &[LayerSpec], local: usize) -> Result<()> {
    c.validate()
        .map_err(|_| Error::Options("Gemma3 configuration"))?;
    if c.num_layers > MAX_LAYERS
        || c.max_positions > 544
        || c.query_heads != 4
        || c.kv_heads != 1
        || ![32, 256].contains(&c.head_dim)
        || ![64, 640].contains(&c.hidden_size)
        || ![128, 2048].contains(&c.intermediate_size)
        || c.vocab_size > 262144
        || c.rms_norm_eps.to_bits() != EPSILON.to_bits()
        || specs.len() != c.num_layers
        || local == 0
        || local > c.max_positions
    {
        return Err(Error::Options("Gemma3 bounded leaf geometry"));
    }
    for &spec in specs {
        let expected = if spec.kind == KvKind::Sliding {
            10000.0f32
        } else {
            1000000.0f32
        };
        if spec.theta.to_bits() != expected.to_bits() {
            return Err(Error::Options("Gemma3 ordinary local/global theta"));
        }
    }
    // Validate these leaves once without GPU allocation; sums here are not
    // added to the ledger because the actual owned operands are shared.
    rms::Plan::new(B, c.hidden_size, c.hidden_size, c.hidden_size, EPSILON)?.operand_bytes()?;
    rms::Plan::new(B * 4, c.head_dim, c.head_dim, c.head_dim, EPSILON)?.operand_bytes()?;
    pointwise::Plan::new(
        pointwise::Kind::Gelu,
        B,
        c.intermediate_size,
        c.intermediate_size,
        c.intermediate_size,
    )?
    .operand_bytes()?;
    pointwise::Plan::new(
        pointwise::Kind::Scale(bf16::from_f64((c.hidden_size as f64).sqrt()).to_f32()),
        B,
        c.hidden_size,
        c.hidden_size,
        c.hidden_size,
    )?
    .operand_bytes()?;
    Ok(())
}
fn matrix_bytes(m: &Matrix<'_>, rows: usize, columns: usize) -> Result<usize> {
    let l = m.layout();
    if l.rows() != rows
        || l.columns() != columns
        || l.bits() != 4
        || l.group_size() != 64
        || l.metadata_dtype() != MetadataDtype::BF16
    {
        return Err(Error::Options("Gemma3 packed4 group64 BF16 shape"));
    }
    mlx::validate_resident_layout(l)?;
    mlx::packed_bytes(l)
}
fn gamma(raw: &[bf16], width: usize) -> Result<()> {
    if raw.len() != width
        || raw
            .iter()
            .any(|g| !g.to_f32().is_finite() || !(1.0 + g.to_f32()).is_finite())
    {
        return Err(Error::Options("Gemma3 raw BF16 gamma"));
    }
    Ok(())
}
struct Norms {
    input: Buffer<u32>,
    post_attention: Buffer<u32>,
    pre_ff: Buffer<u32>,
    post_ff: Buffer<u32>,
    q: Buffer<u32>,
    k: Buffer<u32>,
}
struct Layer {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
    norms: Norms,
    spec: LayerSpec,
}
pub struct Model {
    gpu: Gpu,
    config: Config,
    layers: Vec<Layer>,
    embedding: Tensor,
    final_norm: Buffer<u32>,
    zero_bias: Buffer<u16>,
    local_window: usize,
    embedding_scale: f32,
    rms: rms::Operation,
    pointwise: pointwise::Operation,
    attention: attention::Operation,
    cache: cache::Operation,
    cache_bytes: usize,
    memory: Memory,
}
fn norm_upload(gpu: &Gpu, raw: &[bf16]) -> Result<Buffer<u32>> {
    let mut buffer = Buffer::new(gpu, raw.len() + GUARD)?;
    buffer.with_mut(|out| {
        out.fill(0x5a170031);
        for (dst, value) in out.iter_mut().zip(raw) {
            *dst = (1.0 + value.to_f32()).to_bits();
        }
    });
    Ok(buffer)
}
fn packed_upload(
    gpu: &Gpu,
    original: &Matrix<'_>,
    pipelines: &[[mlx::Pipeline; 2]],
) -> Result<Tensor> {
    let l = original.layout();
    let mut buffer = Buffer::new(gpu, l.weight_bytes() / 2)?;
    let mut packed = mlx::Packed::allocate(gpu, l, pipelines)?;
    packed.upload_matrix(original, &mut buffer)?;
    // Logical metadata only: this is not a dense allocation or MPS weight view.
    let layout = Layout {
        rows: l.rows(),
        columns: l.columns(),
        stride: l.columns(),
        elements: l.rows() * l.columns(),
    };
    Ok(Tensor {
        buffer,
        layout,
        storage: TensorStorage::Mlx(packed),
    })
}
impl Model {
    /// Upload authenticated original packed matrices exactly once. `source_bytes`
    /// must include the caller's live artifact/container and raw gamma owners;
    /// this cold overlap is checked before GPU creation and numerical allocation.
    /// Only this API's explicit raw-gamma contract authorizes FP32(1+gamma).
    pub fn from_raw_gamma(
        config: Config,
        specs: Vec<LayerSpec>,
        local_window: usize,
        weights: Weights<Matrix<'_>>,
        source_bytes: usize,
        memory_limit: usize,
    ) -> Result<Self> {
        budget(memory_limit)?;
        geometry(&config, &specs, local_window)?;
        let c = &config;
        if weights.layers.len() != c.num_layers {
            return Err(Error::Options("Gemma3 weight layers"));
        }
        let q = 4 * c.head_dim;
        let mut packed = matrix_bytes(&weights.embedding, c.vocab_size, c.hidden_size)?;
        let mut source_minimum = weights.embedding.layout().total_bytes();
        gamma(&weights.final_gamma, c.hidden_size)?;
        source_minimum = add(source_minimum, mul(weights.final_gamma.capacity(), 2)?)?;
        source_minimum = add(
            source_minimum,
            mul(
                weights.layers.capacity(),
                size_of::<crate::model::gemma3::LayerWeights<Matrix<'_>>>(),
            )?,
        )?;
        source_minimum = add(
            source_minimum,
            mul(specs.capacity(), size_of::<LayerSpec>())?,
        )?;
        for w in &weights.layers {
            for (m, rows, columns) in [
                (&w.q, q, c.hidden_size),
                (&w.k, c.head_dim, c.hidden_size),
                (&w.v, c.head_dim, c.hidden_size),
                (&w.o, c.hidden_size, q),
                (&w.gate, c.intermediate_size, c.hidden_size),
                (&w.up, c.intermediate_size, c.hidden_size),
                (&w.down, c.hidden_size, c.intermediate_size),
            ] {
                packed = add(packed, matrix_bytes(m, rows, columns)?)?;
                source_minimum = add(source_minimum, m.layout().total_bytes())?;
            }
            for raw in [
                &w.gamma.input,
                &w.gamma.post_attention,
                &w.gamma.pre_ff,
                &w.gamma.post_ff,
                &w.gamma.q,
                &w.gamma.k,
            ] {
                source_minimum = add(source_minimum, mul(raw.capacity(), 2)?)?;
            }
            for raw in [
                &w.gamma.input,
                &w.gamma.post_attention,
                &w.gamma.pre_ff,
                &w.gamma.post_ff,
            ] {
                gamma(raw, c.hidden_size)?;
            }
            gamma(&w.gamma.q, c.head_dim)?;
            gamma(&w.gamma.k, c.head_dim)?;
        }
        if source_bytes < source_minimum {
            return Err(Error::Options("Gemma3 cold source overlap understated"));
        }
        let norms = mul(
            add(
                mul(
                    c.num_layers,
                    add(mul(4, c.hidden_size + GUARD)?, mul(2, c.head_dim + GUARD)?)?,
                )?,
                c.hidden_size + GUARD,
            )?,
            4,
        )?;
        let zero = zero_bias_bytes(c)?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(c.num_layers)
            .map_err(|_| Error::Options("Gemma3 layer metadata allocation"))?;
        let metadata = add(
            size_of::<Self>(),
            mul(layers.capacity(), size_of::<Layer>())?,
        )?;
        let model_bytes = add(add(add(packed, norms)?, zero)?, add(metadata, ALLOWANCE)?)?;
        let cold_peak = add(model_bytes, source_bytes)?;
        admit(cold_peak, memory_limit)?;
        autoreleasepool(|_| {
            let c = &config;
            let gpu = Gpu::new()?;
            // Query every scratch view width before accepting tight head rows.
            for width in [c.hidden_size, q, c.head_dim, c.intermediate_size] {
                tight(width)?;
            }
            let maximum = gpu.device.maxBufferLength();
            let validate = |m: &Matrix<'_>| -> Result<()> {
                let l = m.layout();
                validate_buffer_bytes(l.weight_bytes(), maximum)?;
                validate_buffer_bytes(l.metadata_bytes().div_ceil(4) * 4, maximum)
            };
            validate(&weights.embedding)?;
            for w in &weights.layers {
                for m in [&w.q, &w.k, &w.v, &w.o, &w.gate, &w.up, &w.down] {
                    validate(m)?;
                }
            }
            validate_buffer_bytes((c.hidden_size + GUARD) * 4, maximum)?;
            validate_buffer_bytes((c.head_dim + GUARD) * 4, maximum)?;
            validate_buffer_bytes(zero, maximum)?;
            let mut configs = [cache::OwnerConfig {
                dim: 32,
                window: None,
            }; MAX_LAYERS];
            for (out, &spec) in configs.iter_mut().zip(&specs) {
                *out = cache::OwnerConfig {
                    dim: c.head_dim,
                    window: window(spec, local_window),
                };
            }
            let cache_plan = cache::Plan::new(c.max_positions, &configs[..c.num_layers])?;
            let cache_bytes = cache_plan.session_bytes()?;
            // One shared framework/compiler allowance in Memory, not four
            // fictitious independently reserved physical working sets.
            let rms = rms::Operation::new(&gpu, ALLOWANCE + size_of::<rms::Operation>())?;
            let pointwise =
                pointwise::Operation::new(&gpu, ALLOWANCE + size_of::<pointwise::Operation>())?;
            let attention =
                attention::Operation::new(&gpu, ALLOWANCE + size_of::<attention::Operation>())?;
            let cache =
                cache::Operation::new(&gpu, cache_plan, ALLOWANCE + size_of::<cache::Operation>())?;
            let pipelines = mlx::compile(&gpu)?;
            let embedding = packed_upload(&gpu, &weights.embedding, &pipelines)?;
            let final_norm = norm_upload(&gpu, &weights.final_gamma)?;
            for (w, spec) in weights.layers.into_iter().zip(specs) {
                let g = w.gamma;
                layers.push(Layer {
                    q: packed_upload(&gpu, &w.q, &pipelines)?,
                    k: packed_upload(&gpu, &w.k, &pipelines)?,
                    v: packed_upload(&gpu, &w.v, &pipelines)?,
                    o: packed_upload(&gpu, &w.o, &pipelines)?,
                    gate: packed_upload(&gpu, &w.gate, &pipelines)?,
                    up: packed_upload(&gpu, &w.up, &pipelines)?,
                    down: packed_upload(&gpu, &w.down, &pipelines)?,
                    spec,
                    norms: Norms {
                        input: norm_upload(&gpu, &g.input)?,
                        post_attention: norm_upload(&gpu, &g.post_attention)?,
                        pre_ff: norm_upload(&gpu, &g.pre_ff)?,
                        post_ff: norm_upload(&gpu, &g.post_ff)?,
                        q: norm_upload(&gpu, &g.q)?,
                        k: norm_upload(&gpu, &g.k)?,
                    },
                });
            }
            let zero_bias = Buffer::new(&gpu, zero / 2)?;
            let embedding_scale = bf16::from_f64((c.hidden_size as f64).sqrt()).to_f32();
            Ok(Self {
                gpu,
                config,
                layers,
                embedding,
                final_norm,
                zero_bias,
                local_window,
                embedding_scale,
                rms,
                pointwise,
                attention,
                cache,
                cache_bytes,
                memory: Memory {
                    packed_weights: packed,
                    effective_norms: norms,
                    zero_bias: zero,
                    metadata,
                    shared_framework_allowance: ALLOWANCE,
                    admitted: model_bytes,
                    cold_peak,
                    ..Memory::default()
                },
            })
        })
    }
    pub fn memory(&self) -> Memory {
        self.memory
    }
    pub fn device_name(&self) -> String {
        self.gpu.device.name().to_string()
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn session(&self, options: Options) -> Result<Session<'_>> {
        Session::new(self, options)
    }
}
fn tight(width: usize) -> Result<Layout> {
    let layout = Layout::new(B, width, true)?;
    if layout.stride != width {
        return Err(Error::Unsupported(
            "Gemma3 requires queried tight Half rows",
        ));
    }
    Ok(layout)
}
struct Scratch {
    hidden: Buffer<u16>,
    norm: Buffer<u16>,
    branch: Buffer<u16>,
    next: Buffer<u16>,
    q: Buffer<u16>,
    work: Buffer<u16>,
    k_norm: Buffer<u16>,
    gate: Buffer<u16>,
    up: Buffer<u16>,
    activated: Buffer<u16>,
    scores: Buffer<u32>,
    probabilities: Buffer<u32>,
    tokens: Buffer<u32>,
}
struct Views {
    norm: Vec<Retained<MPSMatrix>>,
    context: Vec<Retained<MPSMatrix>>,
    gate: Vec<Retained<MPSMatrix>>,
    last: Vec<Retained<MPSMatrix>>,
}
pub struct Session<'m> {
    model: &'m Model,
    cache: cache::Set<'m>,
    scratch: Scratch,
    views: Views,
    candidate: Buffer<u32>,
    published: Buffer<u32>,
    history: Vec<u32>,
    options: Options,
    memory: Memory,
    published_rows: usize,
    snapshots: Vec<snapshots::Snapshot>,
    diagnostic_rows: usize,
    diagnostic_start: usize,
    #[cfg(test)]
    test_fault: TestFault,
    #[cfg(test)]
    test_phase_a: Option<(bool, bool)>,
}
fn bank() -> Result<Vec<Retained<MPSMatrix>>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(B)
        .map_err(|_| Error::Options("Gemma3 view metadata allocation"))?;
    Ok(result)
}
impl<'m> Session<'m> {
    fn new(model: &'m Model, options: Options) -> Result<Self> {
        budget(options.memory_limit_bytes)?;
        if options.diagnostics && !options.all_logits {
            return Err(Error::Options("Gemma3 diagnostics require all-row logits"));
        }
        let c = &model.config;
        let (h, q, d, i, cap) = (
            c.hidden_size,
            4 * c.head_dim,
            c.head_dim,
            c.intermediate_size,
            c.max_positions,
        );
        let mut views = Views {
            norm: bank()?,
            context: bank()?,
            gate: bank()?,
            last: bank()?,
        };
        let mut history = Vec::new();
        history
            .try_reserve_exact(cap)
            .map_err(|_| Error::Options("Gemma3 history allocation"))?;
        let mut metadata = add(
            model.memory.metadata,
            size_of::<Self>() - size_of::<cache::Set<'_>>(),
        )?;
        for view in [&views.norm, &views.context, &views.gate, &views.last] {
            metadata = add(
                metadata,
                mul(view.capacity(), size_of::<Retained<MPSMatrix>>())?,
            )?;
        }
        let (snapshots, diagnostic_gpu, diagnostic_readback, snapshot_metadata) =
            snapshots::plan(c, options.diagnostics)?;
        metadata = add(metadata, snapshot_metadata)?;
        let half_elements = add(
            add(mul(4, extent(B, h)?)?, mul(2, extent(B, q)?)?)?,
            add(extent(B, d)?, mul(3, extent(B, i)?)?)?,
        )?;
        let score_elements = extent(4 * B, cap)?;
        let scratch_bytes = add(mul(half_elements, 2)?, mul(score_elements, 8)?)?;
        let logits_elements = extent(if options.all_logits { B } else { 1 }, c.vocab_size)?;
        let logits_bytes = mul(logits_elements, 8)?;
        let tokens = add(mul(B + GUARD, 4)?, mul(history.capacity(), 4)?)?;
        let admitted = [
            model.memory.packed_weights,
            model.memory.effective_norms,
            model.memory.zero_bias,
            model.cache_bytes,
            scratch_bytes,
            logits_bytes,
            tokens,
            diagnostic_gpu,
            diagnostic_readback,
            metadata,
            ALLOWANCE,
            options.external_resident_bytes,
        ]
        .into_iter()
        .try_fold(0, add)?;
        admit(admitted, options.memory_limit_bytes)?;
        for (width, scalar) in [
            (extent(B, h)?, 2),
            (extent(B, q)?, 2),
            (extent(B, d)?, 2),
            (extent(B, i)?, 2),
            (score_elements, 4),
            (logits_elements, 4),
            (B + GUARD, 4),
        ] {
            validate_buffer_bytes(mul(width, scalar)?, model.gpu.device.maxBufferLength())?;
        }
        snapshots::validate(&snapshots, model.gpu.device.maxBufferLength())?;
        let (hl, ql, il) = (tight(h)?, tight(q)?, tight(i)?);
        autoreleasepool(|_| {
            let gpu = &model.gpu;
            let cache = cache::Set::new(gpu, &model.cache, model.cache_bytes)?;
            let scratch = Scratch {
                hidden: Buffer::new(gpu, hl.elements)?,
                norm: Buffer::new(gpu, hl.elements)?,
                branch: Buffer::new(gpu, hl.elements)?,
                next: Buffer::new(gpu, hl.elements)?,
                q: Buffer::new(gpu, ql.elements)?,
                work: Buffer::new(gpu, ql.elements)?,
                k_norm: Buffer::new(gpu, extent(B, d)?)?,
                gate: Buffer::new(gpu, il.elements)?,
                up: Buffer::new(gpu, il.elements)?,
                activated: Buffer::new(gpu, il.elements)?,
                scores: Buffer::new(gpu, score_elements)?,
                probabilities: Buffer::new(gpu, score_elements)?,
                tokens: Buffer::new(gpu, B + GUARD)?,
            };
            for n in 1..=B {
                views
                    .norm
                    .push(matrix(&scratch.norm, hl.view(n, h)?, true)?);
                views
                    .context
                    .push(matrix(&scratch.work, ql.view(n, q)?, true)?);
                views
                    .gate
                    .push(matrix(&scratch.gate, il.view(n, i)?, true)?);
                views.last.push(matrix_offset(
                    &scratch.norm,
                    hl.view(1, h)?,
                    true,
                    (n - 1) * h,
                )?);
            }
            history.resize(cap, 0);
            let snapshots = snapshots::allocate(gpu, snapshots)?;
            Ok(Self {
                model,
                cache,
                scratch,
                views,
                candidate: Buffer::new(gpu, logits_elements)?,
                published: Buffer::new(gpu, logits_elements)?,
                history,
                options,
                published_rows: 0,
                snapshots,
                diagnostic_rows: 0,
                diagnostic_start: 0,
                #[cfg(test)]
                test_fault: TestFault::None,
                #[cfg(test)]
                test_phase_a: None,
                memory: Memory {
                    cache: model.cache_bytes,
                    scratch: scratch_bytes,
                    logits: logits_bytes,
                    tokens_and_history: tokens,
                    diagnostic_gpu,
                    diagnostic_readback_reserved: diagnostic_readback,
                    metadata,
                    external_resident: options.external_resident_bytes,
                    admitted,
                    cold_peak: accounted_peak(model.memory.cold_peak, admitted),
                    ..model.memory
                },
            })
        })
    }
    pub fn memory(&self) -> Memory {
        self.memory
    }
    pub fn is_poisoned(&self) -> bool {
        self.cache.is_poisoned()
    }
    pub fn committed_tokens(&self) -> Result<&[u32]> {
        if self.is_poisoned() {
            return Err(Error::ReplayRequired);
        }
        Ok(&self.history[..self.cache.offset()])
    }
    pub fn reset(&mut self) -> Result<()> {
        self.cache.reset()?;
        self.published_rows = 0;
        self.diagnostic_rows = 0;
        Ok(())
    }
    pub fn copy_logits(&self, output: &mut [f32]) -> Result<()> {
        if self.is_poisoned() {
            return Err(Error::ReplayRequired);
        }
        let v = self.model.config.vocab_size;
        if self.published_rows == 0 || output.len() != v {
            return Err(Error::Options("Gemma3 no logits/output shape"));
        }
        let start = (self.published_rows - 1) * v;
        self.published.with_slice(|bits| {
            for (out, &value) in output.iter_mut().zip(&bits[start..start + v]) {
                *out = f32::from_bits(value);
            }
        });
        Ok(())
    }
    pub fn copy_all_logits(&self, output: &mut [f32]) -> Result<()> {
        if self.is_poisoned() {
            return Err(Error::ReplayRequired);
        }
        if !self.options.all_logits
            || self.published_rows == 0
            || output.len() != self.published_rows * self.model.config.vocab_size
        {
            return Err(Error::Options("Gemma3 all logits/output shape"));
        }
        self.published.with_slice(|bits| {
            for (out, &value) in output.iter_mut().zip(bits) {
                *out = f32::from_bits(value);
            }
        });
        Ok(())
    }
    /// One all-layer/head Phase A, then one all-owner Phase B. No allocations in
    /// the Rust numerical path after construction; driver encoders are separate.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<()> {
        if self.is_poisoned() {
            return Err(Error::ReplayRequired);
        }
        let n = tokens.len();
        let c = &self.model.config;
        if n == 0
            || n > B
            || tokens.iter().any(|&id| id as usize >= c.vocab_size)
            || self
                .cache
                .offset()
                .checked_add(n)
                .is_none_or(|end| end > c.max_positions)
        {
            return Err(Error::Options("Gemma3 token/count/position admission"));
        }
        autoreleasepool(|_| {
            self.scratch
                .tokens
                .with_mut(|out| out[..n].copy_from_slice(tokens));
            let batch = self.cache.begin(&self.model.gpu, n)?;
            // A failed candidate cannot expose overwritten diagnostic snapshots;
            // previous committed logits/history remain usable after finite math failure.
            self.diagnostic_rows = 0;
            if let Err(error) = self.encode(batch.frames(), n) {
                self.cache.discard()?;
                return Err(error);
            }
            let finished = self.cache.finish_math(batch.frames());
            #[cfg(test)]
            {
                self.test_phase_a = self.cache.test_completion();
            }
            let ready = match finished {
                Ok(ready) => ready,
                Err(error) => {
                    if !self.is_poisoned() {
                        self.cache.discard()?;
                    }
                    return Err(error);
                }
            };
            #[cfg(test)]
            let committed = if self.test_fault == TestFault::FirstCopy {
                self.cache
                    .interrupt_copy_after_first_run(&self.model.gpu, &ready)
            } else {
                self.cache.commit(&self.model.gpu, &ready)
            };
            #[cfg(not(test))]
            let committed = self.cache.commit(&self.model.gpu, &ready);
            let info = match committed {
                Ok(info) => info,
                Err(error) => {
                    if !self.is_poisoned() {
                        self.cache.discard()?;
                    }
                    return Err(error);
                }
            };
            // All fallible work is complete. Fixed history capacity was checked
            // before begin; candidate/published owners swap without allocation.
            debug_assert_eq!(info.owners, self.model.layers.len());
            debug_assert!(info.generation > 0);
            self.history[info.start..info.end].copy_from_slice(tokens);
            std::mem::swap(&mut self.candidate, &mut self.published);
            self.published_rows = if self.options.all_logits { n } else { 1 };
            self.diagnostic_start = info.start;
            self.diagnostic_rows = n;
            Ok(())
        })
    }
    fn encode(&self, frames: &[cache::Frame], n: usize) -> Result<()> {
        let m = self.model;
        let c = &m.config;
        let (h, d, i) = (c.hidden_size, c.head_dim, c.intermediate_size);
        let s = &self.scratch;
        let first = self.cache.view(&frames[0])?;
        mlx::gather(
            &first.command,
            &m.embedding,
            &s.tokens,
            &s.norm,
            h,
            n,
            first.status,
        )?;
        self.capture(&first.command, 0, &s.norm.raw)?;
        m.pointwise.encode(
            &first.command,
            pointwise::Plan::new(pointwise::Kind::Scale(m.embedding_scale), n, h, h, h)?,
            pointwise::Invocation {
                input: &s.norm,
                output: &s.hidden,
                status: first.status,
            },
        )?;
        self.capture(&first.command, 1, &s.hidden.raw)?;
        for (index, layer) in m.layers.iter().enumerate() {
            let view = self.cache.view(&frames[index])?;
            debug_assert_eq!(view.tokens, n);
            debug_assert_eq!(view.capacity, c.max_positions);
            debug_assert_eq!(view.config.dim, d);
            let command = &view.command;
            let status = view.status;
            let base = 2 + index * 22;
            let rms =
                |input: &Buffer<u16>, scales: &Buffer<u32>, output: &Buffer<u16>, rows, width| {
                    m.rms.encode(
                        command,
                        rms::Plan::new(rows, width, width, width, EPSILON)?,
                        rms::Invocation {
                            input,
                            scales,
                            output,
                            status,
                        },
                    )
                };
            rms(&s.hidden, &layer.norms.input, &s.norm, n, h)?;
            self.capture(command, base, &s.norm.raw)?;
            for (offset, weight, out) in [
                (1, &layer.q, &s.q),
                (2, &layer.k, view.new_k),
                (3, &layer.v, view.new_v),
            ] {
                mlx::product(
                    command,
                    weight,
                    &m.zero_bias,
                    &self.views.norm[n - 1],
                    &out.raw,
                    weight.layout.rows,
                    n,
                    true,
                    status,
                )?;
                self.capture(command, base + offset, &out.raw)?;
            }
            rms(&s.q, &layer.norms.q, &s.work, n * 4, d)?;
            self.capture(command, base + 4, &s.work.raw)?;
            rms(view.new_k, &layer.norms.k, &s.k_norm, n, d)?;
            self.capture(command, base + 5, &s.k_norm.raw)?;
            for (heads, input, out, offset) in
                [(4, &s.work, &s.q, 6), (1, &s.k_norm, view.new_k, 7)]
            {
                m.gpu.encode(
                    command,
                    Kernel::Rotary,
                    &[&input.raw, &out.raw],
                    &[
                        n as u32,
                        heads,
                        d as u32,
                        view.offset as u32,
                        layer.spec.theta.to_bits(),
                    ],
                    n * heads as usize * d / 2,
                    Some(status),
                )?;
                self.capture(command, base + offset, &out.raw)?;
            }
            let plan = attention::Plan::new(
                c.max_positions,
                d,
                window(layer.spec, m.local_window),
                view.offset,
                n,
            )?;
            m.attention.encode(
                command,
                plan,
                attention::Invocation {
                    q: &s.q,
                    old_k: view.old_k,
                    old_v: view.old_v,
                    new_k: view.new_k,
                    new_v: view.new_v,
                    scores: &s.scores,
                    probabilities: &s.probabilities,
                    context: &s.work,
                    status,
                },
            )?;
            self.capture(command, base + 8, &s.scores.raw)?;
            self.capture(command, base + 9, &s.probabilities.raw)?;
            self.capture(command, base + 10, &s.work.raw)?;
            mlx::product(
                command,
                &layer.o,
                &m.zero_bias,
                &self.views.context[n - 1],
                &s.branch.raw,
                h,
                n,
                true,
                status,
            )?;
            self.capture(command, base + 11, &s.branch.raw)?;
            rms(&s.branch, &layer.norms.post_attention, &s.next, n, h)?;
            self.capture(command, base + 12, &s.next.raw)?;
            m.gpu.encode(
                command,
                Kernel::ResidualClamp,
                &[&s.next.raw, &s.hidden.raw],
                &[(n * h) as u32],
                n * h,
                Some(status),
            )?;
            self.capture(command, base + 13, &s.next.raw)?;
            rms(&s.next, &layer.norms.pre_ff, &s.norm, n, h)?;
            self.capture(command, base + 14, &s.norm.raw)?;
            for (offset, weight, out) in [(15, &layer.gate, &s.gate), (16, &layer.up, &s.up)] {
                mlx::product(
                    command,
                    weight,
                    &m.zero_bias,
                    &self.views.norm[n - 1],
                    &out.raw,
                    i,
                    n,
                    true,
                    status,
                )?;
                self.capture(command, base + offset, &out.raw)?;
            }
            m.pointwise.encode(
                command,
                pointwise::Plan::new(pointwise::Kind::Gelu, n, i, i, i)?,
                pointwise::Invocation {
                    input: &s.gate,
                    output: &s.activated,
                    status,
                },
            )?;
            self.capture(command, base + 17, &s.activated.raw)?;
            m.gpu.encode(
                command,
                Kernel::Gating,
                &[&s.activated.raw, &s.up.raw, &s.gate.raw],
                &[(n * i) as u32],
                n * i,
                Some(status),
            )?;
            self.capture(command, base + 18, &s.gate.raw)?;
            mlx::product(
                command,
                &layer.down,
                &m.zero_bias,
                &self.views.gate[n - 1],
                &s.branch.raw,
                h,
                n,
                true,
                status,
            )?;
            self.capture(command, base + 19, &s.branch.raw)?;
            rms(&s.branch, &layer.norms.post_ff, &s.hidden, n, h)?;
            self.capture(command, base + 20, &s.hidden.raw)?;
            m.gpu.encode(
                command,
                Kernel::ResidualClamp,
                &[&s.hidden.raw, &s.next.raw],
                &[(n * h) as u32],
                n * h,
                Some(status),
            )?;
            #[cfg(test)]
            if index + 1 == m.layers.len() && self.test_fault == TestFault::LateLayer {
                gpu_tests::inject_nan(command, &s.hidden.raw)?;
            }
            self.capture(command, base + 21, &s.hidden.raw)?;
        }
        let final_view = self.cache.view(&frames[0])?;
        let command = &final_view.command;
        m.rms.encode(
            command,
            rms::Plan::new(n, h, h, h, EPSILON)?,
            rms::Invocation {
                input: &s.hidden,
                scales: &m.final_norm,
                output: &s.norm,
                status: final_view.status,
            },
        )?;
        self.capture(command, 2 + 22 * c.num_layers, &s.norm.raw)?;
        let (left, rows) = if self.options.all_logits {
            (&self.views.norm[n - 1], n)
        } else {
            (&self.views.last[n - 1], 1)
        };
        mlx::product(
            command,
            &m.embedding,
            &m.zero_bias,
            left,
            &self.candidate.raw,
            c.vocab_size,
            rows,
            false,
            final_view.status,
        )?;
        #[cfg(test)]
        if self.test_fault == TestFault::Head {
            gpu_tests::inject_nan(command, &self.candidate.raw)?;
        }
        m.gpu.encode(
            command,
            Kernel::Validate,
            &[&self.candidate.raw],
            &[rows as u32, c.vocab_size as u32, c.vocab_size as u32, 0],
            rows * c.vocab_size,
            Some(final_view.status),
        )?;
        self.capture(command, 3 + 22 * c.num_layers, &self.candidate.raw)?;
        Ok(())
    }
}
