//! Bounded resident Conv→Attention→Conv over original packed weights.
//! Direct BF16 gains, explicit Half stores, and one mixed transaction per chunk.
//! This numerical API admits only the tiny qualified geometry, not a loader,
//! backend registry, tokenizer, sampler, or real-model/application factory.
use super::{
    ALLOWANCE, Error, Result, Tensor, TensorStorage, add, causal_conv3_half as conv,
    half_rms_f32 as rms,
    memory::{Buffer, Gpu, Kernel, Layout, matrix, matrix_offset, validate_buffer_bytes},
    mlx,
    profile::Command,
    staged_attention as attention, staged_cache as cache, strided_half_product as product,
};
use crate::{
    mlx::affine::{Matrix, MetadataDtype},
    model::{
        Config,
        lfm2::{Kind, Weights},
    },
};
use half::bf16;
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::NSRange;
use objc2_metal::*;
use objc2_metal_performance_shaders::MPSMatrix;
use std::mem::size_of;

mod forward;
mod snapshots;
pub use snapshots::SnapshotInfo;
#[cfg(test)]
mod gpu_tests;
#[cfg(test)]
mod tests;
const H: usize = 256;
const I: usize = 256;
const V: usize = 64;
const Q: usize = 256;
const K: usize = 128;
const D: usize = 64;
const CAP: usize = 12;
const B: usize = 32;
const GUARD: usize = 7;
const EPSILON: f32 = 1e-5;
const THETA: f32 = 1e6;
const HALF_PAD: u16 = 0x7e55;
const WORD_PAD: u32 = 0x5a170031;

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub memory_limit_bytes: usize,
    /// Caller-owned artifact/application/extra diagnostic allocations still live.
    pub external_resident_bytes: usize,
    pub all_logits: bool,
    pub diagnostics: bool,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Memory {
    pub packed_weights: usize,
    pub direct_norms: usize,
    pub convolution_coefficients: usize,
    pub zero_bias: usize,
    /// Numerical buffers plus Set and transaction metadata (not counted again).
    pub cache: usize,
    pub convolution_history: usize,
    pub scratch: usize,
    pub logits: usize,
    pub tokens: usize,
    pub diagnostic_gpu: usize,
    pub diagnostic_readback_reserved: usize,
    pub metadata: usize,
    pub shared_framework_allowance: usize,
    pub external_resident: usize,
    pub admitted: usize,
    /// Largest accounted construction/admitted bound, not a measured peak.
    pub cold_peak: usize,
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or(Error::Options("LFM2 memory overflow"))
}
fn extent(rows: usize, width: usize) -> Result<usize> {
    add(mul(rows, width)?, GUARD)
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
        Err(Error::Options("LFM2 initial memory budget"))
    } else {
        Ok(())
    }
}
fn geometry(c: &Config) -> Result<()> {
    c.validate()
        .map_err(|_| Error::Options("LFM2 configuration"))?;
    if (
        c.hidden_size,
        c.intermediate_size,
        c.vocab_size,
        c.num_layers,
        c.query_heads,
        c.kv_heads,
        c.head_dim,
        c.max_positions,
    ) != (H, I, V, 3, 4, 2, D, CAP)
        || c.rms_norm_eps.to_bits() != EPSILON.to_bits()
        || c.rope_theta.to_bits() != THETA.to_bits()
    {
        return Err(Error::Options("LFM2 bounded tiny profile"));
    }
    // Admit each reused leaf's concrete layout cold. These are shared operand
    // bounds, not extra physical allocations to add to the graph ledger.
    rms::Plan::new(B, H, H, H, EPSILON)?.operand_bytes()?;
    rms::Plan::new(B * 4, D, D, D, EPSILON)?.operand_bytes()?;
    conv::Plan::new(B, H, H, H)?.operand_bytes()?;

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
        return Err(Error::Options("LFM2 packed4 group64 BF16 shape"));
    }
    mlx::validate_resident_layout(l)?;
    mlx::packed_bytes(l)
}
fn raw(values: &[bf16], n: usize) -> Result<()> {
    if values.len() != n || values.iter().any(|v| !v.is_finite()) {
        Err(Error::Options("LFM2 direct BF16 coefficient shape/domain"))
    } else {
        Ok(())
    }
}
fn upload_raw(gpu: &Gpu, values: &[bf16]) -> Result<Buffer<u32>> {
    let mut out = Buffer::new(gpu, values.len() + GUARD)?;
    out.with_mut(|dst| {
        dst.fill(WORD_PAD);
        for (d, v) in dst.iter_mut().zip(values) {
            *d = v.to_f32().to_bits();
        }
    });
    Ok(out)
}
fn packed_upload(gpu: &Gpu, m: &Matrix<'_>, pipelines: &[[mlx::Pipeline; 2]]) -> Result<Tensor> {
    let l = m.layout();
    let mut buffer = Buffer::new(gpu, l.weight_bytes() / 2)?;
    let mut packed = mlx::Packed::allocate(gpu, l, pipelines)?;
    packed.upload_matrix(m, &mut buffer)?;
    Ok(Tensor {
        buffer,
        layout: Layout {
            rows: l.rows(),
            columns: l.columns(),
            stride: l.columns(),
            elements: l.rows() * l.columns(),
        },
        storage: TensorStorage::Mlx(packed),
    })
}
fn tight(width: usize) -> Result<Layout> {
    let l = Layout::new(B, width, true)?;
    if l.stride != width {
        return Err(Error::Unsupported("LFM2 requires queried tight Half rows"));
    }
    Ok(l)
}
struct Layer {
    kind: Kind,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
    operator_norm: Buffer<u32>,
    ffn_norm: Buffer<u32>,
}
struct Attention {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    q_norm: Buffer<u32>,
    k_norm: Buffer<u32>,
}
struct Convolution {
    bcx: Tensor,
    output: Tensor,
    coefficients: Buffer<u32>,
}
pub struct Model {
    gpu: Gpu,
    config: Config,
    embedding: Tensor,
    layers: Vec<Layer>,
    attention: Attention,
    convolution: Vec<Convolution>,
    final_norm: Buffer<u32>,
    zero_bias: Buffer<u16>,
    rms: rms::Operation,
    convolution_op: conv::Operation,
    product: product::Operation,
    attention_op: attention::Operation,
    cache: cache::Operation,
    cache_bytes: usize,
    memory: Memory,
}
impl Model {
    /// Caller authenticates original bytes and direct-gain semantics. Source
    /// overlap includes its whole retained container and raw vector capacities.
    /// No CPU-store borrow survives upload; tied head uses the embedding owner.
    pub fn from_raw_weights(
        config: Config,
        weights: Weights<'_>,
        source_bytes: usize,
        memory_limit: usize,
    ) -> Result<Self> {
        budget(memory_limit)?;
        geometry(&config)?;
        if weights.layers.len() != 3
            || weights.attention.len() != 1
            || weights.convolution.len() != 2
        {
            return Err(Error::Options("LFM2 owner inventory"));
        }
        let mut packed = matrix_bytes(&weights.embedding, V, H)?;
        let mut source_minimum = weights.embedding.layout().total_bytes();
        for (layer, kind) in weights.layers.iter().zip([
            Kind::Convolution(0),
            Kind::Attention(0),
            Kind::Convolution(1),
        ]) {
            if layer.kind != kind {
                return Err(Error::Options("LFM2 fixed owner routes"));
            }
            for (m, r, c) in [(&layer.gate, I, H), (&layer.up, I, H), (&layer.down, H, I)] {
                packed = add(packed, matrix_bytes(m, r, c)?)?;
                source_minimum = add(source_minimum, m.layout().total_bytes())?;
            }
            raw(&layer.operator_gamma, H)?;
            raw(&layer.ffn_gamma, H)?;
            source_minimum = add(
                source_minimum,
                mul(
                    add(layer.operator_gamma.capacity(), layer.ffn_gamma.capacity())?,
                    2,
                )?,
            )?;
        }
        let a = &weights.attention[0];
        for (m, r, c) in [(&a.q, Q, H), (&a.k, K, H), (&a.v, K, H), (&a.o, H, Q)] {
            packed = add(packed, matrix_bytes(m, r, c)?)?;
            source_minimum = add(source_minimum, m.layout().total_bytes())?;
        }
        raw(&a.q_gamma, D)?;
        raw(&a.k_gamma, D)?;
        source_minimum = add(
            source_minimum,
            mul(add(a.q_gamma.capacity(), a.k_gamma.capacity())?, 2)?,
        )?;
        for conv in &weights.convolution {
            for (m, r, c) in [(&conv.bcx, 3 * H, H), (&conv.output, H, H)] {
                packed = add(packed, matrix_bytes(m, r, c)?)?;
                source_minimum = add(source_minimum, m.layout().total_bytes())?;
            }
            raw(&conv.coefficients, 3 * H)?;
            source_minimum = add(source_minimum, mul(conv.coefficients.capacity(), 2)?)?;
        }
        raw(&weights.final_gamma, H)?;
        source_minimum = add(source_minimum, mul(weights.final_gamma.capacity(), 2)?)?;
        for bytes in [
            mul(
                weights.layers.capacity(),
                size_of::<crate::model::lfm2::LayerWeights<'_>>(),
            )?,
            mul(
                weights.attention.capacity(),
                size_of::<crate::model::lfm2::AttentionWeights<'_>>(),
            )?,
            mul(
                weights.convolution.capacity(),
                size_of::<crate::model::lfm2::ConvWeights<'_>>(),
            )?,
        ] {
            source_minimum = add(source_minimum, bytes)?;
        }
        if source_bytes < source_minimum {
            return Err(Error::Options("LFM2 cold source overlap understated"));
        }
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(3)
            .map_err(|_| Error::Options("LFM2 layer metadata allocation"))?;
        let mut convolution = Vec::new();
        convolution
            .try_reserve_exact(2)
            .map_err(|_| Error::Options("LFM2 convolution metadata allocation"))?;
        let metadata = add(
            size_of::<Self>(),
            add(
                mul(layers.capacity(), size_of::<Layer>())?,
                mul(convolution.capacity(), size_of::<Convolution>())?,
            )?,
        )?;
        let norms = (7 * (H + GUARD) + 2 * (D + GUARD)) * 4;
        let coefficients = 2 * (3 * H + GUARD) * 4;
        // BCx has3H output rows, larger than every unfused projection here.
        let zero = (3 * H + GUARD) * 2;
        let admitted = [packed, norms, coefficients, zero, metadata, ALLOWANCE]
            .into_iter()
            .try_fold(0, add)?;
        let cold_peak = add(admitted, source_bytes)?;
        admit(cold_peak, memory_limit)?;
        autoreleasepool(|_| {
            let gpu = Gpu::new()?;
            for w in [H, I, Q, K, 3 * H] {
                tight(w)?;
            }
            let maximum = gpu.device.maxBufferLength();
            // Matrix sizes were checked cold. Each upload also independently
            // validates actual allocation/device bounds in Buffer::new.
            for bytes in [packed, norms, coefficients, zero] {
                validate_buffer_bytes(bytes, maximum)?;
            }
            let plan = cache::Plan::new(
                CAP,
                &[cache::OwnerConfig {
                    kv_heads: 2,
                    dim: D,
                    window: None,
                }],
            )?;
            let cache_bytes = plan.session_bytes()?;
            let rms = rms::Operation::new(&gpu, ALLOWANCE + size_of::<rms::Operation>())?;
            let convolution_op =
                conv::Operation::new(&gpu, ALLOWANCE + size_of::<conv::Operation>())?;
            let product =
                product::Operation::new(&gpu, ALLOWANCE + size_of::<product::Operation>())?;
            let attention_op =
                attention::Operation::new(&gpu, ALLOWANCE + size_of::<attention::Operation>())?;
            let cache =
                cache::Operation::new(&gpu, plan, ALLOWANCE + size_of::<cache::Operation>())?;
            let pipelines = mlx::compile(&gpu)?;
            let embedding = packed_upload(&gpu, &weights.embedding, &pipelines)?;
            let final_norm = upload_raw(&gpu, &weights.final_gamma)?;
            for l in weights.layers {
                layers.push(Layer {
                    kind: l.kind,
                    gate: packed_upload(&gpu, &l.gate, &pipelines)?,
                    up: packed_upload(&gpu, &l.up, &pipelines)?,
                    down: packed_upload(&gpu, &l.down, &pipelines)?,
                    operator_norm: upload_raw(&gpu, &l.operator_gamma)?,
                    ffn_norm: upload_raw(&gpu, &l.ffn_gamma)?,
                });
            }
            let attention = Attention {
                q: packed_upload(&gpu, &a.q, &pipelines)?,
                k: packed_upload(&gpu, &a.k, &pipelines)?,
                v: packed_upload(&gpu, &a.v, &pipelines)?,
                o: packed_upload(&gpu, &a.o, &pipelines)?,
                q_norm: upload_raw(&gpu, &a.q_gamma)?,
                k_norm: upload_raw(&gpu, &a.k_gamma)?,
            };
            for w in weights.convolution {
                convolution.push(Convolution {
                    bcx: packed_upload(&gpu, &w.bcx, &pipelines)?,
                    output: packed_upload(&gpu, &w.output, &pipelines)?,
                    coefficients: upload_raw(&gpu, &w.coefficients)?,
                });
            }
            let zero_bias = Buffer::new(&gpu, zero / 2)?;
            Ok(Self {
                gpu,
                config,
                embedding,
                layers,
                attention,
                convolution,
                final_norm,
                zero_bias,
                rms,
                convolution_op,
                product,
                attention_op,
                cache,
                cache_bytes,
                memory: Memory {
                    packed_weights: packed,
                    direct_norms: norms,
                    convolution_coefficients: coefficients,
                    zero_bias: zero,
                    metadata,
                    shared_framework_allowance: ALLOWANCE,
                    admitted,
                    cold_peak,
                    ..Memory::default()
                },
            })
        })
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn memory(&self) -> Memory {
        self.memory
    }
    pub fn device_name(&self) -> String {
        self.gpu.device.name().to_string()
    }
    pub fn session(&self, options: Options) -> Result<Session<'_>> {
        Session::new(self, options)
    }
}
struct Scratch {
    hidden: Buffer<u16>,
    norm: Buffer<u16>,
    branch: Buffer<u16>,
    bcx: Buffer<u16>,
    u: Buffer<u16>,
    z: Buffer<u16>,
    q: Buffer<u16>,
    work: Buffer<u16>,
    k: Buffer<u16>,
    k_norm: Buffer<u16>,
    v: Buffer<u16>,
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
    u: Vec<Retained<MPSMatrix>>,
    gate: Vec<Retained<MPSMatrix>>,
    last: Vec<Retained<MPSMatrix>>,
}
struct History {
    old: Buffer<u16>,
    candidate: Buffer<u16>,
}
pub struct Session<'m> {
    model: &'m Model,
    cache: cache::Set<'m>,
    histories: [History; 2],
    scratch: Scratch,
    views: Views,
    candidate: Buffer<u32>,
    published: Buffer<u32>,
    tokens: Vec<u32>,
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
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TestFault {
    None,
    OmitHistory(usize),
    PartialHistory(usize),
    AfterLayer(usize),
    Head,
    FirstCopy,
}
fn bank() -> Result<Vec<Retained<MPSMatrix>>> {
    let mut v = Vec::new();
    v.try_reserve_exact(CAP)
        .map_err(|_| Error::Options("LFM2 view metadata allocation"))?;
    Ok(v)
}
fn half_buffer(gpu: &Gpu, rows: usize, width: usize) -> Result<Buffer<u16>> {
    let mut b = Buffer::new(gpu, extent(rows, width)?)?;
    b.with_mut(|v| v[rows * width..].fill(HALF_PAD));
    Ok(b)
}
fn word_buffer(gpu: &Gpu, active: usize) -> Result<Buffer<u32>> {
    let mut b = Buffer::new(gpu, active + GUARD)?;
    b.with_mut(|v| v[active..].fill(WORD_PAD));
    Ok(b)
}
impl<'m> Session<'m> {
    fn new(model: &'m Model, options: Options) -> Result<Self> {
        budget(options.memory_limit_bytes)?;
        if options.diagnostics && !options.all_logits {
            return Err(Error::Options("LFM2 diagnostics require all-row logits"));
        }
        let mut views = Views {
            norm: bank()?,
            context: bank()?,
            u: bank()?,
            gate: bank()?,
            last: bank()?,
        };
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(CAP)
            .map_err(|_| Error::Options("LFM2 token allocation"))?;
        let mut metadata = add(
            model.memory.metadata,
            size_of::<Self>() - size_of::<cache::Set<'_>>(),
        )?;
        for v in [
            &views.norm,
            &views.context,
            &views.u,
            &views.gate,
            &views.last,
        ] {
            metadata = add(
                metadata,
                mul(v.capacity(), size_of::<Retained<MPSMatrix>>())?,
            )?;
        }
        let (snapshots, diagnostic_gpu, readback, snapshot_metadata) =
            snapshots::plan(options.diagnostics)?;
        metadata = add(metadata, snapshot_metadata)?;
        let half_elements = add(mul(B, 8 * H + 2 * Q + 3 * K + 3 * I)?, 14 * GUARD)?;
        let scratch_bytes = add(mul(half_elements, 2)?, mul(extent(4 * B, CAP)?, 8)?)?;
        let history_bytes = 4 * (2 * H + GUARD) * 2;
        let logit_rows = if options.all_logits { B } else { 1 };
        let logits_bytes = mul(extent(logit_rows, V)?, 8)?;
        let token_bytes = add((B + GUARD) * 4, mul(tokens.capacity(), 4)?)?;
        let admitted = [
            model.memory.packed_weights,
            model.memory.direct_norms,
            model.memory.convolution_coefficients,
            model.memory.zero_bias,
            model.cache_bytes,
            history_bytes,
            scratch_bytes,
            logits_bytes,
            token_bytes,
            diagnostic_gpu,
            readback,
            metadata,
            ALLOWANCE,
            options.external_resident_bytes,
        ]
        .into_iter()
        .try_fold(0, add)?;
        admit(admitted, options.memory_limit_bytes)?;
        for (n, scalar) in [
            (extent(B, 3 * H)?, 2),
            (extent(4 * B, CAP)?, 4),
            (extent(logit_rows, V)?, 4),
            (2 * H + GUARD, 2),
            (B + GUARD, 4),
        ] {
            validate_buffer_bytes(mul(n, scalar)?, model.gpu.device.maxBufferLength())?;
        }
        snapshots::validate(&snapshots, model.gpu.device.maxBufferLength())?;
        autoreleasepool(|_| {
            let g = &model.gpu;
            let cache = cache::Set::new(g, &model.cache, model.cache_bytes)?;
            let histories = [
                History {
                    old: half_buffer(g, 2, H)?,
                    candidate: half_buffer(g, 2, H)?,
                },
                History {
                    old: half_buffer(g, 2, H)?,
                    candidate: half_buffer(g, 2, H)?,
                },
            ];
            let scratch = Scratch {
                hidden: half_buffer(g, B, H)?,
                norm: half_buffer(g, B, H)?,
                branch: half_buffer(g, B, H)?,
                bcx: half_buffer(g, B, 3 * H)?,
                u: half_buffer(g, B, H)?,
                z: half_buffer(g, B, H)?,
                q: half_buffer(g, B, Q)?,
                work: half_buffer(g, B, Q)?,
                k: half_buffer(g, B, K)?,
                k_norm: half_buffer(g, B, K)?,
                v: half_buffer(g, B, K)?,
                gate: half_buffer(g, B, I)?,
                up: half_buffer(g, B, I)?,
                activated: half_buffer(g, B, I)?,
                scores: word_buffer(g, 4 * B * CAP)?,
                probabilities: word_buffer(g, 4 * B * CAP)?,
                tokens: word_buffer(g, B)?,
            };
            for n in 1..=CAP {
                views
                    .norm
                    .push(matrix(&scratch.norm, tight(H)?.view(n, H)?, true)?);
                views
                    .context
                    .push(matrix(&scratch.work, tight(Q)?.view(n, Q)?, true)?);
                views
                    .u
                    .push(matrix(&scratch.u, tight(H)?.view(n, H)?, true)?);
                views
                    .gate
                    .push(matrix(&scratch.gate, tight(I)?.view(n, I)?, true)?);
                views.last.push(matrix_offset(
                    &scratch.norm,
                    tight(H)?.view(1, H)?,
                    true,
                    (n - 1) * H,
                )?);
            }
            tokens.resize(CAP, 0);
            Ok(Self {
                model,
                cache,
                histories,
                scratch,
                views,
                candidate: word_buffer(g, logit_rows * V)?,
                published: word_buffer(g, logit_rows * V)?,
                tokens,
                options,
                published_rows: 0,
                snapshots: snapshots::allocate(g, snapshots)?,
                diagnostic_rows: 0,
                diagnostic_start: 0,
                #[cfg(test)]
                test_fault: TestFault::None,
                #[cfg(test)]
                test_phase_a: None,
                memory: Memory {
                    cache: model.cache_bytes,
                    convolution_history: history_bytes,
                    scratch: scratch_bytes,
                    logits: logits_bytes,
                    tokens: token_bytes,
                    diagnostic_gpu,
                    diagnostic_readback_reserved: readback,
                    metadata,
                    external_resident: options.external_resident_bytes,
                    admitted,
                    cold_peak: model.memory.cold_peak.max(admitted),
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
        Ok(&self.tokens[..self.cache.offset()])
    }
    pub fn reset(&mut self) -> Result<()> {
        // Validate every destination before Set checks terminal state and both
        // counters. Once reset succeeds, all remaining stores are infallible.
        for h in &self.histories {
            for b in [&h.old, &h.candidate] {
                if b.len != 2 * H + GUARD
                    || !std::ptr::eq(&*b.raw.device(), &*self.model.gpu.device)
                {
                    return Err(Error::Options("LFM2 history reset extent/device"));
                }
            }
        }
        self.cache.reset()?;
        for h in &mut self.histories {
            h.old.with_mut(|v| v[..2 * H].fill(0));
            h.candidate.with_mut(|v| v[..2 * H].fill(0));
        }
        self.published_rows = 0;
        self.diagnostic_rows = 0;
        Ok(())
    }
    pub fn copy_logits(&self, out: &mut [f32]) -> Result<()> {
        if self.is_poisoned() {
            return Err(Error::ReplayRequired);
        }
        if self.published_rows == 0 || out.len() != V {
            return Err(Error::Options("LFM2 logits unavailable/length"));
        }
        let start = (self.published_rows - 1) * V;
        self.published.with_slice(|v| {
            for (dst, &bits) in out.iter_mut().zip(&v[start..start + V]) {
                *dst = f32::from_bits(bits);
            }
        });
        Ok(())
    }
    pub fn copy_all_logits(&self, out: &mut [f32]) -> Result<()> {
        if self.is_poisoned() {
            return Err(Error::ReplayRequired);
        }
        if !self.options.all_logits
            || self.published_rows == 0
            || out.len() != self.published_rows * V
        {
            return Err(Error::Options("LFM2 all logits unavailable/length"));
        }
        self.published.with_slice(|v| {
            for (dst, &bits) in out.iter_mut().zip(v) {
                *dst = f32::from_bits(bits);
            }
        });
        Ok(())
    }
    pub fn forward(&mut self, tokens: &[u32]) -> Result<()> {
        if self.is_poisoned() {
            return Err(Error::ReplayRequired);
        }
        let n = tokens.len();
        if n == 0
            || n > CAP
            || tokens.iter().any(|&v| v as usize >= V)
            || self
                .cache
                .offset()
                .checked_add(n)
                .is_none_or(|end| end > CAP)
        {
            return Err(Error::Options("LFM2 token/count/position admission"));
        }
        // Snapshots are scratch, not double-banked publication. Invalidate them
        // before any Phase-A mutation, even if begin subsequently fails.
        self.diagnostic_rows = 0;
        autoreleasepool(|_| {
            self.scratch
                .tokens
                .with_mut(|v| v[..n].copy_from_slice(tokens));
            let batch = self.cache.begin(&self.model.gpu, n)?;
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
                Ok(v) => v,
                Err(e) => {
                    if !self.is_poisoned() {
                        self.cache.discard()?;
                    }
                    return Err(e);
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
                Ok(v) => v,
                Err(e) => {
                    if !self.is_poisoned() {
                        self.cache.discard()?;
                    }
                    return Err(e);
                }
            };
            // All checks/callbacks/allocations precede publication. Fixed ranges
            // follow begin's validated offset/count and the immutable capacity.
            self.tokens[info.start..info.end].copy_from_slice(tokens);
            for h in &mut self.histories {
                std::mem::swap(&mut h.old, &mut h.candidate);
            }
            std::mem::swap(&mut self.published, &mut self.candidate);
            self.published_rows = if self.options.all_logits { n } else { 1 };
            self.diagnostic_start = info.start;
            self.diagnostic_rows = n;
            Ok(())
        })
    }
}
