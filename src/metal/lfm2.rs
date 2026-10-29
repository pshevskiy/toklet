//! Bounded resident LFM2 hybrid composition over original packed weights.
//! Direct BF16 gains, explicit Half stores, and one mixed transaction per chunk.
//! Cold selection supports the qualified tiny geometry and one bounded real
//! geometry. This is not a backend registry, text/app factory or a claim that
//! the real graph has passed numerical qualification.
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
mod real_capture;
#[cfg(test)]
mod real_gpu_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
const H: usize = 256;
#[cfg(test)]
const I: usize = 256;
#[cfg(test)]
const V: usize = 64;
#[cfg(test)]
const Q: usize = 256;
#[cfg(test)]
const K: usize = 128;
const D: usize = 64;
#[cfg(test)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Profile {
    Tiny,
    SelectedReal,
}
#[derive(Clone, Copy, Debug)]
struct Geometry {
    profile: Profile,
    h: usize,
    i: usize,
    v: usize,
    q: usize,
    k: usize,
    query_heads: usize,
    kv_heads: usize,
    layers: usize,
    attention: usize,
    convolution: usize,
    capacity: usize,
}
impl Geometry {
    fn kind(self, index: usize) -> Kind {
        let attention = match self.profile {
            Profile::Tiny => &[1][..],
            Profile::SelectedReal => &[2, 5, 8, 10, 12, 14][..],
        };
        match attention.iter().position(|&layer| layer == index) {
            Some(owner) => Kind::Attention(owner),
            None => {
                Kind::Convolution(index - attention.iter().filter(|&&layer| layer < index).count())
            }
        }
    }
    fn views(self) -> usize {
        self.capacity.min(B)
    }
    fn admit_step(self, offset: usize, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty()
            || tokens.len() > self.views()
            || tokens.iter().any(|&token| token as usize >= self.v)
            || offset
                .checked_add(tokens.len())
                .is_none_or(|end| end > self.capacity)
        {
            return Err(Error::Options("LFM2 token/count/position admission"));
        }
        Ok(())
    }
    fn stages(self) -> usize {
        3 + self.convolution * 17 + self.attention * 20
    }
    fn half_width(self) -> usize {
        (3 * self.h).max(self.i).max(self.q).max(self.k).max(self.h)
    }
    fn zero_bytes(self) -> Result<usize> {
        mul(
            add(self.v.max(3 * self.h).max(self.i).max(self.q), GUARD)?,
            2,
        )
    }
    fn norm_bytes(self) -> Result<usize> {
        mul(
            add(
                mul(2 * self.layers + 1, self.h + GUARD)?,
                mul(2 * self.attention, D + GUARD)?,
            )?,
            4,
        )
    }
    fn coefficient_bytes(self) -> Result<usize> {
        mul(mul(self.convolution, 3 * self.h + GUARD)?, 4)
    }
    fn history_bytes(self) -> Result<usize> {
        mul(mul(self.convolution, 2)?, mul(extent(2, self.h)?, 2)?)
    }
    fn scratch_bytes(self) -> Result<usize> {
        let half = add(
            mul(B, 8 * self.h + 2 * self.q + 3 * self.k + 3 * self.i)?,
            14 * GUARD,
        )?;
        add(
            mul(half, 2)?,
            mul(extent(self.query_heads * B, self.capacity)?, 8)?,
        )
    }
    fn cache_plan(self) -> Result<cache::Plan> {
        let owners = [cache::OwnerConfig {
            kv_heads: self.kv_heads,
            dim: D,
            window: None,
        }; 6];
        cache::Plan::new(self.capacity, &owners[..self.attention])
    }
}
fn geometry(c: &Config) -> Result<Geometry> {
    c.validate()
        .map_err(|_| Error::Options("LFM2 configuration"))?;
    if c.rms_norm_eps.to_bits() != EPSILON.to_bits()
        || c.rope_theta.to_bits() != THETA.to_bits()
        || c.head_dim != D
    {
        return Err(Error::Options("LFM2 fixed direct norm/rotary profile"));
    }
    let shape = (
        c.hidden_size,
        c.intermediate_size,
        c.vocab_size,
        c.num_layers,
        c.query_heads,
        c.kv_heads,
    );
    let (profile, attention, convolution) = match shape {
        (256, 256, 64, 3, 4, 2) if c.max_positions == 12 => (Profile::Tiny, 1, 2),
        (1024, 4608, 65536, 16, 16, 8) if (1..=512).contains(&c.max_positions) => {
            (Profile::SelectedReal, 6, 10)
        }
        _ => return Err(Error::Options("LFM2 bounded tiny/selected-real geometry")),
    };
    let g = Geometry {
        profile,
        h: c.hidden_size,
        i: c.intermediate_size,
        v: c.vocab_size,
        q: c.query_heads * D,
        k: c.kv_heads * D,
        query_heads: c.query_heads,
        kv_heads: c.kv_heads,
        layers: c.num_layers,
        attention,
        convolution,
        capacity: c.max_positions,
    };
    // Shared leaves admit these concrete operand layouts cold; they are not
    // additional allocations in the model/session ledger.
    rms::Plan::new(B, g.h, g.h, g.h, EPSILON)?.operand_bytes()?;
    rms::Plan::new(B * g.query_heads, D, D, D, EPSILON)?.operand_bytes()?;
    conv::Plan::new(B, g.h, g.h, g.h)?.operand_bytes()?;
    attention::Plan::new_gqa(g.capacity, g.query_heads, g.kv_heads, None, 0, g.views())?
        .operand_bytes()?;
    g.cache_plan()?.session_bytes()?;
    Ok(g)
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
    stage_base: usize,
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
    geometry: Geometry,
    embedding: Tensor,
    layers: Vec<Layer>,
    attention: Vec<Attention>,
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
        let g = geometry(&config)?;
        if weights.layers.len() != g.layers
            || weights.attention.len() != g.attention
            || weights.convolution.len() != g.convolution
        {
            return Err(Error::Options("LFM2 owner inventory"));
        }
        let mut packed = matrix_bytes(&weights.embedding, g.v, g.h)?;
        let mut source_minimum = weights.embedding.layout().total_bytes();
        for (index, layer) in weights.layers.iter().enumerate() {
            let kind = g.kind(index);
            if layer.kind != kind {
                return Err(Error::Options("LFM2 fixed owner routes"));
            }
            for (m, r, c) in [
                (&layer.gate, g.i, g.h),
                (&layer.up, g.i, g.h),
                (&layer.down, g.h, g.i),
            ] {
                packed = add(packed, matrix_bytes(m, r, c)?)?;
                source_minimum = add(source_minimum, m.layout().total_bytes())?;
            }
            raw(&layer.operator_gamma, g.h)?;
            raw(&layer.ffn_gamma, g.h)?;
            source_minimum = add(
                source_minimum,
                mul(
                    add(layer.operator_gamma.capacity(), layer.ffn_gamma.capacity())?,
                    2,
                )?,
            )?;
        }
        for a in &weights.attention {
            for (m, r, c) in [
                (&a.q, g.q, g.h),
                (&a.k, g.k, g.h),
                (&a.v, g.k, g.h),
                (&a.o, g.h, g.q),
            ] {
                packed = add(packed, matrix_bytes(m, r, c)?)?;
                source_minimum = add(source_minimum, m.layout().total_bytes())?;
            }
            raw(&a.q_gamma, D)?;
            raw(&a.k_gamma, D)?;
            source_minimum = add(
                source_minimum,
                mul(add(a.q_gamma.capacity(), a.k_gamma.capacity())?, 2)?,
            )?;
        }
        for conv in &weights.convolution {
            for (m, r, c) in [(&conv.bcx, 3 * g.h, g.h), (&conv.output, g.h, g.h)] {
                packed = add(packed, matrix_bytes(m, r, c)?)?;
                source_minimum = add(source_minimum, m.layout().total_bytes())?;
            }
            raw(&conv.coefficients, 3 * g.h)?;
            source_minimum = add(source_minimum, mul(conv.coefficients.capacity(), 2)?)?;
        }
        raw(&weights.final_gamma, g.h)?;
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
            .try_reserve_exact(g.layers)
            .map_err(|_| Error::Options("LFM2 layer metadata allocation"))?;
        let mut convolution = Vec::new();
        convolution
            .try_reserve_exact(g.convolution)
            .map_err(|_| Error::Options("LFM2 convolution metadata allocation"))?;
        let mut attention = Vec::new();
        attention
            .try_reserve_exact(g.attention)
            .map_err(|_| Error::Options("LFM2 attention metadata allocation"))?;
        let metadata = add(
            size_of::<Self>(),
            add(
                add(
                    mul(layers.capacity(), size_of::<Layer>())?,
                    mul(convolution.capacity(), size_of::<Convolution>())?,
                )?,
                mul(attention.capacity(), size_of::<Attention>())?,
            )?,
        )?;
        let norms = g.norm_bytes()?;
        let coefficients = g.coefficient_bytes()?;
        // The selected real tied head is wider than BCx and the FFN outputs.
        let zero = g.zero_bytes()?;
        let admitted = [packed, norms, coefficients, zero, metadata, ALLOWANCE]
            .into_iter()
            .try_fold(0, add)?;
        let cold_peak = add(admitted, source_bytes)?;
        admit(cold_peak, memory_limit)?;
        autoreleasepool(|_| {
            let gpu = Gpu::new()?;
            for w in [g.h, g.i, g.q, g.k, 3 * g.h] {
                tight(w)?;
            }
            let maximum = gpu.device.maxBufferLength();
            // Matrix sizes were checked cold. Each upload also independently
            // validates actual allocation/device bounds in Buffer::new.
            for bytes in [packed, norms, coefficients, zero] {
                validate_buffer_bytes(bytes, maximum)?;
            }
            let plan = g.cache_plan()?;
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
            let mut stage_base = 1;
            for l in weights.layers {
                layers.push(Layer {
                    kind: l.kind,
                    stage_base,
                    gate: packed_upload(&gpu, &l.gate, &pipelines)?,
                    up: packed_upload(&gpu, &l.up, &pipelines)?,
                    down: packed_upload(&gpu, &l.down, &pipelines)?,
                    operator_norm: upload_raw(&gpu, &l.operator_gamma)?,
                    ffn_norm: upload_raw(&gpu, &l.ffn_gamma)?,
                });
                stage_base += match l.kind {
                    Kind::Convolution(_) => 17,
                    Kind::Attention(_) => 20,
                };
            }
            for a in &weights.attention {
                attention.push(Attention {
                    q: packed_upload(&gpu, &a.q, &pipelines)?,
                    k: packed_upload(&gpu, &a.k, &pipelines)?,
                    v: packed_upload(&gpu, &a.v, &pipelines)?,
                    o: packed_upload(&gpu, &a.o, &pipelines)?,
                    q_norm: upload_raw(&gpu, &a.q_gamma)?,
                    k_norm: upload_raw(&gpu, &a.k_gamma)?,
                });
            }
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
                geometry: g,
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
    histories: Vec<History>,
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
fn bank(count: usize) -> Result<Vec<Retained<MPSMatrix>>> {
    let mut v = Vec::new();
    v.try_reserve_exact(count)
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
        let g = model.geometry;
        if options.diagnostics && !options.all_logits {
            return Err(Error::Options("LFM2 diagnostics require all-row logits"));
        }
        let mut views = Views {
            norm: bank(g.views())?,
            context: bank(g.views())?,
            u: bank(g.views())?,
            gate: bank(g.views())?,
            last: bank(g.views())?,
        };
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(g.capacity)
            .map_err(|_| Error::Options("LFM2 token allocation"))?;
        let mut histories = Vec::new();
        histories
            .try_reserve_exact(g.convolution)
            .map_err(|_| Error::Options("LFM2 history metadata allocation"))?;
        let mut metadata = add(
            model.memory.metadata,
            size_of::<Self>() - size_of::<cache::Set<'_>>(),
        )?;
        metadata = add(metadata, mul(histories.capacity(), size_of::<History>())?)?;
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
            snapshots::plan(options.diagnostics, g)?;
        metadata = add(metadata, snapshot_metadata)?;
        let scratch_bytes = g.scratch_bytes()?;
        let history_bytes = g.history_bytes()?;
        let logit_rows = if options.all_logits { B } else { 1 };
        let logits_bytes = mul(extent(logit_rows, g.v)?, 8)?;
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
            (extent(B, g.half_width())?, 2),
            (extent(g.query_heads * B, g.capacity)?, 4),
            (extent(logit_rows, g.v)?, 4),
            (2 * g.h + GUARD, 2),
            (B + GUARD, 4),
        ] {
            validate_buffer_bytes(mul(n, scalar)?, model.gpu.device.maxBufferLength())?;
        }
        snapshots::validate(&snapshots, model.gpu.device.maxBufferLength())?;
        autoreleasepool(|_| {
            let gpu = &model.gpu;
            let cache = cache::Set::new(gpu, &model.cache, model.cache_bytes)?;
            for _ in 0..g.convolution {
                histories.push(History {
                    old: half_buffer(gpu, 2, g.h)?,
                    candidate: half_buffer(gpu, 2, g.h)?,
                });
            }
            let scratch = Scratch {
                hidden: half_buffer(gpu, B, g.h)?,
                norm: half_buffer(gpu, B, g.h)?,
                branch: half_buffer(gpu, B, g.h)?,
                bcx: half_buffer(gpu, B, 3 * g.h)?,
                u: half_buffer(gpu, B, g.h)?,
                z: half_buffer(gpu, B, g.h)?,
                q: half_buffer(gpu, B, g.q)?,
                work: half_buffer(gpu, B, g.q)?,
                k: half_buffer(gpu, B, g.k)?,
                k_norm: half_buffer(gpu, B, g.k)?,
                v: half_buffer(gpu, B, g.k)?,
                gate: half_buffer(gpu, B, g.i)?,
                up: half_buffer(gpu, B, g.i)?,
                activated: half_buffer(gpu, B, g.i)?,
                scores: word_buffer(gpu, g.query_heads * B * g.capacity)?,
                probabilities: word_buffer(gpu, g.query_heads * B * g.capacity)?,
                tokens: word_buffer(gpu, B)?,
            };
            for n in 1..=g.views() {
                views
                    .norm
                    .push(matrix(&scratch.norm, tight(g.h)?.view(n, g.h)?, true)?);
                views
                    .context
                    .push(matrix(&scratch.work, tight(g.q)?.view(n, g.q)?, true)?);
                views
                    .u
                    .push(matrix(&scratch.u, tight(g.h)?.view(n, g.h)?, true)?);
                views
                    .gate
                    .push(matrix(&scratch.gate, tight(g.i)?.view(n, g.i)?, true)?);
                views.last.push(matrix_offset(
                    &scratch.norm,
                    tight(g.h)?.view(1, g.h)?,
                    true,
                    (n - 1) * g.h,
                )?);
            }
            tokens.resize(g.capacity, 0);
            Ok(Self {
                model,
                cache,
                histories,
                scratch,
                views,
                candidate: word_buffer(gpu, logit_rows * g.v)?,
                published: word_buffer(gpu, logit_rows * g.v)?,
                tokens,
                options,
                published_rows: 0,
                snapshots: snapshots::allocate(gpu, snapshots)?,
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
    fn history_preflight(&self) -> Result<()> {
        let g = self.model.geometry;
        if self.histories.len() != g.convolution || !self.histories.len().is_multiple_of(2) {
            return Err(Error::Options("LFM2 history inventory"));
        }
        for (owner, h) in self.histories.iter().enumerate() {
            for b in [&h.old, &h.candidate] {
                if b.len != extent(2, g.h)?
                    || b.raw.length() != mul(b.len, 2)?
                    || !std::ptr::eq(&*b.raw.device(), &*self.model.gpu.device)
                {
                    return Err(Error::Options("LFM2 history reset extent/device"));
                }
                for previous in &self.histories[..owner] {
                    if [&previous.old, &previous.candidate]
                        .iter()
                        .any(|p| std::ptr::eq(&*b.raw, &*p.raw))
                    {
                        return Err(Error::Options("LFM2 history owner alias"));
                    }
                }
            }
            if std::ptr::eq(&*h.old.raw, &*h.candidate.raw) {
                return Err(Error::Options("LFM2 history bank alias"));
            }
        }
        Ok(())
    }
    pub fn reset(&mut self) -> Result<()> {
        let g = self.model.geometry;
        self.history_preflight()?;
        // Cache checks terminal state/counters after every history destination
        // is validated. All remaining stores and invalidations are infallible.
        self.cache.reset()?;
        for h in &mut self.histories {
            h.old.with_mut(|v| v[..2 * g.h].fill(0));
            h.candidate.with_mut(|v| v[..2 * g.h].fill(0));
        }
        self.published_rows = 0;
        self.diagnostic_rows = 0;
        Ok(())
    }
    pub fn copy_logits(&self, out: &mut [f32]) -> Result<()> {
        let g = self.model.geometry;
        if self.is_poisoned() {
            return Err(Error::ReplayRequired);
        }
        if self.published_rows == 0 || out.len() != g.v {
            return Err(Error::Options("LFM2 logits unavailable/length"));
        }
        let start = (self.published_rows - 1) * g.v;
        self.published.with_slice(|v| {
            for (dst, &bits) in out.iter_mut().zip(&v[start..start + g.v]) {
                *dst = f32::from_bits(bits);
            }
        });
        Ok(())
    }
    pub fn copy_all_logits(&self, out: &mut [f32]) -> Result<()> {
        let g = self.model.geometry;
        if self.is_poisoned() {
            return Err(Error::ReplayRequired);
        }
        if !self.options.all_logits
            || self.published_rows == 0
            || out.len() != self.published_rows * g.v
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
        let g = self.model.geometry;
        if self.is_poisoned() {
            return Err(Error::ReplayRequired);
        }
        let n = tokens.len();
        g.admit_step(self.cache.offset(), tokens)?;
        self.history_preflight()?;
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
