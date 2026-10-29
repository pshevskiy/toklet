//! Bounded scalar Conv→Attention→Conv composition over original packed weights.
//!
//! This is the bounded fixture-sized numerical primitive, not a real-model
//! source factory, native Half path or production text/session admission.
//! Gains and convolution coefficients are exact BF16→FP32 widenings (no +1).
use super::packed_operations::{Operations, Scalar};
use super::short_conv3::History;
use super::window_cache::WindowKvCache;
use super::{Config, Error, allocation_bytes, total_bytes, zeros};
use crate::cpu::{self, finite, length, product};
use crate::mlx::affine::{Matrix, MetadataDtype};
use half::bf16;

mod forward;
const LAYERS: usize = 3;
const KV_OWNERS: usize = 1;
const CONV_OWNERS: usize = 2;
const MAX_CHUNK: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Convolution(usize),
    Attention(usize),
}
pub struct LayerWeights<'a> {
    pub kind: Kind,
    pub gate: Matrix<'a>,
    pub up: Matrix<'a>,
    pub down: Matrix<'a>,
    pub operator_gamma: Vec<bf16>,
    pub ffn_gamma: Vec<bf16>,
}
pub struct AttentionWeights<'a> {
    pub q: Matrix<'a>,
    pub k: Matrix<'a>,
    pub v: Matrix<'a>,
    pub o: Matrix<'a>,
    pub q_gamma: Vec<bf16>,
    pub k_gamma: Vec<bf16>,
}
pub struct ConvWeights<'a> {
    /// One fused physical parent, output rows B then C then x.
    pub bcx: Matrix<'a>,
    pub output: Matrix<'a>,
    /// Original [H,3,1] flattened with oldest-to-current taps, no bias.
    pub coefficients: Vec<bf16>,
}
pub struct Weights<'a> {
    pub embedding: Matrix<'a>,
    pub layers: Vec<LayerWeights<'a>>,
    pub attention: Vec<AttentionWeights<'a>>,
    pub convolution: Vec<ConvWeights<'a>>,
    pub final_gamma: Vec<bf16>,
}
struct Layer<'a> {
    kind: Kind,
    gate: Matrix<'a>,
    up: Matrix<'a>,
    down: Matrix<'a>,
    operator_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
}
struct Attention<'a> {
    q: Matrix<'a>,
    k: Matrix<'a>,
    v: Matrix<'a>,
    o: Matrix<'a>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}
struct Convolution<'a> {
    bcx: Matrix<'a>,
    output: Matrix<'a>,
    coefficients: Vec<f32>,
}
pub struct Model<'a> {
    config: Config,
    embedding: Matrix<'a>,
    layers: Vec<Layer<'a>>,
    attention: Vec<Attention<'a>>,
    convolution: Vec<Convolution<'a>>,
    final_norm: Vec<f32>,
    bytes: usize,
}
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(cpu::Error::SizeOverflow.into())
}
fn matrix(m: &Matrix<'_>, rows: usize, columns: usize) -> Result<usize, Error> {
    let l = m.layout();
    if l.rows() != rows
        || l.columns() != columns
        || l.bits() != 4
        || l.group_size() != 64
        || l.metadata_dtype() != MetadataDtype::BF16
    {
        return Err(Error::InvalidConfig(
            "LFM2 scalar Q4/G64/BF16 matrix geometry",
        ));
    }
    Ok(l.total_bytes())
}
fn raw(values: &[bf16], n: usize) -> Result<(), Error> {
    length("LFM2 direct BF16 coefficient", values.len(), n)?;
    for (i, v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(cpu::Error::NonFinite {
                tensor: "LFM2 direct BF16 coefficient",
                index: i,
            }
            .into());
        }
    }
    Ok(())
}
fn widen(values: Vec<bf16>) -> Result<Vec<f32>, Error> {
    let mut result = zeros(values.len())?;
    for (out, v) in result.iter_mut().zip(values) {
        *out = v.to_f32();
    }
    Ok(result)
}
fn reserved<T>(n: usize) -> Result<Vec<T>, Error> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| Error::Allocation)?;
    Ok(v)
}
impl<'a> Model<'a> {
    pub fn new(config: Config, weights: Weights<'a>) -> Result<Self, Error> {
        config.validate()?;
        if (
            config.hidden_size,
            config.intermediate_size,
            config.vocab_size,
            config.num_layers,
            config.query_heads,
            config.kv_heads,
            config.head_dim,
            config.max_positions,
        ) != (256, 256, 64, 3, 4, 2, 64, 12)
            || config.rms_norm_eps != 1e-5_f32
            || config.rope_theta != 1e6_f32
        {
            return Err(Error::InvalidConfig("LFM2 scalar tiny profile"));
        }
        length("LFM2 layers", weights.layers.len(), LAYERS)?;
        length("LFM2 attention owners", weights.attention.len(), KV_OWNERS)?;
        length(
            "LFM2 convolution owners",
            weights.convolution.len(),
            CONV_OWNERS,
        )?;
        let c = &config;
        let h = c.hidden_size;
        let f = c.intermediate_size;
        let q = product(c.query_heads, c.head_dim)?;
        let kv = product(c.kv_heads, c.head_dim)?;
        let mut tensors = matrix(&weights.embedding, c.vocab_size, h)?;
        raw(&weights.final_gamma, h)?;
        for (l, kind) in weights.layers.iter().zip([
            Kind::Convolution(0),
            Kind::Attention(0),
            Kind::Convolution(1),
        ]) {
            if l.kind != kind {
                return Err(Error::InvalidConfig("LFM2 scalar layer route"));
            }
            for (m, r, col) in [(&l.gate, f, h), (&l.up, f, h), (&l.down, h, f)] {
                tensors = add(tensors, matrix(m, r, col)?)?;
            }
            raw(&l.operator_gamma, h)?;
            raw(&l.ffn_gamma, h)?;
        }
        for a in &weights.attention {
            for (m, r, col) in [(&a.q, q, h), (&a.k, kv, h), (&a.v, kv, h), (&a.o, h, q)] {
                tensors = add(tensors, matrix(m, r, col)?)?;
            }
            raw(&a.q_gamma, c.head_dim)?;
            raw(&a.k_gamma, c.head_dim)?;
        }
        for a in &weights.convolution {
            tensors = add(tensors, matrix(&a.bcx, product(3, h)?, h)?)?;
            tensors = add(tensors, matrix(&a.output, h, h)?)?;
            raw(&a.coefficients, product(3, h)?)?;
        }
        // All original shapes/finite coefficients validated before conversion.
        let mut layers = reserved(LAYERS)?;
        for l in weights.layers {
            layers.push(Layer {
                kind: l.kind,
                gate: l.gate,
                up: l.up,
                down: l.down,
                operator_norm: widen(l.operator_gamma)?,
                ffn_norm: widen(l.ffn_gamma)?,
            });
        }
        let mut attention = reserved(KV_OWNERS)?;
        for w in weights.attention {
            attention.push(Attention {
                q: w.q,
                k: w.k,
                v: w.v,
                o: w.o,
                q_norm: widen(w.q_gamma)?,
                k_norm: widen(w.k_gamma)?,
            });
        }
        let mut convolution = reserved(CONV_OWNERS)?;
        for w in weights.convolution {
            convolution.push(Convolution {
                bcx: w.bcx,
                output: w.output,
                coefficients: widen(w.coefficients)?,
            });
        }
        let final_norm = widen(weights.final_gamma)?;
        let mut bytes = add(std::mem::size_of::<Self>(), tensors)?;
        bytes = add(
            bytes,
            product(layers.capacity(), std::mem::size_of::<Layer<'_>>())?,
        )?;
        bytes = add(
            bytes,
            product(attention.capacity(), std::mem::size_of::<Attention<'_>>())?,
        )?;
        bytes = add(
            bytes,
            product(
                convolution.capacity(),
                std::mem::size_of::<Convolution<'_>>(),
            )?,
        )?;
        bytes = add(bytes, allocation_bytes(final_norm.capacity())?)?;
        for l in &layers {
            bytes = add(
                bytes,
                total_bytes(&[l.operator_norm.capacity(), l.ffn_norm.capacity()])?,
            )?;
        }
        for a in &attention {
            bytes = add(
                bytes,
                total_bytes(&[a.q_norm.capacity(), a.k_norm.capacity()])?,
            )?;
        }
        for a in &convolution {
            bytes = add(bytes, allocation_bytes(a.coefficients.capacity())?)?;
        }
        Ok(Self {
            config,
            embedding: weights.embedding,
            layers,
            attention,
            convolution,
            final_norm,
            bytes,
        })
    }
    /// Accounted per-field borrowed extents plus owned widened coefficients/metadata.
    /// The tied embedding is one field; arbitrary caller aliases conservatively
    /// count again rather than claiming unique physical allocation accounting.
    /// Caller container padding, conversion overlap, diagnostics and process
    /// owners are separate. This is not a full application memory estimate.
    pub fn model_bytes(&self) -> usize {
        self.bytes
    }
    pub fn planned_session_bytes(&self, chunk: usize) -> Result<usize, Error> {
        let c = &self.config;
        if chunk == 0 || chunk > MAX_CHUNK || chunk > c.max_positions {
            return Err(Error::InvalidConfig("LFM2 scalar chunk capacity"));
        }
        let mut bytes = std::mem::size_of::<Session<'_, '_>>();
        bytes = add(
            bytes,
            product(
                KV_OWNERS,
                WindowKvCache::planned_bytes(
                    c.kv_heads,
                    c.head_dim,
                    c.max_positions,
                    c.max_positions,
                    None,
                )?,
            )?,
        )?;
        bytes = add(
            bytes,
            product(
                CONV_OWNERS,
                History::planned_bytes(c.hidden_size, c.max_positions)?,
            )?,
        )?;
        bytes = add(bytes, product(KV_OWNERS, std::mem::size_of::<KvStage>())?)?;
        bytes = add(
            bytes,
            product(CONV_OWNERS, std::mem::size_of::<Vec<f32>>())?,
        )?;
        bytes = add(
            bytes,
            allocation_bytes(product(
                product(product(KV_OWNERS, chunk)?, c.kv_heads)?,
                product(c.head_dim, 2)?,
            )?)?,
        )?;
        bytes = add(
            bytes,
            allocation_bytes(product(product(CONV_OWNERS, 2)?, c.hidden_size)?)?,
        )?;
        bytes = add(bytes, allocation_bytes(c.max_positions)?)?;
        add(bytes, total_bytes(&Scratch::counts(c, chunk)?)?)
    }
    pub fn new_session(
        &self,
        chunk: usize,
        budget: usize,
    ) -> Result<Session<'_, 'a>, crate::session::Error> {
        let required = add(self.bytes, self.planned_session_bytes(chunk)?)?;
        if required > budget {
            return Err(crate::session::Error::MemoryBudget {
                required,
                limit: budget,
            });
        }
        let c = &self.config;
        let mut caches = reserved(KV_OWNERS)?;
        let mut histories = reserved(CONV_OWNERS)?;
        let mut kv_stage = reserved(KV_OWNERS)?;
        let mut conv_stage = reserved(CONV_OWNERS)?;
        let kv = product(product(chunk, c.kv_heads).map_err(Error::from)?, c.head_dim)
            .map_err(Error::from)?;
        for _ in 0..KV_OWNERS {
            caches.push(WindowKvCache::new(
                c.kv_heads,
                c.head_dim,
                c.max_positions,
                c.max_positions,
                None,
            )?);
            kv_stage.push(KvStage {
                k: zeros(kv)?,
                v: zeros(kv)?,
            });
        }
        for _ in 0..CONV_OWNERS {
            histories.push(History::new(c.hidden_size, c.max_positions)?);
            conv_stage.push(zeros(2 * c.hidden_size)?);
        }
        let mut tokens = reserved(c.max_positions)?;
        tokens.resize(c.max_positions, 0);
        let s = Session {
            model: self,
            operations: Scalar,
            caches,
            histories,
            kv_stage,
            conv_stage,
            scratch: Scratch::new(c, chunk)?,
            tokens,
            committed: 0,
            generation: 0,
            published_rows: 0,
            chunk,
            #[cfg(test)]
            fault: None,
        };
        let actual = add(self.bytes, s.allocated_bytes()?)?;
        if actual > budget {
            return Err(crate::session::Error::MemoryBudget {
                required: actual,
                limit: budget,
            });
        }
        Ok(s)
    }
}
struct KvStage {
    k: Vec<f32>,
    v: Vec<f32>,
}
struct Scratch {
    hidden: Vec<f32>,
    next: Vec<f32>,
    norm: Vec<f32>,
    bcx: Vec<f32>,
    u: Vec<f32>,
    conv: Vec<f32>,
    context: Vec<f32>,
    projected: Vec<f32>,
    q: Vec<f32>,
    qn: Vec<f32>,
    k: Vec<f32>,
    kn: Vec<f32>,
    v: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    gated: Vec<f32>,
    scores: Vec<f32>,
    probabilities: Vec<f32>,
    logits: Vec<f32>,
    published: Vec<f32>,
    work: Vec<f32>,
}
impl Scratch {
    fn counts(c: &Config, n: usize) -> Result<[usize; 21], Error> {
        let h = product(n, c.hidden_size)?;
        let q = product(product(n, c.query_heads)?, c.head_dim)?;
        let kv = product(product(n, c.kv_heads)?, c.head_dim)?;
        let ff = product(n, c.intermediate_size)?;
        let scores = product(product(n, c.query_heads)?, c.max_positions)?;
        let logits = product(n, c.vocab_size)?;
        Ok([
            h,
            h,
            h,
            product(3, h)?,
            h,
            h,
            q,
            h,
            q,
            q,
            kv,
            kv,
            kv,
            ff,
            ff,
            ff,
            scores,
            scores,
            logits,
            logits,
            product(2, c.hidden_size)?,
        ])
    }
    fn new(c: &Config, n: usize) -> Result<Self, Error> {
        let [
            h,
            nx,
            norm,
            bcx,
            u,
            conv,
            ct,
            p,
            q,
            qn,
            k,
            kn,
            v,
            g,
            up,
            gd,
            sc,
            pr,
            lo,
            publo,
            work,
        ] = Self::counts(c, n)?;
        Ok(Self {
            hidden: zeros(h)?,
            next: zeros(nx)?,
            norm: zeros(norm)?,
            bcx: zeros(bcx)?,
            u: zeros(u)?,
            conv: zeros(conv)?,
            context: zeros(ct)?,
            projected: zeros(p)?,
            q: zeros(q)?,
            qn: zeros(qn)?,
            k: zeros(k)?,
            kn: zeros(kn)?,
            v: zeros(v)?,
            gate: zeros(g)?,
            up: zeros(up)?,
            gated: zeros(gd)?,
            scores: zeros(sc)?,
            probabilities: zeros(pr)?,
            logits: zeros(lo)?,
            published: zeros(publo)?,
            work: zeros(work)?,
        })
    }
    fn capacities(&self) -> [usize; 21] {
        [
            &self.hidden,
            &self.next,
            &self.norm,
            &self.bcx,
            &self.u,
            &self.conv,
            &self.context,
            &self.projected,
            &self.q,
            &self.qn,
            &self.k,
            &self.kn,
            &self.v,
            &self.gate,
            &self.up,
            &self.gated,
            &self.scores,
            &self.probabilities,
            &self.logits,
            &self.published,
            &self.work,
        ]
        .map(Vec::capacity)
    }
}
/// Append-only scalar numerical chunks. No dense prefix rewind, text, pending
/// sampled-token machinery or implicit model/state sharing is provided.
pub struct Session<'model, 'weights> {
    model: &'model Model<'weights>,
    operations: Scalar,
    caches: Vec<WindowKvCache>,
    histories: Vec<History>,
    kv_stage: Vec<KvStage>,
    conv_stage: Vec<Vec<f32>>,
    scratch: Scratch,
    tokens: Vec<u32>,
    committed: usize,
    generation: u64,
    published_rows: usize,
    chunk: usize,
    #[cfg(test)]
    fault: Option<usize>,
}
impl Session<'_, '_> {
    pub fn tokens(&self) -> &[u32] {
        &self.tokens[..self.committed]
    }
    pub fn logits(&self) -> &[f32] {
        &self.scratch.published[..self.published_rows * self.model.config.vocab_size]
    }
    pub fn allocated_bytes(&self) -> Result<usize, Error> {
        let mut bytes = std::mem::size_of::<Self>();
        // Owner structs are already included by each owner's allocated_bytes.
        bytes = add(
            bytes,
            product(
                self.caches.capacity() - self.caches.len(),
                std::mem::size_of::<WindowKvCache>(),
            )?,
        )?;
        bytes = add(
            bytes,
            product(
                self.histories.capacity() - self.histories.len(),
                std::mem::size_of::<History>(),
            )?,
        )?;
        for k in &self.caches {
            bytes = add(bytes, k.allocated_bytes())?;
        }
        for h in &self.histories {
            bytes = add(bytes, h.allocated_bytes()?)?;
        }
        bytes = add(
            bytes,
            product(self.kv_stage.capacity(), std::mem::size_of::<KvStage>())?,
        )?;
        bytes = add(
            bytes,
            product(self.conv_stage.capacity(), std::mem::size_of::<Vec<f32>>())?,
        )?;
        for s in &self.kv_stage {
            bytes = add(bytes, total_bytes(&[s.k.capacity(), s.v.capacity()])?)?;
        }
        for s in &self.conv_stage {
            bytes = add(bytes, allocation_bytes(s.capacity())?)?;
        }
        bytes = add(bytes, allocation_bytes(self.tokens.capacity())?)?;
        add(bytes, total_bytes(&self.scratch.capacities())?)
    }
    pub fn reset(&mut self) -> Result<(), Error> {
        self.check_inventory()?;
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(cpu::Error::SizeOverflow)?;
        for k in &self.caches {
            k.validate_reset()?;
        }
        for h in &self.histories {
            h.validate_reset()?;
        }
        // All families and session checked. No fallible operation after here.
        for k in &mut self.caches {
            k.reset_validated();
        }
        for h in &mut self.histories {
            h.reset_validated();
        }
        self.committed = 0;
        self.generation = generation;
        self.published_rows = 0;
        Ok(())
    }
    fn check_inventory(&self) -> Result<(), Error> {
        if self.caches.len() != KV_OWNERS
            || self.histories.len() != CONV_OWNERS
            || self.kv_stage.len() != KV_OWNERS
            || self.conv_stage.len() != CONV_OWNERS
            || self
                .caches
                .iter()
                .any(|k| k.offset() != self.committed || k.generation() != self.generation)
            || self
                .histories
                .iter()
                .any(|h| h.offset() != self.committed || h.generation() != self.generation)
        {
            return Err(Error::InvalidConfig("LFM2 mixed committed inventory"));
        }
        Ok(())
    }
    pub fn forward(&mut self, tokens: &[u32]) -> Result<&[f32], Error> {
        forward::execute::<false, _>(self, tokens, &mut |_| Ok(()))?;
        Ok(self.logits())
    }
}
#[cfg(test)]
mod tests;
