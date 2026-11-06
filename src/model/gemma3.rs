//! Concrete FP32 Gemma3 packed graph. Raw BF16 gamma becomes an owned FP32
//! unit-offset coefficient once at construction. This is not a Half profile.
//! Source loading/text eligibility remain outside this numerical primitive.
pub use super::packed_operations::{MatrixSource, Operations, Scalar};
use super::shared_kv::{KvKind, LayerKvGeometry, SharedKvPlan};
use super::window_cache::{WindowKvCache, reset_batch};
use super::{Config, Error, allocation_bytes, total_bytes, zeros};
use crate::{
    cpu::{self, finite, length, product},
    mlx::affine::MetadataDtype,
};
use half::bf16;

mod forward;
const MAX_LAYERS: usize = 32;
const MAX_CAPACITY: usize = 544;
const MAX_CHUNK: usize = 32;

#[derive(Clone, Copy, Debug)]
pub struct LayerSpec {
    pub kind: KvKind,
    pub theta: f32,
}
pub struct Gamma {
    pub input: Vec<bf16>,
    pub post_attention: Vec<bf16>,
    pub pre_ff: Vec<bf16>,
    pub post_ff: Vec<bf16>,
    pub q: Vec<bf16>,
    pub k: Vec<bf16>,
}
pub struct LayerWeights<M> {
    pub q: M,
    pub k: M,
    pub v: M,
    pub o: M,
    pub gate: M,
    pub up: M,
    pub down: M,
    pub gamma: Gamma,
}
pub struct Weights<M> {
    pub embedding: M,
    pub layers: Vec<LayerWeights<M>>,
    pub final_gamma: Vec<bf16>,
}
struct Norms {
    input: Vec<f32>,
    post_attention: Vec<f32>,
    pre_ff: Vec<f32>,
    post_ff: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
}
struct Layer<M> {
    q: M,
    k: M,
    v: M,
    o: M,
    gate: M,
    up: M,
    down: M,
    norms: Norms,
    spec: LayerSpec,
}
pub struct Model<M> {
    config: Config,
    embedding: M,
    layers: Vec<Layer<M>>,
    final_norm: Vec<f32>,
    plan: SharedKvPlan,
    window: usize,
    embedding_scale: f32,
    bytes: usize,
}

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(cpu::Error::SizeOverflow.into())
}
fn offset_norm(raw: Vec<bf16>, width: usize) -> Result<Vec<f32>, Error> {
    length("Gemma3 raw gamma", raw.len(), width)?;
    let mut result = zeros(width)?;
    for (i, (out, gamma)) in result.iter_mut().zip(raw).enumerate() {
        let gamma = gamma.to_f32();
        if !gamma.is_finite() {
            return Err(cpu::Error::NonFinite {
                tensor: "Gemma3 raw gamma",
                index: i,
            }
            .into());
        }
        *out = 1.0 + gamma;
        if !out.is_finite() {
            return Err(cpu::Error::NonFinite {
                tensor: "Gemma3 offset coefficient",
                index: i,
            }
            .into());
        }
    }
    Ok(result)
}
fn matrix(m: &impl MatrixSource, rows: usize, columns: usize) -> Result<usize, Error> {
    let super::packed_operations::MatrixDescriptor::Affine(l) = m.descriptor() else {
        return Err(Error::InvalidConfig(
            "Gemma3 packed4/group64/BF16 matrix shape",
        ));
    };
    if l.rows() != rows
        || l.columns() != columns
        || l.bits() != 4
        || l.group_size() != 64
        || l.metadata_dtype() != MetadataDtype::BF16
    {
        return Err(Error::InvalidConfig(
            "Gemma3 packed4/group64/BF16 matrix shape",
        ));
    }
    Ok(l.total_bytes())
}
impl<M: MatrixSource> Model<M> {
    /// The source adapter must separately authenticate buffers/provenance. This
    /// constructor validates the explicit mathematical graph and bounded storage.
    pub fn new(
        config: Config,
        specs: Vec<LayerSpec>,
        window: usize,
        weights: Weights<M>,
    ) -> Result<Self, Error> {
        config.validate()?;
        if config.num_layers > MAX_LAYERS
            || config.max_positions > MAX_CAPACITY
            || config.kv_heads != 1
            || window == 0
            || window > config.max_positions
            || specs.len() != config.num_layers
            || weights.layers.len() != config.num_layers
        {
            return Err(Error::InvalidConfig(
                "Gemma3 bounded graph/layer/KV profile",
            ));
        }
        for spec in &specs {
            if !spec.theta.is_finite() || spec.theta <= 0.0 {
                return Err(Error::InvalidConfig("Gemma3 layer theta"));
            }
        }
        let c = &config;
        let q = product(c.query_heads, c.head_dim)?;
        let mut tensor_bytes = matrix(&weights.embedding, c.vocab_size, c.hidden_size)?;
        // Validate the complete original matrix/norm inventory before converting
        // any norm or publishing a model. No pointer/name lookup occurs in forward.
        length(
            "Gemma3 final gamma",
            weights.final_gamma.len(),
            c.hidden_size,
        )?;
        for w in &weights.layers {
            for (m, rows, cols) in [
                (&w.q, q, c.hidden_size),
                (&w.k, c.head_dim, c.hidden_size),
                (&w.v, c.head_dim, c.hidden_size),
                (&w.o, c.hidden_size, q),
                (&w.gate, c.intermediate_size, c.hidden_size),
                (&w.up, c.intermediate_size, c.hidden_size),
                (&w.down, c.hidden_size, c.intermediate_size),
            ] {
                tensor_bytes = add(tensor_bytes, matrix(m, rows, cols)?)?;
            }
            for n in [
                &w.gamma.input,
                &w.gamma.post_attention,
                &w.gamma.pre_ff,
                &w.gamma.post_ff,
            ] {
                length("Gemma3 hidden gamma", n.len(), c.hidden_size)?;
            }
            for n in [&w.gamma.q, &w.gamma.k] {
                length("Gemma3 head gamma", n.len(), c.head_dim)?;
            }
        }
        let mut geometry = Vec::new();
        geometry
            .try_reserve_exact(c.num_layers)
            .map_err(|_| Error::Allocation)?;
        for spec in &specs {
            geometry.push(LayerKvGeometry {
                kind: spec.kind,
                heads: 1,
                head_dim: c.head_dim,
            });
        }
        let plan = SharedKvPlan::new(&geometry, 0, window, c.max_positions, c.max_positions)?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(c.num_layers)
            .map_err(|_| Error::Allocation)?;
        for (w, spec) in weights.layers.into_iter().zip(specs) {
            let g = w.gamma;
            layers.push(Layer {
                q: w.q,
                k: w.k,
                v: w.v,
                o: w.o,
                gate: w.gate,
                up: w.up,
                down: w.down,
                spec,
                norms: Norms {
                    input: offset_norm(g.input, c.hidden_size)?,
                    post_attention: offset_norm(g.post_attention, c.hidden_size)?,
                    pre_ff: offset_norm(g.pre_ff, c.hidden_size)?,
                    post_ff: offset_norm(g.post_ff, c.hidden_size)?,
                    q: offset_norm(g.q, c.head_dim)?,
                    k: offset_norm(g.k, c.head_dim)?,
                },
            });
        }
        let final_norm = offset_norm(weights.final_gamma, c.hidden_size)?;
        let embedding_scale = bf16::from_f64((c.hidden_size as f64).sqrt()).to_f32();
        let mut bytes = add(std::mem::size_of::<Self>(), tensor_bytes)?;
        bytes = add(
            bytes,
            product(layers.capacity(), std::mem::size_of::<Layer<M>>())?,
        )?;
        bytes = add(
            bytes,
            plan.allocated_metadata_bytes() - std::mem::size_of::<SharedKvPlan>(),
        )?;
        bytes = add(bytes, allocation_bytes(final_norm.capacity())?)?;
        for l in &layers {
            bytes = add(
                bytes,
                total_bytes(&[
                    l.norms.input.capacity(),
                    l.norms.post_attention.capacity(),
                    l.norms.pre_ff.capacity(),
                    l.norms.post_ff.capacity(),
                    l.norms.q.capacity(),
                    l.norms.k.capacity(),
                ])?,
            )?;
        }
        Ok(Self {
            config,
            embedding: weights.embedding,
            layers,
            final_norm,
            plan,
            window,
            embedding_scale,
            bytes,
        })
    }
    /// Includes borrowed packed tensor extents and owned FP32 norms/metadata.
    /// Caller padding/source metadata, load conversion overlap and process/driver
    /// owners remain separate. This is not a complete application memory budget.
    pub fn model_bytes(&self) -> usize {
        self.bytes
    }
    pub fn embedding_scale(&self) -> f32 {
        self.embedding_scale
    }
    pub fn planned_session_bytes(&self, chunk: usize) -> Result<usize, Error> {
        self.session_bytes::<Scalar>(chunk)
    }
    pub fn new_session(
        &self,
        chunk: usize,
        budget: usize,
    ) -> Result<Session<'_, M>, crate::session::Error>
    where
        Scalar: Operations<M>,
    {
        self.session_with_backend(chunk, budget, Scalar)
    }
    fn session_bytes<B>(&self, chunk: usize) -> Result<usize, Error> {
        let c = &self.config;
        if chunk == 0 || chunk > MAX_CHUNK || chunk > c.max_positions {
            return Err(Error::InvalidConfig("Gemma3 chunk capacity"));
        }
        let mut bytes = std::mem::size_of::<Session<'_, M, B>>();
        bytes = add(
            bytes,
            self.plan.planned_cache_bytes()? - std::mem::size_of::<Vec<WindowKvCache>>(),
        )?;
        bytes = add(bytes, allocation_bytes(c.max_positions)?)?;
        bytes = add(bytes, total_bytes(&Scratch::counts(c, chunk)?)?)?;
        bytes = add(
            bytes,
            product(c.num_layers, std::mem::size_of::<Staging>())?,
        )?;
        bytes = add(
            bytes,
            allocation_bytes(product(
                product(product(c.num_layers, chunk)?, c.head_dim)?,
                2,
            )?)?,
        )?;
        Ok(bytes)
    }
    fn session_with_backend<B: Operations<M>>(
        &self,
        chunk: usize,
        budget: usize,
        operations: B,
    ) -> Result<Session<'_, M, B>, crate::session::Error> {
        let external = operations.memory()?.total()?;
        let required = add(add(self.bytes, self.session_bytes::<B>(chunk)?)?, external)?;
        if required > budget {
            return Err(crate::session::Error::MemoryBudget {
                required,
                limit: budget,
            });
        }
        let mut staging = Vec::new();
        staging
            .try_reserve_exact(self.layers.len())
            .map_err(|_| Error::Allocation)?;
        let stage_elements = product(chunk, self.config.head_dim).map_err(Error::from)?;
        for _ in &self.layers {
            staging.push(Staging {
                k: zeros(stage_elements)?,
                v: zeros(stage_elements)?,
            });
        }
        let mut history = Vec::new();
        history
            .try_reserve_exact(self.config.max_positions)
            .map_err(|_| Error::Allocation)?;
        history.resize(self.config.max_positions, 0);
        let session = Session {
            model: self,
            operations,
            caches: self.plan.new_caches()?,
            staging,
            scratch: Scratch::new(&self.config, chunk)?,
            history,
            committed: 0,
            published_rows: 0,
            chunk,
            #[cfg(test)]
            fault: None,
        };
        let actual = add(self.bytes, session.allocated_bytes()?)?;
        if actual > budget {
            return Err(crate::session::Error::MemoryBudget {
                required: actual,
                limit: budget,
            });
        }
        Ok(session)
    }
}

struct Staging {
    k: Vec<f32>,
    v: Vec<f32>,
}
struct Scratch {
    hidden: Vec<f32>,
    next: Vec<f32>,
    norm: Vec<f32>,
    q: Vec<f32>,
    q_norm: Vec<f32>,
    kv: Vec<f32>,
    kv_norm: Vec<f32>,
    context: Vec<f32>,
    attention: Vec<f32>,
    residual: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    gated: Vec<f32>,
    scores: Vec<f32>,
    probabilities: Vec<f32>,
    logits: Vec<f32>,
    published: Vec<f32>,
}
impl Scratch {
    fn counts(c: &Config, chunk: usize) -> Result<[usize; 17], Error> {
        let h = product(chunk, c.hidden_size)?;
        let q = product(product(chunk, c.query_heads)?, c.head_dim)?;
        let kv = product(chunk, c.head_dim)?;
        let f = product(chunk, c.intermediate_size)?;
        let scores = product(product(chunk, c.query_heads)?, c.max_positions)?;
        let logits = product(chunk, c.vocab_size)?;
        Ok([
            h, h, h, q, q, kv, kv, q, h, h, f, f, f, scores, scores, logits, logits,
        ])
    }
    fn new(c: &Config, chunk: usize) -> Result<Self, Error> {
        let [
            h,
            norm_next,
            norm,
            q,
            qn,
            kv,
            kvn,
            ct,
            a,
            r,
            g,
            u,
            p,
            s,
            pr,
            l,
            pub_l,
        ] = Self::counts(c, chunk)?;
        Ok(Self {
            hidden: zeros(h)?,
            next: zeros(norm_next)?,
            norm: zeros(norm)?,
            q: zeros(q)?,
            q_norm: zeros(qn)?,
            kv: zeros(kv)?,
            kv_norm: zeros(kvn)?,
            context: zeros(ct)?,
            attention: zeros(a)?,
            residual: zeros(r)?,
            gate: zeros(g)?,
            up: zeros(u)?,
            gated: zeros(p)?,
            scores: zeros(s)?,
            probabilities: zeros(pr)?,
            logits: zeros(l)?,
            published: zeros(pub_l)?,
        })
    }
    fn capacities(&self) -> [usize; 17] {
        [
            &self.hidden,
            &self.next,
            &self.norm,
            &self.q,
            &self.q_norm,
            &self.kv,
            &self.kv_norm,
            &self.context,
            &self.attention,
            &self.residual,
            &self.gate,
            &self.up,
            &self.gated,
            &self.scores,
            &self.probabilities,
            &self.logits,
            &self.published,
        ]
        .map(Vec::capacity)
    }
}

/// Append-only new-chunk primitive. Prefix rewinds, sampled pending tokens and
/// production generation/text integration require a separate replay wrapper.
pub struct Session<'model, M, B = Scalar> {
    model: &'model Model<M>,
    operations: B,
    caches: Vec<WindowKvCache>,
    staging: Vec<Staging>,
    scratch: Scratch,
    history: Vec<u32>,
    committed: usize,
    published_rows: usize,
    chunk: usize,
    #[cfg(test)]
    fault: Option<usize>,
}
impl<M: MatrixSource, B: Operations<M>> Session<'_, M, B> {
    pub fn tokens(&self) -> &[u32] {
        &self.history[..self.committed]
    }
    pub fn logits(&self) -> &[f32] {
        &self.scratch.published[..self.published_rows * self.model.config.vocab_size]
    }
    pub fn allocated_bytes(&self) -> Result<usize, Error> {
        let mut bytes = add(
            std::mem::size_of::<Self>(),
            self.operations.memory()?.total()?,
        )?;
        bytes = add(
            bytes,
            self.model.plan.allocated_cache_bytes(&self.caches)?
                - std::mem::size_of::<Vec<WindowKvCache>>(),
        )?;
        bytes = add(bytes, allocation_bytes(self.history.capacity())?)?;
        bytes = add(bytes, total_bytes(&self.scratch.capacities())?)?;
        bytes = add(
            bytes,
            product(self.staging.capacity(), std::mem::size_of::<Staging>())?,
        )?;
        for stage in &self.staging {
            bytes = add(
                bytes,
                total_bytes(&[stage.k.capacity(), stage.v.capacity()])?,
            )?;
        }
        Ok(bytes)
    }
    pub fn reset(&mut self) -> Result<(), B::Error> {
        self.operations.ready()?;
        reset_batch(&mut self.caches)?;
        self.history.fill(0);
        self.committed = 0;
        self.published_rows = 0;
        Ok(())
    }
    pub fn forward(&mut self, tokens: &[u32]) -> Result<&[f32], B::Error> {
        forward::run(self, tokens)?;
        Ok(self.logits())
    }
}

#[cfg(test)]
mod tests;
