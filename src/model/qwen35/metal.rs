//! Resident packed GPU projection/gather; existing CPU state transaction/math.
use super::*;
use crate::{
    metal::gemma_affine::{Context, Gather, Projection, Weights as GpuWeights},
    mlx::{affine::Layout, qwen35::payload::metal::MetalStore},
};
use std::mem::size_of;

const CHUNK: usize = 16384;
impl From<crate::metal::Error> for Error {
    fn from(error: crate::metal::Error) -> Self {
        Self::Metal(error)
    }
}

/// Resolved immutable GPU source slot; no name lookup during numerical work.
pub struct MetalMatrix<'a> {
    weights: &'a GpuWeights,
    slot: usize,
}
#[cfg(test)]
impl<'a> MetalMatrix<'a> {
    pub(crate) fn diagnostic_binding(weights: &'a GpuWeights, slot: usize) -> Self {
        Self { weights, slot }
    }
}
impl execution::sealed::Matrix for MetalMatrix<'_> {}
impl MatrixStorage for MetalMatrix<'_> {
    fn layout(&self) -> Layout {
        self.weights.layout()
    }
}
pub type MetalModel<'a> = Model<'a, MetalMatrix<'a>>;
pub type MetalSession<'model, 'weights, 'context> =
    Session<'model, 'weights, MetalMatrix<'weights>, MetalExecution<'context>>;

fn bindings<'a>(model: &MetalModel<'a>) -> Result<Vec<&'a GpuWeights>> {
    let c = model.config;
    let count = add(
        1,
        add(mul(c.delta_count(), 8)?, mul(c.attention_count(), 7)?)?,
    )?;
    let mut slots = reserved(count)?;
    slots.resize(count, None);
    let mut bind = |m: &MetalMatrix<'a>| -> Result<()> {
        let slot = slots
            .get_mut(m.slot)
            .ok_or_else(|| invalid("Qwen GPU matrix slot outside model"))?;
        if slot.is_some() {
            return Err(invalid("Qwen GPU matrix slot reused"));
        }
        *slot = Some(m.weights);
        Ok(())
    };
    bind(&model.embedding)?;
    for layer in &model.layers {
        let w = &layer.weights;
        for m in [&w.gate, &w.up, &w.down] {
            bind(m)?;
        }
        match &w.mixing {
            MixingWeights::Delta(d) => {
                for m in [&d.qkv, &d.z, &d.a, &d.b, &d.out] {
                    bind(m)?;
                }
            }
            MixingWeights::Attention(a) => {
                for m in [&a.q_gate, &a.k, &a.v, &a.out] {
                    bind(m)?;
                }
            }
        }
    }
    slots
        .into_iter()
        .map(|s| s.ok_or_else(|| invalid("Qwen GPU matrix slot missing")))
        .collect()
}

/// Mutable per-session projection/gather workspaces. CPU recurrent/history/KV
/// banks remain in Session. No source bytes, model clone or token controller.
pub struct MetalExecution<'a> {
    projections: Vec<Projection<'a>>,
    gather: Gather<'a>,
    poisoned: bool,
}
impl execution::sealed::Executor for MetalExecution<'_> {}
impl<'a> MetalExecution<'a> {
    fn planned(model: &MetalModel<'_>) -> Result<usize> {
        let mut bytes = Gather::minimum_admitted_bytes(model.config.hidden, 1)?
            .checked_sub(size_of::<Gather<'_>>())
            .ok_or(cpu::Error::SizeOverflow)?;
        for matrix in bindings(model)? {
            bytes = add(
                bytes,
                Projection::minimum_owned_bytes(matrix.layout(), CHUNK, 1)?,
            )?;
        }
        Ok(bytes)
    }
    fn new(context: &'a Context, model: &MetalModel<'_>, limit: usize) -> Result<Self> {
        let planned = Self::planned(model)?;
        if planned > limit {
            return Err(Error::MemoryBudget {
                required: planned,
                limit,
            });
        }
        let source = bindings(model)?;
        let mut projections = reserved(source.len())?;
        for matrix in source {
            let bytes = Projection::minimum_owned_bytes(matrix.layout(), CHUNK, 1)?;
            projections.push(context.projection_session(matrix.layout(), CHUNK, 1, bytes)?);
        }
        let bytes = Gather::minimum_admitted_bytes(model.config.hidden, 1)?;
        let gather = context.gather_session(model.config.hidden, 1, bytes)?;
        let result = Self {
            projections,
            gather,
            poisoned: false,
        };
        let actual = result.owned_bytes()?;
        if actual > limit {
            return Err(Error::MemoryBudget {
                required: actual,
                limit,
            });
        }
        Ok(result)
    }
}
impl Execution<MetalMatrix<'_>> for MetalExecution<'_> {
    fn owned_bytes(&self) -> Result<usize> {
        // Projection owner structs live in their Vec slots and count once.
        // Gather itself is embedded in Session metadata, so subtract that body.
        let mut bytes = self
            .gather
            .admitted_bytes()
            .checked_sub(size_of::<Gather<'_>>())
            .ok_or(cpu::Error::SizeOverflow)?;
        bytes = add(
            bytes,
            mul(
                self.projections.capacity() - self.projections.len(),
                size_of::<Projection<'_>>(),
            )?,
        )?;
        self.projections
            .iter()
            .try_fold(bytes, |n, p| add(n, p.owned_bytes()))
    }
    fn project(
        &mut self,
        matrix: &MetalMatrix<'_>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        if self.poisoned {
            return Err(Error::ExecutionPoisoned);
        }
        let workspace = self
            .projections
            .get_mut(matrix.slot)
            .ok_or_else(|| invalid("Qwen GPU matrix slot outside session"))?;
        let result = workspace.apply(matrix.weights, input, output, 1);
        self.poisoned |= workspace.is_poisoned();
        Ok(result?)
    }
    fn gather(&mut self, matrix: &MetalMatrix<'_>, token: u32, output: &mut [f32]) -> Result<()> {
        if self.poisoned {
            return Err(Error::ExecutionPoisoned);
        }
        let result = self
            .gather
            .apply_weights(matrix.weights, &[token as usize], output);
        self.poisoned |= self.gather.is_poisoned();
        Ok(result?)
    }
}
impl<'a> MetalModel<'a> {
    pub fn from_metal_store(store: &'a MetalStore) -> Result<Self> {
        Self::bind_source(
            store.plan(),
            |name| {
                let (slot, weights) = store.matrix(name)?;
                Ok(MetalMatrix { weights, slot })
            },
            |name| Ok(store.vector(name)?),
            |name| Ok(store.convolution(name)?),
            store.report().retained_policy_bytes,
        )
    }
    pub fn planned_session_memory(&self, options: Options) -> Result<Memory> {
        MetalSession::planned_with_execution(self, options, MetalExecution::planned(self)?)
    }
    pub fn session<'model, 'context>(
        &'model self,
        context: &'context Context,
        options: Options,
    ) -> Result<MetalSession<'model, 'a, 'context>> {
        let planned = self.planned_session_memory(options)?;
        planned.admit(options.memory_limit_bytes)?;
        let bytes = MetalExecution::planned(self)?;
        let execution = MetalExecution::new(context, self, bytes)?;
        MetalSession::with_execution(self, options, execution)
    }
}
impl<'model, 'weights, 'context> MetalSession<'model, 'weights, 'context> {
    /// True only after an execution workspace was poisoned. Observer/cancel
    /// rejection preserves reusable execution; reset alone does not repair GPU.
    pub fn requires_repair(&self) -> bool {
        self.execution.poisoned
    }
    /// Cold, explicitly admitted replacement; committed state/published logits
    /// remain untouched. Charge simultaneous old/new GPU scratch before upload.
    pub fn repair_execution(&mut self, context: &'context Context) -> Result<()> {
        let minimum = MetalExecution::planned(self.model)?;
        let mut peak = self.memory;
        peak.scratch_bytes = add(peak.scratch_bytes, minimum)?;
        peak.sum()?;
        peak.admit(self.options.memory_limit_bytes)?;
        let replacement = MetalExecution::new(context, self.model, minimum)?;
        let old = self.execution.owned_bytes()?;
        let new = replacement.owned_bytes()?;
        peak.scratch_bytes = add(self.memory.scratch_bytes, new)?;
        peak.sum()?;
        peak.admit(self.options.memory_limit_bytes)?;
        let scratch = self
            .memory
            .scratch_bytes
            .checked_sub(old)
            .ok_or(cpu::Error::SizeOverflow)?;
        let mut final_memory = self.memory;
        final_memory.scratch_bytes = add(scratch, new)?;
        final_memory.sum()?;
        final_memory.admit(self.options.memory_limit_bytes)?;
        self.execution = replacement;
        self.memory = final_memory;
        Ok(())
    }
}
