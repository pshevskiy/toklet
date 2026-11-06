//! Short hybrid diagnostic: original GPU products/gathers, shared CPU graph.
//! The caller keeps Context, immutable source, norm copy and graph alive.
use super::*;
use crate::model::gemma_packed as graph;
use crate::session::{self, Options};
use graph::{BackendMemory, MatrixSource, Operations, operations::sealed};
use std::fmt;
#[cfg(test)]
pub(crate) mod test_support;
mod workspace;

#[derive(Debug)]
pub enum Error {
    Model(crate::model::Error),
    Metal(crate::metal::Error),
}
impl From<crate::model::Error> for Error {
    fn from(value: crate::model::Error) -> Self {
        Self::Model(value)
    }
}
impl From<crate::cpu::Error> for Error {
    fn from(value: crate::cpu::Error) -> Self {
        Self::Model(value.into())
    }
}
impl From<crate::metal::Error> for Error {
    fn from(value: crate::metal::Error) -> Self {
        Self::Metal(value)
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(e) => e.fmt(f),
            Self::Metal(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(e) => Some(e),
            Self::Metal(e) => Some(e),
        }
    }
}
impl From<Error> for session::Error {
    fn from(value: Error) -> Self {
        match value {
            Error::Model(e) => e.into(),
            Error::Metal(e) => Self::Backend(e.to_string()),
        }
    }
}
type HybridResult<T> = std::result::Result<T, Error>;
pub type Model<'a, 'weights> = graph::Model<'a, MatrixRef<'a, 'weights>>;
pub type Session<'model, 'a, 'weights, 'context> =
    graph::Session<'model, 'a, MatrixRef<'a, 'weights>, Backend<'a, 'weights, 'context>>;
pub type GenerationSession<'model, 'a, 'weights, 'context> =
    graph::GenerationSession<'model, 'a, MatrixRef<'a, 'weights>, Backend<'a, 'weights, 'context>>;

/// Bound immutable matrix reference. Only this admitted norm/source owner can
/// create it; role slots are resolved cold, outside numerical loops.
pub struct MatrixRef<'a, 'weights> {
    source: &'a Norms<'weights>,
    index: usize,
}
impl sealed::Matrix for MatrixRef<'_, '_> {}
impl MatrixSource for MatrixRef<'_, '_> {
    fn descriptor(&self) -> graph::MatrixDescriptor {
        graph::MatrixDescriptor::Affine(self.source.matrices[self.index].layout())
    }
}
/// Small immutable CPU vector copy tied to the exact resident source. Keeps the
/// GPU copy resident too, and accounts for both. No source packed model copy.
pub struct Norms<'weights> {
    source: Option<&'weights E2bWeights>,
    matrices: &'weights [Weights],
    values: Vec<Vec<f32>>,
    shapes: Vec<crate::mlx::affine::Layout>,
    slots: Vec<usize>,
    gather_indices: [usize; 2],
    owned: usize,
    payload: usize,
}
impl<'weights> Norms<'weights> {
    pub fn new(source: &'weights E2bWeights, available_bytes: usize) -> Result<Self> {
        Self::new_for(source, available_bytes, false)
    }

    fn new_for(source: &'weights E2bWeights, available_bytes: usize, mixed: bool) -> Result<Self> {
        let gathers = [
            source.bindings.embedding_index(),
            source.bindings.ple_embedding_index(),
        ];
        let matrices = &source.storage.matrices;
        if gathers
            .iter()
            .any(|&index| matrices[index].layout().bits() != 4)
        {
            return Err(super::Error::Options("Gemma hybrid gather format"));
        }
        let shape_capacity = if mixed {
            // An exact cold count before reserving. Layout equality includes
            // the packing width, so otherwise equal shapes keep distinct slots.
            matrices
                .iter()
                .enumerate()
                .filter(|(i, matrix)| {
                    *i != gathers[1]
                        && !matrices[..*i].iter().enumerate().any(|(prior, other)| {
                            prior != gathers[1] && other.layout() == matrix.layout()
                        })
                })
                .count()
        } else {
            14
        };
        let payload = source
            .storage
            .vectors
            .iter()
            .try_fold(0usize, |sum, v| add(sum, multiply(v.values, 4)?))?;
        let minimum_metadata = [
            size_of::<Self>(),
            multiply(262, size_of::<Vec<f32>>())?,
            multiply(278, size_of::<usize>())?,
            multiply(shape_capacity, size_of::<crate::mlx::affine::Layout>())?,
        ]
        .into_iter()
        .try_fold(0, add)?;
        let minimum = add(minimum_metadata, payload)?;
        admit(
            add(source.report.resident_requested_bytes, minimum)?,
            available_bytes,
        )?;
        let mut values = owner_vec::<Vec<f32>>(262)?;
        let mut shapes = owner_vec::<crate::mlx::affine::Layout>(shape_capacity)?;
        let mut slots = owner_vec::<usize>(278)?;
        let metadata = add(
            size_of::<Self>(),
            add(
                multiply(values.capacity(), size_of::<Vec<f32>>())?,
                add(
                    multiply(slots.capacity(), size_of::<usize>())?,
                    multiply(shapes.capacity(), size_of::<crate::mlx::affine::Layout>())?,
                )?,
            )?,
        )?;
        admit(
            add(
                source.report.resident_requested_bytes,
                add(metadata, payload)?,
            )?,
            available_bytes,
        )?;
        let mut actual = 0;
        let mut completed = 0;
        for vector in &source.storage.vectors {
            let mut copy = owner_vec::<f32>(vector.values)?;
            actual = add(actual, multiply(copy.capacity(), 4)?)?;
            completed = add(completed, multiply(vector.values, 4)?)?;
            admit(
                add(
                    source.report.resident_requested_bytes,
                    add(metadata, add(actual, subtract(payload, completed)?)?)?,
                )?,
                available_bytes,
            )?;
            vector.buffer.with_slice(|words| {
                copy.extend(words[..vector.values].iter().map(|&v| f32::from_bits(v)))
            });
            values.push(copy);
        }
        for (index, matrix) in matrices.iter().enumerate() {
            if index == gathers[1] {
                slots.push(usize::MAX);
                continue;
            }
            let layout = matrix.layout();
            let slot = if let Some(slot) = shapes.iter().position(|v| *v == layout) {
                slot
            } else {
                if shapes.len() == shape_capacity {
                    return Err(super::Error::Options(
                        "Gemma hybrid geometry inventory differs",
                    ));
                }
                shapes.push(layout);
                shapes.len() - 1
            };
            slots.push(slot);
        }
        let expected_shapes = if mixed { shape_capacity } else { 13 };
        if shapes.len() != expected_shapes {
            return Err(super::Error::Options(
                "Gemma hybrid projection geometry count",
            ));
        }
        Ok(Self {
            source: Some(source),
            matrices,
            values,
            shapes,
            slots,
            gather_indices: gathers,
            owned: add(metadata, actual)?,
            payload: actual,
        })
    }
    pub fn owned_bytes(&self) -> usize {
        self.owned
    }
    pub fn source_identity(&self) -> Option<&Identity> {
        self.source.map(E2bWeights::identity)
    }
    pub fn model(&self) -> HybridResult<Model<'_, 'weights>> {
        self.model_for_context_probe(graph::MAX_POSITIONS)
    }
    /// Explicit diagnostic ceiling over the same immutable packed source.
    ///
    /// The default model remains32 positions. This probe does not qualify wider
    /// numerical execution or app/codec identity. Per-call scratch stays<=32.
    pub fn model_for_context_probe(
        &self,
        position_limit: usize,
    ) -> HybridResult<Model<'_, 'weights>> {
        self.model_for_format(position_limit, false)
    }

    fn model_for_format(
        &self,
        position_limit: usize,
        mixed: bool,
    ) -> HybridResult<Model<'_, 'weights>> {
        if !(1..=graph::MAX_E2B_DIAGNOSTIC_POSITIONS).contains(&position_limit) {
            return Err(
                crate::model::Error::InvalidConfig("Gemma E2B diagnostic position limit").into(),
            );
        }
        let source = self.source.ok_or(super::Error::Unsupported(
            "Gemma hybrid synthetic source is test-only",
        ))?;
        let weights = source.bindings.graph_weights(self)?;
        let physical = source.storage.matrices.iter().try_fold(0usize, |sum, w| {
            w.views().try_fold(sum, |sum, v| {
                add(sum, multiply(v.codes.len + v.scales.len + v.biases.len, 4)?)
            })
        })?;
        let physical = source
            .storage
            .vectors
            .iter()
            .try_fold(physical, |sum, v| add(sum, multiply(v.buffer.len, 4)?))?;
        let backing = graph::Backing {
            tensor_bytes: add(physical, self.payload)?,
            metadata_bytes: add(
                subtract(source.report.resident_requested_bytes, physical)?,
                subtract(self.owned, self.payload)?,
            )?,
            metadata_allowance_bytes: 0,
        };
        Ok(graph::Model::from_e2b_parts(
            weights,
            &source.bindings.geometry(),
            backing,
            position_limit,
            mixed,
        )?)
    }
    pub(super) fn matrix(&self, index: usize) -> Result<MatrixRef<'_, 'weights>> {
        if index >= self.matrices.len() {
            return Err(super::Error::Options("Gemma hybrid matrix role"));
        }
        Ok(MatrixRef {
            source: self,
            index,
        })
    }
    pub(super) fn vector(&self, index: usize) -> Result<&[f32]> {
        self.values
            .get(index)
            .map(Vec::as_slice)
            .ok_or(super::Error::Options("Gemma hybrid vector role"))
    }
    fn check_model(&self, model: &Model<'_, 'weights>) -> Result<()> {
        if model
            .matrix_handles()
            .any(|m| !std::ptr::eq(m.source, self))
        {
            return Err(super::Error::Options("Gemma hybrid model/source mismatch"));
        }
        Ok(())
    }
    pub fn new_session<'model, 'a, 'context>(
        &'a self,
        model: &'model Model<'a, 'weights>,
        context: &'context Context,
        budget: usize,
    ) -> HybridResult<Session<'model, 'a, 'weights, 'context>> {
        self.new_session_for_chunk(model, context, model.default_chunk_rows(), budget)
    }

    fn new_session_for_chunk<'model, 'a, 'context>(
        &'a self,
        model: &'model Model<'a, 'weights>,
        context: &'context Context,
        chunk_rows: usize,
        budget: usize,
    ) -> HybridResult<Session<'model, 'a, 'weights, 'context>> {
        self.check_model(model)?;
        let numerical = model
            .planned_backend_session_bytes_for_chunk::<Backend<'a, 'weights, 'context>>(
                chunk_rows,
            )?;
        admit(
            add(
                add(model.model_bytes(), numerical)?,
                Backend::minimum_bytes(self, context, chunk_rows)?,
            )?,
            budget,
        )?;
        let remaining = subtract(budget, add(model.model_bytes(), numerical)?)?;
        let backend = Backend::new(self, context, chunk_rows, remaining)?;
        model.session_with_backend_for_chunk(backend, chunk_rows, budget)
    }
    pub fn new_generation_session<'model, 'a, 'context>(
        &'a self,
        model: &'model Model<'a, 'weights>,
        context: &'context Context,
        options: Options,
    ) -> std::result::Result<GenerationSession<'model, 'a, 'weights, 'context>, session::Error>
    {
        self.check_model(model)
            .map_err(Error::from)
            .map_err(session::Error::from)?;
        graph::generation::validate_options(model, options)?;
        let raw = model
            .planned_backend_session_bytes_for_capacity::<Backend<'a, 'weights, 'context>>(
                options.capacity,
                options.prefill_chunk_size,
            )?;
        let extra = GenerationSession::<'model, 'a, 'weights, 'context>::planned_wrapper_bytes(
            options.capacity,
        )
        .map_err(Error::from)
        .map_err(session::Error::from)?;
        let required = add(model.model_bytes(), add(raw, extra).map_err(Error::from)?)
            .map_err(Error::from)
            .map_err(session::Error::from)?;
        let full = add(
            required,
            Backend::minimum_bytes(self, context, options.prefill_chunk_size)
                .map_err(Error::from)
                .map_err(session::Error::from)?,
        )
        .map_err(Error::from)
        .map_err(session::Error::from)?;
        if full > options.memory_limit_bytes {
            return Err(session::Error::MemoryBudget {
                required: full,
                limit: options.memory_limit_bytes,
            });
        }
        let remaining = options.memory_limit_bytes.checked_sub(required).ok_or(
            session::Error::MemoryBudget {
                required,
                limit: options.memory_limit_bytes,
            },
        )?;
        let backend = Backend::new(self, context, options.prefill_chunk_size, remaining)
            .map_err(session::Error::from)?;
        graph::GenerationSession::with_backend(model, backend, options)
    }
}

/// Small norm copy for the authenticated mixed QAT text diagnostic. This
/// wrapper grants only the explicit mixed-context session.
pub struct QatNorms<'weights>(Norms<'weights>);

impl<'weights> QatNorms<'weights> {
    pub const MAX_POSITIONS: usize = 512;

    pub fn new(source: &'weights QatWeights, available_bytes: usize) -> Result<Self> {
        Norms::new_for(&source.0, available_bytes, true).map(Self)
    }

    pub fn owned_bytes(&self) -> usize {
        self.0.owned_bytes()
    }

    pub fn model(&self, position_limit: usize) -> HybridResult<Model<'_, 'weights>> {
        if position_limit > Self::MAX_POSITIONS {
            return Err(crate::model::Error::InvalidConfig("Gemma QAT diagnostic limit").into());
        }
        self.0.model_for_format(position_limit, true)
    }

    #[cfg(test)]
    pub(crate) fn model_for_quality_probe(
        &self,
        position_limit: usize,
    ) -> HybridResult<Model<'_, 'weights>> {
        self.0.model_for_format(position_limit, true)
    }

    #[cfg(test)]
    pub(crate) fn new_quality_generation_session<'model, 'a, 'context>(
        &'a self,
        model: &'model Model<'a, 'weights>,
        context: &'context MixedContext,
        options: Options,
    ) -> std::result::Result<GenerationSession<'model, 'a, 'weights, 'context>, session::Error>
    {
        self.0.new_generation_session(model, &context.0, options)
    }

    pub fn new_session<'model, 'a, 'context>(
        &'a self,
        model: &'model Model<'a, 'weights>,
        context: &'context MixedContext,
        chunk_rows: usize,
        budget: usize,
    ) -> HybridResult<Session<'model, 'a, 'weights, 'context>> {
        self.0
            .new_session_for_chunk(model, &context.0, chunk_rows, budget)
    }
}

/// Per-session pooled scratch, bound to one immutable source and external Context.
pub struct Backend<'a, 'weights, 'context> {
    source: &'a Norms<'weights>,
    projections: Vec<Projection<'context>>,
    gathers: [Gather<'context>; 2],
    failed: bool,
    #[cfg(test)]
    test_fault: Option<(bool, usize)>,
}
impl sealed::Operations for Backend<'_, '_, '_> {}
impl<'a, 'weights> Operations<MatrixRef<'a, 'weights>> for Backend<'a, 'weights, '_> {
    type Error = Error;
    fn project(
        &mut self,
        input: &[f32],
        weight: &MatrixRef<'a, 'weights>,
        output: &mut [f32],
        tokens: usize,
    ) -> HybridResult<()> {
        self.ready()?;
        if !std::ptr::eq(weight.source, self.source) {
            return Err(super::Error::Options("Gemma hybrid foreign projection").into());
        }
        let slot = self.source.slots[weight.index];
        if slot >= self.projections.len() {
            return Err(super::Error::Options("Gemma hybrid projection role").into());
        }
        #[cfg(test)]
        if self.test_failure(false) {
            return Err(super::Error::Options("injected Gemma hybrid projection failure").into());
        }
        let result = self.projections[slot].apply(
            &self.source.matrices[weight.index],
            input,
            output,
            tokens,
        );
        if result.is_err() {
            self.failed = true;
        }
        Ok(result?)
    }
    fn gather(
        &mut self,
        weight: &MatrixRef<'a, 'weights>,
        ids: &[u32],
        output: &mut [f32],
    ) -> HybridResult<()> {
        self.ready()?;
        if !std::ptr::eq(weight.source, self.source) {
            return Err(super::Error::Options("Gemma hybrid foreign gather").into());
        }
        let slot = self
            .source
            .gather_indices
            .iter()
            .position(|&index| index == weight.index)
            .ok_or(super::Error::Options("Gemma hybrid gather role"))?;
        if ids.len() > graph::MAX_POSITIONS {
            return Err(super::Error::Options("Gemma hybrid gather ID bound").into());
        }
        let mut converted = [0usize; graph::MAX_POSITIONS];
        for (to, &id) in converted.iter_mut().zip(ids) {
            *to = id as usize;
        }
        #[cfg(test)]
        if self.test_failure(true) {
            return Err(super::Error::Options("injected Gemma hybrid gather failure").into());
        }
        let result = self.gathers[slot].apply_weights(
            &self.source.matrices[weight.index],
            &converted[..ids.len()],
            output,
        );
        if result.is_err() {
            self.failed = true;
        }
        Ok(result?)
    }
    fn ready(&self) -> HybridResult<()> {
        if self.failed
            || self.projections.iter().any(Projection::is_poisoned)
            || self.gathers.iter().any(Gather::is_poisoned)
        {
            return Err(super::Error::Options(
                "Gemma hybrid scratch failed; recreate session and replay",
            )
            .into());
        }
        Ok(())
    }
    fn memory(&self) -> std::result::Result<BackendMemory, crate::model::Error> {
        self.memory_report().map_err(|_| {
            crate::model::Error::InvalidConfig("Gemma hybrid scratch accounting overflow")
        })
    }
}
