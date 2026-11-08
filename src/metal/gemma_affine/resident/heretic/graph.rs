//! Borrowed mixed source and the shared Gemma graph; all names resolve cold.
use super::*;
use crate::{
    model::gemma_packed as engine,
    session::{self, Options},
};
use engine::{BackendMemory, MatrixDescriptor, MatrixSource, Operations, operations::sealed};
type HResult<T> = std::result::Result<T, super::super::hybrid::Error>;
mod workspace;

pub type Model<'a, 'weights> = engine::Model<'a, MatrixRef<'a, 'weights>>;
pub type Session<'model, 'a, 'weights, 'context> =
    engine::Session<'model, 'a, MatrixRef<'a, 'weights>, Backend<'a, 'weights, 'context>>;
pub type GenerationSession<'model, 'a, 'weights, 'context> =
    engine::GenerationSession<'model, 'a, MatrixRef<'a, 'weights>, Backend<'a, 'weights, 'context>>;

pub struct MatrixRef<'a, 'weights> {
    source: &'a Norms<'weights>,
    index: usize,
}
impl sealed::Matrix for MatrixRef<'_, '_> {}
impl MatrixSource for MatrixRef<'_, '_> {
    fn descriptor(&self) -> MatrixDescriptor {
        match self.source.source.slots[self.index] {
            MatrixSlot::Affine(i) => {
                MatrixDescriptor::Affine(self.source.source.storage.matrices[i].layout())
            }
            MatrixSlot::Bf16 => {
                let (rows, columns) = self.source.source.bf16.shape();
                MatrixDescriptor::Bf16 {
                    rows,
                    columns,
                    file_backed: false,
                }
            }
        }
    }
}

/// Immutable FP32 norm copy and cold projection slots. Original GPU arrays stay
/// in the separately owned source; both copies of the small vectors are charged.
pub struct Norms<'weights> {
    source: &'weights HereticWeights,
    values: Vec<Vec<f32>>,
    shapes: Vec<Layout>,
    slots: Vec<Option<usize>>,
    gathers: [usize; 2],
    owned: usize,
    payload: usize,
}
impl<'weights> Norms<'weights> {
    pub const MAX_POSITIONS: usize = 512;
    pub fn new(source: &'weights HereticWeights, available: usize) -> Result<Self> {
        if available == 0 || available > source.limit {
            return Err(Error::Options("Heretic norm budget exceeds source limit"));
        }
        let payload = source
            .storage
            .vectors
            .iter()
            .try_fold(0usize, |s, v| add(s, multiply(v.values, 4)?))?;
        let minimum = add(
            size_of::<Self>(),
            add(
                multiply(262, size_of::<Vec<f32>>())?,
                add(
                    multiply(12, size_of::<Layout>())?,
                    multiply(278, size_of::<Option<usize>>())?,
                )?,
            )?,
        )?;
        admit(
            add(
                source.report.resident_requested_bytes,
                add(CONTEXT_BYTES, add(minimum, payload)?)?,
            )?,
            available,
        )?;
        let mut values = owner_vec::<Vec<f32>>(262)?;
        let mut shapes = owner_vec::<Layout>(12)?;
        let mut slots = owner_vec::<Option<usize>>(278)?;
        let metadata = add(
            size_of::<Self>(),
            add(
                multiply(values.capacity(), size_of::<Vec<f32>>())?,
                add(
                    multiply(shapes.capacity(), size_of::<Layout>())?,
                    multiply(slots.capacity(), size_of::<Option<usize>>())?,
                )?,
            )?,
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
                    add(
                        CONTEXT_BYTES,
                        add(metadata, add(actual, subtract(payload, completed)?)?)?,
                    )?,
                )?,
                available,
            )?;
            vector
                .buffer
                .with_slice(|s| copy.extend(s[..vector.values].iter().map(|&v| f32::from_bits(v))));
            values.push(copy);
        }
        let gathers = [
            source.bindings.embedding_index(),
            source.bindings.ple_embedding_index(),
        ];
        for (logical, slot) in source.slots.iter().enumerate() {
            if logical == gathers[1] || matches!(slot, MatrixSlot::Bf16) {
                slots.push(None);
                continue;
            }
            let MatrixSlot::Affine(i) = *slot else {
                unreachable!()
            };
            let layout = source.storage.matrices[i].layout();
            let index = if let Some(index) = shapes.iter().position(|&l| l == layout) {
                index
            } else {
                if shapes.len() == 12 {
                    return Err(Error::Options("Heretic projection inventory exceeds bound"));
                }
                shapes.push(layout);
                shapes.len() - 1
            };
            slots.push(Some(index));
        }
        if shapes.len() != 12 || slots.len() != 278 || values.len() != 262 {
            return Err(Error::Options("Heretic cold session inventory differs"));
        }
        Ok(Self {
            source,
            values,
            shapes,
            slots,
            gathers,
            owned: add(metadata, actual)?,
            payload: actual,
        })
    }
    pub fn owned_bytes(&self) -> usize {
        self.owned
    }
    pub fn model(&self, positions: usize) -> HResult<Model<'_, 'weights>> {
        if !(1..=Self::MAX_POSITIONS).contains(&positions) {
            return Err(Error::Options("Heretic diagnostic model limit512").into());
        }
        self.model_for_limit(positions)
    }
    fn model_for_limit(&self, positions: usize) -> HResult<Model<'_, 'weights>> {
        let weights = self
            .source
            .bindings
            .graph_weights_for(|i| self.matrix(i), |i| self.vector(i))?;
        let mut physical = self.source.bf16.physical_bytes();
        for w in &self.source.storage.matrices {
            physical = w.views().try_fold(physical, |s, v| {
                add(s, multiply(v.codes.len + v.scales.len + v.biases.len, 4)?)
            })?;
        }
        for v in &self.source.storage.vectors {
            physical = add(physical, multiply(v.buffer.len, 4)?)?;
        }
        let backing = engine::Backing {
            tensor_bytes: add(physical, self.payload)?,
            metadata_bytes: add(
                subtract(self.source.report.resident_requested_bytes, physical)?,
                subtract(self.owned, self.payload)?,
            )?,
            metadata_allowance_bytes: 0,
        };
        Ok(engine::Model::from_heretic_parts(
            weights,
            &self.source.bindings.geometry(),
            backing,
            positions,
        )?)
    }
    fn matrix(&self, index: usize) -> Result<MatrixRef<'_, 'weights>> {
        if index >= self.source.slots.len() {
            return Err(Error::Options("Heretic matrix role outside source"));
        }
        Ok(MatrixRef {
            source: self,
            index,
        })
    }
    fn vector(&self, index: usize) -> Result<&[f32]> {
        self.values
            .get(index)
            .map(Vec::as_slice)
            .ok_or(Error::Options("Heretic norm role outside source"))
    }
    fn affine_weight(&self, index: usize) -> Result<&Weights> {
        match self.source.slots.get(index) {
            Some(MatrixSlot::Affine(i)) => Ok(&self.source.storage.matrices[*i]),
            _ => Err(Error::Options("Heretic role is not an affine matrix")),
        }
    }
    fn check_model(&self, model: &Model<'_, 'weights>, budget: usize) -> Result<()> {
        if budget == 0
            || budget > self.source.limit
            || model
                .matrix_handles()
                .any(|m| !std::ptr::eq(m.source, self))
        {
            return Err(Error::Options("Heretic session source or budget differs"));
        }
        Ok(())
    }
    pub fn new_session<'model, 'a, 'context>(
        &'a self,
        model: &'model Model<'a, 'weights>,
        context: &'context HereticContext,
        chunk_rows: usize,
        budget: usize,
    ) -> HResult<Session<'model, 'a, 'weights, 'context>> {
        self.check_model(model, budget)?;
        let numerical = model
            .planned_backend_session_bytes_for_chunk::<Backend<'a, 'weights, 'context>>(
                chunk_rows,
            )?;
        admit(
            add(
                model.model_bytes(),
                add(numerical, Backend::minimum_bytes(self, chunk_rows)?)?,
            )?,
            budget,
        )?;
        let backend = Backend::new(
            self,
            context,
            chunk_rows,
            subtract(budget, add(model.model_bytes(), numerical)?)?,
        )?;
        model.session_with_backend_for_chunk(backend, chunk_rows, budget)
    }
    /// Admission includes this source and session once. Reserve other live
    /// sessions, caller/codec/app storage separately before supplying the budget.
    pub fn new_generation_session<'model, 'a, 'context>(
        &'a self,
        model: &'model Model<'a, 'weights>,
        context: &'context HereticContext,
        options: Options,
    ) -> std::result::Result<GenerationSession<'model, 'a, 'weights, 'context>, session::Error>
    {
        self.generation_with_prefill(model, context, options, true)
    }
    #[cfg(test)]
    pub(super) fn reference_generation_session<'model, 'a, 'context>(
        &'a self,
        model: &'model Model<'a, 'weights>,
        context: &'context HereticContext,
        options: Options,
    ) -> std::result::Result<GenerationSession<'model, 'a, 'weights, 'context>, session::Error>
    {
        self.generation_with_prefill(model, context, options, false)
    }
    fn generation_with_prefill<'model, 'a, 'context>(
        &'a self,
        model: &'model Model<'a, 'weights>,
        context: &'context HereticContext,
        options: Options,
        last_row: bool,
    ) -> std::result::Result<GenerationSession<'model, 'a, 'weights, 'context>, session::Error>
    {
        let create = || -> HResult<Backend<'a, 'weights, 'context>> {
            self.check_model(model, options.memory_limit_bytes)?;
            let raw = model
                .planned_backend_session_bytes_for_capacity::<Backend<'a, 'weights, 'context>>(
                    options.capacity,
                    options.prefill_chunk_size,
                )?;
            let extra = GenerationSession::<'model, 'a, 'weights, 'context>::planned_wrapper_bytes(
                options.capacity,
            )?;
            let required = add(model.model_bytes(), add(raw, extra)?)?;
            admit(
                add(
                    required,
                    Backend::minimum_bytes(self, options.prefill_chunk_size)?,
                )?,
                options.memory_limit_bytes,
            )?;
            let backend = Backend::new(
                self,
                context,
                options.prefill_chunk_size,
                subtract(options.memory_limit_bytes, required)?,
            )?;
            Ok(backend)
        };
        engine::generation::validate_options(model, options)?;
        let backend = create().map_err(session::Error::from)?;
        let mut session = engine::GenerationSession::with_backend(model, backend, options)?;
        if last_row {
            session.use_last_row_prefill();
        }
        Ok(session)
    }
    #[cfg(test)]
    pub(super) fn faulted_generation_session<'model, 'a, 'context>(
        &'a self,
        model: &'model Model<'a, 'weights>,
        context: &'context HereticContext,
        options: Options,
        after_projections: usize,
    ) -> std::result::Result<GenerationSession<'model, 'a, 'weights, 'context>, session::Error>
    {
        engine::generation::validate_options(model, options)?;
        let create = || -> HResult<Backend<'a, 'weights, 'context>> {
            self.check_model(model, options.memory_limit_bytes)?;
            let raw = model
                .planned_backend_session_bytes_for_capacity::<Backend<'a, 'weights, 'context>>(
                    options.capacity,
                    options.prefill_chunk_size,
                )?;
            let extra = GenerationSession::<'model, 'a, 'weights, 'context>::planned_wrapper_bytes(
                options.capacity,
            )?;
            let required = add(model.model_bytes(), add(raw, extra)?)?;
            admit(
                add(
                    required,
                    Backend::minimum_bytes(self, options.prefill_chunk_size)?,
                )?,
                options.memory_limit_bytes,
            )?;
            let mut backend = Backend::new(
                self,
                context,
                options.prefill_chunk_size,
                subtract(options.memory_limit_bytes, required)?,
            )?;
            backend.test_fail_after = Some(after_projections);
            Ok(backend)
        };
        let mut session = engine::GenerationSession::with_backend(
            model,
            create().map_err(session::Error::from)?,
            options,
        )?;
        session.use_last_row_prefill();
        Ok(session)
    }
}

pub struct Backend<'a, 'weights, 'context> {
    source: &'a Norms<'weights>,
    projections: Vec<Projection<'context>>,
    gathers: [Gather<'context>; 2],
    bf16: Bf16Workspace<'context>,
    failed: bool,
    #[cfg(test)]
    test_fail_after: Option<usize>,
}
impl sealed::Operations for Backend<'_, '_, '_> {}
impl<'a, 'weights> Operations<MatrixRef<'a, 'weights>> for Backend<'a, 'weights, '_> {
    type Error = super::super::hybrid::Error;
    fn project(
        &mut self,
        input: &[f32],
        weight: &MatrixRef<'a, 'weights>,
        output: &mut [f32],
        tokens: usize,
    ) -> HResult<()> {
        self.ready()?;
        if !std::ptr::eq(weight.source, self.source) {
            return Err(Error::Options("Heretic foreign projection").into());
        }
        #[cfg(test)]
        if let Some(remaining) = self.test_fail_after.as_mut() {
            if *remaining == 0 {
                self.failed = true;
                return Err(Error::Options("injected Heretic projection failure").into());
            }
            *remaining -= 1;
        }
        let result = match self.source.source.slots[weight.index] {
            MatrixSlot::Affine(i) => {
                let slot = self.source.slots[weight.index]
                    .ok_or(Error::Options("Heretic matrix is gather-only"))?;
                self.projections[slot].apply(
                    &self.source.source.storage.matrices[i],
                    input,
                    output,
                    tokens,
                )
            }
            MatrixSlot::Bf16 => self
                .bf16
                .apply(&self.source.source.bf16, input, output, tokens),
        };
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
    ) -> HResult<()> {
        self.ready()?;
        if !std::ptr::eq(weight.source, self.source) || ids.len() > engine::MAX_POSITIONS {
            return Err(Error::Options("Heretic gather source or count differs").into());
        }
        let slot = self
            .source
            .gathers
            .iter()
            .position(|&i| i == weight.index)
            .ok_or(Error::Options("Heretic gather role differs"))?;
        let MatrixSlot::Affine(index) = self.source.source.slots[weight.index] else {
            return Err(Error::Options("Heretic gather format differs").into());
        };
        let mut converted = [0usize; engine::MAX_POSITIONS];
        for (to, &id) in converted.iter_mut().zip(ids) {
            *to = id as usize;
        }
        let result = self.gathers[slot].apply_weights(
            &self.source.source.storage.matrices[index],
            &converted[..ids.len()],
            output,
        );
        if result.is_err() {
            self.failed = true;
        }
        Ok(result?)
    }
    fn ready(&self) -> HResult<()> {
        if self.failed
            || self.bf16.is_poisoned()
            || self.projections.iter().any(Projection::is_poisoned)
            || self.gathers.iter().any(Gather::is_poisoned)
        {
            return Err(Error::Options("Heretic scratch failed; recreate and replay").into());
        }
        Ok(())
    }
    fn memory(&self) -> std::result::Result<BackendMemory, crate::model::Error> {
        self.memory_report()
            .map_err(|_| crate::model::Error::InvalidConfig("Heretic scratch accounting overflow"))
    }
}
