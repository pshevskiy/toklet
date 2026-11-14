//! Short official BF16 graph over authenticated shared buffers and file-backed PLE.
use super::profile::{Bf16Trace, ProfileOptions};
use super::*;
use crate::{
    metal::gemma_affine::project::bf16_batch,
    metal::gemma_affine::resident::hybrid::Error as GraphError,
    mlx::gemma4::bf16::rows::Workspace as PleWorkspace,
    model::gemma_packed::{
        self as graph, BackendMemory, MatrixDescriptor, MatrixSource, Operations,
        operations::sealed,
    },
};

type GraphResult<T> = std::result::Result<T, GraphError>;

#[derive(Clone, Copy, Default)]
struct SessionPolicy {
    profile: Option<ProfileOptions>,
    single_row_batching: bool,
}
pub type Bf16Model<'weights> = graph::Model<'weights, MatrixRef<'weights>>;
pub type Bf16Session<'model, 'weights, 'context> =
    graph::Session<'model, 'weights, MatrixRef<'weights>, Backend<'weights, 'context>>;

pub type Bf16GenerationSession<'model, 'weights, 'context> =
    graph::GenerationSession<'model, 'weights, MatrixRef<'weights>, Backend<'weights, 'context>>;

impl<'model, 'weights, 'context>
    graph::GenerationSession<'model, 'weights, MatrixRef<'weights>, Backend<'weights, 'context>>
{
    /// Explicit diagnostic export; an unselected, failed or invalidated frame
    /// is unavailable rather than represented by a zero-duration trace.
    pub fn bf16_profile(&self) -> GraphResult<profile::FrameProfile> {
        let backend = self.operations();
        backend.ready()?;
        Ok(backend
            .trace
            .as_ref()
            .ok_or(Error::Options("BF16 profiling disabled"))?
            .export()?)
    }
    pub fn bf16_profile_status(&self) -> Option<profile::ProfileStatus> {
        let backend = self.operations();
        backend.trace.as_ref().map(|trace| {
            let mut status = trace.status();
            if backend.failed || backend.pending.is_some() {
                status.state = profile::CaptureState::Invalid;
            }
            status
        })
    }
}

pub struct MatrixRef<'weights> {
    source: &'weights Bf16Weights,
    index: usize,
    operation: crate::mlx::gemma4::bf16::execution::OperationId,
}

impl sealed::Matrix for MatrixRef<'_> {}

impl MatrixSource for MatrixRef<'_> {
    fn descriptor(&self) -> MatrixDescriptor {
        if let Some(matrix) = &self.source.matrices[self.index] {
            MatrixDescriptor::Bf16 {
                rows: matrix.rows,
                columns: matrix.columns,
                file_backed: false,
            }
        } else {
            let (rows, columns) = self.source.source.shape();
            MatrixDescriptor::Bf16 {
                rows,
                columns,
                file_backed: true,
            }
        }
    }
}

impl Bf16Weights {
    /// Raw diagnostic only. Position capacity is independent of the source
    /// checkpoint's declared context and is bounded to 32 for this entry point.
    pub fn model(&self, position_limit: usize) -> GraphResult<Bf16Model<'_>> {
        if !(1..=graph::MAX_POSITIONS).contains(&position_limit) {
            return Err(
                crate::model::Error::InvalidConfig("Gemma BF16 diagnostic position limit").into(),
            );
        }
        self.build_model(position_limit)
    }

    #[cfg(test)]
    pub(crate) fn model_for_quality_probe(
        &self,
        position_limit: usize,
    ) -> GraphResult<Bf16Model<'_>> {
        self.build_model(position_limit)
    }

    pub(in crate::metal::gemma_affine::bf16) fn build_model(
        &self,
        position_limit: usize,
    ) -> GraphResult<Bf16Model<'_>> {
        let weights = self.bindings.graph_weights_for(
            |index| {
                Ok(MatrixRef {
                    source: self,
                    index,
                    operation: self.bindings.operation_identity(index)?,
                })
            },
            |index| {
                self.vectors
                    .get(index)
                    .map(Vec::as_slice)
                    .ok_or(Error::Options("Gemma BF16 graph vector role"))
            },
        )?;
        let matrix_bytes = self.matrices.iter().flatten().try_fold(0usize, |sum, m| {
            m.tiles
                .iter()
                .try_fold(sum, |sum, tile| add(sum, multiply(tile.len, 4)?))
        })?;
        let tensor_bytes = self
            .vectors
            .iter()
            .try_fold(matrix_bytes, |sum, v| add(sum, multiply(v.capacity(), 4)?))?;
        let backing = graph::Backing {
            tensor_bytes,
            metadata_bytes: self
                .report
                .retained_bytes
                .checked_sub(tensor_bytes)
                .ok_or(Error::Options("Gemma BF16 backing accounting differs"))?,
            metadata_allowance_bytes: 0,
        };
        Ok(graph::Model::from_official_bf16_parts(
            weights,
            &self.bindings.geometry(),
            backing,
            position_limit,
        )?)
    }

    pub fn new_session<'model, 'weights, 'context>(
        &'weights self,
        model: &'model Bf16Model<'weights>,
        context: &'context Bf16Context,
        chunk_rows: usize,
        budget: usize,
    ) -> GraphResult<Bf16Session<'model, 'weights, 'context>> {
        self.new_session_with_policy(model, context, chunk_rows, budget, SessionPolicy::default())
    }

    /// Test-only raw numerical qualification of the explicitly enabled policy.
    /// Shares all validation and admission with the unchanged default factory.
    #[cfg(test)]
    pub(crate) fn new_batched_session<'model, 'weights, 'context>(
        &'weights self,
        model: &'model Bf16Model<'weights>,
        context: &'context Bf16Context,
        chunk_rows: usize,
        budget: usize,
    ) -> GraphResult<Bf16Session<'model, 'weights, 'context>> {
        self.new_session_with_policy(
            model,
            context,
            chunk_rows,
            budget,
            SessionPolicy {
                profile: None,
                single_row_batching: true,
            },
        )
    }

    fn new_session_with_policy<'model, 'weights, 'context>(
        &'weights self,
        model: &'model Bf16Model<'weights>,
        context: &'context Bf16Context,
        chunk_rows: usize,
        budget: usize,
        policy: SessionPolicy,
    ) -> GraphResult<Bf16Session<'model, 'weights, 'context>> {
        let numerical = model
            .planned_backend_session_bytes_for_chunk::<Backend<'weights, 'context>>(chunk_rows)?;
        let fixed = add(model.model_bytes(), numerical)?;
        let backend = self.admit_backend(model, context, chunk_rows, fixed, budget, policy)?;
        #[cfg(test)]
        if policy.single_row_batching {
            assert!(
                backend.single_row_batching,
                "raw candidate policy was not enabled"
            );
        }
        model.session_with_backend_for_chunk(backend, chunk_rows, budget)
    }

    pub(crate) fn new_generation_session<'model, 'weights, 'context>(
        &'weights self,
        model: &'model Bf16Model<'weights>,
        context: &'context Bf16Context,
        options: crate::session::Options,
    ) -> std::result::Result<Bf16GenerationSession<'model, 'weights, 'context>, crate::session::Error>
    {
        self.new_generation_session_with_profile(model, context, options, None)
    }

    pub(crate) fn new_generation_session_with_profile<'model, 'weights, 'context>(
        &'weights self,
        model: &'model Bf16Model<'weights>,
        context: &'context Bf16Context,
        options: crate::session::Options,
        profile: Option<ProfileOptions>,
    ) -> std::result::Result<Bf16GenerationSession<'model, 'weights, 'context>, crate::session::Error>
    {
        self.new_generation_session_with_policy(model, context, options, profile, false)
    }

    pub(crate) fn new_generation_session_with_policy<'model, 'weights, 'context>(
        &'weights self,
        model: &'model Bf16Model<'weights>,
        context: &'context Bf16Context,
        options: crate::session::Options,
        profile: Option<ProfileOptions>,
        single_row_batching: bool,
    ) -> std::result::Result<Bf16GenerationSession<'model, 'weights, 'context>, crate::session::Error>
    {
        use crate::session::Error as SessionError;
        graph::generation::validate_options(model, options)?;
        if let Some(profile) = profile {
            profile.validate().map_err(GraphError::from)?;
        }
        let fixed = Self::generation_fixed_bytes(model, options)?;
        let backend = self
            .admit_backend(
                model,
                context,
                options.prefill_chunk_size,
                fixed,
                options.memory_limit_bytes,
                SessionPolicy {
                    profile,
                    single_row_batching,
                },
            )
            .map_err(|error| match error {
                GraphError::Metal(Error::Budget { required, limit }) => {
                    SessionError::MemoryBudget { required, limit }
                }
                other => other.into(),
            })?;
        graph::GenerationSession::with_backend(model, backend, options)
    }

    fn generation_fixed_bytes(
        model: &Bf16Model<'_>,
        options: crate::session::Options,
    ) -> GraphResult<usize> {
        let numerical = model.planned_backend_session_bytes_for_capacity::<Backend<'_, '_>>(
            options.capacity,
            options.prefill_chunk_size,
        )?;
        let wrapper = Bf16GenerationSession::<'_, '_, '_>::planned_wrapper_bytes(options.capacity)?;
        Ok(add(model.model_bytes(), add(numerical, wrapper)?)?)
    }

    pub(crate) fn planned_generation_admission(
        &self,
        model: &Bf16Model<'_>,
        options: crate::session::Options,
    ) -> std::result::Result<usize, crate::session::Error> {
        graph::generation::validate_options(model, options)?;
        let fixed = Self::generation_fixed_bytes(model, options)?;
        let counts = Counts::new(self, options.prefill_chunk_size).map_err(GraphError::from)?;
        let backend = counts
            .memory(counts.tile, counts.ple)
            .map_err(GraphError::from)?
            .total()
            .map_err(GraphError::from)?;
        add(fixed, backend)
            .map_err(GraphError::from)
            .map_err(Into::into)
    }

    fn admit_backend<'weights, 'context>(
        &'weights self,
        model: &Bf16Model<'weights>,
        context: &'context Bf16Context,
        chunk_rows: usize,
        fixed: usize,
        budget: usize,
        policy: SessionPolicy,
    ) -> GraphResult<Backend<'weights, 'context>> {
        if !std::ptr::eq(&*self.device, &*context.0.gpu.device)
            || model
                .matrix_handles()
                .any(|m| !std::ptr::eq(m.source, self))
        {
            return Err(Error::Options("Gemma BF16 model/source/device differs").into());
        }
        let counts = Counts::new(self, chunk_rows)?;
        let minimum = counts.memory(counts.tile, counts.ple)?.total()?;
        let diagnostic = if policy.profile.is_some() {
            profile::ALLOWANCE_BYTES
        } else {
            0
        };
        check_budget(add(add(fixed, minimum)?, diagnostic)?, budget)?;
        Backend::new(self, context, chunk_rows, counts, budget - fixed, policy)
    }
}

pub(super) struct Counts {
    input: usize,
    tile: usize,
    ple: usize,
}

impl Counts {
    fn new(source: &Bf16Weights, chunk: usize) -> Result<Self> {
        let matrices = source.matrices.iter().flatten();
        let columns = matrices.clone().map(|m| m.columns).max().unwrap_or(0);
        let rows = matrices.map(|m| m.rows.min(TILE_ROWS)).max().unwrap_or(0);
        Self::from_dimensions(columns, rows, source.source.workspace_bytes(chunk)?, chunk)
    }
    pub(super) fn from_plan(plan: &TextPlan, chunk: usize, ple: usize) -> Result<Self> {
        let tensors = plan
            .text_tensors()
            .values()
            .filter(|t| t.shape.len() == 2 && !std::ptr::eq(*t, plan.per_layer_embedding()));
        let columns = tensors.clone().map(|t| t.shape[1]).max().unwrap_or(0);
        let rows = tensors
            .map(|t| t.shape[0].min(TILE_ROWS))
            .max()
            .unwrap_or(0);
        Self::from_dimensions(columns, rows, ple, chunk)
    }
    fn from_dimensions(columns: usize, rows: usize, ple: usize, chunk: usize) -> Result<Self> {
        Ok(Self {
            input: multiply(chunk, columns)?,
            tile: multiply(chunk, rows)?,
            ple,
        })
    }
    pub(super) fn minimum_memory(&self) -> Result<BackendMemory> {
        self.memory(self.tile, self.ple)
    }

    fn memory(&self, cpu_tile: usize, ple_owned: usize) -> Result<BackendMemory> {
        let gpu = multiply(add(add(self.input, self.tile)?, 1 + 3 * GUARD)?, 4)?;
        let host = add(
            multiply(cpu_tile, 4)?,
            ple_owned
                .checked_sub(size_of::<PleWorkspace>())
                .ok_or(Error::Options(
                    "Gemma BF16 PLE workspace accounting differs",
                ))?,
        )?;
        Ok(BackendMemory {
            scratch_bytes: add(gpu, host)?,
            metadata_bytes: 0,
            metadata_allowance_bytes: CONTEXT_ALLOWANCE_BYTES,
        })
    }
}

/// Private construction binds all scratch and immutable owners to one session.
/// Partial projection tiles touch only private graph scratch. A late failure
/// poisons the backend and requires recreation; published state stays valid.
pub struct Backend<'weights, 'context> {
    source: &'weights Bf16Weights,
    context: &'context Context,
    chunk: usize,
    input: Buffer<u32>,
    output: Buffer<u32>,
    status: Buffer<u32>,
    tile: Vec<f32>,
    ple: PleWorkspace,
    pending: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    memory: BackendMemory,
    failed: bool,
    trace: Option<Box<Bf16Trace>>,
    single_row_batching: bool,
    #[cfg(test)]
    fault: Option<Fault>,
}

#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fault {
    AfterFirstHeadTile,
    AfterPle,
}

impl<'weights, 'context> Backend<'weights, 'context> {
    fn new(
        source: &'weights Bf16Weights,
        context: &'context Bf16Context,
        chunk: usize,
        counts: Counts,
        budget: usize,
        policy: SessionPolicy,
    ) -> GraphResult<Self> {
        let diagnostic = if policy.profile.is_some() {
            profile::ALLOWANCE_BYTES
        } else {
            0
        };
        let budget = budget
            .checked_sub(diagnostic)
            .ok_or(Error::Options("BF16 diagnostic budget missing"))?;
        let mut tile = reserve::<f32>(counts.tile)?;
        let with_tile = counts.memory(tile.capacity(), counts.ple)?.total()?;
        check_budget(with_tile, budget)?;
        let ple_budget = add(counts.ple, budget - with_tile)?;
        let ple = source
            .source
            .workspace(chunk, ple_budget)
            .map_err(Error::from)?;
        let mut memory = counts.memory(tile.capacity(), ple.owned_bytes())?;
        check_budget(memory.total()?, budget)?;
        memory.metadata_allowance_bytes = add(memory.metadata_allowance_bytes, diagnostic)?;
        tile.resize(counts.tile, 0.);
        let gpu = &context.0.gpu;
        let trace = policy
            .profile
            .map(|options| {
                Bf16Trace::new(source, gpu, options, policy.single_row_batching, tile.len())
                    .map(Box::new)
            })
            .transpose()?;
        Ok(Self {
            source,
            context: &context.0,
            chunk,
            input: Buffer::new(gpu, counts.input + GUARD)?,
            output: Buffer::new(gpu, counts.tile + GUARD)?,
            status: Buffer::new(gpu, 1 + GUARD)?,
            tile,
            ple,
            pending: None,
            memory,
            failed: false,
            trace,
            single_row_batching: policy.single_row_batching,
            #[cfg(test)]
            fault: None,
        })
    }

    #[cfg(test)]
    pub(crate) fn set_fault(&mut self, fault: Fault) {
        self.fault = Some(fault);
    }
    #[cfg(test)]
    pub(crate) fn limit_profile_events_for_test(&mut self, capacity: usize) {
        self.trace
            .as_mut()
            .expect("profile enabled")
            .limit_events_for_test(capacity);
    }
}

impl sealed::Operations for Backend<'_, '_> {}

impl<'weights> Operations<MatrixRef<'weights>> for Backend<'weights, '_> {
    type Error = GraphError;

    fn begin_generation_step(&mut self, position: usize, rows: usize, last_row_only: bool) {
        if let Some(trace) = &mut self.trace {
            trace.begin(&self.context.gpu, position, rows, last_row_only);
        }
    }
    fn finish_generation_step(&mut self, committed: Option<usize>) {
        if let Some(trace) = &mut self.trace {
            trace.finish(&self.context.gpu, committed);
        }
    }
    fn generation_history_replaced(&mut self) {
        if let Some(trace) = &mut self.trace {
            trace.replace_history();
        }
    }

    fn project(
        &mut self,
        input: &[f32],
        weight: &MatrixRef<'weights>,
        output: &mut [f32],
        tokens: usize,
    ) -> GraphResult<()> {
        self.ready()?;
        if !std::ptr::eq(weight.source, self.source) {
            return Err(Error::Options("Gemma BF16 foreign projection").into());
        }
        let matrix = self.source.matrices[weight.index]
            .as_ref()
            .ok_or(Error::Options("Gemma BF16 file table cannot project"))?;
        if !(1..=self.chunk).contains(&tokens)
            || input.len() != tokens * matrix.columns
            || output.len() != tokens * matrix.rows
            || input.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Options("Gemma BF16 graph projection shape differs").into());
        }
        admit(matrix.columns, matrix.maximum, input)?;
        self.failed = true;
        let mut operation = weight.operation;
        if operation.role == crate::mlx::gemma4::bf16::execution::Role::TokenEmbedding {
            operation.role = crate::mlx::gemma4::bf16::execution::Role::OutputHead;
        }
        if self.single_row_batching
            && tokens == 1
            && matrix.tiles.len() > 1
            && matrix.rows <= self.tile.len()
            && self
                .output
                .len
                .checked_sub(GUARD)
                .is_some_and(|n| matrix.rows <= n)
        {
            let observation = match self.trace.as_mut() {
                Some(trace) => trace.register_batch(
                    operation,
                    weight.index,
                    matrix.rows,
                    matrix.columns,
                    matrix.tiles.len(),
                )?,
                None => None,
            };
            let tiled = bf16_batch::TiledBf16 {
                tiles: &matrix.tiles,
                rows: matrix.rows,
                columns: matrix.columns,
            };
            let buffers = project::Buffers {
                input: &mut self.input,
                output: &mut self.output,
                status: &mut self.status,
                pending: &mut self.pending,
            };
            #[cfg(test)]
            {
                // Preserve the late fault after real GPU work and the first
                // private CPU tile, before parent graph/KV publication.
                let fault = (weight.index == self.source.bindings.embedding_index()
                    && self.fault == Some(Fault::AfterFirstHeadTile))
                .then_some(bf16_batch::Fault::AfterFirstCopy);
                bf16_batch::run_with_fault(
                    self.context,
                    tiled,
                    buffers,
                    input,
                    &mut self.tile[..matrix.rows],
                    observation,
                    fault,
                )?;
            }
            #[cfg(not(test))]
            bf16_batch::run(
                self.context,
                tiled,
                buffers,
                input,
                &mut self.tile[..matrix.rows],
                observation,
            )?;
            output.copy_from_slice(&self.tile[..matrix.rows]);
            self.failed = false;
            return Ok(());
        }
        for (tile_index, weights) in matrix.tiles.iter().enumerate() {
            let first = tile_index * TILE_ROWS;
            let rows = (matrix.rows - first).min(TILE_ROWS);
            let active = &mut self.tile[..rows * tokens];
            let observation = self.trace.as_mut().and_then(|trace| {
                trace.observation(project::TileAttribution {
                    operation,
                    source_matrix_slot: weight.index,
                    tile_index,
                    source_first_row: first,
                    source_rows: rows,
                    input_rows: tokens,
                    output_features: matrix.rows,
                    input_features: matrix.columns,
                })
            });
            project::run_observed(
                self.context,
                project::Matrix::bf16(rows, matrix.columns, weights),
                project::Buffers {
                    input: &mut self.input,
                    output: &mut self.output,
                    status: &mut self.status,
                    pending: &mut self.pending,
                },
                input,
                active,
                tokens,
                observation,
            )?;
            for (to, from) in output
                .chunks_exact_mut(matrix.rows)
                .zip(active.chunks_exact(rows))
            {
                to[first..first + rows].copy_from_slice(from);
            }
            #[cfg(test)]
            if tile_index == 0
                && weight.index == self.source.bindings.embedding_index()
                && self.fault == Some(Fault::AfterFirstHeadTile)
            {
                return Err(Error::Options("injected Gemma BF16 late head tile failure").into());
            }
        }
        self.failed = false;
        Ok(())
    }

    fn gather(
        &mut self,
        weight: &MatrixRef<'weights>,
        ids: &[u32],
        output: &mut [f32],
    ) -> GraphResult<()> {
        self.ready()?;
        if !std::ptr::eq(weight.source, self.source) || ids.len() > self.chunk {
            return Err(Error::Options("Gemma BF16 foreign gather or row bound").into());
        }
        let begin = self.trace.as_ref().and_then(|trace| trace.gather_start());
        if weight.index == self.source.bindings.embedding_index() {
            self.source.gather_embeddings(ids, output)?;
        } else if weight.index == self.source.bindings.ple_embedding_index() {
            self.failed = true;
            self.source
                .source
                .gather(&mut self.ple, ids, output)
                .map_err(Error::from)?;
            #[cfg(test)]
            if self.fault == Some(Fault::AfterPle) {
                return Err(Error::Options("injected Gemma BF16 late PLE failure").into());
            }
            self.failed = false;
        } else {
            return Err(Error::Options("Gemma BF16 graph gather role differs").into());
        }
        if let (Some(trace), Some(begin)) = (&mut self.trace, begin) {
            self.failed = true;
            let columns = match weight.descriptor() {
                MatrixDescriptor::Bf16 { columns, .. } => columns,
                _ => unreachable!(),
            };
            trace.gather(weight.operation, weight.index, ids.len(), columns, begin)?;
            self.failed = false;
        }
        Ok(())
    }

    fn ready(&self) -> GraphResult<()> {
        if self.failed || self.pending.is_some() {
            return Err(Error::Options("Gemma BF16 graph backend poisoned; recreate").into());
        }
        Ok(())
    }

    fn memory(&self) -> std::result::Result<BackendMemory, crate::model::Error> {
        Ok(self.memory)
    }
}

impl Drop for Backend<'_, '_> {
    fn drop(&mut self) {
        // Only submitted commands are stored here. Failed/injected work must
        // finish GPU access before any session-owned shared scratch is freed.
        if let Some(command) = self.pending.take() {
            command.waitUntilCompleted();
        }
    }
}
