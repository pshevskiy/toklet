//! Short official BF16 graph over authenticated shared buffers and file-backed PLE.
use super::*;
use crate::{
    metal::gemma_affine::resident::hybrid::Error as GraphError,
    mlx::gemma4::bf16::rows::Workspace as PleWorkspace,
    model::gemma_packed::{
        self as graph, BackendMemory, MatrixDescriptor, MatrixSource, Operations,
        operations::sealed,
    },
};

type GraphResult<T> = std::result::Result<T, GraphError>;
pub type Bf16Model<'weights> = graph::Model<'weights, MatrixRef<'weights>>;
pub type Bf16Session<'model, 'weights, 'context> =
    graph::Session<'model, 'weights, MatrixRef<'weights>, Backend<'weights, 'context>>;

#[cfg(test)]
type QualitySession<'model, 'weights, 'context> =
    graph::GenerationSession<'model, 'weights, MatrixRef<'weights>, Backend<'weights, 'context>>;

pub struct MatrixRef<'weights> {
    source: &'weights Bf16Weights,
    index: usize,
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

    fn build_model(&self, position_limit: usize) -> GraphResult<Bf16Model<'_>> {
        let weights = self.bindings.graph_weights_for(
            |index| {
                Ok(MatrixRef {
                    source: self,
                    index,
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
        let numerical = model
            .planned_backend_session_bytes_for_chunk::<Backend<'weights, 'context>>(chunk_rows)?;
        let fixed = add(model.model_bytes(), numerical)?;
        let backend = self.admit_backend(model, context, chunk_rows, fixed, budget)?;
        model.session_with_backend_for_chunk(backend, chunk_rows, budget)
    }

    #[cfg(test)]
    pub(crate) fn new_quality_generation_session<'model, 'weights, 'context>(
        &'weights self,
        model: &'model Bf16Model<'weights>,
        context: &'context Bf16Context,
        options: crate::session::Options,
    ) -> std::result::Result<QualitySession<'model, 'weights, 'context>, crate::session::Error>
    {
        use crate::session::Error as SessionError;
        graph::generation::validate_options(model, options)?;
        let numerical = model
            .planned_backend_session_bytes_for_capacity::<Backend<'weights, 'context>>(
                options.capacity,
                options.prefill_chunk_size,
            )?;
        let wrapper =
            QualitySession::<'model, 'weights, 'context>::planned_wrapper_bytes(options.capacity)
                .map_err(GraphError::from)?;
        let fixed = add(
            model.model_bytes(),
            add(numerical, wrapper).map_err(GraphError::from)?,
        )
        .map_err(GraphError::from)?;
        let backend = self
            .admit_backend(
                model,
                context,
                options.prefill_chunk_size,
                fixed,
                options.memory_limit_bytes,
            )
            .map_err(|error| match error {
                GraphError::Metal(Error::Budget { required, limit }) => {
                    SessionError::MemoryBudget { required, limit }
                }
                other => other.into(),
            })?;
        graph::GenerationSession::with_backend(model, backend, options)
    }

    fn admit_backend<'weights, 'context>(
        &'weights self,
        model: &Bf16Model<'weights>,
        context: &'context Bf16Context,
        chunk_rows: usize,
        fixed: usize,
        budget: usize,
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
        check_budget(add(fixed, minimum)?, budget)?;
        Backend::new(self, context, chunk_rows, counts, budget - fixed)
    }
}

struct Counts {
    input: usize,
    tile: usize,
    ple: usize,
}

impl Counts {
    fn new(source: &Bf16Weights, chunk: usize) -> Result<Self> {
        let matrices = source.matrices.iter().flatten();
        let columns = matrices.clone().map(|m| m.columns).max().unwrap_or(0);
        let rows = matrices.map(|m| m.rows.min(TILE_ROWS)).max().unwrap_or(0);
        Ok(Self {
            input: multiply(chunk, columns)?,
            tile: multiply(chunk, rows)?,
            ple: source.source.workspace_bytes(chunk)?,
        })
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
    ) -> GraphResult<Self> {
        let mut tile = reserve::<f32>(counts.tile)?;
        let with_tile = counts.memory(tile.capacity(), counts.ple)?.total()?;
        check_budget(with_tile, budget)?;
        let ple_budget = add(counts.ple, budget - with_tile)?;
        let ple = source
            .source
            .workspace(chunk, ple_budget)
            .map_err(Error::from)?;
        let memory = counts.memory(tile.capacity(), ple.owned_bytes())?;
        check_budget(memory.total()?, budget)?;
        tile.resize(counts.tile, 0.);
        let gpu = &context.0.gpu;
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
            #[cfg(test)]
            fault: None,
        })
    }

    #[cfg(test)]
    pub(crate) fn set_fault(&mut self, fault: Fault) {
        self.fault = Some(fault);
    }
}

impl sealed::Operations for Backend<'_, '_> {}

impl<'weights> Operations<MatrixRef<'weights>> for Backend<'weights, '_> {
    type Error = GraphError;

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
        for (tile_index, weights) in matrix.tiles.iter().enumerate() {
            let first = tile_index * TILE_ROWS;
            let rows = (matrix.rows - first).min(TILE_ROWS);
            let active = &mut self.tile[..rows * tokens];
            project::run(
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
