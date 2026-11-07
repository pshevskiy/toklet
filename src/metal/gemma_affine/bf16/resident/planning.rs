//! Concrete official BF16 factory explanation. No numerical owners in inspection.
use super::*;
use crate::{
    mlx::gemma4::bf16::{
        CONFIG_BYTES,
        execution::{self, Execution, Operation, Role},
    },
    model::gemma_packed,
    session,
    weights::{self, audit::open_regular},
};
use serde::Serialize;
use std::path::Path;

type InspectResult<T> = std::result::Result<T, crate::metal::gemma_affine::resident::hybrid::Error>;

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct SelectedOperation {
    #[serde(flatten)]
    pub logical: Operation,
    pub coefficient_layout: &'static str,
    pub kernel_entry: Option<&'static str>,
    pub projection_tiles: usize,
    pub predicted_commands_per_chunk: usize,
    pub predicted_dispatches_per_chunk: usize,
    pub predicted_host_waits_per_chunk: usize,
    pub modelled_prefill_output_host_bytes: usize,
    pub modelled_decode_output_host_bytes: usize,
}

fn selected(logical: Operation, tiles: usize) -> Result<SelectedOperation> {
    let projection = logical.execution == Execution::TiledBf16MetalProjection;
    let output = if projection {
        multiply(logical.source_rows, 4)?
    } else {
        0
    };
    Ok(SelectedOperation {
        modelled_prefill_output_host_bytes: multiply(output, logical.prefill_rows)?,
        modelled_decode_output_host_bytes: multiply(output, logical.decode_rows)?,
        coefficient_layout: "row-major [out,in], original little-endian BF16, two coefficients/u32; FP32 activations/output",
        kernel_entry: projection.then_some("gemma_bf16_product"),
        projection_tiles: tiles,
        predicted_commands_per_chunk: tiles,
        predicted_dispatches_per_chunk: tiles,
        predicted_host_waits_per_chunk: tiles,
        logical,
    })
}

#[derive(Debug, Serialize)]
pub struct MinimumMemory {
    pub owner_load_admitted_bytes: usize,
    pub graph_load_admitted_bytes: usize,
    pub load_admitted_bytes: usize,
    pub shared_graph_and_codec_admitted_bytes: usize,
    pub numerical_session_inline_and_buffers_bytes: usize,
    pub generation_wrapper_bytes: usize,
    pub external_backend_bytes: usize,
    /// Overlapping part of numerical_session_inline_and_buffers_bytes.
    pub kv_cache_and_proposal_payload_bytes: usize,
    pub session_admitted_bytes: usize,
    pub codec_owner_reserved_bytes: usize,
    pub driver_context_allowance_bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct OfficialBf16Plan {
    pub schema_version: u32,
    pub route: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
    pub device_name: String,
    pub maximum_buffer_bytes: usize,
    pub projection_threadgroup_width: usize,
    pub requested_load_context: usize,
    pub requested_capacity: usize,
    pub requested_chunk_rows: usize,
    pub requested_load_budget_bytes: usize,
    pub requested_session_budget_bytes: usize,
    pub source_container_bytes: u64,
    pub resident_matrix_owners: usize,
    pub resident_gpu_tile_owners: usize,
    pub vector_owners: usize,
    pub file_backed_ple_logical_bytes: usize,
    pub excluded_media_bytes: usize,
    pub excluded_unused_shared_kv_bytes: usize,
    pub policy: &'static str,
    pub operations: Vec<SelectedOperation>,
    pub cpu_recipe: [&'static str; 5],
    pub minimum_memory: MinimumMemory,
    pub minimum_fits_load_budget: bool,
    pub minimum_fits_session_budget: bool,
    pub numerical_buffers_allocated: bool,
    pub payload_validation_required: bool,
    pub qualification_evaluated: bool,
    pub limitations: [&'static str; 5],
}

/// Query the actual device and precise BF16 pipeline, then describe a full
/// configured chunk in G2 and single-token decode. Reads bounded config/header;
/// payload/codec validation and numerical admission are deferred to loading.
pub fn inspect_official_bf16(
    directory: &Path,
    load: weights::LoadOptions,
    options: session::Options,
) -> InspectResult<OfficialBf16Plan> {
    if !(1..=OfficialBf16::MAX_POSITIONS).contains(&load.context_limit)
        || !(1..=load.context_limit).contains(&options.capacity)
        || !(1..=options.capacity.min(32)).contains(&options.prefill_chunk_size)
        || options.position_offset != 0
        || load.memory_limit_bytes == 0
        || load.memory_limit_bytes > weights::INITIAL_MEMORY_LIMIT_BYTES
        || options.memory_limit_bytes == 0
        || options.memory_limit_bytes > load.memory_limit_bytes
    {
        return Err(Error::Options("official BF16 plan requires capacity<=load<=512, chunk<=min(capacity,32), offset0 and0<session budget<=load budget<=6GiB").into());
    }
    let mut config_file = open_regular(&directory.join("config.json")).map_err(Error::from)?;
    let mut config = [0; CONFIG_BYTES];
    config_file
        .read_exact(&mut config)
        .map_err(|e| Error::Native(e.to_string()))?;
    if config_file
        .read(&mut [0])
        .map_err(|e| Error::Native(e.to_string()))?
        != 0
    {
        return Err(Error::Options("Gemma BF16 config exceeds pinned size").into());
    }
    let file = open_regular(&directory.join("model.safetensors")).map_err(Error::from)?;
    let prepared =
        Prepared::new(file, &config, weights::INITIAL_MEMORY_LIMIT_BYTES).map_err(Error::from)?;
    let source = prepared.minimum_source_memory().map_err(Error::from)?;
    let plan = prepared.plan();
    let context = Bf16Context::new(weights::INITIAL_MEMORY_LIMIT_BYTES)?;
    let maximum = context.0.gpu.device.maxBufferLength();
    let storage = plan_storage(plan, maximum)?;
    let reserved = OfficialBf16::reserved_bytes()?;
    let owner_load = add(
        add(storage, source.load_bytes)?,
        add(
            add(CONTEXT_ALLOWANCE_BYTES, VECTOR_STAGE_BYTES)?,
            add(reserved, CONFIG_BYTES)?,
        )?,
    )?;
    let (metadata, numerical_session) =
        gemma_packed::official_bf16_planned_layout::<graph::MatrixRef<'_>, graph::Backend<'_, '_>>(
            load.context_limit,
            options.capacity,
            options.prefill_chunk_size,
        )?;
    let shared = add(
        add(add(storage, source.retained_bytes)?, metadata)?,
        reserved,
    )?;
    let wrapper =
        graph::Bf16GenerationSession::<'_, '_, '_>::planned_wrapper_bytes(options.capacity)?;
    let backend = graph::Counts::from_plan(
        plan,
        options.prefill_chunk_size,
        prepared
            .workspace_bytes(options.prefill_chunk_size)
            .map_err(Error::from)?,
    )?
    .minimum_memory()?
    .total()?;
    let graph_load = add(shared, CONTEXT_ALLOWANCE_BYTES)?;
    let load_admitted = owner_load.max(graph_load);
    let admitted = add(add(add(shared, numerical_session)?, wrapper)?, backend)?;
    let logical = execution::inspect(plan, options.prefill_chunk_size).map_err(Error::from)?;
    let mut operations = Vec::new();
    operations
        .try_reserve_exact(logical.operations.len())
        .map_err(|_| Error::Native("Gemma operation inventory reservation failed".into()))?;
    for operation in logical.operations {
        let tiles = if operation.execution == Execution::TiledBf16MetalProjection {
            operation.source_rows.div_ceil(TILE_ROWS)
        } else {
            0
        };
        operations.push(selected(operation, tiles)?);
    }
    let tensors = plan.text_tensors().values();
    let gpu_matrices = tensors
        .clone()
        .filter(|t| t.shape.len() == 2 && !std::ptr::eq(*t, plan.per_layer_embedding()));
    Ok(OfficialBf16Plan {
        schema_version: 1,
        route: "official_gemma_bf16",
        repository: OfficialBf16::REPOSITORY,
        revision: OfficialBf16::REVISION,
        device_name: context.device_name(),
        maximum_buffer_bytes: maximum,
        projection_threadgroup_width: context.0.width,
        requested_load_context: load.context_limit,
        requested_capacity: options.capacity,
        requested_chunk_rows: options.prefill_chunk_size,
        requested_load_budget_bytes: load.memory_limit_bytes,
        requested_session_budget_bytes: options.memory_limit_bytes,
        source_container_bytes: crate::mlx::gemma4::bf16::CONTAINER_BYTES,
        resident_matrix_owners: gpu_matrices.clone().count(),
        resident_gpu_tile_owners: gpu_matrices.map(|t| t.shape[0].div_ceil(TILE_ROWS)).sum(),
        vector_owners: tensors.filter(|t| t.shape.len() == 1).count(),
        file_backed_ple_logical_bytes: plan.per_layer_embedding().range.len(),
        excluded_media_bytes: plan.media_bytes(),
        excluded_unused_shared_kv_bytes: plan.unused_shared_kv_bytes(),
        policy: logical.policy,
        operations,
        cpu_recipe: logical.cpu_recipe,
        minimum_memory: MinimumMemory {
            owner_load_admitted_bytes: owner_load,
            graph_load_admitted_bytes: graph_load,
            load_admitted_bytes: load_admitted,
            shared_graph_and_codec_admitted_bytes: shared,
            numerical_session_inline_and_buffers_bytes: numerical_session,
            generation_wrapper_bytes: wrapper,
            external_backend_bytes: backend,
            kv_cache_and_proposal_payload_bytes: gemma_packed::official_bf16_planned_kv(
                options.capacity,
                options.prefill_chunk_size,
            )?,
            session_admitted_bytes: admitted,
            codec_owner_reserved_bytes: reserved,
            driver_context_allowance_bytes: CONTEXT_ALLOWANCE_BYTES,
        },
        minimum_fits_load_budget: load_admitted <= load.memory_limit_bytes,
        minimum_fits_session_budget: admitted <= options.memory_limit_bytes,
        numerical_buffers_allocated: false,
        payload_validation_required: true,
        qualification_evaluated: false,
        limitations: [
            "Minimum capacities only; allocator overreservation may increase actual admission. Loading/session creation revalidate.",
            "Codec and whole tensor payload are not authenticated by inspection; fixed source metadata alone is insufficient.",
            "One full configured chunk/single-token decode recipe, not request work or observed dispatch counters. CPU stages are not individually timed.",
            "Modelled host-visible active projection output bytes exclude guards/status and are not measured DRAM/DMA traffic; shared storage counted once.",
            "Driver64MiB and codec owner allowance are policies, not measured residency; no performance/quality/device qualification.",
        ],
    })
}

#[derive(Debug, Serialize)]
pub struct RuntimeInventory {
    pub payload_authenticated: bool,
    pub resident_matrix_owners: usize,
    pub resident_gpu_tile_owners: usize,
    pub vector_owners: usize,
    pub retained_weight_owner_bytes: usize,
    pub operations: Vec<SelectedOperation>,
}

impl Bf16Weights {
    /// Explicit diagnostic inventory from live immutable buffer owners. The
    /// returned command/readback counts remain predictions, not GPU observations.
    pub fn runtime_inventory(&self, chunk_rows: usize) -> Result<RuntimeInventory> {
        let geometry = self.bindings.geometry();
        let ids = execution::operation_ids(&geometry).map_err(Error::from)?;
        let mut operations = Vec::new();
        operations
            .try_reserve_exact(ids.len())
            .map_err(|_| Error::Native("Gemma runtime inventory reservation failed".into()))?;
        for id in ids {
            let index = self.bindings.operation_source(id)?;
            let (rows, columns, tiles, kind) = if id.role == Role::PerLayerEmbedding {
                let (rows, columns) = self.source.shape();
                (rows, columns, 0, Execution::AuthenticatedFileBf16CpuGather)
            } else {
                let matrix = self
                    .matrices
                    .get(index)
                    .and_then(Option::as_ref)
                    .ok_or(Error::Options("Gemma runtime matrix binding absent"))?;
                if id.role == Role::TokenEmbedding {
                    (
                        matrix.rows,
                        matrix.columns,
                        0,
                        Execution::ResidentBf16CpuGather,
                    )
                } else {
                    (
                        matrix.rows,
                        matrix.columns,
                        matrix.tiles.len(),
                        Execution::TiledBf16MetalProjection,
                    )
                }
            };
            let logical = Operation {
                id,
                source_matrix_slot: index,
                source_tensor: id.tensor_name().map_err(Error::from)?,
                source_rows: rows,
                source_columns: columns,
                source_bytes: multiply(multiply(rows, columns)?, 2)?,
                prefill_rows: execution::prefill_rows(id, &geometry, chunk_rows)
                    .map_err(Error::from)?,
                decode_rows: 1,
                execution: kind,
            };
            operations.push(selected(logical, tiles)?);
        }
        Ok(RuntimeInventory {
            payload_authenticated: true,
            resident_matrix_owners: self.matrix_count(),
            resident_gpu_tile_owners: self.matrices.iter().flatten().map(|m| m.tiles.len()).sum(),
            vector_owners: self.vector_count(),
            retained_weight_owner_bytes: self.report.retained_bytes,
            operations,
        })
    }
}

#[cfg(test)]
mod tests;
