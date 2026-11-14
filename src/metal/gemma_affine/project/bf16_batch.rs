//! One checked submission for independent single-row original BF16 tiles.
//! Private callers retain authenticated same-device weights and apply the
//! existing finite/domain/admission proof before touching numerical scratch.
use super::*;

const TILE_ROWS: usize = 8192;
const MAX_ROWS: usize = 262144;

#[derive(Clone, Copy)]
pub(in crate::metal::gemma_affine) struct TiledBf16<'a> {
    pub tiles: &'a [Buffer<u32>],
    pub rows: usize,
    pub columns: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub(in crate::metal::gemma_affine) struct Counts {
    pub dispatches_encoded: usize,
    pub commands_submitted: usize,
    pub host_wait_calls: usize,
    /// Logical active copies, not measured DRAM traffic or parent scatter.
    pub input_copy_bytes: usize,
    pub output_copy_bytes: usize,
}

#[cfg(any(test, feature = "gemma-head-batch-mac-probe"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::metal::gemma_affine) enum Fault {
    AfterFirstEncode,
    AfterSubmit,
    AfterWait,
    CorruptStatus,
    CorruptGuard,
    AfterFirstCopy,
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or(Error::Options("BF16 batch size overflow"))
}
fn multiply(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or(Error::Options("BF16 batch size overflow"))
}
fn validate_matrix(matrix: TiledBf16<'_>) -> Result<()> {
    if !(1..=MAX_ROWS).contains(&matrix.rows)
        || !(64..=12288).contains(&matrix.columns)
        || !matrix.columns.is_multiple_of(64)
        || matrix.tiles.len() != matrix.rows.div_ceil(TILE_ROWS)
    {
        return Err(Error::Options("BF16 batch matrix shape differs"));
    }
    for (index, weights) in matrix.tiles.iter().enumerate() {
        let first = multiply(index, TILE_ROWS)?;
        let rows = (matrix.rows - first).min(TILE_ROWS);
        if weights.len != add(multiply(rows, matrix.columns)? / 2, GUARD)? {
            return Err(Error::Options("BF16 batch coefficient extent differs"));
        }
    }
    Ok(())
}
fn elapsed(observation: &Option<BatchObservation<'_>>) -> f64 {
    observation.as_ref().map_or(0., |o| o.trace.elapsed())
}

pub(in crate::metal::gemma_affine) fn run(
    context: &Context,
    matrix: TiledBf16<'_>,
    buffers: Buffers<'_>,
    input: &[f32],
    cpu_tile: &mut [f32],
    observation: Option<BatchObservation<'_>>,
) -> Result<Counts> {
    autoreleasepool(|_| {
        #[cfg(any(test, feature = "gemma-head-batch-mac-probe"))]
        {
            execute(context, matrix, buffers, input, cpu_tile, observation, None)
        }
        #[cfg(not(any(test, feature = "gemma-head-batch-mac-probe")))]
        {
            execute(context, matrix, buffers, input, cpu_tile, observation)
        }
    })
}

#[cfg(any(test, feature = "gemma-head-batch-mac-probe"))]
pub(in crate::metal::gemma_affine) fn run_with_fault(
    context: &Context,
    matrix: TiledBf16<'_>,
    buffers: Buffers<'_>,
    input: &[f32],
    cpu_tile: &mut [f32],
    observation: Option<BatchObservation<'_>>,
    fault: Option<Fault>,
) -> Result<Counts> {
    autoreleasepool(|_| {
        execute(
            context,
            matrix,
            buffers,
            input,
            cpu_tile,
            observation,
            fault,
        )
    })
}

fn execute(
    context: &Context,
    matrix: TiledBf16<'_>,
    buffers: Buffers<'_>,
    input: &[f32],
    cpu_tile: &mut [f32],
    mut observation: Option<BatchObservation<'_>>,
    #[cfg(any(test, feature = "gemma-head-batch-mac-probe"))] fault: Option<Fault>,
) -> Result<Counts> {
    validate_matrix(matrix)?;
    if buffers.pending.is_some()
        || input.len() != matrix.columns
        || cpu_tile.len() != matrix.rows
        || buffers.input.len < add(matrix.columns, GUARD)?
        || buffers.output.len < add(matrix.rows, GUARD)?
        || buffers.status.len != 1 + GUARD
    {
        return Err(Error::Options("BF16 batch scratch or request differs"));
    }
    if let Some(o) = &mut observation {
        let group = o.group;
        let end = add(group.first_counter_slot, group.tile_count)?;
        if end > o.capacity || end > o.records.capacity() {
            *o.overflowed = true;
            return Err(Error::Options("BF16 batch event capacity exceeded"));
        }
        if group.first_counter_slot != o.records.len()
            || group.owner_counter_slot != group.first_counter_slot
            || group.tile_count != matrix.tiles.len()
            || group.tile_count <= 1
            || group.input_rows != 1
            || group.input_features != matrix.columns
            || group.output_features != matrix.rows
        {
            return Err(Error::Options("BF16 batch group attribution differs"));
        }
    }
    let (pipeline, width) = context.projection_pipeline(16)?;
    let staging_start = elapsed(&observation);
    buffers.input.with_mut(|words| {
        words.fill(SENTINEL);
        for (word, value) in words.iter_mut().zip(input) {
            *word = value.to_bits();
        }
    });
    buffers.output.with_mut(|words| words.fill(SENTINEL));
    buffers.status.with_mut(|words| {
        words.fill(SENTINEL);
        words[0] = 0;
    });
    let staging_end = elapsed(&observation);
    let mut counts = Counts {
        input_copy_bytes: multiply(matrix.columns, 4)?,
        ..Counts::default()
    };
    let command = context.gpu.command()?;
    for (index, weights) in matrix.tiles.iter().enumerate() {
        let first = index * TILE_ROWS;
        let rows = (matrix.rows - first).min(TILE_ROWS);
        let offset = multiply(first, 4)?;
        let encoding_start = if index == 0 {
            staging_end
        } else {
            elapsed(&observation)
        };
        let (encoder, slot) = if let Some(o) = observation.as_mut() {
            let (slot, encoder) = o.trace.encoder(&command, "gemma_bf16_product")?;
            if slot != o.records.len() || slot != o.group.first_counter_slot + index {
                return Err(Error::Options("BF16 batch counter slot differs"));
            }
            o.records.push(TileRecord {
                counter_slot: slot,
                attribution: TileAttribution {
                    operation: o.group.operation,
                    source_matrix_slot: o.group.source_matrix_slot,
                    tile_index: index,
                    source_first_row: first,
                    source_rows: rows,
                    input_rows: 1,
                    output_features: matrix.rows,
                    input_features: matrix.columns,
                },
                host_staging_seconds: if index == 0 {
                    staging_end - staging_start
                } else {
                    0.
                },
                host_encoding_seconds: 0.,
                host_commit_seconds: 0.,
                host_wait_seconds: 0.,
                host_validation_and_copy_seconds: 0.,
                dispatches_encoded: 0,
                commands_submitted: 0,
                host_wait_calls: 0,
                input_copy_bytes: if index == 0 {
                    counts.input_copy_bytes
                } else {
                    0
                },
                output_copy_bytes: 0,
            });
            (encoder, Some(slot))
        } else {
            (
                command
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::Native("BF16 batch encoder absent".into()))?,
                None,
            )
        };
        encoder.setComputePipelineState(pipeline);
        let shape = [1_u32, matrix.columns as u32, rows as u32];
        // SAFETY: the caller owns authenticated same-device guarded BF16
        // tiles and admitted finite input. Checked sizes bound each unchanged
        // kernel to a disjoint output slice. Shape bytes copy immediately;
        // original atomic OR status is initialized once. Shared CPU views
        // wait for checked completion, and pending retains the whole command.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&weights.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&buffers.input.raw), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&buffers.output.raw), offset, 4);
            encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), size_of::<[u32; 3]>(), 5);
            encoder.setBuffer_offset_atIndex(Some(&buffers.status.raw), 0, 6);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: rows,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        counts.dispatches_encoded += 1;
        encoder.endEncoding();
        if let (Some(o), Some(slot)) = (observation.as_mut(), slot) {
            o.trace.finish_encoder(slot);
            o.records[slot].dispatches_encoded = 1;
            o.records[slot].host_encoding_seconds = o.trace.elapsed() - encoding_start;
        }
        #[cfg(any(test, feature = "gemma-head-batch-mac-probe"))]
        if index == 0 && fault == Some(Fault::AfterFirstEncode) {
            return Err(Error::Options("injected BF16 batch encode failure"));
        }
    }
    *buffers.pending = Some(command);
    let command = buffers.pending.as_ref().unwrap();
    let commit_start = elapsed(&observation);
    command.commit();
    let commit_end = elapsed(&observation);
    counts.commands_submitted = 1;
    if let Some(o) = observation.as_mut() {
        let record = &mut o.records[o.group.owner_counter_slot];
        record.commands_submitted = 1;
        record.host_commit_seconds = commit_end - commit_start;
    }
    #[cfg(any(test, feature = "gemma-head-batch-mac-probe"))]
    if fault == Some(Fault::AfterSubmit) {
        return Err(Error::Options("injected BF16 batch submitted failure"));
    }
    command.waitUntilCompleted();
    let wait_end = elapsed(&observation);
    counts.host_wait_calls = 1;
    if let Some(o) = observation.as_mut() {
        let record = &mut o.records[o.group.owner_counter_slot];
        record.host_wait_calls = 1;
        record.host_wait_seconds = wait_end - commit_end;
    }
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(Error::Completion);
    }
    *buffers.pending = None;
    #[cfg(any(test, feature = "gemma-head-batch-mac-probe"))]
    if fault == Some(Fault::AfterWait) {
        return Err(Error::Options("injected BF16 batch waited failure"));
    }
    #[cfg(any(test, feature = "gemma-head-batch-mac-probe"))]
    if fault == Some(Fault::CorruptStatus) {
        buffers.status.with_mut(|words| words[0] = 1);
    }
    #[cfg(any(test, feature = "gemma-head-batch-mac-probe"))]
    if fault == Some(Fault::CorruptGuard) {
        buffers.output.with_mut(|words| {
            if let Some(last) = words.last_mut() {
                *last ^= 1;
            }
        });
    }
    validate_completed(
        matrix,
        buffers.input,
        buffers.output,
        buffers.status,
        buffers.pending,
    )?;
    buffers.output.with_slice(|words| -> Result<()> {
        for first in (0..matrix.rows).step_by(TILE_ROWS) {
            let end = (first + TILE_ROWS).min(matrix.rows);
            for (value, &bits) in cpu_tile[first..end].iter_mut().zip(&words[first..end]) {
                *value = f32::from_bits(bits);
            }
            #[cfg(any(test, feature = "gemma-head-batch-mac-probe"))]
            if first == 0 && fault == Some(Fault::AfterFirstCopy) {
                return Err(Error::Options("injected BF16 batch private-copy failure"));
            }
        }
        Ok(())
    })?;
    counts.output_copy_bytes = multiply(matrix.rows, 4)?;
    if let Some(o) = observation.as_mut() {
        let record = &mut o.records[o.group.owner_counter_slot];
        record.output_copy_bytes = counts.output_copy_bytes;
        record.host_validation_and_copy_seconds = o.trace.elapsed() - wait_end;
    }
    Ok(counts)
}

pub(in crate::metal::gemma_affine) fn validate_completed(
    matrix: TiledBf16<'_>,
    input: &Buffer<u32>,
    output: &Buffer<u32>,
    status: &Buffer<u32>,
    pending: &Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
) -> Result<()> {
    if pending.is_some() {
        return Err(Error::Options("BF16 batch completion unavailable"));
    }
    validate_matrix(matrix)?;
    if input.len < add(matrix.columns, GUARD)?
        || output.len < add(matrix.rows, GUARD)?
        || status.len != 1 + GUARD
    {
        return Err(Error::Options("BF16 batch completed scratch differs"));
    }
    for (index, weights) in matrix.tiles.iter().enumerate() {
        let rows = (matrix.rows - index * TILE_ROWS).min(TILE_ROWS);
        if !weights.with_slice(|words| sentinel_tail_unchanged(words, rows * matrix.columns / 2)) {
            return Err(Error::Options("BF16 batch immutable guard changed"));
        }
    }
    if !input.with_slice(|words| sentinel_tail_unchanged(words, matrix.columns))
        || !output.with_slice(|words| sentinel_tail_unchanged(words, matrix.rows))
        || !status.with_slice(|words| sentinel_tail_unchanged(words, 1))
    {
        return Err(Error::Options("BF16 batch guard/inactive capacity changed"));
    }
    if status.with_slice(|words| words[0]) != 0
        || !output.with_slice(|words| {
            words[..matrix.rows]
                .iter()
                .all(|&bits| f32::from_bits(bits).is_finite())
        })
    {
        return Err(Error::Nonfinite);
    }
    Ok(())
}
