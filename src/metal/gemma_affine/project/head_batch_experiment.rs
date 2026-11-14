//! Test-only component screen. No graph selection or profile schema changes.
use super::*;

const TILE_ROWS: usize = 8192;
const MAX_ROWS: usize = 262144;

pub(in crate::metal::gemma_affine) use super::bf16_batch::{Counts, Fault};

/// Weights/context and caller slices have separate external owners/budgets.
/// Successful batch publication is atomic with respect to the caller slice.
pub(in crate::metal::gemma_affine) struct Scratch<'a> {
    context: &'a Context,
    tiles: &'a [Buffer<u32>],
    rows: usize,
    columns: usize,
    input_capacity: usize,
    output_capacity: usize,
    input: Buffer<u32>,
    output: Buffer<u32>,
    status: Buffer<u32>,
    tile: Vec<f32>,
    pending: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    maximum: f64,
    owned: usize,
    poisoned: bool,
    fault: Option<Fault>,
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or(Error::Options("head batch size overflow"))
}
fn multiply(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or(Error::Options("head batch size overflow"))
}
fn budget(required: usize, limit: usize) -> Result<()> {
    if required > limit {
        Err(Error::Budget { required, limit })
    } else {
        Ok(())
    }
}

impl<'a> Scratch<'a> {
    pub(in crate::metal::gemma_affine) fn new(
        context: &'a Context,
        tiles: &'a [Buffer<u32>],
        rows: usize,
        columns: usize,
        input_capacity: usize,
        output_capacity: usize,
        available_bytes: usize,
    ) -> Result<Self> {
        if !(1..=MAX_ROWS).contains(&rows)
            || !(64..=12288).contains(&columns)
            || !columns.is_multiple_of(64)
            || input_capacity < columns
            || output_capacity < rows.min(TILE_ROWS)
            || tiles.len() != rows.div_ceil(TILE_ROWS)
        {
            return Err(Error::Options("head batch geometry or capacity differs"));
        }
        // Validate the precise existing BF16 pipeline before any allocation.
        context.projection_pipeline(16)?;
        let input_words = add(input_capacity, GUARD)?;
        let output_words = add(output_capacity, GUARD)?;
        let status_words = add(1, GUARD)?;
        let gpu_bytes = multiply(add(add(input_words, output_words)?, status_words)?, 4)?;
        let owned_for = |capacity| {
            add(
                size_of::<Self>(),
                add(gpu_bytes, multiply(capacity, size_of::<f32>())?)?,
            )
        };
        budget(owned_for(output_capacity)?, available_bytes)?;
        for words in [input_words, output_words, status_words] {
            validate_buffer_bytes(multiply(words, 4)?, context.gpu.device.maxBufferLength())?;
        }
        let mut maximum = 0_f64;
        for (index, weights) in tiles.iter().enumerate() {
            let first = multiply(index, TILE_ROWS)?;
            let active_rows = (rows - first).min(TILE_ROWS);
            let active_words = multiply(active_rows, columns)? / 2;
            let expected_words = add(active_words, GUARD)?;
            let expected_bytes = multiply(expected_words, 4)?;
            if weights.len != expected_words
                || weights.raw.length() != expected_bytes
                || !std::ptr::eq(&*weights.raw.device(), &*context.gpu.device)
                || weights.raw.storageMode() != MTLStorageMode::Shared
                || weights.raw.hazardTrackingMode() != MTLHazardTrackingMode::Tracked
            {
                return Err(Error::Options("head batch BF16 tile owner differs"));
            }
            // The original BF16 domain/admission is preserved. Packed halves
            // are examined directly without constructing a temporary model.
            let tile_maximum = weights.with_slice(|words| -> Result<f64> {
                if !sentinel_tail_unchanged(words, active_words) {
                    return Err(Error::Options("head batch immutable guard differs"));
                }
                let mut largest = 0_u16;
                for &word in &words[..active_words] {
                    for half in [word as u16, (word >> 16) as u16] {
                        let bits = half & 0x7fff;
                        if bits >= 0x7f80 || (1..0x80).contains(&bits) {
                            return Err(Error::Unsupported("Gemma BF16 coefficient domain"));
                        }
                        largest = largest.max(bits);
                    }
                }
                let value = f32::from_bits(u32::from(largest) << 16);
                if value > MAX_WEIGHT {
                    return Err(Error::Unsupported("Gemma BF16 coefficient magnitude"));
                }
                Ok(f64::from(value))
            })?;
            maximum = maximum.max(tile_maximum);
        }
        let mut tile = Vec::new();
        tile.try_reserve_exact(output_capacity)
            .map_err(|_| Error::Native("head batch CPU scratch reservation failed".into()))?;
        let owned = owned_for(tile.capacity())?;
        budget(owned, available_bytes)?;
        tile.resize(output_capacity, 0.);
        Ok(Self {
            context,
            tiles,
            rows,
            columns,
            input_capacity,
            output_capacity,
            input: Buffer::new(&context.gpu, input_words)?,
            output: Buffer::new(&context.gpu, output_words)?,
            status: Buffer::new(&context.gpu, status_words)?,
            tile,
            pending: None,
            maximum,
            owned,
            poisoned: false,
            fault: None,
        })
    }

    pub(in crate::metal::gemma_affine) fn owned_bytes(&self) -> usize {
        self.owned
    }
    pub(in crate::metal::gemma_affine) fn can_batch(&self) -> bool {
        self.rows <= self.output_capacity
    }
    pub(in crate::metal::gemma_affine) fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    pub(in crate::metal::gemma_affine) fn has_pending(&self) -> bool {
        self.pending.is_some()
    }
    pub(in crate::metal::gemma_affine) fn set_fault(&mut self, fault: Fault) {
        self.fault = Some(fault);
    }

    fn validate_request(&self, input: &[f32], output: &[f32], batched: bool) -> Result<()> {
        if self.poisoned || self.pending.is_some() {
            return Err(Error::Options("head batch scratch poisoned; recreate"));
        }
        if input.len() != self.columns
            || output.len() != self.rows
            || input.iter().any(|value| !value.is_finite())
            || self.input.len != add(self.input_capacity, GUARD)?
            || self.output.len != add(self.output_capacity, GUARD)?
            || self.status.len != 1 + GUARD
            || self.tile.len() != self.output_capacity
            || (batched && !self.can_batch())
        {
            return Err(Error::Options("head batch request or scratch differs"));
        }
        admit(self.columns, self.maximum, input)
    }

    /// Original per-tile execution and scatter. Like the current graph's
    /// private parent scratch, a failed legacy output is discardable.
    pub(in crate::metal::gemma_affine) fn legacy(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<Counts> {
        self.validate_request(input, output, false)?;
        if self.fault.is_some() {
            return Err(Error::Options("head batch faults require the batched path"));
        }
        self.poisoned = true;
        let mut counts = Counts::default();
        for (index, weights) in self.tiles.iter().enumerate() {
            let first = index * TILE_ROWS;
            let rows = (self.rows - first).min(TILE_ROWS);
            super::run(
                self.context,
                Matrix::bf16(rows, self.columns, weights),
                Buffers {
                    input: &mut self.input,
                    output: &mut self.output,
                    status: &mut self.status,
                    pending: &mut self.pending,
                },
                input,
                &mut self.tile[..rows],
                1,
            )?;
            counts.dispatches_encoded += 1;
            counts.commands_submitted += 1;
            counts.host_wait_calls += 1;
            counts.input_copy_bytes += self.columns * 4;
            counts.output_copy_bytes += rows * 4;
            output[first..first + rows].copy_from_slice(&self.tile[..rows]);
        }
        self.poisoned = false;
        Ok(counts)
    }

    pub(in crate::metal::gemma_affine) fn batched(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<Counts> {
        self.validate_request(input, output, true)?;
        self.poisoned = true;
        let matrix = super::bf16_batch::TiledBf16 {
            tiles: self.tiles,
            rows: self.rows,
            columns: self.columns,
        };
        let buffers = Buffers {
            input: &mut self.input,
            output: &mut self.output,
            status: &mut self.status,
            pending: &mut self.pending,
        };
        let counts = match self.fault {
            None => super::bf16_batch::run(
                self.context,
                matrix,
                buffers,
                input,
                &mut self.tile[..self.rows],
                None,
            ),
            Some(fault) => super::bf16_batch::run_with_fault(
                self.context,
                matrix,
                buffers,
                input,
                &mut self.tile[..self.rows],
                None,
                Some(fault),
            ),
        }?;
        // Shared core has checked every GPU result and copied only into our
        // private initialized CPU scratch. No fallible work follows publish.
        output.copy_from_slice(&self.tile[..self.rows]);
        self.poisoned = false;
        Ok(counts)
    }

    fn validate_completed(&self) -> Result<()> {
        super::bf16_batch::validate_completed(
            super::bf16_batch::TiledBf16 {
                tiles: self.tiles,
                rows: self.rows,
                columns: self.columns,
            },
            &self.input,
            &self.output,
            &self.status,
            &self.pending,
        )
    }
}

impl Drop for Scratch<'_> {
    fn drop(&mut self) {
        // Only committed commands enter pending. Even an injected failure
        // after submission must drain GPU access before scratch is released.
        if let Some(command) = self.pending.take() {
            command.waitUntilCompleted();
        }
    }
}

#[cfg(all(feature = "gemma-head-batch-mac-probe", not(test)))]
pub(in crate::metal::gemma_affine) fn run_controls(start: fn(), finish: fn() -> usize) {
    tests::run_all(start, finish);
}

#[cfg(any(test, feature = "gemma-head-batch-mac-probe"))]
mod tests;
