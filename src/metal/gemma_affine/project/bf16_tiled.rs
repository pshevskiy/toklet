//! Private qualification of original compensated BF16 dots with full-row stores.
//! No staging, reset, admission dispatch, submit, wait, replay or publication.
use super::super::*;
use super::bf16_encode::Bf16Tile;
use crate::metal::memory::{BufferRange, validate_encode_ranges};
#[cfg(test)]
use std::cell::Cell;

const TILE_ROWS: usize = 8192;
const MAX_ROWS: usize = 262144;
const PIPELINE_POLICY_BYTES: usize = 64 * 1024 * 1024;
const ORIGINAL: &str = include_str!("../../gemma_affine.metal");
const SUFFIX: &str = include_str!("bf16_strided.metal");

#[derive(Clone, Copy)]
struct Geometry {
    rows: usize,
    columns: usize,
    tokens: usize,
    input: usize,
    output: usize,
    tiles: usize,
}
impl Geometry {
    fn new(rows: usize, columns: usize, tokens: usize) -> Result<Self> {
        if !(1..=MAX_ROWS).contains(&rows)
            || !(64..=12288).contains(&columns)
            || !columns.is_multiple_of(64)
            || !(1..=32).contains(&tokens)
        {
            return Err(Error::Options("BF16 tiled geometry outside bounds"));
        }
        let input = checked_product(tokens, columns)?;
        let output = checked_product(tokens, rows)?;
        if input > u32::MAX as usize || output > u32::MAX as usize {
            return Err(Error::Options("BF16 tiled shader index overflow"));
        }
        Ok(Self {
            rows,
            columns,
            tokens,
            input,
            output,
            tiles: rows.div_ceil(TILE_ROWS),
        })
    }
    fn tile(self, index: usize) -> Result<TileGeometry> {
        if index >= self.tiles {
            return Err(Error::Options("BF16 tiled index outside coverage"));
        }
        let first = checked_product(index, TILE_ROWS)?;
        let rows = (self.rows - first).min(TILE_ROWS);
        let coefficients = checked_product(rows, self.columns)? / 2;
        let groups = checked_product(self.tokens, rows)?;
        let last = checked_product(self.tokens - 1, self.rows)?
            .checked_add(first)
            .and_then(|n| n.checked_add(rows - 1))
            .ok_or(Error::Options("BF16 tiled final store overflow"))?;
        if last >= self.output || groups > u32::MAX as usize {
            return Err(Error::Options("BF16 tiled store outside output"));
        }
        let mut words = [0; 5];
        for (out, value) in
            words
                .iter_mut()
                .zip([self.tokens, self.columns, rows, self.rows, first])
        {
            *out = u32::try_from(value)
                .map_err(|_| Error::Options("BF16 tiled ABI narrowing overflow"))?;
        }
        Ok(TileGeometry {
            words,
            coefficients,
            groups,
        })
    }
}
struct TileGeometry {
    words: [u32; 5],
    coefficients: usize,
    groups: usize,
}
fn checked_product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or(Error::Options("BF16 tiled size overflow"))
}
fn physical_range(range: BufferRange<'_>) -> Result<()> {
    let offset = checked_product(range.offset, 4)?;
    let bytes = checked_product(range.len, 4)?;
    let end = offset
        .checked_add(bytes)
        .ok_or(Error::Options("BF16 tiled range overflow"))?;
    if range.len == 0
        || !range.raw.length().is_multiple_of(4)
        || end > range.raw.length()
        || range.raw.length() > isize::MAX as usize
        || range.raw.storageMode() != MTLStorageMode::Shared
        || range.raw.hazardTrackingMode() != MTLHazardTrackingMode::Tracked
        || !(range.raw.contents().as_ptr() as usize).is_multiple_of(4)
    {
        return Err(Error::Options("BF16 tiled physical range differs"));
    }
    Ok(())
}
fn coefficient_range<'a>(
    tile: &'a Buffer<u32>,
    geometry: &TileGeometry,
) -> Result<BufferRange<'a>> {
    let words = geometry
        .coefficients
        .checked_add(GUARD)
        .ok_or(Error::Options("BF16 tiled coefficient guard overflow"))?;
    if tile.len != words || tile.raw.length() != checked_product(words, 4)? {
        return Err(Error::Options("BF16 tiled coefficient allocation differs"));
    }
    // The entire immutable coefficient owner, including guards, is read-only
    // for alias preflight. The shader still reads only the active payload.
    let range = BufferRange::new(tile, 0, words)?;
    physical_range(range)?;
    Ok(range)
}

/// Cold view over the original immutable ordered tile owners. The admission
/// handle borrows the first tile but carries the maximum scanned over ALL tiles;
/// neither its maximum nor its two proof thresholds are recomputed while encoding.
/// Its coefficient pointer is never used to represent the remaining matrix.
#[derive(Clone, Copy)]
pub(in crate::metal::gemma_affine) struct Bf16MatrixView<'a> {
    tiles: &'a [Buffer<u32>],
    rows: usize,
    columns: usize,
    admission: Bf16Tile<'a>,
}
impl<'a> Bf16MatrixView<'a> {
    /// Only Matrix::borrowed_tiled supplies loader-authenticated source maxima.
    /// Tests may construct small synthetic owners with an independently known max.
    pub(in crate::metal::gemma_affine) fn new(
        context: &Context,
        tiles: &'a [Buffer<u32>],
        rows: usize,
        columns: usize,
        maximum: f64,
    ) -> Result<Self> {
        context.projection_pipeline(16)?;
        let g = Geometry::new(rows, columns, 1)?;
        if tiles.len() != g.tiles {
            return Err(Error::Options("BF16 tiled coefficient count differs"));
        }
        for (index, tile) in tiles.iter().enumerate() {
            let range = coefficient_range(tile, &g.tile(index)?)?;
            if !std::ptr::eq(&*range.raw.device(), &*context.gpu.device) {
                return Err(Error::Options("BF16 tiled foreign coefficient device"));
            }
        }
        let first_rows = rows.min(TILE_ROWS);
        let admission = Bf16Tile::new(&context.gpu, first_rows, columns, &tiles[0], maximum)?;
        Ok(Self {
            tiles,
            rows,
            columns,
            admission,
        })
    }
    /// Encode existing max+L1 admission ONCE for this entire projection, after
    /// its input producer. Keep that status separate from all product statuses.
    pub(in crate::metal::gemma_affine) fn admission_tile(self) -> Bf16Tile<'a> {
        self.admission
    }
    pub(in crate::metal::gemma_affine) fn tile_count(self) -> usize {
        self.tiles.len()
    }
}

#[derive(Clone, Copy)]
pub(in crate::metal::gemma_affine) struct DeviceBuffers<'a> {
    pub input: BufferRange<'a>,
    /// Complete contiguous [tokens,total_rows], not a compact tile stripe.
    pub output: BufferRange<'a>,
    /// One caller-initialized product status, ORed by every tile without reset.
    pub status: BufferRange<'a>,
}
pub(in crate::metal::gemma_affine) struct StridedBf16Pipelines {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    pipeline: Pipeline,
    width: usize,
    #[cfg(test)]
    fail_before_tile: Cell<Option<usize>>,
}
fn validate_width(original: usize, strided: usize, maximum: usize) -> Result<()> {
    if strided == 0 || original != strided || strided > maximum || strided > u16::MAX as usize {
        return Err(Error::Unsupported(
            "BF16 strided original SIMD width differs",
        ));
    }
    Ok(())
}
impl StridedBf16Pipelines {
    pub(in crate::metal::gemma_affine) const fn policy_bytes() -> usize {
        PIPELINE_POLICY_BYTES
    }
    /// Cold compilation only; policy is an allowance, not measured residency.
    /// Original Context construction and original shader bytes stay unchanged.
    pub(in crate::metal::gemma_affine) fn new(context: &Context, available: usize) -> Result<Self> {
        if available < Self::policy_bytes() {
            return Err(Error::Budget {
                required: Self::policy_bytes(),
                limit: available,
            });
        }
        let (_, original_width) = context.projection_pipeline(16)?;
        autoreleasepool(|_| {
            let mut source = String::with_capacity(ORIGINAL.len() + 1 + SUFFIX.len());
            source.push_str(ORIGINAL);
            source.push('\n');
            source.push_str(SUFFIX);
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version2_3);
            options.setMathMode(MTLMathMode::Safe);
            options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
            if options.languageVersion() != MTLLanguageVersion::Version2_3
                || options.mathMode() != MTLMathMode::Safe
                || options.mathFloatingPointFunctions() != MTLMathFloatingPointFunctions::Precise
            {
                return Err(Error::Unsupported(
                    "BF16 strided precise policy unavailable",
                ));
            }
            let library = context
                .gpu
                .device
                .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options))
                .map_err(|e| Error::Native(e.to_string()))?;
            let function = library
                .newFunctionWithName(&NSString::from_str("gemma_bf16_product_strided"))
                .ok_or_else(|| Error::Native("BF16 strided entry absent".into()))?;
            let pipeline = context
                .gpu
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| Error::Native(e.to_string()))?;
            let width = pipeline.threadExecutionWidth();
            validate_width(
                original_width,
                width,
                pipeline.maxTotalThreadsPerThreadgroup(),
            )?;
            Ok(Self {
                device: context.gpu.device.clone(),
                pipeline,
                width,
                #[cfg(test)]
                fail_before_tile: Cell::new(None),
            })
        })
    }
    /// Preflight ALL tiles and full spans before creating the first encoder.
    /// The graph caller retains every borrowed owner/command until terminal
    /// completion and handles partial native encoding failure by draining and
    /// discarding the complete transaction. No tile result is publishable here.
    pub(in crate::metal::gemma_affine) fn encode(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        matrix: Bf16MatrixView<'_>,
        buffers: DeviceBuffers<'_>,
        tokens: usize,
    ) -> Result<()> {
        let g = Geometry::new(matrix.rows, matrix.columns, tokens)?;
        if matrix.tiles.len() != g.tiles
            || buffers.input.len != g.input
            || buffers.output.len != g.output
            || buffers.status.len != 1
        {
            return Err(Error::Options("BF16 tiled full buffer lengths differ"));
        }
        for range in [buffers.input, buffers.output, buffers.status] {
            physical_range(range)?;
        }
        // First pass creates no encoders and never mutates the command/buffers.
        for (index, tile) in matrix.tiles.iter().enumerate() {
            let coefficient = coefficient_range(tile, &g.tile(index)?)?;
            validate_encode_ranges(
                command,
                &self.device,
                &[coefficient, buffers.input, buffers.output, buffers.status],
                2,
            )?;
        }
        // Second pass writes disjoint logical row stripes into one full output.
        for (index, tile) in matrix.tiles.iter().enumerate() {
            #[cfg(test)]
            if self.fail_before_tile.get() == Some(index) {
                return Err(Error::Native("injected BF16 late encoder absence".into()));
            }
            let shape = g.tile(index)?;
            let encoder = command
                .computeCommandEncoder()
                .ok_or_else(|| Error::Native("BF16 tiled encoder absent".into()))?;
            encoder.setComputePipelineState(&self.pipeline);
            // SAFETY: the complete first pass established checked u32 shape,
            // physical initialized shared tracked allocations, same-device
            // retained NotEnqueued command, exact coefficient/input/full output
            // spans and every writable nonoverlap. Full coverage makes tile
            // stripes disjoint. Shape bytes copy now. The caller owns admission,
            // resource lifetimes, completion/guards/finiteness and publication.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&tile.raw), 0, 0);
                encoder.setBuffer_offset_atIndex(
                    Some(buffers.input.raw),
                    buffers.input.offset * 4,
                    3,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(buffers.output.raw),
                    buffers.output.offset * 4,
                    4,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::from(&shape.words).cast(),
                    size_of::<[u32; 5]>(),
                    5,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(buffers.status.raw),
                    buffers.status.offset * 4,
                    6,
                );
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: shape.groups,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.width,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
