//! Private checked Q4/group64/BF16 encoding over the existing immutable owner.
//! No staging, reset, submission, wait, CPU view, replay or publication here.
use super::super::*;
use super::bf16_encode::AdmissionPipeline;
use crate::metal::memory::{BufferRange, validate_encode_ranges};
use std::cell::Cell;

const MAX_ROWS: usize = 262144;
const MAX_CHUNK_ROWS: usize = 16384;
const MAX_CHUNKS: usize = 64;
const MAX_FINITE_BITS: u32 = 0x7f7f_ffff;
// Separate existing product and admission family allowances, not physical PSO
// measurements. The original Context allowance and caller storage are separate.
const POLICY_BYTES: usize = 2 * 64 * 1024 * 1024;
const ORIGINAL: &str = include_str!("../../gemma_affine.metal");
const SUFFIX: &str = include_str!("affine4_strided.metal");

fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or(Error::Options("Q4 strided size overflow"))
}
#[derive(Clone, Copy)]
struct Geometry {
    rows: usize,
    columns: usize,
    tokens: usize,
    input: usize,
    output: usize,
}
impl Geometry {
    fn new(rows: usize, columns: usize, tokens: usize) -> Result<Self> {
        if !(1..=MAX_ROWS).contains(&rows)
            || !(64..=12288).contains(&columns)
            || !columns.is_multiple_of(64)
            || !(1..=32).contains(&tokens)
        {
            return Err(Error::Options("Q4 strided geometry outside bounds"));
        }
        let input = product(tokens, columns)?;
        let output = product(tokens, rows)?;
        if input > u32::MAX as usize || output > u32::MAX as usize {
            return Err(Error::Options("Q4 strided shader index overflow"));
        }
        Ok(Self {
            rows,
            columns,
            tokens,
            input,
            output,
        })
    }
    fn chunk(self, child: Layout, first: usize) -> Result<ChunkGeometry> {
        if !(1..=MAX_CHUNK_ROWS).contains(&child.rows())
            || child.columns() != self.columns
            || child.bits() != 4
            || child.group_size() != 64
            || child.metadata_dtype() != MetadataDtype::BF16
            || first
                .checked_add(child.rows())
                .is_none_or(|end| end > self.rows)
        {
            return Err(Error::Options("Q4 strided child layout/coverage differs"));
        }
        let values = product(child.rows(), self.columns)?;
        let codes = values / 8;
        let metadata = (values / 64).div_ceil(2);
        let groups = product(self.tokens, child.rows())?;
        let last = product(self.tokens - 1, self.rows)?
            .checked_add(first)
            .and_then(|n| n.checked_add(child.rows() - 1))
            .ok_or(Error::Options("Q4 strided final store overflow"))?;
        if last >= self.output
            || groups > u32::MAX as usize
            || codes > u32::MAX as usize
            || metadata > u32::MAX as usize
        {
            return Err(Error::Options("Q4 strided child index overflow"));
        }
        let mut words = [0; 5];
        for (word, value) in
            words
                .iter_mut()
                .zip([self.tokens, self.columns, child.rows(), self.rows, first])
        {
            *word = u32::try_from(value)
                .map_err(|_| Error::Options("Q4 strided ABI narrowing overflow"))?;
        }
        Ok(ChunkGeometry {
            words,
            codes,
            metadata,
            groups,
        })
    }
}
struct ChunkGeometry {
    words: [u32; 5],
    codes: usize,
    metadata: usize,
    groups: usize,
}

pub(super) fn physical_range(range: BufferRange<'_>) -> Result<()> {
    let end = range
        .offset
        .checked_add(range.len)
        .and_then(|n| n.checked_mul(4));
    if range.len == 0
        || end.is_none_or(|end| end > range.raw.length())
        || !range.raw.length().is_multiple_of(4)
        || range.raw.length() > isize::MAX as usize
        || range.raw.storageMode() != MTLStorageMode::Shared
        || range.raw.hazardTrackingMode() != MTLHazardTrackingMode::Tracked
        || !(range.raw.contents().as_ptr() as usize).is_multiple_of(4)
    {
        return Err(Error::Options("Q4 strided physical range differs"));
    }
    Ok(())
}
fn owned_range(buffer: &Buffer<u32>, active: usize) -> Result<BufferRange<'_>> {
    let words = active
        .checked_add(GUARD)
        .ok_or(Error::Options("Q4 immutable guard size overflow"))?;
    if buffer.len != words || buffer.raw.length() != product(words, 4)? {
        return Err(Error::Options("Q4 immutable physical allocation differs"));
    }
    let range = BufferRange::new(buffer, 0, words)?;
    physical_range(range)?;
    Ok(range)
}
fn coefficient_ranges<'a>(view: WeightView<'a>, g: &ChunkGeometry) -> Result<[BufferRange<'a>; 3]> {
    Ok([
        owned_range(view.codes, g.codes)?,
        owned_range(view.scales, g.metadata)?,
        owned_range(view.biases, g.metadata)?,
    ])
}

fn threshold(mut passes: impl FnMut(u32) -> bool) -> Result<u32> {
    let (mut low, mut high) = (0, MAX_FINITE_BITS);
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if passes(middle) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    if !passes(low) || (low < MAX_FINITE_BITS && passes(low + 1)) {
        return Err(Error::Options("Q4 original proof endpoint differs"));
    }
    Ok(low)
}

#[derive(Clone, Copy)]
struct InputProof {
    maximum_bits: u32,
    max_bits: u32,
    l1_bits: u32,
}
impl InputProof {
    fn new(columns: usize, maximum: f64) -> Result<Self> {
        let stored = maximum as f32;
        if !(64..=12288).contains(&columns)
            || !columns.is_multiple_of(64)
            || !maximum.is_finite()
            || maximum < 0.
            || maximum > f64::from(MAX_WEIGHT)
            || f64::from(stored) != maximum
            || stored.is_subnormal()
        {
            return Err(Error::Options(
                "Q4 original codebook maximum domain differs",
            ));
        }
        // Unlike BF16 tiles, reconstructed Q4 weights are general FP32. Their
        // exact low16 bits must never be discarded or required to be zero.
        let max_bits = threshold(|bits| {
            let (s, u) = bounds(columns, maximum, f32::from_bits(bits));
            s <= f64::from(f32::MAX) / 4. && u <= ORDERED_ERROR_LIMIT
        })?;
        let mut row = vec![0.; columns];
        let l1_bits = threshold(|bits| {
            row[0] = f32::from_bits(bits);
            let (s, u) = l1_bounds(columns, maximum, &row);
            s <= f64::from(f32::MAX) / 4. && u <= ORDERED_ERROR_LIMIT
        })?;
        Ok(Self {
            maximum_bits: stored.to_bits() & 0x7fff_ffff,
            max_bits,
            l1_bits,
        })
    }
}

/// Cold borrow over all original chunk allocations. The maximum is loader-
/// authenticated over every codebook value, including unused codes, and covers
/// the entire matrix. No coefficients are cloned, expanded or rescanned here.
#[derive(Clone, Copy)]
pub(in crate::metal::gemma_affine) struct Affine4MatrixView<'w> {
    weights: &'w Weights,
    layout: Layout,
    chunks: usize,
    maximum_chunk_rows: usize,
    proof: InputProof,
}
impl<'w> Affine4MatrixView<'w> {
    pub(in crate::metal::gemma_affine) fn new(
        context: &Context,
        weights: &'w Weights,
    ) -> Result<Self> {
        context.projection_pipeline(4)?;
        let layout = weights.layout();
        let g = Geometry::new(layout.rows(), layout.columns(), 1)?;
        if !weights.same_device(context)
            || layout.bits() != 4
            || layout.group_size() != 64
            || layout.metadata_dtype() != MetadataDtype::BF16
        {
            return Err(Error::Options(
                "Q4 original immutable owner format/device differs",
            ));
        }
        let mut first = 0usize;
        let mut chunks = 0usize;
        let mut maximum_chunk_rows = 0;
        for view in weights.views() {
            chunks += 1;
            if chunks > MAX_CHUNKS {
                return Err(Error::Options("Q4 immutable chunk count outside bounds"));
            }
            let child = g.chunk(view.layout, first)?;
            for range in coefficient_ranges(view, &child)? {
                if !std::ptr::eq(&*range.raw.device(), &*context.gpu.device) {
                    return Err(Error::Options("Q4 immutable child device differs"));
                }
            }
            first = first
                .checked_add(view.layout.rows())
                .ok_or(Error::Options("Q4 immutable row coverage overflow"))?;
            maximum_chunk_rows = maximum_chunk_rows.max(view.layout.rows());
        }
        if chunks == 0 || first != g.rows {
            return Err(Error::Options("Q4 immutable complete row coverage differs"));
        }
        let proof = InputProof::new(g.columns, weights.max_abs_weight())?;
        Ok(Self {
            weights,
            layout,
            chunks,
            maximum_chunk_rows,
            proof,
        })
    }
    pub(in crate::metal::gemma_affine) fn chunk_count(self) -> usize {
        self.chunks
    }
    pub(in crate::metal::gemma_affine) fn maximum_chunk_rows(self) -> usize {
        self.maximum_chunk_rows
    }
    pub(in crate::metal::gemma_affine) fn maximum_bits(self) -> u32 {
        self.proof.maximum_bits
    }
    pub(super) fn admission_words(self, tokens: usize) -> Result<[u32; 4]> {
        let g = Geometry::new(self.layout.rows(), self.layout.columns(), tokens)?;
        Ok([
            g.tokens as u32,
            g.columns as u32,
            self.proof.max_bits,
            self.proof.l1_bits,
        ])
    }
}
#[derive(Clone, Copy)]
pub(in crate::metal::gemma_affine) struct DeviceBuffers<'a> {
    pub input: BufferRange<'a>,
    pub output: BufferRange<'a>,
    pub admission_status: BufferRange<'a>,
    pub product_status: BufferRange<'a>,
}
#[derive(Debug, PartialEq, Eq)]
pub(in crate::metal::gemma_affine) enum Decision {
    Ready,
    ReferenceReplay,
}
/// Only after checked completion and GLOBAL guards/structural integrity. A proof
/// miss discards speculative product failures and replays SAME Weights/input via
/// unchanged Projection::apply; that path may still return its original error.
pub(in crate::metal::gemma_affine) fn decision(admission: u32, product: u32) -> Result<Decision> {
    if admission & !3 != 0 || product & !1 != 0 {
        return Err(Error::Options(
            "Q4 admission/product status contract differs",
        ));
    }
    if admission & 1 != 0 {
        return Err(Error::Nonfinite);
    }
    if admission & 2 != 0 {
        return Ok(Decision::ReferenceReplay);
    }
    if product & 1 != 0 {
        return Err(Error::Nonfinite);
    }
    Ok(Decision::Ready)
}
pub(in crate::metal::gemma_affine) struct Affine4Pipelines {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    pipeline: Pipeline,
    width: usize,
    admission: AdmissionPipeline,
    fail_before_chunk: Cell<Option<usize>>,
}
impl Affine4Pipelines {
    pub(in crate::metal::gemma_affine) const fn policy_bytes() -> usize {
        POLICY_BYTES
    }
    pub(in crate::metal::gemma_affine) fn new(context: &Context, available: usize) -> Result<Self> {
        if available < Self::policy_bytes() {
            return Err(Error::Budget {
                required: Self::policy_bytes(),
                limit: available,
            });
        }
        let (_, original_width) = context.projection_pipeline(4)?;
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
                return Err(Error::Unsupported("Q4 strided precise policy unavailable"));
            }
            let library = context
                .gpu
                .device
                .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options))
                .map_err(|e| Error::Native(e.to_string()))?;
            let function = library
                .newFunctionWithName(&NSString::from_str("gemma_affine4_product_strided"))
                .ok_or_else(|| Error::Native("Q4 strided entry absent".into()))?;
            let pipeline = context
                .gpu
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| Error::Native(e.to_string()))?;
            let width = pipeline.threadExecutionWidth();
            if width == 0
                || width != original_width
                || width > pipeline.maxTotalThreadsPerThreadgroup()
                || width > u16::MAX as usize
            {
                return Err(Error::Unsupported("Q4 strided original SIMD width differs"));
            }
            let admission = AdmissionPipeline::new(&context.gpu)?;
            Ok(Self {
                device: context.gpu.device.clone(),
                pipeline,
                width,
                admission,
                fail_before_chunk: Cell::new(None),
            })
        })
    }
    pub(in crate::metal::gemma_affine) fn preflight(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        matrix: Affine4MatrixView<'_>,
        buffers: DeviceBuffers<'_>,
        tokens: usize,
    ) -> Result<()> {
        let g = Geometry::new(matrix.layout.rows(), matrix.layout.columns(), tokens)?;
        if buffers.input.len != g.input
            || buffers.output.len != g.output
            || buffers.admission_status.len != 1
            || buffers.product_status.len != 1
        {
            return Err(Error::Options("Q4 full operation buffer lengths differ"));
        }
        for range in [
            buffers.input,
            buffers.output,
            buffers.admission_status,
            buffers.product_status,
        ] {
            physical_range(range)?;
        }
        let mut first = 0usize;
        let mut count = 0usize;
        for view in matrix.weights.views() {
            count += 1;
            if count > MAX_CHUNKS {
                return Err(Error::Options("Q4 immutable chunk count outside bounds"));
            }
            let shape = g.chunk(view.layout, first)?;
            let [codes, scales, biases] = coefficient_ranges(view, &shape)?;
            validate_encode_ranges(
                command,
                &self.device,
                &[
                    codes,
                    scales,
                    biases,
                    buffers.input,
                    buffers.output,
                    buffers.admission_status,
                    buffers.product_status,
                ],
                4,
            )?;
            first = first
                .checked_add(view.layout.rows())
                .ok_or(Error::Options("Q4 whole row coverage overflow"))?;
        }
        if count != matrix.chunks || first != g.rows {
            return Err(Error::Options(
                "Q4 complete immutable chunk coverage differs",
            ));
        }
        self.admission.preflight_affine4_with_l1(
            command,
            matrix,
            buffers.input,
            buffers.admission_status,
            tokens,
        )
    }
    /// Successful return is the actual1+chunks dispatch count. Any error must
    /// make the caller's aggregate count unavailable and require drain/discard;
    /// preserve this original error. Nothing has published and no waits occur.
    pub(in crate::metal::gemma_affine) fn encode(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        matrix: Affine4MatrixView<'_>,
        buffers: DeviceBuffers<'_>,
        tokens: usize,
    ) -> Result<usize> {
        self.preflight(command, matrix, buffers, tokens)?;
        let g = Geometry::new(matrix.layout.rows(), matrix.layout.columns(), tokens)?;
        self.admission.encode_affine4_with_l1(
            command,
            matrix,
            buffers.input,
            buffers.admission_status,
            tokens,
        )?;
        let mut first = 0usize;
        for (index, view) in matrix.weights.views().enumerate() {
            if self.fail_before_chunk.get() == Some(index) {
                return Err(Error::Native("injected Q4 late encoder absence".into()));
            }
            let shape = g.chunk(view.layout, first)?;
            let encoder = command
                .computeCommandEncoder()
                .ok_or_else(|| Error::Native("Q4 strided encoder absent".into()))?;
            encoder.setComputePipelineState(&self.pipeline);
            // SAFETY: all children/full physical coefficient guards and both
            // distinct statuses passed preflight before the first encoder.
            // Read-only sharing is allowed; every write is disjoint. The
            // same-device NotEnqueued command retains tracked resources.
            // Full-output indexing is bounded for every token/row. Caller
            // retains owners through completion/replay/atomic publication.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&view.codes.raw), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&view.scales.raw), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&view.biases.raw), 0, 2);
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
                    Some(buffers.product_status.raw),
                    buffers.product_status.offset * 4,
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
            first += view.layout.rows();
        }
        Ok(1 + matrix.chunks)
    }
}

#[cfg(test)]
mod tests;
