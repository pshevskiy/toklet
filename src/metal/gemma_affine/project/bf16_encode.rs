//! Checked encode-only seam for one original BF16 tile and its max proof.
//!
//! This is a private primitive experiment. The ordinary synchronous projection
//! route, original shader arithmetic, host admit/L1 and publication stay intact.
use super::super::*;
use crate::metal::memory::{BufferRange, validate_encode_ranges};

const MAX_FINITE_BITS: u32 = 0x7f7f_ffff;
const SINGLE_TILE_ROWS: usize = 8192;
const INPUT_NONFINITE: u32 = 1;
const MAX_PROOF_MISS: u32 = 2;

#[derive(Clone, Copy)]
struct MaxInputProof {
    bits: u32,
}
impl MaxInputProof {
    /// Cold search through exactly the unchanged upward-rounded scalar proof.
    /// Nonnegative finite FP32 bit order equals value order. Both original
    /// bounds are monotone here; no analytically rounded threshold is used.
    fn new(columns: usize, maximum: f64) -> Result<Self> {
        let stored = maximum as f32;
        if !(64..=12288).contains(&columns)
            || !columns.is_multiple_of(64)
            || !maximum.is_finite()
            || maximum < 0.
            || maximum > f64::from(MAX_WEIGHT)
            || f64::from(stored) != maximum
            || stored.is_subnormal()
            || stored.to_bits() & 0xffff != 0
        {
            return Err(Error::Options("BF16 max proof source domain differs"));
        }
        let passes = |bits| {
            let (s, u) = bounds(columns, maximum, f32::from_bits(bits));
            s <= f64::from(f32::MAX) / 4. && u <= ORDERED_ERROR_LIMIT
        };
        let (mut low, mut high) = (0, MAX_FINITE_BITS);
        while low < high {
            let middle = low + (high - low).div_ceil(2);
            if passes(middle) {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        // Retain endpoint checks, including the zero-matrix/all-finite case.
        if !passes(low) || (low < MAX_FINITE_BITS && passes(low + 1)) {
            return Err(Error::Options("BF16 max proof endpoint differs"));
        }
        Ok(Self { bits: low })
    }
}

/// Largest finite FP32 magnitude accepted by the unchanged host L1 proof.
/// The singleton-nonzero row makes row_l1 exactly equal to the candidate, while
/// retaining the original upward products, epsilon and error priority contract.
#[derive(Clone, Copy)]
struct L1InputProof {
    bits: u32,
}
impl L1InputProof {
    fn new(columns: usize, maximum: f64) -> Result<Self> {
        MaxInputProof::new(columns, maximum)?;
        let mut row = vec![0.; columns];
        let mut passes = |bits| {
            row[0] = f32::from_bits(bits);
            let (s, u) = l1_bounds(columns, maximum, &row);
            s <= f64::from(f32::MAX) / 4. && u <= ORDERED_ERROR_LIMIT
        };
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
            return Err(Error::Options("BF16 L1 proof endpoint differs"));
        }
        Ok(Self { bits: low })
    }
}

/// Borrows one existing immutable resident tile. Its scanned maximum, exact
/// shape and proof are fixed together at cold owner construction. No weights
/// are cloned, expanded or rescanned; Matrix::single_tile is the production
/// construction site and supplies the loader-authenticated maximum.
#[derive(Clone, Copy)]
pub(in crate::metal::gemma_affine) struct Bf16Tile<'a> {
    coefficients: BufferRange<'a>,
    rows: usize,
    columns: usize,
    proof: MaxInputProof,
    l1_proof: L1InputProof,
    maximum_bits: u32,
}
impl<'a> Bf16Tile<'a> {
    pub(in crate::metal::gemma_affine) fn new(
        gpu: &Gpu,
        rows: usize,
        columns: usize,
        coefficients: &'a Buffer<u32>,
        maximum: f64,
    ) -> Result<Self> {
        let geometry = Geometry::new(rows, columns, 1)?;
        let proof = MaxInputProof::new(columns, maximum)?;
        let l1_proof = L1InputProof::new(columns, maximum)?;
        if coefficients.len != geometry.coefficients + GUARD
            || !std::ptr::eq(&*coefficients.raw.device(), &*gpu.device)
        {
            return Err(Error::Options("BF16 single tile storage/device differs"));
        }
        Ok(Self {
            coefficients: BufferRange::new(coefficients, 0, geometry.coefficients)?,
            rows,
            columns,
            proof,
            l1_proof,
            maximum_bits: (maximum as f32).to_bits() & 0x7fff_ffff,
        })
    }

    pub(in crate::metal::gemma_affine) fn max_input_bits(self) -> u32 {
        self.proof.bits
    }
    pub(in crate::metal::gemma_affine) fn l1_input_bits(self) -> u32 {
        self.l1_proof.bits
    }
    pub(in crate::metal::gemma_affine) fn maximum_bits(self) -> u32 {
        self.maximum_bits
    }
}

struct Geometry {
    words: [u32; 3],
    coefficients: usize,
    input: usize,
    output: usize,
}
impl Geometry {
    fn new(rows: usize, columns: usize, tokens: usize) -> Result<Self> {
        if !(1..=SINGLE_TILE_ROWS).contains(&rows)
            || !(64..=12288).contains(&columns)
            || !columns.is_multiple_of(64)
            || !(1..=32).contains(&tokens)
        {
            return Err(Error::Options("BF16 single tile geometry outside bounds"));
        }
        let product = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or(Error::Options("BF16 single tile geometry overflow"))
        };
        Ok(Self {
            words: [tokens as u32, columns as u32, rows as u32],
            coefficients: product(rows, columns)? / 2,
            input: product(tokens, columns)?,
            output: product(tokens, rows)?,
        })
    }
}

/// Encode the unchanged gemma_bf16_product ABI, with one SIMD group per output.
///
/// This does NOT establish dynamic input admission. The caller must encode the
/// corresponding input max check after its producer, keep admission and product
/// statuses separate, and inspect statuses in dependency order after terminal
/// completion and guard checks. A proof miss discards speculative downstream
/// output and requests the unchanged reference admit/L1 path. The caller owns
/// synchronization, finite/guard validation, poisoning and atomic publication.
pub(in crate::metal::gemma_affine) fn encode_bf16(
    context: &Context,
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    tile: Bf16Tile<'_>,
    input: BufferRange<'_>,
    output: BufferRange<'_>,
    status: BufferRange<'_>,
    tokens: usize,
) -> Result<()> {
    let g = Geometry::new(tile.rows, tile.columns, tokens)?;
    if tile.coefficients.len != g.coefficients
        || input.len != g.input
        || output.len != g.output
        || status.len != 1
    {
        return Err(Error::Options("BF16 borrowed projection lengths differ"));
    }
    let (pipeline, width) = context.projection_pipeline(16)?;
    validate_encode_ranges(
        command,
        &context.gpu.device,
        &[tile.coefficients, input, output, status],
        2,
    )?;
    let encoder = command
        .computeCommandEncoder()
        .ok_or_else(|| Error::Native("BF16 borrowed projection encoder absent".into()))?;
    encoder.setComputePipelineState(pipeline);
    // SAFETY: checked exact ABI lengths/shape, same-device retained command,
    // initialized tracked u32 storage and disjoint writable ranges. Slots 1/2
    // are absent from the BF16 shader ABI. Shape bytes copy immediately; caller
    // owns admission, completion and publication for the entire transaction.
    unsafe {
        encoder.setBuffer_offset_atIndex(
            Some(tile.coefficients.raw),
            tile.coefficients.offset * 4,
            0,
        );
        encoder.setBuffer_offset_atIndex(Some(input.raw), input.offset * 4, 3);
        encoder.setBuffer_offset_atIndex(Some(output.raw), output.offset * 4, 4);
        encoder.setBytes_length_atIndex(NonNull::from(&g.words).cast(), size_of::<[u32; 3]>(), 5);
        encoder.setBuffer_offset_atIndex(Some(status.raw), status.offset * 4, 6);
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: g.output,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
pub(in crate::metal::gemma_affine) enum AdmissionDecision {
    Admitted,
    ReferenceReplay,
}
/// Decode only after checked terminal completion and guards. These bits belong
/// to a dedicated admission slot, not the original projection status slot.
pub(in crate::metal::gemma_affine) fn admission_decision(status: u32) -> Result<AdmissionDecision> {
    if status & !(INPUT_NONFINITE | MAX_PROOF_MISS) != 0 {
        return Err(Error::Options("BF16 admission status contract differs"));
    }
    if status & INPUT_NONFINITE != 0 {
        return Err(Error::Nonfinite);
    }
    Ok(if status & MAX_PROOF_MISS != 0 {
        AdmissionDecision::ReferenceReplay
    } else {
        AdmissionDecision::Admitted
    })
}

/// Cached integer-only finite/magnitude proof pipeline on the existing GPU.
/// No second queue/context or numerical scratch allocation is introduced.
pub(in crate::metal::gemma_affine) struct AdmissionPipeline {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    pipeline: Pipeline,
    width: usize,
    l1_pipeline: Pipeline,
    l1_width: usize,
    #[cfg(test)]
    diagnostic: Pipeline,
    #[cfg(test)]
    diagnostic_width: usize,
}
impl AdmissionPipeline {
    pub(in crate::metal::gemma_affine) fn new(gpu: &Gpu) -> Result<Self> {
        autoreleasepool(|_| {
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version2_3);
            options.setMathMode(MTLMathMode::Safe);
            options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
            if options.languageVersion() != MTLLanguageVersion::Version2_3
                || options.mathMode() != MTLMathMode::Safe
                || options.mathFloatingPointFunctions() != MTLMathFloatingPointFunctions::Precise
            {
                return Err(Error::Unsupported(
                    "BF16 admission precise policy unavailable",
                ));
            }
            let library = gpu
                .device
                .newLibraryWithSource_options_error(
                    &NSString::from_str(include_str!("bf16_admission.metal")),
                    Some(&options),
                )
                .map_err(|e| Error::Native(e.to_string()))?;
            let function = library
                .newFunctionWithName(&NSString::from_str("gemma_bf16_admit_max"))
                .ok_or_else(|| Error::Native("BF16 admission kernel absent".into()))?;
            let pipeline = gpu
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| Error::Native(e.to_string()))?;
            let width = pipeline.threadExecutionWidth();
            if width == 0
                || width > pipeline.maxTotalThreadsPerThreadgroup()
                || width > u16::MAX as usize
            {
                return Err(Error::Unsupported(
                    "BF16 admission launch width unavailable",
                ));
            }
            let l1_function = library
                .newFunctionWithName(&NSString::from_str("gemma_bf16_admit_max_l1"))
                .ok_or_else(|| Error::Native("BF16 L1 admission kernel absent".into()))?;
            let l1_pipeline = gpu
                .device
                .newComputePipelineStateWithFunction_error(&l1_function)
                .map_err(|e| Error::Native(e.to_string()))?;
            let l1_width = l1_pipeline.threadExecutionWidth();
            if l1_width == 0
                || l1_width > l1_pipeline.maxTotalThreadsPerThreadgroup()
                || l1_width > u16::MAX as usize
            {
                return Err(Error::Unsupported(
                    "BF16 L1 admission launch width unavailable",
                ));
            }
            #[cfg(test)]
            let diagnostic = library
                .newFunctionWithName(&NSString::from_str("gemma_bf16_admit_l1_diagnostic"))
                .ok_or_else(|| Error::Native("BF16 L1 diagnostic kernel absent".into()))?;
            #[cfg(test)]
            let diagnostic = gpu
                .device
                .newComputePipelineStateWithFunction_error(&diagnostic)
                .map_err(|e| Error::Native(e.to_string()))?;
            #[cfg(test)]
            let diagnostic_width = diagnostic.threadExecutionWidth();
            #[cfg(test)]
            if diagnostic_width == 0
                || diagnostic_width > diagnostic.maxTotalThreadsPerThreadgroup()
            {
                return Err(Error::Unsupported("BF16 L1 diagnostic width unavailable"));
            }
            Ok(Self {
                device: gpu.device.clone(),
                pipeline,
                width,
                l1_pipeline,
                l1_width,
                #[cfg(test)]
                diagnostic,
                #[cfg(test)]
                diagnostic_width,
            })
        })
    }

    /// Every actual input word is inspected after its producer. Integer
    /// magnitudes preserve subnormal bits and signed-zero semantics. A max miss
    /// requests reference replay because unchanged host admit() may pass L1.
    /// Caller initializes a distinct status slot to zero and retains the input
    /// checkpoint until completion/replay/publication. No CPU access/wait here.
    pub(in crate::metal::gemma_affine) fn encode(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        tile: Bf16Tile<'_>,
        input: BufferRange<'_>,
        status: BufferRange<'_>,
        tokens: usize,
    ) -> Result<()> {
        let g = Geometry::new(tile.rows, tile.columns, tokens)?;
        if input.len != g.input || status.len != 1 {
            return Err(Error::Options("BF16 borrowed admission lengths differ"));
        }
        validate_encode_ranges(
            command,
            &self.device,
            &[tile.coefficients, input, status],
            2,
        )?;
        let words = [tokens as u32, tile.columns as u32, tile.max_input_bits()];
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::Native("BF16 admission encoder absent".into()))?;
        encoder.setComputePipelineState(&self.pipeline);
        // SAFETY: exact integer ABI, admitted row geometry, checked resource
        // lengths/device/retention and disjoint status. The command retains
        // tracked storage; caller owns completion and status interpretation.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw), input.offset * 4, 0);
            encoder.setBytes_length_atIndex(NonNull::from(&words).cast(), size_of::<[u32; 3]>(), 1);
            encoder.setBuffer_offset_atIndex(Some(status.raw), status.offset * 4, 2);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: tokens,
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
        Ok(())
    }
    /// Supplement the max proof with a conservative source-order L1 enclosure.
    /// This remains one dispatch: only lane0 on max-miss rows runs the serial
    /// positive sum. Finite inputs are accepted only if either unchanged proof
    /// is established; uncertain/overflow rows request whole reference replay.
    pub(in crate::metal::gemma_affine) fn encode_with_l1(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        tile: Bf16Tile<'_>,
        input: BufferRange<'_>,
        status: BufferRange<'_>,
        tokens: usize,
    ) -> Result<()> {
        let g = Geometry::new(tile.rows, tile.columns, tokens)?;
        if input.len != g.input || status.len != 1 {
            return Err(Error::Options("BF16 borrowed L1 admission lengths differ"));
        }
        validate_encode_ranges(
            command,
            &self.device,
            &[tile.coefficients, input, status],
            2,
        )?;
        let words = [
            tokens as u32,
            tile.columns as u32,
            tile.max_input_bits(),
            tile.l1_input_bits(),
        ];
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::Native("BF16 L1 admission encoder absent".into()))?;
        encoder.setComputePipelineState(&self.l1_pipeline);
        // SAFETY: exact four-u32 ABI, finite cold proofs tied to the immutable
        // tile, checked ranges/device/command and disjoint dedicated status.
        // No CPU view/wait or status clearing occurs inside this encoder.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw), input.offset * 4, 0);
            encoder.setBytes_length_atIndex(NonNull::from(&words).cast(), size_of::<[u32; 4]>(), 1);
            encoder.setBuffer_offset_atIndex(Some(status.raw), status.offset * 4, 2);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: tokens,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.l1_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        Ok(())
    }

    #[cfg(test)]
    pub(in crate::metal::gemma_affine) fn encode_l1_diagnostic(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        tile: Bf16Tile<'_>,
        input: BufferRange<'_>,
        trace: BufferRange<'_>,
        tokens: usize,
    ) -> Result<()> {
        let g = Geometry::new(tile.rows, tile.columns, tokens)?;
        if input.len != g.input || trace.len != 4 * tokens {
            return Err(Error::Options("BF16 L1 diagnostic extent differs"));
        }
        validate_encode_ranges(command, &self.device, &[tile.coefficients, input, trace], 2)?;
        let words = [
            tokens as u32,
            tile.columns as u32,
            tile.max_input_bits(),
            tile.l1_input_bits(),
        ];
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::Native("BF16 L1 diagnostic encoder absent".into()))?;
        encoder.setComputePipelineState(&self.diagnostic);
        // SAFETY: same checked ABI/source proof as production; the bounded
        // test-only trace is disjoint, initialized and retained through wait.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw), input.offset * 4, 0);
            encoder.setBytes_length_atIndex(NonNull::from(&words).cast(), size_of::<[u32; 4]>(), 1);
            encoder.setBuffer_offset_atIndex(Some(trace.raw), trace.offset * 4, 2);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: tokens,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.diagnostic_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        Ok(())
    }
}

#[cfg(test)]
mod tests;
