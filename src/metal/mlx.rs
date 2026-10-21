//! Opt-in resident MLX affine operator qualification surface.
//!
//! This bounded operator uses Half activations/bias with FP32 reconstruction
//! and compensated reduction. It is not complete-model or MLX-native rounding
//! equivalence. Original packed tensors remain resident between calls.

use super::{
    Error, Result,
    memory::{Buffer, Gpu},
};
use crate::mlx::affine::{Layout, Matrix, MetadataDtype};
use half::f16;
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::{mem::size_of_val, ptr::NonNull};

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
const GUARD: usize = 7;
const HALF_GUARD: u16 = 0x3555;
const FLOAT_GUARD: u32 = 0x3eaaaaab;

/// One explicit context caches the 18 useful bits/dtype pairs. Creating it does
/// not change the existing engine's default library/pipelines or execution.
pub struct Context {
    gpu: Gpu,
    pipelines: Vec<[Pipeline; 2]>,
}
impl Context {
    pub fn new() -> Result<Self> {
        autoreleasepool(|_| {
            let gpu = Gpu::new()?;
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version2_3);
            options.setMathMode(MTLMathMode::Safe);
            options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
            if options.languageVersion() != MTLLanguageVersion::Version2_3
                || options.mathMode() != MTLMathMode::Safe
                || options.mathFloatingPointFunctions() != MTLMathFloatingPointFunctions::Precise
            {
                return Err(Error::Unsupported("MLX precise math policy unavailable"));
            }
            let library = gpu
                .device
                .newLibraryWithSource_options_error(
                    &NSString::from_str(include_str!("mlx.metal")),
                    Some(&options),
                )
                .map_err(|e| Error::Native(e.to_string()))?;
            let mut pipelines = Vec::new();
            for bits in [2, 3, 4, 5, 6, 8] {
                for dtype in 0..3 {
                    let compile = |operation| -> Result<Pipeline> {
                        let f = library
                            .newFunctionWithName(&NSString::from_str(&format!(
                                "mlx_{operation}_{bits}_{dtype}"
                            )))
                            .ok_or_else(|| Error::Native("MLX kernel absent".into()))?;
                        gpu.device
                            .newComputePipelineStateWithFunction_error(&f)
                            .map_err(|e| Error::Native(e.to_string()))
                    };
                    let product = compile("product")?;
                    let width = product.threadExecutionWidth();
                    if !width.is_power_of_two() || width > product.maxTotalThreadsPerThreadgroup() {
                        return Err(Error::Unsupported(
                            "MLX reduction requires supported power-of-two SIMD width",
                        ));
                    }
                    pipelines.push([product, compile("gather")?]);
                }
            }
            Ok(Self { gpu, pipelines })
        })
    }
    pub fn device_name(&self) -> String {
        self.gpu.device.name().to_string()
    }
    /// Allocate one resident packed tensor and bounded reusable probe scratch.
    /// Admission includes actual buffer lengths plus a 64 MiB library/metadata
    /// estimate, not a measured physical peak or a complete session budget.
    /// GPU domain excludes FP32-subnormal metadata and requires every possible
    /// fused weight to be finite and either zero or at least 2^-79 in magnitude.
    /// The broader scalar/storage reader remains available for other values.
    pub fn affine<'a>(
        &'a self,
        layout: Layout,
        weights: &[u8],
        scales: &[u8],
        biases: &[u8],
        capacity: usize,
        memory_limit: usize,
    ) -> Result<Affine<'a>> {
        Matrix::from_le_bytes(layout, weights, scales, biases).map_err(|e| Error::Options(e.0))?;
        validate_domain(layout, scales, biases)?;
        let sizes = Sizes::new(layout, capacity, memory_limit)?;
        let upload = |bytes: &[u8]| -> Result<Buffer<u32>> {
            let mut buffer = Buffer::new(&self.gpu, bytes.len().div_ceil(4))?;
            buffer.with_mut(|target| {
                for (word, raw) in target.iter_mut().zip(bytes.chunks(4)) {
                    let mut padded = [0; 4];
                    padded[..raw.len()].copy_from_slice(raw);
                    *word = u32::from_le_bytes(padded);
                }
            });
            Ok(buffer)
        };
        Ok(Affine {
            context: self,
            layout,
            capacity,
            allocated_bytes: sizes.bytes,
            weights: upload(weights)?,
            scales: upload(scales)?,
            biases: upload(biases)?,
            input: Buffer::new(&self.gpu, sizes.input)?,
            linear_bias: Buffer::new(&self.gpu, layout.rows())?,
            half_output: Buffer::new(&self.gpu, sizes.output)?,
            float_output: Buffer::new(&self.gpu, sizes.output)?,
            tokens: Buffer::new(&self.gpu, capacity)?,
            status: Buffer::new(&self.gpu, 1)?,
        })
    }
}
struct Sizes {
    input: usize,
    output: usize,
    bytes: usize,
}
impl Sizes {
    fn new(layout: Layout, capacity: usize, limit: usize) -> Result<Self> {
        if capacity == 0 || limit == 0 || limit > crate::weights::INITIAL_MEMORY_LIMIT_BYTES {
            return Err(Error::Options("MLX operator capacity/budget invalid"));
        }
        let product = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or(Error::Options("MLX operator size overflow"))
        };
        let input = product(capacity, layout.columns())?;
        let output = product(capacity, layout.rows().max(layout.columns()))?
            .checked_add(GUARD)
            .ok_or(Error::Options("MLX guard overflow"))?;
        let metadata = product(layout.rows(), layout.columns() / layout.group_size())?;
        for n in [
            layout.rows(),
            layout.columns(),
            product(layout.columns(), usize::from(layout.bits()))?,
            layout.weight_bytes(),
            metadata,
            input,
            output,
            capacity,
        ] {
            if n > u32::MAX as usize {
                return Err(Error::Options("MLX shader indexing overflow"));
            }
        }
        let mut bytes = layout.weight_bytes();
        for count in [
            product(layout.metadata_bytes().div_ceil(4), 8)?,
            product(input, 2)?,
            product(output, 6)?,
            product(layout.rows(), 2)?,
            product(capacity, 4)?,
            4,
        ] {
            bytes = bytes
                .checked_add(count)
                .ok_or(Error::Options("MLX buffer size overflow"))?;
        }
        let required = bytes
            .checked_add(64 * 1024 * 1024)
            .ok_or(Error::Options("MLX admission overflow"))?;
        if required > limit {
            return Err(Error::Budget { required, limit });
        }
        Ok(Self {
            input,
            output,
            bytes,
        })
    }
}

/// Qualification operator with reusable allocations. Ordinary calls upload only
/// activations/bias or token IDs and read results after one completion boundary.
/// It owns no KV/model graph; it cannot qualify complete-model performance.
pub struct Affine<'a> {
    context: &'a Context,
    layout: Layout,
    capacity: usize,
    allocated_bytes: usize,
    weights: Buffer<u32>,
    scales: Buffer<u32>,
    biases: Buffer<u32>,
    input: Buffer<u16>,
    linear_bias: Buffer<u16>,
    half_output: Buffer<u16>,
    float_output: Buffer<u32>,
    tokens: Buffer<u32>,
    status: Buffer<u32>,
}
impl Affine<'_> {
    pub fn allocated_buffer_bytes(&self) -> usize {
        self.allocated_bytes
    }
    pub fn execution_width(&self) -> usize {
        self.pipeline(false).threadExecutionWidth()
    }
    fn pipeline(&self, gather: bool) -> &ProtocolObject<dyn MTLComputePipelineState> {
        let bits = match self.layout.bits() {
            2 => 0,
            3 => 1,
            4 => 2,
            5 => 3,
            6 => 4,
            8 => 5,
            _ => unreachable!(),
        };
        let dtype = match self.layout.metadata_dtype() {
            MetadataDtype::F16 => 0,
            MetadataDtype::BF16 => 1,
            MetadataDtype::F32 => 2,
        };
        &self.context.pipelines[bits * 3 + dtype][usize::from(gather)]
    }
    /// Row-major [batch, columns] input, [batch, rows] output. Input and linear
    /// bias are rounded once to Half; choose Half or FP32 output per operation.
    /// Caller output is unchanged on every failure; internal scratch is reusable.
    pub fn matmul(
        &mut self,
        batch: usize,
        input: &[f32],
        linear_bias: &[f32],
        output: &mut [f32],
        half_output: bool,
    ) -> Result<()> {
        if batch == 0
            || batch > self.capacity
            || input.len() != batch * self.layout.columns()
            || output.len() != batch * self.layout.rows()
            || linear_bias.len() != self.layout.rows()
        {
            return Err(Error::Options("MLX matmul shape differs"));
        }
        validate_half(input)?;
        validate_half(linear_bias)?;
        self.input.with_mut(|target| {
            for (y, &x) in target.iter_mut().zip(input) {
                *y = f16::from_f32(x).to_bits();
            }
        });
        self.linear_bias.with_mut(|target| {
            for (y, &x) in target.iter_mut().zip(linear_bias) {
                *y = f16::from_f32(x).to_bits();
            }
        });
        self.run(batch, false, output, half_output)
    }
    /// Gather packed embedding rows. IDs are validated before submission and
    /// checked again by the kernel before any weight access.
    pub fn gather(&mut self, tokens: &[u32], output: &mut [f32], half_output: bool) -> Result<()> {
        if tokens.is_empty()
            || tokens.len() > self.capacity
            || output.len() != tokens.len() * self.layout.columns()
            || tokens.iter().any(|&id| id as usize >= self.layout.rows())
        {
            return Err(Error::Options("MLX gather shape/token differs"));
        }
        self.tokens
            .with_mut(|target| target[..tokens.len()].copy_from_slice(tokens));
        self.run(tokens.len(), true, output, half_output)
    }
    fn run(
        &mut self,
        batch: usize,
        gather: bool,
        output: &mut [f32],
        half_output: bool,
    ) -> Result<()> {
        autoreleasepool(|_| {
            self.status.with_mut(|s| s[0] = 0);
            self.half_output.with_mut(|s| s.fill(HALF_GUARD));
            self.float_output.with_mut(|s| s.fill(FLOAT_GUARD));
            let destination = if half_output {
                &self.half_output.raw
            } else {
                &self.float_output.raw
            };
            let shape = [
                batch as u32,
                self.layout.columns() as u32,
                self.layout.rows() as u32,
                self.layout.group_size() as u32,
                u32::from(half_output),
            ];
            let bindings: [&ProtocolObject<dyn MTLBuffer>; 6] = [
                &self.weights.raw,
                &self.scales.raw,
                &self.biases.raw,
                if gather {
                    &self.tokens.raw
                } else {
                    &self.linear_bias.raw
                },
                if gather { destination } else { &self.input.raw },
                destination,
            ];
            let buffers = &bindings[..if gather { 5 } else { 6 }];
            let pipeline = self.pipeline(gather);
            let width = pipeline.threadExecutionWidth();
            if width == 0 || width > pipeline.maxTotalThreadsPerThreadgroup() {
                return Err(Error::Unsupported("MLX launch width unsupported"));
            }
            let count = if gather {
                output.len().div_ceil(width)
            } else {
                output.len()
            };
            if count
                .checked_mul(width)
                .is_none_or(|n| n > u32::MAX as usize)
            {
                return Err(Error::Options("MLX dispatch overflow/width invalid"));
            }
            let command = self.context.gpu.command()?;
            let encoder = command
                .computeCommandEncoder()
                .ok_or_else(|| Error::Native("MLX encoder absent".into()))?;
            encoder.setComputePipelineState(pipeline);
            // SAFETY: same-context retained shared allocations, exact validated
            // extents and closed ABI; immutable packed buffers, disjoint input,
            // output and status. Copied u32 shape selects output dtype only at
            // operation boundary. Owners survive checked completion, excluding
            // all host buffer access while the command is in flight.
            unsafe {
                for (i, buffer) in buffers.iter().enumerate() {
                    encoder.setBuffer_offset_atIndex(Some(buffer), 0, i);
                }
                encoder.setBytes_length_atIndex(
                    NonNull::new(shape.as_ptr().cast_mut()).unwrap().cast(),
                    size_of_val(&shape),
                    buffers.len(),
                );
                encoder.setBuffer_offset_atIndex(Some(&self.status.raw), 0, buffers.len() + 1);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: count,
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
            command.commit();
            command.waitUntilCompleted();
            if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
                return Err(Error::Completion);
            }
            if self.status.with_slice(|s| s[0]) != 0 {
                return Err(Error::Nonfinite);
            }
            let count = output.len();
            if half_output {
                self.half_output.with_slice(|s| {
                    if s[count..].iter().any(|&v| v != HALF_GUARD) {
                        return Err(Error::Options("MLX Half output guard changed"));
                    }
                    if s[..count].iter().any(|&v| !f16::from_bits(v).is_finite()) {
                        return Err(Error::Nonfinite);
                    }
                    for (y, &x) in output.iter_mut().zip(s) {
                        *y = f16::from_bits(x).to_f32();
                    }
                    Ok(())
                })
            } else {
                self.float_output.with_slice(|s| {
                    if s[count..].iter().any(|&v| v != FLOAT_GUARD) {
                        return Err(Error::Options("MLX FP32 output guard changed"));
                    }
                    if s[..count].iter().any(|&v| !f32::from_bits(v).is_finite()) {
                        return Err(Error::Nonfinite);
                    }
                    for (y, &x) in output.iter_mut().zip(s) {
                        *y = f32::from_bits(x);
                    }
                    Ok(())
                })
            }
        })
    }
}
fn validate_half(values: &[f32]) -> Result<()> {
    if values
        .iter()
        .any(|&v| !v.is_finite() || !f16::from_f32(v).is_finite())
    {
        return Err(Error::Nonfinite);
    }
    Ok(())
}

// Metal can flush FP32 subnormals even under Safe/Precise. Validate once at
// upload, without expanding/storing weights. Conservatively consider every
// code, including unused codes. Normal metadata and finite fused weights that
// are zero or >= 2^-79 ensure a 2^-102 weight lattice. Half inputs are multiples
// of 2^-24, so exact products and reduction residuals/differences lie on the
// FP32-normal 2^-126 lattice (or are zero). Overflow remains a runtime error.
fn validate_domain(layout: Layout, scales: &[u8], biases: &[u8]) -> Result<()> {
    fn check<const N: usize>(
        scales: &[u8],
        biases: &[u8],
        bits: u8,
        decode: impl Fn([u8; N]) -> f32,
    ) -> Result<()> {
        let minimum = f32::from_bits((127 - 79) << 23);
        for (scale, bias) in scales.chunks_exact(N).zip(biases.chunks_exact(N)) {
            let scale = decode(scale.try_into().unwrap());
            let bias = decode(bias.try_into().unwrap());
            if scale.is_subnormal() || bias.is_subnormal() {
                return Err(Error::Unsupported(
                    "MLX GPU excludes FP32-subnormal metadata",
                ));
            }
            for code in 0..(1u16 << bits) {
                let weight = scale.mul_add(f32::from(code), bias);
                if !weight.is_finite() || (weight != 0.0 && weight.abs() < minimum) {
                    return Err(Error::Unsupported(
                        "MLX GPU requires every possible nonzero fused weight >= 2^-79 and finite",
                    ));
                }
            }
        }
        Ok(())
    }
    match layout.metadata_dtype() {
        MetadataDtype::F16 => check(scales, biases, layout.bits(), |b| {
            f16::from_le_bytes(b).to_f32()
        }),
        MetadataDtype::BF16 => check(scales, biases, layout.bits(), |b| {
            half::bf16::from_le_bytes(b).to_f32()
        }),
        MetadataDtype::F32 => check(scales, biases, layout.bits(), f32::from_le_bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admission_checks_sizes_before_any_device_or_payload_allocation() {
        let layout = Layout::new(3, 32, 4, 32, MetadataDtype::F16).unwrap();
        let s = Sizes::new(layout, 5, crate::weights::INITIAL_MEMORY_LIMIT_BYTES).unwrap();
        // Metadata each round from 6 to 8 bytes; both output precisions retained.
        let expected = 48 + 16 + (5 * 32) * 2 + (5 * 32 + GUARD) * 6 + 3 * 2 + 5 * 4 + 4;
        assert_eq!(s.bytes, expected);
        assert!(Sizes::new(layout, 0, usize::MAX).is_err());
        assert!(
            Sizes::new(
                layout,
                usize::MAX,
                crate::weights::INITIAL_MEMORY_LIMIT_BYTES
            )
            .is_err()
        );
        assert!(matches!(
            Sizes::new(layout, 5, expected + 64 * 1024 * 1024 - 1),
            Err(Error::Budget { .. })
        ));
        assert!(Sizes::new(layout, 5, expected + 64 * 1024 * 1024).is_ok());
    }

    #[test]
    fn upload_domain_rejects_tiny_metadata_cancellation_and_product_risks() {
        let layout = Layout::new(1, 32, 4, 32, MetadataDtype::F32).unwrap();
        let check = |s: f32, b: f32| validate_domain(layout, &s.to_le_bytes(), &b.to_le_bytes());
        assert!(check(0., f32::from_bits(1)).is_err());
        // Normal operands can reconstruct a subnormal by cancellation.
        assert!(check(f32::MIN_POSITIVE, -f32::from_bits(0x00800001)).is_err());
        // Code zero is safe; code one cancels below the floor. Checking only
        // metadata or the code-zero bias would incorrectly admit this group.
        let scale = 2f32.powi(-56);
        let bias = -(scale - 2f32.powi(-80));
        assert!(bias.abs() >= 2f32.powi(-79));
        assert_eq!(scale.mul_add(1., bias), 2f32.powi(-80));
        assert!(check(scale, bias).is_err());
        assert!(check(0., 2f32.powi(-80)).is_err());
        assert!(check(0., 2f32.powi(-79)).is_ok());
        assert!(check(0., -2f32.powi(-79)).is_ok());
        assert!(check(-0., 0.).is_ok());
        assert!(check(-0.5, 1.).is_ok());
        assert!(check(f32::MAX, 0.).is_err());
        let layout = Layout::new(1, 32, 4, 32, MetadataDtype::BF16).unwrap();
        assert!(validate_domain(layout, &0u16.to_le_bytes(), &1u16.to_le_bytes()).is_err());
        let layout = Layout::new(1, 32, 4, 32, MetadataDtype::F16).unwrap();
        assert!(validate_domain(layout, &1u16.to_le_bytes(), &0u16.to_le_bytes()).is_ok());
    }
}
