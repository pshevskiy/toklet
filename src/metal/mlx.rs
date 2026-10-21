//! Opt-in resident MLX affine operator qualification surface.
//!
//! This bounded operator uses Half activations/bias with FP32 reconstruction
//! and compensated reduction. It is not complete-model or MLX-native rounding
//! equivalence. Original packed tensors remain resident between calls.

use super::{
    Error, Result,
    memory::{Buffer, Gpu},
    profile::Command,
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

pub(super) type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
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
            let pipelines = compile(&gpu)?;
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
pub(super) fn compile(gpu: &Gpu) -> Result<Vec<[Pipeline; 2]>> {
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
    Ok(pipelines)
}

pub(super) struct Packed {
    pub layout: Layout,
    scales: Buffer<u32>,
    biases: Buffer<u32>,
    pipelines: [Pipeline; 2],
}
fn pipeline_index(layout: Layout) -> usize {
    let bits = match layout.bits() {
        2 => 0,
        3 => 1,
        4 => 2,
        5 => 3,
        6 => 4,
        8 => 5,
        _ => unreachable!(),
    };
    let dtype = match layout.metadata_dtype() {
        MetadataDtype::F16 => 0,
        MetadataDtype::BF16 => 1,
        MetadataDtype::F32 => 2,
    };
    bits * 3 + dtype
}
pub(super) fn packed_bytes(layout: Layout) -> Result<usize> {
    layout
        .metadata_bytes()
        .div_ceil(4)
        .checked_mul(8)
        .and_then(|n| n.checked_add(layout.weight_bytes()))
        .ok_or(Error::Options("MLX resident weight bytes overflow"))
}
impl Packed {
    pub fn allocate(gpu: &Gpu, layout: Layout, pipelines: &[[Pipeline; 2]]) -> Result<Self> {
        Sizes::new(layout, 1, crate::weights::INITIAL_MEMORY_LIMIT_BYTES)?;
        let pair = &pipelines[pipeline_index(layout)];
        Ok(Self {
            layout,
            scales: Buffer::new(gpu, layout.metadata_bytes().div_ceil(4))?,
            biases: Buffer::new(gpu, layout.metadata_bytes().div_ceil(4))?,
            pipelines: [pair[0].clone(), pair[1].clone()],
        })
    }
    pub fn upload(
        &mut self,
        source: &mut crate::mlx::qwen::Source,
        name: &str,
        weights: &mut Buffer<u16>,
    ) -> Result<()> {
        let layout = self.layout;
        let size = layout.metadata_bytes();
        weights.with_bytes_mut(|w| {
            self.scales.with_bytes_mut(|sc| {
                self.biases.with_bytes_mut(|bi| {
                    source.read_affine(name, w, &mut sc[..size], &mut bi[..size])?;
                    validate_domain(layout, &sc[..size], &bi[..size])
                })
            })
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn product(
    command: &Command<'_>,
    weight: &super::Tensor,
    bias: &Buffer<u16>,
    left: &objc2_metal_performance_shaders::MPSMatrix,
    output: &ProtocolObject<dyn MTLBuffer>,
    output_stride: usize,
    n: usize,
    half_output: bool,
    status: &Buffer<u32>,
) -> Result<()> {
    let super::TensorStorage::Mlx(p) = &weight.storage else {
        return Err(Error::Options("MLX projection storage differs"));
    };
    // SAFETY: private model/session callers pass immutable views of admitted
    // same-device Half buffers, retained by the session through command wait.
    let (offset, row_bytes, rows, columns, input) = unsafe {
        (
            left.offset(),
            left.rowBytes(),
            left.rows(),
            left.columns(),
            left.data(),
        )
    };
    let width = p.layout.columns();
    let out = p.layout.rows();
    if n == 0
        || rows != n
        || columns != width
        || offset % 2 != 0
        || row_bytes % 2 != 0
        || row_bytes / 2 < width
        || output_stride < out
        || bias.len < out
    {
        return Err(Error::Options("MLX input/output view differs"));
    }
    let input_end = (n - 1)
        .checked_mul(row_bytes / 2)
        .and_then(|v| v.checked_add(offset / 2))
        .and_then(|v| v.checked_add(width));
    let output_end = (n - 1)
        .checked_mul(output_stride)
        .and_then(|v| v.checked_add(out));
    if input_end.is_none_or(|v| v > input.length() / 2)
        || output_end.is_none_or(|v| v > output.length() / if half_output { 2 } else { 4 })
    {
        return Err(Error::Options("MLX projection buffer extent differs"));
    }
    let shape = shape([
        n,
        width,
        out,
        p.layout.group_size(),
        usize::from(half_output),
        row_bytes / 2,
        output_stride,
        offset / 2,
    ])?;
    encode(
        command,
        &p.pipelines[0],
        &[
            &weight.buffer.raw,
            &p.scales.raw,
            &p.biases.raw,
            &bias.raw,
            &input,
            output,
        ],
        &shape,
        n.checked_mul(out)
            .ok_or(Error::Options("MLX dispatch overflow"))?,
        false,
        status,
    )
}

pub(super) fn gather(
    command: &Command<'_>,
    weight: &super::Tensor,
    tokens: &Buffer<u32>,
    output: &Buffer<u16>,
    stride: usize,
    n: usize,
    status: &Buffer<u32>,
) -> Result<()> {
    let super::TensorStorage::Mlx(p) = &weight.storage else {
        return Err(Error::Options("MLX gather storage differs"));
    };
    let width = p.layout.columns();
    if n == 0
        || n > tokens.len
        || stride < width
        || (n - 1)
            .checked_mul(stride)
            .and_then(|v| v.checked_add(width))
            .is_none_or(|v| v > output.len)
    {
        return Err(Error::Options("MLX gather extent differs"));
    }
    let shape = shape([
        n,
        width,
        p.layout.rows(),
        p.layout.group_size(),
        1,
        0,
        stride,
        0,
    ])?;
    encode(
        command,
        &p.pipelines[1],
        &[
            &weight.buffer.raw,
            &p.scales.raw,
            &p.biases.raw,
            &tokens.raw,
            &output.raw,
        ],
        &shape,
        n.checked_mul(width)
            .ok_or(Error::Options("MLX gather dispatch overflow"))?,
        true,
        status,
    )
}
fn shape(values: [usize; 8]) -> Result<[u32; 8]> {
    let mut output = [0; 8];
    for (y, x) in output.iter_mut().zip(values) {
        *y = u32::try_from(x).map_err(|_| Error::Options("MLX shape index overflow"))?;
    }
    Ok(output)
}
fn encode(
    command: &Command<'_>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[&ProtocolObject<dyn MTLBuffer>],
    shape: &[u32; 8],
    items: usize,
    gather: bool,
    status: &Buffer<u32>,
) -> Result<()> {
    let width = pipeline.threadExecutionWidth();
    if width == 0 || width > pipeline.maxTotalThreadsPerThreadgroup() || status.len < 1 {
        return Err(Error::Unsupported("MLX launch width/status invalid"));
    }
    let groups = if gather { items.div_ceil(width) } else { items };
    if items == 0
        || groups
            .checked_mul(width)
            .is_none_or(|v| v > u32::MAX as usize)
    {
        return Err(Error::Options("MLX rounded dispatch overflow"));
    }
    let name = if gather { "mlx_gather" } else { "mlx_product" };
    let (index, encoder) = if let Some(trace) = command.trace {
        let (i, e) = trace.encoder(command.raw, name)?;
        (Some(i), e)
    } else {
        (
            None,
            command
                .computeCommandEncoder()
                .ok_or_else(|| Error::Native("MLX encoder absent".into()))?,
        )
    };
    encoder.setComputePipelineState(pipeline);
    // SAFETY: closed private ABIs, validated same-device retained inputs/outputs
    // and exact u32 indices. Upload is complete; no CPU access until completion.
    // Shape bytes copy immediately; tracked hazards order every encoder.
    unsafe {
        for (i, b) in buffers.iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(b), 0, i);
        }
        encoder.setBytes_length_atIndex(
            NonNull::new(shape.as_ptr().cast_mut()).unwrap().cast(),
            size_of_val(shape),
            buffers.len(),
        );
        encoder.setBuffer_offset_atIndex(Some(&status.raw), 0, buffers.len() + 1);
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: groups,
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
    if let Some(index) = index {
        command.trace.unwrap().finish_encoder(index);
    }
    Ok(())
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
                self.layout.columns() as u32,
                if gather {
                    self.layout.columns()
                } else {
                    self.layout.rows()
                } as u32,
                0,
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

    #[test]
    #[ignore = "Explicit Apple GPU MLX strided views, nonzero offset, padding and guards"]
    fn strided_projection_and_gather_honor_offsets_and_preserve_padding() {
        use super::super::memory::{Layout as DenseLayout, matrix_offset};
        autoreleasepool(|_| {
            let gpu = Gpu::new().unwrap();
            let pipelines = compile(&gpu).unwrap();
            let layout = Layout::new(3, 32, 4, 32, MetadataDtype::F16).unwrap();
            let raw = vec![0x21; layout.weight_bytes()];
            let sc: Vec<u8> = [0.5f32, -0.25, 0.125]
                .into_iter()
                .flat_map(|x| f16::from_f32(x).to_le_bytes())
                .collect();
            let bi: Vec<u8> = [-1f32, 0.5, 0.25]
                .into_iter()
                .flat_map(|x| f16::from_f32(x).to_le_bytes())
                .collect();
            let oracle = Matrix::from_le_bytes(layout, &raw, &sc, &bi).unwrap();
            let mut packed = Packed::allocate(&gpu, layout, &pipelines).unwrap();
            packed
                .scales
                .with_bytes_mut(|b| b[..6].copy_from_slice(&sc));
            packed
                .biases
                .with_bytes_mut(|b| b[..6].copy_from_slice(&bi));
            let mut weight = super::super::Tensor {
                buffer: Buffer::new(&gpu, raw.len() / 2).unwrap(),
                layout: DenseLayout::new(3, 32, true).unwrap(),
                storage: super::super::TensorStorage::Mlx(packed),
            };
            weight.buffer.with_bytes_mut(|b| b.copy_from_slice(&raw));
            let mut input = Buffer::<u16>::new(&gpu, 3 * 40 + 7).unwrap();
            input.with_mut(|b| {
                b.fill(f16::NAN.to_bits());
                for t in 0..3 {
                    for c in 0..32 {
                        b[t * 40 + c] =
                            f16::from_f32(((t * 5 + c) % 9) as f32 / 8. - 0.5).to_bits();
                    }
                }
            });
            let view = matrix_offset(
                &input,
                DenseLayout {
                    rows: 2,
                    columns: 32,
                    stride: 40,
                    elements: 3 * 40 + 7,
                },
                true,
                40,
            )
            .unwrap();
            let mut bias = Buffer::<u16>::new(&gpu, 3).unwrap();
            bias.with_mut(|b| {
                b[0] = f16::from_f32(0.125).to_bits();
            });
            let mut status = Buffer::<u32>::new(&gpu, 1).unwrap();
            for half in [false, true] {
                let mut out_h = Buffer::<u16>::new(&gpu, 2 * 7 + 7).unwrap();
                out_h.with_mut(|b| b.fill(HALF_GUARD));
                let mut out_f = Buffer::<u32>::new(&gpu, 2 * 7 + 7).unwrap();
                out_f.with_mut(|b| b.fill(FLOAT_GUARD));
                status.with_mut(|b| b[0] = 0);
                let raw_command = gpu.command().unwrap();
                let command = Command {
                    raw: &raw_command,
                    trace: None,
                };
                product(
                    &command,
                    &weight,
                    &bias,
                    &view,
                    if half { &out_h.raw } else { &out_f.raw },
                    7,
                    2,
                    half,
                    &status,
                )
                .unwrap();
                raw_command.commit();
                raw_command.waitUntilCompleted();
                assert_eq!(raw_command.status(), MTLCommandBufferStatus::Completed);
                assert!(raw_command.error().is_none());
                assert_eq!(status.with_slice(|b| b[0]), 0);
                for i in 0..2 * 7 + 7 {
                    if i / 7 < 2 && i % 7 < 3 {
                        let token = i / 7 + 1;
                        let row = i % 7;
                        let mut sum = if row == 0 { 0.125 } else { 0. };
                        for c in 0..32 {
                            sum += f64::from(oracle.value(row, c).unwrap())
                                * f64::from(((token * 5 + c) % 9) as f32 / 8. - 0.5);
                        }
                        let expected = if half {
                            f16::from_f64(sum).to_f32()
                        } else {
                            sum as f32
                        };
                        let actual = if half {
                            out_h.with_slice(|b| f16::from_bits(b[i]).to_f32())
                        } else {
                            out_f.with_slice(|b| f32::from_bits(b[i]))
                        };
                        assert_eq!(actual.to_bits(), expected.to_bits());
                    } else if half {
                        assert_eq!(out_h.with_slice(|b| b[i]), HALF_GUARD);
                    } else {
                        assert_eq!(out_f.with_slice(|b| b[i]), FLOAT_GUARD);
                    }
                }
            }
            let mut ids = Buffer::<u32>::new(&gpu, 2).unwrap();
            ids.with_mut(|b| b.copy_from_slice(&[2, 0]));
            let mut output = Buffer::<u16>::new(&gpu, 2 * 40 + 7).unwrap();
            output.with_mut(|b| b.fill(HALF_GUARD));
            status.with_mut(|b| b[0] = 0);
            let raw_command = gpu.command().unwrap();
            let command = Command {
                raw: &raw_command,
                trace: None,
            };
            gather(&command, &weight, &ids, &output, 40, 2, &status).unwrap();
            raw_command.commit();
            raw_command.waitUntilCompleted();
            assert_eq!(raw_command.status(), MTLCommandBufferStatus::Completed);
            assert_eq!(status.with_slice(|b| b[0]), 0);
            output.with_slice(|b| {
                for (i, &x) in b.iter().enumerate() {
                    if i / 40 < 2 && i % 40 < 32 {
                        let row = if i / 40 == 0 { 2 } else { 0 };
                        assert_eq!(
                            x,
                            f16::from_f32(oracle.value(row, i % 40).unwrap()).to_bits()
                        );
                    } else {
                        assert_eq!(x, HALF_GUARD);
                    }
                }
            });
        });
    }

    #[test]
    #[ignore = "Explicit Apple GPU MLX post-completion rejection and replay publication boundary"]
    fn native_mlx_failures_preserve_publication_and_require_completion_replay() {
        use super::super::{Model, NormalizationMode, ProjectionMode, ReadbackMode, Session};
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/mlx-qwen-v1/model");
        let model = Model::load_mlx(
            &path,
            crate::weights::LoadOptions {
                context_limit: 32,
                ..Default::default()
            },
        )
        .unwrap();
        for readback in [ReadbackMode::FullLogits, ReadbackMode::GreedyToken] {
            let options = crate::session::Options {
                capacity: 32,
                prefill_chunk_size: 3,
                position_offset: 0,
                memory_limit_bytes: 256 * 1024 * 1024,
            };
            let mut s = Session::with_readback_mode(
                &model,
                options,
                ProjectionMode::MlxAffine,
                NormalizationMode::Compensated,
                readback,
            )
            .unwrap();
            s.set_tokens(&[1, 7, 13, 3]).unwrap();
            assert_eq!(s.evaluate_next_chunk().unwrap(), 3);
            let committed = s.logits.clone();
            let selected = s.selected;
            s.scratch.tokens.with_mut(|b| b[0] = 3);
            s.scratch.status.with_mut(|b| b.fill(0));
            let timing = autoreleasepool(|_| s.run_chunk(1, 3)).unwrap();
            assert_eq!(timing.command_count, 1);
            assert!(matches!(
                s.finish_chunk(1, Err(Error::Nonfinite)),
                Err(Error::Nonfinite)
            ));
            assert_eq!(s.evaluated_len(), 3);
            assert_eq!(s.logits, committed);
            assert_eq!(s.selected, selected);
            assert!(s.greedy_token().is_none());
            s.evaluate().unwrap();
            let expected = s.greedy_token().unwrap().unwrap();
            s.append(&[47]).unwrap();
            assert!(matches!(
                s.finish_chunk(1, Err(Error::Completion)),
                Err(Error::Completion)
            ));
            assert_eq!(s.evaluated_len(), 0);
            assert!(matches!(s.append(&[1]), Err(Error::ReplayRequired)));
            s.set_tokens(&[1, 7, 13, 3]).unwrap();
            s.evaluate().unwrap();
            assert_eq!(s.greedy_token().unwrap().unwrap(), expected);
        }
    }
}
