//! Resident packed group64/BF16 projection with FP32 activations and outputs.
//!
//! Standalone bounded correctness surface, not a whole Gemma graph or a tuned
//! GEMM. Direct affine reconstruction and dot accumulation have independent
//! references. Conservative admission excludes incompatible scalar/exact dots.
use super::{
    Error, Result,
    memory::{Buffer, Gpu, validate_buffer_bytes},
};
use crate::mlx::affine::{Layout, Matrix, MetadataDtype};
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::{mem::size_of, ptr::NonNull};

mod bf16;
mod loading;
mod partitioned;
mod project;
mod projection;
pub mod resident;
mod weights;
pub use bf16::{Bf16Context, Bf16Projection};
pub use partitioned::{Gather, PartitionedAffine};
pub use projection::Projection;
pub use weights::Weights;

pub const CONTEXT_ALLOWANCE_BYTES: usize = 64 * 1024 * 1024;
const MIXED_CONTEXT_ALLOWANCE_BYTES: usize = 2 * CONTEXT_ALLOWANCE_BYTES;
const GUARD: usize = 7;
const SENTINEL: u32 = 0x5a17_0031;
const MAX_WEIGHT: f32 = 4294967296.;
const ORDERED_ERROR_LIMIT: f64 = 1e-5 / 16.;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// One cached device/queue and precise affine4 projection pipeline.
/// Context allowance is policy, not measured residency;
/// admit caller storage and each owner's actual buffers separately.
pub struct Context {
    gpu: Gpu,
    pipeline: Pipeline,
    width: usize,
    projection_bits: u8,
    affine8: Option<(Pipeline, usize)>,
}
impl Context {
    pub fn new(available_bytes: usize) -> Result<Self> {
        Self::with_projection_pipeline(available_bytes, 4)
    }

    fn with_projection_pipeline(available_bytes: usize, projection_bits: u8) -> Result<Self> {
        let name = match projection_bits {
            4 => "gemma_affine_product",
            8 => "gemma_affine_product_8",
            16 => "gemma_bf16_product",
            _ => return Err(Error::Options("Gemma affine projection width")),
        };
        Self::compile(available_bytes, name, projection_bits, false)
    }

    fn compile(
        available_bytes: usize,
        name: &str,
        projection_bits: u8,
        mixed: bool,
    ) -> Result<Self> {
        let allowance = if mixed {
            MIXED_CONTEXT_ALLOWANCE_BYTES
        } else {
            CONTEXT_ALLOWANCE_BYTES
        };
        if available_bytes < allowance {
            return Err(Error::Budget {
                required: allowance,
                limit: available_bytes,
            });
        }
        autoreleasepool(|_| {
            let gpu = Gpu::new()?;
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version2_3);
            options.setMathMode(MTLMathMode::Safe);
            options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
            if options.mathMode() != MTLMathMode::Safe
                || options.mathFloatingPointFunctions() != MTLMathFloatingPointFunctions::Precise
                || options.languageVersion() != MTLLanguageVersion::Version2_3
            {
                return Err(Error::Unsupported("Gemma affine precise math unavailable"));
            }
            let library = gpu
                .device
                .newLibraryWithSource_options_error(
                    &NSString::from_str(include_str!("gemma_affine.metal")),
                    Some(&options),
                )
                .map_err(|e| Error::Native(e.to_string()))?;
            let compile_pipeline = |entry: &str| -> Result<(Pipeline, usize)> {
                let function = library
                    .newFunctionWithName(&NSString::from_str(entry))
                    .ok_or_else(|| Error::Native("Gemma affine kernel absent".into()))?;
                let pipeline = gpu
                    .device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|e| Error::Native(e.to_string()))?;
                let width = pipeline.threadExecutionWidth();
                if width == 0
                    || width > pipeline.maxTotalThreadsPerThreadgroup()
                    || width > u16::MAX as usize
                {
                    return Err(Error::Unsupported("Gemma affine launch width unavailable"));
                }
                Ok((pipeline, width))
            };
            let (pipeline, width) = compile_pipeline(name)?;
            let affine8 = mixed
                .then(|| compile_pipeline("gemma_affine_product_8"))
                .transpose()?;
            Ok(Self {
                gpu,
                pipeline,
                width,
                projection_bits,
                affine8,
            })
        })
    }

    fn projection_pipeline(&self, bits: u8) -> Result<(&Pipeline, usize)> {
        if bits == self.projection_bits {
            return Ok((&self.pipeline, self.width));
        }
        if bits == 8
            && let Some((pipeline, width)) = &self.affine8
        {
            return Ok((pipeline, *width));
        }
        Err(Error::Options("Gemma affine projection width"))
    }

    fn allowance_bytes(&self) -> usize {
        if self.affine8.is_some() {
            MIXED_CONTEXT_ALLOWANCE_BYTES
        } else {
            CONTEXT_ALLOWANCE_BYTES
        }
    }

    pub fn device_name(&self) -> String {
        self.gpu.device.name().to_string()
    }

    /// Upload original contiguous packed tensors once. Batches1..capacity reuse
    /// the same weights and scratch; no dense matrix is expanded. The format is
    /// affine4/group64/BF16, outputs1..16384, columns64..12288 and capacity1..32.
    pub fn affine(
        &self,
        layout: Layout,
        weights: &[u8],
        scales: &[u8],
        biases: &[u8],
        capacity: usize,
        available_owned_bytes: usize,
    ) -> Result<Affine<'_>> {
        if layout.bits() != self.projection_bits {
            return Err(Error::Options("Gemma affine projection width"));
        }
        let sizes = Sizes::projection(layout, capacity)?;
        if sizes.owned > available_owned_bytes {
            return Err(Error::Budget {
                required: sizes.owned,
                limit: available_owned_bytes,
            });
        }
        Matrix::from_le_bytes(layout, weights, scales, biases).map_err(|e| Error::Options(e.0))?;
        let max_weight = codebook(scales, biases, layout.bits())?;
        for words in sizes.buffers() {
            validate_buffer_bytes((words + GUARD) * 4, self.gpu.device.maxBufferLength())?;
        }
        Ok(Affine {
            context: self,
            layout,
            capacity,
            sizes,
            max_weight,
            codes: loading::upload(&self.gpu, weights)?,
            scales: loading::upload(&self.gpu, scales)?,
            biases: loading::upload(&self.gpu, biases)?,
            input: Buffer::new(&self.gpu, sizes.input + GUARD)?,
            output: Buffer::new(&self.gpu, sizes.output + GUARD)?,
            status: Buffer::new(&self.gpu, 1 + GUARD)?,
            poisoned: false,
            pending: None,
        })
    }
}

/// Standalone affine8 projection context. It owns one pipeline under the same
/// context allowance as affine4. Its distinct type keeps the existing
/// affine4 resident and gather entry points outside this capability.
pub struct Affine8Context(Context);

impl Affine8Context {
    pub fn new(available_bytes: usize) -> Result<Self> {
        Context::with_projection_pipeline(available_bytes, 8).map(Self)
    }

    /// Upload original packed affine8/group64/BF16 rows and admit guarded
    /// FP32 scratch for the selected batch capacity.
    pub fn affine(
        &self,
        layout: Layout,
        weights: &[u8],
        scales: &[u8],
        biases: &[u8],
        capacity: usize,
        available_owned_bytes: usize,
    ) -> Result<Affine<'_>> {
        self.0.affine(
            layout,
            weights,
            scales,
            biases,
            capacity,
            available_owned_bytes,
        )
    }
}
/// One device and queue with both precise projection pipelines. This context
/// is reserved for the explicit mixed QAT diagnostic session.
pub struct MixedContext(Context);

impl MixedContext {
    pub fn new(available_bytes: usize) -> Result<Self> {
        Context::compile(available_bytes, "gemma_affine_product", 4, true).map(Self)
    }

    pub fn device_name(&self) -> String {
        self.0.device_name()
    }
}

#[derive(Clone, Copy)]
struct Sizes {
    codes: usize,
    metadata: usize,
    input: usize,
    output: usize,
    owned: usize,
}
impl Sizes {
    fn buffers(self) -> [usize; 6] {
        [
            self.codes,
            self.metadata,
            self.metadata,
            self.input,
            self.output,
            1,
        ]
    }
    fn new(layout: Layout, capacity: usize) -> Result<Self> {
        if layout.bits() != 4 {
            return Err(Error::Options(
                "Gemma affine fixed geometry/format outside bounds",
            ));
        }
        Self::projection(layout, capacity)
    }
    fn projection(layout: Layout, capacity: usize) -> Result<Self> {
        if !matches!(layout.bits(), 4 | 8)
            || layout.group_size() != 64
            || layout.metadata_dtype() != MetadataDtype::BF16
            || layout.rows() > 16384
            || layout.columns() > 12288
            || !(1..=32).contains(&capacity)
        {
            return Err(Error::Options(
                "Gemma affine fixed geometry/format outside bounds",
            ));
        }
        let mut s = Self {
            codes: layout.weight_bytes().div_ceil(4),
            metadata: layout.metadata_bytes().div_ceil(4),
            input: capacity
                .checked_mul(layout.columns())
                .ok_or(Error::Options("Gemma affine input size overflow"))?,
            output: capacity
                .checked_mul(layout.rows())
                .ok_or(Error::Options("Gemma affine output size overflow"))?,
            owned: 0,
        };
        s.owned = s
            .buffers()
            .into_iter()
            .try_fold(0usize, |sum, n| sum.checked_add(n)?.checked_add(GUARD))
            .and_then(|words| words.checked_mul(4))
            .and_then(|bytes| bytes.checked_add(size_of::<Affine<'_>>()))
            .ok_or(Error::Options("Gemma affine owned size overflow"))?;
        Ok(s)
    }
}
// Called only after exact BF16 lengths/finiteness pass Matrix admission. Bound
// all possible codes, including unused ones, without expanding packed weights.
fn codebook(scales: &[u8], biases: &[u8], bits: u8) -> Result<f64> {
    let read = |bytes: &[u8]| {
        f32::from_bits(u32::from(u16::from_le_bytes(bytes.try_into().unwrap())) << 16)
    };
    let mut maximum = 0_f32;
    for (scale, bias) in scales.chunks_exact(2).zip(biases.chunks_exact(2)) {
        let (scale, bias) = (read(scale), read(bias));
        if scale.is_subnormal() || bias.is_subnormal() {
            return Err(Error::Unsupported("Gemma affine metadata domain"));
        }
        for code in 0..(1u32 << bits) {
            let value = (f64::from(scale) * f64::from(code) + f64::from(bias)) as f32;
            if !value.is_finite() {
                return Err(Error::Unsupported(
                    "Gemma affine reconstruction must be finite",
                ));
            }
            if value.is_subnormal() || value.abs() > MAX_WEIGHT {
                return Err(Error::Unsupported("Gemma affine weight domain"));
            }
            maximum = maximum.max(value.abs());
        }
    }
    Ok(f64::from(maximum))
}
fn up_product(a: f64, b: f64) -> f64 {
    if a == 0. || b == 0. {
        0.
    } else {
        (a * b).next_up()
    }
}
// Products of finite FP32 values are exact in FP64. For K<=12288,
// gamma_K<=2*K*2^-53 bounds ordered summation against exact accumulation.
// Upward products make S>=K*maxW*maxX and U>=2*K*2^-53*S. Exact-zero bounds
// stay zero. This is scalar/exact compatibility admission, not a GPU error proof.
fn bounds(columns: usize, max_weight: f64, max_input: f32) -> (f64, f64) {
    let s = up_product(up_product(columns as f64, max_weight), f64::from(max_input));
    let u = up_product(
        up_product(up_product(2., columns as f64), f64::EPSILON / 2.),
        s,
    );
    (s, u)
}
fn admit(columns: usize, max_weight: f64, input: &[f32]) -> Result<()> {
    let maximum = input.iter().fold(0_f32, |a, x| a.max(x.abs()));
    let (s, u) = bounds(columns, max_weight, maximum);
    let error = if s > f64::from(f32::MAX) / 4. {
        "Gemma affine input absolute-product bound"
    } else if u > ORDERED_ERROR_LIMIT {
        "Gemma affine input ordered-sum accuracy bound"
    } else {
        return Ok(());
    };
    let (s, u) = l1_bounds(columns, max_weight, input);
    if s <= f64::from(f32::MAX) / 4. && u <= ORDERED_ERROR_LIMIT {
        Ok(())
    } else {
        // Both proofs fail before staging; preserve the original error priority.
        Err(Error::Options(error))
    }
}
// Called only for finite, complete [tokens, columns] rows after the old max
// proof rejects. FP32 magnitudes convert exactly. Skip zero and the first
// positive addition; otherwise nearest-add then next_up conservatively encloses
// every positive term. Independent rows share no sum or temporary allocation.
fn row_l1(row: &[f32]) -> f64 {
    let mut total = 0.;
    for &value in row {
        let magnitude = f64::from(value.abs());
        if magnitude != 0. {
            total = if total == 0. {
                magnitude
            } else {
                (total + magnitude).next_up()
            };
        }
    }
    total
}
fn l1_bounds(columns: usize, max_weight: f64, input: &[f32]) -> (f64, f64) {
    #[cfg(test)]
    tests::L1_CALLS.with(|calls| calls.set(calls.get() + 1));
    let magnitude = input
        .chunks_exact(columns)
        .map(row_l1)
        .fold(0_f64, f64::max);
    let s = up_product(max_weight, magnitude);
    let u = up_product(
        up_product(up_product(2., columns as f64), f64::EPSILON / 2.),
        s,
    );
    (s, u)
}
/// Resident immutable tensors and reusable bounded standalone scratch. Input
/// admission preserves output and reusability. Failure after staging begins
/// permanently poisons the owner; command/resources remain retained. Driver or
/// in-flight failure injection is not qualified by successful fixture execution.
pub struct Affine<'a> {
    context: &'a Context,
    layout: Layout,
    capacity: usize,
    sizes: Sizes,
    max_weight: f64,
    codes: Buffer<u32>,
    scales: Buffer<u32>,
    biases: Buffer<u32>,
    input: Buffer<u32>,
    output: Buffer<u32>,
    status: Buffer<u32>,
    poisoned: bool,
    pending: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}
impl Affine<'_> {
    pub fn owned_bytes(&self) -> usize {
        self.sizes.owned
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// Complete one batch1..capacity, retaining weights. Caller output changes
    /// only after completion, all active/inactive guards, status and finiteness
    /// pass. No shared CPU view escapes or overlaps in-flight GPU access.
    pub fn apply(&mut self, input: &[f32], output: &mut [f32], tokens: usize) -> Result<()> {
        if self.poisoned || self.pending.is_some() {
            return Err(Error::Options("Gemma affine owner poisoned; recreate"));
        }
        if tokens == 0
            || tokens > self.capacity
            || input.len() != tokens * self.layout.columns()
            || output.len() != tokens * self.layout.rows()
            || input.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Options("Gemma affine input shape/nonfinite value"));
        }
        admit(self.layout.columns(), self.max_weight, input)?;
        self.poisoned = true;
        // Explicit field borrows keep immutable weights separate from pending
        // command and mutable scratch without changing the legacy owner layout.
        let weights = WeightView {
            layout: self.layout,
            codes: &self.codes,
            scales: &self.scales,
            biases: &self.biases,
        };
        project::run(
            self.context,
            project::Matrix::affine(weights),
            project::Buffers {
                input: &mut self.input,
                output: &mut self.output,
                status: &mut self.status,
                pending: &mut self.pending,
            },
            input,
            output,
            tokens,
        )?;
        self.poisoned = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests;

/// Internal immutable operation view; native resource owners retain lifetimes.
#[derive(Clone, Copy)]
struct WeightView<'a> {
    layout: Layout,
    codes: &'a Buffer<u32>,
    scales: &'a Buffer<u32>,
    biases: &'a Buffer<u32>,
}
impl Affine<'_> {
    fn weight_view(&self) -> WeightView<'_> {
        WeightView {
            layout: self.layout,
            codes: &self.codes,
            scales: &self.scales,
            biases: &self.biases,
        }
    }
}
