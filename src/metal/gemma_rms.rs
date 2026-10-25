//! Standalone resident FP32 weighted/unweighted Gemma RMS correctness operator.
//!
//! Direct learned scales (no +1), exact fixed FP32 epsilon, scaled compensated
//! reduction and exponent-aware weighted output. Qualified against independent
//! FP64 mathematical fixtures; no source parity, whole graph or speed claim.
use super::{
    Error, Result,
    memory::{Buffer, Gpu, validate_buffer_bytes},
};
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::{mem::size_of, ptr::NonNull};

pub const CONTEXT_ALLOWANCE_BYTES: usize = 64 * 1024 * 1024;
const EPSILON: f32 = 1e-6;
const GUARD: usize = 7;
const SENTINEL: u32 = 0x5a17_0031;

/// Cached Gpu device/queue/default pipelines plus two precise RMS pipelines.
/// Context allowance is compiler/resource policy, not measured residency. Count
/// it once, separately from each owner's admitted buffers and caller storage.
pub struct Context {
    gpu: Gpu,
    pipelines: [Retained<ProtocolObject<dyn MTLComputePipelineState>>; 2],
    widths: [usize; 2],
}
impl Context {
    pub fn new(available_bytes: usize) -> Result<Self> {
        if available_bytes < CONTEXT_ALLOWANCE_BYTES {
            return Err(Error::Budget {
                required: CONTEXT_ALLOWANCE_BYTES,
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
                return Err(Error::Unsupported("Gemma RMS precise math unavailable"));
            }
            let library = gpu
                .device
                .newLibraryWithSource_options_error(
                    &NSString::from_str(include_str!("gemma_rms.metal")),
                    Some(&options),
                )
                .map_err(|e| Error::Native(e.to_string()))?;
            let pipeline = |name| {
                let function = library
                    .newFunctionWithName(&NSString::from_str(name))
                    .ok_or_else(|| Error::Native("Gemma RMS kernel absent".into()))?;
                gpu.device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|e| Error::Native(e.to_string()))
            };
            let pipelines = [
                pipeline("gemma_rms_unlearned")?,
                pipeline("gemma_rms_learned")?,
            ];
            let widths = pipelines.each_ref().map(|p| p.threadExecutionWidth());
            for (pipeline, &width) in pipelines.iter().zip(&widths) {
                if width == 0
                    || width > pipeline.maxTotalThreadsPerThreadgroup()
                    || width > u16::MAX as usize
                {
                    return Err(Error::Unsupported("Gemma RMS launch width unavailable"));
                }
            }
            Ok(Self {
                gpu,
                pipelines,
                widths,
            })
        })
    }
    pub fn device_name(&self) -> String {
        self.gpu.device.name().to_string()
    }

    /// Admit fixed row-major geometry and immutable optional direct scales.
    /// Rows1..256, width1..1536 and exact 1e-6_f32 only. Caller inputs/outputs
    /// need separate admission. No numerical allocations precede full preflight.
    pub fn rms(
        &self,
        rows: usize,
        width: usize,
        epsilon: f32,
        weights: Option<&[f32]>,
        available_owned_bytes: usize,
    ) -> Result<Rms<'_>> {
        let sizes = Sizes::new(rows, width, epsilon, weights)?;
        if sizes.owned > available_owned_bytes {
            return Err(Error::Budget {
                required: sizes.owned,
                limit: available_owned_bytes,
            });
        }
        for count in [sizes.values + GUARD, sizes.weights + GUARD, 1 + GUARD] {
            validate_buffer_bytes(count * 4, self.gpu.device.maxBufferLength())?;
        }
        let mut weight_buffer = Buffer::new(&self.gpu, sizes.weights + GUARD)?;
        weight_buffer.with_mut(|out| {
            out.fill(SENTINEL);
            if let Some(weights) = weights {
                for (out, weight) in out.iter_mut().zip(weights) {
                    *out = weight.to_bits();
                }
            } else {
                out[0] = 1_f32.to_bits();
            }
        });
        Ok(Rms {
            context: self,
            rows,
            width,
            sizes,
            variant: usize::from(weights.is_some()),
            root_epsilon: (f64::from(epsilon).sqrt() as f32).to_bits(),
            input: Buffer::new(&self.gpu, sizes.values + GUARD)?,
            output: Buffer::new(&self.gpu, sizes.values + GUARD)?,
            weights: weight_buffer,
            status: Buffer::new(&self.gpu, 1 + GUARD)?,
            poisoned: false,
            pending: None,
        })
    }
}
#[derive(Clone, Copy)]
struct Sizes {
    values: usize,
    weights: usize,
    owned: usize,
}
impl Sizes {
    fn new(rows: usize, width: usize, epsilon: f32, weights: Option<&[f32]>) -> Result<Self> {
        if !(1..=256).contains(&rows)
            || !(1..=1536).contains(&width)
            || epsilon.to_bits() != EPSILON.to_bits()
            || weights.is_some_and(|w| w.len() != width || w.iter().any(|x| !x.is_finite()))
        {
            return Err(Error::Options(
                "Gemma RMS fixed geometry/epsilon/weights outside bounds",
            ));
        }
        let values = rows
            .checked_mul(width)
            .ok_or(Error::Options("Gemma RMS size overflow"))?;
        let weights = if weights.is_some() { width } else { 1 };
        // Dimensions are bounded above before these additions/multiplications.
        let words = 2 * (values + GUARD) + weights + GUARD + 1 + GUARD;
        let owned = words
            .checked_mul(4)
            .and_then(|n| n.checked_add(size_of::<Rms<'_>>()))
            .ok_or(Error::Options("Gemma RMS owned size overflow"))?;
        Ok(Self {
            values,
            weights,
            owned,
        })
    }
}
/// Persistent disjoint buffers, borrowing their Context. Preflight errors leave
/// the owner reusable and caller output unchanged. Every failure after staging
/// begins poisons the owner permanently; retain pending command/resources and
/// recreate. No CPU access is permitted before checked command completion.
pub struct Rms<'a> {
    context: &'a Context,
    rows: usize,
    width: usize,
    sizes: Sizes,
    variant: usize,
    root_epsilon: u32,
    input: Buffer<u32>,
    output: Buffer<u32>,
    weights: Buffer<u32>,
    status: Buffer<u32>,
    poisoned: bool,
    pending: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}
impl Rms<'_> {
    pub fn owned_bytes(&self) -> usize {
        self.sizes.owned
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// Completed standalone operation; publish caller output only after all
    /// completion, guard and finite-result checks succeed. No view escapes.
    pub fn apply(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if self.poisoned || self.pending.is_some() {
            return Err(Error::Options("Gemma RMS owner poisoned; recreate"));
        }
        if input.len() != self.sizes.values
            || output.len() != self.sizes.values
            || input.iter().any(|x| !x.is_finite())
        {
            return Err(Error::Options("Gemma RMS input shape/nonfinite value"));
        }
        self.poisoned = true;
        autoreleasepool(|_| {
            self.input.with_mut(|out| {
                out.fill(SENTINEL);
                for (y, x) in out.iter_mut().zip(input) {
                    *y = x.to_bits();
                }
            });
            self.output.with_mut(|out| out.fill(SENTINEL));
            self.status.with_mut(|out| {
                out.fill(SENTINEL);
                out[0] = 0;
            });
            let raw = self.context.gpu.command()?;
            let encoder = raw
                .computeCommandEncoder()
                .ok_or_else(|| Error::Native("Gemma RMS encoder absent".into()))?;
            encoder.setComputePipelineState(&self.context.pipelines[self.variant]);
            let shape = [self.rows as u32, self.width as u32, self.root_epsilon];
            // SAFETY: private shader ABI, initialized same-device aligned u32
            // buffers, disjoint and admitted with guards. Validated dimensions
            // bound indices; parameters copy immediately. Tracked hazards and
            // retained command/resources protect GPU lifetimes. CPU views are
            // excluded until checked completion below.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&self.input.raw), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&self.weights.raw), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&self.output.raw), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&shape).cast(),
                    size_of::<[u32; 3]>(),
                    3,
                );
                encoder.setBuffer_offset_atIndex(Some(&self.status.raw), 0, 4);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: self.rows,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.context.widths[self.variant],
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            self.pending = Some(raw);
            let command = self.pending.as_ref().unwrap();
            command.commit();
            command.waitUntilCompleted();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(Error::Completion);
            }
            let tail = |words: &[u32], n| words[n..].iter().all(|&w| w == SENTINEL);
            if !self.input.with_slice(|s| tail(s, self.sizes.values))
                || !self.output.with_slice(|s| tail(s, self.sizes.values))
                || !self.weights.with_slice(|s| tail(s, self.sizes.weights))
                || !self.status.with_slice(|s| tail(s, 1))
            {
                return Err(Error::Options("Gemma RMS guard changed"));
            }
            if self.status.with_slice(|s| s[0]) != 0
                || !self.output.with_slice(|s| {
                    s[..self.sizes.values]
                        .iter()
                        .all(|&v| f32::from_bits(v).is_finite())
                })
            {
                return Err(Error::Nonfinite);
            }
            self.output.with_slice(|s| {
                for (out, &bits) in output.iter_mut().zip(s) {
                    *out = f32::from_bits(bits);
                }
            });
            self.pending = None;
            self.poisoned = false;
            Ok(())
        })
    }
}
#[cfg(test)]
mod tests;
