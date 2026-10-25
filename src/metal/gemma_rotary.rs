//! Standalone fixed-shape proportional rotary correctness operator.
//!
//! Cached host FP64 phase/trig, rounded FP32 coefficients and noncontracting GPU
//! FP32 products/sums are qualified against mathematical/CPU fixtures. This does not
//! emulate upstream approximations or qualify a Gemma graph/context/performance.
//! Active subnormal arithmetic is not bitwise qualified; inactive bits copy exactly.
use super::{
    Error, Result,
    memory::{Buffer, Gpu},
};
use crate::cpu::ProportionalRotaryShape;
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::{mem::size_of, ptr::NonNull};

pub const CONTEXT_ALLOWANCE_BYTES: usize = 64 * 1024 * 1024;
const GUARD: usize = 7;
const SENTINEL: u32 = 0x5a17_0031;

/// One cached device/queue with Gpu::new's existing default pipelines plus one
/// separate precise pipeline. The 64 MiB allowance
/// is compiler/context policy, not measured physical residency. Caller-owned
/// inputs/outputs and multiple simultaneous operators need their own allowance.
pub struct Context {
    gpu: Gpu,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    width: usize,
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
                return Err(Error::Unsupported("Gemma rotary precise math unavailable"));
            }
            let library = gpu
                .device
                .newLibraryWithSource_options_error(
                    &NSString::from_str(include_str!("gemma_rotary.metal")),
                    Some(&options),
                )
                .map_err(|e| Error::Native(e.to_string()))?;
            let function = library
                .newFunctionWithName(&NSString::from_str("gemma_proportional_rotary"))
                .ok_or_else(|| Error::Native("Gemma rotary kernel absent".into()))?;
            let pipeline = gpu
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| Error::Native(e.to_string()))?;
            let width = pipeline.threadExecutionWidth();
            if width == 0
                || width > pipeline.maxTotalThreadsPerThreadgroup()
                || width > u32::MAX as usize
            {
                return Err(Error::Unsupported("Gemma rotary launch width unavailable"));
            }
            Ok(Self {
                gpu,
                pipeline,
                width,
            })
        })
    }
    pub fn device_name(&self) -> String {
        self.gpu.device.name().to_string()
    }
    /// Cache coefficients for this immutable shape/offset. Admit owned bytes
    /// separately from the shared context, counted once across all operators.
    pub fn rotary(
        &self,
        shape: ProportionalRotaryShape,
        available_owned_bytes: usize,
    ) -> Result<Rotary<'_>> {
        let sizes = Sizes::new(&shape)?;
        if sizes.owned > available_owned_bytes {
            return Err(Error::Budget {
                required: sizes.owned,
                limit: available_owned_bytes,
            });
        }
        let mut coefficients = Buffer::<u32>::new(&self.gpu, sizes.coefficients + GUARD)?;
        coefficients.with_mut(|out| {
            out.fill(SENTINEL);
            for pair in 0..shape.rotated_dims / 2 {
                let inverse = f64::from(shape.base)
                    .powf(-2.0 * pair as f64 / shape.head_dim as f64)
                    / f64::from(shape.factor);
                for token in 0..shape.tokens {
                    let (sine, cosine) = ((shape.offset + token) as f64 * inverse).sin_cos();
                    let index = 2 * (token * (shape.rotated_dims / 2) + pair);
                    out[index] = (cosine as f32).to_bits();
                    out[index + 1] = (sine as f32).to_bits();
                }
            }
        });
        Ok(Rotary {
            context: self,
            shape,
            sizes,
            input: Buffer::new(&self.gpu, sizes.values + GUARD)?,
            output: Buffer::new(&self.gpu, sizes.values + GUARD)?,
            coefficients,
            status: Buffer::new(&self.gpu, 1 + GUARD)?,
            poisoned: false,
            pending: None,
        })
    }
}
#[derive(Clone, Copy)]
struct Sizes {
    values: usize,
    coefficients: usize,
    owned: usize,
}
impl Sizes {
    fn new(s: &ProportionalRotaryShape) -> Result<Self> {
        if !(1..=8).contains(&s.tokens)
            || !(1..=8).contains(&s.heads)
            || !(2..=512).contains(&s.head_dim)
            || !s.head_dim.is_multiple_of(2)
            || s.rotated_dims > s.head_dim
            || !s.rotated_dims.is_multiple_of(2)
            || !s.base.is_finite()
            || !(1.0..=1e6).contains(&s.base)
            || !s.factor.is_finite()
            || !(0.125..=8.0).contains(&s.factor)
            || s.offset
                .checked_add(s.tokens - 1)
                .is_none_or(|p| p > 131071)
        {
            return Err(Error::Options(
                "Gemma rotary fixed geometry outside qualified bounds",
            ));
        }
        let values = s
            .tokens
            .checked_mul(s.heads)
            .and_then(|n| n.checked_mul(s.head_dim))
            .ok_or(Error::Options("Gemma rotary size overflow"))?;
        let coefficients = s
            .tokens
            .checked_mul(s.rotated_dims)
            .ok_or(Error::Options("Gemma rotary coefficient size overflow"))?;
        let words = (values + GUARD) * 2 + coefficients + GUARD + 1 + GUARD;
        let owned = words
            .checked_mul(4)
            .and_then(|n| n.checked_add(size_of::<Rotary<'_>>()))
            .ok_or(Error::Options("Gemma rotary owned size overflow"))?;
        Ok(Self {
            values,
            coefficients,
            owned,
        })
    }
}
/// Resident buffers borrow their context. Offset changes require a new admitted
/// coefficient table/owner; efficient autoregressive graph use is not qualified.
/// Invalid input rejects before submission and remains reusable. Every later
/// failure permanently poisons this owner; recreate it rather than read/reuse it.
pub struct Rotary<'a> {
    context: &'a Context,
    shape: ProportionalRotaryShape,
    sizes: Sizes,
    input: Buffer<u32>,
    output: Buffer<u32>,
    coefficients: Buffer<u32>,
    status: Buffer<u32>,
    poisoned: bool,
    pending: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}
impl Rotary<'_> {
    pub fn owned_bytes(&self) -> usize {
        self.sizes.owned
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// One completed standalone invocation. Caller output is unchanged on every
    /// error. No internal shared-buffer reference or pending result escapes.
    pub fn apply(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if self.poisoned || self.pending.is_some() {
            return Err(Error::Options("Gemma rotary owner poisoned; recreate"));
        }
        if input.len() != self.sizes.values
            || output.len() != self.sizes.values
            || input.iter().any(|x| !x.is_finite())
        {
            return Err(Error::Options("Gemma rotary input shape/nonfinite value"));
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
                .ok_or_else(|| Error::Native("Gemma rotary encoder absent".into()))?;
            encoder.setComputePipelineState(&self.context.pipeline);
            let shape = [
                self.shape.tokens as u32,
                self.shape.heads as u32,
                self.shape.head_dim as u32,
                (self.shape.rotated_dims / 2) as u32,
            ];
            // SAFETY: private fixed shader ABI. All u32 buffers are initialized,
            // aligned, same-device, disjoint and admitted with trailing guards.
            // Validated shapes bound every index; parameters copy immediately.
            // Tracked hazards and retained command resources protect lifetimes.
            // CPU views are excluded until checked command completion below.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&self.input.raw), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&self.coefficients.raw), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&self.output.raw), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&shape).cast(),
                    size_of::<[u32; 4]>(),
                    3,
                );
                encoder.setBuffer_offset_atIndex(Some(&self.status.raw), 0, 4);
            }
            let count = self.sizes.values / 2;
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: count.div_ceil(self.context.width),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.context.width,
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
            // Completion excludes all device access before these CPU reads.
            let tail = |words: &[u32], n| words[n..].iter().all(|&w| w == SENTINEL);
            if !self.input.with_slice(|s| tail(s, self.sizes.values))
                || !self.output.with_slice(|s| tail(s, self.sizes.values))
                || !self
                    .coefficients
                    .with_slice(|s| tail(s, self.sizes.coefficients))
                || !self.status.with_slice(|s| tail(s, 1))
            {
                return Err(Error::Options("Gemma rotary guard changed"));
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
