//! Standalone resident FP32 weighted/unweighted Gemma RMS correctness operator.
//!
//! Direct learned scales (no +1), exact fixed FP32 epsilon, scaled compensated
//! reduction and exponent-aware weighted output. Qualified against independent
//! FP64 mathematical fixtures; no source parity, whole graph or speed claim.
use super::{
    Error, Result,
    memory::{Buffer, BufferRange, Gpu, validate_buffer_bytes, validate_encode_ranges},
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
    pipelines: Pipelines,
}
pub(super) struct Pipelines {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
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
            let pipelines = Pipelines::new(&gpu)?;
            Ok(Self { gpu, pipelines })
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
            input: Buffer::new(&self.gpu, sizes.values + GUARD)?,
            output: Buffer::new(&self.gpu, sizes.values + GUARD)?,
            weights: weight_buffer,
            status: Buffer::new(&self.gpu, 1 + GUARD)?,
            poisoned: false,
            pending: None,
        })
    }
}
impl Pipelines {
    pub(super) fn new(gpu: &Gpu) -> Result<Self> {
        autoreleasepool(|_| {
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
                device: gpu.device.clone(),
                pipelines,
                widths,
            })
        })
    }
    /// Encode only. Caller initializes status, excludes CPU views through terminal
    /// completion, validates status/guards and publishes all dependent outputs atomically.
    /// Invalid geometry/resources/commands reject before creating an encoder.
    pub(super) fn encode(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        buffers: &DeviceBuffers<'_>,
        rows: usize,
        width: usize,
        epsilon: f32,
        learned: bool,
    ) -> Result<()> {
        let g = Geometry::new(rows, width, epsilon, learned)?;
        if buffers.input.len != g.values
            || buffers.output.len != g.values
            || buffers.weights.len != g.weights
            || buffers.status.len != 1
        {
            return Err(Error::Options("Gemma RMS borrowed buffer length"));
        }
        validate_encode_ranges(
            command,
            &self.device,
            &[
                buffers.input,
                buffers.weights,
                buffers.output,
                buffers.status,
            ],
            2,
        )?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::Native("Gemma RMS encoder absent".into()))?;
        let variant = usize::from(learned);
        encoder.setComputePipelineState(&self.pipelines[variant]);
        // SAFETY: checked same-device retained/tracked u32 ranges, ABI lengths
        // and disjoint writable ranges. Shape bytes copy immediately. Caller
        // retains synchronization/guard validation and publication responsibility.
        unsafe {
            for (slot, b) in [buffers.input, buffers.weights, buffers.output]
                .iter()
                .enumerate()
            {
                encoder.setBuffer_offset_atIndex(Some(b.raw), b.offset * 4, slot);
            }
            encoder.setBytes_length_atIndex(
                NonNull::from(&g.words).cast(),
                size_of::<[u32; 3]>(),
                3,
            );
            encoder.setBuffer_offset_atIndex(
                Some(buffers.status.raw),
                buffers.status.offset * 4,
                4,
            );
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: rows,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.widths[variant],
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        Ok(())
    }
}
pub(super) struct DeviceBuffers<'a> {
    pub input: BufferRange<'a>,
    pub weights: BufferRange<'a>,
    pub output: BufferRange<'a>,
    pub status: BufferRange<'a>,
}
struct Geometry {
    words: [u32; 3],
    values: usize,
    weights: usize,
}
impl Geometry {
    fn new(rows: usize, width: usize, epsilon: f32, learned: bool) -> Result<Self> {
        if !(1..=256).contains(&rows)
            || !(1..=1536).contains(&width)
            || epsilon.to_bits() != EPSILON.to_bits()
        {
            return Err(Error::Options(
                "Gemma RMS fixed geometry/epsilon outside bounds",
            ));
        }
        let values = rows
            .checked_mul(width)
            .ok_or(Error::Options("Gemma RMS size overflow"))?;
        Ok(Self {
            words: [
                rows as u32,
                width as u32,
                (f64::from(epsilon).sqrt() as f32).to_bits(),
            ],
            values,
            weights: if learned { width } else { 1 },
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
        let g = Geometry::new(rows, width, epsilon, weights.is_some())?;
        if weights.is_some_and(|w| w.len() != width || w.iter().any(|x| !x.is_finite())) {
            return Err(Error::Options("Gemma RMS weights outside bounds"));
        }
        let values = g.values;
        let weights = g.weights;
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
            self.context.pipelines.encode(
                &raw,
                &DeviceBuffers {
                    input: BufferRange::new(&self.input, 0, self.sizes.values)?,
                    weights: BufferRange::new(&self.weights, 0, self.sizes.weights)?,
                    output: BufferRange::new(&self.output, 0, self.sizes.values)?,
                    status: BufferRange::new(&self.status, 0, 1)?,
                },
                self.rows,
                self.width,
                EPSILON,
                self.variant != 0,
            )?;
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
