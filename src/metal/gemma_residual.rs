//! Private encode-only FP32 residual addition and separate layer scalar multiply.
//!
//! Contiguous [tokens,1536] (at most32rows) with distinct immutable operands,
//! private output and per-operation status. No fused add/scale or graph route.
//! Exact-copy cases preserve signed zeros/subnormals; uncertain FTZ-sensitive
//! arithmetic requests reference replay instead of publishing an approximation.
use super::{
    Error, Result,
    memory::{BufferRange, Gpu, validate_encode_ranges},
};
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::{mem::size_of, ptr::NonNull};

#[cfg(test)]
pub(super) mod pool_audit;

const MAX_VALUES: usize = 32 * 1536;
const FATAL: u32 = 1;
const REPLAY: u32 = 2;

pub(super) struct Pipelines {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    pipelines: [Retained<ProtocolObject<dyn MTLComputePipelineState>>; 2],
    widths: [usize; 2],
    #[cfg(test)]
    pub(super) pool_audit: Option<pool_audit::Pipeline>,
}
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Decision {
    Accepted,
    ReferenceReplay,
}
/// Inspect only after checked terminal completion and guards. Every operation
/// has its own status slot. In graph order, an earlier replay discards all
/// speculative descendants; later nonfinites cannot replace that decision.
pub(super) fn decision(status: u32) -> Result<Decision> {
    if status & !(FATAL | REPLAY) != 0 {
        return Err(Error::Options(
            "Gemma FP32 residual status contract differs",
        ));
    }
    if status & FATAL != 0 {
        return Err(Error::Nonfinite);
    }
    Ok(if status & REPLAY != 0 {
        Decision::ReferenceReplay
    } else {
        Decision::Accepted
    })
}
fn geometry(count: usize, output: usize, status: usize, width: usize) -> Result<usize> {
    if !(1..=MAX_VALUES).contains(&count) || output != count || status != 1 || width == 0 {
        return Err(Error::Options("Gemma FP32 residual extent differs"));
    }
    let groups = count.div_ceil(width);
    if groups
        .checked_mul(width)
        .is_none_or(|n| n > u32::MAX as usize)
    {
        return Err(Error::Options("Gemma FP32 residual dispatch overflow"));
    }
    Ok(groups)
}
impl Pipelines {
    pub(super) fn new(gpu: &Gpu) -> Result<Self> {
        autoreleasepool(|_| {
            let library = Self::library(gpu, include_str!("gemma_residual.metal"))?;
            Self::from_library(gpu, &library)
        })
    }
    #[cfg(test)]
    pub(super) fn new_for_pool(gpu: &Gpu) -> Result<Self> {
        autoreleasepool(|_| {
            let source = concat!(
                include_str!("gemma_residual.metal"),
                "\n",
                include_str!("gemma_residual/pool_audit.metal")
            );
            let library = Self::library(gpu, source)?;
            let mut result = Self::from_library(gpu, &library)?;
            result.pool_audit = Some(pool_audit::Pipeline::new(gpu, &library)?);
            Ok(result)
        })
    }
    fn library(gpu: &Gpu, source: &str) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version2_3);
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        if options.languageVersion() != MTLLanguageVersion::Version2_3
            || options.mathMode() != MTLMathMode::Safe
            || options.mathFloatingPointFunctions() != MTLMathFloatingPointFunctions::Precise
        {
            return Err(Error::Unsupported(
                "Gemma FP32 residual precise policy unavailable",
            ));
        }
        let library = gpu
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
            .map_err(|e| Error::Native(e.to_string()))?;
        Ok(library)
    }
    fn from_library(gpu: &Gpu, library: &ProtocolObject<dyn MTLLibrary>) -> Result<Self> {
        let compile = |name| {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| Error::Native("Gemma FP32 residual kernel absent".into()))?;
            gpu.device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| Error::Native(e.to_string()))
        };
        let pipelines = [
            compile("gemma_fp32_residual_add")?,
            compile("gemma_fp32_scalar_multiply")?,
        ];
        let widths = pipelines.each_ref().map(|p| p.threadExecutionWidth());
        for (p, &width) in pipelines.iter().zip(&widths) {
            if width == 0 || width > p.maxTotalThreadsPerThreadgroup() || width > u16::MAX as usize
            {
                return Err(Error::Unsupported(
                    "Gemma FP32 residual launch width unavailable",
                ));
            }
        }
        Ok(Self {
            device: gpu.device.clone(),
            pipelines,
            widths,
            #[cfg(test)]
            pool_audit: None,
        })
    }

    /// Separate FP32 addition, same operand order as cpu::residual_add(left,right).
    /// Read-only operands may alias. Every writable overlap rejects before an
    /// encoder is created, including in-place output and status/source overlap.
    /// Caller initializes status, excludes CPU views until terminal completion,
    /// checks guards/decision, and owns replay/poisoning/atomic publication.
    pub(super) fn encode_add(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        left: BufferRange<'_>,
        right: BufferRange<'_>,
        output: BufferRange<'_>,
        status: BufferRange<'_>,
    ) -> Result<()> {
        let groups = geometry(left.len, output.len, status.len, self.widths[0])?;
        if right.len != left.len {
            return Err(Error::Options("Gemma FP32 residual operand lengths differ"));
        }
        validate_encode_ranges(command, &self.device, &[left, right, output, status], 2)?;
        let shape = [left.len as u32];
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::Native("Gemma FP32 residual add encoder absent".into()))?;
        encoder.setComputePipelineState(&self.pipelines[0]);
        // SAFETY: exact private ABI, checked typed same-device retained/tracked
        // ranges and disjoint writes. Shape copies immediately. Caller owns
        // completion and ordered status/guard validation before publication.
        unsafe {
            for (slot, b) in [left, right, output].iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(b.raw), b.offset * 4, slot);
            }
            encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), size_of::<[u32; 1]>(), 3);
            encoder.setBuffer_offset_atIndex(Some(status.raw), status.offset * 4, 4);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.widths[0],
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        Ok(())
    }

    /// One distinct FP32 multiply after the last residual addition. Scalar is
    /// exactly the supplied finite FP32 learned value: no +1, clamp or fusion.
    /// Subnormal/underflow-sensitive non-identity arithmetic requests replay.
    /// No host wait or numerical staging/publication occurs in this method.
    pub(super) fn encode_scale(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        status: BufferRange<'_>,
        scalar: f32,
    ) -> Result<()> {
        let groups = geometry(input.len, output.len, status.len, self.widths[1])?;
        if !scalar.is_finite() {
            return Err(Error::Options("Gemma FP32 layer scalar must be finite"));
        }
        validate_encode_ranges(command, &self.device, &[input, output, status], 1)?;
        let shape = [input.len as u32, scalar.to_bits()];
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::Native("Gemma FP32 scalar encoder absent".into()))?;
        encoder.setComputePipelineState(&self.pipelines[1]);
        // SAFETY: exact private ABI, validated finite scalar, extent/resource
        // contracts and disjoint output/status. Caller owns synchronization,
        // guard/status validation and publication of the whole transaction.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw), input.offset * 4, 0);
            encoder.setBuffer_offset_atIndex(Some(output.raw), output.offset * 4, 1);
            encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), size_of::<[u32; 2]>(), 2);
            encoder.setBuffer_offset_atIndex(Some(status.raw), status.offset * 4, 3);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.widths[1],
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
