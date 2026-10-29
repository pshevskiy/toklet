//! Experimental borrowed Half RMS with model-owned FP32 effective scales.
//! Owned and consumed by the concrete resident Gemma3 graph.
//! No buffer upload, allocation, host wait or status reset occurs in encode.
use super::{
    Error, Result,
    memory::{Buffer, Gpu},
    profile::Command,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::{mem::size_of, ptr::NonNull};

pub(super) const GUARD: usize = 7;
pub(super) const PIPELINE_ALLOWANCE: usize = 64 * 1024 * 1024;
const EPSILON: f32 = 1e-6;

#[derive(Clone, Copy, Debug)]
pub(super) struct Plan {
    rows: usize,
    width: usize,
    input_stride: usize,
    output_stride: usize,
    input_len: usize,
    output_len: usize,
}
impl Plan {
    pub(super) fn new(
        rows: usize,
        width: usize,
        input_stride: usize,
        output_stride: usize,
        epsilon: f32,
    ) -> Result<Self> {
        if rows == 0
            || rows > 128
            || ![32, 64, 256, 640].contains(&width)
            || input_stride < width
            || output_stride < width
            || epsilon.to_bits() != EPSILON.to_bits()
        {
            return Err(Error::Options("Half FP32-scale RMS shape/epsilon profile"));
        }
        let extent = |stride: usize| {
            rows.checked_mul(stride)
                .filter(|&n| n <= u32::MAX as usize)
                .and_then(|n| n.checked_add(GUARD))
                .filter(|&n| n <= isize::MAX as usize / 2)
                .ok_or(Error::Options("Half FP32-scale RMS UInt32 extent overflow"))
        };
        let input_len = extent(input_stride)?;
        let output_len = extent(output_stride)?;
        Ok(Self {
            rows,
            width,
            input_stride,
            output_stride,
            input_len,
            output_len,
        })
    }
    pub(super) fn input_len(self) -> usize {
        self.input_len
    }
    pub(super) fn output_len(self) -> usize {
        self.output_len
    }
    pub(super) fn scale_len(self) -> usize {
        self.width + GUARD
    }
    pub(super) fn status_len(self) -> usize {
        1 + GUARD
    }
    /// Numerical operand allocations only; model may share immutable coefficients
    /// across invocations. Driver/pipeline and caller/diagnostic owners are separate.
    pub(super) fn operand_bytes(self) -> Result<usize> {
        self.input_len
            .checked_add(self.output_len)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| {
                self.scale_len()
                    .checked_add(self.status_len())
                    .and_then(|m| m.checked_mul(4))
                    .and_then(|m| n.checked_add(m))
            })
            .ok_or(Error::Options("Half FP32-scale RMS operand bytes overflow"))
    }
}

pub(super) struct Operation {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    width: usize,
}
impl Operation {
    pub(super) fn new(gpu: &Gpu, available: usize) -> Result<Self> {
        let required = PIPELINE_ALLOWANCE + size_of::<Self>();
        if available < required {
            return Err(Error::Budget {
                required,
                limit: available,
            });
        }
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version2_3);
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        if options.languageVersion() != MTLLanguageVersion::Version2_3
            || options.mathMode() != MTLMathMode::Safe
            || options.mathFloatingPointFunctions() != MTLMathFloatingPointFunctions::Precise
        {
            return Err(Error::Unsupported(
                "Half FP32-scale RMS precise math unavailable",
            ));
        }
        let library = gpu
            .device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("half_rms_f32.metal")),
                Some(&options),
            )
            .map_err(|e| Error::Native(e.to_string()))?;
        let function = library
            .newFunctionWithName(&NSString::from_str("half_rms_f32_scale"))
            .ok_or_else(|| Error::Native("Half FP32-scale RMS kernel absent".into()))?;
        let pipeline = gpu
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| Error::Native(e.to_string()))?;
        let width = pipeline.threadExecutionWidth();
        if width == 0
            || width > pipeline.maxTotalThreadsPerThreadgroup()
            || width > u32::MAX as usize
        {
            return Err(Error::Unsupported(
                "Half FP32-scale RMS launch width unavailable",
            ));
        }
        Ok(Self { pipeline, width })
    }
    pub(super) fn encode(
        &self,
        command: &Command<'_>,
        plan: Plan,
        io: Invocation<'_>,
    ) -> Result<()> {
        if io.input.len < plan.input_len()
            || io.output.len < plan.output_len()
            || io.scales.len < plan.scale_len()
            || io.status.len < plan.status_len()
        {
            return Err(Error::Options("Half FP32-scale RMS operand extent"));
        }
        let buffers = [
            &*io.input.raw,
            &*io.scales.raw,
            &*io.output.raw,
            &*io.status.raw,
        ];
        for (index, buffer) in buffers.iter().enumerate() {
            if buffers[..index]
                .iter()
                .any(|other| std::ptr::eq(*buffer, *other))
            {
                return Err(Error::Options("Half FP32-scale RMS operand alias"));
            }
        }
        let device = self.pipeline.device();
        if !std::ptr::eq(&*command.device(), &*device)
            || buffers
                .iter()
                .any(|b| !std::ptr::eq(&*b.device(), &*device))
        {
            return Err(Error::Options("Half FP32-scale RMS device identity"));
        }
        if !command.retainedReferences() {
            return Err(Error::Unsupported(
                "Half FP32-scale RMS unretained resources",
            ));
        }
        let groups = plan.rows.div_ceil(self.width);
        if groups
            .checked_mul(self.width)
            .is_none_or(|n| n > u32::MAX as usize)
        {
            return Err(Error::Options("Half FP32-scale RMS rounded dispatch"));
        }
        let (index, encoder) = if let Some(trace) = command.trace {
            let (i, e) = trace.encoder(command.raw, "half_rms_f32_scale")?;
            (Some(i), e)
        } else {
            (
                None,
                command
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::Native("Half FP32-scale RMS encoder absent".into()))?,
            )
        };
        encoder.setComputePipelineState(&self.pipeline);
        let args = [
            plan.rows as u32,
            plan.width as u32,
            plan.input_stride as u32,
            plan.output_stride as u32,
            EPSILON.to_bits(),
        ];
        // SAFETY: private fixed ABI; same-device initialized allocations are
        // pairwise disjoint and every UInt32 offset/extent/rounded lane checked.
        // u16/u32 storage meets Half/FP32 alignment; stack shape is copied now.
        // Retained command resources live through completion. Caller excludes
        // CPU access/reuse until completion and owns shared status publication.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&io.input.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&io.scales.raw), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&io.output.raw), 0, 2);
            encoder.setBytes_length_atIndex(NonNull::from(&args).cast(), size_of::<[u32; 5]>(), 3);
            encoder.setBuffer_offset_atIndex(Some(&io.status.raw), 0, 4);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups,
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
        if let (Some(trace), Some(index)) = (command.trace, index) {
            trace.finish_encoder(index);
        }
        Ok(())
    }
}
pub(super) struct Invocation<'a> {
    pub input: &'a Buffer<u16>,
    pub scales: &'a Buffer<u32>,
    pub output: &'a Buffer<u16>,
    pub status: &'a Buffer<u32>,
}

#[cfg(test)]
mod tests;
