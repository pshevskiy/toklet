//! Standalone borrowed causal Conv3 over Half u with original BF16→FP32 taps.
//! Test-only until the mixed resident graph owns this operation and its state.
//! Candidate history is not publication: caller discards it on any failed status.
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
    ) -> Result<Self> {
        if rows == 0
            || rows > 32
            || ![5, 256, 1024].contains(&width)
            || input_stride < width
            || output_stride < width
        {
            return Err(Error::Options("Half Conv3 shape profile"));
        }
        let extent = |stride: usize| {
            rows.checked_mul(stride)
                .filter(|&n| n <= u32::MAX as usize)
                .and_then(|n| n.checked_add(GUARD))
                .filter(|&n| n <= isize::MAX as usize / 2)
                .ok_or(Error::Options("Half Conv3 UInt32 extent overflow"))
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
        3 * self.width + GUARD
    }
    pub(super) fn history_len(self) -> usize {
        2 * self.width + GUARD
    }
    pub(super) fn status_len(self) -> usize {
        1 + GUARD
    }
    /// Numerical operand allocations only; model may share immutable coefficients
    /// across invocations. Driver/pipeline and caller/diagnostic owners are separate.
    pub(super) fn operand_bytes(self) -> Result<usize> {
        self.input_len
            .checked_add(self.output_len)
            .and_then(|n| {
                self.history_len()
                    .checked_mul(2)
                    .and_then(|h| n.checked_add(h))
            })
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| {
                self.scale_len()
                    .checked_add(self.status_len())
                    .and_then(|m| m.checked_mul(4))
                    .and_then(|m| n.checked_add(m))
            })
            .ok_or(Error::Options("Half Conv3 operand bytes overflow"))
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
            return Err(Error::Unsupported("Half Conv3 precise math unavailable"));
        }
        let library = gpu
            .device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("causal_conv3_half.metal")),
                Some(&options),
            )
            .map_err(|e| Error::Native(e.to_string()))?;
        let function = library
            .newFunctionWithName(&NSString::from_str("causal_conv3_half"))
            .ok_or_else(|| Error::Native("Half Conv3 kernel absent".into()))?;
        let pipeline = gpu
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| Error::Native(e.to_string()))?;
        let width = pipeline.threadExecutionWidth();
        if width == 0
            || width > pipeline.maxTotalThreadsPerThreadgroup()
            || width > u32::MAX as usize
        {
            return Err(Error::Unsupported("Half Conv3 launch width unavailable"));
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
            || io.old.len < plan.history_len()
            || io.candidate.len < plan.history_len()
            || io.status.len < plan.status_len()
        {
            return Err(Error::Options("Half Conv3 operand extent"));
        }
        let buffers = [
            &*io.input.raw,
            &*io.scales.raw,
            &*io.output.raw,
            &*io.old.raw,
            &*io.candidate.raw,
            &*io.status.raw,
        ];
        for (index, buffer) in buffers.iter().enumerate() {
            if buffers[..index]
                .iter()
                .any(|other| std::ptr::eq(*buffer, *other))
            {
                return Err(Error::Options("Half Conv3 operand alias"));
            }
        }
        let device = self.pipeline.device();
        if !std::ptr::eq(&*command.device(), &*device)
            || buffers
                .iter()
                .any(|b| !std::ptr::eq(&*b.device(), &*device))
        {
            return Err(Error::Options("Half Conv3 device identity"));
        }
        if !command.retainedReferences() {
            return Err(Error::Unsupported("Half Conv3 unretained resources"));
        }
        let groups = (plan.rows * plan.width).div_ceil(self.width);
        if groups
            .checked_mul(self.width)
            .is_none_or(|n| n > u32::MAX as usize)
        {
            return Err(Error::Options("Half Conv3 rounded dispatch"));
        }
        let (index, encoder) = if let Some(trace) = command.trace {
            let (i, e) = trace.encoder(command.raw, "causal_conv3_half")?;
            (Some(i), e)
        } else {
            (
                None,
                command
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::Native("Half Conv3 encoder absent".into()))?,
            )
        };
        encoder.setComputePipelineState(&self.pipeline);
        let args = [
            plan.rows as u32,
            plan.width as u32,
            plan.input_stride as u32,
            plan.output_stride as u32,
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
            encoder.setBuffer_offset_atIndex(Some(&io.old.raw), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&io.candidate.raw), 0, 4);
            encoder.setBytes_length_atIndex(NonNull::from(&args).cast(), size_of::<[u32; 4]>(), 5);
            encoder.setBuffer_offset_atIndex(Some(&io.status.raw), 0, 6);
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
    pub old: &'a Buffer<u16>,
    pub candidate: &'a Buffer<u16>,
    pub status: &'a Buffer<u32>,
}

#[cfg(test)]
mod tests;
