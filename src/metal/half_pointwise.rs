//! Borrowed Half GELU and scalar scale, with explicit Half output stores.
//! Test-only until the complete resident graph owns/uses this optional pipeline.
//! Encode performs no operand upload/allocation, host wait or status reset.
//! Rust allocation qualification is release-only; driver encoders are separate.
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
pub(super) enum Kind {
    Gelu,
    Scale(f32),
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Plan {
    kind: Kind,
    rows: usize,
    width: usize,
    input_stride: usize,
    output_stride: usize,
    input_len: usize,
    output_len: usize,
}
impl Plan {
    pub(super) fn new(
        kind: Kind,
        rows: usize,
        width: usize,
        input_stride: usize,
        output_stride: usize,
    ) -> Result<Self> {
        let accepted = match kind {
            Kind::Gelu => [128, 2048].contains(&width),
            Kind::Scale(factor) => {
                [64, 640].contains(&width)
                    && [8.0f32.to_bits(), 25.25f32.to_bits()].contains(&factor.to_bits())
            }
        };
        if rows == 0 || rows > 32 || !accepted || input_stride < width || output_stride < width {
            return Err(Error::Options("Half pointwise shape/factor profile"));
        }
        let extent = |stride: usize| {
            rows.checked_mul(stride)
                .filter(|&n| n <= u32::MAX as usize)
                .and_then(|n| n.checked_add(GUARD))
                .filter(|&n| n <= isize::MAX as usize / 2)
                .ok_or(Error::Options("Half pointwise UInt32 extent overflow"))
        };
        let input_len = extent(input_stride)?;
        let output_len = extent(output_stride)?;
        Ok(Self {
            kind,
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
    pub(super) fn status_len(self) -> usize {
        1 + GUARD
    }
    /// Numerical input/output/status allocations; compiler/context/caller and
    /// diagnostic owners are separate. No immutable coefficient buffer.
    pub(super) fn operand_bytes(self) -> Result<usize> {
        self.input_len
            .checked_add(self.output_len)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| {
                self.status_len()
                    .checked_mul(4)
                    .and_then(|m| n.checked_add(m))
            })
            .ok_or(Error::Options("Half pointwise operand bytes overflow"))
    }
}

pub(super) struct Operation {
    pipelines: [Retained<ProtocolObject<dyn MTLComputePipelineState>>; 2],
    widths: [usize; 2],
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
                "Half pointwise precise math unavailable",
            ));
        }
        let library = gpu
            .device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("half_pointwise.metal")),
                Some(&options),
            )
            .map_err(|e| Error::Native(e.to_string()))?;
        let compile = |name: &str| -> Result<_> {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| Error::Native("Half pointwise kernel absent".into()))?;
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
                    "Half pointwise launch width unavailable",
                ));
            }
            Ok((pipeline, width))
        };
        let (gelu, gelu_width) = compile("half_tanh_gelu")?;
        let (scale, scale_width) = compile("half_scalar_scale")?;
        Ok(Self {
            pipelines: [gelu, scale],
            widths: [gelu_width, scale_width],
        })
    }
    pub(super) fn encode(
        &self,
        command: &Command<'_>,
        plan: Plan,
        io: Invocation<'_>,
    ) -> Result<()> {
        if io.input.len < plan.input_len()
            || io.output.len < plan.output_len()
            || io.status.len < plan.status_len()
        {
            return Err(Error::Options("Half pointwise operand extent"));
        }
        let buffers = [&*io.input.raw, &*io.output.raw, &*io.status.raw];
        for (index, buffer) in buffers.iter().enumerate() {
            if buffers[..index]
                .iter()
                .any(|other| std::ptr::eq(*buffer, *other))
            {
                return Err(Error::Options("Half pointwise operand alias"));
            }
        }
        let (variant, factor, label) = match plan.kind {
            Kind::Gelu => (0, 0, "half_tanh_gelu"),
            Kind::Scale(factor) => (1, factor.to_bits(), "half_scalar_scale"),
        };
        let pipeline = &self.pipelines[variant];
        let width = self.widths[variant];
        let device = pipeline.device();
        if !std::ptr::eq(&*command.device(), &*device)
            || buffers
                .iter()
                .any(|b| !std::ptr::eq(&*b.device(), &*device))
        {
            return Err(Error::Options("Half pointwise device identity"));
        }
        if !command.retainedReferences() {
            return Err(Error::Unsupported("Half pointwise unretained resources"));
        }
        let groups = (plan.rows * plan.width).div_ceil(width);
        if groups
            .checked_mul(width)
            .is_none_or(|n| n > u32::MAX as usize)
        {
            return Err(Error::Options("Half pointwise rounded dispatch"));
        }
        let (index, encoder) = if let Some(trace) = command.trace {
            let (i, e) = trace.encoder(command.raw, label)?;
            (Some(i), e)
        } else {
            (
                None,
                command
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::Native("Half pointwise encoder absent".into()))?,
            )
        };
        encoder.setComputePipelineState(pipeline);
        let args = [
            plan.rows as u32,
            plan.width as u32,
            plan.input_stride as u32,
            plan.output_stride as u32,
            factor,
        ];
        // SAFETY: private fixed ABI; same-device initialized allocations are
        // pairwise disjoint and every UInt32 offset/extent/rounded lane checked.
        // u16/u32 storage meets Half/FP32 alignment; stack shape is copied now.
        // Retained command resources live through completion. Caller excludes
        // CPU access/reuse until completion and owns shared status publication.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&io.input.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&io.output.raw), 0, 1);
            encoder.setBytes_length_atIndex(NonNull::from(&args).cast(), size_of::<[u32; 5]>(), 2);
            encoder.setBuffer_offset_atIndex(Some(&io.status.raw), 0, 3);
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
        if let (Some(trace), Some(index)) = (command.trace, index) {
            trace.finish_encoder(index);
        }
        Ok(())
    }
}
pub(super) struct Invocation<'a> {
    pub input: &'a Buffer<u16>,
    pub output: &'a Buffer<u16>,
    pub status: &'a Buffer<u32>,
}

#[cfg(test)]
mod tests;
