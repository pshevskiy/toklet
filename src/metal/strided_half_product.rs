//! Borrowed finite Half product over checked row views, no activation fusion.
//! Test-only until a concrete graph owns this operation. No allocation, wait,
//! upload, status reset or CPU value access occurs in encode. Failed output is
//! discarded by the caller. Readonly BCx parent aliases are intentional.
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
pub(super) struct View {
    pub offset: usize,
    pub stride: usize,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Plan {
    rows: usize,
    width: usize,
    a: View,
    b: View,
    output: View,
    a_len: usize,
    b_len: usize,
    output_len: usize,
}
impl Plan {
    pub(super) fn new(rows: usize, width: usize, a: View, b: View, output: View) -> Result<Self> {
        if rows == 0 || rows > 32 || ![256, 1024].contains(&width) {
            return Err(Error::Options("Half product shape profile"));
        }
        let extent = |v: View, writable: bool| {
            if v.stride < width || v.offset > u32::MAX as usize || v.stride > u32::MAX as usize {
                return Err(Error::Options("Half product view profile"));
            }
            let end = if writable {
                rows.checked_mul(v.stride)
            } else {
                (rows - 1)
                    .checked_mul(v.stride)
                    .and_then(|n| n.checked_add(width))
            };
            end.and_then(|n| n.checked_add(v.offset))
                .and_then(|n| n.checked_add(GUARD))
                .filter(|&n| n <= u32::MAX as usize && n <= isize::MAX as usize / 2)
                .ok_or(Error::Options("Half product UInt32 extent overflow"))
        };
        Ok(Self {
            rows,
            width,
            a,
            b,
            output,
            a_len: extent(a, false)?,
            b_len: extent(b, false)?,
            output_len: extent(output, true)?,
        })
    }
    /// Minimum view extents, NOT an allocation total: a and b may share a parent.
    /// The caller accounts actual unique physical owners and framework separately.
    pub(super) fn required_lengths(self) -> [usize; 4] {
        [self.a_len, self.b_len, self.output_len, 1 + GUARD]
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
            return Err(Error::Unsupported("Half product precise math unavailable"));
        }
        let library = gpu
            .device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("strided_half_product.metal")),
                Some(&options),
            )
            .map_err(|e| Error::Native(e.to_string()))?;
        let function = library
            .newFunctionWithName(&NSString::from_str("strided_half_product"))
            .ok_or_else(|| Error::Native("Half product kernel absent".into()))?;
        let pipeline = gpu
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| Error::Native(e.to_string()))?;
        let width = pipeline.threadExecutionWidth();
        if width == 0
            || width > pipeline.maxTotalThreadsPerThreadgroup()
            || width > u32::MAX as usize
        {
            return Err(Error::Unsupported("Half product launch width unavailable"));
        }
        Ok(Self { pipeline, width })
    }
    pub(super) fn encode(
        &self,
        command: &Command<'_>,
        plan: Plan,
        io: Invocation<'_>,
    ) -> Result<()> {
        let [a_len, b_len, output_len, status_len] = plan.required_lengths();
        if io.a.len < a_len
            || io.b.len < b_len
            || io.output.len < output_len
            || io.status.len < status_len
        {
            return Err(Error::Options("Half product operand extent"));
        }
        let buffers = [&*io.a.raw, &*io.b.raw, &*io.output.raw, &*io.status.raw];
        // Read/read identity is legal (two BCx views); all writable aliases fail.
        if std::ptr::eq(buffers[2], buffers[0])
            || std::ptr::eq(buffers[2], buffers[1])
            || std::ptr::eq(buffers[3], buffers[0])
            || std::ptr::eq(buffers[3], buffers[1])
            || std::ptr::eq(buffers[3], buffers[2])
        {
            return Err(Error::Options("Half product writable alias"));
        }
        let device = self.pipeline.device();
        if !std::ptr::eq(&*command.device(), &*device)
            || buffers
                .iter()
                .any(|b| !std::ptr::eq(&*b.device(), &*device))
        {
            return Err(Error::Options("Half product device identity"));
        }
        if !command.retainedReferences() {
            return Err(Error::Unsupported("Half product unretained resources"));
        }
        let groups = (plan.rows * plan.width).div_ceil(self.width);
        if groups
            .checked_mul(self.width)
            .is_none_or(|n| n > u32::MAX as usize)
        {
            return Err(Error::Options("Half product rounded dispatch"));
        }
        let (index, encoder) = if let Some(trace) = command.trace {
            let (i, e) = trace.encoder(command.raw, "strided_half_product")?;
            (Some(i), e)
        } else {
            (
                None,
                command
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::Native("Half product encoder absent".into()))?,
            )
        };
        encoder.setComputePipelineState(&self.pipeline);
        let args = [
            plan.rows as u32,
            plan.width as u32,
            plan.a.offset as u32,
            plan.a.stride as u32,
            plan.b.offset as u32,
            plan.b.stride as u32,
            plan.output.offset as u32,
            plan.output.stride as u32,
        ];
        // SAFETY: checked UInt32 ABI and lengths; initialized same-device buffers.
        // The readonly inputs may share one parent; writable resources are disjoint.
        // u16/u32 alignment meets Half/status requirements. setBytes copies args.
        // Retained command resources outlive completion; caller excludes all CPU
        // access/reuse until completion and owns sticky-status publication.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&io.a.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&io.b.raw), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&io.output.raw), 0, 2);
            encoder.setBytes_length_atIndex(NonNull::from(&args).cast(), size_of::<[u32; 8]>(), 3);
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
    pub a: &'a Buffer<u16>,
    pub b: &'a Buffer<u16>,
    pub output: &'a Buffer<u16>,
    pub status: &'a Buffer<u32>,
}

#[cfg(test)]
mod tests;
