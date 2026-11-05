//! Borrowed resident Half-score router, qualified here before graph integration.
//! Candidate outputs must be discarded after any nonzero shared status.
#[cfg(test)]
use super::memory::validate_buffer_bytes;
use super::{
    Error, Result,
    memory::{Buffer, Gpu},
    profile::Command,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::{mem::size_of, ptr::NonNull};
const GUARD: usize = 7;
const STATUS: usize = 1 + GUARD;
pub(super) const PIPELINE_RESERVATION: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub(super) struct Plan {
    capacity: usize,
    experts: usize,
    selected: usize,
    pub(super) input_stride: usize,
    output_stride: usize,
    pub(super) input_len: usize,
    pub(super) output_len: usize,
    #[cfg(test)]
    operand_bytes: usize,
}
impl Plan {
    pub(super) fn new(
        capacity: usize,
        experts: usize,
        selected: usize,
        input_stride: usize,
        output_stride: usize,
    ) -> Result<Self> {
        if !(1..=32).contains(&capacity)
            || !(1..=32).contains(&experts)
            || selected == 0
            || selected > experts
            || input_stride < experts
            || output_stride < selected
        {
            return Err(Error::Options("MoE router geometry/stride"));
        }
        let extent = |stride: usize, width: usize| {
            capacity
                .checked_mul(stride)
                .filter(|&n| n <= u32::MAX as usize)
                .and_then(|n| n.checked_add(GUARD))
                .filter(|&n| n <= isize::MAX as usize / width)
                .ok_or(Error::Options("MoE router guarded extent overflow"))
        };
        let input_len = extent(input_stride, 2)?;
        let output_len = extent(output_stride, 4)?;
        #[cfg(test)]
        let operand_bytes = input_len
            .checked_mul(2)
            .and_then(|n| output_len.checked_mul(8).and_then(|o| n.checked_add(o)))
            .and_then(|n| n.checked_add(STATUS * 4))
            .ok_or(Error::Options("MoE router byte overflow"))?;
        Ok(Self {
            capacity,
            experts,
            selected,
            input_stride,
            output_stride,
            input_len,
            output_len,
            #[cfg(test)]
            operand_bytes,
        })
    }
    #[cfg(test)]
    fn admit(self, maximum: usize, budget: usize) -> Result<usize> {
        let required = self
            .operand_bytes
            .checked_add(size_of::<Resident>())
            .ok_or(Error::Options("MoE router owner overflow"))?;
        if required > budget {
            return Err(Error::Budget {
                required,
                limit: budget,
            });
        }
        for bytes in [self.input_len * 2, self.output_len * 4, STATUS * 4] {
            validate_buffer_bytes(bytes, maximum)?;
        }
        Ok(required)
    }
}
/// Concrete qualification owner consumes the exact guarded allocation plan.
/// A later session can borrow already-owned score/status resources instead.
#[cfg(test)]
struct Resident {
    plan: Plan,
    input: Buffer<u16>,
    ids: Buffer<u32>,
    gates: Buffer<u32>,
    status: Buffer<u32>,
}
#[cfg(test)]
impl Resident {
    fn new(gpu: &Gpu, plan: Plan, budget: usize) -> Result<Self> {
        plan.admit(gpu.device.maxBufferLength(), budget)?;
        Ok(Self {
            plan,
            input: Buffer::new(gpu, plan.input_len)?,
            ids: Buffer::new(gpu, plan.output_len)?,
            gates: Buffer::new(gpu, plan.output_len)?,
            status: Buffer::new(gpu, STATUS)?,
        })
    }
    fn invocation(&self) -> Invocation<'_> {
        Invocation {
            input: &self.input,
            ids: &self.ids,
            gates: &self.gates,
            status: &self.status,
        }
    }
}
pub(super) struct Invocation<'a> {
    pub(super) input: &'a Buffer<u16>,
    pub(super) ids: &'a Buffer<u32>,
    pub(super) gates: &'a Buffer<u32>,
    pub(super) status: &'a Buffer<u32>,
}
pub(super) struct Operation {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    width: usize,
}
impl Operation {
    pub(super) fn new(gpu: &Gpu, budget: usize) -> Result<Self> {
        let required = PIPELINE_RESERVATION + size_of::<Self>();
        if budget < required {
            return Err(Error::Budget {
                required,
                limit: budget,
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
            return Err(Error::Unsupported("MoE router precise math unavailable"));
        }
        let library = gpu
            .device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("moe_router_half.metal")),
                Some(&options),
            )
            .map_err(|e| Error::Native(e.to_string()))?;
        let function = library
            .newFunctionWithName(&NSString::from_str("moe_router_half"))
            .ok_or_else(|| Error::Native("MoE router kernel absent".into()))?;
        let pipeline = gpu
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| Error::Native(e.to_string()))?;
        let width = pipeline.threadExecutionWidth();
        if width == 0
            || width > pipeline.maxTotalThreadsPerThreadgroup()
            || width > u32::MAX as usize
        {
            return Err(Error::Unsupported("MoE router launch width unavailable"));
        }
        Ok(Self { pipeline, width })
    }
    /// No upload, readback, allocation, wait or status reset in this encode.
    pub(super) fn encode(
        &self,
        command: &Command<'_>,
        plan: Plan,
        rows: usize,
        io: Invocation<'_>,
    ) -> Result<()> {
        if rows == 0
            || rows > plan.capacity
            || io.input.len < plan.input_len
            || io.ids.len < plan.output_len
            || io.gates.len < plan.output_len
            || io.status.len < STATUS
        {
            return Err(Error::Options("MoE router invocation extent"));
        }
        let buffers = [
            &*io.input.raw,
            &*io.ids.raw,
            &*io.gates.raw,
            &*io.status.raw,
        ];
        let device = self.pipeline.device();
        if !std::ptr::eq(&*command.device(), &*device)
            || buffers
                .iter()
                .any(|b| !std::ptr::eq(&*b.device(), &*device))
        {
            return Err(Error::Options("MoE router device identity"));
        }
        for (i, b) in buffers.iter().enumerate() {
            if buffers[..i].iter().any(|a| std::ptr::eq(*a, *b)) {
                return Err(Error::Options("MoE router operand alias"));
            }
        }
        if !command.retainedReferences() {
            return Err(Error::Unsupported("MoE router unretained resources"));
        }
        let groups = rows.div_ceil(self.width);
        if groups
            .checked_mul(self.width)
            .is_none_or(|n| n > u32::MAX as usize)
        {
            return Err(Error::Options("MoE router rounded grid overflow"));
        }
        let (index, encoder) = if let Some(trace) = command.trace {
            let (i, e) = trace.encoder(command.raw, "moe_router_half")?;
            (Some(i), e)
        } else {
            (
                None,
                command
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::Native("MoE router encoder absent".into()))?,
            )
        };
        encoder.setComputePipelineState(&self.pipeline);
        let args = [
            rows,
            plan.experts,
            plan.selected,
            plan.input_stride,
            plan.output_stride,
        ]
        .map(|n| n as u32);
        // SAFETY: checked UInt32 extents and active rows bound all shader offsets.
        // Tracked, same-device, disjoint buffers meet Half/u32 alignment. Shape
        // bytes are copied now; resources are retained through completion.
        // Caller excludes CPU access during execution and owns status publication.
        unsafe {
            for (i, b) in buffers.into_iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(b), 0, i);
            }
            encoder.setBytes_length_atIndex(NonNull::from(&args).cast(), size_of::<[u32; 5]>(), 4);
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
        if let (Some(trace), Some(i)) = (command.trace, index) {
            trace.finish_encoder(i);
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests;
