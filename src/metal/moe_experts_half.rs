//! Resident correctness leaves for the Granite fused expert layout.
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
pub(super) const PIPELINE_RESERVATION: usize = 64 * 1024 * 1024;
const LABELS: [&str; 4] = [
    "expert_gate_up",
    "expert_glu",
    "expert_down",
    "expert_reduce",
];

#[derive(Clone, Copy)]
pub(super) struct Plan {
    shape: [u32; 10],
    // Binding order: input, fused weights, down weights, gate, up, GLU,
    // expert values, reduced output, then IDs, FP32 gate words and status.
    pub(super) half: [usize; 8],
    pub(super) word: [usize; 3],
    #[cfg(test)]
    operand_bytes: usize,
}

impl Plan {
    pub(super) fn new(geometry: [usize; 5], strides: [usize; 5]) -> Result<Self> {
        let [capacity, experts, selected, hidden, intermediate] = geometry;
        let [input, routing, activation, values, output] = strides;
        if !(1..=32).contains(&capacity)
            || !(1..=32).contains(&experts)
            || !(1..=experts).contains(&selected)
            || hidden == 0
            || intermediate == 0
            || input < hidden
            || routing < selected
            || activation < intermediate
            || values < hidden
            || output < hidden
        {
            return Err(Error::Options("MoE expert geometry/strides"));
        }
        let product = |a: usize, b: usize| {
            a.checked_mul(b)
                .filter(|&n| n <= u32::MAX as usize)
                .ok_or(Error::Options("MoE expert UInt32 extent"))
        };
        let extent = |rows, stride| {
            product(rows, stride)?
                .checked_add(GUARD)
                .filter(|&n| n <= isize::MAX as usize / 4)
                .ok_or(Error::Options("MoE expert guarded extent"))
        };
        let slots = product(capacity, selected)?;
        let matrix = product(hidden, intermediate)?;
        let expert_weights = product(experts, matrix)?;
        let scratch = extent(slots, activation)?;
        let half = [
            extent(capacity, input)?,
            extent(2, expert_weights)?,
            extent(1, expert_weights)?,
            scratch,
            scratch,
            scratch,
            extent(slots, values)?,
            extent(capacity, output)?,
        ];
        let route = extent(capacity, routing)?;
        let word = [route, route, 1 + GUARD];
        #[cfg(test)]
        let operand_bytes = half
            .iter()
            .map(|&n| n * 2)
            .chain(word.iter().map(|&n| n * 4))
            .try_fold(0_usize, |total, bytes| total.checked_add(bytes))
            .ok_or(Error::Options("MoE expert operand bytes"))?;
        Ok(Self {
            shape: [
                capacity,
                experts,
                selected,
                hidden,
                intermediate,
                input,
                routing,
                activation,
                values,
                output,
            ]
            .map(|n| n as u32),
            half,
            word,
            #[cfg(test)]
            operand_bytes,
        })
    }

    #[cfg(test)]
    fn admit(self, maximum: usize, budget: usize) -> Result<usize> {
        let required = self
            .operand_bytes
            .checked_add(size_of::<Resident>())
            .ok_or(Error::Options("MoE expert owner bytes"))?;
        if required > budget {
            return Err(Error::Budget {
                required,
                limit: budget,
            });
        }
        for bytes in self
            .half
            .map(|n| n * 2)
            .into_iter()
            .chain(self.word.map(|n| n * 4))
        {
            validate_buffer_bytes(bytes, maximum)?;
        }
        Ok(required)
    }
}

#[cfg(test)]
struct Resident {
    plan: Plan,
    half: [Buffer<u16>; 8],
    word: [Buffer<u32>; 3],
}
#[cfg(test)]
impl Resident {
    fn new(gpu: &Gpu, plan: Plan, budget: usize) -> Result<Self> {
        plan.admit(gpu.device.maxBufferLength(), budget)?;
        let [input, fused, down, gate, up, glu, values, output] = plan.half;
        let [ids, gates, status] = plan.word;
        Ok(Self {
            plan,
            half: [
                Buffer::new(gpu, input)?,
                Buffer::new(gpu, fused)?,
                Buffer::new(gpu, down)?,
                Buffer::new(gpu, gate)?,
                Buffer::new(gpu, up)?,
                Buffer::new(gpu, glu)?,
                Buffer::new(gpu, values)?,
                Buffer::new(gpu, output)?,
            ],
            word: [
                Buffer::new(gpu, ids)?,
                Buffer::new(gpu, gates)?,
                Buffer::new(gpu, status)?,
            ],
        })
    }
    fn invocation(&self) -> Invocation<'_> {
        Invocation {
            half: self.half.each_ref(),
            word: self.word.each_ref(),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Invocation<'a> {
    pub(super) half: [&'a Buffer<u16>; 8],
    pub(super) word: [&'a Buffer<u32>; 3],
}

pub(super) struct Operation {
    pipelines: [Retained<ProtocolObject<dyn MTLComputePipelineState>>; 4],
    widths: [usize; 4],
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
            return Err(Error::Unsupported("MoE expert precise math unavailable"));
        }
        let library = gpu
            .device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("moe_experts_half.metal")),
                Some(&options),
            )
            .map_err(|e| Error::Native(e.to_string()))?;
        let compile = |label| {
            let function = library
                .newFunctionWithName(&NSString::from_str(label))
                .ok_or_else(|| Error::Native("MoE expert function absent".into()))?;
            gpu.device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| Error::Native(e.to_string()))
        };
        let [gate_up, glu, down, reduce] = LABELS;
        let pipelines = [
            compile(gate_up)?,
            compile(glu)?,
            compile(down)?,
            compile(reduce)?,
        ];
        let widths = pipelines.each_ref().map(|p| p.threadExecutionWidth());
        if pipelines
            .iter()
            .zip(widths)
            .any(|(p, w)| w == 0 || w > p.maxTotalThreadsPerThreadgroup())
        {
            return Err(Error::Unsupported("MoE expert launch width unavailable"));
        }
        Ok(Self { pipelines, widths })
    }

    /// The caller owns shared status and discards all candidates after failure.
    pub(super) fn encode(
        &self,
        command: &Command<'_>,
        plan: Plan,
        rows: usize,
        io: Invocation<'_>,
    ) -> Result<()> {
        self.encode_stages(command, plan, rows, io, 0)
    }

    fn encode_stages(
        &self,
        command: &Command<'_>,
        plan: Plan,
        rows: usize,
        io: Invocation<'_>,
        first: usize,
    ) -> Result<()> {
        if rows == 0
            || rows > plan.shape[0] as usize
            || io.half.iter().zip(plan.half).any(|(b, n)| b.len < n)
            || io.word.iter().zip(plan.word).any(|(b, n)| b.len < n)
        {
            return Err(Error::Options("MoE expert invocation extent"));
        }
        let [input, fused, down, gate, up, glu, values, output] = io.half.map(|b| &*b.raw);
        let [ids, gates, status] = io.word.map(|b| &*b.raw);
        let buffers = [
            input, fused, down, gate, up, glu, values, output, ids, gates, status,
        ];
        let device = self.pipelines[0].device();
        if !std::ptr::eq(&*command.device(), &*device)
            || buffers
                .iter()
                .any(|b| !std::ptr::eq(&*b.device(), &*device))
        {
            return Err(Error::Options("MoE expert device identity"));
        }
        for (index, buffer) in buffers.iter().enumerate() {
            if buffers[..index]
                .iter()
                .any(|other| std::ptr::eq(*buffer, *other))
            {
                return Err(Error::Options("MoE expert operand alias"));
            }
        }
        if !command.retainedReferences() {
            return Err(Error::Unsupported("MoE expert unretained resources"));
        }
        let mut shape = plan.shape;
        shape[0] = rows as u32;
        let slots = rows * shape[2] as usize;
        let hidden = shape[3] as usize;
        let intermediate = shape[4] as usize;
        let counts = [
            slots * intermediate,
            slots * intermediate,
            slots * hidden,
            rows * hidden,
        ];
        let groups = std::array::from_fn::<_, 4, _>(|i| counts[i].div_ceil(self.widths[i]));
        if groups
            .iter()
            .zip(self.widths)
            .any(|(g, w)| g.checked_mul(w).is_none_or(|n| n > u32::MAX as usize))
        {
            return Err(Error::Options("MoE expert rounded dispatch"));
        }
        for (i, &groups) in groups.iter().enumerate().skip(first) {
            let (trace_index, encoder) = if let Some(trace) = command.trace {
                let (index, encoder) = trace.encoder(command.raw, LABELS[i])?;
                (Some(index), encoder)
            } else {
                (
                    None,
                    command
                        .computeCommandEncoder()
                        .ok_or_else(|| Error::Native("MoE expert encoder absent".into()))?,
                )
            };
            encoder.setComputePipelineState(&self.pipelines[i]);
            // SAFETY: checked extents bound all UInt32 accesses. Expert IDs
            // are checked in the shader before each weight dereference. Buffers
            // are disjoint, tracked, aligned and retained on this device. Metal
            // copies the shape now. Caller excludes CPU access until completion.
            unsafe {
                for (index, buffer) in buffers.into_iter().enumerate() {
                    encoder.setBuffer_offset_atIndex(Some(buffer), 0, index);
                }
                encoder.setBytes_length_atIndex(
                    NonNull::from(&shape).cast(),
                    size_of::<[u32; 10]>(),
                    11,
                );
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.widths[i],
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            if let (Some(trace), Some(index)) = (command.trace, trace_index) {
                trace.finish_encoder(index);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
