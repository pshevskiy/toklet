//! Borrowed serial Reference attention over old ring and immutable new Half KV.
//! Test-only until the complete resident graph owns/uses this optional pipeline.
//! No operand upload/allocation, host wait, cache write or status reset in encode.
//! Rust allocation qualification is release-only; driver encoders are separate.
use super::{
    Error, Result,
    memory::{Buffer, Gpu},
    profile::Command,
};
use crate::capacity::{KvOwner, KvStorage, Retention};
use crate::model::window_rows;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::{mem::size_of, ptr::NonNull};

pub(super) const GUARD: usize = 7;
pub(super) const CHUNK: usize = 32;
pub(super) const HEADS: usize = 4;
pub(super) const PIPELINE_ALLOWANCE: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub(super) struct Plan {
    capacity: usize,
    dim: usize,
    retained: usize,
    committed: usize,
    tokens: usize,
    window: Option<usize>,
    start: usize,
    keys: usize,
    old: [window_rows::OldSpan; 2],
}
impl Plan {
    pub(super) fn new(
        capacity: usize,
        dim: usize,
        window: Option<usize>,
        committed: usize,
        tokens: usize,
    ) -> Result<Self> {
        if capacity == 0
            || capacity > 544
            || ![32, 256].contains(&dim)
            || window.is_some_and(|w| w == 0 || w > capacity)
            || tokens == 0
            || tokens > CHUNK
        {
            return Err(Error::Options("staged attention shape profile"));
        }
        let end = committed
            .checked_add(tokens)
            .filter(|&end| end <= capacity)
            .ok_or(Error::Options("staged attention position admission"))?;
        let retained = window.unwrap_or(capacity);
        let start = window_rows::view_start(committed, window);
        let old = window_rows::old_spans(retained, committed, window);
        let plan = Self {
            capacity,
            dim,
            retained,
            committed,
            tokens,
            window,
            start,
            keys: end - start,
            old,
        };
        plan.operand_bytes()?;
        Ok(plan)
    }
    pub(super) fn q_len(self) -> usize {
        CHUNK * HEADS * self.dim + GUARD
    }
    pub(super) fn old_len(self) -> usize {
        self.retained * self.dim + GUARD
    }
    pub(super) fn staged_len(self) -> usize {
        CHUNK * self.dim + GUARD
    }
    pub(super) fn score_len(self) -> usize {
        HEADS * CHUNK * self.capacity + GUARD
    }
    pub(super) fn context_len(self) -> usize {
        self.q_len()
    }
    pub(super) fn status_len(self) -> usize {
        1 + GUARD
    }
    /// All guarded physical operands; immutable weights/GPU framework/compiler and
    /// caller/diagnostic owners are separate. KV1 is never expanded to Q4.
    pub(super) fn operand_bytes(self) -> Result<usize> {
        let old = KvOwner {
            heads: 1,
            head_dim: self.dim,
            storage: KvStorage::F16,
            retention: self
                .window
                .map_or(Retention::Full, |window| Retention::Sliding { window }),
            guard_elements: GUARD,
            copies: 1,
        }
        .at_capacity(self.capacity)
        .map_err(|_| Error::Options("staged attention old KV bytes"))?;
        debug_assert_eq!(old.elements_per_array, self.old_len());
        let half_scratch = self
            .q_len()
            .checked_add(self.context_len())
            .and_then(|n| {
                self.staged_len()
                    .checked_mul(2)
                    .and_then(|m| n.checked_add(m))
            })
            .and_then(|n| n.checked_mul(2));
        let float_scratch = self
            .score_len()
            .checked_mul(2)
            .and_then(|n| n.checked_add(self.status_len()))
            .and_then(|n| n.checked_mul(4));
        half_scratch
            .and_then(|n| float_scratch.and_then(|m| n.checked_add(m)))
            .and_then(|n| n.checked_add(old.total_bytes))
            .ok_or(Error::Options("staged attention operand bytes overflow"))
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
                "staged attention precise math unavailable",
            ));
        }
        let library = gpu
            .device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("staged_attention.metal")),
                Some(&options),
            )
            .map_err(|e| Error::Native(e.to_string()))?;
        let function = library
            .newFunctionWithName(&NSString::from_str("staged_reference_attention"))
            .ok_or_else(|| Error::Native("staged attention kernel absent".into()))?;
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
                "staged attention launch width unavailable",
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
        if io.q.len < plan.q_len()
            || io.old_k.len < plan.old_len()
            || io.old_v.len < plan.old_len()
            || io.new_k.len < plan.staged_len()
            || io.new_v.len < plan.staged_len()
            || io.scores.len < plan.score_len()
            || io.probabilities.len < plan.score_len()
            || io.context.len < plan.context_len()
            || io.status.len < plan.status_len()
        {
            return Err(Error::Options("staged attention operand extent"));
        }
        let buffers = [
            &*io.q.raw,
            &*io.old_k.raw,
            &*io.old_v.raw,
            &*io.new_k.raw,
            &*io.new_v.raw,
            &*io.scores.raw,
            &*io.probabilities.raw,
            &*io.context.raw,
            &*io.status.raw,
        ];
        for (index, buffer) in buffers.iter().enumerate() {
            if buffers[..index]
                .iter()
                .any(|other| std::ptr::eq(*buffer, *other))
            {
                return Err(Error::Options("staged attention operand alias"));
            }
        }
        let device = self.pipeline.device();
        if !std::ptr::eq(&*command.device(), &*device)
            || buffers
                .iter()
                .any(|b| !std::ptr::eq(&*b.device(), &*device))
        {
            return Err(Error::Options("staged attention device identity"));
        }
        if !command.retainedReferences() {
            return Err(Error::Unsupported("staged attention unretained resources"));
        }
        // Shape bounds make every Q/K/V/score address and rounded lane UInt32.
        // Rows/strides are derived once from the shared host row descriptors.
        let count = plan.tokens * HEADS;
        let groups = count.div_ceil(self.width);
        if groups
            .checked_mul(self.width)
            .is_none_or(|n| n > u32::MAX as usize)
        {
            return Err(Error::Options("staged attention rounded dispatch"));
        }
        let (index, encoder) = if let Some(trace) = command.trace {
            let (i, e) = trace.encoder(command.raw, "staged_reference_attention")?;
            (Some(i), e)
        } else {
            (
                None,
                command
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::Native("staged attention encoder absent".into()))?,
            )
        };
        encoder.setComputePipelineState(&self.pipeline);
        let args = [
            plan.tokens as u32,
            plan.dim as u32,
            plan.capacity as u32,
            plan.committed as u32,
            plan.start as u32,
            plan.keys as u32,
            plan.old[0].slot as u32,
            plan.old[0].rows as u32,
            plan.old[1].rows as u32,
            plan.window.unwrap_or(0) as u32,
            CHUNK as u32,
        ];
        // SAFETY: fixed private ABI; initialized, same-device, pairwise disjoint
        // allocations with checked lengths/index domain and Half/FP32 alignment.
        // Each thread owns disjoint active scores/probs/context. Status is atomic.
        // Scalars are copied immediately; retained resources survive completion.
        // Caller excludes all CPU access/reuse until command completion.
        unsafe {
            for (index, buffer) in buffers[..8].iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(*buffer), 0, index);
            }
            encoder.setBytes_length_atIndex(NonNull::from(&args).cast(), size_of::<[u32; 11]>(), 8);
            encoder.setBuffer_offset_atIndex(Some(&io.status.raw), 0, 9);
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
    pub q: &'a Buffer<u16>,
    pub old_k: &'a Buffer<u16>,
    pub old_v: &'a Buffer<u16>,
    pub new_k: &'a Buffer<u16>,
    pub new_v: &'a Buffer<u16>,
    pub scores: &'a Buffer<u32>,
    pub probabilities: &'a Buffer<u32>,
    pub context: &'a Buffer<u16>,
    pub status: &'a Buffer<u32>,
}

#[cfg(test)]
mod tests;
