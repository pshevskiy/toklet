//! Explicit Mac-only staged SIMD attention for authenticated Llama3.
use super::{
    ALLOWANCE, AttentionMode, Buffer, Command, Error, Gpu, NormalizationMode, ProjectionMode,
    ReadbackMode, Result, Session, add, mlx,
};
use crate::weights::content::DenseProfile;
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};
use std::ptr::NonNull;

const NAME: &str = "attention_staged_simd";

pub(super) struct Pipeline {
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    label: Retained<NSString>,
    width: usize,
}

impl Pipeline {
    pub(super) fn new(gpu: &Gpu) -> Result<Self> {
        let library = mlx::precise_library(
            gpu,
            concat!(
                include_str!("kernels.metal"),
                "\n",
                include_str!("llama_staged.metal")
            ),
        )?;
        let label = NSString::from_str(NAME);
        let function = library
            .newFunctionWithName(&label)
            .ok_or_else(|| Error::Native("missing staged Llama kernel".into()))?;
        let state = gpu
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| Error::Native(error.to_string()))?;
        let width = state.threadExecutionWidth();
        if !(32..=state.maxTotalThreadsPerThreadgroup()).contains(&width) {
            return Err(Error::Unsupported("staged Llama SIMD width"));
        }
        Ok(Self {
            state,
            label,
            width,
        })
    }

    pub(super) fn encode(
        &self,
        command: &Command<'_>,
        buffers: [&ProtocolObject<dyn MTLBuffer>; 6],
        shape: [u32; 8],
        status: &Buffer<u32>,
    ) -> Result<()> {
        let [rows, heads, kv_heads, dim, cap, chunk, offset, valid] = shape;
        if [heads, kv_heads, dim, cap, chunk] != [32, 8, 64, 512, 32]
            || !(1..=chunk).contains(&rows)
            || offset.checked_add(rows) != Some(valid)
            || valid > cap
            || status.len == 0
        {
            return Err(Error::Options("staged Llama shape differs"));
        }
        let query = (rows * heads * dim * 2) as usize;
        let cache = (kv_heads * cap * dim * 2) as usize;
        let workspace = (heads * chunk * cap * 4) as usize;
        if buffers
            .iter()
            .zip([query, cache, cache, workspace, workspace, query])
            .any(|(buffer, required)| buffer.length() < required)
        {
            return Err(Error::Options("staged Llama buffer extent differs"));
        }
        let (index, encoder) = if let Some(trace) = command.trace {
            let (index, encoder) = trace.encoder(command.raw, NAME)?;
            (Some(index), encoder)
        } else {
            (
                None,
                command
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::Native("staged Llama encoder absent".into()))?,
            )
        };
        encoder.setLabel(Some(&self.label));
        encoder.setComputePipelineState(&self.state);
        // SAFETY: The checked ABI binds disjoint, tracked owners that remain
        // live through completion. Each SIMD group writes its own score,
        // exponent and output regions. CPU access follows completion.
        unsafe {
            for (slot, buffer) in buffers.into_iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(buffer), 0, slot);
            }
            encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), size_of_val(&shape), 6);
            encoder.setBuffer_offset_atIndex(Some(&status.raw), 0, 7);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (rows * heads) as usize,
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

impl Session<'_> {
    /// Report explicit selection independently of the Reference workspace mode.
    pub fn llama3_staged_simd_enabled(&self) -> bool {
        self.llama_staged.is_some()
    }

    /// Admit the optional stage for the authenticated, bounded Mac profile.
    pub fn enable_llama3_staged_simd(&mut self) -> Result<()> {
        let model = self.model;
        let config = &model.config;
        if model
            .mlx_dense_profile_content_identity(DenseProfile::Llama3)
            .is_none()
            || (
                config.num_layers,
                config.query_heads,
                config.kv_heads,
                config.head_dim,
            ) != (16, 32, 8, 64)
            || model.scales.attention.is_some()
            || model.positional.is_none()
            || self.options.capacity != 512
            || self.options.prefill_chunk_size != 32
            || self.options.position_offset != 0
            || self.attention_mode != AttentionMode::Reference
            || self.projection_mode != ProjectionMode::MlxAffine
            || self.normalization_mode != NormalizationMode::Compensated
            || self.readback_mode != ReadbackMode::FullLogits
            || !self.history.is_empty()
            || self.evaluated != 0
            || self.llama_staged.is_some()
            || self.affine_tile4.is_some()
            || self.parallel_reference.is_some()
        {
            return Err(Error::Unsupported(
                "staged attention requires a fresh authenticated Llama3 Mac512/chunk32 session",
            ));
        }
        let admitted = add(self.memory.admitted_bytes, ALLOWANCE)?;
        if admitted > self.options.memory_limit_bytes {
            return Err(Error::Budget {
                required: admitted,
                limit: self.options.memory_limit_bytes,
            });
        }
        let allowance = add(self.memory.metadata_allowance_bytes, ALLOWANCE)?;
        let pipeline = autoreleasepool(|_| self.model.gpu.llama_staged())?;
        self.llama_staged = Some(pipeline);
        self.memory.metadata_allowance_bytes = allowance;
        self.memory.admitted_bytes = admitted;
        Ok(())
    }
}

#[cfg(test)]
mod witness;
