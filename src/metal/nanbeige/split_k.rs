//! Fixed T128 K/V split-K projections using the pinned MLX 0.32.4 exports.
use super::{
    Buffer, Error, HIDDEN, KV_WIDTH, Model, Result,
    kernel::{Encoder, RawBuffer},
    qmm,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::ptr::NonNull;

pub(super) const ROWS: usize = 128;
const SPLITS: usize = 4;
pub(super) const PARTIAL_WORDS: usize = SPLITS * ROWS * KV_WIDTH;
pub(super) const PARTIAL_BYTES: usize = PARTIAL_WORDS * 4;
const SPLIT: &str = "affine_qmm_t_splitk_bfloat16_t_gs_64_b_4_alN_true";
const MERGE: &str = "steel_gemm_splitk_accum_bfloat16_float32";

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

pub(super) struct SplitK {
    split: Pipeline,
    merge: Pipeline,
}

impl SplitK {
    pub fn prepare(model: &Model, library: &ProtocolObject<dyn MTLLibrary>) -> Result<Self> {
        let device = &model.context.device;
        let maximum = device.maxThreadsPerThreadgroup();
        if HIDDEN != 3072
            || KV_WIDTH != 1024
            || !device.hasUnifiedMemory()
            || maximum.width < 32
            || maximum.height < 32
            || maximum.depth < 2
        {
            return Err(Error::Unsupported("Nanbeige Steel split-K geometry"));
        }
        for block in &model.plan.blocks {
            for matrix in [block.key, block.value] {
                if (matrix.layout.columns(), matrix.layout.rows()) != (HIDDEN, KV_WIDTH)
                    || !qmm::disjoint_exact_ranges([
                        (&model.weights[matrix.weights].raw, HIDDEN * KV_WIDTH / 2),
                        (&model.weights[matrix.scales].raw, HIDDEN * KV_WIDTH / 32),
                        (&model.weights[matrix.biases].raw, HIDDEN * KV_WIDTH / 32),
                    ])
                {
                    return Err(Error::Unsupported(
                        "Nanbeige Steel split-K packed K/V layout",
                    ));
                }
            }
        }
        super::super::memory::validate_buffer_bytes(PARTIAL_BYTES, device.maxBufferLength())?;
        let pipeline = |name: &str, threads| -> Result<Pipeline> {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or(Error::Unsupported("Nanbeige Steel split-K exact export"))?;
            let state = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| Error::Native(error.to_string()))?;
            if state.threadExecutionWidth() != 32
                || state.maxTotalThreadsPerThreadgroup() < threads
                || state.staticThreadgroupMemoryLength() > device.maxThreadgroupMemoryLength()
            {
                return Err(Error::Unsupported(
                    "Nanbeige Steel split-K pipeline geometry",
                ));
            }
            Ok(state)
        };
        let split = pipeline(SPLIT, 128)?;
        let merge = pipeline(MERGE, 1024)?;
        Ok(Self { split, merge })
    }

    pub fn encode(
        &self,
        encoder: &Encoder,
        banks: [&RawBuffer; 3],
        normalized: &RawBuffer,
        partials: &Buffer<u32>,
        output: &RawBuffer,
    ) {
        let [weights, scales, biases] = banks;
        encoder.setComputePipelineState(&self.split);
        bind(
            encoder,
            &[weights, scales, biases, normalized, &partials.raw],
            &[
                HIDDEN as u32,
                KV_WIDTH as u32,
                ROWS as u32,
                (HIDDEN / SPLITS) as u32,
                (ROWS * KV_WIDTH) as u32,
            ],
        );
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            grid(KV_WIDTH / 32, ROWS / 32, SPLITS),
            grid(32, 2, 2),
        );
        encoder.setComputePipelineState(&self.merge);
        bind(
            encoder,
            &[&partials.raw, output],
            &[SPLITS as u32, (ROWS * KV_WIDTH) as u32, KV_WIDTH as u32],
        );
        encoder.dispatchThreads_threadsPerThreadgroup(grid(KV_WIDTH, ROWS, 1), grid(32, 32, 1));
    }
}

fn grid(width: usize, height: usize, depth: usize) -> MTLSize {
    MTLSize {
        width,
        height,
        depth,
    }
}

fn bind(encoder: &Encoder, buffers: &[&RawBuffer], constants: &[u32]) {
    // SAFETY: admission fixes all bank, input, partial and output extents.
    // Session scratch is disjoint and retained through completion. The serial
    // encoder orders split writes, merge reads, scan reads and all later reuse
    // of the same tracked partial allocation. Metal copies every four-byte scalar.
    unsafe {
        for (index, buffer) in buffers.iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(buffer), 0, index);
        }
        for (index, value) in constants.iter().enumerate() {
            encoder.setBytes_length_atIndex(NonNull::from(value).cast(), 4, buffers.len() + index);
        }
    }
}
