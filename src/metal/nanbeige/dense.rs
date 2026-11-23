//! Prefill-only companion to the canonical packed banks.
use super::{
    Buffer, Context, Error, HIDDEN, INTERMEDIATE, Options, Result, Sizes, add,
    kernel::{Encoder, RawBuffer, encode_pipeline},
    plan,
};
use crate::metal::{memory::validate_buffer_bytes, mlx::precise_library_on};
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::{NSObjectProtocol, ProtocolObject},
    sel,
};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::time::Instant;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
const ROWS: usize = super::split_k::ROWS;
const THREADS: usize = 128;
const FRAGMENT_WORDS: usize = 4;
const TOKEN_BLOCK: usize = 16;
const PACKED_BYTES: usize = ROWS * HIDDEN * std::mem::size_of::<u16>();
const _: () = assert!(PACKED_BYTES <= ROWS * INTERMEDIATE * std::mem::size_of::<u16>());
const _: () = assert!((TOKEN_BLOCK * HIDDEN).is_multiple_of(THREADS * FRAGMENT_WORDS));
const _: () = assert!(ROWS.is_multiple_of(TOKEN_BLOCK));
const PAIR_BYTES: usize = HIDDEN * INTERMEDIATE * 4;
pub(super) const ALL_BYTES: usize = plan::PHYSICAL_LAYERS * PAIR_BYTES;

pub(super) struct Prepacked {
    pack: Pipeline,
    consume: Pipeline,
}

impl Prepacked {
    pub(super) fn new(pack: Pipeline, consume: Pipeline) -> Result<Self> {
        for pipeline in [&pack, &consume] {
            if pipeline.threadExecutionWidth() != 32
                || pipeline.maxTotalThreadsPerThreadgroup() < THREADS
                || pipeline.staticThreadgroupMemoryLength() != 0
            {
                return Err(Error::Unsupported(
                    "Nanbeige Steel dense prepack pipeline geometry",
                ));
            }
        }
        Ok(Self { pack, consume })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DensePrefillReceipt {
    pub physical_pairs: usize,
    pub startup_basis_seconds: f64,
    pub payload_bytes: usize,
    pub allocated_bytes: usize,
    pub allocation_seconds: f64,
    pub prepare_host_seconds: f64,
    pub prepare_gpu_seconds: f64,
    pub device_before_bytes: usize,
    pub device_after_bytes: usize,
    pub device_budget_bytes: usize,
    pub device_recommended_bytes: u64,
    pub distinct_address_ranges: usize,
}

pub(super) struct DensePrefill {
    pairs: Vec<Buffer<u32>>,
    pipeline: Pipeline,
    pub(super) receipt: DensePrefillReceipt,
}

fn budget(base: usize, extra: usize, limit: usize) -> Result<()> {
    let required = add(base, extra)?;
    if required > limit {
        return Err(Error::Budget { required, limit });
    }
    Ok(())
}

fn remaining_session(sizes: &Sizes) -> usize {
    sizes.memory.admitted_bytes - sizes.memory.weights_bytes
}

impl DensePrefill {
    pub(super) fn prepare(
        context: &Context,
        weights: &[Buffer<u32>],
        plan: &plan::Plan,
        sizes: &Sizes,
        options: Options,
    ) -> Result<Self> {
        budget(
            sizes.memory.admitted_bytes,
            ALL_BYTES,
            options.memory_limit_bytes,
        )?;
        let device = &context.device;
        if !cfg!(target_os = "macos")
            || options.prefill_chunk_size != 128
            || !device.supportsFamily(MTLGPUFamily::Apple7)
            || !context.has_gate_up_compact()
            || device.name().to_string() != "Apple M1 Pro"
            || device.maxThreadgroupMemoryLength() < 4608
        {
            return Err(Error::Unsupported("Nanbeige dense prefill capability"));
        }
        let budget_bytes = device.recommendedMaxWorkingSetSize();
        let device_limit = usize::try_from(budget_bytes)
            .unwrap_or(usize::MAX)
            .min(options.memory_limit_bytes);
        if device_limit == 0 {
            return Err(Error::Unsupported(
                "Nanbeige dense device budget unavailable",
            ));
        }
        validate_buffer_bytes(PAIR_BYTES, device.maxBufferLength())?;
        let before = device.currentAllocatedSize();
        budget(
            add(before, remaining_session(sizes))?,
            ALL_BYTES,
            device_limit,
        )?;

        let library = precise_library_on(
            device,
            concat!(
                include_str!("../nanbeige.metal"),
                "\n",
                include_str!("dense.metal"),
                "\n",
                include_str!("prepare.metal")
            ),
        )?;
        let pipeline = |name: &str, bytes| -> Result<Pipeline> {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or(Error::Unsupported("Nanbeige dense prefill function"))?;
            let pipe = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| Error::Native(e.to_string()))?;
            if pipe.threadExecutionWidth() != 32
                || pipe.maxTotalThreadsPerThreadgroup() < 128
                || pipe.staticThreadgroupMemoryLength() != bytes
            {
                return Err(Error::Unsupported(
                    "Nanbeige dense prefill pipeline geometry",
                ));
            }
            Ok(pipe)
        };
        let consume = pipeline("nanbeige_ffn_fused_gate_up_swiglu_dense_k64", 4608)?;
        let prepare = pipeline("nanbeige_prepare_dense_gate_up", 0)?;
        // Temporary basis owners are dropped before any model pair allocates.
        const BASIS_RESERVE: usize = 1 << 20;
        budget(
            add(device.currentAllocatedSize(), remaining_session(sizes))?,
            BASIS_RESERVE,
            device_limit,
        )?;
        let basis_started = Instant::now();
        autoreleasepool(|_| super::dense_basis::check(context, &prepare, &consume))?;
        let basis_seconds = basis_started.elapsed().as_secs_f64();

        let mut pairs: Vec<Buffer<u32>> = Vec::with_capacity(plan::PHYSICAL_LAYERS);
        let mut receipt = DensePrefillReceipt {
            physical_pairs: 0,
            startup_basis_seconds: basis_seconds,
            payload_bytes: ALL_BYTES,
            allocated_bytes: 0,
            allocation_seconds: 0.0,
            prepare_host_seconds: 0.0,
            prepare_gpu_seconds: 0.0,
            device_before_bytes: before,
            device_after_bytes: 0,
            device_budget_bytes: device_limit,
            device_recommended_bytes: budget_bytes,
            distinct_address_ranges: 0,
        };
        for block in &plan.blocks {
            let pending = (plan::PHYSICAL_LAYERS - pairs.len()) * PAIR_BYTES;
            budget(
                add(sizes.memory.admitted_bytes, receipt.allocated_bytes)?,
                pending,
                options.memory_limit_bytes,
            )?;
            budget(
                add(device.currentAllocatedSize(), remaining_session(sizes))?,
                pending,
                device_limit,
            )?;
            let allocation = Instant::now();
            let pair = Buffer::<u32>::on_device(device, PAIR_BYTES / 4)?;
            receipt.allocation_seconds += allocation.elapsed().as_secs_f64();
            let raw = &*pair.raw;
            if !raw.respondsToSelector(sel!(gpuAddress))
                || raw.gpuAddress() == 0
                || !raw.gpuAddress().is_multiple_of(16)
            {
                return Err(Error::Unsupported("Nanbeige dense allocation alignment"));
            }
            let begin = raw.gpuAddress();
            let end = begin
                .checked_add(PAIR_BYTES as u64)
                .ok_or(Error::Unsupported("Nanbeige dense GPU address extent"))?;
            for existing in &pairs {
                let address = existing.raw.gpuAddress();
                if begin < address + PAIR_BYTES as u64 && address < end {
                    return Err(Error::Unsupported("Nanbeige dense GPU address alias"));
                }
            }
            let allocated = MTLResource::allocatedSize(raw);
            if allocated < PAIR_BYTES {
                return Err(Error::Native("Nanbeige dense resource size differs".into()));
            }
            receipt.allocated_bytes = add(receipt.allocated_bytes, allocated)?;
            budget(
                add(sizes.memory.admitted_bytes, receipt.allocated_bytes)?,
                pending - PAIR_BYTES,
                options.memory_limit_bytes,
            )?;
            budget(
                add(device.currentAllocatedSize(), remaining_session(sizes))?,
                pending - PAIR_BYTES,
                device_limit,
            )?;

            let (host_seconds, gpu_seconds) = autoreleasepool(|_| {
                let host = Instant::now();
                let cmd = context.command()?;
                let encoder = cmd
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::Native("Nanbeige dense prepare encoder".into()))?;
                encode_pipeline(
                    &encoder,
                    &prepare,
                    128,
                    &[
                        &weights[block.gate.weights].raw,
                        &weights[block.gate.scales].raw,
                        &weights[block.gate.biases].raw,
                        &weights[block.up.weights].raw,
                        &weights[block.up.scales].raw,
                        &weights[block.up.biases].raw,
                        raw,
                    ],
                    &[128, HIDDEN as u32, INTERMEDIATE as u32, 0],
                    [(PAIR_BYTES / 16).div_ceil(128), 1],
                );
                encoder.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
                let seconds = host.elapsed().as_secs_f64();
                let start = cmd.GPUStartTime();
                let elapsed = cmd.GPUEndTime() - start;
                if cmd.status() != MTLCommandBufferStatus::Completed
                    || start <= 0.0
                    || !elapsed.is_finite()
                    || elapsed <= 0.0
                    || elapsed > seconds
                {
                    return Err(Error::Native("Nanbeige dense prepare completion".into()));
                }
                Ok((seconds, elapsed))
            })?;
            receipt.prepare_host_seconds += host_seconds;
            receipt.prepare_gpu_seconds += gpu_seconds;
            pairs.push(pair);
        }
        receipt.physical_pairs = pairs.len();
        receipt.distinct_address_ranges = pairs.len();
        receipt.device_after_bytes = device.currentAllocatedSize();
        budget(
            receipt.device_after_bytes,
            remaining_session(sizes),
            device_limit,
        )?;
        Ok(Self {
            pairs,
            pipeline: consume,
            receipt,
        })
    }

    pub(super) fn encode(
        &self,
        encoder: &Encoder,
        layer: usize,
        input: &RawBuffer,
        output: &RawBuffer,
        status: &RawBuffer,
    ) {
        encode_pipeline(
            encoder,
            &self.pipeline,
            128,
            &[&self.pairs[layer].raw, input, output, status],
            &[128, HIDDEN as u32, INTERMEDIATE as u32, 0],
            [INTERMEDIATE / 32, 4],
        );
    }

    pub(super) fn encode_prepacked(
        &self,
        pipeline: &Prepacked,
        encoder: &Encoder,
        layer: usize,
        [input, packed, output, status]: [&RawBuffer; 4],
    ) {
        let shape = [ROWS as u32, HIDDEN as u32, INTERMEDIATE as u32, 0];
        // The serial compute encoder orders both dispatches on tracked scratch.
        // Every visit overwrites all PACKED_BYTES before its consumer starts.
        encode_pipeline(
            encoder,
            &pipeline.pack,
            THREADS,
            &[input, packed],
            &shape,
            [
                TOKEN_BLOCK * HIDDEN / (FRAGMENT_WORDS * THREADS),
                ROWS / TOKEN_BLOCK,
            ],
        );
        encode_pipeline(
            encoder,
            &pipeline.consume,
            THREADS,
            &[&self.pairs[layer].raw, packed, output, status],
            &shape,
            [INTERMEDIATE / 32, ROWS / 64],
        );
    }
}
