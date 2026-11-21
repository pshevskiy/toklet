use super::super::{
    Error, Result,
    memory::{Buffer, system_device},
    mlx::precise_library_on,
};
use objc2::{
    rc::Retained,
    runtime::{NSObjectProtocol, ProtocolObject},
    sel,
};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::ptr::NonNull;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub(super) type RawBuffer = ProtocolObject<dyn MTLBuffer>;
pub(super) type Encoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

#[derive(Clone, Copy)]
pub(super) enum Op {
    Embedding,
    Vector,
    Tiled,
    Rms,
    RotateCache,
    Attention,
    Add,
    Swiglu,
    Copy,
    Greedy,
    FfnM32,
    GateUpCompact,
    DownK64,
}

impl Op {
    const ALL: [Self; 10] = [
        Self::Embedding,
        Self::Vector,
        Self::Tiled,
        Self::Rms,
        Self::RotateCache,
        Self::Attention,
        Self::Add,
        Self::Swiglu,
        Self::Copy,
        Self::Greedy,
    ];

    fn name(self) -> &'static str {
        [
            "nanbeige_embedding",
            "nanbeige_vector",
            "nanbeige_tiled",
            "nanbeige_rms",
            "nanbeige_rotate_cache",
            "nanbeige_attention",
            "nanbeige_add",
            "nanbeige_swiglu",
            "nanbeige_copy",
            "nanbeige_greedy",
            "nanbeige_ffn_m32",
            "nanbeige_ffn_compact",
            "nanbeige_down_k64",
        ][self as usize]
    }

    fn group_threads(self) -> usize {
        match self {
            Self::Vector => 32,
            Self::Attention | Self::GateUpCompact | Self::DownK64 => 128,
            Self::Tiled | Self::Rms | Self::Greedy | Self::FfnM32 => 256,
            _ => 32,
        }
    }
}

pub(super) struct Context {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: [Pipeline; Op::ALL.len()],
    ffn_m32: Option<Pipeline>,
    gate_up_compact: Option<Pipeline>,
    down_k64: Option<Pipeline>,
}

// K64 loads four adjacent BF16 words per vector. Both admitted K strides
// and all tile offsets are multiples of four words.
pub(super) fn k64_input_aligned(input: &RawBuffer) -> bool {
    if !input.respondsToSelector(sel!(gpuAddress)) {
        return false;
    }
    let start = input.gpuAddress();
    start != 0 && start.is_multiple_of(8)
}

impl Context {
    pub fn new() -> Result<Self> {
        let device = system_device()?;
        if !device.hasUnifiedMemory() || !device.supportsFamily(MTLGPUFamily::Apple7) {
            return Err(Error::Unsupported(
                "Nanbeige Metal requires Apple7 or later with unified memory",
            ));
        }
        let library = precise_library_on(&device, include_str!("../nanbeige.metal"))?;
        let mut pipelines = Vec::with_capacity(Op::ALL.len());
        for op in Op::ALL {
            let function = library
                .newFunctionWithName(&NSString::from_str(op.name()))
                .ok_or_else(|| Error::Native(format!("missing {} Metal function", op.name())))?;
            let pipeline = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| Error::Native(e.to_string()))?;
            if pipeline.threadExecutionWidth() != 32
                || pipeline.maxTotalThreadsPerThreadgroup() < op.group_threads()
            {
                return Err(Error::Unsupported(
                    "Nanbeige shader launch geometry unavailable",
                ));
            }
            pipelines.push(pipeline);
        }
        // Prepare the Mac prefill specialization while preserving the
        // released library and its admitted devices.
        let ffn_m32 =
            if cfg!(target_os = "macos") && device.maxThreadgroupMemoryLength() >= 12 * 1024 {
                (|| {
                    let library = precise_library_on(
                        &device,
                        concat!(
                            include_str!("../nanbeige.metal"),
                            "\n",
                            include_str!("ffn_m32.metal"),
                        ),
                    )
                    .ok()?;
                    let function =
                        library.newFunctionWithName(&NSString::from_str(Op::FfnM32.name()))?;
                    let pipeline = device
                        .newComputePipelineStateWithFunction_error(&function)
                        .ok()?;
                    (pipeline.threadExecutionWidth() == 32
                        && pipeline.maxTotalThreadsPerThreadgroup() >= Op::FfnM32.group_threads()
                        && pipeline.staticThreadgroupMemoryLength()
                            <= device.maxThreadgroupMemoryLength())
                    .then_some(pipeline)
                })()
            } else {
                None
            };
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| Error::Native("Nanbeige queue absent".into()))?;
        let mut context = Self {
            device,
            queue,
            pipelines: pipelines
                .try_into()
                .map_err(|_| Error::Options("Nanbeige pipeline count"))?,
            ffn_m32,
            gate_up_compact: None,
            down_k64: None,
        };
        if cfg!(target_os = "macos") {
            context.gate_up_compact = context.prepare_compact(
                Op::GateUpCompact,
                concat!(
                    include_str!("../nanbeige.metal"),
                    "\n",
                    include_str!("gate_up_compact.metal"),
                ),
                5 * 1024,
            );
            context.down_k64 = context.prepare_compact(
                Op::DownK64,
                concat!(
                    include_str!("../nanbeige.metal"),
                    "\n",
                    include_str!("down_k64.metal"),
                ),
                9 * 1024,
            );
        }
        Ok(context)
    }

    pub fn has_ffn_m32(&self) -> bool {
        self.ffn_m32.is_some()
    }

    pub fn has_gate_up_compact(&self) -> bool {
        self.gate_up_compact.is_some()
    }

    pub fn has_down_k64(&self) -> bool {
        self.down_k64.is_some()
    }

    fn prepare_compact(&self, op: Op, source: &str, scratch_bytes: usize) -> Option<Pipeline> {
        if self.device.maxThreadgroupMemoryLength() < scratch_bytes {
            return None;
        }
        let library = precise_library_on(&self.device, source).ok()?;
        let function = library.newFunctionWithName(&NSString::from_str(op.name()))?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .ok()?;
        if pipeline.threadExecutionWidth() != 32
            || pipeline.maxTotalThreadsPerThreadgroup() < op.group_threads()
            || pipeline.staticThreadgroupMemoryLength() > self.device.maxThreadgroupMemoryLength()
        {
            return None;
        }
        self.compact_basis(op, &pipeline)?.then_some(pipeline)
    }

    /// Metal does not specify fragment lane layout. Test the compiled pipeline
    /// with signed one-hot weights spanning both affine groups in one tile.
    /// Each expected BF16 word is indexed directly from the distinct inputs.
    fn compact_basis(&self, op: Op, pipeline: &Pipeline) -> Option<bool> {
        const M: usize = 32;
        const N: usize = 32;
        const K: usize = 128;
        let column = |n| (37 * n + 11) % K;
        let negative = |n, group| (n + group) % 2 == 1;
        let mut packed = Buffer::<u32>::on_device(&self.device, N * K / 8).ok()?;
        let mut scales = Buffer::<u16>::on_device(&self.device, N * K / 64).ok()?;
        let biases = Buffer::<u16>::on_device(&self.device, N * K / 64).ok()?;
        let mut input = Buffer::<u16>::on_device(&self.device, M * K).ok()?;
        let mut output = Buffer::<u16>::on_device(&self.device, (M + 1) * N).ok()?;
        let status = Buffer::<u32>::on_device(&self.device, 1).ok()?;
        if matches!(op, Op::DownK64) && !k64_input_aligned(&input.raw) {
            return Some(false);
        }
        packed.with_mut(|words| {
            for n in 0..N {
                let k = column(n);
                words[n * K / 8 + k / 8] = 1 << (k % 8 * 4);
            }
        });
        scales.with_mut(|words| {
            for (i, word) in words.iter_mut().enumerate() {
                *word = if negative(i / 2, i % 2) {
                    0xbf80
                } else {
                    0x3f80
                };
            }
        });
        input.with_mut(|words| {
            for (i, word) in words.iter_mut().enumerate() {
                *word = 0x2000 + i as u16;
            }
        });
        output.with_mut(|words| words.fill(u16::MAX));
        let command = self.command().ok()?;
        let encoder = command.computeCommandEncoder()?;
        encode_pipeline(
            &encoder,
            pipeline,
            op.group_threads(),
            &[
                &packed.raw,
                &scales.raw,
                &biases.raw,
                &input.raw,
                &output.raw,
                &status.raw,
            ],
            &[M as u32, K as u32, N as u32, 0],
            [1, 1],
        );
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed
            || command.error().is_some()
            || status.with_slice(|words| words[0]) != 0
        {
            return Some(false);
        }
        Some(output.with_slice(|words| {
            words[M * N..].iter().all(|&word| word == u16::MAX)
                && words[..M * N].iter().enumerate().all(|(i, &word)| {
                    let n = i % N;
                    let k = column(n);
                    let signed = if negative(n, k / 64) { 0x8000 } else { 0 };
                    word == (signed | (0x2000 + (i / N * K + k) as u16))
                })
        }))
    }

    pub fn command(&self) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Error::Native("Nanbeige command absent".into()))?;
        if !command.retainedReferences() {
            return Err(Error::Unsupported(
                "Nanbeige command requires retained references",
            ));
        }
        Ok(command)
    }

    /// All dimensions and buffer extents come from the fixed accepted plan or
    /// the checked chunk. The ordinary compute encoder is serial. Its ordered
    /// dispatches and the tracked buffer owners carry each dependency forward.
    pub fn encode(
        &self,
        encoder: &Encoder,
        op: Op,
        buffers: &[&RawBuffer],
        constants: &[u32],
        groups: [usize; 2],
    ) {
        let pipeline = match op {
            Op::FfnM32 => self.ffn_m32.as_ref().expect("prepared M32 FFN pipeline"),
            Op::GateUpCompact => self
                .gate_up_compact
                .as_ref()
                .expect("prepared compact gate/up pipeline"),
            Op::DownK64 => self.down_k64.as_ref().expect("prepared K64 Down pipeline"),
            _ => &self.pipelines[op as usize],
        };
        encode_pipeline(
            encoder,
            pipeline,
            op.group_threads(),
            buffers,
            constants,
            groups,
        );
    }
}

fn encode_pipeline(
    encoder: &Encoder,
    pipeline: &Pipeline,
    threads: usize,
    buffers: &[&RawBuffer],
    constants: &[u32],
    groups: [usize; 2],
) {
    encoder.setComputePipelineState(pipeline);
    // SAFETY: the private graph and startup basis bind same-device initialized,
    // disjoint owners except explicitly in-place pointwise operations. Indices
    // match the shader signature. Metal copies at most 24 bytes of plain constants.
    unsafe {
        for (i, buffer) in buffers.iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(buffer), 0, i);
        }
        encoder.setBytes_length_atIndex(
            NonNull::from(constants).cast(),
            std::mem::size_of_val(constants),
            buffers.len(),
        );
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: groups[0],
            height: groups[1],
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;
    use objc2::rc::autoreleasepool;

    #[test]
    #[ignore = "Explicit Apple GPU check of the Nanbeige projection arithmetic"]
    fn projection_rounding_midpoint() {
        autoreleasepool(|_| {
            let context = Context::new().unwrap();
            let words = |n| Buffer::<u16>::on_device(&context.device, n).unwrap();
            let mut packed = Buffer::<u32>::on_device(&context.device, 32 * 512 / 8).unwrap();
            let mut scales = words(32 * 512 / 64);
            let mut biases = words(32 * 512 / 64);
            let mut input = words(128 * 512);
            let vector_output = words(32);
            let tiled_output = words(128 * 32);
            let status = Buffer::<u32>::on_device(&context.device, 1).unwrap();
            let b = |x| bf16::from_f32(x).to_bits();
            packed.with_mut(|v| v[0] = 1);
            scales.with_mut(|v| v.fill(b(1.)));
            biases.with_mut(|v| {
                v[0] = b(1. / 256.);
                v[1] = b(-1.);
            });
            input.with_mut(|v| {
                for row in v.chunks_exact_mut(512) {
                    row[0] = b(1.);
                    row[64] = b(1.);
                }
            });

            let command = context.command().unwrap();
            let encoder = command.computeCommandEncoder().unwrap();
            for (op, shape, groups, output) in [
                (Op::Vector, [1, 512, 32, 0], [8, 1], &vector_output),
                (Op::Tiled, [128, 512, 32, 0], [1, 8], &tiled_output),
            ] {
                context.encode(
                    &encoder,
                    op,
                    &[
                        &packed.raw,
                        &scales.raw,
                        &biases.raw,
                        &input.raw,
                        &output.raw,
                        &status.raw,
                    ],
                    &shape,
                    groups,
                );
            }
            encoder.endEncoding();
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            assert_eq!(status.with_slice(|v| v[0]), 0);
            assert_eq!(vector_output.with_slice(|v| v[0]), b(1. / 256.));
            tiled_output.with_slice(|v| assert!(v.iter().all(|&x| x == 0)));

            packed.with_mut(|v| v[0] = 3);
            scales.with_mut(|v| v[0] = b(129. / 128.));
            biases.with_mut(|v| v[0] = b(-3.));
            let tokens = Buffer::<u32>::on_device(&context.device, 1).unwrap();
            let embedding = words(512);
            let command = context.command().unwrap();
            let encoder = command.computeCommandEncoder().unwrap();
            context.encode(
                &encoder,
                Op::Embedding,
                &[
                    &packed.raw,
                    &scales.raw,
                    &biases.raw,
                    &tokens.raw,
                    &embedding.raw,
                    &status.raw,
                ],
                &[1, 512, 32, 0],
                [16, 1],
            );
            encoder.endEncoding();
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            assert_eq!(status.with_slice(|v| v[0]), 0);
            assert_eq!(embedding.with_slice(|v| v[0]), b(3. / 128.));
        });
    }
}
