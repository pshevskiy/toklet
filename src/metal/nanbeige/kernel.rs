use super::super::{Error, Result, memory::system_device, mlx::precise_library_on};
use objc2::{rc::Retained, runtime::ProtocolObject};
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
        ][self as usize]
    }

    fn group_threads(self) -> usize {
        match self {
            Self::Vector => 32,
            Self::Attention => 128,
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
        Ok(Self {
            device,
            queue,
            pipelines: pipelines
                .try_into()
                .map_err(|_| Error::Options("Nanbeige pipeline count"))?,
            ffn_m32,
        })
    }

    pub fn has_ffn_m32(&self) -> bool {
        self.ffn_m32.is_some()
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
            _ => &self.pipelines[op as usize],
        };
        encoder.setComputePipelineState(pipeline);
        // SAFETY: the private graph binds same-device initialized, disjoint
        // owners except explicitly in-place pointwise operations. Indices
        // match the corresponding shader signature, and Metal copies these
        // at most 24 bytes of plain constants before the call returns.
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
                width: op.group_threads(),
                height: 1,
                depth: 1,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::memory::Buffer;
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
