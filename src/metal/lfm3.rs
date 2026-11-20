//! Fixed exact BF16 LFM3 Metal execution.
use super::{
    Buffer, Gpu, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLDevice, Retained,
};
use half::bf16;

const I: usize = crate::model::lfm3::HIDDEN;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandEncoder, MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLLanguageVersion, MTLLibrary, MTLMathFloatingPointFunctions, MTLMathMode, MTLSize,
};
use std::{mem::size_of, ptr::NonNull, time::Instant};

const SENTINEL: u16 = 0x7fc1;
const STATUS_GUARD: u32 = 0x3fac715e;
const ROUTE_GUARD: u32 = u32::MAX;

struct Kernels {
    admission: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    projection: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl Kernels {
    fn with_shader(gpu: &Gpu, shader: &str) -> Option<Self> {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version2_3);
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        if options.languageVersion() != MTLLanguageVersion::Version2_3
            || options.mathMode() != MTLMathMode::Safe
            || options.mathFloatingPointFunctions() != MTLMathFloatingPointFunctions::Precise
            || !gpu.device.hasUnifiedMemory()
            || gpu.device.maxThreadsPerThreadgroup().width < 32
        {
            return None;
        }
        let lib = gpu
            .device
            .newLibraryWithSource_options_error(
                &NSString::from_str(
                    &shader.replace("__LFM3_DYADIC__", include_str!("lfm3/dyadic_shared.h")),
                ),
                Some(&options),
            )
            .ok()?;
        let pipeline = |name: &str| {
            let f = lib.newFunctionWithName(&NSString::from_str(name))?;
            gpu.device
                .newComputePipelineStateWithFunction_error(&f)
                .ok()
        };
        let admission = pipeline("lfm3_admit")?;
        let projection = pipeline("lfm3_ordered")?;
        if [&admission, &projection]
            .iter()
            .any(|pso| pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 32)
        {
            return None;
        }
        Some(Self {
            admission,
            projection,
        })
    }
}

mod attention_workspace;
mod conv_workspace;
mod down;
mod output_workspace;
mod pair;
mod projection_tile;
mod qkv_workspace;
mod shared_mlp;
mod shared_tied;

pub(crate) use shared_mlp::SharedMlpWorkspace;
