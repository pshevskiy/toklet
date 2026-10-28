//! Isolated, test-only qualification before model admission. The future model
//! owns Operation; sessions supply borrowed mutable buffers and completion.
use super::{
    Error, Result,
    memory::{Buffer, Gpu},
    profile::Command,
};
use crate::cpu::Llama3Scaling;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::{mem::size_of, ptr::NonNull};

const GUARD: usize = 7;
const SENTINEL: u32 = 0x5a17_0031;

/// Immutable optional pipeline and one resident inverse-frequency allocation.
/// Ordinary Gpu construction/pipelines and the ordinary rotary shader stay
/// unchanged. Compiler/context policy is separate from these owned bytes.
struct Operation {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    inverse: Buffer<u32>,
    dimension: usize,
    width: usize,
}

fn frequency(theta: f32, dimension: usize, pair: usize, scaling: Llama3Scaling) -> Result<u32> {
    let ordinary = f64::from(theta).powf(-2.0 * pair as f64 / dimension as f64);
    let inverse = scaling
        .inverse_frequency(ordinary)
        .map_err(|_| Error::Options("invalid scaled rotary frequency"))? as f32;
    if !inverse.is_finite() || inverse <= 0.0 {
        return Err(Error::Options(
            "scaled rotary frequency exceeds positive FP32",
        ));
    }
    Ok(inverse.to_bits())
}
fn owned_bytes(dimension: usize) -> Result<usize> {
    if dimension == 0 || !dimension.is_multiple_of(2) || dimension > u32::MAX as usize {
        return Err(Error::Options(
            "scaled rotary requires even UInt32 dimension",
        ));
    }
    (dimension / 2)
        .checked_add(GUARD)
        .and_then(|n| n.checked_mul(4))
        .filter(|&n| n <= isize::MAX as usize)
        .and_then(|n| n.checked_add(size_of::<Operation>()))
        .ok_or(Error::Options("scaled rotary allocation overflow"))
}

impl Operation {
    fn new(
        gpu: &Gpu,
        dimension: usize,
        theta: f32,
        scaling: Llama3Scaling,
        budget: usize,
    ) -> Result<Self> {
        let required = owned_bytes(dimension)?;
        if required > budget {
            return Err(Error::Budget {
                required,
                limit: budget,
            });
        }
        super::memory::validate_buffer_bytes(
            required - size_of::<Operation>(),
            gpu.device.maxBufferLength(),
        )?;
        if !theta.is_finite() || theta <= 0.0 {
            return Err(Error::Options("scaled rotary theta"));
        }
        // Validate conversion before allocating/compiling. Recompute directly
        // into shared immutable storage below, avoiding a temporary host table.
        for pair in 0..dimension / 2 {
            frequency(theta, dimension, pair, scaling)?;
        }
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version2_3);
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        if options.languageVersion() != MTLLanguageVersion::Version2_3
            || options.mathMode() != MTLMathMode::Safe
            || options.mathFloatingPointFunctions() != MTLMathFloatingPointFunctions::Precise
        {
            return Err(Error::Unsupported("scaled rotary precise math unavailable"));
        }
        let library = gpu
            .device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("llama3_rotary.metal")),
                Some(&options),
            )
            .map_err(|e| Error::Native(e.to_string()))?;
        let function = library
            .newFunctionWithName(&NSString::from_str("llama3_rotary"))
            .ok_or_else(|| Error::Native("scaled rotary kernel absent".into()))?;
        let pipeline = gpu
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| Error::Native(e.to_string()))?;
        let width = pipeline.threadExecutionWidth();
        if width == 0
            || width > pipeline.maxTotalThreadsPerThreadgroup()
            || width > u32::MAX as usize
        {
            return Err(Error::Unsupported("scaled rotary launch width unavailable"));
        }
        let mut inverse = Buffer::new(gpu, dimension / 2 + GUARD)?;
        inverse.with_mut(|out| -> Result<()> {
            out.fill(SENTINEL);
            for (pair, word) in out[..dimension / 2].iter_mut().enumerate() {
                *word = frequency(theta, dimension, pair, scaling)?;
            }
            Ok(())
        })?;
        Ok(Self {
            pipeline,
            inverse,
            dimension,
            width,
        })
    }

    fn encode(&self, command: &Command<'_>, invocation: Invocation<'_>) -> Result<()> {
        let n = extent(
            invocation.rows,
            invocation.heads,
            self.dimension,
            invocation.offset,
        )?;
        if invocation.input.len < n + GUARD
            || invocation.output.len < n + GUARD
            || invocation.status.len < 1 + GUARD
            || std::ptr::eq(&*invocation.input.raw, &*invocation.output.raw)
        {
            return Err(Error::Options("scaled rotary buffer extent/alias"));
        }
        let groups = (n / 2).div_ceil(self.width);
        if groups
            .checked_mul(self.width)
            .is_none_or(|n| n > u32::MAX as usize)
        {
            return Err(Error::Options("scaled rotary rounded dispatch overflow"));
        }
        let (index, encoder) = if let Some(trace) = command.trace {
            let (index, encoder) = trace.encoder(command.raw, "llama3_rotary")?;
            (Some(index), encoder)
        } else {
            (
                None,
                command
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::Native("scaled rotary encoder absent".into()))?,
            )
        };
        encoder.setComputePipelineState(&self.pipeline);
        let shape = [
            invocation.rows as u32,
            invocation.heads as u32,
            self.dimension as u32,
            invocation.offset as u32,
        ];
        // SAFETY: closed private ABI with initialized same-device disjoint
        // buffers. Checked products bound every UInt32 intermediate and index;
        // contiguous Half operands and FP32 words have their native alignment.
        // Stack args are copied immediately. The retained command keeps all
        // resources live; callers exclude CPU access until checked completion.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&invocation.input.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.inverse.raw), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&invocation.output.raw), 0, 2);
            encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), size_of::<[u32; 4]>(), 3);
            encoder.setBuffer_offset_atIndex(Some(&invocation.status.raw), 0, 4);
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

struct Invocation<'a> {
    input: &'a Buffer<u16>,
    output: &'a Buffer<u16>,
    status: &'a Buffer<u32>,
    rows: usize,
    heads: usize,
    offset: usize,
}
fn extent(rows: usize, heads: usize, dimension: usize, offset: usize) -> Result<usize> {
    if rows == 0
        || heads == 0
        || dimension == 0
        || !dimension.is_multiple_of(2)
        || offset.checked_add(rows).is_none_or(|end| end > 512)
    {
        return Err(Error::Options(
            "scaled rotary requires absolute positions0..511",
        ));
    }
    rows.checked_mul(heads)
        .and_then(|n| n.checked_mul(dimension))
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or(Error::Options("scaled rotary UInt32 extent overflow"))
}

mod tests;
