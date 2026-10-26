//! Shared exact command/completion path for legacy and separate projection scratch.
use super::*;

pub(super) struct Buffers<'a> {
    pub input: &'a mut Buffer<u32>,
    pub output: &'a mut Buffer<u32>,
    pub status: &'a mut Buffer<u32>,
    pub pending: &'a mut Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}
// Private callers validate same device, original layout, finite/admitted input,
// active dimensions and all actual scratch capacities before staging. They own
// sticky poison until the complete parent transaction publishes successfully.
pub(super) fn run(
    context: &Context,
    weights: WeightView<'_>,
    buffers: Buffers<'_>,
    input: &[f32],
    output: &mut [f32],
    tokens: usize,
) -> Result<()> {
    autoreleasepool(|_| {
        buffers.input.with_mut(|out| {
            out.fill(SENTINEL);
            for (y, x) in out.iter_mut().zip(input) {
                *y = x.to_bits();
            }
        });
        buffers.output.with_mut(|out| out.fill(SENTINEL));
        buffers.status.with_mut(|out| {
            out.fill(SENTINEL);
            out[0] = 0;
        });
        let raw = context.gpu.command()?;
        let encoder = raw
            .computeCommandEncoder()
            .ok_or_else(|| Error::Native("Gemma affine encoder absent".into()))?;
        encoder.setComputePipelineState(&context.pipeline);
        let shape = [
            tokens as u32,
            weights.layout.columns() as u32,
            weights.layout.rows() as u32,
        ];
        // SAFETY: private fixed ABI; all same-device u32 buffers initialized,
        // aligned, disjoint, admitted with guards and validated dimensions.
        // Metadata's padded final word allows aligned ushort reads. Parameter
        // bytes copy immediately. Tracked hazards and retained command
        // resources protect lifetimes; CPU views wait for checked completion.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&weights.codes.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&weights.scales.raw), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&weights.biases.raw), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&buffers.input.raw), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&buffers.output.raw), 0, 4);
            encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), size_of::<[u32; 3]>(), 5);
            encoder.setBuffer_offset_atIndex(Some(&buffers.status.raw), 0, 6);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: tokens * weights.layout.rows(),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: context.width,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        *buffers.pending = Some(raw);
        let command = buffers.pending.as_ref().unwrap();
        command.commit();
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Completion);
        }
        let tail = |words: &[u32], active| words[active..].iter().all(|&v| v == SENTINEL);
        if !weights
            .codes
            .with_slice(|s| tail(s, weights.layout.weight_bytes().div_ceil(4)))
            || !weights
                .scales
                .with_slice(|s| tail(s, weights.layout.metadata_bytes().div_ceil(4)))
            || !weights
                .biases
                .with_slice(|s| tail(s, weights.layout.metadata_bytes().div_ceil(4)))
            || !buffers.input.with_slice(|s| tail(s, input.len()))
            || !buffers.output.with_slice(|s| tail(s, output.len()))
            || !buffers.status.with_slice(|s| tail(s, 1))
        {
            return Err(Error::Options(
                "Gemma affine guard/inactive capacity changed",
            ));
        }
        if buffers.status.with_slice(|s| s[0]) != 0
            || !buffers.output.with_slice(|s| {
                s[..output.len()]
                    .iter()
                    .all(|&v| f32::from_bits(v).is_finite())
            })
        {
            return Err(Error::Nonfinite);
        }
        buffers.output.with_slice(|s| {
            for (out, &bits) in output.iter_mut().zip(s) {
                *out = f32::from_bits(bits);
            }
        });
        *buffers.pending = None;
        Ok(())
    })
}
