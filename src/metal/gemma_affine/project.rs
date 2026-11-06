//! Shared exact command/completion path for legacy and separate projection scratch.
use super::*;

pub(super) struct Buffers<'a> {
    pub input: &'a mut Buffer<u32>,
    pub output: &'a mut Buffer<u32>,
    pub status: &'a mut Buffer<u32>,
    pub pending: &'a mut Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}
#[derive(Clone, Copy)]
struct ImmutableBuffer<'a> {
    buffer: &'a Buffer<u32>,
    words: usize,
}

pub(super) struct Matrix<'a> {
    rows: usize,
    columns: usize,
    bits: u8,
    coefficients: ImmutableBuffer<'a>,
    affine_metadata: Option<[ImmutableBuffer<'a>; 2]>,
}

impl<'a> Matrix<'a> {
    pub(super) fn affine(view: WeightView<'a>) -> Self {
        let metadata = view.layout.metadata_bytes().div_ceil(4);
        Self {
            rows: view.layout.rows(),
            columns: view.layout.columns(),
            bits: view.layout.bits(),
            coefficients: ImmutableBuffer {
                buffer: view.codes,
                words: view.layout.weight_bytes().div_ceil(4),
            },
            affine_metadata: Some([
                ImmutableBuffer {
                    buffer: view.scales,
                    words: metadata,
                },
                ImmutableBuffer {
                    buffer: view.biases,
                    words: metadata,
                },
            ]),
        }
    }

    pub(super) fn bf16(rows: usize, columns: usize, coefficients: &'a Buffer<u32>) -> Self {
        Self {
            rows,
            columns,
            bits: 16,
            coefficients: ImmutableBuffer {
                buffer: coefficients,
                words: rows * columns / 2,
            },
            affine_metadata: None,
        }
    }
}

// Private callers validate same device, original layout, finite/admitted input,
// active dimensions and all actual scratch capacities before staging. They own
// sticky poison until the complete parent transaction publishes successfully.
pub(super) fn run(
    context: &Context,
    weights: Matrix<'_>,
    buffers: Buffers<'_>,
    input: &[f32],
    output: &mut [f32],
    tokens: usize,
) -> Result<()> {
    let (pipeline, width) = context.projection_pipeline(weights.bits)?;
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
        encoder.setComputePipelineState(pipeline);
        let shape = [tokens as u32, weights.columns as u32, weights.rows as u32];
        // SAFETY: private fixed ABI; all same-device u32 buffers initialized,
        // aligned, disjoint, admitted with guards and validated dimensions.
        // Metadata's padded final word allows aligned ushort reads. Parameter
        // bytes copy immediately. Tracked hazards and retained command
        // resources protect lifetimes; CPU views wait for checked completion.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&weights.coefficients.buffer.raw), 0, 0);
            if let Some([scales, biases]) = weights.affine_metadata {
                encoder.setBuffer_offset_atIndex(Some(&scales.buffer.raw), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&biases.buffer.raw), 0, 2);
            }
            encoder.setBuffer_offset_atIndex(Some(&buffers.input.raw), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&buffers.output.raw), 0, 4);
            encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), size_of::<[u32; 3]>(), 5);
            encoder.setBuffer_offset_atIndex(Some(&buffers.status.raw), 0, 6);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: tokens * weights.rows,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
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
        let guarded = |part: ImmutableBuffer<'_>| part.buffer.with_slice(|s| tail(s, part.words));
        if !guarded(weights.coefficients)
            || weights
                .affine_metadata
                .is_some_and(|parts| !parts.into_iter().all(guarded))
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
