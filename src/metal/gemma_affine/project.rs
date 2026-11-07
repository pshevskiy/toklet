//! Shared exact command/completion path for legacy and separate projection scratch.
use super::*;
use crate::{metal::profile::Trace, mlx::gemma4::bf16::execution::OperationId};

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct TileAttribution {
    pub operation: OperationId,
    pub source_matrix_slot: usize,
    pub tile_index: usize,
    pub source_first_row: usize,
    pub source_rows: usize,
    pub input_rows: usize,
    pub output_features: usize,
    pub input_features: usize,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct TileRecord {
    pub counter_slot: usize,
    pub attribution: TileAttribution,
    pub host_staging_seconds: f64,
    pub host_encoding_seconds: f64,
    pub host_commit_seconds: f64,
    pub host_wait_seconds: f64,
    pub host_validation_and_copy_seconds: f64,
    pub dispatches_encoded: usize,
    pub commands_submitted: usize,
    pub host_wait_calls: usize,
    pub input_copy_bytes: usize,
    pub output_copy_bytes: usize,
}

pub(super) struct Observation<'a> {
    pub trace: &'a Trace,
    pub records: &'a mut Vec<TileRecord>,
    pub capacity: usize,
    pub overflowed: &'a mut bool,
    pub attribution: TileAttribution,
}

// OR cannot cancel differences, so zero is equivalent to every tail word
// matching the sentinel. Integer reduction permits compiler vectorization while
// preserving complete guard/inactive-capacity coverage and checked slice bounds.
fn sentinel_tail_unchanged(words: &[u32], active: usize) -> bool {
    words[active..]
        .iter()
        .fold(0_u32, |different, &word| different | (word ^ SENTINEL))
        == 0
}

fn elapsed(observation: &Option<Observation<'_>>) -> f64 {
    observation.as_ref().map_or(0., |o| o.trace.elapsed())
}

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
    execute(context, weights, buffers, input, output, tokens, None)
}

#[cfg(target_os = "macos")]
pub(super) fn run_observed(
    context: &Context,
    weights: Matrix<'_>,
    buffers: Buffers<'_>,
    input: &[f32],
    output: &mut [f32],
    tokens: usize,
    observation: Option<Observation<'_>>,
) -> Result<()> {
    execute(
        context,
        weights,
        buffers,
        input,
        output,
        tokens,
        observation,
    )
}

fn execute(
    context: &Context,
    weights: Matrix<'_>,
    buffers: Buffers<'_>,
    input: &[f32],
    output: &mut [f32],
    tokens: usize,
    mut observation: Option<Observation<'_>>,
) -> Result<()> {
    if let Some(o) = &observation {
        let a = o.attribution;
        if weights.bits != 16
            || a.input_rows != tokens
            || a.input_features != weights.columns
            || a.source_rows != weights.rows
            || a.source_first_row
                .checked_add(a.source_rows)
                .is_none_or(|end| end > a.output_features)
            || input.len()
                != tokens
                    .checked_mul(weights.columns)
                    .ok_or(Error::Options("BF16 profile input shape overflow"))?
            || output.len()
                != tokens
                    .checked_mul(weights.rows)
                    .ok_or(Error::Options("BF16 profile output shape overflow"))?
        {
            return Err(Error::Options("BF16 profile attribution shape differs"));
        }
    }
    let (pipeline, width) = context.projection_pipeline(weights.bits)?;
    autoreleasepool(|_| {
        let staging_start = elapsed(&observation);
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
        let staging_end = elapsed(&observation);
        let raw = context.gpu.command()?;
        let (encoder, slot) = if let Some(observation) = observation.as_mut() {
            if observation.records.len() >= observation.capacity {
                *observation.overflowed = true;
                return Err(Error::Options("BF16 profile event capacity exceeded"));
            }
            let (index, encoder) = observation.trace.encoder(&raw, "gemma_bf16_product")?;
            if index != observation.records.len() || index >= observation.records.capacity() {
                return Err(Error::Options("BF16 profile slot attribution differs"));
            }
            // Register role/layer/tile/shape now, before dispatch encoding. No
            // later graph-order reconstruction and no vector growth are used.
            observation.records.push(TileRecord {
                counter_slot: index,
                attribution: observation.attribution,
                host_staging_seconds: staging_end - staging_start,
                host_encoding_seconds: 0.,
                host_commit_seconds: 0.,
                host_wait_seconds: 0.,
                host_validation_and_copy_seconds: 0.,
                dispatches_encoded: 0,
                commands_submitted: 0,
                host_wait_calls: 0,
                input_copy_bytes: input
                    .len()
                    .checked_mul(4)
                    .ok_or(Error::Options("BF16 profile input bytes overflow"))?,
                output_copy_bytes: 0,
            });
            (encoder, Some(index))
        } else {
            (
                raw.computeCommandEncoder()
                    .ok_or_else(|| Error::Native("Gemma affine encoder absent".into()))?,
                None,
            )
        };
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
        if let (Some(o), Some(index)) = (observation.as_mut(), slot) {
            o.records[index].dispatches_encoded += 1;
        }
        encoder.endEncoding();
        if let (Some(o), Some(index)) = (observation.as_mut(), slot) {
            o.trace.finish_encoder(index);
            o.records[index].host_encoding_seconds = o.trace.elapsed() - staging_end;
        }
        *buffers.pending = Some(raw);
        let command = buffers.pending.as_ref().unwrap();
        let commit_start = elapsed(&observation);
        command.commit();
        let commit_end = elapsed(&observation);
        if let (Some(o), Some(index)) = (observation.as_mut(), slot) {
            o.records[index].commands_submitted += 1;
            o.records[index].host_wait_calls += 1;
            o.records[index].host_commit_seconds = commit_end - commit_start;
        }
        command.waitUntilCompleted();
        let wait_end = elapsed(&observation);
        if let (Some(o), Some(index)) = (observation.as_mut(), slot) {
            o.records[index].host_wait_seconds = wait_end - commit_end;
        }
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Completion);
        }
        let guarded = |part: ImmutableBuffer<'_>| {
            part.buffer
                .with_slice(|s| sentinel_tail_unchanged(s, part.words))
        };
        if !guarded(weights.coefficients)
            || weights
                .affine_metadata
                .is_some_and(|parts| !parts.into_iter().all(guarded))
            || !buffers
                .input
                .with_slice(|s| sentinel_tail_unchanged(s, input.len()))
            || !buffers
                .output
                .with_slice(|s| sentinel_tail_unchanged(s, output.len()))
            || !buffers.status.with_slice(|s| sentinel_tail_unchanged(s, 1))
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
        if let (Some(o), Some(index)) = (observation.as_mut(), slot) {
            o.records[index].host_validation_and_copy_seconds = o.trace.elapsed() - wait_end;
            o.records[index].output_copy_bytes = output
                .len()
                .checked_mul(4)
                .ok_or(Error::Options("BF16 profile output bytes overflow"))?;
        }
        *buffers.pending = None;
        Ok(())
    })
}

#[cfg(test)]
mod guard_tests {
    use super::{SENTINEL, sentinel_tail_unchanged};
    fn scalar_reference(words: &[u32], active: usize) -> bool {
        for &word in &words[active..] {
            if word != SENTINEL {
                return false;
            }
        }
        true
    }
    #[test]
    fn all_bits_positions_and_active_boundaries_match_scalar_reference() {
        for len in [
            0, 1, 2, 3, 4, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257,
        ] {
            for active in 0..=len {
                let mut words = vec![SENTINEL; len];
                assert!(sentinel_tail_unchanged(&words, active));
                for index in 0..len {
                    for bit in 0..32 {
                        words[index] ^= 1 << bit;
                        assert_eq!(sentinel_tail_unchanged(&words, active), index < active);
                        assert_eq!(
                            sentinel_tail_unchanged(&words, active),
                            scalar_reference(&words, active)
                        );
                        words[index] ^= 1 << bit;
                    }
                }
            }
        }
    }
    #[test]
    fn identical_corruptions_in_two_tail_words_never_cancel() {
        for active in 0..32 {
            for bit in 0..32 {
                let mut words = [SENTINEL; 65];
                words[active] ^= 1 << bit;
                words[64] ^= 1 << bit;
                assert!(!sentinel_tail_unchanged(&words, active));
                assert_eq!(
                    sentinel_tail_unchanged(&words, active),
                    scalar_reference(&words, active)
                );
            }
        }
    }
    #[test]
    fn real_capacity_tails_preserve_all_word_alignments_and_last_word() {
        for len in [393_223, 262_151] {
            for offset in 0..8 {
                let mut storage = vec![SENTINEL; len + 8];
                let words = &mut storage[offset..offset + len];
                for active in [0, 1, 1536, 8192, 12288, len - 7, len] {
                    assert!(sentinel_tail_unchanged(words, active));
                    if active == len {
                        continue;
                    }
                    for index in [active, active + (len - active) / 2, len - 1] {
                        words[index] ^= u32::MAX;
                        assert!(!sentinel_tail_unchanged(words, active));
                        assert_eq!(
                            sentinel_tail_unchanged(words, active),
                            scalar_reference(words, active)
                        );
                        words[index] ^= u32::MAX;
                    }
                }
            }
        }
    }
}
