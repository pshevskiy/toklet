//! Experimental retained late-T128 attention over borrowed graph KV spans.
use super::*;
use crate::{
    cpu::{KvSegment, SpanAttentionShape},
    model::Error,
};

const QUERIES: usize = 128;
const HEADS: usize = 16;
const K_HEADS: usize = 4;
const DIM: usize = 128;
const WIDTH: usize = QUERIES * HEADS * DIM;
const PLANE: usize = QUERIES * HEADS * 512;
const GUARD: usize = 8;
const SENTINEL: u16 = 0x7fc1;
const STATUS_SENTINEL: u32 = 0x73a8_fff1;
const DOMAIN: u32 = 7;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

fn operation_error() -> Error {
    Error::InvalidConfig("LFM3 late attention Metal operation failed")
}

pub(super) struct Workspace {
    selected: bool,
    pipelines: [Pipeline; 3],
    q: Buffer<u16>,
    k: Buffer<u16>,
    v: Buffer<u16>,
    plane: Buffer<u16>,
    context: Buffer<u16>,
    status: Buffer<u32>,
    counters: [u64; 5],
}

impl Workspace {
    pub(super) const BUFFER_BYTES: [usize; 6] = [
        2 * (WIDTH + GUARD),
        2 * (WIDTH + GUARD),
        2 * (WIDTH + GUARD),
        2 * (PLANE + GUARD),
        2 * (WIDTH + GUARD),
        4 * (1 + GUARD),
    ];
    // Opaque pipeline/command preparation allowance, separately charged.
    pub(super) const NATIVE_ALLOWANCE: usize = 16 << 20;

    pub(super) fn new(gpu: &Gpu, selected: bool) -> std::result::Result<Self, Error> {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version2_3);
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        if options.languageVersion() != MTLLanguageVersion::Version2_3
            || options.mathMode() != MTLMathMode::Safe
            || options.mathFloatingPointFunctions() != MTLMathFloatingPointFunctions::Precise
        {
            return Err(operation_error());
        }
        let shader = NSString::from_str(include_str!("attention.metal"));
        let library = gpu
            .device
            .newLibraryWithSource_options_error(&shader, Some(&options))
            .map_err(|_| operation_error())?;
        let pipeline = |name| {
            let f = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(operation_error)?;
            gpu.device
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|_| operation_error())
        };
        let pipelines = [
            pipeline("lfm3_late_qk")?,
            pipeline("lfm3_late_probability")?,
            pipeline("lfm3_late_context")?,
        ];
        if pipelines[0].threadExecutionWidth() != 32
            || pipelines[1].threadExecutionWidth() != 32
            || pipelines
                .iter()
                .zip([128, 32, 32])
                .any(|(p, n)| p.maxTotalThreadsPerThreadgroup() < n)
        {
            return Err(operation_error());
        }
        Ok(Self {
            selected,
            pipelines,
            q: Buffer::new(gpu, WIDTH + GUARD).map_err(|_| Error::Allocation)?,
            k: Buffer::new(gpu, WIDTH + GUARD).map_err(|_| Error::Allocation)?,
            v: Buffer::new(gpu, WIDTH + GUARD).map_err(|_| Error::Allocation)?,
            plane: Buffer::new(gpu, PLANE + GUARD).map_err(|_| Error::Allocation)?,
            context: Buffer::new(gpu, WIDTH + GUARD).map_err(|_| Error::Allocation)?,
            status: Buffer::new(gpu, 1 + GUARD).map_err(|_| Error::Allocation)?,
            counters: [0; 5],
        })
    }

    pub(super) fn stats(&self) -> [u64; 5] {
        self.counters
    }

    fn count(&mut self, index: usize) -> std::result::Result<(), Error> {
        self.counters[index] = self.counters[index]
            .checked_add(1)
            .ok_or(crate::cpu::Error::SizeOverflow)?;
        Ok(())
    }

    pub(super) fn apply(
        &mut self,
        gpu: &Gpu,
        _layer: usize,
        q: &[bf16],
        spans: &[KvSegment<'_, bf16>; 3],
        context: &mut [bf16],
        shape: &SpanAttentionShape,
    ) -> std::result::Result<bool, Error> {
        if !self.selected
            || shape.queries != QUERIES
            || !matches!(shape.query_start, 128 | 256 | 384)
        {
            return Ok(false);
        }
        let start = shape.query_start;
        let first = shape
            .window
            .map_or(0, |w| start.saturating_sub(w.saturating_sub(1)));
        let old = start - first;
        let local = shape.window == Some(256);
        let lead = if local {
            old.min(256 - first % 256)
        } else {
            old
        };
        let stride = if local { 32_768 } else { 262_144 };
        let tail = old - lead;
        let expected = [
            (first, lead, stride),
            (
                if tail == 0 { start } else { first + lead },
                tail,
                if tail == 0 { 0 } else { stride },
            ),
            (start, 128, 16_384),
        ];
        let exact = (shape.query_heads, shape.kv_heads, shape.head_dim) == (HEADS, K_HEADS, DIM)
            && matches!(shape.window, None | Some(256))
            && shape.scale.to_bits() == 0x3db5_04f3
            && shape.key_start == first
            && shape.keys == old + 128
            && q.len() == WIDTH
            && context.len() == WIDTH
            && spans.iter().zip(expected).all(|(s, (pos, rows, stride))| {
                let n = if rows == 0 {
                    0
                } else {
                    (K_HEADS - 1) * stride + rows * DIM
                };
                (s.key_start, s.keys, s.head_stride) == (pos, rows, stride)
                    && s.k.len() == n
                    && s.v.len() == n
            });
        if !exact {
            return Err(operation_error());
        }
        self.count(0)?;
        let keys = shape.keys;
        let boxes = QUERIES * HEADS * keys;
        let key_words = K_HEADS * keys * DIM;
        let geometry = [
            keys as u32,
            start as u32,
            first as u32,
            shape.window.unwrap_or(0) as u32,
        ];
        self.q.with_mut(|dst| {
            for (to, from) in dst[..WIDTH].iter_mut().zip(q) {
                *to = from.to_bits();
            }
            dst[WIDTH..].fill(SENTINEL);
        });
        let pack = |out: &mut Buffer<u16>, values| {
            out.with_mut(|dst| {
                dst[key_words..].fill(SENTINEL);
                for (h, head) in dst[..key_words].chunks_exact_mut(keys * DIM).enumerate() {
                    let mut offset = 0;
                    for span in spans.iter().filter(|s| s.keys != 0) {
                        let words = span.keys * DIM;
                        let data = if values { span.v } else { span.k };
                        let start = h * span.head_stride;
                        for (to, from) in head[offset..offset + words]
                            .iter_mut()
                            .zip(&data[start..start + words])
                        {
                            *to = from.to_bits();
                        }
                        offset += words;
                    }
                }
            });
        };
        pack(&mut self.k, false);
        pack(&mut self.v, true);
        self.plane.with_mut(|v| v.fill(SENTINEL));
        self.context.with_mut(|v| v.fill(SENTINEL));
        self.status.with_mut(|v| {
            v.fill(STATUS_SENTINEL);
            v[0] = 0;
        });

        let command = gpu.command().map_err(|_| operation_error())?;
        if command.status() != MTLCommandBufferStatus::NotEnqueued {
            return Err(operation_error());
        }
        let size = |width| MTLSize {
            width,
            height: 1,
            depth: 1,
        };
        let buffers: [&[&ProtocolObject<dyn MTLBuffer>]; 3] = [
            &[&self.q.raw, &self.k.raw, &self.plane.raw, &self.status.raw],
            &[&self.plane.raw, &self.status.raw],
            &[
                &self.plane.raw,
                &self.v.raw,
                &self.context.raw,
                &self.status.raw,
            ],
        ];
        for (i, (groups, width)) in [(boxes / 4, 128), (QUERIES * HEADS, 32), (WIDTH / 32, 32)]
            .into_iter()
            .enumerate()
        {
            let encoder = command
                .computeCommandEncoder()
                .ok_or_else(operation_error)?;
            encoder.setComputePipelineState(&self.pipelines[i]);
            // SAFETY: exact checked active dimensions; same-device private
            // tracked buffers are retained until completion. Only the
            // sequential plane is reused across separate ordered encoders.
            unsafe {
                for (slot, &buffer) in buffers[i].iter().enumerate() {
                    encoder.setBuffer_offset_atIndex(Some(buffer), 0, slot);
                }
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&geometry).cast(),
                    std::mem::size_of_val(&geometry),
                    buffers[i].len(),
                );
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(width));
            encoder.endEncoding();
        }
        self.count(1)?;
        command.commit();
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
            return Err(operation_error());
        }
        self.count(2)?;
        let status = self
            .status
            .with_slice(|v| v[1..].iter().all(|&x| x == STATUS_SENTINEL).then_some(v[0]))
            .ok_or_else(operation_error)?;
        if status & !DOMAIN != 0
            || ![
                (&self.q, WIDTH),
                (&self.k, key_words),
                (&self.v, key_words),
                (&self.plane, boxes),
                (&self.context, WIDTH),
            ]
            .into_iter()
            .all(|(buf, n)| buf.with_slice(|v| v[n..].iter().all(|&x| x == SENTINEL)))
        {
            return Err(operation_error());
        }
        if status != 0 {
            self.count(4)?;
            return Ok(false);
        }
        // Validate every output before copying any word into caller scratch.
        if !self
            .context
            .with_slice(|src| src[..WIDTH].iter().all(|&w| bf16::from_bits(w).is_finite()))
        {
            return Err(operation_error());
        }
        self.count(3)?;
        self.context.with_slice(|src| {
            for (dst, &bits) in context.iter_mut().zip(src) {
                *dst = bf16::from_bits(bits);
            }
        });
        Ok(true)
    }
}
