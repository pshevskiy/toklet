//! Bounded FP32 Gemma attention building block over three immutable KV spans.
//!
//! This standalone path qualifies one mechanism before resident block integration.
//! Q/context use [queries,heads,D]; scores/probabilities [heads,queries,keys].
//! It retains the tested compensated dot/context and precise FP32 softmax shader.
//! No model/session route selects it automatically.
use super::{
    Error, Result,
    memory::{Buffer, Gpu, validate_buffer_bytes},
};
use crate::cpu::{AttentionKvSegment, SpanAttentionShape};
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::{mem::size_of, ptr::NonNull};

pub const CONTEXT_ALLOWANCE_BYTES: usize = 64 * 1024 * 1024;
const GUARD: usize = 7;
const SENTINEL: u32 = 0x7fc1_2345;

/// Fixed capacities. The currently qualified Gemma geometry has one KV head,
/// at most eight query heads, 32 query rows, 1,025 keys and D<=512.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub max_queries: usize,
    pub max_keys: usize,
    pub query_heads: usize,
    pub head_dim: usize,
}
impl Options {
    /// Logical owned buffer payload plus Workspace value; context/compiler and
    /// caller arrays are separate. This is admission, not process footprint.
    pub fn planned_bytes(self) -> Result<usize> {
        Ok(Sizes::new(self)?.owned)
    }
}

/// Cached device, queue and four precise pipelines. Count its policy allowance
/// once, separately from every workspace and caller/diagnostic array.
pub struct Context {
    gpu: Gpu,
    pipelines: Pipelines,
}
/// Concrete pipelines shared by standalone and future resident block owners.
/// Construction uses the caller's device; no second queue/context is required.
pub(super) struct Pipelines {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    pipelines: [Retained<ProtocolObject<dyn MTLComputePipelineState>>; 4],
    widths: [usize; 4],
}
impl Context {
    pub fn new(available_bytes: usize) -> Result<Self> {
        if available_bytes < CONTEXT_ALLOWANCE_BYTES {
            return Err(Error::Budget {
                required: CONTEXT_ALLOWANCE_BYTES,
                limit: available_bytes,
            });
        }
        autoreleasepool(|_| {
            let gpu = Gpu::new()?;
            let pipelines = Pipelines::new(&gpu)?;
            Ok(Self { gpu, pipelines })
        })
    }
    pub fn device_name(&self) -> String {
        self.gpu.device.name().to_string()
    }
}
impl Pipelines {
    pub(super) fn new(gpu: &Gpu) -> Result<Self> {
        autoreleasepool(|_| {
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version2_3);
            options.setMathMode(MTLMathMode::Safe);
            options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
            if options.languageVersion() != MTLLanguageVersion::Version2_3
                || options.mathMode() != MTLMathMode::Safe
                || options.mathFloatingPointFunctions() != MTLMathFloatingPointFunctions::Precise
            {
                return Err(Error::Unsupported(
                    "Gemma attention precise math unavailable",
                ));
            }
            let library = gpu
                .device
                .newLibraryWithSource_options_error(
                    &NSString::from_str(include_str!("gemma_attention.metal")),
                    Some(&options),
                )
                .map_err(|e| Error::Native(e.to_string()))?;
            let pipeline = |name| {
                let function = library
                    .newFunctionWithName(&NSString::from_str(name))
                    .ok_or_else(|| Error::Native("Gemma attention kernel absent".into()))?;
                gpu.device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|e| Error::Native(e.to_string()))
            };
            let pipelines = [
                pipeline("validate_inputs")?,
                pipeline("scores3")?,
                pipeline("softmax3")?,
                pipeline("context3")?,
            ];
            let widths = pipelines.each_ref().map(|p| p.threadExecutionWidth());
            for (pipeline, &width) in pipelines.iter().zip(&widths) {
                if width == 0
                    || width > pipeline.maxTotalThreadsPerThreadgroup()
                    || width > u16::MAX as usize
                {
                    return Err(Error::Unsupported(
                        "Gemma attention launch width unavailable",
                    ));
                }
            }
            Ok(Self {
                device: gpu.device.clone(),
                pipelines,
                widths,
            })
        })
    }
}
impl Context {
    /// Preflight every payload and the exact whole workspace budget before the
    /// first numerical buffer allocation. Each of six KV buffers has max_keys
    /// capacity; unused capacity remains guarded. No expanded query-head KV.
    pub fn workspace(
        &self,
        options: Options,
        available_owned_bytes: usize,
    ) -> Result<Workspace<'_>> {
        let sizes = Sizes::new(options)?;
        if sizes.owned > available_owned_bytes {
            return Err(Error::Budget {
                required: sizes.owned,
                limit: available_owned_bytes,
            });
        }
        for n in [sizes.query, sizes.kv, sizes.scores, 1] {
            validate_buffer_bytes((n + GUARD) * 4, self.gpu.device.maxBufferLength())?;
        }
        let make = |n| Buffer::new(&self.gpu, n + GUARD);
        Ok(Workspace {
            context: self,
            options,
            sizes,
            q: make(sizes.query)?,
            kv: [
                make(sizes.kv)?,
                make(sizes.kv)?,
                make(sizes.kv)?,
                make(sizes.kv)?,
                make(sizes.kv)?,
                make(sizes.kv)?,
            ],
            scores: make(sizes.scores)?,
            probabilities: make(sizes.scores)?,
            output: make(sizes.query)?,
            status: make(1)?,
            poisoned: false,
            pending: None,
            completed: None,
        })
    }
}
#[derive(Clone, Copy)]
struct Sizes {
    query: usize,
    kv: usize,
    scores: usize,
    owned: usize,
}
impl Sizes {
    fn new(o: Options) -> Result<Self> {
        if !(1..=32).contains(&o.max_queries)
            || !(1..=1025).contains(&o.max_keys)
            || !(1..=8).contains(&o.query_heads)
            || !(1..=512).contains(&o.head_dim)
        {
            return Err(Error::Options("Gemma attention capacity outside bounds"));
        }
        // All dimensions are bounded before arithmetic; retain checked totals
        // so later changes cannot silently weaken allocation admission.
        let query = o.max_queries * o.query_heads * o.head_dim;
        let kv = o.max_keys * o.head_dim;
        let scores = o.query_heads * o.max_queries * o.max_keys;
        let words = 2 * (query + GUARD) + 6 * (kv + GUARD) + 2 * (scores + GUARD) + 1 + GUARD;
        let owned = words
            .checked_mul(4)
            .and_then(|n| n.checked_add(size_of::<Workspace<'_>>()))
            .ok_or(Error::Options("Gemma attention allocation overflow"))?;
        Ok(Self {
            query,
            kv,
            scores,
            owned,
        })
    }
}
#[derive(Clone, Copy)]
struct Geometry {
    words: [u32; 15],
    query: usize,
    scores: usize,
    segment: [usize; 3],
}
#[derive(Clone, Copy)]
struct SegmentLayout {
    keys: usize,
    head_stride: usize,
    key_start: usize,
    k_len: usize,
    v_len: usize,
}
impl Geometry {
    fn new(
        o: Options,
        qlen: usize,
        output_len: usize,
        spans: &[AttentionKvSegment<'_>; 3],
        s: &SpanAttentionShape,
    ) -> Result<Self> {
        Self::from_lengths(
            o,
            qlen,
            output_len,
            spans.each_ref().map(|span| SegmentLayout {
                keys: span.keys,
                head_stride: span.head_stride,
                key_start: span.key_start,
                k_len: span.k.len(),
                v_len: span.v.len(),
            }),
            s,
        )
    }
    fn from_lengths(
        o: Options,
        qlen: usize,
        output_len: usize,
        spans: [SegmentLayout; 3],
        s: &SpanAttentionShape,
    ) -> Result<Self> {
        // Callers of the internal resident seam need the same capacity bounds
        // before multiplication as callers constructing a standalone owner.
        Sizes::new(o)?;
        if s.queries == 0
            || s.queries > o.max_queries
            || s.keys == 0
            || s.keys > o.max_keys
            || s.query_heads != o.query_heads
            || s.kv_heads != 1
            || s.head_dim != o.head_dim
            || s.window == Some(0)
            || s.scale.to_bits() != 1f32.to_bits()
        {
            return Err(Error::Options(
                "Gemma attention request outside admitted geometry",
            ));
        }
        let query = s.queries * s.query_heads * s.head_dim;
        let scores = s.query_heads * s.queries * s.keys;
        if qlen != query || output_len != query {
            return Err(Error::Options("Gemma attention query/output length"));
        }
        let end = s
            .key_start
            .checked_add(s.keys - 1)
            .ok_or(Error::Options("Gemma attention key end overflow"))?;
        let query_end = s
            .query_start
            .checked_add(s.queries - 1)
            .ok_or(Error::Options("Gemma attention query end overflow"))?;
        if end > u32::MAX as usize
            || query_end > u32::MAX as usize
            || s.window.is_some_and(|w| w > u32::MAX as usize)
        {
            return Err(Error::Options(
                "Gemma attention position exceeds shader index",
            ));
        }
        let mut keys = 0usize;
        let mut segment = [0usize; 3];
        for (i, span) in spans.iter().enumerate() {
            if span.keys > s.keys {
                return Err(Error::Options("Gemma attention segment exceeds keys"));
            }
            let count = span.keys * s.head_dim;
            if span.k_len != count || span.v_len != count {
                return Err(Error::Options("Gemma attention segment length"));
            }
            if span.keys != 0
                && (span.head_stride < count
                    || s.key_start.checked_add(keys) != Some(span.key_start))
            {
                return Err(Error::Options("Gemma attention segment stride/origin"));
            }
            keys = keys
                .checked_add(span.keys)
                .ok_or(Error::Options("Gemma attention segment sum overflow"))?;
            segment[i] = count;
        }
        if keys != s.keys {
            return Err(Error::Options("Gemma attention segment coverage"));
        }
        for row in 0..s.queries {
            let position = s.query_start + row;
            let first = s.window.map_or(s.key_start, |w| {
                s.key_start.max(position.saturating_sub(w - 1))
            });
            if first > end.min(position) {
                return Err(Error::Options("Gemma attention fully masked query"));
            }
        }
        let words = [
            s.queries,
            s.keys,
            s.query_heads,
            1,
            s.head_dim,
            s.query_start,
            s.key_start,
            s.window.unwrap_or(0),
            spans[0].keys,
            spans[1].keys,
            spans[2].keys,
            segment[0],
            segment[1],
            segment[2],
            1f32.to_bits() as usize,
        ]
        .map(|x| x as u32); // Every field bounded above, including absolute endpoints.
        Ok(Self {
            words,
            query,
            scores,
            segment,
        })
    }
}

pub(super) use super::memory::BufferRange;
pub(super) struct DeviceSegment<'a> {
    pub k: BufferRange<'a>,
    pub v: BufferRange<'a>,
    pub keys: usize,
    pub head_stride: usize,
    pub key_start: usize,
}
pub(super) struct DeviceBuffers<'a> {
    pub q: BufferRange<'a>,
    pub segments: [DeviceSegment<'a>; 3],
    pub scores: BufferRange<'a>,
    pub probabilities: BufferRange<'a>,
    pub output: BufferRange<'a>,
    pub status: BufferRange<'a>,
}
impl Pipelines {
    /// Encode only: the caller initializes status to zero before submission,
    /// owns the command and all buffers, and excludes CPU access until checked
    /// terminal completion. Completion/status/guards must pass before publishing
    /// outputs; any encode/submission/completion failure invalidates that owner.
    /// Resources are retained by the checked command. Read-only ranges may
    /// share backing storage (including wrapped KV ring spans); every range
    /// overlapping a writable range is rejected before an encoder is created.
    pub(super) fn encode(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        b: &DeviceBuffers<'_>,
        shape: &SpanAttentionShape,
    ) -> Result<()> {
        let g = Geometry::from_lengths(
            Options {
                max_queries: 32,
                max_keys: 1025,
                query_heads: shape.query_heads,
                head_dim: shape.head_dim,
            },
            b.q.len,
            b.output.len,
            b.segments.each_ref().map(|s| SegmentLayout {
                keys: s.keys,
                head_stride: s.head_stride,
                key_start: s.key_start,
                k_len: s.k.len,
                v_len: s.v.len,
            }),
            shape,
        )?;
        if b.scores.len != g.scores || b.probabilities.len != g.scores || b.status.len != 1 {
            return Err(Error::Options("Gemma attention scratch/status length"));
        }
        if !command.retainedReferences()
            || command.status() != MTLCommandBufferStatus::NotEnqueued
            || !std::ptr::eq(&*command.device(), &*self.device)
        {
            return Err(Error::Options("Gemma attention command contract differs"));
        }
        let buffers = [
            b.q,
            b.segments[0].k,
            b.segments[0].v,
            b.segments[1].k,
            b.segments[1].v,
            b.segments[2].k,
            b.segments[2].v,
            b.scores,
            b.probabilities,
            b.output,
            b.status,
        ];
        for (i, &buffer) in buffers.iter().enumerate() {
            if !std::ptr::eq(&*buffer.raw.device(), &*self.device) {
                return Err(Error::Options("Gemma attention foreign device buffer"));
            }
            for &other in &buffers[..i] {
                if i >= 7 && buffer.overlaps(other) {
                    return Err(Error::Options("Gemma attention writable buffer overlap"));
                }
            }
        }
        let totals = [
            g.query.max(shape.keys * shape.head_dim),
            g.scores,
            shape.queries * shape.query_heads,
            g.query,
        ];
        for (index, &total) in totals.iter().enumerate() {
            let encoder = command
                .computeCommandEncoder()
                .ok_or_else(|| Error::Native("Gemma attention encoder absent".into()))?;
            encoder.setComputePipelineState(&self.pipelines[index]);
            // SAFETY: private fixed shader ABI; all typed ranges, lengths,
            // endpoints, same-device resources and writable disjointness were
            // checked above. setBytes copies shape immediately; the command
            // retains tracked buffers. Caller owns synchronization/publication.
            unsafe {
                for (slot, buffer) in buffers[..10].iter().enumerate() {
                    encoder.setBuffer_offset_atIndex(Some(buffer.raw), buffer.offset * 4, slot);
                }
                encoder.setBytes_length_atIndex(
                    NonNull::from(&g.words).cast(),
                    size_of::<[u32; 15]>(),
                    10,
                );
                encoder.setBuffer_offset_atIndex(Some(b.status.raw), b.status.offset * 4, 11);
            }
            let width = self.widths[index];
            let groups = if index == 1 {
                total
            } else {
                total.div_ceil(width)
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups,
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
        }
        Ok(())
    }
}

/// Fixed buffers are reused across shapes within capacity. Preflight errors
/// leave the owner and caller output unchanged. Any error after staging begins
/// invalidates diagnostics and permanently poisons the owner; recreate it.
/// Buffers and retained command stay owned through checked GPU completion.
pub struct Workspace<'a> {
    context: &'a Context,
    options: Options,
    sizes: Sizes,
    q: Buffer<u32>,
    kv: [Buffer<u32>; 6],
    scores: Buffer<u32>,
    probabilities: Buffer<u32>,
    output: Buffer<u32>,
    status: Buffer<u32>,
    poisoned: bool,
    pending: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    completed: Option<Geometry>,
}
fn tail(words: &[u32], active: usize) -> bool {
    words[active..]
        .iter()
        .fold(0u32, |difference, &word| difference | (word ^ SENTINEL))
        == 0
}
fn stage(buffer: &mut Buffer<u32>, input: &[f32]) {
    buffer.with_mut(|out| {
        out.fill(SENTINEL);
        for (out, value) in out.iter_mut().zip(input) {
            *out = value.to_bits();
        }
    });
}
fn clear(buffer: &mut Buffer<u32>) {
    buffer.with_mut(|out| out.fill(SENTINEL));
}
impl Workspace<'_> {
    pub fn owned_bytes(&self) -> usize {
        self.sizes.owned
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// Synchronous standalone boundary. Four dependent encoders, one command,
    /// one completed wait. Copies only context to the caller on success;
    /// scores/probabilities require an explicit diagnostic call afterward.
    pub fn apply(
        &mut self,
        q: &[f32],
        spans: &[AttentionKvSegment<'_>; 3],
        output: &mut [f32],
        shape: &SpanAttentionShape,
    ) -> Result<()> {
        if self.poisoned || self.pending.is_some() {
            return Err(Error::ReplayRequired);
        }
        let g = Geometry::new(self.options, q.len(), output.len(), spans, shape)?;
        if q.iter()
            .chain(spans.iter().flat_map(|s| s.k.iter().chain(s.v)))
            .any(|x| !x.is_finite())
        {
            return Err(Error::Options("Gemma attention nonfinite input"));
        }
        self.poisoned = true;
        self.completed = None;
        autoreleasepool(|_| {
            stage(&mut self.q, q);
            for (i, span) in spans.iter().enumerate() {
                stage(&mut self.kv[2 * i], span.k);
                stage(&mut self.kv[2 * i + 1], span.v);
            }
            clear(&mut self.scores);
            clear(&mut self.probabilities);
            clear(&mut self.output);
            clear(&mut self.status);
            self.status.with_mut(|s| s[0] = 0);
            let command = self.context.gpu.command()?;
            fn range(buffer: &Buffer<u32>, len: usize) -> Result<BufferRange<'_>> {
                BufferRange::new(buffer, 0, len)
            }
            let segment = |i: usize| -> Result<DeviceSegment<'_>> {
                Ok(DeviceSegment {
                    k: range(&self.kv[2 * i], g.segment[i])?,
                    v: range(&self.kv[2 * i + 1], g.segment[i])?,
                    keys: spans[i].keys,
                    head_stride: spans[i].head_stride,
                    key_start: spans[i].key_start,
                })
            };
            let segments = [segment(0)?, segment(1)?, segment(2)?];
            self.context.pipelines.encode(
                &command,
                &DeviceBuffers {
                    q: range(&self.q, g.query)?,
                    segments,
                    scores: range(&self.scores, g.scores)?,
                    probabilities: range(&self.probabilities, g.scores)?,
                    output: range(&self.output, g.query)?,
                    status: range(&self.status, 1)?,
                },
                shape,
            )?;
            self.pending = Some(command);
            let command = self.pending.as_ref().unwrap();
            command.commit();
            command.waitUntilCompleted();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(Error::Completion);
            }
            self.validate_completed(&g)?;
            self.output.with_slice(|s| {
                for (out, &word) in output.iter_mut().zip(&s[..g.query]) {
                    *out = f32::from_bits(word);
                }
            });
            self.pending = None;
            self.poisoned = false;
            self.completed = Some(g);
            Ok(())
        })
    }
    fn validate_completed(&self, g: &Geometry) -> Result<()> {
        if !self.q.with_slice(|s| tail(s, g.query))
            || self
                .kv
                .iter()
                .enumerate()
                .any(|(i, b)| !b.with_slice(|s| tail(s, g.segment[i / 2])))
            || !self.scores.with_slice(|s| tail(s, g.scores))
            || !self.probabilities.with_slice(|s| tail(s, g.scores))
            || !self.output.with_slice(|s| tail(s, g.query))
            || !self.status.with_slice(|s| tail(s, 1))
        {
            return Err(Error::Options("Gemma attention guard changed"));
        }
        if self.status.with_slice(|s| s[0]) != 0
            || [&self.scores, &self.probabilities, &self.output]
                .iter()
                .zip([g.scores, g.scores, g.query])
                .any(|(b, n)| {
                    b.with_slice(|s| s[..n].iter().any(|&x| !f32::from_bits(x).is_finite()))
                })
        {
            return Err(Error::Nonfinite);
        }
        Ok(())
    }
    /// Copies only the most recent successful operation's diagnostic arrays.
    /// Rejected preflight keeps prior diagnostics; an execution failure removes
    /// them. Invalid lengths reject before either caller array is written.
    pub fn copy_diagnostics(&self, scores: &mut [f32], probabilities: &mut [f32]) -> Result<()> {
        if self.poisoned || self.pending.is_some() {
            return Err(Error::ReplayRequired);
        }
        let g = self
            .completed
            .ok_or(Error::Options("Gemma attention diagnostics unavailable"))?;
        if scores.len() != g.scores || probabilities.len() != g.scores {
            return Err(Error::Options("Gemma attention diagnostic length"));
        }
        for (buffer, out) in [(&self.scores, scores), (&self.probabilities, probabilities)] {
            buffer.with_slice(|s| {
                for (out, &word) in out.iter_mut().zip(&s[..g.scores]) {
                    *out = f32::from_bits(word);
                }
            });
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests;
