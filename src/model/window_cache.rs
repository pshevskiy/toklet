//! Standalone FP32 global/local KV ownership; no graph or numerical dispatch.
//!
//! Separate K/V are `[kv_heads, capacity, head_dim]`, initialized and resident.
//! Local slots use `absolute_position % capacity`, with capacity equal to W.
//! Prepared immutable new chunks are `[kv_heads, tokens, head_dim]`. A view
//! borrows needed old rows in at most two spans plus the new chunk; it never
//! copies or overwrites old keys needed by early queries in a large chunk.
//! Batch commit validates every frame before writes, then copies only new rows.
//! Discarding a prepared frame leaves the exact committed prefix unchanged.

use super::{Error, allocation_bytes, total_bytes, window_rows, zeros};
use crate::capacity::{KvOwner, KvStorage, Retention};
use crate::cpu::{self, finite, length, nonzero, product};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

// One cold identity reservation per cache, never per element or decode step.
// No cache is Clone; moving an owner preserves identity. Exhaustion rejects.
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

pub struct WindowKvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    heads: usize,
    dims: usize,
    capacity: usize,
    max_positions: usize,
    window: Option<usize>,
    offset: usize,
    generation: u64,
    owner: u64,
}

/// Opaque borrowed proposal. It contains no copy of the prior cache. A stale or
/// foreign proposal cannot be viewed or committed. New slices stay immutable
/// until this frame and all its views are dropped.
pub struct PreparedKv<'a> {
    k: &'a [f32],
    v: &'a [f32],
    owner: u64,
    generation: u64,
    offset: usize,
    tokens: usize,
}

/// A checked immutable attention union, separate from the final committed ring.
pub struct KvView<'cache, 'input> {
    cache: &'cache WindowKvCache,
    k: &'input [f32],
    v: &'input [f32],
    tokens: usize,
}

/// One head's contiguous rows `[rows, head_dim]` at an absolute logical origin.
/// Empty spans are permitted; K and V have the same length.
#[derive(Clone, Copy, Debug)]
pub struct KvSpan<'a> {
    pub position: usize,
    pub k: &'a [f32],
    pub v: &'a [f32],
}

impl WindowKvCache {
    /// Includes both FP32 buffers and this owner object. Caller-owned new-token
    /// staging, collections of owners and numerical scratch are separate costs.
    pub fn planned_bytes(
        heads: usize,
        dims: usize,
        capacity: usize,
        max_positions: usize,
        window: Option<usize>,
    ) -> Result<usize, Error> {
        for (name, value) in [
            ("KV heads", heads),
            ("KV head width", dims),
            ("KV capacity", capacity),
            ("KV maximum positions", max_positions),
        ] {
            nonzero(value, name)?;
        }
        if capacity > max_positions || window.is_some_and(|w| w == 0 || w != capacity) {
            return Err(Error::InvalidConfig("KV capacity/window"));
        }
        KvOwner {
            heads,
            head_dim: dims,
            storage: KvStorage::F32,
            retention: match window {
                Some(window) => Retention::Sliding { window },
                None => Retention::Full,
            },
            guard_elements: 0,
            copies: 1,
        }
        .at_capacity(capacity)?
        .total_bytes
        .checked_add(std::mem::size_of::<Self>())
        .ok_or(cpu::Error::SizeOverflow.into())
    }

    pub fn new(
        heads: usize,
        dims: usize,
        capacity: usize,
        max_positions: usize,
        window: Option<usize>,
    ) -> Result<Self, Error> {
        Self::planned_bytes(heads, dims, capacity, max_positions, window)?;
        let elements = product(product(heads, capacity)?, dims)?;
        let k = zeros(elements)?;
        let v = zeros(elements)?;
        total_bytes(&[k.capacity(), v.capacity()])?
            .checked_add(std::mem::size_of::<Self>())
            .ok_or(cpu::Error::SizeOverflow)?;
        let owner = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| Error::InvalidConfig("KV owner identity exhausted"))?;
        Ok(Self {
            k,
            v,
            heads,
            dims,
            capacity,
            max_positions,
            window,
            offset: 0,
            generation: 0,
            owner,
        })
    }

    pub(super) fn matches_geometry(
        &self,
        heads: usize,
        dims: usize,
        capacity: usize,
        max_positions: usize,
        window: Option<usize>,
    ) -> bool {
        (
            self.heads,
            self.dims,
            self.capacity,
            self.max_positions,
            self.window,
        ) == (heads, dims, capacity, max_positions, window)
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    #[cfg(test)]
    pub(crate) fn set_generation_for_test(&mut self, generation: u64) {
        self.generation = generation;
    }

    pub fn allocated_bytes(&self) -> usize {
        // The constructor checks actual capacities and this addition.
        (self.k.capacity() + self.v.capacity()) * std::mem::size_of::<f32>()
            + std::mem::size_of::<Self>()
    }

    /// Physical initialized storage, including unused capacity. Validity is
    /// described by `position_in_slot`; numerical consumers should use a view.
    pub fn committed(&self) -> (&[f32], &[f32]) {
        (&self.k, &self.v)
    }

    pub fn position_in_slot(&self, slot: usize) -> Option<usize> {
        if slot >= self.capacity || self.offset == 0 {
            return None;
        }
        if self.window.is_none() {
            return (slot < self.offset).then_some(slot);
        }
        let last = self.offset - 1;
        let base = last - last % self.capacity;
        let position = if slot <= last % self.capacity {
            base + slot
        } else {
            base.checked_sub(self.capacity - slot)?
        };
        (position >= self.offset.saturating_sub(self.capacity)).then_some(position)
    }

    /// All lengths, finite new values and admission arithmetic are checked
    /// before creating a proposal. Empty sequences make no prepare/commit call.
    pub fn prepare<'a>(
        &self,
        k: &'a [f32],
        v: &'a [f32],
        tokens: usize,
    ) -> Result<PreparedKv<'a>, Error> {
        let elements = self.admit_tokens(tokens)?;
        length("new KV keys", k.len(), elements)?;
        length("new KV values", v.len(), elements)?;
        finite("new KV keys", k)?;
        finite("new KV values", v)?;
        Ok(PreparedKv {
            k,
            v,
            owner: self.owner,
            generation: self.generation,
            offset: self.offset,
            tokens,
        })
    }

    fn admit_tokens(&self, tokens: usize) -> Result<usize, Error> {
        nonzero(tokens, "KV new tokens")?;
        let after = self
            .offset
            .checked_add(tokens)
            .ok_or(cpu::Error::SizeOverflow)?;
        let limit = if self.window.is_some() {
            self.max_positions
        } else {
            self.capacity.min(self.max_positions)
        };
        if after > limit {
            return Err(Error::Capacity {
                offset: self.offset,
                tokens,
                limit,
            });
        }
        self.generation
            .checked_add(1)
            .ok_or(cpu::Error::SizeOverflow)?;
        let elements = product(product(self.heads, tokens)?, self.dims)?;
        allocation_bytes(elements)?;
        Ok(elements)
    }

    fn validate(&self, frame: &PreparedKv<'_>) -> Result<(), Error> {
        // Identity fixes immutable owner geometry; generation/offset fix its
        // committed prefix. Recheck admission and lengths before batch mutation.
        // Finite inputs cannot change through the frame's immutable borrows.
        if frame.owner != self.owner
            || frame.generation != self.generation
            || frame.offset != self.offset
        {
            return Err(Error::InvalidConfig("stale or foreign KV proposal"));
        }
        let elements = self.admit_tokens(frame.tokens)?;
        length("new KV keys", frame.k.len(), elements)?;
        length("new KV values", frame.v.len(), elements)?;
        Ok(())
    }

    pub fn view<'cache, 'input>(
        &'cache self,
        frame: &PreparedKv<'input>,
    ) -> Result<KvView<'cache, 'input>, Error> {
        self.validate(frame)?;
        Ok(KvView {
            cache: self,
            k: frame.k,
            v: frame.v,
            tokens: frame.tokens,
        })
    }

    /// The concrete hybrid boundary validates every state family before copies.
    pub(super) fn validate_step(
        &self,
        frame: &PreparedKv<'_>,
        offset: usize,
        tokens: usize,
        generation: u64,
    ) -> Result<(), Error> {
        if self.offset != offset || self.generation != generation || frame.tokens != tokens {
            return Err(Error::InvalidConfig("mixed KV/session step"));
        }
        self.validate(frame)
    }
    pub(super) fn validate_reset(&self) -> Result<(), Error> {
        self.generation
            .checked_add(1)
            .ok_or(Error::InvalidConfig("KV reset generation exhausted"))?;
        Ok(())
    }
    /// Call only immediately after all participating reset validations succeed.
    pub(super) fn reset_validated(&mut self) {
        self.offset = 0;
        self.generation += 1;
    }

    pub fn commit(&mut self, frame: &PreparedKv<'_>) -> Result<(), Error> {
        commit_batch(std::slice::from_mut(self), std::slice::from_ref(frame))
    }

    pub(super) fn commit_validated(&mut self, frame: &PreparedKv<'_>) {
        // All owners were validated before any writes. Each backend applies
        // the same row descriptors to its own head strides and scalar storage.
        let runs = window_rows::copy_runs(
            self.capacity,
            self.offset,
            frame.tokens,
            self.window.is_some(),
        );
        for head in 0..self.heads {
            for run in runs {
                if run.rows == 0 {
                    continue;
                }
                let source = (head * frame.tokens + run.source_row) * self.dims;
                let target = (head * self.capacity + run.slot) * self.dims;
                let count = run.rows * self.dims;
                self.k[target..target + count].copy_from_slice(&frame.k[source..source + count]);
                self.v[target..target + count].copy_from_slice(&frame.v[source..source + count]);
            }
        }
        self.offset += frame.tokens;
        self.generation += 1;
    }
}

/// All owners require exactly one corresponding valid frame. Missing, foreign,
/// duplicate and stale frames reject before any owner changes. No allocation or
/// fallible arithmetic remains after validation. The caller invokes this only
/// after the entire model step succeeds; this alone is not a graph transaction.
pub fn commit_batch(caches: &mut [WindowKvCache], frames: &[PreparedKv<'_>]) -> Result<(), Error> {
    length("KV commit frames", frames.len(), caches.len())?;
    for (cache, frame) in caches.iter().zip(frames) {
        cache.validate(frame)?;
    }
    for (cache, frame) in caches.iter_mut().zip(frames) {
        cache.commit_validated(frame);
    }
    Ok(())
}

/// Invalidate a complete cache collection without reallocating its buffers.
/// Check every generation before mutation so exhaustion leaves the old prefix.
/// Existing borrowed proposals become stale, even if their old offset was zero.
pub fn reset_batch(caches: &mut [WindowKvCache]) -> Result<(), Error> {
    for cache in caches.iter() {
        cache.validate_reset()?;
    }
    for cache in caches {
        cache.reset_validated();
    }
    Ok(())
}

impl KvView<'_, '_> {
    /// Borrow all heads as two old-ring intervals and a differently strided new
    /// chunk. Empty intervals have empty slices; no old KV is linearized/copied.
    pub fn segments(&self) -> [crate::cpu::AttentionKvSegment<'_>; 3] {
        let empty = crate::cpu::AttentionKvSegment {
            k: &[],
            v: &[],
            keys: 0,
            head_stride: 0,
            key_start: self.cache.offset,
        };
        let mut segments = [empty; 3];
        let stride = self.cache.capacity * self.cache.dims;
        let old = window_rows::old_spans(self.cache.capacity, self.cache.offset, self.cache.window);
        for (index, span) in old.into_iter().enumerate() {
            if span.rows == 0 {
                continue;
            }
            let begin = span.slot * self.cache.dims;
            let end = (self.cache.heads - 1) * stride + begin + span.rows * self.cache.dims;
            segments[index] = crate::cpu::AttentionKvSegment {
                k: &self.cache.k[begin..end],
                v: &self.cache.v[begin..end],
                keys: span.rows,
                head_stride: stride,
                key_start: span.position,
            };
        }
        segments[2] = crate::cpu::AttentionKvSegment {
            k: self.k,
            v: self.v,
            keys: self.tokens,
            head_stride: self.tokens * self.cache.dims,
            key_start: self.cache.offset,
        };
        segments
    }

    /// Describe the borrowed geometry. The numerical entry point validates the
    /// requested query-head count and scale together with its input/output sizes.
    pub fn attention_shape(
        &self,
        query_heads: usize,
        scale: f32,
    ) -> crate::cpu::SpanAttentionShape {
        crate::cpu::SpanAttentionShape {
            queries: self.tokens,
            keys: self.end() - self.start(),
            query_heads,
            kv_heads: self.cache.heads,
            head_dim: self.cache.dims,
            query_start: self.cache.offset,
            key_start: self.start(),
            window: self.cache.window,
            scale,
        }
    }

    pub fn query_start(&self) -> usize {
        self.cache.offset
    }

    pub fn end(&self) -> usize {
        self.cache.offset + self.tokens
    }

    pub fn start(&self) -> usize {
        window_rows::view_start(self.cache.offset, self.cache.window)
    }

    pub fn visible_range(&self, query: usize) -> Result<Range<usize>, Error> {
        if !(self.query_start()..self.end()).contains(&query) {
            return Err(Error::InvalidConfig("KV view query position"));
        }
        let start = window_rows::view_start(query, self.cache.window);
        Ok(start..query + 1)
    }

    /// Spans are in increasing logical order, regardless of physical wrap.
    /// New data remains separate until commit. A view borrows its cache, so safe
    /// Rust cannot mutate the owner while a view is in use.
    pub fn head(&self, head: usize) -> Result<[KvSpan<'_>; 3], Error> {
        if head >= self.cache.heads {
            return Err(Error::InvalidConfig("KV view head"));
        }
        let empty = KvSpan {
            position: self.cache.offset,
            k: &[],
            v: &[],
        };
        let mut spans = [empty; 3];
        let old = window_rows::old_spans(self.cache.capacity, self.cache.offset, self.cache.window);
        if old[0].rows > 0 {
            for (index, span) in old.into_iter().enumerate() {
                let begin = (head * self.cache.capacity + span.slot) * self.cache.dims;
                let end = begin + span.rows * self.cache.dims;
                spans[index] = KvSpan {
                    position: span.position,
                    k: &self.cache.k[begin..end],
                    v: &self.cache.v[begin..end],
                };
            }
        }
        let first = head * self.tokens * self.cache.dims;
        let last = first + self.tokens * self.cache.dims;
        spans[2] = KvSpan {
            position: self.cache.offset,
            k: &self.k[first..last],
            v: &self.v[first..last],
        };
        Ok(spans)
    }
}

#[cfg(test)]
mod tests;
