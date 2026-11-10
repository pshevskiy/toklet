//! Shared typed global/local KV ownership; no graph or numerical dispatch.
//!
//! Separate K/V are `[kv_heads, capacity, head_dim]`, initialized and resident.
//! Local slots use `absolute_position % capacity`, with capacity equal to W.
//! Prepared immutable new chunks are `[kv_heads, tokens, head_dim]`. A view
//! borrows needed old rows in at most two spans plus the new chunk; it never
//! copies or overwrites old keys needed by early queries in a large chunk.
//! Batch commit validates every frame before writes, then copies only new rows.
//! Discarding a prepared frame leaves the exact committed prefix unchanged.

use super::state_element::{allocation_bytes, total_bytes, zeros};
use super::{Error, StateElement, window_rows};
use crate::capacity::{KvOwner, Retention};
use crate::cpu::{self, length, nonzero, product};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

// One cold identity reservation per cache, never per element or decode step.
// No cache is Clone; moving an owner preserves identity. Exhaustion rejects.
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

/// Existing concrete FP32 API. BF16 aliases live in window_cache_bf16.
pub type WindowKvCache = Cache<f32>;
pub type PreparedKv<'a> = Prepared<'a, f32>;
pub type KvView<'cache, 'input> = View<'cache, 'input, f32>;
pub type KvSpan<'a> = Span<'a, f32>;

pub struct Cache<T: StateElement> {
    k: Vec<T>,
    v: Vec<T>,
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
pub struct Prepared<'a, T> {
    k: &'a [T],
    v: &'a [T],
    owner: u64,
    generation: u64,
    offset: usize,
    tokens: usize,
}

#[cfg(test)]
impl<T> Prepared<'_, T> {
    pub(super) fn tokens(&self) -> usize {
        self.tokens
    }
}

/// A checked immutable attention union, separate from the final committed ring.
pub struct View<'cache, 'input, T: StateElement> {
    cache: &'cache Cache<T>,
    k: &'input [T],
    v: &'input [T],
    tokens: usize,
}

/// One head's contiguous rows `[rows, head_dim]` at an absolute logical origin.
/// Empty spans are permitted; K and V have the same length.
#[derive(Clone, Copy, Debug)]
pub struct Span<'a, T> {
    pub position: usize,
    pub k: &'a [T],
    pub v: &'a [T],
}

impl<T: StateElement> Cache<T> {
    /// Includes both typed buffers and this owner object. Caller-owned new-token
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
            storage: T::STORAGE,
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
        let k = zeros::<T>(elements)?;
        let v = zeros::<T>(elements)?;
        total_bytes::<T>(&[k.capacity(), v.capacity()])?
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
        (self.k.capacity() + self.v.capacity()) * std::mem::size_of::<T>()
            + std::mem::size_of::<Self>()
    }

    /// Physical initialized storage, including unused capacity. Validity is
    /// described by `position_in_slot`; numerical consumers should use a view.
    pub fn committed(&self) -> (&[T], &[T]) {
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

    /// Chronological immutable committed rows, including the oldest valid
    /// local slot at a full window. No pending rows or uninitialized capacity
    /// are exposed. The owner cannot be mutated while these views are live.
    pub fn committed_segments(&self) -> [cpu::KvSegment<'_, T>; 3] {
        let start = self.window.map_or(0, |w| self.offset.saturating_sub(w));
        self.segments_for(window_rows::spans(self.capacity, start, self.offset))
    }

    fn segments_for(&self, spans: [window_rows::OldSpan; 2]) -> [cpu::KvSegment<'_, T>; 3] {
        let empty = cpu::KvSegment {
            k: &[],
            v: &[],
            keys: 0,
            head_stride: 0,
            key_start: self.offset,
        };
        let mut segments = [empty; 3];
        let stride = self.capacity * self.dims;
        for (index, span) in spans.into_iter().enumerate() {
            if span.rows == 0 {
                continue;
            }
            let begin = span.slot * self.dims;
            let end = (self.heads - 1) * stride + begin + span.rows * self.dims;
            segments[index] = cpu::KvSegment {
                k: &self.k[begin..end],
                v: &self.v[begin..end],
                keys: span.rows,
                head_stride: stride,
                key_start: span.position,
            };
        }
        segments
    }

    /// All lengths, finite new values and admission arithmetic are checked
    /// before creating a proposal. Empty sequences make no prepare/commit call.
    fn prepare_impl<'a>(
        &self,
        k: &'a [T],
        v: &'a [T],
        tokens: usize,
    ) -> Result<Prepared<'a, T>, Error> {
        let elements = self.admit_tokens(tokens)?;
        length("new KV keys", k.len(), elements)?;
        length("new KV values", v.len(), elements)?;
        T::finite("new KV keys", k)?;
        T::finite("new KV values", v)?;
        Ok(Prepared {
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
        allocation_bytes::<T>(elements)?;
        Ok(elements)
    }

    fn validate(&self, frame: &Prepared<'_, T>) -> Result<(), Error> {
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

    fn view_impl<'cache, 'input>(
        &'cache self,
        frame: &Prepared<'input, T>,
    ) -> Result<View<'cache, 'input, T>, Error> {
        self.validate(frame)?;
        Ok(View {
            cache: self,
            k: frame.k,
            v: frame.v,
            tokens: frame.tokens,
        })
    }

    /// The concrete hybrid boundary validates every state family before copies.
    pub(super) fn validate_step(
        &self,
        frame: &Prepared<'_, T>,
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

    fn commit_impl(&mut self, frame: &Prepared<'_, T>) -> Result<(), Error> {
        commit_batch_impl(std::slice::from_mut(self), std::slice::from_ref(frame))
    }

    pub(super) fn commit_validated(&mut self, frame: &Prepared<'_, T>) {
        self.commit_prefix_validated(frame, frame.tokens);
    }

    /// The routing owner has validated the complete frame and prefix bound.
    /// Source head stride still uses all proposed tokens. Local destination
    /// runs depend on accepted rows, so a rejected suffix cannot overwrite KV.
    pub(super) fn commit_prefix_validated(&mut self, frame: &Prepared<'_, T>, accepted: usize) {
        debug_assert!((1..=frame.tokens).contains(&accepted));
        let runs =
            window_rows::copy_runs(self.capacity, self.offset, accepted, self.window.is_some());
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
        self.offset += accepted;
        self.generation += 1;
    }
}

/// All owners require exactly one corresponding valid frame. Missing, foreign,
/// duplicate and stale frames reject before any owner changes. No allocation or
/// fallible arithmetic remains after validation. The caller invokes this only
/// after the entire model step succeeds; this alone is not a graph transaction.
pub fn commit_batch(caches: &mut [WindowKvCache], frames: &[PreparedKv<'_>]) -> Result<(), Error> {
    commit_batch_impl(caches, frames)
}

pub(super) fn commit_batch_impl<T: StateElement>(
    caches: &mut [Cache<T>],
    frames: &[Prepared<'_, T>],
) -> Result<(), Error> {
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
    reset_batch_impl(caches)
}

pub(super) fn reset_batch_impl<T: StateElement>(caches: &mut [Cache<T>]) -> Result<(), Error> {
    for cache in caches.iter() {
        cache.validate_reset()?;
    }
    for cache in caches {
        cache.reset_validated();
    }
    Ok(())
}

impl<T: StateElement> View<'_, '_, T> {
    /// Borrow all heads as two old-ring intervals and a differently strided new
    /// chunk. Empty intervals have empty slices; no old KV is linearized/copied.
    fn segments_impl(&self) -> [crate::cpu::KvSegment<'_, T>; 3] {
        let spans =
            window_rows::old_spans(self.cache.capacity, self.cache.offset, self.cache.window);
        let mut segments = self.cache.segments_for(spans);
        segments[2] = crate::cpu::KvSegment {
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
    fn attention_shape_impl(
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
    fn head_impl(&self, head: usize) -> Result<[Span<'_, T>; 3], Error> {
        if head >= self.cache.heads {
            return Err(Error::InvalidConfig("KV view head"));
        }
        let empty = Span {
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
                spans[index] = Span {
                    position: span.position,
                    k: &self.cache.k[begin..end],
                    v: &self.cache.v[begin..end],
                };
            }
        }
        let first = head * self.tokens * self.cache.dims;
        let last = first + self.tokens * self.cache.dims;
        spans[2] = Span {
            position: self.cache.offset,
            k: &self.k[first..last],
            v: &self.v[first..last],
        };
        Ok(spans)
    }
}

// Concrete public boundaries instantiate the shared storage work in the engine
// crate, preserving existing FP32 validation/copy optimization across callers.
macro_rules! concrete_api {
    ($ty:ty) => {
        impl Cache<$ty> {
            /// Validate a complete immutable chunk before returning its proposal.
            pub fn prepare<'a>(
                &self,
                k: &'a [$ty],
                v: &'a [$ty],
                tokens: usize,
            ) -> Result<Prepared<'a, $ty>, Error> {
                self.prepare_impl(k, v, tokens)
            }
            pub fn view<'cache, 'input>(
                &'cache self,
                frame: &Prepared<'input, $ty>,
            ) -> Result<View<'cache, 'input, $ty>, Error> {
                self.view_impl(frame)
            }
            pub fn commit(&mut self, frame: &Prepared<'_, $ty>) -> Result<(), Error> {
                self.commit_impl(frame)
            }
        }
        impl View<'_, '_, $ty> {
            pub fn segments(&self) -> [cpu::KvSegment<'_, $ty>; 3] {
                self.segments_impl()
            }
            pub fn attention_shape(
                &self,
                query_heads: usize,
                scale: f32,
            ) -> cpu::SpanAttentionShape {
                self.attention_shape_impl(query_heads, scale)
            }
            pub fn head(&self, head: usize) -> Result<[Span<'_, $ty>; 3], Error> {
                self.head_impl(head)
            }
        }
    };
}
concrete_api!(f32);
concrete_api!(half::bf16);

#[cfg(test)]
mod tests;
