//! Concrete BF16 global/local ownership over the shared checked cache geometry.
//! This defines storage/views, not BF16 attention arithmetic or model admission.
use super::{Error, window_cache};
use half::bf16;

pub type WindowKvCache = window_cache::Cache<bf16>;
pub type PreparedKv<'a> = window_cache::Prepared<'a, bf16>;
pub type KvView<'cache, 'input> = window_cache::View<'cache, 'input, bf16>;
pub type KvSpan<'a> = window_cache::Span<'a, bf16>;

pub fn commit_batch(caches: &mut [WindowKvCache], frames: &[PreparedKv<'_>]) -> Result<(), Error> {
    window_cache::commit_batch_impl(caches, frames)
}
pub fn reset_batch(caches: &mut [WindowKvCache]) -> Result<(), Error> {
    window_cache::reset_batch_impl(caches)
}
