use super::*;
use crate::model::window_rows;
use std::sync::atomic::{AtomicU64, Ordering};
static NEXT_CACHE: AtomicU64 = AtomicU64::new(1);
pub(in crate::metal::gemma_affine::bf16::resident) struct ProducerKv {
    pub k: Buffer<u32>,
    pub v: Buffer<u32>,
    pub capacity: usize,
    pub d: usize,
    pub window: Option<usize>,
    pub offset: usize,
    pub generation: u64,
    pub owner: u64,
    pub block: u64,
    pub history: [u32; 4],
    pub busy: bool,
}
impl ProducerKv {
    pub fn planned_bytes(layer: usize) -> Result<usize> {
        let (d, capacity) = match layer {
            0 => (256, 512),
            4 => (512, 4),
            _ => return Err(Error::Options("producer KV layer outside scope")),
        };
        Ok(2 * (capacity * d + GUARD) * 4 + size_of::<Self>())
    }
    pub fn new(block: &ProducerBlock<'_>, available: usize) -> Result<Self> {
        let required = Self::planned_bytes(block.layer)?;
        check_budget(required, available)?;
        let capacity = if block.layer == 0 { 512 } else { 4 };
        let d = block.layout.d;
        let mut k = Buffer::new(&block.context.0.gpu, capacity * d + GUARD)?;
        let mut v = Buffer::new(&block.context.0.gpu, capacity * d + GUARD)?;
        k.with_mut(|s| s.fill(SENTINEL));
        v.with_mut(|s| s.fill(SENTINEL));
        let owner = NEXT_CACHE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .map_err(|_| Error::Options("producer KV identity exhausted"))?;
        Ok(Self {
            k,
            v,
            capacity,
            d,
            window: (block.layer == 0).then_some(512),
            offset: 0,
            generation: 0,
            owner,
            block: block.identity,
            history: [0; 4],
            busy: false,
        })
    }
    pub fn reset(&mut self) -> Result<()> {
        if self.busy {
            return Err(Error::Options("producer KV pending; cannot reset"));
        }
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or(Error::Options("producer KV generation exhausted"))?;
        self.offset = 0;
        self.history = [0; 4];
        Ok(())
    }
    pub(super) fn guards(&self) -> bool {
        [&self.k, &self.v]
            .iter()
            .all(|b| b.with_slice(|s| s[self.capacity * self.d..].iter().all(|&w| w == SENTINEL)))
    }
    fn storage(&self, block: &ProducerBlock<'_>) -> Result<()> {
        if self.block != block.identity
            || self.d != block.layout.d
            || self.capacity != if block.layer == 0 { 512 } else { 4 }
            || self.window != (block.layer == 0).then_some(512)
        {
            return Err(Error::Options("producer KV storage geometry differs"));
        }
        let words = self.capacity * self.d + GUARD;
        if self.k.len != words
            || self.v.len != words
            || self.k.raw.length() != words * 4
            || self.v.raw.length() != words * 4
            || std::ptr::eq(&*self.k.raw, &*self.v.raw)
            || !std::ptr::eq(&*self.k.raw.device(), &*block.context.0.gpu.device)
            || !std::ptr::eq(&*self.v.raw.device(), &*block.context.0.gpu.device)
        {
            return Err(Error::Options(
                "producer KV physical resource contract differs",
            ));
        }
        Ok(())
    }
    pub(super) fn check(&self, block: &ProducerBlock<'_>, prefix: &[u32]) -> Result<Frame> {
        self.storage(block)?;
        if self.busy
            || self.block != block.identity
            || self.d != block.layout.d
            || self.capacity != if block.layer == 0 { 512 } else { 4 }
            || self.window != (block.layer == 0).then_some(512)
            || prefix.len() > 4
            || prefix.len() <= self.offset
            || prefix[..self.offset] != self.history[..self.offset]
            || prefix.iter().any(|&id| id >= 262144)
        {
            return Err(Error::Options(
                "producer KV identity/prefix/position differs",
            ));
        }
        let rows = prefix.len() - self.offset;
        let mut history = [0; 4];
        history[..prefix.len()].copy_from_slice(prefix);
        self.generation
            .checked_add(1)
            .ok_or(Error::Options("producer KV generation exhausted"))?;
        Ok(Frame {
            owner: self.owner,
            generation: self.generation,
            offset: self.offset,
            rows,
            end: prefix.len(),
            history,
            epoch: block
                .epoch
                .checked_add(1)
                .ok_or(Error::Options("producer epoch exhausted"))?,
        })
    }
    pub(super) fn validate_frame(&self, block: &ProducerBlock<'_>, f: &Frame) -> Result<()> {
        self.storage(block)?;
        if !self.busy
            || self.block != block.identity
            || self.owner != f.owner
            || self.generation != f.generation
            || self.offset != f.offset
            || f.epoch != block.epoch
            || f.end > 4
            || f.rows == 0
            || f.offset.checked_add(f.rows) != Some(f.end)
            || self.history[..f.offset] != f.history[..f.offset]
        {
            return Err(Error::Options("producer stale or foreign frame"));
        }
        Ok(())
    }
    pub(super) fn runs(&self, f: &Frame) -> Result<[window_rows::CopyRun; 2]> {
        if f.end > 4 || f.offset.checked_add(f.rows) != Some(f.end) || f.rows == 0 {
            return Err(Error::Options("producer copy frame differs"));
        }
        let runs = window_rows::copy_runs(self.capacity, f.offset, f.rows, self.window.is_some());
        for r in runs {
            if r.source_row.checked_add(r.rows).is_none_or(|n| n > f.rows)
                || r.slot.checked_add(r.rows).is_none_or(|n| n > self.capacity)
            {
                return Err(Error::Options("producer KV copy exceeds admitted range"));
            }
        }
        Ok(runs)
    }
}
