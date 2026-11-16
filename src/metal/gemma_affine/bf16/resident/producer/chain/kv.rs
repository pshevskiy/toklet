use super::*;
pub(super) struct CacheStorage {
    pub k: Buffer<u32>,
    pub v: Buffer<u32>,
    pub d: usize,
    pub capacity: usize,
    pub window: Option<usize>,
    pub owner: u64,
    pub generation: u64,
    pub offset: usize,
}
impl CacheStorage {
    pub fn planned_bytes(d: usize, capacity: usize) -> Result<usize> {
        capacity
            .checked_mul(d)
            .and_then(|n| n.checked_add(GUARD))
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .ok_or(Error::Options("chain KV byte overflow"))
    }
    pub fn new(
        gpu: &Gpu,
        d: usize,
        capacity: usize,
        window: Option<usize>,
        owner: u64,
    ) -> Result<Self> {
        let mut k = Buffer::new(gpu, capacity * d + GUARD)?;
        let mut v = Buffer::new(gpu, capacity * d + GUARD)?;
        k.with_mut(|s| s.fill(SENTINEL));
        v.with_mut(|s| s.fill(SENTINEL));
        Ok(Self {
            k,
            v,
            d,
            capacity,
            window,
            owner,
            generation: 0,
            offset: 0,
        })
    }
    pub fn bytes(&self) -> usize {
        size_of::<Self>() + (self.k.len + self.v.len) * 4
    }
    pub fn validate(
        &self,
        gpu: &Gpu,
        d: usize,
        capacity: usize,
        window: Option<usize>,
    ) -> Result<()> {
        if self.d != d
            || self.capacity != capacity
            || self.window != window
            || std::ptr::eq(&*self.k.raw, &*self.v.raw)
        {
            return Err(Error::Options("chain KV owner geometry differs"));
        }
        for b in [&self.k, &self.v] {
            validate_storage(gpu, b, capacity * d + GUARD)?;
        }
        Ok(())
    }
    pub fn guards(&self, bytes: &mut usize) -> bool {
        tail(&self.k, self.capacity * self.d, bytes) & tail(&self.v, self.capacity * self.d, bytes)
    }
    pub fn segments<'a>(
        &'a self,
        frame: NodeFrame,
        k: BufferRange<'a>,
        v: BufferRange<'a>,
    ) -> Result<[attention::DeviceSegment<'a>; 3]> {
        let old = window_rows::old_spans(self.capacity, frame.proposal_start, self.window);
        let segment = |i: usize| -> Result<attention::DeviceSegment<'a>> {
            let s = old[i];
            Ok(attention::DeviceSegment {
                k: BufferRange::new(&self.k, s.slot * self.d, s.rows * self.d)?,
                v: BufferRange::new(&self.v, s.slot * self.d, s.rows * self.d)?,
                keys: s.rows,
                head_stride: s.rows * self.d,
                key_start: s.position,
            })
        };
        if k.len != frame.proposal_rows * self.d || v.len != k.len {
            return Err(Error::Options("chain donor proposal extent differs"));
        }
        Ok([
            segment(0)?,
            segment(1)?,
            attention::DeviceSegment {
                k,
                v,
                keys: frame.proposal_rows,
                head_stride: frame.proposal_rows * self.d,
                key_start: frame.proposal_start,
            },
        ])
    }
    pub fn runs(&self, frame: &GraphFrame) -> Result<[window_rows::CopyRun; 2]> {
        if frame.rows == 0
            || frame.offset.checked_add(frame.rows) != Some(frame.end)
            || frame.end > self.capacity
        {
            return Err(Error::Options("chain KV copy frame exceeds capacity"));
        }
        let runs = window_rows::copy_runs(
            self.capacity,
            frame.offset,
            frame.rows,
            self.window.is_some(),
        );
        for r in runs {
            if r.source_row
                .checked_add(r.rows)
                .is_none_or(|n| n > frame.rows)
                || r.slot.checked_add(r.rows).is_none_or(|n| n > self.capacity)
            {
                return Err(Error::Options("chain KV copy run exceeds storage"));
            }
        }
        Ok(runs)
    }
}
