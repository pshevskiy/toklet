use super::*;

#[derive(Clone, Copy)]
pub(super) struct LayerLayout {
    pub regions: [Region; 32],
    pub words: usize,
    pub rows: usize,
    pub keys: usize,
    pub d: usize,
    pub ff: usize,
    pub producer: bool,
}
impl LayerLayout {
    pub fn new(rows: usize, keys: usize, d: usize, ff: usize, producer: bool) -> Result<Self> {
        if !(1..=32).contains(&rows)
            || !(1..=512).contains(&keys)
            || ![256, 512].contains(&d)
            || ff != if producer { 6144 } else { 12288 }
        {
            return Err(Error::Options("chain layer layout outside admitted shape"));
        }
        let mut cursor = 0;
        let mut regions = [Region { start: 0, len: 0 }; 32];
        for state in STATES {
            let len = recipe::values(state, rows, keys, d, ff, producer);
            regions[state as usize] = Region { start: cursor, len };
            cursor = cursor
                .checked_add(len)
                .and_then(|x| x.checked_add(GUARD))
                .ok_or(Error::Options("chain arena overflow"))?;
        }
        cursor
            .checked_mul(4)
            .ok_or(Error::Options("chain arena byte overflow"))?;
        Ok(Self {
            regions,
            words: cursor,
            rows,
            keys,
            d,
            ff,
            producer,
        })
    }
    pub fn active(self, state: State, frame: NodeFrame) -> usize {
        recipe::values(
            state,
            frame.rows,
            frame.keys,
            self.d,
            self.ff,
            self.producer,
        )
    }
    pub fn region(self, state: State) -> Region {
        self.regions[state as usize]
    }
    pub fn range(
        self,
        buffer: &Buffer<u32>,
        state: State,
        frame: NodeFrame,
    ) -> Result<BufferRange<'_>> {
        let n = self.active(state, frame);
        self.region(state).range(buffer, n)
    }
    pub fn states(
        self,
        buffer: &Buffer<u32>,
        frame: NodeFrame,
    ) -> Result<[Option<BufferRange<'_>>; 32]> {
        let mut out = [None; 32];
        for state in STATES {
            if self.active(state, frame) != 0 {
                out[state as usize] = Some(self.range(buffer, state, frame)?);
            }
        }
        Ok(out)
    }
    pub fn rms_plan(self) -> ([Option<Region>; 8], usize) {
        let mut cursor = GUARD;
        let regions = std::array::from_fn(|slot| {
            if !self.producer && (slot == 2 || slot == 7) {
                return None;
            }
            let len = self.rows * if slot == 1 { 8 } else { 1 };
            let region = Region { start: cursor, len };
            cursor = region.end() + GUARD;
            Some(region)
        });
        (regions, cursor)
    }
    pub fn rms_reasons(
        self,
        buffer: &Buffer<u32>,
        frame: NodeFrame,
    ) -> Result<[Option<BufferRange<'_>>; 8]> {
        let (regions, _) = self.rms_plan();
        let mut out = [None; 8];
        for (slot, region) in regions.into_iter().enumerate() {
            if let Some(region) = region {
                out[slot] = Some(region.range(buffer, frame.rows * if slot == 1 { 8 } else { 1 })?);
            }
        }
        Ok(out)
    }
    pub fn check_frame(self, frame: NodeFrame) -> Result<()> {
        if frame.rows == 0
            || frame.rows > self.rows
            || frame.keys == 0
            || frame.keys > self.keys
            || frame.proposal_rows == 0
            || frame.proposal_rows > self.rows
            || frame.key_start > frame.query_start
            || frame.query_start.checked_add(frame.rows).is_none_or(|end| {
                frame
                    .key_start
                    .checked_add(frame.keys)
                    .is_none_or(|limit| end > limit)
            })
        {
            return Err(Error::Options("chain active frame exceeds layout"));
        }
        Ok(())
    }
}
pub(super) struct LayerStorage {
    pub arena: Buffer<u32>,
    pub status: Buffer<u32>,
    pub flags: [Buffer<u32>; 2],
    pub intervals: [Buffer<u32>; 2],
    pub rms_reasons: Buffer<u32>,
}
impl LayerStorage {
    pub fn planned_bytes(layout: LayerLayout) -> Result<usize> {
        let lane = layout.rows * (layout.ff + P);
        let words = layout
            .words
            .checked_add(STATUS_WORDS + GUARD)
            .and_then(|n| n.checked_add(9 * lane + 4 * GUARD))
            .and_then(|n| n.checked_add(layout.rms_plan().1))
            .ok_or(Error::Options("chain layer storage overflow"))?;
        words
            .checked_mul(4)
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .ok_or(Error::Options("chain layer byte overflow"))
    }
    pub fn new(gpu: &Gpu, layout: LayerLayout) -> Result<Self> {
        Ok(Self {
            arena: Buffer::new(gpu, layout.words)?,
            status: Buffer::new(gpu, STATUS_WORDS + GUARD)?,
            flags: [
                Buffer::new(gpu, layout.rows * layout.ff + GUARD)?,
                Buffer::new(gpu, layout.rows * P + GUARD)?,
            ],
            intervals: [
                Buffer::new(gpu, 8 * layout.rows * layout.ff + GUARD)?,
                Buffer::new(gpu, 8 * layout.rows * P + GUARD)?,
            ],
            rms_reasons: Buffer::new(gpu, layout.rms_plan().1)?,
        })
    }
    pub fn buffers(&self) -> [&Buffer<u32>; 7] {
        [
            &self.arena,
            &self.status,
            &self.flags[0],
            &self.flags[1],
            &self.intervals[0],
            &self.intervals[1],
            &self.rms_reasons,
        ]
    }
    pub fn bytes(&self) -> usize {
        size_of::<Self>() + self.buffers().into_iter().map(|b| b.len * 4).sum::<usize>()
    }
    pub fn validate(&self, gpu: &Gpu, layout: LayerLayout) -> Result<()> {
        let lengths = [
            layout.words,
            STATUS_WORDS + GUARD,
            layout.rows * layout.ff + GUARD,
            layout.rows * P + GUARD,
            8 * layout.rows * layout.ff + GUARD,
            8 * layout.rows * P + GUARD,
            layout.rms_plan().1,
        ];
        for (buffer, len) in self.buffers().into_iter().zip(lengths) {
            validate_storage(gpu, buffer, len)?;
        }
        Ok(())
    }
    pub fn reset(&mut self) {
        self.arena.with_mut(|s| s.fill(SENTINEL));
        self.status.with_mut(|s| {
            s.fill(SENTINEL);
            s[..STATUS_WORDS].fill(0);
        });
        self.rms_reasons.with_mut(|s| s.fill(SENTINEL));
        for b in self.flags.iter_mut().chain(&mut self.intervals) {
            b.with_mut(|s| s.fill(SENTINEL));
        }
    }
    pub fn guards(&self, layout: LayerLayout, frame: NodeFrame, bytes: &mut usize) -> bool {
        let mut valid = true;
        self.arena.with_slice(|s| {
            for state in STATES {
                let r = layout.region(state);
                valid &= sentinel(
                    &s[r.start + layout.active(state, frame)..r.end() + GUARD],
                    bytes,
                );
            }
        });
        valid &= tail(&self.status, STATUS_WORDS, bytes);
        self.rms_reasons.with_slice(|s| {
            valid &= sentinel(&s[..GUARD], bytes);
            for (slot, region) in layout.rms_plan().0.into_iter().enumerate() {
                if let Some(region) = region {
                    let active = frame.rows * if slot == 1 { 8 } else { 1 };
                    valid &= sentinel(&s[region.start + active..region.end() + GUARD], bytes);
                }
            }
        });
        for (i, width) in [layout.ff, P].into_iter().enumerate() {
            valid &= tail(&self.flags[i], frame.rows * width, bytes);
            valid &= tail(&self.intervals[i], 8 * frame.rows * width, bytes);
        }
        valid
    }
    pub fn snapshot_rms(
        &self,
        layout: LayerLayout,
        frame: NodeFrame,
        out: &mut [u32; MAX_RMS_REASONS],
    ) -> Result<usize> {
        out.fill(0);
        self.rms_reasons.with_slice(|words| {
            let mut cursor = 0;
            for (slot, region) in layout.rms_plan().0.into_iter().enumerate() {
                if let Some(region) = region {
                    let active = frame.rows * if slot == 1 { 8 } else { 1 };
                    let values = &words[region.start..region.start + active];
                    if values
                        .iter()
                        .any(|&word| word & !rms::precision::KNOWN_REASON_BITS != 0)
                    {
                        return Err(Error::Options("chain RMS reason contract differs"));
                    }
                    out[cursor..cursor + active].copy_from_slice(values);
                    cursor += active;
                }
            }
            Ok(cursor * 4)
        })
    }
    pub fn finite(&self, layout: LayerLayout, frame: NodeFrame, bytes: &mut usize) -> bool {
        self.arena.with_slice(|s| {
            let mut valid = true;
            for state in STATES {
                let r = layout.region(state);
                let n = layout.active(state, frame);
                *bytes += n * 4;
                for &word in &s[r.start..r.start + n] {
                    valid &= f32::from_bits(word).is_finite();
                }
            }
            valid
        })
    }
}
pub(super) fn validate_storage(gpu: &Gpu, buffer: &Buffer<u32>, len: usize) -> Result<()> {
    if buffer.len != len
        || len.checked_mul(4) != Some(buffer.raw.length())
        || !std::ptr::eq(&*buffer.raw.device(), &*gpu.device)
        || buffer.raw.storageMode() != MTLStorageMode::Shared
        || buffer.raw.hazardTrackingMode() != MTLHazardTrackingMode::Tracked
    {
        return Err(Error::Options("chain physical storage contract differs"));
    }
    Ok(())
}
pub(super) fn sentinel(words: &[u32], bytes: &mut usize) -> bool {
    *bytes += words.len() * 4;
    words
        .iter()
        .fold(0, |different, &word| different | (word ^ SENTINEL))
        == 0
}
pub(super) fn tail(buffer: &Buffer<u32>, active: usize, bytes: &mut usize) -> bool {
    buffer.with_slice(|s| sentinel(&s[active..], bytes))
}
