//! Shared cold staging and original-byte upload for concrete packed owners.
use super::*;

pub(super) struct SourceScratch {
    pub(super) arrays: [Vec<u8>; 3],
}
impl SourceScratch {
    pub(super) fn new(lengths: [usize; 3]) -> Result<Self> {
        let reserve = |length| -> Result<Vec<u8>> {
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(length)
                .map_err(|_| Error::Native("stream source reservation failed".into()))?;
            bytes.resize(length, 0);
            Ok(bytes)
        };
        Ok(Self {
            arrays: [
                reserve(lengths[0])?,
                reserve(lengths[1])?,
                reserve(lengths[2])?,
            ],
        })
    }
    pub(super) fn capacities(&self) -> [usize; 3] {
        self.arrays.each_ref().map(Vec::capacity)
    }
}
pub(super) fn upload(gpu: &Gpu, bytes: &[u8]) -> Result<Buffer<u32>> {
    let mut buffer = Buffer::new(gpu, bytes.len().div_ceil(4) + GUARD)?;
    buffer.with_mut(|out| {
        out.fill(SENTINEL);
        for (word, bytes) in out.iter_mut().zip(bytes.chunks(4)) {
            let mut raw = [0; 4];
            raw[..bytes.len()].copy_from_slice(bytes);
            *word = u32::from_le_bytes(raw);
        }
    });
    Ok(buffer)
}
