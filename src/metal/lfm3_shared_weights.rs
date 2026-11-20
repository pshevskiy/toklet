//! Exclusive construction and immutable native bindings for selected LFM3 weights.
use crate::weights::Error;
use half::bf16;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLDevice, MTLHazardTrackingMode, MTLResource,
    MTLResourceOptions, MTLStorageMode,
};

const _: () = {
    assert!(core::mem::size_of::<bf16>() == core::mem::size_of::<u16>());
    assert!(core::mem::align_of::<bf16>() == core::mem::align_of::<u16>());
    assert!(cfg!(target_endian = "little"));
};

fn bad(reason: &str) -> Error {
    Error::Invalid(reason.into())
}

pub(crate) struct BankBuilder {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    words: usize,
    delivered: usize,
}

pub(crate) struct Bank {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    words: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct BankRef<'a>(&'a Bank);

impl BankBuilder {
    pub(crate) fn new(device: &ProtocolObject<dyn MTLDevice>, words: usize) -> Result<Self, Error> {
        let bytes = words
            .checked_mul(2)
            .ok_or_else(|| bad("shared BF16 extent overflow"))?;
        if bytes == 0 || bytes > device.maxBufferLength() || bytes > isize::MAX as usize {
            return Err(bad("shared BF16 extent outside device limits"));
        }
        let raw = device
            .newBufferWithLength_options(
                bytes,
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::HazardTrackingModeTracked,
            )
            .ok_or(Error::Allocation)?;
        if raw.length() != bytes
            || raw.storageMode() != MTLStorageMode::Shared
            || raw.hazardTrackingMode() != MTLHazardTrackingMode::Tracked
            || !std::ptr::eq(&*raw.device(), device)
            || !(raw.contents().as_ptr() as usize).is_multiple_of(core::mem::align_of::<bf16>())
        {
            return Err(bad("shared BF16 buffer contract"));
        }
        Ok(Self {
            raw,
            words,
            delivered: 0,
        })
    }

    pub(crate) fn disjoint(&self, other: &Self) -> bool {
        if std::ptr::eq(&*self.raw, &*other.raw) {
            return false;
        }
        let a = self.raw.contents().as_ptr() as usize;
        let b = other.raw.contents().as_ptr() as usize;
        a.checked_add(self.words * 2).is_some_and(|end| end <= b)
            || b.checked_add(other.words * 2).is_some_and(|end| end <= a)
    }

    pub(crate) fn write(&mut self, start: usize, bytes: &[u8]) -> Result<(), Error> {
        let end = start
            .checked_add(bytes.len() / 2)
            .ok_or_else(|| bad("shared BF16 cursor overflow"))?;
        if start != self.delivered || !bytes.len().is_multiple_of(2) || end > self.words {
            return Err(bad("shared BF16 delivery cursor differs"));
        }
        let base = self.raw.contents().as_ptr().cast::<u16>();
        for (i, pair) in bytes.chunks_exact(2).enumerate() {
            let word = u16::from_le_bytes([pair[0], pair[1]]);
            if !bf16::from_bits(word).is_finite() {
                return Err(bad("nonfinite shared BF16 weight"));
            }
            // SAFETY: Every index is in the exclusively owned aligned destination.
            // No references are constructed to its possibly uninitialized bytes.
            unsafe { base.add(start + i).write(word) };
        }
        self.delivered = end;
        Ok(())
    }

    pub(crate) fn seal(self) -> Result<Bank, Error> {
        if self.delivered != self.words {
            return Err(bad("incomplete shared BF16 bank"));
        }
        Ok(Bank {
            raw: self.raw,
            words: self.words,
        })
    }
}

impl Bank {
    pub(crate) fn words(&self) -> &[bf16] {
        // SAFETY: pinned half::bf16 is repr(transparent) over u16. The consumed
        // builder wrote every aligned word. Native callers may only bind as
        // read-only. Allocation and slice lifetime are both tied to this owner.
        unsafe { std::slice::from_raw_parts(self.raw.contents().as_ptr().cast(), self.words) }
    }

    pub(crate) fn view(&self) -> BankRef<'_> {
        BankRef(self)
    }
}

impl BankRef<'_> {
    pub(crate) fn aliases(self, weights: &[bf16]) -> bool {
        let words = self.0.words();
        words.len() == weights.len() && std::ptr::eq(words.as_ptr(), weights.as_ptr())
    }

    pub(crate) fn words(self) -> usize {
        self.0.words
    }

    pub(crate) fn identity(self) -> usize {
        &*self.0.raw as *const ProtocolObject<dyn MTLBuffer> as usize
    }

    pub(crate) fn device_matches(self, device: &ProtocolObject<dyn MTLDevice>) -> bool {
        std::ptr::eq(&*self.0.raw.device(), device)
    }

    /// Bind the full bank at byte offset zero.
    ///
    /// # Safety
    /// The kernel must never write or alias writable storage to this resource.
    /// The encoder uses this device, with correct contiguous dimensions and a
    /// command that retains resources through completion.
    pub(crate) unsafe fn bind_read_only(
        self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        index: usize,
    ) {
        unsafe { encoder.setBuffer_offset_atIndex(Some(&self.0.raw), 0, index) };
    }
}
