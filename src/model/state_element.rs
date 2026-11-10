//! Sealed storage types for the two concrete CPU state-owner profiles.
//! No numerical arithmetic or runtime dtype selection is defined here.
use super::Error;
use crate::{capacity::KvStorage, cpu};
use half::bf16;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for half::bf16 {}
}

/// Storage-only specialization for concrete FP32 and BF16 cache owners.
/// Sealing bounds reuse to the two implemented formats; it grants no model or
/// numerical-backend support. Selection is static at the owner boundary.
pub trait StateElement: sealed::Sealed + Copy {
    const ZERO: Self;
    const STORAGE: KvStorage;
    fn finite(tensor: &'static str, values: &[Self]) -> Result<(), cpu::Error>;
}

impl StateElement for f32 {
    const ZERO: Self = 0.;
    const STORAGE: KvStorage = KvStorage::F32;
    fn finite(tensor: &'static str, values: &[Self]) -> Result<(), cpu::Error> {
        cpu::finite(tensor, values)
    }
}
impl StateElement for bf16 {
    const ZERO: Self = bf16::ZERO;
    const STORAGE: KvStorage = KvStorage::Bf16;
    fn finite(tensor: &'static str, values: &[Self]) -> Result<(), cpu::Error> {
        if let Some(index) = values.iter().position(|v| !v.is_finite()) {
            Err(cpu::Error::NonFinite { tensor, index })
        } else {
            Ok(())
        }
    }
}

pub(super) fn allocation_bytes<T: StateElement>(count: usize) -> Result<usize, Error> {
    let bytes = cpu::product(count, std::mem::size_of::<T>())?;
    if bytes > isize::MAX as usize {
        return Err(cpu::Error::SizeOverflow.into());
    }
    Ok(bytes)
}
pub(super) fn total_bytes<T: StateElement>(counts: &[usize]) -> Result<usize, Error> {
    counts.iter().try_fold(0_usize, |total, &count| {
        total
            .checked_add(allocation_bytes::<T>(count)?)
            .ok_or(cpu::Error::SizeOverflow.into())
    })
}
pub(super) fn zeros<T: StateElement>(count: usize) -> Result<Vec<T>, Error> {
    allocation_bytes::<T>(count)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Allocation)?;
    values.resize(count, T::ZERO);
    Ok(values)
}
