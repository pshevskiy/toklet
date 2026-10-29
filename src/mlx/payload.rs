//! Small cold payload checks shared by the two concrete Gemma loaders.
//! No shared store, family dispatch or numerical execution policy lives here.
use super::affine::{Layout, Matrix, MetadataDtype};
use crate::weights::Error;
use half::bf16;
use sha2::{Digest, Sha256};
use std::io::Read;

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
pub(super) fn buffer(size: usize) -> Result<Vec<u8>, Error> {
    if size > isize::MAX as usize {
        return Err(invalid("Gemma payload allocation bound"));
    }
    let mut v = Vec::new();
    v.try_reserve_exact(size).map_err(|_| Error::Allocation)?;
    v.resize(size, 0);
    Ok(v)
}
pub(super) fn read_exact<R: Read>(
    reader: &mut R,
    bytes: &mut [u8],
    sha: &mut Sha256,
) -> Result<(), Error> {
    reader
        .read_exact(bytes)
        .map_err(|e| invalid(format!("Gemma payload read: {e}")))?;
    sha.update(bytes);
    Ok(())
}
pub(super) fn validate_q4_bf16(
    layout: Layout,
    weight: &[u8],
    scales: &[u8],
    biases: &[u8],
) -> Result<(), Error> {
    if layout.bits() != 4 || layout.metadata_dtype() != MetadataDtype::BF16 {
        return Err(invalid("Gemma payload packed representation"));
    }
    Matrix::from_le_bytes(layout, weight, scales, biases).map_err(|e| invalid(e.0))?;
    // Preserve the existing Gemma4 stronger endpoint check, including unused
    // codes: finite FP32 reconstruction at 0/15 covers every intervening code.
    for (s, b) in scales.chunks_exact(2).zip(biases.chunks_exact(2)) {
        let scale = f64::from(bf16::from_le_bytes(s.try_into().unwrap()).to_f32());
        let bias = f64::from(bf16::from_le_bytes(b.try_into().unwrap()).to_f32());
        for code in [0., 15.] {
            if !((scale * code + bias) as f32).is_finite() {
                return Err(invalid("Gemma affine reconstruction overflow"));
            }
        }
    }
    Ok(())
}
