use crate::cpu::{Error, finite, nonzero};

/// Select the largest finite logit, choosing the lowest token ID on ties.
/// No allocation, probability normalization, or token-history mutation occurs.
pub fn greedy(logits: &[f32]) -> Result<u32, Error> {
    nonzero(logits.len(), "vocabulary")?;
    if logits.len() - 1 > u32::MAX as usize {
        return Err(Error::InvalidParameter("vocabulary IDs must fit u32"));
    }
    finite("logits", logits)?;
    let mut best = 0;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    Ok(best as u32)
}
