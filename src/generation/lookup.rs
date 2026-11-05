//! Bounded prompt lookup over an already committed history.
use super::Error;

pub(super) const MAX_MATCH: usize = 8;
pub(super) const MAX_PROPOSAL: usize = 8;

/// Copy known continuation IDs for the longest matching suffix, breaking
/// equal-length ties in favor of the most recent eligible occurrence.
/// Both the previous match and the copied continuation end before the current
/// suffix. Supply committed history only. No-match and invalid requests leave
/// the destination untouched, and the return count excludes its unused tail.
pub fn prompt_lookup(
    committed: &[u32],
    max_match: usize,
    proposal: &mut [u32],
) -> Result<usize, Error> {
    if max_match > MAX_MATCH {
        return Err(Error::InvalidRequest(
            "prompt lookup match limit exceeds eight",
        ));
    }
    if proposal.is_empty() {
        return Ok(0);
    }
    // Two disjoint n-grams and one known continuation require at least 2n+1.
    let max_match = max_match.min(committed.len().saturating_sub(1) / 2);
    for length in (1..=max_match).rev() {
        let suffix_start = committed.len() - length;
        let suffix = &committed[suffix_start..];
        let earlier = &committed[..suffix_start];
        if let Some(start) = earlier[..earlier.len() - 1]
            .windows(length)
            .rposition(|candidate| candidate == suffix)
        {
            let continuation = &earlier[start + length..];
            let count = continuation.len().min(proposal.len()).min(MAX_PROPOSAL);
            proposal[..count].copy_from_slice(&continuation[..count]);
            return Ok(count);
        }
    }
    Ok(0)
}
