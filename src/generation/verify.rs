//! Strict greedy comparison over target rows before any session mutation.
use super::Error;
use crate::sampling;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifiedNext {
    Token(u32),
    Stop(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GreedyVerification {
    /// Matching visible proposal IDs. A matched stop stays pending.
    pub accepted: usize,
    /// The first correction, bonus, or hidden stop from the target.
    pub next: VerifiedNext,
}

/// Row zero scores the committed prefix. Each subsequent row scores the
/// prefix after one more proposal, with one final row for the bonus decision.
/// Validate the entire tentative batch even if an earlier decision ends it.
/// This only compares supplied decisions; the caller must establish that each
/// row came from the correct target prefix and handle cache commit or replay.
pub fn verify_greedy(
    proposals: &[u32],
    logits: &[f32],
    vocabulary: usize,
    stop_tokens: &[u32],
) -> Result<GreedyVerification, Error> {
    let row_count = proposals
        .len()
        .checked_add(1)
        .ok_or(Error::InvalidRequest("proposal count overflow"))?;
    if vocabulary == 0
        || vocabulary - 1 > u32::MAX as usize
        || row_count.checked_mul(vocabulary) != Some(logits.len())
    {
        return Err(Error::InvalidRequest("invalid target logits shape"));
    }
    if proposals
        .iter()
        .chain(stop_tokens)
        .any(|&token| token as usize >= vocabulary)
    {
        return Err(Error::InvalidRequest(
            "proposal or stop token out of vocabulary",
        ));
    }
    let mut outcome = None;
    for (index, row) in logits.chunks_exact(vocabulary).enumerate() {
        let selected = sampling::greedy(row).map_err(Error::Sampling)?;
        if outcome.is_some() {
            continue;
        }
        let stop = stop_tokens.contains(&selected);
        if stop || proposals.get(index) != Some(&selected) {
            outcome = Some(GreedyVerification {
                accepted: index,
                next: if stop {
                    VerifiedNext::Stop(selected)
                } else {
                    VerifiedNext::Token(selected)
                },
            });
        }
    }
    // The mandatory bonus row always ends a fully matching proposal.
    Ok(outcome.expect("verified target batch includes a bonus row"))
}
