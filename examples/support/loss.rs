//! Metal worker scoring policy and stable FP64 math.
//! Mirrors the self-contained independent CPU scorer; no backend execution.
use sha2::{Digest, Sha256};
use std::{io::Read, ops::Range, path::Path};

pub(super) const TARGETS: usize = 10_000;
pub(super) const BLOCK: usize = 1024;
pub(super) const CAPACITY: usize = BLOCK + 1;
pub(super) const APP_BYTES: usize = 64 * 1024 * 1024;
pub(super) const POLICY: &str =
    "independent_blocks_of_up_to_1025_tokens_one_token_overlap_score_every_target_once";

#[derive(Default)]
pub(super) struct Sum {
    pub(super) value: f64,
    correction: f64,
}
impl Sum {
    pub(super) fn add(&mut self, value: f64) {
        let adjusted = value - self.correction;
        let next = self.value + adjusted;
        self.correction = (next - self.value) - adjusted;
        self.value = next;
    }
}

pub(super) fn loss(logits: &[f32], target: usize) -> Result<f64, &'static str> {
    if logits.is_empty() || target >= logits.len() {
        return Err("loss target or vocabulary outside bounds");
    }
    let mut maximum = f64::NEG_INFINITY;
    for &value in logits {
        if !value.is_finite() {
            return Err("loss logits must be finite");
        }
        maximum = maximum.max(f64::from(value));
    }
    let mut exponential = Sum::default();
    for &value in logits {
        exponential.add((f64::from(value) - maximum).exp());
    }
    // Subtract before adding log(sum), retaining equal-logit loss even at MAX.
    let result = (maximum - f64::from(logits[target])) + exponential.value.ln();
    if !result.is_finite() || result < 0.0 {
        return Err("loss reduction produced an invalid result");
    }
    Ok(result)
}

pub(super) fn spans(targets: usize) -> impl Iterator<Item = Range<usize>> {
    (0..targets)
        .step_by(BLOCK)
        .map(move |start| start..start.saturating_add(BLOCK).min(targets))
}

pub(super) fn tokens(path: &Path) -> Result<(Vec<u32>, String), Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let size = file.metadata()?.len();
    let limit = (TARGETS + 1) * size_of::<u32>();
    if !(8..=limit as u64).contains(&size) || !size.is_multiple_of(4) {
        return Err("loss input must contain 2..10001 little-endian UInt32 tokens".into());
    }
    let mut data = Vec::new();
    file.take((limit + 1) as u64).read_to_end(&mut data)?;
    if data.len() as u64 != size {
        return Err("loss input changed size during bounded read".into());
    }
    let sha = format!("{:x}", Sha256::digest(&data));
    let ids = data
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect();
    Ok((ids, sha))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_loss_matches_manual_distributions_and_extremes() {
        assert_eq!(loss(&[42.0], 0).unwrap(), 0.0);
        assert!((loss(&[0.0, 0.0], 1).unwrap() - 2.0_f64.ln()).abs() < 1e-15);
        assert!((loss(&[f32::MAX; 3], 2).unwrap() - 3.0_f64.ln()).abs() < 1e-15);
        assert_eq!(loss(&[1000.0, -1000.0], 1).unwrap(), 2000.0);
        assert_eq!(
            loss(&[f32::MAX, -f32::MAX], 1).unwrap(),
            2.0 * f64::from(f32::MAX)
        );
        let expected = 1.0 + (-1.0_f64).exp().ln_1p();
        assert!((loss(&[-1000.0, -999.0], 0).unwrap() - expected).abs() < 1e-15);
        assert_eq!(loss(&[-1000.0, -999.0], 0), loss(&[1000.0, 1001.0], 0));
    }

    #[test]
    fn invalid_distributions_and_targets_fail_closed() {
        assert!(loss(&[], 0).is_err());
        assert!(loss(&[1.0], 1).is_err());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(loss(&[1.0, value], 0).is_err());
        }
    }

    #[test]
    fn frozen_blocks_score_each_target_once_with_one_token_overlap() {
        let blocks: Vec<_> = spans(TARGETS).collect();
        assert_eq!(blocks.len(), 10);
        assert_eq!(blocks[0], 0..1024);
        assert_eq!(blocks[1], 1024..2048);
        assert_eq!(blocks[9], 9216..10000);
        assert_eq!(
            blocks.iter().flat_map(Clone::clone).collect::<Vec<_>>(),
            (0..TARGETS).collect::<Vec<_>>()
        );
        for range in blocks {
            // Token slice is inclusive of both the initial prefix and target:
            // [start, end] has <=1025 IDs. Adjacent blocks share only token end.
            assert!(range.len() < CAPACITY);
        }
    }
}
