//! Half-split YaRN for SenseNova text, where all three MRoPE positions agree.

use super::{Error, TEXT, YARN};
use half::bf16;
use std::array;
use std::f64::consts::TAU;

const ROTATED: usize = (TEXT.head_dim as f64 * YARN.partial_rotary_factor) as usize;
const PAIRS: usize = ROTATED / 2;

/// Precomputed release-specific inverse frequencies.
pub struct TextRotary {
    inverse: [f32; PAIRS],
    scale: f32,
}

impl TextRotary {
    pub fn new() -> Self {
        let correction = |turns: f64| {
            ROTATED as f64 * (YARN.original_max_position_embeddings as f64 / (turns * TAU)).ln()
                / (2.0 * YARN.rope_theta.ln())
        };
        let low = correction(32.0).floor().max(0.0) as f32;
        let high = correction(1.0).ceil().min((ROTATED - 1) as f64) as f32;
        let theta = YARN.rope_theta as f32;
        let factor = YARN.factor as f32;
        let inverse = array::from_fn(|lane| {
            let wavelength = theta.powf((2 * lane) as f32 / ROTATED as f32);
            let extrapolation = 1.0 - ((lane as f32 - low) / (high - low)).clamp(0.0, 1.0);
            (1.0 - extrapolation) * (1.0 / (factor * wavelength))
                + extrapolation * (1.0 / wavelength)
        });
        Self {
            inverse,
            scale: (1.0 + 0.1 * YARN.factor.ln()) as f32,
        }
    }

    /// Apply one text position to normalized, contiguous head-major FP32 Q and K.
    /// The session owns position admission and any BF16 cast boundaries.
    pub fn apply(&self, position: usize, q: &mut [f32], k: &mut [f32]) -> Result<(), Error> {
        self.apply_mode::<false>(position, q, k)
    }

    /// Store normalized Q/K, scaled phase coefficients and rotated outputs at
    /// BF16 boundaries while keeping the borrowed cache storage interface FP32.
    pub fn apply_bf16(&self, position: usize, q: &mut [f32], k: &mut [f32]) -> Result<(), Error> {
        self.apply_mode::<true>(position, q, k)
    }

    fn apply_mode<const BF16: bool>(
        &self,
        position: usize,
        q: &mut [f32],
        k: &mut [f32],
    ) -> Result<(), Error> {
        if q.len() != TEXT.num_attention_heads * TEXT.head_dim
            || k.len() != TEXT.num_key_value_heads * TEXT.head_dim
        {
            return Err(Error::InvalidConfig(
                "SenseNova text rotary Q/K shape differs",
            ));
        }
        let store = |value| {
            if BF16 {
                bf16::from_f32(value).to_f32()
            } else {
                value
            }
        };
        let phases: [(f32, f32); PAIRS] = self.inverse.map(|inverse| {
            let (sin, cos) = (position as f32 * inverse).sin_cos();
            (store(sin * self.scale), store(cos * self.scale))
        });
        for head in q
            .chunks_exact_mut(TEXT.head_dim)
            .chain(k.chunks_exact_mut(TEXT.head_dim))
        {
            if BF16 {
                head.iter_mut().for_each(|value| *value = store(*value));
            }
            let (first, second) = head[..ROTATED].split_at_mut(PAIRS);
            for ((a, b), (sin, cos)) in first.iter_mut().zip(second).zip(phases) {
                let (x, y) = (*a, *b);
                *a = store(x * cos - y * sin);
                *b = store(y * cos + x * sin);
            }
        }
        Ok(())
    }
}

impl Default for TextRotary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_position_zero_rounds_phase_and_full_normalized_heads() {
        let rotary = TextRotary::new();
        let mut q = vec![1.; TEXT.num_attention_heads * TEXT.head_dim];
        let mut k = vec![1.; TEXT.num_key_value_heads * TEXT.head_dim];
        q[ROTATED] = 1.001;
        k[ROTATED] = 1.001;
        rotary.apply_bf16(0, &mut q, &mut k).unwrap();
        for head in q
            .chunks_exact(TEXT.head_dim)
            .chain(k.chunks_exact(TEXT.head_dim))
        {
            assert!(head[..ROTATED].iter().all(|&lane| lane == 1.0703125));
            assert!(head[ROTATED..].iter().all(|&lane| lane == 1.0));
        }
    }

    #[test]
    fn text_position_yarn_matches_reference_pairs_and_preserves_suffix() {
        let rotary = TextRotary::new();
        let lanes = [0, 14, 18, 22, 31];
        for (position, expected) in [
            (
                0,
                [
                    (0.033416085, -0.050124127),
                    (0.50124127, -0.28403673),
                    (0.63490564, -0.35086888),
                    (0.76856995, -0.41770107),
                    (1.0693147, -0.56807345),
                ],
            ),
            (
                1,
                [
                    (0.060232785, 0.0010364843),
                    (0.5014871, -0.28360257),
                    (0.634936, -0.3508139),
                    (0.76857316, -0.41769513),
                    (1.0693147, -0.56807333),
                ],
            ),
            (
                257,
                [
                    (-0.0013610449, -0.060226325),
                    (0.5515719, -0.1663975),
                    (0.6425575, -0.33665115),
                    (0.769395, -0.4161794),
                    (1.0693268, -0.5680507),
                ],
            ),
        ] {
            let mut input = vec![-0.0; TEXT.head_dim];
            for lane in 0..PAIRS {
                input[lane] = (lane + 1) as f32 / 32.0;
                input[lane + PAIRS] = -((lane + 3) as f32) / 64.0;
            }
            let mut q = input.repeat(TEXT.num_attention_heads);
            let mut k = input
                .iter()
                .map(|&x| -2.0 * x)
                .collect::<Vec<_>>()
                .repeat(TEXT.num_key_value_heads);
            let k_initial = k.clone();
            rotary.apply(position, &mut q, &mut k).unwrap();
            for head in q.chunks_exact(TEXT.head_dim) {
                for (lane, (first, second)) in lanes.into_iter().zip(expected) {
                    assert!((head[lane] - first).abs() < 2e-7);
                    assert!((head[lane + PAIRS] - second).abs() < 2e-7);
                }
            }
            for (head, original) in k
                .chunks_exact(TEXT.head_dim)
                .zip(k_initial.chunks_exact(TEXT.head_dim))
            {
                for (lane, (first, second)) in lanes.into_iter().zip(expected) {
                    assert!((head[lane] + 2.0 * first).abs() < 4e-7);
                    assert!((head[lane + PAIRS] + 2.0 * second).abs() < 4e-7);
                }
                assert!(
                    head[ROTATED..]
                        .iter()
                        .zip(&original[ROTATED..])
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
            assert!(
                q.chunks_exact(TEXT.head_dim)
                    .flat_map(|head| &head[ROTATED..])
                    .all(|value| value.to_bits() == (-0.0f32).to_bits())
            );
        }
    }

    #[test]
    fn malformed_key_extent_leaves_query_unchanged() {
        let mut q = vec![1.0; TEXT.num_attention_heads * TEXT.head_dim];
        let mut k = vec![1.0; TEXT.num_key_value_heads * TEXT.head_dim - 1];
        assert!(TextRotary::new().apply(1, &mut q, &mut k).is_err());
        assert!(q.iter().chain(&k).all(|&value| value == 1.0));
    }
}
