//! Scalar two-stream mHC reference for the fixed SenseNova Loop release.
//! Coefficients stay FP32. The attention/FFN branch and residual operations
//! retain the BF16 storage boundaries of the published forward.

use super::super::Error;
use half::bf16;

const STREAMS: usize = 2;
const OUTPUTS: usize = 8;
const ITERATIONS: usize = 20;
const EPSILON: f32 = 1e-6;

pub struct Router<'a> {
    weight: &'a [f32],
    bias: &'a [f32; OUTPUTS],
    hidden: usize,
}

#[derive(Debug)]
pub struct Routing {
    post: [f32; STREAMS],
    mix: [[f32; STREAMS]; STREAMS],
}

fn invalid(reason: &'static str) -> Error {
    Error::InvalidConfig(reason)
}

fn store(value: f32) -> Result<bf16, Error> {
    let value = bf16::from_f32(value);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid("SenseNova mHC nonfinite BF16 output"))
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn lse(a: f32, b: f32) -> f32 {
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

fn sinkhorn(mut values: [[f32; STREAMS]; STREAMS]) -> [[f32; STREAMS]; STREAMS] {
    for _ in 0..ITERATIONS {
        for row in &mut values {
            let normalizer = lse(row[0], row[1]);
            for x in row {
                *x -= normalizer;
            }
        }
        let [top, bottom] = &mut values;
        for (a, b) in top.iter_mut().zip(bottom) {
            let normalizer = lse(*a, *b);
            *a -= normalizer;
            *b -= normalizer;
        }
    }
    values.map(|row| row.map(f32::exp))
}

impl<'a> Router<'a> {
    /// Borrow one original attention or feed-forward router's row-major weights.
    pub fn new(hidden: usize, weight: &'a [f32], bias: &'a [f32; OUTPUTS]) -> Result<Self, Error> {
        if hidden == 0
            || hidden.checked_mul(STREAMS * OUTPUTS) != Some(weight.len())
            || !weight.iter().chain(bias).all(|x| x.is_finite())
        {
            return Err(invalid("SenseNova mHC projection dimensions or parameters"));
        }
        Ok(Self {
            weight,
            bias,
            hidden,
        })
    }

    /// Route one BF16 token into the attention or feed-forward branch.
    pub fn route(
        &self,
        residual: [&[bf16]; STREAMS],
        branch: &mut [bf16],
    ) -> Result<Routing, Error> {
        if residual.iter().any(|row| row.len() != self.hidden)
            || branch.len() != self.hidden
            || !residual.into_iter().flatten().all(|word| word.is_finite())
        {
            return Err(invalid("SenseNova mHC residual/branch geometry or input"));
        }

        let mut square_sum = 0.0_f32;
        for value in residual.into_iter().flatten().copied().map(bf16::to_f32) {
            square_sum += value * value;
        }
        let mean = square_sum / (STREAMS * self.hidden) as f32;
        if !mean.is_finite() {
            return Err(invalid("SenseNova mHC nonfinite RMS"));
        }
        let inverse = (mean + EPSILON).sqrt().recip();
        let mut logits = [0.0_f32; OUTPUTS];
        for ((logit, weight), &bias) in logits
            .iter_mut()
            .zip(self.weight.chunks_exact(STREAMS * self.hidden))
            .zip(self.bias)
        {
            let mut sum = 0.0_f32;
            for (value, &weight) in residual.into_iter().flatten().zip(weight) {
                let normalized = value.to_f32() * inverse;
                sum += normalized * weight;
            }
            *logit = sum + bias;
        }
        if !logits.iter().all(|x| x.is_finite()) {
            return Err(invalid("SenseNova mHC nonfinite routing logits"));
        }
        let [pre0, pre1, post0, post1, mix00, mix01, mix10, mix11] = logits;
        let pre = [bf16::from_f32(sigmoid(pre0)), bf16::from_f32(sigmoid(pre1))];
        let post = [2.0 * sigmoid(post0), 2.0 * sigmoid(post1)];
        let mix = sinkhorn([[mix00, mix01], [mix10, mix11]]);
        if !mix.iter().flatten().all(|x| x.is_finite()) {
            return Err(invalid("SenseNova mHC nonfinite Sinkhorn result"));
        }
        for ((&a, &b), out) in residual[0].iter().zip(residual[1]).zip(branch) {
            *out = store(a.to_f32() * pre[0].to_f32() + b.to_f32() * pre[1].to_f32())?;
        }
        Ok(Routing { post, mix })
    }
}

impl Routing {
    /// Merge the BF16 branch output into the two input streams.
    pub fn merge(
        &self,
        residual: [&[bf16]; STREAMS],
        branch: &[bf16],
        output: [&mut [bf16]; STREAMS],
    ) -> Result<(), Error> {
        if branch.is_empty()
            || residual.iter().any(|row| row.len() != branch.len())
            || output.iter().any(|row| row.len() != branch.len())
            || !residual
                .into_iter()
                .flatten()
                .chain(branch)
                .all(|word| word.is_finite())
        {
            return Err(invalid("SenseNova mHC merge geometry or input"));
        }

        for ((out, &post), mix) in output.into_iter().zip(&self.post).zip(self.mix) {
            let gain = bf16::from_f32(post).to_f32();
            let [p0, p1] = mix.map(|coefficient| bf16::from_f32(coefficient).to_f32());
            for (((&a, &b), &update), value) in
                residual[0].iter().zip(residual[1]).zip(branch).zip(out)
            {
                let mixed = store(a.to_f32() * p0 + b.to_f32() * p1)?;
                let update = store(update.to_f32() * gain)?;
                *value = store(mixed.to_f32() + update.to_f32())?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod checkpoint;

#[cfg(test)]
mod tests {
    use super::*;

    fn words<const N: usize>(input: [f32; N]) -> [bf16; N] {
        input.map(bf16::from_f32)
    }

    #[test]
    fn separate_routers_match_asymmetric_two_stream_oracle() {
        let weight = [0.; 32];
        let log3 = 3.0_f32.ln();
        let attn_bias = [-log3, log3, 0., 0., log3, 0., 0., log3];
        let ffn_bias = [log3, -log3, 0., 0., 0., log3, log3, 0.];
        let attn = Router::new(2, &weight, &attn_bias).unwrap();
        let ffn = Router::new(2, &weight, &ffn_bias).unwrap();

        let a = words([4., -8.]);
        let b = words([20., 8.]);
        let mut branch = words([0.; 2]);
        let attn_routing = attn.route([&a, &b], &mut branch).unwrap();
        assert_eq!(branch, words([16., 4.]));
        let mut first = words([0.; 2]);
        let mut second = words([0.; 2]);
        attn_routing
            .merge([&a, &b], &words([2., 6.]), [&mut first, &mut second])
            .unwrap();
        assert_eq!(first, words([10., 2.]));
        assert_eq!(second, words([18., 10.]));

        let ffn_routing = ffn.route([&first, &second], &mut branch).unwrap();
        assert_eq!(branch, words([12., 4.]));
        let mut a = words([0.; 2]);
        let mut b = words([0.; 2]);
        ffn_routing
            .merge([&first, &second], &words([-1., 3.]), [&mut a, &mut b])
            .unwrap();
        assert_eq!(a, words([15., 11.]));
        assert_eq!(b, words([11., 7.]));
    }

    #[test]
    fn routing_uses_one_flat_fp32_norm_and_validates_inputs() {
        let mut weights = [0.; 32];
        // Flattened [4,0,0,0] normalized once is approximately [2,0,0,0].
        weights[0] = 3.0_f32.ln() * 0.5;
        assert!(Router::new(2, &weights[..31], &[0.; 8]).is_err());
        let router = Router::new(2, &weights, &[0.; 8]).unwrap();
        let left = words([4., 0.]);
        let right = words([0., 0.]);
        let mut branch = words([0.; 2]);
        let routing = router.route([&left, &right], &mut branch).unwrap();
        assert_eq!(branch, words([3., 0.]));
        assert_eq!(routing.post, [1., 1.]);
        assert!(
            routing
                .mix
                .into_iter()
                .flatten()
                .all(|x| (x - 0.5).abs() < f32::EPSILON)
        );
        assert!(router.route([&left[..1], &right], &mut branch).is_err());
        assert!(
            router
                .route([&left, &words([f32::NAN, 0.])], &mut branch)
                .is_err()
        );
        assert!(
            router
                .route([&words([f32::MAX / 2., 0.]), &right], &mut branch)
                .is_err()
        );
    }

    #[test]
    fn merge_stores_matmul_and_branch_product_before_residual_add() {
        let routing = Routing {
            post: [1.5, 1.5],
            mix: [[0.5; 2]; 2],
        };
        let a = words([1., 1.]);
        let b = words([1.0078125, 1.]);
        let branch = [bf16::from_bits(0x3aa0), bf16::from_bits(0x3b2b)];
        let mut first = words([0.; 2]);
        let mut second = words([0.; 2]);
        routing
            .merge([&a, &b], &branch, [&mut first, &mut second])
            .unwrap();
        // Column 0 has a halfway matmul result and a small positive update.
        // Column 1 has a branch product just above half an ulp, which becomes
        // an exact halfway update at the BF16 store and rounds down at the final tie.
        assert_eq!(first, words([1., 1.]));
        for i in 0..2 {
            let wide = (a[i].to_f32() + b[i].to_f32()) * 0.5 + branch[i].to_f32() * 1.5;
            assert_eq!(bf16::from_f32(wide), words([1.0078125])[0]);
        }
        assert_eq!(second, first);
    }
}
