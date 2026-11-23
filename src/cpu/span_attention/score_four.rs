//! Detached T1 score operator. Each dot has its own ordered binary64 carry.
use super::{Error, ordered_span_score};

fn ordered_four(query: &[f32], keys: [&[f32]; 4], scale: f32) -> [f32; 4] {
    let [k0, k1, k2, k3] = keys;
    debug_assert!(keys.iter().all(|k| k.len() == query.len()));
    let (mut s0, mut s1, mut s2, mut s3) = (0_f64, 0_f64, 0_f64, 0_f64);
    for ((((&q, &k0), &k1), &k2), &k3) in query.iter().zip(k0).zip(k1).zip(k2).zip(k3) {
        let q = f64::from(q);
        s0 += q * f64::from(k0);
        s1 += q * f64::from(k1);
        s2 += q * f64::from(k2);
        s3 += q * f64::from(k3);
    }
    [s0, s1, s2, s3].map(|sum| (sum * f64::from(scale)) as f32)
}

fn publish(target: &mut f32, score: f32, index: usize) -> Result<(), Error> {
    if !score.is_finite() {
        return Err(Error::NonFinite {
            tensor: "attention scores",
            index,
        });
    }
    *target = score;
    Ok(())
}

/// Score one prevalidated physical span. Each four-key stripe carries
/// independent dots, but publishes its lanes in original logical order.
pub(super) fn write(
    query: &[f32],
    segment: &[f32],
    output: &mut [f32],
    base: usize,
    scale: f32,
) -> Result<(), Error> {
    let dim = query.len();
    let mut stripes = segment.chunks_exact(4 * dim);
    let mut logical = 0;
    for stripe in &mut stripes {
        let lanes = ordered_four(
            query,
            [
                &stripe[..dim],
                &stripe[dim..2 * dim],
                &stripe[2 * dim..3 * dim],
                &stripe[3 * dim..],
            ],
            scale,
        );
        for score in lanes {
            publish(&mut output[logical], score, base + logical)?;
            logical += 1;
        }
    }
    for key in stripes.remainder().chunks_exact(dim) {
        let score = ordered_span_score(query, key, scale);
        publish(&mut output[logical], score, base + logical)?;
        logical += 1;
    }
    debug_assert_eq!(logical, output.len());
    Ok(())
}
