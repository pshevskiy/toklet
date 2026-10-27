//! Explicit GPU accuracy experiment. The oracle is test-local FP64 arithmetic
//! on stored Half inputs, with the declared FP32 score/probability boundaries.
//! It is not a CPU emulation of Metal transcendental rounding or a model oracle.
use super::*;
use half::f16;
use sha2::{Digest, Sha256};

const GUARD: usize = 7;
const FLOAT_GUARD: u32 = 0x7fa1_2345;
const HALF_GUARD: u16 = 0x3555;

struct Input {
    rows: usize,
    heads: usize,
    kvheads: usize,
    d: usize,
    cap: usize,
    offset: usize,
    valid: usize,
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
}
fn value(bits: u16) -> f64 {
    f64::from(f16::from_bits(bits).to_f32())
}
fn half(x: f64) -> u16 {
    f16::from_f64(x).to_bits()
}
// Half is an exact integer multiple of 2^-24. Sum squared differences in
// units of 2^-48, then round once for reporting. This is stronger than fsum
// of individually rounded f64 squares for widely separated Half magnitudes.
// At most 4096 finite-Half differences have a squared sum below 2^94.
fn squared_half_error(a: u16, b: u16) -> u128 {
    let (a, b) = (value(a), value(b));
    assert!(a.is_finite() && b.is_finite());
    let units = (a - b).abs() * 2f64.powi(24);
    assert!(units.is_finite() && units.fract() == 0. && (0. ..2f64.powi(41)).contains(&units));
    let units = units as u64; // Exact integer and range checked before conversion.
    u128::from(units).checked_mul(u128::from(units)).unwrap()
}
fn squared_half_errors(actual: &[u16], expected: &[u16]) -> u128 {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.len() <= 4096);
    actual
        .iter()
        .zip(expected)
        .try_fold(0u128, |sum, (&a, &b)| {
            sum.checked_add(squared_half_error(a, b))
        })
        .unwrap()
}
// Error-free two-sum expansion, followed by the half-even final rounding
// correction. Independent of shader Neumaier arithmetic and BLAS order.
// Inputs here are finite bounded dots, exponentials or context products.
fn accurate_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut partials = Vec::<f64>::new();
    for mut x in values {
        assert!(x.is_finite());
        let mut kept = 0;
        for i in 0..partials.len() {
            let mut y = partials[i];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            assert!(hi.is_finite());
            let lo = y - (hi - x);
            if lo != 0. {
                partials[kept] = lo;
                kept += 1;
            }
            x = hi;
        }
        partials.truncate(kept);
        partials.push(x);
    }
    let mut hi = partials.pop().unwrap_or(0.);
    let mut lo = 0.;
    while let Some(y) = partials.pop() {
        let x = hi;
        hi = x + y;
        lo = y - (hi - x);
        if lo != 0. {
            break;
        }
    }
    if partials
        .last()
        .is_some_and(|x| (*x > 0. && lo > 0.) || (*x < 0. && lo < 0.))
    {
        let twice = lo * 2.;
        let corrected = hi + twice;
        if corrected - hi == twice {
            hi = corrected;
        }
    }
    hi
}
fn oracle(input: &Input) -> Vec<f64> {
    let mut output = Vec::with_capacity(input.rows * input.heads * input.d);
    for row in 0..input.rows {
        for head in 0..input.heads {
            let kv = head / (input.heads / input.kvheads);
            let scores: Vec<f64> = (0..=input.offset + row)
                .map(|p| {
                    let dot = accurate_sum((0..input.d).map(|f| {
                        value(input.q[(row * input.heads + head) * input.d + f])
                            * value(input.k[(kv * input.cap + p) * input.d + f])
                    }));
                    f64::from((dot / (input.d as f64).sqrt()) as f32)
                })
                .collect();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|x| (x - maximum).exp()).collect();
            let mass = accurate_sum(exps.iter().copied());
            let probabilities: Vec<f64> =
                exps.iter().map(|x| f64::from((x / mass) as f32)).collect();
            for f in 0..input.d {
                output.push(accurate_sum(probabilities.iter().enumerate().map(
                    |(p, weight)| weight * value(input.v[(kv * input.cap + p) * input.d + f]),
                )));
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"kind":"attention_expected", "rows":input.rows, "heads":input.heads,
        "kv_heads":input.kvheads, "head_dim":input.d, "offset":input.offset, "valid":input.valid,
        "output_half_bits":output.iter().map(|x| half(*x)).collect::<Vec<_>>()})
    );
    output
}
struct Observed {
    output: Vec<u16>,
    scores: Vec<f32>,
    probabilities: Vec<f32>,
    status: u32,
}
fn run(gpu: &Gpu, input: &Input, kernel: Kernel) -> Observed {
    let upload = |data: &[u16]| {
        let mut buffer = Buffer::<u16>::new(gpu, data.len() + GUARD).unwrap();
        buffer.with_mut(|b| {
            b.fill(HALF_GUARD);
            b[..data.len()].copy_from_slice(data);
        });
        buffer
    };
    let q = upload(&input.q);
    let k = upload(&input.k);
    let v = upload(&input.v);
    // A padding row exercises allocated scratch strides independently of rows.
    let chunk = input.rows + 1;
    let scratch_len = input.heads * chunk * input.cap;
    let mut scores = Buffer::<u32>::new(gpu, scratch_len + GUARD).unwrap();
    let mut probs = Buffer::<u32>::new(gpu, scratch_len + GUARD).unwrap();
    scores.with_mut(|b| b.fill(FLOAT_GUARD));
    probs.with_mut(|b| b.fill(FLOAT_GUARD));
    let n = input.rows * input.heads * input.d;
    let mut out = Buffer::<u16>::new(gpu, n + GUARD).unwrap();
    out.with_mut(|b| b.fill(HALF_GUARD));
    let status = Buffer::<u32>::new(gpu, 8).unwrap();
    let shape = [
        input.rows,
        input.heads,
        input.kvheads,
        input.d,
        input.cap,
        chunk,
        input.offset,
        input.valid,
    ]
    .map(|x| u32::try_from(x).unwrap());
    let raw = gpu.command().unwrap();
    gpu.encode(
        &Command {
            raw: &raw,
            trace: None,
        },
        kernel,
        &[&q.raw, &k.raw, &v.raw, &scores.raw, &probs.raw, &out.raw],
        &shape,
        input.rows * input.heads,
        Some(&status),
    )
    .unwrap();
    raw.commit();
    raw.waitUntilCompleted();
    // Persist bounded raw output before status, guard or numerical assertions.
    println!(
        "{}",
        serde_json::json!({"kind":"attention_actual", "kernel":kernel.name(), "shape":shape,
        "output_half_bits":out.with_slice(|b| b[..n].to_vec()), "status_words":status.with_slice(|b| b.to_vec())})
    );
    assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
    for (buffer, original) in [(&q, &input.q), (&k, &input.k), (&v, &input.v)] {
        buffer.with_slice(|b| {
            assert_eq!(&b[..original.len()], original);
            assert!(b[original.len()..].iter().all(|x| *x == HALF_GUARD));
        });
    }
    let read_scratch = |buffer: &Buffer<u32>| {
        buffer.with_slice(|b| {
            for head in 0..input.heads {
                for row in 0..chunk {
                    let base = (head * chunk + row) * input.cap;
                    let first = if row < input.rows { input.valid } else { 0 };
                    assert!(
                        b[base + first..base + input.cap]
                            .iter()
                            .all(|x| *x == FLOAT_GUARD)
                    );
                }
            }
            assert!(b[scratch_len..].iter().all(|x| *x == FLOAT_GUARD));
            b[..scratch_len]
                .iter()
                .map(|x| f32::from_bits(*x))
                .collect()
        })
    };
    Observed {
        output: out.with_slice(|b| {
            assert!(b[n..].iter().all(|x| *x == HALF_GUARD));
            b[..n].to_vec()
        }),
        scores: read_scratch(&scores),
        probabilities: read_scratch(&probs),
        status: status.with_slice(|b| {
            assert!(b[1..].iter().all(|x| *x == 0));
            b[0]
        }),
    }
}
fn gate(actual: &[u16], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        // Existing independent attention primitive bound; never fit to this candidate.
        assert!(
            value(a).is_finite() && (value(a) - b).abs() <= 1e-4 + 1e-3 * b.abs(),
            "{} vs {b}",
            value(a)
        );
    }
}
fn patterned(rows: usize, offset: usize, heads: usize, kvheads: usize, d: usize) -> Input {
    let cap = offset + rows + 3;
    let values = |n, seed| {
        (0..n)
            .map(|i| half(((i * 17 + seed) % 101) as f64 / 32. - 1.5))
            .collect()
    };
    let mut input = Input {
        rows,
        heads,
        kvheads,
        d,
        cap,
        offset,
        valid: offset + rows,
        q: values(rows * heads * d, 3),
        k: values(kvheads * cap * d, 11),
        v: values(kvheads * cap * d, 37),
    };
    for head in 0..kvheads {
        for buffer in [&mut input.k, &mut input.v] {
            buffer[(head * cap + input.valid) * d..(head + 1) * cap * d].fill(0x7e00);
        }
    }
    input
}

#[test]
fn compensated_attention_oracle_and_fixture_are_independent_of_gpu() {
    assert_eq!(AttentionMode::default(), AttentionMode::Reference);
    assert_eq!(squared_half_error(half(1.5), half(1.)), 1u128 << 46);
    assert_eq!(squared_half_error(half(0.), half(2f64.powi(-24))), 1);
    assert_eq!(squared_half_error(half(-1.5), half(-1.)), 1u128 << 46);
    assert_eq!(squared_half_error(half(0.), half(-0.)), 0);
    assert_ne!(half(0.), half(-0.));
    let maximum_units = 131008u128 << 24;
    assert_eq!(
        squared_half_error(half(65504.), half(-65504.)),
        maximum_units * maximum_units
    );
    let mixed_units = (65504u128 << 24) - 1;
    assert_eq!(
        squared_half_error(half(65504.), half(2f64.powi(-24))),
        mixed_units * mixed_units
    );
    assert_eq!(
        squared_half_errors(&[half(1.5), half(-1.5)], &[half(1.), half(-1.)]),
        1u128 << 47
    );
    assert_eq!(accurate_sum([1e16, 1., -1e16]), 1.);
    assert_eq!(
        accurate_sum([1., 2f64.powi(-53), 2f64.powi(-107)]),
        1. + 2f64.powi(-52)
    );
    assert_eq!(
        accurate_sum([-1., -2f64.powi(-53), -2f64.powi(-107)]),
        -1. - 2f64.powi(-52)
    );
    assert_eq!(accurate_sum([1., 2f64.powi(-53)]), 1.);
    let mut input = patterned(1, 0, 4, 2, 8);
    input.q.fill(half(0.));
    let result = oracle(&input);
    for head in 0..input.heads {
        for f in 0..input.d {
            assert_eq!(
                result[head * input.d + f],
                value(input.v[((head / 2) * input.cap) * input.d + f])
            );
        }
    }
    let witness = witness();
    assert_eq!(witness.len(), 141312);
}

#[test]
#[ignore = "explicit Apple GPU compensated attention, causal/GQA/offset/stride controls"]
fn compensated_attention_controls() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        for (rows, offset, heads, kvheads, d) in
            [(1, 0, 4, 2, 8), (3, 5, 4, 2, 8), (3, 63, 16, 8, 128)]
        {
            let input = patterned(rows, offset, heads, kvheads, d);
            let expected = oracle(&input);
            for kernel in [Kernel::Attention, Kernel::AttentionCompensated] {
                let full = run(&gpu, &input, kernel);
                assert_eq!(full.status, 0);
                gate(&full.output, &expected);
                if rows == 1 && offset == 0 {
                    assert_eq!(
                        full.output,
                        expected.iter().map(|x| half(*x)).collect::<Vec<_>>()
                    );
                }
                for row in 0..rows {
                    let one = Input {
                        rows: 1,
                        heads,
                        kvheads,
                        d,
                        cap: input.cap,
                        offset: offset + row,
                        valid: offset + row + 1,
                        q: input.q[row * heads * d..(row + 1) * heads * d].to_vec(),
                        k: input.k.clone(),
                        v: input.v.clone(),
                    };
                    let single = run(&gpu, &one, kernel);
                    assert_eq!(single.status, 0);
                    assert_eq!(
                        single.output,
                        full.output[row * heads * d..(row + 1) * heads * d]
                    );
                    for head in 0..heads {
                        let base = (head * (rows + 1) + row) * input.cap;
                        let probability_mass = accurate_sum(
                            full.probabilities[base..base + one.valid]
                                .iter()
                                .map(|x| f64::from(*x)),
                        );
                        assert!((probability_mass - 1.).abs() < 1e-5);
                        assert!(
                            full.probabilities[base + one.valid..base + input.valid]
                                .iter()
                                .all(|x| x.to_bits() == 0)
                        );
                    }
                }
            }
        }
        let mut zero = patterned(3, 5, 4, 2, 8);
        zero.q.fill(half(0.));
        zero.k.fill(half(0.));
        zero.v.fill(half(0.));
        let expected = oracle(&zero);
        for kernel in [Kernel::Attention, Kernel::AttentionCompensated] {
            let observed = run(&gpu, &zero, kernel);
            assert_eq!(observed.status, 0);
            assert_eq!(observed.output, vec![half(0.); expected.len()]);
        }
        let mut invalid = patterned(1, 0, 4, 2, 8);
        invalid.k[0] = 0x7e00;
        assert_ne!(run(&gpu, &invalid, Kernel::AttentionCompensated).status, 0);
    });
}

#[test]
#[ignore = "explicit Apple GPU compensated attention cancellation and signed Half midpoints"]
fn compensated_attention_cancellation_and_midpoints() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let mut input = patterned(1, 15, 2, 1, 8);
        input.q.fill(half(0.));
        input.k.fill(half(0.));
        input.v.fill(half(0.));
        // Uniform sixteen-key probabilities are exactly 1/16. The tiny fifth
        // term determines the side of a Half midpoint after large terms.
        for f in 0..4 {
            let sign = if f == 3 { -1. } else { 1. };
            for (p, x) in [8., 8., 2f64.powi(-8), 2f64.powi(-8)]
                .into_iter()
                .enumerate()
            {
                input.v[p * 8 + f] = half(sign * x);
            }
            input.v[4 * 8 + f] = half(match f {
                0 => 2f64.powi(-24),
                2 | 3 => -2f64.powi(-24),
                _ => 0.,
            });
        }
        for (p, x) in [65504., 2f64.powi(-10), -65504.].into_iter().enumerate() {
            input.v[p * 8 + 4] = half(x);
        }
        let expected = oracle(&input);
        let improved = run(&gpu, &input, Kernel::AttentionCompensated);
        assert_eq!(improved.status, 0);
        assert_eq!(
            improved.output,
            expected.iter().map(|x| half(*x)).collect::<Vec<_>>()
        );
        assert_eq!(
            &improved.output[..5],
            &[
                half(1. + 2f64.powi(-10)),
                half(1.),
                half(1.),
                half(-1. - 2f64.powi(-10)),
                half(2f64.powi(-14))
            ]
        );
        // Half*Half score products are exact; only reduction loses the small
        // middle term in the original ordered FP32 sum.
        input.offset = 1;
        input.valid = 2;
        input.q.fill(half(0.));
        input.k.fill(half(0.));
        input.v.fill(half(0.));
        for h in 0..2 {
            for (f, x) in [2048., 0.25, 2048.].into_iter().enumerate() {
                input.q[h * 8 + f] = half(x);
            }
        }
        for (f, x) in [2048., 0.25, -2048.].into_iter().enumerate() {
            input.k[f] = half(x);
        }
        input.v[0] = half(1.);
        let expected = oracle(&input);
        let improved = run(&gpu, &input, Kernel::AttentionCompensated);
        let original = run(&gpu, &input, Kernel::Attention);
        assert_eq!(improved.status, 0);
        assert_eq!(original.status, 0);
        let expected_score = 0.0625f32 / 8f32.sqrt();
        assert!((improved.scores[0] - expected_score).abs() <= f32::EPSILON * expected_score);
        assert_eq!(original.scores[0], 0.);
        gate(&improved.output, &expected);
    });
}

fn witness() -> Vec<u16> {
    let bytes = include_bytes!("../../tests/fixtures/qwen3-attention-inputs-v1/inputs.f16le");
    assert_eq!(bytes.len(), 282624);
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "1d82f23393652efc0c440366e68fa1932086da98f73105846f835cbf1b1035d9"
    );
    bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}
#[test]
#[ignore = "explicit Apple GPU saved Qwen3 layer0 own-input attention A/B accuracy; no model execution"]
fn compensated_attention_qwen3_own_input_accuracy() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let data = witness();
        let mut total = [0usize; 2];
        let mut squared_error = [0u128; 2];
        for (row, position) in [0usize, 41, 65].into_iter().enumerate() {
            let mut input = Input {
                rows: 1,
                heads: 16,
                kvheads: 8,
                d: 128,
                cap: 69,
                offset: position,
                valid: position + 1,
                q: data[row * 2048..(row + 1) * 2048].to_vec(),
                k: vec![0x7e00; 8 * 69 * 128],
                v: vec![0x7e00; 8 * 69 * 128],
            };
            for (which, target) in [&mut input.k, &mut input.v].into_iter().enumerate() {
                for p in 0..=position {
                    for head in 0..8 {
                        let source = 3 * 2048 + which * 66 * 1024 + (p * 8 + head) * 128;
                        let destination = (head * 69 + p) * 128;
                        target[destination..destination + 128]
                            .copy_from_slice(&data[source..source + 128]);
                    }
                }
            }
            let expected = oracle(&input);
            let expected_half: Vec<_> = expected.iter().map(|x| half(*x)).collect();
            assert_eq!(expected.len(), 2048);
            let mut row_different = [0usize; 2];
            let mut row_squared = [0u128; 2];
            for (i, kernel) in [Kernel::Attention, Kernel::AttentionCompensated]
                .into_iter()
                .enumerate()
            {
                let observed = run(&gpu, &input, kernel);
                assert_eq!(observed.status, 0);
                gate(&observed.output, &expected);
                let different = observed
                    .output
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| **a != half(**b))
                    .count();
                let sum_squared = squared_half_errors(&observed.output, &expected_half);
                row_different[i] = different;
                row_squared[i] = sum_squared;
                if position == 0 {
                    assert_eq!(different, 0);
                } else {
                    total[i] = total[i].checked_add(different).unwrap();
                    squared_error[i] = squared_error[i].checked_add(sum_squared).unwrap();
                }
                let reported_squared = sum_squared as f64 * 2f64.powi(-48);
                println!(
                    "{} position={position} half_differences={different} squared_error={reported_squared:.17e} squared_units_2^-48={sum_squared}",
                    kernel.name()
                );
            }
            assert!(
                row_different[1] <= row_different[0],
                "position {position} Half mismatch regression: {row_different:?}"
            );
            assert!(
                row_squared[1] <= row_squared[0],
                "position {position} squared-error regression: {row_squared:?}"
            );
        }
        // Experiment retention criterion, separate from unchanged primitive
        // bounds and from the still-required whole-model output screen.
        assert!(
            total[1] < total[0],
            "no pooled Half accuracy improvement: {total:?}"
        );
        assert!(
            squared_error[1] < squared_error[0],
            "no pooled squared-error improvement: {squared_error:?}"
        );
    });
}
