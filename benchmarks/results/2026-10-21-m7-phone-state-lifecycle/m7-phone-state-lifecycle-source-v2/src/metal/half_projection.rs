//! Explicit native primitives for the experimental resident Half projection.
use super::*;
use half::f16;

fn verify(
    g: &Gpu,
    x: &[f16],
    w: &[f16],
    bias: &[f16],
    want: &[f64],
    n: usize,
    width: usize,
) -> usize {
    autoreleasepool(|_| {
        assert_eq!(
            g.execution_width(Kernel::HalfCompensated),
            g.execution_width(Kernel::HalfCompensatedFloat)
        );
        let out = bias.len();
        assert_eq!(x.len(), n * width);
        assert_eq!(w.len(), out * width);
        assert_eq!(want.len(), n * out);
        let xs = width + 7;
        let ws = width + 3;
        let os = out + 5;
        let offset = 5;
        let mut input = Buffer::<u16>::new(g, offset + n * xs + 7).unwrap();
        let mut weight = Buffer::<u16>::new(g, out * ws + 7).unwrap();
        let mut biases = Buffer::<u16>::new(g, out + 7).unwrap();
        input.with_mut(|a| {
            a.fill(0x7e00);
            for row in 0..n {
                for col in 0..width {
                    a[offset + row * xs + col] = x[row * width + col].to_bits();
                }
            }
        });
        weight.with_mut(|a| {
            a.fill(0x7e00);
            for row in 0..out {
                for col in 0..width {
                    a[row * ws + col] = w[row * width + col].to_bits();
                }
            }
        });
        biases.with_mut(|a| {
            a.fill(0x7e00);
            for (dst, value) in a.iter_mut().zip(bias) {
                *dst = value.to_bits();
            }
        });
        let mut halves = Buffer::<u16>::new(g, n * os + 7).unwrap();
        let mut floats = Buffer::<u32>::new(g, n * os + 7).unwrap();
        let mut status = Buffer::<u32>::new(g, 8).unwrap();
        for kernel in [Kernel::HalfCompensated, Kernel::HalfCompensatedFloat] {
            halves.with_mut(|a| a.fill(0x7bff));
            floats.with_mut(|a| a.fill(0xdeadbeef));
            status.with_mut(|a| a.fill(0));
            let raw = g.command().unwrap();
            let command = Command {
                raw: &raw,
                trace: None,
            };
            let output = if matches!(kernel, Kernel::HalfCompensated) {
                &halves.raw
            } else {
                &floats.raw
            };
            g.encode(
                &command,
                kernel,
                &[&weight.raw, &biases.raw, &input.raw, output],
                &[
                    n as u32,
                    width as u32,
                    out as u32,
                    ws as u32,
                    xs as u32,
                    os as u32,
                    offset as u32,
                ],
                n * out,
                Some(&status),
            )
            .unwrap();
            raw.commit();
            raw.waitUntilCompleted();
            assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
            assert!(raw.error().is_none());
            let invalid = want.iter().any(|&v| {
                if matches!(kernel, Kernel::HalfCompensated) {
                    !f16::from_f64(v).is_finite()
                } else {
                    !(v as f32).is_finite()
                }
            });
            status.with_slice(|a| {
                assert_eq!(a[0], u32::from(invalid));
                assert!(a[1..].iter().all(|&v| v == 0));
            });
            if matches!(kernel, Kernel::HalfCompensated) {
                halves.with_slice(|a| {
                    for (i, &bits) in a.iter().enumerate() {
                        let row = i / os;
                        let col = i % os;
                        if row < n && col < out {
                            let expected = f16::from_f64(want[row * out + col]);
                            if expected.is_nan() {
                                assert!(f16::from_bits(bits).is_nan());
                            } else {
                                assert_eq!(
                                    bits,
                                    expected.to_bits(),
                                    "Half n{n} width{width} row{row} col{col}"
                                );
                            }
                        } else {
                            assert_eq!(bits, 0x7bff);
                        }
                    }
                });
            } else {
                floats.with_slice(|a| {
                    for (i, &bits) in a.iter().enumerate() {
                        let row = i / os;
                        let col = i % os;
                        if row < n && col < out {
                            let expected = want[row * out + col] as f32;
                            if expected.is_nan() {
                                assert!(f32::from_bits(bits).is_nan());
                            } else {
                                assert_eq!(
                                    bits,
                                    expected.to_bits(),
                                    "Float n{n} width{width} row{row} col{col}"
                                );
                            }
                        } else {
                            assert_eq!(bits, 0xdeadbeef);
                        }
                    }
                });
            }
        }
        n * out * 2
    })
}

#[test]
#[ignore = "Explicit native compensated Half primitive gates; no model required"]
fn half_projection_rounding_cancellation_strides_and_tails() {
    autoreleasepool(|_| {
        let g = Gpu::new().unwrap();
        let mut compared = 0;
        let mut double_rounding = 0;
        // Three nonzero terms have a bounded exponent range and at most 37 bits in
        // the exact sum. FP64 therefore represents these midpoint witnesses exactly.
        for lower in [
            0u16, 1, 0x3ff, 0x400, 0x401, 0x3554, 0x3555, 0x3bff, 0x3c00, 0x3c01, 0x7bfe, 0x7bff,
        ] {
            let overflow = lower == 0x7bff;
            let a = f16::from_bits(lower).to_f32();
            let b = if overflow {
                65536.
            } else {
                f16::from_bits(lower + 1).to_f32()
            };
            let midpoint = (a + b) * 0.5;
            let exponent = ((midpoint.to_bits() >> 23) & 255) as i32 - 127;
            let tiny = f16::from_f32(2f32.powi(exponent - 1).max(2f32.powi(-24)));
            for sign in [1., -1.] {
                for side in [-1., 0., 1.] {
                    let mut x = vec![f16::ZERO; 3 * 65];
                    let mut w = vec![f16::ZERO; 2 * 65];
                    for row in 0..3 {
                        x[row * 65] = f16::from_f32(sign * a);
                        x[row * 65 + 1] = f16::from_f32(sign * if overflow { 16. } else { b });
                        x[row * 65 + 64] = f16::from_f32(sign * side * tiny.to_f32());
                    }
                    for row in 0..2 {
                        w[row * 65] = f16::from_f32(if overflow { 1. } else { 0.5 });
                        w[row * 65 + 1] = w[row * 65];
                        w[row * 65 + 64] = f16::from_bits(1);
                    }
                    let exact = (0..65)
                        .map(|i| f64::from(x[i].to_f32()) * f64::from(w[i].to_f32()))
                        .sum::<f64>();
                    if f16::from_f64(exact) != f16::from_f32(exact as f32) {
                        double_rounding += 1;
                    }
                    compared += verify(&g, &x, &w, &[f16::ZERO; 2], &[exact; 6], 3, 65);
                    compared += verify(&g, &x[..65], &w, &[f16::ZERO; 2], &[exact; 2], 1, 65);
                }
            }
        }
        assert!(double_rounding > 0);
        // Analytical cancellation oracle: the identical large products cancel.
        // Do not lose the tiny middle term in a sequential FP64 reference sum.
        for tiny in [f16::from_bits(1), f16::from_f32(1. / 1024.)] {
            let big = f16::MAX;
            compared += verify(
                &g,
                &[big, tiny, big],
                &[big, f16::ONE, -big],
                &[f16::ZERO],
                &[f64::from(tiny.to_f32())],
                1,
                3,
            );
            compared += verify(
                &g,
                &[f16::MAX, tiny],
                &[f16::ONE, f16::ONE],
                &[-f16::MAX],
                &[f64::from(tiny.to_f32())],
                1,
                2,
            );
        }
        // The exact FP32 midpoint rounds to even. Half's odd-rounding preparation
        // must not be used for the final FP32 vocabulary distribution.
        compared += verify(
            &g,
            &[f16::ONE, f16::from_bits(1)],
            &[f16::ONE, f16::ONE],
            &[f16::ZERO],
            &[1. + 2f64.powi(-24)],
            1,
            2,
        );
        for width in [1, 31, 33, 65, 129, 1536, 8960] {
            for n in [1, 3, 8] {
                let x: Vec<_> = (0..n * width)
                    .map(|i| f16::from_f32((i as i32 % 19 - 9) as f32 / 16.))
                    .collect();
                let w: Vec<_> = (0..3 * width)
                    .map(|i| f16::from_f32((i as i32 % 17 - 8) as f32 / 32.))
                    .collect();
                let bias = [
                    f16::from_f32(0.125),
                    f16::from_f32(-0.25),
                    f16::from_f32(0.5),
                ];
                // Every term is an integer multiple of 2^-9; the bounded sum is exact in FP64.
                let mut want = Vec::new();
                for row in 0..n {
                    for out in 0..3 {
                        want.push(
                            (0..width)
                                .map(|col| {
                                    f64::from(x[row * width + col].to_f32())
                                        * f64::from(w[out * width + col].to_f32())
                                })
                                .sum::<f64>()
                                + f64::from(bias[out].to_f32()),
                        );
                    }
                }
                compared += verify(&g, &x, &w, &bias, &want, n, width);
            }
        }
        compared += verify(
            &g,
            &[f16::NAN],
            &[f16::ONE],
            &[f16::ZERO],
            &[f64::NAN],
            1,
            1,
        );
        println!(
            "Half compensated primitives: {compared} output comparisons, {double_rounding} exact midpoint double-rounding witnesses; no general correctly-rounded claim"
        );
    });
}
