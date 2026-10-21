//! Native-only primitive qualification. The independent FP64 oracle consumes
//! exact stored Half inputs, not production attention or CPU decoder helpers.
use super::*;
use half::f16;

#[test]
#[ignore = "explicit Apple GPU online attention, FP64 oracle and 32K depth"]
fn online_attention_matches_oracle_at_irregular_and_long_depths() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let mut maximum_error = 0.0f64;
        let mut compared = 0;
        for (heads, kvheads, d) in [(4, 2, 8), (14, 2, 64), (12, 2, 128)] {
            for (rows, offset) in [(1, 0), (3, 31), (17, 511), (3, 8191), (17, 32751)] {
                let cap = offset + rows + 3;
                let n = rows * heads * d;
                let kn = kvheads * cap * d;
                let mut q = Buffer::<u16>::new(&gpu, n + 7).unwrap();
                let mut k = Buffer::<u16>::new(&gpu, kn + 7).unwrap();
                let mut v = Buffer::<u16>::new(&gpu, kn + 7).unwrap();
                let fill = |buffer: &mut Buffer<u16>, seed: usize| {
                    buffer.with_mut(|bits| {
                        bits.fill(0x7e00);
                        let len = bits.len() - 7;
                        for (i, value) in bits[..len].iter_mut().enumerate() {
                            *value =
                                f16::from_f32(((i * 17 + seed) % 101) as f32 / 32. - 1.5).to_bits();
                        }
                    });
                };
                fill(&mut q, 3);
                fill(&mut k, 11);
                fill(&mut v, 37);
                // Unwritten future cache slots contain NaNs: online paths must
                // never consume them. Masked future rows inside a chunk remain
                // finite, with causal correctness checked by the oracle.
                k.with_mut(|bits| {
                    for head in 0..kvheads {
                        bits[(head * cap + offset + rows) * d..(head + 1) * cap * d].fill(0x7e00);
                    }
                });
                let before = [&q, &k, &v].map(|b| b.with_slice(|v| v.to_vec()));
                let mut out = Buffer::<u16>::new(&gpu, n + 7).unwrap();
                let mut status = Buffer::<u32>::new(&gpu, 8).unwrap();
                let shape = [
                    rows as u32,
                    heads as u32,
                    kvheads as u32,
                    d as u32,
                    cap as u32,
                    rows as u32,
                    offset as u32,
                    (offset + rows) as u32,
                ];
                let mut oracle = vec![0.; n];
                for row in 0..rows {
                    for head in 0..heads {
                        let kv = head / (heads / kvheads);
                        let mut scores = Vec::with_capacity(offset + row + 1);
                        for p in 0..=offset + row {
                            let mut dot = 0.;
                            for f in 0..d {
                                dot += f64::from(
                                    f16::from_bits(before[0][(row * heads + head) * d + f])
                                        .to_f32(),
                                ) * f64::from(
                                    f16::from_bits(before[1][(kv * cap + p) * d + f]).to_f32(),
                                );
                            }
                            scores.push(dot / (d as f64).sqrt());
                        }
                        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        for s in &mut scores {
                            *s = (*s - maximum).exp();
                        }
                        let mass: f64 = scores.iter().sum();
                        for f in 0..d {
                            let sum: f64 = scores
                                .iter()
                                .enumerate()
                                .map(|(p, w)| {
                                    w * f64::from(
                                        f16::from_bits(before[2][(kv * cap + p) * d + f]).to_f32(),
                                    )
                                })
                                .sum();
                            oracle[(row * heads + head) * d + f] = sum / mass;
                        }
                    }
                }
                for kernel in [Kernel::AttentionDecode, Kernel::AttentionPrefill] {
                    // Decode's one-row ownership is also tested on multi-row
                    // inputs; production uses it only for a single-token step.
                    out.with_mut(|v| v.fill(0x7bff));
                    status.with_mut(|v| v.fill(0));
                    let raw = gpu.command().unwrap();
                    gpu.encode(
                        &Command {
                            raw: &raw,
                            trace: None,
                        },
                        kernel,
                        &[&q.raw, &k.raw, &v.raw, &out.raw],
                        &shape,
                        rows.div_ceil(if matches!(kernel, Kernel::AttentionDecode) {
                            1
                        } else {
                            2
                        }) * heads,
                        Some(&status),
                    )
                    .unwrap();
                    raw.commit();
                    raw.waitUntilCompleted();
                    assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
                    assert_eq!(status.with_slice(|v| v.to_vec()), vec![0; 8]);
                    out.with_slice(|bits| {
                        for (i, &expected) in oracle.iter().enumerate() {
                            let actual = f64::from(f16::from_bits(bits[i]).to_f32());
                            let delta = (actual - expected).abs();
                            maximum_error = maximum_error.max(delta);
                            // Frozen before collection: covers Half rounding
                            // plus FP32 reduction, independent of model gates.
                            assert!(actual.is_finite() && delta <= 1e-4 + 1e-3 * expected.abs(),
                                "{} d={d} rows={rows} offset={offset} i={i}: {actual} vs {expected}", kernel.name());
                            compared += 1;
                        }
                        assert!(bits[n..].iter().all(|v| *v == 0x7bff));
                    });
                    for (buffer, expected) in [&q, &k, &v].into_iter().zip(&before) {
                        assert_eq!(buffer.with_slice(|v| v.to_vec()), *expected);
                    }
                }
            }
        }
        println!("online oracle: {compared} outputs; max_abs_error={maximum_error}");
    });
}

#[test]
#[ignore = "explicit Apple GPU stable maxima, tile edges, causal poison and nonfinite status"]
fn online_attention_adversarial_witnesses() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let (heads, kvheads, d, cap) = (4usize, 2usize, 8usize, 64usize);
        let mut q = Buffer::<u16>::new(&gpu, 3 * heads * d + 7).unwrap();
        let mut k = Buffer::<u16>::new(&gpu, kvheads * cap * d + 7).unwrap();
        let mut v = Buffer::<u16>::new(&gpu, kvheads * cap * d + 7).unwrap();
        let mut out = Buffer::<u16>::new(&gpu, 3 * heads * d + 7).unwrap();
        let mut status = Buffer::<u32>::new(&gpu, 8).unwrap();
        let mut cases = 0;
        for depth in [15usize, 16, 17, 31, 32, 33] {
            for rows in [1usize, 2, 3] {
                let offset = depth - rows;
                for pattern in 0..4 {
                    let query = if pattern == 3 { 65504. } else { 1. };
                    q.with_mut(|b| b.fill(f16::from_f32(query).to_bits()));
                    for buffer in [&mut k, &mut v] {
                        buffer.with_mut(|b| b.fill(0x7e00));
                    }
                    k.with_mut(|b| {
                        for head in 0..kvheads {
                            for p in 0..depth {
                                let key = match pattern {
                                    0 => 0.,                        // uniform distribution
                                    1 => (p / 16) as f32 * 8. - 8., // later maxima
                                    2 => {
                                        if p == depth - 1 {
                                            64.
                                        } else {
                                            -64.
                                        }
                                    }
                                    _ => {
                                        if p == depth - 1 {
                                            65504.
                                        } else {
                                            -65504.
                                        }
                                    }
                                };
                                b[(head * cap + p) * d..(head * cap + p + 1) * d]
                                    .fill(f16::from_f32(key).to_bits());
                            }
                        }
                    });
                    v.with_mut(|b| {
                        for head in 0..kvheads {
                            for p in 0..depth {
                                for f in 0..d {
                                    b[(head * cap + p) * d + f] = f16::from_f32(
                                        (p as f32 - 16.) * 0.125 + head as f32 + f as f32 * 0.0625,
                                    )
                                    .to_bits();
                                }
                            }
                        }
                    });
                    let keys = k.with_slice(|v| v.to_vec());
                    let values = v.with_slice(|v| v.to_vec());
                    let mut results = Vec::new();
                    for kernel in [Kernel::AttentionDecode, Kernel::AttentionPrefill] {
                        out.with_mut(|b| b.fill(0x7bff));
                        status.with_mut(|b| b.fill(0));
                        let raw = gpu.command().unwrap();
                        gpu.encode(
                            &Command {
                                raw: &raw,
                                trace: None,
                            },
                            kernel,
                            &[&q.raw, &k.raw, &v.raw, &out.raw],
                            &[
                                rows as u32,
                                heads as u32,
                                kvheads as u32,
                                d as u32,
                                cap as u32,
                                rows as u32,
                                offset as u32,
                                depth as u32,
                            ],
                            rows.div_ceil(if matches!(kernel, Kernel::AttentionDecode) {
                                1
                            } else {
                                2
                            }) * heads,
                            Some(&status),
                        )
                        .unwrap();
                        raw.commit();
                        raw.waitUntilCompleted();
                        assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
                        assert_eq!(status.with_slice(|b| b[0]), 0);
                        let actual = out.with_slice(|v| v.to_vec());
                        assert!(actual[rows * heads * d..].iter().all(|b| *b == 0x7bff));
                        for row in 0..rows {
                            for head in 0..heads {
                                let kv = head / (heads / kvheads);
                                let scores: Vec<_> = (0..=offset + row)
                                    .map(|p| {
                                        f64::from(query)
                                            * f64::from(
                                                f16::from_bits(keys[(kv * cap + p) * d]).to_f32(),
                                            )
                                            * (d as f64).sqrt()
                                    })
                                    .collect();
                                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                                let weights: Vec<_> =
                                    scores.iter().map(|s| (s - max).exp()).collect();
                                let sum: f64 = weights.iter().sum();
                                for f in 0..d {
                                    let expected: f64 = weights
                                        .iter()
                                        .enumerate()
                                        .map(|(p, w)| {
                                            w * f64::from(
                                                f16::from_bits(values[(kv * cap + p) * d + f])
                                                    .to_f32(),
                                            )
                                        })
                                        .sum::<f64>()
                                        / sum;
                                    let a = f64::from(
                                        f16::from_bits(actual[(row * heads + head) * d + f])
                                            .to_f32(),
                                    );
                                    assert!(
                                        (a - expected).abs() <= 1e-4 + 1e-3 * expected.abs(),
                                        "pattern{pattern} depth{depth} row{row}: {a} vs {expected}"
                                    );
                                }
                            }
                        }
                        results.push(actual);
                    }
                    assert_eq!(results[0], results[1]);
                    cases += 1;
                }
            }
        }
        // Poison active operands independently. With one valid key every Q/K
        // nonfinite still yields invalid softmax, and nonfinite V invalid output.
        for operand in 0..3 {
            for poison in [0x7e00u16, 0x7c00, 0xfc00] {
                for b in [&mut q, &mut k, &mut v] {
                    b.with_mut(|b| b.fill(0));
                }
                [&mut q, &mut k, &mut v][operand].with_mut(|b| b[0] = poison);
                for kernel in [Kernel::AttentionDecode, Kernel::AttentionPrefill] {
                    out.with_mut(|b| b.fill(0x7bff));
                    status.with_mut(|b| b.fill(0));
                    let raw = gpu.command().unwrap();
                    gpu.encode(
                        &Command {
                            raw: &raw,
                            trace: None,
                        },
                        kernel,
                        &[&q.raw, &k.raw, &v.raw, &out.raw],
                        &[1, 4, 2, 8, 64, 1, 0, 1],
                        heads,
                        Some(&status),
                    )
                    .unwrap();
                    raw.commit();
                    raw.waitUntilCompleted();
                    assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
                    assert_ne!(status.with_slice(|b| b[0]), 0);
                    assert!(out.with_slice(|b| b[heads * d..].iter().all(|b| *b == 0x7bff)));
                }
                cases += 1;
            }
        }
        // Later queries' K/V may already be cached. Changing them must have no
        // effect on the earlier row, including when these future slots are NaN.
        for b in [&mut q, &mut k, &mut v] {
            b.with_mut(|b| b.fill(f16::ONE.to_bits()));
        }
        let mut first = None;
        for poison in [f16::ONE.to_bits(), 0x7e00] {
            for b in [&mut k, &mut v] {
                b.with_mut(|b| {
                    for head in 0..kvheads {
                        b[(head * cap + 31) * d..(head * cap + 33) * d].fill(poison);
                    }
                });
            }
            out.with_mut(|b| b.fill(0x7bff));
            status.with_mut(|b| b.fill(0));
            let raw = gpu.command().unwrap();
            gpu.encode(
                &Command {
                    raw: &raw,
                    trace: None,
                },
                Kernel::AttentionPrefill,
                &[&q.raw, &k.raw, &v.raw, &out.raw],
                &[3, 4, 2, 8, 64, 3, 30, 33],
                2 * heads,
                Some(&status),
            )
            .unwrap();
            raw.commit();
            raw.waitUntilCompleted();
            assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
            let actual = out.with_slice(|b| b[..heads * d].to_vec());
            if let Some(ref expected) = first {
                assert_eq!(&actual, expected);
            } else {
                first = Some(actual);
            }
        }
        println!(
            "online adversarial: {cases} uniform/max-jump/extreme/nonfinite witnesses; causal future perturbation passes"
        );
    });
}

#[test]
#[ignore = "explicit Apple GPU high-position split-half RoPE against independent FP64 math"]
fn high_position_rotary_matches_independent_math_envelope() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let mut compared = 0;
        let mut maximum_error = 0f64;
        for d in [64usize, 128] {
            let (rows, heads) = (3usize, 2usize);
            let n = rows * heads * d;
            let mut input = Buffer::<u16>::new(&gpu, n + 7).unwrap();
            input.with_mut(|b| {
                b.fill(0x7e00);
                for (i, v) in b[..n].iter_mut().enumerate() {
                    *v = f16::from_f32(((i * 13) % 31) as f32 * 0.125 - 1.875).to_bits();
                }
            });
            let before = input.with_slice(|b| b.to_vec());
            let mut out = Buffer::<u16>::new(&gpu, n + 7).unwrap();
            let mut status = Buffer::<u32>::new(&gpu, 8).unwrap();
            for offset in [0usize, 2045, 4093, 8189, 16381, 32765] {
                out.with_mut(|b| b.fill(0x7bff));
                status.with_mut(|b| b.fill(0));
                let raw = gpu.command().unwrap();
                gpu.encode(
                    &Command {
                        raw: &raw,
                        trace: None,
                    },
                    Kernel::Rotary,
                    &[&input.raw, &out.raw],
                    &[rows as u32, heads as u32, d as u32, offset as u32],
                    n / 2,
                    Some(&status),
                )
                .unwrap();
                raw.commit();
                raw.waitUntilCompleted();
                assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
                assert_eq!(status.with_slice(|b| b[0]), 0);
                out.with_slice(|bits| {
                    for row in 0..rows {
                        for head in 0..heads {
                            for f in 0..d / 2 {
                                let a = (row * heads + head) * d + f;
                                let b = a + d / 2;
                                let av = f64::from(f16::from_bits(before[a]).to_f32());
                                let bv = f64::from(f16::from_bits(before[b]).to_f32());
                                let frequency = 1e6f64.powf(-2. * f as f64 / d as f64);
                                let angle = (offset + row) as f64 * frequency;
                                let expected = [
                                    av * angle.cos() - bv * angle.sin(),
                                    bv * angle.cos() + av * angle.sin(),
                                ];
                                // Predeclared numerical envelope inspired by
                                // FP32 frequency/angle/trig error (eight epsilons),
                                // plus final Half rounding. Larger positions
                                // amplify frequency error; this is not a model
                                // loss gate, vendor error guarantee or positional extension claim.
                                let scale = av.abs() + bv.abs();
                                let bound = scale
                                    * (8. * f64::from(f32::EPSILON) * (angle.abs() + 1.) + 0.0005)
                                    + 1e-5;
                                for (index, value) in [a, b].into_iter().zip(expected) {
                                    let actual = f64::from(f16::from_bits(bits[index]).to_f32());
                                    let delta = (actual - value).abs();
                                    maximum_error = maximum_error.max(delta);
                                    assert!(
                                        actual.is_finite() && delta <= bound,
                                        "d{d}/pos{}/f{f}: {actual} vs {value}, bound{bound}",
                                        offset + row
                                    );
                                    compared += 1;
                                }
                            }
                        }
                    }
                    assert!(bits[n..].iter().all(|b| *b == 0x7bff));
                });
                assert_eq!(input.with_slice(|b| b.to_vec()), before);
            }
        }
        println!(
            "high-position rotary: {compared} independent values through32767; max_abs_error={maximum_error}"
        );
    });
}
