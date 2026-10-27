//! Analytical controls for the existing native Qwen3 profile, not a new kernel.
use super::*;
use half::f16;

const GUARD: usize = 7;
const HALF_GUARD: u16 = 0x3555;
const FLOAT_GUARD: u32 = 0x7fa1_2345;

fn upload(gpu: &Gpu, values: &[u16]) -> Buffer<u16> {
    let mut buffer = Buffer::new(gpu, values.len() + GUARD).unwrap();
    buffer.with_mut(|b| {
        b.fill(HALF_GUARD);
        b[..values.len()].copy_from_slice(values);
    });
    buffer
}

#[test]
#[ignore = "Explicit GPU analytical native-profile RMS widths128/1024; no model"]
fn rms_qwen3_shapes() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        for width in [128, 1024] {
            // Row0 zero; row1 alternating +/-1. Squared sums are exactly0/width.
            // epsilon1e-6 makes the nonzero normalized magnitude slightly below1,
            // comfortably inside Half's rounding interval for1. Dyadic weights
            // keep each expected result inside the corresponding Half interval.
            let weights: Vec<_> = (0..width)
                .map(|f| f16::from_f32((f % 4 + 1) as f32 / 4.).to_bits())
                .collect();
            let values: Vec<_> = (0..2 * width)
                .map(|i| {
                    f16::from_f32(if i < width {
                        0.
                    } else if i % 2 == 0 {
                        1.
                    } else {
                        -1.
                    })
                    .to_bits()
                })
                .collect();
            let expected: Vec<_> = (0..2 * width)
                .map(|i| {
                    if i < width {
                        0
                    } else {
                        weights[i % width] | if i % 2 == 0 { 0 } else { 0x8000 }
                    }
                })
                .collect();
            let input = upload(&gpu, &values);
            let weight = upload(&gpu, &weights);
            let output = upload(&gpu, &vec![HALF_GUARD; values.len()]);
            let status = Buffer::<u32>::new(&gpu, 8).unwrap();
            let raw = gpu.command().unwrap();
            rms(
                &gpu,
                &Command {
                    raw: &raw,
                    trace: None,
                },
                &input,
                &weight,
                &output,
                2,
                width,
                1e-6,
                NormalizationMode::Compensated,
                &status,
            )
            .unwrap();
            raw.commit();
            raw.waitUntilCompleted();
            let actual = output.with_slice(|b| b.to_vec());
            let status_words = status.with_slice(|b| b.to_vec());
            println!(
                "{}",
                serde_json::json!({"kind":"native_profile_rms", "width":width,
                "expected_half_bits":expected, "actual_half_bits":actual, "status":status_words})
            );
            assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
            assert_eq!(status_words, [0; 8]);
            assert_eq!(&actual[..expected.len()], expected);
            assert!(actual[expected.len()..].iter().all(|x| *x == HALF_GUARD));
            for (buffer, original) in [(&input, &values), (&weight, &weights)] {
                buffer.with_slice(|b| {
                    assert_eq!(&b[..original.len()], original);
                    assert!(b[original.len()..].iter().all(|x| *x == HALF_GUARD));
                });
            }
        }
    });
}

#[test]
#[ignore = "Explicit GPU analytical Reference attention16:8/head128 through512 keys; no model"]
fn reference_attention_qwen3_shapes() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let (heads, kvheads, dim, chunk) = (16, 8, 128, 2);
        for keys in [1, 128, 256, 512] {
            for zero in [false, true] {
                let cap = keys + 3;
                let valid = keys + usize::from(keys < 512);
                let q_values = vec![0; heads * dim];
                let mut k_values = vec![0x7e00; kvheads * cap * dim];
                let mut v_values = k_values.clone();
                for head in 0..kvheads {
                    for pos in 0..keys {
                        for feature in 0..dim {
                            let at = (head * cap + pos) * dim + feature;
                            k_values[at] = 0;
                            // Zero scores -> exactly1/keys probabilities for these
                            // power-of-two counts. Constant dyadic V makes ordered
                            // FP32 accumulation and the final Half result exact.
                            let value = if zero {
                                0.
                            } else {
                                (head + 1) as f32 / 16. * if feature % 2 == 0 { 1. } else { -1. }
                            };
                            v_values[at] = f16::from_f32(value).to_bits();
                        }
                    }
                    // A valid but causally future row has finite K and NaN V.
                    // Its score must be ignored by softmax and its V unread.
                    // At512 keys retain the model's512-valid-position limit.
                    if valid > keys {
                        k_values[(head * cap + keys) * dim..(head * cap + valid) * dim].fill(0);
                    }
                }
                let expected: Vec<_> = (0..heads * dim)
                    .map(|i| v_values[((i / dim / 2) * cap) * dim + i % dim])
                    .collect();
                let q = upload(&gpu, &q_values);
                let k = upload(&gpu, &k_values);
                let v = upload(&gpu, &v_values);
                let out = upload(&gpu, &vec![HALF_GUARD; expected.len()]);
                let scratch_len = heads * chunk * cap;
                let mut scores = Buffer::<u32>::new(&gpu, scratch_len + GUARD).unwrap();
                let mut probs = Buffer::<u32>::new(&gpu, scratch_len + GUARD).unwrap();
                scores.with_mut(|b| b.fill(FLOAT_GUARD));
                probs.with_mut(|b| b.fill(FLOAT_GUARD));
                let status = Buffer::<u32>::new(&gpu, 8).unwrap();
                let shape = [1, heads, kvheads, dim, cap, chunk, keys - 1, valid].map(|x| x as u32);
                let raw = gpu.command().unwrap();
                gpu.encode(
                    &Command {
                        raw: &raw,
                        trace: None,
                    },
                    Kernel::Attention,
                    &[&q.raw, &k.raw, &v.raw, &scores.raw, &probs.raw, &out.raw],
                    &shape,
                    heads,
                    Some(&status),
                )
                .unwrap();
                raw.commit();
                raw.waitUntilCompleted();
                let actual = out.with_slice(|b| b.to_vec());
                let status_words = status.with_slice(|b| b.to_vec());
                println!(
                    "{}",
                    serde_json::json!({"kind":"native_profile_attention", "shape":shape,
                    "zero":zero, "expected_half_bits":expected, "actual_half_bits":actual, "status":status_words})
                );
                assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
                assert_eq!(status_words, [0; 8]);
                assert_eq!(&actual[..expected.len()], expected);
                assert!(actual[expected.len()..].iter().all(|x| *x == HALF_GUARD));
                for (buffer, original) in [(&q, &q_values), (&k, &k_values), (&v, &v_values)] {
                    buffer.with_slice(|b| {
                        assert_eq!(&b[..original.len()], original);
                        assert!(b[original.len()..].iter().all(|x| *x == HALF_GUARD));
                    });
                }
                for (buffer, exact) in [(&scores, 0f32), (&probs, 1. / keys as f32)] {
                    buffer.with_slice(|b| {
                        for head in 0..heads {
                            let base = head * chunk * cap;
                            assert!(b[base..base + keys].iter().all(|x| *x == exact.to_bits()));
                            assert!(b[base + keys..base + valid].iter().all(|x| *x == 0));
                            assert!(
                                b[base + valid..base + chunk * cap]
                                    .iter()
                                    .all(|x| *x == FLOAT_GUARD)
                            );
                        }
                        assert!(b[scratch_len..].iter().all(|x| *x == FLOAT_GUARD));
                    });
                }
            }
        }
    });
}
