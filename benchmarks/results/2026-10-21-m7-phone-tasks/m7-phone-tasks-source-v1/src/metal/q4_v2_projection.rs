//! Exact dyadic primitive witnesses, independent of the production quantizer.
use super::*;
use half::f16;

fn packed(rows: usize, columns: usize, subnormal: bool) -> Vec<u8> {
    let stride = (columns.div_ceil(8) * 8).div_ceil(64) * 64;
    let mut bytes = vec![0; 64 + rows * stride];
    bytes[..8].copy_from_slice(b"TOKQ4V2\0");
    bytes[8..12].copy_from_slice(&2u32.to_le_bytes());
    bytes[12..16].copy_from_slice(&8u32.to_le_bytes());
    for (offset, value) in [(16, rows), (24, columns), (32, stride), (40, rows * stride)] {
        bytes[offset..offset + 8].copy_from_slice(&(value as u64).to_le_bytes());
    }
    for row in 0..rows {
        for col in 0..columns {
            let start = 64 + row * stride + col / 8 * 8;
            let scale = if subnormal {
                1u16
            } else {
                f16::from_f32(0.125).to_bits()
            };
            bytes[start..start + 2].copy_from_slice(&scale.to_le_bytes());
            let bias = if subnormal {
                0
            } else {
                f16::from_f32(-(row as f32) * 0.25).to_bits()
            };
            bytes[start + 2..start + 4].copy_from_slice(&bias.to_le_bytes());
            bytes[start + 4 + col % 8 / 2] |= (((col + row) % 16) as u8) << (4 * (col % 2));
        }
    }
    bytes
}

#[test]
#[ignore = "Explicit Apple GPU v2 vector/tile exact FP64 oracle, tails and buffer guards"]
fn v2_vector_tile_match_exact_oracle_and_preserve_guards() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let mut compared = 0;
        for columns in [1, 7, 8, 9, 63, 64, 65, 257] {
            for subnormal in [false, true] {
                let data = packed(3, columns, subnormal);
                let oracle = q4::Matrix::parse(&data).unwrap();
                let layout = Layout::new(3, columns, true).unwrap();
                let weight = Tensor::packed(&gpu, layout, &data).unwrap();
                let mut bias = Buffer::<u16>::new(&gpu, 3).unwrap();
                bias.with_mut(|v| {
                    for (i, b) in v.iter_mut().enumerate() {
                        *b = f16::from_f32(i as f32 * 0.125).to_bits()
                    }
                });
                for n in [1usize, 7, 8, 9, 17] {
                    let input_layout = Layout::new(n, columns, true).unwrap();
                    let out_layout = Layout::new(n, 3, true).unwrap();
                    let mut input = Buffer::<u16>::new(&gpu, input_layout.elements).unwrap();
                    input.with_mut(|v| {
                        v.fill(0x7e00);
                        for token in 0..n {
                            for col in 0..columns {
                                v[token * input_layout.stride + col] =
                                    f16::from_f32(((col + token) % 9) as f32 * 0.5 - 2.).to_bits()
                            }
                        }
                    });
                    let mut out = Buffer::<u16>::new(&gpu, out_layout.elements).unwrap();
                    let mut status = Buffer::<u32>::new(&gpu, 4).unwrap();
                    let mut previous = None;
                    for kernel in [Kernel::Q4V2Compensated, Kernel::Q4V2CompensatedPrefill] {
                        let kernel_name = kernel.name();
                        out.with_mut(|v| v.fill(0x7bff));
                        status.with_mut(|v| v.fill(0));
                        let raw = gpu.command().unwrap();
                        let command = Command {
                            raw: &raw,
                            trace: None,
                        };
                        gpu.encode(
                            &command,
                            kernel,
                            &[&weight.buffer.raw, &bias.raw, &input.raw, &out.raw],
                            &[
                                n as u32,
                                columns as u32,
                                3,
                                oracle.row_stride() as u32,
                                input_layout.stride as u32,
                                out_layout.stride as u32,
                            ],
                            if matches!(kernel, Kernel::Q4V2CompensatedPrefill) {
                                n.div_ceil(8) * 3
                            } else {
                                n * 3
                            },
                            Some(&status),
                        )
                        .unwrap();
                        raw.commit();
                        raw.waitUntilCompleted();
                        assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
                        assert_eq!(status.with_slice(|v| v[0]), 0);
                        let values = out.with_slice(|v| v.to_vec());
                        for token in 0..n {
                            for row in 0..3 {
                                let mut exact =
                                    bias.with_slice(|v| f64::from(f16::from_bits(v[row]).to_f32()));
                                for col in 0..columns {
                                    exact += f64::from(oracle.value(row, col).unwrap())
                                        * input.with_slice(|v| {
                                            f64::from(
                                                f16::from_bits(
                                                    v[token * input_layout.stride + col],
                                                )
                                                .to_f32(),
                                            )
                                        });
                                }
                                assert_eq!(
                                    values[token * out_layout.stride + row],
                                    f16::from_f64(exact).to_bits(),
                                    "{kernel_name} n{n} columns{columns} row{row} token{token} subnormal{subnormal}"
                                );
                                compared += 1;
                            }
                        }
                        for (i, &value) in values.iter().enumerate() {
                            if i / out_layout.stride >= n || i % out_layout.stride >= 3 {
                                assert_eq!(value, 0x7bff, "output guard {i}");
                            }
                        }
                        if let Some(before) = &previous {
                            assert_eq!(&values, before, "vector/tile bits");
                        }
                        previous = Some(values);
                    }
                }
            }
        }
        println!(
            "V2 exact FP64-rounded Half comparisons {compared}; vector/tile queried SIMD widths {}/{}; all inactive rows/padding guards pass",
            gpu.execution_width(Kernel::Q4V2Compensated),
            gpu.execution_width(Kernel::Q4V2CompensatedPrefill)
        );
    });
}

#[test]
#[ignore = "Explicit Apple GPU v2 signed Half midpoint, product residual and overflow status"]
fn v2_signed_midpoints_and_overflow_preserve_status_and_guards() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let layout = Layout::new(1, 17, true).unwrap();
        let mut data = packed(1, 17, false);
        // Unit weights: independently authored code1/scale1/bias0 for each feature.
        for group in 0..3 {
            let offset = 64 + group * 8;
            data[offset..offset + 2].copy_from_slice(&f16::ONE.to_bits().to_le_bytes());
            data[offset + 2..offset + 4].fill(0);
            data[offset + 4..offset + 8].fill(0x11);
        }
        data[64 + 16 + 4] = 1;
        data[64 + 16 + 5..64 + 24].fill(0);
        let weight = Tensor::packed(&gpu, layout, &data).unwrap();
        let mut bias = Buffer::<u16>::new(&gpu, 1).unwrap();
        bias.with_mut(|v| v.fill(0));
        let input_layout = Layout::new(9, 17, true).unwrap();
        let out_layout = Layout::new(9, 1, true).unwrap();
        let mut input = Buffer::<u16>::new(&gpu, input_layout.elements).unwrap();
        let mut out = Buffer::<u16>::new(&gpu, out_layout.elements).unwrap();
        let mut status = Buffer::<u32>::new(&gpu, 4).unwrap();
        let mut cases = 0;
        for sign in [-1., 1.] {
            for side in [-1., 1.] {
                for overflow in [false, true] {
                    input.with_mut(|v| {
                        v.fill(0);
                        for token in 0..9 {
                            v[token * input_layout.stride] =
                                f16::from_f32(sign * if overflow { 65504. } else { 2048. })
                                    .to_bits();
                            v[token * input_layout.stride + 1] =
                                f16::from_f32(sign * if overflow { 65504. } else { 1. }).to_bits();
                            v[token * input_layout.stride + 2] =
                                f16::from_f32(sign * side * 2f32.powi(-24)).to_bits();
                        }
                    });
                    let expected = f16::from_f64(
                        f64::from(sign)
                            * (if overflow { 131008. } else { 2049. }
                                + f64::from(side) * 2f64.powi(-24)),
                    );
                    for kernel in [Kernel::Q4V2Compensated, Kernel::Q4V2CompensatedPrefill] {
                        out.with_mut(|v| v.fill(0x7bff));
                        status.with_mut(|v| v.fill(0));
                        let raw = gpu.command().unwrap();
                        let command = Command {
                            raw: &raw,
                            trace: None,
                        };
                        gpu.encode(
                            &command,
                            kernel,
                            &[&weight.buffer.raw, &bias.raw, &input.raw, &out.raw],
                            &[
                                9,
                                17,
                                1,
                                64,
                                input_layout.stride as u32,
                                out_layout.stride as u32,
                            ],
                            if matches!(kernel, Kernel::Q4V2CompensatedPrefill) {
                                2
                            } else {
                                9
                            },
                            Some(&status),
                        )
                        .unwrap();
                        raw.commit();
                        raw.waitUntilCompleted();
                        assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
                        assert_eq!(status.with_slice(|v| v[0]), u32::from(overflow));
                        out.with_slice(|v| {
                            for (i, &bits) in v.iter().enumerate() {
                                if i / out_layout.stride < 9 && i % out_layout.stride == 0 {
                                    if !overflow {
                                        assert_eq!(bits, expected.to_bits());
                                    }
                                } else {
                                    assert_eq!(bits, 0x7bff);
                                }
                            }
                        });
                        cases += 1;
                    }
                }
            }
        }
        println!(
            "V2 {cases} signed midpoint/overflow dispatch witnesses; unchanged failure bits and guards"
        );
    });
}
