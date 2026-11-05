use super::super::memory::{Layout as DenseLayout, matrix_offset};
use super::*;

#[test]
#[ignore = "Explicit Apple GPU affine8 tile4 oracle, fallback, tails and guards"]
fn affine_tile4_matches_baseline_and_independent_rows() {
    const COLUMNS: usize = 128;
    const OUT: usize = 5;
    const INPUT_STRIDE: usize = 136;
    const OUTPUT_STRIDE: usize = 8;
    const CAPACITY: usize = 128;
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let pipelines = compile(&gpu).unwrap();
        let tile = gpu.affine_tile4().unwrap();
        assert!(std::ptr::eq(tile, gpu.affine_tile4().unwrap()));
        let layout = Layout::new(OUT, COLUMNS, 8, 64, MetadataDtype::F16).unwrap();
        let mut raw: Vec<_> = (0..layout.weight_bytes())
            .map(|i| ((i * 17 + i / COLUMNS) % 137) as u8)
            .collect();
        raw[..3].fill(64); // Reconstruct 1.0 in row zero.
        let scales: Vec<_> = (0..OUT * 2)
            .flat_map(|i| f16::from_f32(if i % 2 == 0 { 0.03125 } else { -0.015625 }).to_le_bytes())
            .collect();
        let biases: Vec<_> = (0..OUT * 2)
            .flat_map(|i| f16::from_f32((i % 3) as f32 - 1.).to_le_bytes())
            .collect();
        let oracle = Matrix::from_le_bytes(layout, &raw, &scales, &biases).unwrap();
        let mut packed = Packed::allocate(&gpu, layout, &pipelines).unwrap();
        assert_eq!(
            tile.threadExecutionWidth(),
            packed.pipelines[0].threadExecutionWidth()
        );
        let mut buffer = Buffer::new(&gpu, raw.len() / 2).unwrap();
        packed.upload_matrix(&oracle, &mut buffer).unwrap();
        let weight = super::super::Tensor {
            buffer,
            layout: DenseLayout::new(OUT, COLUMNS, true).unwrap(),
            storage: super::super::TensorStorage::Mlx(packed),
        };
        let value = |t, c| {
            if t < 3 {
                // Row zero lands on a Half midpoint or one exact FP32 term
                // to either side after its linear bias.
                match c {
                    0 => 1.25,
                    1 => 2f32.powi(-11),
                    2 => (t as f32 - 1.) * 2f32.powi(-24),
                    _ => 0.,
                }
            } else {
                ((t * 5 + c * 7) % 17) as f32 / 8. - 1.
            }
        };
        let mut input = Buffer::<u16>::new(&gpu, (CAPACITY + 1) * INPUT_STRIDE + 7).unwrap();
        input.with_mut(|b| {
            b.fill(f16::NAN.to_bits());
            for t in 0..CAPACITY {
                for c in 0..COLUMNS {
                    b[(t + 1) * INPUT_STRIDE + c] = f16::from_f32(value(t, c)).to_bits();
                }
            }
        });
        let mut bias = Buffer::<u16>::new(&gpu, OUT).unwrap();
        bias.with_mut(|b| {
            for (row, word) in b.iter_mut().enumerate() {
                *word = f16::from_f32((row as f32 - 2.) / 16.).to_bits();
            }
        });
        let mut status = Buffer::<u32>::new(&gpu, 1).unwrap();
        let mut output = Buffer::<u16>::new(&gpu, CAPACITY * OUTPUT_STRIDE + 7).unwrap();
        for (batch, overflow) in [
            (1, false),
            (3, false),
            (4, false),
            (5, false),
            (7, false),
            (8, false),
            (9, false),
            (128, false),
            (5, true),
        ] {
            if overflow {
                input.with_mut(|b| {
                    b[5 * INPUT_STRIDE..5 * INPUT_STRIDE + COLUMNS].fill(f16::MAX.to_bits())
                });
            }
            let view = matrix_offset(
                &input,
                DenseLayout {
                    rows: batch,
                    columns: COLUMNS,
                    stride: INPUT_STRIDE,
                    elements: input.len,
                },
                true,
                INPUT_STRIDE,
            )
            .unwrap();
            let mut baseline = Vec::new();
            for tile4 in [None, Some(tile)] {
                output.with_mut(|b| b.fill(HALF_GUARD));
                status.with_mut(|b| b[0] = 4);
                let raw_command = gpu.command().unwrap();
                let command = Command {
                    raw: &raw_command,
                    trace: None,
                };
                product_scheduled(
                    &command,
                    &weight,
                    &bias,
                    &view,
                    &output.raw,
                    OUTPUT_STRIDE,
                    batch,
                    true,
                    tile4,
                    &status,
                )
                .unwrap();
                raw_command.commit();
                raw_command.waitUntilCompleted();
                assert_eq!(raw_command.status(), MTLCommandBufferStatus::Completed);
                assert_eq!(status.with_slice(|b| b[0]), 4 | u32::from(overflow));
                output.with_slice(|actual| {
                    if tile4.is_none() {
                        baseline.extend_from_slice(actual);
                    } else {
                        assert_eq!(actual, baseline, "batch={batch} overflow={overflow}");
                    }
                    for (i, &word) in actual.iter().enumerate() {
                        let (t, r) = (i / OUTPUT_STRIDE, i % OUTPUT_STRIDE);
                        if t >= batch || r >= OUT {
                            assert_eq!(word, HALF_GUARD);
                            continue;
                        }
                        let mut want = (r as f64 - 2.) / 16.;
                        for c in 0..COLUMNS {
                            let x = if overflow && t == 4 {
                                f16::MAX.to_f32()
                            } else {
                                value(t, c)
                            };
                            want += f64::from(oracle.value(r, c).unwrap()) * f64::from(x);
                        }
                        assert_eq!(
                            word,
                            f16::from_f64(want).to_bits(),
                            "batch={batch} row={t} channel={r}"
                        );
                    }
                });
            }
        }
    });
}
