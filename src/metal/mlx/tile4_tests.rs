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

#[test]
#[ignore = "Explicit pinned Qwen tile4 cancellation, failure replay and capacity"]
fn affine_tile4_real_state_witness() {
    use crate::{generation, session, weights};
    use sha2::{Digest, Sha256};
    use std::{
        path::Path,
        sync::atomic::{AtomicBool, Ordering},
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = super::super::Model::load_mlx_qwen2_audited(
        &root.join("models/MLX-Qwen2.5-0.5B-Instruct-8bit-6474d4c"),
        weights::LoadOptions {
            context_limit: 1152,
            memory_limit_bytes: 1536 * 1024 * 1024,
        },
    )
    .unwrap();
    let manifest_bytes = std::fs::read(root.join("benchmarks/contract-v1/manifest.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&manifest_bytes)),
        "3f09dcf70f6d9620ce5172dcbaa6428f3487a3edf412f2bfce57937faab72c66"
    );
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes).unwrap();
    let tokens = |key: &str, count: usize| {
        let record = &manifest["arrays"][key];
        let data = std::fs::read(
            root.join("benchmarks/contract-v1")
                .join(record["file"].as_str().unwrap()),
        )
        .unwrap();
        assert_eq!(data.len(), count * 4);
        assert_eq!(record["sha256"], format!("{:x}", Sha256::digest(&data)));
        data.chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let prompt = tokens("prompt_512", 512);
    let continuation = tokens("continuation_128", 128);
    let mut options = session::Options {
        capacity: 1152,
        position_offset: 0,
        prefill_chunk_size: 128,
        memory_limit_bytes: 1536 * 1024 * 1024,
    };
    let open = |options| {
        super::super::Session::with_attention_mode(
            &model,
            options,
            super::super::ProjectionMode::MlxAffine,
            super::super::NormalizationMode::Compensated,
            super::super::ReadbackMode::FullLogits,
            super::super::AttentionMode::Online,
        )
        .unwrap()
    };
    let mut base = open(options);
    let initial = base.memory();
    options.memory_limit_bytes = initial.admitted_bytes + super::super::ALLOWANCE - 1;
    let mut rejected = open(options);
    assert!(matches!(
        rejected.enable_affine_prefill_tile4(),
        Err(Error::Budget { .. })
    ));
    assert_eq!(rejected.memory(), initial);
    drop(rejected);
    options.memory_limit_bytes =
        1536 * 1024 * 1024 - (initial.admitted_bytes - model.model_resident_bytes().unwrap());
    let mut tile = open(options);
    tile.enable_affine_prefill_tile4().unwrap();
    assert_eq!(
        tile.memory().admitted_bytes,
        initial.admitted_bytes + super::super::ALLOWANCE
    );
    tile.enable_profile().unwrap();
    for s in [&mut base, &mut tile] {
        s.append(&prompt).unwrap();
        s.evaluate().unwrap();
    }
    assert_eq!(
        tile.profile()
            .unwrap()
            .events
            .iter()
            .filter(|e| e.name == "mlx_product_tile4")
            .count(),
        168
    );
    let parity = |base: &super::super::Session<'_>, tile: &super::super::Session<'_>| {
        assert_eq!(base.history(), tile.history());
        assert_eq!(base.evaluated_len(), tile.evaluated_len());
        assert_eq!(base.pending_len(), tile.pending_len());
        match (base.logits(), tile.logits()) {
            (Some(a), Some(b)) => assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())),
            (None, None) => {}
            _ => panic!("distribution availability differs"),
        }
    };
    parity(&base, &tile);

    tile.append(&continuation).unwrap();
    // Inject a malformed private input after public append. Embedding rejects
    // it through GPU status, and the candidate writes only after the prefix.
    tile.scratch.tokens.with_mut(|b| {
        b[..128].copy_from_slice(&continuation);
        b[0] = model.config().vocab_size as u32;
    });
    tile.scratch.status.with_mut(|b| b.fill(0));
    tile.diagnostic_rows = 0;
    let result = autoreleasepool(|_| tile.run_chunk(128, 512, None));
    assert!(matches!(
        tile.finish_chunk(128, result),
        Err(Error::Nonfinite)
    ));
    assert_eq!(tile.evaluated_len(), 512);
    assert!(tile.logits().is_none());
    assert!(tile.profile().is_err());
    tile.discard_pending();
    parity(&base, &tile);
    assert!(matches!(
        tile.finish_chunk(1, Err(Error::Completion)),
        Err(Error::Completion)
    ));
    assert!(matches!(
        tile.evaluate_next_chunk(),
        Err(Error::ReplayRequired)
    ));
    tile.set_tokens(&prompt).unwrap();
    tile.evaluate().unwrap();
    parity(&base, &tile);

    let changed = tokens("changed_prompt_512", 512);
    for s in [&mut base, &mut tile] {
        s.set_tokens(&changed).unwrap();
        s.evaluate().unwrap();
    }
    parity(&base, &tile);
    for s in [&mut base, &mut tile] {
        let cancelled = AtomicBool::new(false);
        let result = generation::generate(
            s,
            generation::Request {
                input: generation::Input::Reset(&prompt),
                max_new_tokens: 4,
                stop_tokens: &[],
            },
            &cancelled,
            |_| {
                cancelled.store(true, Ordering::Relaxed);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(result.reason, generation::StopReason::Cancelled);
        assert_eq!(
            (
                result.sampled_tokens,
                result.evaluated_tokens,
                result.pending_tokens
            ),
            (1, 512, 1)
        );
    }
    parity(&base, &tile);
    let mut boundary = tokens("prompt_1024", 1024);
    boundary.extend(&continuation);
    for s in [&mut base, &mut tile] {
        s.reset();
        s.append(&boundary).unwrap();
        s.evaluate().unwrap();
        assert!(s.append(&[0]).is_err());
        assert_eq!(s.evaluated_len(), 1152);
        assert_eq!(s.history(), boundary);
    }
    parity(&base, &tile);
    println!(
        "TILE4_STATE {}",
        serde_json::json!({"capacity":1152,"chunk":128,
        "invalid_candidate_preserves_prefix":true,"completion_requires_replay":true,
        "cancellation":true,"changed_prefix":true,"overflow_preserved":true,
        "candidate_memory":tile.memory()})
    );
}
