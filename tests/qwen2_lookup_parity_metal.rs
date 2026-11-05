#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use std::path::Path;
use toklet::{
    generation::{GreedyVerification, VerifiedNext, verify_greedy},
    metal::{self, AttentionMode, NormalizationMode, ProjectionMode, ReadbackMode},
    sampling,
    session::Options,
    weights::LoadOptions,
};

#[test]
#[ignore = "explicit pinned Qwen2.5 MLX8 Metal lookup parity and rollback"]
fn qwen2_metal_lookup_batches_preserve_serial_decisions() {
    const BUDGET: usize = 1280 * 1024 * 1024;
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("models/MLX-Qwen2.5-0.5B-Instruct-8bit-6474d4c");
    let model = metal::Model::load_mlx_qwen2_audited(
        &path,
        LoadOptions {
            context_limit: 256,
            memory_limit_bytes: BUDGET,
        },
    )
    .unwrap();
    assert!(model.mlx_qwen2_source_content_identity().is_some());
    let prompt: Vec<_> = include_bytes!("../benchmarks/contract-v1/prompt_128.u32le")
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    assert_eq!(prompt.len(), 128);
    let width = model.config().vocab_size;
    let open = |diagnostic| {
        let mut session = metal::Session::with_attention_mode(
            &model,
            Options {
                capacity: 256,
                prefill_chunk_size: 8,
                position_offset: 0,
                memory_limit_bytes: BUDGET,
            },
            ProjectionMode::MlxAffine,
            NormalizationMode::Compensated,
            ReadbackMode::FullLogits,
            AttentionMode::Online,
        )
        .unwrap();
        if diagnostic {
            session.enable_diagnostics(&[]).unwrap();
        }
        session.set_tokens(&prompt).unwrap();
        session.evaluate().unwrap();
        session
    };
    let mut original = open(false);
    let base = original.logits().unwrap().to_vec();
    let mut generated = [0; 9];
    for (index, id) in generated.iter_mut().enumerate() {
        *id = original.greedy_token().unwrap().unwrap();
        if index < 8 {
            original.append(&[*id]).unwrap();
            original.evaluate().unwrap();
        }
    }
    drop(original);

    let mut compared = 0;
    let mut greedy_differences = Vec::new();
    let mut min_margin = f32::INFINITY;
    let mut max_logit_difference = 0.0f32;
    for (case, reject, stop, abandon) in [
        ("full", None, None, false),
        ("first", Some(0), None, false),
        ("middle", Some(4), None, false),
        ("stop", None, Some(2), false),
        ("abandon", None, None, true),
    ] {
        let mut proposals: [u32; 8] = generated[..8].try_into().unwrap();
        if let Some(index) = reject {
            proposals[index] = (proposals[index] + 1) % u32::try_from(width).unwrap();
        }
        // A repeated greedy ID could stop before the selected row. The
        // independent expected index uses the first actual occurrence.
        let stop_tokens: Vec<_> = stop.map(|index| generated[index]).into_iter().collect();
        let accepted = reject.unwrap_or_else(|| {
            generated[..8]
                .iter()
                .position(|id| stop_tokens.contains(id))
                .unwrap_or(8)
        });
        let next = generated[accepted];
        let expected = GreedyVerification {
            accepted,
            next: if stop_tokens.contains(&next) {
                VerifiedNext::Stop(next)
            } else {
                VerifiedNext::Token(next)
            },
        };
        let mut serial = open(false);
        let mut batch = open(true);
        assert_eq!(serial.logits().unwrap(), base);
        assert_eq!(batch.logits().unwrap(), base);
        assert!(serial.memory().admitted_bytes + batch.memory().admitted_bytes <= 2 * BUDGET);
        batch.append(&proposals).unwrap();
        assert_eq!(batch.evaluate_next_chunk().unwrap(), proposals.len());
        let snapshot = batch
            .diagnostics()
            .unwrap()
            .into_iter()
            .find(|stage| stage.stage == "logits")
            .unwrap();
        assert_eq!((snapshot.rows, snapshot.columns), (8, width));
        let mut rows = base.clone();
        rows.extend_from_slice(&snapshot.values);
        for (index, &token) in proposals.iter().enumerate() {
            serial.append(&[token]).unwrap();
            assert_eq!(serial.evaluate_next_chunk().unwrap(), 1);
            let actual = &snapshot.values[index * width..][..width];
            let expected = serial.logits().unwrap();
            let top = sampling::greedy(expected).unwrap() as usize;
            let runner_up = expected
                .iter()
                .enumerate()
                .filter(|&(id, _)| id != top)
                .map(|(_, &value)| value)
                .fold(f32::NEG_INFINITY, f32::max);
            min_margin = min_margin.min(expected[top] - runner_up);
            let candidate = sampling::greedy(actual).unwrap();
            if top as u32 != candidate {
                greedy_differences.push((case, index + 1, top as u32, candidate));
            }
            for (&x, &y) in actual.iter().zip(expected) {
                assert!(x.is_finite() && y.is_finite());
                max_logit_difference = max_logit_difference.max((x - y).abs());
                compared += 1;
            }
        }
        let observed = verify_greedy(&proposals, &rows, width, &stop_tokens).unwrap();
        assert_eq!(observed, expected, "{case}");
        let keep = if abandon { 0 } else { observed.accepted };
        let mut recovered = prompt.clone();
        recovered.extend_from_slice(&proposals[..keep]);
        batch.set_tokens(&recovered).unwrap();
        batch.evaluate().unwrap();
        assert_eq!(batch.history(), recovered);
        assert_eq!(batch.evaluated_len(), recovered.len());
        assert_eq!(batch.pending_len(), 0);
        assert_eq!(batch.greedy_token().unwrap().unwrap(), generated[keep]);
        let recovered_logits = batch.logits().unwrap().to_vec();
        if abandon {
            assert_eq!(recovered_logits, base);
        } else {
            batch.append(&[next]).unwrap();
            assert!(batch.logits().is_none());
            assert_eq!(batch.pending_len(), 1);
            assert_eq!(batch.history().last(), Some(&next));
            batch.discard_pending();
            assert_eq!(batch.logits().unwrap(), recovered_logits);
        }
    }
    println!(
        "LOOKUP_METAL {}",
        serde_json::json!({
            "cases":5,"compared":compared,"greedy_differences":greedy_differences,
            "minimum_serial_margin":min_margin,"maximum_logit_difference":max_logit_difference
        })
    );
    assert!(
        greedy_differences.is_empty(),
        "strict target decision mismatch"
    );
}
