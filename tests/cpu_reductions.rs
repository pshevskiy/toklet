mod support;

use support::Fixture;
use toklet::cpu::{self, Error, MlpWeights};

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs(),
            "index {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn all_normalizations_match_the_frozen_numpy_reference() {
    let fixture = Fixture::tiny();
    for case in ["prompt", "future_changed", "offset", "single", "capacity"] {
        let tokens = fixture.u32(&format!("cases.{case}.tokens")).len();
        for layer in 0..2 {
            let prefix = format!("cases.{case}.layers.{layer}");
            let input_name = if layer == 0 {
                format!("cases.{case}.embedding")
            } else {
                format!("cases.{case}.layers.0.output")
            };
            for (input_name, weight_name, expected_name) in [
                (
                    input_name,
                    format!("weights.layers.{layer}.input_layernorm"),
                    format!("{prefix}.input_norm"),
                ),
                (
                    format!("{prefix}.attention_residual"),
                    format!("weights.layers.{layer}.post_attention_layernorm"),
                    format!("{prefix}.post_attention_norm"),
                ),
            ] {
                let mut output = vec![0.0; tokens * 32];
                cpu::rmsnorm(
                    &fixture.f32(&input_name),
                    &fixture.f32(&weight_name),
                    &mut output,
                    tokens,
                    32,
                    1e-6,
                )
                .unwrap();
                close(&output, &fixture.f32(&expected_name));
            }
        }
        let mut output = vec![0.0; tokens * 32];
        cpu::rmsnorm(
            &fixture.f32(&format!("cases.{case}.layers.1.output")),
            &fixture.f32("weights.norm"),
            &mut output,
            tokens,
            32,
            1e-6,
        )
        .unwrap();
        close(&output, &fixture.f32(&format!("cases.{case}.final_norm")));
    }
}

#[test]
fn softmax_matches_all_causal_probability_arrays() {
    let fixture = Fixture::tiny();
    for case in ["prompt", "future_changed", "offset", "single", "capacity"] {
        let tokens = fixture.u32(&format!("cases.{case}.tokens")).len();
        let mask: Vec<bool> = (0..4 * tokens)
            .flat_map(|row| (0..tokens).map(move |column| column <= row % tokens))
            .collect();
        for layer in 0..2 {
            let prefix = format!("cases.{case}.layers.{layer}");
            let scores = fixture.f32(&format!("{prefix}.attention_scores"));
            let mut output = vec![0.0; scores.len()];
            cpu::softmax(&scores, Some(&mask), &mut output, 4 * tokens, tokens).unwrap();
            close(
                &output,
                &fixture.f32(&format!("{prefix}.attention_probabilities")),
            );
            for row in output.chunks_exact(tokens) {
                assert!((row.iter().sum::<f32>() - 1.0).abs() <= 1e-6);
            }
            assert!(
                output
                    .iter()
                    .zip(&mask)
                    .all(|(&x, &included)| included || x == 0.0)
            );
        }
    }
}

#[test]
fn activations_and_complete_gated_mlp_match_every_saved_layer() {
    let fixture = Fixture::tiny();
    for case in ["prompt", "future_changed", "offset", "single", "capacity"] {
        let tokens = fixture.u32(&format!("cases.{case}.tokens")).len();
        for layer in 0..2 {
            let prefix = format!("cases.{case}.layers.{layer}");
            let gate = fixture.f32(&format!("{prefix}.gate_projection"));
            let up = fixture.f32(&format!("{prefix}.up_projection"));
            let mut activation = vec![0.0; gate.len()];
            cpu::silu(&gate, &mut activation).unwrap();
            close(&activation, &fixture.f32(&format!("{prefix}.gate_silu")));
            cpu::gated_silu(&gate, &up, &mut activation).unwrap();
            close(&activation, &fixture.f32(&format!("{prefix}.gated")));
            let weight_prefix = format!("weights.layers.{layer}");
            let gate_weight = fixture.f32(&format!("{weight_prefix}.gate_proj.weight"));
            let up_weight = fixture.f32(&format!("{weight_prefix}.up_proj.weight"));
            let down_weight = fixture.f32(&format!("{weight_prefix}.down_proj.weight"));
            let weights = MlpWeights {
                gate: &gate_weight,
                up: &up_weight,
                down: &down_weight,
                hidden: 32,
                intermediate: 64,
            };
            let mut gate_scratch = vec![0.0; tokens * 64];
            let mut up_scratch = vec![0.0; tokens * 64];
            let mut output = vec![0.0; tokens * 32];
            cpu::gated_mlp(
                &fixture.f32(&format!("{prefix}.post_attention_norm")),
                &weights,
                &mut output,
                &mut gate_scratch,
                &mut up_scratch,
                tokens,
            )
            .unwrap();
            close(&output, &fixture.f32(&format!("{prefix}.down_projection")));
            close(&gate_scratch, &fixture.f32(&format!("{prefix}.gated")));
            close(&up_scratch, &up);
        }
    }
}

#[test]
fn stable_operations_cover_zero_and_extreme_finite_values() {
    let fixture = Fixture::tiny();
    let mut zero = [123.0; 32];
    cpu::rmsnorm(&[0.0; 32], &[1.0; 32], &mut zero, 1, 32, 1e-6).unwrap();
    close(&zero, &fixture.f32("edge.rmsnorm.zero_output"));
    let mut output = [0.0; 4];
    cpu::rmsnorm(
        &fixture.f32("edge.rmsnorm.large_input"),
        &[1.0; 4],
        &mut output,
        1,
        4,
        1e-6,
    )
    .unwrap();
    close(&output, &fixture.f32("edge.rmsnorm.large_output"));
    cpu::rmsnorm(
        &[f32::MAX, -f32::MAX, 0.0, f32::MAX],
        &[1.0; 4],
        &mut output,
        1,
        4,
        f32::from_bits(1),
    )
    .unwrap();
    close(
        &output,
        &[
            2.0 / 3.0_f32.sqrt(),
            -2.0 / 3.0_f32.sqrt(),
            0.0,
            2.0 / 3.0_f32.sqrt(),
        ],
    );
    let mut probabilities = [0.0; 3];
    cpu::softmax(
        &fixture.f32("edge.softmax.input"),
        None,
        &mut probabilities,
        1,
        3,
    )
    .unwrap();
    close(&probabilities, &fixture.f32("edge.softmax.output"));
    cpu::softmax(
        &[f32::MAX, -f32::MAX, f32::MAX],
        None,
        &mut probabilities,
        1,
        3,
    )
    .unwrap();
    assert_eq!(probabilities, [0.5, 0.0, 0.5]);
    cpu::softmax(
        &[0.0; 3],
        Some(&[false, true, false]),
        &mut probabilities,
        1,
        3,
    )
    .unwrap();
    assert_eq!(probabilities, [0.0, 1.0, 0.0]);
    let mut activation = [0.0; 5];
    cpu::silu(&fixture.f32("edge.silu.input"), &mut activation).unwrap();
    close(&activation, &fixture.f32("edge.silu.output"));
    cpu::silu(&[-f32::MAX, f32::MAX], &mut activation[..2]).unwrap();
    assert_eq!(&activation[..2], &[-0.0, f32::MAX]);
}

#[test]
fn invalid_reduction_inputs_are_rejected_before_output_changes() {
    let mut output = [123.0; 4];
    for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            cpu::rmsnorm(&[1.0; 4], &[1.0; 2], &mut output, 2, 2, epsilon),
            Err(Error::InvalidParameter(_))
        ));
    }
    assert!(matches!(
        cpu::rmsnorm(&[1.0; 3], &[1.0; 2], &mut output, 2, 2, 1e-6),
        Err(Error::Length {
            tensor: "input",
            ..
        })
    ));
    assert!(matches!(
        cpu::rmsnorm(&[1.0; 4], &[f32::NAN; 2], &mut output, 2, 2, 1e-6),
        Err(Error::NonFinite {
            tensor: "norm weight",
            ..
        })
    ));
    assert_eq!(
        cpu::rmsnorm(&[], &[], &mut [], usize::MAX, 2, 1e-6),
        Err(Error::SizeOverflow)
    );
    assert_eq!(
        cpu::softmax(
            &[1.0; 4],
            Some(&[true, false, false, false]),
            &mut output,
            2,
            2
        ),
        Err(Error::FullyMasked { row: 1 })
    );
    assert!(matches!(
        cpu::softmax(&[1.0; 4], Some(&[true; 3]), &mut output, 2, 2),
        Err(Error::Length { tensor: "mask", .. })
    ));
    assert!(matches!(
        cpu::softmax(
            &[1.0, f32::NEG_INFINITY, 1.0, 1.0],
            Some(&[true, false, true, true]),
            &mut output,
            2,
            2
        ),
        Err(Error::NonFinite {
            tensor: "scores",
            index: 1
        })
    ));
    assert_eq!(
        cpu::softmax(&[], None, &mut [], 1, 0),
        Err(Error::ZeroDimension("columns"))
    );
    assert!(matches!(
        cpu::silu(&[f32::NAN; 4], &mut output),
        Err(Error::NonFinite { .. })
    ));
    assert!(matches!(
        cpu::gated_silu(&[1.0; 4], &[f32::INFINITY; 4], &mut output),
        Err(Error::NonFinite { tensor: "up", .. })
    ));
    assert_eq!(output, [123.0; 4]);
}

#[test]
fn mlp_validates_all_buffers_and_reports_arithmetic_overflow() {
    let mut output = [123.0; 2];
    let mut gate = [456.0; 3];
    let mut up = [789.0; 3];
    let mut weights = MlpWeights {
        gate: &[1.0; 6],
        up: &[1.0; 6],
        down: &[1.0; 5],
        hidden: 2,
        intermediate: 3,
    };
    assert!(matches!(
        cpu::gated_mlp(&[1.0; 2], &weights, &mut output, &mut gate, &mut up, 1),
        Err(Error::Length {
            tensor: "down weight",
            ..
        })
    ));
    weights.down = &[f32::NAN; 6];
    assert!(matches!(
        cpu::gated_mlp(&[1.0; 2], &weights, &mut output, &mut gate, &mut up, 1),
        Err(Error::NonFinite {
            tensor: "down weight",
            ..
        })
    ));
    weights.down = &[1.0; 6];
    assert!(matches!(
        cpu::gated_mlp(&[1.0; 2], &weights, &mut output, &mut gate[..2], &mut up, 1),
        Err(Error::Length {
            tensor: "gate scratch",
            ..
        })
    ));
    assert_eq!(output, [123.0; 2]);
    assert_eq!(gate, [456.0; 3]);
    assert_eq!(up, [789.0; 3]);
    assert!(matches!(
        cpu::gated_silu(&[f32::MAX], &[2.0], &mut [0.0]),
        Err(Error::NonFinite { .. })
    ));
    assert!(matches!(
        cpu::rmsnorm(&[1.0, 0.0], &[f32::MAX; 2], &mut output, 1, 2, 1e-6),
        Err(Error::NonFinite { .. })
    ));
}
