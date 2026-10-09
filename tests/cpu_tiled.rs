mod support;

use support::Fixture;
use toklet::cpu::{self, Error};

fn exact_projection(
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    tokens: usize,
    features: usize,
    outputs: usize,
) -> Vec<f32> {
    let mut scalar = vec![0.0; tokens * outputs];
    let mut tiled = scalar.clone();
    cpu::matmul_f64(input, weights, bias, &mut scalar, tokens, features, outputs).unwrap();
    cpu::matmul_f64_tiled(input, weights, bias, &mut tiled, tokens, features, outputs).unwrap();
    for (index, (&actual, &expected)) in tiled.iter().zip(&scalar).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "index {index}");
    }
    tiled
}

#[test]
fn independent_tiny_projections_keep_frozen_tolerances_and_exact_scalar_bits() {
    let fixture = Fixture::tiny();
    for case in ["prompt", "future_changed", "offset", "single", "capacity"] {
        let tokens = fixture.u32(&format!("cases.{case}.tokens"));
        for layer in 0..2 {
            let weight_prefix = format!("weights.layers.{layer}");
            let values = format!("cases.{case}.layers.{layer}");
            for (name, input_name, expected_name, has_bias) in [
                ("q", "input_norm", "q_projection", true),
                ("k", "input_norm", "k_projection", true),
                ("v", "input_norm", "v_projection", true),
                ("o", "attention_context", "attention_projection", false),
                ("gate", "post_attention_norm", "gate_projection", false),
                ("up", "post_attention_norm", "up_projection", false),
                ("down", "gated", "down_projection", false),
            ] {
                let name = format!("{weight_prefix}.{name}_proj.weight");
                let shape = fixture.shape(&name);
                let bias = has_bias.then(|| fixture.f32(&name.replace(".weight", ".bias")));
                let actual = exact_projection(
                    &fixture.f32(&format!("{values}.{input_name}")),
                    &fixture.f32(&name),
                    bias.as_deref(),
                    tokens.len(),
                    shape[1],
                    shape[0],
                );
                let expected = fixture.f32(&format!("{values}.{expected_name}"));
                for (&actual, &expected) in actual.iter().zip(&expected) {
                    assert!((actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs());
                }
            }
        }
        exact_projection(
            &fixture.f32(&format!("cases.{case}.final_norm")),
            &fixture.f32("weights.embed_tokens"),
            None,
            tokens.len(),
            32,
            64,
        );
    }
}

#[test]
fn rectangular_tiles_and_all_tails_preserve_bias_and_ordered_cancellation() {
    for tokens in [1, 2, 3, 4, 5, 7, 8, 9] {
        for outputs in [1, 2, 3, 4, 5, 7, 8, 9] {
            for features in [1, 3, 7, 16, 31] {
                let input: Vec<_> = (0..tokens * features)
                    .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
                    .collect();
                let weights: Vec<_> = (0..outputs * features)
                    .map(|i| ((i % 23) as f32 - 11.0) / 16.0)
                    .collect();
                let bias: Vec<_> = (0..outputs).map(|i| i as f32 / 4.0).collect();
                for bias in [None, Some(bias.as_slice())] {
                    exact_projection(&input, &weights, bias, tokens, features, outputs);
                }
            }
        }
    }
    // A tree reduction or FP32 accumulation would change these small terms.
    let input = [1.0; 5 * 7];
    let weights = [1e8, 1.0, -1e8, 1e20, 1.0, -1e20, 0.5].repeat(7);
    let actual = exact_projection(&input, &weights, Some(&[0.25; 7]), 5, 7, 7);
    assert_eq!(actual, vec![0.75; 35]);
    let fixture = Fixture::tiny();
    assert_eq!(
        exact_projection(
            &fixture.f32("manual.inputs"),
            &fixture.f32("manual.weight"),
            Some(&fixture.f32("manual.bias")),
            2,
            3,
            2,
        ),
        [-3.75, 14.5, -2.75, 5.0]
    );
}

#[test]
fn fp32_range_subnormals_and_signed_zeros_match_the_scalar_oracle() {
    let input = [
        f32::MAX,
        -f32::MAX,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        -0.0,
        0.0,
    ]
    .repeat(5);
    let weights = [0.5, 0.5, -1.0, 1.0, -0.0, 0.0].repeat(7);
    exact_projection(&input, &weights, None, 5, 6, 7);
}

#[test]
fn invalid_boundaries_fail_before_writes_and_arithmetic_reports_global_index() {
    let cases = [
        (0, 1, 1, Error::ZeroDimension("tokens")),
        (1, 0, 1, Error::ZeroDimension("in_features")),
        (1, 1, 0, Error::ZeroDimension("out_features")),
        (usize::MAX, 2, 1, Error::SizeOverflow),
    ];
    for (tokens, features, outputs, error) in cases {
        assert_eq!(
            cpu::matmul_f64_tiled(&[], &[], None, &mut [], tokens, features, outputs),
            Err(error)
        );
    }
    let mut output = [123.0; 20];
    for (input, weights, bias, error_tensor) in [
        (&[1.0; 3][..], &[1.0; 4][..], None, "input"),
        (&[1.0; 5][..], &[1.0; 3][..], None, "weights"),
        (&[1.0; 5][..], &[1.0; 4][..], Some(&[0.0; 3][..]), "bias"),
    ] {
        assert!(matches!(
            cpu::matmul_f64_tiled(input, weights, bias, &mut output, 5, 1, 4),
            Err(Error::Length { tensor, .. }) if tensor == error_tensor
        ));
        assert_eq!(output, [123.0; 20]);
    }
    for (input, weights, bias, tensor, index) in [
        (
            &[1.0, 1.0, 1.0, 1.0, f32::NAN][..],
            &[1.0; 4][..],
            None,
            "input",
            4,
        ),
        (
            &[1.0; 5][..],
            &[1.0, 1.0, f32::INFINITY, 1.0][..],
            None,
            "weights",
            2,
        ),
        (
            &[1.0; 5][..],
            &[1.0; 4][..],
            Some(&[0.0, f32::NAN, 0.0, 0.0][..]),
            "bias",
            1,
        ),
    ] {
        assert_eq!(
            cpu::matmul_f64_tiled(input, weights, bias, &mut output, 5, 1, 4),
            Err(Error::NonFinite { tensor, index })
        );
        assert_eq!(output, [123.0; 20]);
    }
    assert!(matches!(
        cpu::matmul_f64_tiled(&[1.0; 5], &[1.0; 4], None, &mut [], 5, 1, 4),
        Err(Error::Length {
            tensor: "output",
            ..
        })
    ));
    assert_eq!(
        cpu::matmul_f64_tiled(
            &[1.0, 1.0, 1.0, 1.0, f32::MAX],
            &[1.0, 2.0, 1.0, 1.0],
            None,
            &mut output,
            5,
            1,
            4,
        ),
        Err(Error::NonFinite {
            tensor: "output",
            index: 17
        })
    );
}
