mod support;

use support::Fixture;
use toklet::cpu::{self, Error};

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
fn rectangular_manual_projection_supports_batch_bias_and_matvec() {
    let fixture = Fixture::tiny();
    let input = fixture.f32("manual.inputs");
    let weights = fixture.f32("manual.weight");
    let bias = fixture.f32("manual.bias");
    let mut output = [0.0; 4];
    cpu::matmul(&input, &weights, Some(&bias), &mut output, 2, 3, 2).unwrap();
    assert_eq!(output, [-3.75, 14.5, -2.75, 5.0]);
    let mut vector = [0.0; 2];
    cpu::matvec(&input[..3], &weights, Some(&bias), &mut vector, 3, 2).unwrap();
    assert_eq!(vector, output[..2]);
    cpu::matvec(&input[..3], &weights, None, &mut vector, 3, 2).unwrap();
    assert_eq!(vector, [-4.0, 15.0]);
}

#[test]
fn every_tiny_linear_projection_agrees_with_independent_numpy_values() {
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
                let weight_name = format!("{weight_prefix}.{name}_proj.weight");
                let shape = fixture.shape(&weight_name);
                let weights = fixture.f32(&weight_name);
                let bias =
                    has_bias.then(|| fixture.f32(&format!("{weight_prefix}.{name}_proj.bias")));
                let input = fixture.f32(&format!("{values}.{input_name}"));
                let expected = fixture.f32(&format!("{values}.{expected_name}"));
                let mut output = vec![0.0; expected.len()];
                cpu::matmul(
                    &input,
                    &weights,
                    bias.as_deref(),
                    &mut output,
                    tokens.len(),
                    shape[1],
                    shape[0],
                )
                .unwrap();
                close(&output, &expected);
                for (row, expected_row) in input
                    .chunks_exact(shape[1])
                    .zip(expected.chunks_exact(shape[0]))
                {
                    let mut output = vec![0.0; shape[0]];
                    cpu::matvec(
                        row,
                        &weights,
                        bias.as_deref(),
                        &mut output,
                        shape[1],
                        shape[0],
                    )
                    .unwrap();
                    close(&output, expected_row);
                }
            }
        }
        let mut logits = vec![0.0; tokens.len() * 64];
        cpu::matmul(
            &fixture.f32(&format!("cases.{case}.final_norm")),
            &fixture.f32("weights.embed_tokens"),
            None,
            &mut logits,
            tokens.len(),
            32,
            64,
        )
        .unwrap();
        close(&logits, &fixture.f32(&format!("cases.{case}.logits")));
    }
}

#[test]
fn embeddings_and_in_place_residuals_match_saved_values() {
    let fixture = Fixture::tiny();
    let table = fixture.f32("weights.embed_tokens");
    for case in ["prompt", "future_changed", "offset", "single", "capacity"] {
        let tokens = fixture.u32(&format!("cases.{case}.tokens"));
        let mut activation = vec![0.0; tokens.len() * 32];
        cpu::embedding(&tokens, &table, &mut activation, 64, 32).unwrap();
        close(
            &activation,
            &fixture.f32(&format!("cases.{case}.embedding")),
        );
        for layer in 0..2 {
            let prefix = format!("cases.{case}.layers.{layer}");
            cpu::residual_add(
                &mut activation,
                &fixture.f32(&format!("{prefix}.attention_projection")),
            )
            .unwrap();
            close(
                &activation,
                &fixture.f32(&format!("{prefix}.attention_residual")),
            );
            cpu::residual_add(
                &mut activation,
                &fixture.f32(&format!("{prefix}.down_projection")),
            )
            .unwrap();
            close(&activation, &fixture.f32(&format!("{prefix}.output")));
        }
    }
}

#[test]
fn invalid_projection_inputs_fail_before_writing_output() {
    let mut output = [123.0; 2];
    assert!(matches!(
        cpu::matvec(&[1.0], &[1.0; 6], None, &mut output, 3, 2),
        Err(Error::Length {
            tensor: "input",
            ..
        })
    ));
    assert!(matches!(
        cpu::matvec(&[1.0; 3], &[1.0; 5], None, &mut output, 3, 2),
        Err(Error::Length {
            tensor: "weights",
            ..
        })
    ));
    assert!(matches!(
        cpu::matvec(&[1.0; 3], &[1.0; 6], Some(&[0.0]), &mut output, 3, 2),
        Err(Error::Length { tensor: "bias", .. })
    ));
    assert!(matches!(
        cpu::matvec(&[1.0; 3], &[f32::NAN; 6], None, &mut output, 3, 2),
        Err(Error::NonFinite {
            tensor: "weights",
            ..
        })
    ));
    assert_eq!(output, [123.0; 2]);
    assert_eq!(
        cpu::matmul(&[], &[], None, &mut [], 0, 3, 2),
        Err(Error::ZeroDimension("tokens"))
    );
    assert_eq!(
        cpu::matmul(&[], &[], None, &mut [], usize::MAX, 2, 1),
        Err(Error::SizeOverflow)
    );
    assert!(matches!(
        cpu::matvec(&[1.0; 3], &[1.0; 6], None, &mut [], 3, 2),
        Err(Error::Length {
            tensor: "output",
            ..
        })
    ));
}

#[test]
fn invalid_embedding_and_residual_inputs_fail_before_mutation() {
    let mut output = [123.0; 4];
    assert_eq!(
        cpu::embedding(&[0, 2], &[1.0; 4], &mut output, 2, 2),
        Err(Error::Token {
            position: 1,
            token: 2,
            vocab_size: 2
        })
    );
    assert!(cpu::embedding(&[0, 1], &[1.0; 3], &mut output, 2, 2).is_err());
    assert!(cpu::residual_add(&mut output, &[1.0]).is_err());
    assert!(cpu::residual_add(&mut output, &[f32::INFINITY; 4]).is_err());
    assert_eq!(output, [123.0; 4]);
}

#[test]
fn arithmetic_overflow_is_reported() {
    let mut output = [0.0];
    assert_eq!(
        cpu::matvec(&[f32::MAX], &[2.0], None, &mut output, 1, 1),
        Err(Error::NonFinite {
            tensor: "output",
            index: 0
        })
    );
    assert_eq!(
        cpu::residual_add(&mut [f32::MAX], &[f32::MAX]),
        Err(Error::NonFinite {
            tensor: "output",
            index: 0
        })
    );
}
