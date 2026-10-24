use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn fixture() -> Value {
    let receipt =
        include_bytes!("../../../benchmarks/results/2026-10-24-gemma-gelu-reference-v1.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(receipt)),
        "278af5ea3fe5ddd18dc93484566b6e4786df6e46476e2268f7d6f5be20ada02b"
    );
    let accepted: Value = serde_json::from_slice(receipt).unwrap();
    let raw = include_bytes!("../../../tests/fixtures/gemma-gelu-v1/math.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        accepted["fixture_sha256"].as_str().unwrap()
    );
    serde_json::from_slice(raw).unwrap()
}

fn close(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && (f64::from(a) - e).abs() <= 1e-5 + 1e-4 * e.abs(),
            "{a} differs from {e}"
        );
    }
}

#[test]
fn frozen_independent_gelu_and_geglu_match() {
    for row in fixture()["cases"].as_array().unwrap() {
        let values = |key: &str| {
            row["arrays"][key]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect::<Vec<_>>()
        };
        let gate: Vec<f32> = values("gate").iter().map(|&x| x as f32).collect();
        let up: Vec<f32> = values("up").iter().map(|&x| x as f32).collect();
        assert_eq!(gate.len(), row["width"].as_u64().unwrap() as usize);
        if gate.is_empty() {
            continue; // Empty sequences make no activation calls.
        }
        let mut activation = vec![f32::NAN; gate.len()];
        let mut output = activation.clone();
        gelu_tanh(&gate, &mut activation).unwrap();
        gated_gelu_tanh(&gate, &up, &mut output).unwrap();
        close(&activation, &values("activated"));
        close(&output, &values("output"));
        for ((&a, &u), &out) in activation.iter().zip(&up).zip(&output) {
            assert_eq!((a * u).to_bits(), out.to_bits());
        }
    }
}

#[test]
fn hand_values_tiny_inputs_and_extremes_preserve_tanh_semantics() {
    let mut output = [f32::NAN; 3];
    gelu_tanh(&[-1., 0., 1.], &mut output).unwrap();
    for (&a, e) in output
        .iter()
        .zip([-0.15880800939172324, 0., 0.8411919906082768])
    {
        assert!((f64::from(a) - e).abs() < 3e-8);
    }
    gated_gelu_tanh(&[-1., 0., 1.], &[2., 3., -4.], &mut output).unwrap();
    close(&output, &[-0.3176160187834465, 0., -3.364767962433107]);
    // Relative checks ensure an all-zero near-origin implementation fails.
    gelu_tanh(&[-1e-8, 0., 1e-8], &mut output).unwrap();
    assert!((output[0] / -5e-9 - 1.0).abs() < 1e-6);
    assert!((output[2] / 5e-9 - 1.0).abs() < 1e-6);
    // These FP32 extrema are mathematical boundary tests, not upstream fixtures.
    gelu_tanh(&[-f32::MAX, 32., f32::MAX], &mut output).unwrap();
    assert_eq!(output, [0., 32., f32::MAX]);
}

#[test]
fn invalid_boundaries_preserve_output_before_writes() {
    let mut output = [7.; 2];
    assert!(gelu_tanh(&[], &mut []).is_err());
    assert!(gated_gelu_tanh(&[], &[], &mut []).is_err());
    for size in [1, 3] {
        let mut wrong_output = [7.; 3];
        assert!(gelu_tanh(&[1.; 2], &mut wrong_output[..size]).is_err());
        assert_eq!(wrong_output, [7.; 3]);
        assert!(gated_gelu_tanh(&[1.; 2], &[2.; 3][..size], &mut output).is_err());
        assert!(gated_gelu_tanh(&[1.; 2], &[2.; 2], &mut wrong_output[..size]).is_err());
        assert_eq!(wrong_output, [7.; 3]);
        assert_eq!(output, [7.; 2]);
    }
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(gelu_tanh(&[1., invalid], &mut output).is_err());
        assert_eq!(output, [7.; 2]);
        assert!(gated_gelu_tanh(&[1., invalid], &[2.; 2], &mut output).is_err());
        assert_eq!(output, [7.; 2]);
        assert!(gated_gelu_tanh(&[1.; 2], &[2., invalid], &mut output).is_err());
        assert_eq!(output, [7.; 2]);
    }
}

#[test]
fn gated_overflow_preserves_inputs_and_allows_replay() {
    let gate = [1., f32::MAX];
    let up = [2., 2.];
    let mut output = [7.; 2];
    assert_eq!(
        gated_gelu_tanh(&gate, &up, &mut output),
        Err(Error::NonFinite {
            tensor: "GELU output",
            index: 1,
        })
    );
    assert_ne!(output[0], 7.);
    assert_eq!(output[1], 7.);
    assert_eq!(gate, [1., f32::MAX]);
    assert_eq!(up, [2., 2.]);
    gated_gelu_tanh(&gate, &[2., 0.5], &mut output).unwrap();
    assert!(output.iter().all(|x| x.is_finite()));
    assert_eq!(output[1], f32::MAX * 0.5);
}
