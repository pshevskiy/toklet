use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

const BINARY: &[u8] = include_bytes!("../../../tests/fixtures/causal-conv-v1/arrays.bin");
const MANIFEST: &str = include_str!("../../../tests/fixtures/causal-conv-v1/manifest.json");

fn fixture() -> Value {
    let raw = include_str!("../../../benchmarks/results/2026-10-24-causal-conv-reference-v1.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw.as_bytes())),
        "945b0fc825adc144a8ab1b8ec74d30693478345c59b4cde30a67edd3ab955359"
    );
    let receipt: Value = serde_json::from_str(raw).unwrap();
    for (bytes, key) in [
        (MANIFEST.as_bytes(), "fixture_manifest_sha256"),
        (BINARY, "fixture_binary_sha256"),
    ] {
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            receipt[key].as_str().unwrap()
        );
    }
    serde_json::from_str(MANIFEST).unwrap()
}

fn values(row: &Value, name: &str) -> Vec<f64> {
    let item = &row["arrays"][name];
    let offset = item["offset"].as_u64().unwrap() as usize;
    let bytes = item["bytes"].as_u64().unwrap() as usize;
    let raw = &BINARY[offset..offset + bytes];
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        item["sha256"].as_str().unwrap()
    );
    match item["dtype"].as_str().unwrap() {
        "<f4" => raw
            .chunks_exact(4)
            .map(|x| f64::from(f32::from_le_bytes(x.try_into().unwrap())))
            .collect(),
        "<f8" => raw
            .chunks_exact(8)
            .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
            .collect(),
        _ => panic!("Unexpected frozen convolution dtype"),
    }
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
fn frozen_convolution_silu_and_each_raw_history_match() {
    for row in fixture()["cases"].as_array().unwrap() {
        let n = |key: &str| row[key].as_u64().unwrap() as usize;
        let (tokens, channels, kernel) = (n("tokens"), n("channels"), n("kernel"));
        let input = |name| {
            values(row, name)
                .into_iter()
                .map(|x| x as f32)
                .collect::<Vec<_>>()
        };
        let x = input("input");
        let weight = input("weight");
        let initial = input("initial_history");
        let mut history = initial.clone();
        let mut candidate = vec![f32::NAN; history.len()];
        let mut output = vec![f32::NAN; channels];
        let mut activated = vec![f32::NAN; channels];
        let expected_output = values(row, "output");
        let expected_activation = values(row, "activated");
        let expected_history = values(row, "history");
        for token in 0..tokens {
            let before = history.clone();
            causal_depthwise_conv_step(
                &x[token * channels..(token + 1) * channels],
                &weight,
                &history,
                &mut candidate,
                &mut output,
                channels,
                kernel,
            )
            .unwrap();
            assert_eq!(history, before);
            close(
                &output,
                &expected_output[token * channels..(token + 1) * channels],
            );
            super::super::silu(&output, &mut activated).unwrap();
            close(
                &activated,
                &expected_activation[token * channels..(token + 1) * channels],
            );
            assert_eq!(
                candidate.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
                expected_history[token * history.len()..(token + 1) * history.len()]
            );
            std::mem::swap(&mut history, &mut candidate);
        }
        assert_eq!(
            history.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
            values(row, "final_history")
        );
        assert_eq!(initial, input("initial_history"));
    }
}

#[test]
fn asymmetric_hand_case_and_kernel_one_preserve_channel_and_tap_order() {
    let mut candidate = [f32::NAN; 4];
    let mut output = [f32::NAN; 2];
    causal_depthwise_conv_step(
        &[3., 30.],
        &[2., -1., 3., -2., 0.5, 1.],
        &[1., 10., 2., 20.],
        &mut candidate,
        &mut output,
        2,
        3,
    )
    .unwrap();
    assert_eq!(output, [9., 20.]);
    assert_eq!(candidate, [2., 20., 3., 30.]);
    let old = candidate;
    causal_depthwise_conv_step(
        &[4., 40.],
        &[2., -1., 3., -2., 0.5, 1.],
        &old,
        &mut candidate,
        &mut output,
        2,
        3,
    )
    .unwrap();
    assert_eq!(output, [13., 15.]);
    assert_eq!(candidate, [3., 30., 4., 40.]);
    causal_depthwise_conv_step(&[3., 5.], &[2., -1.], &[], &mut [], &mut output, 2, 1).unwrap();
    assert_eq!(output, [6., -5.]);
    // Products exceeding FP32 can cancel to a finite output in FP64.
    causal_depthwise_conv_step(
        &[f32::MAX],
        &[2., -2.],
        &[f32::MAX],
        &mut [0.],
        &mut output[..1],
        1,
        2,
    )
    .unwrap();
    assert_eq!(output[0], 0.);
}

#[test]
fn invalid_dimensions_lengths_and_nonfinite_inputs_preserve_outputs() {
    let x = [1., 2.];
    let weight = [1.; 6];
    let history = [2.; 4];
    for (channels, kernel) in [(0, 3), (2, 0), (usize::MAX, 2), (2, usize::MAX)] {
        let mut candidate = [7.; 4];
        let mut output = [8.; 2];
        assert!(
            causal_depthwise_conv_step(
                &x,
                &weight,
                &history,
                &mut candidate,
                &mut output,
                channels,
                kernel
            )
            .is_err()
        );
        assert_eq!(candidate, [7.; 4]);
        assert_eq!(output, [8.; 2]);
    }
    for bad in 0..5 {
        let mut candidate = [7.; 4];
        let mut output = [8.; 2];
        let result = causal_depthwise_conv_step(
            &x[..if bad == 0 { 1 } else { 2 }],
            &weight[..if bad == 1 { 5 } else { 6 }],
            &history[..if bad == 2 { 3 } else { 4 }],
            &mut candidate[..if bad == 3 { 3 } else { 4 }],
            &mut output[..if bad == 4 { 1 } else { 2 }],
            2,
            3,
        );
        assert!(result.is_err());
        assert_eq!(candidate, [7.; 4]);
        assert_eq!(output, [8.; 2]);
    }
    for tensor in 0..3 {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut x = x;
            let mut weight = weight;
            let mut history = history;
            match tensor {
                0 => x[1] = invalid,
                1 => weight[5] = invalid,
                _ => history[3] = invalid,
            }
            let mut candidate = [7.; 4];
            let mut output = [8.; 2];
            assert!(
                causal_depthwise_conv_step(
                    &x,
                    &weight,
                    &history,
                    &mut candidate,
                    &mut output,
                    2,
                    3
                )
                .is_err()
            );
            assert_eq!(candidate, [7.; 4]);
            assert_eq!(output, [8.; 2]);
        }
    }
}

#[test]
fn output_overflow_keeps_previous_history_and_replay_succeeds() {
    let history = [2., f32::MAX];
    let before = history;
    let mut candidate = [7.; 2];
    let mut output = [8.; 2];
    assert!(matches!(
        causal_depthwise_conv_step(
            &[1., 1.],
            &[1., 1., 2., 1.],
            &history,
            &mut candidate,
            &mut output,
            2,
            2
        ),
        Err(Error::NonFinite { index: 1, .. })
    ));
    assert_eq!(history, before);
    assert_eq!(candidate, [7.; 2]);
    assert_eq!(output[0], 3.);
    causal_depthwise_conv_step(
        &[1., 1.],
        &[1., 1., 0., 1.],
        &history,
        &mut candidate,
        &mut output,
        2,
        2,
    )
    .unwrap();
    assert_eq!(output, [3., 1.]);
    assert_eq!(candidate, [1., 1.]);
    assert_eq!(history, before);
}
