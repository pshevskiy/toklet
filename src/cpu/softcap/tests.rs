use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn close(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && (f64::from(a) - e).abs() <= 1e-5 + 1e-4 * e.abs(),
            "{a} vs {e}"
        );
    }
}

#[test]
fn frozen_softcap_matches_both_independent_contracts_with_reused_storage() {
    let receipt =
        include_bytes!("../../../benchmarks/results/2026-10-25-gemma-softcap-reference-v1.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(receipt)),
        "917bb5eeb25e12b7db6930d38046e9e88f3bb91721a7df9120337eb6642123f4"
    );
    let accepted: Value = serde_json::from_slice(receipt).unwrap();
    let raw = include_bytes!("../../../tests/fixtures/gemma-softcap-v1/math.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        accepted["fixture_sha256"].as_str().unwrap()
    );
    let data: Value = serde_json::from_slice(raw).unwrap();
    let mut checked = 0;
    for case in data["cases"].as_array().unwrap() {
        let width = case["width"].as_u64().unwrap() as usize;
        if width == 0 {
            continue;
        }
        let values = |key: &str| {
            case[key]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect::<Vec<_>>()
        };
        let input = values("input")
            .iter()
            .map(|&v| v as f32)
            .collect::<Vec<_>>();
        let cap = case["cap"].as_f64().unwrap() as f32;
        let mut output = input.clone();
        let pointer = output.as_ptr();
        for _ in 0..3 {
            output.copy_from_slice(&input);
            logit_softcap(&mut output, cap).unwrap();
            close(&output, &values("math"));
            close(&output, &values("staged_fp32"));
            assert_eq!(pointer, output.as_ptr());
            for (&before, &after) in input.iter().zip(&output) {
                if before == 0. {
                    assert_eq!(before.to_bits(), after.to_bits());
                }
            }
        }
        checked += 1;
    }
    assert_eq!(checked, 5);
}

#[test]
fn all_admission_errors_preserve_every_input_output_bit() {
    let input = [1., -2., 0., -0., f32::MAX];
    for cap in [0., -0., -1., f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut output = input;
        assert_eq!(
            logit_softcap(&mut output, cap),
            Err(Error::InvalidParameter("logit softcap"))
        );
        assert_eq!(bits(&output), bits(&input));
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut output = input;
        output[4] = value;
        let before = bits(&output);
        assert_eq!(
            logit_softcap(&mut output, 30.),
            Err(Error::NonFinite {
                tensor: "logits",
                index: 4
            })
        );
        assert_eq!(bits(&output), before);
    }
    assert_eq!(
        logit_softcap(&mut [], 30.),
        Err(Error::ZeroDimension("logit softcap width"))
    );
}

#[test]
fn native_full_positive_cap_domain_stays_finite_odd_monotone_and_bounded() {
    // Extreme-cap/subnormal cases are mathematical/native robustness coverage,
    // separate from the upstream witness's cap30 and synthetic smaller caps.
    let tiny = f32::from_bits(1);
    let input = [
        -f32::MAX,
        -30.,
        -1.,
        -tiny,
        -0.,
        0.,
        tiny,
        1.,
        30.,
        f32::MAX,
    ];
    for cap in [tiny, f32::MIN_POSITIVE, 1. / 1024., 0.5, 30., f32::MAX] {
        let mut output = input;
        logit_softcap(&mut output, cap).unwrap();
        assert!(output.iter().all(|v| v.is_finite() && v.abs() <= cap));
        assert!(output.windows(2).all(|pair| pair[0] <= pair[1]));
        for i in 0..output.len() {
            assert_eq!(
                output[i].to_bits(),
                (-output[output.len() - 1 - i]).to_bits()
            );
        }
        assert_eq!(output[4].to_bits(), (-0_f32).to_bits());
        assert_eq!(output[5].to_bits(), 0_f32.to_bits());
        if cap <= 30. {
            assert_eq!(output[0], -cap);
            assert_eq!(output[9], cap);
        }
    }
    let mut hand = [1., 30.];
    logit_softcap(&mut hand, 30.).unwrap();
    assert!(hand[0] > 0.99 && hand[0] < 1.);
    assert!(hand[1] > 22.84 && hand[1] < 22.85);
}

#[test]
fn independent_chunks_match_full_vector_bitwise_without_capacity_changes() {
    let input = (0..1025)
        .map(|i| (i as f32 - 512.) * 0.125)
        .collect::<Vec<_>>();
    let mut full = input.clone();
    logit_softcap(&mut full, 30.).unwrap();
    let mut chunks = input.clone();
    let pointer = chunks.as_ptr();
    let capacity = chunks.capacity();
    for size in [1, 3, 7, 128, 1025] {
        chunks.copy_from_slice(&input);
        for part in chunks.chunks_mut(size) {
            logit_softcap(part, 30.).unwrap();
        }
        assert_eq!(bits(&chunks), bits(&full));
        assert_eq!(pointer, chunks.as_ptr());
        assert_eq!(capacity, chunks.capacity());
    }
}
