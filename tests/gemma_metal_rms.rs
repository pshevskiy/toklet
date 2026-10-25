//! Download/GPU-independent scalar check of the frozen FP32 Metal RMS contract.
use serde_json::Value;
use sha2::{Digest, Sha256};
use toklet::cpu;

fn reference() -> Value {
    let raw = include_bytes!("fixtures/gemma-metal-rms-v1/math.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "935e1ecac3d8418cd4f668b5200e3e58730c0798391561bebe1b945391e859bb"
    );
    let manifest = include_bytes!("fixtures/gemma-metal-rms-v1/manifest.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(manifest)),
        "e889c6c2461c5fe98f77d5f2dde3165261d355462fb0dafa949255fc3a1c57f5"
    );
    serde_json::from_slice(raw).unwrap()
}
#[test]
fn scalar_matches_all_independent_finite_rms_cases() {
    let data = reference();
    let mut checked = 0;
    for case in data["cases"].as_array().unwrap() {
        let width = case["width"].as_u64().unwrap() as usize;
        let rows = case["rows"].as_u64().unwrap() as usize;
        let input: Vec<f32> = case["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let weights: Vec<f32> = case["weights"].as_array().map_or_else(
            || vec![1.; width],
            |w| w.iter().map(|v| v.as_f64().unwrap() as f32).collect(),
        );
        let mut output = vec![f32::NAN; input.len()];
        cpu::rmsnorm(&input, &weights, &mut output, rows, width, 1e-6).unwrap();
        for (&actual, expected) in output.iter().zip(case["math_fp64"].as_array().unwrap()) {
            let expected = expected.as_f64().unwrap();
            assert!(
                actual.is_finite()
                    && (f64::from(actual) - expected).abs() <= 1e-5 + 1e-4 * expected.abs(),
                "{}: {actual} vs {expected}",
                case["name"]
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 57685);
}
#[test]
fn scalar_rejects_the_frozen_finite_input_weighted_output_overflow() {
    let data = reference();
    let case = &data["overflow_case"];
    let values = |key: &str| {
        case[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect::<Vec<_>>()
    };
    let mut output = [42.; 2];
    assert!(matches!(
        cpu::rmsnorm(
            &values("input"),
            &values("weights"),
            &mut output,
            1,
            2,
            1e-6
        ),
        Err(cpu::Error::NonFinite {
            tensor: "output",
            index: 0
        })
    ));
    assert_eq!(output, [42.; 2]);
}
