//! Scalar mapping of unlearned per-D RMS onto the existing learned RMS kernel.
//! Exact-one scales are persistent caller/model data, not checkpoint norm weights.
use serde_json::Value;
use sha2::{Digest, Sha256};
use toklet::cpu::{self, Error};

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
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
fn accepted_unlearned_v_rows_match_both_contracts_and_exact_staged_bits() {
    let receipt = include_bytes!("../benchmarks/results/2026-10-24-gemma-v-norm-reference-v1.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(receipt)),
        "f239fe651e26ace320de099b4344045963f0137d85ec8805551c90bc6fd4dd84"
    );
    let accepted: Value = serde_json::from_slice(receipt).unwrap();
    let raw = include_bytes!("fixtures/gemma-v-norm-v1/math.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        accepted["fixture_sha256"].as_str().unwrap()
    );
    let data: Value = serde_json::from_slice(raw).unwrap();
    // One immutable allocation serves every head/row and both qualified widths.
    // A future model allocation plan must count this shared scale buffer once.
    let unit_scale = vec![1_f32; 512];
    let scale_pointer = unit_scale.as_ptr();
    let mut checked = 0;
    for case in data["cases"].as_array().unwrap() {
        let n = |k: &str| case[k].as_u64().unwrap() as usize;
        let rows = n("batch") * n("tokens") * n("heads");
        if rows == 0 {
            continue;
        }
        let width = n("width");
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
        let before = bits(&input);
        let mut output = vec![f32::NAN; input.len()];
        let pointer = output.as_ptr();
        let epsilon = case["epsilon"].as_f64().unwrap() as f32;
        cpu::rmsnorm(
            &input,
            &unit_scale[..width],
            &mut output,
            rows,
            width,
            epsilon,
        )
        .unwrap();
        close(&output, &values("math"));
        close(&output, &values("staged_fp32"));
        assert_eq!(
            bits(&output),
            bits(
                &values("staged_fp32")
                    .iter()
                    .map(|&v| v as f32)
                    .collect::<Vec<_>>()
            )
        );
        let expected = bits(&output);
        for chunk in [1, 2, rows] {
            for start in (0..rows).step_by(chunk) {
                let stop = (start + chunk).min(rows);
                cpu::rmsnorm(
                    &input[start * width..stop * width],
                    &unit_scale[..width],
                    &mut output[start * width..stop * width],
                    stop - start,
                    width,
                    epsilon,
                )
                .unwrap();
            }
            assert_eq!(bits(&output), expected);
            assert_eq!(pointer, output.as_ptr());
        }
        assert_eq!(before, bits(&input));
        assert_eq!(scale_pointer, unit_scale.as_ptr());
        assert!(unit_scale.iter().all(|v| v.to_bits() == 1_f32.to_bits()));
        checked += 1;
    }
    assert_eq!(checked, 7);
}

#[test]
fn native_robustness_and_signed_zero_are_separate_from_fast_upstream_scope() {
    let scale = [1.; 4];
    let input = [f32::MAX, 0., 0., 0., -f32::MAX, 0., 0., 0.];
    let before = bits(&input);
    let mut output = [0.; 8];
    cpu::rmsnorm(&input, &scale, &mut output, 2, 4, 1e-6).unwrap();
    assert_eq!(output, [2., 0., 0., 0., -2., 0., 0., 0.]);
    assert_eq!(before, bits(&input));
    let tiny = f32::from_bits(1);
    let input = [-0., 0., tiny, -tiny];
    let mut output = [f32::NAN; 4];
    cpu::rmsnorm(&input, &scale, &mut output, 1, 4, 1e-6).unwrap();
    let denom = (input.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / 4.
        + f64::from(1e-6_f32))
    .sqrt();
    let expected = input.map(|v| (f64::from(v) / denom) as f32);
    assert_eq!(bits(&output), bits(&expected));
    assert_eq!(output[0].to_bits(), (-0_f32).to_bits());
    assert_ne!(output[2], 0.);
    // Explicitly guard mean-square rather than sum-only L2 normalization.
    let input = [2., 0., 0., 0.];
    cpu::rmsnorm(&input, &scale, &mut output, 1, 4, 1e-6).unwrap();
    assert!(output[0] > 1.99);
}

#[test]
fn invalid_mapping_inputs_preserve_output_and_all_borrowed_data() {
    let input = [1., -2., 3., 4.];
    let scale = [1.; 4];
    let mut output = [-17.; 4];
    let before = bits(&output);
    for epsilon in [0., -1., f32::NAN, f32::INFINITY] {
        assert!(cpu::rmsnorm(&input, &scale, &mut output, 1, 4, epsilon).is_err());
        assert_eq!(before, bits(&output));
    }
    for (rows, width) in [(0, 4), (1, 0), (usize::MAX, 4), (1, usize::MAX)] {
        assert!(cpu::rmsnorm(&input, &scale, &mut output, rows, width, 1e-6).is_err());
        assert_eq!(before, bits(&output));
    }
    let mut nonfinite = input;
    nonfinite[3] = f32::NAN;
    assert!(matches!(
        cpu::rmsnorm(&nonfinite, &scale, &mut output, 1, 4, 1e-6),
        Err(Error::NonFinite { index: 3, .. })
    ));
    assert_eq!(before, bits(&output));
    assert!(cpu::rmsnorm(&input[..3], &scale, &mut output, 1, 4, 1e-6).is_err());
    assert_eq!(before, bits(&output));
    assert!(cpu::rmsnorm(&input, &scale[..3], &mut output, 1, 4, 1e-6).is_err());
    assert_eq!(before, bits(&output));
    assert!(cpu::rmsnorm(&input, &scale, &mut output[..3], 1, 4, 1e-6).is_err());
    assert_eq!(before, bits(&output));
    assert_eq!(input, [1., -2., 3., 4.]);
    assert_eq!(scale, [1.; 4]);
}
