use super::*;
use crate::cpu::{embedding, matmul_f64};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn shape() -> PerLayerInputShape {
    PerLayerInputShape {
        tokens: 1,
        hidden: 4,
        layers: 2,
        width: 2,
        epsilon: 1e-6,
    }
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
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn frozen_full_path_and_stages_match_both_independent_contracts() {
    let receipt =
        include_bytes!("../../../benchmarks/results/2026-10-24-gemma-per-layer-reference-v1.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(receipt)),
        "e5b4e055f519db2cd5fd0ac3d769a535a92f772fc4b9b9760868afbc375c222a"
    );
    let accepted: Value = serde_json::from_slice(receipt).unwrap();
    let raw = include_bytes!("../../../tests/fixtures/gemma-per-layer-v1/math.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        accepted["fixture_sha256"].as_str().unwrap()
    );
    let data: Value = serde_json::from_slice(raw).unwrap();
    let mut checked = 0;
    for case in data["cases"].as_array().unwrap() {
        let n = |key: &str| case[key].as_u64().unwrap() as usize;
        let tokens = n("batch") * n("tokens");
        if tokens == 0 {
            continue;
        }
        let s = PerLayerInputShape {
            tokens,
            hidden: n("hidden"),
            layers: n("layers"),
            width: n("width"),
            epsilon: case["epsilon"].as_f64().unwrap() as f32,
        };
        let features = s.layers * s.width;
        let values = |group: &str, key: &str| {
            case[group][key]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect::<Vec<_>>()
        };
        let input = |key: &str| {
            values("inputs", key)
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<_>>()
        };
        let main = input("raw_embedding");
        let table = input("table");
        let weights = input("projection_weight");
        let norm = input("norm_weight");
        let ids = case["inputs"]["ids"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let before = [bits(&main), bits(&table), bits(&weights), bits(&norm)];
        let mut scaled = vec![0.; main.len()];
        let mut projected = vec![0.; tokens * features];
        let mut lookup = projected.clone();
        let mut scratch = projected.clone();
        let mut output = projected.clone();
        prepare_per_layer_projection_input(&main, &mut scaled, tokens, s.hidden).unwrap();
        matmul_f64(
            &scaled,
            &weights,
            None,
            &mut projected,
            tokens,
            s.hidden,
            features,
        )
        .unwrap();
        embedding(&ids, &table, &mut lookup, 4, features).unwrap();
        finish_per_layer_inputs(
            &projected,
            case["with_lookup"].as_bool().unwrap().then_some(&lookup),
            &norm,
            &mut scratch,
            &mut output,
            &s,
        )
        .unwrap();
        for group in ["math", "staged_fp32"] {
            for (name, actual) in [
                ("scaled_embedding", &scaled),
                ("projection", &projected),
                ("lookup_raw", &lookup),
                ("projection_scaled", &scratch),
                ("output", &output),
            ] {
                close(actual, &values(group, name));
            }
        }
        assert_eq!(
            bits(&scaled),
            bits(
                &values("staged_fp32", "scaled_embedding")
                    .iter()
                    .map(|&x| x as f32)
                    .collect::<Vec<_>>()
            )
        );
        assert_eq!(
            lookup,
            values("staged_fp32", "lookup_raw")
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            [bits(&main), bits(&table), bits(&weights), bits(&norm)],
            before
        );
        checked += 1;
    }
    assert_eq!(checked, 6);
}

#[test]
fn hand_norm_axis_direct_scale_and_projection_only_output() {
    let s = shape();
    let projected = [2., 0., 0., 4.];
    let lookup = [1., 2., 3., 4.];
    let norm = [0.75, -0.5];
    let (mut scratch, mut output) = ([0.; 4], [0.; 4]);
    finish_per_layer_inputs(&projected, None, &norm, &mut scratch, &mut output, &s).unwrap();
    assert_eq!(scratch, [1., 0., 0., 2.]);
    let expected = [
        0.75_f64 / (0.5 + f64::from(s.epsilon)).sqrt(),
        0.,
        0.,
        -1_f64 / (2. + f64::from(s.epsilon)).sqrt(),
    ];
    close(&output, &expected);
    let plain = output;
    finish_per_layer_inputs(
        &projected,
        Some(&lookup),
        &norm,
        &mut scratch,
        &mut output,
        &s,
    )
    .unwrap();
    let expected = std::array::from_fn::<_, 4, _>(|i| {
        f64::from((plain[i] + lookup[i] * (2_f64.sqrt() as f32)) * (2_f64.powf(-0.5) as f32))
    });
    close(&output, &expected);
    assert!(plain[3] < 0.);
    assert!((f64::from(plain[0]) - 0.75 / (1.25 + f64::from(s.epsilon)).sqrt()).abs() > 0.3);
}

#[test]
fn all_geometry_lengths_epsilon_and_late_nonfinite_values_are_prewrite() {
    let projection = [1.; 4];
    let lookup = [2.; 4];
    let norm = [1.; 2];
    for mode in 0..16 {
        let mut s = shape();
        let mut p = projection;
        let mut l = lookup;
        let mut w = norm;
        match mode {
            0 => s.tokens = 0,
            1 => s.hidden = 0,
            2 => s.layers = 0,
            3 => s.width = 0,
            4 => s.layers = usize::MAX,
            5 => {
                s.tokens = usize::MAX;
                s.layers = 2;
            }
            6 => s.epsilon = 0.,
            7 => s.epsilon = f32::NAN,
            8 => s.epsilon = f32::INFINITY,
            9 => p[3] = f32::INFINITY,
            10 => l[3] = f32::NAN,
            11 => w[1] = f32::NEG_INFINITY,
            _ => {}
        }
        let (mut scratch, mut output) = ([77.; 4], [77.; 4]);
        let result = finish_per_layer_inputs(
            if mode == 12 { &p[..3] } else { &p },
            Some(if mode == 13 { &l[..3] } else { &l }),
            if mode == 14 { &w[..1] } else { &w },
            if mode == 15 {
                &mut scratch[..3]
            } else {
                &mut scratch
            },
            &mut output,
            &s,
        );
        assert!(result.is_err());
        assert_eq!(scratch, [77.; 4]);
        assert_eq!(output, [77.; 4]);
    }
    let (mut scratch, mut output) = ([77.; 4], [77.; 3]);
    assert!(
        finish_per_layer_inputs(
            &projection,
            None,
            &norm,
            &mut scratch,
            &mut output,
            &shape()
        )
        .is_err()
    );
    assert_eq!(scratch, [77.; 4]);
    assert_eq!(output, [77.; 3]);
    for mode in 0..5 {
        let input = [1., 1., 1., if mode == 0 { f32::NAN } else { 1. }];
        let mut scaled = [77.; 4];
        assert!(
            prepare_per_layer_projection_input(
                if mode == 1 { &input[..3] } else { &input },
                if mode == 2 {
                    &mut scaled[..3]
                } else {
                    &mut scaled
                },
                if mode == 3 { 0 } else { 1 },
                if mode == 4 { 0 } else { 4 }
            )
            .is_err()
        );
        assert_eq!(scaled, [77.; 4]);
    }
}

#[test]
fn intermediate_overflow_rejects_without_scale_cancellation_and_supports_replay() {
    let input = [0., f32::MAX];
    let before = bits(&input);
    let mut scaled = [77.; 2];
    assert!(prepare_per_layer_projection_input(&input, &mut scaled, 1, 2).is_err());
    assert_eq!(bits(&input), before);
    prepare_per_layer_projection_input(&[0., 1.], &mut scaled, 1, 2).unwrap();
    assert_eq!(scaled, [0., 2_f32.sqrt()]);
    let s = shape();
    let projection = [1.; 4];
    let lookup = [0., 0., 0., f32::MAX];
    let norm = [1.; 2];
    let before = [bits(&projection), bits(&lookup), bits(&norm)];
    let (mut scratch, mut output) = ([0.; 4], [0.; 4]);
    assert!(matches!(
        finish_per_layer_inputs(
            &projection,
            Some(&lookup),
            &norm,
            &mut scratch,
            &mut output,
            &s
        ),
        Err(Error::NonFinite {
            tensor: "per-layer scaled lookup",
            index: 3
        })
    ));
    assert_eq!([bits(&projection), bits(&lookup), bits(&norm)], before);
    finish_per_layer_inputs(
        &projection,
        Some(&[0.; 4]),
        &norm,
        &mut scratch,
        &mut output,
        &s,
    )
    .unwrap();
    assert!(output.iter().all(|x| x.is_finite()));
    let norm = [f32::MAX; 2];
    let before = bits(&norm);
    assert!(
        finish_per_layer_inputs(
            &[2., 0., 0., 4.],
            None,
            &norm,
            &mut scratch,
            &mut output,
            &s
        )
        .is_err()
    );
    assert_eq!(bits(&norm), before);
    let s = PerLayerInputShape {
        tokens: 1,
        hidden: 1,
        layers: 1,
        width: 1,
        epsilon: 1e-6,
    };
    let (mut scratch, mut output) = ([0.], [0.]);
    assert!(matches!(
        finish_per_layer_inputs(
            &[1.],
            Some(&[f32::MAX]),
            &[f32::MAX],
            &mut scratch,
            &mut output,
            &s
        ),
        Err(Error::NonFinite {
            tensor: "per-layer combined input",
            index: 0
        })
    ));
}

#[test]
fn independent_rows_chunks_and_reused_buffers_preserve_exact_results() {
    let s = PerLayerInputShape {
        tokens: 4,
        hidden: 5,
        layers: 2,
        width: 3,
        epsilon: 1e-6,
    };
    let raw = (0..20).map(|i| (i as f32 - 7.) * 0.125).collect::<Vec<_>>();
    let projected = (0..24)
        .map(|i| (i as f32 - 11.) * 0.125)
        .collect::<Vec<_>>();
    let lookup = (0..24)
        .map(|i| (i as f32 - 13.) * 0.0625)
        .collect::<Vec<_>>();
    let norm = [0.75, -0.5, 1.25];
    let before = [bits(&raw), bits(&projected), bits(&lookup), bits(&norm)];
    let mut scaled = vec![0.; 20];
    let mut scratch = vec![0.; 24];
    let mut output = vec![0.; 24];
    let pointers = (scaled.as_ptr(), scratch.as_ptr(), output.as_ptr());
    prepare_per_layer_projection_input(&raw, &mut scaled, 4, 5).unwrap();
    finish_per_layer_inputs(
        &projected,
        Some(&lookup),
        &norm,
        &mut scratch,
        &mut output,
        &s,
    )
    .unwrap();
    let expected = [bits(&scaled), bits(&scratch), bits(&output)];
    for size in [1, 2] {
        for start in (0..4).step_by(size) {
            let shape = PerLayerInputShape { tokens: size, ..s };
            let range = start * 6..(start + size) * 6;
            prepare_per_layer_projection_input(
                &raw[start * 5..(start + size) * 5],
                &mut scaled[start * 5..(start + size) * 5],
                size,
                5,
            )
            .unwrap();
            finish_per_layer_inputs(
                &projected[range.clone()],
                Some(&lookup[range.clone()]),
                &norm,
                &mut scratch[range.clone()],
                &mut output[range],
                &shape,
            )
            .unwrap();
        }
        assert_eq!([bits(&scaled), bits(&scratch), bits(&output)], expected);
    }
    assert_eq!(
        (scaled.as_ptr(), scratch.as_ptr(), output.as_ptr()),
        pointers
    );
    assert_eq!(
        [bits(&raw), bits(&projected), bits(&lookup), bits(&norm)],
        before
    );
}

#[test]
fn empty_and_overflow_admission_is_explicit_with_signed_zero_copying() {
    let mut out = [];
    assert!(prepare_per_layer_projection_input(&[], &mut out, 0, 4).is_err());
    assert!(prepare_per_layer_projection_input(&[], &mut out, usize::MAX, 2).is_err());
    let mut scratch = [];
    let s = PerLayerInputShape {
        tokens: 0,
        ..shape()
    };
    assert!(finish_per_layer_inputs(&[], Some(&[]), &[1.; 2], &mut scratch, &mut out, &s).is_err());
    let raw = [-0., 0.];
    let mut scaled = [1.; 2];
    prepare_per_layer_projection_input(&raw, &mut scaled, 1, 2).unwrap();
    assert_eq!(bits(&raw), bits(&scaled));
}
