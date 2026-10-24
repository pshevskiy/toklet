use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

const BINARY: &[u8] = include_bytes!("../../../tests/fixtures/gated-delta-v1/arrays.bin");
const MANIFEST: &str = include_str!("../../../tests/fixtures/gated-delta-v1/manifest.json");

fn scalar() -> GatedDeltaShape {
    GatedDeltaShape {
        key_heads: 1,
        value_heads: 1,
        key_dim: 1,
        value_dim: 1,
    }
}

fn values(row: &Value, name: &str) -> Vec<f64> {
    let d = &row["arrays"][name];
    let offset = d["offset"].as_u64().unwrap() as usize;
    let bytes = d["bytes"].as_u64().unwrap() as usize;
    let raw = &BINARY[offset..offset + bytes];
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        d["sha256"].as_str().unwrap()
    );
    match d["dtype"].as_str().unwrap() {
        "<f4" => raw
            .chunks_exact(4)
            .map(|x| f64::from(f32::from_le_bytes(x.try_into().unwrap())))
            .collect(),
        "<f8" => raw
            .chunks_exact(8)
            .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
            .collect(),
        _ => panic!("unexpected frozen dtype"),
    }
}

fn close(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && (f64::from(*a) - e).abs() <= 1e-5 + 1e-4 * e.abs(),
            "{a} differs from {e}"
        );
    }
}

fn fixture() -> Value {
    let receipt_raw =
        include_str!("../../../benchmarks/results/2026-10-24-gated-delta-oracle-v1.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(receipt_raw.as_bytes())),
        "69299394de716f79d02c6465c51a1104b3a5078e3f5d78ce352bd7135c995ef7"
    );
    let receipt: Value = serde_json::from_str(receipt_raw).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(MANIFEST.as_bytes())),
        receipt["fixture_manifest_sha256"].as_str().unwrap()
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(BINARY)),
        receipt["fixture_binary_sha256"].as_str().unwrap()
    );
    serde_json::from_str(MANIFEST).unwrap()
}

#[test]
fn frozen_normalized_recurrence_matches_each_state_and_output() {
    let manifest = fixture();
    for row in manifest["cases"].as_array().unwrap() {
        let g = &row["geometry"];
        let n = |key: &str| g[key].as_u64().unwrap() as usize;
        let shape = GatedDeltaShape {
            key_heads: n("key_heads"),
            value_heads: n("value_heads"),
            key_dim: n("key_dim"),
            value_dim: n("value_dim"),
        };
        let tokens = n("tokens");
        let qs = shape.key_heads * shape.key_dim;
        let vs = shape.value_heads * shape.value_dim;
        let input = |name| {
            values(row, name)
                .into_iter()
                .map(|x| x as f32)
                .collect::<Vec<_>>()
        };
        let q = input("normalized_q");
        let k = input("normalized_k");
        let v = input("v");
        let decay = input("decay");
        let beta = input("beta");
        let initial = input("initial_state");
        let mut state = initial.clone();
        let mut candidate = vec![f32::NAN; state.len()];
        let mut output = vec![f32::NAN; vs];
        let expected_state = values(row, "states");
        let expected_output = values(row, "output");
        for t in 0..tokens {
            let before = state.clone();
            gated_delta_step(
                &q[t * qs..(t + 1) * qs],
                &k[t * qs..(t + 1) * qs],
                &v[t * vs..(t + 1) * vs],
                &decay[t * shape.value_heads..(t + 1) * shape.value_heads],
                &beta[t * shape.value_heads..(t + 1) * shape.value_heads],
                &state,
                &mut candidate,
                &mut output,
                &shape,
            )
            .unwrap();
            assert_eq!(state, before);
            close(
                &candidate,
                &expected_state[t * state.len()..(t + 1) * state.len()],
            );
            close(&output, &expected_output[t * vs..(t + 1) * vs]);
            std::mem::swap(&mut state, &mut candidate);
        }
        close(&state, &values(row, "final_state"));
        assert_eq!(initial, input("initial_state"));
    }
}

#[test]
fn hand_computed_decay_correction_and_group_mapping() {
    let mut candidate = [0.];
    let mut out = [0.];
    gated_delta_step(
        &[1.],
        &[1.],
        &[3.],
        &[0.5],
        &[0.5],
        &[4.],
        &mut candidate,
        &mut out,
        &scalar(),
    )
    .unwrap();
    assert_eq!(candidate, [2.5]);
    assert_eq!(out, [2.5]);
    let shape = GatedDeltaShape {
        key_heads: 2,
        value_heads: 4,
        key_dim: 1,
        value_dim: 1,
    };
    let mut candidate = [0.; 4];
    let mut out = [0.; 4];
    gated_delta_step(
        &[1., 2.],
        &[1., 1.],
        &[1., 2., 3., 4.],
        &[1.; 4],
        &[1.; 4],
        &[0.; 4],
        &mut candidate,
        &mut out,
        &shape,
    )
    .unwrap();
    assert_eq!(out, [1., 2., 6., 8.]);
}

#[test]
fn invalid_admission_preserves_both_outputs_and_input_state() {
    let state = [2.];
    for (q, gate, shape) in [
        (f32::NAN, 1., scalar()),
        (1., -0.01, scalar()),
        (1., 1.01, scalar()),
        (
            1.,
            1.,
            GatedDeltaShape {
                key_dim: 0,
                ..scalar()
            },
        ),
        (
            1.,
            1.,
            GatedDeltaShape {
                key_heads: 2,
                value_heads: 3,
                ..scalar()
            },
        ),
        (
            1.,
            1.,
            GatedDeltaShape {
                key_dim: usize::MAX,
                value_dim: 2,
                ..scalar()
            },
        ),
    ] {
        let mut candidate = [91.];
        let mut out = [92.];
        assert!(
            gated_delta_step(
                &[q],
                &[1.],
                &[1.],
                &[gate],
                &[1.],
                &state,
                &mut candidate,
                &mut out,
                &shape
            )
            .is_err()
        );
        assert_eq!(candidate, [91.]);
        assert_eq!(out, [92.]);
        assert_eq!(state, [2.]);
    }
    let mut candidate = [91.];
    let mut out = [92.];
    assert!(
        gated_delta_step(
            &[],
            &[1.],
            &[1.],
            &[1.],
            &[1.],
            &state,
            &mut candidate,
            &mut out,
            &scalar()
        )
        .is_err()
    );
    assert_eq!(candidate, [91.]);
    assert_eq!(out, [92.]);
}

#[test]
fn arithmetic_failure_keeps_old_state_available_for_replay() {
    let state = [f32::MAX];
    let mut candidate = [0.];
    let mut out = [0.];
    assert!(matches!(
        gated_delta_step(
            &[1.],
            &[2.],
            &[0.],
            &[1.],
            &[1.],
            &state,
            &mut candidate,
            &mut out,
            &scalar()
        ),
        Err(Error::NonFinite { .. })
    ));
    assert_eq!(state, [f32::MAX]);
    gated_delta_step(
        &[1.],
        &[1.],
        &[0.],
        &[0.],
        &[1.],
        &state,
        &mut candidate,
        &mut out,
        &scalar(),
    )
    .unwrap();
    assert_eq!(candidate, [0.]);
    assert_eq!(out, [0.]);
}

#[test]
fn every_nonfinite_input_and_invalid_beta_reject_before_writes() {
    let mut cases = Vec::new();
    for index in 0..6 {
        for invalid in [f32::NAN, f32::INFINITY] {
            let mut inputs = [1_f32; 6];
            inputs[index] = invalid;
            cases.push(inputs);
        }
    }
    for invalid in [-0.01, 1.01] {
        let mut inputs = [1_f32; 6];
        inputs[4] = invalid;
        cases.push(inputs);
    }
    for [q, k, v, decay, beta, state] in cases {
        let mut candidate = [91.];
        let mut out = [92.];
        assert!(
            gated_delta_step(
                &[q],
                &[k],
                &[v],
                &[decay],
                &[beta],
                &[state],
                &mut candidate,
                &mut out,
                &scalar()
            )
            .is_err()
        );
        assert_eq!(candidate, [91.]);
        assert_eq!(out, [92.]);
    }
}

#[test]
fn correction_candidate_and_readout_overflow_preserve_old_state() {
    for (q, k, v, previous, tensor) in [
        (1., 1., f32::MAX, -f32::MAX, "delta correction"),
        (1., 2., f32::MAX, 0., "delta candidate"),
        (2., 1., f32::MAX, 0., "delta output"),
    ] {
        let state = [previous];
        let mut candidate = [0.];
        let mut out = [0.];
        assert!(
            matches!(gated_delta_step(&[q], &[k], &[v], &[1.], &[1.], &state,
            &mut candidate, &mut out, &scalar()), Err(Error::NonFinite {tensor: actual, ..}) if actual == tensor)
        );
        assert_eq!(state, [previous]);
    }
}

#[test]
fn raw_normalization_matches_frozen_math_and_composed_recurrence() {
    let manifest = fixture();
    for row in manifest["cases"].as_array().unwrap() {
        let g = &row["geometry"];
        let n = |key: &str| g[key].as_u64().unwrap() as usize;
        let shape = GatedDeltaShape {
            key_heads: n("key_heads"),
            value_heads: n("value_heads"),
            key_dim: n("key_dim"),
            value_dim: n("value_dim"),
        };
        let input = |name| {
            values(row, name)
                .into_iter()
                .map(|x| x as f32)
                .collect::<Vec<_>>()
        };
        let q = input("raw_q");
        let k = input("raw_k");
        let v = input("v");
        let decay = input("decay");
        let beta = input("beta");
        let expected_q = values(row, "normalized_q");
        let expected_k = values(row, "normalized_k");
        let expected_state = values(row, "states");
        let expected_output = values(row, "output");
        let qs = shape.key_heads * shape.key_dim;
        let vs = shape.value_heads * shape.value_dim;
        let mut normalized_q = vec![f32::NAN; qs];
        let mut normalized_k = vec![f32::NAN; qs];
        let mut state = input("initial_state");
        let mut candidate = vec![f32::NAN; state.len()];
        let mut output = vec![f32::NAN; vs];
        for token in 0..n("tokens") {
            gated_delta_normalize(
                &q[token * qs..(token + 1) * qs],
                &k[token * qs..(token + 1) * qs],
                &mut normalized_q,
                &mut normalized_k,
                shape.key_heads,
                shape.key_dim,
                g["epsilon"].as_f64().unwrap(),
            )
            .unwrap();
            close(&normalized_q, &expected_q[token * qs..(token + 1) * qs]);
            close(&normalized_k, &expected_k[token * qs..(token + 1) * qs]);
            gated_delta_step(
                &normalized_q,
                &normalized_k,
                &v[token * vs..(token + 1) * vs],
                &decay[token * shape.value_heads..(token + 1) * shape.value_heads],
                &beta[token * shape.value_heads..(token + 1) * shape.value_heads],
                &state,
                &mut candidate,
                &mut output,
                &shape,
            )
            .unwrap();
            close(
                &candidate,
                &expected_state[token * state.len()..(token + 1) * state.len()],
            );
            close(&output, &expected_output[token * vs..(token + 1) * vs]);
            std::mem::swap(&mut state, &mut candidate);
        }
        close(&state, &values(row, "final_state"));
    }
}

#[test]
fn l2_sum_epsilon_and_single_query_scale_are_not_rmsnorm() {
    let mut q = [0.; 2];
    let mut k = [0.; 2];
    gated_delta_normalize(&[3., 4.], &[3., 4.], &mut q, &mut k, 1, 2, 1.).unwrap();
    close(
        &q,
        &[
            3. / 26_f64.sqrt() / 2_f64.sqrt(),
            4. / 26_f64.sqrt() / 2_f64.sqrt(),
        ],
    );
    close(&k, &[3. / 26_f64.sqrt(), 4. / 26_f64.sqrt()]);
    gated_delta_normalize(&[0., 0.], &[1e-4, 1e-4], &mut q, &mut k, 1, 2, 1e-6).unwrap();
    assert_eq!(q, [0., 0.]);
    // Epsilon dominates the sum; a zero result must still fail the policy.
    close(&k, &[1e-4 / (2e-8 + 1e-6_f64).sqrt(); 2]);
    gated_delta_normalize(
        &[f32::MAX, -f32::MAX],
        &[f32::MAX; 2],
        &mut q,
        &mut k,
        1,
        2,
        1e-6,
    )
    .unwrap();
    close(&q, &[0.5, -0.5]);
    close(&k, &[2_f64.sqrt().recip(); 2]);
}

#[test]
fn invalid_normalization_never_writes_outputs() {
    for epsilon in [0., -1., f64::NAN, f64::INFINITY] {
        let mut q = [91.];
        let mut k = [92.];
        assert!(gated_delta_normalize(&[1.], &[1.], &mut q, &mut k, 1, 1, epsilon).is_err());
        assert_eq!(q, [91.]);
        assert_eq!(k, [92.]);
    }
    for (input, heads, dim) in [
        (f32::NAN, 1, 1),
        (1., 0, 1),
        (1., 2, usize::MAX),
        (1., 1, 2),
    ] {
        let mut q = [91.];
        let mut k = [92.];
        assert!(gated_delta_normalize(&[input], &[1.], &mut q, &mut k, heads, dim, 1e-6).is_err());
        assert_eq!(q, [91.]);
        assert_eq!(k, [92.]);
    }
    let mut q = [91.];
    let mut k = [92.];
    assert!(gated_delta_normalize(&[1.], &[f32::NAN], &mut q, &mut k, 1, 1, 1e-6).is_err());
    assert_eq!(q, [91.]);
    assert_eq!(k, [92.]);
    let mut q = [91.; 2];
    assert!(gated_delta_normalize(&[1.], &[1.], &mut q, &mut k, 1, 1, 1e-6).is_err());
    assert_eq!(q, [91.; 2]);
    assert_eq!(k, [92.]);
}

#[test]
fn raw_gates_and_full_scalar_pipeline_match_frozen_witnesses() {
    let manifest = fixture();
    for row in manifest["cases"].as_array().unwrap() {
        let g = &row["geometry"];
        let n = |key: &str| g[key].as_u64().unwrap() as usize;
        let shape = GatedDeltaShape {
            key_heads: n("key_heads"),
            value_heads: n("value_heads"),
            key_dim: n("key_dim"),
            value_dim: n("value_dim"),
        };
        let input = |name| {
            values(row, name)
                .into_iter()
                .map(|x| x as f32)
                .collect::<Vec<_>>()
        };
        let q = input("raw_q");
        let k = input("raw_k");
        let v = input("v");
        let a = input("a");
        let b = input("b");
        let a_log = input("a_log");
        let bias = input("dt_bias");
        let expected_decay = values(row, "decay");
        let expected_beta = values(row, "beta");
        let expected_state = values(row, "states");
        let expected_output = values(row, "output");
        let qs = shape.key_heads * shape.key_dim;
        let vs = shape.value_heads * shape.value_dim;
        let hv = shape.value_heads;
        let mut normalized_q = vec![f32::NAN; qs];
        let mut normalized_k = vec![f32::NAN; qs];
        let mut decay = vec![f32::NAN; hv];
        let mut beta = vec![f32::NAN; hv];
        let mut state = input("initial_state");
        let mut candidate = vec![f32::NAN; state.len()];
        let mut output = vec![f32::NAN; vs];
        for t in 0..n("tokens") {
            gated_delta_gates(
                &a[t * hv..(t + 1) * hv],
                &b[t * hv..(t + 1) * hv],
                &a_log,
                &bias,
                &mut decay,
                &mut beta,
            )
            .unwrap();
            close(&decay, &expected_decay[t * hv..(t + 1) * hv]);
            close(&beta, &expected_beta[t * hv..(t + 1) * hv]);
            gated_delta_normalize(
                &q[t * qs..(t + 1) * qs],
                &k[t * qs..(t + 1) * qs],
                &mut normalized_q,
                &mut normalized_k,
                shape.key_heads,
                shape.key_dim,
                g["epsilon"].as_f64().unwrap(),
            )
            .unwrap();
            gated_delta_step(
                &normalized_q,
                &normalized_k,
                &v[t * vs..(t + 1) * vs],
                &decay,
                &beta,
                &state,
                &mut candidate,
                &mut output,
                &shape,
            )
            .unwrap();
            close(
                &candidate,
                &expected_state[t * state.len()..(t + 1) * state.len()],
            );
            close(&output, &expected_output[t * vs..(t + 1) * vs]);
            std::mem::swap(&mut state, &mut candidate);
        }
        close(&state, &values(row, "final_state"));
    }
}

#[test]
fn gate_hand_values_and_saturated_inputs_are_bounded() {
    let mut decay = [0.];
    let mut beta = [0.];
    gated_delta_gates(&[0.], &[0.], &[0.], &[0.], &mut decay, &mut beta).unwrap();
    assert_eq!(decay, [0.5]);
    assert_eq!(beta, [0.5]);
    gated_delta_gates(&[-0.5], &[0.], &[2_f32.ln()], &[0.5], &mut decay, &mut beta).unwrap();
    close(&decay, &[0.25]);
    assert_eq!(beta, [0.5]);
    for (a, b, g, u) in [(1000., f32::MAX, 0., 1.), (-1000., -f32::MAX, 1., 0.)] {
        gated_delta_gates(&[a], &[b], &[0.], &[0.], &mut decay, &mut beta).unwrap();
        assert_eq!(decay, [g]);
        assert_eq!(beta, [u]);
    }
    gated_delta_gates(&[-24.], &[-24.], &[0.], &[0.], &mut decay, &mut beta).unwrap();
    let expected = (-24_f64).exp() / (1.0 + (-24_f64).exp());
    assert!(beta[0] > 0.0 && (f64::from(beta[0]) - expected).abs() <= expected * 1e-6);
}

#[test]
fn gate_admission_preserves_outputs_and_intermediate_overflow_rejects() {
    for slot in 0..4 {
        for invalid in [f32::NAN, f32::INFINITY] {
            let mut values = [0.; 4];
            values[slot] = invalid;
            let mut decay = [91.];
            let mut beta = [92.];
            assert!(
                gated_delta_gates(
                    &values[..1],
                    &values[1..2],
                    &values[2..3],
                    &values[3..],
                    &mut decay,
                    &mut beta
                )
                .is_err()
            );
            assert_eq!(decay, [91.]);
            assert_eq!(beta, [92.]);
        }
    }
    let mut decay = [91.];
    let mut beta = [92.];
    assert!(gated_delta_gates(&[0.], &[], &[0.], &[0.], &mut decay, &mut beta).is_err());
    assert_eq!(decay, [91.]);
    assert_eq!(beta, [92.]);
    for (a, log) in [(0., 1000.), (f32::MAX, 700.)] {
        assert!(matches!(
            gated_delta_gates(&[a], &[0.], &[log], &[0.], &mut decay, &mut beta),
            Err(Error::NonFinite { .. })
        ));
    }
}
