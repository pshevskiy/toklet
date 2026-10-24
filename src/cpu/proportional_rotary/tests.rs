use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn fixture() -> Value {
    let receipt =
        include_bytes!("../../../benchmarks/results/2026-10-24-gemma-rotary-math-v1.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(receipt)),
        "a16a944f4ba20956f8c8a19796afb78727bb15aa38b78b0d29a4b25ee8beb63d"
    );
    let accepted: Value = serde_json::from_slice(receipt).unwrap();
    let raw = include_bytes!("../../../tests/fixtures/gemma-rotary-v1/math.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        accepted["fixture_sha256"].as_str().unwrap()
    );
    serde_json::from_slice(raw).unwrap()
}

fn shape() -> ProportionalRotaryShape {
    ProportionalRotaryShape {
        tokens: 1,
        heads: 1,
        head_dim: 16,
        rotated_dims: 4,
        offset: 2,
        base: 256.,
        factor: 1.,
    }
}

#[test]
fn frozen_mathematical_rotary_matches_and_inactive_copy_is_exact() {
    for row in fixture()["cases"].as_array().unwrap() {
        let n = |key: &str| row[key].as_u64().unwrap() as usize;
        let geometry = ProportionalRotaryShape {
            tokens: n("tokens"),
            heads: n("heads"),
            head_dim: n("dims"),
            rotated_dims: n("rotated_dims"),
            offset: n("offset"),
            base: row["base"].as_f64().unwrap() as f32,
            factor: row["factor"].as_f64().unwrap() as f32,
        };
        let values = |name: &str| {
            row["arrays"][name]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect::<Vec<_>>()
        };
        let x: Vec<f32> = values("input").iter().map(|&v| v as f32).collect();
        let expected = values("output");
        if geometry.tokens == 0 {
            continue; // Empty sequences make no operator calls.
        }
        let mut output = vec![f32::NAN; x.len()];
        proportional_rotary(&x, &mut output, &geometry).unwrap();
        for (&a, &e) in output.iter().zip(&expected) {
            assert!(a.is_finite() && (f64::from(a) - e).abs() <= 1e-5 + 1e-4 * e.abs());
        }
        for first in (0..x.len()).step_by(geometry.head_dim) {
            for pair in geometry.rotated_dims / 2..geometry.head_dim / 2 {
                for i in [first + pair, first + pair + geometry.head_dim / 2] {
                    assert_eq!(output[i].to_bits(), x[i].to_bits());
                }
            }
        }
    }
}

#[test]
fn hand_half_pairs_and_signed_zero_copy() {
    let mut x = [0.; 16];
    x[0] = 2.;
    x[8] = 3.;
    x[1] = 4.;
    x[9] = 5.;
    x[2] = -0.;
    x[10] = -0.;
    let mut output = [f32::NAN; 16];
    let mut geometry = shape();
    proportional_rotary(&x, &mut output, &geometry).unwrap();
    for (i, e) in [
        (0, -3.5601859535713305),
        (8, 0.5701543378319363),
        (1, -2.0461456984751235),
        (9, 6.067395468572284),
    ] {
        assert!((f64::from(output[i]) - e).abs() < 1e-6);
    }
    for i in [2, 10] {
        assert_eq!(output[i].to_bits(), x[i].to_bits());
    }
    geometry.rotated_dims = 0;
    proportional_rotary(&x, &mut output, &geometry).unwrap();
    assert_eq!(output.map(f32::to_bits), x.map(f32::to_bits));
}

#[test]
fn full_rotation_matches_existing_reference_and_chunks_match() {
    let x: Vec<f32> = (0..3 * 2 * 16)
        .map(|i| ((i * 7 % 31) as f32 - 15.) * 0.125)
        .collect();
    let mut geometry = ProportionalRotaryShape {
        tokens: 3,
        heads: 2,
        head_dim: 16,
        rotated_dims: 16,
        offset: 131069,
        base: 1e6,
        factor: 1.,
    };
    let mut full = vec![0.; x.len()];
    let mut old = vec![0.; x.len()];
    proportional_rotary(&x, &mut full, &geometry).unwrap();
    super::super::rotary(&x, &mut old, 3, 2, 16, 131069, 1e6).unwrap();
    assert_eq!(full, old);
    geometry.rotated_dims = 4;
    geometry.factor = 2.5;
    proportional_rotary(&x, &mut full, &geometry).unwrap();
    for chunk in [1, 2] {
        let mut joined = vec![0.; x.len()];
        for first in (0..3).step_by(chunk) {
            let count = chunk.min(3 - first);
            let part = ProportionalRotaryShape {
                tokens: count,
                offset: 131069 + first,
                ..geometry
            };
            proportional_rotary(
                &x[first * 32..(first + count) * 32],
                &mut joined[first * 32..(first + count) * 32],
                &part,
            )
            .unwrap();
        }
        assert_eq!(joined, full);
    }
}

#[test]
fn invalid_geometry_lengths_and_nonfinite_input_preserve_output() {
    let mut invalid = Vec::new();
    for value in [0, 3] {
        invalid.push(ProportionalRotaryShape {
            head_dim: value,
            ..shape()
        });
    }
    for value in [3, 18] {
        invalid.push(ProportionalRotaryShape {
            rotated_dims: value,
            ..shape()
        });
    }
    invalid.extend([
        ProportionalRotaryShape {
            tokens: 0,
            ..shape()
        },
        ProportionalRotaryShape {
            heads: 0,
            ..shape()
        },
        ProportionalRotaryShape {
            heads: usize::MAX,
            ..shape()
        },
        ProportionalRotaryShape {
            tokens: 2,
            heads: usize::MAX / 16,
            ..shape()
        },
        ProportionalRotaryShape {
            tokens: 2,
            offset: usize::MAX,
            ..shape()
        },
    ]);
    if let Ok(offset) = usize::try_from((1_u64 << 53) + 1) {
        invalid.push(ProportionalRotaryShape { offset, ..shape() });
    }
    for value in [0., -1., f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        invalid.push(ProportionalRotaryShape {
            base: value,
            ..shape()
        });
        invalid.push(ProportionalRotaryShape {
            factor: value,
            ..shape()
        });
    }
    let x = [1.; 16];
    let mut output = [7.; 17];
    for geometry in invalid {
        assert!(proportional_rotary(&x, &mut output[..16], &geometry).is_err());
        assert_eq!(output, [7.; 17]);
    }
    for count in [15, 17] {
        assert!(proportional_rotary(&[1.; 17][..count], &mut output[..16], &shape()).is_err());
        assert!(proportional_rotary(&x, &mut output[..count], &shape()).is_err());
        assert_eq!(output, [7.; 17]);
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bad = x;
        bad[15] = value; // Admission includes inactive coordinates.
        assert!(proportional_rotary(&bad, &mut output[..16], &shape()).is_err());
        assert_eq!(output, [7.; 17]);
    }
}

#[test]
fn arithmetic_overflow_preserves_input_and_allows_replay() {
    for sign in [1., -1.] {
        let mut x = [0.; 16];
        x[0] = 1.;
        x[8] = 2.;
        x[1] = f32::MAX;
        x[9] = sign * f32::MAX;
        let before = x;
        let mut output = [7.; 16];
        let geometry = ProportionalRotaryShape {
            offset: 1,
            ..shape()
        };
        assert_eq!(
            proportional_rotary(&x, &mut output, &geometry),
            Err(Error::NonFinite {
                tensor: "proportional rotary output",
                index: if sign > 0. { 9 } else { 1 },
            })
        );
        assert_eq!(x, before);
        assert_ne!(output[0], x[0]);
        let mut valid = x;
        valid[1] = 1.;
        valid[9] = 2.;
        proportional_rotary(&valid, &mut output, &geometry).unwrap();
        assert!(output.iter().all(|x| x.is_finite()));
    }
}

#[test]
fn extreme_finite_parameters_and_exact_position_boundary_remain_checked() {
    let offset = usize::try_from(1_u64 << 53).unwrap_or(0);
    let tiny = f32::from_bits(1);
    for (base, factor) in [
        (tiny, tiny),
        (tiny, f32::MAX),
        (f32::MAX, tiny),
        (f32::MAX, f32::MAX),
    ] {
        let mut output = [0.; 16];
        let geometry = ProportionalRotaryShape {
            rotated_dims: 16,
            offset,
            base,
            factor,
            ..shape()
        };
        proportional_rotary(&[0.25; 16], &mut output, &geometry).unwrap();
        assert!(output.iter().all(|x| x.is_finite()));
        if offset > 0 {
            let before = [7.; 32];
            let mut rejected = before;
            assert!(
                proportional_rotary(
                    &[0.25; 32],
                    &mut rejected,
                    &ProportionalRotaryShape {
                        tokens: 2,
                        ..geometry
                    },
                )
                .is_err()
            );
            assert_eq!(rejected, before);
        }
    }
}
