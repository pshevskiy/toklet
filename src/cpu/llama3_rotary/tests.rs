use super::*;
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
    controls: Vec<Control>,
    producer_sha256: String,
}
#[derive(Deserialize)]
struct Control {
    ordinary: f64,
    scaled: f64,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    tokens: usize,
    heads: usize,
    head_dim: usize,
    offset: usize,
    theta: f32,
    factor: f32,
    low: f32,
    high: f32,
    original: usize,
    input: Vec<f32>,
    inverse_frequencies: Vec<f64>,
    output: Vec<f64>,
}
fn fixture() -> Fixture {
    let raw = include_bytes!("../../../tests/fixtures/llama3-rotary-v1/math.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "980c8835bc5485c7d064b77e868dab44891a55c2bbff8365ea3b30c96d4c1692"
    );
    let fixture: Fixture = serde_json::from_slice(raw).unwrap();
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../../../tools/reference/llama3_rotary.py"))
        ),
        fixture.producer_sha256
    );
    fixture
}
fn scaling() -> Llama3Scaling {
    Llama3Scaling::new(32., 1., 4., 8192).unwrap()
}
fn shape() -> Llama3RotaryShape {
    Llama3RotaryShape {
        tokens: 1,
        heads: 1,
        head_dim: 16,
        offset: 1,
        theta: 500000.,
    }
}

#[test]
fn independent_decimal_frequencies_and_rotations_match_frozen_gates() {
    for case in fixture().cases {
        let scaling = Llama3Scaling::new(case.factor, case.low, case.high, case.original).unwrap();
        assert_eq!(case.inverse_frequencies.len(), case.head_dim / 2);
        for (pair, &expected) in case.inverse_frequencies.iter().enumerate() {
            let ordinary = f64::from(case.theta).powf(-2.0 * pair as f64 / case.head_dim as f64);
            let actual = scaling.inverse_frequency(ordinary).unwrap();
            assert!(
                (actual - expected).abs() <= 1e-15 + 1e-13 * expected.abs(),
                "{} frequency {pair}: {actual} vs {expected}",
                case.name
            );
        }
        let geometry = Llama3RotaryShape {
            tokens: case.tokens,
            heads: case.heads,
            head_dim: case.head_dim,
            offset: case.offset,
            theta: case.theta,
        };
        let mut output = vec![f32::NAN; case.input.len()];
        llama3_rotary(&case.input, &mut output, &geometry, &scaling).unwrap();
        assert_eq!(output.len(), case.output.len());
        for (i, (&actual, &expected)) in output.iter().zip(&case.output).enumerate() {
            assert!(
                actual.is_finite()
                    && (f64::from(actual) - expected).abs() <= 1e-5 + 1e-4 * expected.abs(),
                "{} output {i}: {actual} vs {expected}",
                case.name
            );
        }
    }
}

#[test]
fn transition_edges_and_candidate_bands_are_explicit() {
    let s = scaling();
    for control in fixture().controls {
        let actual = s.inverse_frequency(control.ordinary).unwrap();
        assert!((actual - control.scaled).abs() <= 1e-17 + 1e-13 * control.scaled.abs());
    }
    let high = std::f64::consts::TAU * 4.0 / 8192.0;
    let low = std::f64::consts::TAU / 8192.0;
    assert_eq!(s.inverse_frequency(high).unwrap(), high);
    assert_eq!(s.inverse_frequency(low).unwrap(), low / 32.0);
    for boundary in [low, high] {
        for inverse in [boundary.next_down(), boundary.next_up()] {
            let actual = s.inverse_frequency(inverse).unwrap();
            let expected_edge = s.inverse_frequency(boundary).unwrap();
            assert!((actual - expected_edge).abs() < 1e-17);
        }
    }
    let mut bands = [0; 3];
    for pair in 0..32 {
        let ordinary = 500000_f64.powf(-2.0 * f64::from(pair) / 64.0);
        let wavelength = std::f64::consts::TAU / ordinary;
        assert_ne!(wavelength, 2048.0);
        assert_ne!(wavelength, 8192.0);
        let band = if wavelength < 2048.0 {
            0
        } else if wavelength > 8192.0 {
            2
        } else {
            1
        };
        bands[band] += 1;
    }
    assert_eq!(bands, [15, 3, 14]);
}

#[test]
fn identity_and_high_frequency_paths_match_ordinary_bits() {
    let input: Vec<f32> = (0..96)
        .map(|i| ((i * 7 % 31) as f32 - 15.) * 0.125)
        .collect();
    for (theta, s) in [
        (500000., Llama3Scaling::new(1., 1., 4., 8192).unwrap()),
        (2., scaling()),
    ] {
        let geometry = Llama3RotaryShape {
            tokens: 3,
            heads: 2,
            offset: 131069,
            theta,
            ..shape()
        };
        let mut ordinary = vec![0.; input.len()];
        let mut actual = vec![0.; input.len()];
        super::super::rotary(&input, &mut ordinary, 3, 2, 16, 131069, theta).unwrap();
        llama3_rotary(&input, &mut actual, &geometry, &s).unwrap();
        assert!(
            actual
                .iter()
                .zip(&ordinary)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
}

#[test]
fn chunked_absolute_positions_match_and_scaling_applies_to_short_inputs() {
    let input: Vec<f32> = (0..384)
        .map(|i| ((i * 7 % 31) as f32 - 15.) * 0.125)
        .collect();
    let mut geometry = Llama3RotaryShape {
        tokens: 3,
        heads: 2,
        head_dim: 64,
        offset: 131069,
        ..shape()
    };
    let mut whole = vec![0.; input.len()];
    llama3_rotary(&input, &mut whole, &geometry, &scaling()).unwrap();
    for chunk in [1, 2] {
        let mut joined = vec![0.; input.len()];
        for first in (0..3).step_by(chunk) {
            let count = chunk.min(3 - first);
            let part = Llama3RotaryShape {
                tokens: count,
                offset: geometry.offset + first,
                ..geometry
            };
            llama3_rotary(
                &input[first * 128..(first + count) * 128],
                &mut joined[first * 128..(first + count) * 128],
                &part,
                &scaling(),
            )
            .unwrap();
        }
        assert_eq!(joined, whole);
    }
    geometry.offset = 1;
    let mut ordinary = vec![0.; input.len()];
    super::super::rotary(&input, &mut ordinary, 3, 2, 64, 1, geometry.theta).unwrap();
    llama3_rotary(&input, &mut whole, &geometry, &scaling()).unwrap();
    assert!(
        whole
            .iter()
            .zip(&ordinary)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    );
}

#[test]
fn invalid_parameters_and_inputs_reject_before_writes() {
    for value in [0., -1., 0.5, f32::NAN, f32::INFINITY] {
        assert!(Llama3Scaling::new(value, 1., 4., 8192).is_err());
    }
    for value in [0., -1., f32::NAN, f32::INFINITY] {
        assert!(Llama3Scaling::new(32., value, 4., 8192).is_err());
        assert!(Llama3Scaling::new(32., 1., value, 8192).is_err());
    }
    assert!(Llama3Scaling::new(32., 4., 4., 8192).is_err());
    assert!(Llama3Scaling::new(32., 5., 4., 8192).is_err());
    assert!(Llama3Scaling::new(32., 1., 4., 0).is_err());
    if let Ok(original) = usize::try_from((1_u64 << 53) + 1) {
        assert!(Llama3Scaling::new(32., 1., 4., original).is_err());
    }
    for value in [0., -1., f64::NAN, f64::INFINITY, f64::from_bits(1)] {
        assert!(scaling().inverse_frequency(value).is_err());
    }
    let mut bad = vec![
        Llama3RotaryShape {
            tokens: 0,
            ..shape()
        },
        Llama3RotaryShape {
            heads: 0,
            ..shape()
        },
        Llama3RotaryShape {
            heads: usize::MAX,
            ..shape()
        },
        Llama3RotaryShape {
            head_dim: 0,
            ..shape()
        },
        Llama3RotaryShape {
            head_dim: 3,
            ..shape()
        },
        Llama3RotaryShape {
            offset: usize::MAX,
            tokens: 2,
            ..shape()
        },
    ];
    if let Ok(offset) = usize::try_from((1_u64 << 53) + 1) {
        bad.push(Llama3RotaryShape { offset, ..shape() });
    }
    for theta in [0., -1., f32::NAN, f32::INFINITY] {
        bad.push(Llama3RotaryShape { theta, ..shape() });
    }
    let mut output = [7.; 17];
    for geometry in bad {
        assert!(llama3_rotary(&[1.; 16], &mut output[..16], &geometry, &scaling()).is_err());
        assert_eq!(output, [7.; 17]);
    }
    for count in [15, 17] {
        assert!(
            llama3_rotary(&[1.; 17][..count], &mut output[..16], &shape(), &scaling()).is_err()
        );
        assert!(llama3_rotary(&[1.; 16], &mut output[..count], &shape(), &scaling()).is_err());
        assert_eq!(output, [7.; 17]);
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut input = [1.; 16];
        input[15] = value;
        assert!(llama3_rotary(&input, &mut output[..16], &shape(), &scaling()).is_err());
        assert_eq!(output, [7.; 17]);
    }
}

#[test]
fn arithmetic_failure_is_explicit_and_exact_position_limit_is_accepted() {
    let input = [f32::MAX, -f32::MAX];
    let mut output = [0.; 2];
    assert!(matches!(
        llama3_rotary(
            &input,
            &mut output,
            &Llama3RotaryShape {
                head_dim: 2,
                ..shape()
            },
            &scaling()
        ),
        Err(Error::NonFinite { .. })
    ));
    if let Ok(offset) = usize::try_from(1_u64 << 53) {
        llama3_rotary(
            &[1.; 16],
            &mut [0.; 16],
            &Llama3RotaryShape { offset, ..shape() },
            &scaling(),
        )
        .unwrap();
    }
}
