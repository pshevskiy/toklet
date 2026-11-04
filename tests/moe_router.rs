use serde_json::Value;
use sha2::{Digest, Sha256};
use toklet::cpu::{RouterShape, route_topk};

#[test]
fn selected_router_matches_frozen_independent_fixture() {
    let raw = include_bytes!("fixtures/moe-router-v1/fixture.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "68ad524d8abb8b7ed6ed6a4a6c6d2cdf0e2fcba394e667a4a9afc39e98bb0034"
    );
    let fixture: Value = serde_json::from_slice(raw).unwrap();
    assert_eq!(
        fixture["producer_sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../tools/reference/moe_router_fixture.py"))
        )
    );
    for case in fixture["cases"].as_array().unwrap() {
        let scores: Vec<_> = case["score_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| f32::from_bits(v.as_u64().unwrap() as u32))
            .collect();
        let shape = RouterShape {
            tokens: case["tokens"].as_u64().unwrap() as usize,
            experts: case["experts"].as_u64().unwrap() as usize,
            selected: case["selected"].as_u64().unwrap() as usize,
        };
        let mut ids = vec![u32::MAX; shape.tokens * shape.selected];
        let mut gates = vec![f32::NAN; ids.len()];
        route_topk(&scores, &mut ids, &mut gates, shape).unwrap();
        let expected: Vec<_> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(ids, expected, "{}", case["name"]);
        for (i, &gate) in gates.iter().enumerate() {
            let expected = f32::from_bits(case["gate_bits"][i].as_u64().unwrap() as u32);
            let tolerance = fixture["gate_atol"].as_f64().unwrap()
                + fixture["gate_rtol"].as_f64().unwrap() * f64::from(expected).abs();
            assert!(
                gate.is_finite()
                    && gate >= 0.
                    && (f64::from(gate) - f64::from(expected)).abs() <= tolerance,
                "{} slot{i}: {gate} vs {expected}",
                case["name"]
            );
        }
        for row in gates.chunks_exact(shape.selected) {
            assert!((row.iter().map(|&x| f64::from(x)).sum::<f64>() - 1.).abs() < 1e-6);
        }
    }
}

#[test]
fn invalid_router_inputs_preserve_every_output() {
    let mut ids = [99; 4];
    let mut gates = [-7.; 4];
    let valid = RouterShape {
        tokens: 2,
        experts: 4,
        selected: 2,
    };
    for shape in [
        RouterShape { tokens: 0, ..valid },
        RouterShape {
            experts: 0,
            ..valid
        },
        RouterShape {
            selected: 0,
            ..valid
        },
        RouterShape {
            selected: 5,
            ..valid
        },
        RouterShape {
            experts: 33,
            ..valid
        },
        RouterShape {
            tokens: usize::MAX,
            ..valid
        },
    ] {
        assert!(route_topk(&[0.; 8], &mut ids, &mut gates, shape).is_err());
        assert_eq!(ids, [99; 4]);
        assert_eq!(gates, [-7.; 4]);
    }
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut scores = [0.; 8];
        scores[7] = bad;
        assert!(route_topk(&scores, &mut ids, &mut gates, valid).is_err());
        assert_eq!(ids, [99; 4]);
        assert_eq!(gates, [-7.; 4]);
    }
    assert!(route_topk(&[0.; 7], &mut ids, &mut gates, valid).is_err());
    assert!(route_topk(&[0.; 8], &mut ids[..3], &mut gates, valid).is_err());
    assert!(route_topk(&[0.; 8], &mut ids, &mut gates[..3], valid).is_err());
    assert_eq!(ids, [99; 4]);
    assert_eq!(gates, [-7.; 4]);
}

#[test]
fn router_retains_ranked_fp32_mass_boundary() {
    // exp(-17) < half an ULP at 1: each ranked FP32 addition rounds back to1.
    // Summing the seven small terms first, or summing in FP64, changes gate0.
    let mut scores = [-17.; 32];
    scores[0] = 0.;
    let mut ids = [u32::MAX; 8];
    let mut gates = [0.; 8];
    route_topk(
        &scores,
        &mut ids,
        &mut gates,
        RouterShape {
            tokens: 1,
            experts: 32,
            selected: 8,
        },
    )
    .unwrap();
    assert_eq!(ids, [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(gates[0].to_bits(), 1_f32.to_bits());
    assert!(
        gates[1..]
            .iter()
            .all(|&gate| gate > 0. && gate < 2_f32.powi(-24))
    );
    let wide = 1. / (1. + 7. * (-17_f64).exp());
    assert_ne!((wide as f32).to_bits(), gates[0].to_bits());
    let mut pair_ids = [u32::MAX; 2];
    let mut pair_gates = [-1.; 2];
    route_topk(
        &[f32::MAX, -f32::MAX],
        &mut pair_ids,
        &mut pair_gates,
        RouterShape {
            tokens: 1,
            experts: 2,
            selected: 2,
        },
    )
    .unwrap();
    assert_eq!(pair_ids, [0, 1]);
    assert_eq!(
        pair_gates.map(f32::to_bits),
        [1_f32.to_bits(), 0_f32.to_bits()]
    );
    let mut tied = [-1.; 32];
    tied[0] = 2.;
    tied[1..10].fill(1.);
    route_topk(
        &tied,
        &mut ids,
        &mut gates,
        RouterShape {
            tokens: 1,
            experts: 32,
            selected: 8,
        },
    )
    .unwrap();
    assert_eq!(ids, [0, 1, 2, 3, 4, 5, 6, 7]);
}
