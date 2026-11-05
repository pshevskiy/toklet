use serde_json::Value;
use sha2::{Digest, Sha256};
use toklet::cpu::{
    ExpertWeights, SelectedExpertShape as Shape, reduce_selected_experts, selected_expert_glu,
};
fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|n| f32::from_bits(n.as_u64().unwrap() as u32))
        .collect()
}
fn close(a: &[f32], b: &[f32], fixture: &Value) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!(
            x.is_finite()
                && (f64::from(x) - f64::from(y)).abs()
                    <= fixture["atol"].as_f64().unwrap()
                        + fixture["rtol"].as_f64().unwrap() * f64::from(y).abs(),
            "{i}: {x} vs {y}"
        );
    }
}
fn fixture() -> Value {
    let raw = include_bytes!("fixtures/moe-experts-v1/fixture.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "add465495c770e1f7f6582c7741596fd551dad447cbbe88aac2899d6ecc0b287"
    );
    let f: Value = serde_json::from_slice(raw).unwrap();
    assert_eq!(
        f["producer_sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../tools/reference/moe_experts_fixture.py"))
        )
    );
    f
}
#[test]
fn selected_experts_match_independent_stages_and_token_permutations() {
    let f = fixture();
    for case in f["cases"].as_array().unwrap() {
        let n = |name: &str| case[name].as_u64().unwrap() as usize;
        let s = Shape {
            tokens: n("tokens"),
            experts: n("experts"),
            selected: n("selected"),
            hidden: n("hidden"),
            intermediate: n("intermediate"),
        };
        let a = &case["arrays"];
        let input = floats(&a["input"]);
        let gate_up = floats(&a["gate_up"]);
        let down = floats(&a["down"]);
        let gates = floats(&a["gates"]);
        let ids: Vec<_> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap() as u32)
            .collect();
        let weights = ExpertWeights {
            gate_up: &gate_up,
            down: &down,
        };
        let mut out = vec![f32::NAN; s.tokens * s.selected * s.hidden];
        let mut gate = vec![f32::NAN; s.tokens * s.selected * s.intermediate];
        let mut up = gate.clone();
        selected_expert_glu(&input, &ids, &weights, &mut out, &mut gate, &mut up, s).unwrap();
        close(&gate, &floats(&a["activated"]), &f);
        close(&up, &floats(&a["up"]), &f);
        close(&out, &floats(&a["expert_output"]), &f);
        let mut reduced = vec![f32::NAN; s.tokens * s.hidden];
        reduce_selected_experts(&out, &gates, &mut reduced, s.tokens, s.selected, s.hidden)
            .unwrap();
        close(&reduced, &floats(&a["output"]), &f);
        // Reorder complete token rows; keep each token's supplied slot order.
        let reverse = |v: &[f32], width| {
            v.chunks_exact(width)
                .rev()
                .flatten()
                .copied()
                .collect::<Vec<_>>()
        };
        let ids_rev: Vec<_> = ids
            .chunks_exact(s.selected)
            .rev()
            .flatten()
            .copied()
            .collect();
        selected_expert_glu(
            &reverse(&input, s.hidden),
            &ids_rev,
            &weights,
            &mut out,
            &mut gate,
            &mut up,
            s,
        )
        .unwrap();
        close(
            &out,
            &reverse(&floats(&a["expert_output"]), s.selected * s.hidden),
            &f,
        );
        reduce_selected_experts(
            &out,
            &reverse(&gates, s.selected),
            &mut reduced,
            s.tokens,
            s.selected,
            s.hidden,
        )
        .unwrap();
        close(&reduced, &reverse(&floats(&a["output"]), s.hidden), &f);
    }
}
#[test]
fn reduction_has_separate_products_and_supplied_slot_order() {
    let f = fixture();
    for c in f["reduction_controls"].as_array().unwrap() {
        let floats = |name: &str| {
            c[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let values = floats("values");
        let gates = floats("gates");
        let mut out = [f32::NAN];
        reduce_selected_experts(&values, &gates, &mut out, 1, gates.len(), 1).unwrap();
        assert_eq!(
            out[0].to_bits(),
            (c["expected"].as_f64().unwrap() as f32).to_bits(),
            "{}",
            c["name"]
        );
    }
}
fn reject(
    input: &[f32],
    ids: &[u32],
    gate_up: &[f32],
    down: &[f32],
    shape: Shape,
    lens: [usize; 3],
) {
    let mut out = vec![7.; lens[0]];
    let mut gate = vec![8.; lens[1]];
    let mut up = vec![9.; lens[2]];
    assert!(
        selected_expert_glu(
            input,
            ids,
            &ExpertWeights { gate_up, down },
            &mut out,
            &mut gate,
            &mut up,
            shape
        )
        .is_err()
    );
    assert!(
        out.iter().all(|&v| v == 7.)
            && gate.iter().all(|&v| v == 8.)
            && up.iter().all(|&v| v == 9.)
    );
}
#[test]
fn expert_validation_preserves_every_buffer_including_unselected_weights() {
    let s = Shape {
        tokens: 2,
        experts: 4,
        selected: 2,
        hidden: 3,
        intermediate: 5,
    };
    let input = [1.; 6];
    let ids = [3, 1, 2, 0];
    let gate = [0.25; 120];
    let down = [0.5; 60];
    for shape in [
        Shape { tokens: 0, ..s },
        Shape { experts: 0, ..s },
        Shape { selected: 0, ..s },
        Shape { hidden: 0, ..s },
        Shape {
            intermediate: 0,
            ..s
        },
        Shape { experts: 33, ..s },
        Shape { selected: 5, ..s },
        Shape {
            tokens: usize::MAX,
            ..s
        },
        Shape {
            hidden: usize::MAX,
            ..s
        },
    ] {
        reject(&input, &ids, &gate, &down, shape, [12, 20, 20]);
    }
    for lens in [[11, 20, 20], [12, 19, 20], [12, 20, 19]] {
        reject(&input, &ids, &gate, &down, s, lens);
    }
    reject(&input[..5], &ids, &gate, &down, s, [12, 20, 20]);
    reject(&input, &ids[..3], &gate, &down, s, [12, 20, 20]);
    reject(&input, &ids, &gate[..119], &down, s, [12, 20, 20]);
    reject(&input, &ids, &gate, &down[..59], s, [12, 20, 20]);
    for ids in [[3, 1, 2, 2], [3, 1, 2, 4], [3, 1, 2, u32::MAX]] {
        reject(&input, &ids, &gate, &down, s, [12, 20, 20]);
    }
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut x = input;
        x[5] = bad;
        reject(&x, &ids, &gate, &down, s, [12, 20, 20]);
        let mut w = gate;
        w[119] = bad;
        reject(&input, &ids, &w, &down, s, [12, 20, 20]);
        let mut w = down;
        w[59] = bad;
        reject(&input, &ids, &gate, &w, s, [12, 20, 20]);
        // Expert0 is unselected here, yet full immutable weights are validated.
        let mut w = gate;
        w[0] = bad;
        reject(&input, &[3, 1, 3, 1], &w, &down, s, [12, 20, 20]);
    }
}
#[test]
fn reduction_validation_and_arithmetic_failures_are_distinct() {
    for (values, gates, t, k, h) in [
        (vec![1.; 4], vec![1.; 2], 0, 2, 2),
        (vec![1.; 4], vec![1.; 2], 1, 0, 2),
        (vec![1.; 4], vec![1.; 2], 1, 2, 0),
        (vec![1.; 4], vec![1.; 2], usize::MAX, 2, 2),
        (vec![1.; 4], vec![1.; 2], 1, 33, 2),
        (vec![1.; 3], vec![1.; 2], 1, 2, 2),
        (vec![1.; 4], vec![1.], 1, 2, 2),
        (vec![1.; 4], vec![1., -0.1], 1, 2, 2),
        (vec![1.; 4], vec![1., f32::NAN], 1, 2, 2),
        (vec![1., 1., 1., f32::INFINITY], vec![1., 0.], 1, 2, 2),
    ] {
        let mut out = [7.; 2];
        assert!(reduce_selected_experts(&values, &gates, &mut out, t, k, h).is_err());
        assert_eq!(out, [7.; 2]);
    }
    assert!(reduce_selected_experts(&[1.; 4], &[1.; 2], &mut [7.; 1], 1, 2, 2).is_err());
    assert!(reduce_selected_experts(&[f32::MAX], &[2.], &mut [0.], 1, 1, 1).is_err());
    assert!(reduce_selected_experts(&[f32::MAX, f32::MAX], &[1., 1.], &mut [0.], 1, 2, 1).is_err());
    let s = Shape {
        tokens: 1,
        experts: 1,
        selected: 1,
        hidden: 1,
        intermediate: 1,
    };
    assert!(
        selected_expert_glu(
            &[f32::MAX],
            &[0],
            &ExpertWeights {
                gate_up: &[2., 1.],
                down: &[1.]
            },
            &mut [0.],
            &mut [0.],
            &mut [0.],
            s
        )
        .is_err()
    );
}
