use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

type Inputs = BTreeMap<String, Vec<f32>>;
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
fn weights_at_width(
    x: &Inputs,
    width: Option<usize>,
    scalar: Option<f32>,
) -> DenseDecoderWeights<'_> {
    DenseDecoderWeights {
        post_attention_norm: &x["post_attention_norm"],
        pre_ff_norm: &x["pre_ff_norm"],
        post_ff_norm: &x["post_ff_norm"],
        ff_gate: &x["gate_weight"],
        ff_up: &x["up_weight"],
        ff_down: &x["down_weight"],
        per_layer: width.map(|width| PerLayerGateWeights {
            width,
            gate: &x["ple_gate_weight"],
            projection: &x["ple_projection_weight"],
            post_norm: &x["post_ple_norm"],
        }),
        scalar,
    }
}
struct Buffers {
    work: Vec<f32>,
    norm: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    product: Vec<f32>,
    output: Vec<f32>,
}
impl Buffers {
    fn new(s: &DenseDecoderShape, width: usize, fill: f32) -> Self {
        Self {
            work: vec![fill; s.tokens * s.hidden],
            norm: vec![fill; s.tokens * s.hidden],
            gate: vec![fill; s.tokens * s.intermediate.max(width)],
            up: vec![fill; s.tokens * s.intermediate],
            product: vec![fill; s.tokens * s.intermediate.max(width)],
            output: vec![fill; s.tokens * s.hidden],
        }
    }
    fn all(&self) -> [&[f32]; 6] {
        [
            &self.work,
            &self.norm,
            &self.gate,
            &self.up,
            &self.product,
            &self.output,
        ]
    }
    fn snapshot(&self) -> Vec<Vec<u32>> {
        self.all().map(bits).to_vec()
    }
    fn run(
        &mut self,
        x: &Inputs,
        input: Option<&[f32]>,
        w: &DenseDecoderWeights<'_>,
        s: &DenseDecoderShape,
    ) -> Result<(), Error> {
        finish_dense_decoder(
            &x["hidden"],
            &x["attention_output"],
            input,
            w,
            &mut DenseDecoderScratch {
                hidden_work: &mut self.work,
                hidden_norm: &mut self.norm,
                gate: &mut self.gate,
                up: &mut self.up,
                product: &mut self.product,
            },
            &mut self.output,
            s,
        )
    }
}
fn hand() -> (DenseDecoderShape, Inputs) {
    let s = DenseDecoderShape {
        tokens: 1,
        hidden: 2,
        intermediate: 2,
        epsilon: 1e-6,
    };
    let x = [
        ("hidden", vec![0.5, -0.25]),
        ("attention_output", vec![1., -0.5]),
        ("per_layer_input", vec![0.75, -0.5]),
        ("post_attention_norm", vec![0.75, -0.5]),
        ("pre_ff_norm", vec![-0.5, 1.25]),
        ("post_ff_norm", vec![1.25, 0.375]),
        ("post_ple_norm", vec![0.375, 0.75]),
        ("gate_weight", vec![0.5, -0.25, 0.125, 0.75]),
        ("up_weight", vec![-0.5, 0.25, 0.25, -0.75]),
        ("down_weight", vec![0.25, -0.5, 0.75, 0.125]),
        ("ple_gate_weight", vec![0.125, -0.25, -0.5, 0.75]),
        ("ple_projection_weight", vec![0.25, -0.125, 0.75, -0.5]),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), v))
    .collect();
    (s, x)
}
fn unchanged(x: &Inputs) -> Vec<(String, Vec<u32>)> {
    x.iter().map(|(k, v)| (k.clone(), bits(v))).collect()
}

#[test]
fn frozen_order_and_reused_stage_buffers_match_both_contracts() {
    let receipt = include_bytes!(
        "../../../benchmarks/results/2026-10-24-gemma-decoder-order-reference-v1.json"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(receipt)),
        "b15cedce8d253f91b55ef01bd7826f9c27f13e1b8f55b7d465fe2fb633ae538e"
    );
    let accepted: Value = serde_json::from_slice(receipt).unwrap();
    let raw = include_bytes!("../../../tests/fixtures/gemma-decoder-order-v1/math.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        accepted["fixture_sha256"].as_str().unwrap()
    );
    let data: Value = serde_json::from_slice(raw).unwrap();
    let mut checked = 0;
    for case in data["cases"].as_array().unwrap() {
        let n = |k: &str| case[k].as_u64().unwrap() as usize;
        let tokens = n("batch") * n("tokens");
        if tokens == 0 {
            continue;
        }
        let s = DenseDecoderShape {
            tokens,
            hidden: n("hidden"),
            intermediate: n("intermediate"),
            epsilon: case["epsilon"].as_f64().unwrap() as f32,
        };
        let values = |g: &str, k: &str| {
            case[g][k]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect::<Vec<_>>()
        };
        let x: Inputs = case["inputs"]
            .as_object()
            .unwrap()
            .keys()
            .map(|k| {
                (
                    k.clone(),
                    values("inputs", k).iter().map(|&v| v as f32).collect(),
                )
            })
            .collect();
        let mode = case["ple_mode"].as_str().unwrap();
        let present = matches!(mode, "full" | "no_input");
        let scalar = case["with_scalar"]
            .as_bool()
            .unwrap()
            .then_some(x["layer_scalar"][0]);
        let w = weights_at_width(&x, present.then_some(n("width")), scalar);
        // Source's partial-module guards are represented as coherent native disabled
        // PLE; a future adapter must reject partial checkpoint weight inventories.
        let ple = (mode == "full").then_some(x["per_layer_input"].as_slice());
        let before = unchanged(&x);
        let mut b = Buffers::new(&s, if present { n("width") } else { 0 }, f32::NAN);
        let pointers = b.all().map(|v| v.as_ptr());
        let mut normalized = vec![0.; tokens * s.hidden];
        rmsnorm(
            &x["hidden"],
            &x["input_norm"],
            &mut normalized,
            tokens,
            s.hidden,
            s.epsilon,
        )
        .unwrap();
        b.run(&x, ple, &w, &s).unwrap();
        for group in ["math", "staged_fp32"] {
            close(&normalized, &values(group, "attention_input"));
            close(&b.output, &values(group, "output"));
            close(&b.up, &values(group, "ff_up"));
            let (gate, product, work, norm, width) = if mode == "full" {
                (
                    "ple_gate",
                    "ple_product",
                    "ple_projected",
                    "ple_norm",
                    n("width"),
                )
            } else {
                (
                    "ff_gate",
                    "ff_product",
                    "ff_down",
                    "ff_norm",
                    s.intermediate,
                )
            };
            close(&b.gate[..tokens * width], &values(group, gate));
            close(&b.product[..tokens * width], &values(group, product));
            close(&b.work, &values(group, work));
            close(&b.norm, &values(group, norm));
        }
        assert_eq!(before, unchanged(&x));
        assert_eq!(pointers, b.all().map(|v| v.as_ptr()));
        checked += 1;
    }
    assert_eq!(checked, 11);
}

fn rejects_without_writes(
    x: &Inputs,
    w: &DenseDecoderWeights<'_>,
    s: &DenseDecoderShape,
    ple: Option<&[f32]>,
) {
    let mut b = Buffers::new(s, w.per_layer.as_ref().map_or(0, |v| v.width), -123.25);
    let before = b.snapshot();
    let inputs = unchanged(x);
    assert!(b.run(x, ple, w, s).is_err());
    assert_eq!(before, b.snapshot());
    assert_eq!(inputs, unchanged(x));
}
#[test]
fn every_late_weight_input_scalar_and_scratch_error_precedes_writes() {
    let (s, x) = hand();
    for key in x.keys() {
        let mut changed = x.clone();
        *changed.get_mut(key).unwrap().last_mut().unwrap() = f32::NAN;
        let w = weights_at_width(&changed, Some(2), Some(-1.25));
        rejects_without_writes(&changed, &w, &s, Some(&changed["per_layer_input"]));
        // Every PLE weight is admitted even if supplied input is absent.
        if key.starts_with("ple_") || key == "post_ple_norm" {
            rejects_without_writes(&changed, &w, &s, None);
        }
        let mut changed = x.clone();
        changed.get_mut(key).unwrap().pop();
        let w = weights_at_width(&changed, Some(2), Some(-1.25));
        rejects_without_writes(&changed, &w, &s, Some(&changed["per_layer_input"]));
    }
    for scalar in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        rejects_without_writes(
            &x,
            &weights_at_width(&x, Some(2), Some(scalar)),
            &s,
            Some(&x["per_layer_input"]),
        );
    }
    let w = weights_at_width(&x, Some(2), Some(-1.25));
    for index in 0..6 {
        let mut b = Buffers::new(&s, 2, -123.25);
        match index {
            0 => b.work.pop(),
            1 => b.norm.pop(),
            2 => b.gate.pop(),
            3 => b.up.pop(),
            4 => b.product.pop(),
            _ => b.output.pop(),
        };
        let before = b.snapshot();
        assert!(b.run(&x, Some(&x["per_layer_input"]), &w, &s).is_err());
        assert_eq!(before, b.snapshot());
    }
    rejects_without_writes(
        &x,
        &weights_at_width(&x, None, Some(-1.25)),
        &s,
        Some(&x["per_layer_input"]),
    );
}

// Pin the failed operation and its successful prerequisites, so a later-stage
// witness cannot accidentally pass because an earlier projection overflowed.
fn assert_overflow_stage(
    stage: usize,
    x: &Inputs,
    s: &DenseDecoderShape,
    b: &Buffers,
    error: &Error,
) {
    let mut a = vec![0.; 2];
    if stage == 0 {
        let expected = rmsnorm(
            &x["attention_output"],
            &x["post_attention_norm"],
            &mut a,
            1,
            2,
            s.epsilon,
        )
        .unwrap_err();
        assert_eq!(*error, expected);
        assert!(b.all()[..5].iter().all(|v| bits(v) == bits(&[-123.25; 2])));
        return;
    }
    rmsnorm(
        &x["attention_output"],
        &x["post_attention_norm"],
        &mut a,
        1,
        2,
        s.epsilon,
    )
    .unwrap();
    if stage == 1 {
        assert_eq!(bits(&b.output), bits(&a));
        let expected = residual_add(&mut a, &x["hidden"]).unwrap_err();
        assert_eq!(*error, expected);
        assert!(b.all()[..5].iter().all(|v| bits(v) == bits(&[-123.25; 2])));
        return;
    }
    residual_add(&mut a, &x["hidden"]).unwrap();
    if stage <= 4 {
        let mut input = vec![0.; 2];
        rmsnorm(&a, &x["pre_ff_norm"], &mut input, 1, 2, s.epsilon).unwrap();
        assert_eq!(bits(&b.output), bits(&a));
        assert_eq!(bits(&b.norm), bits(&[-123.25; 2]));
        let mut gate = vec![0.; 2];
        if stage == 2 {
            assert_eq!(bits(&b.work), bits(&input));
            let expected =
                matmul_f64(&input, &x["gate_weight"], None, &mut gate, 1, 2, 2).unwrap_err();
            assert_eq!(*error, expected);
            assert_eq!(bits(&b.up), bits(&[-123.25; 2]));
            assert_eq!(bits(&b.product), bits(&[-123.25; 2]));
            return;
        }
        matmul_f64(&input, &x["gate_weight"], None, &mut gate, 1, 2, 2).unwrap();
        let mut up = vec![0.; 2];
        matmul_f64(&input, &x["up_weight"], None, &mut up, 1, 2, 2).unwrap();
        assert_eq!(bits(&b.gate), bits(&gate));
        assert_eq!(bits(&b.up), bits(&up));
        let mut product = vec![0.; 2];
        if stage == 3 {
            assert_eq!(bits(&b.work), bits(&input));
            let expected = gated_gelu_tanh(&gate, &up, &mut product).unwrap_err();
            assert_eq!(*error, expected);
            return;
        }
        gated_gelu_tanh(&gate, &up, &mut product).unwrap();
        assert_eq!(bits(&b.product), bits(&product));
        let mut down = vec![0.; 2];
        let expected =
            matmul_f64(&product, &x["down_weight"], None, &mut down, 1, 2, 2).unwrap_err();
        assert_eq!(*error, expected);
        return;
    }
    let mut baseline = Buffers::new(s, 2, 0.);
    let input = (stage == 6).then_some(x["per_layer_input"].as_slice());
    baseline
        .run(x, input, &weights_at_width(x, Some(2), None), s)
        .unwrap();
    if stage == 5 {
        assert_eq!(bits(&b.output), bits(&baseline.output));
        assert_eq!(bits(&b.work), bits(&baseline.work));
        assert_eq!(bits(&b.norm), bits(&baseline.norm));
        assert_eq!(bits(&b.up), bits(&baseline.up));
        let mut gate = vec![0.; 2];
        matmul_f64(
            &baseline.output,
            &x["ple_gate_weight"],
            None,
            &mut gate,
            1,
            2,
            2,
        )
        .unwrap();
        assert_eq!(bits(&b.gate), bits(&gate));
        let mut product = vec![0.; 2];
        let expected = gated_gelu_tanh(&gate, &x["per_layer_input"], &mut product).unwrap_err();
        assert_eq!(*error, expected);
    } else {
        assert_eq!(b.snapshot()[..5], baseline.snapshot()[..5]);
        let index = baseline
            .output
            .iter()
            .position(|v| !(v * f32::MAX).is_finite())
            .unwrap();
        assert_eq!(
            *error,
            Error::NonFinite {
                tensor: "dense scalar output",
                index
            }
        );
        for (j, (&actual, &unscaled)) in b.output.iter().zip(&baseline.output).enumerate() {
            let expected = if j < index {
                unscaled * f32::MAX
            } else {
                unscaled
            };
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
}

#[test]
fn arithmetic_failures_preserve_inputs_and_allow_fresh_replay() {
    let (s, base) = hand();
    let mut reference = Buffers::new(&s, 2, 0.);
    reference
        .run(
            &base,
            Some(&base["per_layer_input"]),
            &weights_at_width(&base, Some(2), Some(-1.25)),
            &s,
        )
        .unwrap();
    for stage in 0..7 {
        let mut x = base.clone();
        let mut scalar = Some(-1.25);
        match stage {
            0 => {
                x.insert("attention_output".into(), vec![1., 0.]);
                x.insert("post_attention_norm".into(), vec![f32::MAX, 1.]);
            }
            1 => {
                x.insert("hidden".into(), vec![f32::MAX, 0.]);
                x.insert("attention_output".into(), vec![1., 0.]);
                x.insert("post_attention_norm".into(), vec![f32::MAX / 2., 1.]);
            }
            2 => {
                x.insert("gate_weight".into(), vec![f32::MAX; 4]);
                x.insert("pre_ff_norm".into(), vec![16.; 2]);
            }
            3 => {
                x.insert("pre_ff_norm".into(), vec![1.; 2]);
                x.insert("gate_weight".into(), vec![f32::MAX / 16.; 4]);
                x.insert("up_weight".into(), vec![f32::MAX / 16.; 4]);
            }
            4 => {
                x.insert("pre_ff_norm".into(), vec![1.; 2]);
                x.insert("gate_weight".into(), vec![16.; 4]);
                x.insert("up_weight".into(), vec![16.; 4]);
                x.insert("down_weight".into(), vec![f32::MAX / 2.; 4]);
            }
            5 => {
                x.insert("ple_gate_weight".into(), vec![16.; 4]);
                x.insert("per_layer_input".into(), vec![f32::MAX; 2]);
            }
            _ => scalar = Some(f32::MAX),
        }
        let before = unchanged(&x);
        let mut b = Buffers::new(&s, 2, -123.25);
        let result = b.run(
            &x,
            Some(&x["per_layer_input"]),
            &weights_at_width(&x, Some(2), scalar),
            &s,
        );
        assert!(
            matches!(result, Err(Error::NonFinite { .. })),
            "stage {stage}: {result:?}"
        );
        assert_overflow_stage(stage, &x, &s, &b, result.as_ref().unwrap_err());
        assert_eq!(before, unchanged(&x));
        b.run(
            &base,
            Some(&base["per_layer_input"]),
            &weights_at_width(&base, Some(2), Some(-1.25)),
            &s,
        )
        .unwrap();
        assert_eq!(bits(&b.output), bits(&reference.output));
    }
}

#[test]
fn exact_independent_chunks_reuse_buffers_without_mutating_operands() {
    let (mut s, mut x) = hand();
    let row_keys = ["hidden", "attention_output", "per_layer_input"];
    for key in row_keys {
        let row = x[key].clone();
        x.insert(key.into(), row.repeat(3));
    }
    s.tokens = 3;
    let before = unchanged(&x);
    let mut full = Buffers::new(&s, 2, 0.);
    full.run(
        &x,
        Some(&x["per_layer_input"]),
        &weights_at_width(&x, Some(2), Some(-1.25)),
        &s,
    )
    .unwrap();
    for (start, end) in [(0, 1), (1, 3)] {
        let mut chunk = x.clone();
        for key in row_keys {
            chunk.insert(key.into(), x[key][start * 2..end * 2].to_vec());
        }
        let shape = DenseDecoderShape {
            tokens: end - start,
            ..s
        };
        let mut b = Buffers::new(&shape, 2, f32::NAN);
        let pointers = b.all().map(|v| v.as_ptr());
        for _ in 0..3 {
            b.run(
                &chunk,
                Some(&chunk["per_layer_input"]),
                &weights_at_width(&chunk, Some(2), Some(-1.25)),
                &shape,
            )
            .unwrap();
            assert_eq!(bits(&b.output), bits(&full.output[start * 2..end * 2]));
            assert_eq!(pointers, b.all().map(|v| v.as_ptr()));
        }
    }
    assert_eq!(before, unchanged(&x));
}

#[test]
fn disabled_and_missing_input_skip_gate_but_scalar_is_always_last() {
    let (s, x) = hand();
    let mut a = Buffers::new(&s, 2, 0.);
    a.run(&x, None, &weights_at_width(&x, Some(2), Some(-1.25)), &s)
        .unwrap();
    let mut b = Buffers::new(&s, 0, 0.);
    b.run(&x, None, &weights_at_width(&x, None, Some(-1.25)), &s)
        .unwrap();
    assert_eq!(bits(&a.output), bits(&b.output));
    b.run(&x, None, &weights_at_width(&x, None, None), &s)
        .unwrap();
    for (&with, &without) in a.output.iter().zip(&b.output) {
        assert_eq!(with.to_bits(), (without * -1.25).to_bits());
    }
    let mut z = x.clone();
    for key in ["hidden", "attention_output"] {
        z.insert(key.into(), vec![0.; 2]);
    }
    b.run(&z, None, &weights_at_width(&z, None, Some(-0.)), &s)
        .unwrap();
    assert_eq!(bits(&b.output), vec![(-0_f32).to_bits(); 2]);
}

#[test]
fn geometry_overflow_empty_and_invalid_epsilon_reject_without_writes() {
    let (s, x) = hand();
    let w = weights_at_width(&x, Some(2), Some(1.));
    for epsilon in [0., -1., f32::NAN, f32::INFINITY] {
        let shape = DenseDecoderShape { epsilon, ..s };
        rejects_without_writes(&x, &w, &shape, Some(&x["per_layer_input"]));
    }
    for (tokens, hidden, intermediate) in [
        (0, 2, 2),
        (1, 0, 2),
        (1, 2, 0),
        (usize::MAX, 2, 2),
        (1, usize::MAX, 2),
        (1, 2, usize::MAX),
    ] {
        let shape = DenseDecoderShape {
            tokens,
            hidden,
            intermediate,
            ..s
        };
        let mut scratch = [0.; 5];
        let mut output = [-17.; 2];
        let before = bits(&output);
        let [a, b, c, d, e] = &mut scratch;
        assert!(
            finish_dense_decoder(
                &[],
                &[],
                None,
                &w,
                &mut DenseDecoderScratch {
                    hidden_work: std::slice::from_mut(a),
                    hidden_norm: std::slice::from_mut(b),
                    gate: std::slice::from_mut(c),
                    up: std::slice::from_mut(d),
                    product: std::slice::from_mut(e)
                },
                &mut output,
                &shape
            )
            .is_err()
        );
        assert_eq!(bits(&output), before);
        assert_eq!(scratch, [0.; 5]);
    }
    let w = weights_at_width(&x, Some(0), Some(1.));
    rejects_without_writes(&x, &w, &s, None);
}
