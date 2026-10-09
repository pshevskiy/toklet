mod support;

use support::Fixture;
use toklet::{
    cpu,
    model::{Config, Error, LayerWeights, Model, Projection, Weights},
    sampling,
};

fn tiny_parts(f: &Fixture) -> (Config, Weights) {
    let m = &f.manifest["model"];
    let integer = |name: &str| m[name].as_u64().unwrap() as usize;
    let config = Config {
        vocab_size: integer("vocab_size"),
        hidden_size: integer("hidden_size"),
        intermediate_size: integer("intermediate_size"),
        num_layers: integer("num_layers"),
        query_heads: integer("num_query_heads"),
        kv_heads: integer("num_kv_heads"),
        head_dim: integer("head_dim"),
        max_positions: integer("capacity"),
        rms_norm_eps: m["rms_norm_eps"].as_f64().unwrap() as f32,
        rope_theta: m["rope_theta"].as_f64().unwrap() as f32,
    };
    let layers = (0..config.num_layers)
        .map(|index| {
            let prefix = format!("weights.layers.{index}");
            let projection = |name| Projection {
                weight: f.f32(&format!("{prefix}.{name}_proj.weight")),
                bias: f.f32(&format!("{prefix}.{name}_proj.bias")),
            };
            LayerWeights {
                input_norm: f.f32(&format!("{prefix}.input_layernorm")),
                post_attention_norm: f.f32(&format!("{prefix}.post_attention_layernorm")),
                q: projection("q"),
                k: projection("k"),
                v: projection("v"),
                o: f.f32(&format!("{prefix}.o_proj.weight")),
                gate: f.f32(&format!("{prefix}.gate_proj.weight")),
                up: f.f32(&format!("{prefix}.up_proj.weight")),
                down: f.f32(&format!("{prefix}.down_proj.weight")),
            }
        })
        .collect();
    (
        config,
        Weights {
            embedding: f.f32("weights.embed_tokens"),
            layers,
            final_norm: f.f32("weights.norm"),
        },
    )
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs(),
            "index {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn composed_decoder_matches_every_named_array_across_all_cases() {
    let fixture = Fixture::tiny();
    let (config, weights) = tiny_parts(&fixture);
    let model = Model::new(config, weights).unwrap();
    let mut scratch = model.scratch(32).unwrap();
    assert_eq!(scratch.capacity(), 32);
    let allocated = scratch.allocated_bytes();
    assert!(allocated > 0);
    for case in ["prompt", "future_changed", "offset", "single", "capacity"] {
        let tokens = fixture.u32(&format!("cases.{case}.tokens"));
        let offset = fixture.manifest["cases"][case]["position_offset"]
            .as_u64()
            .unwrap() as usize;
        let trace = model
            .forward_diagnostic(&tokens, offset, &mut scratch)
            .unwrap();
        close(
            &trace.embedding,
            &fixture.f32(&format!("cases.{case}.embedding")),
        );
        close(
            &trace.final_norm,
            &fixture.f32(&format!("cases.{case}.final_norm")),
        );
        close(&trace.logits, &fixture.f32(&format!("cases.{case}.logits")));
        for (index, layer) in trace.layers.iter().enumerate() {
            let prefix = format!("cases.{case}.layers.{index}");
            for (name, values) in [
                ("input_norm", &layer.input_norm),
                ("q_projection", &layer.q_projection),
                ("k_projection", &layer.k_projection),
                ("v_projection", &layer.v_projection),
                ("q_rotary", &layer.q_rotary),
                ("k_rotary", &layer.k_rotary),
                ("attention_scores", &layer.attention_scores),
                ("attention_probabilities", &layer.attention_probabilities),
                ("attention_context", &layer.attention_context),
                ("attention_projection", &layer.attention_projection),
                ("attention_residual", &layer.attention_residual),
                ("post_attention_norm", &layer.post_attention_norm),
                ("gate_projection", &layer.gate_projection),
                ("up_projection", &layer.up_projection),
                ("gate_silu", &layer.gate_silu),
                ("gated", &layer.gated),
                ("down_projection", &layer.down_projection),
                ("output", &layer.output),
            ] {
                close(values, &fixture.f32(&format!("{prefix}.{name}")));
            }
        }
        let mut last = [0.0; 64];
        model
            .forward(&tokens, offset, &mut scratch, &mut last)
            .unwrap();
        assert_eq!(last, trace.logits[(tokens.len() - 1) * 64..]);
        assert_eq!(scratch.allocated_bytes(), allocated);
    }
}

#[test]
fn full_prefix_greedy_generation_matches_saved_logits_and_tokens() {
    let fixture = Fixture::tiny();
    let (config, weights) = tiny_parts(&fixture);
    let model = Model::new(config, weights).unwrap();
    let mut scratch = model.scratch(32).unwrap();
    let mut history: Vec<u32> = fixture.manifest["greedy"]["prompt_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.as_u64().unwrap() as u32)
        .collect();
    for (step, expected) in fixture.manifest["greedy"]["generated_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let mut logits = [0.0; 64];
        model
            .forward(&history, 0, &mut scratch, &mut logits)
            .unwrap();
        close(&logits, &fixture.f32(&format!("greedy.step_{step}.logits")));
        let token = sampling::greedy(&logits).unwrap();
        assert_eq!(u64::from(token), expected.as_u64().unwrap());
        history.push(token);
    }
}

#[test]
fn composed_decoder_preserves_causal_outputs_when_future_tokens_change() {
    let fixture = Fixture::tiny();
    let (config, weights) = tiny_parts(&fixture);
    let model = Model::new(config, weights).unwrap();
    let mut scratch = model.scratch(4).unwrap();
    let first = model
        .forward_diagnostic(&fixture.u32("cases.prompt.tokens"), 0, &mut scratch)
        .unwrap();
    let second = model
        .forward_diagnostic(&fixture.u32("cases.future_changed.tokens"), 0, &mut scratch)
        .unwrap();
    assert_eq!(first.logits[..3 * 64], second.logits[..3 * 64]);
    assert_ne!(first.logits[3 * 64..], second.logits[3 * 64..]);
    for (first, second) in first.layers.iter().zip(&second.layers) {
        assert_eq!(first.output[..3 * 32], second.output[..3 * 32]);
    }
}

#[test]
fn greedy_uses_lowest_id_ties_and_rejects_invalid_distributions() {
    assert_eq!(sampling::greedy(&[3.0, 3.0, 1.0]).unwrap(), 0);
    assert_eq!(sampling::greedy(&[-4.0, -2.0, -2.0]).unwrap(), 1);
    assert_eq!(
        sampling::greedy(&[]),
        Err(cpu::Error::ZeroDimension("vocabulary"))
    );
    for values in [
        [f32::NAN, 1.0],
        [1.0, f32::INFINITY],
        [1.0, f32::NEG_INFINITY],
    ] {
        assert!(matches!(
            sampling::greedy(&values),
            Err(cpu::Error::NonFinite {
                tensor: "logits",
                ..
            })
        ));
    }
}

#[test]
fn constructor_rejects_invalid_configuration_shapes_biases_and_weights() {
    let fixture = Fixture::tiny();
    for variant in 0..6 {
        let (mut c, w) = tiny_parts(&fixture);
        match variant {
            0 => c.hidden_size = 31,
            1 => c.kv_heads = 3,
            2 => c.head_dim = 7,
            3 => c.rope_theta = f32::NAN,
            4 => c.rms_norm_eps = 0.0,
            _ => c.intermediate_size = usize::MAX,
        }
        assert!(Model::new(c, w).is_err());
    }
    let (c, mut w) = tiny_parts(&fixture);
    w.layers.pop();
    assert!(matches!(
        Model::new(c, w),
        Err(Error::LayerCount {
            expected: 2,
            actual: 1
        })
    ));
    let (c, mut w) = tiny_parts(&fixture);
    w.layers[1].q.bias.pop();
    assert!(matches!(
        Model::new(c, w),
        Err(Error::Kernel(cpu::Error::Length {
            tensor: "QKV bias",
            ..
        }))
    ));
    let (c, mut w) = tiny_parts(&fixture);
    w.layers[1].down[17] = f32::NAN;
    assert!(matches!(
        Model::new(c, w),
        Err(Error::Kernel(cpu::Error::NonFinite {
            tensor: "MLP weight",
            index: 17
        }))
    ));
}

#[test]
fn admission_rejects_invalid_requests_and_scratch_remains_reusable() {
    let fixture = Fixture::tiny();
    let (c, w) = tiny_parts(&fixture);
    let model = Model::new(c, w).unwrap();
    let mut scratch = model.scratch(4).unwrap();
    let mut logits = [123.0; 64];
    assert!(matches!(
        model.forward(&[1, 2, 3, 4, 5], 0, &mut scratch, &mut logits),
        Err(Error::ScratchCapacity { .. })
    ));
    assert!(matches!(
        model.forward(&[1, 2, 3, 4], 29, &mut scratch, &mut logits),
        Err(Error::Capacity { .. })
    ));
    assert!(matches!(
        model.forward(&[1], usize::MAX, &mut scratch, &mut logits),
        Err(Error::Kernel(cpu::Error::SizeOverflow))
    ));
    assert!(matches!(
        model.forward(&[64], 0, &mut scratch, &mut logits),
        Err(Error::Kernel(cpu::Error::Token { .. }))
    ));
    assert!(model.forward(&[], 0, &mut scratch, &mut logits).is_err());
    assert!(
        model
            .forward(&[1], 0, &mut scratch, &mut logits[..63])
            .is_err()
    );
    assert_eq!(logits, [123.0; 64]);
    assert!(model.scratch(0).is_err());
    assert!(model.scratch(33).is_err());
    let (mut c, w) = tiny_parts(&fixture);
    c.rope_theta = 10000.0;
    let different_model = Model::new(c, w).unwrap();
    assert_eq!(
        different_model.forward(&[1], 0, &mut scratch, &mut logits),
        Err(Error::ScratchMismatch)
    );
    model
        .forward(
            &fixture.u32("cases.prompt.tokens"),
            0,
            &mut scratch,
            &mut logits,
        )
        .unwrap();
    close(&logits, &fixture.f32("cases.prompt.logits")[3 * 64..]);
}

#[test]
fn output_projection_shares_embedding_weights() {
    let fixture = Fixture::tiny();
    let (c, mut w) = tiny_parts(&fixture);
    // Token 63 is absent from the input. Changing its row can affect only its
    // output logit because the output uses the same immutable embedding table.
    w.embedding[63 * 32..].fill(0.0);
    let model = Model::new(c, w).unwrap();
    let mut scratch = model.scratch(4).unwrap();
    let mut logits = [0.0; 64];
    model
        .forward(
            &fixture.u32("cases.prompt.tokens"),
            0,
            &mut scratch,
            &mut logits,
        )
        .unwrap();
    let expected = fixture.f32("cases.prompt.logits");
    close(&logits[..63], &expected[3 * 64..4 * 64 - 1]);
    assert_eq!(logits[63], 0.0);
}
