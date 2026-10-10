use toklet::model::{LayerTrace, Trace};

fn arrays(layer: &LayerTrace) -> [(&'static str, &[f32]); 18] {
    [
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
    ]
}

/// Every element, including signed zero, must have identical FP32 bits.
pub fn assert_bits(actual: &Trace, expected: &Trace) -> usize {
    let equal = |name: &str, actual: &[f32], expected: &[f32]| {
        assert_eq!(actual.len(), expected.len(), "{name} shape");
        for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{name}[{index}]: {a} versus {b}");
        }
    };
    for (name, a, b) in [
        ("embedding", &actual.embedding, &expected.embedding),
        ("final_norm", &actual.final_norm, &expected.final_norm),
        ("logits", &actual.logits, &expected.logits),
    ] {
        equal(name, a, b);
    }
    assert_eq!(actual.layers.len(), expected.layers.len());
    for (a, b) in actual.layers.iter().zip(&expected.layers) {
        for ((name, a), (_, b)) in arrays(a).into_iter().zip(arrays(b)) {
            equal(name, a, b);
        }
    }
    3 + actual.layers.len() * 18
}
