mod support;

use serde_json::json;
use support::Fixture;

#[test]
fn every_frozen_array_has_the_declared_shape_hash_and_finite_values() {
    let fixture = Fixture::tiny();
    let arrays = fixture.manifest["arrays"].as_object().unwrap();
    assert_eq!(arrays.len(), 242);
    for (name, metadata) in arrays {
        match metadata["dtype"].as_str().unwrap() {
            "f32le" => assert!(fixture.f32(name).iter().all(|value| value.is_finite())),
            "u32le" => assert!(fixture.u32(name).iter().all(|&token| token < 64)),
            dtype => panic!("unsupported fixture dtype: {dtype}"),
        }
    }
}

#[test]
fn topology_and_numerical_policy_match_the_learning_model() {
    let fixture = Fixture::tiny();
    assert_eq!(fixture.manifest["schema_version"], 1);
    assert_eq!(fixture.manifest["fixture_id"], "toklet-tiny-fp32-v1");
    assert_eq!(
        fixture.manifest["model"],
        json!({
            "vocab_size": 64, "num_layers": 2, "hidden_size": 32,
            "num_query_heads": 4, "num_kv_heads": 2, "head_dim": 8,
            "intermediate_size": 64, "capacity": 32,
            "rms_norm_eps": 1e-6, "rope_theta": 1_000_000.0,
            "rope_pairing": "split_half", "tie_word_embeddings": true,
            "projection_biases": ["q", "k", "v"]
        })
    );
    let policy = &fixture.manifest["numerical_policy"];
    for tolerance in policy["tolerances"].as_object().unwrap().values() {
        assert_eq!(tolerance, &json!({ "atol": 1e-5, "rtol": 1e-4 }));
    }
    for envelope in policy["measured_fp32_vs_fp64"]
        .as_object()
        .unwrap()
        .values()
    {
        let fraction = envelope["max_fraction_of_tolerance"].as_f64().unwrap();
        assert!(fraction.is_finite() && (0.0..=1.0).contains(&fraction));
    }
    assert_eq!(fixture.shape("weights.embed_tokens"), [64, 32]);
    assert!(fixture.manifest["arrays"].get("weights.lm_head").is_none());
    assert_eq!(fixture.shape("cases.capacity.logits"), [32, 64]);
    assert_eq!(fixture.manifest["cases"]["offset"]["position_offset"], 5);
}

#[test]
fn saved_linear_example_matches_hand_calculation() {
    let fixture = Fixture::tiny();
    assert_eq!(fixture.shape("manual.weight"), [2, 3]);
    assert_eq!(
        fixture.f32("manual.weight"),
        [1.0, 2.0, -1.0, 0.5, -2.0, 3.0]
    );
    assert_eq!(
        fixture.f32("manual.inputs"),
        [2.0, -1.0, 4.0, -1.0, 0.0, 2.0]
    );
    assert_eq!(fixture.f32("manual.bias"), [0.25, -0.5]);
    assert_eq!(fixture.f32("manual.output"), [-3.75, 14.5, -2.75, 5.0]);
}

#[test]
fn causal_fixture_preserves_logits_before_the_changed_future_token() {
    let fixture = Fixture::tiny();
    assert_eq!(fixture.u32("cases.prompt.tokens"), [1, 7, 13, 3]);
    assert_eq!(fixture.u32("cases.future_changed.tokens"), [1, 7, 13, 17]);
    let original = fixture.f32("cases.prompt.logits");
    let changed = fixture.f32("cases.future_changed.logits");
    assert_eq!(&original[..3 * 64], &changed[..3 * 64]);
    assert_ne!(&original[3 * 64..], &changed[3 * 64..]);
    assert_eq!(fixture.u32("greedy.tokens").len(), 4);
}
