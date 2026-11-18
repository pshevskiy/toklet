use toklet::model::sensenova_loop::{Attention, Plan, ResidualStreams};

const CONFIG: &[u8] = include_bytes!("fixtures/sensenova_loop/official-awq-config.json");
const MLX: &[u8] = include_bytes!("fixtures/sensenova_loop/mlx-config.json");

#[test]
fn official_awq_visits_share_weights_and_own_distinct_states() {
    let plan = Plan::from_official_awq_config(CONFIG).unwrap();
    let visits = plan.visits();
    let physical: Vec<_> = visits.iter().map(|visit| visit.physical_layer).collect();
    assert_eq!(
        physical,
        (0..4)
            .chain(4..28)
            .chain(4..28)
            .chain(28..32)
            .collect::<Vec<_>>()
    );
    assert!(
        visits
            .iter()
            .enumerate()
            .all(|(slot, visit)| visit.state_slot == slot)
    );
    assert_eq!(
        visits
            .iter()
            .filter(|visit| visit.attention == Attention::FullAttention)
            .count(),
        14
    );
    for slot in 0..24 {
        let a = visits[4 + slot];
        let b = visits[28 + slot];
        assert_eq!(a.physical_layer, b.physical_layer);
        assert_eq!(a.attention, b.attention);
        assert_eq!(a.residual_streams, b.residual_streams);
        assert_ne!(a.state_slot, b.state_slot);
        assert_eq!(
            a.residual_streams,
            match slot {
                0 => ResidualStreams::Expand,
                23 => ResidualStreams::Reduce,
                _ => ResidualStreams::Carry,
            }
        );
    }
    assert!(
        visits[..4]
            .iter()
            .chain(&visits[52..])
            .all(|visit| visit.residual_streams == ResidualStreams::Plain)
    );
    assert_eq!(plan.text_shape().max_position_embeddings, 262144);
    assert_eq!(plan.yarn().mrope_section, [11, 11, 10]);
}

#[test]
fn altered_execution_and_position_fields_are_rejected() {
    let original: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    for (field, wrong, reason) in [
        (
            "/loop_settings/eval_count",
            serde_json::json!(1),
            "SenseNova fixed decoder loop",
        ),
        (
            "/loop_settings/use_gate",
            serde_json::json!(true),
            "SenseNova fixed decoder loop",
        ),
        (
            "/mhc/iterations",
            serde_json::json!(19),
            "SenseNova two-stream mHC routing",
        ),
        (
            "/mhc/start",
            serde_json::json!(5),
            "SenseNova two-stream mHC routing",
        ),
        (
            "/text_config/layer_types/0",
            serde_json::json!("full_attention"),
            "SenseNova physical attention types",
        ),
        (
            "/text_config/mlp_only_layers",
            serde_json::json!([0]),
            "SenseNova official AWQ configuration JSON",
        ),
        (
            "/text_config/mamba_ssm_dtype",
            serde_json::json!("bfloat16"),
            "SenseNova text backbone geometry",
        ),
        (
            "/text_config/rope_parameters/rope_type",
            serde_json::json!("default"),
            "SenseNova text YaRN positions",
        ),
        (
            "/text_config/rope_parameters/factor",
            serde_json::json!(3.0),
            "SenseNova text YaRN positions",
        ),
        (
            "/text_config/max_position_embeddings",
            serde_json::json!(524288),
            "SenseNova text backbone geometry",
        ),
    ] {
        let mut altered = original.clone();
        *altered.pointer_mut(field).unwrap() = wrong;
        let result = Plan::from_official_awq_config(&serde_json::to_vec(&altered).unwrap());
        assert!(
            matches!(result, Err(toklet::model::Error::InvalidConfig(actual)) if actual == reason),
            "{field} must reach the intended rejection: {reason}"
        );
    }
}

#[test]
fn mlx_geometry_keeps_published_context_and_cache_settings() {
    let awq = Plan::from_official_awq_config(CONFIG).unwrap();
    let mlx = Plan::from_mlx_config(MLX).unwrap();
    assert_eq!(mlx.visits(), awq.visits());
    assert_eq!(mlx.yarn(), awq.yarn());
    assert_eq!(mlx.text_shape().max_position_embeddings, 524288);
    assert!(mlx.default_use_cache());
    assert!(!awq.default_use_cache());

    let original: serde_json::Value = serde_json::from_slice(MLX).unwrap();
    for (field, wrong, reason) in [
        (
            "/loop_settings/use_gate",
            serde_json::json!(true),
            "SenseNova fixed decoder loop",
        ),
        (
            "/text_config/max_position_embeddings",
            serde_json::json!(262144),
            "SenseNova text backbone geometry",
        ),
        (
            "/text_config/use_cache",
            serde_json::json!(false),
            "SenseNova text cache default",
        ),
        (
            "/use_cache",
            serde_json::json!(false),
            "SenseNova text cache default",
        ),
    ] {
        let mut altered = original.clone();
        if field == "/loop_settings/use_gate" {
            altered["loop_settings"]["use_gate"] = wrong;
        } else {
            *altered.pointer_mut(field).unwrap() = wrong;
        }
        let result = Plan::from_mlx_config(&serde_json::to_vec(&altered).unwrap());
        assert!(
            matches!(result, Err(toklet::model::Error::InvalidConfig(actual)) if actual == reason),
            "{field} must reach the intended rejection: {reason}"
        );
    }
    assert!(matches!(
        Plan::from_mlx_config(CONFIG),
        Err(toklet::model::Error::InvalidConfig(
            "SenseNova MLX architecture"
        ))
    ));
}
