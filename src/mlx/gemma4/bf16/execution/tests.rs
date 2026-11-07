use super::*;
use std::collections::BTreeSet;

fn source() -> TextPlan {
    super::super::inspect(
        include_bytes!("../../../../../tests/fixtures/gemma-unquantized-header-v1/config.json"),
        include_bytes!("../../../../../tests/fixtures/gemma-unquantized-header-v1/header.json"),
        super::super::CONTAINER_BYTES,
        crate::mlx::gemma4::ModalityPolicy::TextOnly,
    )
    .unwrap()
}

#[test]
fn logical_inventory_matches_independent_source_shapes_and_unique_identities() {
    let plan = inspect(&source(), 32).unwrap();
    let layout: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../../../tests/fixtures/gemma-unquantized-header-v1/layout.json"
    ))
    .unwrap();
    assert_eq!(
        (plan.layer_count, plan.producer_count, plan.operations.len()),
        (35, 15, 279)
    );
    let mut identities = BTreeSet::new();
    let mut owners = BTreeSet::new();
    for operation in &plan.operations {
        assert!(identities.insert(operation.id));
        owners.insert(operation.source_matrix_slot);
        let reference = &layout["selected_text"][&operation.source_tensor];
        assert_eq!(
            serde_json::json!([operation.source_rows, operation.source_columns]),
            reference["shape"]
        );
        assert_eq!(
            serde_json::json!(operation.source_bytes),
            reference["bytes"]
        );
    }
    assert_eq!(owners.len(), 278);
    assert_eq!(
        plan.operations[0].source_matrix_slot,
        plan.operations.last().unwrap().source_matrix_slot
    );
    assert_eq!(
        plan.operations[0].source_tensor,
        plan.operations.last().unwrap().source_tensor
    );
    assert!(!plan.payload_authenticated && !plan.numerical_buffers_allocated);
}

#[test]
fn producer_consumer_g2_rows_and_kv_roles_match_literal_schedule() {
    let plan = inspect(&source(), 32).unwrap();
    let find = |role, layer| {
        plan.operations
            .iter()
            .find(|op| op.id == OperationId { role, layer })
            .unwrap()
    };
    assert_eq!(find(Role::PerLayerInputProjection, None).prefill_rows, 32);
    assert_eq!(find(Role::Query, Some(14)).prefill_rows, 32);
    assert_eq!(find(Role::Query, Some(15)).prefill_rows, 1);
    assert_eq!(find(Role::OutputHead, None).prefill_rows, 1);
    assert!(plan.operations.iter().all(|op| op.decode_rows == 1));
    let layer0: Vec<_> = plan
        .operations
        .iter()
        .filter(|op| op.id.layer == Some(0))
        .map(|op| op.id.role)
        .collect();
    assert_eq!(
        layer0,
        [
            Role::Query,
            Role::Key,
            Role::Value,
            Role::AttentionOutput,
            Role::FeedForwardGate,
            Role::FeedForwardUp,
            Role::FeedForwardDown,
            Role::PerLayerGate,
            Role::PerLayerProjection
        ]
    );
    assert!(
        !plan
            .operations
            .iter()
            .any(|op| op.id.layer == Some(34) && matches!(op.id.role, Role::Key | Role::Value))
    );
    assert!(
        inspect(&source(), 1)
            .unwrap()
            .operations
            .iter()
            .all(|op| op.prefill_rows == 1)
    );
}

#[test]
fn rejects_invalid_chunks_and_role_scopes() {
    for rows in [0, 33, usize::MAX] {
        assert!(inspect(&source(), rows).is_err());
    }
    for id in [
        OperationId {
            role: Role::Query,
            layer: None,
        },
        OperationId {
            role: Role::OutputHead,
            layer: Some(0),
        },
        OperationId {
            role: Role::Key,
            layer: Some(35),
        },
    ] {
        assert!(id.tensor_name().is_err());
    }
}
