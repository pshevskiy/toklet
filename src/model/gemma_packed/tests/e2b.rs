//! Offline admission/accounting only. No model download or large E2B allocation.
use super::*;
use crate::mlx::gemma4::payload::LoadReport;
use crate::model::gemma_reference::tests::Measurement;

#[test]
fn actual_backing_capacity_replaces_borrowed_tensor_sum_once() {
    let data = fixture::Fixture::new();
    let plain = Model::new(data.weights()).unwrap();
    // Compute the independent frozen byte-backed tensor inventory, excluding
    // source scalar vectors copied into LayerWeights rather than borrowed.
    let packed = data.data["packed_original_bf16"].as_object().unwrap();
    let mut borrowed = 0;
    for item in packed.values() {
        for key in ["weight", "scales", "biases"] {
            borrowed += item[key]["bytes"].as_u64().unwrap() as usize;
        }
    }
    for (name, item) in data.data["weights"].as_object().unwrap() {
        if !packed.contains_key(name) && !name.ends_with(".layer_scalar") {
            borrowed += item["bytes"].as_u64().unwrap() as usize;
        }
    }
    let source_scalars = 4 * LAYERS;
    let excess = 1024;
    let mut report = LoadReport {
        container_bytes: 0,
        container_sha256: [0; 32],
        text_source_bytes: 0,
        excluded_source_bytes: 0,
        packed_matrices: packed.len(),
        fp32_vectors: 0,
        tensor_capacity_bytes: borrowed + source_scalars + excess,
        metadata_reservation_bytes: 64 * 1024 * 1024,
        // Load-only stream/conversion overlap must never be counted here.
        admitted_load_bytes: usize::MAX,
    };
    let model = Model::with_storage(data.weights(), Layout::witness(), Some(&report)).unwrap();
    assert_eq!(
        model.model_bytes(),
        plain.model_bytes() + source_scalars + excess + report.metadata_reservation_bytes
    );
    let budget = model.model_bytes() + model.planned_session_bytes().unwrap();
    let measured = Measurement::start();
    let rejected = model.new_session(budget - 1).is_err();
    assert_eq!(measured.finish(), 0);
    assert!(rejected);
    let mut session = model.new_session(budget).unwrap();
    both(
        &data,
        "logits",
        session.forward(&data.ids).unwrap(),
        0,
        MAX_POSITIONS,
    );
    assert_eq!(
        session.allocated_bytes().unwrap(),
        model.planned_session_bytes().unwrap()
    );
    for invalid in ["capacity", "overflow"] {
        if invalid == "capacity" {
            report.tensor_capacity_bytes = borrowed - 1;
            report.metadata_reservation_bytes = 0;
        } else {
            report.tensor_capacity_bytes = borrowed;
            report.metadata_reservation_bytes = usize::MAX;
        }
        let weights = data.weights();
        let layout = Layout::witness();
        let measured = Measurement::start();
        let rejected = Model::with_storage(weights, layout, Some(&report)).is_err();
        assert_eq!(measured.finish(), 0);
        assert!(rejected);
    }
}
