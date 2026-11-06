use super::*;
use sha2::Digest;
use std::io::Cursor;

mod gpu;
mod real;

fn real_metadata() -> (source::Prepared, Cursor<Vec<u8>>, LoadOptions) {
    let config = include_bytes!("../../../../tests/fixtures/gemma4-e2b-metadata-v1/config.json");
    let index = include_bytes!("../../../../tests/fixtures/gemma4-e2b-metadata-v1/index.json");
    let header = include_bytes!("../../../../tests/fixtures/gemma4-e2b-metadata-v1/header.json");
    let mut raw = (header.len() as u64).to_le_bytes().to_vec();
    raw.extend_from_slice(header);
    let mut reader = Cursor::new(raw);
    let options = LoadOptions {
        container_bytes: 3550670554,
        container_sha256: [0; 32],
        available_bytes: 6 * 1024 * 1024 * 1024,
        modality: ModalityPolicy::TextOnly,
    };
    let prepared =
        source::Prepared::read(config, index, &mut reader, &options, SourceKind::Affine4).unwrap();
    (prepared, reader, options)
}
#[test]
fn full_plan_matches_frozen_inventory_and_counts_new_owner_abi_without_allocating_payloads() {
    let raw = include_bytes!(
        "../../../../tests/fixtures/gemma-resident-storage-reference-v1/inventory.json"
    );
    assert_eq!(
        format!("{:x}", sha2::Sha256::digest(raw)),
        "a3dff3b9d825fcf4fa6f391262732c5f429345682d683d6a39bc551ddacd68e2"
    );
    let reference: serde_json::Value = serde_json::from_slice(raw).unwrap();
    let (prepared, _, _) = real_metadata();
    let plan = Plan::new(&prepared.inputs, 278, 262).unwrap();
    assert_eq!(size_of::<Weights>(), 112);
    assert_eq!(size_of::<ResidentVector>(), 24);
    let known = reference["accounting"]["known_matrix_resident_requested_bytes"]
        .as_u64()
        .unwrap() as usize;
    let vector = reference["accounting"]["vector_guarded_buffer_bytes"]
        .as_u64()
        .unwrap() as usize;
    assert_eq!(
        plan.base + plan.payload,
        known + vector + 262 * 24 + size_of::<E2bWeights>()
    );
    assert_eq!(plan.source, 82575432);
    assert_eq!(plan.maximum_vector_raw, 3072);
    let required = plan.working(0, 0, plan.source).unwrap();
    admit(required, required).unwrap();
    assert!(
        matches!(admit(required,required-1),Err(Error::Budget{required:r,limit})if r==required&&limit==required-1)
    );
    plan.validate_device(&prepared.inputs, usize::MAX).unwrap();
    assert!(plan.validate_device(&prepared.inputs, 1024).is_err());
    println!(
        "GEMMA_RESIDENT_PLAN bundle={} vector={} resident={} source={} policy={}",
        size_of::<E2bWeights>(),
        size_of::<ResidentVector>(),
        plan.base + plan.payload,
        plan.source,
        required
    );
}
#[test]
fn aggregate_plan_replaces_minima_with_actual_slots_and_prior_payloads_once() {
    let (prepared, _, _) = real_metadata();
    let plan = Plan::new(&prepared.inputs, 279, 264).unwrap();
    let plain = Plan::new(&prepared.inputs, 278, 262).unwrap();
    assert_eq!(plan.base - plain.base, 112 + 2 * 24);
    let completed = plan.matrix_payloads[0] + plan.matrix_payloads[1];
    let actual = completed + 2 * 112; // Two extra actual chunk descriptor slots.
    assert_eq!(
        plan.working(actual, completed, plan.source + 3).unwrap(),
        plan.working(0, 0, plan.source).unwrap() + 2 * 112 + 3
    );
    assert!(plan.working(0, plan.payload + 1, 0).is_err());
    assert!(Plan::new(&prepared.inputs, 277, 262).is_err());
    assert!(Plan::new(&prepared.inputs, 278, 261).is_err());
    assert!(Plan::new(&prepared.inputs, usize::MAX, 262).is_err());
    assert!(plan.working(usize::MAX, 0, 0).is_err());
}
#[test]
fn named_bindings_cover_every_matrix_vector_and_shared_geometry_once() {
    let (prepared, _, _) = real_metadata();
    let bindings = Bindings::new(&prepared.cold).unwrap();
    for index in 0..35 {
        assert_eq!(
            bindings.layer_geometry(index),
            Some(prepared.cold.layers()[index])
        );
    }
    assert!(bindings.layer_geometry(35).is_none());
    assert_eq!(prepared.inputs.matrices.len(), 278);
    assert_eq!(prepared.inputs.vectors.len(), 262);
}
