//! Tiny consuming two-table storage proof; no real-source/GPU graph eligibility.
use super::*;
use crate::metal::{
    Error as MetalError,
    gemma_affine::{CONTEXT_ALLOWANCE_BYTES, Context},
};

const LIMIT: usize = 6 * 1024 * 1024 * 1024;

#[test]
#[ignore = "explicit tiny9B-shaped source-store consuming GPU ownership and cleanup"]
fn consuming_distinct_tables_original_rows_budget_and_failure_cleanup() {
    let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let policy = CONTEXT_ALLOWANCE_BYTES + 64 * 1024 * 1024;
    let required = tiny().report().retained_tensor_capacity_bytes + policy;
    assert!(matches!(tiny().into_metal(&context, required - 1),
        Err(MetalError::Budget { required: r, limit }) if r == required && limit == required - 1));
    let store = tiny();
    let identity = store.identity();
    let vectors = store.vector(NORM).unwrap().to_vec();
    let convolution = store.convolution(CONV).unwrap().to_vec();
    let gpu = store.into_metal(&context, LIMIT).unwrap();
    assert_eq!(
        gpu.identity().container_sha256(),
        identity.container_sha256()
    );
    assert_eq!(gpu.plan().output_head().name, HEAD);
    assert_eq!(gpu.vector(NORM).unwrap(), vectors);
    assert_eq!(gpu.convolution(CONV).unwrap(), convolution);
    let report = gpu.report();
    assert_eq!(report.matrices, 2);
    assert_eq!(report.retained_cpu_matrix_bytes, 0);
    assert_eq!(
        report.retained_policy_bytes,
        report.retained_requested_bytes + policy
    );
    assert!(report.admitted_load_policy_bytes >= report.retained_policy_bytes);
    assert!(report.admitted_load_policy_bytes <= LIMIT);
    let (embedding_slot, _) = gpu.matrix(EMBED).unwrap();
    let (head_slot, _) = gpu.matrix(HEAD).unwrap();
    assert_ne!(embedding_slot, head_slot);
    let fixture = inventory();
    for name in [EMBED, HEAD] {
        let (_, weights) = gpu.matrix(name).unwrap();
        let rows = fixture["expected_rows"][name].as_array().unwrap();
        let mut gather = context.gather_session(64, 1, LIMIT).unwrap();
        for (row, expected) in rows.iter().enumerate() {
            let mut actual = [0f32; 64];
            gather.apply_weights(weights, &[row], &mut actual).unwrap();
            for (value, expected) in actual.into_iter().zip(expected.as_array().unwrap()) {
                assert_eq!(
                    value.to_bits(),
                    (expected.as_f64().unwrap() as f32).to_bits()
                );
            }
        }
    }
    // HEAD sorts first; malformed EMBED metadata fails after a GPU owner exists.
    // Error drops unpublished GPU/source owners. The same Context remains usable.
    drop(gpu);
    let mut bad = tiny();
    bad.components.matrices.get_mut(EMBED).unwrap().scales[..2]
        .copy_from_slice(&0x7f80u16.to_le_bytes());
    assert!(bad.into_metal(&context, LIMIT).is_err());
    let recovered = tiny().into_metal(&context, LIMIT).unwrap();
    assert_eq!(recovered.report().matrices, 2);
    assert_eq!(recovered.report().retained_cpu_matrix_bytes, 0);
    println!(
        "\nQWEN35_9B_TINY_UPLOAD {}",
        json!({"passed":true,"matrices":2,"separate_slots":true,
        "rows_bit_exact":true,"budget_minus_one_rejected":true,"failure_cleanup_context_reusable":true,
        "retained_cpu_matrix_bytes":0,"real_source_qualified":false})
    );
}
