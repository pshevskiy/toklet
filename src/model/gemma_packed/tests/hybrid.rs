//! Explicit tiny GPU graph check against the existing frozen numerical contracts.
use super::*;
use crate::metal::gemma_affine::{self, resident::hybrid};
use hybrid::test_support::Uploaded;
use objc2::rc::autoreleasepool;
mod capture;
mod controls;
mod real;

#[test]
#[ignore = "explicit complete tiny hybrid Gemma graph qualification"]
fn frozen_tiny_gpu_products_and_gathers_share_cpu_graph_and_cache_transactions() {
    let mut capture = capture::Capture::new();
    let data = fixture::Fixture::new();
    let uploaded = autoreleasepool(|_| {
        let source_context =
            gemma_affine::Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
        Uploaded::new(&source_context, data.weights())
    });
    let before = uploaded.snapshot();
    let norms = uploaded.norms();
    assert!(norms.source_identity().is_none());
    assert!(norms.model().is_err()); // Synthetic data cannot enter the public E2B constructor.
    let model = Model::with_storage(
        uploaded.graph_weights(&norms),
        Layout::witness(),
        Some(uploaded.backing(&norms)),
    )
    .unwrap();
    let context = gemma_affine::Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
    frozen_graph_checks(
        &data,
        || {
            norms
                .new_session(&model, &context, 512 * 1024 * 1024)
                .unwrap()
        },
        |name, values, before, count| capture.record(name, values, before, count),
    );
    assert_eq!(uploaded.snapshot(), before);
    capture.finish();
    println!(
        "GEMMA_HYBRID_TINY_COMPLETE frozen_graph=true scalar_order_unchanged=true full_e2b_qualified=false"
    );
}
