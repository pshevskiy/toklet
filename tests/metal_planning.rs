//! Explicit device resource witnesses; ordinary CPU tests never create a GPU.
#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
use toklet::{
    metal::{self, AttentionMode, NormalizationMode, ProjectionMode, ReadbackMode},
    session, weights,
};

#[test]
#[ignore = "Explicit Apple GPU: compare padded resource plans with actual owners"]
fn mlx_resource_plans_match_allocated_owners() {
    let frozen: serde_json::Value =
        serde_json::from_slice(include_bytes!("fixtures/metal-resource-v1/baseline.json")).unwrap();
    let mut witnessed = 0;
    for directory in [
        "mlx-qwen-v1/model",
        "mlx-llama-v1/tied",
        "mlx-llama-v1/untied",
        "mlx-mistral-v1/untied",
        "mlx-qwen3-v1/tied",
        "mlx-qwen3-v1/untied",
    ] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(directory);
        let load = weights::LoadOptions {
            context_limit: 32,
            memory_limit_bytes: 256 * 1024 * 1024,
        };
        let model = metal::Model::load_mlx(&path, load).unwrap();
        for chunk in [1, 3, 32] {
            for readback in [ReadbackMode::FullLogits, ReadbackMode::GreedyToken] {
                for attention in [AttentionMode::Reference, AttentionMode::Online] {
                    if attention == AttentionMode::Online && !directory.starts_with("mlx-qwen-v1/")
                    {
                        continue;
                    }
                    let options = session::Options {
                        capacity: 32,
                        prefill_chunk_size: chunk,
                        position_offset: 0,
                        memory_limit_bytes: load.memory_limit_bytes,
                    };
                    let actual = metal::Session::with_attention_mode(
                        &model,
                        options,
                        ProjectionMode::MlxAffine,
                        NormalizationMode::Compensated,
                        readback,
                        attention,
                    )
                    .unwrap();
                    let plan = metal::planning::inspect_mlx(
                        &path,
                        load,
                        options,
                        NormalizationMode::Compensated,
                        readback,
                        attention,
                    )
                    .unwrap();
                    assert_eq!(plan.session_memory, actual.memory());
                    assert_eq!(plan.resident_weights_bytes, model.weights_bytes());
                    assert!(
                        plan.fits_requested_budgets
                            && !plan.numerical_buffers_allocated
                            && plan.payload_validation_required
                            && !plan.qualification_evaluated
                    );
                    let case = &frozen["cases"][witnessed];
                    assert_eq!(case["directory"], directory);
                    assert_eq!(case["chunk"], chunk);
                    assert_eq!(case["readback"], serde_json::to_value(readback).unwrap());
                    assert_eq!(case["attention"], serde_json::to_value(attention).unwrap());
                    assert_eq!(
                        case["memory"],
                        serde_json::to_value(plan.session_memory).unwrap()
                    );
                    witnessed += 1;
                    println!(
                        "MLX_RESOURCE_PLAN {}",
                        serde_json::to_string(&plan).unwrap()
                    );
                }
            }
        }
    }
    assert_eq!(witnessed, 42);
}

#[test]
#[ignore = "Explicit Apple GPU: runtime reports distinct source/load/session budgets"]
fn mlx_resource_budget_and_offset_boundaries() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mlx-qwen-v1/model");
    let load = weights::LoadOptions {
        context_limit: 32,
        memory_limit_bytes: 256 * 1024 * 1024,
    };
    let options = session::Options {
        capacity: 31,
        prefill_chunk_size: 3,
        position_offset: 1,
        memory_limit_bytes: load.memory_limit_bytes,
    };
    let probe = |load, options| {
        metal::planning::inspect_mlx(
            &path,
            load,
            options,
            NormalizationMode::Original,
            ReadbackMode::FullLogits,
            AttentionMode::Reference,
        )
        .unwrap()
    };
    let baseline = probe(load, options);
    let exact_load = baseline
        .source_admitted_load_bytes
        .max(baseline.resident_admitted_load_bytes);
    let exact_session = baseline.session_memory.admitted_bytes;
    let exact = probe(
        weights::LoadOptions {
            memory_limit_bytes: exact_load,
            ..load
        },
        session::Options {
            memory_limit_bytes: exact_session,
            ..options
        },
    );
    assert!(exact.fits_requested_budgets);
    let below_load = probe(
        weights::LoadOptions {
            memory_limit_bytes: exact_load - 1,
            ..load
        },
        options,
    );
    assert!(!below_load.fits_requested_budgets);
    assert_eq!(
        below_load.fits_source_load_budget,
        baseline.source_admitted_load_bytes < exact_load
    );
    assert_eq!(
        below_load.fits_resident_load_budget,
        baseline.resident_admitted_load_bytes < exact_load
    );
    let below_session = probe(
        load,
        session::Options {
            memory_limit_bytes: exact_session - 1,
            ..options
        },
    );
    assert!(!below_session.fits_session_budget && !below_session.fits_requested_budgets);
    assert_eq!(baseline.session_memory, below_session.session_memory);
    let zero_session = probe(
        load,
        session::Options {
            memory_limit_bytes: 0,
            ..options
        },
    );
    assert!(!zero_session.fits_session_budget);
    for options in [
        session::Options {
            capacity: 32,
            ..options
        },
        session::Options {
            capacity: 0,
            ..options
        },
        session::Options {
            prefill_chunk_size: 0,
            ..options
        },
        session::Options {
            prefill_chunk_size: 32,
            ..options
        },
        session::Options {
            position_offset: usize::MAX,
            ..options
        },
        session::Options {
            memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES + 1,
            ..options
        },
    ] {
        assert!(matches!(
            metal::planning::inspect_mlx(
                &path,
                load,
                options,
                NormalizationMode::Original,
                ReadbackMode::FullLogits,
                AttentionMode::Reference
            ),
            Err(metal::Error::Options(_))
        ));
    }
    let model = metal::Model::load_mlx(&path, load).unwrap();
    let create = |limit| {
        metal::Session::with_attention_mode(
            &model,
            session::Options {
                memory_limit_bytes: limit,
                ..options
            },
            ProjectionMode::MlxAffine,
            NormalizationMode::Original,
            ReadbackMode::FullLogits,
            AttentionMode::Reference,
        )
    };
    assert_eq!(
        create(exact_session).unwrap().memory(),
        baseline.session_memory
    );
    assert!(
        matches!(create(exact_session-1),Err(metal::Error::Budget{required,limit}) if required==exact_session && limit==exact_session-1)
    );
    assert!(matches!(
        metal::Model::load_mlx(
            &path,
            weights::LoadOptions {
                memory_limit_bytes: exact_load - 1,
                ..load
            }
        ),
        Err(metal::Error::Budget { .. })
            | Err(metal::Error::Weights(weights::Error::Budget { .. }))
    ));
    assert_eq!(
        metal::Model::load_mlx(
            &path,
            weights::LoadOptions {
                memory_limit_bytes: exact_load,
                ..load
            }
        )
        .unwrap()
        .weights_bytes(),
        baseline.resident_weights_bytes
    );
    println!(
        "MLX_RESOURCE_BOUNDARIES {}",
        serde_json::json!({"load_minimum":exact_load,"session_minimum":exact_session,"offset":1,"exact_and_below_passed":true})
    );
}
