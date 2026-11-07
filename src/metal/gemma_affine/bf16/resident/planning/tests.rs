use super::*;

fn source() -> TextPlan {
    crate::mlx::gemma4::bf16::inspect(
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/gemma-unquantized-header-v1/config.json"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/gemma-unquantized-header-v1/header.json"
        )),
        crate::mlx::gemma4::bf16::CONTAINER_BYTES,
        crate::mlx::gemma4::ModalityPolicy::TextOnly,
    )
    .unwrap()
}

#[test]
fn concrete_bindings_match_independent_metadata_ids_shapes_and_tied_head() {
    let source = source();
    let logical = execution::inspect(&source, 32).unwrap();
    let bindings = Bindings::bf16(&source).unwrap();
    let mut commands = 0;
    for operation in logical.operations {
        assert_eq!(
            bindings.operation_source(operation.id).unwrap(),
            operation.source_matrix_slot
        );
        let mut expected = operation.id;
        if expected.role == Role::OutputHead {
            expected.role = Role::TokenEmbedding;
        }
        assert_eq!(
            bindings
                .operation_identity(operation.source_matrix_slot)
                .unwrap(),
            expected
        );
        let tiles = if operation.execution == Execution::TiledBf16MetalProjection {
            operation.source_rows.div_ceil(TILE_ROWS)
        } else {
            0
        };
        let selected = selected(operation, tiles).unwrap();
        commands += selected.predicted_commands_per_chunk;
        if selected.logical.id.role == Role::OutputHead {
            assert_eq!(selected.projection_tiles, 32);
            assert_eq!(selected.modelled_prefill_output_host_bytes, 1_048_576);
            assert_eq!(
                selected.logical.source_matrix_slot,
                bindings.embedding_index()
            );
        }
        if selected.logical.id.role == Role::PerLayerEmbedding {
            assert_eq!(selected.projection_tiles, 0);
            assert_eq!(selected.kernel_entry, None);
        }
    }
    assert!(commands > 277);
    for id in [
        execution::OperationId {
            role: Role::Key,
            layer: Some(15),
        },
        execution::OperationId {
            role: Role::OutputHead,
            layer: Some(0),
        },
    ] {
        assert!(bindings.operation_source(id).is_err());
    }
}

#[test]
fn cold_layout_counts_include_exact_independent_fp32_kv_and_stages() {
    // Twelve256-dim and three512-dim producer owners, separate K/V, FP32.
    let dimensions = 12 * 256 + 3 * 512;
    for (capacity, chunk) in [(128, 16), (512, 32)] {
        let (model, session) = gemma_packed::official_bf16_planned_layout::<
            graph::MatrixRef<'_>,
            graph::Backend<'_, '_>,
        >(512, capacity, chunk)
        .unwrap();
        let kv = gemma_packed::official_bf16_planned_kv(capacity, chunk).unwrap();
        assert_eq!(kv, (capacity + chunk) * dimensions * 2 * 4);
        assert!(session > kv && model > 0);
    }
    for (load, capacity, chunk) in [(513, 512, 32), (512, 513, 32), (512, 0, 32), (512, 512, 0)] {
        assert!(
            gemma_packed::official_bf16_planned_layout::<
                graph::MatrixRef<'_>,
                graph::Backend<'_, '_>,
            >(load, capacity, chunk)
            .is_err()
        );
    }
}

#[test]
fn impossible_options_reject_before_io_or_device_initialization() {
    let load = weights::LoadOptions {
        context_limit: 512,
        memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES,
    };
    let options = session::Options {
        capacity: 512,
        prefill_chunk_size: 32,
        position_offset: 0,
        memory_limit_bytes: load.memory_limit_bytes,
    };
    for options in [
        session::Options {
            capacity: 513,
            ..options
        },
        session::Options {
            prefill_chunk_size: 33,
            ..options
        },
        session::Options {
            position_offset: 1,
            ..options
        },
        session::Options {
            memory_limit_bytes: 0,
            ..options
        },
    ] {
        assert!(matches!(
            inspect_official_bf16(Path::new("missing"), load, options),
            Err(crate::metal::gemma_affine::resident::hybrid::Error::Metal(
                Error::Options(_)
            ))
        ));
    }
}

#[test]
#[ignore = "Explicit official BF16 Mac cold-plan/live-owner/session reconciliation"]
fn official_runtime_inventory_and_minimum_admission_match_live_owners() {
    let path = std::path::PathBuf::from(std::env::var_os("TOKLET_GEMMA_BF16_MODEL").unwrap());
    let budget = weights::INITIAL_MEMORY_LIMIT_BYTES;
    let load = weights::LoadOptions {
        context_limit: 512,
        memory_limit_bytes: budget,
    };
    let options = session::Options {
        capacity: 512,
        prefill_chunk_size: 32,
        position_offset: 0,
        memory_limit_bytes: budget,
    };
    let cold = inspect_official_bf16(&path, load, options).unwrap();
    assert!(cold.minimum_fits_load_budget && cold.minimum_fits_session_budget);
    assert!(
        !cold.numerical_buffers_allocated
            && cold.payload_validation_required
            && !cold.qualification_evaluated
    );
    let owner = OfficialBf16::load(&path, 512, budget).unwrap();
    let live = owner.runtime_inventory(32).unwrap();
    assert_eq!(cold.operations, live.operations);
    assert_eq!(
        (
            cold.resident_matrix_owners,
            cold.resident_gpu_tile_owners,
            cold.vector_owners
        ),
        (
            live.resident_matrix_owners,
            live.resident_gpu_tile_owners,
            live.vector_owners
        )
    );
    assert_eq!(
        (live.resident_matrix_owners, live.vector_owners),
        (277, 262)
    );
    assert!(owner.memory().load_admitted_bytes >= cold.minimum_memory.owner_load_admitted_bytes);
    let model = owner.model().unwrap();
    let inference = owner.new_session(&model, options).unwrap();
    let actual = inference.memory().admitted_bytes + owner.memory().session_reserved_bytes;
    assert!(actual >= cold.minimum_memory.session_admitted_bytes && actual <= budget);
    let mut footprint = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: flavor4 matches the libc/Apple ABI of the owned writable stack
    // object. No alias is used during the call; status0 permits reading it.
    let status = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            footprint.as_mut_ptr().cast(),
        )
    };
    assert_eq!(status, 0, "process physical footprint unavailable");
    // SAFETY: zero initialization plus successful flavor4 copy initializes all fields.
    let footprint = unsafe { footprint.assume_init() };
    assert!(
        footprint.ri_phys_footprint > 0
            && footprint.ri_phys_footprint <= footprint.ri_lifetime_max_phys_footprint
    );
    assert!(footprint.ri_lifetime_max_phys_footprint <= weights::INITIAL_MEMORY_LIMIT_BYTES as u64);
    println!(
        "OFFICIAL_BF16_RUNTIME_PLAN {}",
        serde_json::json!({"operations":cold.operations.len(),
        "resident_matrix_owners":live.resident_matrix_owners,"gpu_tile_owners":live.resident_gpu_tile_owners,
        "minimum_session_admitted_bytes":cold.minimum_memory.session_admitted_bytes,"actual_session_admitted_bytes":actual,
        "capacity_overreservation_bytes":actual-cold.minimum_memory.session_admitted_bytes,
        "process_physical_footprint_bytes":footprint.ri_phys_footprint,
        "process_lifetime_peak_physical_footprint_bytes":footprint.ri_lifetime_max_phys_footprint,
        "memory_scope":"whole Rust test process, including cold inspection/load/session; not isolated load peak","passed":true})
    );
}
