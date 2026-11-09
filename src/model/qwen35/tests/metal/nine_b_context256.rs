//! Pinned9B128/256 complete checkpoints, all-logit partitions and deep-prefix repair.
use super::*;
use crate::mlx::qwen35::{
    ModalityPolicy,
    nine_b::{self as source, payload},
};
use std::{
    fs::File,
    io::{Read, Write},
    mem::size_of,
    path::PathBuf,
    time::Instant,
};

const IDS: [u32; 17] = [1, 7, 13, 3, 9, 5, 11, 2, 17, 4, 6, 10, 19, 8, 21, 23, 29];
const DIAGNOSTIC: usize = 320 * 1024 * 1024;

struct Loaded {
    context: Context,
    store: payload::metal::MetalStore,
    cpu_load_ns: u128,
    upload_ns: u128,
    context_ns: u128,
}
fn directory() -> PathBuf {
    PathBuf::from(std::env::var_os("TOKLET_QWEN35_9B_MODEL_DIRECTORY").unwrap())
}
fn config() -> Vec<u8> {
    let file = File::open(directory().join("config.json")).unwrap();
    assert!(file.metadata().unwrap().is_file());
    assert_eq!(file.metadata().unwrap().len(), source::CONFIG_BYTES as u64);
    let mut bytes = Vec::new();
    file.take(source::CONFIG_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .unwrap();
    bytes
}
fn load_with_diagnostic(diagnostic: usize) -> Loaded {
    assert!(diagnostic > 0 && diagnostic < MEMORY_LIMIT - CONTEXT_ALLOWANCE_BYTES);
    let start = Instant::now();
    let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let context_ns = start.elapsed().as_nanos();
    let config = config();
    let start = Instant::now();
    let cpu = payload::load(
        &config,
        File::open(directory().join("model.safetensors")).unwrap(),
        payload::LoadOptions {
            memory_limit_bytes: MEMORY_LIMIT - diagnostic - CONTEXT_ALLOWANCE_BYTES,
            modality: ModalityPolicy::TextOnly,
        },
    )
    .unwrap();
    let cpu_load_ns = start.elapsed().as_nanos();
    let start = Instant::now();
    let store = cpu.into_metal(&context, MEMORY_LIMIT - diagnostic).unwrap();
    let upload_ns = start.elapsed().as_nanos();
    assert_eq!(store.report().matrices, 250);
    assert_eq!(store.report().retained_cpu_matrix_bytes, 0);
    let hex: String = store
        .identity()
        .container_sha256()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, source::CONTAINER_SHA256);
    Loaded {
        context,
        store,
        cpu_load_ns,
        upload_ns,
        context_ns,
    }
}
fn work(tokens: usize) -> Work {
    Work {
        outcome: Outcome::Success,
        tokens_completed: tokens,
        tokens_committed: tokens,
        projections_completed: tokens * 249,
        gathers_completed: tokens,
        convolution_steps_completed: tokens * 24,
        recurrence_steps_completed: tokens * 24,
        attention_heads_completed: tokens * 128,
    }
}
fn layout() -> Value {
    serde_json::json!({"model":size_of::<MetalModel<'_>>(),"layer":size_of::<Layer<'_,MetalMatrix<'_>>>(),
    "session":size_of::<MetalSession<'_,'_,'_>>(),"delta_state":size_of::<state::DeltaState>(),"kv":size_of::<state::Kv>(),
    "projection":size_of::<crate::metal::gemma_affine::Projection<'_>>(),"gather":size_of::<crate::metal::gemma_affine::Gather<'_>>(),
    "metal_store":size_of::<payload::metal::MetalStore>(),"gpu_weights":size_of::<GpuWeights>()})
}

const CHECKPOINTS: [usize; 2] = [128, 256];
const CAPTURE_VALUES: [usize; 2] = [18_773_504, 20_903_424];
const INPUTS: [u32; 256] = {
    let mut ids = [0; 256];
    let mut i = 0;
    while i < ids.len() {
        ids[i] = IDS[i % IDS.len()];
        i += 1;
    }
    ids
};
#[derive(serde::Serialize)]
struct Checkpoint {
    inputs: usize,
    snapshot: String,
    greedy_lowest_id: usize,
}
fn snapshot_is(session: &MetalSession<'_, '_, '_>, expected: &Checkpoint) {
    assert_eq!(session.history().len(), expected.inputs);
    assert_eq!(snapshot(session), expected.snapshot);
    assert_eq!(greedy(session.logits().unwrap()), expected.greedy_lowest_id);
}

#[test]
#[ignore = "explicit pinned9B128/256 independent checkpoints, all logits and prefix128 recovery"]
fn actual_source_context256() {
    println!();
    let output = PathBuf::from(std::env::var_os("TOKLET_QWEN35_9B_CONTEXT256_CAPTURE").unwrap());
    std::fs::create_dir(&output).unwrap();
    let loaded = load_with_diagnostic(DIAGNOSTIC);
    let bind = Instant::now();
    let model = MetalModel::from_nine_b_metal_store(&loaded.store).unwrap();
    let bind_ns = bind.elapsed().as_nanos();
    let options = Options {
        capacity: 512,
        chunk: 32,
        position_offset: 37,
        memory_limit_bytes: MEMORY_LIMIT - DIAGNOSTIC,
    };
    let planned = model.planned_session_memory(options).unwrap();
    assert!(planned.admitted_bytes + DIAGNOSTIC <= MEMORY_LIMIT);
    assert!(matches!(model.session(&loaded.context, Options {
        memory_limit_bytes: planned.admitted_bytes - 1, ..options
    }), Err(Error::MemoryBudget { required, .. }) if required == planned.admitted_bytes));
    let start = Instant::now();
    let mut session = model.session(&loaded.context, options).unwrap();
    let session_ns = start.elapsed().as_nanos();
    assert_eq!(session.memory(), planned);
    let memory = session.memory();
    let execution_bytes = session.execution.owned_bytes().unwrap();
    let mut body = std::io::BufWriter::with_capacity(
        64 * 1024,
        File::create(output.join("rows.f32le")).unwrap(),
    );
    // One external diagnostic owner holds every logit row. No second full output
    // vector or file is retained; ordinary session output remains chunk-bounded.
    let mut all = Vec::with_capacity(INPUTS.len().checked_mul(model.config.vocab).unwrap());
    let mut checkpoints = Vec::with_capacity(2);
    let diagnostic_requested_bytes = all.capacity().checked_mul(4).unwrap()
        + body.capacity()
        + checkpoints.capacity() * size_of::<Checkpoint>();
    assert!(diagnostic_requested_bytes < DIAGNOSTIC);
    let mut bytes = 0usize;
    let mut arrays = [0usize; 2];
    let mut values = [0usize; 2];
    let observed = Instant::now();
    for input in INPUTS.chunks(32) {
        let logits = session
            .forward_observed(input, &AtomicBool::new(false), &mut |o| {
                if o.stage == "before_commit" {
                    return Ok(());
                }
                let Some(slot) = CHECKPOINTS.iter().position(|&n| n == o.token + 1) else {
                    return Ok(());
                };
                let count = o.values.len();
                let size = count.checked_mul(4).unwrap();
                assert!(bytes.checked_add(size).unwrap() <= 158_707_712);
                let mut digest = Sha256::new();
                for value in o.values {
                    assert!(value.is_finite());
                    let raw = value.to_le_bytes();
                    body.write_all(&raw).unwrap();
                    digest.update(raw);
                }
                println!(
                    "QWEN35_9B_CONTEXT256_ARRAY {}",
                    serde_json::json!({
                        "checkpoint": CHECKPOINTS[slot], "token": o.token, "position": o.position,
                        "layer": o.layer, "stage": o.stage, "head": o.head, "count": count,
                        "offset": bytes, "bytes": size, "sha256": format!("{:x}", digest.finalize())
                    })
                );
                bytes += size;
                arrays[slot] += 1;
                values[slot] += count;
                Ok(())
            })
            .unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
        all.extend_from_slice(logits);
        assert_eq!(session.work(), work(input.len()));
        if CHECKPOINTS.contains(&session.history().len()) {
            checkpoints.push(Checkpoint {
                inputs: session.history().len(),
                snapshot: snapshot(&session),
                greedy_lowest_id: greedy(session.logits().unwrap()),
            });
        }
    }
    let observed_forward_ns = observed.elapsed().as_nanos();
    assert_eq!(arrays, [1051, 1051]);
    assert_eq!(values, CAPTURE_VALUES);
    assert_eq!(bytes, 158_707_712);
    assert_eq!(checkpoints.len(), 2);
    assert_eq!(all.len(), 256 * model.config.vocab);
    assert_eq!(all.capacity() * 4, 254_279_680);
    body.flush().unwrap();
    body.get_ref().sync_all().unwrap();
    drop(body);
    let validation = Instant::now();
    let mut partitions = Vec::with_capacity(3);
    for chunk in [32, 17, 1] {
        let measure = Measurement::start();
        session.reset();
        assert_eq!(measure.finish(), 0);
        let mut token = 0;
        let mut calls = Vec::with_capacity(256);
        let mut forward_ns = 0u128;
        for checkpoint in &checkpoints {
            while token < checkpoint.inputs {
                let count = chunk.min(checkpoint.inputs - token);
                let measure = Measurement::start();
                let start = Instant::now();
                let result = session.forward(&INPUTS[token..token + count]);
                forward_ns += start.elapsed().as_nanos();
                let allocations = measure.finish();
                assert_float_bits(
                    result.unwrap(),
                    &all[token * model.config.vocab..(token + count) * model.config.vocab],
                );
                assert_eq!(allocations, 0);
                assert_eq!(session.work(), work(count));
                calls.push(count);
                token += count;
            }
            snapshot_is(&session, checkpoint);
        }
        assert_eq!(session.history(), INPUTS);
        assert_eq!(session.memory(), memory);
        partitions.push(serde_json::json!({"chunk":chunk,"calls":calls,"full_logits_bits_exact":true,
            "checkpoint_state_bits_exact":true,"warmed_rust_allocations":0,"forward_ns":forward_ns}));
    }
    session.reset();
    for input in INPUTS[..128].chunks(32) {
        session.forward(input).unwrap();
    }
    snapshot_is(&session, &checkpoints[0]);
    for cache in &mut session.kv {
        for head in 0..model.config.kv_heads {
            let start = (head * 512 + 128) * model.config.head_dim;
            let end = (head + 1) * 512 * model.config.head_dim;
            cache.k[start..end].fill(f32::NAN);
            cache.v[start..end].fill(f32::NAN);
        }
    }
    let cancel = AtomicBool::new(false);
    assert!(
        session
            .forward_observed(&INPUTS[128..160], &cancel, &mut |o| {
                if o.token == 130 && o.layer == Some(0) && o.stage == "delta_output" {
                    return Err(Error::Observation(
                        "9B context256 partial-third-frame failure",
                    ));
                }
                Ok(())
            })
            .is_err()
    );
    let failed_work = session.work();
    assert_eq!(
        failed_work,
        Work {
            outcome: Outcome::Error,
            tokens_completed: 2,
            tokens_committed: 0,
            projections_completed: 502,
            gathers_completed: 3,
            convolution_steps_completed: 49,
            recurrence_steps_completed: 49,
            attention_heads_completed: 256
        }
    );
    snapshot_is(&session, &checkpoints[0]);
    assert!(!session.requires_repair());
    assert!(matches!(
        session.forward_observed(&INPUTS[128..160], &cancel, &mut |o| {
            if o.stage == "delta_qkv" {
                cancel.store(true, Ordering::Relaxed);
            }
            Ok(())
        }),
        Err(Error::Cancelled)
    ));
    let cancelled_work = session.work();
    assert_eq!(
        cancelled_work,
        Work {
            outcome: Outcome::Cancelled,
            projections_completed: 1,
            gathers_completed: 1,
            ..Work::default()
        }
    );
    snapshot_is(&session, &checkpoints[0]);
    assert!(!session.requires_repair());
    with_projection_failure(2, || assert!(session.forward(&INPUTS[128..160]).is_err()));
    let poisoned_work = session.work();
    assert_eq!(
        poisoned_work,
        Work {
            outcome: Outcome::Error,
            projections_completed: 248,
            gathers_completed: 1,
            convolution_steps_completed: 24,
            recurrence_steps_completed: 24,
            attention_heads_completed: 128,
            ..Work::default()
        }
    );
    assert!(session.requires_repair());
    snapshot_is(&session, &checkpoints[0]);
    assert!(matches!(
        session.forward(&INPUTS[128..160]),
        Err(Error::ExecutionPoisoned)
    ));
    let blocked_work = session.work();
    assert_eq!(
        blocked_work,
        Work {
            outcome: Outcome::Error,
            ..Work::default()
        }
    );
    snapshot_is(&session, &checkpoints[0]);
    let repair_peak = memory.admitted_bytes.checked_add(execution_bytes).unwrap();
    session.options.memory_limit_bytes = repair_peak - 1;
    assert!(
        matches!(session.repair_execution(&loaded.context), Err(Error::MemoryBudget {required,limit})
        if required == repair_peak && limit == repair_peak - 1)
    );
    assert!(session.requires_repair());
    snapshot_is(&session, &checkpoints[0]);
    session.options.memory_limit_bytes = options.memory_limit_bytes;
    let repair = Instant::now();
    session.repair_execution(&loaded.context).unwrap();
    let repair_ns = repair.elapsed().as_nanos();
    assert!(!session.requires_repair());
    snapshot_is(&session, &checkpoints[0]);
    let mut token = 128;
    for input in INPUTS[128..].chunks(32) {
        let measure = Measurement::start();
        let result = session.forward(input);
        let allocations = measure.finish();
        assert_float_bits(
            result.unwrap(),
            &all[token * model.config.vocab..(token + input.len()) * model.config.vocab],
        );
        assert_eq!(allocations, 0);
        assert_eq!(session.work(), work(input.len()));
        token += input.len();
    }
    snapshot_is(&session, &checkpoints[1]);
    assert_eq!(session.memory(), memory);
    let before = snapshot(&session);
    for invalid in [&[][..], &[1; 33][..], &[248320][..]] {
        assert!(session.forward(invalid).is_err());
        assert_eq!(snapshot(&session), before);
        assert_eq!(
            session.work(),
            Work {
                outcome: Outcome::Error,
                ..Work::default()
            }
        );
    }
    let rejected_work = session.work();
    let measure = Measurement::start();
    session.reset();
    assert_eq!(measure.finish(), 0);
    assert!(session.history().is_empty() && session.logits().is_none());
    assert_eq!(session.work(), Work::default());
    assert_eq!(session.memory(), memory);
    let adapter = GenerationSession::new(session, options.memory_limit_bytes).unwrap();
    let state = adapter.state();
    let unsupported = [1u32; 1024];
    for prompt in [512, 1024] {
        assert!(
            crate::generation::request_admission(
                &adapter,
                &crate::generation::Request {
                    input: crate::generation::Input::Prompt(&unsupported[..prompt]),
                    max_new_tokens: 128,
                    stop_tokens: &[]
                }
            )
            .is_err()
        );
        assert_eq!(adapter.state(), state);
    }
    let admitted = crate::generation::request_admission(
        &adapter,
        &crate::generation::Request {
            input: crate::generation::Input::Prompt(&INPUTS[..128]),
            max_new_tokens: 128,
            stop_tokens: &[],
        },
    )
    .unwrap();
    assert_eq!(admitted.required_capacity, 256);
    assert_eq!(adapter.state(), state);
    let validation_ns = validation.elapsed().as_nanos();
    let endpoint = Instant::now();
    let mut source_file = File::open(directory().join("model.safetensors")).unwrap();
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    let mut source_bytes = 0u64;
    loop {
        let count = source_file.read(&mut buffer).unwrap();
        if count == 0 {
            break;
        }
        source_bytes += count as u64;
        assert!(source_bytes <= source::CONTAINER_BYTES);
        digest.update(&buffer[..count]);
    }
    assert_eq!(source_bytes, source::CONTAINER_BYTES);
    let source_endpoint_sha256 = format!("{:x}", digest.finalize());
    assert_eq!(source_endpoint_sha256, source::CONTAINER_SHA256);
    let source_endpoint_ns = endpoint.elapsed().as_nanos();
    let all_logits_sha256 = all.iter().fold(Sha256::new(), |mut digest, value| {
        digest.update(value.to_le_bytes());
        digest
    });
    let mut report = serde_json::Map::new();
    for fields in [
        serde_json::json!({"phase":"context256","source_sha256":source::CONTAINER_SHA256,
            "config_sha256":source::CONFIG_SHA256,"header_sha256":source::HEADER_SHA256,
            "source_bytes":source_bytes,"source_endpoint_sha256":source_endpoint_sha256,
            "tokens":INPUTS.as_slice(),"position_offset":37,"capacity":512,"chunk":32,
            "config":model.config(),"load_report":loaded.store.report(),"memory":memory,"wrapper_memory":adapter.memory_report(),
            "layout":layout(),"model_owned_bytes":model.owned_bytes,"model_learned_bytes":model.learned_bytes,
            "execution_owned_bytes":execution_bytes,"distinct_head":true,"diagnostic_allowance_bytes":DIAGNOSTIC,
            "diagnostic_requested_bytes":diagnostic_requested_bytes,"all_logit_bytes":all.len()*4,
            "all_logits_sha256":format!("{:x}",all_logits_sha256.finalize()),"capture_bytes":bytes,
            "capture_arrays":arrays,"capture_values":values,"checkpoints":checkpoints,"partitions":partitions}),
        serde_json::json!({"context_ns":loaded.context_ns,"cpu_load_ns":loaded.cpu_load_ns,"upload_ns":loaded.upload_ns,
            "bind_ns":bind_ns,"session_ns":session_ns,"observed_forward_ns":observed_forward_ns,"validation_ns":validation_ns,
            "source_endpoint_ns":source_endpoint_ns,"repair_ns":repair_ns,"repair_peak_admitted_bytes":repair_peak,
            "failed_work":failed_work,"cancelled_work":cancelled_work,"poisoned_work":poisoned_work,
            "blocked_work":blocked_work,"rejected_work":rejected_work,"final_work":adapter.state().last_numerical_work,
            "prefix_tokens":128,"prefix_preserved":true,"stale_future_nan_masked":true,
            "repair_overlap_minus_one_rejected":true,"retry_all_logits_bits_exact":true,"retry_warmed_rust_allocations":0,
            "reset_exact":true,"admission_minus_one_rejected":true,"request128plus128_required_capacity":admitted.required_capacity,
            "request512plus128_rejected":true,"request1024plus128_rejected":true,"warmed_rust_allocations":0,
            "whole_gpu_graph_qualified":false,"independent_fidelity_qualified":false,"request_metrics_qualified":false,
            "instrumentation_overhead_qualified":false,"quality_qualified":false,"managed_adoption_qualified":false,
            "allocator_environment":{"MallocLargeCache":std::env::var("MallocLargeCache").ok(),
                "MallocNanoZone":std::env::var("MallocNanoZone").ok(),"MallocAggressiveMadvise":std::env::var("MallocAggressiveMadvise").ok()}}),
    ] {
        let Value::Object(fields) = fields else {
            unreachable!()
        };
        report.extend(fields);
    }
    println!("QWEN35_9B_CONTEXT256_REPORT {}", Value::Object(report));
}
