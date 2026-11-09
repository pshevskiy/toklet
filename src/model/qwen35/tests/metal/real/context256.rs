//! Full source depth and bounded all-logit replay; no request/quality route.
use super::*;
use std::io::{BufReader, BufWriter, Seek, SeekFrom};

const CHECKPOINTS: [usize; 2] = [128, 256];
const CAPTURE_VALUES: [usize; 2] = [18_180_608, 20_310_528];
const DIAGNOSTIC_BYTES: usize = 256 * 1024 * 1024;
const INPUTS: [u32; 256] = {
    let mut result = [0; 256];
    let mut index = 0;
    while index < result.len() {
        result[index] = IDS[index % IDS.len()];
        index += 1;
    }
    result
};

#[derive(serde::Serialize)]
struct Checkpoint {
    inputs: usize,
    snapshot: String,
    greedy_lowest_id: u32,
}
fn step_options() -> Options {
    Options {
        memory_limit_bytes: MEMORY_LIMIT - DIAGNOSTIC_BYTES,
        ..opts(0)
    }
}
fn compare_rows(values: &[f32], expected: &mut BufReader<File>, buffer: &mut [u8]) {
    let bytes = values.len().checked_mul(4).unwrap();
    assert!(bytes <= buffer.len());
    expected.read_exact(&mut buffer[..bytes]).unwrap();
    for (index, (value, raw)) in values
        .iter()
        .zip(buffer[..bytes].chunks_exact(4))
        .enumerate()
    {
        assert_eq!(
            value.to_bits(),
            u32::from_le_bytes(raw.try_into().unwrap()),
            "logit bits at{index}"
        );
    }
}
fn row_offset(token: usize, vocab: usize) -> u64 {
    token.checked_mul(vocab).unwrap().checked_mul(4).unwrap() as u64
}
fn snapshot_is(session: &MetalSession<'_, '_, '_>, expected: &Checkpoint) {
    assert_eq!(session.history().len(), expected.inputs);
    assert_eq!(snapshot(session), expected.snapshot);
    assert_eq!(
        crate::sampling::greedy(session.logits().unwrap()).unwrap(),
        expected.greedy_lowest_id
    );
}

#[test]
#[ignore = "explicit authenticated Qwen128/256 full context, all-logit partitions and recovery"]
fn actual_source_context256() {
    println!();
    let output = PathBuf::from(std::env::var_os("TOKLET_QWEN35_CONTEXT256_CAPTURE").unwrap());
    std::fs::create_dir(&output).unwrap();
    let loaded = load_with_diagnostic(DIAGNOSTIC_BYTES);
    let model = MetalModel::from_metal_store(&loaded.store).unwrap();
    let options = step_options();
    let planned = model.planned_session_memory(options).unwrap();
    assert!(planned.admitted_bytes + DIAGNOSTIC_BYTES <= MEMORY_LIMIT);
    assert!(
        matches!(model.session(&loaded.context, Options { memory_limit_bytes: planned.admitted_bytes - 1, ..options }), Err(Error::MemoryBudget { required, .. }) if required == planned.admitted_bytes)
    );
    let mut session = model.session(&loaded.context, options).unwrap();
    assert_eq!(session.memory(), planned);
    let memory = session.memory();
    let mut body =
        BufWriter::with_capacity(64 * 1024, File::create(output.join("rows.f32le")).unwrap());
    let mut logits_file = BufWriter::with_capacity(
        64 * 1024,
        File::create(output.join("logits.f32le")).unwrap(),
    );
    let mut checkpoints = Vec::with_capacity(2);
    let mut bytes = 0;
    let mut arrays = [0; 2];
    let mut values = [0; 2];
    let start = Instant::now();
    for inputs in INPUTS.chunks(32) {
        let all = session.forward_observed(inputs, &AtomicBool::new(false), &mut |observation| {
            if observation.stage == "before_commit" { return Ok(()); }
            let Some(slot) = CHECKPOINTS.iter().position(|&n| n == observation.token + 1) else { return Ok(()); };
            let count = observation.values.len();
            let size = count.checked_mul(4).unwrap();
            assert!(bytes + size <= 256*1024*1024);
            let mut digest = Sha256::new();
            for value in observation.values {
                assert!(value.is_finite());
                let raw = value.to_le_bytes();body.write_all(&raw).unwrap();digest.update(raw);
            }
            println!("QWEN35_CONTEXT256_ARRAY {}",serde_json::json!({"checkpoint":CHECKPOINTS[slot],
                "token":observation.token,"position":observation.position,"layer":observation.layer,
                "stage":observation.stage,"head":observation.head,"count":count,"offset":bytes,"bytes":size,
                "sha256":format!("{:x}",digest.finalize())}));
            bytes += size;arrays[slot] += 1;values[slot] += count;
            Ok(())
        }).unwrap();
        for value in all {
            assert!(value.is_finite());
            logits_file.write_all(&value.to_le_bytes()).unwrap();
        }
        assert_eq!(session.work(), work(inputs.len()));
        if CHECKPOINTS.contains(&session.history().len()) {
            checkpoints.push(Checkpoint {
                inputs: session.history().len(),
                snapshot: snapshot(&session),
                greedy_lowest_id: crate::sampling::greedy(session.logits().unwrap()).unwrap(),
            });
        }
    }
    let diagnostic_capture_ns = start.elapsed().as_nanos();
    assert_eq!(arrays, [1051, 1051]);
    assert_eq!(values, CAPTURE_VALUES);
    assert_eq!(bytes, 153_964_544);
    assert_eq!(checkpoints.len(), 2);
    body.flush().unwrap();
    body.get_ref().sync_all().unwrap();
    drop(body);
    logits_file.flush().unwrap();
    logits_file.get_ref().sync_all().unwrap();
    drop(logits_file);
    assert_eq!(
        std::fs::metadata(output.join("logits.f32le"))
            .unwrap()
            .len(),
        row_offset(256, model.config.vocab)
    );
    let mut expected =
        BufReader::with_capacity(64 * 1024, File::open(output.join("logits.f32le")).unwrap());
    let mut buffer = vec![0u8; 32 * model.config.vocab * 4];
    assert!(buffer.capacity() + 2 * 64 * 1024 + 64 * 1024 < DIAGNOSTIC_BYTES);
    let mut partitions = Vec::with_capacity(3);
    for chunk in [32, 17, 1] {
        let measure = Measurement::start();
        session.reset();
        assert_eq!(measure.finish(), 0);
        expected.seek(SeekFrom::Start(0)).unwrap();
        let mut token = 0;
        let mut calls = Vec::with_capacity(256);
        let mut forward_ns = 0;
        for checkpoint in &checkpoints {
            while token < checkpoint.inputs {
                let count = chunk.min(checkpoint.inputs - token);
                let measure = Measurement::start();
                let start = Instant::now();
                let all = session.forward(&INPUTS[token..token + count]);
                forward_ns += start.elapsed().as_nanos();
                let allocations = measure.finish();
                compare_rows(all.unwrap(), &mut expected, &mut buffer);
                assert_eq!(allocations, 0);
                assert_eq!(session.work(), work(count));
                calls.push(count);
                token += count;
            }
            snapshot_is(&session, checkpoint);
        }
        assert_eq!(expected.read(&mut [0]).unwrap(), 0);
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
    // Inactive tails contain invalid values so future-cache visibility cannot
    // accidentally pass just because a prior clean replay populated those rows.
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
                    return Err(Error::Observation("context256 late-third-frame failure"));
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
    let gpu_failed_work = session.work();
    assert_eq!(
        gpu_failed_work,
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
    snapshot_is(&session, &checkpoints[0]);
    assert!(session.requires_repair());
    session.repair_execution(&loaded.context).unwrap();
    assert!(!session.requires_repair());
    snapshot_is(&session, &checkpoints[0]);
    expected
        .seek(SeekFrom::Start(row_offset(128, model.config.vocab)))
        .unwrap();
    for input in INPUTS[128..].chunks(32) {
        let measure = Measurement::start();
        let all = session.forward(input);
        let allocations = measure.finish();
        compare_rows(all.unwrap(), &mut expected, &mut buffer);
        assert_eq!(allocations, 0);
        assert_eq!(session.work(), work(input.len()));
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
    let boundary_work = session.work();
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
    let mut report = serde_json::Map::new();
    for fields in [
        serde_json::json!({
        "source_sha256":source::CONTAINER_SHA256,"tokens":INPUTS.as_slice(),"position_offset":0,
        "capacity":512,"chunk":32,"memory":memory,"wrapper_memory":adapter.memory_report(),
        "diagnostic_allowance_bytes":DIAGNOSTIC_BYTES,"load_report":loaded.store.report(),"layout":layout(),
        "model_owned_bytes":model.owned_bytes,"model_learned_bytes":model.learned_bytes,
        "execution_owned_bytes":adapter.numerical_for_tests().execution.owned_bytes().unwrap(),
        "checkpoints":checkpoints,"capture_bytes":bytes,"capture_arrays":arrays,"capture_values":values,
        "all_logit_bytes":row_offset(256,model.config.vocab),"partitions":partitions}),
        serde_json::json!({"failed_work":failed_work,
        "cancelled_work":cancelled_work,"gpu_failed_work":gpu_failed_work,"boundary_work":boundary_work,
        "final_work":adapter.state().last_numerical_work,"failed_prefix_preserved":true,"retry_bits_exact":true,
        "repair_prefix_preserved":true,"stale_future_nan_masked":true,"reset_exact":true,
        "admission_minus_one_rejected":true,"request128plus128_required_capacity":admitted.required_capacity,
        "request512plus128_rejected":true,"request1024plus128_rejected":true,"warmed_rust_allocations":0,
        "diagnostic_capture_ns":diagnostic_capture_ns,"independent_fidelity_qualified":false,
        "request_metrics_qualified":false,"instrumentation_overhead_qualified":false,
        "quality_qualified":false,"managed_adoption_qualified":false,
        "allocator_environment":{"MallocLargeCache":std::env::var("MallocLargeCache").ok(),
            "MallocNanoZone":std::env::var("MallocNanoZone").ok(),"MallocAggressiveMadvise":std::env::var("MallocAggressiveMadvise").ok()}}),
    ] {
        let serde_json::Value::Object(fields) = fields else {
            unreachable!()
        };
        report.extend(fields);
    }
    println!(
        "QWEN35_CONTEXT256_REPORT {}",
        serde_json::Value::Object(report)
    );
}
