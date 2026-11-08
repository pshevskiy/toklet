//! Real-source closed GPU projection/CPU-state probes, separate from ordinary tests.
use super::*;
use crate::mlx::qwen35::{self as source, payload};
use std::{
    fs::File,
    io::{Read, Write},
    mem::size_of,
    path::PathBuf,
    time::Instant,
};

const IDS: [u32; 17] = [1, 7, 13, 3, 9, 5, 11, 2, 17, 4, 6, 10, 19, 8, 21, 23, 29];
const DIAGNOSTIC: usize = 128 * 1024 * 1024;

struct Loaded {
    context: Context,
    store: payload::metal::MetalStore,
    cpu_load_ns: u128,
    upload_ns: u128,
    context_ns: u128,
}
fn directory() -> PathBuf {
    PathBuf::from(std::env::var_os("TOKLET_QWEN35_MODEL_DIRECTORY").unwrap())
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
fn load() -> Loaded {
    let start = Instant::now();
    let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let context_ns = start.elapsed().as_nanos();
    let config = config();
    let start = Instant::now();
    let cpu = payload::load(
        &config,
        File::open(directory().join("model.safetensors")).unwrap(),
        payload::LoadOptions {
            memory_limit_bytes: MEMORY_LIMIT - DIAGNOSTIC - CONTEXT_ALLOWANCE_BYTES,
            modality: source::ModalityPolicy::TextOnly,
        },
    )
    .unwrap();
    let cpu_load_ns = start.elapsed().as_nanos();
    let start = Instant::now();
    let store = cpu.into_metal(&context, MEMORY_LIMIT - DIAGNOSTIC).unwrap();
    let upload_ns = start.elapsed().as_nanos();
    assert_eq!(store.report().matrices, 249);
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
fn workload() -> (&'static str, &'static [u32], usize) {
    match std::env::var("TOKLET_QWEN35_METAL_PHASE").unwrap().as_str() {
        "one" => ("one", &IDS[..1], 0),
        "context17" => ("context17", &IDS, 37),
        _ => panic!("frozen phase required"),
    }
}
fn opts(offset: usize) -> Options {
    Options {
        capacity: 512,
        chunk: 32,
        position_offset: offset,
        memory_limit_bytes: MEMORY_LIMIT - DIAGNOSTIC,
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

#[test]
#[ignore = "explicit pinned Qwen Metal projection/CPU-state source capture and contextual recovery"]
fn actual_source_coupled() {
    println!();
    let (phase, ids, offset) = workload();
    let output = PathBuf::from(std::env::var_os("TOKLET_QWEN35_METAL_CAPTURE").unwrap());
    std::fs::create_dir(&output).unwrap();
    let loaded = load();
    let start = Instant::now();
    let model = MetalModel::from_metal_store(&loaded.store).unwrap();
    let bind_ns = start.elapsed().as_nanos();
    let options = opts(offset);
    let planned = model.planned_session_memory(options).unwrap();
    assert!(
        matches!(model.session(&loaded.context,Options {memory_limit_bytes:planned.admitted_bytes-1,..options}),Err(Error::MemoryBudget {required,..}) if required==planned.admitted_bytes)
    );
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
    let mut bytes = 0usize;
    let mut arrays = 0usize;
    let observed = Instant::now();
    let all=session.forward_observed(ids,&AtomicBool::new(false),&mut|o| {
        if o.stage=="before_commit" || o.token!=ids.len()-1 {return Ok(());}
        let mut digest=Sha256::new();let count=o.values.len();let size=count.checked_mul(4).unwrap();
        assert!(bytes.checked_add(size).unwrap()<=128*1024*1024);
        for value in o.values {let raw=value.to_le_bytes();body.write_all(&raw).unwrap();digest.update(raw);}
        println!("QWEN35_METAL_ARRAY {}",serde_json::json!({"token":o.token,"position":o.position,"layer":o.layer,"stage":o.stage,"head":o.head,"count":count,"offset":bytes,"bytes":size,"sha256":format!("{:x}",digest.finalize())}));
        bytes+=size;arrays+=1;Ok(())
    }).unwrap().to_vec();
    let observed_forward_ns = observed.elapsed().as_nanos();
    let observed_work = session.work();
    assert_eq!(observed_work, work(ids.len()));
    assert_eq!(arrays, 1051);
    assert_eq!(bytes, if phase == "one" { 64269312 } else { 65334272 });
    body.flush().unwrap();
    body.get_ref().sync_all().unwrap();
    let final_state = snapshot(&session);
    let mut partitions = Vec::new();
    for chunk in [17, 1, 3] {
        session.reset();
        let mut forward_ns = 0u128;
        let mut token = 0;
        for input in ids.chunks(chunk) {
            let measure = Measurement::start();
            let start = Instant::now();
            let result = session.forward(input);
            let elapsed = start.elapsed().as_nanos();
            let allocations = measure.finish();
            assert_float_bits(
                result.unwrap(),
                &all[token * model.config.vocab..(token + input.len()) * model.config.vocab],
            );
            assert_eq!(allocations, 0);
            assert_eq!(session.work(), work(input.len()));
            forward_ns += elapsed;
            token += input.len();
        }
        assert_eq!(snapshot(&session), final_state);
        assert_eq!(session.history(), ids);
        partitions.push(serde_json::json!({"chunk":chunk,"snapshot":snapshot(&session),"all_logits_bits_exact":true,"warmed_rust_allocations":0,"forward_ns":forward_ns}));
    }
    let mut reset_allocations = Vec::new();
    for trial in 0..3 {
        let measure = Measurement::start();
        session.reset();
        let allocations = measure.finish();
        assert_eq!(allocations, 0);
        assert!(session.history().is_empty() && session.logits().is_none());
        assert_eq!(session.work(), Work::default());
        assert_eq!(session.memory(), memory);
        reset_allocations.push(allocations);
        if trial < 2 {
            assert_float_bits(session.forward(ids).unwrap(), &all);
            assert_eq!(snapshot(&session), final_state);
        }
    }
    let mut recovery = Value::Null;
    if phase == "context17" {
        session.forward(&ids[..5]).unwrap();
        let prefix = snapshot(&session);
        for cache in &mut session.kv {
            for head in 0..model.config.kv_heads {
                let start = (head * 512 + 5) * model.config.head_dim;
                let end = (head + 1) * 512 * model.config.head_dim;
                cache.k[start..end].fill(f32::NAN);
                cache.v[start..end].fill(f32::NAN);
            }
        }
        assert!(
            session
                .forward_observed(&ids[5..], &AtomicBool::new(false), &mut |o| {
                    if o.token == 7 && o.layer == Some(0) && o.stage == "delta_output" {
                        return Err(Error::Observation("frozen partial-third-frame failure"));
                    }
                    Ok(())
                })
                .is_err()
        );
        let failed_work = session.work();
        assert_eq!(snapshot(&session), prefix);
        assert!(!session.requires_repair());
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
        let cancel = AtomicBool::new(false);
        assert!(matches!(
            session.forward_observed(&ids[5..], &cancel, &mut |o| {
                if o.stage == "delta_qkv" {
                    cancel.store(true, Ordering::Relaxed);
                }
                Ok(())
            }),
            Err(Error::Cancelled)
        ));
        let cancelled_work = session.work();
        assert_eq!(snapshot(&session), prefix);
        assert!(!session.requires_repair());
        assert_eq!(
            cancelled_work,
            Work {
                outcome: Outcome::Cancelled,
                tokens_completed: 0,
                tokens_committed: 0,
                projections_completed: 1,
                gathers_completed: 1,
                convolution_steps_completed: 0,
                recurrence_steps_completed: 0,
                attention_heads_completed: 0
            }
        );
        with_projection_failure(2, || assert!(session.forward(&ids[5..]).is_err()));
        let poisoned_work = session.work();
        assert_eq!(
            poisoned_work,
            Work {
                outcome: Outcome::Error,
                tokens_completed: 0,
                tokens_committed: 0,
                projections_completed: 248,
                gathers_completed: 1,
                convolution_steps_completed: 24,
                recurrence_steps_completed: 24,
                attention_heads_completed: 128,
            }
        );
        assert!(session.requires_repair());
        assert_eq!(snapshot(&session), prefix);
        assert!(matches!(
            session.forward(&ids[5..]),
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
        assert_eq!(snapshot(&session), prefix);
        let repair_peak = memory.admitted_bytes.checked_add(execution_bytes).unwrap();
        session.options.memory_limit_bytes = repair_peak - 1;
        assert!(
            matches!(session.repair_execution(&loaded.context),Err(Error::MemoryBudget {required,limit}) if required==repair_peak && limit==repair_peak-1)
        );
        assert!(session.requires_repair());
        assert_eq!(snapshot(&session), prefix);
        session.options.memory_limit_bytes = options.memory_limit_bytes;
        let repair = Instant::now();
        session.repair_execution(&loaded.context).unwrap();
        let repair_ns = repair.elapsed().as_nanos();
        assert!(!session.requires_repair());
        assert_eq!(session.memory(), memory);
        assert_eq!(snapshot(&session), prefix);
        let measure = Measurement::start();
        let result = session.forward(&ids[5..]);
        let allocations = measure.finish();
        assert_float_bits(result.unwrap(), &all[5 * model.config.vocab..]);
        assert_eq!(allocations, 0);
        let retry_work = session.work();
        assert_eq!(retry_work, work(12));
        assert_eq!(snapshot(&session), final_state);
        assert!(session.forward(&[1; 33]).is_err());
        let rejected_work = session.work();
        assert_eq!(
            rejected_work,
            Work {
                outcome: Outcome::Error,
                ..Work::default()
            }
        );
        assert_eq!(snapshot(&session), final_state);
        recovery = serde_json::json!({"prefix_tokens":5,"prefix_preserved":true,"stale_nonfinite_kv_masked":true,"failed_work":failed_work,"cancelled_work":cancelled_work,"poisoned_work":poisoned_work,"blocked_work":blocked_work,"retry_work":retry_work,"rejected_work":rejected_work,"repair_overlap_minus_one_rejected":true,"repair_peak_admitted_bytes":repair_peak,"repair_ns":repair_ns,"retry_warmed_rust_allocations":allocations,"retry_all_logits_bits_exact":true});
    }
    println!(
        "QWEN35_METAL_REPORT {}",
        serde_json::json!({"phase":phase,"source_sha256":source::CONTAINER_SHA256,"config_sha256":source::CONFIG_SHA256,"header_sha256":source::HEADER_SHA256,"source_bytes":source::CONTAINER_BYTES,
        "tokens":ids,"position_offset":offset,"capacity":512,"chunk":32,"config":model.config(),"load_report":loaded.store.report(),"model_owned_bytes":model.owned_bytes,"model_learned_bytes":model.learned_bytes,"memory":memory,"execution_owned_bytes":execution_bytes,"layout":layout(),
        "context_ns":loaded.context_ns,"cpu_load_ns":loaded.cpu_load_ns,"upload_ns":loaded.upload_ns,"bind_ns":bind_ns,"session_ns":session_ns,"observed_forward_ns":observed_forward_ns,"observed_work":observed_work,"capture_bytes":bytes,"arrays":arrays,"partitions":partitions,"reset_allocations":reset_allocations,"recovery":recovery,"all_logits_bits_exact":true,"greedy_lowest_id":greedy(&all[(ids.len()-1)*model.config.vocab..]),"warmed_rust_allocations":0,"admission_minus_one_rejected":true,"whole_gpu_graph_qualified":false,"independent_fidelity_qualified":false,"quality_qualified":false,"managed_adoption_qualified":false})
    );
}

#[test]
#[ignore = "explicit release arm64 ABI layout observation for frozen report accounting"]
fn abi_layout_probe() {
    println!("\nQWEN35_METAL_LAYOUT {}", layout());
}

fn timed<'s, M: MatrixStorage, E: Execution<M>>(
    session: &'s mut Session<'_, '_, M, E>,
    ids: &[u32],
) -> (u128, &'s [f32]) {
    let start = Instant::now();
    let output = session.forward(ids);
    let ns = start.elapsed().as_nanos();
    (ns, std::hint::black_box(output.unwrap()))
}
fn compare_logits(cpu: &[f32], gpu: &[f32], vocab: usize) -> f64 {
    assert_eq!(cpu.len(), 17 * vocab);
    assert_eq!(gpu.len(), cpu.len());
    let mut maximum = 0.0f64;
    for (&reference, &actual) in cpu.iter().zip(gpu) {
        let error = (f64::from(actual) - f64::from(reference)).abs()
            / (1e-5 + 1e-4 * f64::from(reference).abs());
        assert!(actual.is_finite() && reference.is_finite() && error <= 1.);
        maximum = maximum.max(error);
    }
    for (c, g) in cpu.chunks_exact(vocab).zip(gpu.chunks_exact(vocab)) {
        assert_eq!(greedy(c), greedy(g));
    }
    maximum
}

#[test]
#[ignore = "explicit paired full17input scalar/Metal forward,3warm+10measure;~30min per process"]
fn matched_forward17_benchmark() {
    println!();
    let (phase, ids, offset) = workload();
    assert_eq!(phase, "context17");
    let cohort: usize = std::env::var("TOKLET_QWEN35_METAL_COHORT")
        .unwrap()
        .parse()
        .unwrap();
    assert!(cohort < 2);
    let loaded = load();
    let gpu_model = MetalModel::from_metal_store(&loaded.store).unwrap();
    let mut gpu = gpu_model.session(&loaded.context, opts(offset)).unwrap();
    let gpu_memory = gpu.memory();
    // This comparison deliberately owns a separate packed scalar reference.
    // It is diagnostic overlap; the ordinary GPU owner retains no CPU matrices.
    let available = MEMORY_LIMIT
        .checked_sub(DIAGNOSTIC)
        .unwrap()
        .checked_sub(gpu_memory.admitted_bytes)
        .unwrap();
    let cfg = config();
    let start = Instant::now();
    let cpu_store = payload::load(
        &cfg,
        File::open(directory().join("model.safetensors")).unwrap(),
        payload::LoadOptions {
            memory_limit_bytes: available,
            modality: source::ModalityPolicy::TextOnly,
        },
    )
    .unwrap();
    let scalar_load_ns = start.elapsed().as_nanos();
    let load_peak = gpu_memory
        .admitted_bytes
        .checked_add(cpu_store.report().admitted_load_bytes)
        .unwrap()
        .checked_add(DIAGNOSTIC)
        .unwrap();
    assert!(load_peak <= MEMORY_LIMIT);
    let cpu_model = Model::from_cpu_store(&cpu_store).unwrap();
    let cpu_options = Options {
        memory_limit_bytes: available,
        ..opts(offset)
    };
    let mut cpu = cpu_model.session(cpu_options).unwrap();
    let cpu_memory = cpu.memory();
    let admitted = gpu_memory
        .admitted_bytes
        .checked_add(cpu_memory.admitted_bytes)
        .unwrap()
        .checked_add(DIAGNOSTIC)
        .unwrap();
    assert!(admitted <= MEMORY_LIMIT);
    let mut maximum = 0.0f64;
    for iteration in 0..13 {
        cpu.reset();
        gpu.reset();
        assert!(cpu.history().is_empty() && gpu.history().is_empty());
        let cpu_first = (iteration + cohort).is_multiple_of(2);
        let (cpu_ns, gpu_ns, ratio) = if cpu_first {
            let (cn, c) = timed(&mut cpu, ids);
            let (gn, g) = timed(&mut gpu, ids);
            (cn, gn, compare_logits(c, g, gpu_model.config.vocab))
        } else {
            let (gn, g) = timed(&mut gpu, ids);
            let (cn, c) = timed(&mut cpu, ids);
            (cn, gn, compare_logits(c, g, gpu_model.config.vocab))
        };
        maximum = maximum.max(ratio);
        assert_eq!(cpu.work(), work(17));
        assert_eq!(gpu.work(), work(17));
        assert_eq!(cpu.history(), ids);
        assert_eq!(gpu.history(), ids);
        for slot in 0..2 {
            let scalar = (slot == 0) == cpu_first;
            println!(
                "QWEN35_METAL_SAMPLE {}",
                serde_json::json!({"cohort":cohort,"iteration":iteration,"warmup":iteration<3,"engine":if scalar {"scalar_cpu"}else{"metal_projection_cpu_state"},"elapsed_ns":if scalar {cpu_ns}else{gpu_ns},"tokens":17,"position_offset":37,"work":if scalar {cpu.work()}else{gpu.work()},"maximum_cpu_gpu_logit_gate_ratio":ratio})
            );
        }
    }
    println!(
        "QWEN35_METAL_BENCH_REPORT {}",
        serde_json::json!({"cohort":cohort,"phase":phase,"source_sha256":source::CONTAINER_SHA256,"config_sha256":source::CONFIG_SHA256,"header_sha256":source::HEADER_SHA256,"tokens":ids,"position_offset":offset,"capacity":512,"chunk":32,"cpu_memory":cpu_memory,"gpu_memory":gpu_memory,"diagnostic_allowance_bytes":DIAGNOSTIC,"combined_admitted_bytes":admitted,"cpu_load_overlap_admitted_bytes":load_peak,"gpu_load_report":loaded.store.report(),"scalar_load_ns":scalar_load_ns,"gpu_cpu_load_ns":loaded.cpu_load_ns,"gpu_upload_ns":loaded.upload_ns,"context_ns":loaded.context_ns,"maximum_cpu_gpu_logit_gate_ratio":maximum,"all17_greedy_ids_match":true,"warmups_per_engine":3,"measurements_per_engine":10,"timing_scope":"Completed full17input/offset37 forward only; resident weights, empty numerical prefix after reset; reset/load/compare/diagnostic I/O outside timer","whole_gpu_graph_qualified":false,"request_ttft_qualified":false,"sustained_qualified":false,"quality_qualified":false,"managed_adoption_qualified":false})
    );
}
