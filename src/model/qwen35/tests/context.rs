//! Explicit real-source recurrent/context controls; no ordinary model download.
use super::*;
use crate::mlx::qwen35::{self as source, payload};
use std::{
    fs::File,
    io::{Read, Write},
    path::PathBuf,
    time::Instant,
};

const IDS: [u32; 17] = [1, 7, 13, 3, 9, 5, 11, 2, 17, 4, 6, 10, 19, 8, 21, 23, 29];
const OFFSET: usize = 37;

fn complete_work(tokens: usize) -> Work {
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

#[test]
#[ignore = "explicit pinned Qwen3.5-4B CPU17input/offset37 coupled capture and recovery"]
fn actual_source_context17() {
    let directory = PathBuf::from(std::env::var_os("TOKLET_QWEN35_MODEL_DIRECTORY").unwrap());
    let output = PathBuf::from(std::env::var_os("TOKLET_QWEN35_GRAPH_CAPTURE").unwrap());
    std::fs::create_dir(&output).unwrap();
    let file = File::open(directory.join("config.json")).unwrap();
    assert_eq!(file.metadata().unwrap().len(), source::CONFIG_BYTES as u64);
    let mut config = Vec::with_capacity(source::CONFIG_BYTES + 1);
    file.take((source::CONFIG_BYTES + 1) as u64)
        .read_to_end(&mut config)
        .unwrap();
    assert_eq!(config.len(), source::CONFIG_BYTES);
    let start = Instant::now();
    let store = payload::load(
        &config,
        File::open(directory.join("model.safetensors")).unwrap(),
        payload::LoadOptions {
            memory_limit_bytes: MEMORY_LIMIT - 128 * 1024 * 1024,
            modality: source::ModalityPolicy::TextOnly,
        },
    )
    .unwrap();
    let load_ns = start.elapsed().as_nanos();
    let bind = Instant::now();
    let model = Model::from_cpu_store(&store).unwrap();
    let bind_ns = bind.elapsed().as_nanos();
    let options = Options {
        capacity: 512,
        chunk: 32,
        position_offset: OFFSET,
        memory_limit_bytes: MEMORY_LIMIT - 128 * 1024 * 1024,
    };
    let memory = Session::planned_memory(&model, options).unwrap();
    assert!(
        matches!(model.session(Options { memory_limit_bytes: memory.admitted_bytes - 1, ..options }),
        Err(Error::MemoryBudget { required, .. }) if required == memory.admitted_bytes)
    );
    let mut session = model
        .session(Options {
            memory_limit_bytes: memory.admitted_bytes,
            ..options
        })
        .unwrap();
    assert_eq!(session.memory(), memory);
    let mut body = std::io::BufWriter::with_capacity(
        64 * 1024,
        File::create(output.join("rows.f32le")).unwrap(),
    );
    let mut bytes = 0usize;
    let mut arrays = 0;
    println!();
    let observed = Instant::now();
    let expected = session.forward_observed(&IDS, &AtomicBool::new(false), &mut |o| {
        assert_eq!(o.position, OFFSET + o.token);
        if o.stage == "before_commit" || o.token != IDS.len() - 1 { return Ok(()); }
        let count = o.values.len();
        let n = count.checked_mul(4).ok_or(Error::Observation("capture size"))?;
        if bytes.checked_add(n).is_none_or(|v| v > 128*1024*1024) { return Err(Error::Observation("capture bound")); }
        let mut hash = Sha256::new();
        for &value in o.values {
            let raw = value.to_le_bytes();
            body.write_all(&raw).map_err(|_| Error::Observation("capture write"))?;
            hash.update(raw);
        }
        println!("QWEN35_CONTEXT_ARRAY {}", serde_json::json!({"token":o.token,"position":o.position,"layer":o.layer,"stage":o.stage,"head":o.head,"count":count,"offset":bytes,"bytes":n,"sha256":format!("{:x}",hash.finalize())}));
        bytes += n; arrays += 1; Ok(())
    }).unwrap().to_vec();
    let observed_ns = observed.elapsed().as_nanos();
    body.flush().unwrap();
    body.get_ref().sync_all().unwrap();
    assert_eq!((arrays, bytes), (1051, 65334272));
    let observed_work = session.work();
    assert_eq!(observed_work, complete_work(17));
    assert_eq!(session.history(), IDS);
    let end = snapshot(&session);
    let mut partition_ns = Vec::new();
    for chunk in [17, 1, 3] {
        session.reset();
        let begin = Instant::now();
        for (index, part) in IDS.chunks(chunk).enumerate() {
            let first = index * chunk;
            let measure = Measurement::start();
            let result = session.forward(part);
            let allocations = measure.finish();
            assert_float_bits(
                result.unwrap(),
                &expected[first * model.config.vocab..(first + part.len()) * model.config.vocab],
            );
            assert_eq!(allocations, 0);
            assert_eq!(session.work(), complete_work(part.len()));
            assert_eq!(session.memory(), memory);
        }
        assert_eq!(snapshot(&session), end);
        partition_ns.push(begin.elapsed().as_nanos());
        println!(
            "QWEN35_CONTEXT_PARTITION {}",
            serde_json::json!({"chunk":chunk,"snapshot":end,"all_logits_bits_exact":true,"warmed_allocations":0})
        );
    }
    session.reset();
    session.forward(&IDS[..5]).unwrap();
    let prefix = snapshot(&session);
    let error = session
        .forward_observed(&IDS[5..], &AtomicBool::new(false), &mut |o| {
            if o.token == 7 && o.layer == Some(0) && o.stage == "delta_state" {
                Err(Error::Observation("context injected failure"))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
    assert!(matches!(
        error,
        Error::Observation("context injected failure")
    ));
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
    assert_eq!(session.history(), &IDS[..5]);
    assert_eq!(snapshot(&session), prefix);
    let cancel = AtomicBool::new(false);
    let error = session
        .forward_observed(&IDS[5..], &cancel, &mut |o| {
            if o.token == 5 && o.layer == Some(0) && o.stage == "delta_qkv" {
                cancel.store(true, Ordering::Relaxed);
            }
            Ok(())
        })
        .unwrap_err();
    assert!(matches!(error, Error::Cancelled));
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
    assert_eq!(snapshot(&session), prefix);
    let measure = Measurement::start();
    let result = session.forward(&IDS[5..]);
    let allocations = measure.finish();
    assert_float_bits(result.unwrap(), &expected[5 * model.config.vocab..]);
    assert_eq!(allocations, 0);
    assert_eq!(snapshot(&session), end);
    assert_eq!(session.work(), complete_work(12));
    assert!(session.forward(&[0; 33]).is_err());
    assert_eq!(snapshot(&session), end);
    assert_eq!(session.memory(), memory);
    println!(
        "QWEN35_CONTEXT_REPORT {}",
        serde_json::json!({"source_sha256":source::CONTAINER_SHA256,"container_bytes":source::CONTAINER_BYTES,"config_sha256":source::CONFIG_SHA256,"header_sha256":source::HEADER_SHA256,"tokens":IDS,"position_offset":OFFSET,"capture_token":16,"capacity":512,"chunk":32,"memory":memory,"capture_bytes":bytes,"arrays":arrays,"load_ns":load_ns,"bind_ns":bind_ns,"observed_forward_ns":observed_ns,"observed_work":observed_work,"partition_forward_ns":partition_ns,"partitions":[17,1,3],"all_logits_bits_exact":true,"warmed_rust_allocations":0,"admission_minus_one_rejected":true,"failed_prefix_preserved":true,"cancelled_prefix_preserved":true,"retry_exact":true,"failed_work":failed_work,"cancelled_work":cancelled_work,"final_work":session.work(),"greedy_lowest_id":greedy(session.logits().unwrap()),"independent_context_fidelity_qualified":false,"quality_qualified":false,"managed_adoption_qualified":false,"timing_scope":"Diagnostic CPU observations/control replays; no controlled engine latency/overhead qualification"})
    );
}
