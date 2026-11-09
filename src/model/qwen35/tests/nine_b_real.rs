//! Explicit pinned9B CPU capture and prefix/replay witness.
use super::*;

#[test]
#[ignore = "explicit authenticated9B CPU all-stage/state/logit capture; no codec or GPU"]
fn actual_source_one() {
    use crate::mlx::qwen35::{ModalityPolicy, nine_b as source, nine_b::payload};
    use std::{
        fs::File,
        io::{Read, Write},
        path::PathBuf,
        time::Instant,
    };
    let directory = PathBuf::from(std::env::var_os("TOKLET_QWEN35_9B_MODEL_DIRECTORY").unwrap());
    let output = PathBuf::from(std::env::var_os("TOKLET_QWEN35_9B_GRAPH_CAPTURE").unwrap());
    std::fs::create_dir(&output).unwrap();
    let config_file = File::open(directory.join("config.json")).unwrap();
    assert_eq!(
        config_file.metadata().unwrap().len(),
        source::CONFIG_BYTES as u64
    );
    let mut config = Vec::with_capacity(source::CONFIG_BYTES + 1);
    config_file
        .take((source::CONFIG_BYTES + 1) as u64)
        .read_to_end(&mut config)
        .unwrap();
    assert_eq!(config.len(), source::CONFIG_BYTES);
    let input = File::open(directory.join("model.safetensors")).unwrap();
    assert!(input.metadata().unwrap().is_file());
    assert_eq!(input.metadata().unwrap().len(), source::CONTAINER_BYTES);
    let start = Instant::now();
    let store = payload::load(
        &config,
        input,
        payload::LoadOptions {
            memory_limit_bytes: MEMORY_LIMIT - 128 * 1024 * 1024,
            modality: ModalityPolicy::TextOnly,
        },
    )
    .unwrap();
    let load_ns = start.elapsed().as_nanos();
    let bind = Instant::now();
    let model = Model::from_nine_b_cpu_store(&store).unwrap();
    let bind_ns = bind.elapsed().as_nanos();
    assert_eq!(
        (
            model.config.layers,
            model.config.hidden,
            model.config.intermediate,
            model.config.vocab
        ),
        (32, 4096, 12288, 248320)
    );
    let options = Options {
        capacity: 512,
        chunk: 32,
        position_offset: 0,
        memory_limit_bytes: MEMORY_LIMIT - 128 * 1024 * 1024,
    };
    let planned = Session::planned_memory(&model, options).unwrap();
    assert!(
        matches!(model.session(Options{memory_limit_bytes:planned.admitted_bytes-1,..options}),Err(Error::MemoryBudget{required,..})if required==planned.admitted_bytes)
    );
    let mut session = model
        .session(Options {
            memory_limit_bytes: planned.admitted_bytes,
            ..options
        })
        .unwrap();
    assert_eq!(session.memory(), planned);
    let mut body = std::io::BufWriter::with_capacity(
        64 * 1024,
        File::create(output.join("rows.f32le")).unwrap(),
    );
    let mut offset = 0;
    let mut arrays = 0;
    println!();
    let observed = Instant::now();
    session.forward_observed(&[1],&AtomicBool::new(false),&mut |o|{
        if o.stage=="before_commit"{return Ok(());}
        let bytes=o.values.len()*4;if offset+bytes>128*1024*1024{return Err(Error::Observation("bounded capture"));}
        let mut hash=Sha256::new();for &value in o.values{let b=value.to_le_bytes();body.write_all(&b).map_err(|_|Error::Observation("capture write"))?;hash.update(b);}
        println!("QWEN35_9B_GRAPH_ARRAY {}",serde_json::json!({"token":o.token,"position":o.position,"layer":o.layer,"stage":o.stage,"head":o.head,"count":o.values.len(),"offset":offset,"bytes":bytes,"sha256":format!("{:x}",hash.finalize())}));offset+=bytes;arrays+=1;Ok(())
    }).unwrap();
    let observed_ns = observed.elapsed().as_nanos();
    body.flush().unwrap();
    body.get_ref().sync_all().unwrap();
    let expected = session.logits().unwrap().to_vec();
    let before = snapshot(&session);
    let work = session.work();
    assert_eq!(session.history(), [1]);
    assert_eq!(
        (
            work.projections_completed,
            work.gathers_completed,
            work.convolution_steps_completed,
            work.recurrence_steps_completed,
            work.attention_heads_completed
        ),
        (249, 1, 24, 24, 128)
    );
    session.reset();
    let plain = Instant::now();
    let measure = Measurement::start();
    let result = session.forward(&[1]);
    let allocations = measure.finish();
    result.unwrap();
    let plain_ns = plain.elapsed().as_nanos();
    assert_eq!(allocations, 0);
    assert_eq!(session.logits().unwrap(), expected);
    assert_eq!(snapshot(&session), before);
    assert_eq!(arrays, 1051);
    assert_eq!(offset, 66_640_896);
    assert!(model.output_head.is_some());
    assert!(matches!(
        session.forward_observed(&[7], &AtomicBool::new(false), &mut |o| {
            if o.stage == "logits" {
                Err(Error::Observation("9B output observer"))
            } else {
                Ok(())
            }
        }),
        Err(Error::Observation("9B output observer"))
    ));
    let failed_work = session.work();
    assert_eq!(failed_work.projections_completed, 249);
    assert_eq!(failed_work.tokens_committed, 0);
    assert_eq!(snapshot(&session), before);
    assert_float_bits(session.logits().unwrap(), &expected);
    assert!(matches!(
        session.forward_observed(&[7], &AtomicBool::new(true), &mut |_| Ok(())),
        Err(Error::Cancelled)
    ));
    assert_eq!(snapshot(&session), before);
    assert_float_bits(session.logits().unwrap(), &expected);
    session.reset();
    let measure = Measurement::start();
    let result = session.forward(&[1]);
    let retry_allocations = measure.finish();
    result.unwrap();
    assert_eq!(retry_allocations, 0);
    assert_float_bits(session.logits().unwrap(), &expected);
    assert_eq!(snapshot(&session), before);
    let endpoint = Instant::now();
    let mut source_file = File::open(directory.join("model.safetensors")).unwrap();
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    let mut source_bytes = 0u64;
    loop {
        let n = source_file.read(&mut buffer).unwrap();
        if n == 0 {
            break;
        }
        source_bytes += n as u64;
        assert!(source_bytes <= source::CONTAINER_BYTES);
        hash.update(&buffer[..n]);
    }
    assert_eq!(source_bytes, source::CONTAINER_BYTES);
    let source_endpoint_sha256 = format!("{:x}", hash.finalize());
    assert_eq!(source_endpoint_sha256, source::CONTAINER_SHA256);
    let source_endpoint_ns = endpoint.elapsed().as_nanos();
    println!(
        "QWEN35_9B_GRAPH_REPORT {}",
        serde_json::json!({"source_sha256":source::CONTAINER_SHA256,"source_endpoint_sha256":source_endpoint_sha256,"source_endpoint_ns":source_endpoint_ns,"distinct_head":true,"container_bytes":source::CONTAINER_BYTES,"config_sha256":source::CONFIG_SHA256,"header_sha256":source::HEADER_SHA256,"tokens":[1],"position_offset":0,"capacity":512,"chunk":32,"memory":session.memory(),"work":work,"capture_bytes":offset,"arrays":arrays,"load_ns":load_ns,"bind_ns":bind_ns,"observed_forward_ns":observed_ns,"plain_forward_ns":plain_ns,"warmed_rust_allocations":allocations,"late_head_failure_work":failed_work,"late_head_failure_prefix_preserved":true,"cancelled_prefix_preserved":true,"retry_bits_equal":true,"retry_warmed_rust_allocations":retry_allocations,"greedy_lowest_id":greedy(session.logits().unwrap()),"first_token_executed":true,"independent_real_source_fidelity_qualified":false,"quality_qualified":false,"managed_adoption_qualified":false,"timing_scope":"CPU raw-input capture/replay and late-head failure/cancel/reset; no controlled performance or independent fidelity claim"})
    );
}
