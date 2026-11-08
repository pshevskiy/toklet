//! Explicit allocator witness for the actual source's admitted reset buffers.
use super::*;
use crate::mlx::qwen35::{self as source, payload};
use std::{fs::File, io::Read, path::PathBuf};

#[test]
#[ignore = "explicit pinned Qwen3.5-4B warmed reset allocator witness"]
fn actual_source_warmed_reset_allocations() {
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
    let store = payload::load(
        &config,
        File::open(directory.join("model.safetensors")).unwrap(),
        payload::LoadOptions {
            memory_limit_bytes: MEMORY_LIMIT - 128 * 1024 * 1024,
            modality: source::ModalityPolicy::TextOnly,
        },
    )
    .unwrap();
    let model = Model::from_cpu_store(&store).unwrap();
    let mut session = model
        .session(Options {
            capacity: 512,
            chunk: 32,
            position_offset: 37,
            memory_limit_bytes: MEMORY_LIMIT - 128 * 1024 * 1024,
        })
        .unwrap();
    let memory = session.memory();
    for _ in 0..3 {
        session.forward(&[1]).unwrap();
        assert!(session.logits().is_some());
        let measure = Measurement::start();
        session.reset();
        assert_eq!(measure.finish(), 0);
        assert_eq!(session.memory(), memory);
        assert!(session.history().is_empty() && session.logits().is_none());
        assert_eq!(session.work(), Work::default());
    }
    println!(
        "QWEN35_RESET_REPORT {}",
        serde_json::json!({"source_sha256":source::CONTAINER_SHA256,
        "capacity":512,"chunk":32,"position_offset":37,"repetitions":3,"allocations":[0,0,0],
        "history_empty":true,"logits_absent":true,"work_idle":true,"memory":memory,
        "model_or_quality_adoption_qualified":false})
    );
}
