//! Standalone owned-process storage measurement, linked against a pinned rlib.
//! Compile-time source selection keeps old/new4B workloads exactly identical.
use std::{fs::File, io::Read, path::PathBuf, time::Instant};
use toklet::mlx::qwen35::ModalityPolicy;
#[cfg(toklet_9b)]
use toklet::mlx::qwen35::nine_b::payload as engine;
#[cfg(not(toklet_9b))]
use toklet::mlx::qwen35::payload as engine;

fn main() {
    let folder = PathBuf::from(std::env::args_os().nth(1).expect("model directory"));
    let mut config = Vec::new();
    let f = File::open(folder.join("config.json")).unwrap();
    assert!(f.metadata().unwrap().is_file());
    assert!(f.metadata().unwrap().len() <= 65536);
    f.take(65537).read_to_end(&mut config).unwrap();
    assert!(config.len() <= 65536);
    for ordinal in 0..13 {
        let file = File::open(folder.join("model.safetensors")).unwrap();
        assert!(file.metadata().unwrap().is_file());
        let start = Instant::now();
        let store = engine::load(
            &config,
            file,
            engine::LoadOptions {
                memory_limit_bytes: 6 * 1024 * 1024 * 1024,
                modality: ModalityPolicy::TextOnly,
            },
        )
        .unwrap();
        let load_ns = start.elapsed().as_nanos();
        let r = store.report();
        let bytes = r.source_payload_bytes;
        let capacity = r.retained_tensor_capacity_bytes;
        let admitted = r.admitted_load_bytes;
        let matrices = r.packed_matrices;
        let vectors = r.fp32_vectors;
        let convolutions = r.fp32_convolutions;
        let container_bytes = store.identity().container_bytes();
        #[cfg(toklet_9b)]
        assert_eq!(
            (
                matrices,
                vectors,
                convolutions,
                container_bytes,
                capacity,
                admitted
            ),
            (250, 153, 24, 5_038_160_953, 5_040_162_816, 5_107_402_752)
        );
        #[cfg(not(toklet_9b))]
        assert_eq!(
            (
                matrices,
                vectors,
                convolutions,
                container_bytes,
                capacity,
                admitted
            ),
            (249, 153, 24, 2_367_237_149, 2_369_040_384, 2_436_280_320)
        );
        drop(store);
        let load_and_drop_ns = start.elapsed().as_nanos();
        println!(
            "STORAGE_SAMPLE {{\"ordinal\":{ordinal},\"warmup\":{},\"load_ns\":{load_ns},\"load_and_drop_ns\":{load_and_drop_ns},\"source_payload_bytes\":{bytes},\"container_bytes\":{container_bytes},\"retained_tensor_capacity_bytes\":{capacity},\"admitted_load_bytes\":{admitted},\"packed_matrices\":{matrices},\"fp32_vectors\":{vectors},\"fp32_convolutions\":{convolutions},\"graph_executed\":false,\"runtime_qualified\":false}}",
            ordinal < 3
        );
    }
}
