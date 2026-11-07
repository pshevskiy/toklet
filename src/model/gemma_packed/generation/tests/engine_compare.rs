//! Explicit local candidate inspection; no managed model adoption is implied.
use std::{
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
};
fn bounded(path: &Path) -> Vec<u8> {
    let mut bytes = Vec::new();
    File::open(path)
        .unwrap()
        .take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .unwrap();
    assert!(bytes.len() <= 1024 * 1024);
    bytes
}
#[test]
#[ignore = "Explicit pinned uncensored MLX candidate cold metadata, without GPU/model allocation"]
fn uncensored_mlx_candidate_shape_inspection() {
    let directory = PathBuf::from(
        std::env::var_os("TOKLET_GEMMA_CANDIDATE_DIRECTORY").expect("verified pinned candidate"),
    );
    let config = bounded(&directory.join("config.json"));
    let index = bounded(&directory.join("model.safetensors.index.json"));
    let mut file = File::open(directory.join("model.safetensors")).unwrap();
    assert_eq!(file.metadata().unwrap().len(), 3_581_100_878);
    let mut prefix = [0; 8];
    file.read_exact(&mut prefix).unwrap();
    let length = usize::try_from(u64::from_le_bytes(prefix)).unwrap();
    assert!(length <= 1024 * 1024);
    let mut header = vec![0; length];
    file.read_exact(&mut header).unwrap();
    let inspected = crate::mlx::gemma4::inspect(
        &config,
        &index,
        &header,
        3_581_100_878,
        crate::mlx::gemma4::ModalityPolicy::TextOnly,
    );
    let report = serde_json::json!({"inspection_completed":true,"native_shape_compatible":inspected.is_ok(),
        "error":inspected.as_ref().err().map(ToString::to_string),"numerical_buffers_allocated":false,
        "payload_validation_required":true,"revision":"cde01efcb11f8a12e88a953b752fc16062aa1486",
        "scope":"Cold shape/schema compatibility only; crypto/execution/quality/adoption remain separate"});
    let output = PathBuf::from(
        std::env::var_os("TOKLET_GEMMA_CANDIDATE_INSPECTION").expect("fresh output file"),
    );
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    file.write_all(&serde_json::to_vec_pretty(&report).unwrap())
        .unwrap();
    println!("GEMMA_CANDIDATE_INSPECTION {report}");
}
