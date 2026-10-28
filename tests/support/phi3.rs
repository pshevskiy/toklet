//! Frozen independent Phi source arrays shared by portable and explicit GPU tests.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

const MANIFEST_SHA: &str = "bd297f60a0b6e280ee3495688b31c2c4042afc124b10735c4539262ead01ef26";

pub fn open() -> (PathBuf, Value, Vec<u8>) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mlx-phi3-v1");
    let raw = fs::read(root.join("manifest.json")).unwrap();
    assert!(raw.len() <= 1024 * 1024);
    assert_eq!(format!("{:x}", Sha256::digest(&raw)), MANIFEST_SHA);
    let manifest: Value = serde_json::from_slice(&raw).unwrap();
    assert_eq!(manifest["format"], "toklet-phi3-tiny-graph-v1");
    assert_eq!(manifest["stage_groups"], 195);
    assert_eq!(manifest["stored_arrays"], 585);
    assert_eq!(manifest["native_diagnostic_stages_per_case"], 35);
    assert_eq!(
        manifest["prospective_policy_sha256"],
        "f2eba6a4a8aa5c00020ddde373cdc4adc9365686cf62f54f6949359a593e1613"
    );
    assert_eq!(
        manifest["producer"]["sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../../tools/reference/phi3_graph.py"))
        )
    );
    assert_eq!(manifest["files"].as_object().unwrap().len(), 5);
    for (name, item) in manifest["files"].as_object().unwrap() {
        let bytes = fs::read(root.join(name)).unwrap();
        assert_eq!(bytes.len() as u64, item["bytes"]);
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), item["sha256"]);
    }
    let arrays = fs::read(root.join("arrays.bin")).unwrap();
    assert!(arrays.len() <= 16 * 1024 * 1024);
    (root, manifest, arrays)
}

pub fn values(raw: &[u8], descriptor: &Value) -> Vec<f32> {
    assert_eq!(descriptor["file"], "arrays.bin");
    let start = usize::try_from(descriptor["offset"].as_u64().unwrap()).unwrap();
    let length = usize::try_from(descriptor["bytes"].as_u64().unwrap()).unwrap();
    let bytes = raw.get(start..start.checked_add(length).unwrap()).unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(bytes)), descriptor["sha256"]);
    let result: Vec<f32> = match descriptor["dtype"].as_str().unwrap() {
        "f16le" => {
            assert!(bytes.len().is_multiple_of(2));
            bytes
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes(b.try_into().unwrap()).to_f32())
                .collect()
        }
        "f32le" => {
            assert!(bytes.len().is_multiple_of(4));
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        }
        _ => panic!("unexpected source dtype"),
    };
    assert!(result.iter().all(|v| v.is_finite()));
    assert_eq!(
        result.len(),
        descriptor["shape"]
            .as_array()
            .unwrap()
            .iter()
            .try_fold(1usize, |n, v| n.checked_mul(v.as_u64().unwrap() as usize))
            .unwrap()
    );
    result
}
