use std::{fs, path::PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

pub struct Fixture {
    root: PathBuf,
    pub manifest: Value,
}

impl Fixture {
    pub fn tiny() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-v1");
        let manifest = serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap())
            .expect("tiny fixture manifest should be valid JSON");
        Self { root, manifest }
    }

    pub fn bytes(&self, name: &str) -> Vec<u8> {
        let metadata = &self.manifest["arrays"][name];
        let bytes = fs::read(self.root.join(metadata["file"].as_str().unwrap())).unwrap();
        assert_eq!(bytes.len() as u64, metadata["bytes"].as_u64().unwrap());
        let elements = self
            .shape(name)
            .iter()
            .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension));
        assert_eq!(
            elements.and_then(|count| count.checked_mul(4)),
            Some(bytes.len())
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            metadata["sha256"].as_str().unwrap(),
            "hash mismatch for {name}"
        );
        bytes
    }

    pub fn shape(&self, name: &str) -> Vec<usize> {
        self.manifest["arrays"][name]["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|dimension| usize::try_from(dimension.as_u64().unwrap()).unwrap())
            .collect()
    }

    pub fn f32(&self, name: &str) -> Vec<f32> {
        assert_eq!(self.manifest["arrays"][name]["dtype"], "f32le");
        self.bytes(name)
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    pub fn u32(&self, name: &str) -> Vec<u32> {
        assert_eq!(self.manifest["arrays"][name]["dtype"], "u32le");
        self.bytes(name)
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }
}
