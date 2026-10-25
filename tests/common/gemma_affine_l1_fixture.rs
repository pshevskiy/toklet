//! Immutable pre-target inputs/dots and explicit constant-row packed matrices.
use super::engine;
use engine::mlx::affine::{Layout, MetadataDtype};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub struct Case {
    pub reference: Value,
    pub layout: Layout,
    pub input: Vec<f32>,
    pub codes: Vec<u8>,
    pub scales: Vec<u8>,
    pub biases: Vec<u8>,
}
fn pin(bytes: &[u8], expected: &str) {
    assert_eq!(format!("{:x}", Sha256::digest(bytes)), expected);
}
pub fn cases() -> Vec<Case> {
    let metadata = include_bytes!("../fixtures/gemma-metal-affine-l1-v1/math.json");
    let binary = include_bytes!("../fixtures/gemma-metal-affine-l1-v1/math.bin");
    let manifest = include_bytes!("../fixtures/gemma-metal-affine-l1-v1/manifest.json");
    pin(
        metadata,
        "24043c085202e0ef90f558f67e7b53f63b83f1f0c84cd34bd0f9574855efb967",
    );
    pin(
        binary,
        "81c0f8522b4100d4e34cf23b9d8b7dfa7c9972cf326fa496f9f39f75b458b63a",
    );
    pin(
        manifest,
        "deedb1dc5c1dfaed7a50b1b3a97049f280fda3b7535412de62445dbb0747053e",
    );
    let metadata: Value = serde_json::from_slice(metadata).unwrap();
    assert_eq!(metadata["format"], "toklet-gemma-affine-l1-math-v1");
    assert_eq!(
        metadata["matrix_representation"],
        "packed4/group64/BF16; all codes0, scales0, each output row bias equals weights[row]"
    );
    let mut end = 0;
    let cases: Vec<_> = metadata["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|reference| {
            let k = reference["width"].as_u64().unwrap() as usize;
            let batch = reference["batch"].as_u64().unwrap() as usize;
            let weights = reference["weights"].as_array().unwrap();
            let layout = Layout::new(weights.len(), k, 4, 64, MetadataDtype::BF16).unwrap();
            let array = &reference["arrays"]["input"];
            assert_eq!(array["dtype"], "F32");
            assert_eq!(array["shape"], serde_json::json!([batch, k]));
            let offset = array["offset"].as_u64().unwrap() as usize;
            let bytes = array["bytes"].as_u64().unwrap() as usize;
            assert_eq!(offset, end);
            assert_eq!(bytes, batch * k * 4);
            end = offset.checked_add(bytes).unwrap();
            let body = &binary[offset..end];
            pin(body, array["sha256"].as_str().unwrap());
            let input: Vec<_> = body
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            assert!(input.iter().all(|v| v.is_finite()));
            let mut biases = Vec::with_capacity(layout.metadata_bytes());
            for value in weights {
                let value = value.as_f64().unwrap();
                let weight = value as f32;
                assert_eq!(f64::from(weight), value);
                assert_eq!(weight.to_bits() & 65535, 0);
                for _ in 0..k / 64 {
                    biases.extend_from_slice(&((weight.to_bits() >> 16) as u16).to_le_bytes());
                }
            }
            assert_eq!(biases.len(), layout.metadata_bytes());
            Case {
                reference: reference.clone(),
                layout,
                input,
                codes: vec![0; layout.weight_bytes()],
                scales: vec![0; layout.metadata_bytes()],
                biases,
            }
        })
        .collect();
    assert_eq!(cases.len(), 19);
    assert_eq!(end, binary.len());
    cases
}
