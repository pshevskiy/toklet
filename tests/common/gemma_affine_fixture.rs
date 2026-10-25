//! Shared test-only, authenticated new/previous packed projection fixtures.
//! Previous ordered outputs are already rounded FP32, promoted for comparison;
//! their ideal affine reference is historical, not a new exact-rational oracle.
use super::engine;
use engine::mlx::affine::{Layout, MetadataDtype};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn pinned(raw: &[u8], expected: &str) {
    assert_eq!(format!("{:x}", Sha256::digest(raw)), expected);
}
fn part<'a>(raw: &'a [u8], case: &Value, name: &str) -> &'a [u8] {
    let spec = &case["arrays"][name];
    let offset = spec["offset"].as_u64().unwrap() as usize;
    let count = spec["bytes"].as_u64().unwrap() as usize;
    let bytes = &raw[offset..offset.checked_add(count).unwrap()];
    pinned(bytes, spec["sha256"].as_str().unwrap());
    bytes
}
pub struct Fixture {
    pub cases: Vec<Value>,
    raw: &'static [u8],
}
impl Fixture {
    pub fn bytes(&self, case: &Value, name: &str) -> &[u8] {
        part(self.raw, case, name)
    }
    pub fn input(&self, case: &Value) -> Vec<f32> {
        self.bytes(case, "input")
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
    pub fn layout(&self, case: &Value) -> Layout {
        Layout::new(
            case["outputs"].as_u64().unwrap() as usize,
            case["width"].as_u64().unwrap() as usize,
            4,
            64,
            MetadataDtype::BF16,
        )
        .unwrap()
    }
}
pub fn fixtures() -> [Fixture; 2] {
    let meta = include_bytes!("../fixtures/gemma-metal-affine-v1/math.json");
    let raw = include_bytes!("../fixtures/gemma-metal-affine-v1/math.bin");
    let manifest = include_bytes!("../fixtures/gemma-metal-affine-v1/manifest.json");
    pinned(
        meta,
        "61e78bfd5dfbc9131b2d6804bb7f8c8ff228cce249716c690ed3df90d40a0c53",
    );
    pinned(
        raw,
        "b7c4d03b8095368ba2c41abed1429b215ff5d6a4f1b77bf3dade338fd677817e",
    );
    pinned(
        manifest,
        "361ec669193df298ec4eef71eb9624d536cbd8d1db59b6703ddab580e4949457",
    );
    let new: Value = serde_json::from_slice(meta).unwrap();
    let receipt =
        include_bytes!("../../benchmarks/results/2026-10-25-gemma-packed-reference-v1.json");
    pinned(
        receipt,
        "a8349f8cf8ba1f30aac6d8480ff014160b68135b16c2574a8fd9ff7bafea5b19",
    );
    let accepted: Value = serde_json::from_slice(receipt).unwrap();
    let old_meta = include_bytes!("../fixtures/gemma-packed-v1/math.json");
    let old_raw = include_bytes!("../fixtures/gemma-packed-v1/math.bin");
    pinned(
        old_meta,
        accepted["fixture_manifest_sha256"].as_str().unwrap(),
    );
    pinned(old_raw, accepted["fixture_binary_sha256"].as_str().unwrap());
    let old: Value = serde_json::from_slice(old_meta).unwrap();
    let cases=old["cases"].as_array().unwrap().iter().map(|case| {
        let math:Vec<f64>=part(old_raw,case,"projection_ideal").chunks_exact(8).map(|b|f64::from_le_bytes(b.try_into().unwrap())).collect();
        let ordered:Vec<f64>=part(old_raw,case,"projection_ordered_f32").chunks_exact(4).map(|b|f64::from(f32::from_le_bytes(b.try_into().unwrap()))).collect();
        json!({"name":format!("previous-width-{}",case["width"]),"width":case["width"],"outputs":3,"batch":2,
            "constructor":{"admitted":true},"input_admission":{"admitted":true},
            "math_fp64":math,"ordered_fp64":ordered,
            "reference_contract":{"math":"previous affine-ideal FP64","ordered":"previous output-rounded FP32 promoted for comparison"},
            "arrays":{"weight":case["arrays"]["weight"],"scales":case["arrays"]["scales"],"biases":case["arrays"]["biases"],"input":case["arrays"]["input_f32"]}})
    }).collect();
    let mut fresh = new["cases"].as_array().unwrap().clone();
    for case in &mut fresh {
        case["reference_contract"] = json!({"math":"exact Fraction dot over once-reconstructed FP32 weights converted to FP64","ordered":"ordered FP64 dot over once-reconstructed FP32 weights"});
    }
    [
        Fixture { cases: fresh, raw },
        Fixture {
            cases,
            raw: old_raw,
        },
    ]
}
