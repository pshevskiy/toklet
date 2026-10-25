//! Authenticated immutable pre-target arrays, shared by CPU/GPU diagnostics.
use super::engine;
use engine::mlx::affine::{Layout, Matrix, MetadataDtype};
use serde_json::Value;
use sha2::{Digest, Sha256};

const DATA: &[u8] = include_bytes!("../fixtures/gemma-table-math-v1/math.bin");
fn pin(bytes: &[u8], wanted: &str) {
    assert_eq!(format!("{:x}", Sha256::digest(bytes)), wanted);
}
pub fn fixture() -> Value {
    let metadata = include_bytes!("../fixtures/gemma-table-math-v1/math.json");
    let manifest = include_bytes!("../fixtures/gemma-table-math-v1/manifest.json");
    pin(
        metadata,
        "9083f619b0a12202955156631e8b65e7824cb7120641edcad6b906d551aa721f",
    );
    pin(
        DATA,
        "cc34796740e66853ecccb950028ff994de37b2bc5a7e12a1cc9877b0ce3a549e",
    );
    pin(
        manifest,
        "36aeea18b3143e0024d33e097091ebf47d6fdde83beaf69784d1e2f498cc6683",
    );
    let data: Value = serde_json::from_slice(metadata).unwrap();
    assert_eq!(data["passed"], true);
    assert_eq!(data["binary_bytes"], DATA.len());
    data
}
pub fn number(value: &Value) -> usize {
    usize::try_from(value.as_u64().unwrap()).unwrap()
}
pub fn array(entry: &Value) -> &'static [u8] {
    let start = number(&entry["offset"]);
    let end = start.checked_add(number(&entry["bytes"])).unwrap();
    let raw = &DATA[start..end];
    pin(raw, entry["sha256"].as_str().unwrap());
    raw
}
pub fn floats(entry: &Value) -> Vec<f32> {
    assert_eq!(entry["dtype"], "F32");
    array(entry)
        .chunks_exact(4)
        .map(|raw| f32::from_le_bytes(raw.try_into().unwrap()))
        .collect()
}
pub fn matrix(case: &Value) -> Matrix<'static> {
    let layout = Layout::new(
        number(&case["rows"]),
        number(&case["columns"]),
        4,
        64,
        MetadataDtype::BF16,
    )
    .unwrap();
    let arrays = &case["arrays"];
    Matrix::from_le_bytes(
        layout,
        array(&arrays["codes"]),
        array(&arrays["scales"]),
        array(&arrays["biases"]),
    )
    .unwrap()
}
