//! Native immutable row views against pre-target bytes and rounded scalar dots.
//! No model files or Apple GPU are required by these tests.
use serde_json::Value;
use sha2::{Digest, Sha256};
use toklet::mlx::affine::{Layout, Matrix, MetadataDtype};

const DATA: &[u8] = include_bytes!("fixtures/gemma-table-math-v1/math.bin");
fn pin(bytes: &[u8], wanted: &str) {
    assert_eq!(format!("{:x}", Sha256::digest(bytes)), wanted);
}
fn fixture() -> Value {
    let metadata = include_bytes!("fixtures/gemma-table-math-v1/math.json");
    let manifest = include_bytes!("fixtures/gemma-table-math-v1/manifest.json");
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
fn number(value: &Value) -> usize {
    usize::try_from(value.as_u64().unwrap()).unwrap()
}
fn array(entry: &Value) -> &'static [u8] {
    let start = number(&entry["offset"]);
    let end = start.checked_add(number(&entry["bytes"])).unwrap();
    let raw = &DATA[start..end];
    pin(raw, entry["sha256"].as_str().unwrap());
    raw
}
fn floats(entry: &Value) -> Vec<f32> {
    assert_eq!(entry["dtype"], "F32");
    array(entry)
        .chunks_exact(4)
        .map(|raw| f32::from_le_bytes(raw.try_into().unwrap()))
        .collect()
}
fn matrix(case: &Value) -> Matrix<'static> {
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
#[test]
fn frozen_row_views_gather_all_selected_values_and_zero_signs() {
    let data = fixture();
    let mut values = 0;
    let mut negative_zeros = 0;
    for case in data["cases"].as_array().unwrap() {
        let matrix = matrix(case);
        let k = matrix.layout().columns();
        let expected = floats(&case["arrays"]["gather"]);
        let mut actual = vec![42.; k];
        for (selected, id) in case["gather_ids"].as_array().unwrap().iter().enumerate() {
            let row = number(id);
            let view = matrix.row_range(row..row + 1).unwrap();
            view.copy_row(0, &mut actual).unwrap();
            for (&value, &reference) in actual
                .iter()
                .zip(&expected[selected * k..(selected + 1) * k])
            {
                assert_eq!(value.to_bits(), reference.to_bits());
                negative_zeros += usize::from(value.to_bits() == 0x8000_0000);
                values += 1;
            }
        }
    }
    assert_eq!(values, 960);
    assert_eq!(negative_zeros, 192);
}
#[test]
fn frozen_partitioned_row_views_preserve_all_scalar_outputs() {
    let data = fixture();
    let mut values = 0;
    for case in data["cases"].as_array().unwrap() {
        let matrix = matrix(case);
        let input = floats(&case["arrays"]["input"]);
        let expected = floats(&case["arrays"]["output_f32"]);
        let rows = matrix.layout().rows();
        for partition in case["partitions"].as_array().unwrap() {
            let mut combined = vec![42_f32; expected.len()];
            for chunk in partition["chunks"].as_array().unwrap() {
                let first = number(&chunk["first_row"]);
                let count = number(&chunk["rows"]);
                let view = matrix.row_range(first..first + count).unwrap();
                for (token, row) in input.chunks_exact(matrix.layout().columns()).enumerate() {
                    let destination =
                        &mut combined[token * rows + first..token * rows + first + count];
                    view.matvec(row, destination).unwrap();
                }
            }
            for (actual, reference) in combined.iter().zip(&expected) {
                assert_eq!(actual.to_bits(), reference.to_bits());
                values += 1;
            }
        }
    }
    assert_eq!(values, 360);
}
#[test]
fn invalid_and_nested_row_ranges_preserve_original_views() {
    let data = fixture();
    for case in data["cases"].as_array().unwrap() {
        let parent = matrix(case);
        let rows = parent.layout().rows();
        for range in [
            0..0,
            rows..rows,
            rows..rows + 1,
            std::ops::Range { start: 7, end: 6 },
            0..usize::MAX,
            usize::MAX..usize::MAX,
        ] {
            assert_eq!(
                parent.row_range(range).err().unwrap().0,
                "MLX affine row range out of bounds"
            );
        }
        let whole = parent.row_range(0..rows).unwrap();
        let nested = whole.row_range(1..rows).unwrap().row_range(1..2).unwrap();
        let mut actual = vec![42.; parent.layout().columns()];
        nested.copy_row(0, &mut actual).unwrap();
        let expected = floats(&case["arrays"]["reconstructed"]);
        for (&value, &reference) in actual
            .iter()
            .zip(&expected[2 * actual.len()..3 * actual.len()])
        {
            assert_eq!(value.to_bits(), reference.to_bits());
        }
    }
}
