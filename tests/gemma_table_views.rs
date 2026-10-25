//! Native immutable row views against pre-target bytes and rounded scalar dots.
//! No model files or Apple GPU are required by these tests.
use toklet as engine;
#[path = "common/gemma_table_fixture.rs"]
mod fixture;
use fixture::{fixture, floats, matrix, number};
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
