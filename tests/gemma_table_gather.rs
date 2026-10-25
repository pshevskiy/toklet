//! Atomic multirow gathers over authenticated pre-target original packed bytes.
use toklet as engine;
#[path = "common/gemma_table_fixture.rs"]
mod fixture;
use fixture::{fixture, floats, matrix, number};

fn same_bits(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[test]
fn frozen_gather_preserves_all_values_order_duplicates_and_zero_signs() {
    let data = fixture();
    let mut values = 0;
    let mut negative_zero = 0;
    for case in data["cases"].as_array().unwrap() {
        let matrix = matrix(case);
        let ids: Vec<_> = case["gather_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(number)
            .collect();
        let expected = floats(&case["arrays"]["gather"]);
        let mut actual = vec![42.; expected.len()];
        matrix.gather_rows(&ids, &mut actual).unwrap();
        same_bits(&actual, &expected);
        values += actual.len();
        negative_zero += actual.iter().filter(|v| v.to_bits() == 0x80000000).count();
        actual.fill(42.);
        matrix.gather_rows(&ids, &mut actual).unwrap();
        same_bits(&actual, &expected);
    }
    assert_eq!((values, negative_zero), (960, 192));
}

#[test]
fn one_and_thirty_two_selected_rows_match_frozen_repeated_reordered_values() {
    let data = fixture();
    for case in data["cases"].as_array().unwrap() {
        let matrix = matrix(case);
        let k = matrix.layout().columns();
        let ids: Vec<_> = case["gather_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(number)
            .collect();
        let frozen = floats(&case["arrays"]["gather"]);
        for count in [1, 32] {
            let choices: Vec<_> = (0..count).map(|i| (i * 3 + 4) % 5).collect();
            let selected: Vec<_> = choices.iter().map(|&i| ids[i]).collect();
            let expected: Vec<_> = choices
                .iter()
                .flat_map(|&i| frozen[i * k..(i + 1) * k].iter().copied())
                .collect();
            let mut output = vec![42.; count * k];
            matrix.gather_rows(&selected, &mut output).unwrap();
            same_bits(&output, &expected);
        }
    }
}

#[test]
fn invalid_request_leaves_entire_destination_untouched_and_recovers() {
    let data = fixture();
    for case in data["cases"].as_array().unwrap() {
        let matrix = matrix(case);
        let n = matrix.layout().rows();
        let k = matrix.layout().columns();
        for (ids, length) in [
            (vec![], 0),
            (vec![0; 33], 33 * k),
            (vec![0, 1, n, 0], 4 * k),
            (vec![0, 1, usize::MAX, 0], 4 * k),
            (vec![0, 1], 2 * k - 1),
            (vec![0, 1], 2 * k + 1),
        ] {
            let mut output = vec![-0.; length];
            let before = output.clone();
            assert_eq!(
                matrix.gather_rows(&ids, &mut output),
                Err(engine::mlx::affine::Error(
                    "MLX affine gather destination or indices mismatch"
                ))
            );
            same_bits(&output, &before);
        }
        let ids: Vec<_> = case["gather_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(number)
            .collect();
        let mut output = vec![42.; ids.len() * k];
        matrix.gather_rows(&ids, &mut output).unwrap();
        same_bits(&output, &floats(&case["arrays"]["gather"]));
    }
}

#[test]
fn later_overflow_preserves_whole_destination_then_valid_selection_succeeds() {
    let data = fixture();
    let case = &data["overflow"];
    let matrix = matrix(case);
    let before = floats(&case["arrays"]["destination"]);
    let mut output = before.clone();
    let ids: Vec<_> = case["gather_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(number)
        .collect();
    assert_eq!(
        matrix.gather_rows(&ids, &mut output),
        Err(engine::mlx::affine::Error(
            "MLX affine row reconstruction overflow"
        ))
    );
    same_bits(&output, &before);
    matrix.gather_rows(&[1, 0], &mut output[..128]).unwrap();
    same_bits(&output[..128], &[0.; 128]);
    same_bits(&output[128..], &before[128..]);
}
