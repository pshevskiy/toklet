//! Scalar reconstruction of frozen L1 witnesses; no Apple GPU/download needed.
use toklet as engine;
#[path = "common/gemma_affine_l1_fixture.rs"]
mod fixture;
use engine::mlx::affine::Matrix;

#[test]
fn frozen_l1_scalar_ordered_and_mathematical_subset() {
    let mut outputs = 0;
    let mut admitted = 0;
    for case in fixture::cases() {
        let matrix =
            Matrix::from_le_bytes(case.layout, &case.codes, &case.scales, &case.biases).unwrap();
        let mut actual = vec![0.; case.layout.rows()];
        for (token, row) in case.input.chunks_exact(case.layout.columns()).enumerate() {
            matrix.matvec(row, &mut actual).unwrap();
            for (out, &value) in actual.iter().enumerate() {
                let index = token * case.layout.rows() + out;
                let ordered = case.reference["ordered_fp64"][index].as_f64().unwrap();
                assert_eq!(
                    value.to_bits(),
                    (ordered as f32).to_bits(),
                    "{}",
                    case.reference["name"]
                );
                outputs += 1;
                if case.reference["combined_admitted"] == true {
                    let ideal = case.reference["mathematical_fp64"][index].as_f64().unwrap();
                    assert!((f64::from(value) - ideal).abs() <= 1e-5 + 1e-4 * ideal.abs());
                    admitted += 1;
                }
            }
        }
    }
    assert_eq!(outputs, 47);
    assert_eq!(admitted, 42);
}
