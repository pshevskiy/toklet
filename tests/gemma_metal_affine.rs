//! Ordinary CPU tests authenticate both packed fixtures; no GPU/download needed.
use toklet as engine;
#[path = "common/gemma_affine_fixture.rs"]
mod fixture;
use engine::mlx::affine::Matrix;

#[test]
fn scalar_ordered_reference_and_admitted_mathematical_subset() {
    let mut ordered_count = 0;
    let mut mathematical_count = 0;
    for fixture in fixture::fixtures() {
        for case in &fixture.cases {
            let layout = fixture.layout(case);
            let matrix = Matrix::from_le_bytes(
                layout,
                fixture.bytes(case, "weight"),
                fixture.bytes(case, "scales"),
                fixture.bytes(case, "biases"),
            )
            .unwrap();
            let input = fixture.input(case);
            let tokens = case["batch"].as_u64().unwrap() as usize;
            let mut output = vec![42.; tokens * layout.rows()];
            for (row, out) in input
                .chunks_exact(layout.columns())
                .zip(output.chunks_exact_mut(layout.rows()))
            {
                matrix.matvec(row, out).unwrap();
            }
            let admitted = case["constructor"]["admitted"] == true
                && case["input_admission"]["admitted"] == true;
            for (i, &actual) in output.iter().enumerate() {
                let ordered = case["ordered_fp64"][i].as_f64().unwrap();
                assert_eq!(
                    actual.to_bits(),
                    (ordered as f32).to_bits(),
                    "{}",
                    case["name"]
                );
                ordered_count += 1;
                if admitted {
                    let expected = case["math_fp64"][i].as_f64().unwrap();
                    assert!((f64::from(actual) - expected).abs() <= 1e-5 + 1e-4 * expected.abs());
                    mathematical_count += 1;
                }
            }
            if case["name"] == "ordered-exact-conflict" {
                assert_eq!(output, [0.]);
                assert_eq!(case["math_fp64"][0], 1.);
                assert!(!admitted);
            }
        }
    }
    assert_eq!(ordered_count, 330);
    assert_eq!(mathematical_count, 324);
}
