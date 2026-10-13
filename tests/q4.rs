use sha2::{Digest, Sha256};
use toklet::q4::Matrix;

const PACKED: &[u8] = include_bytes!("fixtures/q4/rows.q4");

#[test]
fn python_artifact_roundtrips_and_reconstructs_manual_values() {
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/q4/manifest.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(PACKED)),
        manifest["packed_sha256"].as_str().unwrap()
    );
    let matrix = Matrix::parse(PACKED).unwrap();
    assert_eq!(
        (matrix.rows(), matrix.columns(), matrix.row_stride()),
        (2, 65, 128)
    );
    assert_eq!(matrix.as_bytes(), PACKED);
    for (row, values) in manifest["reconstructed_rows"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        for (column, expected) in values.as_array().unwrap().iter().enumerate() {
            assert_eq!(
                matrix.value(row, column).unwrap(),
                expected.as_f64().unwrap() as f32
            );
        }
    }
    assert!(matrix.value(2, 0).is_err());
    assert!(matrix.value(0, 65).is_err());
    let mut result = [99.0; 2];
    let mut input = [1.0; 65];
    input[64] = 2.0;
    matrix.matvec(&input, &mut result).unwrap();
    // Four repetitions of [-2, -1.5, ..., 5.5] sum to 112; tail adds 6.5.
    assert_eq!(result, [118.5, 0.0]);
}

#[test]
fn rejects_bad_headers_overflow_metadata_and_noncanonical_padding() {
    for offset in [0, 8, 12, 32, 40, 48, 104, 136] {
        let mut corrupt = PACKED.to_vec();
        corrupt[offset] ^= 1;
        assert!(Matrix::parse(&corrupt).is_err(), "offset {offset}");
    }
    for offset in [16, 24] {
        for dimension in [0_u64, u64::MAX] {
            let mut corrupt = PACKED.to_vec();
            corrupt[offset..offset + 8].copy_from_slice(&dimension.to_le_bytes());
            assert!(Matrix::parse(&corrupt).is_err());
        }
    }
    for (offset, bits) in [(64, 0x8000_u16), (64, 0xbc00), (64, 0x7c00), (66, 0x7e00)] {
        let mut corrupt = PACKED.to_vec();
        corrupt[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
        assert!(Matrix::parse(&corrupt).is_err());
    }
    let mut corrupt = PACKED.to_vec();
    corrupt[196] = 1; // zero-scale constant group may only contain zero codes
    assert!(Matrix::parse(&corrupt).is_err());
    for length in [0, 63, 64, PACKED.len() - 1] {
        assert!(Matrix::parse(&PACKED[..length]).is_err());
    }
    let mut corrupt = PACKED.to_vec();
    corrupt.push(0);
    assert!(Matrix::parse(&corrupt).is_err());
}

#[test]
fn matvec_errors_preserve_all_output_and_half_extrema_use_fp32_reconstruction() {
    let matrix = Matrix::parse(PACKED).unwrap();
    let mut output = [42.0; 2];
    assert!(matrix.matvec(&[1.0], &mut output).is_err());
    assert_eq!(output, [42.0; 2]);
    let mut input = [1.0; 65];
    input[50] = f32::NAN;
    assert!(matrix.matvec(&input, &mut output).is_err());
    assert_eq!(output, [42.0; 2]);
    input.fill(f32::MAX);
    assert!(matrix.matvec(&input, &mut output).is_err());
    assert_eq!(output, [42.0; 2]);
    let mut extreme = PACKED.to_vec();
    extreme[64..66].copy_from_slice(&half::f16::from_f32(8736.0).to_bits().to_le_bytes());
    extreme[66..68].copy_from_slice(&half::f16::from_f32(-65504.0).to_bits().to_le_bytes());
    let matrix = Matrix::parse(&extreme).unwrap();
    assert_eq!(matrix.value(0, 15).unwrap(), 65536.0);
}
