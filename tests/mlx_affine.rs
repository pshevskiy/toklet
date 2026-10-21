use half::{bf16, f16};
use sha2::{Digest, Sha256};
use toklet::mlx::affine::{Layout, Matrix, MetadataDtype};

fn metadata(dtype: MetadataDtype, values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| match dtype {
            MetadataDtype::F16 => f16::from_f32(v).to_le_bytes().to_vec(),
            MetadataDtype::BF16 => bf16::from_f32(v).to_le_bytes().to_vec(),
            MetadataDtype::F32 => v.to_le_bytes().to_vec(),
        })
        .collect()
}
fn hex(text: &str) -> Vec<u8> {
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}
fn all_code_pattern(bits: u8) -> Vec<u8> {
    // Literal unsigned codes 0..2^bits-1 in order; no bit-packing helper.
    match bits {
        2 => vec![0xe4],
        3 => vec![0x88, 0xc6, 0xfa],
        4 => vec![0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe],
        5 => hex("2088418a3928a9c59a7b30ca49abbd38ebcdbbff"),
        6 => hex(
            "40200c44611c48a22c4ce33c50244d54655d58a66d5ce77d60288e64699e68aaae6cebbe702ccf746ddf78aeef7cefff",
        ),
        8 => (0..=255).collect(),
        _ => unreachable!(),
    }
}

#[test]
fn every_unsigned_code_crosses_bytes_words_groups_and_rows() {
    for bits in [2u8, 3, 4, 5, 6, 8] {
        let pattern = all_code_pattern(bits);
        let row = pattern.repeat(256 * usize::from(bits) / 8 / pattern.len());
        let packed: Vec<u8> = row
            .iter()
            .copied()
            .chain(row.iter().map(|v| v ^ 255))
            .collect();
        for group in [32, 64, 128] {
            let layout = Layout::new(2, 256, bits, group, MetadataDtype::F32).unwrap();
            let scales = metadata(MetadataDtype::F32, &vec![1.0; 512 / group]);
            let biases = metadata(MetadataDtype::F32, &vec![0.0; 512 / group]);
            let matrix = Matrix::from_le_bytes(layout, &packed, &scales, &biases).unwrap();
            let mask = (1usize << bits) - 1;
            for c in 0..256 {
                assert_eq!(matrix.value(0, c).unwrap(), (c & mask) as f32);
                assert_eq!(matrix.value(1, c).unwrap(), (mask - (c & mask)) as f32);
            }
            assert_eq!(layout.weight_bytes(), packed.len());
            assert_eq!(layout.metadata_bytes(), scales.len());
            assert_eq!(
                layout.total_bytes(),
                packed.len() + scales.len() + biases.len()
            );
            assert_eq!(layout.row_bytes(), row.len());
            assert_eq!(layout.bits(), bits);
            assert_eq!(layout.group_size(), group);
            assert_eq!(matrix.layout(), layout);
            assert!(matrix.value(2, 0).is_err());
            assert!(matrix.value(0, 256).is_err());
        }
    }
}

#[test]
fn pinned_mlx_cpu_gpu_oracle_replays_without_python_or_apple_frameworks() {
    const RAW: &[u8] = include_bytes!("fixtures/mlx-affine-v1/arrays.bin");
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/mlx-affine-v1/manifest.json")).unwrap();
    assert_eq!(manifest["mlx_version"], "0.32.3");
    assert_eq!(manifest["bytes"].as_u64().unwrap() as usize, RAW.len());
    assert_eq!(format!("{:x}", Sha256::digest(RAW)), manifest["sha256"]);
    let array = |case: &serde_json::Value, name: &str| {
        let entry = &case["arrays"][name];
        let offset = entry["offset"].as_u64().unwrap() as usize;
        let size = entry["bytes"].as_u64().unwrap() as usize;
        let data = &RAW[offset..offset + size];
        assert_eq!(format!("{:x}", Sha256::digest(data)), entry["sha256"]);
        data
    };
    let mut inventory = std::collections::BTreeSet::new();
    let cases = manifest["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 54);
    for case in cases {
        let dtype = match case["metadata_dtype"].as_str().unwrap() {
            "F16" => MetadataDtype::F16,
            "BF16" => MetadataDtype::BF16,
            "F32" => MetadataDtype::F32,
            _ => panic!("fixture dtype"),
        };
        let bits = case["bits"].as_u64().unwrap() as u8;
        let group = case["group_size"].as_u64().unwrap() as usize;
        assert!(inventory.insert((bits, group, case["metadata_dtype"].as_str().unwrap())));
        let layout = Layout::new(2, 256, bits, group, dtype).unwrap();
        assert_eq!(layout.metadata_dtype(), dtype);
        let matrix = Matrix::from_le_bytes(
            layout,
            array(case, "weight"),
            array(case, "scales"),
            array(case, "biases"),
        )
        .unwrap();
        let expected = array(case, "reconstructed_f32");
        for row in 0..2 {
            for column in 0..256 {
                let i = (row * 256 + column) * 4;
                assert_eq!(
                    matrix.value(row, column).unwrap().to_le_bytes(),
                    expected[i..i + 4]
                );
            }
        }
        let input: Vec<f32> = array(case, "input_f32")
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let mut output = [99.0; 2];
        matrix.matvec(&input, &mut output).unwrap();
        let actual: Vec<u8> = output.into_iter().flat_map(f32::to_le_bytes).collect();
        assert_eq!(actual, array(case, "matvec_f32"));
        // These are separate native-dtype rounding witnesses, not the Rust
        // FP32 reconstruction oracle. Verify their identities, not equivalence.
        let cpu = array(case, "native_cpu");
        let gpu = array(case, "native_gpu");
        assert_eq!(
            cpu == gpu,
            case["native_dtype_cpu_gpu_equal"].as_bool().unwrap()
        );
    }
}

#[test]
fn rejects_invalid_shapes_parameters_and_size_overflow_before_borrowing() {
    for (rows, columns, bits, group) in [
        (0, 32, 4, 32),
        (1, 0, 4, 32),
        (1, 31, 4, 32),
        (1, 33, 4, 32),
        (1, 32, 0, 32),
        (1, 32, 1, 32),
        (1, 32, 7, 32),
        (1, 32, 9, 32),
        (1, 32, 4, 0),
        (1, 32, 4, 16),
        (1, 32, 4, 256),
        (usize::MAX, 128, 8, 128),
        (1, usize::MAX / 128 * 128, 8, 128),
        (isize::MAX as usize / 16, 32, 4, 32),
    ] {
        assert!(Layout::new(rows, columns, bits, group, MetadataDtype::F32).is_err());
    }
}

#[test]
fn byte_lengths_nonfinite_metadata_and_unaligned_slices_are_checked() {
    for dtype in [MetadataDtype::F16, MetadataDtype::BF16, MetadataDtype::F32] {
        let layout = Layout::new(2, 64, 4, 32, dtype).unwrap();
        let weights = vec![0x21; layout.weight_bytes()];
        let scales = metadata(dtype, &[0.5; 4]);
        let biases = metadata(dtype, &[0.0; 4]);
        for selected in 0..3 {
            for delta in [-1isize, 1] {
                let mut tensors = [weights.clone(), scales.clone(), biases.clone()];
                if delta < 0 {
                    tensors[selected].pop();
                } else {
                    tensors[selected].push(0);
                }
                assert!(
                    Matrix::from_le_bytes(layout, &tensors[0], &tensors[1], &tensors[2]).is_err()
                );
            }
        }
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut corrupt = scales.clone();
            corrupt[..dtype.bytes()].copy_from_slice(&metadata(dtype, &[invalid]));
            assert!(Matrix::from_le_bytes(layout, &weights, &corrupt, &biases).is_err());
            assert!(Matrix::from_le_bytes(layout, &weights, &scales, &corrupt).is_err());
        }
        let offset = |data: &[u8]| {
            let mut value = vec![0xff];
            value.extend_from_slice(data);
            value
        };
        let w = offset(&weights);
        let s = offset(&scales);
        let b = offset(&biases);
        let unaligned = Matrix::from_le_bytes(layout, &w[1..], &s[1..], &b[1..]).unwrap();
        assert_eq!(unaligned.value(0, 0).unwrap(), 0.5);
        assert_eq!(unaligned.value(0, 1).unwrap(), 1.0);
        assert_eq!(unaligned.value(1, 63).unwrap(), 1.0);
    }
}

#[test]
fn signed_zero_negative_and_subnormal_scales_do_not_use_toklet_q4_rules() {
    for dtype in [MetadataDtype::F16, MetadataDtype::BF16, MetadataDtype::F32] {
        let layout = Layout::new(2, 64, 4, 32, dtype).unwrap();
        let weights = vec![0xff; layout.weight_bytes()];
        let scales = metadata(dtype, &[-0.5, 0.25, 0.0, -0.0]);
        let biases = metadata(dtype, &[2.0, -1.0, 3.0, -0.0]);
        let matrix = Matrix::from_le_bytes(layout, &weights, &scales, &biases).unwrap();
        assert_eq!(matrix.value(0, 0).unwrap(), -5.5);
        assert_eq!(matrix.value(0, 63).unwrap(), 2.75);
        assert_eq!(matrix.value(1, 0).unwrap(), 3.0); // Zero scale permits nonzero codes.
        assert_eq!(matrix.value(1, 63).unwrap().to_bits(), (-0.0f32).to_bits());
        let layout = Layout::new(1, 32, 4, 32, dtype).unwrap();
        let weights = [0x11; 16];
        let scales = match dtype {
            MetadataDtype::F16 | MetadataDtype::BF16 => vec![1, 0],
            MetadataDtype::F32 => vec![1, 0, 0, 0],
        };
        let biases = vec![0; dtype.bytes()];
        let matrix = Matrix::from_le_bytes(layout, &weights, &scales, &biases).unwrap();
        let expected = match dtype {
            MetadataDtype::F16 => f32::from_bits(0x33800000),
            MetadataDtype::BF16 => f32::from_bits(0x00010000),
            MetadataDtype::F32 => f32::from_bits(1),
        };
        assert_eq!(matrix.value(0, 0).unwrap().to_bits(), expected.to_bits());
    }
}

#[test]
fn reconstruction_precision_is_explicit_and_is_not_metadata_dtype_emulation() {
    let layout = Layout::new(1, 32, 4, 32, MetadataDtype::F32).unwrap();
    let weights = [0x33; 16];
    let scale = f32::from_bits(0x3f800001); // 1 + 2^-23
    let scales = scale.to_le_bytes();
    let biases = (-3.0f32).to_le_bytes();
    let matrix = Matrix::from_le_bytes(layout, &weights, &scales, &biases).unwrap();
    assert_eq!(matrix.value(0, 0).unwrap(), 3.0 * 2.0f32.powi(-23));
    assert_ne!(matrix.value(0, 0).unwrap(), scale * 3.0 - 3.0);
    let layout = Layout::new(1, 32, 4, 32, MetadataDtype::F16).unwrap();
    let scales = f16::MAX.to_le_bytes();
    let biases = [0, 0];
    let matrix = Matrix::from_le_bytes(layout, &[0xff; 16], &scales, &biases).unwrap();
    assert_eq!(matrix.value(0, 0).unwrap(), 65504.0 * 15.0); // No Half output cast.
}

#[test]
fn all_matvec_failures_preserve_every_output_element() {
    let layout = Layout::new(2, 32, 4, 32, MetadataDtype::F32).unwrap();
    let weights = [0xff; 32];
    let biases = [0; 8];
    let scales = metadata(MetadataDtype::F32, &[0.0, f32::MAX / 16.0]);
    let matrix = Matrix::from_le_bytes(layout, &weights, &scales, &biases).unwrap();
    assert!(matrix.value(1, 0).unwrap().is_finite());
    let mut output = [42.0, 43.0];
    for input in [
        &[1.0f32][..],
        &[f32::MAX; 32][..],
        &[f32::NAN; 32][..],
        &[f32::INFINITY; 32][..],
    ] {
        assert!(matrix.matvec(input, &mut output).is_err());
        assert_eq!(output, [42.0, 43.0]);
    }
    let mut short = [99.0];
    assert!(matrix.matvec(&[1.0; 32], &mut short).is_err());
    assert_eq!(short, [99.0]);
    let scales = metadata(MetadataDtype::F32, &[0.0, f32::MAX]);
    let matrix = Matrix::from_le_bytes(layout, &weights, &scales, &biases).unwrap();
    assert!(matrix.value(1, 0).is_err());
    assert!(matrix.matvec(&[0.0; 32], &mut output).is_err());
    assert_eq!(output, [42.0, 43.0]);
}
