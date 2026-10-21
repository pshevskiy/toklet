#![cfg(all(
    feature = "metal",
    any(target_os = "macos", target_os = "ios"),
    target_arch = "aarch64"
))]

use half::f16;
use serde_json::Value;
use sha2::{Digest, Sha256};
use toklet::{
    metal::{Error, mlx::Context},
    mlx::affine::{Layout, Matrix, MetadataDtype},
};

#[test]
#[ignore = "Explicit Apple GPU native MLX affine/gather oracle, batches and resident reuse"]
fn resident_affine_and_embedding_match_all_pinned_storage_cases() {
    const RAW: &[u8] = include_bytes!("fixtures/mlx-affine-v1/arrays.bin");
    let manifest: Value =
        serde_json::from_str(include_str!("fixtures/mlx-affine-v1/manifest.json")).unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(RAW)), manifest["sha256"]);
    let context = Context::new().unwrap();
    let mut compared = 0;
    let mut calls = 0;
    let mut widths = std::collections::BTreeSet::new();
    for case in manifest["cases"].as_array().unwrap() {
        let array = |name: &str| {
            let a = &case["arrays"][name];
            let offset = a["offset"].as_u64().unwrap() as usize;
            let len = a["bytes"].as_u64().unwrap() as usize;
            let data = &RAW[offset..offset + len];
            assert_eq!(format!("{:x}", Sha256::digest(data)), a["sha256"]);
            data
        };
        let dtype = match case["metadata_dtype"].as_str().unwrap() {
            "F16" => MetadataDtype::F16,
            "BF16" => MetadataDtype::BF16,
            "F32" => MetadataDtype::F32,
            _ => panic!(),
        };
        let layout = Layout::new(
            2,
            256,
            case["bits"].as_u64().unwrap() as u8,
            case["group_size"].as_u64().unwrap() as usize,
            dtype,
        )
        .unwrap();
        let oracle =
            Matrix::from_le_bytes(layout, array("weight"), array("scales"), array("biases"))
                .unwrap();
        let mut affine = context
            .affine(
                layout,
                array("weight"),
                array("scales"),
                array("biases"),
                9,
                128 * 1024 * 1024,
            )
            .unwrap();
        widths.insert(affine.execution_width());
        let allocated = affine.allocated_buffer_bytes();
        for batch in [1, 3, 9, 1] {
            for half_output in [false, true] {
                let input: Vec<f32> = (0..batch * 256)
                    .map(|i| {
                        f16::from_f32(((i * 13 + i / 256) % 29) as f32 / 16.0 - 0.875).to_f32()
                    })
                    .collect();
                let bias = [0.125, -0.25];
                let mut output = vec![99.; batch * 2];
                affine
                    .matmul(batch, &input, &bias, &mut output, half_output)
                    .unwrap();
                calls += 1;
                for token in 0..batch {
                    for row in 0..2 {
                        let mut sum = f64::from(bias[row]);
                        for col in 0..256 {
                            sum += f64::from(oracle.value(row, col).unwrap())
                                * f64::from(input[token * 256 + col]);
                        }
                        let expected = if half_output {
                            f16::from_f64(sum).to_f32()
                        } else {
                            sum as f32
                        };
                        assert_eq!(
                            output[token * 2 + row].to_bits(),
                            expected.to_bits(),
                            "case={case:?} batch={batch} token={token} row={row} half={half_output}"
                        );
                        compared += 1;
                    }
                }
                let tokens: Vec<u32> = (0..batch).map(|t| (t % 2) as u32).collect();
                let mut embedding = vec![99.; batch * 256];
                affine.gather(&tokens, &mut embedding, half_output).unwrap();
                calls += 1;
                for (i, &actual) in embedding.iter().enumerate() {
                    let value = oracle.value(tokens[i / 256] as usize, i % 256).unwrap();
                    let expected = if half_output {
                        f16::from_f32(value).to_f32()
                    } else {
                        value
                    };
                    assert_eq!(actual.to_bits(), expected.to_bits());
                    compared += 1;
                }
                assert_eq!(affine.allocated_buffer_bytes(), allocated);
            }
        }
    }
    println!(
        "MLX_AFFINE_GPU {}",
        serde_json::json!({"device":context.device_name(),"storage_cases":54,"completed_calls":calls,"exact_values":compared,"execution_widths":widths,"model_execution_qualified":false,"performance_qualified":false})
    );
}

#[test]
#[ignore = "Explicit Apple GPU native MLX precision midpoint and failure/reuse boundary"]
fn midpoint_precision_nonfinite_failure_and_reuse_preserve_caller_output() {
    let context = Context::new().unwrap();
    let layout = Layout::new(1, 32, 4, 32, MetadataDtype::F32).unwrap();
    let weights = [0u8; 16];
    let scales = 0f32.to_le_bytes();
    let biases = 1f32.to_le_bytes();
    let mut affine = context
        .affine(layout, &weights, &scales, &biases, 3, 128 * 1024 * 1024)
        .unwrap();
    let mut input = vec![0.; 32];
    input[0] = 1.;
    input[1] = 2f32.powi(-11);
    input[2] = 2f32.powi(-24);
    let exact = 1f64 + 2f64.powi(-11) + 2f64.powi(-24);
    let mut out = [99.];
    affine.matmul(1, &input, &[0.], &mut out, true).unwrap();
    assert_eq!(out[0], f16::from_f64(exact).to_f32());
    assert!(out[0] > 1.);
    affine.matmul(1, &input, &[0.], &mut out, false).unwrap();
    assert_eq!(out[0], exact as f32);
    for sign in [-1f64, 1f64] {
        for side in [-1f64, 0f64, 1f64] {
            input.fill(0.);
            input[0] = sign as f32;
            input[1] = (sign * 2f64.powi(-11)) as f32;
            input[2] = (sign * side * 2f64.powi(-24)) as f32;
            let sum = sign * (1f64 + 2f64.powi(-11) + side * 2f64.powi(-24));
            affine.matmul(1, &input, &[0.], &mut out, true).unwrap();
            assert_eq!(out[0].to_bits(), f16::from_f64(sum).to_f32().to_bits());
            affine.matmul(1, &input, &[0.], &mut out, false).unwrap();
            assert_eq!(out[0].to_bits(), (sum as f32).to_bits());
        }
    }
    for invalid in [f32::NAN, f32::INFINITY, 1e10] {
        input[0] = invalid;
        out = [99.];
        assert!(affine.matmul(1, &input, &[0.], &mut out, false).is_err());
        assert_eq!(out, [99.]);
    }
    out = [99.];
    assert!(affine.gather(&[1], &mut [99.; 32], false).is_err());
    assert!(affine.matmul(0, &[], &[0.], &mut out, false).is_err());
    assert_eq!(out, [99.]);
    input.fill(65504.);
    assert!(matches!(
        affine.matmul(1, &input, &[0.], &mut out, true),
        Err(Error::Nonfinite)
    ));
    assert_eq!(out, [99.]);
    input.fill(0.5);
    affine.matmul(1, &input, &[0.25], &mut out, true).unwrap();
    assert_eq!(out, [16.25]);
    // Finite metadata can reconstruct beyond Half; gather must reject safely.
    let huge = 1e10f32.to_le_bytes();
    let mut affine = context
        .affine(layout, &weights, &scales, &huge, 1, 128 * 1024 * 1024)
        .unwrap();
    let mut output = vec![99.; 32];
    assert!(matches!(
        affine.gather(&[0], &mut output, true),
        Err(Error::Nonfinite)
    ));
    assert_eq!(output, vec![99.; 32]);
    affine.gather(&[0], &mut output, false).unwrap();
    assert_eq!(output, vec![1e10; 32]);
    println!(
        "MLX_AFFINE_BOUNDARIES {}",
        serde_json::json!({"device":context.device_name(),"midpoint_correct":true,"failure_atomic":true,"reused_after_nonfinite":true,"model_execution_qualified":false})
    );
}

#[test]
#[ignore = "Explicit Apple GPU odd Half/BF16 metadata padding and shrinking batch reuse"]
fn odd_metadata_counts_preserve_distinct_rows_and_padding() {
    let context = Context::new().unwrap();
    for dtype in [MetadataDtype::F16, MetadataDtype::BF16] {
        let encode = |values: &[f32]| -> Vec<u8> {
            values
                .iter()
                .flat_map(|&x| match dtype {
                    MetadataDtype::F16 => f16::from_f32(x).to_le_bytes(),
                    MetadataDtype::BF16 => half::bf16::from_f32(x).to_le_bytes(),
                    _ => unreachable!(),
                })
                .collect()
        };
        let layout = Layout::new(3, 32, 4, 32, dtype).unwrap();
        let weights = vec![0x21; layout.weight_bytes()];
        let scales = encode(&[0.5, -0.25, 0.125]);
        let biases = encode(&[-1., 0.5, 0.25]);
        assert_eq!(scales.len(), 6);
        let oracle = Matrix::from_le_bytes(layout, &weights, &scales, &biases).unwrap();
        let mut affine = context
            .affine(layout, &weights, &scales, &biases, 7, 128 * 1024 * 1024)
            .unwrap();
        for batch in [7, 1, 3, 1] {
            for half_output in [false, true] {
                let input: Vec<f32> = (0..batch * 32)
                    .map(|i| ((i * 3 + i / 32) % 11) as f32 / 8. - 0.5)
                    .collect();
                let bias = [0.25, -0.5, 0.125];
                let mut output = vec![99.; batch * 3];
                affine
                    .matmul(batch, &input, &bias, &mut output, half_output)
                    .unwrap();
                for token in 0..batch {
                    for row in 0..3 {
                        let mut sum = f64::from(bias[row]);
                        for c in 0..32 {
                            sum += f64::from(oracle.value(row, c).unwrap())
                                * f64::from(input[token * 32 + c]);
                        }
                        let expected = if half_output {
                            f16::from_f64(sum).to_f32()
                        } else {
                            sum as f32
                        };
                        assert_eq!(output[token * 3 + row].to_bits(), expected.to_bits());
                    }
                }
                let ids: Vec<u32> = (0..batch).map(|t| (t % 3) as u32).collect();
                let mut output = vec![99.; batch * 32];
                affine.gather(&ids, &mut output, half_output).unwrap();
                for (i, &actual) in output.iter().enumerate() {
                    let x = oracle.value(ids[i / 32] as usize, i % 32).unwrap();
                    let expected = if half_output {
                        f16::from_f32(x).to_f32()
                    } else {
                        x
                    };
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
            }
        }
    }
    println!(
        "MLX_AFFINE_ODD_METADATA {}",
        serde_json::json!({"device":context.device_name(),"metadata_dtypes":2,"rows":3,"metadata_payload_bytes_each":6,"metadata_allocation_bytes_each":8,"model_execution_qualified":false})
    );
}

#[test]
#[ignore = "Explicit Apple GPU restricted domain and Half-subnormal conversion witnesses"]
fn tiny_metadata_rejects_and_admitted_half_subnormals_are_preserved() {
    let context = Context::new().unwrap();
    let layout = Layout::new(1, 32, 4, 32, MetadataDtype::F32).unwrap();
    let zero = 0f32.to_le_bytes();
    let tiny = f32::from_bits(1).to_le_bytes();
    let weights = [0u8; 16];
    let cpu = Matrix::from_le_bytes(layout, &weights, &zero, &tiny).unwrap();
    assert_eq!(cpu.value(0, 0).unwrap().to_bits(), 1);
    assert!(matches!(
        context.affine(layout, &weights, &zero, &tiny, 1, 128 * 1024 * 1024),
        Err(Error::Unsupported(_))
    ));
    let minimum = 2f32.powi(-79).to_le_bytes();
    let mut affine = context
        .affine(layout, &weights, &zero, &minimum, 1, 128 * 1024 * 1024)
        .unwrap();
    let mut output = [99.; 32];
    affine.gather(&[0], &mut output, false).unwrap();
    assert_eq!(output, [2f32.powi(-79); 32]);
    let input = [f16::from_bits(1).to_f32(); 32];
    let mut dot = [99.];
    affine.matmul(1, &input, &[0.], &mut dot, false).unwrap();
    assert_eq!(dot, [2f32.powi(-98)]);
    let layout = Layout::new(1, 32, 4, 32, MetadataDtype::F16).unwrap();
    let mut affine = context
        .affine(
            layout,
            &weights,
            &0u16.to_le_bytes(),
            &1u16.to_le_bytes(),
            1,
            128 * 1024 * 1024,
        )
        .unwrap();
    affine.gather(&[0], &mut output, true).unwrap();
    assert_eq!(output, [f16::from_bits(1).to_f32(); 32]);
    // Half subnormal activations and source metadata become normal FP32 values.
    affine.matmul(1, &input, &[0.], &mut dot, false).unwrap();
    assert_eq!(dot, [2f32.powi(-43)]);
    affine.matmul(1, &[1.; 32], &[0.], &mut dot, true).unwrap();
    assert_eq!(dot, [f16::from_bits(32).to_f32()]);
    // Product residual plus cancellation reaches the exact admitted FP32 floor.
    let layout = Layout::new(1, 32, 4, 32, MetadataDtype::F32).unwrap();
    let mut weights = [0u8; 16];
    weights[0] = 0x11;
    let mut affine = context
        .affine(
            layout,
            &weights,
            &2f32.powi(-102).to_le_bytes(),
            &2f32.powi(-79).to_le_bytes(),
            1,
            128 * 1024 * 1024,
        )
        .unwrap();
    let mut input = [0.; 32];
    for sign in [-1f32, 1f32] {
        input[0] = sign * (2f32.powi(-14) + 2f32.powi(-24));
        input[1] = -sign * 2f32.powi(-14);
        input[2] = -input[0];
        input[3] = -input[1];
        affine.matmul(1, &input, &[0.], &mut dot, false).unwrap();
        assert_eq!(dot[0].to_bits(), (sign * f32::MIN_POSITIVE).to_bits());
    }
    println!(
        "MLX_AFFINE_DOMAIN {}",
        serde_json::json!({"device":context.device_name(),"fp32_subnormal_metadata_rejected":true,"minimum_fused_weight":2f64.powi(-79),"half_subnormal_conversion_preserved":true,"signed_fp32_residual_floor_preserved":true,"model_execution_qualified":false})
    );
}
