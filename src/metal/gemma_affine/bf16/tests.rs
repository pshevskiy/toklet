use super::*;
use half::bf16;
use std::io::{Read, Seek, SeekFrom};

fn oracle(weights: &[u8], input: &[f32], columns: usize) -> Vec<f64> {
    input
        .chunks_exact(columns)
        .flat_map(|row| {
            weights.chunks_exact(columns * 2).map(move |coefficients| {
                coefficients
                    .chunks_exact(2)
                    .zip(row)
                    .map(|(w, &x)| {
                        let weight = bf16::from_bits(u16::from_le_bytes([w[0], w[1]]));
                        f64::from(weight.to_f32()) * f64::from(x)
                    })
                    .sum()
            })
        })
        .collect()
}

#[test]
fn exact_bf16_plan_and_coefficient_domain() {
    let (sizes, owned) = projection_sizes(5, 128, 3).unwrap();
    assert_eq!(sizes, [320, 384, 15, 1]);
    assert_eq!(
        owned,
        (320 + 384 + 15 + 1 + 4 * GUARD) * 4 + size_of::<Bf16Projection<'_>>()
    );
    for (rows, columns, capacity) in [(0, 128, 3), (5, 65, 3), (5, 128, 33), (usize::MAX, 128, 1)] {
        assert!(projection_sizes(rows, columns, capacity).is_err());
    }
    assert_eq!(
        maximum_coefficient(&[0, 0xbf, 0x81, 0x3f]).unwrap(),
        1.0078125
    );
    for word in [0x0001_u16, 0x8001, 0x7fc0, 0xff80, 0x5000] {
        assert!(maximum_coefficient(&word.to_le_bytes()).is_err());
    }
}

#[test]
#[ignore = "explicit Metal BF16 arithmetic, memory and publication checks"]
fn bf16_projection_preserves_residuals_chunks_and_guards() {
    assert!(matches!(
        Bf16Context::new(CONTEXT_ALLOWANCE_BYTES - 1),
        Err(Error::Budget { required: CONTEXT_ALLOWANCE_BYTES, limit })
        if limit == CONTEXT_ALLOWANCE_BYTES - 1
    ));
    let context = Bf16Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let (rows, columns, capacity) = (5, 128, 3);
    let mut weights = vec![0u8; rows * columns * 2];
    for (i, word) in weights.chunks_exact_mut(2).enumerate().skip(columns) {
        let value = ((i % 29) as f32 - 14.) / 32.;
        word.copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
    }
    weights[..2].copy_from_slice(&0x3f81_u16.to_le_bytes());
    weights[64..66].copy_from_slice(&0xbf80_u16.to_le_bytes());
    let mut input: Vec<f32> = (0..capacity * columns)
        .map(|i| ((i % 13) as f32 - 6.) / 8.)
        .collect();
    input[0] = 1_f32.next_up();
    input[32] = (f64::from(input[0]) * 1.0078125) as f32;
    let expected: Vec<_> = oracle(&weights, &input, columns)
        .into_iter()
        .map(|v| (v as f32).to_bits())
        .collect();
    assert_eq!(expected[0], (2_f32.powi(-30)).to_bits());
    let owned = Bf16Projection::minimum_owned_bytes(rows, columns, capacity).unwrap();
    assert!(matches!(
        context.projection(rows, columns, &weights, capacity, owned - 1),
        Err(Error::Budget { required, limit }) if required == owned && limit == owned - 1
    ));
    let mut active = context
        .projection(rows, columns, &weights, capacity, owned)
        .unwrap();
    assert_eq!(active.owned_bytes(), owned);
    let mut actual = vec![99.; capacity * rows];
    active.apply(&input, &mut actual, capacity).unwrap();
    assert!(
        actual
            .iter()
            .map(|v| v.to_bits())
            .eq(expected.iter().copied())
    );
    actual.fill(99.);
    active
        .apply(&input[..columns], &mut actual[..rows], 1)
        .unwrap();
    active
        .apply(&input[columns..], &mut actual[rows..], 2)
        .unwrap();
    assert!(
        actual
            .iter()
            .map(|v| v.to_bits())
            .eq(expected.iter().copied())
    );
    actual.fill(99.);
    assert!(
        active
            .apply(&input[..columns - 1], &mut actual[..rows], 1)
            .is_err()
    );
    assert!(!active.is_poisoned() && actual.iter().all(|&v| v == 99.));
    active.coefficients.with_mut(|v| *v.last_mut().unwrap() = 0);
    assert!(active.apply(&input, &mut actual, capacity).is_err());
    assert!(active.is_poisoned() && actual.iter().all(|&v| v == 99.));
    println!(
        "BF16_PROJECTION_AUTHORED device={} owned={} exact_outputs={}",
        context.device_name(),
        owned,
        expected.len()
    );
}

#[test]
#[ignore = "explicit Metal BF16 midpoint and SIMD merge controls"]
fn bf16_projection_resolves_three_term_midpoints() {
    let context = Bf16Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let lanes = context.0.width;
    let half_ulp = 2_f32.powi(-24);
    let tail = 2_f32.powi(-60);
    let next = 1_f32.next_up();
    let subnormal = f32::from_bits(1);
    // The first two terms put the exact dot on a midpoint. The last term
    // distinguishes both sides of that midpoint without changing either
    // adjacent FP32 word. Include signed, cancellation and tiny controls.
    let cases = [
        ([1., half_ulp, tail], next),
        ([1., half_ulp, 0.], 1.),
        ([1., half_ulp, -tail], 1.),
        ([-1., -half_ulp, -tail], -next),
        ([-1., -half_ulp, tail], -1.),
        ([next, half_ulp, -tail], next),
        ([next, half_ulp, 0.], next.next_up()),
        ([next, half_ulp, tail], next.next_up()),
        ([1., -1., f32::MIN_POSITIVE], f32::MIN_POSITIVE),
        ([1., -1., -subnormal], -subnormal),
    ];
    let count = 2 * cases.len();
    for columns in (64..=12288).step_by(64) {
        let weights = 1_f32.to_bits().to_le_bytes()[2..].repeat(columns);
        let within_lane = if 2 * lanes < columns {
            [0, lanes, 2 * lanes]
        } else {
            [0, 1, 2]
        };
        let across_lanes = [0, 1, columns - 1];
        let mut input = vec![0.; count * columns];
        let mut expected = Vec::with_capacity(count);
        for (row, (positions, (values, answer))) in [within_lane, across_lanes]
            .into_iter()
            .flat_map(|positions| cases.map(|case| (positions, case)))
            .enumerate()
        {
            for (column, value) in positions.into_iter().zip(values) {
                input[row * columns + column] = value;
            }
            expected.push(answer.to_bits());
        }
        let bytes = Bf16Projection::minimum_owned_bytes(1, columns, count).unwrap();
        let mut projection = context
            .projection(1, columns, &weights, count, bytes)
            .unwrap();
        let mut actual = vec![0.; count];
        projection.apply(&input, &mut actual, count).unwrap();
        for (row, (value, bits)) in actual.into_iter().zip(expected).enumerate() {
            // Precise Metal may flush FP32 subnormals. Accept only the
            // exact subnormal or zero for the existing subnormal control.
            if f32::from_bits(bits).is_subnormal() && value == 0.0 {
                continue;
            }
            assert_eq!(value.to_bits(), bits, "width {columns}, authored row {row}");
        }
    }
    println!(
        "BF16_MIDPOINTS lanes={lanes} widths=192 normal_exact={} subnormal_controls=384",
        192 * (count - 2)
    );
}

#[test]
#[ignore = "explicit original BF16 source rows and independent Metal dot check"]
fn original_bf16_attention_and_feed_forward_rows() {
    use crate::mlx::gemma4::{ModalityPolicy, bf16 as source};
    let root = std::path::PathBuf::from(std::env::var_os("TOKLET_GEMMA_BF16_MODEL").unwrap());
    let config = std::fs::read(root.join("config.json")).unwrap();
    let mut file = std::fs::File::open(root.join("model.safetensors")).unwrap();
    let size = file.metadata().unwrap().len();
    let mut length = [0; 8];
    file.read_exact(&mut length).unwrap();
    let header_bytes = u64::from_le_bytes(length);
    assert_eq!(header_bytes, 263952);
    let mut header = vec![0; header_bytes as usize];
    file.read_exact(&mut header).unwrap();
    let source = source::inspect(&config, &header, size, ModalityPolicy::TextOnly).unwrap();
    let context = Bf16Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    for (name, first, rows) in [
        ("layers.0.self_attn.q_proj.weight", 7, 5),
        ("layers.16.mlp.down_proj.weight", 47, 3),
    ] {
        let tensor = &source.text_tensors()[&format!("model.language_model.{name}")];
        let columns = tensor.shape[1];
        let start = source.payload_offset() + tensor.range.start + first * columns * 2;
        file.seek(SeekFrom::Start(start as u64)).unwrap();
        let mut weights = vec![0; rows * columns * 2];
        file.read_exact(&mut weights).unwrap();
        let input: Vec<_> = (0..3 * columns)
            .map(|i| ((i % 31) as f32 - 15.) / 32.)
            .collect();
        let expected = oracle(&weights, &input, columns);
        let bytes = Bf16Projection::minimum_owned_bytes(rows, columns, 3).unwrap();
        let mut projection = context
            .projection(rows, columns, &weights, 3, bytes)
            .unwrap();
        let mut output = vec![0.; 3 * rows];
        projection.apply(&input, &mut output, 3).unwrap();
        for (&actual, want) in output.iter().zip(expected) {
            let error = (f64::from(actual) - want).abs();
            assert!(error <= 1e-5 + 1e-4 * want.abs());
        }
        println!(
            "BF16_PROJECTION_REAL name={name} first={first} rows={rows} columns={columns} checked={} owned={bytes}",
            output.len()
        );
    }
}
