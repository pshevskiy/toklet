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
