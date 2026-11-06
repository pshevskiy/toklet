use super::*;

const ROWS: usize = 3;
const COLUMNS: usize = 192;
const CAPACITY: usize = 32;

fn metadata(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

#[test]
fn affine8_bounds_keep_the_existing_resident_layout_gate() {
    assert!(matches!(
        Affine8Context::new(CONTEXT_ALLOWANCE_BYTES - 1),
        Err(Error::Budget { required, limit })
            if required == CONTEXT_ALLOWANCE_BYTES && limit == required - 1
    ));
    let layout = Layout::new(ROWS, COLUMNS, 8, 64, MetadataDtype::BF16).unwrap();
    assert!(Sizes::new(layout, CAPACITY).is_err());
    let sizes = Sizes::projection(layout, CAPACITY).unwrap();
    let payload_words = ROWS * COLUMNS / 4
        + 2 * (ROWS * (COLUMNS / 64) * 2).div_ceil(4)
        + CAPACITY * (COLUMNS + ROWS)
        + 1;
    assert_eq!(
        sizes.owned,
        (payload_words + 6 * GUARD) * 4 + size_of::<Affine<'_>>()
    );
    for invalid in [
        Layout::new(ROWS, COLUMNS, 8, 32, MetadataDtype::BF16).unwrap(),
        Layout::new(ROWS, COLUMNS, 8, 64, MetadataDtype::F16).unwrap(),
    ] {
        assert!(Sizes::projection(invalid, CAPACITY).is_err());
    }
    for batch in [0, 33, usize::MAX] {
        assert!(Sizes::projection(layout, batch).is_err());
    }
    let high = metadata(&[2_f32.powi(25)]);
    let zero = metadata(&[0.]);
    assert!(codebook(&high, &zero, 4).is_ok());
    assert!(codebook(&high, &zero, 8).is_err());
}

#[test]
#[ignore = "explicit Apple GPU standalone affine8 projection checks"]
fn affine8_all_codes_batches_and_residual_match_independent_dots() {
    let context = Affine8Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let layout = Layout::new(ROWS, COLUMNS, 8, 64, MetadataDtype::BF16).unwrap();
    let codes: Vec<_> = (0..ROWS * COLUMNS)
        .map(|index| (index % 256) as u8)
        .collect();
    let scales = [
        0.03125, -0.0625, 0.00390625, 0., 0.125, -0.25, 0.5, -0.015625, 0.25,
    ];
    let biases = [0.25, -0.5, 1.0078125, -2., 0.5, 0., -0.25, 1., -1.];
    let scale_bytes = metadata(&scales);
    let bias_bytes = metadata(&biases);
    let owned = Sizes::projection(layout, CAPACITY).unwrap().owned;
    assert!(matches!(
        context.affine(layout, &codes, &scale_bytes, &bias_bytes, CAPACITY, owned - 1),
        Err(Error::Budget { required, limit }) if required == owned && limit == owned - 1
    ));
    let invalid = Layout::new(ROWS, COLUMNS, 4, 64, MetadataDtype::BF16).unwrap();
    assert!(
        context
            .affine(invalid, &codes, &scale_bytes, &bias_bytes, CAPACITY, owned)
            .is_err()
    );
    let invalid_scale = metadata(&[2_f32.powi(25); 9]);
    assert!(
        context
            .affine(
                layout,
                &[0; ROWS * COLUMNS],
                &invalid_scale,
                &bias_bytes,
                CAPACITY,
                owned
            )
            .is_err()
    );
    assert!(
        context
            .affine(
                layout,
                &codes[..codes.len() - 1],
                &scale_bytes,
                &bias_bytes,
                CAPACITY,
                owned
            )
            .is_err()
    );
    let mut projection = context
        .affine(layout, &codes, &scale_bytes, &bias_bytes, CAPACITY, owned)
        .unwrap();
    let input: Vec<_> = (0..CAPACITY * COLUMNS)
        .map(|index| ((index * 7 % 31) as f32 - 15.) / 16.)
        .collect();
    for batch in [1, 3, 32, 1] {
        let rows = &input[..batch * COLUMNS];
        let mut output = vec![f32::NAN; batch * ROWS];
        projection.apply(rows, &mut output, batch).unwrap();
        for (token, input_row) in rows.chunks_exact(COLUMNS).enumerate() {
            for (row, code_row) in codes.chunks_exact(COLUMNS).enumerate() {
                let expected = code_row
                    .iter()
                    .zip(input_row)
                    .enumerate()
                    .map(|(column, (&code, &x))| {
                        let group = row * (COLUMNS / 64) + column / 64;
                        let weight = (f64::from(scales[group]) * f64::from(code)
                            + f64::from(biases[group])) as f32;
                        f64::from(weight) * f64::from(x)
                    })
                    .sum::<f64>() as f32;
                assert_eq!(output[token * ROWS + row].to_bits(), expected.to_bits());
            }
        }
        assert_eq!(projection.owned_bytes(), owned);
    }
    let mut preserved = [7.; ROWS];
    let mut bad = [0.; COLUMNS];
    bad[0] = f32::NAN;
    assert!(projection.apply(&bad, &mut preserved, 1).is_err());
    assert_eq!(preserved, [7.; ROWS]);
    assert!(!projection.is_poisoned());

    let residual_layout = Layout::new(1, 64, 8, 64, MetadataDtype::BF16).unwrap();
    let mut residual = context
        .affine(
            residual_layout,
            &[0; 64],
            &metadata(&[0.]),
            &metadata(&[1.0078125]),
            1,
            owned,
        )
        .unwrap();
    let mut input = [0.; 64];
    input[..2].copy_from_slice(&[1. + f32::EPSILON, -1.]);
    let mut output = [0.];
    residual.apply(&input, &mut output, 1).unwrap();
    assert_eq!(output[0].to_bits(), (1.0078125 * f32::EPSILON).to_bits());
}
