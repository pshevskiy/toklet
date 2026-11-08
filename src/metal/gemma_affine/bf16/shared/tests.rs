use super::*;

#[test]
fn shared_weight_and_scratch_sizes_have_separate_owners_and_checked_bounds() {
    let (rows, cols, cap) = (3, 64, 2);
    assert_eq!(
        Matrix::payload_bytes(rows, cols).unwrap(),
        (rows * cols / 2 + GUARD) * 4
    );
    assert_eq!(
        Workspace::minimum_bytes(rows, cols, cap).unwrap(),
        size_of::<Workspace<'_>>() + (cap * cols + cap * rows + 1 + 3 * GUARD) * 4
    );
    assert_eq!(
        Workspace::minimum_bytes(rows, cols, cap).unwrap() - size_of::<Workspace<'_>>(),
        Bf16Projection::minimum_owned_bytes(rows, cols, cap).unwrap()
            - size_of::<Bf16Projection<'_>>()
            - Matrix::payload_bytes(rows, cols).unwrap()
    );
    for (r, c, n) in [
        (0, 64, 1),
        (16385, 64, 1),
        (3, 63, 1),
        (3, 65, 1),
        (3, 64, 0),
        (3, 64, 33),
    ] {
        assert!(Workspace::minimum_bytes(r, c, n).is_err());
    }
}

#[test]
#[ignore = "explicit Mac BF16 immutable-matrix/shared-scratch GPU fixture"]
fn shared_bf16_reuses_weights_matches_original_projection_and_preserves_poison() {
    let ctx = Bf16Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let (rows, cols, cap) = (3, 64, 2);
    let coefficients: Vec<f32> = (0..rows * cols)
        .map(|i| (i as i32 % 19 - 9) as f32 / 16.)
        .collect();
    let raw: Vec<u8> = coefficients
        .iter()
        .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
        .collect();
    let needed = Matrix::payload_bytes(rows, cols).unwrap() + size_of::<Matrix>();
    assert!(
        matches!(ctx.resident_matrix(rows,cols,&raw,needed-1),Err(Error::Budget {required,limit}) if required==needed && limit==needed-1)
    );
    let mut matrix = ctx.resident_matrix(rows, cols, &raw, needed).unwrap();
    let scratch = Workspace::minimum_bytes(rows, cols, cap).unwrap();
    assert!(
        matches!(ctx.shared_workspace(rows,cols,cap,scratch-1),Err(Error::Budget {required,limit}) if required==scratch && limit==scratch-1)
    );
    let mut a = ctx.shared_workspace(rows, cols, cap, scratch).unwrap();
    let mut b = ctx.shared_workspace(rows, cols, cap, scratch).unwrap();
    let mut original = ctx
        .projection(
            rows,
            cols,
            &raw,
            cap,
            Bf16Projection::minimum_owned_bytes(rows, cols, cap).unwrap(),
        )
        .unwrap();
    for tokens in [1, 2, 1, 2] {
        let x: Vec<f32> = (0..tokens * cols)
            .map(|i| (i as i32 % 13 - 6) as f32 / 32.)
            .collect();
        let mut expected = vec![0.; tokens * rows];
        original.apply(&x, &mut expected, tokens).unwrap();
        for s in [&mut a, &mut b] {
            let mut actual = vec![f32::NAN; tokens * rows];
            s.apply(&matrix, &x, &mut actual, tokens).unwrap();
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            for (token, y) in actual.chunks_exact(rows).enumerate() {
                for (row, &value) in y.iter().enumerate() {
                    let oracle = (0..cols)
                        .map(|c| {
                            f64::from(x[token * cols + c]) * f64::from(coefficients[row * cols + c])
                        })
                        .sum::<f64>();
                    assert!(
                        (f64::from(value) - oracle).abs()
                            <= ORDERED_ERROR_LIMIT * (1. + oracle.abs())
                    );
                }
            }
        }
    }
    let mut y = [123.; 3];
    assert!(a.apply(&matrix, &[1.; 63], &mut y, 1).is_err());
    assert!(!a.is_poisoned());
    assert_eq!(y, [123.; 3]);
    matrix
        .coefficients
        .with_mut(|words| *words.last_mut().unwrap() ^= 1);
    assert!(a.apply(&matrix, &[1.; 64], &mut y, 1).is_err());
    assert!(a.is_poisoned());
    assert_eq!(y, [123.; 3]);
    matrix
        .coefficients
        .with_mut(|words| *words.last_mut().unwrap() ^= 1);
    assert!(a.apply(&matrix, &[1.; 64], &mut y, 1).is_err());
    b.apply(&matrix, &[1.; 64], &mut y, 1).unwrap();
    assert_eq!(
        matrix.physical_bytes(),
        Matrix::payload_bytes(rows, cols).unwrap()
    );
}
