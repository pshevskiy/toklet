use super::*;

#[test]
fn cold_threshold_is_largest_passing_fp32_through_unchanged_bounds() {
    for columns in (64..=12288).step_by(64) {
        for maximum in [
            0.,
            f64::from(f32::MIN_POSITIVE),
            1. / 128.,
            1.,
            f64::from(MAX_WEIGHT),
        ] {
            let proof = MaxInputProof::new(columns, maximum).unwrap();
            let passes = |bits| {
                let (s, u) = bounds(columns, maximum, f32::from_bits(bits));
                s <= f64::from(f32::MAX) / 4. && u <= ORDERED_ERROR_LIMIT
            };
            assert!(passes(proof.bits));
            if proof.bits < MAX_FINITE_BITS {
                assert!(!passes(proof.bits + 1));
            }
            for bits in [
                0,
                1,
                0x007f_ffff,
                0x0080_0000,
                1_f32.to_bits(),
                MAX_FINITE_BITS,
            ] {
                assert_eq!(passes(bits), bits <= proof.bits);
            }
            if maximum == 0. {
                assert_eq!(proof.bits, MAX_FINITE_BITS);
            }
        }
    }
    for columns in [0, 1, 63, 65, 12289, usize::MAX] {
        assert!(MaxInputProof::new(columns, 1.).is_err());
    }
    for maximum in [
        f64::NAN,
        f64::INFINITY,
        -1.,
        f64::from(MAX_WEIGHT) * 2.,
        f64::from(f32::from_bits(1)),
        1.00000000000001,
        f64::from(1_f32.next_up()),
    ] {
        assert!(MaxInputProof::new(64, maximum).is_err());
    }
}

#[test]
fn threshold_miss_preserves_sparse_l1_success_and_original_error_priority() {
    for columns in [64, 1536, 4096, 6144, 12288] {
        let proof = MaxInputProof::new(columns, 1.).unwrap();
        let miss = f32::from_bits(proof.bits + 1);
        let (s, u) = bounds(columns, 1., miss);
        assert!(s > f64::from(f32::MAX) / 4. || u > ORDERED_ERROR_LIMIT);
        let mut sparse = vec![0.; columns];
        sparse[columns - 1] = -miss;
        admit(columns, 1., &sparse).unwrap();
        let (s, u) = l1_bounds(columns, 1., &sparse);
        assert!(s <= f64::from(f32::MAX) / 4. && u <= ORDERED_ERROR_LIMIT);
        assert!(matches!(
            admit(columns, 1., &vec![miss; columns]),
            Err(Error::Options(
                "Gemma affine input ordered-sum accuracy bound"
            ))
        ));
    }
    assert!(matches!(
        admit(64, f64::from(MAX_WEIGHT), &[f32::MAX; 64]),
        Err(Error::Options("Gemma affine input absolute-product bound"))
    ));
}

#[test]
fn bounded_projection_geometry_and_ordered_status_decisions() {
    let g = Geometry::new(8192, 12288, 32).unwrap();
    assert_eq!(g.words, [32, 12288, 8192]);
    assert_eq!(g.coefficients, 8192 * 12288 / 2);
    assert_eq!(g.input, 32 * 12288);
    assert_eq!(g.output, 32 * 8192);
    for (rows, columns, tokens) in [
        (0, 64, 1),
        (8193, 64, 1),
        (1, 0, 1),
        (1, 65, 1),
        (1, 12352, 1),
        (1, 64, 0),
        (1, 64, 33),
        (usize::MAX, 64, 1),
        (1, usize::MAX, 1),
        (1, 64, usize::MAX),
    ] {
        assert!(Geometry::new(rows, columns, tokens).is_err());
    }
    assert_eq!(admission_decision(0).unwrap(), AdmissionDecision::Admitted);
    assert_eq!(
        admission_decision(2).unwrap(),
        AdmissionDecision::ReferenceReplay
    );
    for flags in [1, 3] {
        assert!(matches!(admission_decision(flags), Err(Error::Nonfinite)));
    }
    for flags in [4, 5, 6, 7, u32::MAX] {
        assert!(matches!(admission_decision(flags), Err(Error::Options(_))));
    }
}

// Independent host simulation of the new integer-bit/FP32 enclosure. The CPU
// compatibility oracle remains row_l1/l1_bounds; no CPU proof implementation changes.
fn two_up_enclosure(row: &[f32]) -> Option<f32> {
    let mut upper = 0_f32;
    for &value in row {
        let magnitude = value.to_bits() & 0x7fff_ffff;
        if magnitude >= 0x7f80_0000 {
            return None;
        }
        if magnitude == 0 {
            continue;
        }
        let term = f32::from_bits(magnitude.max(0x0080_0000));
        if upper == 0. {
            upper = term;
            continue;
        }
        let rounded = (upper + term).to_bits();
        if rounded > 0x7f7f_fffd {
            return None;
        }
        upper = f32::from_bits(rounded + 2);
    }
    Some(upper)
}

#[test]
fn cold_l1_threshold_is_largest_through_unchanged_singleton_l1_bounds() {
    for columns in [64, 256, 1536, 2048, 4096, 6144, 12288] {
        for maximum in [
            0.,
            f64::from(f32::MIN_POSITIVE),
            1. / 128.,
            1.,
            f64::from(MAX_WEIGHT),
        ] {
            let proof = L1InputProof::new(columns, maximum).unwrap();
            let mut row = vec![0.; columns];
            let mut passes = |bits| {
                row[0] = f32::from_bits(bits);
                assert_eq!(row_l1(&row), f64::from(row[0]));
                let (s, u) = l1_bounds(columns, maximum, &row);
                s <= f64::from(f32::MAX) / 4. && u <= ORDERED_ERROR_LIMIT
            };
            assert!(passes(proof.bits));
            if proof.bits < MAX_FINITE_BITS {
                assert!(!passes(proof.bits + 1));
            }
            for bits in [
                0,
                1,
                0x007f_ffff,
                0x0080_0000,
                1_f32.to_bits(),
                MAX_FINITE_BITS,
            ] {
                assert_eq!(passes(bits), bits <= proof.bits);
            }
            if maximum == 0. {
                assert_eq!(proof.bits, MAX_FINITE_BITS);
            }
            let max = MaxInputProof::new(columns, maximum).unwrap();
            let next = max.bits.checked_add(1).filter(|&b| b <= MAX_FINITE_BITS);
            if let Some(next) = next {
                let mut sparse = vec![0.; columns];
                sparse[columns - 1] = -f32::from_bits(next);
                assert!(two_up_enclosure(&sparse).unwrap().to_bits() <= proof.bits);
                admit(columns, maximum, &sparse).unwrap();
            }
        }
    }
    for columns in [0, 63, 65, usize::MAX] {
        assert!(L1InputProof::new(columns, 1.).is_err());
    }
    for maximum in [
        f64::NAN,
        f64::INFINITY,
        -1.,
        f64::from(MAX_WEIGHT) * 2.,
        f64::from(f32::from_bits(1)),
    ] {
        assert!(L1InputProof::new(64, maximum).is_err());
    }
}

#[test]
fn sequential_two_up_encloses_original_fp64_upward_l1_or_requests_replay() {
    for row in [
        vec![0., -0.],
        vec![f32::from_bits(1), -f32::from_bits(0x007f_ffff)],
        vec![f32::MIN_POSITIVE, 1., f32::EPSILON, -1.],
        vec![1.; 12288],
        vec![f32::MAX],
        vec![f32::MAX, 1.],
        vec![f32::MAX, f32::MAX],
    ] {
        if let Some(upper) = two_up_enclosure(&row) {
            assert!(f64::from(upper) >= row_l1(&row));
        }
    }
    assert_eq!(two_up_enclosure(&[0., -0.]), Some(0.));
    assert_eq!(
        two_up_enclosure(&[f32::from_bits(1)]),
        Some(f32::MIN_POSITIVE)
    );
    assert_eq!(
        two_up_enclosure(&[1., 1.]),
        Some(f32::from_bits(2_f32.to_bits() + 2))
    );
    assert!(two_up_enclosure(&[f32::MAX, 1.]).is_none());
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(two_up_enclosure(&[0., value]).is_none());
    }
    let mut seed = 0x5a17_0031_u32;
    for length in [1, 2, 3, 64, 256, 1536, 6144, 12288] {
        let mut row = Vec::with_capacity(length);
        for _ in 0..length {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let exponent = seed % 255;
            row.push(f32::from_bits((seed & 0x807f_ffff) | (exponent << 23)));
        }
        if let Some(upper) = two_up_enclosure(&row) {
            assert!(f64::from(upper) >= row_l1(&row));
        }
    }
}
