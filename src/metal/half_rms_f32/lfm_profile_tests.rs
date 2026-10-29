use super::*;

#[test]
fn direct_lfm_shapes_retain_admitted_epsilon_and_exact_extents() {
    for (width, rows) in [
        (64, 1),
        (64, 128),
        (64, 256),
        (64, 512),
        (256, 1),
        (256, 32),
        (1024, 1),
        (1024, 32),
    ] {
        let p = Plan::new(rows, width, width + 3, width + 5, 1e-5).unwrap();
        assert_eq!(p.epsilon_bits, 1e-5_f32.to_bits());
        assert_eq!(p.input_len(), rows * (width + 3) + GUARD);
        assert_eq!(p.output_len(), rows * (width + 5) + GUARD);
        let bytes = 2 * (p.input_len() + p.output_len()) + 4 * (width + GUARD + 1 + GUARD);
        assert_eq!(p.operand_bytes().unwrap(), bytes);
    }
    for (width, rows) in [
        (32, 1),
        (640, 1),
        (64, 513),
        (256, 33),
        (1024, 33),
        (1024, 512),
    ] {
        assert!(Plan::new(rows, width, width, width, 1e-5).is_err());
    }
    for epsilon in [
        0.,
        f32::NAN,
        f32::INFINITY,
        f32::from_bits(1e-5_f32.to_bits() + 1),
    ] {
        assert!(Plan::new(1, 64, 64, 64, epsilon).is_err());
    }
    assert!(Plan::new(0, 64, 64, 64, 1e-5).is_err());
    assert!(Plan::new(512, 64, usize::MAX, 64, 1e-5).is_err());
    assert!(Plan::new(512, 64, 64, usize::MAX, 1e-5).is_err());
    assert!(Plan::new(1, 64, 63, 64, 1e-5).is_err());
    assert!(Plan::new(1, 64, 64, 63, 1e-5).is_err());
}

#[test]
fn historical_gemma_admission_and_epsilon_remain_distinct() {
    for width in [32, 64, 256, 640] {
        let p = Plan::new(128, width, width, width, 1e-6).unwrap();
        assert_eq!(p.epsilon_bits, 1e-6_f32.to_bits());
        assert!(Plan::new(129, width, width, width, 1e-6).is_err());
    }
    assert!(Plan::new(1, 1024, 1024, 1024, 1e-6).is_err());
    assert_ne!(
        Plan::new(1, 64, 64, 64, 1e-6).unwrap().epsilon_bits,
        Plan::new(1, 64, 64, 64, 1e-5).unwrap().epsilon_bits
    );
}
