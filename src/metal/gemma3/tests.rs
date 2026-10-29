use super::*;
fn config() -> Config {
    Config {
        vocab_size: 64,
        hidden_size: 64,
        intermediate_size: 128,
        num_layers: 2,
        query_heads: 4,
        kv_heads: 1,
        head_dim: 32,
        max_positions: 12,
        rms_norm_eps: EPSILON,
        rope_theta: 1000000.,
    }
}
fn specs() -> Vec<LayerSpec> {
    vec![
        LayerSpec {
            kind: KvKind::Sliding,
            theta: 10000.,
        },
        LayerSpec {
            kind: KvKind::Full,
            theta: 1000000.,
        },
    ]
}
#[test]
fn bounded_leaf_geometry_preserves_distinct_query_and_hidden_widths() {
    let c = config();
    geometry(&c, &specs(), 4).unwrap();
    assert_ne!(c.hidden_size, c.query_heads * c.head_dim);
    let mut larger = c.clone();
    larger.hidden_size = 640;
    larger.intermediate_size = 2048;
    larger.head_dim = 256;
    larger.max_positions = 544;
    geometry(&larger, &specs(), 512).unwrap();
    for mutation in 0..8 {
        let mut invalid = c.clone();
        match mutation {
            0 => invalid.query_heads = 2,
            1 => invalid.kv_heads = 2,
            2 => invalid.head_dim = 64,
            3 => invalid.hidden_size = 128,
            4 => invalid.intermediate_size = 256,
            5 => invalid.max_positions = 545,
            6 => invalid.rms_norm_eps = 1e-5,
            _ => invalid.num_layers = 33,
        }
        assert!(geometry(&invalid, &specs(), 4).is_err());
    }
    assert!(geometry(&c, &specs(), 0).is_err());
    assert!(geometry(&c, &specs(), 13).is_err());
    let mut wrong = specs();
    wrong[0].theta = 1000000.;
    assert!(geometry(&c, &wrong, 4).is_err());
}
#[test]
fn raw_gamma_contract_keeps_fp32_effective_coefficients() {
    let raw = [
        bf16::from_bits(0xbf80),
        bf16::from_bits(0x3980),
        bf16::from_bits(0xbf7f),
        bf16::from_bits(0xbf81),
    ];
    gamma(&raw, 4).unwrap();
    let effective = raw.map(|x| 1.0 + x.to_f32());
    assert_eq!(effective, [0., 1. + 2f32.powi(-12), 1. / 256., -1. / 128.]);
    assert_ne!(half::f16::from_f32(effective[1]).to_f32(), effective[1]);
    assert!(gamma(&raw, 3).is_err());
    assert!(gamma(&[bf16::from_bits(0x7fc0)], 1).is_err());
}
#[test]
fn checked_scratch_and_snapshot_accounting_match_unique_physical_owners() {
    let c = config();
    let half_elements = 4 * extent(B, 64).unwrap()
        + 2 * extent(B, 128).unwrap()
        + extent(B, 32).unwrap()
        + 3 * extent(B, 128).unwrap();
    assert_eq!(half_elements * 2, 59532);
    assert_eq!(extent(4 * B, 12).unwrap() * 8, 12344);
    assert_eq!(extent(1, 64).unwrap() * 8, 568);
    assert_eq!(extent(B, 64).unwrap() * 8, 16440);
    let (plans, gpu, readback, metadata) = snapshots::plan(&c, true).unwrap();
    assert_eq!(plans.len(), 48);
    // 3 global hidden Half, 2 layers of 8H+4Q+4D+4I Half;
    // global logits plus two score arrays per layer in FP32.
    let halves = 3 * B * 64 + 2 * B * (8 * 64 + 4 * 128 + 4 * 32 + 4 * 128);
    let floats = B * 64 + 2 * 2 * 4 * B * 12;
    assert_eq!(readback, (halves + floats) * 4);
    // Each of 48 allocations includes seven guard elements, Half allocations
    // get a further two bytes from u32 backing. 43 Half +5 Float arrays.
    assert_eq!(gpu, halves * 2 + floats * 4 + 43 * 16 + 5 * 28);
    assert!(metadata >= 48 * size_of::<snapshots::Snapshot>());
    assert!(mul(usize::MAX, 2).is_err());
    assert!(extent(usize::MAX, 2).is_err());
    assert!(admit(101, 100).is_err());
    assert_eq!(snapshots::plan(&c, false).unwrap().1, 0);
}

#[test]
fn shared_zero_bias_sizes_every_projection_for_mixed_leaf_geometry() {
    let mut mixed = config();
    mixed.hidden_size = 640;
    geometry(&mixed, &specs(), 4).unwrap();
    assert_eq!(
        (mixed.intermediate_size, mixed.head_dim, mixed.vocab_size),
        (128, 32, 64)
    );
    assert_eq!(zero_bias_bytes(&mixed).unwrap(), (640 + GUARD) * 2);
    assert_eq!(zero_bias_bytes(&config()).unwrap(), (128 + GUARD) * 2);
    let mut overflow = mixed;
    overflow.query_heads = usize::MAX;
    assert!(zero_bias_bytes(&overflow).is_err());
}

#[test]
fn accounted_peak_covers_both_upload_and_session_phases() {
    assert_eq!(accounted_peak(500, 800), 800);
    assert_eq!(accounted_peak(800, 500), 800);
    assert_eq!(accounted_peak(800, 800), 800);
}
