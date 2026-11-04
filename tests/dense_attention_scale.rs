use toklet::cpu::{
    AttentionShape, CachedAttentionShape, attention_cached_scaled, attention_scaled,
};

#[test]
fn explicit_scale_preserves_causality_and_cached_mapping() {
    let qk = [1., 0., 0., 1.];
    let v = [2., 0., 0., 2.];
    let mut out = [0.; 4];
    let mut scores = [0.; 4];
    let mut probs = [0.; 4];
    let shape = AttentionShape {
        tokens: 2,
        query_heads: 1,
        kv_heads: 1,
        head_dim: 2,
    };
    attention_scaled(&qk, &qk, &v, &mut out, &mut scores, &mut probs, &shape, 2.).unwrap();
    assert_eq!(&out[..2], &[2., 0.]);
    assert_eq!(scores, [2., 0., 0., 2.]);
    assert_eq!(probs[1], 0.);
    let expected = [0.23840584, 1.7615942];
    for (a, b) in out[2..].iter().zip(expected) {
        assert!((a - b).abs() < 1e-6);
    }
    let mut cached = [0.; 2];
    attention_cached_scaled(
        &qk[2..],
        &qk,
        &v,
        &mut cached,
        &mut scores[..2],
        &mut probs[..2],
        &CachedAttentionShape {
            tokens: 1,
            query_heads: 1,
            kv_heads: 1,
            head_dim: 2,
            capacity: 2,
            offset: 1,
        },
        2.,
    )
    .unwrap();
    assert_eq!(cached, out[2..]);
    let before = out;
    assert!(attention_scaled(&qk, &qk, &v, &mut out, &mut scores, &mut probs, &shape, 0.).is_err());
    assert_eq!(out, before);
}
