mod support;
use support::Fixture;
use toklet::cpu::{self, CachedAttentionShape};

#[test]
fn head_major_cache_with_stale_tails_matches_frozen_gqa_at_all_offsets() {
    let f = Fixture::tiny();
    for case in ["prompt", "future_changed", "offset", "single", "capacity"] {
        for layer in 0..2 {
            let prefix = format!("cases.{case}.layers.{layer}");
            let q = f.f32(&format!("{prefix}.q_rotary"));
            let k = f.f32(&format!("{prefix}.k_rotary"));
            let v = f.f32(&format!("{prefix}.v_projection"));
            let expected = f.f32(&format!("{prefix}.attention_context"));
            let tokens = f.u32(&format!("cases.{case}.tokens")).len();
            assert_eq!(q.len(), tokens * 32);
            let capacity = tokens + 3;
            let mut kc = vec![f32::NAN; 2 * capacity * 8];
            let mut vc = kc.clone();
            for head in 0..2 {
                for token in 0..tokens {
                    let source = token * 16 + head * 8;
                    let target = (head * capacity + token) * 8;
                    kc[target..target + 8].copy_from_slice(&k[source..source + 8]);
                    vc[target..target + 8].copy_from_slice(&v[source..source + 8]);
                }
            }
            // Each suffix query sees exactly its earlier cached keys, including
            // unequal GQA head groups and a final irregular query chunk.
            for offset in 0..tokens {
                let n = tokens - offset;
                let mut context = vec![0.0; n * 32];
                cpu::attention_cached(
                    &q[offset * 32..],
                    &kc,
                    &vc,
                    &mut context,
                    &mut vec![0.0; capacity],
                    &mut vec![0.0; capacity],
                    &CachedAttentionShape {
                        tokens: n,
                        query_heads: 4,
                        kv_heads: 2,
                        head_dim: 8,
                        capacity,
                        offset,
                    },
                )
                .unwrap();
                for (&a, &e) in context.iter().zip(&expected[offset * 32..]) {
                    assert!((a - e).abs() <= 1e-5 + 1e-4 * e.abs());
                }
            }
        }
    }
}

#[test]
fn cached_attention_rejects_invalid_shapes_or_valid_prefix_before_writes() {
    for variant in 0..7 {
        let mut shape = CachedAttentionShape {
            tokens: 1,
            query_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            capacity: 3,
            offset: 1,
        };
        let mut k = [0.0; 6];
        let mut v = [0.0; 6];
        let q = [0.0; 4];
        match variant {
            0 => shape.offset = usize::MAX,
            1 => shape.offset = 3,
            2 => shape.kv_heads = 3,
            3 => shape.tokens = 0,
            4 => shape.head_dim = usize::MAX,
            5 => k[0] = f32::NAN,
            _ => v[2] = f32::INFINITY,
        }
        let mut context = [123.0; 4];
        let mut scores = [456.0; 3];
        let mut probabilities = [789.0; 3];
        assert!(
            cpu::attention_cached(
                &q,
                &k,
                &v,
                &mut context,
                &mut scores,
                &mut probabilities,
                &shape
            )
            .is_err()
        );
        assert_eq!(context, [123.0; 4]);
        assert_eq!(scores, [456.0; 3]);
        assert_eq!(probabilities, [789.0; 3]);
    }
}
