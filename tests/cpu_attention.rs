mod support;

use support::Fixture;
use toklet::cpu::{self, AttentionShape, Error};

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs(),
            "index {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn rotary_and_gqa_attention_match_all_saved_intermediates() {
    let fixture = Fixture::tiny();
    for case in ["prompt", "future_changed", "offset", "single", "capacity"] {
        let tokens = fixture.u32(&format!("cases.{case}.tokens")).len();
        let offset = fixture.manifest["cases"][case]["position_offset"]
            .as_u64()
            .unwrap() as usize;
        for layer in 0..2 {
            let prefix = format!("cases.{case}.layers.{layer}");
            let mut q = vec![0.0; tokens * 32];
            let mut k = vec![0.0; tokens * 16];
            cpu::rotary(
                &fixture.f32(&format!("{prefix}.q_projection")),
                &mut q,
                tokens,
                4,
                8,
                offset,
                1e6,
            )
            .unwrap();
            cpu::rotary(
                &fixture.f32(&format!("{prefix}.k_projection")),
                &mut k,
                tokens,
                2,
                8,
                offset,
                1e6,
            )
            .unwrap();
            close(&q, &fixture.f32(&format!("{prefix}.q_rotary")));
            close(&k, &fixture.f32(&format!("{prefix}.k_rotary")));
            let mut scores = vec![0.0; 4 * tokens * tokens];
            let mut probabilities = vec![0.0; scores.len()];
            let mut context = vec![0.0; tokens * 32];
            cpu::attention(
                &q,
                &k,
                &fixture.f32(&format!("{prefix}.v_projection")),
                &mut context,
                &mut scores,
                &mut probabilities,
                &AttentionShape {
                    tokens,
                    query_heads: 4,
                    kv_heads: 2,
                    head_dim: 8,
                },
            )
            .unwrap();
            close(&scores, &fixture.f32(&format!("{prefix}.attention_scores")));
            close(
                &probabilities,
                &fixture.f32(&format!("{prefix}.attention_probabilities")),
            );
            close(
                &context,
                &fixture.f32(&format!("{prefix}.attention_context")),
            );
        }
    }
}

#[test]
fn rotary_uses_split_half_pairing_and_absolute_chunk_offsets() {
    let input = [1.0, 2.0, 3.0, 4.0];
    let mut output = [0.0; 4];
    cpu::rotary(&input, &mut output, 1, 1, 4, 0, 1.0).unwrap();
    assert_eq!(output, input);
    cpu::rotary(&input, &mut output, 1, 1, 4, 1, 1.0).unwrap();
    let (sin, cos) = 1.0_f32.sin_cos();
    close(
        &output,
        &[
            cos - 3.0 * sin,
            2.0 * cos - 4.0 * sin,
            sin + 3.0 * cos,
            2.0 * sin + 4.0 * cos,
        ],
    );
    let fixture = Fixture::tiny();
    let input = fixture.f32("cases.capacity.layers.0.q_projection");
    let mut whole = vec![0.0; input.len()];
    cpu::rotary(&input, &mut whole, 32, 4, 8, 5, 1e6).unwrap();
    for token in 0..32 {
        let mut one = [0.0; 32];
        cpu::rotary(
            &input[token * 32..(token + 1) * 32],
            &mut one,
            1,
            4,
            8,
            5 + token,
            1e6,
        )
        .unwrap();
        assert_eq!(one, whole[token * 32..(token + 1) * 32]);
        for head in 0..4 {
            let start = token * 32 + head * 8;
            let before: f64 = input[start..start + 8]
                .iter()
                .map(|&x| f64::from(x).powi(2))
                .sum();
            let after: f64 = whole[start..start + 8]
                .iter()
                .map(|&x| f64::from(x).powi(2))
                .sum();
            assert!((before - after).abs() <= 1e-6 * before.max(1.0));
        }
    }
}

#[test]
fn grouped_heads_read_their_shared_value_head_without_expansion() {
    let mut context = [0.0; 8];
    let mut scores = [0.0; 4];
    let mut probabilities = [0.0; 4];
    cpu::attention(
        &[0.0; 8],
        &[0.0; 4],
        &[1.0, 2.0, 3.0, 4.0],
        &mut context,
        &mut scores,
        &mut probabilities,
        &AttentionShape {
            tokens: 1,
            query_heads: 4,
            kv_heads: 2,
            head_dim: 2,
        },
    )
    .unwrap();
    assert_eq!(context, [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    assert_eq!(scores, [0.0; 4]);
    assert_eq!(probabilities, [1.0; 4]);
}

#[test]
fn changing_future_keys_queries_and_values_preserves_earlier_context() {
    let fixture = Fixture::tiny();
    let mut q = fixture.f32("cases.prompt.layers.0.q_rotary");
    let mut k = fixture.f32("cases.prompt.layers.0.k_rotary");
    let mut v = fixture.f32("cases.prompt.layers.0.v_projection");
    let mut before = [0.0; 128];
    let mut after = [0.0; 128];
    let mut scores = [0.0; 64];
    let mut probabilities = [0.0; 64];
    let shape = AttentionShape {
        tokens: 4,
        query_heads: 4,
        kv_heads: 2,
        head_dim: 8,
    };
    cpu::attention(
        &q,
        &k,
        &v,
        &mut before,
        &mut scores,
        &mut probabilities,
        &shape,
    )
    .unwrap();
    q[96..].fill(7.0);
    k[48..].fill(-5.0);
    v[48..].fill(100.0);
    cpu::attention(
        &q,
        &k,
        &v,
        &mut after,
        &mut scores,
        &mut probabilities,
        &shape,
    )
    .unwrap();
    assert_eq!(before[..96], after[..96]);
    assert_ne!(before[96..], after[96..]);
    for head in 0..4 {
        for query in 0..4 {
            let row = &probabilities[(head * 4 + query) * 4..(head * 4 + query + 1) * 4];
            assert_eq!(&row[query + 1..], vec![0.0; 3 - query]);
        }
    }
}

#[test]
fn invalid_position_and_attention_inputs_fail_before_mutation() {
    let mut output = [123.0; 4];
    for theta in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            cpu::rotary(&[1.0; 4], &mut output, 1, 1, 4, 0, theta),
            Err(Error::InvalidParameter(_))
        ));
    }
    assert!(matches!(
        cpu::rotary(&[1.0; 3], &mut output[..3], 1, 1, 3, 0, 1e6),
        Err(Error::InvalidParameter(_))
    ));
    assert_eq!(
        cpu::rotary(&[1.0; 4], &mut output, 2, 1, 2, usize::MAX, 1e6),
        Err(Error::SizeOverflow)
    );
    assert_eq!(
        cpu::rotary(&[], &mut [], 1, usize::MAX, 2, 0, 1e6),
        Err(Error::SizeOverflow)
    );
    assert!(matches!(
        cpu::rotary(&[f32::NAN; 4], &mut output, 1, 1, 4, 0, 1e6),
        Err(Error::NonFinite { .. })
    ));
    let mut scores = [456.0; 2];
    let mut probabilities = [789.0; 2];
    let mut shape = AttentionShape {
        tokens: 1,
        query_heads: 2,
        kv_heads: 3,
        head_dim: 2,
    };
    assert!(matches!(
        cpu::attention(
            &[1.0; 4],
            &[1.0; 6],
            &[1.0; 6],
            &mut output,
            &mut scores,
            &mut probabilities,
            &shape
        ),
        Err(Error::InvalidParameter(_))
    ));
    shape.kv_heads = 1;
    assert!(matches!(
        cpu::attention(
            &[1.0; 4],
            &[1.0; 2],
            &[f32::NAN; 2],
            &mut output,
            &mut scores,
            &mut probabilities,
            &shape
        ),
        Err(Error::NonFinite { tensor: "v", .. })
    ));
    assert!(matches!(
        cpu::attention(
            &[1.0; 4],
            &[1.0; 1],
            &[1.0; 2],
            &mut output,
            &mut scores,
            &mut probabilities,
            &shape
        ),
        Err(Error::Length { tensor: "k", .. })
    ));
    assert_eq!(output, [123.0; 4]);
    assert_eq!(scores, [456.0; 2]);
    assert_eq!(probabilities, [789.0; 2]);
    shape.tokens = usize::MAX;
    assert_eq!(
        cpu::attention(&[], &[], &[], &mut [], &mut [], &mut [], &shape),
        Err(Error::SizeOverflow)
    );
}

#[test]
fn arithmetic_overflow_requires_discarding_partial_outputs() {
    assert!(matches!(
        cpu::rotary(&[f32::MAX; 2], &mut [0.0; 2], 1, 1, 2, 1, 1.0),
        Err(Error::NonFinite {
            tensor: "rotary output",
            ..
        })
    ));
    assert!(matches!(
        cpu::attention(
            &[f32::MAX],
            &[2.0],
            &[0.0],
            &mut [0.0],
            &mut [0.0],
            &mut [0.0],
            &AttentionShape {
                tokens: 1,
                query_heads: 1,
                kv_heads: 1,
                head_dim: 1
            }
        ),
        Err(Error::NonFinite {
            tensor: "attention scores",
            ..
        })
    ));
}
