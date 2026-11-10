use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

const FIXTURE: &[u8] =
    include_bytes!("../../../tests/fixtures/lfm3-attention-bf16-v1/fixture.json");
const CONTRACT: &[u8] =
    include_bytes!("../../../tests/fixtures/lfm3-attention-bf16-v1/contract.md");
const EMPTY: KvSegment<'static, bf16> = KvSegment {
    k: &[],
    v: &[],
    keys: 0,
    head_stride: 0,
    key_start: 0,
};

fn words(len: usize, seed: u32, profile: &str) -> Vec<bf16> {
    (0..len)
        .map(|i| {
            let mut x = (i as u32).wrapping_add(seed);
            x ^= x >> 16;
            x = x.wrapping_mul(0x7feb_352d);
            x ^= x >> 15;
            x = x.wrapping_mul(0x846c_a68b);
            x ^= x >> 16;
            let bits = match profile {
                "edges" => [0, 1, 0x7f, 0x80, 0x3700, 0x3b80, 0x3f80, 0x4000][(x % 8) as usize],
                "cancellation" => 0x3600 + (x & 0x7ff),
                "mixed" => 0x3d00 + (x & 0x3ff),
                _ => panic!("Unknown fixture input profile"),
            };
            bf16::from_bits((bits | ((x >> 16) & 0x8000)) as u16)
        })
        .collect()
}

fn digest(data: &[bf16]) -> String {
    let bytes: Vec<_> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    format!("{:x}", Sha256::digest(bytes))
}

fn segments<'a>(
    k: &'a [bf16],
    v: &'a [bf16],
    s: &SpanAttentionShape,
    cuts: [usize; 2],
) -> [KvSegment<'a, bf16>; 3] {
    let stride = s.keys * WIDTH;
    [(0, cuts[0]), (cuts[0], cuts[1]), (cuts[1], s.keys)].map(|(first, end)| {
        if first == end {
            EMPTY
        } else {
            let span = first * WIDTH..(KV_HEADS - 1) * stride + end * WIDTH;
            KvSegment {
                k: &k[span.clone()],
                v: &v[span],
                keys: end - first,
                head_stride: stride,
                key_start: s.key_start + first,
            }
        }
    })
}

fn number(row: &Value, key: &str) -> usize {
    row[key].as_u64().unwrap().try_into().unwrap()
}

fn approx(actual: f32, reference: f64) {
    let error = (f64::from(actual) - reference).abs();
    assert!(
        error <= 1e-5 + 1e-4 * reference.abs(),
        "{actual} vs {reference}"
    );
}

fn check_digest(i: usize, name: &str, actual: &[bf16], expected: &Value) {
    assert_eq!(actual.len(), number(expected, "words"));
    let actual_sha = digest(actual);
    if actual_sha != expected["sha256"].as_str().unwrap() {
        if let Ok(path) = std::env::var("TOKLET_ATTENTION_REFERENCE_RAW") {
            let raw = std::fs::read(format!("{path}/{i:02}-{name}.bf16le")).unwrap();
            let differences = actual
                .iter()
                .zip(raw.chunks_exact(2))
                .enumerate()
                .filter_map(|(index, (value, bytes))| {
                    let expected = bf16::from_le_bytes(bytes.try_into().unwrap());
                    (value.to_bits() != expected.to_bits()).then_some((index, *value, expected))
                })
                .collect::<Vec<_>>();
            eprintln!(
                "{i}:{name} {} differences, first {:?}",
                differences.len(),
                &differences[..differences.len().min(8)]
            );
        }
        panic!("{i}: {name} differs ({actual_sha})");
    }
}

fn check_case(i: usize, case: &Value) {
    let s = SpanAttentionShape {
        queries: number(case, "tokens"),
        keys: number(case, "keys"),
        query_heads: QUERY_HEADS,
        kv_heads: KV_HEADS,
        head_dim: WIDTH,
        query_start: number(case, "prefix"),
        key_start: number(case, "key_start"),
        window: case["window"].as_u64().map(|x| x as usize),
        scale: LFM3_ATTENTION_SCALE,
    };
    let profile = case["profile"].as_str().unwrap();
    let seed = number(case, "seed") as u32;
    let q = words(s.queries * QUERY_HEADS * WIDTH, seed, profile);
    let k = words(s.keys * KV_HEADS * WIDTH, seed + 0x397, profile);
    let v = words(s.keys * KV_HEADS * WIDTH, seed + 0xb21, profile);
    for (name, input) in [("q", &q), ("k", &k), ("v", &v)] {
        assert_eq!(digest(input), case["inputs"][name].as_str().unwrap());
    }
    let spans = segments(&k, &v, &s, [s.keys / 3, s.keys * 2 / 3]);
    let mut context = vec![bf16::NAN; q.len()];
    let mut scratch = vec![f32::NAN; s.keys];
    lfm3_attention_bf16(&q, &spans, &mut context, &mut scratch, &s).unwrap();

    let mut qk = Vec::new();
    let mut scaled = Vec::new();
    let mut scores = Vec::new();
    let mut probabilities = Vec::new();
    let small = s.queries * s.keys <= 3;
    for head in 0..QUERY_HEADS {
        let kv_head = head / 4;
        for token in 0..s.queries {
            let q_base = (token * QUERY_HEADS + head) * WIDTH;
            let query = &q[q_base..q_base + WIDTH];
            let position = s.query_start + token;
            let first = s.window.map_or(0, |w| position.saturating_sub(w - 1));
            let sample_key = (head * 53 + token * 37) % s.keys;
            for (j, value) in scratch.iter_mut().enumerate() {
                let key_base = (kv_head * s.keys + j) * WIDTH;
                let key = &k[key_base..key_base + WIDTH];
                let raw = if small {
                    qk_dot::<true>(query, key)
                } else {
                    qk_dot::<false>(query, key)
                };
                if j == sample_key {
                    let expected = query
                        .iter()
                        .zip(key)
                        .map(|(a, b)| f64::from(a.to_f32()) * f64::from(b.to_f32()))
                        .sum();
                    approx(raw, expected);
                }
                let raw = bf16::from_f32(raw);
                let scale = scale_score(raw);
                let visible = (first..=position).contains(&(s.key_start + j));
                *value = mask_score(scale, visible);
                qk.push(raw);
                scaled.push(scale);
                scores.push(bf16::from_f32(*value));
            }
            let maximum = scratch.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator: f64 = scratch
                .iter()
                .map(|x| (f64::from(*x) - f64::from(maximum)).exp())
                .sum();
            let expected =
                (f64::from(scratch[sample_key]) - f64::from(maximum)).exp() / denominator;
            softmax(&mut scratch);
            approx(scratch[sample_key], expected);
            for probability in &mut scratch {
                let word = bf16::from_f32(*probability);
                probabilities.push(word);
                *probability = word.to_f32();
            }
            let mixed = if small {
                mix_values::<true>(&scratch, &spans, kv_head)
            } else {
                mix_values::<false>(&scratch, &spans, kv_head)
            };
            let feature = (head * 61 + token * 17) % WIDTH;
            let expected = scratch
                .iter()
                .enumerate()
                .map(|(j, p)| {
                    f64::from(*p) * f64::from(v[(kv_head * s.keys + j) * WIDTH + feature].to_f32())
                })
                .sum();
            approx(mixed[feature], expected);
        }
    }

    for (name, actual) in [
        ("qk", qk),
        ("scaled", scaled),
        ("scores", scores),
        ("probabilities", probabilities),
        ("context", context),
    ] {
        check_digest(i, name, &actual, &case["stages"][name]);
    }
}

fn fixture() -> Value {
    assert_eq!(
        format!("{:x}", Sha256::digest(FIXTURE)),
        "574a71182ea5503842ad140fd5e4545b5b914040c6bc2af959ef0a848e281676",
    );
    let data: Value = serde_json::from_slice(FIXTURE).unwrap();
    assert_eq!(
        data["contract_sha256"],
        format!("{:x}", Sha256::digest(CONTRACT))
    );
    let producer = include_bytes!("../../../tools/reference/lfm3_attention_bf16_fixture.py");
    assert_eq!(
        data["producer_sha256"],
        format!("{:x}", Sha256::digest(producer))
    );
    data
}

#[test]
fn source_dispatch_and_mask_stages() {
    let f = fixture();
    for (i, case) in f["cases"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .filter(|(_, c)| number(c, "tokens") * number(c, "keys") <= 8200)
    {
        check_case(i, case);
    }
}

#[test]
#[ignore = "large exact source component cohorts, run explicitly in release"]
fn deep_source_attention_stages() {
    let f = fixture();
    for (i, case) in f["cases"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .filter(|(_, c)| number(c, "tokens") * number(c, "keys") > 8200)
    {
        check_case(i, case);
    }
}

#[test]
fn source_small_bmm_contracts_before_subnormal_rounding() {
    // The last exact product is half of the smallest FP32 subnormal. The
    // previous sum is one FP32 subnormal below a BF16 midpoint that rounds up.
    let mut q = [bf16::ZERO; WIDTH];
    let mut k = [bf16::ZERO; WIDTH];
    for (dst, x) in q
        .iter_mut()
        .zip([2f32.powi(-66), -2f32.powi(-74), 2f32.powi(-75)])
    {
        *dst = bf16::from_f32(x);
    }
    for (dst, x) in k
        .iter_mut()
        .zip([3.0 * 2f32.powi(-68), 2f32.powi(-75), 2f32.powi(-75)])
    {
        *dst = bf16::from_f32(x);
    }
    assert_eq!(bf16::from_f32(qk_dot::<true>(&q, &k)).to_bits(), 2);
    assert_eq!(bf16::from_f32(qk_dot::<false>(&q, &k)).to_bits(), 1);
}

#[test]
fn wrapped_cache_attention_matches_contiguous_and_does_not_commit() {
    use crate::model::window_cache_bf16::WindowKvCache;
    let prefix = 513;
    let tokens = 4;
    let mut cache = WindowKvCache::new(KV_HEADS, WIDTH, WINDOW, 2048, Some(WINDOW)).unwrap();
    let k = words(KV_HEADS * prefix * WIDTH, 1234, "mixed");
    let v = words(KV_HEADS * prefix * WIDTH, 4567, "mixed");
    cache
        .commit(&cache.prepare(&k, &v, prefix).unwrap())
        .unwrap();
    let generation = cache.generation();
    let (old_k, old_v) = cache.committed();
    let original = (digest(old_k), digest(old_v));
    let new_k = words(KV_HEADS * tokens * WIDTH, 8910, "mixed");
    let new_v = words(KV_HEADS * tokens * WIDTH, 1112, "mixed");
    let frame = cache.prepare(&new_k, &new_v, tokens).unwrap();
    let view = cache.view(&frame).unwrap();
    let shape = view.attention_shape(QUERY_HEADS, LFM3_ATTENTION_SCALE);
    let spans = view.segments();
    assert_eq!(spans.map(|span| span.keys), [254, 1, 4]);
    let linear = |before: &[bf16], fresh: &[bf16]| {
        before
            .chunks_exact(prefix * WIDTH)
            .zip(fresh.chunks_exact(tokens * WIDTH))
            .flat_map(|(a, b)| a[(prefix - WINDOW + 1) * WIDTH..].iter().chain(b).copied())
            .collect::<Vec<_>>()
    };
    let expected_k = linear(&k, &new_k);
    let expected_v = linear(&v, &new_v);
    let contiguous = segments(&expected_k, &expected_v, &shape, [0, 0]);
    let q = words(tokens * QUERY_HEADS * WIDTH, 1314, "mixed");
    let mut actual = vec![bf16::NAN; q.len()];
    let mut expected = actual.clone();
    let mut scratch = vec![f32::NAN; shape.keys];
    lfm3_attention_bf16(&q, &spans, &mut actual, &mut scratch, &shape).unwrap();
    lfm3_attention_bf16(&q, &contiguous, &mut expected, &mut scratch, &shape).unwrap();
    assert_eq!(digest(&actual), digest(&expected));
    let (old_k, old_v) = cache.committed();
    assert_eq!((digest(old_k), digest(old_v)), original);
    assert_eq!((cache.offset(), cache.generation()), (prefix, generation));
}

#[test]
fn admission_rejects_before_writes_and_padding_is_unread() {
    let s = SpanAttentionShape {
        queries: 2,
        keys: 2,
        query_heads: QUERY_HEADS,
        kv_heads: KV_HEADS,
        head_dim: WIDTH,
        query_start: 0,
        key_start: 0,
        window: None,
        scale: LFM3_ATTENTION_SCALE,
    };
    let q = vec![bf16::ONE; s.queries * QUERY_HEADS * WIDTH];
    let stride = 3 * WIDTH;
    let mut k = vec![bf16::ONE; (KV_HEADS - 1) * stride + 2 * WIDTH];
    for h in 0..KV_HEADS - 1 {
        k[h * stride + 2 * WIDTH..(h + 1) * stride].fill(bf16::NAN);
    }
    let span = |k| KvSegment {
        k,
        v: k,
        keys: 2,
        head_stride: stride,
        key_start: 0,
    };
    let mut context = vec![bf16::MAX; q.len()];
    let mut scratch = [3.0, 3.0];
    let good = span(&k);
    let bad_origin = KvSegment {
        key_start: 1,
        ..good
    };
    let overlapping = KvSegment {
        head_stride: WIDTH,
        ..good
    };
    let short = KvSegment {
        k: &good.k[..good.k.len() - 1],
        ..good
    };
    for bad in [bad_origin, overlapping, short] {
        assert!(
            lfm3_attention_bf16(&q, &[EMPTY, EMPTY, bad], &mut context, &mut scratch, &s).is_err()
        );
    }
    assert!(
        lfm3_attention_bf16(
            &q,
            &[EMPTY, EMPTY, good],
            &mut context,
            &mut scratch[..1],
            &s
        )
        .is_err()
    );
    let mut bad_q = q.clone();
    *bad_q.last_mut().unwrap() = bf16::INFINITY;
    assert!(
        lfm3_attention_bf16(
            &bad_q,
            &[EMPTY, EMPTY, good],
            &mut context,
            &mut scratch,
            &s
        )
        .is_err()
    );
    let mut bad_keys = k.clone();
    bad_keys[WIDTH] = bf16::NAN; // Future for the first query.
    assert!(
        lfm3_attention_bf16(
            &q,
            &[EMPTY, EMPTY, span(&bad_keys)],
            &mut context,
            &mut scratch,
            &s
        )
        .is_err()
    );
    assert!(context.iter().all(|x| *x == bf16::MAX));
    assert_eq!(scratch, [3.0, 3.0]);
    lfm3_attention_bf16(&q, &[EMPTY, EMPTY, good], &mut context, &mut scratch, &s).unwrap();
    assert!(context.iter().all(|x| *x == bf16::ONE));

    let mut extreme_keys = k.clone();
    extreme_keys[..WIDTH].fill(bf16::ZERO);
    extreme_keys[WIDTH] = bf16::MAX;
    let extreme_q = vec![bf16::MAX; q.len()];
    let overflow = lfm3_attention_bf16(
        &extreme_q,
        &[EMPTY, EMPTY, span(&extreme_keys)],
        &mut context,
        &mut scratch,
        &s,
    );
    assert!(matches!(
        overflow,
        Err(Error::NonFinite {
            tensor: "LFM3 attention QK",
            index: 1
        })
    ));
}
