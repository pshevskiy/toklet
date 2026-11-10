//! Release baseline for one BF16 CPU attention layer with a prepared cache.
use half::bf16;
use sha2::{Digest, Sha256};
use std::{hint::black_box, time::Instant};
use toklet::{
    cpu::{LFM3_ATTENTION_SCALE, lfm3_attention_bf16},
    model::window_cache_bf16::WindowKvCache,
};

const WIDTH: usize = 128;
const KV_HEADS: usize = 4;
const QUERY_HEADS: usize = 16;

fn words(count: usize, seed: u32) -> Vec<bf16> {
    (0..count)
        .map(|i| {
            let mut x = (i as u32).wrapping_add(seed);
            x ^= x >> 16;
            x = x.wrapping_mul(0x7feb_352d);
            x ^= x >> 15;
            x = x.wrapping_mul(0x846c_a68b);
            x ^= x >> 16;
            bf16::from_bits(((0x3d00 + (x & 0x3ff)) | ((x >> 16) & 0x8000)) as u16)
        })
        .collect()
}

fn sha(words: &[bf16]) -> String {
    let mut hash = Sha256::new();
    for word in words {
        hash.update(word.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err("release build required".into());
    }
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let [queries, prefix, mode] = args.as_slice() else {
        return Err("usage: lfm3_attention_bf16_probe QUERIES PREFIX full|window".into());
    };
    let queries: usize = queries.parse()?;
    let prefix: usize = prefix.parse()?;
    if !matches!(queries, 1 | 32 | 128) || !matches!(prefix, 0 | 513 | 1023) {
        return Err("unsupported query or prefix size".into());
    }
    let window = match mode.as_str() {
        "full" => None,
        "window" => Some(256),
        _ => return Err("unsupported attention mode".into()),
    };
    let started = Instant::now();
    let mut cache = WindowKvCache::new(KV_HEADS, WIDTH, window.unwrap_or(2048), 2048, window)?;
    let old_k = words(prefix * KV_HEADS * WIDTH, 0x118f);
    let old_v = words(prefix * KV_HEADS * WIDTH, 0x252b);
    if prefix != 0 {
        cache.commit(&cache.prepare(&old_k, &old_v, prefix)?)?;
    }
    let q = words(queries * QUERY_HEADS * WIDTH, 0x397f);
    let new_k = words(queries * KV_HEADS * WIDTH, 0x440b);
    let new_v = words(queries * KV_HEADS * WIDTH, 0x59f9);
    let frame = cache.prepare(&new_k, &new_v, queries)?;
    let view = cache.view(&frame)?;
    let shape = view.attention_shape(QUERY_HEADS, LFM3_ATTENTION_SCALE);
    let segments = view.segments();
    let mut output = vec![bf16::ZERO; q.len()];
    let mut scratch = vec![0.0_f32; shape.keys];
    let prepared_ns = started.elapsed().as_nanos();
    let mut samples = [0; 14];
    for duration in &mut samples {
        let begin = Instant::now();
        lfm3_attention_bf16(black_box(&q), &segments, &mut output, &mut scratch, &shape)?;
        *duration = begin.elapsed().as_nanos();
        black_box(&output);
    }
    assert_eq!(cache.offset(), prefix);
    let hashes = [&old_k, &old_v, &q, &new_k, &new_v, &output].map(|x| sha(x));
    println!(
        "{}",
        serde_json::json!({
            "format": "toklet-lfm3-bf16-attention-v1",
            "queries": queries, "prefix": prefix, "window": window, "logical_keys": shape.keys,
            "segment_rows": segments.map(|s| s.keys), "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS, "head_dim": WIDTH, "dtype": "BF16", "capacity": 2048,
            "prepared_ns": prepared_ns, "first_ns": samples[0],
            "warmup_ns": &samples[1..4], "samples_ns": &samples[4..],
            "cache_owned_bytes": cache.allocated_bytes(),
            "query_bytes": q.capacity()*2, "context_bytes": output.capacity()*2,
            "prefix_staging_bytes": (old_k.capacity()+old_v.capacity())*2,
            "proposal_bytes": (new_k.capacity()+new_v.capacity())*2,
            "softmax_scratch_bytes": scratch.capacity()*4,
            "hashes": {"old_k": hashes[0], "old_v": hashes[1], "q": hashes[2],
                       "new_k": hashes[3], "new_v": hashes[4], "output": hashes[5]},
        })
    );
    Ok(())
}
