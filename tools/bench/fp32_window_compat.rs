//! Fixed existing FP32 state API workload, reused unchanged before/after BF16 storage support.
use smallvec::SmallVec;
use std::{hint::black_box, time::Instant};
use toklet::model::{
    Error,
    window_cache::{PreparedKv, WindowKvCache, commit_batch, reset_batch},
};

fn step(caches: &mut [WindowKvCache], k: &[f32], v: &[f32], tokens: usize) -> Result<(), Error> {
    let mut frames: SmallVec<[PreparedKv<'_>; 11]> = SmallVec::new();
    for cache in caches.iter() {
        let frame = cache.prepare(k, v, tokens)?;
        let view = cache.view(&frame)?;
        black_box(view.segments());
        black_box(view.attention_shape(16, 0.125));
        for head in 0..4 {
            black_box(view.head(head)?);
        }
        black_box(view.visible_range(cache.offset() + tokens - 1)?);
        frames.push(frame);
    }
    commit_batch(caches, &frames)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err("release build required".into());
    }
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 || args[0] != "--kind" || args[2] != "--tokens" {
        return Err("usage: fp32_window_compat --kind full|sliding --tokens 1|32|128".into());
    }
    let window = match args[1].as_str() {
        "full" => None,
        "sliding" => Some(512),
        _ => return Err("invalid kind".into()),
    };
    let tokens: usize = args[3].parse()?;
    if !matches!(tokens, 1 | 32 | 128) {
        return Err("invalid token count".into());
    }
    let capacity = window.unwrap_or(2048);
    let mut caches = (0..11)
        .map(|_| WindowKvCache::new(4, 128, capacity, 2048, window))
        .collect::<Result<Vec<_>, _>>()?;
    let k: Vec<_> = (0..4 * 511 * 128)
        .map(|i| ((i * 7 % 257) as f32 - 128.) / 32.)
        .collect();
    let v: Vec<_> = (0..k.len())
        .map(|i| ((i * 11 % 257) as f32 - 128.) / 32.)
        .collect();
    let mut samples = Vec::with_capacity(10);
    let repeats = (64 / tokens).max(1);
    let mut raw = Vec::with_capacity(10 * repeats);
    for cohort in 0..13 {
        let mut total = 0_u128;
        for _ in 0..repeats {
            reset_batch(&mut caches)?;
            step(&mut caches, &k, &v, 511)?;
            let begin = Instant::now();
            step(
                &mut caches,
                black_box(&k[..4 * tokens * 128]),
                black_box(&v[..4 * tokens * 128]),
                tokens,
            )?;
            let elapsed = begin.elapsed().as_nanos();
            black_box(caches[0].committed());
            total += elapsed;
            if cohort >= 3 {
                raw.push(elapsed);
            }
        }
        if cohort >= 3 {
            samples.push(total as f64 / repeats as f64);
        }
    }
    let mut ordered = samples.clone();
    ordered.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({
            "format":"toklet-fp32-window-compat-micro-v1","kind":args[1],"storage":"F32",
            "heads":4,"head_dim":128,"owners":11,"capacity":capacity,"max_positions":2048,
            "window":window,"offset":511,"tokens_per_owner":tokens,"warmup_cohorts":3,
            "measured_cohorts":10,"requests_per_cohort":repeats,
            "samples_ns_per_atomic_batch":samples,"raw_request_ns":raw,
            "median_ns_per_atomic_batch":(ordered[4]+ordered[5])*0.5,
            "owner_bytes":caches.iter().map(WindowKvCache::allocated_bytes).sum::<usize>(),
            "caller_seed_and_input_bytes":(k.capacity()+v.capacity())*4,
            "timing_boundary":"Prepare11FP32chunks, borrow segments/all4headviews/lastqueryrange, atomiccommit; reset/511rowseed outside timing",
            "all_numerical_buffers_reused":true,"model_execution_qualified":false
        })
    );
    Ok(())
}
