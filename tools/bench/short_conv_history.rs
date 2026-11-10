// Shared body for concrete three/seven-tap release-only console probes.
use profile::{History, KERNEL, Prepared, commit_batch, reset_batch};
use smallvec::SmallVec;
use std::{hint::black_box, time::Instant};

fn step(
    states: &mut [History],
    candidates: &mut [Vec<f32>],
    work: &mut [f32],
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    tokens: usize,
) -> Result<(), toklet::model::Error> {
    let mut frames: SmallVec<[Prepared<'_>; 13]> = SmallVec::new();
    for (history, candidate) in states.iter().zip(candidates) {
        frames.push(history.prepare_chunk(input, weight, candidate, work, output, tokens)?);
    }
    commit_batch(states, &frames)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err("release build required".into());
    }
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 || args[1] != "--tokens" {
        return Err("usage: short_conv_history --tokens 1|32|128".into());
    }
    let tokens: usize = args[2].parse()?;
    if ![1, 32, 128].contains(&tokens) {
        return Err("tokens must be 1, 32 or 128".into());
    }
    let channels = 2048;
    let owners = 13;
    let repeats = (64 / tokens).max(1);
    let mut states = (0..owners)
        .map(|_| History::new(channels, tokens + 6))
        .collect::<Result<Vec<_>, _>>()?;
    let bank = (KERNEL - 1) * channels;
    let mut candidates = (0..owners).map(|_| vec![0.; bank]).collect::<Vec<_>>();
    let mut work = vec![0.; bank];
    let weight = (0..KERNEL * channels)
        .map(|i| ((i * 11 + 3) % 31) as f32 / 32. - 0.5)
        .collect::<Vec<_>>();
    let input = (0..tokens.max(6) * channels)
        .map(|i| ((i * 7 + 4) % 31) as f32 / 16. - 1.)
        .collect::<Vec<_>>();
    let mut output = vec![0.; input.len()];
    let mut samples = Vec::with_capacity(10);
    let mut raw = Vec::with_capacity(10 * repeats);
    for cohort in 0..13 {
        let mut total = 0_u128;
        for _ in 0..repeats {
            reset_batch(&mut states)?;
            step(
                &mut states,
                &mut candidates,
                &mut work,
                &input[..6 * channels],
                &weight,
                &mut output[..6 * channels],
                6,
            )?;
            let start = Instant::now();
            step(
                &mut states,
                &mut candidates,
                &mut work,
                black_box(&input[..tokens * channels]),
                black_box(&weight),
                black_box(&mut output[..tokens * channels]),
                tokens,
            )?;
            let elapsed = start.elapsed().as_nanos();
            black_box(states[0].committed());
            black_box(&output);
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
    let median = (ordered[4] + ordered[5]) * 0.5;
    let owner_bytes = states
        .iter()
        .try_fold(0_usize, |sum, h| h.allocated_bytes().map(|n| sum + n))?;
    let candidate_bytes = candidates.iter().map(|v| v.capacity() * 4).sum::<usize>();
    let record = serde_json::json!({
        "format":"toklet-short-conv-history-bench-v1","kernel":KERNEL,"channels":channels,"owners":owners,"tokens_per_owner":tokens,
        "warmup_cohorts":3,"measured_cohorts":10,"requests_per_cohort":repeats,
        "samples_ns_per_atomic_batch":samples,"raw_request_ns":raw,"median_ns_per_atomic_batch":median,
        "timing_boundary":"Prepare all13 chunks and atomic batch commit; reset/six-row seed outside timing; loaded weights and reused buffers",
        "history_owner_bytes":owner_bytes,"candidate_data_bytes":candidate_bytes,"shared_work_data_bytes":work.capacity()*4,
        "weight_data_bytes":weight.capacity()*4,"input_data_bytes":input.capacity()*4,"output_data_bytes":output.capacity()*4,
        "all_numerical_buffers_reused":true,"engine_model_execution":false,"full_model_memory_qualified":false,"performance_adoption_qualified":false
    });
    println!("{}", serde_json::to_string(&record)?);
    Ok(())
}
