//! First component screen. No generation, graph scheduling or profile adoption.
use super::*;
use crate::metal::gemma_affine::project::head_batch_experiment::{Counts, Scratch};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{io::Write, time::Instant};

const CALLER: usize = 16 * 1024 * 1024;
const HELPER: usize = 32 * 1024 * 1024;

#[derive(Serialize)]
struct Pair {
    pair_index: usize,
    order: &'static str,
    legacy_seconds: f64,
    batched_seconds: f64,
    outputs_bitwise: bool,
    output_sha256: String,
    legacy_counts: Counts,
    batched_counts: Counts,
}

fn digest(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn pairs(
    scratch: &mut Scratch<'_>,
    input: &[f32],
    legacy: &mut [f32],
    batched: &mut [f32],
    count: usize,
    cohort: usize,
) -> Vec<Pair> {
    let mut result = Vec::with_capacity(count);
    for pair_index in 0..count {
        let legacy_first = (pair_index + cohort - 1).is_multiple_of(2);
        let (legacy_seconds, legacy_counts, batched_seconds, batched_counts) = if legacy_first {
            let started = Instant::now();
            let before = scratch.legacy(input, legacy).unwrap();
            let before_seconds = started.elapsed().as_secs_f64();
            let started = Instant::now();
            let after = scratch.batched(input, batched).unwrap();
            (
                before_seconds,
                before,
                started.elapsed().as_secs_f64(),
                after,
            )
        } else {
            let started = Instant::now();
            let after = scratch.batched(input, batched).unwrap();
            let after_seconds = started.elapsed().as_secs_f64();
            let started = Instant::now();
            let before = scratch.legacy(input, legacy).unwrap();
            (
                started.elapsed().as_secs_f64(),
                before,
                after_seconds,
                after,
            )
        };
        assert!(legacy_seconds.is_finite() && legacy_seconds > 0.);
        assert!(batched_seconds.is_finite() && batched_seconds > 0.);
        assert_eq!(legacy.len(), batched.len());
        for (index, (before, after)) in legacy.iter().zip(batched.iter()).enumerate() {
            assert_eq!(before.to_bits(), after.to_bits(), "full head bit {index}");
        }
        assert_eq!(
            legacy_counts,
            Counts {
                dispatches_encoded: 32,
                commands_submitted: 32,
                host_wait_calls: 32,
                input_copy_bytes: 32 * 1536 * 4,
                output_copy_bytes: 262144 * 4,
            }
        );
        assert_eq!(
            batched_counts,
            Counts {
                dispatches_encoded: 32,
                commands_submitted: 1,
                host_wait_calls: 1,
                input_copy_bytes: 1536 * 4,
                output_copy_bytes: 262144 * 4,
            }
        );
        result.push(Pair {
            pair_index,
            order: if legacy_first { "AB" } else { "BA" },
            legacy_seconds,
            batched_seconds,
            outputs_bitwise: true,
            output_sha256: digest(legacy),
            legacy_counts,
            batched_counts,
        });
    }
    result
}

pub(super) fn official_head_batch_fixed_screen() {
    let directory = std::path::PathBuf::from(
        std::env::var_os("TOKLET_GEMMA_OFFICIAL_DIRECTORY").expect("pinned offline source"),
    );
    let output = std::path::PathBuf::from(
        std::env::var_os("TOKLET_HEAD_BATCH_OUTPUT").expect("fresh JSON output"),
    );
    let cohort: usize = std::env::var("TOKLET_HEAD_BATCH_COHORT")
        .expect("explicit cohort1/2")
        .parse()
        .unwrap();
    assert!((1..=2).contains(&cohort));
    assert!(!output.exists());
    let limit = INITIAL_MEMORY_LIMIT_BYTES;
    let owner = OfficialBf16::load(&directory, 512, limit - CALLER - HELPER).unwrap();
    let super::super::resident::HeadBatchParts {
        tiles,
        rows,
        columns,
        maximum,
        input_capacity,
        output_capacity,
    } = owner.weights.head_batch_parts().unwrap();
    assert_eq!((rows, columns, tiles.len()), (262144, 1536, 32));
    assert_eq!((input_capacity, output_capacity), (393216, 262144));
    let owner_load = owner.memory().load_admitted_bytes;
    let budget = limit
        .checked_sub(owner_load)
        .and_then(|left| left.checked_sub(CALLER + HELPER))
        .unwrap();
    let mut scratch = Scratch::new(
        &owner.context.0,
        tiles,
        rows,
        columns,
        input_capacity,
        output_capacity,
        budget,
    )
    .unwrap();
    assert!(scratch.can_batch());
    let admitted = owner_load
        .checked_add(scratch.owned_bytes())
        .and_then(|bytes| bytes.checked_add(CALLER + HELPER))
        .unwrap();
    assert!(admitted <= limit);
    // Declared synthetic, exact dyadic activations; no claim these came from
    // a generation trace. Their identity is fixed across both fresh cohorts.
    let input: Vec<f32> = (0..columns)
        .map(|i| ((i % 31) as i32 - 15) as f32 / 32.)
        .collect();
    crate::metal::gemma_affine::admit(columns, maximum, &input).unwrap();
    let mut legacy = vec![0.; rows];
    let mut batched = vec![0.; rows];
    let warmups = pairs(&mut scratch, &input, &mut legacy, &mut batched, 3, cohort);
    let samples = pairs(&mut scratch, &input, &mut legacy, &mut batched, 10, cohort);
    assert!(!scratch.is_poisoned() && !scratch.has_pending());
    let report = serde_json::json!({
        "format":"toklet-gemma-head-batch-screen-v1", "cohort":cohort,
        "repository":OfficialBf16::REPOSITORY,"revision":OfficialBf16::REVISION,
        "geometry":{"rows":rows,"columns":columns,"tile_rows":8192,
            "tiles":tiles.len(),"input_capacity":input_capacity,
            "output_capacity":output_capacity},
        "input_sha256":digest(&input), "maximum_abs_coefficient":maximum,
        "owner_load_admitted_bytes":owner_load,"scratch_owned_bytes":scratch.owned_bytes(),
        "caller_allowance_bytes":CALLER,"helper_allowance_bytes":HELPER,"limit_bytes":limit,
        "warmups":warmups,"samples":samples
    });
    let raw = serde_json::to_vec_pretty(&report).unwrap();
    assert!(raw.len() <= 64 * 1024);
    let mut file = std::fs::File::create_new(output).unwrap();
    file.write_all(&raw).unwrap();
    file.write_all(b"\n").unwrap();
    file.sync_all().unwrap();
    println!(
        "HEAD_BATCH_SCREEN {}",
        serde_json::json!({"cohort":cohort,"pairs":13,"full_bit_comparisons":13*rows,
            "admitted_with_caller_helper":admitted,"diagnostic_only":true,
            "engine_adoption_qualified":false})
    );
}
