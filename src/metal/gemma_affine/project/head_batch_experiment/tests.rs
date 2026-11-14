//! Explicit GPU controls for the isolated BF16 submission experiment.
use super::{Counts, Fault, Scratch};
use crate::metal::{
    Error,
    gemma_affine::{CONTEXT_ALLOWANCE_BYTES, Context, GUARD, SENTINEL, loading},
    memory::Buffer,
};
#[cfg(test)]
use crate::model::gemma_reference::tests::Measurement;
use objc2::rc::autoreleasepool;
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus};
#[cfg(not(test))]
use std::sync::OnceLock;

#[cfg(not(test))]
type MeasurementHooks = (fn(), fn() -> usize);

#[cfg(not(test))]
static MEASUREMENT_HOOKS: OnceLock<MeasurementHooks> = OnceLock::new();

// The small probe binary owns the existing kind of thread-local allocator.
// These callbacks run only at coarse measurement boundaries, outside kernels.
#[cfg(not(test))]
struct Measurement {
    finished: bool,
}

#[cfg(not(test))]
impl Measurement {
    fn start() -> Self {
        let (start, _) = MEASUREMENT_HOOKS
            .get()
            .expect("measurement hooks installed");
        start();
        Self { finished: false }
    }

    fn finish(mut self) -> usize {
        self.finished = true;
        let (_, finish) = MEASUREMENT_HOOKS
            .get()
            .expect("measurement hooks installed");
        finish()
    }
}

#[cfg(not(test))]
impl Drop for Measurement {
    fn drop(&mut self) {
        if !self.finished {
            let (_, finish) = MEASUREMENT_HOOKS
                .get()
                .expect("measurement hooks installed");
            let _ = finish();
        }
    }
}

const TILE_ROWS: usize = 8192;
const SCRATCH_BUDGET: usize = 16 * 1024 * 1024;
const CALLER_SENTINEL: f32 = f32::from_bits(0x5a31_2079);

fn context() -> Context {
    Context::with_projection_pipeline(CONTEXT_ALLOWANCE_BYTES, 16).unwrap()
}

fn upload(
    context: &Context,
    rows: usize,
    columns: usize,
    coefficient: impl Fn(usize, usize) -> f32,
) -> Vec<Buffer<u32>> {
    (0..rows)
        .step_by(TILE_ROWS)
        .map(|first| {
            let count = (rows - first).min(TILE_ROWS);
            let mut bytes = Vec::with_capacity(count * columns * 2);
            for row in first..first + count {
                for column in 0..columns {
                    let bits = coefficient(row, column).to_bits();
                    assert_eq!(bits & 0xffff, 0, "fixture must already be exact BF16");
                    bytes.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
                }
            }
            loading::upload(&context.gpu, &bytes).unwrap()
        })
        .collect()
}

fn scale(row: usize) -> f32 {
    let magnitude = [0.25, 0.5, 1.0, 2.0, 4.0][row % 5];
    if row.is_multiple_of(2) {
        magnitude
    } else {
        -magnitude
    }
}

fn exact_tiles(context: &Context, rows: usize, columns: usize) -> Vec<Buffer<u32>> {
    upload(context, rows, columns, |row, column| {
        scale(row) * [1.0, 2.0, -1.0, -2.0][column % 4]
    })
}

fn exact_input(columns: usize) -> Vec<f32> {
    (0..columns)
        .map(|column| [0.5, 0.25, -0.5, -0.25][column % 4])
        .collect()
}

fn exact_output(rows: usize, columns: usize) -> Vec<f32> {
    // Every product is exactly scale(row)/2. This analytic identity does not
    // reproduce the kernel's lane partition, compensated sum or reduction.
    (0..rows)
        .map(|row| scale(row) * (columns / 2) as f32)
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn assert_counts(counts: Counts, tiles: usize, batched: bool, rows: usize, columns: usize) {
    assert_eq!(counts.dispatches_encoded, tiles);
    assert_eq!(counts.commands_submitted, if batched { 1 } else { tiles });
    assert_eq!(counts.host_wait_calls, if batched { 1 } else { tiles });
    assert_eq!(
        counts.input_copy_bytes,
        columns * 4 * if batched { 1 } else { tiles }
    );
    assert_eq!(counts.output_copy_bytes, rows * 4);
}

#[cfg_attr(test, test)]
#[cfg_attr(
    test,
    ignore = "explicit GPU: independent BF16 oracle at two and three tiles with tails"
)]
fn head_batch_tail_shapes_match_independent_dyadic_oracle_and_actual_counts() {
    autoreleasepool(|_| {
        let context = context();
        for rows in [8193, 16386] {
            for columns in [64, 128] {
                let weights = exact_tiles(&context, rows, columns);
                let input = exact_input(columns);
                let expected = bits(&exact_output(rows, columns));
                let mut old = vec![CALLER_SENTINEL; rows];
                let mut new = old.clone();
                let mut scratch = Scratch::new(
                    &context,
                    &weights,
                    rows,
                    columns,
                    columns + 3,
                    rows + 5,
                    SCRATCH_BUDGET,
                )
                .unwrap();
                assert!(scratch.can_batch());
                let legacy = scratch.legacy(&input, &mut old).unwrap();
                let batched = scratch.batched(&input, &mut new).unwrap();
                assert_eq!(bits(&old), expected, "legacy {rows}x{columns}");
                assert_eq!(bits(&new), expected, "batched {rows}x{columns}");
                assert_counts(legacy, rows.div_ceil(TILE_ROWS), false, rows, columns);
                assert_counts(batched, rows.div_ceil(TILE_ROWS), true, rows, columns);
                assert!(!scratch.is_poisoned());
                assert!(!scratch.has_pending());
            }
        }
    });
}

#[cfg_attr(test, test)]
#[cfg_attr(
    test,
    ignore = "explicit GPU: unchanged cancellation, signed-zero and subnormal numerical policy"
)]
fn head_batch_preserves_legacy_bits_for_cancellation_zero_and_subnormal_inputs() {
    autoreleasepool(|_| {
        let context = context();
        let rows = 8193;
        let columns = 64;
        let weights = upload(&context, rows, columns, |_, _| 1.0);
        let mut scratch = Scratch::new(
            &context,
            &weights,
            rows,
            columns,
            columns,
            rows,
            SCRATCH_BUDGET,
        )
        .unwrap();
        let mut cancellation = vec![0.0; columns];
        cancellation[..3].copy_from_slice(&[16777216.0, 1.0, -16777216.0]);
        cancellation[3] = -0.0;
        let mut subnormal = vec![0.0; columns];
        subnormal[0] = f32::from_bits(1);
        for (name, input) in [
            ("cancellation", cancellation),
            ("signed_zero", vec![-0.0; columns]),
            ("subnormal", subnormal),
        ] {
            let mut old = vec![CALLER_SENTINEL; rows];
            let mut new = old.clone();
            scratch.legacy(&input, &mut old).unwrap();
            scratch.batched(&input, &mut new).unwrap();
            assert_eq!(bits(&new), bits(&old), "{name}");
            assert!(new.iter().all(|value| value.is_finite()));
            if name == "cancellation" {
                assert!(new.iter().all(|value| value.to_bits() == 1.0_f32.to_bits()));
            }
            // Subnormal handling is compared to the unchanged precise pipeline;
            // this control does not impose a new cross-device denormal policy.
        }
    });
}

#[cfg_attr(test, test)]
#[cfg_attr(
    test,
    ignore = "explicit GPU: constructor shape, pipeline, BF16 domain and exact guarded tile admission"
)]
fn head_batch_constructor_rejects_incompatible_shapes_pipeline_tiles_and_weights() {
    autoreleasepool(|_| {
        let context = context();
        let weights = exact_tiles(&context, 1, 64);
        for (rows, columns) in [
            (0, 64),
            (262145, 64),
            (1, 0),
            (1, 63),
            (1, 65),
            (1, 12352),
            (usize::MAX, 64),
        ] {
            assert!(
                Scratch::new(&context, &weights, rows, columns, 64, 1, SCRATCH_BUDGET).is_err()
            );
        }
        assert!(Scratch::new(&context, &[], 1, 64, 64, 1, SCRATCH_BUDGET).is_err());
        assert!(Scratch::new(&context, &weights, 8193, 64, 64, 8193, SCRATCH_BUDGET).is_err());
        let affine_context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        assert!(Scratch::new(&affine_context, &weights, 1, 64, 64, 1, SCRATCH_BUDGET).is_err());

        for extra in [-1_isize, 1] {
            let length = (64 / 2 + GUARD).checked_add_signed(extra).unwrap();
            let mut buffer = Buffer::new(&context.gpu, length).unwrap();
            buffer.with_mut(|words| words.fill(0x3f80_3f80));
            assert!(Scratch::new(&context, &[buffer], 1, 64, 64, 1, SCRATCH_BUDGET).is_err());
        }
        for value in [
            f32::INFINITY,
            f32::NAN,
            f32::from_bits(0x0001_0000),
            f32::from_bits(0x5000_0000),
        ] {
            let bad = upload(
                &context,
                1,
                64,
                |_, column| if column == 0 { value } else { 1.0 },
            );
            assert!(Scratch::new(&context, &bad, 1, 64, 64, 1, SCRATCH_BUDGET).is_err());
        }
        for index in 0..GUARD {
            let mut bad_guard = exact_tiles(&context, 1, 64);
            bad_guard[0].with_mut(|words| words[32 + index] ^= 1);
            assert!(Scratch::new(&context, &bad_guard, 1, 64, 64, 1, SCRATCH_BUDGET).is_err());
        }
    });
}

#[cfg_attr(test, test)]
#[cfg_attr(
    test,
    ignore = "explicit GPU: exact budget and legacy-only output capacity preserve admission boundaries"
)]
fn head_batch_budget_and_legacy_only_capacity_are_explicit() {
    autoreleasepool(|_| {
        let context = context();
        let rows = 8193;
        let columns = 64;
        let weights = exact_tiles(&context, rows, columns);
        let initial = Scratch::new(
            &context,
            &weights,
            rows,
            columns,
            columns,
            rows,
            SCRATCH_BUDGET,
        )
        .unwrap();
        let required = initial.owned_bytes();
        assert!(required > 0 && required <= SCRATCH_BUDGET);
        drop(initial);
        assert!(matches!(
            Scratch::new(
                &context,
                &weights,
                rows,
                columns,
                columns,
                rows,
                required - 1
            ),
            Err(Error::Budget { .. })
        ));
        assert_eq!(
            Scratch::new(&context, &weights, rows, columns, columns, rows, required)
                .unwrap()
                .owned_bytes(),
            required
        );
        for (input_capacity, output_capacity) in [(63, rows), (64, 0), (64, 8191)] {
            assert!(
                Scratch::new(
                    &context,
                    &weights,
                    rows,
                    columns,
                    input_capacity,
                    output_capacity,
                    SCRATCH_BUDGET
                )
                .is_err()
            );
        }
        let mut legacy_only = Scratch::new(
            &context,
            &weights,
            rows,
            columns,
            columns,
            TILE_ROWS,
            SCRATCH_BUDGET,
        )
        .unwrap();
        assert!(!legacy_only.can_batch());
        let input = exact_input(columns);
        let mut output = vec![CALLER_SENTINEL; rows];
        assert!(legacy_only.batched(&input, &mut output).is_err());
        assert!(
            output
                .iter()
                .all(|value| value.to_bits() == CALLER_SENTINEL.to_bits())
        );
        assert!(!legacy_only.is_poisoned());
        assert!(!legacy_only.has_pending());
        let counts = legacy_only.legacy(&input, &mut output).unwrap();
        assert_eq!(bits(&output), bits(&exact_output(rows, columns)));
        assert_counts(counts, 2, false, rows, columns);
    });
}

#[cfg_attr(test, test)]
#[cfg_attr(
    test,
    ignore = "explicit GPU: request bounds and numerical admission fail before touching output or poison"
)]
fn head_batch_request_rejections_leave_caller_output_and_reusability_unchanged() {
    autoreleasepool(|_| {
        let context = context();
        let rows = 9;
        let columns = 64;
        let weights = exact_tiles(&context, rows, columns);
        let input = exact_input(columns);
        let mut scratch = Scratch::new(
            &context,
            &weights,
            rows,
            columns,
            columns + 5,
            rows + 7,
            SCRATCH_BUDGET,
        )
        .unwrap();
        let mut cases = vec![
            (input[..columns - 1].to_vec(), rows),
            ([input.clone(), vec![0.0]].concat(), rows),
            (input.clone(), rows - 1),
            (input.clone(), rows + 1),
        ];
        for invalid in [f32::NAN, f32::INFINITY, f32::MAX] {
            let mut changed = input.clone();
            changed[0] = invalid;
            cases.push((changed, rows));
        }
        for (input, length) in cases {
            for batched in [false, true] {
                let mut output = vec![CALLER_SENTINEL; length];
                let result = if batched {
                    scratch.batched(&input, &mut output)
                } else {
                    scratch.legacy(&input, &mut output)
                };
                assert!(result.is_err());
                assert!(
                    output
                        .iter()
                        .all(|value| value.to_bits() == CALLER_SENTINEL.to_bits())
                );
                assert!(!scratch.is_poisoned());
                assert!(!scratch.has_pending());
            }
        }
        let mut output = vec![CALLER_SENTINEL; rows];
        scratch.batched(&input, &mut output).unwrap();
        assert_eq!(bits(&output), bits(&exact_output(rows, columns)));
    });
}

#[cfg_attr(test, test)]
#[cfg_attr(
    test,
    ignore = "explicit GPU: completed validation covers every inactive and guard word plus status and active finiteness"
)]
fn head_batch_completed_validation_checks_full_scratch_capacity_and_status() {
    autoreleasepool(|_| {
        let context = context();
        let rows = 9;
        let columns = 64;
        let weights = exact_tiles(&context, rows, columns);
        let mut scratch = Scratch::new(
            &context,
            &weights,
            rows,
            columns,
            columns + 9,
            rows + 11,
            SCRATCH_BUDGET,
        )
        .unwrap();
        let mut output = vec![CALLER_SENTINEL; rows];
        scratch.batched(&exact_input(columns), &mut output).unwrap();
        assert!(!scratch.has_pending());
        scratch.validate_completed().unwrap();
        for index in columns..scratch.input.len {
            scratch.input.with_mut(|words| words[index] ^= 1);
            assert!(
                scratch.validate_completed().is_err(),
                "input inactive/guard {index}"
            );
            scratch.input.with_mut(|words| words[index] = SENTINEL);
        }
        for index in rows..scratch.output.len {
            scratch.output.with_mut(|words| words[index] ^= 1);
            assert!(
                scratch.validate_completed().is_err(),
                "output inactive/guard {index}"
            );
            scratch.output.with_mut(|words| words[index] = SENTINEL);
        }
        for index in 1..scratch.status.len {
            scratch.status.with_mut(|words| words[index] ^= 1);
            assert!(
                scratch.validate_completed().is_err(),
                "status guard {index}"
            );
            scratch.status.with_mut(|words| words[index] = SENTINEL);
        }
        scratch.status.with_mut(|words| words[0] = 1);
        assert!(scratch.validate_completed().is_err());
        scratch.status.with_mut(|words| words[0] = 0);
        let original = scratch.output.with_slice(|words| words[0]);
        scratch
            .output
            .with_mut(|words| words[0] = f32::NAN.to_bits());
        assert!(scratch.validate_completed().is_err());
        scratch.output.with_mut(|words| words[0] = original);
        scratch.validate_completed().unwrap();
        assert_eq!(bits(&output), bits(&exact_output(rows, columns)));
        // The read-only validator reports corruption; transaction methods own
        // poison state, covered independently by the injected-failure controls.
        assert!(!scratch.is_poisoned());
    });
}

#[cfg_attr(test, test)]
#[cfg_attr(
    test,
    ignore = "explicit GPU: all post-touch fault stages retain transactional caller output and sticky poison"
)]
fn head_batch_faults_do_not_publish_partial_output_or_allow_reuse() {
    autoreleasepool(|_| {
        let context = context();
        let rows = 8193;
        let columns = 64;
        let weights = exact_tiles(&context, rows, columns);
        let input = exact_input(columns);
        for (name, fault) in [
            ("encode", Fault::AfterFirstEncode),
            ("submit", Fault::AfterSubmit),
            ("wait", Fault::AfterWait),
            ("status", Fault::CorruptStatus),
            ("guard", Fault::CorruptGuard),
            ("copy", Fault::AfterFirstCopy),
        ] {
            let mut scratch = Scratch::new(
                &context,
                &weights,
                rows,
                columns,
                columns + 1,
                rows + 1,
                SCRATCH_BUDGET,
            )
            .unwrap();
            let mut output = vec![CALLER_SENTINEL; rows];
            scratch.set_fault(fault);
            assert!(scratch.batched(&input, &mut output).is_err(), "{name}");
            assert!(
                output
                    .iter()
                    .all(|value| value.to_bits() == CALLER_SENTINEL.to_bits()),
                "{name}"
            );
            assert!(scratch.is_poisoned(), "{name}");
            assert_eq!(scratch.has_pending(), name == "submit", "{name}");
            assert!(scratch.batched(&input, &mut output).is_err(), "{name}");
            assert!(scratch.legacy(&input, &mut output).is_err(), "{name}");
            assert!(
                output
                    .iter()
                    .all(|value| value.to_bits() == CALLER_SENTINEL.to_bits()),
                "reuse {name}"
            );
        }
    });
}

#[cfg_attr(test, test)]
#[cfg_attr(
    test,
    ignore = "explicit GPU: dropping a submitted pending batch waits for its retained command"
)]
fn head_batch_drop_completes_a_retained_pending_command_before_resource_release() {
    autoreleasepool(|_| {
        let context = context();
        let rows = 8193;
        let columns = 64;
        let weights = exact_tiles(&context, rows, columns);
        let input = exact_input(columns);
        let mut output = vec![CALLER_SENTINEL; rows];
        let mut scratch = Scratch::new(
            &context,
            &weights,
            rows,
            columns,
            columns,
            rows,
            SCRATCH_BUDGET,
        )
        .unwrap();
        scratch.set_fault(Fault::AfterSubmit);
        assert!(scratch.batched(&input, &mut output).is_err());
        assert!(scratch.has_pending());
        assert!(scratch.validate_completed().is_err());
        let command = scratch.pending.as_ref().unwrap().clone();
        drop(scratch);
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(
            output
                .iter()
                .all(|value| value.to_bits() == CALLER_SENTINEL.to_bits())
        );
        let mut replacement = Scratch::new(
            &context,
            &weights,
            rows,
            columns,
            columns,
            rows,
            SCRATCH_BUDGET,
        )
        .unwrap();
        replacement.batched(&input, &mut output).unwrap();
        assert_eq!(bits(&output), bits(&exact_output(rows, columns)));
    });
}

#[cfg_attr(test, test)]
#[cfg_attr(
    test,
    ignore = "explicit GPU: warmed legacy and batched calls allocate no Rust heap storage on this thread"
)]
fn head_batch_warmed_submissions_use_existing_scratch_without_rust_allocations() {
    autoreleasepool(|_| {
        let context = context();
        let rows = 8193;
        let columns = 64;
        let weights = exact_tiles(&context, rows, columns);
        let input = exact_input(columns);
        let mut old = vec![CALLER_SENTINEL; rows];
        let mut new = old.clone();
        let mut scratch = Scratch::new(
            &context,
            &weights,
            rows,
            columns,
            columns,
            rows,
            SCRATCH_BUDGET,
        )
        .unwrap();
        scratch.legacy(&input, &mut old).unwrap();
        scratch.batched(&input, &mut new).unwrap();
        let measurement = Measurement::start();
        let legacy = scratch.legacy(&input, &mut old);
        let batched = scratch.batched(&input, &mut new);
        let allocations = measurement.finish();
        assert_eq!(allocations, 0);
        assert_counts(legacy.unwrap(), 2, false, rows, columns);
        assert_counts(batched.unwrap(), 2, true, rows, columns);
        assert_eq!(bits(&new), bits(&old));
    });
}

#[cfg(all(feature = "gemma-head-batch-mac-probe", not(test)))]
pub(super) fn run_all(start: fn(), finish: fn() -> usize) {
    assert!(
        MEASUREMENT_HOOKS.set((start, finish)).is_ok(),
        "measurement hooks already installed"
    );
    head_batch_tail_shapes_match_independent_dyadic_oracle_and_actual_counts();
    head_batch_preserves_legacy_bits_for_cancellation_zero_and_subnormal_inputs();
    head_batch_constructor_rejects_incompatible_shapes_pipeline_tiles_and_weights();
    head_batch_budget_and_legacy_only_capacity_are_explicit();
    head_batch_request_rejections_leave_caller_output_and_reusability_unchanged();
    head_batch_completed_validation_checks_full_scratch_capacity_and_status();
    head_batch_faults_do_not_publish_partial_output_or_allow_reuse();
    head_batch_drop_completes_a_retained_pending_command_before_resource_release();
    head_batch_warmed_submissions_use_existing_scratch_without_rust_allocations();
}
