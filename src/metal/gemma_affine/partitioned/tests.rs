use super::*;
use crate as engine;
use sha2::Digest;
#[path = "../../../../tests/common/gemma_table_fixture.rs"]
pub(super) mod fixture;
mod head;

#[test]
fn aggregate_plan_counts_padding_scratch_and_owner_slots_without_gpu() {
    let layout = Layout::new(3, 64, 4, 64, MetadataDtype::BF16).unwrap();
    // Independent bytes: GPU buffers1036+992, CPU full/chunk scratch36+24.
    // Live structs are already part of children_owned and counted once.
    let expected = 2088 + 2 * size_of::<Affine<'_>>() + size_of::<PartitionedAffine<'_>>();
    let plan = Plan::new(layout, 3, 2, expected).unwrap();
    assert_eq!(plan.minimum_owned, expected);
    assert_eq!(plan.full_values, 9);
    assert_eq!(plan.chunk_values, 6);
    assert_eq!(plan.chunks, 2);
    assert!(
        matches!(Plan::new(layout, 3, 2, expected - 1), Err(Error::Budget { required, limit }) if required == expected && limit == expected - 1)
    );
    let extra = plan.owned(3, 10, 7).unwrap();
    assert_eq!(extra, expected + size_of::<Affine<'_>>() + 8);
    for (capacity, maximum) in [(0, 2), (33, 2), (usize::MAX, 2), (3, 0), (3, 16385)] {
        assert!(Plan::new(layout, capacity, maximum, usize::MAX).is_err());
    }
    let large = Layout::new(262144, 1536, 4, 64, MetadataDtype::BF16).unwrap();
    assert!(Plan::new(large, 4, 1, usize::MAX).is_err());
    assert_eq!(Plan::new(large, 4, 16384, usize::MAX).unwrap().chunks, 16);
}

#[test]
fn frozen_plan_preserves_all_source_rows_and_owner_only_budget_without_gpu() {
    let data = fixture::fixture();
    for case in data["cases"].as_array().unwrap() {
        let matrix = fixture::matrix(case);
        for partition in case["partitions"].as_array().unwrap() {
            let plan = Plan::new(
                matrix.layout(),
                3,
                fixture::number(&partition["maximum_rows"]),
                usize::MAX,
            )
            .unwrap();
            assert_eq!(plan.chunks, partition["chunks"].as_array().unwrap().len());
            for (index, expected) in partition["chunks"].as_array().unwrap().iter().enumerate() {
                let (first, layout) = plan.child(index).unwrap();
                assert_eq!(first, fixture::number(&expected["first_row"]));
                assert_eq!(layout.rows(), fixture::number(&expected["rows"]));
                let view = matrix.row_range(first..first + layout.rows()).unwrap();
                let (codes, scales, biases) = view.original_tensors();
                for (role, bytes) in [("codes", codes), ("scales", scales), ("biases", biases)] {
                    let spec = &expected["views"][role];
                    assert_eq!(bytes.len(), fixture::number(&spec["bytes"]));
                    let offset = fixture::number(&spec["source_offset"]);
                    let source = fixture::array(&case["arrays"][role]);
                    assert_eq!(bytes.as_ptr(), source[offset..].as_ptr());
                    assert_eq!(
                        format!("{:x}", sha2::Sha256::digest(bytes)),
                        spec["sha256"].as_str().unwrap()
                    );
                }
            }
            assert!(plan.child(plan.chunks).is_err());
        }
    }
}

fn report(label: &str, value: serde_json::Value) {
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    writeln!(out, "{label} {value}").unwrap();
    out.flush().unwrap();
}
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn observe(owner: &PartitionedAffine<'_>, caller: &[f32]) -> serde_json::Value {
    use serde_json::json;
    let buffer = |buffer: &Buffer<u32>| json!({"allocation":format!("{:p}",&*buffer.raw),"words":buffer.with_slice(<[u32]>::to_vec)});
    let children: Vec<_> = owner.chunks.iter().map(|child| {
        let safe = child.pending.as_ref().is_none_or(|command| command.status() == MTLCommandBufferStatus::Completed);
        let device = safe.then(|| json!({"codes":buffer(&child.codes),"scales":buffer(&child.scales),"biases":buffer(&child.biases),"input":buffer(&child.input),"output":buffer(&child.output),"status":buffer(&child.status)}));
        json!({"rows":child.layout.rows(),"capacity":child.capacity,"owned_bytes":child.owned_bytes(),"poisoned":child.is_poisoned(),"pending":child.pending.is_some(),"shared_read_safe":safe,"device":device})
    }).collect();
    json!({"owned_bytes":owner.owned_bytes(),"poisoned":owner.is_poisoned(),"rows":owner.rows,"columns":owner.columns,"capacity":owner.capacity,
        "owner_slots":owner.chunks.capacity(),"full_scratch_capacity":owner.staged.capacity(),"chunk_scratch_capacity":owner.chunk_output.capacity(),
        "caller_bits":bits(caller),"staged_bits":bits(&owner.staged),"chunk_scratch_bits":bits(&owner.chunk_output),"children":children})
}
fn step(
    owner: &mut PartitionedAffine<'_>,
    key: &str,
    phase: &str,
    index: usize,
    input: &[f32],
    tokens: usize,
    caller_len: usize,
) -> (Vec<f32>, Result<()>, serde_json::Value) {
    use serde_json::json;
    report(
        "GEMMA_PARTITION_STEP_ATTEMPT",
        json!({"key":key,"phase":phase,"index":index,"tokens":tokens,"input_bits":bits(input),"caller_len":caller_len}),
    );
    let mut caller = vec![42.; caller_len];
    let result = owner.apply(input, &mut caller, tokens);
    // Snapshot every safe actual device allocation and complete CPU views
    // before any assertion or fallible publication. Publish the original cause
    // first, before a larger diagnostic write could fail.
    let observation = observe(owner, &caller);
    report(
        "GEMMA_PARTITION_STEP_OUTCOME",
        json!({"key":key,"phase":phase,"index":index,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"poisoned":owner.is_poisoned()}),
    );
    report(
        "GEMMA_PARTITION_STEP",
        json!({"key":key,"phase":phase,"index":index,"tokens":tokens,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"observation":observation}),
    );
    (caller, result, observation)
}

#[test]
#[ignore = "explicit small resident partitioned affine GPU/transaction qualification"]
fn frozen_partitioned_projection_reuse_transactions() {
    use serde_json::json;
    autoreleasepool(|_| {
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        report(
            "GEMMA_PARTITION_META",
            json!({"device":context.device_name(),"simd_width":context.width,"context_allowance_bytes":CONTEXT_ALLOWANCE_BYTES,"owner_struct_bytes":size_of::<PartitionedAffine<'_>>(),"child_struct_bytes":size_of::<Affine<'_>>(),"real_head_qualified":false,"whole_graph_qualified":false}),
        );
        let data = fixture::fixture();
        let mut cases = 0;
        let mut composition_failures = 0;
        for case in data["cases"].as_array().unwrap() {
            let matrix = fixture::matrix(case);
            let input = fixture::floats(&case["arrays"]["input"]);
            let expected = fixture::floats(&case["arrays"]["output_f32"]);
            let ordered: Vec<_> = fixture::array(&case["arrays"]["ordered_fp64"])
                .chunks_exact(8)
                .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
                .collect();
            for partition in case["partitions"].as_array().unwrap() {
                let maximum = fixture::number(&partition["maximum_rows"]);
                let key = format!("{}/max{maximum}", case["name"].as_str().unwrap());
                let plan = Plan::new(matrix.layout(), 3, maximum, usize::MAX).unwrap();
                let rejected =
                    context.partitioned_affine(&matrix, maximum, 3, plan.minimum_owned - 1);
                report(
                    "GEMMA_PARTITION_BUDGET",
                    json!({"key":key,"required":plan.minimum_owned,"limit":plan.minimum_owned-1,"rejected":rejected.is_err(),"error":rejected.as_ref().err().map(ToString::to_string)}),
                );
                assert!(
                    matches!(rejected, Err(Error::Budget { required, limit }) if required == plan.minimum_owned && limit == plan.minimum_owned-1)
                );
                report(
                    "GEMMA_PARTITION_CONSTRUCTOR_ATTEMPT",
                    json!({"key":key,"rows":matrix.layout().rows(),"columns":matrix.layout().columns(),"maximum_rows":maximum,"capacity":3,"required":plan.minimum_owned}),
                );
                let constructed =
                    context.partitioned_affine(&matrix, maximum, 3, plan.minimum_owned);
                report(
                    "GEMMA_PARTITION_CONSTRUCTOR_OUTCOME",
                    json!({"key":key,"completed":constructed.is_ok(),"error":constructed.as_ref().err().map(ToString::to_string)}),
                );
                let mut owner = constructed.unwrap();
                let cold = observe(&owner, &[]);
                report(
                    "GEMMA_PARTITION_CONSTRUCTOR",
                    json!({"key":key,"observation":cold}),
                );
                let n = matrix.layout().rows();
                let k = matrix.layout().columns();
                let mut baseline = Vec::new();
                for repeat in 0..2 {
                    let (actual, result, _) =
                        step(&mut owner, &key, "repeat", repeat, &input, 3, 3 * n);
                    result.unwrap();
                    for ((&actual, &rounded), &ordered) in
                        actual.iter().zip(&expected).zip(&ordered)
                    {
                        assert_eq!(actual.to_bits(), rounded.to_bits());
                        assert!((f64::from(actual) - ordered).abs() <= 1e-5 + 1e-4 * ordered.abs());
                    }
                    if repeat == 0 {
                        baseline = bits(&actual);
                    } else {
                        assert_eq!(bits(&actual), baseline);
                    }
                }
                for token in 0..3 {
                    let (actual, result, _) = step(
                        &mut owner,
                        &key,
                        "part",
                        token,
                        &input[token * k..(token + 1) * k],
                        1,
                        n,
                    );
                    result.unwrap();
                    assert_eq!(bits(&actual), baseline[token * n..(token + 1) * n]);
                }
                let (actual, result, _) = step(&mut owner, &key, "resume", 0, &input, 3, 3 * n);
                result.unwrap();
                assert_eq!(bits(&actual), baseline);
                if cases == 0 {
                    let before = observe(&owner, &[]);
                    let changed: Vec<_> = input.iter().map(|x| x + 1.).collect();
                    FAILURE_AFTER.with(|v| v.set(Some(2)));
                    let (actual, result, failed) = step(
                        &mut owner,
                        &key,
                        "composition-failure",
                        0,
                        &changed,
                        3,
                        3 * n,
                    );
                    FAILURE_AFTER.with(|v| v.set(None));
                    assert!(matches!(
                        result,
                        Err(Error::Options(
                            "Gemma partitioned affine injected composition failure"
                        ))
                    ));
                    assert_eq!(bits(&actual), vec![42_f32.to_bits(); 3 * n]);
                    assert!(owner.is_poisoned());
                    assert_ne!(failed["staged_bits"], before["staged_bits"]);
                    let (actual, result, blocked) =
                        step(&mut owner, &key, "poison-retry", 0, &input, 3, 3 * n);
                    assert!(matches!(
                        result,
                        Err(Error::Options(
                            "Gemma partitioned affine owner poisoned; recreate"
                        ))
                    ));
                    assert_eq!(bits(&actual), vec![42_f32.to_bits(); 3 * n]);
                    assert_eq!(failed, blocked);
                    // The failed owner's completed children stay retained until
                    // explicit teardown. Recreate the exact same geometry.
                    drop(owner);
                    let replacement_key = format!("{key}/recreated");
                    report(
                        "GEMMA_PARTITION_CONSTRUCTOR_ATTEMPT",
                        json!({"key":replacement_key,"rows":matrix.layout().rows(),"columns":matrix.layout().columns(),"maximum_rows":maximum,"capacity":3,"required":plan.minimum_owned}),
                    );
                    let replacement =
                        context.partitioned_affine(&matrix, maximum, 3, plan.minimum_owned);
                    report(
                        "GEMMA_PARTITION_CONSTRUCTOR_OUTCOME",
                        json!({"key":replacement_key,"completed":replacement.is_ok(),"error":replacement.as_ref().err().map(ToString::to_string)}),
                    );
                    let mut replacement = replacement.unwrap();
                    let cold = observe(&replacement, &[]);
                    report(
                        "GEMMA_PARTITION_CONSTRUCTOR",
                        json!({"key":replacement_key,"observation":cold}),
                    );
                    let (actual, result, _) = step(
                        &mut replacement,
                        &replacement_key,
                        "recreate",
                        0,
                        &input,
                        3,
                        3 * n,
                    );
                    result.unwrap();
                    assert_eq!(bits(&actual), baseline);
                    composition_failures += 1;
                }
                cases += 1;
            }
        }
        // First chunk has zero weights; the later unit-weight chunk rejects
        // the historical ordered-zero/exact-one input before either stages.
        let layout = Layout::new(2, 64, 4, 64, MetadataDtype::BF16).unwrap();
        let codes = [0; 64];
        let scales = [0; 4];
        let biases = [0, 0, 0x80, 0x3f];
        let matrix = Matrix::from_le_bytes(layout, &codes, &scales, &biases).unwrap();
        let plan = Plan::new(layout, 1, 1, usize::MAX).unwrap();
        let key = "later-admission";
        report(
            "GEMMA_PARTITION_CONSTRUCTOR_ATTEMPT",
            json!({"key":key,"rows":2,"columns":64,"maximum_rows":1,"capacity":1,"required":plan.minimum_owned}),
        );
        let constructed = context.partitioned_affine(&matrix, 1, 1, plan.minimum_owned);
        report(
            "GEMMA_PARTITION_CONSTRUCTOR_OUTCOME",
            json!({"key":key,"completed":constructed.is_ok(),"error":constructed.as_ref().err().map(ToString::to_string)}),
        );
        let mut owner = constructed.unwrap();
        let cold = observe(&owner, &[]);
        report(
            "GEMMA_PARTITION_CONSTRUCTOR",
            json!({"key":key,"observation":cold}),
        );
        let (_, result, _) = step(&mut owner, key, "warm", 0, &[0.; 64], 1, 2);
        result.unwrap();
        let before = observe(&owner, &[42.; 2]);
        let mut conflict = [0.; 64];
        conflict[0] = 2_f32.powi(100);
        conflict[1] = 1.;
        conflict[2] = -2_f32.powi(100);
        let (actual, result, rejected) = step(&mut owner, key, "reject", 0, &conflict, 1, 2);
        assert!(matches!(
            result,
            Err(Error::Options(
                "Gemma affine input ordered-sum accuracy bound"
            ))
        ));
        assert_eq!(bits(&actual), vec![42_f32.to_bits(); 2]);
        assert_eq!(before, rejected);
        assert!(!owner.is_poisoned());
        let mut nan = [0.; 64];
        nan[0] = f32::NAN;
        let invalids = [
            (nan.to_vec(), 1, 2),
            (vec![0.; 63], 1, 2),
            (vec![0.; 64], 1, 3),
            (vec![], 0, 0),
            (vec![0.; 128], 2, 4),
            (vec![], usize::MAX, 0),
        ];
        for (index, (input, tokens, caller_len)) in invalids.iter().enumerate() {
            let before = observe(&owner, &vec![42.; *caller_len]);
            let (actual, result, after) = step(
                &mut owner,
                key,
                "preflight",
                index,
                input,
                *tokens,
                *caller_len,
            );
            assert!(matches!(
                result,
                Err(Error::Options(
                    "Gemma partitioned affine input shape/nonfinite value"
                ))
            ));
            assert_eq!(bits(&actual), vec![42_f32.to_bits(); *caller_len]);
            assert_eq!(before, after);
        }
        let (actual, result, _) = step(&mut owner, key, "recover", 0, &[0.; 64], 1, 2);
        result.unwrap();
        assert_eq!(bits(&actual), vec![0; 2]);
        let invalid = fixture::matrix(&data["overflow"]);
        let rejected = context.partitioned_affine(&invalid, 1, 1, 1024 * 1024);
        report(
            "GEMMA_PARTITION_DOMAIN",
            json!({"rejected":rejected.is_err(),"error":rejected.as_ref().err().map(ToString::to_string)}),
        );
        assert!(matches!(
            rejected,
            Err(Error::Unsupported("Gemma affine weight domain"))
        ));
        let head = Layout::new(262144, 1536, 4, 64, MetadataDtype::BF16).unwrap();
        let rejected = context.affine(head, &[], &[], &[], 1, 1024 * 1024 * 1024);
        report(
            "GEMMA_PARTITION_OLD_HEAD",
            json!({"rejected":rejected.is_err(),"error":rejected.as_ref().err().map(ToString::to_string),"large_spans_read":false}),
        );
        assert!(matches!(
            rejected,
            Err(Error::Options(
                "Gemma affine fixed geometry/format outside bounds"
            ))
        ));
        report(
            "GEMMA_PARTITION_FINAL",
            json!({"cases":cases,"repeats":2,"partitions":24,"resumes":8,"budget_rejections":8,"composition_failures":composition_failures,"same_geometry_recreations":1,"late_preflight_rejections":1,"recoveries":1,"domain_rejections":1,"preflight_rejections":6,"old_head_geometry_rejected":true,"driver_or_inflight_failure_injected":false,"real_head_qualified":false,"whole_graph_qualified":false}),
        );
        assert_eq!((cases, composition_failures), (8, 1));
    });
}
