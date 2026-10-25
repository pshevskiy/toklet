use super::super::tests::fixture;
use super::*;
use serde_json::{Value, json};

#[test]
fn workspace_plan_counts_guards_cpu_capacity_owner_and_pipeline_policy_without_gpu() {
    // GPU9+135+8 words, two128-value CPU arrays, actual owner once.
    let expected = 1632 + size_of::<Gather<'_>>() + PIPELINE_ALLOWANCE;
    let plan = Sizes::new(64, 2, expected).unwrap();
    assert_eq!(plan.minimum_admitted, expected);
    assert_eq!(plan.admitted(129, 130).unwrap(), expected + 12);
    assert!(
        matches!(Sizes::new(64,2,expected-1),Err(Error::Budget{required,limit}) if required==expected && limit==expected-1)
    );
    for (columns, capacity) in [
        (0, 2),
        (63, 2),
        (65, 2),
        (12289, 2),
        (usize::MAX, 2),
        (64, 0),
        (64, 33),
        (64, usize::MAX),
    ] {
        assert!(Sizes::new(columns, capacity, usize::MAX).is_err());
    }
}
fn report(label: &str, value: Value) {
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    writeln!(out, "{label} {value}").unwrap();
    out.flush().unwrap();
}
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn buffer(buffer: &Buffer<u32>) -> Value {
    buffer.with_slice(|v| json!({"allocation":format!("{:p}",&*buffer.raw),"words":v}))
}
fn matrix_snapshot(matrix: &PartitionedAffine<'_>, safe: bool) -> Value {
    let children:Vec<_>=matrix.chunks.iter().map(|child|{
        let child_safe=safe && child.pending.as_ref().is_none_or(|c|c.status()==MTLCommandBufferStatus::Completed);
        let device=child_safe.then(||json!({"codes":buffer(&child.codes),"scales":buffer(&child.scales),"biases":buffer(&child.biases),"input":buffer(&child.input),"output":buffer(&child.output),"status":buffer(&child.status)}));
        json!({"rows":child.layout.rows(),"capacity":child.capacity,"poisoned":child.poisoned,"pending":child.pending.is_some(),"shared_read_safe":child_safe,"device":device})
    }).collect();
    json!({"rows":matrix.rows,"columns":matrix.columns,"capacity":matrix.capacity,"poisoned":matrix.poisoned,"admitted_bytes":matrix.owned_bytes(),"slots":matrix.chunks.capacity(),
        "full_scratch_capacity":matrix.staged.capacity(),"chunk_scratch_capacity":matrix.chunk_output.capacity(),"staged_bits":bits(&matrix.staged),"chunk_bits":bits(&matrix.chunk_output),"children":children})
}
fn snapshot(session: &Gather<'_>, matrix: &PartitionedAffine<'_>, caller: &[f32]) -> Value {
    let safe = session
        .pending
        .as_ref()
        .is_none_or(|c| c.status() == MTLCommandBufferStatus::Completed);
    json!({"admitted_bytes":session.admitted_bytes(),"columns":session.columns,"capacity":session.capacity,"poisoned":session.is_poisoned(),"pending":session.pending.is_some(),"shared_read_safe":safe,
        "full_scratch_capacity":session.staged.capacity(),"chunk_scratch_capacity":session.chunk.capacity(),"staged_bits":bits(&session.staged),"chunk_bits":bits(&session.chunk),"caller_bits":bits(caller),
        "workspace":safe.then(||json!({"ids":buffer(&session.ids),"output":buffer(&session.output),"status":buffer(&session.status)})),
        "matrix":matrix_snapshot(matrix,safe)})
}
fn step(
    session: &mut Gather<'_>,
    matrix: &PartitionedAffine<'_>,
    key: &str,
    phase: &str,
    index: usize,
    ids: &[usize],
    length: usize,
) -> (Result<()>, Value, Vec<f32>) {
    report(
        "GEMMA_GATHER_STEP_ATTEMPT",
        json!({"key":key,"phase":phase,"index":index,"ids":ids,"caller_len":length}),
    );
    let mut caller = vec![42.; length];
    let result = session.apply(matrix, ids, &mut caller);
    let observation = snapshot(session, matrix, &caller);
    // Complete safe output views precede numerical assertions/fallible later
    // publication; retain original cause before the larger observation write.
    report(
        "GEMMA_GATHER_STEP_OUTCOME",
        json!({"key":key,"phase":phase,"index":index,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"poisoned":session.is_poisoned()}),
    );
    report(
        "GEMMA_GATHER_STEP",
        json!({"key":key,"phase":phase,"index":index,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"observation":observation}),
    );
    (result, observation, caller)
}
fn affine(matrix: &mut PartitionedAffine<'_>, case: &Value, key: &str, index: usize) {
    let n = matrix.rows;
    let k = matrix.columns;
    let input = fixture::floats(&case["arrays"]["input"]);
    let expected = fixture::floats(&case["arrays"]["output_f32"]);
    let mut output = vec![42.; n];
    report(
        "GEMMA_GATHER_AFFINE_ATTEMPT",
        json!({"key":key,"index":index,"input_bits":bits(&input[k..2*k]),"caller_len":n}),
    );
    let result = matrix.apply(&input[k..2 * k], &mut output, 1);
    report(
        "GEMMA_GATHER_AFFINE_OUTCOME",
        json!({"key":key,"index":index,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string)}),
    );
    report(
        "GEMMA_GATHER_AFFINE",
        json!({"key":key,"index":index,"input_bits":bits(&input[k..2*k]),"output_bits":bits(&output),"matrix":matrix_snapshot(matrix,true)}),
    );
    result.unwrap();
    assert_eq!(bits(&output), bits(&expected[n..2 * n]));
}
#[test]
#[ignore = "explicit resident lookup, original-byte sharing and transactional GPU qualification"]
fn frozen_resident_table_gathers_share_weights_and_preserve_transactions() {
    autoreleasepool(|_| {
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        report(
            "GEMMA_GATHER_META",
            json!({"device":context.device_name(),"context_allowance_bytes":CONTEXT_ALLOWANCE_BYTES,"pipeline_policy_bytes":PIPELINE_ALLOWANCE,"owner_struct_bytes":size_of::<Gather<'_>>(),"routing_stack_bytes":32*size_of::<u32>()+32*size_of::<usize>(),"whole_graph_qualified":false,"actual_large_table_qualified":false}),
        );
        let data = fixture::fixture();
        let mut cases = 0;
        for case in data["cases"].as_array().unwrap() {
            let source = fixture::matrix(case);
            let k = source.layout().columns();
            let canonical: Vec<_> = case["gather_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(fixture::number)
                .collect();
            let expected = fixture::floats(&case["arrays"]["gather"]);
            for maximum in [7, 8] {
                let key = format!("{}/max{maximum}", case["name"].as_str().unwrap());
                // Affine capacity1 is independent of gather request capacity32.
                report(
                    "GEMMA_GATHER_MATRIX_ATTEMPT",
                    json!({"key":key,"rows":source.layout().rows(),"columns":k,"maximum":maximum,"capacity":1,"budget":1024*1024}),
                );
                let constructed = context.partitioned_affine(&source, maximum, 1, 1024 * 1024);
                report(
                    "GEMMA_GATHER_MATRIX_OUTCOME",
                    json!({"key":key,"completed":constructed.is_ok(),"error":constructed.as_ref().err().map(ToString::to_string)}),
                );
                let mut matrix = constructed.unwrap();
                affine(&mut matrix, case, &key, 0);
                let plan = Sizes::new(k, 32, usize::MAX).unwrap();
                let rejected = context.gather_session(k, 32, plan.minimum_admitted - 1);
                report(
                    "GEMMA_GATHER_BUDGET",
                    json!({"key":key,"required":plan.minimum_admitted,"limit":plan.minimum_admitted-1,"rejected":rejected.is_err(),"error":rejected.as_ref().err().map(ToString::to_string)}),
                );
                assert!(
                    matches!(rejected,Err(Error::Budget{required,limit}) if required==plan.minimum_admitted && limit==plan.minimum_admitted-1)
                );
                report(
                    "GEMMA_GATHER_CONSTRUCTOR_ATTEMPT",
                    json!({"key":key,"columns":k,"capacity":32,"required":plan.minimum_admitted}),
                );
                let constructed = context.gather_session(k, 32, plan.minimum_admitted);
                report(
                    "GEMMA_GATHER_CONSTRUCTOR_OUTCOME",
                    json!({"key":key,"completed":constructed.is_ok(),"error":constructed.as_ref().err().map(ToString::to_string)}),
                );
                let mut session = constructed.unwrap();
                report(
                    "GEMMA_GATHER_CONSTRUCTOR",
                    json!({"key":key,"simd_width":session.width,"observation":snapshot(&session,&matrix,&[])}),
                );
                for repeat in 0..2 {
                    let (result, _, output) = step(
                        &mut session,
                        &matrix,
                        &key,
                        "canonical",
                        repeat,
                        &canonical,
                        expected.len(),
                    );
                    result.unwrap();
                    assert_eq!(bits(&output), bits(&expected));
                }
                for count in [1, 32] {
                    let choices: Vec<_> = (0..count).map(|i| (i * 3 + 4) % 5).collect();
                    let ids: Vec<_> = choices.iter().map(|&i| canonical[i]).collect();
                    let wanted: Vec<_> = choices
                        .iter()
                        .flat_map(|&i| expected[i * k..(i + 1) * k].iter().copied())
                        .collect();
                    let (result, _, output) = step(
                        &mut session,
                        &matrix,
                        &key,
                        "boundary",
                        count,
                        &ids,
                        count * k,
                    );
                    result.unwrap();
                    assert_eq!(bits(&output), bits(&wanted));
                }
                let before = snapshot(&session, &matrix, &[]);
                for (index, (ids, len)) in [
                    (vec![], 0),
                    (vec![0; 33], 33 * k),
                    (vec![0, source.layout().rows()], 2 * k),
                    (vec![0, usize::MAX], 2 * k),
                    (vec![0, 1], 2 * k - 1),
                    (vec![0, 1], 2 * k + 1),
                ]
                .iter()
                .enumerate()
                {
                    let (result, after, caller) = step(
                        &mut session,
                        &matrix,
                        &key,
                        "request-reject",
                        index,
                        ids,
                        *len,
                    );
                    assert!(matches!(
                        result,
                        Err(Error::Options(
                            "Gemma table gather request shape or indices"
                        ))
                    ));
                    let mut before = before.clone();
                    assert_eq!(bits(&caller), vec![42_f32.to_bits(); *len]);
                    before["caller_bits"] = json!(vec![42_f32.to_bits(); *len]);
                    assert_eq!(before, after);
                }
                if cases == 0 {
                    // All state rejections happen against warmed scratch.
                    let other = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
                    let foreign = other
                        .partitioned_affine(&source, maximum, 1, 1024 * 1024)
                        .unwrap();
                    let wrong_columns = fixture::matrix(&data["cases"][1]);
                    let wrong = context
                        .partitioned_affine(&wrong_columns, maximum, 1, 1024 * 1024)
                        .unwrap();
                    for (index, table) in [(0, &foreign), (1, &wrong)] {
                        let warmed = snapshot(&session, &matrix, &vec![42.; k]);
                        let before = snapshot(&session, table, &vec![42.; k]);
                        let (result, after, caller) =
                            step(&mut session, table, &key, "matrix-reject", index, &[0], k);
                        assert!(matches!(
                            result,
                            Err(Error::Options(
                                "Gemma table gather matrix Context/shape/state mismatch"
                            ))
                        ));
                        assert_eq!(bits(&caller), vec![42_f32.to_bits(); k]);
                        assert_eq!(before, after);
                        assert_eq!(snapshot(&session, &matrix, &vec![42.; k]), warmed);
                    }
                    matrix.poisoned = true;
                    let before = snapshot(&session, &matrix, &vec![42.; k]);
                    let (result, after, caller) =
                        step(&mut session, &matrix, &key, "matrix-reject", 2, &[0], k);
                    assert!(matches!(
                        result,
                        Err(Error::Options(
                            "Gemma table gather matrix Context/shape/state mismatch"
                        ))
                    ));
                    assert_eq!(bits(&caller), vec![42_f32.to_bits(); k]);
                    assert_eq!(before, after);
                    matrix.poisoned = false;
                    matrix.chunks[0].poisoned = true;
                    let before = snapshot(&session, &matrix, &vec![42.; k]);
                    let (result, after, caller) =
                        step(&mut session, &matrix, &key, "matrix-reject", 3, &[0], k);
                    assert!(matches!(
                        result,
                        Err(Error::Options(
                            "Gemma table gather matrix Context/shape/state mismatch"
                        ))
                    ));
                    assert_eq!(bits(&caller), vec![42_f32.to_bits(); k]);
                    assert_eq!(before, after);
                    matrix.chunks[0].poisoned = false;
                    let pending = context.gpu.command().unwrap();
                    matrix.chunks[0].pending = Some(pending);
                    let before = snapshot(&session, &matrix, &vec![42.; k]);
                    let (result, after, caller) =
                        step(&mut session, &matrix, &key, "matrix-reject", 4, &[0], k);
                    assert!(matches!(
                        result,
                        Err(Error::Options(
                            "Gemma table gather matrix Context/shape/state mismatch"
                        ))
                    ));
                    assert_eq!(bits(&caller), vec![42_f32.to_bits(); k]);
                    assert_eq!(before, after);
                    matrix.chunks[0].pending = None;
                    let before = snapshot(&session, &matrix, &vec![42.; canonical.len() * k]);
                    FAILURE_AFTER.with(|v| v.set(Some(2)));
                    let changed = [18, 7, 6, 0, 7];
                    let (result, failed, caller) = step(
                        &mut session,
                        &matrix,
                        &key,
                        "composition-failure",
                        0,
                        &changed,
                        changed.len() * k,
                    );
                    FAILURE_AFTER.with(|v| v.set(None));
                    assert!(matches!(
                        result,
                        Err(Error::Options(
                            "Gemma table gather injected composition failure"
                        ))
                    ));
                    assert!(session.is_poisoned());
                    assert_eq!(bits(&caller), vec![42_f32.to_bits(); caller.len()]);
                    assert_ne!(before["staged_bits"], failed["staged_bits"]);
                    let (result, blocked, _) = step(
                        &mut session,
                        &matrix,
                        &key,
                        "poison-retry",
                        0,
                        &canonical,
                        canonical.len() * k,
                    );
                    assert!(matches!(
                        result,
                        Err(Error::Options(
                            "Gemma table gather poisoned; recreate workspace"
                        ))
                    ));
                    assert_eq!(failed, blocked);
                    let replacement = context.gather_session(k, 32, plan.minimum_admitted);
                    report(
                        "GEMMA_GATHER_RECREATE_OUTCOME",
                        json!({"key":key,"completed":replacement.is_ok(),"error":replacement.as_ref().err().map(ToString::to_string)}),
                    );
                    session = replacement.unwrap();
                    report(
                        "GEMMA_GATHER_RECREATED",
                        json!({"key":key,"observation":snapshot(&session,&matrix,&[])}),
                    );
                    let (result, _, output) = step(
                        &mut session,
                        &matrix,
                        &key,
                        "recover",
                        0,
                        &canonical,
                        expected.len(),
                    );
                    result.unwrap();
                    assert_eq!(bits(&output), bits(&expected));
                }
                affine(&mut matrix, case, &key, 1);
                let (result, _, output) = step(
                    &mut session,
                    &matrix,
                    &key,
                    "resume",
                    0,
                    &canonical,
                    expected.len(),
                );
                result.unwrap();
                assert_eq!(bits(&output), bits(&expected));
                report(
                    "GEMMA_GATHER_CASE_END",
                    json!({"key":key,"observation":snapshot(&session,&matrix,&[])}),
                );
                cases += 1;
            }
        }
        report(
            "GEMMA_GATHER_FINAL",
            json!({"cases":cases,"composition_failures":1,"recreated_workspaces":1,"affine_gather_alternations":4,"matrix_weights_shared":true,"matrix_affine_capacity":1,"gather_capacity":32,"driver_or_inflight_failure_injected":false,"actual_large_table_qualified":false,"whole_graph_qualified":false,"performance_qualified":false}),
        );
        assert_eq!(cases, 4);
    });
}
