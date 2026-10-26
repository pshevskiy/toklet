//! Explicit tiny immutable-weight lifetime and existing-path qualification.
use super::*;

fn emit(label: &str, value: Value) {
    report(&format!("GEMMA_WEIGHTS_{label}"), value);
}
fn observe(weights: &Weights) -> Value {
    observe_if(weights, true)
}
fn observe_if(weights: &Weights, safe: bool) -> Value {
    let (slots, chunk_struct_bytes, device) = weights.diagnostic_storage();
    let children:Vec<_>=weights.views().map(|v|json!({"rows":v.layout.rows(),"shared_read_safe":safe,"allocations":[format!("{:p}",&*v.codes.raw),format!("{:p}",&*v.scales.raw),format!("{:p}",&*v.biases.raw)],"codes":safe.then(||buffer(v.codes)),"scales":safe.then(||buffer(v.scales)),"biases":safe.then(||buffer(v.biases))})).collect();
    json!({"rows":weights.layout().rows(),"columns":weights.layout().columns(),"owned_bytes":weights.owned_bytes(),"max_abs_weight":weights.max_abs_weight(),"slots":slots,"chunk_struct_bytes":chunk_struct_bytes,"device":device,"children":children})
}
fn workspace(session: &Gather<'_>, caller: &[f32]) -> Value {
    let safe = session
        .pending
        .as_ref()
        .is_none_or(|c| c.status() == MTLCommandBufferStatus::Completed);
    json!({"admitted_bytes":session.admitted_bytes(),"columns":session.columns,"capacity":session.capacity,"poisoned":session.poisoned,"pending":session.pending.is_some(),"shared_read_safe":safe,
        "staged_bits":bits(&session.staged),"chunk_bits":bits(&session.chunk),"caller_bits":bits(caller),"full_scratch_capacity":session.staged.capacity(),"chunk_scratch_capacity":session.chunk.capacity(),
        "device":safe.then(||json!({"ids":buffer(&session.ids),"output":buffer(&session.output),"status":buffer(&session.status)}))})
}
fn minimum_budget(layout: Layout, maximum: usize) -> usize {
    let (owner, chunk) = Weights::diagnostic_sizes();
    let chunks = layout.rows().div_ceil(maximum);
    let mut resident = owner + chunks * chunk;
    for first in (0..layout.rows()).step_by(maximum) {
        let rows = maximum.min(layout.rows() - first);
        resident += (rows * layout.columns() / 2 + GUARD * 4)
            + 2 * ((rows * layout.columns() / 32).div_ceil(4) * 4 + GUARD * 4);
    }
    resident
        + maximum.min(layout.rows()) * layout.columns() / 2
        + 2 * maximum.min(layout.rows()) * layout.columns() / 32
        + 3 * size_of::<Vec<u8>>()
}
fn fill(
    source: &Matrix<'_>,
    first: usize,
    child: Layout,
    codes: &mut [u8],
    scales: &mut [u8],
    biases: &mut [u8],
) -> Result<()> {
    let view = source
        .row_range(first..first + child.rows())
        .map_err(|e| Error::Options(e.0))?;
    let (c, s, b) = view.original_tensors();
    codes.copy_from_slice(c);
    scales.copy_from_slice(s);
    biases.copy_from_slice(b);
    Ok(())
}
fn construction_fault(
    context: &Context,
    weights: &Weights,
    source: &Matrix<'_>,
    maximum: usize,
    budget: usize,
    key: &str,
    kind: &str,
) {
    let mut callbacks = 0;
    let limit = if kind == "budget" { budget - 1 } else { budget };
    emit(
        "FAULT_ATTEMPT",
        json!({"key":key,"kind":kind,"budget":limit,"minimum":budget}),
    );
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        context.weights_from_fill(source.layout(),maximum,limit,|first,child,c,s,b|{
        callbacks+=1;
        emit("FAULT_CALLBACK_ATTEMPT",json!({"key":key,"kind":kind,"first_row":first,"rows":child.rows(),"codes_before":c,"scales_before":s,"biases_before":b}));
        if kind=="first-source"{return Err(Error::Options("immutable original first source"));}
        fill(source,first,child,c,s,b)?;
        if first>0 && kind=="domain"{s[..2].copy_from_slice(&0x7f7f_u16.to_le_bytes());b[..2].copy_from_slice(&0x7f7f_u16.to_le_bytes());}
        emit("FAULT_CALLBACK",json!({"key":key,"kind":kind,"first_row":first,"codes":c,"scales":s,"biases":b}));
        if first>0 && kind=="late-source"{return Err(Error::Native("immutable original late source".into()));}
        if first>0 && kind=="panic"{std::panic::panic_any("immutable original panic");}
        Ok(())
    })
    }));
    let original_error = match &caught {
        Ok(Err(e)) => Some(e.to_string()),
        _ => None,
    };
    let panic_payload = caught
        .as_ref()
        .err()
        .and_then(|e| e.downcast_ref::<&str>().copied());
    emit(
        "FAULT_OUTCOME",
        json!({"key":key,"kind":kind,"callbacks":callbacks,"unwound":caught.is_err(),"published":matches!(&caught,Ok(Ok(_))),"error":original_error,"panic_payload":panic_payload}),
    );
    if let Ok(Ok(unexpected)) = &caught {
        emit(
            "FAULT_UNEXPECTED_OWNER",
            json!({"key":key,"kind":kind,"weights":observe(unexpected)}),
        );
    }
    emit(
        "FAULT_BASELINE",
        json!({"key":key,"kind":kind,"weights":observe(weights)}),
    );
    match kind {
        "budget" => {
            assert!(
                matches!(caught,Ok(Err(Error::Budget{required,limit:l})) if required==budget && l==limit)
            );
            assert_eq!(callbacks, 0);
        }
        "first-source" => {
            assert!(matches!(
                caught,
                Ok(Err(Error::Options("immutable original first source")))
            ));
            assert_eq!(callbacks, 1);
        }
        "late-source" => {
            assert!(
                matches!(caught,Ok(Err(Error::Native(ref e))) if e=="immutable original late source")
            );
            assert_eq!(callbacks, 2);
        }
        "domain" => {
            assert!(matches!(
                caught,
                Ok(Err(Error::Unsupported("Gemma affine weight domain")))
            ));
            assert_eq!(callbacks, 2);
        }
        "panic" => {
            assert_eq!(panic_payload, Some("immutable original panic"));
            assert_eq!(callbacks, 2);
        }
        _ => panic!("unknown construction fault"),
    }
}
fn step(
    session: &mut Gather<'_>,
    weights: &Weights,
    key: &str,
    index: usize,
    ids: &[usize],
    len: usize,
) -> (Result<()>, Vec<f32>) {
    let mut caller = vec![42.; len];
    emit(
        "STEP_ATTEMPT",
        json!({"key":key,"index":index,"ids":ids,"caller_len":len}),
    );
    let result = session.apply_weights(weights, ids, &mut caller);
    emit(
        "STEP_OUTCOME",
        json!({"key":key,"index":index,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string)}),
    );
    emit(
        "STEP",
        json!({"key":key,"index":index,"workspace":workspace(session,&caller),"weights":observe_if(weights,session.pending.as_ref().is_none_or(|c|c.status()==MTLCommandBufferStatus::Completed))}),
    );
    (result, caller)
}
#[test]
#[ignore = "explicit immutable original-weight lifetime, workspace recovery and legacy-path GPU qualification"]
fn immutable_weights_outlive_context_and_reuse_fresh_workspaces() {
    autoreleasepool(|_| {
        let categories = json!({"two_context_policies":2*CONTEXT_ALLOWANCE_BYTES,"two_workspace_policies":2*32*1024*1024,"live_weight_owner_allowance":1024*1024,"legacy_owner_allowance":1024*1024,"failed_upload_overlap_allowance":4*1024*1024,"source_staging_allowance":1024*1024,"fixture_host_policy":32*1024*1024,"serialization_policy":64*1024*1024,"caller_policy":1024*1024});
        let total = categories
            .as_object()
            .unwrap()
            .values()
            .try_fold(0_u64, |sum, v| sum.checked_add(v.as_u64().unwrap()))
            .unwrap();
        assert!(total <= 512 * 1024 * 1024);
        emit(
            "PLAN",
            json!({"categories":categories,"total_policy_bytes":total,"native_process_allowance_bytes":512*1024*1024,"exact_physical_residency":false}),
        );
        emit(
            "META",
            json!({"context_policy_bytes":CONTEXT_ALLOWANCE_BYTES,"owner_struct_bytes":size_of::<Weights>(),"session_struct_bytes":size_of::<Gather<'_>>(),"whole_graph_qualified":false,"performance_qualified":false}),
        );
        let data = fixture::fixture();
        for case in data["cases"].as_array().unwrap() {
            let matrix = fixture::matrix(case);
            let layout = matrix.layout();
            let k = layout.columns();
            let ids: Vec<_> = case["gather_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(fixture::number)
                .collect();
            let expected = fixture::floats(&case["arrays"]["gather"]);
            for maximum in [7, 8] {
                let key = format!("{}/max{maximum}", case["name"].as_str().unwrap());
                let weights = autoreleasepool(|_| {
                    let source_context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
                    emit(
                        "CONTEXT",
                        json!({"key":key,"phase":"construction","device":source_context.device_name(),"identity":format!("{:p}",&*source_context.gpu.device)}),
                    );
                    let mut calls = 0;
                    let budget = minimum_budget(layout, maximum);
                    emit(
                        "CONSTRUCT_ATTEMPT",
                        json!({"key":key,"rows":layout.rows(),"columns":k,"maximum":maximum,"budget":budget}),
                    );
                    let constructed=source_context.weights_from_fill(layout,maximum,budget,|first,child,c,s,b|{
                        calls+=1;emit("FILL_ATTEMPT",json!({"key":key,"first_row":first,"rows":child.rows(),"addresses":[format!("{:p}",c.as_ptr()),format!("{:p}",s.as_ptr()),format!("{:p}",b.as_ptr())]}));
                        fill(&matrix,first,child,c,s,b)?;
                        emit("FILL",json!({"key":key,"first_row":first,"codes":c,"scales":s,"biases":b}));Ok(())
                    });
                    emit(
                        "CONSTRUCT_OUTCOME",
                        json!({"key":key,"completed":constructed.is_ok(),"error":constructed.as_ref().err().map(ToString::to_string),"callbacks":calls}),
                    );
                    let weights = constructed.unwrap();
                    emit(
                        "WEIGHTS",
                        json!({"key":key,"observation":observe(&weights)}),
                    );
                    assert!(weights.same_device(&source_context));
                    if layout.rows() == 19 && maximum == 7 {
                        for kind in ["budget", "first-source", "late-source", "domain", "panic"] {
                            construction_fault(
                                &source_context,
                                &weights,
                                &matrix,
                                maximum,
                                budget,
                                &key,
                                kind,
                            );
                        }
                    }
                    weights
                });
                // No borrowed Context or unsafe lifetime extension: the source
                // Context and its autorelease pool have gone out of scope.
                emit(
                    "CONSTRUCTION_CONTEXT_DROPPED",
                    json!({"key":key,"weights":observe(&weights)}),
                );
                let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
                emit(
                    "CONTEXT",
                    json!({"key":key,"phase":"reuse","device":context.device_name(),"identity":format!("{:p}",&*context.gpu.device),"same_device":weights.same_device(&context)}),
                );
                assert!(weights.same_device(&context));
                let mut session = context.gather_session(k, 32, 32 * 1024 * 1024).unwrap();
                emit(
                    "WORKSPACE",
                    json!({"key":key,"workspace":workspace(&session,&[])}),
                );
                for index in 0..2 {
                    let (result, caller) =
                        step(&mut session, &weights, &key, index, &ids, expected.len());
                    result.unwrap();
                    assert_eq!(bits(&caller), bits(&expected));
                }
                let reverse: Vec<_> = ids.iter().rev().copied().collect();
                let (result, caller) =
                    step(&mut session, &weights, &key, 2, &reverse, expected.len());
                result.unwrap();
                let reversed: Vec<_> = expected.chunks_exact(k).rev().flatten().copied().collect();
                assert_eq!(bits(&caller), bits(&reversed));
                for (index, count) in [(3, 1), (4, 32)] {
                    let request = vec![layout.rows() - 1; count];
                    let (result, caller) =
                        step(&mut session, &weights, &key, index, &request, count * k);
                    result.unwrap();
                    let mut row = vec![0.; k];
                    matrix.copy_row(layout.rows() - 1, &mut row).unwrap();
                    assert!(caller.chunks_exact(k).all(|v| bits(v) == bits(&row)));
                }
                let (result, caller) = step(&mut session, &weights, &key, 5, &[layout.rows()], k);
                assert!(result.is_err());
                assert!(caller.iter().all(|v| *v == 42.));
                FAILURE_AFTER.with(|v| v.set(Some(2)));
                let (result, caller) = step(&mut session, &weights, &key, 6, &ids, expected.len());
                FAILURE_AFTER.with(|v| v.set(None));
                assert!(matches!(
                    result,
                    Err(Error::Options(
                        "Gemma table gather injected composition failure"
                    ))
                ));
                assert!(caller.iter().all(|v| *v == 42.));
                assert!(session.poisoned);
                let (result, caller) = step(&mut session, &weights, &key, 7, &ids, expected.len());
                assert!(result.is_err());
                assert!(caller.iter().all(|v| *v == 42.));
                let mut recovered = context.gather_session(k, 32, 32 * 1024 * 1024).unwrap();
                emit(
                    "RECOVERED_WORKSPACE",
                    json!({"key":key,"workspace":workspace(&recovered,&[])}),
                );
                let (result, caller) =
                    step(&mut recovered, &weights, &key, 8, &ids, expected.len());
                result.unwrap();
                assert_eq!(bits(&caller), bits(&expected));
                let other_case = data["cases"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|c| fixture::number(&c["columns"]) != k)
                    .unwrap();
                let other_source = fixture::matrix(other_case);
                let other_layout = other_source.layout();
                let other_budget = minimum_budget(other_layout, maximum);
                emit(
                    "MISMATCH_OWNER_ATTEMPT",
                    json!({"key":key,"rows":other_layout.rows(),"columns":other_layout.columns(),"maximum":maximum,"budget":other_budget}),
                );
                let mut other_callbacks = 0;
                let other_result=context.weights_from_fill(other_layout,maximum,other_budget,|first,child,c,s,b|{
                    other_callbacks+=1;fill(&other_source,first,child,c,s,b)?;
                    emit("MISMATCH_FILL",json!({"key":key,"first_row":first,"rows":child.rows(),"codes":c,"scales":s,"biases":b}));Ok(())
                });
                emit(
                    "MISMATCH_OWNER_OUTCOME",
                    json!({"key":key,"completed":other_result.is_ok(),"error":other_result.as_ref().err().map(ToString::to_string),"callbacks":other_callbacks}),
                );
                let other_weights = other_result.unwrap();
                emit(
                    "MISMATCH_OWNER",
                    json!({"key":key,"weights":observe(&other_weights)}),
                );
                let mut rejected = vec![42.; k];
                let before = json!({"workspace":workspace(&recovered,&rejected),"weights":observe(&weights),"other_weights":observe(&other_weights)});
                emit("MISMATCH_BEFORE", json!({"key":key,"observation":before}));
                let rejected_result = recovered.apply_weights(&other_weights, &[0], &mut rejected);
                emit(
                    "MISMATCH_OUTCOME",
                    json!({"key":key,"completed":rejected_result.is_ok(),"error":rejected_result.as_ref().err().map(ToString::to_string)}),
                );
                let after = json!({"workspace":workspace(&recovered,&rejected),"weights":observe(&weights),"other_weights":observe(&other_weights)});
                emit("MISMATCH_AFTER", json!({"key":key,"observation":after}));
                assert!(matches!(
                    rejected_result,
                    Err(Error::Options(
                        "Gemma table gather weights device/shape mismatch"
                    ))
                ));
                assert_eq!(before, after);
                let (result, caller) =
                    step(&mut recovered, &weights, &key, 9, &ids, expected.len());
                result.unwrap();
                assert_eq!(bits(&caller), bits(&expected));
                // The original API and its exact Context checks stay distinct;
                // both sources now use the same encode/completion implementation.
                let old = context
                    .partitioned_affine(&matrix, maximum, 1, 1024 * 1024)
                    .unwrap();
                let mut caller = vec![42.; expected.len()];
                let result = recovered.apply(&old, &ids, &mut caller);
                emit(
                    "LEGACY",
                    json!({"key":key,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"observation":snapshot(&recovered,&old,&caller)}),
                );
                result.unwrap();
                assert_eq!(bits(&caller), bits(&expected));
                emit("FINAL", json!({"key":key,"weights":observe(&weights)}));
            }
        }
        emit(
            "RESULT",
            json!({"cases":4,"steps":40,"source_contexts_dropped":4,"foreign_device_qualified":false,"whole_graph_qualified":false,"driver_or_inflight_failure_injected":false,"performance_qualified":false}),
        );
    });
}
