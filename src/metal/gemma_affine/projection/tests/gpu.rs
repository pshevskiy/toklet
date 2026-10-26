//! Explicit tiny independent source/state witness. No ordinary test needs Metal.
use super::*;
use serde_json::{Value, json};
use std::io::Write;

const HOT: f32 = 16777216.;
const CASES: [(&str, usize); 6] = [
    ("original", 0),
    ("double", 1),
    ("hot", 2),
    ("oversized", 0),
    ("rows", 3),
    ("width", 3),
];
fn emit(label: &str, value: Value) {
    let mut out = std::io::stdout().lock();
    writeln!(out, "GEMMA_PROJECTION_{label} {value}").unwrap();
    out.flush().unwrap();
}
fn buffer(buffer: &Buffer<u32>, safe: bool) -> Value {
    json!({"allocation":format!("{:p}",&*buffer.raw),"length":buffer.len,
        "words":safe.then(||buffer.with_slice(<[u32]>::to_vec))})
}
fn weight(weights: &Weights, safe: bool) -> Value {
    let (slots, chunk_bytes, device) = weights.diagnostic_storage();
    let chunks: Vec<_> = weights
        .views()
        .map(|v| {
            json!({"rows":v.layout.rows(),"columns":v.layout.columns(),
        "codes":buffer(v.codes,safe),"scales":buffer(v.scales,safe),"biases":buffer(v.biases,safe)})
        })
        .collect();
    json!({"rows":weights.layout().rows(),"columns":weights.layout().columns(),"owned_bytes":weights.owned_bytes(),
        "owner_slots":slots,"chunk_struct_bytes":chunk_bytes,"device":device,"maximum_bits":weights.max_abs_weight().to_bits(),"chunks":chunks})
}
fn safe(session: &Projection<'_>) -> bool {
    session
        .pending
        .as_ref()
        .is_none_or(|v| v.status() == MTLCommandBufferStatus::Completed)
}
fn observation(session: &Projection<'_>) -> Value {
    let read = safe(session);
    json!({"rows":session.plan.layout.rows(),"columns":session.plan.layout.columns(),"maximum_rows":session.plan.maximum_rows,
        "capacity":session.plan.capacity,"owned_bytes":session.owned_bytes(),"poisoned":session.is_poisoned(),
        "pending":session.pending.is_some(),"shared_read_safe":read,
        "staged_capacity":session.staged.capacity(),"chunk_capacity":session.chunk_output.capacity(),
        "staged_bits":session.staged.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
        "chunk_bits":session.chunk_output.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
        "input":buffer(&session.input,read),"output":buffer(&session.output,read),"status":buffer(&session.status,read)})
}
fn policy() -> Value {
    let categories = json!({"two_contexts":2*CONTEXT_ALLOWANCE_BYTES,"immutable_weights":8*1024*1024,
        "source_loading":4*1024*1024,"two_projection_workspaces":8*1024*1024,"frozen_source_fixture":32*1024*1024,
        "serialization":64*1024*1024,"diagnostic_state":64*1024*1024,"caller":1024*1024});
    let total: u64 = categories
        .as_object()
        .unwrap()
        .values()
        .map(|v| v.as_u64().unwrap())
        .sum();
    assert!(total < 512 * 1024 * 1024);
    json!({"categories":categories,"total_policy_bytes":total,"native_process_allowance_bytes":512*1024*1024,"exact_physical_residency":false})
}
fn layout(rows: usize, columns: usize) -> Layout {
    Layout::new(rows, columns, 4, 64, MetadataDtype::BF16).unwrap()
}
fn upload(
    context: &Context,
    matrix: &Matrix<'_>,
    role: &str,
    variant: usize,
    maximum: usize,
) -> Weights {
    let source = matrix.layout();
    let rows = source.rows() + usize::from(role == "rows");
    let columns = if role == "width" {
        if source.columns() == 64 { 128 } else { 64 }
    } else {
        source.columns()
    };
    let target = layout(rows, columns);
    emit(
        "UPLOAD_ATTEMPT",
        json!({"role":role,"variant":variant,"rows":rows,"columns":columns,"maximum_rows":maximum,"working_budget":1024*1024}),
    );
    let original = matrix.original_tensors();
    let result = context.weights_from_fill(
        target,
        maximum,
        1024 * 1024,
        |first, child, codes, scales, biases| {
            for (index, array) in [codes, scales, biases].into_iter().enumerate() {
                if variant >= 2 {
                    let value: u16 = if index == 2 {
                        if variant == 2 { 0x4b80 } else { 0x3f80 }
                    } else {
                        0
                    };
                    for raw in array.chunks_exact_mut(2) {
                        raw.copy_from_slice(&value.to_le_bytes());
                    }
                } else {
                    let source_bytes = [original.0, original.1, original.2][index];
                    let stride = source_bytes.len() / source.rows();
                    let start = first * stride;
                    array.copy_from_slice(&source_bytes[start..start + array.len()]);
                    if variant == 1 && index != 0 {
                        for raw in array.chunks_exact_mut(2) {
                            let value = f32::from_bits(
                                u32::from(u16::from_le_bytes(raw.try_into().unwrap())) << 16,
                            ) * 2.;
                            raw.copy_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
                        }
                    }
                }
            }
            emit(
                "FILL",
                json!({"role":role,"first_row":first,"rows":child.rows()}),
            );
            Ok(())
        },
    );
    emit(
        "UPLOAD_OUTCOME",
        json!({"role":role,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string)}),
    );
    result.unwrap()
}
fn wanted_error(index: usize) -> Option<&'static str> {
    match index {
        5 => Some("Gemma affine input ordered-sum accuracy bound"),
        7..=9 => Some(MISMATCH),
        10..=15 => Some("Gemma projection input shape/nonfinite value"),
        17 => Some("Gemma projection injected composition failure"),
        18 => Some("Gemma projection workspace poisoned; recreate"),
        _ => None,
    }
}
#[test]
#[ignore = "explicit immutable projection scratch, old/new Context teardown and full state replay"]
fn immutable_projection_reuses_weights_and_preserves_complete_transactions() {
    autoreleasepool(|_| {
        emit("PLAN", policy());
        let data = fixture::fixture();
        let mut cases = 0;
        for case in data["cases"].as_array().unwrap() {
            let matrix = fixture::matrix(case);
            let shape = matrix.layout();
            let input = fixture::floats(&case["arrays"]["input"]);
            let expected = fixture::floats(&case["arrays"]["output_f32"]);
            for partition in [7, 8] {
                let key = format!("{}/{}", case["name"].as_str().unwrap(), partition);
                emit(
                    "CASE",
                    json!({"key":key,"rows":shape.rows(),"columns":shape.columns(),"partition":partition,
                    "projection_struct_bytes":size_of::<Projection<'_>>(),"weights_struct_bytes":size_of::<Weights>()}),
                );
                let weights = autoreleasepool(|_| {
                    let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
                    emit(
                        "CONSTRUCTION_CONTEXT",
                        json!({"device":context.device_name(),"identity":format!("{:p}",&*context.gpu.device)}),
                    );
                    CASES.map(|(role, variant)| {
                        let maximum = match role {
                            "double" => 15 - partition,
                            "oversized" => 9,
                            "rows" | "width" => 8,
                            _ => partition,
                        };
                        let owner = upload(&context, &matrix, role, variant, maximum);
                        emit(
                            "WEIGHTS",
                            json!({"role":role,"observation":weight(&owner,true)}),
                        );
                        owner
                    })
                });
                emit(
                    "CONSTRUCTION_CONTEXT_DROPPED",
                    json!({"owners":6,"devices":weights.each_ref().map(|v|v.diagnostic_storage().2)}),
                );
                let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
                emit(
                    "LOOKUP_CONTEXT",
                    json!({"device":context.device_name(),"identity":format!("{:p}",&*context.gpu.device),"same_devices":weights.each_ref().map(|v|v.same_device(&context))}),
                );
                let required = Plan::new(shape, 8, 3, usize::MAX).unwrap().minimum_owned;
                let rejected = context.projection_session(shape, 8, 3, required - 1);
                emit(
                    "BUDGET_OUTCOME",
                    json!({"required":required,"limit":required-1,"completed":rejected.is_ok(),"error":rejected.as_ref().err().map(ToString::to_string)}),
                );
                assert!(matches!(rejected, Err(Error::Budget { .. })));
                emit(
                    "WORKSPACE_ATTEMPT",
                    json!({"phase":"initial","budget":required}),
                );
                let constructed = context.projection_session(shape, 8, 3, required);
                emit(
                    "WORKSPACE_OUTCOME",
                    json!({"phase":"initial","completed":constructed.is_ok(),"error":constructed.as_ref().err().map(ToString::to_string)}),
                );
                let mut session = constructed.unwrap();
                let mut retired = None;
                emit("COLD", observation(&session));
                for index in 0..22 {
                    if index == 19 {
                        emit(
                            "WORKSPACE_ATTEMPT",
                            json!({"phase":"fresh","budget":required}),
                        );
                        let constructed = context.projection_session(shape, 8, 3, required);
                        emit(
                            "WORKSPACE_OUTCOME",
                            json!({"phase":"fresh","completed":constructed.is_ok(),"error":constructed.as_ref().err().map(ToString::to_string)}),
                        );
                        let replacement = constructed.unwrap();
                        retired = Some(std::mem::replace(&mut session, replacement));
                        emit(
                            "RECOVERY",
                            json!({"retired":observation(retired.as_ref().unwrap()),"fresh":observation(&session)}),
                        );
                    }
                    let owner = match index {
                        1 | 3 | 17 | 19 => 1,
                        4 | 5 | 20 => 2,
                        7 => 4,
                        8 => 5,
                        9 => 3,
                        _ => 0,
                    };
                    let tokens = match index {
                        0 | 1 | 6 | 16..=21 => 3,
                        3 => 2,
                        10 => 4,
                        15 => 0,
                        _ => 1,
                    };
                    let mut actual_input: Vec<_> = (0..tokens * shape.columns())
                        .map(|i| input[i % input.len()])
                        .collect();
                    if matches!(index, 4 | 20) {
                        actual_input.fill(0.);
                        for row in actual_input.chunks_exact_mut(shape.columns()) {
                            row[0] = 1.;
                        }
                    }
                    if index == 5 {
                        actual_input.fill(1.);
                    }
                    if index == 11 {
                        actual_input[0] = f32::NAN;
                    }
                    if index == 14 {
                        actual_input.pop();
                    }
                    let length = if index == 13 {
                        tokens * shape.rows() - 1
                    } else {
                        tokens * shape.rows()
                    };
                    let mut caller = vec![42.; length];
                    if index == 12 {
                        caller.push(42.);
                    }
                    emit(
                        "STEP_ATTEMPT",
                        json!({"index":index,"role":CASES[owner].0,"tokens":tokens,"input_bits":actual_input.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"caller_length":caller.len()}),
                    );
                    if index == 17 {
                        FAILURE_AFTER.with(|v| v.set(Some(2)));
                    }
                    super::super::super::tests::L1_CALLS.with(|v| v.set(0));
                    let result = session.apply(&weights[owner], &actual_input, &mut caller, tokens);
                    let l1 = super::super::super::tests::L1_CALLS.with(std::cell::Cell::get);
                    FAILURE_AFTER.with(|v| v.set(None));
                    emit(
                        "STEP_OUTCOME",
                        json!({"index":index,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"l1_calls":l1}),
                    );
                    let read = safe(&session) && retired.as_ref().is_none_or(safe);
                    emit(
                        "STEP",
                        json!({"index":index,"caller_bits":caller.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"workspace":observation(&session),
                        "retired":retired.as_ref().map(observation),"weights":weights.each_ref().map(|v|weight(v,read))}),
                    );
                    if let Some(error) = wanted_error(index) {
                        assert_eq!(result.unwrap_err().to_string(), error);
                        assert!(caller.iter().all(|&v| v == 42.));
                    } else {
                        result.unwrap();
                        let expected: Vec<_> = if matches!(index, 4 | 20) {
                            vec![HOT; tokens * shape.rows()]
                        } else {
                            expected[..tokens * shape.rows()]
                                .iter()
                                .map(|v| v * if owner == 1 { 2. } else { 1. })
                                .collect()
                        };
                        assert_eq!(
                            caller.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                        );
                    }
                }
                for (index, owner) in weights.iter().enumerate() {
                    emit(
                        "FINAL_WEIGHTS",
                        json!({"role":CASES[index].0,"observation":weight(owner,true)}),
                    );
                }
                emit(
                    "FINAL",
                    json!({"key":key,"workspace":observation(&session),"retired":observation(retired.as_ref().unwrap())}),
                );
                cases += 1;
            }
        }
        emit(
            "RESULT",
            json!({"cases":cases,"steps":88,"construction_contexts_dropped":4,"same_geometry_weight_swaps":true,
            "foreign_device_qualified":false,"driver_or_inflight_failure_injected":false,"whole_graph_qualified":false,"performance_qualified":false}),
        );
    });
}

mod real;
