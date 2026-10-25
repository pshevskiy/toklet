use super::super::super::tests::{fixture, observe, step};
use super::*;
use serde_json::{Value, json};

fn report(label: &str, body: Value) {
    use std::io::Write;
    let mut output = std::io::stdout().lock();
    writeln!(output, "GEMMA_STREAM_{label} {body}").unwrap();
    output.flush().unwrap();
}
#[derive(Clone, Copy)]
enum Fault {
    None,
    Source(usize),
    Domain(usize),
    Panic(usize),
}
impl Fault {
    fn action(self, index: usize) -> &'static str {
        match self {
            Self::Source(at) if index == at => "source-error",
            Self::Domain(at) if index == at => "invalid-domain",
            Self::Panic(at) if index == at => "panic",
            _ => "success",
        }
    }
}
fn construct<'a>(
    context: &'a Context,
    source: &Matrix<'_>,
    maximum: usize,
    tag: (&str, &str),
    limit: usize,
    fault: Fault,
    calls: &mut usize,
) -> Result<PartitionedAffine<'a>> {
    let (key, phase) = tag;
    let layout = source.layout();
    let plan = UploadPlan::new(layout, maximum, 3, usize::MAX).unwrap();
    report(
        "ATTEMPT",
        json!({"key":key,"phase":phase,"rows":layout.rows(),"columns":layout.columns(),"maximum":maximum,"capacity":3,"minimum_working":plan.minimum_working,"limit":limit,"source_capacities":plan.source_bytes}),
    );
    let mut addresses = None;
    let result = context.partitioned_affine_from_fill(
        layout,
        maximum,
        3,
        limit,
        |first, child, codes, scales, biases| {
            let index = *calls;
            *calls += 1;
            let pointers = [codes.as_ptr() as usize, scales.as_ptr() as usize, biases.as_ptr() as usize];
            let before = json!({"codes":codes,"scales":scales,"biases":biases});
            report("FILL_ATTEMPT",json!({"key":key,"phase":phase,"index":index,"first":first,"rows":child.rows(),"columns":child.columns(),"addresses":pointers.map(|p|format!("0x{p:x}")),"before":before}));
            // Record complete safe prefixes before source-shape/address assertions.
            if let Some(previous) = addresses {
                assert_eq!(previous,pointers);
            }
            addresses = Some(pointers);
            let view = source.row_range(first..first+child.rows()).unwrap();
            assert_eq!(view.layout(),child);
            let (weights, original_scales, original_biases) = view.original_tensors();
            codes.copy_from_slice(weights);
            scales.copy_from_slice(original_scales);
            biases.copy_from_slice(original_biases);
            let action = fault.action(index);
            if action=="invalid-domain" {
                scales[..2].copy_from_slice(&0x7f7f_u16.to_le_bytes());
                biases[..2].copy_from_slice(&0_u16.to_le_bytes());
            }
            report("FILL_RESULT",json!({"key":key,"phase":phase,"index":index,"action":action,"after":{"codes":codes,"scales":scales,"biases":biases}}));
            match action {
                "source-error"=>Err(Error::Native(format!("stream source error at {index}"))),
                "panic"=>panic!("original stream callback panic"),
                _=>Ok(()),
            }
        },
    );
    report(
        "OUTCOME",
        json!({"key":key,"phase":phase,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"panic":null,"callback_calls":*calls}),
    );
    if let Ok(owner) = &result {
        report(
            "OWNER",
            json!({"key":key,"phase":phase,"observation":observe(owner,&[])}),
        );
    }
    result
}
#[test]
#[ignore = "explicit bounded source upload and construction-failure GPU qualification"]
fn frozen_stream_upload_reuses_source_storage_and_recovers_after_failed_construction() {
    autoreleasepool(|_| {
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        report(
            "META",
            json!({"device":context.device_name(),"simd_width":context.width,"context_allowance_bytes":CONTEXT_ALLOWANCE_BYTES,"source_descriptors_bytes":size_of::<SourceScratch>(),"parent_bytes":size_of::<PartitionedAffine<'_>>(),"child_bytes":size_of::<Affine<'_>>(),"actual_large_table_qualified":false,"whole_graph_qualified":false}),
        );
        let data = fixture::fixture();
        let mut cases = 0;
        for case in data["cases"].as_array().unwrap() {
            let source = fixture::matrix(case);
            let input = fixture::floats(&case["arrays"]["input"]);
            let expected = fixture::floats(&case["arrays"]["output_f32"]);
            for maximum in [7, 8] {
                let key = format!("{}/max{maximum}", case["name"].as_str().unwrap());
                let plan = UploadPlan::new(source.layout(), maximum, 3, usize::MAX).unwrap();
                let mut calls = 0;
                let mut owner = construct(
                    &context,
                    &source,
                    maximum,
                    (&key, "main"),
                    plan.minimum_working,
                    Fault::None,
                    &mut calls,
                )
                .unwrap();
                assert_eq!(calls, plan.resident.chunks);
                for repeat in 0..2 {
                    let (output, result, _) = step(
                        &mut owner,
                        &key,
                        "repeat",
                        repeat,
                        &input,
                        3,
                        expected.len(),
                    );
                    result.unwrap();
                    assert_eq!(
                        output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
                for token in 0..3 {
                    let k = source.layout().columns();
                    let n = source.layout().rows();
                    let (output, result, _) = step(
                        &mut owner,
                        &key,
                        "part",
                        token,
                        &input[token * k..(token + 1) * k],
                        1,
                        n,
                    );
                    result.unwrap();
                    assert_eq!(
                        output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected[token * n..(token + 1) * n]
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    );
                }
                let (output, result, _) =
                    step(&mut owner, &key, "resume", 0, &input, 3, expected.len());
                result.unwrap();
                assert_eq!(
                    output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                if cases == 0 {
                    let warmed = observe(&owner, &[]);
                    for (phase, fault, limit, wanted_calls) in [
                        ("budget", Fault::None, plan.minimum_working - 1, 0),
                        ("first-source", Fault::Source(0), plan.minimum_working, 1),
                        ("late-source", Fault::Source(1), plan.minimum_working, 2),
                        ("late-domain", Fault::Domain(1), plan.minimum_working, 2),
                    ] {
                        let mut calls = 0;
                        let result = construct(
                            &context,
                            &source,
                            maximum,
                            (&key, phase),
                            limit,
                            fault,
                            &mut calls,
                        );
                        let after = observe(&owner, &[]);
                        report(
                            "UNCHANGED_OWNER",
                            json!({"key":key,"phase":phase,"observation":after}),
                        );
                        assert!(result.is_err());
                        assert_eq!(calls, wanted_calls);
                        match phase {
                            "budget" => assert!(
                                matches!(result,Err(Error::Budget{required,limit}) if required==plan.minimum_working && limit==plan.minimum_working-1)
                            ),
                            "first-source" => assert!(
                                matches!(result,Err(Error::Native(ref text)) if text=="stream source error at 0")
                            ),
                            "late-source" => assert!(
                                matches!(result,Err(Error::Native(ref text)) if text=="stream source error at 1")
                            ),
                            _ => assert!(matches!(
                                result,
                                Err(Error::Unsupported("Gemma affine weight domain"))
                            )),
                        }
                        assert_eq!(warmed, after);
                    }
                    let mut calls = 0;
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        construct(
                            &context,
                            &source,
                            maximum,
                            (&key, "panic"),
                            plan.minimum_working,
                            Fault::Panic(1),
                            &mut calls,
                        )
                    }));
                    let panic = result
                        .as_ref()
                        .err()
                        .and_then(|p| p.downcast_ref::<&str>())
                        .copied();
                    let unexpected_error = match &result {
                        Ok(Err(error)) => Some(error.to_string()),
                        _ => None,
                    };
                    report(
                        "OUTCOME",
                        json!({"key":key,"phase":"panic","completed":matches!(&result,Ok(Ok(_))),"error":unexpected_error,"panic":panic,"callback_calls":calls}),
                    );
                    if let Ok(Ok(owner)) = &result {
                        report(
                            "OWNER",
                            json!({"key":key,"phase":"panic","observation":observe(owner,&[])}),
                        );
                    }
                    let after = observe(&owner, &[]);
                    report(
                        "UNCHANGED_OWNER",
                        json!({"key":key,"phase":"panic","observation":after}),
                    );
                    assert_eq!(panic, Some("original stream callback panic"));
                    assert_eq!(calls, 2);
                    assert_eq!(warmed, after);
                    let mut calls = 0;
                    let recovered = construct(
                        &context,
                        &source,
                        maximum,
                        (&key, "recover"),
                        plan.minimum_working,
                        Fault::None,
                        &mut calls,
                    );
                    let after = observe(&owner, &[]);
                    report(
                        "UNCHANGED_OWNER",
                        json!({"key":key,"phase":"recover-construct","observation":after}),
                    );
                    assert_eq!(warmed, after);
                    let mut recovered = recovered.unwrap();
                    let (output, result, _) = step(
                        &mut recovered,
                        &(key.clone() + "/recover"),
                        "recover",
                        0,
                        &input,
                        3,
                        expected.len(),
                    );
                    let after = observe(&owner, &[]);
                    report(
                        "UNCHANGED_OWNER",
                        json!({"key":key,"phase":"recover","observation":after}),
                    );
                    result.unwrap();
                    assert_eq!(
                        output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    assert_eq!(after, warmed);
                }
                report(
                    "CASE_END",
                    json!({"key":key,"observation":observe(&owner,&[])}),
                );
                cases += 1;
            }
        }
        report(
            "FINAL",
            json!({"cases":cases,"successful_owners":5,"projection_steps":25,"source_errors":2,"late_domain_errors":1,"callback_panics":1,"budget_rejections":1,"callback_side_effects_rolled_back":false,"gpu_commands_submitted_during_upload":false,"actual_large_table_qualified":false,"whole_graph_qualified":false,"performance_qualified":false}),
        );
        assert_eq!(cases, 4);
    });
}
