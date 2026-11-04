use super::*;
use crate as engine;
#[path = "../../../tests/common/gemma_affine_fixture.rs"]
mod fixture;
#[path = "../../../tests/common/gemma_affine_l1_fixture.rs"]
mod l1_fixture;
use serde_json::{Value, json};

mod native_arithmetic;
mod real;

thread_local! {
    pub(super) static L1_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[test]
fn frozen_l1_fallback_rows_bounds_and_fast_path_without_gpu() {
    let before = L1_CALLS.with(std::cell::Cell::get);
    admit(64, 1., &[1.; 64]).unwrap();
    assert_eq!(L1_CALLS.with(std::cell::Cell::get), before);
    let mut new = 0;
    for case in l1_fixture::cases() {
        let maximum = codebook(&case.scales, &case.biases).unwrap();
        Matrix::from_le_bytes(case.layout, &case.codes, &case.scales, &case.biases).unwrap();
        let k = case.layout.columns();
        for (row, expected) in case
            .input
            .chunks_exact(k)
            .zip(case.reference["per_token_l1"].as_array().unwrap())
        {
            assert_eq!(row_l1(row).to_bits(), expected.as_f64().unwrap().to_bits());
        }
        let (s, u) = l1_bounds(k, maximum, &case.input);
        assert_eq!(
            s.to_bits(),
            case.reference["l1"]["absolute_products"]
                .as_f64()
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            u.to_bits(),
            case.reference["l1"]["ordered_error"]
                .as_f64()
                .unwrap()
                .to_bits()
        );
        let old = bounds(
            k,
            maximum,
            case.input.iter().fold(0_f32, |a, x| a.max(x.abs())),
        );
        assert_eq!(
            old.0.to_bits(),
            case.reference["legacy_max"]["absolute_products"]
                .as_f64()
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            old.1.to_bits(),
            case.reference["legacy_max"]["ordered_error"]
                .as_f64()
                .unwrap()
                .to_bits()
        );
        let before = L1_CALLS.with(std::cell::Cell::get);
        let result = admit(k, maximum, &case.input);
        assert_eq!(
            result.is_ok(),
            case.reference["combined_admitted"] == true,
            "{}",
            case.reference["name"]
        );
        let old_passed = case.reference["legacy_max"]["admitted"] == true;
        assert_eq!(
            L1_CALLS.with(std::cell::Cell::get),
            before + usize::from(!old_passed)
        );
        if result.is_ok() && !old_passed {
            new += 1;
        }
        if let Err(error) = result {
            let expected = if old.0 > f64::from(f32::MAX) / 4. {
                "Gemma affine input absolute-product bound"
            } else {
                "Gemma affine input ordered-sum accuracy bound"
            };
            assert!(matches!(error, Error::Options(actual) if actual == expected));
        }
    }
    assert_eq!(new, 12);
}

#[test]
fn bounded_shapes_codebooks_and_upward_admission_without_gpu() {
    assert!(matches!(
        Context::new(CONTEXT_ALLOWANCE_BYTES - 1),
        Err(Error::Budget { .. })
    ));
    let layout = Layout::new(3, 64, 4, 64, MetadataDtype::BF16).unwrap();
    for capacity in [0, 33, usize::MAX] {
        assert!(Sizes::new(layout, capacity).is_err());
    }
    for bad in [
        Layout::new(3, 64, 8, 64, MetadataDtype::BF16).unwrap(),
        Layout::new(3, 64, 4, 32, MetadataDtype::BF16).unwrap(),
        Layout::new(3, 64, 4, 64, MetadataDtype::F16).unwrap(),
        Layout::new(16385, 64, 4, 64, MetadataDtype::BF16).unwrap(),
        Layout::new(1, 12352, 4, 64, MetadataDtype::BF16).unwrap(),
    ] {
        assert!(Sizes::new(bad, 1).is_err());
    }
    let maximum = Sizes::new(
        Layout::new(16384, 12288, 4, 64, MetadataDtype::BF16).unwrap(),
        32,
    )
    .unwrap();
    assert_eq!(maximum.codes, 25165824);
    assert_eq!(maximum.metadata, 1572864);
    assert_eq!(maximum.input, 393216);
    assert_eq!(maximum.output, 524288);
    assert_eq!(
        maximum.owned,
        maximum
            .buffers()
            .iter()
            .map(|n| (n + GUARD) * 4)
            .sum::<usize>()
            + size_of::<Affine<'_>>()
    );
    let mut count = 0;
    for fixture in fixture::fixtures() {
        for case in &fixture.cases {
            let layout = fixture.layout(case);
            let result = codebook(fixture.bytes(case, "scales"), fixture.bytes(case, "biases"));
            if case["constructor"]["admitted"] == true {
                let max_weight = result.unwrap();
                if let Some(expected) = case["constructor"]["max_weight"].as_f64() {
                    assert_eq!(max_weight, expected);
                }
                let input = fixture.input(case);
                let allowed = case["input_admission"]["admitted"] == true;
                assert_eq!(
                    admit(layout.columns(), max_weight, &input).is_ok(),
                    allowed,
                    "{}",
                    case["name"]
                );
                if let Some(s) = case["input_admission"]["absolute_products_bound"].as_f64() {
                    let (actual_s, actual_u) = bounds(
                        layout.columns(),
                        max_weight,
                        input.iter().fold(0_f32, |a, x| a.max(x.abs())),
                    );
                    assert_eq!(actual_s, s);
                    assert_eq!(
                        actual_u,
                        case["input_admission"]["ordered_error_bound"]
                            .as_f64()
                            .unwrap()
                    );
                }
            } else {
                assert!(result.is_err(), "{}", case["name"]);
            }
            count += 1;
        }
    }
    assert_eq!(count, 31);
}
fn bits(input: &[f32]) -> Vec<u32> {
    input.iter().map(|v| v.to_bits()).collect()
}
fn report(label: &str, value: Value) {
    use std::io::Write;
    println!("{label} {value}");
    std::io::stdout().flush().unwrap();
}
// Capture every raw completed buffer BEFORE numerical/guard assertions. No CPU
// read is allowed after failed/unfinished commands; they remain retained.
fn device(operator: &Affine<'_>) -> Option<Value> {
    if let Some(command) = &operator.pending {
        if command.status() != MTLCommandBufferStatus::Completed {
            return None;
        }
    } else if operator.poisoned {
        return None;
    }
    Some(
        json!({"codes":operator.codes.with_slice(|s|s.to_vec()),"scales":operator.scales.with_slice(|s|s.to_vec()),"biases":operator.biases.with_slice(|s|s.to_vec()),"input":operator.input.with_slice(|s|s.to_vec()),"output":operator.output.with_slice(|s|s.to_vec()),"status":operator.status.with_slice(|s|s.to_vec())}),
    )
}

#[test]
#[ignore = "explicit Apple GPU L1 fallback qualification"]
fn real_frozen_l1_projection_reuse_partitions_and_rejections() {
    autoreleasepool(|_| {
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        report(
            "GEMMA_AFFINE_L1_META",
            json!({"device":context.device_name(),"context_allowance_bytes":CONTEXT_ALLOWANCE_BYTES,"simd_width":context.width,"owner_struct_bytes":size_of::<Affine<'_>>(),"graph_qualified":false,"source_parity_qualified":false}),
        );
        let mut accepted = 0;
        let mut rejected = 0;
        let mut parts = 0;
        for case in l1_fixture::cases() {
            let capacity = case.reference["batch"].as_u64().unwrap() as usize;
            let sizes = Sizes::new(case.layout, capacity).unwrap();
            let mut operator = context
                .affine(
                    case.layout,
                    &case.codes,
                    &case.scales,
                    &case.biases,
                    capacity,
                    sizes.owned,
                )
                .unwrap();
            let mut output = vec![42.; capacity * case.layout.rows()];
            if case.reference["combined_admitted"] != true {
                let result = operator.apply(&case.input, &mut output, capacity);
                report(
                    "GEMMA_AFFINE_L1_REJECTION",
                    json!({"name":case.reference["name"],"rejected":result.is_err(),"error":result.as_ref().err().map(ToString::to_string),"output_bits":bits(&output),"poisoned":operator.is_poisoned(),"pending":operator.pending.is_some()}),
                );
                assert!(result.is_err());
                assert!(!operator.is_poisoned() && operator.pending.is_none());
                assert_eq!(output, vec![42.; output.len()]);
                let zeros = vec![0.; case.input.len()];
                let result = operator.apply(&zeros, &mut output, capacity);
                report(
                    "GEMMA_AFFINE_L1_RECOVERY",
                    json!({"name":case.reference["name"],"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"output_bits":bits(&output),"poisoned":operator.is_poisoned(),"owned_bytes":operator.owned_bytes(),"device":device(&operator)}),
                );
                result.unwrap();
                assert!(output.iter().all(|&v| v == 0.));
                rejected += 1;
                continue;
            }
            let mut baseline = Vec::new();
            for repeat in 0..2 {
                let result = operator.apply(&case.input, &mut output, capacity);
                report(
                    "GEMMA_AFFINE_L1_CASE",
                    json!({"name":case.reference["name"],"repeat":repeat,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"output_bits":bits(&output),"poisoned":operator.is_poisoned(),"owned_bytes":operator.owned_bytes(),"device":device(&operator)}),
                );
                result.unwrap();
                assert!(!operator.is_poisoned());
                assert_eq!(operator.owned_bytes(), sizes.owned);
                if repeat == 0 {
                    baseline = bits(&output);
                } else {
                    assert_eq!(baseline, bits(&output));
                }
                for contract in ["ordered_fp64", "mathematical_fp64"] {
                    for (&value, expected) in output
                        .iter()
                        .zip(case.reference[contract].as_array().unwrap())
                    {
                        let expected = expected.as_f64().unwrap();
                        assert!(
                            value.is_finite()
                                && (f64::from(value) - expected).abs()
                                    <= 1e-5 + 1e-4 * expected.abs(),
                            "{} {contract}: {value} vs {expected}",
                            case.reference["name"]
                        );
                    }
                }
            }
            for token in 0..capacity {
                let row =
                    &case.input[token * case.layout.columns()..(token + 1) * case.layout.columns()];
                let mut part = vec![42.; case.layout.rows()];
                let result = operator.apply(row, &mut part, 1);
                report(
                    "GEMMA_AFFINE_L1_PART",
                    json!({"name":case.reference["name"],"token":token,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"output_bits":bits(&part),"poisoned":operator.is_poisoned(),"owned_bytes":operator.owned_bytes(),"device":device(&operator)}),
                );
                result.unwrap();
                assert_eq!(
                    bits(&part),
                    baseline[token * case.layout.rows()..(token + 1) * case.layout.rows()]
                );
                parts += 1;
            }
            let result = operator.apply(&case.input, &mut output, capacity);
            report(
                "GEMMA_AFFINE_L1_RESUME",
                json!({"name":case.reference["name"],"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"output_bits":bits(&output),"poisoned":operator.is_poisoned(),"owned_bytes":operator.owned_bytes(),"device":device(&operator)}),
            );
            result.unwrap();
            assert_eq!(bits(&output), baseline);
            accepted += 1;
        }
        // A later failing row cannot stage even the earlier valid row. Warm the
        // same owner first so every raw device buffer is initialized for comparison.
        let case = l1_fixture::cases()
            .into_iter()
            .find(|c| c.reference["name"] == "ordered-exact-conflict")
            .unwrap();
        let sizes = Sizes::new(case.layout, 2).unwrap();
        let mut operator = context
            .affine(
                case.layout,
                &case.codes,
                &case.scales,
                &case.biases,
                2,
                sizes.owned,
            )
            .unwrap();
        let zeros = vec![0.; case.layout.columns() * 2];
        let mut output = vec![0.; 2];
        let warm = operator.apply(&zeros, &mut output, 2);
        report(
            "GEMMA_AFFINE_L1_LATE_WARM",
            json!({"completed":warm.is_ok(),"error":warm.as_ref().err().map(ToString::to_string),"output_bits":bits(&output),"device":device(&operator)}),
        );
        warm.unwrap();
        let before = device(&operator);
        let mut input = zeros.clone();
        input[0] = 1e7; // This sparse first row needs and passes the L1 fallback.
        assert!(bounds(64, 1., 1e7).1 > ORDERED_ERROR_LIMIT);
        admit(64, 1., &input[..64]).unwrap();
        input[case.layout.columns()..].copy_from_slice(&case.input);
        output.fill(42.);
        let result = operator.apply(&input, &mut output, 2);
        let after = device(&operator);
        report(
            "GEMMA_AFFINE_L1_LATE_REJECTION",
            json!({"rejected":result.is_err(),"error":result.as_ref().err().map(ToString::to_string),"output_bits":bits(&output),"poisoned":operator.is_poisoned(),"pending":operator.pending.is_some(),"device_before":before,"device_after":after}),
        );
        assert!(result.is_err());
        assert_eq!(output, [42.; 2]);
        assert!(!operator.is_poisoned() && operator.pending.is_none());
        assert_eq!(before, after);
        let result = operator.apply(&zeros, &mut output, 2);
        report(
            "GEMMA_AFFINE_L1_LATE_RECOVERY",
            json!({"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"output_bits":bits(&output),"poisoned":operator.is_poisoned(),"owned_bytes":operator.owned_bytes(),"device":device(&operator)}),
        );
        result.unwrap();
        assert_eq!(output, [0.; 2]);
        report(
            "GEMMA_AFFINE_L1_FINAL",
            json!({"accepted_cases":accepted,"rejected_cases":rejected,"partitions":parts,"late_row_rejected":true,"whole_graph_qualified":false}),
        );
        assert_eq!((accepted, rejected, parts), (15, 4, 16));
    });
}
#[test]
#[ignore = "explicit Apple GPU FP32 packed projection qualification"]
fn real_frozen_projection_reuse_partitions_and_rejections() {
    autoreleasepool(|_| {
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        report(
            "GEMMA_AFFINE_META",
            json!({"device":context.device_name(),"context_allowance_bytes":CONTEXT_ALLOWANCE_BYTES,"simd_width":context.width,"owner_struct_bytes":size_of::<Affine<'_>>(),"graph_qualified":false,"source_parity_qualified":false}),
        );
        let mut cases = 0;
        let mut rejected_constructor = 0;
        let mut rejected_input = 0;
        let mut parts = 0;
        for fixture in fixture::fixtures() {
            for case in &fixture.cases {
                let layout = fixture.layout(case);
                let capacity = case["batch"].as_u64().unwrap() as usize;
                let weights = fixture.bytes(case, "weight");
                let scales = fixture.bytes(case, "scales");
                let biases = fixture.bytes(case, "biases");
                let sizes = Sizes::new(layout, capacity).unwrap();
                assert!(matches!(
                    context.affine(layout, weights, scales, biases, capacity, sizes.owned - 1),
                    Err(Error::Budget { .. })
                ));
                let result = context.affine(layout, weights, scales, biases, capacity, sizes.owned);
                if case["constructor"]["admitted"] != true {
                    report(
                        "GEMMA_AFFINE_CONSTRUCTOR_REJECTION",
                        json!({"name":case["name"],"rejected":result.is_err(),"error":result.as_ref().err().map(ToString::to_string)}),
                    );
                    assert!(result.is_err());
                    rejected_constructor += 1;
                    continue;
                }
                let mut operator = result.unwrap();
                let owned = operator.owned_bytes();
                let input = fixture.input(case);
                let mut output = vec![42.; capacity * layout.rows()];
                if case["input_admission"]["admitted"] != true {
                    let result = operator.apply(&input, &mut output, capacity);
                    report(
                        "GEMMA_AFFINE_INPUT_REJECTION",
                        json!({"name":case["name"],"rejected":result.is_err(),"error":result.as_ref().err().map(ToString::to_string),"output_bits":bits(&output),"poisoned":operator.is_poisoned(),"pending":operator.pending.is_some()}),
                    );
                    assert!(result.is_err());
                    assert!(!operator.is_poisoned());
                    assert!(operator.pending.is_none());
                    assert_eq!(output, vec![42.; output.len()]);
                    let zeros = vec![0.; input.len()];
                    let result = operator.apply(&zeros, &mut output, capacity);
                    report(
                        "GEMMA_AFFINE_REJECTION_RECOVERY",
                        json!({"name":case["name"],"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"output_bits":bits(&output),"poisoned":operator.is_poisoned(),"owned_bytes":owned,"device":device(&operator)}),
                    );
                    result.unwrap();
                    assert!(output.iter().all(|&v| v == 0.));
                    rejected_input += 1;
                    continue;
                }
                let mut baseline = Vec::new();
                for repeat in 0..2 {
                    let result = operator.apply(&input, &mut output, capacity);
                    report(
                        "GEMMA_AFFINE_CASE",
                        json!({"name":case["name"],"reference_contract":case["reference_contract"],"repeat":repeat,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"output_bits":bits(&output),"poisoned":operator.is_poisoned(),"owned_bytes":owned,"device":device(&operator)}),
                    );
                    result.unwrap();
                    assert!(!operator.is_poisoned());
                    assert_eq!(operator.owned_bytes(), owned);
                    if repeat == 0 {
                        baseline = bits(&output);
                    } else {
                        assert_eq!(baseline, bits(&output));
                    }
                    for key in ["math_fp64", "ordered_fp64"] {
                        for (&actual, expected) in output.iter().zip(case[key].as_array().unwrap())
                        {
                            let expected = expected.as_f64().unwrap();
                            assert!(
                                actual.is_finite()
                                    && (f64::from(actual) - expected).abs()
                                        <= 1e-5 + 1e-4 * expected.abs(),
                                "{} {key}: {actual} vs {expected}",
                                case["name"]
                            );
                        }
                    }
                }
                for token in 0..capacity {
                    let slice = &input[token * layout.columns()..(token + 1) * layout.columns()];
                    let mut output = vec![42.; layout.rows()];
                    let result = operator.apply(slice, &mut output, 1);
                    report(
                        "GEMMA_AFFINE_PART",
                        json!({"name":case["name"],"token":token,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"output_bits":bits(&output),"poisoned":operator.is_poisoned(),"owned_bytes":operator.owned_bytes(),"device":device(&operator)}),
                    );
                    result.unwrap();
                    assert_eq!(
                        bits(&output),
                        baseline[token * layout.rows()..(token + 1) * layout.rows()]
                    );
                    assert_eq!(operator.owned_bytes(), owned);
                    parts += 1;
                }
                // Return from smaller batches to the originally admitted capacity.
                let result = operator.apply(&input, &mut output, capacity);
                report(
                    "GEMMA_AFFINE_RESUME",
                    json!({"name":case["name"],"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"output_bits":bits(&output),"poisoned":operator.is_poisoned(),"owned_bytes":operator.owned_bytes(),"device":device(&operator)}),
                );
                result.unwrap();
                assert_eq!(bits(&output), baseline);
                cases += 1;
            }
        }
        let layout = Layout::new(1, 64, 4, 64, MetadataDtype::BF16).unwrap();
        let weight = [0_u8; 32];
        let coefficient = [0_u8; 2];
        let mut operator = context
            .affine(layout, &weight, &coefficient, &coefficient, 1, 4096)
            .unwrap();
        for (label, input, count, tokens) in [
            ("nan", vec![f32::NAN; 64], 1, 1),
            ("input-shape", vec![0.; 63], 1, 1),
            ("output-shape", vec![0.; 64], 2, 1),
            ("zero-tokens", vec![], 1, 0),
            ("above-capacity", vec![0.; 128], 2, 2),
            ("overflow-tokens", vec![], 1, usize::MAX),
        ] {
            let mut output = vec![42.; count];
            let result = operator.apply(&input, &mut output, tokens);
            report(
                "GEMMA_AFFINE_PREFLIGHT",
                json!({"label":label,"rejected":result.is_err(),"output_bits":bits(&output),"poisoned":operator.is_poisoned(),"pending":operator.pending.is_some()}),
            );
            assert!(result.is_err());
            assert!(!operator.is_poisoned());
            assert!(operator.pending.is_none());
            assert_eq!(output, vec![42.; count]);
        }
        let mut output = [42.];
        let result = operator.apply(&[f32::MAX; 64], &mut output, 1);
        report(
            "GEMMA_AFFINE_PREFLIGHT_RECOVERY",
            json!({"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"output_bits":bits(&output),"poisoned":operator.is_poisoned(),"device":device(&operator)}),
        );
        result.unwrap();
        assert_eq!(output, [0.]);
        report(
            "GEMMA_AFFINE_FINAL",
            json!({"passed":true,"cases":cases,"repeats":2,"partitions":parts,"constructor_rejections":rejected_constructor,"input_rejections":rejected_input,"driver_failure_injected":false,"actual_model_qualified":false}),
        );
    });
}
