use super::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
fn reference() -> Value {
    let raw = include_bytes!("../../../tests/fixtures/gemma-metal-rms-v1/math.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "935e1ecac3d8418cd4f668b5200e3e58730c0798391561bebe1b945391e859bb"
    );
    serde_json::from_slice(raw).unwrap()
}
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|x| x.to_bits()).collect()
}
fn floats(row: &Value, key: &str) -> Vec<f32> {
    row[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
#[test]
fn bounded_geometry_epsilon_weights_and_admission_without_gpu() {
    assert!(matches!(
        Context::new(CONTEXT_ALLOWANCE_BYTES - 1),
        Err(Error::Budget { .. })
    ));
    for (rows, width) in [
        (0, 1),
        (257, 1),
        (1, 0),
        (1, 1537),
        (usize::MAX, 1),
        (1, usize::MAX),
    ] {
        assert!(Sizes::new(rows, width, EPSILON, None).is_err());
    }
    for epsilon in [
        0.,
        -1.,
        f32::NAN,
        f32::INFINITY,
        f32::from_bits(EPSILON.to_bits() + 1),
    ] {
        assert!(Sizes::new(1, 1, epsilon, None).is_err());
    }
    for weights in [
        &[][..],
        &[f32::NAN][..],
        &[f32::INFINITY][..],
        &[1., 2.][..],
    ] {
        assert!(Sizes::new(1, 1, EPSILON, Some(weights)).is_err());
    }
    let weights = vec![1.; 1536];
    for learned in [false, true] {
        let s = Sizes::new(256, 1536, EPSILON, learned.then_some(weights.as_slice())).unwrap();
        assert_eq!(s.values, 393216);
        assert_eq!(s.weights, if learned { 1536 } else { 1 });
        assert_eq!(
            s.owned,
            (2 * (393216 + GUARD) + s.weights + GUARD + 1 + GUARD) * 4 + size_of::<Rms<'_>>()
        );
    }
}
// Safe only after success (apply checked completion), or an explicitly completed
// failed command. Preserve raw device data/status/guards BEFORE numerical gates.
fn device(operator: &Rms<'_>) -> Option<Value> {
    if let Some(command) = &operator.pending {
        if command.status() != MTLCommandBufferStatus::Completed {
            return None;
        }
    } else if operator.poisoned {
        return None;
    }
    Some(
        json!({"output_words":operator.output.with_slice(|s|s.to_vec()),
        "input_guard":operator.input.with_slice(|s|s[operator.sizes.values..].to_vec()),
        "weight_words":operator.weights.with_slice(|s|s.to_vec()),
        "status_words":operator.status.with_slice(|s|s.to_vec())}),
    )
}
fn report(label: &str, row: Value) {
    use std::io::Write;
    println!("{label} {row}");
    std::io::stdout().flush().unwrap();
}
#[test]
#[ignore = "explicit Apple GPU FP32 RMS mathematical qualification"]
fn real_frozen_math_guards_partitions_and_failure_state() {
    autoreleasepool(|_| {
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        report(
            "GEMMA_RMS_META",
            json!({"device":context.device_name(),"context_allowance_bytes":CONTEXT_ALLOWANCE_BYTES,"simd_widths":context.widths,"graph_qualified":false,"source_parity_qualified":false}),
        );
        let data = reference();
        let mut partitions = 0;
        let mut baselines = Vec::new();
        for row in data["cases"].as_array().unwrap() {
            let rows = row["rows"].as_u64().unwrap() as usize;
            let width = row["width"].as_u64().unwrap() as usize;
            let input = floats(row, "input");
            let weights = (!row["weights"].is_null()).then(|| floats(row, "weights"));
            let w = weights.as_deref();
            let sizes = Sizes::new(rows, width, EPSILON, w).unwrap();
            assert!(matches!(
                context.rms(rows, width, EPSILON, w, sizes.owned - 1),
                Err(Error::Budget { .. })
            ));
            let mut operator = context.rms(rows, width, EPSILON, w, sizes.owned).unwrap();
            let stored_weights = operator.weights.with_slice(|s| s.to_vec());
            let mut baseline = Vec::new();
            for repeat in 0..2 {
                let mut output = vec![42.; input.len()];
                let result = operator.apply(&input, &mut output);
                report(
                    "GEMMA_RMS_CASE",
                    json!({"name":row["name"],"repeat":repeat,"input_bits":bits(&input),"output_bits":bits(&output),"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"poisoned":operator.is_poisoned(),"owned_bytes":operator.owned_bytes(),"device":device(&operator)}),
                );
                result.unwrap();
                assert!(!operator.is_poisoned());
                assert_eq!(operator.owned_bytes(), sizes.owned);
                if repeat == 0 {
                    baseline = bits(&output);
                } else {
                    assert_eq!(bits(&output), baseline);
                }
                for (&actual, expected) in output.iter().zip(row["math_fp64"].as_array().unwrap()) {
                    let expected = expected.as_f64().unwrap();
                    assert!(
                        actual.is_finite()
                            && (f64::from(actual) - expected).abs() <= 1e-5 + 1e-4 * expected.abs(),
                        "{}: {actual} vs {expected}",
                        row["name"]
                    );
                }
                assert_eq!(operator.weights.with_slice(|s| s.to_vec()), stored_weights);
                assert_eq!(
                    operator.input.with_slice(|s| s[..input.len()].to_vec()),
                    bits(&input)
                );
            }
            // Row independence also covers odd tails and actual width1536.
            for start in (0..rows).step_by(7) {
                let count = (rows - start).min(7);
                let slice = &input[start * width..(start + count) * width];
                let mut part = context.rms(count, width, EPSILON, w, sizes.owned).unwrap();
                let mut output = vec![42.; slice.len()];
                let result = part.apply(slice, &mut output);
                report(
                    "GEMMA_RMS_PART",
                    json!({"name":row["name"],"start":start,"rows":count,"input_bits":bits(slice),"output_bits":bits(&output),"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"poisoned":part.is_poisoned(),"device":device(&part)}),
                );
                result.unwrap();
                assert_eq!(
                    bits(&output),
                    baseline[start * width..(start + count) * width]
                );
                partitions += 1;
            }
            baselines.push((row["name"].as_str().unwrap().to_owned(), baseline));
        }
        let get = |name| &baselines.iter().find(|(n, _)| n == name).unwrap().1;
        for (&a, &b) in get("global-v").iter().zip(get("global-v-ones")) {
            let a = f64::from(f32::from_bits(a));
            let b = f64::from(f32::from_bits(b));
            assert!((a - b).abs() <= 1e-5 + 1e-4 * a.abs());
        }
        let mut operator = context
            .rms(1, 2, EPSILON, Some(&[f32::MAX, 1.]), 4096)
            .unwrap();
        for (label, input, n) in [
            ("nonfinite", vec![1., f32::NAN], 2),
            ("input-shape", vec![1.], 2),
            ("output-shape", vec![1., 2.], 1),
        ] {
            let mut output = vec![42.; n];
            let result = operator.apply(&input, &mut output);
            report(
                "GEMMA_RMS_PREFLIGHT",
                json!({"label":label,"rejected":result.is_err(),"output_bits":bits(&output),"poisoned":operator.is_poisoned()}),
            );
            assert!(result.is_err());
            assert!(!operator.is_poisoned());
            assert_eq!(output, vec![42.; n]);
        }
        let mut output = [42.; 2];
        let reused = operator.apply(&[0., 0.], &mut output);
        report(
            "GEMMA_RMS_REUSE",
            json!({"completed":reused.is_ok(),"error":reused.as_ref().err().map(ToString::to_string),"output_bits":bits(&output),"poisoned":operator.is_poisoned(),"device":device(&operator)}),
        );
        reused.unwrap();
        assert_eq!(output, [0.; 2]);
        output.fill(42.);
        let result = operator.apply(&[1., 0.], &mut output);
        report(
            "GEMMA_RMS_FAILURE",
            json!({"rejected":result.is_err(),"error":result.as_ref().err().map(ToString::to_string),"output_bits":bits(&output),"poisoned":operator.is_poisoned(),"device":device(&operator)}),
        );
        assert!(matches!(result, Err(Error::Nonfinite)));
        assert_eq!(output, [42.; 2]);
        assert!(operator.is_poisoned());
        assert_eq!(operator.status.with_slice(|s| s[0]), 1);
        assert_eq!(
            operator.pending.as_ref().unwrap().status(),
            MTLCommandBufferStatus::Completed
        );
        let retry = operator.apply(&[0., 0.], &mut output);
        report(
            "GEMMA_RMS_RETRY",
            json!({"rejected":retry.is_err(),"output_bits":bits(&output),"poisoned":operator.is_poisoned()}),
        );
        assert!(retry.is_err());
        assert_eq!(output, [42.; 2]);
        let mut fresh = context
            .rms(1, 2, EPSILON, Some(&[f32::MAX, 1.]), 4096)
            .unwrap();
        let result = fresh.apply(&[0., 0.], &mut output);
        report(
            "GEMMA_RMS_RECREATE",
            json!({"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"output_bits":bits(&output),"poisoned":fresh.is_poisoned(),"device":device(&fresh)}),
        );
        result.unwrap();
        assert_eq!(output, [0.; 2]);
        assert!(!fresh.is_poisoned());
        report(
            "GEMMA_RMS_FINAL",
            json!({"passed":true,"cases":16,"repeats":2,"partitions":partitions,"weighted_ones_compared":true,"overflow_poison_and_recreate":true,"driver_failure_injected":false}),
        );
    });
}
