use super::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn shape() -> ProportionalRotaryShape {
    ProportionalRotaryShape {
        tokens: 1,
        heads: 1,
        head_dim: 2,
        rotated_dims: 2,
        offset: 1,
        base: 1.,
        factor: 1.,
    }
}
fn frozen(raw: &[u8], expected: &str) -> Value {
    assert_eq!(format!("{:x}", Sha256::digest(raw)), expected);
    serde_json::from_slice(raw).unwrap()
}
fn math_cases() -> Vec<Value> {
    let receipt =
        include_bytes!("../../../benchmarks/results/2026-10-24-gemma-rotary-math-v1.json");
    let reference = frozen(
        receipt,
        "a16a944f4ba20956f8c8a19796afb78727bb15aa38b78b0d29a4b25ee8beb63d",
    );
    let original = frozen(
        include_bytes!("../../../tests/fixtures/gemma-rotary-v1/math.json"),
        reference["fixture_sha256"].as_str().unwrap(),
    );
    let mut cases: Vec<_> = original["cases"].as_array().unwrap().iter()
        .filter(|r| r["tokens"].as_u64().unwrap()!=0)
        .map(|r| json!({"name":r["name"],"tokens":r["tokens"],"heads":r["heads"],"dims":r["dims"],"rotated_dims":r["rotated_dims"],"offset":r["offset"],"base":r["base"],"factor":r["factor"],"input":r["arrays"]["input"]["values"],"expected_math_fp64":r["arrays"]["output"]["values"]})).collect();
    let extra = frozen(
        include_bytes!("../../../tests/fixtures/gemma-metal-rotary-v1/math.json"),
        "9969cb2211ec664acb92faaae2055341025a04a0e45cc39f4ff48b41c82ecd3a",
    );
    cases.extend(extra["cases"].as_array().unwrap().iter().cloned());
    cases
}
#[test]
fn bounded_geometry_and_owned_capacity_reject_without_gpu() {
    let s = shape();
    let sizes = Sizes::new(&s).unwrap();
    assert_eq!(sizes.values, 2);
    assert_eq!(sizes.coefficients, 2);
    assert_eq!(
        sizes.owned,
        (2 * (2 + GUARD) + 2 + GUARD + 1 + GUARD) * 4 + size_of::<Rotary<'_>>()
    );
    assert!(matches!(
        Context::new(CONTEXT_ALLOWANCE_BYTES - 1),
        Err(Error::Budget { .. })
    ));
    for key in 0..13 {
        let mut invalid = s;
        match key {
            0 => invalid.tokens = 0,
            1 => invalid.tokens = 9,
            2 => invalid.heads = 0,
            3 => invalid.heads = 9,
            4 => invalid.head_dim = 513,
            5 => invalid.head_dim = 3,
            6 => invalid.rotated_dims = 3,
            7 => invalid.offset = usize::MAX,
            8 => invalid.offset = 131072,
            9 => invalid.base = f32::NAN,
            10 => invalid.base = 0.,
            11 => invalid.factor = f32::INFINITY,
            _ => invalid.factor = 0.0625,
        }
        assert!(Sizes::new(&invalid).is_err(), "{key}");
    }
    for row in math_cases() {
        let s = row_shape(&row);
        let sizes = Sizes::new(&s).unwrap();
        assert_eq!(sizes.values, row["input"].as_array().unwrap().len());
    }
}
fn row_shape(r: &Value) -> ProportionalRotaryShape {
    let n = |key: &str| r[key].as_u64().unwrap() as usize;
    ProportionalRotaryShape {
        tokens: n("tokens"),
        heads: n("heads"),
        head_dim: n("dims"),
        rotated_dims: n("rotated_dims"),
        offset: n("offset"),
        base: r["base"].as_f64().unwrap() as f32,
        factor: r["factor"].as_f64().unwrap() as f32,
    }
}
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|x| x.to_bits()).collect()
}

// Failed driver/unfinished commands never permit shared CPU reads. Completed
// numerical/guard failures retain the actual device result before assertions.
fn completed_device(operator: &Rotary<'_>) -> Option<Value> {
    let command = operator.pending.as_ref()?;
    if command.status() != MTLCommandBufferStatus::Completed {
        return None;
    }
    Some(json!({"completed":true,
        "output_bits":operator.output.with_slice(|s|s.to_vec()),
        "status_words":operator.status.with_slice(|s|s.to_vec()),
        "input_guard":operator.input.with_slice(|s|s[operator.sizes.values..].to_vec()),
        "coefficient_guard":operator.coefficients.with_slice(|s|s[operator.sizes.coefficients..].to_vec())}))
}

#[test]
#[ignore = "explicit Apple GPU proportional rotary mathematical qualification"]
fn real_frozen_math_rounding_guards_and_failure_state() {
    autoreleasepool(|_| {
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        println!(
            "GEMMA_ROTARY_META {}",
            json!({"device":context.device_name(),"context_allowance_bytes":CONTEXT_ALLOWANCE_BYTES,"graph_qualified":false,"upstream_parity_qualified":false})
        );
        for row in math_cases() {
            let s = row_shape(&row);
            let input: Vec<f32> = row["input"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let expected: Vec<f64> = row["expected_math_fp64"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let sizes = Sizes::new(&s).unwrap();
            assert!(matches!(
                context.rotary(s, sizes.owned - 1),
                Err(Error::Budget { .. })
            ));
            let mut operator = context.rotary(s, sizes.owned).unwrap();
            let coefficients = operator.coefficients.with_slice(|v| v.to_vec());
            let owned = operator.owned_bytes();
            let mut baseline = Vec::new();
            for repeat in 0..2 {
                let mut output = vec![f32::from_bits(SENTINEL); input.len()];
                let result = operator.apply(&input, &mut output);
                // Retain complete actual output/status BEFORE numerical assertions.
                println!(
                    "GEMMA_ROTARY_CASE {}",
                    json!({"name":row["name"],"repeat":repeat,"input_bits":bits(&input),"output_bits":bits(&output),"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"poisoned":operator.is_poisoned(),"owned_bytes":operator.owned_bytes(),"completed_device_failure":completed_device(&operator)})
                );
                result.unwrap();
                assert!(!operator.is_poisoned());
                assert_eq!(operator.owned_bytes(), owned);
                if repeat == 0 {
                    baseline = bits(&output);
                } else {
                    assert_eq!(bits(&output), baseline);
                }

                for (&a, &e) in output.iter().zip(&expected) {
                    assert!(
                        a.is_finite() && (f64::from(a) - e).abs() <= 1e-5 + 1e-4 * e.abs(),
                        "{}: {a} vs {e}",
                        row["name"]
                    );
                }
                for base in (0..input.len()).step_by(s.head_dim) {
                    for pair in s.rotated_dims / 2..s.head_dim / 2 {
                        for i in [base + pair, base + pair + s.head_dim / 2] {
                            assert_eq!(output[i].to_bits(), input[i].to_bits());
                        }
                    }
                }
                assert_eq!(
                    operator.coefficients.with_slice(|v| v.to_vec()),
                    coefficients
                );
                assert_eq!(
                    operator.input.with_slice(|v| v[..input.len()].to_vec()),
                    bits(&input)
                );
            }
            if s.heads == 8 {
                let stride = s.heads * s.head_dim;
                for token in 0..s.tokens {
                    let mut part_shape = s;
                    part_shape.tokens = 1;
                    part_shape.offset += token;
                    let mut part = context.rotary(part_shape, sizes.owned).unwrap();
                    let slice = &input[token * stride..(token + 1) * stride];
                    let mut output = vec![f32::from_bits(SENTINEL); stride];
                    let result = part.apply(slice, &mut output);
                    println!(
                        "GEMMA_ROTARY_PART {}",
                        json!({"name":row["name"],"token":token,"input_bits":bits(slice),"output_bits":bits(&output),"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"poisoned":part.is_poisoned(),"completed_device_failure":completed_device(&part)})
                    );
                    result.unwrap();
                    assert_eq!(
                        bits(&output),
                        baseline[token * stride..(token + 1) * stride]
                    );
                }
            }
        }
        let rounding = frozen(
            include_bytes!("../../../tests/fixtures/gemma-metal-rotary-v1/rounding.json"),
            "314f461b34a80f534ec45e67d57c568eb12ae72c7dba28a026b5b4ed9b247448",
        );
        let input: Vec<f32> = rounding["input_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| f32::from_bits(v.as_u64().unwrap() as u32))
            .collect();
        let mut operator = context.rotary(shape(), 4096).unwrap();
        let mut output = [42.; 2];
        let result = operator.apply(&input, &mut output);
        println!(
            "GEMMA_ROTARY_ROUNDING {}",
            json!({"output_bits":bits(&output),"completed":result.is_ok(),"poisoned":operator.is_poisoned(),"completed_device_failure":completed_device(&operator)})
        );
        result.unwrap();
        let expected: Vec<u32> = rounding["noncontract_output_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(bits(&output), expected);
        // Both fields describe alternative contractions of FIRST coordinate.
        for key in [
            "contracting_first_output_bits",
            "contracting_second_output_bits",
        ] {
            assert_ne!(output[0].to_bits(), rounding[key].as_u64().unwrap() as u32);
        }
        let mut unchanged = [42.; 2];
        for input in [&[f32::NAN, 1.][..], &[1.][..]] {
            assert!(operator.apply(input, &mut unchanged).is_err());
            assert!(!operator.is_poisoned());
            assert_eq!(unchanged, [42.; 2]);
        }
        let mut short = [42.; 1];
        let rejection = operator.apply(&[1., 2.], &mut short);
        println!(
            "GEMMA_ROTARY_OUTPUT_BOUND {}",
            json!({"rejected":rejection.is_err(),"output_bits":bits(&short),"poisoned":operator.is_poisoned()})
        );
        assert!(rejection.is_err());
        assert!(!operator.is_poisoned());
        assert_eq!(short, [42.; 1]);
        operator.apply(&[1., 2.], &mut unchanged).unwrap();
        let mut untouched = [42.; 2];
        let failure = operator.apply(&[f32::MAX, f32::MAX], &mut untouched);
        println!(
            "GEMMA_ROTARY_FAILURE {}",
            json!({"rejected":failure.is_err(),"error":failure.as_ref().err().map(ToString::to_string),"output_bits":bits(&untouched),"poisoned":operator.is_poisoned(),"completed_device_failure":completed_device(&operator)})
        );
        assert!(matches!(failure, Err(Error::Nonfinite)));
        assert_eq!(
            operator.pending.as_ref().unwrap().status(),
            MTLCommandBufferStatus::Completed
        );
        assert_ne!(operator.status.with_slice(|s| s[0]) & 1, 0);
        assert!(operator.is_poisoned());
        assert_eq!(untouched, [42.; 2]);
        assert!(operator.apply(&[1., 2.], &mut untouched).is_err());
        assert_eq!(untouched, [42.; 2]);
        let mut fresh = context.rotary(shape(), 4096).unwrap();
        let recreated = fresh.apply(&[1., 2.], &mut untouched);
        println!(
            "GEMMA_ROTARY_RECREATE {}",
            json!({"completed":recreated.is_ok(),"error":recreated.as_ref().err().map(ToString::to_string),"output_bits":bits(&untouched),"poisoned":fresh.is_poisoned(),"completed_device_failure":completed_device(&fresh)})
        );
        recreated.unwrap();
        let c = f32::from_bits(rounding["cosine_bits"].as_u64().unwrap() as u32);
        let s = f32::from_bits(rounding["sine_bits"].as_u64().unwrap() as u32);
        assert_eq!(bits(&untouched), bits(&[c - 2. * s, 2. * c + s]));
        // Zero rotation copies every bit even when FP32 arithmetic would flush.
        let mut zero = shape();
        zero.rotated_dims = 0;
        let mut copy = context.rotary(zero, 4096).unwrap();
        let signed = [-0.0, f32::from_bits(1)];
        let mut out = [1.; 2];
        let copied = copy.apply(&signed, &mut out);
        println!(
            "GEMMA_ROTARY_COPY {}",
            json!({"completed":copied.is_ok(),"error":copied.as_ref().err().map(ToString::to_string),"input_bits":bits(&signed),"output_bits":bits(&out),"poisoned":copy.is_poisoned(),"completed_device_failure":completed_device(&copy)})
        );
        copied.unwrap();
        assert_eq!(bits(&signed), bits(&out));
        println!(
            "GEMMA_ROTARY_FINAL {}",
            json!({"passed":true,"cases":11,"repeats":2,"partitions":12,"negative_zero_and_subnormal_inactive_copy":true,"noncontraction_checked":true,"overflow_poison_and_recreate":true,"driver_failure_injected":false})
        );
    });
}
