use super::*;
use half::f16;
use objc2::rc::autoreleasepool;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn fixture() -> Value {
    let raw = include_bytes!("../../../tests/fixtures/llama3-metal-rotary-v1/fixture.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "29c138081b7681ecab8fbc4f5353862e85cc5209754aefc86619e7d0e2b3c809"
    );
    let result: Value = serde_json::from_slice(raw).unwrap();
    assert_eq!(
        result["producer_sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../../../tools/reference/llama3_metal_rotary.py"
            ))
        )
    );
    assert_eq!(
        result["positional_producer_sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../../../tools/reference/llama3_rotary.py"))
        )
    );
    result
}
fn scaling(row: &Value) -> Llama3Scaling {
    Llama3Scaling::new(
        row["factor"].as_f64().unwrap() as f32,
        row["low"].as_f64().unwrap() as f32,
        row["high"].as_f64().unwrap() as f32,
        row["original"].as_u64().unwrap() as usize,
    )
    .unwrap()
}

#[test]
fn allocation_frequency_and_uint_bounds_are_cold_checked() {
    let s = Llama3Scaling::new(32., 1., 4., 8192).unwrap();
    assert_eq!(owned_bytes(64).unwrap(), 39 * 4 + size_of::<Operation>());
    for dimension in [0, 1, 3, usize::MAX] {
        assert!(owned_bytes(dimension).is_err());
    }
    for args in [
        (0, 1, 64, 0),
        (1, 0, 64, 0),
        (1, 1, 0, 0),
        (1, 1, 3, 0),
        (1, 1, 64, 512),
        (2, 1, 64, 511),
        (2, 1, 64, usize::MAX),
        (1, usize::MAX, 64, 0),
        (1, 2, u32::MAX as usize - 1, 0),
    ] {
        assert!(extent(args.0, args.1, args.2, args.3).is_err(), "{args:?}");
    }
    assert_eq!(extent(3, 32, 64, 509).unwrap(), 6144);
    assert!(frequency(f32::NAN, 64, 1, s).is_err());
    assert!(frequency(f32::INFINITY, 64, 1, s).is_err());
    assert!(
        frequency(
            1e38,
            128,
            63,
            Llama3Scaling::new(f32::MAX, 1., 4., 8192).unwrap()
        )
        .is_err()
    );
    for row in fixture()["cases"].as_array().unwrap() {
        let d = row["head_dim"].as_u64().unwrap() as usize;
        for (pair, expected) in row["inverse_bits"].as_array().unwrap().iter().enumerate() {
            assert_eq!(
                frequency(row["theta"].as_f64().unwrap() as f32, d, pair, scaling(row)).unwrap()
                    as u64,
                expected.as_u64().unwrap()
            );
        }
    }
}

fn run(
    gpu: &Gpu,
    op: &Operation,
    input: &[f32],
    rows: usize,
    heads: usize,
    offset: usize,
) -> Vec<u16> {
    let n = input.len();
    assert_eq!(extent(rows, heads, op.dimension, offset).unwrap(), n);
    let mut x = Buffer::<u16>::new(gpu, n + GUARD).unwrap();
    x.with_mut(|out| {
        out.fill(0x5a17);
        for (out, &v) in out.iter_mut().zip(input) {
            let value = f16::from_f32(v);
            assert_eq!(value.to_f32(), v);
            *out = value.to_bits();
        }
    });
    let mut output = Buffer::<u16>::new(gpu, n + GUARD).unwrap();
    output.with_mut(|out| out.fill(0x5a17));
    let mut status = Buffer::<u32>::new(gpu, 1 + GUARD).unwrap();
    status.with_mut(|out| {
        out.fill(SENTINEL);
        out[0] = 0;
    });
    let input_before = x.with_slice(|v| v.to_vec());
    let frequency_before = op.inverse.with_slice(|v| v.to_vec());
    let raw = gpu.command().unwrap();
    op.encode(
        &Command {
            raw: &raw,
            trace: None,
        },
        Invocation {
            input: &x,
            output: &output,
            status: &status,
            rows,
            heads,
            offset,
        },
    )
    .unwrap();
    raw.commit();
    raw.waitUntilCompleted();
    println!(
        "LLAMA3_ROTARY_COMPLETION {}",
        json!({"rows":rows,"heads":heads,"dimension":op.dimension,"offset":offset,"completed":raw.status()==MTLCommandBufferStatus::Completed})
    );
    assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
    let actual = output.with_slice(|v| v.to_vec());
    let flags = status.with_slice(|v| v.to_vec());
    println!(
        "LLAMA3_ROTARY_RAW {}",
        json!({"bits":actual,"status":flags})
    );
    assert!(actual[n..].iter().all(|&v| v == 0x5a17));
    assert!(flags[1..].iter().all(|&v| v == SENTINEL));
    assert_eq!(x.with_slice(|v| v.to_vec()), input_before);
    assert_eq!(op.inverse.with_slice(|v| v.to_vec()), frequency_before);
    assert!(
        frequency_before[op.dimension / 2..]
            .iter()
            .all(|&v| v == SENTINEL)
    );
    assert_eq!(flags[0], 0);
    actual[..n].to_vec()
}

#[test]
#[ignore = "explicit Apple GPU isolated Llama3 Half rotary dual-gate qualification"]
fn isolated_half_rotary_matches_prospective_dual_gates_and_guards() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        println!(
            "LLAMA3_ROTARY_META {}",
            json!({"device":gpu.device.name().to_string(),"positions":"0..511","model_qualified":false,"policy_sha256":fixture()["astra_policy_sha256"]})
        );
        for row in fixture()["cases"].as_array().unwrap() {
            let n = |key: &str| row[key].as_u64().unwrap() as usize;
            let (rows, heads, d, offset) = (n("tokens"), n("heads"), n("head_dim"), n("offset"));
            let required = owned_bytes(d).unwrap();
            assert!(matches!(
                Operation::new(
                    &gpu,
                    d,
                    row["theta"].as_f64().unwrap() as f32,
                    scaling(row),
                    required - 1
                ),
                Err(Error::Budget { .. })
            ));
            let op = Operation::new(
                &gpu,
                d,
                row["theta"].as_f64().unwrap() as f32,
                scaling(row),
                required,
            )
            .unwrap();
            assert_eq!(op.inverse.raw.length() + size_of::<Operation>(), required);
            let input: Vec<f32> = row["input"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let mut first = Vec::new();
            for repeat in 0..2 {
                let actual = run(&gpu, &op, &input, rows, heads, offset);
                if repeat == 0 {
                    first = actual.clone();
                } else {
                    assert_eq!(actual, first);
                }
                let (mut math_error, mut profile_error, mut math_fraction, mut profile_fraction) =
                    (0f64, 0f64, 0f64, 0f64);
                for (i, &bits) in actual.iter().enumerate() {
                    let value = f64::from(f16::from_bits(bits).to_f32());
                    let m = row["mathematical"][i].as_f64().unwrap();
                    let p = f64::from(
                        f16::from_bits(row["profile_half_bits"][i].as_u64().unwrap() as u16)
                            .to_f32(),
                    );
                    let mb = row["mathematical_bound"][i].as_f64().unwrap();
                    let pb = row["profile_bound"][i].as_f64().unwrap();
                    let (me, pe) = ((value - m).abs(), (value - p).abs());
                    math_error = math_error.max(me);
                    profile_error = profile_error.max(pe);
                    math_fraction = math_fraction.max(me / mb);
                    profile_fraction = profile_fraction.max(pe / pb);
                    assert!(
                        value.is_finite() && me <= mb && pe <= pb,
                        "{}/i{i}: {value}, math{m} bound{mb}, profile{p} bound{pb}",
                        row["name"]
                    );
                    if offset == 0 && i < heads * d && input[i] != 0. {
                        assert_eq!(bits, f16::from_f32(input[i]).to_bits());
                    }
                }
                println!(
                    "LLAMA3_ROTARY_CASE {}",
                    json!({"name":row["name"],"repeat":repeat,"values":actual.len(),"max_math_error":math_error,"max_profile_error":profile_error,"max_math_fraction":math_fraction,"max_profile_fraction":profile_fraction})
                );
            }
            for chunk in [1, 2] {
                let mut joined = Vec::new();
                for token in (0..rows).step_by(chunk) {
                    let count = chunk.min(rows - token);
                    joined.extend(run(
                        &gpu,
                        &op,
                        &input[token * heads * d..(token + count) * heads * d],
                        count,
                        heads,
                        offset + token,
                    ));
                }
                assert_eq!(joined, first);
            }
        }
        // A valid Half input can overflow the output representation; status
        // must reject publication even when the FP32 intermediate is finite.
        let op = Operation::new(
            &gpu,
            2,
            500000.,
            Llama3Scaling::new(32., 1., 4., 8192).unwrap(),
            owned_bytes(2).unwrap(),
        )
        .unwrap();
        let mut input = Buffer::<u16>::new(&gpu, 2 + GUARD).unwrap();
        input.with_mut(|s| {
            s.fill(0x5a17);
            s[0] = f16::MAX.to_bits();
            s[1] = (-f16::MAX).to_bits();
        });
        let mut output = Buffer::<u16>::new(&gpu, 2 + GUARD).unwrap();
        output.with_mut(|s| s.fill(0x5a17));
        let mut status = Buffer::<u32>::new(&gpu, 1 + GUARD).unwrap();
        status.with_mut(|s| {
            s.fill(SENTINEL);
            s[0] = 0;
        });
        let input_before = input.with_slice(|v| v.to_vec());
        let frequencies_before = op.inverse.with_slice(|v| v.to_vec());
        let raw = gpu.command().unwrap();
        assert!(
            op.encode(
                &Command {
                    raw: &raw,
                    trace: None
                },
                Invocation {
                    input: &input,
                    output: &output,
                    status: &status,
                    rows: 2,
                    heads: 1,
                    offset: 511
                }
            )
            .is_err()
        );
        assert!(
            op.encode(
                &Command {
                    raw: &raw,
                    trace: None
                },
                Invocation {
                    input: &input,
                    output: &input,
                    status: &status,
                    rows: 1,
                    heads: 1,
                    offset: 1
                }
            )
            .is_err()
        );
        op.encode(
            &Command {
                raw: &raw,
                trace: None,
            },
            Invocation {
                input: &input,
                output: &output,
                status: &status,
                rows: 1,
                heads: 1,
                offset: 1,
            },
        )
        .unwrap();
        raw.commit();
        raw.waitUntilCompleted();
        assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
        println!(
            "LLAMA3_ROTARY_OVERFLOW {}",
            json!({"output":output.with_slice(|v|v.to_vec()),"status":status.with_slice(|v|v.to_vec())})
        );
        assert_ne!(status.with_slice(|v| v[0]), 0);
        assert!(output.with_slice(|s| s[2..].iter().all(|&v| v == 0x5a17)));
        assert!(status.with_slice(|s| s[1..].iter().all(|&v| v == SENTINEL)));
        assert_eq!(input.with_slice(|v| v.to_vec()), input_before);
        assert_eq!(op.inverse.with_slice(|v| v.to_vec()), frequencies_before);
        run(&gpu, &op, &[1., 1.], 1, 1, 1);
    });
}
