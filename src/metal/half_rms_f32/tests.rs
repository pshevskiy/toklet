use super::*;
use crate::model::gemma_reference::tests::Measurement;
use half::{bf16, f16};
use objc2::rc::autoreleasepool;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const RAW: &[u8] = include_bytes!("../../../tests/fixtures/gemma3-half-rms-v1/arrays.bin");
fn fixture() -> Value {
    let raw = include_bytes!("../../../tests/fixtures/gemma3-half-rms-v1/manifest.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "aaca461bcbca3176132a5b030d3cee5cc0ff2d382de26b8695af81b4ede49b8a"
    );
    let m: Value = serde_json::from_slice(raw).unwrap();
    assert_eq!(m["format"], "toklet-gemma3-half-rms-v1");
    assert_eq!(
        m["protocol_sha256"],
        "01d5b70be1a408b03dd849bd7b166a108b25489a55300bd779fa0e30d855e845"
    );
    assert_eq!(m["source_reference_executed"], true);
    assert_eq!(m["target_gpu_executed"], false);
    assert_eq!(m["finite_output_values"], 54272);
    assert_eq!(m["cases"].as_array().unwrap().len(), 6);
    assert_eq!(
        m["producer"]["sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../../../tools/reference/gemma3_half_rms.py"
            ))
        )
    );
    assert_eq!(
        m["files"]["arrays.bin"]["sha256"],
        format!("{:x}", Sha256::digest(RAW))
    );
    let errors = include_bytes!("../../../tests/fixtures/gemma3-half-rms-v1/errors.json");
    assert_eq!(
        m["files"]["errors.json"]["sha256"],
        format!("{:x}", Sha256::digest(errors))
    );
    assert_eq!(m["policy"]["fatal_reference"], "main");
    assert_eq!(m["policy"]["atol_floor"].as_f64(), Some(1e-4));
    assert_eq!(m["policy"]["rtol"].as_f64(), Some(1e-3));
    m
}
fn bytes(d: &Value) -> &'static [u8] {
    assert_eq!(d["file"], "arrays.bin");
    let start = d["offset"].as_u64().unwrap() as usize;
    let len = d["bytes"].as_u64().unwrap() as usize;
    let raw = &RAW[start..start + len];
    assert_eq!(d["sha256"], format!("{:x}", Sha256::digest(raw)));
    raw
}
fn words16(d: &Value) -> Vec<u16> {
    assert_eq!(d["dtype"], "u16le");
    bytes(d)
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn words32(d: &Value) -> Vec<u32> {
    assert_eq!(d["dtype"], "u32le");
    bytes(d)
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn hash16(v: &[u16]) -> String {
    let mut h = Sha256::new();
    for w in v {
        h.update(w.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn hash32(v: &[u32]) -> String {
    let mut h = Sha256::new();
    for w in v {
        h.update(w.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

#[test]
fn checked_shape_bytes_and_index_boundaries() {
    let p = Plan::new(1, 32, 35, 37, 1e-6).unwrap();
    assert_eq!(
        (p.input_len(), p.output_len(), p.scale_len(), p.status_len()),
        (42, 44, 39, 8)
    );
    assert_eq!(p.operand_bytes().unwrap(), 360);
    for width in [32, 64, 256, 640] {
        assert!(Plan::new(128, width, width + 3, width + 5, 1e-6).is_ok());
    }
    for (rows, width, input, output, eps) in [
        (0, 32, 32, 32, 1e-6),
        (129, 32, 32, 32, 1e-6),
        (1, 0, 32, 32, 1e-6),
        (1, 33, 33, 33, 1e-6),
        (1, 32, 31, 32, 1e-6),
        (1, 32, 32, 31, 1e-6),
        (1, 32, 32, 32, 1e-5),
        (1, 32, 32, 32, f32::NAN),
        (2, 32, usize::MAX, 32, 1e-6),
        (128, 32, (u32::MAX as usize / 128) + 1, 32, 1e-6),
    ] {
        assert!(Plan::new(rows, width, input, output, eps).is_err());
    }
}
#[test]
fn frozen_policy_shapes_words_and_exact_scale_counterexample() {
    let m = fixture();
    let mut values = 0;
    for c in m["cases"].as_array().unwrap() {
        let rows = c["rows"].as_u64().unwrap() as usize;
        let width = c["width"].as_u64().unwrap() as usize;
        let plan = Plan::new(
            rows,
            width,
            c["input_stride"].as_u64().unwrap() as usize,
            c["output_stride"].as_u64().unwrap() as usize,
            f32::from_bits(m["epsilon_bits"].as_u64().unwrap() as u32),
        )
        .unwrap();
        assert_eq!(words16(&c["arrays"]["input"]).len(), plan.input_len());
        assert_eq!(words32(&c["arrays"]["scales"]).len(), plan.scale_len());
        let a = words16(&c["arrays"]["main"]);
        let b = words16(&c["arrays"]["wide"]);
        assert_eq!(a.len(), rows * width);
        assert_eq!(a.len(), b.len());
        let envelope = a
            .iter()
            .zip(&b)
            .map(|(&a, &b)| {
                (f64::from(f16::from_bits(a).to_f32()) - f64::from(f16::from_bits(b).to_f32()))
                    .abs()
            })
            .fold(0.0, f64::max);
        assert_eq!(c["envelope"].as_f64(), Some(envelope));
        assert_eq!(c["gate"]["atol"].as_f64(), Some((2.0 * envelope).max(1e-4)));
        assert_eq!(c["gate"]["rtol"].as_f64(), Some(1e-3));
        values += a.len();
        for exact in c["exact"].as_array().unwrap() {
            let i = exact["row"].as_u64().unwrap() as usize * width
                + exact["column"].as_u64().unwrap() as usize;
            assert_eq!(a[i], exact["expected_word"].as_u64().unwrap() as u16);
            assert_eq!(b[i], a[i]);
            assert_ne!(
                a[i],
                exact["wrong_half_scale_word"].as_u64().unwrap() as u16
            );
        }
    }
    assert_eq!(values, 54272);
}

struct Operands {
    plan: Plan,
    input: Buffer<u16>,
    scales: Buffer<u32>,
    output: Buffer<u16>,
    status: Buffer<u32>,
    input_hash: String,
    scale_hash: String,
}
impl Operands {
    fn new(gpu: &Gpu, plan: Plan, input: &[u16], scales: &[u32]) -> Self {
        assert_eq!(input.len(), plan.input_len());
        assert_eq!(scales.len(), plan.scale_len());
        let mut x = Buffer::new(gpu, input.len()).unwrap();
        x.with_mut(|v| v.copy_from_slice(input));
        let mut w = Buffer::new(gpu, scales.len()).unwrap();
        w.with_mut(|v| v.copy_from_slice(scales));
        let mut y = Buffer::new(gpu, plan.output_len()).unwrap();
        y.with_mut(|v| v.fill(0x5a17));
        let mut status = Buffer::new(gpu, plan.status_len()).unwrap();
        status.with_mut(|v| {
            v.fill(0x5a170031);
            v[0] = 0;
        });
        assert_eq!(
            x.raw.length() + w.raw.length() + y.raw.length() + status.raw.length(),
            plan.operand_bytes().unwrap()
        );
        Self {
            plan,
            input: x,
            scales: w,
            output: y,
            status,
            input_hash: hash16(input),
            scale_hash: hash32(scales),
        }
    }
    fn invocation(&self) -> Invocation<'_> {
        Invocation {
            input: &self.input,
            scales: &self.scales,
            output: &self.output,
            status: &self.status,
        }
    }
    fn record(&self, label: &str, completed: bool) -> (Vec<u16>, Vec<u32>) {
        let output = self.output.with_slice(|v| v.to_vec());
        let status = self.status.with_slice(|v| v.to_vec());
        let x = self.input.with_slice(hash16);
        let w = self.scales.with_slice(hash32);
        println!(
            "HALF_RMS_RAW {}",
            json!({"label":label,"completed":completed,"rows":self.plan.rows,"width":self.plan.width,
            "input_stride":self.plan.input_stride,"output_stride":self.plan.output_stride,
            "input_before_sha256":self.input_hash,"input_after_sha256":x,"scales_before_sha256":self.scale_hash,"scales_after_sha256":w,
            "output_words":output,"status_words":status,"operand_bytes":self.plan.operand_bytes().unwrap()})
        );
        assert_eq!(x, self.input_hash);
        assert_eq!(w, self.scale_hash);
        for (i, &word) in output.iter().enumerate() {
            if i >= self.plan.rows * self.plan.output_stride
                || i % self.plan.output_stride >= self.plan.width
            {
                assert_eq!(word, 0x5a17, "output padding/guard {i}");
            }
        }
        assert!(status[1..].iter().all(|&w| w == 0x5a170031));
        (output, status)
    }
}
fn run(gpu: &Gpu, op: &Operation, io: &Operands, label: &str) -> (Vec<u16>, Vec<u32>) {
    let raw = gpu.command().unwrap();
    let command = Command {
        raw: &raw,
        trace: None,
    };
    op.encode(&command, io.plan, io.invocation()).unwrap();
    raw.commit();
    raw.waitUntilCompleted();
    let completed = raw.status() == MTLCommandBufferStatus::Completed;
    let result = io.record(label, completed);
    assert!(completed);
    result
}
fn compare(c: &Value, p: Plan, out: &[u16]) -> (usize, f64, f64, usize) {
    let main = words16(&c["arrays"]["main"]);
    let wide = words16(&c["arrays"]["wide"]);
    let mut maximum = 0.0f64;
    let mut fraction = 0.0f64;
    let mut wide_exceedances = 0;
    let atol = c["gate"]["atol"].as_f64().unwrap();
    let rtol = c["gate"]["rtol"].as_f64().unwrap();
    for row in 0..p.rows {
        for column in 0..p.width {
            let index = row * p.width + column;
            let actual = f16::from_bits(out[row * p.output_stride + column]).to_f32();
            let expected = f16::from_bits(main[index]).to_f32();
            assert!(actual.is_finite());
            let error = (f64::from(actual) - f64::from(expected)).abs();
            let limit = atol + rtol * f64::from(expected).abs();
            maximum = maximum.max(error);
            fraction = fraction.max(error / limit);
            assert!(
                error <= limit,
                "{}[{row},{column}] actual={actual} bits={} expected={expected} bits={} error={error} limit={limit}",
                c["name"],
                out[row * p.output_stride + column],
                main[index]
            );
            let w = f64::from(f16::from_bits(wide[index]).to_f32());
            wide_exceedances += usize::from((f64::from(actual) - w).abs() > atol + rtol * w.abs());
        }
    }
    for exact in c["exact"].as_array().unwrap() {
        let i = exact["row"].as_u64().unwrap() as usize * p.output_stride
            + exact["column"].as_u64().unwrap() as usize;
        assert_eq!(out[i], exact["expected_word"].as_u64().unwrap() as u16);
    }
    (main.len(), maximum, fraction, wide_exceedances)
}

#[test]
#[ignore = "explicit Apple GPU Half/F32-scale RMS source comparisons"]
fn gpu_source_arrays_batched_encode_and_preflight() {
    autoreleasepool(|_| {
        let m = fixture();
        let gpu = Gpu::new().unwrap();
        let required = PIPELINE_ALLOWANCE + size_of::<Operation>();
        assert!(
            matches!(Operation::new(&gpu,required-1),Err(Error::Budget{required:r,limit})if r==required&&limit==required-1)
        );
        let op = Operation::new(&gpu, required).unwrap();
        // Equal extents reach the alias check instead of failing length first.
        let alias_plan = Plan::new(1, 32, 37, 37, 1e-6).unwrap();
        let alias_io = Operands::new(
            &gpu,
            alias_plan,
            &vec![0; alias_plan.input_len()],
            &vec![1.0f32.to_bits(); alias_plan.scale_len()],
        );
        let raw = gpu.command().unwrap();
        let command = Command {
            raw: &raw,
            trace: None,
        };
        assert!(matches!(
            op.encode(
                &command,
                alias_plan,
                Invocation {
                    output: &alias_io.input,
                    ..alias_io.invocation()
                }
            ),
            Err(Error::Options("Half FP32-scale RMS operand alias"))
        ));
        let short_input = Buffer::new(&gpu, alias_plan.input_len() - 1).unwrap();
        let short_output = Buffer::new(&gpu, alias_plan.output_len() - 1).unwrap();
        let short_scales = Buffer::new(&gpu, alias_plan.scale_len() - 1).unwrap();
        let short_status = Buffer::new(&gpu, alias_plan.status_len() - 1).unwrap();
        for invocation in [
            Invocation {
                input: &short_input,
                ..alias_io.invocation()
            },
            Invocation {
                output: &short_output,
                ..alias_io.invocation()
            },
            Invocation {
                scales: &short_scales,
                ..alias_io.invocation()
            },
            Invocation {
                status: &short_status,
                ..alias_io.invocation()
            },
        ] {
            assert!(matches!(
                op.encode(&command, alias_plan, invocation),
                Err(Error::Options("Half FP32-scale RMS operand extent"))
            ));
        }
        println!(
            "HALF_RMS_PREFLIGHT {}",
            json!({"equal_extent_input_output_alias_rejected":true,"short_operand_rejections":4})
        );
        let mut total = 0;
        let mut maximum = 0.0f64;
        let mut fraction = 0.0f64;
        let mut wide = 0;
        for c in m["cases"].as_array().unwrap() {
            let p = Plan::new(
                c["rows"].as_u64().unwrap() as usize,
                c["width"].as_u64().unwrap() as usize,
                c["input_stride"].as_u64().unwrap() as usize,
                c["output_stride"].as_u64().unwrap() as usize,
                1e-6,
            )
            .unwrap();
            let input = words16(&c["arrays"]["input"]);
            let scales = words32(&c["arrays"]["scales"]);
            let io = Operands::new(&gpu, p, &input, &scales);
            let raw = gpu.command().unwrap();
            let command = Command {
                raw: &raw,
                trace: None,
            };
            assert!(matches!(
                op.encode(
                    &command,
                    p,
                    Invocation {
                        output: &io.input,
                        ..io.invocation()
                    }
                ),
                Err(Error::Options(
                    "Half FP32-scale RMS operand extent" | "Half FP32-scale RMS operand alias"
                ))
            ));
            assert!(matches!(
                op.encode(
                    &command,
                    p,
                    Invocation {
                        status: &io.scales,
                        ..io.invocation()
                    }
                ),
                Err(Error::Options("Half FP32-scale RMS operand alias"))
            ));
            let mut second = Buffer::new(&gpu, p.output_len()).unwrap();
            second.with_mut(|v| v.fill(0x5a17));
            let measured = Measurement::start();
            let first_result = op.encode(&command, p, io.invocation());
            let first_allocations = measured.finish();
            first_result.unwrap();
            let measured = Measurement::start();
            let second_result = op.encode(
                &command,
                p,
                Invocation {
                    output: &second,
                    ..io.invocation()
                },
            );
            let second_allocations = measured.finish();
            second_result.unwrap();
            raw.commit();
            raw.waitUntilCompleted();
            let completed = raw.status() == MTLCommandBufferStatus::Completed;
            let (out, status) = io.record(c["name"].as_str().unwrap(), completed);
            let twin = second.with_slice(|v| v.to_vec());
            println!(
                "HALF_RMS_TWIN {}",
                json!({"case":c["name"],"output_words":twin,"encoded_operations":2,"submissions":1,"completion_waits":1,
                "encode_rust_allocations":[first_allocations,second_allocations]})
            );
            assert_eq!([first_allocations, second_allocations], [0, 0]);
            assert!(completed);
            assert_eq!(status[0], 0);
            assert_eq!(out, twin);
            for result in [&out, &twin] {
                let (n, e, f, w) = compare(c, p, result);
                total += n;
                maximum = maximum.max(e);
                fraction = fraction.max(f);
                wide += w;
            }
        }
        assert_eq!(total, 108544);
        println!(
            "HALF_RMS_RESULT {}",
            json!({"device":gpu.device.name().to_string(),"source_cases":6,"target_values":total,"max_absolute_error":maximum,
        "max_gate_fraction":fraction,"wide_diagnostic_exceedances":wide,"wide_fatal":false,"exact_scale_word":0x3fa7,
        "batched_operations_per_command":2,"default_pipelines_changed":false,"whole_graph_qualified":false})
        );
    });
}

#[test]
#[ignore = "explicit Apple GPU Half/F32-scale RMS status and recovery"]
fn gpu_nonfinite_status_sticky_and_recovery() {
    autoreleasepool(|_| {
        let m = fixture();
        let gpu = Gpu::new().unwrap();
        let op = Operation::new(&gpu, PIPELINE_ALLOWANCE + size_of::<Operation>()).unwrap();
        let errors: Value = serde_json::from_slice(include_bytes!(
            "../../../tests/fixtures/gemma3-half-rms-v1/errors.json"
        ))
        .unwrap();
        let baseline = &m["cases"][0];
        let input = words16(&baseline["arrays"]["input"]);
        let scales = words32(&baseline["arrays"]["scales"]);
        for c in errors["controls"].as_array().unwrap() {
            let p = Plan::new(1, 32, 35, 37, 1e-6).unwrap();
            let bad: Vec<u16> = c["input_words"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u16)
                .collect();
            let mut coeff: Vec<u32> = c["gamma_words"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| (1.0 + bf16::from_bits(v.as_u64().unwrap() as u16).to_f32()).to_bits())
                .collect();
            coeff.extend([0x5a170031; GUARD]);
            let mut io = Operands::new(&gpu, p, &bad, &coeff);
            let (_, status) = run(&gpu, &op, &io, c["name"].as_str().unwrap());
            assert_ne!(status[0], 0);
            io.input.with_mut(|v| v.copy_from_slice(&input));
            io.scales.with_mut(|v| v.copy_from_slice(&scales));
            io.input_hash = hash16(&input);
            io.scale_hash = hash32(&scales);
            io.output.with_mut(|v| v.fill(0x5a17));
            let (out, sticky) = run(&gpu, &op, &io, "valid_with_sticky_status");
            assert_eq!(sticky[0], status[0]);
            compare(baseline, p, &out);
            io.status.with_mut(|v| v[0] = 0);
            io.output.with_mut(|v| v.fill(0x5a17));
            let (out, cleared) = run(&gpu, &op, &io, "valid_after_explicit_reset");
            assert_eq!(cleared[0], 0);
            compare(baseline, p, &out);
        }
    });
}

#[test]
#[ignore = "explicit Apple GPU exact Half subnormal preservation"]
fn gpu_exact_subnormal_words() {
    autoreleasepool(|_| {
        let raw = include_bytes!("../../../tests/fixtures/gemma3-half-rms-v1/subnormal.json");
        assert_eq!(
            format!("{:x}", Sha256::digest(raw)),
            "a8d753607dcc1293e40978899145fbca5bcbf9061b45e6964daac15c3bca2b36"
        );
        let control: Value = serde_json::from_slice(raw).unwrap();
        let gpu = Gpu::new().unwrap();
        let op = Operation::new(&gpu, PIPELINE_ALLOWANCE + size_of::<Operation>()).unwrap();
        let p = Plan::new(1, 32, 35, 37, 1e-6).unwrap();
        let mut scales = vec![1.0f32.to_bits(); p.scale_len()];
        scales[32..].fill(0x5a170031);
        for c in control["cases"].as_array().unwrap() {
            let first = c["first_input_word"].as_u64().unwrap() as u16;
            let mut input = vec![0x7e55; p.input_len()];
            input[..32].fill(0);
            input[0] = first;
            let io = Operands::new(&gpu, p, &input, &scales);
            let (out, status) = run(&gpu, &op, &io, &format!("subnormal_{first:04x}"));
            assert_eq!(status[0], 0);
            assert_eq!(
                out[0],
                c["expected_first_main_word"].as_u64().unwrap() as u16
            );
            assert!(out[1..32].iter().all(|&word| word == 0));
        }
    });
}
