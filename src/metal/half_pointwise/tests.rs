use super::super::memory::Kernel;
use super::*;
use crate::model::gemma_reference::tests::Measurement;
use half::f16;
use objc2::rc::autoreleasepool;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const RAW: &[u8] = include_bytes!("../../../tests/fixtures/gemma3-half-pointwise-v1/arrays.bin");
const CANARY: u16 = 0x5a17;
const STATUS_GUARD: u32 = 0x5a170031;
fn fixture() -> Value {
    let raw = include_bytes!("../../../tests/fixtures/gemma3-half-pointwise-v1/manifest.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "a0814112d9db5b77cf55832625e062723d90ab79c3fe1a16ae07a9efccc52cea"
    );
    let m: Value = serde_json::from_slice(raw).unwrap();
    assert_eq!(m["format"], "toklet-gemma3-half-pointwise-v1");
    assert_eq!(
        m["protocol_sha256"],
        "fd84f903bc1f078b3413e4a6ff1a195f404ab3eac723fd111761276c3265e45b"
    );
    assert_eq!(m["source_reference_executed"], true);
    assert_eq!(m["target_gpu_executed"], false);
    assert_eq!(m["values_per_profile"], 152512);
    assert_eq!(m["cases"].as_array().unwrap().len(), 4);
    assert_eq!(
        m["producer"]["sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../../../tools/reference/gemma3_half_pointwise.py"
            ))
        )
    );
    assert_eq!(
        m["files"]["arrays.bin"]["sha256"],
        format!("{:x}", Sha256::digest(RAW))
    );
    assert_eq!(
        m["files"]["controls.json"]["sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../../../tests/fixtures/gemma3-half-pointwise-v1/controls.json"
            ))
        )
    );
    assert_eq!(m["policy"]["fatal_reference"], "main");
    assert_eq!(m["policy"]["atol_floor"].as_f64(), Some(1e-4));
    assert_eq!(m["policy"]["rtol"].as_f64(), Some(1e-3));
    m
}
fn words(d: &Value) -> Vec<u16> {
    assert_eq!(d["file"], "arrays.bin");
    assert_eq!(d["dtype"], "u16le");
    let start = d["offset"].as_u64().unwrap() as usize;
    let size = d["bytes"].as_u64().unwrap() as usize;
    let bytes = &RAW[start..start + size];
    assert_eq!(d["sha256"], format!("{:x}", Sha256::digest(bytes)));
    bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn hash(v: &[u16]) -> String {
    let mut h = Sha256::new();
    for word in v {
        h.update(word.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn upload(gpu: &Gpu, words: &[u16]) -> Buffer<u16> {
    let mut b = Buffer::new(gpu, words.len()).unwrap();
    b.with_mut(|v| v.copy_from_slice(words));
    b
}
fn empty(gpu: &Gpu, len: usize) -> Buffer<u16> {
    let mut b = Buffer::new(gpu, len).unwrap();
    b.with_mut(|v| v.fill(CANARY));
    b
}
fn plan(c: &Value, tight: bool) -> Plan {
    let kind = if c["operation"] == "scale" {
        Kind::Scale(c["factor"].as_f64().unwrap() as f32)
    } else {
        Kind::Gelu
    };
    let rows = c["rows"].as_u64().unwrap() as usize;
    let width = c["width"].as_u64().unwrap() as usize;
    Plan::new(
        kind,
        rows,
        width,
        c["input_stride"].as_u64().unwrap() as usize,
        if tight {
            width
        } else {
            c["output_stride"].as_u64().unwrap() as usize
        },
    )
    .unwrap()
}
#[test]
fn checked_shapes_bytes_and_profiles() {
    let p = Plan::new(Kind::Gelu, 1, 128, 131, 133).unwrap();
    assert_eq!(
        (p.input_len(), p.output_len(), p.status_len()),
        (138, 140, 8)
    );
    assert_eq!(p.operand_bytes().unwrap(), 588);
    for (kind, width) in [
        (Kind::Gelu, 128),
        (Kind::Gelu, 2048),
        (Kind::Scale(8.0), 64),
        (Kind::Scale(25.25), 640),
    ] {
        assert!(Plan::new(kind, 32, width, width + 3, width + 5).is_ok());
        for (rows, input, output) in [
            (0, width, width),
            (33, width, width),
            (1, width - 1, width),
            (1, width, width - 1),
            (2, usize::MAX, width),
            (32, u32::MAX as usize, width),
        ] {
            assert!(Plan::new(kind, rows, width, input, output).is_err());
        }
    }
    for (kind, width) in [
        (Kind::Gelu, 640),
        (Kind::Scale(8.0), 128),
        (Kind::Scale(25.29822), 640),
        (Kind::Scale(f32::NAN), 64),
    ] {
        assert!(Plan::new(kind, 1, width, width, width).is_err());
    }
}
#[test]
fn frozen_policy_shapes_and_exact_store_witnesses() {
    let m = fixture();
    let mut total = 0;
    for c in m["cases"].as_array().unwrap() {
        let p = plan(c, false);
        assert_eq!(words(&c["inputs"]["input"]).len(), p.input_len());
        for (stage, d) in c["arrays"].as_object().unwrap() {
            let main = words(&d["main"]);
            let wide = words(&d["wide"]);
            assert_eq!(main.len(), p.rows * p.width);
            assert_eq!(wide.len(), main.len());
            let envelope = main
                .iter()
                .zip(&wide)
                .map(|(&a, &b)| {
                    (f64::from(f16::from_bits(a).to_f32()) - f64::from(f16::from_bits(b).to_f32()))
                        .abs()
                })
                .fold(0.0, f64::max);
            assert_eq!(d["envelope"].as_f64(), Some(envelope));
            assert_eq!(d["gate"]["atol"].as_f64(), Some((2.0 * envelope).max(1e-4)));
            assert_eq!(d["gate"]["rtol"].as_f64(), Some(1e-3));
            for point in c["exact"].as_array().unwrap() {
                let index = point["row"].as_u64().unwrap() as usize * p.width
                    + point["column"].as_u64().unwrap() as usize;
                assert_eq!(main[index], point[stage].as_u64().unwrap() as u16);
                assert_eq!(main[index], wide[index]);
            }
            total += main.len();
        }
    }
    assert_eq!(total, 152512);
}
struct Io {
    p: Plan,
    input: Buffer<u16>,
    output: Buffer<u16>,
    status: Buffer<u32>,
    input_hash: String,
}
impl Io {
    fn new(gpu: &Gpu, p: Plan, input: &[u16]) -> Self {
        assert_eq!(input.len(), p.input_len());
        let mut status = Buffer::new(gpu, p.status_len()).unwrap();
        status.with_mut(|v| {
            v.fill(STATUS_GUARD);
            v[0] = 0;
        });
        let io = Self {
            p,
            input: upload(gpu, input),
            output: empty(gpu, p.output_len()),
            status,
            input_hash: hash(input),
        };
        assert_eq!(
            io.input.raw.length() + io.output.raw.length() + io.status.raw.length(),
            p.operand_bytes().unwrap()
        );
        io
    }
    fn invocation(&self) -> Invocation<'_> {
        Invocation {
            input: &self.input,
            output: &self.output,
            status: &self.status,
        }
    }
    fn record(&self, label: &str, completion: &Value) -> (Vec<u16>, Vec<u32>) {
        let out = self.output.with_slice(|v| v.to_vec());
        let status = self.status.with_slice(|v| v.to_vec());
        let after = self.input.with_slice(hash);
        println!(
            "POINTWISE_RAW {}",
            json!({"label":label,"completion":completion,"rows":self.p.rows,"width":self.p.width,
            "input_stride":self.p.input_stride,"output_stride":self.p.output_stride,"input_before_sha256":self.input_hash,"input_after_sha256":after,
            "output_words":out,"status_words":status,"operand_bytes":self.p.operand_bytes().unwrap()})
        );
        (out, status)
    }
    fn validate(&self, out: &[u16], status: &[u32]) {
        assert_eq!(self.input.with_slice(hash), self.input_hash);
        guards(out, self.p.rows, self.p.width, self.p.output_stride);
        assert!(status[1..].iter().all(|&w| w == STATUS_GUARD));
    }
}
fn completion(raw: &ProtocolObject<dyn MTLCommandBuffer>) -> Value {
    let status = raw.status();
    let error = raw.error().map(|e| e.to_string());
    json!({"completed":status==MTLCommandBufferStatus::Completed && error.is_none(),
        "status":format!("{status:?}"),"error":error})
}
fn guards(out: &[u16], rows: usize, width: usize, stride: usize) {
    for (i, &word) in out.iter().enumerate() {
        if i >= rows * stride || i % stride >= width {
            assert_eq!(word, CANARY, "guard/pad {i}");
        }
    }
}
fn compare(c: &Value, stage: &str, out: &[u16], stride: usize) -> (usize, f64, f64, usize) {
    let d = &c["arrays"][stage];
    let main = words(&d["main"]);
    let wide = words(&d["wide"]);
    let width = c["width"].as_u64().unwrap() as usize;
    let atol = d["gate"]["atol"].as_f64().unwrap();
    let rtol = d["gate"]["rtol"].as_f64().unwrap();
    let (mut maximum, mut fraction, mut diagnostics) = (0.0f64, 0.0f64, 0);
    for (i, &expected) in main.iter().enumerate() {
        let actual = f64::from(f16::from_bits(out[i / width * stride + i % width]).to_f32());
        let expected = f64::from(f16::from_bits(expected).to_f32());
        let w = f64::from(f16::from_bits(wide[i]).to_f32());
        let error = (actual - expected).abs();
        let limit = atol + rtol * expected.abs();
        assert!(
            actual.is_finite() && error <= limit,
            "{} {stage}[{i}] {actual} vs {expected}, limit{limit}",
            c["name"]
        );
        maximum = maximum.max(error);
        fraction = fraction.max(error / limit);
        diagnostics += usize::from((actual - w).abs() > atol + rtol * w.abs());
    }
    for point in c["exact"].as_array().unwrap() {
        let i = point["row"].as_u64().unwrap() as usize * stride
            + point["column"].as_u64().unwrap() as usize;
        assert_eq!(out[i], point[stage].as_u64().unwrap() as u16);
    }
    (main.len(), maximum, fraction, diagnostics)
}
fn measured_encode(op: &Operation, command: &Command<'_>, p: Plan, io: Invocation<'_>) -> usize {
    let measured = Measurement::start();
    let result = op.encode(command, p, io);
    let count = measured.finish();
    result.unwrap();
    count
}
fn gating(
    gpu: &Gpu,
    command: &Command<'_>,
    a: &Buffer<u16>,
    up: &Buffer<u16>,
    out: &Buffer<u16>,
    status: &Buffer<u32>,
    n: usize,
) -> Result<()> {
    gpu.encode(
        command,
        Kernel::Gating,
        &[&a.raw, &up.raw, &out.raw],
        &[n as u32],
        n,
        Some(status),
    )
}

#[test]
#[ignore = "explicit release Metal Half GELU/scale and stored-Half gating"]
fn gpu_source_arrays_physical_store_and_preflight() {
    autoreleasepool(|_| {
        let m = fixture();
        let gpu = Gpu::new().unwrap();
        let required = PIPELINE_ALLOWANCE + size_of::<Operation>();
        assert!(
            matches!(Operation::new(&gpu,required-1),Err(Error::Budget{required:r,limit})if r==required&&limit==required-1)
        );
        let op = Operation::new(&gpu, required).unwrap();
        let p = Plan::new(Kind::Gelu, 1, 128, 128, 128).unwrap();
        let io = Io::new(&gpu, p, &vec![0; p.input_len()]);
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
            Err(Error::Options("Half pointwise operand alias"))
        ));
        let short_input = empty(&gpu, p.input_len() - 1);
        let short_output = empty(&gpu, p.output_len() - 1);
        let short_status = Buffer::new(&gpu, p.status_len() - 1).unwrap();
        for invocation in [
            Invocation {
                input: &short_input,
                ..io.invocation()
            },
            Invocation {
                output: &short_output,
                ..io.invocation()
            },
            Invocation {
                status: &short_status,
                ..io.invocation()
            },
        ] {
            assert!(matches!(
                op.encode(&command, p, invocation),
                Err(Error::Options("Half pointwise operand extent"))
            ));
        }
        println!(
            "POINTWISE_PREFLIGHT {}",
            json!({"equal_extent_alias_rejected":true,"short_buffer_rejections":3})
        );
        let (mut total, mut maximum, mut fraction, mut diagnostics, mut calls) =
            (0, 0.0f64, 0.0f64, 0, 0);
        for c in m["cases"].as_array().unwrap() {
            let p = plan(c, false);
            let input = words(&c["inputs"]["input"]);
            let io = Io::new(&gpu, p, &input);
            let composite = if c["operation"] == "gelu_gated" {
                let up = words(&c["inputs"]["up"]);
                let h = hash(&up);
                Some((
                    upload(&gpu, &up),
                    empty(&gpu, p.rows * p.width + GUARD),
                    empty(&gpu, p.rows * p.width + GUARD),
                    h,
                ))
            } else {
                None
            };
            let raw = gpu.command().unwrap();
            let command = Command {
                raw: &raw,
                trace: None,
            };
            let mut allocations = vec![measured_encode(&op, &command, p, io.invocation())];
            if let Some((up, activation, product, _)) = &composite {
                allocations.push(measured_encode(
                    &op,
                    &command,
                    plan(c, true),
                    Invocation {
                        output: activation,
                        ..io.invocation()
                    },
                ));
                let measured = Measurement::start();
                let result = gating(
                    &gpu,
                    &command,
                    activation,
                    up,
                    product,
                    &io.status,
                    p.rows * p.width,
                );
                allocations.push(measured.finish());
                result.unwrap();
            }
            raw.commit();
            raw.waitUntilCompleted();
            let completion = completion(&raw);
            let completed = completion["completed"].as_bool().unwrap();
            let (out, status) = io.record(c["name"].as_str().unwrap(), &completion);
            let extra=composite.as_ref().map(|(up,a,y,before)|{
                let a=a.with_slice(|v|v.to_vec());let y=y.with_slice(|v|v.to_vec());let after=up.with_slice(hash);
                println!("POINTWISE_COMPOSED {}",json!({"case":c["name"],"activation_words":a,"product_words":y,"up_before_sha256":before,"up_after_sha256":after,
                    "numerical_allocated_bytes":io.p.operand_bytes().unwrap()+up.raw.length()+2*(p.rows*p.width+GUARD)*2,
                    "submissions":1,"completion_waits":1,"encoded_operations":3}));
                (a,y,after)
            });
            println!(
                "POINTWISE_ALLOCATIONS {}",
                json!({"case":c["name"],"release_rust_allocations":allocations,"driver_allocations_measured":false})
            );
            assert!(completed);
            io.validate(&out, &status);
            assert_eq!(status[0], 0);
            assert!(allocations.iter().all(|&n| n == 0));
            calls += allocations.len();
            let mut results = Vec::new();
            if let Some((a, y, after)) = extra {
                assert_eq!(&after, &composite.as_ref().unwrap().3);
                guards(&a, p.rows, p.width, p.width);
                guards(&y, p.rows, p.width, p.width);
                results.push(compare(c, "activation", &out, p.output_stride));
                results.push(compare(c, "activation", &a, p.width));
                results.push(compare(c, "product", &y, p.width));
                for row in 0..p.rows {
                    assert_eq!(
                        &out[row * p.output_stride..row * p.output_stride + p.width],
                        &a[row * p.width..(row + 1) * p.width]
                    );
                }
            } else {
                results.push(compare(c, "scaled", &out, p.output_stride));
            }
            for (n, e, f, w) in results {
                total += n;
                maximum = maximum.max(e);
                fraction = fraction.max(f);
                diagnostics += w;
            }
        }
        assert_eq!(total, 218432);
        assert_eq!(calls, 8);
        println!(
            "POINTWISE_RESULT {}",
            json!({"device":gpu.device.name().to_string(),"source_cases":4,"target_values":total,"max_absolute_error":maximum,"max_gate_fraction":fraction,
            "wide_diagnostic_exceedances":diagnostics,"wide_fatal":false,"measured_encode_calls":calls,"default_pipelines_changed":false,"whole_graph_qualified":false})
        );
    });
}

#[test]
#[ignore = "explicit release Metal pointwise invalid/sticky/reset controls"]
fn gpu_nonfinite_status_and_recovery() {
    autoreleasepool(|_| {
        let _ = fixture();
        let controls: Value = serde_json::from_slice(include_bytes!(
            "../../../tests/fixtures/gemma3-half-pointwise-v1/controls.json"
        ))
        .unwrap();
        assert_eq!(controls["error_output_words_not_a_gate"], true);
        let gpu = Gpu::new().unwrap();
        let op = Operation::new(&gpu, PIPELINE_ALLOWANCE + size_of::<Operation>()).unwrap();
        for (index, c) in controls["errors"].as_array().unwrap().iter().enumerate() {
            let scale = c["operation"] == "scale";
            let width = if scale { 640 } else { 128 };
            let kind = if scale {
                Kind::Scale(25.25)
            } else {
                Kind::Gelu
            };
            let p = Plan::new(kind, 1, width, width, width).unwrap();
            let mut input = vec![0; p.input_len()];
            input[width..].fill(0x7e55);
            input[0] = c["input_word"].as_u64().unwrap() as u16;
            let mut io = Io::new(&gpu, p, &input);
            let mut up_words = vec![0x3c00; width + GUARD];
            up_words[width..].fill(0x7e55);
            if !scale {
                up_words[0] = c["up_word"].as_u64().unwrap() as u16;
            }
            let mut up = upload(&gpu, &up_words);
            let mut product = empty(&gpu, width + GUARD);
            let mut sticky = 0;
            for phase in 0..3 {
                if phase > 0 {
                    input[0] = if scale { 0x3c00 } else { 0x3400 };
                    io.input.with_mut(|v| v.copy_from_slice(&input));
                    io.input_hash = hash(&input);
                    up_words[0] = 0x4500;
                    up.with_mut(|v| v.copy_from_slice(&up_words));
                    io.output.with_mut(|v| v.fill(CANARY));
                    product.with_mut(|v| v.fill(CANARY));
                }
                if phase == 2 {
                    io.status.with_mut(|v| v[0] = 0);
                }
                let raw = gpu.command().unwrap();
                let command = Command {
                    raw: &raw,
                    trace: None,
                };
                op.encode(&command, p, io.invocation()).unwrap();
                if !scale {
                    gating(&gpu, &command, &io.output, &up, &product, &io.status, width).unwrap();
                }
                raw.commit();
                raw.waitUntilCompleted();
                let completion = completion(&raw);
                let completed = completion["completed"].as_bool().unwrap();
                let (out, status) = io.record(&format!("error{index}_phase{phase}"), &completion);
                let y = product.with_slice(|v| v.to_vec());
                let after = up.with_slice(hash);
                println!(
                    "POINTWISE_RECOVERY {}",
                    json!({"control":index,"phase":phase,"operation":c["operation"],"product_words":y,"up_before_sha256":hash(&up_words),"up_after_sha256":after})
                );
                assert!(completed);
                io.validate(&out, &status);
                assert_eq!(after, hash(&up_words));
                guards(&y, 1, width, width);
                if phase == 0 {
                    assert_ne!(status[0], 0);
                    sticky = status[0];
                } else {
                    assert_eq!(status[0], if phase == 1 { sticky } else { 0 });
                    assert_eq!(out[0], if scale { 0x4e50 } else { 0x30ca });
                    assert!(out[1..width].iter().all(|&w| w == 0));
                    if !scale {
                        assert_eq!(y[0], 0x39fc);
                        assert!(y[1..width].iter().all(|&w| w == 0));
                    }
                }
            }
        }
    });
}
