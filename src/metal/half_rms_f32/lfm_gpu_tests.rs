use super::*;
use crate::model::gemma_reference::tests::Measurement;
use half::{bf16, f16};
use objc2::rc::autoreleasepool;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const RAW: &[u8] = include_bytes!("../../../tests/fixtures/lfm2-half-rms-v1/arrays.bin");
fn fixture() -> Value {
    let raw = include_bytes!("../../../tests/fixtures/lfm2-half-rms-v1/manifest.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "abdbe293bc547a458c4177c2c67a07020be5ebfe3e5cc809e8aebd509d70557c"
    );
    let m: Value = serde_json::from_slice(raw).unwrap();
    assert_eq!(m["format"], "toklet-lfm2-half-rms-v1");
    assert_eq!(
        m["protocol_sha256"],
        "f2518543fc49139d4d982bcad3f0aad5bf4a41b99267aca75cf0cacea1cb3476"
    );
    assert_eq!(m["source_reference_executed"], true);
    assert_eq!(m["target_gpu_executed"], false);
    assert_eq!(m["finite_output_values"], 75008);
    assert_eq!(m["cases"].as_array().unwrap().len(), 7);
    assert_eq!(
        m["producer"]["sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../../../tools/reference/lfm2_half_rms.py"))
        )
    );
    assert_eq!(
        m["files"]["arrays.bin"]["sha256"],
        format!("{:x}", Sha256::digest(RAW))
    );
    let errors = include_bytes!("../../../tests/fixtures/lfm2-half-rms-v1/errors.json");
    assert_eq!(
        m["files"]["errors.json"]["sha256"],
        format!("{:x}", Sha256::digest(errors))
    );
    assert_eq!(m["policy"]["fatal_reference"], "main");
    assert_eq!(m["policy"]["atol_floor"].as_f64(), Some(1e-4));
    assert_eq!(m["policy"]["rtol"].as_f64(), Some(1e-3));
    assert_eq!(m["epsilon_bits"], 1e-5_f32.to_bits());
    assert_eq!(m["policy"]["envelope_factor"].as_f64(), Some(2.0));
    assert_eq!(
        m["helper"]["sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../../../tools/reference/gemma3_half_rms.py"
            ))
        )
    );
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
        let gamma = words16(&c["arrays"]["gamma"]);
        let scales = words32(&c["arrays"]["scales"]);
        assert_eq!(gamma.len(), width);
        for (i, &word) in gamma.iter().enumerate() {
            assert_eq!(scales[i], bf16::from_bits(word).to_f32().to_bits());
        }
        assert!(scales[width..].iter().all(|&w| w == 0x5a170031));
        let input = words16(&c["arrays"]["input"]);
        for (row, chunk) in input[..rows * plan.input_stride]
            .chunks_exact(plan.input_stride)
            .enumerate()
        {
            assert!(chunk[width..].iter().all(|&w| w == 0x7e55));
            if rows == 512 && row == 511 {
                assert!(chunk[..width].iter().any(|&w| w != 0));
            }
        }
        assert!(
            input[rows * plan.input_stride..]
                .iter()
                .all(|&w| w == 0x7e55)
        );
        assert_eq!(c["arrays"]["main"]["shape"], json!([rows, width]));
        assert_eq!(c["arrays"]["wide"]["shape"], json!([rows, width]));
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
            for key in ["wrong_half_scale_word", "wrong_unit_offset_word"] {
                if let Some(wrong) = exact[key].as_u64() {
                    assert_ne!(a[i], wrong as u16);
                }
            }
        }
    }
    assert_eq!(values, 75008);
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
struct Capture {
    output: Vec<u16>,
    status: Vec<u32>,
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
    fn record(&self, label: &str, command: &Value) -> Capture {
        let c = Capture {
            output: self.output.with_slice(|v| v.to_vec()),
            status: self.status.with_slice(|v| v.to_vec()),
            input_hash: self.input.with_slice(hash16),
            scale_hash: self.scales.with_slice(hash32),
        };
        println!(
            "LFM_RMS_RAW {}",
            json!({"label":label,"command":command,"rows":self.plan.rows,"width":self.plan.width,"epsilon_bits":self.plan.epsilon_bits,"input_stride":self.plan.input_stride,"output_stride":self.plan.output_stride,"input_before_sha256":self.input_hash,"input_after_sha256":c.input_hash,"scales_before_sha256":self.scale_hash,"scales_after_sha256":c.scale_hash,"output_words":c.output,"status_words":c.status,"operand_bytes":self.plan.operand_bytes().unwrap()})
        );
        c
    }
    fn validate(&self, c: &Capture) {
        assert_eq!(c.input_hash, self.input_hash);
        assert_eq!(c.scale_hash, self.scale_hash);
        for (i, &word) in c.output.iter().enumerate() {
            if i >= self.plan.rows * self.plan.output_stride
                || i % self.plan.output_stride >= self.plan.width
            {
                assert_eq!(word, 0x5a17, "output padding/guard {i}");
            }
        }
        assert!(c.status[1..].iter().all(|&w| w == 0x5a170031));
    }
}
fn completed(raw: &ProtocolObject<dyn MTLCommandBuffer>) -> Value {
    let status = raw.status();
    let error = raw.error().map(|e| e.to_string());
    // Never map an in-flight operand, including a failed completion contract.
    assert!(matches!(
        status,
        MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
    ));
    json!({"completed":status==MTLCommandBufferStatus::Completed&&error.is_none(),"status":status.0,"error":error})
}
fn run(gpu: &Gpu, op: &Operation, io: &Operands, label: &str) -> Capture {
    let raw = gpu.command().unwrap();
    let command = Command {
        raw: &raw,
        trace: None,
    };
    op.encode(&command, io.plan, io.invocation()).unwrap();
    raw.commit();
    raw.waitUntilCompleted();
    let command = completed(&raw);
    let capture = io.record(label, &command);
    io.validate(&capture);
    assert_eq!(command["completed"], true);
    capture
}
#[derive(Default)]
struct Metrics {
    values: usize,
    max_error: f64,
    max_fraction: f64,
    wide_exceedances: usize,
    wide_max_error: f64,
    wide_max_fraction: f64,
}
impl Metrics {
    fn compare(&mut self, c: &Value, p: Plan, out: &[u16]) {
        let main = words16(&c["arrays"]["main"]);
        let wide = words16(&c["arrays"]["wide"]);
        let atol = c["gate"]["atol"].as_f64().unwrap();
        let rtol = c["gate"]["rtol"].as_f64().unwrap();
        for row in 0..p.rows {
            for col in 0..p.width {
                let i = row * p.width + col;
                let word = out[row * p.output_stride + col];
                let actual = f64::from(f16::from_bits(word).to_f32());
                let expected = f64::from(f16::from_bits(main[i]).to_f32());
                let w = f64::from(f16::from_bits(wide[i]).to_f32());
                assert!(actual.is_finite() && expected.is_finite() && w.is_finite());
                let error = (actual - expected).abs();
                let limit = atol + rtol * expected.abs();
                self.values += 1;
                self.max_error = self.max_error.max(error);
                self.max_fraction = self.max_fraction.max(error / limit);
                let we = (actual - w).abs();
                let wl = atol + rtol * w.abs();
                self.wide_exceedances += usize::from(we > wl);
                self.wide_max_error = self.wide_max_error.max(we);
                self.wide_max_fraction = self.wide_max_fraction.max(we / wl);
                assert!(
                    error <= limit,
                    "{}[{row},{col}] native={word:04x} main={:04x} wide={:04x} error={error} limit={limit}",
                    c["name"],
                    main[i],
                    wide[i]
                );
            }
        }
        for e in c["exact"].as_array().unwrap() {
            let i = e["row"].as_u64().unwrap() as usize * p.output_stride
                + e["column"].as_u64().unwrap() as usize;
            assert_eq!(out[i], e["expected_word"].as_u64().unwrap() as u16);
        }
    }
}
#[test]
#[ignore = "explicit Apple GPU LFM direct-gain RMS strided/tight source qualification"]
fn gpu_lfm_source_words_strided_tight_and_zero_allocation() {
    autoreleasepool(|_| {
        let m = fixture();
        let gpu = Gpu::new().unwrap();
        let required = PIPELINE_ALLOWANCE + size_of::<Operation>();
        assert!(
            matches!(Operation::new(&gpu,required-1),Err(Error::Budget{required:r,limit})if r==required&&limit==required-1)
        );
        let op = Operation::new(&gpu, required).unwrap();
        let alias = Plan::new(1, 64, 69, 69, 1e-5).unwrap();
        let io = Operands::new(
            &gpu,
            alias,
            &vec![0; alias.input_len()],
            &vec![1f32.to_bits(); alias.scale_len()],
        );
        let raw = gpu.command().unwrap();
        let command = Command {
            raw: &raw,
            trace: None,
        };
        assert!(matches!(
            op.encode(
                &command,
                alias,
                Invocation {
                    output: &io.input,
                    ..io.invocation()
                }
            ),
            Err(Error::Options("Half FP32-scale RMS operand alias"))
        ));
        assert!(matches!(
            op.encode(
                &command,
                alias,
                Invocation {
                    status: &io.scales,
                    ..io.invocation()
                }
            ),
            Err(Error::Options("Half FP32-scale RMS operand alias"))
        ));
        let short_input = Buffer::new(&gpu, alias.input_len() - 1).unwrap();
        let short_output = Buffer::new(&gpu, alias.output_len() - 1).unwrap();
        let short_scales = Buffer::new(&gpu, alias.scale_len() - 1).unwrap();
        let short_status = Buffer::new(&gpu, alias.status_len() - 1).unwrap();
        for v in [
            Invocation {
                input: &short_input,
                ..io.invocation()
            },
            Invocation {
                output: &short_output,
                ..io.invocation()
            },
            Invocation {
                scales: &short_scales,
                ..io.invocation()
            },
            Invocation {
                status: &short_status,
                ..io.invocation()
            },
        ] {
            assert!(matches!(
                op.encode(&command, alias, v),
                Err(Error::Options("Half FP32-scale RMS operand extent"))
            ));
        }
        println!(
            "LFM_RMS_PREFLIGHT {}",
            json!({"typed_alias_rejections":2,"short_operands":4,"pipeline_budget_exact_and_minus_one":true})
        );
        let mut metrics = Metrics::default();
        let mut allocations = 0;
        for c in m["cases"].as_array().unwrap() {
            let rows = c["rows"].as_u64().unwrap() as usize;
            let width = c["width"].as_u64().unwrap() as usize;
            let name = c["name"].as_str().unwrap();
            let p = Plan::new(
                rows,
                width,
                c["input_stride"].as_u64().unwrap() as usize,
                c["output_stride"].as_u64().unwrap() as usize,
                1e-5,
            )
            .unwrap();
            let tight = Plan::new(rows, width, width, width, 1e-5).unwrap();
            let input = words16(&c["arrays"]["input"]);
            let scales = words32(&c["arrays"]["scales"]);
            let mut ti = vec![0x7e55; tight.input_len()];
            for row in 0..rows {
                ti[row * width..(row + 1) * width]
                    .copy_from_slice(&input[row * p.input_stride..row * p.input_stride + width]);
            }
            let strided = Operands::new(&gpu, p, &input, &scales);
            let contiguous = Operands::new(&gpu, tight, &ti, &scales);
            let raw = gpu.command().unwrap();
            let command = Command {
                raw: &raw,
                trace: None,
            };
            let measure = Measurement::start();
            let first = op.encode(&command, p, strided.invocation());
            let a = measure.finish();
            println!(
                "LFM_RMS_ALLOCATION {}",
                json!({"case":name,"layout":"strided","rust_allocations":a,"encode_ok":first.is_ok(),"release_required_zero":!cfg!(debug_assertions)})
            );
            first.unwrap();
            let measure = Measurement::start();
            let second = op.encode(&command, tight, contiguous.invocation());
            let b = measure.finish();
            println!(
                "LFM_RMS_ALLOCATION {}",
                json!({"case":name,"layout":"tight","rust_allocations":b,"encode_ok":second.is_ok(),"release_required_zero":!cfg!(debug_assertions)})
            );
            second.unwrap();
            raw.commit();
            raw.waitUntilCompleted();
            let completion = completed(&raw);
            let x = strided.record(&format!("{name}/strided"), &completion);
            let y = contiguous.record(&format!("{name}/tight"), &completion);
            // Both complete raw outputs and allocation records precede gates.
            strided.validate(&x);
            contiguous.validate(&y);
            assert_eq!(completion["completed"], true);
            assert_eq!(x.status[0], 0);
            assert_eq!(y.status[0], 0);
            if !cfg!(debug_assertions) {
                assert_eq!([a, b], [0, 0]);
            }
            allocations += 2;
            for row in 0..rows {
                assert_eq!(
                    &x.output[row * p.output_stride..row * p.output_stride + width],
                    &y.output[row * width..(row + 1) * width],
                    "strided/tight exact"
                );
            }
            metrics.compare(c, p, &x.output);
            metrics.compare(c, tight, &y.output);
        }
        assert_eq!(metrics.values, 150016);
        assert_eq!(allocations, 14);
        println!(
            "LFM_RMS_RESULT {}",
            json!({"device":gpu.device.name().to_string(),"source_cases":7,"main_gate_values":metrics.values,"main_failures":0,"max_absolute_error":metrics.max_error,"max_gate_fraction":metrics.max_fraction,"wide_diagnostic_values":metrics.values,"wide_exceedances":metrics.wide_exceedances,"wide_max_error":metrics.wide_max_error,"wide_max_fraction":metrics.wide_max_fraction,"wide_fatal":false,"rust_encode_measurements":allocations,"release_required_zero":!cfg!(debug_assertions),"exact_gain_word":4,"exact_epsilon_words":[0x013c,0x813c],"whole_graph_qualified":false})
        );
    });
}
#[test]
#[ignore = "explicit Apple GPU LFM RMS invalid operand/sticky status/reset recovery"]
fn gpu_lfm_nonfinite_sticky_and_recovery() {
    autoreleasepool(|_| {
        let m = fixture();
        let gpu = Gpu::new().unwrap();
        let op = Operation::new(&gpu, PIPELINE_ALLOWANCE + size_of::<Operation>()).unwrap();
        let errors: Value = serde_json::from_slice(include_bytes!(
            "../../../tests/fixtures/lfm2-half-rms-v1/errors.json"
        ))
        .unwrap();
        assert_eq!(errors["controls"].as_array().unwrap().len(), 5);
        let baseline = &m["cases"][1];
        assert_eq!(baseline["name"], "coefficient_underflow");
        let input = words16(&baseline["arrays"]["input"]);
        let scales = words32(&baseline["arrays"]["scales"]);
        let mut metrics = Metrics::default();
        for c in errors["controls"].as_array().unwrap() {
            let name = c["name"].as_str().unwrap();
            let p = Plan::new(1, 64, 67, 69, 1e-5).unwrap();
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
                .map(|v| {
                    bf16::from_bits(v.as_u64().unwrap() as u16)
                        .to_f32()
                        .to_bits()
                })
                .collect();
            coeff.extend([0x5a170031; GUARD]);
            let mut io = Operands::new(&gpu, p, &bad, &coeff);
            let failure = run(&gpu, &op, &io, &format!("{name}/invalid"));
            assert_ne!(failure.status[0], 0);
            io.input.with_mut(|v| v.copy_from_slice(&input));
            io.scales.with_mut(|v| v.copy_from_slice(&scales));
            io.input_hash = hash16(&input);
            io.scale_hash = hash32(&scales);
            io.output.with_mut(|v| v.fill(0x5a17));
            let sticky = run(&gpu, &op, &io, &format!("{name}/valid_sticky"));
            assert_eq!(sticky.status[0], failure.status[0]);
            metrics.compare(baseline, p, &sticky.output);
            io.status.with_mut(|v| v[0] = 0);
            io.output.with_mut(|v| v.fill(0x5a17));
            let cleared = run(&gpu, &op, &io, &format!("{name}/valid_reset"));
            assert_eq!(cleared.status[0], 0);
            metrics.compare(baseline, p, &cleared.output);
        }
        assert_eq!(metrics.values, 640);
        println!(
            "LFM_RMS_RECOVERY_RESULT {}",
            json!({"invalid_sites":5,"submitted_phases":15,"valid_recovery_values":metrics.values,"main_failures":0,"max_error":metrics.max_error,"max_fraction":metrics.max_fraction,"invalid_output_bits_are_oracle":false})
        );
    });
}
