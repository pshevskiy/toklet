use super::*;
use crate::model::gemma_reference::tests::Measurement;
use half::{bf16, f16};
use objc2::rc::autoreleasepool;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const RAW: &[u8] = include_bytes!("../../../tests/fixtures/causal-conv3-half-v1/arrays.bin");
const ERRORS: &[u8] = include_bytes!("../../../tests/fixtures/causal-conv3-half-v1/errors.json");
fn fixture() -> Value {
    let raw = include_bytes!("../../../tests/fixtures/causal-conv3-half-v1/manifest.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "ca946c68bce4b38bcf231a6e8a1f55dfa86ed7dc9719667c087414404cc867cc"
    );
    let m: Value = serde_json::from_slice(raw).unwrap();
    assert_eq!(m["format"], "toklet-causal-conv3-half-v1");
    assert_eq!(
        m["protocol_sha256"],
        "251197ef28690222f29cc1e2315a5c2fa9a245a8efc1da02daf5a41de09b11b3"
    );
    assert_eq!(m["source_reference_executed"], true);
    assert_eq!(m["target_gpu_executed"], false);
    assert_eq!(m["finite_output_values"], 46571);
    assert_eq!(m["candidate_words"], 7238);
    assert_eq!(m["cases"].as_array().unwrap().len(), 12);
    for (name, raw) in [("arrays.bin", RAW), ("errors.json", ERRORS)] {
        assert_eq!(m["files"][name]["bytes"], raw.len());
        assert_eq!(
            m["files"][name]["sha256"],
            format!("{:x}", Sha256::digest(raw))
        );
    }
    for (name, raw) in [
        (
            "producer",
            include_bytes!("../../../tools/reference/causal_conv3_half.py").as_slice(),
        ),
        (
            "helper",
            include_bytes!("../../../tools/reference/gemma3_half_rms.py").as_slice(),
        ),
    ] {
        assert_eq!(m[name]["sha256"], format!("{:x}", Sha256::digest(raw)));
    }
    assert_eq!(
        m["policy"],
        json!({"atol_floor":1e-4,"envelope_factor":2,"rtol":1e-3,"fatal_reference":"main"})
    );
    m
}
fn bytes(d: &Value, dtype: &str, item: usize) -> &'static [u8] {
    assert_eq!(d["file"], "arrays.bin");
    assert_eq!(d["dtype"], dtype);
    let start = usize::try_from(d["offset"].as_u64().unwrap()).unwrap();
    let len = usize::try_from(d["bytes"].as_u64().unwrap()).unwrap();
    let elements = d["shape"]
        .as_array()
        .unwrap()
        .iter()
        .try_fold(1usize, |n, d| {
            n.checked_mul(usize::try_from(d.as_u64().unwrap()).unwrap())
        })
        .unwrap();
    assert_eq!(elements.checked_mul(item).unwrap(), len);
    let raw = &RAW[start..start.checked_add(len).unwrap()];
    assert_eq!(d["sha256"], format!("{:x}", Sha256::digest(raw)));
    raw
}
fn words16(d: &Value) -> Vec<u16> {
    bytes(d, "u16le", 2)
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn words32(d: &Value) -> Vec<u32> {
    bytes(d, "u32le", 4)
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn json16(v: &Value) -> Vec<u16> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|w| u16::try_from(w.as_u64().unwrap()).unwrap())
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
fn case_plan(c: &Value) -> Plan {
    Plan::new(
        c["rows"].as_u64().unwrap() as usize,
        c["width"].as_u64().unwrap() as usize,
        c["input_stride"].as_u64().unwrap() as usize,
        c["output_stride"].as_u64().unwrap() as usize,
    )
    .unwrap()
}
fn active(plan: Plan, output: &[u16]) -> Vec<u16> {
    (0..plan.rows)
        .flat_map(|r| {
            output[r * plan.output_stride..r * plan.output_stride + plan.width]
                .iter()
                .copied()
        })
        .collect()
}
fn suffix(
    original_old: &[u16],
    input: &[u16],
    width: usize,
    stride: usize,
    end: usize,
) -> Vec<u16> {
    // Independent absolute row selection, not the shader's row/T branch.
    let mut history = original_old[..2 * width].to_vec();
    for row in 0..end {
        history.extend_from_slice(&input[row * stride..row * stride + width]);
    }
    history[history.len() - 2 * width..].to_vec()
}

#[test]
fn shape_bounds_checked_extents_and_accounting() {
    for h in [5, 256, 1024] {
        for n in [1, 2, 3, 32] {
            let p = Plan::new(n, h, h + 3, h + 5).unwrap();
            assert_eq!(p.input_len(), n * (h + 3) + 7);
            assert_eq!(p.output_len(), n * (h + 5) + 7);
            assert_eq!(p.history_len(), 2 * h + 7);
            assert_eq!(p.scale_len(), 3 * h + 7);
            assert_eq!(
                p.operand_bytes().unwrap(),
                2 * (p.input_len() + p.output_len() + 2 * p.history_len())
                    + 4 * (p.scale_len() + p.status_len())
            );
        }
    }
    for (n, h) in [(0, 5), (33, 5), (1, 4), (1, 64), (1, 2048)] {
        assert!(Plan::new(n, h, h, h).is_err());
    }
    for (a, b) in [
        (4, 5),
        (5, 4),
        (usize::MAX, 5),
        (5, usize::MAX),
        (u32::MAX as usize, 5),
        (5, u32::MAX as usize),
    ] {
        assert!(Plan::new(32, 5, a, b).is_err());
    }
}
#[test]
fn frozen_coefficients_stores_state_partitions_and_gates() {
    let m = fixture();
    let (mut values, mut candidates, mut schedules, mut boundaries) = (0, 0, 0, 0);
    for c in m["cases"].as_array().unwrap() {
        let p = case_plan(c);
        let a = &c["arrays"];
        let input = words16(&a["u"]);
        let old = words16(&a["old"]);
        let gamma = words16(&a["gamma"]);
        let scales = words32(&a["scales"]);
        assert_eq!(input.len(), p.input_len());
        assert_eq!(old.len(), p.history_len());
        assert_eq!(scales.len(), p.scale_len());
        assert_eq!(gamma.len(), 3 * p.width);
        for (i, &word) in gamma.iter().enumerate() {
            assert_eq!(scales[i], bf16::from_bits(word).to_f32().to_bits());
        }
        assert!(scales[3 * p.width..].iter().all(|&v| v == 0x5a170031));
        for r in 0..p.rows {
            assert!(
                input[r * p.input_stride + p.width..(r + 1) * p.input_stride]
                    .iter()
                    .all(|&v| v == 0x7e55)
            );
        }
        assert!(
            input[p.rows * p.input_stride..]
                .iter()
                .all(|&v| v == 0x7e55)
        );
        assert!(old[2 * p.width..].iter().all(|&v| v == 0x5a17));
        let main = words16(&a["main"]);
        let wide = words16(&a["wide"]);
        let candidate = words16(&a["candidate"]);
        assert_eq!(main.len(), p.rows * p.width);
        assert_eq!(wide.len(), main.len());
        assert_eq!(
            candidate,
            suffix(&old, &input, p.width, p.input_stride, p.rows)
        );
        let mut env = 0f64;
        for (&x, &y) in main.iter().zip(&wide) {
            let x = f64::from(f16::from_bits(x).to_f32());
            let y = f64::from(f16::from_bits(y).to_f32());
            assert!(x.is_finite() && y.is_finite());
            env = env.max((x - y).abs());
        }
        assert_eq!(c["envelope"].as_f64(), Some(env));
        assert_eq!(c["gate"]["atol"].as_f64(), Some((2. * env).max(1e-4)));
        assert_eq!(c["gate"]["rtol"].as_f64(), Some(1e-3));
        let exact = json16(&c["exact_words"]);
        if !exact.is_empty() {
            assert_eq!(main, exact);
            assert_eq!(wide, exact);
        }
        for schedule in c["partitions"].as_array().unwrap() {
            let mut at = 0;
            let mut before = old[..2 * p.width].to_vec();
            for (n, b) in schedule["chunks"]
                .as_array()
                .unwrap()
                .iter()
                .zip(schedule["boundaries"].as_array().unwrap())
            {
                at += n.as_u64().unwrap() as usize;
                let after = suffix(&old, &input, p.width, p.input_stride, at);
                assert_eq!(b["end"], at);
                assert_eq!(b["before"]["sha256"], hash16(&before));
                assert_eq!(b["after"]["sha256"], hash16(&after));
                before = after;
                boundaries += 1;
            }
            assert_eq!(at, p.rows);
            schedules += 1;
        }
        values += main.len();
        candidates += candidate.len();
    }
    assert_eq!(
        (values, candidates, schedules, boundaries),
        (46571, 7238, 35, 233)
    );
}

struct Operands {
    plan: Plan,
    input: Buffer<u16>,
    old: Buffer<u16>,
    scales: Buffer<u32>,
    output: Buffer<u16>,
    candidate: Buffer<u16>,
    status: Buffer<u32>,
    input_hash: String,
    old_hash: String,
    scale_hash: String,
}
struct Capture {
    output: Vec<u16>,
    candidate: Vec<u16>,
    status: Vec<u32>,
    input_hash: String,
    old_hash: String,
    scale_hash: String,
}
impl Operands {
    fn new(gpu: &Gpu, plan: Plan, input: &[u16], old: &[u16], scales: &[u32]) -> Self {
        assert_eq!(input.len(), plan.input_len());
        assert_eq!(old.len(), plan.history_len());
        assert_eq!(scales.len(), plan.scale_len());
        let mut x = Buffer::new(gpu, input.len()).unwrap();
        x.with_mut(|v| v.copy_from_slice(input));
        let mut o = Buffer::new(gpu, old.len()).unwrap();
        o.with_mut(|v| v.copy_from_slice(old));
        let mut w = Buffer::new(gpu, scales.len()).unwrap();
        w.with_mut(|v| v.copy_from_slice(scales));
        let mut y = Buffer::new(gpu, plan.output_len()).unwrap();
        y.with_mut(|v| v.fill(0x5a17));
        let mut c = Buffer::new(gpu, plan.history_len()).unwrap();
        c.with_mut(|v| v.fill(0x5a17));
        let mut s = Buffer::new(gpu, plan.status_len()).unwrap();
        s.with_mut(|v| {
            v.fill(0x5a170031);
            v[0] = 0;
        });
        assert_eq!(
            x.raw.length()
                + o.raw.length()
                + w.raw.length()
                + y.raw.length()
                + c.raw.length()
                + s.raw.length(),
            plan.operand_bytes().unwrap()
        );
        Self {
            plan,
            input: x,
            old: o,
            scales: w,
            output: y,
            candidate: c,
            status: s,
            input_hash: hash16(input),
            old_hash: hash16(old),
            scale_hash: hash32(scales),
        }
    }
    fn invocation(&self) -> Invocation<'_> {
        Invocation {
            input: &self.input,
            old: &self.old,
            scales: &self.scales,
            output: &self.output,
            candidate: &self.candidate,
            status: &self.status,
        }
    }
    fn record(&self, label: &str, command: &Value) -> Capture {
        let c = Capture {
            output: self.output.with_slice(|v| v.to_vec()),
            candidate: self.candidate.with_slice(|v| v.to_vec()),
            status: self.status.with_slice(|v| v.to_vec()),
            input_hash: self.input.with_slice(hash16),
            old_hash: self.old.with_slice(hash16),
            scale_hash: self.scales.with_slice(hash32),
        };
        println!(
            "CONV3_RAW {}",
            json!({"label":label,"command":command,"rows":self.plan.rows,"width":self.plan.width,"input_stride":self.plan.input_stride,"output_stride":self.plan.output_stride,"input_before_sha256":self.input_hash,"input_after_sha256":c.input_hash,"old_before_sha256":self.old_hash,"old_after_sha256":c.old_hash,"scales_before_sha256":self.scale_hash,"scales_after_sha256":c.scale_hash,"output_words":c.output,"candidate_words":c.candidate,"status_words":c.status,"operand_bytes":self.plan.operand_bytes().unwrap()})
        );
        c
    }
    fn validate(&self, c: &Capture) {
        assert_eq!(c.input_hash, self.input_hash);
        assert_eq!(c.old_hash, self.old_hash);
        assert_eq!(c.scale_hash, self.scale_hash);
        for (i, &w) in c.output.iter().enumerate() {
            if i >= self.plan.rows * self.plan.output_stride
                || i % self.plan.output_stride >= self.plan.width
            {
                assert_eq!(w, 0x5a17, "output canary {i}");
            }
        }
        assert!(
            c.candidate[2 * self.plan.width..]
                .iter()
                .all(|&w| w == 0x5a17)
        );
        assert!(c.status[1..].iter().all(|&w| w == 0x5a170031));
    }
}
fn completed(raw: &ProtocolObject<dyn MTLCommandBuffer>) -> Value {
    let status = raw.status();
    let error = raw.error().map(|e| e.to_string());
    assert!(matches!(
        status,
        MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
    ));
    json!({"completed":status==MTLCommandBufferStatus::Completed&&error.is_none(),"status":status.0,"error":error})
}
fn run(gpu: &Gpu, op: &Operation, io: &Operands, label: &str) -> Capture {
    let raw = gpu.command().unwrap();
    let c = Command {
        raw: &raw,
        trace: None,
    };
    op.encode(&c, io.plan, io.invocation()).unwrap();
    raw.commit();
    raw.waitUntilCompleted();
    let done = completed(&raw);
    let capture = io.record(label, &done);
    io.validate(&capture);
    assert_eq!(done["completed"], true);
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
    fn compare(&mut self, c: &Value, p: Plan, out: &[u16], start: usize) {
        let main = words16(&c["arrays"]["main"]);
        let wide = words16(&c["arrays"]["wide"]);
        let exact = json16(&c["exact_words"]);
        let atol = c["gate"]["atol"].as_f64().unwrap();
        let rtol = c["gate"]["rtol"].as_f64().unwrap();
        for row in 0..p.rows {
            for col in 0..p.width {
                let i = (start + row) * p.width + col;
                let word = out[row * p.output_stride + col];
                let a = f64::from(f16::from_bits(word).to_f32());
                let b = f64::from(f16::from_bits(main[i]).to_f32());
                let w = f64::from(f16::from_bits(wide[i]).to_f32());
                assert!(a.is_finite() && b.is_finite() && w.is_finite());
                let e = (a - b).abs();
                let limit = atol + rtol * b.abs();
                let we = (a - w).abs();
                let wl = atol + rtol * w.abs();
                self.values += 1;
                self.max_error = self.max_error.max(e);
                self.max_fraction = self.max_fraction.max(e / limit);
                self.wide_exceedances += usize::from(we > wl);
                self.wide_max_error = self.wide_max_error.max(we);
                self.wide_max_fraction = self.wide_max_fraction.max(we / wl);
                assert!(
                    e <= limit,
                    "{} row{} col{col} native{word:04x} main{:04x} wide{:04x} error{e} limit{limit}",
                    c["name"],
                    start + row,
                    main[i],
                    wide[i]
                );
                if !exact.is_empty() {
                    assert_eq!(word, exact[i]);
                }
            }
        }
    }
}
fn preflight(gpu: &Gpu, op: &Operation) {
    let p = Plan::new(2, 5, 5, 5).unwrap();
    let io = Operands::new(
        gpu,
        p,
        &vec![0; p.input_len()],
        &vec![0; p.history_len()],
        &vec![1f32.to_bits(); p.scale_len()],
    );
    let raw = gpu.command().unwrap();
    let command = Command {
        raw: &raw,
        trace: None,
    };
    // All four Half allocations have identical extents, so these hit alias,
    // not a preceding size rejection. The u32 status borrows the larger taps.
    for v in [
        Invocation {
            output: &io.input,
            ..io.invocation()
        },
        Invocation {
            old: &io.input,
            ..io.invocation()
        },
        Invocation {
            candidate: &io.input,
            ..io.invocation()
        },
        Invocation {
            output: &io.old,
            ..io.invocation()
        },
        Invocation {
            candidate: &io.old,
            ..io.invocation()
        },
        Invocation {
            candidate: &io.output,
            ..io.invocation()
        },
        Invocation {
            status: &io.scales,
            ..io.invocation()
        },
    ] {
        assert!(matches!(
            op.encode(&command, p, v),
            Err(Error::Options("Half Conv3 operand alias"))
        ));
    }
    let x = Buffer::new(gpu, p.input_len() - 1).unwrap();
    let o = Buffer::new(gpu, p.history_len() - 1).unwrap();
    let w = Buffer::new(gpu, p.scale_len() - 1).unwrap();
    let y = Buffer::new(gpu, p.output_len() - 1).unwrap();
    let n = Buffer::new(gpu, p.history_len() - 1).unwrap();
    let s = Buffer::new(gpu, p.status_len() - 1).unwrap();
    for v in [
        Invocation {
            input: &x,
            ..io.invocation()
        },
        Invocation {
            old: &o,
            ..io.invocation()
        },
        Invocation {
            scales: &w,
            ..io.invocation()
        },
        Invocation {
            output: &y,
            ..io.invocation()
        },
        Invocation {
            candidate: &n,
            ..io.invocation()
        },
        Invocation {
            status: &s,
            ..io.invocation()
        },
    ] {
        assert!(matches!(
            op.encode(&command, p, v),
            Err(Error::Options("Half Conv3 operand extent"))
        ));
    }
    println!(
        "CONV3_PREFLIGHT {}",
        json!({"equal_extent_half_aliases":6,"u32_aliases":1,"short_operands":6,"pipeline_budget_exact_minus_one":true})
    );
}

#[test]
#[ignore = "explicit Apple GPU raw Conv3 source/strides/chunks/exact candidate qualification"]
fn gpu_source_partitions_candidate_and_zero_allocation() {
    autoreleasepool(|_| {
        let m = fixture();
        let gpu = Gpu::new().unwrap();
        let required = PIPELINE_ALLOWANCE + size_of::<Operation>();
        assert!(
            matches!(Operation::new(&gpu,required-1),Err(Error::Budget{required:r,limit})if r==required&&limit==required-1)
        );
        let op = Operation::new(&gpu, required).unwrap();
        preflight(&gpu, &op);
        let mut metrics = Metrics::default();
        let mut partition_metrics = Metrics::default();
        let (mut allocations, mut boundaries, mut schedules, mut candidate_words) = (0, 0, 0, 0);
        for c in m["cases"].as_array().unwrap() {
            let p = case_plan(c);
            let name = c["name"].as_str().unwrap();
            let input = words16(&c["arrays"]["u"]);
            let old = words16(&c["arrays"]["old"]);
            let scales = words32(&c["arrays"]["scales"]);
            let tight = Plan::new(p.rows, p.width, p.width, p.width).unwrap();
            let mut ti = vec![0x7e55; tight.input_len()];
            for r in 0..p.rows {
                ti[r * p.width..(r + 1) * p.width]
                    .copy_from_slice(&input[r * p.input_stride..r * p.input_stride + p.width]);
            }
            let io = Operands::new(&gpu, p, &input, &old, &scales);
            let other = Operands::new(&gpu, tight, &ti, &old, &scales);
            let raw = gpu.command().unwrap();
            let command = Command {
                raw: &raw,
                trace: None,
            };
            let mut counts = [0; 2];
            for (i, owner) in [&io, &other].into_iter().enumerate() {
                let measured = Measurement::start();
                let result = op.encode(&command, owner.plan, owner.invocation());
                counts[i] = measured.finish();
                println!(
                    "CONV3_ALLOCATION {}",
                    json!({"case":name,"layout":i,"rust_allocations":counts[i],"encode_ok":result.is_ok(),"release_required_zero":!cfg!(debug_assertions)})
                );
                result.unwrap();
            }
            raw.commit();
            raw.waitUntilCompleted();
            let done = completed(&raw);
            let a = io.record(&format!("{name}/strided"), &done);
            let b = other.record(&format!("{name}/tight"), &done);
            // Both physical captures and allocation counts precede every gate.
            io.validate(&a);
            other.validate(&b);
            assert_eq!(done["completed"], true);
            assert_eq!((a.status[0], b.status[0]), (0, 0));
            if !cfg!(debug_assertions) {
                assert_eq!(counts, [0, 0]);
            }
            allocations += 2;
            let baseline = active(p, &a.output);
            assert_eq!(baseline, active(tight, &b.output));
            let expected = words16(&c["arrays"]["candidate"]);
            assert_eq!(&a.candidate[..2 * p.width], expected);
            assert_eq!(&b.candidate[..2 * p.width], expected);
            candidate_words += 4 * p.width;
            metrics.compare(c, p, &a.output, 0);
            metrics.compare(c, tight, &b.output, 0);
            for (schedule_index, schedule) in c["partitions"].as_array().unwrap().iter().enumerate()
            {
                let mut state = old.clone();
                let mut at = 0;
                let mut combined = Vec::new();
                for (chunk_index, count) in
                    schedule["chunks"].as_array().unwrap().iter().enumerate()
                {
                    let count = count.as_u64().unwrap() as usize;
                    let q = Plan::new(count, p.width, p.input_stride, p.output_stride).unwrap();
                    let mut u = vec![0x7e55; q.input_len()];
                    for r in 0..count {
                        u[r * q.input_stride..r * q.input_stride + p.width].copy_from_slice(
                            &input[(at + r) * p.input_stride..(at + r) * p.input_stride + p.width],
                        );
                    }
                    let boundary = &schedule["boundaries"][chunk_index];
                    assert_eq!(boundary["before"]["sha256"], hash16(&state[..2 * p.width]));
                    let chunk = Operands::new(&gpu, q, &u, &state, &scales);
                    let got = run(
                        &gpu,
                        &op,
                        &chunk,
                        &format!("{name}/partition{schedule_index}/chunk{chunk_index}/start{at}"),
                    );
                    assert_eq!(got.status[0], 0);
                    partition_metrics.compare(c, q, &got.output, at);
                    combined.extend(active(q, &got.output));
                    at += count;
                    let expected = suffix(&old, &input, p.width, p.input_stride, at);
                    assert_eq!(&got.candidate[..2 * p.width], expected);
                    assert_eq!(boundary["after"]["sha256"], hash16(&expected));
                    state[..2 * p.width].copy_from_slice(&got.candidate[..2 * p.width]);
                    candidate_words += 2 * p.width;
                    boundaries += 1;
                }
                assert_eq!(at, p.rows);
                assert_eq!(combined, baseline, "same-native partition exact {name}");
                schedules += 1;
            }
        }
        assert_eq!(
            (
                metrics.values,
                partition_metrics.values,
                allocations,
                schedules,
                boundaries,
                candidate_words
            ),
            (93142, 219388, 24, 35, 233, 200750)
        );
        println!(
            "CONV3_RESULT {}",
            json!({"device":gpu.device.name().to_string(),"cases":12,"normal_main_values":metrics.values,"partition_main_values":partition_metrics.values,"main_failures":0,"normal_max_error":metrics.max_error,"normal_max_fraction":metrics.max_fraction,"partition_max_error":partition_metrics.max_error,"partition_max_fraction":partition_metrics.max_fraction,"wide_diagnostic_values":metrics.values+partition_metrics.values,"wide_exceedances":metrics.wide_exceedances+partition_metrics.wide_exceedances,"wide_max_error":metrics.wide_max_error.max(partition_metrics.wide_max_error),"wide_max_fraction":metrics.wide_max_fraction.max(partition_metrics.wide_max_fraction),"wide_fatal":false,"candidate_exact_words":candidate_words,"schedules":schedules,"boundaries":boundaries,"rust_encode_measurements":allocations,"release_required_zero":!cfg!(debug_assertions),"resident_transaction_qualified":false})
        );
        // Future-only changed final input: actual native pair, not fabricated rows.
        let c = &m["cases"][0];
        let p = case_plan(c);
        let input = words16(&c["arrays"]["u"]);
        let old = words16(&c["arrays"]["old"]);
        let scales = words32(&c["arrays"]["scales"]);
        let mut changed = input.clone();
        changed[(p.rows - 1) * p.input_stride..(p.rows - 1) * p.input_stride + p.width]
            .fill(f16::from_f32(8.).to_bits());
        let left = Operands::new(&gpu, p, &input, &old, &scales);
        let right = Operands::new(&gpu, p, &changed, &old, &scales);
        let a = run(&gpu, &op, &left, "causal/baseline");
        let b = run(&gpu, &op, &right, "causal/changed_last");
        assert_eq!((a.status[0], b.status[0]), (0, 0));
        assert_eq!(
            &active(p, &a.output)[..(p.rows - 1) * p.width],
            &active(p, &b.output)[..(p.rows - 1) * p.width]
        );
        assert_ne!(active(p, &a.output), active(p, &b.output));
        assert!(
            active(p, &b.output)
                .iter()
                .all(|&w| f16::from_bits(w).is_finite())
        );
        assert_eq!(
            &a.candidate[..2 * p.width],
            suffix(&old, &input, p.width, p.input_stride, p.rows)
        );
        assert_eq!(
            &b.candidate[..2 * p.width],
            suffix(&old, &changed, p.width, p.input_stride, p.rows)
        );
        println!(
            "CONV3_CAUSAL_RESULT {}",
            json!({"native_equal_prefix_words":20,"changed_last_observed":true,"candidate_exact_words":20,"submitted_commands":2})
        );
    });
}

#[test]
#[ignore = "explicit Apple GPU raw Conv3 invalid operands/sticky status/reset recovery"]
fn gpu_invalid_sticky_and_recovery() {
    autoreleasepool(|_| {
        let m = fixture();
        let gpu = Gpu::new().unwrap();
        let op = Operation::new(&gpu, PIPELINE_ALLOWANCE + size_of::<Operation>()).unwrap();
        let errors: Value = serde_json::from_slice(ERRORS).unwrap();
        assert_eq!(errors["controls"].as_array().unwrap().len(), 9);
        let baseline = &m["cases"][10];
        assert_eq!(baseline["name"], "stores");
        let p = case_plan(baseline);
        let input = words16(&baseline["arrays"]["u"]);
        let old = words16(&baseline["arrays"]["old"]);
        let scales = words32(&baseline["arrays"]["scales"]);
        let expected = words16(&baseline["arrays"]["candidate"]);
        let mut metrics = Metrics::default();
        for error in errors["controls"].as_array().unwrap() {
            let bad = json16(&error["u_words"]);
            let bad_old = json16(&error["old_words"]);
            let gamma = json16(&error["gamma_words"]);
            let mut coeff: Vec<u32> = gamma
                .iter()
                .map(|&w| bf16::from_bits(w).to_f32().to_bits())
                .collect();
            coeff.extend([0x5a170031; GUARD]);
            let mut io = Operands::new(&gpu, p, &bad, &bad_old, &coeff);
            let name = error["name"].as_str().unwrap();
            let failed = run(&gpu, &op, &io, &format!("{name}/invalid"));
            assert_ne!(failed.status[0], 0);
            // No committed owner publication. All mutation is terminal scratch repair.
            io.input.with_mut(|v| v.copy_from_slice(&input));
            io.old.with_mut(|v| v.copy_from_slice(&old));
            io.scales.with_mut(|v| v.copy_from_slice(&scales));
            io.input_hash = hash16(&input);
            io.old_hash = hash16(&old);
            io.scale_hash = hash32(&scales);
            for reset in [false, true] {
                if reset {
                    io.status.with_mut(|v| v[0] = 0);
                }
                io.output.with_mut(|v| v.fill(0x5a17));
                io.candidate.with_mut(|v| v.fill(0x5a17));
                let got = run(
                    &gpu,
                    &op,
                    &io,
                    &format!(
                        "{name}/{}",
                        if reset { "valid_reset" } else { "valid_sticky" }
                    ),
                );
                assert_eq!(got.status[0], if reset { 0 } else { failed.status[0] });
                metrics.compare(baseline, p, &got.output, 0);
                assert_eq!(&got.candidate[..2 * p.width], expected);
            }
        }
        assert_eq!(metrics.values, 90);
        println!(
            "CONV3_RECOVERY_RESULT {}",
            json!({"invalid_sites":9,"submitted_phases":27,"valid_recovery_values":metrics.values,"main_failures":0,"max_error":metrics.max_error,"max_fraction":metrics.max_fraction,"wide_exceedances":metrics.wide_exceedances,"candidate_exact_words":180,"invalid_output_bits_are_oracle":false,"partial_overflow_status_does_not_isolate_intermediate_check":true})
        );
    });
}
