use super::*;
use crate::model::gemma_reference::tests::Measurement;
use half::f16;
use objc2::rc::autoreleasepool;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::Write;

const RAW: &[u8] = include_bytes!("../../../tests/fixtures/gemma3-staged-attention-v1/arrays.bin");
const HALF_PAD: u16 = 0x5a17;
const FLOAT_PAD: u32 = 0x5a170031;
const INPUTS: [&str; 5] = ["q", "old_k", "old_v", "new_k", "new_v"];
fn number(v: &Value) -> usize {
    usize::try_from(v.as_u64().unwrap()).unwrap()
}
fn fixture() -> Value {
    let bytes = include_bytes!("../../../tests/fixtures/gemma3-staged-attention-v1/manifest.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "90656e1a724b1bfb8d3c14fd742166348d5abf64dedbaf3f8d83645e5e5ca7cb"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(RAW)),
        "12339d584e5d67602d386b75fcdca2dcfe62087ef1f4ac13eb0a5109525c150e"
    );
    let m: Value = serde_json::from_slice(bytes).unwrap();
    assert_eq!(m["format"], "toklet-gemma3-staged-attention-v1");
    assert_eq!(
        m["protocol_sha256"],
        "7b4508605e7432c39fd70ce409ad4dd658541054361b2e02937acfa02055e30f"
    );
    assert_eq!(
        m["producer"]["sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../../../tools/reference/gemma3_staged_attention.py"
            ))
        )
    );
    assert_eq!(m["source_reference_executed"], true);
    assert_eq!(m["target_gpu_executed"], false);
    assert_eq!(m["full_graph_executed"], false);
    assert_eq!(m["policy"]["fatal_reference"], "main");
    assert_eq!(m["policy"]["atol_floor"].as_f64(), Some(1e-4));
    assert_eq!(m["policy"]["envelope_factor"], 2);
    assert_eq!(m["policy"]["rtol"].as_f64(), Some(1e-3));
    assert_eq!(number(&m["files"]["arrays.bin"]["bytes"]), RAW.len());
    assert_eq!(
        m["files"]["arrays.bin"]["sha256"],
        format!("{:x}", Sha256::digest(RAW))
    );
    assert_eq!(m["stored_arrays"], 104);
    assert_eq!(m["numerical_profile_arrays"], 48);
    assert_eq!(m["physical_state_arrays"], 56);
    assert_eq!(m["guard"], GUARD);
    assert_eq!(m["cases"].as_array().unwrap().len(), 8);
    m
}
fn bytes(d: &Value) -> &[u8] {
    assert_eq!(d["file"], "arrays.bin");
    let start = number(&d["offset"]);
    let length = number(&d["bytes"]);
    let b = &RAW[start..start.checked_add(length).unwrap()];
    assert_eq!(d["sha256"], format!("{:x}", Sha256::digest(b)));
    b
}
fn words(d: &Value) -> Vec<u16> {
    assert_eq!(d["dtype"], "u16le");
    bytes(d)
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn values(d: &Value) -> Vec<f64> {
    match d["dtype"].as_str().unwrap() {
        "u16le" => words(d)
            .iter()
            .map(|&v| f64::from(f16::from_bits(v).to_f32()))
            .collect(),
        "f32le" => bytes(d)
            .chunks_exact(4)
            .map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        other => panic!("unexpected fixture type {other}"),
    }
}
fn hash(v: &[u16]) -> String {
    let mut h = Sha256::new();
    for word in v {
        h.update(word.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn plan(c: &Value) -> Plan {
    Plan::new(
        number(&c["capacity"]),
        number(&c["dim"]),
        c["window"].as_u64().map(|n| usize::try_from(n).unwrap()),
        number(&c["committed"]),
        number(&c["tokens"]),
    )
    .unwrap()
}
fn input_lengths(p: Plan) -> [usize; 5] {
    [
        p.q_len(),
        p.old_len(),
        p.old_len(),
        p.staged_len(),
        p.staged_len(),
    ]
}
fn source_inputs(c: &Value) -> [Vec<u16>; 5] {
    INPUTS.map(|name| words(&c["inputs"][name]))
}
#[test]
fn checked_plan_indices_spans_and_bytes() {
    for capacity in [1, 12, 544] {
        for dim in [32, 256] {
            for window in [None, Some(1), Some(capacity)] {
                for committed in [0, capacity / 2, capacity - 1] {
                    let tokens = (capacity - committed).min(CHUNK);
                    let p = Plan::new(capacity, dim, window, committed, tokens).unwrap();
                    let start = window.map_or(0, |w| committed.saturating_sub(w - 1));
                    assert_eq!(p.start, start);
                    assert_eq!(p.keys, committed + tokens - start);
                    let actual: Vec<_> = p
                        .old
                        .into_iter()
                        .flat_map(|s| (0..s.rows).map(move |r| (s.position + r, s.slot + r)))
                        .collect();
                    let expected: Vec<_> = (start..committed)
                        .map(|pos| (pos, window.map_or(pos, |w| pos % w)))
                        .collect();
                    assert_eq!(actual, expected);
                    let expected_bytes = (2 * p.retained * dim
                        + 2 * CHUNK * HEADS * dim
                        + 2 * CHUNK * dim
                        + 6 * GUARD)
                        * 2
                        + (2 * HEADS * CHUNK * capacity + 2 * GUARD + 1 + GUARD) * 4;
                    assert_eq!(p.operand_bytes().unwrap(), expected_bytes);
                    assert!(HEADS * CHUNK * capacity < u32::MAX as usize);
                    assert!(tokens * HEADS * dim <= p.q_len() - GUARD);
                    for span in p.old {
                        assert!((span.slot + span.rows) * dim <= p.old_len() - GUARD);
                    }
                    assert!(p.keys <= capacity);
                    assert!(
                        (HEADS - 1) * CHUNK * capacity + (tokens - 1) * capacity + p.keys
                            <= p.score_len() - GUARD
                    );
                }
            }
        }
    }
    for (capacity, dim, window, committed, tokens) in [
        (0, 32, None, 0, 1),
        (545, 32, None, 0, 1),
        (12, 64, None, 0, 1),
        (12, 32, Some(0), 0, 1),
        (12, 32, Some(13), 0, 1),
        (12, 32, None, 0, 0),
        (544, 256, None, 0, 33),
        (12, 32, None, 12, 1),
        (12, 32, Some(4), usize::MAX, 1),
    ] {
        assert!(Plan::new(capacity, dim, window, committed, tokens).is_err());
    }
    let w1 = Plan::new(12, 32, Some(1), 5, 7).unwrap();
    assert_eq!(
        (w1.start, w1.keys, w1.old[0].rows + w1.old[1].rows),
        (5, 7, 0)
    );
}
#[test]
fn frozen_arrays_policy_and_source_geometry() {
    let m = fixture();
    let mut extents = Vec::new();
    let mut total = 0;
    for c in m["cases"].as_array().unwrap() {
        let p = plan(c);
        assert_eq!(c["query_heads"], HEADS);
        assert_eq!(c["kv_heads"], 1);
        assert_eq!(c["chunk_capacity"], CHUNK);
        assert_eq!(number(&c["view"]["start"]), p.start);
        assert_eq!(number(&c["view"]["end"]), p.committed + p.tokens);
        let spans: Vec<_> = p
            .old
            .iter()
            .filter(|s| s.rows != 0)
            .map(|s| json!({"absolute_start":s.position,"slot":s.slot,"rows":s.rows}))
            .collect();
        assert_eq!(c["view"]["old_spans"], json!(spans));
        for (name, length) in INPUTS.into_iter().zip(input_lengths(p)).chain([
            ("expected_after_k", p.old_len()),
            ("expected_after_v", p.old_len()),
        ]) {
            let d = &c["inputs"][name];
            assert_eq!(words(d).len(), length);
            assert_eq!(d["shape"], json!([length]));
            extents.push((number(&d["offset"]), number(&d["bytes"])));
        }
        for stage in ["scores", "probabilities", "context"] {
            let d = &c["arrays"][stage];
            let main = values(&d["main"]);
            let wide = values(&d["wide"]);
            let shape = if stage == "context" {
                vec![p.tokens, HEADS, p.dim]
            } else {
                vec![HEADS, p.tokens, p.keys]
            };
            assert_eq!(main.len(), shape.iter().product::<usize>());
            assert_eq!(main.len(), wide.len());
            assert!(main.iter().chain(&wide).all(|v| v.is_finite()));
            let envelope = main
                .iter()
                .zip(&wide)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert_eq!(d["envelope"].as_f64(), Some(envelope));
            assert_eq!(d["gate"]["atol"].as_f64(), Some((2.0 * envelope).max(1e-4)));
            assert_eq!(d["gate"]["rtol"].as_f64(), Some(1e-3));
            for profile in ["main", "wide"] {
                let a = &d[profile];
                assert_eq!(a["shape"], json!(shape));
                assert_eq!(
                    a["dtype"],
                    if stage == "context" { "u16le" } else { "f32le" }
                );
                extents.push((number(&a["offset"]), number(&a["bytes"])));
            }
            total += main.len();
        }
    }
    extents.sort_unstable();
    assert_eq!(extents.len(), 104);
    let mut end = 0;
    for (start, count) in extents {
        assert_eq!(start, end);
        end += count;
    }
    assert_eq!(end, RAW.len());
    assert_eq!(total, 361448);
}

fn half_buffer(gpu: &Gpu, input: &[u16]) -> Buffer<u16> {
    let mut b = Buffer::new(gpu, input.len()).unwrap();
    b.with_mut(|v| v.copy_from_slice(input));
    b
}
fn float_buffer(gpu: &Gpu, len: usize) -> Buffer<u32> {
    let mut b = Buffer::new(gpu, len).unwrap();
    b.with_mut(|v| v.fill(FLOAT_PAD));
    b
}
struct Io {
    p: Plan,
    inputs: [Buffer<u16>; 5],
    before: [String; 5],
    scores: Buffer<u32>,
    probabilities: Buffer<u32>,
    context: Buffer<u16>,
    status: Buffer<u32>,
}
struct Capture {
    scores: Vec<u32>,
    probabilities: Vec<u32>,
    context: Vec<u16>,
    status: Vec<u32>,
    after: [String; 5],
    completed: bool,
}
impl Io {
    fn new(gpu: &Gpu, p: Plan, input: &[Vec<u16>; 5]) -> Self {
        for (v, n) in input.iter().zip(input_lengths(p)) {
            assert_eq!(v.len(), n);
        }
        let mut status = float_buffer(gpu, p.status_len());
        status.with_mut(|v| v[0] = 0);
        let io = Self {
            p,
            inputs: input.each_ref().map(|v| half_buffer(gpu, v)),
            before: input.each_ref().map(|v| hash(v)),
            scores: float_buffer(gpu, p.score_len()),
            probabilities: float_buffer(gpu, p.score_len()),
            context: half_buffer(gpu, &vec![HALF_PAD; p.context_len()]),
            status,
        };
        assert_eq!(
            io.inputs.iter().map(|b| b.raw.length()).sum::<usize>()
                + io.scores.raw.length()
                + io.probabilities.raw.length()
                + io.context.raw.length()
                + io.status.raw.length(),
            p.operand_bytes().unwrap()
        );
        io
    }
    fn invocation(&self) -> Invocation<'_> {
        Invocation {
            q: &self.inputs[0],
            old_k: &self.inputs[1],
            old_v: &self.inputs[2],
            new_k: &self.inputs[3],
            new_v: &self.inputs[4],
            scores: &self.scores,
            probabilities: &self.probabilities,
            context: &self.context,
            status: &self.status,
        }
    }
    fn reset_outputs(&mut self) {
        self.scores.with_mut(|v| v.fill(FLOAT_PAD));
        self.probabilities.with_mut(|v| v.fill(FLOAT_PAD));
        self.context.with_mut(|v| v.fill(HALF_PAD));
    }
    fn replace_inputs(&mut self, input: &[Vec<u16>; 5]) {
        for (b, v) in self.inputs.iter_mut().zip(input) {
            b.with_mut(|x| x.copy_from_slice(v));
        }
        self.before = input.each_ref().map(|v| hash(v));
    }
    // No assertions here: serialize every available raw output and input identity first.
    fn record(&self, label: &str, raw: &ProtocolObject<dyn MTLCommandBuffer>) -> Capture {
        let state = raw.status();
        let error = raw.error().map(|e| e.to_string());
        let c = Capture {
            scores: self.scores.with_slice(|v| v.to_vec()),
            probabilities: self.probabilities.with_slice(|v| v.to_vec()),
            context: self.context.with_slice(|v| v.to_vec()),
            status: self.status.with_slice(|v| v.to_vec()),
            after: self.inputs.each_ref().map(|b| b.with_slice(hash)),
            completed: state == MTLCommandBufferStatus::Completed && error.is_none(),
        };
        println!(
            "STAGED_ATTENTION_RAW {}",
            json!({"label":label,"completion":{"completed":c.completed,"status":format!("{state:?}"),"error":error},
            "capacity":self.p.capacity,"dim":self.p.dim,"window":self.p.window,"committed":self.p.committed,"tokens":self.p.tokens,"view_start":self.p.start,"keys":self.p.keys,
            "input_names":INPUTS,"input_before_sha256":self.before,"input_after_sha256":c.after,"scores_u32":c.scores,"probabilities_u32":c.probabilities,"context_u16":c.context,"status_u32":c.status,
            "operand_bytes":self.p.operand_bytes().unwrap(),"cache_copy_executed":false})
        );
        std::io::stdout().flush().unwrap();
        c
    }
    fn validate(&self, c: &Capture) {
        assert!(c.completed);
        assert_eq!(c.after, self.before);
        assert!(c.status[1..].iter().all(|&v| v == FLOAT_PAD));
        assert!(
            c.context[self.p.tokens * HEADS * self.p.dim..]
                .iter()
                .all(|&v| v == HALF_PAD)
        );
        for output in [&c.scores, &c.probabilities] {
            for (i, &word) in output.iter().enumerate() {
                let active = i < HEADS * CHUNK * self.p.capacity
                    && (i / self.p.capacity) % CHUNK < self.p.tokens
                    && i % self.p.capacity < self.p.keys;
                if !active {
                    assert_eq!(word, FLOAT_PAD, "inactive score/probability element {i}");
                }
            }
        }
    }
}
fn active(p: Plan, c: &Capture, stage: &str) -> Vec<f64> {
    if stage == "context" {
        c.context[..p.tokens * HEADS * p.dim]
            .iter()
            .map(|&w| f64::from(f16::from_bits(w).to_f32()))
            .collect()
    } else {
        let words = if stage == "scores" {
            &c.scores
        } else {
            &c.probabilities
        };
        (0..HEADS)
            .flat_map(|h| {
                (0..p.tokens).flat_map(move |r| {
                    (0..p.keys).map(move |k| {
                        f64::from(f32::from_bits(words[(h * CHUNK + r) * p.capacity + k]))
                    })
                })
            })
            .collect()
    }
}
fn compare(p: Plan, c: &Value, capture: &Capture) -> (usize, f64, f64, usize) {
    let (mut total, mut maximum, mut fraction, mut wide_exceedances) = (0, 0.0f64, 0.0f64, 0);
    for stage in ["scores", "probabilities", "context"] {
        let d = &c["arrays"][stage];
        let actual = active(p, capture, stage);
        let main = values(&d["main"]);
        let wide = values(&d["wide"]);
        assert_eq!(actual.len(), main.len());
        let atol = d["gate"]["atol"].as_f64().unwrap();
        let rtol = d["gate"]["rtol"].as_f64().unwrap();
        let (mut stage_max, mut stage_ratio, mut diagnostic) = (0.0f64, 0.0f64, 0);
        for (i, ((&a, &m), &w)) in actual.iter().zip(&main).zip(&wide).enumerate() {
            let err = (a - m).abs();
            let limit = atol + rtol * m.abs();
            assert!(
                a.is_finite() && m.is_finite() && w.is_finite() && err <= limit,
                "{} {stage}[{i}]: actual {a}, main {m}, limit {limit}",
                c["name"]
            );
            stage_max = stage_max.max(err);
            stage_ratio = stage_ratio.max(err / limit);
            diagnostic += usize::from((a - w).abs() > atol + rtol * w.abs());
        }
        println!(
            "STAGED_ATTENTION_STAT {}",
            json!({"case":c["name"],"stage":stage,"main_values":actual.len(),"max_absolute_error":stage_max,"max_gate_fraction":stage_ratio,"wide_diagnostic_exceedances":diagnostic,"wide_fatal":false})
        );
        total += actual.len();
        maximum = maximum.max(stage_max);
        fraction = fraction.max(stage_ratio);
        wide_exceedances += diagnostic;
    }
    for h in 0..HEADS {
        for row in 0..p.tokens {
            for key in 0..p.keys {
                let absolute = p.start + key;
                let query = p.committed + row;
                let visible = absolute <= query && p.window.is_none_or(|w| query - absolute < w);
                if !visible {
                    assert_eq!(
                        f32::from_bits(capture.probabilities[(h * CHUNK + row) * p.capacity + key]),
                        0.0
                    );
                }
            }
        }
    }
    if let Some(exact) = c["exact_context_first_four_per_head"].as_array() {
        assert_eq!(p.tokens, 1);
        for h in 0..HEADS {
            for (f, word) in exact.iter().enumerate() {
                assert_eq!(capture.context[h * p.dim + f], number(word) as u16);
            }
        }
    }
    (total, maximum, fraction, wide_exceedances)
}
fn execute(gpu: &Gpu, op: &Operation, io: &Io, label: &str, measure: bool) -> (Capture, usize) {
    let raw = gpu.command().unwrap();
    let command = Command {
        raw: &raw,
        trace: None,
    };
    let (result, allocations) = if measure {
        let measured = Measurement::start();
        let result = op.encode(&command, io.p, io.invocation());
        (result, measured.finish())
    } else {
        (op.encode(&command, io.p, io.invocation()), 0)
    };
    result.unwrap();
    raw.commit();
    raw.waitUntilCompleted();
    let c = io.record(label, &raw);
    if measure {
        println!(
            "STAGED_ATTENTION_ALLOCATIONS {}",
            json!({"case":label,"rust_allocations":allocations,"release":!cfg!(debug_assertions),"driver_allocations_measured":false})
        );
    }
    (c, allocations)
}
fn preflight(gpu: &Gpu, op: &Operation) {
    let p = Plan::new(32, 32, None, 0, 1).unwrap();
    let input = input_lengths(p).map(|n| vec![0; n]);
    let io = Io::new(gpu, p, &input);
    let raw = gpu.command().unwrap();
    let command = Command {
        raw: &raw,
        trace: None,
    };
    let half_lengths = [
        p.q_len(),
        p.old_len(),
        p.old_len(),
        p.staged_len(),
        p.staged_len(),
        p.context_len(),
    ];
    let short_half = half_lengths.map(|n| half_buffer(gpu, &vec![0; n - 1]));
    let short_float =
        [p.score_len(), p.score_len(), p.status_len()].map(|n| float_buffer(gpu, n - 1));
    for index in 0..9 {
        let mut invocation = io.invocation();
        match index {
            0 => invocation.q = &short_half[0],
            1 => invocation.old_k = &short_half[1],
            2 => invocation.old_v = &short_half[2],
            3 => invocation.new_k = &short_half[3],
            4 => invocation.new_v = &short_half[4],
            5 => invocation.context = &short_half[5],
            6 => invocation.scores = &short_float[0],
            7 => invocation.probabilities = &short_float[1],
            8 => invocation.status = &short_float[2],
            _ => unreachable!(),
        }
        assert!(matches!(
            op.encode(&command, p, invocation),
            Err(Error::Options("staged attention operand extent"))
        ));
    }
    assert_eq!(io.inputs[0].len, io.context.len);
    assert_eq!(io.inputs[1].len, io.inputs[2].len);
    assert_eq!(io.inputs[3].len, io.inputs[4].len);
    assert_eq!(io.scores.len, io.probabilities.len);
    for index in 0..4 {
        let mut invocation = io.invocation();
        match index {
            0 => invocation.context = &io.inputs[0],
            1 => invocation.old_v = &io.inputs[1],
            2 => invocation.new_v = &io.inputs[3],
            3 => invocation.probabilities = &io.scores,
            _ => unreachable!(),
        }
        assert!(matches!(
            op.encode(&command, p, invocation),
            Err(Error::Options("staged attention operand alias"))
        ));
    }
    println!(
        "STAGED_ATTENTION_PREFLIGHT {}",
        json!({"short_operand_rejections":9,"equal_extent_typed_alias_rejections":4,"command_submitted":false})
    );
}
// New bounded exact identities. These do not change the eight source arrays or
// their gates: a single visible value has probability one, for every query head.
fn analytical(gpu: &Gpu, op: &Operation, tiny: &Value) -> (usize, usize) {
    let source = source_inputs(tiny);
    let (mut contexts, mut probabilities) = (0, 0);
    for (label, window, committed, tokens) in [
        ("identity_w1", Some(1), 5, 7),
        ("identity_c0_local", Some(4), 0, 1),
        ("identity_c0_full", None, 0, 1),
    ] {
        let p = Plan::new(12, 32, window, committed, tokens).unwrap();
        assert_eq!(p.old[0].rows + p.old[1].rows, 0);
        let input = [
            source[0].clone(),
            vec![0x7e55; p.old_len()],
            vec![0x7e55; p.old_len()],
            source[3].clone(),
            source[4].clone(),
        ];
        let io = Io::new(gpu, p, &input);
        let (c, _) = execute(gpu, op, &io, label, false);
        io.validate(&c);
        assert_eq!(c.status[0], 0);
        assert!(active(p, &c, "scores").iter().all(|v| v.is_finite()));
        for h in 0..HEADS {
            for row in 0..tokens {
                for key in 0..p.keys {
                    assert_eq!(
                        f32::from_bits(c.probabilities[(h * CHUNK + row) * p.capacity + key]),
                        if key == row { 1.0 } else { 0.0 }
                    );
                    probabilities += 1;
                }
                for f in 0..p.dim {
                    let expected = input[4][row * p.dim + f];
                    assert!(f16::from_bits(expected).is_finite());
                    assert_ne!(expected, 0x8000, "identity source has positive zero only");
                    assert_eq!(c.context[(row * HEADS + h) * p.dim + f], expected);
                    contexts += 1;
                }
            }
        }
    }
    assert_eq!((contexts, probabilities), (1152, 204));
    println!(
        "STAGED_ATTENTION_ANALYTICAL {}",
        json!({"cases":3,"exact_context_words":contexts,"exact_probability_values":probabilities,"not_in_fixture_gate_count":true,"cache_copy_executed":false})
    );
    (contexts, probabilities)
}
#[test]
#[ignore = "explicit release Metal staged attention source and analytical controls"]
fn gpu_source_arrays_guards_and_analytical_edges() {
    autoreleasepool(|_| {
        let m = fixture();
        let gpu = Gpu::new().unwrap();
        let required = PIPELINE_ALLOWANCE + size_of::<Operation>();
        assert!(
            matches!(Operation::new(&gpu,required-1),Err(Error::Budget{required:r,limit})if r==required&&limit==required-1)
        );
        let op = Operation::new(&gpu, required).unwrap();
        preflight(&gpu, &op);
        let (mut total, mut maximum, mut fraction, mut diagnostic, mut calls) =
            (0, 0.0f64, 0.0f64, 0, 0);
        for c in m["cases"].as_array().unwrap() {
            let p = plan(c);
            let io = Io::new(&gpu, p, &source_inputs(c));
            let (raw, allocations) = execute(&gpu, &op, &io, c["name"].as_str().unwrap(), true);
            io.validate(&raw);
            assert_eq!(raw.status[0], 0);
            assert!(
                !cfg!(debug_assertions),
                "allocation qualification requires release"
            );
            assert_eq!(allocations, 0);
            let (n, e, f, w) = compare(p, c, &raw);
            total += n;
            maximum = maximum.max(e);
            fraction = fraction.max(f);
            diagnostic += w;
            calls += 1;
        }
        let (identity_context, identity_probability) = analytical(&gpu, &op, &m["cases"][0]);
        assert_eq!((total, calls), (361448, 8));
        println!(
            "STAGED_ATTENTION_RESULT {}",
            json!({"device":gpu.device.name().to_string(),"source_cases":8,"main_values":total,"wide_diagnostic_values":total,"max_absolute_error":maximum,"max_gate_fraction":fraction,"wide_diagnostic_exceedances":diagnostic,"wide_fatal":false,"release_measured_encode_calls":calls,"analytical_cases":3,"analytical_context_words":identity_context,"analytical_probability_values":identity_probability,"cache_copy_executed":false,"whole_graph_qualified":false})
        );
    });
}
#[test]
#[ignore = "explicit release Metal staged attention invalid/sticky/reset controls"]
fn gpu_active_invalid_discard_and_recovery() {
    autoreleasepool(|_| {
        let m = fixture();
        let c = &m["cases"][0];
        let p = plan(c);
        let source = source_inputs(c);
        let gpu = Gpu::new().unwrap();
        let op = Operation::new(&gpu, PIPELINE_ALLOWANCE + size_of::<Operation>()).unwrap();
        let mut valid_capture = None;
        // Every active owner is checked; late-stage poison is not visible to the
        // first query, but must still prevent a subsequent all-layer commit.
        let sites = [
            (0, 0, 0x7e01),
            (1, p.old[0].slot * p.dim, 0x7c00),
            (2, p.old[0].slot * p.dim, 0x7e01),
            (3, (p.tokens - 1) * p.dim, 0x7c00),
            (4, (p.tokens - 1) * p.dim, 0x7e01),
        ];
        for (control, (owner, index, word)) in sites.into_iter().enumerate() {
            let mut bad = source.clone();
            bad[owner][index] = word;
            let mut io = Io::new(&gpu, p, &bad);
            let mut sticky = 0;
            for phase in 0..3 {
                if phase > 0 {
                    io.replace_inputs(&source);
                    io.reset_outputs();
                }
                if phase == 2 {
                    io.status.with_mut(|v| v[0] = 0);
                }
                let label = format!("invalid_{control}_phase_{phase}");
                let (raw, _) = execute(&gpu, &op, &io, &label, false);
                println!(
                    "STAGED_ATTENTION_RECOVERY {}",
                    json!({"control":control,"phase":phase,"corrupted_owner":INPUTS[owner],"corrupted_element":index,"corrupt_half_word":word,"math_output_discarded":phase==0,"cache_copy_executed":false,"status_reset":phase==2})
                );
                io.validate(&raw);
                if phase == 0 {
                    assert_ne!(raw.status[0], 0);
                    sticky = raw.status[0];
                } else {
                    if phase == 1 {
                        assert_eq!(raw.status[0], sticky);
                    } else {
                        assert_eq!(raw.status[0], 0);
                    }
                    compare(p, c, &raw);
                    if phase == 2 {
                        valid_capture = Some(raw);
                    }
                }
            }
        }
        // Physical retention W includes position1; the minimal old view starts2.
        // Poison both unused row1 owners without touching any visible row.
        assert_eq!((p.start, p.retained), (2, 4));
        let mut unused = source.clone();
        unused[1][p.dim..2 * p.dim].fill(0x7e55);
        unused[2][p.dim..2 * p.dim].fill(0x7e55);
        let io = Io::new(&gpu, p, &unused);
        let (raw, _) = execute(&gpu, &op, &io, "unused_old_local_row_poison", false);
        io.validate(&raw);
        assert_eq!(raw.status[0], 0);
        compare(p, c, &raw);
        let expected = valid_capture.unwrap();
        assert_eq!(raw.scores, expected.scores);
        assert_eq!(raw.probabilities, expected.probabilities);
        assert_eq!(raw.context, expected.context);
        println!(
            "STAGED_ATTENTION_ERRORS_RESULT {}",
            json!({"active_input_controls":5,"phases_each":3,"unused_old_row_controls":1,"submissions":16,"cache_inputs_unchanged":true,"sticky_status_and_reset":true,"discard_is_caller_policy":true,"cache_copy_executed":false,"transaction_qualification":false})
        );
    });
}
