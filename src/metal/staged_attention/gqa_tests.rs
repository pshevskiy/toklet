use super::*;
use crate::metal::staged_gqa_fixture as f;
use crate::model::gemma_reference::tests::Measurement;
use half::f16;
use objc2::rc::autoreleasepool;
use serde_json::{Value, json};
use std::io::Write;
const HP: u16 = 0x5a17;
const FP: u32 = 0x5a170031;
fn plan(c: &Value) -> Plan {
    Plan::new_gqa(
        f::number(&c["capacity"]),
        f::number(&c["query_heads"]),
        f::number(&c["kv_heads"]),
        c["window"].as_u64().map(|v| v as usize),
        f::number(&c["committed"]),
        f::number(&c["tokens"]),
    )
    .unwrap()
}
fn inputs(c: &Value) -> [Vec<u16>; 5] {
    f::INPUTS.map(|key| f::words(&c["inputs"][key]))
}
fn lengths(p: Plan) -> [usize; 5] {
    [
        p.q_len(),
        p.old_len(),
        p.old_len(),
        p.staged_len(),
        p.staged_len(),
    ]
}
#[test]
fn explicit_head_profiles_preserve_legacy_and_exact_physical_bytes() {
    for (cap, q, kv) in [(64, 4, 2), (544, 16, 8)] {
        for window in [None, Some(4), Some(cap)] {
            for offset in [0, 1, cap - 1] {
                let n = (cap - offset).min(32);
                let p = Plan::new_gqa(cap, q, kv, window, offset, n).unwrap();
                assert_eq!(p.q_len(), 32 * q * 64 + 7);
                assert_eq!(p.staged_len(), kv * 32 * 64 + 7);
                assert_eq!(p.old_len(), kv * window.unwrap_or(cap) * 64 + 7);
                assert_eq!(p.score_len(), q * 32 * cap + 7);
                assert_eq!(
                    p.operand_bytes().unwrap(),
                    2 * (2 * p.q_len() + 2 * p.old_len() + 2 * p.staged_len())
                        + 4 * (2 * p.score_len() + p.status_len())
                );
            }
        }
    }
    assert_eq!(
        Plan::new_gqa(544, 16, 8, None, 512, 32)
            .unwrap()
            .operand_bytes()
            .unwrap(),
        3539116
    );
    for (q, kv) in [(0, 0), (4, 1), (4, 3), (16, 2), (16, 16), (usize::MAX, 8)] {
        assert!(Plan::new_gqa(544, q, kv, None, 0, 1).is_err());
    }
    for c in [0, 545, usize::MAX] {
        assert!(Plan::new_gqa(c, 4, 2, None, 0, 1).is_err());
    }
    assert!(Plan::new_gqa(544, 16, 8, None, usize::MAX, 1).is_err());
    assert!(Plan::new_gqa(544, 16, 8, None, 513, 32).is_err());
    for d in [32, 256] {
        assert!(Plan::new(544, d, None, 512, 32).is_ok());
    }
    assert!(Plan::new(544, 64, None, 0, 1).is_err());
}
fn hb(gpu: &Gpu, v: &[u16]) -> Buffer<u16> {
    let mut b = Buffer::new(gpu, v.len()).unwrap();
    b.with_mut(|x| x.copy_from_slice(v));
    b
}
fn fb(gpu: &Gpu, n: usize) -> Buffer<u32> {
    let mut b = Buffer::new(gpu, n).unwrap();
    b.with_mut(|x| x.fill(FP));
    b
}
struct Io {
    p: Plan,
    input: [Buffer<u16>; 5],
    before: [String; 5],
    scores: Buffer<u32>,
    probs: Buffer<u32>,
    context: Buffer<u16>,
    status: Buffer<u32>,
}
struct Capture {
    scores: Vec<u32>,
    probs: Vec<u32>,
    context: Vec<u16>,
    status: Vec<u32>,
    full: [String; 3],
    after: [String; 5],
    completed: bool,
}
impl Io {
    fn new(gpu: &Gpu, p: Plan, input: &[Vec<u16>; 5]) -> Self {
        for (v, n) in input.iter().zip(lengths(p)) {
            assert_eq!(v.len(), n);
        }
        let mut status = fb(gpu, p.status_len());
        status.with_mut(|v| v[0] = 0);
        let io = Self {
            p,
            input: input.each_ref().map(|v| hb(gpu, v)),
            before: input.each_ref().map(|v| f::hash16(v)),
            scores: fb(gpu, p.score_len()),
            probs: fb(gpu, p.score_len()),
            context: hb(gpu, &vec![HP; p.context_len()]),
            status,
        };
        assert_eq!(
            io.input.iter().map(|b| b.raw.length()).sum::<usize>()
                + io.scores.raw.length()
                + io.probs.raw.length()
                + io.context.raw.length()
                + io.status.raw.length(),
            p.operand_bytes().unwrap()
        );
        io
    }
    fn invocation(&self) -> Invocation<'_> {
        Invocation {
            q: &self.input[0],
            old_k: &self.input[1],
            old_v: &self.input[2],
            new_k: &self.input[3],
            new_v: &self.input[4],
            scores: &self.scores,
            probabilities: &self.probs,
            context: &self.context,
            status: &self.status,
        }
    }
    fn record(&self, label: &str, raw: &ProtocolObject<dyn MTLCommandBuffer>) -> Capture {
        let state = raw.status();
        let error = raw.error().map(|e| e.to_string());
        assert!(matches!(
            state,
            MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
        ));
        let p = self.p;
        let select = |v: &[u32]| {
            (0..p.query_heads)
                .flat_map(|h| {
                    (0..p.tokens).flat_map(move |t| {
                        (0..p.keys).map(move |k| v[(h * CHUNK + t) * p.capacity + k])
                    })
                })
                .collect()
        };
        let c = Capture {
            scores: self.scores.with_slice(select),
            probs: self.probs.with_slice(select),
            context: self
                .context
                .with_slice(|v| v[..p.tokens * p.query_heads * p.dim].to_vec()),
            status: self.status.with_slice(|v| v.to_vec()),
            full: [
                self.scores.with_slice(f::hash32),
                self.probs.with_slice(f::hash32),
                self.context.with_slice(f::hash16),
            ],
            after: self.input.each_ref().map(|b| b.with_slice(f::hash16)),
            completed: state == MTLCommandBufferStatus::Completed && error.is_none(),
        };
        println!(
            "GQA_ATTENTION_RAW {}",
            json!({"label":label,"completion":{"completed":c.completed,"status":state.0,"error":error},"capacity":p.capacity,"query_heads":p.query_heads,"kv_heads":p.kv_heads,"dim":p.dim,"committed":p.committed,"tokens":p.tokens,"window":p.window,"view_start":p.start,"keys":p.keys,"chunk_capacity":CHUNK,"active_score_shape":[p.query_heads,p.tokens,p.keys],"active_context_shape":[p.tokens,p.query_heads,p.dim],"scores_active_u32le_hex":f::hex32(&c.scores),"probabilities_active_u32le_hex":f::hex32(&c.probs),"context_active_u16le_hex":f::hex16(&c.context),"physical_lengths":[p.score_len(),p.score_len(),p.context_len()],"physical_sha256":c.full,"inactive_expected_canaries":[FP,FP,HP as u32],"status_words":c.status,"input_names":f::INPUTS,"input_before_sha256":self.before,"input_after_sha256":c.after,"operand_bytes":p.operand_bytes().unwrap(),"capture_scope":"all actual active cells; inactive cells validated by reconstruction against actual full physical hash"})
        );
        std::io::stdout().flush().unwrap();
        c
    }
    fn validate(&self, c: &Capture) {
        let p = self.p;
        assert!(c.completed);
        assert_eq!(c.after, self.before);
        assert!(c.status[1..].iter().all(|&w| w == FP));
        // Independently rebuild complete physical owners from captured active
        // cells and admission canaries; hash equality authenticates every gap.
        for (active, actual) in [(&c.scores, &c.full[0]), (&c.probs, &c.full[1])] {
            let mut physical = vec![FP; p.score_len()];
            let mut i = 0;
            for h in 0..p.query_heads {
                for t in 0..p.tokens {
                    for k in 0..p.keys {
                        physical[(h * CHUNK + t) * p.capacity + k] = active[i];
                        i += 1;
                    }
                }
            }
            assert_eq!(i, active.len());
            assert_eq!(&f::hash32(&physical), actual);
        }
        let mut physical = vec![HP; p.context_len()];
        physical[..c.context.len()].copy_from_slice(&c.context);
        assert_eq!(f::hash16(&physical), c.full[2]);
    }
    fn repair(&mut self, input: &[Vec<u16>; 5]) {
        for (b, v) in self.input.iter_mut().zip(input) {
            b.with_mut(|x| x.copy_from_slice(v));
        }
        self.before = input.each_ref().map(|v| f::hash16(v));
    }
    fn clear_outputs(&mut self) {
        self.scores.with_mut(|v| v.fill(FP));
        self.probs.with_mut(|v| v.fill(FP));
        self.context.with_mut(|v| v.fill(HP));
    }
}
fn run(gpu: &Gpu, op: &Operation, io: &Io, label: &str, measure: bool) -> Capture {
    let raw = gpu.command().unwrap();
    let command = Command {
        raw: &raw,
        trace: None,
    };
    let mut count = None;
    let result = if measure {
        let m = Measurement::start();
        let r = op.encode(&command, io.p, io.invocation());
        count = Some(m.finish());
        r
    } else {
        op.encode(&command, io.p, io.invocation())
    };
    if let Some(n) = count {
        println!(
            "GQA_ATTENTION_ALLOCATION {}",
            json!({"label":label,"rust_allocations":n,"encode_ok":result.is_ok(),"release_required_zero":!cfg!(debug_assertions)})
        );
    }
    result.unwrap();
    raw.commit();
    raw.waitUntilCompleted();
    let c = io.record(label, &raw);
    io.validate(&c);
    if !cfg!(debug_assertions)
        && let Some(n) = count
    {
        assert_eq!(n, 0);
    }
    c
}
#[derive(Default)]
struct Metrics {
    values: usize,
    max_error: f64,
    max_fraction: f64,
    wide_exceedances: usize,
    wide_max_error: f64,
    wide_max_fraction: f64,
    masked: usize,
}
impl Metrics {
    fn compare(&mut self, source: &Value, p: Plan, c: &Capture) {
        let full = plan(source);
        let row_offset = p.committed - full.committed;
        assert!(p.start >= full.start);
        let key_offset = p.start - full.start;
        for stage in ["scores", "probabilities", "context"] {
            let d = &source["arrays"][stage];
            let main = f::values(&d["main"]);
            let wide = f::values(&d["wide"]);
            let atol = d["gate"]["atol"].as_f64().unwrap();
            let rtol = d["gate"]["rtol"].as_f64().unwrap();
            let actual: Vec<f64> = match stage {
                "scores" => c
                    .scores
                    .iter()
                    .map(|&w| f64::from(f32::from_bits(w)))
                    .collect(),
                "probabilities" => c
                    .probs
                    .iter()
                    .map(|&w| f64::from(f32::from_bits(w)))
                    .collect(),
                _ => c
                    .context
                    .iter()
                    .map(|&w| f64::from(f16::from_bits(w).to_f32()))
                    .collect(),
            };
            for (i, &a) in actual.iter().enumerate() {
                let original = if stage == "context" {
                    (row_offset + i / (p.query_heads * p.dim)) * p.query_heads * p.dim
                        + i % (p.query_heads * p.dim)
                } else {
                    let h = i / (p.tokens * p.keys);
                    let t = (i / p.keys) % p.tokens;
                    let k = i % p.keys;
                    (h * full.tokens + row_offset + t) * full.keys + key_offset + k
                };
                let b = main[original];
                let w = wide[original];
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
                    a.is_finite() && b.is_finite() && w.is_finite() && e <= limit,
                    "{} {stage} queryoffset{row_offset} index{i}: {a} versus {b}, limit{limit}",
                    source["name"]
                );
            }
        }
        for h in 0..p.query_heads {
            for t in 0..p.tokens {
                for k in 0..p.keys {
                    let absolute = p.start + k;
                    let query = p.committed + t;
                    let visible =
                        absolute <= query && p.window.is_none_or(|w| query - absolute < w);
                    if !visible {
                        assert_eq!(c.probs[(h * p.tokens + t) * p.keys + k], 0);
                        self.masked += 1;
                    }
                }
            }
        }
        if source["exact_one_key_context"] == true {
            assert_eq!(p.tokens, 1);
            assert!(c.probs.iter().all(|&w| w == 1f32.to_bits()));
            let v = f::words(&source["inputs"]["new_v"]);
            for h in 0..p.query_heads {
                let kv = h / (p.query_heads / p.kv_heads);
                assert_eq!(
                    &c.context[h * p.dim..(h + 1) * p.dim],
                    &v[kv * CHUNK * p.dim..kv * CHUNK * p.dim + p.dim]
                );
            }
        }
    }
}
fn preflight(gpu: &Gpu, op: &Operation) {
    let p = Plan::new_gqa(64, 4, 2, None, 0, 3).unwrap();
    let input = lengths(p).map(|n| vec![0; n]);
    let io = Io::new(gpu, p, &input);
    let raw = gpu.command().unwrap();
    let command = Command {
        raw: &raw,
        trace: None,
    };
    for v in [
        Invocation {
            old_v: &io.input[1],
            ..io.invocation()
        },
        Invocation {
            new_v: &io.input[3],
            ..io.invocation()
        },
        Invocation {
            context: &io.input[0],
            ..io.invocation()
        },
        Invocation {
            probabilities: &io.scores,
            ..io.invocation()
        },
    ] {
        assert!(matches!(
            op.encode(&command, p, v),
            Err(Error::Options("staged attention operand alias"))
        ));
    }
    let half = lengths(p).map(|n| Buffer::<u16>::new(gpu, n - 1).unwrap());
    let scores = Buffer::new(gpu, p.score_len() - 1).unwrap();
    let probs = Buffer::new(gpu, p.score_len() - 1).unwrap();
    let context = Buffer::new(gpu, p.context_len() - 1).unwrap();
    let status = Buffer::new(gpu, p.status_len() - 1).unwrap();
    for v in [
        Invocation {
            q: &half[0],
            ..io.invocation()
        },
        Invocation {
            old_k: &half[1],
            ..io.invocation()
        },
        Invocation {
            old_v: &half[2],
            ..io.invocation()
        },
        Invocation {
            new_k: &half[3],
            ..io.invocation()
        },
        Invocation {
            new_v: &half[4],
            ..io.invocation()
        },
        Invocation {
            scores: &scores,
            ..io.invocation()
        },
        Invocation {
            probabilities: &probs,
            ..io.invocation()
        },
        Invocation {
            context: &context,
            ..io.invocation()
        },
        Invocation {
            status: &status,
            ..io.invocation()
        },
    ] {
        assert!(matches!(
            op.encode(&command, p, v),
            Err(Error::Options("staged attention operand extent"))
        ));
    }
    println!(
        "GQA_ATTENTION_PREFLIGHT {}",
        json!({"typed_equal_extent_aliases":4,"short_operands":9,"pipeline_budget_exact_minus_one":true})
    );
}
#[test]
#[ignore = "explicit Apple GPU all-query GQA Reference source plus deep partitions"]
fn gpu_all_source_queries_and_deep_partitions() {
    autoreleasepool(|_| {
        let m = f::fixture();
        let gpu = Gpu::new().unwrap();
        let required = PIPELINE_ALLOWANCE + size_of::<Operation>();
        assert!(
            matches!(Operation::new(&gpu,required-1),Err(Error::Budget{required:r,limit})if r==required&&limit==required-1)
        );
        let op = Operation::new(&gpu, required).unwrap();
        preflight(&gpu, &op);
        let mut metrics = Metrics::default();
        let mut partition_metrics = Metrics::default();
        let mut boundaries = 0;
        for source in m["cases"].as_array().unwrap() {
            let p = plan(source);
            let input = inputs(source);
            let io = Io::new(&gpu, p, &input);
            let baseline = run(&gpu, &op, &io, source["name"].as_str().unwrap(), true);
            assert_eq!(baseline.status[0], 0);
            metrics.compare(source, p, &baseline);
            if source["name"] != "real_deep32" {
                continue;
            }
            for (schedule_index, chunks) in
                [vec![1; 32], vec![1, 2, 3, 7, 19]].into_iter().enumerate()
            {
                let mut old = [input[1].clone(), input[2].clone()];
                let mut at = 0;
                for count in chunks {
                    let q = Plan::new_gqa(
                        p.capacity,
                        p.query_heads,
                        p.kv_heads,
                        None,
                        p.committed + at,
                        count,
                    )
                    .unwrap();
                    let mut chunk = lengths(q).map(|n| vec![0x7e55; n]);
                    chunk[1] = old[0].clone();
                    chunk[2] = old[1].clone();
                    chunk[0][..count * p.query_heads * p.dim].copy_from_slice(
                        &input[0][at * p.query_heads * p.dim..(at + count) * p.query_heads * p.dim],
                    );
                    for head in 0..p.kv_heads {
                        for kind in [3, 4] {
                            let start = head * CHUNK * p.dim;
                            chunk[kind][start..start + count * p.dim].copy_from_slice(
                                &input[kind][start + at * p.dim..start + (at + count) * p.dim],
                            );
                        }
                    }
                    let owner = Io::new(&gpu, q, &chunk);
                    let c = run(
                        &gpu,
                        &op,
                        &owner,
                        &format!("real_deep32/partition{schedule_index}/start{at}"),
                        false,
                    );
                    assert_eq!(c.status[0], 0);
                    partition_metrics.compare(source, q, &c);
                    assert_eq!(
                        &c.context,
                        &baseline.context
                            [at * p.query_heads * p.dim..(at + count) * p.query_heads * p.dim]
                    );
                    for h in 0..p.query_heads {
                        for t in 0..count {
                            for k in 0..q.keys {
                                let local = (h * count + t) * q.keys + k;
                                let whole = (h * p.tokens + at + t) * p.keys + k;
                                assert_eq!(c.scores[local], baseline.scores[whole]);
                                assert_eq!(c.probs[local], baseline.probs[whole]);
                            }
                        }
                    }
                    for head in 0..p.kv_heads {
                        for (kind, words) in old.iter_mut().enumerate() {
                            let dst = (head * p.retained + p.committed + at) * p.dim;
                            let src = head * CHUNK * p.dim;
                            words[dst..dst + count * p.dim]
                                .copy_from_slice(&chunk[kind + 3][src..src + count * p.dim]);
                        }
                    }
                    at += count;
                    boundaries += 1;
                }
                assert_eq!(at, 32);
            }
        }
        assert_eq!(metrics.values, 753432);
        assert_eq!(boundaries, 37);
        println!(
            "GQA_ATTENTION_RESULT {}",
            json!({"cases":11,"source_main_values":metrics.values,"partition_main_values":partition_metrics.values,"main_failures":0,"max_error":metrics.max_error.max(partition_metrics.max_error),"max_fraction":metrics.max_fraction.max(partition_metrics.max_fraction),"wide_diagnostic_values":metrics.values+partition_metrics.values,"wide_exceedances":metrics.wide_exceedances+partition_metrics.wide_exceedances,"wide_max_error":metrics.wide_max_error.max(partition_metrics.wide_max_error),"wide_max_fraction":metrics.wide_max_fraction.max(partition_metrics.wide_max_fraction),"wide_fatal":false,"masked_positive_zero_words":metrics.masked+partition_metrics.masked,"deep_partition_boundaries":boundaries,"release_encode_measurements":11,"cache_copy_executed":false,"all_queries":true})
        );
    });
}
#[test]
#[ignore = "explicit Apple GPU late GQA-head invalid/sticky/reset and padding controls"]
fn gpu_late_head_invalid_sticky_recovery() {
    autoreleasepool(|_| {
        let m = f::fixture();
        let source = &m["cases"][5];
        assert_eq!(source["name"], "real_partial");
        let p = plan(source);
        let input = inputs(source);
        let gpu = Gpu::new().unwrap();
        let op = Operation::new(&gpu, PIPELINE_ALLOWANCE + size_of::<Operation>()).unwrap();
        let mut metrics = Metrics::default();
        let sites = [
            (0, (p.tokens * p.query_heads - 1) * p.dim + 63, 0x7e00),
            (1, (7 * p.retained + p.committed - 1) * p.dim + 63, 0x7c00),
            (2, (7 * p.retained + p.committed - 1) * p.dim + 63, 0x7e00),
            (3, (7 * CHUNK + p.tokens - 1) * p.dim + 63, 0x7e00),
            (4, (7 * CHUNK + p.tokens - 1) * p.dim + 63, 0x7c00),
        ];
        for (site, (owner, index, word)) in sites.into_iter().enumerate() {
            let mut bad = input.clone();
            bad[owner][index] = word;
            let mut io = Io::new(&gpu, p, &bad);
            let c = run(&gpu, &op, &io, &format!("bad{site}/invalid"), false);
            assert_ne!(c.status[0], 0);
            io.repair(&input);
            for reset in [false, true] {
                if reset {
                    io.status.with_mut(|v| v[0] = 0);
                }
                io.clear_outputs();
                let got = run(
                    &gpu,
                    &op,
                    &io,
                    &format!("bad{site}/{}", if reset { "reset" } else { "sticky" }),
                    false,
                );
                assert_eq!(got.status[0], if reset { 0 } else { c.status[0] });
                metrics.compare(source, p, &got);
            }
        }
        let mut poison = input.clone();
        for kind in [3, 4] {
            for head in 0..p.kv_heads {
                poison[kind][(head * CHUNK + p.tokens) * p.dim..(head + 1) * CHUNK * p.dim]
                    .fill(0x7c00);
            }
        }
        let io = Io::new(&gpu, p, &poison);
        let c = run(&gpu, &op, &io, "inactive-stage-poison", false);
        assert_eq!(c.status[0], 0);
        metrics.compare(source, p, &c);
        println!(
            "GQA_ATTENTION_RECOVERY_RESULT {}",
            json!({"invalid_sites":5,"commands":16,"valid_recovery_values":metrics.values,"main_failures":0,"max_error":metrics.max_error,"max_fraction":metrics.max_fraction,"wide_exceedances":metrics.wide_exceedances,"invalid_output_bits_are_oracle":false})
        );
    });
}
