use super::*;
use crate::model::gemma_reference::tests::Measurement;
use half::f16;
use objc2::rc::autoreleasepool;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const RAW: &[u8] = include_bytes!("../../../tests/fixtures/strided-half-product-v1/arrays.bin");
const CONTROLS: &[u8] =
    include_bytes!("../../../tests/fixtures/strided-half-product-v1/controls.json");
const CANARY: u16 = 0x5a17;
const STATUS_GUARD: u32 = 0x5a170031;
fn sha(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}
fn hash16(v: &[u16]) -> String {
    let mut h = Sha256::new();
    for w in v {
        h.update(w.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn fixture() -> Value {
    let bytes = include_bytes!("../../../tests/fixtures/strided-half-product-v1/manifest.json");
    assert_eq!(
        sha(bytes),
        "294a521b372897eef69fe9670f979f3e582abc3ca58570331a27b666d652e0d6"
    );
    let m: Value = serde_json::from_slice(bytes).unwrap();
    assert_eq!(m["format"], "toklet-strided-half-product-v1");
    assert_eq!(
        m["protocol_sha256"],
        "eb77913b363e5ba0b58fb7bebfb84adf1137da412c48bf98f2202801b79da8d6"
    );
    assert_eq!(m["source_reference_executed"], true);
    assert_eq!(m["target_gpu_executed"], false);
    assert_eq!(m["finite_output_values"], 67072);
    assert_eq!(m["stored_arrays"], 17);
    assert_eq!(
        m["policy"],
        json!({"atol_floor":1e-4,"envelope_factor":2,"rtol":1e-3,"fatal_reference":"main"})
    );
    for (name, raw) in [("arrays.bin", RAW), ("controls.json", CONTROLS)] {
        assert_eq!(m["files"][name]["bytes"], raw.len());
        assert_eq!(m["files"][name]["sha256"], sha(raw));
    }
    for (name, raw) in [
        (
            "producer",
            include_bytes!("../../../tools/reference/strided_half_product.py").as_slice(),
        ),
        (
            "helper",
            include_bytes!("../../../tools/reference/gemma3_half_rms.py").as_slice(),
        ),
    ] {
        assert_eq!(m[name]["sha256"], sha(raw));
    }
    m
}
fn number(v: &Value) -> usize {
    usize::try_from(v.as_u64().unwrap()).unwrap()
}
fn words(d: &Value) -> Vec<u16> {
    assert_eq!(d["file"], "arrays.bin");
    assert_eq!(d["dtype"], "u16le");
    let count = d["shape"]
        .as_array()
        .unwrap()
        .iter()
        .try_fold(1usize, |n, x| n.checked_mul(number(x)))
        .unwrap();
    let begin = number(&d["offset"]);
    let bytes = number(&d["bytes"]);
    assert_eq!(bytes, count * 2);
    let raw = &RAW[begin..begin.checked_add(bytes).unwrap()];
    assert_eq!(d["sha256"], sha(raw));
    raw.chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn json_words(v: &Value) -> Vec<u16> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| u16::try_from(number(x)).unwrap())
        .collect()
}
fn view(v: &Value) -> View {
    View {
        offset: number(&v["offset"]),
        stride: number(&v["stride"]),
    }
}
fn plan(c: &Value) -> Plan {
    Plan::new(
        number(&c["rows"]),
        number(&c["width"]),
        view(&c["a"]),
        view(&c["b"]),
        view(&c["output"]),
    )
    .unwrap()
}
fn active(p: Plan, out: &[u16]) -> Vec<u16> {
    (0..p.rows)
        .flat_map(|r| {
            out[p.output.offset + r * p.output.stride
                ..p.output.offset + r * p.output.stride + p.width]
                .iter()
                .copied()
        })
        .collect()
}
fn input_values(c: &Value) -> (Vec<u16>, Option<Vec<u16>>) {
    let a = words(&c["inputs"][c["a"]["owner"].as_str().unwrap()]);
    let b = if c["a"]["owner"] == c["b"]["owner"] {
        None
    } else {
        Some(words(&c["inputs"][c["b"]["owner"].as_str().unwrap()]))
    };
    (a, b)
}
#[test]
fn checked_views_and_frozen_physical_contract() {
    let m = fixture();
    let (mut total, mut schedules, mut boundaries) = (0, 0, 0);
    for c in m["cases"].as_array().unwrap() {
        let p = plan(c);
        let (a, b) = input_values(c);
        let [al, bl, ol, sl] = p.required_lengths();
        assert!(a.len() >= al);
        assert!(b.as_ref().unwrap_or(&a).len() >= bl);
        assert_eq!(ol, number(&c["output_elements"]));
        assert_eq!(sl, 8);
        assert_eq!(
            a.len() * 2 + b.as_ref().map_or(0, |v| v.len() * 2),
            number(&c["physical_input_bytes"])
        );
        assert_eq!(b.is_none(), c["readonly_same_parent"].as_bool().unwrap());
        let main = words(&c["arrays"]["main"]);
        let wide = words(&c["arrays"]["wide"]);
        assert_eq!(main, wide);
        assert_eq!(main.len(), p.rows * p.width);
        assert!(main.iter().all(|w| f16::from_bits(*w).is_finite()));
        assert_eq!(c["gate"], json!({"atol":1e-4,"rtol":1e-3}));
        let mut physical = vec![CANARY; ol];
        for r in 0..p.rows {
            physical[p.output.offset + r * p.output.stride
                ..p.output.offset + r * p.output.stride + p.width]
                .copy_from_slice(&main[r * p.width..(r + 1) * p.width]);
        }
        assert_eq!(c["expected_physical_output"]["sha256"], hash16(&physical));
        let exact = json_words(&c["exact_prefix_words"]);
        assert_eq!(&main[..exact.len()], exact);
        for s in c["partitions"].as_array().unwrap() {
            assert_eq!(
                s.as_array().unwrap().iter().map(number).sum::<usize>(),
                p.rows
            );
            schedules += 1;
            boundaries += s.as_array().unwrap().len();
        }
        total += main.len();
    }
    assert_eq!((total, schedules, boundaries), (67072, 11, 96));
}
#[test]
fn shape_offset_stride_overflow_rejections() {
    for width in [256, 1024] {
        for rows in [1, 2, 3, 32] {
            let a = View {
                offset: width,
                stride: 3 * width,
            };
            let b = View {
                offset: 2,
                stride: width + 5,
            };
            let out = View {
                offset: 3,
                stride: width + 7,
            };
            let p = Plan::new(rows, width, a, b, out).unwrap();
            assert_eq!(
                p.required_lengths(),
                [
                    width + (rows - 1) * 3 * width + width + 7,
                    2 + (rows - 1) * (width + 5) + width + 7,
                    3 + rows * (width + 7) + 7,
                    8
                ]
            );
        }
    }
    let v = View {
        offset: 0,
        stride: 256,
    };
    for (n, w) in [(0, 256), (33, 256), (1, 5), (1, 64), (1, 2048)] {
        assert!(Plan::new(n, w, v, v, v).is_err());
    }
    for bad in [
        View {
            offset: usize::MAX,
            stride: 256,
        },
        View {
            offset: 0,
            stride: usize::MAX,
        },
        View {
            offset: u32::MAX as usize,
            stride: 256,
        },
        View {
            offset: 0,
            stride: u32::MAX as usize,
        },
        View {
            offset: 0,
            stride: 255,
        },
    ] {
        assert!(Plan::new(32, 256, bad, v, v).is_err());
        assert!(Plan::new(32, 256, v, bad, v).is_err());
        assert!(Plan::new(32, 256, v, v, bad).is_err());
    }
}

struct Operands {
    p: Plan,
    a: Buffer<u16>,
    b: Option<Buffer<u16>>,
    output: Buffer<u16>,
    status: Buffer<u32>,
}
struct Capture {
    out: Vec<u16>,
    status: Vec<u32>,
    a: String,
    b: Option<String>,
}
impl Operands {
    fn new(gpu: &Gpu, p: Plan, a: &[u16], b: Option<&[u16]>) -> Self {
        let mut aa = Buffer::new(gpu, a.len()).unwrap();
        aa.with_mut(|v| v.copy_from_slice(a));
        let bb = b.map(|b| {
            let mut out = Buffer::new(gpu, b.len()).unwrap();
            out.with_mut(|v| v.copy_from_slice(b));
            out
        });
        let mut output = Buffer::new(gpu, p.output_len).unwrap();
        output.with_mut(|v| v.fill(CANARY));
        let mut status = Buffer::new(gpu, 8).unwrap();
        status.with_mut(|v| {
            v.fill(STATUS_GUARD);
            v[0] = 0;
        });
        Self {
            p,
            a: aa,
            b: bb,
            output,
            status,
        }
    }
    fn invocation(&self) -> Invocation<'_> {
        Invocation {
            a: &self.a,
            b: self.b.as_ref().unwrap_or(&self.a),
            output: &self.output,
            status: &self.status,
        }
    }
    fn hashes(&self) -> (String, Option<String>) {
        (
            self.a.with_slice(hash16),
            self.b.as_ref().map(|b| b.with_slice(hash16)),
        )
    }
    fn bytes(&self) -> usize {
        self.a.raw.length()
            + self.b.as_ref().map_or(0, |b| b.raw.length())
            + self.output.raw.length()
            + self.status.raw.length()
    }
    fn record(&self, label: &str, before: &(String, Option<String>), done: &Value) -> Capture {
        let (a, b) = self.hashes();
        let c = Capture {
            out: self.output.with_slice(|v| v.to_vec()),
            status: self.status.with_slice(|v| v.to_vec()),
            a,
            b,
        };
        println!(
            "PRODUCT_RAW {}",
            json!({"label":label,"command":done,"rows":self.p.rows,"width":self.p.width,"a_view":[self.p.a.offset,self.p.a.stride],"b_view":[self.p.b.offset,self.p.b.stride],"output_view":[self.p.output.offset,self.p.output.stride],"same_readonly_parent":self.b.is_none(),"before_hashes":before,"after_hashes":[c.a,c.b],"output_words":c.out,"status_words":c.status,"physical_owner_bytes":self.bytes(),"physical_owner_count":if self.b.is_none(){3}else{4}})
        );
        c
    }
    fn validate(&self, c: &Capture, before: &(String, Option<String>)) {
        assert_eq!((&c.a, &c.b), (&before.0, &before.1));
        for (i, &word) in c.out.iter().enumerate() {
            let active = i.checked_sub(self.p.output.offset).is_some_and(|k| {
                k / self.p.output.stride < self.p.rows && k % self.p.output.stride < self.p.width
            });
            if !active {
                assert_eq!(word, CANARY, "canary {i}");
            }
        }
        assert!(c.status[1..].iter().all(|&v| v == STATUS_GUARD));
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
    let before = io.hashes();
    let raw = gpu.command().unwrap();
    let command = Command {
        raw: &raw,
        trace: None,
    };
    op.encode(&command, io.p, io.invocation()).unwrap();
    raw.commit();
    raw.waitUntilCompleted();
    let done = completed(&raw);
    let c = io.record(label, &before, &done);
    io.validate(&c, &before);
    assert_eq!(done["completed"], true);
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
}
impl Metrics {
    fn compare(&mut self, c: &Value, p: Plan, out: &[u16], start: usize) {
        let main = words(&c["arrays"]["main"]);
        let wide = words(&c["arrays"]["wide"]);
        let exact = json_words(&c["exact_prefix_words"]);
        let atol = c["gate"]["atol"].as_f64().unwrap();
        let rtol = c["gate"]["rtol"].as_f64().unwrap();
        for (r, row) in active(p, out).chunks_exact(p.width).enumerate() {
            for (col, &word) in row.iter().enumerate() {
                let i = (start + r) * p.width + col;
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
                    "{} row{} col{col} bits{word:04x} main{:04x} error{e} gate{limit}",
                    c["name"],
                    start + r,
                    main[i]
                );
                if i < exact.len() {
                    assert_eq!(word, exact[i]);
                }
            }
        }
    }
}
fn preflight(gpu: &Gpu, op: &Operation) {
    let v = View {
        offset: 3,
        stride: 259,
    };
    let p = Plan::new(2, 256, v, v, v).unwrap();
    let a = vec![0; p.output_len];
    let io = Operands::new(gpu, p, &a, Some(&a));
    let raw = gpu.command().unwrap();
    let command = Command {
        raw: &raw,
        trace: None,
    };
    // Equal extent Half aliases exercise identity rather than short-buffer rejection.
    for inv in [
        Invocation {
            output: &io.a,
            ..io.invocation()
        },
        Invocation {
            output: io.b.as_ref().unwrap(),
            ..io.invocation()
        },
    ] {
        assert!(matches!(
            op.encode(&command, p, inv),
            Err(Error::Options("Half product writable alias"))
        ));
    }
    for owner in [&io.a, io.b.as_ref().unwrap(), &io.output] {
        let mut status = Buffer::<u32>::new(gpu, 8).unwrap();
        status.raw = owner.raw.clone();
        assert!(matches!(
            op.encode(
                &command,
                p,
                Invocation {
                    status: &status,
                    ..io.invocation()
                }
            ),
            Err(Error::Options("Half product writable alias"))
        ));
    }
    let aa = Buffer::new(gpu, p.a_len - 1).unwrap();
    let bb = Buffer::new(gpu, p.b_len - 1).unwrap();
    let oo = Buffer::new(gpu, p.output_len - 1).unwrap();
    let ss = Buffer::new(gpu, 7).unwrap();
    for inv in [
        Invocation {
            a: &aa,
            ..io.invocation()
        },
        Invocation {
            b: &bb,
            ..io.invocation()
        },
        Invocation {
            output: &oo,
            ..io.invocation()
        },
        Invocation {
            status: &ss,
            ..io.invocation()
        },
    ] {
        assert!(matches!(
            op.encode(&command, p, inv),
            Err(Error::Options("Half product operand extent"))
        ));
    }
    assert_eq!(raw.status(), MTLCommandBufferStatus::NotEnqueued);
    println!(
        "PRODUCT_PREFLIGHT {}",
        json!({"equal_extent_half_writable_aliases":2,"typed_status_aliases":3,"short_operands":4,"not_enqueued":true})
    );
}

#[test]
#[ignore = "explicit Apple GPU strided Half product source, physical parents and partitions"]
fn gpu_source_views_partitions_and_zero_allocation() {
    autoreleasepool(|_| {
        let m = fixture();
        let gpu = Gpu::new().unwrap();
        let required = PIPELINE_ALLOWANCE + size_of::<Operation>();
        assert!(
            matches!(Operation::new(&gpu,required-1),Err(Error::Budget{required:r,limit})if r==required&&limit==required-1)
        );
        let op = Operation::new(&gpu, required).unwrap();
        preflight(&gpu, &op);
        let (mut metrics, mut partitions) = (Metrics::default(), Metrics::default());
        let (mut schedules, mut boundaries, mut allocations) = (0, 0, 0);
        for c in m["cases"].as_array().unwrap() {
            let p = plan(c);
            let (a, b) = input_values(c);
            let name = c["name"].as_str().unwrap();
            let io = Operands::new(&gpu, p, &a, b.as_deref());
            assert_eq!(
                io.bytes(),
                number(&c["physical_input_bytes"]) + p.output_len * 2 + 8 * 4
            );
            let tightv = View {
                offset: 0,
                stride: p.width,
            };
            let tp = Plan::new(p.rows, p.width, tightv, tightv, tightv).unwrap();
            let mut ta = vec![0x7e55; tp.a_len];
            let mut tb = vec![0x7e55; tp.b_len];
            let original_b = b.as_ref().unwrap_or(&a);
            for r in 0..p.rows {
                ta[r * p.width..(r + 1) * p.width].copy_from_slice(
                    &a[p.a.offset + r * p.a.stride..p.a.offset + r * p.a.stride + p.width],
                );
                tb[r * p.width..(r + 1) * p.width].copy_from_slice(
                    &original_b[p.b.offset + r * p.b.stride..p.b.offset + r * p.b.stride + p.width],
                );
            }
            let tight = Operands::new(&gpu, tp, &ta, Some(&tb));
            let before = io.hashes();
            let tbefore = tight.hashes();
            let raw = gpu.command().unwrap();
            let command = Command {
                raw: &raw,
                trace: None,
            };
            let mut counts = [0; 2];
            for (i, owner) in [&io, &tight].into_iter().enumerate() {
                let measure = Measurement::start();
                let result = op.encode(&command, owner.p, owner.invocation());
                counts[i] = measure.finish();
                println!(
                    "PRODUCT_ALLOCATION {}",
                    json!({"case":name,"layout":i,"rust_allocations":counts[i],"encode_ok":result.is_ok(),"release_required_zero":!cfg!(debug_assertions)})
                );
                result.unwrap();
            }
            raw.commit();
            raw.waitUntilCompleted();
            let done = completed(&raw);
            let cap = io.record(&format!("{name}/strided"), &before, &done);
            let tcap = tight.record(&format!("{name}/tight"), &tbefore, &done);
            io.validate(&cap, &before);
            tight.validate(&tcap, &tbefore);
            assert_eq!(done["completed"], true);
            assert_eq!((cap.status[0], tcap.status[0]), (0, 0));
            if !cfg!(debug_assertions) {
                assert_eq!(counts, [0, 0]);
            }
            allocations += 2;
            let baseline = active(p, &cap.out);
            assert_eq!(baseline, active(tp, &tcap.out));
            assert_eq!(hash16(&cap.out), c["expected_physical_output"]["sha256"]);
            metrics.compare(c, p, &cap.out, 0);
            metrics.compare(c, tp, &tcap.out, 0);
            for (si, s) in c["partitions"].as_array().unwrap().iter().enumerate() {
                let mut start = 0;
                let mut combined = Vec::new();
                for (ci, n) in s.as_array().unwrap().iter().enumerate() {
                    let n = number(n);
                    let q = Plan::new(
                        n,
                        p.width,
                        View {
                            offset: p.a.offset + start * p.a.stride,
                            ..p.a
                        },
                        View {
                            offset: p.b.offset + start * p.b.stride,
                            ..p.b
                        },
                        p.output,
                    )
                    .unwrap();
                    let chunk = Operands::new(&gpu, q, &a, b.as_deref());
                    let got = run(
                        &gpu,
                        &op,
                        &chunk,
                        &format!("{name}/schedule{si}/chunk{ci}/start{start}"),
                    );
                    assert_eq!(got.status[0], 0);
                    partitions.compare(c, q, &got.out, start);
                    combined.extend(active(q, &got.out));
                    start += n;
                    boundaries += 1;
                }
                assert_eq!(start, p.rows);
                assert_eq!(combined, baseline);
                schedules += 1;
            }
        }
        assert_eq!(
            (metrics.values, schedules, boundaries, allocations),
            (134144, 11, 96, 10)
        );
        assert_eq!(partitions.values, 199424);
        println!(
            "PRODUCT_RESULT {}",
            json!({"device":gpu.device.name().to_string(),"normal_main_values":metrics.values,"partition_main_values":partitions.values,"main_failures":0,"max_error":metrics.max_error.max(partitions.max_error),"max_fraction":metrics.max_fraction.max(partitions.max_fraction),"wide_diagnostic_values":metrics.values+partitions.values,"wide_exceedances":metrics.wide_exceedances+partitions.wide_exceedances,"wide_max_error":metrics.wide_max_error.max(partitions.wide_max_error),"wide_max_fraction":metrics.wide_max_fraction.max(partitions.wide_max_fraction),"wide_fatal":false,"schedules":schedules,"boundaries":boundaries,"rust_encode_measurements":allocations,"model_qualified":false})
        );
    });
}

#[test]
#[ignore = "explicit Apple GPU strided product invalid operands, sticky status and reset recovery"]
fn gpu_invalid_sticky_recovery_and_inactive_inputs() {
    autoreleasepool(|_| {
        let m = fixture();
        let controls: Value = serde_json::from_slice(CONTROLS).unwrap();
        let c = &m["cases"][2];
        assert_eq!(c["name"], controls["invalid_base_case"]);
        let p = plan(c);
        let (a, b) = input_values(c);
        assert!(b.is_none());
        let gpu = Gpu::new().unwrap();
        let op = Operation::new(&gpu, PIPELINE_ALLOWANCE + size_of::<Operation>()).unwrap();
        let mut recovery = Metrics::default();
        for error in controls["invalid"].as_array().unwrap() {
            let mut io = Operands::new(&gpu, p, &a, None);
            let row = number(&error["row"]);
            let col = number(&error["column"]);
            let (bad, other) = if error["view"] == "a" {
                (p.a, p.b)
            } else {
                (p.b, p.a)
            };
            io.a.with_mut(|v| {
                v[bad.offset + row * bad.stride + col] =
                    u16::try_from(number(&error["word"])).unwrap();
                v[other.offset + row * other.stride + col] =
                    u16::try_from(number(&error["other_word"])).unwrap();
            });
            let label = error["name"].as_str().unwrap();
            let cap = run(&gpu, &op, &io, &format!("invalid/{label}"));
            assert_eq!(cap.status[0], 1);
            // Restoring finite operands does not reset shared status. Bad output bits
            // are retained above but deliberately do not define an invalid oracle.
            io.a.with_mut(|v| v.copy_from_slice(&a));
            let sticky = run(&gpu, &op, &io, &format!("sticky/{label}"));
            assert_eq!(sticky.status[0], 1);
            recovery.compare(c, p, &sticky.out, 0);
            io.status.with_mut(|v| v[0] = 0);
            let good = run(&gpu, &op, &io, &format!("recovery/{label}"));
            assert_eq!(good.status[0], 0);
            recovery.compare(c, p, &good.out, 0);
        }
        // Poison every non-view parent word; neither active view may depend on it.
        let mut poison = a.clone();
        for (i, w) in poison.iter_mut().enumerate() {
            let visible = |v: View| {
                i.checked_sub(v.offset)
                    .is_some_and(|k| k / v.stride < p.rows && k % v.stride < p.width)
            };
            if !visible(p.a) && !visible(p.b) {
                *w = 0xfc00;
            }
        }
        let left = Operands::new(&gpu, p, &a, None);
        let right = Operands::new(&gpu, p, &poison, None);
        let l = run(&gpu, &op, &left, "inactive/baseline");
        let r = run(&gpu, &op, &right, "inactive/poisoned");
        assert_eq!((l.status[0], r.status[0]), (0, 0));
        assert_eq!(active(p, &l.out), active(p, &r.out));
        recovery.compare(c, p, &l.out, 0);
        recovery.compare(c, p, &r.out, 0);
        // Future last-row change cannot change an earlier row of this pure product.
        let mut changed = a.clone();
        changed[p.b.offset + (p.rows - 1) * p.b.stride
            ..p.b.offset + (p.rows - 1) * p.b.stride + p.width]
            .fill(f16::from_f32(3.).to_bits());
        let io = Operands::new(&gpu, p, &changed, None);
        let tail = run(&gpu, &op, &io, "causal/changed_last");
        assert_eq!(tail.status[0], 0);
        let x = active(p, &l.out);
        let y = active(p, &tail.out);
        assert_eq!(&x[..p.width], &y[..p.width]);
        assert_ne!(&x[p.width..], &y[p.width..]);
        assert!(y.iter().all(|w| f16::from_bits(*w).is_finite()));
        assert_eq!(controls["invalid"].as_array().unwrap().len(), 8);
        assert_eq!(recovery.values, 9216);
        println!(
            "PRODUCT_RECOVERY_RESULT {}",
            json!({"invalid_sites":8,"sticky_reset_phases":24,"extra_inactive_causal_commands":3,"recovery_main_values":recovery.values,"main_failures":0,"max_error":recovery.max_error,"max_fraction":recovery.max_fraction,"wide_exceedances":recovery.wide_exceedances,"invalid_output_words_retained_not_gated":true,"causal_equal_prefix_words":256})
        );
    });
}
