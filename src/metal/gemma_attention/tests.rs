use super::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn options() -> Options {
    Options {
        max_queries: 32,
        max_keys: 1025,
        query_heads: 8,
        head_dim: 512,
    }
}
fn shape() -> SpanAttentionShape {
    SpanAttentionShape {
        queries: 1,
        keys: 1,
        query_heads: 8,
        kv_heads: 1,
        head_dim: 512,
        query_start: 0,
        key_start: 0,
        window: None,
        scale: 1.,
    }
}
fn empty() -> AttentionKvSegment<'static> {
    AttentionKvSegment {
        k: &[],
        v: &[],
        keys: 0,
        head_stride: usize::MAX,
        key_start: usize::MAX,
    }
}
#[test]
fn capacity_and_admission_precede_gpu_allocation() {
    assert!(matches!(
        Context::new(CONTEXT_ALLOWANCE_BYTES - 1),
        Err(Error::Budget { .. })
    ));
    for index in 0..4 {
        for invalid in [0, usize::MAX] {
            let mut o = options();
            match index {
                0 => o.max_queries = invalid,
                1 => o.max_keys = invalid,
                2 => o.query_heads = invalid,
                _ => o.head_dim = invalid,
            }
            assert!(o.planned_bytes().is_err());
        }
    }
    for o in [
        Options {
            max_queries: 33,
            ..options()
        },
        Options {
            max_keys: 1026,
            ..options()
        },
        Options {
            query_heads: 9,
            ..options()
        },
        Options {
            head_dim: 513,
            ..options()
        },
    ] {
        assert!(o.planned_bytes().is_err());
    }
    let sizes = Sizes::new(options()).unwrap();
    assert_eq!(sizes.query, 32 * 8 * 512);
    assert_eq!(sizes.kv, 1025 * 512);
    assert_eq!(sizes.scores, 8 * 32 * 1025);
    assert_eq!(
        sizes.owned,
        (2 * (32 * 8 * 512 + GUARD)
            + 6 * (1025 * 512 + GUARD)
            + 2 * (8 * 32 * 1025 + GUARD)
            + 1
            + GUARD)
            * 4
            + size_of::<Workspace<'_>>()
    );
}
#[test]
fn checked_geometry_positions_origins_empty_spans_and_visibility() {
    let data = [0.; 512];
    let mut s = shape();
    let spans = [
        AttentionKvSegment {
            k: &data,
            v: &data,
            keys: 1,
            head_stride: 512,
            key_start: 0,
        },
        empty(),
        empty(),
    ];
    let g = Geometry::new(options(), 4096, 4096, &spans, &s).unwrap();
    assert_eq!(g.words[11..14], [512, 0, 0]);
    assert!(Geometry::new(options(), 4095, 4096, &spans, &s).is_err());
    assert!(Geometry::new(options(), 4096, 4095, &spans, &s).is_err());
    for invalid in [0, 1, 9, usize::MAX] {
        s.query_heads = invalid;
        assert!(Geometry::new(options(), 4096, 4096, &spans, &s).is_err());
    }
    s = shape();
    for invalid in [0, 256, 513, usize::MAX] {
        s.head_dim = invalid;
        assert!(Geometry::new(options(), 4096, 4096, &spans, &s).is_err());
    }
    s = shape();
    for invalid in [0, 33, usize::MAX] {
        s.queries = invalid;
        assert!(Geometry::new(options(), 4096, 4096, &spans, &s).is_err());
    }
    s = shape();
    for invalid in [0, 1026, usize::MAX] {
        s.keys = invalid;
        assert!(Geometry::new(options(), 4096, 4096, &spans, &s).is_err());
    }
    s = shape();
    for scale in [
        0.,
        -1.,
        f32::NAN,
        f32::INFINITY,
        f32::from_bits(1f32.to_bits() + 1),
    ] {
        s.scale = scale;
        assert!(Geometry::new(options(), 4096, 4096, &spans, &s).is_err());
    }
    s = shape();
    for window in [Some(0), Some(usize::MAX)] {
        s.window = window;
        assert!(Geometry::new(options(), 4096, 4096, &spans, &s).is_err());
    }
    s = shape();
    s.kv_heads = 2;
    assert!(Geometry::new(options(), 4096, 4096, &spans, &s).is_err());
    s = shape();
    s.query_start = usize::MAX;
    assert!(Geometry::new(options(), 4096, 4096, &spans, &s).is_err());
    s = shape();
    s.key_start = usize::MAX;
    assert!(Geometry::new(options(), 4096, 4096, &spans, &s).is_err());
    s = shape();
    s.query_start = 9;
    s.window = Some(4);
    assert!(Geometry::new(options(), 4096, 4096, &spans, &s).is_err());
    s = shape();
    s.key_start = u32::MAX as usize;
    s.query_start = s.key_start;
    let translated = [
        AttentionKvSegment {
            key_start: s.key_start,
            ..spans[0]
        },
        empty(),
        empty(),
    ];
    assert!(Geometry::new(options(), 4096, 4096, &translated, &s).is_ok());
    s.keys = 2;
    assert!(Geometry::new(options(), 4096, 4096, &translated, &s).is_err());
    s.keys = 1;
    s.queries = 2;
    assert!(Geometry::new(options(), 8192, 8192, &translated, &s).is_err());
    s = shape();
    s.key_start = 1;
    let future = [
        AttentionKvSegment {
            key_start: 1,
            ..spans[0]
        },
        empty(),
        empty(),
    ];
    assert!(Geometry::new(options(), 4096, 4096, &future, &s).is_err());
    for bad in [
        AttentionKvSegment {
            key_start: 1,
            ..spans[0]
        },
        AttentionKvSegment {
            head_stride: 511,
            ..spans[0]
        },
        AttentionKvSegment {
            keys: usize::MAX,
            ..spans[0]
        },
        AttentionKvSegment { v: &[], ..spans[0] },
    ] {
        assert!(Geometry::new(options(), 4096, 4096, &[bad, empty(), empty()], &shape()).is_err());
    }
}
struct Fixture {
    manifest: Value,
    arrays: Vec<u8>,
}
fn fixture_files() -> &'static Fixture {
    static FIXTURE: std::sync::OnceLock<Fixture> = std::sync::OnceLock::new();
    FIXTURE.get_or_init(|| {
        // Runtime loading avoids embedding 5.6 MiB into every test compilation.
        // Fixed repository-only paths and exact bounded reads keep tests offline
        // and reject oversized/replaced files before parsing any expectations.
        fn read(name: &str, length: usize, digest: &str) -> Vec<u8> {
            use std::io::Read;
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/gemma-reference-attention-v1")
                .join(name);
            let mut file = std::fs::File::open(path).unwrap();
            assert_eq!(file.metadata().unwrap().len(), length as u64);
            let mut bytes = vec![0; length];
            file.read_exact(&mut bytes).unwrap();
            assert_eq!(file.read(&mut [0u8; 1]).unwrap(), 0);
            assert_eq!(format!("{:x}", Sha256::digest(&bytes)), digest);
            bytes
        }
        let manifest = read(
            "manifest.json",
            34549,
            "36365f416c5f9572bfad7d6ff42c51760e43085025857648905350f86924edd8",
        );
        let arrays = read(
            "arrays.bin",
            5597808,
            "f7a13f39536c816d566a5caf167b6f439f11b59b34ae206561b93d5ac28a1b7f",
        );
        Fixture {
            manifest: serde_json::from_slice(&manifest).unwrap(),
            arrays,
        }
    })
}
fn fixture() -> &'static Value {
    &fixture_files().manifest
}
fn n(v: &Value, k: &str) -> usize {
    v[k].as_u64().unwrap() as usize
}
fn recipe(case: &str, kind: &str, head: usize, p: usize, f: usize) -> f32 {
    if case == "cancellation" {
        return match kind {
            "q" => [1024., 1., -1024., -1.][f % 4],
            "k" => {
                (1. + (p % 2) as f64 * 2f64.powi(-20)
                    + if f % 4 == 1 {
                        (p % 7) as f64 * 2f64.powi(-24)
                    } else {
                        0.
                    }) as f32
            }
            _ => {
                ((if p == 512 {
                    0.
                } else {
                    [1024., 1., -1024., -1.][p % 4]
                }) + (f % 7) as f64 * 2f64.powi(-8)) as f32
            }
        };
    }
    let shift = match kind {
        "q" => 3,
        "k" => 11,
        _ => 23,
    };
    ((((head * 31 + p * 17 + f * 7 + shift) % 53) as f64 - 26.) / 13.) as f32
}
fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn input(data: &Value, case: &Value, kind: &str) -> Vec<f32> {
    let name = case["name"].as_str().unwrap();
    let d = n(case, "head_dim");
    let h = if kind == "q" { n(case, "heads") } else { 1 };
    let rows = if kind == "q" {
        n(case, "queries")
    } else {
        n(case, "keys")
    };
    let start = n(
        case,
        if kind == "q" {
            "query_start"
        } else {
            "key_start"
        },
    );
    let mut a = Vec::new();
    for head in 0..h {
        for row in 0..rows {
            for f in 0..d {
                a.push(recipe(name, kind, head, start + row, f));
            }
        }
    }
    let desc = data["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|x| x["case"] == case["name"] && x["contract"] == "staged_fp32" && x["name"] == kind)
        .unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes(&a))),
        desc["sha256"].as_str().unwrap()
    );
    if kind != "q" {
        return a;
    }
    let mut out = vec![0.; a.len()];
    for head in 0..h {
        for row in 0..rows {
            for f in 0..d {
                out[(row * h + head) * d + f] = a[(head * rows + row) * d + f];
            }
        }
    }
    out
}
#[test]
fn independent_fixture_and_input_recipes_authenticate_without_gpu() {
    let data = fixture();
    for case in data["cases"].as_array().unwrap() {
        for kind in ["q", "k", "v"] {
            let _ = input(data, case, kind);
        }
    }
}
fn expected(data: &Value, case: &Value, key: &str) -> Vec<f32> {
    let desc = data["arrays"]
        .as_array()
        .unwrap()
        .iter()
        .find(|x| x["case"] == case["name"] && x["contract"] == "staged_fp32" && x["name"] == key)
        .unwrap();
    let raw = &fixture_files().arrays;
    let slice = &raw[n(desc, "offset")..n(desc, "offset") + n(desc, "bytes")];
    assert_eq!(
        format!("{:x}", Sha256::digest(slice)),
        desc["sha256"].as_str().unwrap()
    );
    slice
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn capture(w: &Workspace<'_>, name: &str, g: &Geometry, result: &Result<()>) {
    use std::io::Write;
    // Read raw buffers only after a checked terminal completion. A failed native
    // command is unavailable; no shared-memory read is used to diagnose it.
    let readable = w.pending.as_ref().map_or(!w.poisoned, |c| {
        c.status() == MTLCommandBufferStatus::Completed
    });
    let mut record = json!({"case":name,"success":result.is_ok(),"poisoned":w.is_poisoned(),"owned_bytes":w.owned_bytes(),"shape":g.words,"readable":readable});
    if readable {
        let mut raw = Vec::new();
        for (buffer, count) in [
            (&w.scores, g.scores),
            (&w.probabilities, g.scores),
            (&w.output, g.query),
            (&w.status, 1),
        ] {
            buffer.with_slice(|s| {
                for &x in &s[..count + GUARD] {
                    raw.extend_from_slice(&x.to_le_bytes());
                }
            });
        }
        record["raw_sha256"] = json!(format!("{:x}", Sha256::digest(&raw)));
        record["raw_bytes"] = json!(raw.len());
        record["status"] = json!(w.status.with_slice(|s| s[0]));
        if let Some(directory) = std::env::var_os("TOKLET_GEMMA_ATTENTION_CAPTURE") {
            let path = std::path::PathBuf::from(directory).join(format!("{name}.u32le"));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, &raw).unwrap();
            record["raw_path"] = json!(path);
        }
    }
    println!("GEMMA_ATTENTION_CAPTURE {record}");
    std::io::stdout().flush().unwrap();
}
fn close(actual: &[f32], expected: &[f32]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0f64;
    for (&a, &e) in actual.iter().zip(expected) {
        let fraction = (f64::from(a) - f64::from(e)).abs() / (1e-5 + 1e-4 * f64::from(e).abs());
        assert!(a.is_finite() && fraction <= 1., "{a} vs {e}: {fraction}");
        maximum = maximum.max(fraction);
    }
    maximum
}
#[test]
#[ignore = "explicit Apple GPU reusable FP32 attention frozen numerical qualification"]
fn frozen_math_segments_reuse_and_diagnostics() {
    let data = fixture();
    let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let mut total = 0;
    let mut maximum = 0f64;
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let qn = n(case, "queries");
        let keys = n(case, "keys");
        let h = n(case, "heads");
        let d = n(case, "head_dim");
        let options = Options {
            query_heads: h,
            head_dim: d,
            ..options()
        };
        let budget = options.planned_bytes().unwrap();
        assert!(matches!(
            context.workspace(options, budget - 1),
            Err(Error::Budget { .. })
        ));
        let mut w = context.workspace(options, budget).unwrap();
        let q = input(data, case, "q");
        let k = input(data, case, "k");
        let v = input(data, case, "v");
        let shape = SpanAttentionShape {
            queries: qn,
            keys,
            query_heads: h,
            kv_heads: 1,
            head_dim: d,
            query_start: n(case, "query_start"),
            key_start: n(case, "key_start"),
            window: case["window"].as_u64().map(|x| x as usize),
            scale: 1.,
        };
        let mut baseline = None;
        for layout in 0..2 {
            let old = keys - qn;
            let parts = if layout == 0 {
                [keys, 0, 0]
            } else {
                [old / 2, old - old / 2, qn]
            };
            let mut cursor = 0;
            let spans = parts.map(|count| {
                let start = cursor;
                cursor += count;
                AttentionKvSegment {
                    k: &k[start * d..cursor * d],
                    v: &v[start * d..cursor * d],
                    keys: count,
                    head_stride: count * d + 3,
                    key_start: shape.key_start + start,
                }
            });
            let g = Geometry::new(options, q.len(), q.len(), &spans, &shape).unwrap();
            let mut out = vec![42.; q.len()];
            let result = w.apply(&q, &spans, &mut out, &shape);
            capture(&w, &format!("{name}-layout{layout}"), &g, &result);
            result.unwrap();
            assert_eq!(w.owned_bytes(), budget);
            assert!(!w.is_poisoned());
            assert_eq!(
                w.q.with_slice(|s| s[..q.len()].to_vec()),
                q.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            let mut scores = vec![0.; g.scores];
            let mut probs = vec![0.; g.scores];
            w.copy_diagnostics(&mut scores, &mut probs).unwrap();
            let mut context_expected = expected(data, case, "context");
            let head_major = context_expected.clone();
            for t in 0..qn {
                for head in 0..h {
                    for f in 0..d {
                        context_expected[(t * h + head) * d + f] =
                            head_major[(head * qn + t) * d + f];
                    }
                }
            }
            for (a, e) in [
                (&scores, expected(data, case, "scores")),
                (&probs, expected(data, case, "probabilities")),
                (&out, context_expected),
            ] {
                maximum = maximum.max(close(a, &e));
                total += a.len();
            }
            for head in 0..h {
                for row in 0..qn {
                    for key in 0..keys {
                        let absolute = shape.key_start + key;
                        let query = shape.query_start + row;
                        if absolute > query
                            || shape
                                .window
                                .is_some_and(|window| query.saturating_sub(absolute) >= window)
                        {
                            assert_eq!(probs[(head * qn + row) * keys + key].to_bits(), 0);
                        }
                    }
                }
            }
            let bits = out.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            if let Some(old) = &baseline {
                assert_eq!(&bits, old);
            } else {
                baseline = Some(bits);
            }
            let mut untouched = [17.; 1];
            assert!(w.copy_diagnostics(&mut untouched, &mut probs).is_err());
            assert_eq!(untouched, [17.; 1]);
        }
    }
    assert_eq!(total, 932968);
    println!(
        "GEMMA_ATTENTION_NUMERICS {}",
        json!({"values":total,"max_gate_fraction":maximum,"graph_qualified":false})
    );
}
fn one_span<'a>(
    k: &'a [f32],
    v: &'a [f32],
    keys: usize,
    dim: usize,
) -> [AttentionKvSegment<'a>; 3] {
    [
        AttentionKvSegment {
            k,
            v,
            keys,
            head_stride: keys * dim,
            key_start: 0,
        },
        empty(),
        empty(),
    ]
}
#[test]
#[ignore = "explicit Apple GPU attention preflight preservation, poison and shape reuse"]
fn preflight_preserves_and_execution_failure_poisoning() {
    let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let o = Options {
        max_queries: 3,
        max_keys: 7,
        query_heads: 2,
        head_dim: 64,
    };
    let mut w = context.workspace(o, o.planned_bytes().unwrap()).unwrap();
    let mut s = SpanAttentionShape {
        queries: 3,
        keys: 7,
        query_heads: 2,
        kv_heads: 1,
        head_dim: 64,
        query_start: 4,
        key_start: 0,
        window: Some(4),
        scale: 1.,
    };
    let mut q = vec![0.25; 3 * 2 * 64];
    let mut k = vec![0.5; 7 * 64];
    let mut v = vec![1.; 7 * 64];
    let mut out = vec![42.; q.len()];

    w.apply(&q, &one_span(&k, &v, 7, 64), &mut out, &s).unwrap();
    let prior = out.clone();
    let mut prior_sc = vec![0.; 42];
    let mut prior_pr = prior_sc.clone();
    w.copy_diagnostics(&mut prior_sc, &mut prior_pr).unwrap();
    q[0] = f32::NAN;
    assert!(w.apply(&q, &one_span(&k, &v, 7, 64), &mut out, &s).is_err());
    assert_eq!(out, prior);
    assert!(!w.is_poisoned());
    q[0] = 0.25;
    // Masked rows are still validated before staging, preserving prior success.
    for key_input in [true, false] {
        if key_input {
            k[6 * 64] = f32::INFINITY;
        } else {
            v[6 * 64] = f32::NAN;
        }
        assert!(w.apply(&q, &one_span(&k, &v, 7, 64), &mut out, &s).is_err());
        assert_eq!(out, prior);
        assert!(!w.is_poisoned());
        k[6 * 64] = 0.5;
        v[6 * 64] = 1.;
    }
    let mut sc = vec![0.; 42];
    let mut pr = sc.clone();
    w.copy_diagnostics(&mut sc, &mut pr).unwrap();
    assert_eq!(sc, prior_sc);
    assert_eq!(pr, prior_pr);
    s.query_start = 100;
    assert!(w.apply(&q, &one_span(&k, &v, 7, 64), &mut out, &s).is_err());
    assert_eq!(out, prior);
    assert!(!w.is_poisoned());
    s.query_start = 4;
    // Isolate only the future-masked (query0,head0,key6) overflow.
    for row in q.chunks_exact_mut(64) {
        row[0] = 0.;
    }
    for row in k.chunks_exact_mut(64) {
        row[0] = 0.;
    }
    q[0] = f32::MAX;
    k[6 * 64] = f32::MAX;
    let mut overflow = Vec::new();
    for query in 0..3 {
        for head in 0..2 {
            for key in 0..7 {
                let score: f64 = (0..64)
                    .map(|f| f64::from(q[(query * 2 + head) * 64 + f]) * f64::from(k[key * 64 + f]))
                    .sum();
                if !(score as f32).is_finite() {
                    overflow.push((query, head, key));
                }
            }
        }
    }
    assert_eq!(overflow, [(0, 0, 6)]);
    let g = Geometry::new(o, q.len(), out.len(), &one_span(&k, &v, 7, 64), &s).unwrap();
    let result = w.apply(&q, &one_span(&k, &v, 7, 64), &mut out, &s);
    capture(&w, "isolated-masked-overflow", &g, &result);
    assert!(matches!(result, Err(Error::Nonfinite)));
    assert_eq!(out, prior);
    assert!(w.is_poisoned());
    assert!(matches!(
        w.apply(&q, &one_span(&k, &v, 7, 64), &mut out, &s),
        Err(Error::ReplayRequired)
    ));
    assert!(w.copy_diagnostics(&mut sc, &mut pr).is_err());
    // Recreate only after a poisoned execution; capacities then handle shrink/grow.
    let mut w = context.workspace(o, o.planned_bytes().unwrap()).unwrap();
    q.fill(0.25);
    k.fill(0.5);
    let owned = w.owned_bytes();
    for (rows, keys) in [(3, 7), (1, 1), (2, 4), (3, 7)] {
        s.queries = rows;
        s.keys = keys;
        s.query_start = keys - rows;
        let mut out = vec![42.; rows * 2 * 64];
        w.apply(
            &q[..out.len()],
            &one_span(&k[..keys * 64], &v[..keys * 64], keys, 64),
            &mut out,
            &s,
        )
        .unwrap();
        assert!(out.iter().all(|&x| (x - 1.).abs() < 1e-6));
        assert!(!w.is_poisoned());
        assert_eq!(w.owned_bytes(), owned);
    }
}

#[test]
#[ignore = "explicit Apple GPU borrowed offset/ring attention encoding and binding admission"]
fn borrowed_ranges_allow_ring_inputs_and_disjoint_shared_scratch() {
    let gpu = Gpu::new().unwrap();
    let pipelines = Pipelines::new(&gpu).unwrap();
    let mut q = Buffer::new(&gpu, 2 * 2 * 3 + GUARD).unwrap();
    q.with_mut(|s| {
        s.fill(SENTINEL);
        s[..12].fill(0);
    });
    let mut k = Buffer::new(&gpu, 24).unwrap();
    let mut v = Buffer::new(&gpu, 24).unwrap();
    k.with_mut(|s| s.fill(0));
    v.with_mut(|s| s.fill(SENTINEL));
    // Logical keys 0,1 live after 2,3 in the physical ring; key4 is proposed.
    let offsets = [9, 1, 18];
    for (part, (start, count)) in [(0, 2), (2, 2), (4, 1)].into_iter().enumerate() {
        v.with_mut(|s| {
            for key in 0..count {
                for f in 0..3 {
                    s[offsets[part] + key * 3 + f] =
                        ((start + key + 1) as f32 + f as f32 * 0.125).to_bits();
                }
            }
        });
    }
    let before_q = q.with_slice(<[u32]>::to_vec);
    let before_k = k.with_slice(<[u32]>::to_vec);
    let before_v = v.with_slice(<[u32]>::to_vec);
    let mut scratch = Buffer::new(&gpu, 53 + GUARD).unwrap();
    scratch.with_mut(|s| {
        s.fill(SENTINEL);
        s[52] = 0;
    });
    assert!(BufferRange::new(&scratch, usize::MAX, 1).is_err());
    assert!(BufferRange::new(&scratch, 52, usize::MAX).is_err());
    assert!(BufferRange::new(&scratch, scratch.len, 0).is_err());
    assert!(BufferRange::new(&scratch, scratch.len - 1, 2).is_err());
    let segment = |i: usize, keys, key_start| DeviceSegment {
        k: BufferRange::new(&k, offsets[i], keys * 3).unwrap(),
        v: BufferRange::new(&v, offsets[i], keys * 3).unwrap(),
        keys,
        head_stride: keys * 3,
        key_start,
    };
    let mut buffers = DeviceBuffers {
        q: BufferRange::new(&q, 0, 12).unwrap(),
        segments: [segment(0, 2, 0), segment(1, 2, 2), segment(2, 1, 4)],
        scores: BufferRange::new(&scratch, 0, 20).unwrap(),
        probabilities: BufferRange::new(&scratch, 20, 20).unwrap(),
        output: BufferRange::new(&scratch, 40, 12).unwrap(),
        status: BufferRange::new(&scratch, 52, 1).unwrap(),
    };
    let s = SpanAttentionShape {
        queries: 2,
        keys: 5,
        query_heads: 2,
        kv_heads: 1,
        head_dim: 3,
        query_start: 3,
        key_start: 0,
        window: Some(3),
        scale: 1.,
    };
    let command = gpu.command().unwrap();
    let output = buffers.output;
    buffers.output = buffers.q;
    assert!(pipelines.encode(&command, &buffers, &s).is_err());
    buffers.output = output;
    let status = buffers.status;
    buffers.status = BufferRange::new(&scratch, 20, 1).unwrap();
    assert!(pipelines.encode(&command, &buffers, &s).is_err());
    buffers.status = status;
    let scores = buffers.scores;
    buffers.scores.len -= 1;
    assert!(pipelines.encode(&command, &buffers, &s).is_err());
    buffers.scores = scores;
    assert_eq!(command.status(), MTLCommandBufferStatus::NotEnqueued);
    pipelines.encode(&command, &buffers, &s).unwrap();
    command.commit();
    command.waitUntilCompleted();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    scratch.with_slice(|s| {
        assert_eq!(s[52], 0);
        assert!(tail(s, 53));
        for row in 0..2 {
            for head in 0..2 {
                for f in 0..3 {
                    let actual = f32::from_bits(s[40 + (row * 2 + head) * 3 + f]);
                    let expected = (row + 3) as f32 + f as f32 * 0.125;
                    assert!((actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs());
                }
            }
        }
        for head in 0..2 {
            for row in 0..2 {
                for key in 0..5 {
                    if key > row + 3 || key < row + 1 {
                        assert_eq!(s[20 + (head * 2 + row) * 5 + key], 0);
                    }
                }
            }
        }
    });
    assert_eq!(q.with_slice(<[u32]>::to_vec), before_q);
    assert_eq!(k.with_slice(<[u32]>::to_vec), before_k);
    assert_eq!(v.with_slice(<[u32]>::to_vec), before_v);
    assert!(pipelines.encode(&command, &buffers, &s).is_err());
}
