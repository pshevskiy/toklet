use super::*;
use crate::metal::memory::Buffer;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::OpenOptions,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
};
const GUARD: usize = 7;
const SENTINEL: u32 = 0x7fc1_2345;
#[derive(Clone, Copy)]
struct Region {
    start: usize,
    len: usize,
}
impl Region {
    fn end(self) -> usize {
        self.start + self.len
    }
    fn range(self, arena: &Buffer<u32>) -> BufferRange<'_> {
        BufferRange::new(arena, self.start, self.len).unwrap()
    }
}
fn regions(lengths: &[usize]) -> (Vec<Region>, usize) {
    let mut cursor = GUARD;
    let out = lengths
        .iter()
        .map(|&len| {
            let r = Region { start: cursor, len };
            cursor = r.end() + GUARD;
            r
        })
        .collect();
    (out, cursor)
}
#[derive(Serialize)]
struct Sidecar {
    path: String,
    words: usize,
    bytes: usize,
    sha256: String,
}
fn raw_dir() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = if let Some(path) = std::env::var_os("TOKLET_RESIDUAL_RAW_DIR") {
        let path = PathBuf::from(path);
        std::fs::create_dir_all(&path).unwrap();
        path
    } else {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "toklet-residual-{}-{nanos}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        path
    };
    path.canonicalize().unwrap()
}
fn capture(dir: &Path, case: &str, phase: &str, words: &[u32]) -> Sidecar {
    let path = dir.join(format!("{case}-{phase}.u32le"));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    let mut out = BufWriter::with_capacity(16384, file);
    let mut sha = Sha256::new();
    let mut block = [0u8; 4096];
    for chunk in words.chunks(block.len() / 4) {
        for (word, bytes) in chunk.iter().zip(block.chunks_exact_mut(4)) {
            bytes.copy_from_slice(&word.to_le_bytes());
        }
        let raw = &block[..chunk.len() * 4];
        out.write_all(raw).unwrap();
        sha.update(raw);
    }
    out.flush().unwrap();
    assert_eq!(
        out.get_ref().metadata().unwrap().len(),
        (words.len() * 4) as u64
    );
    Sidecar {
        path: path.to_str().unwrap().into(),
        words: words.len(),
        bytes: words.len() * 4,
        sha256: format!("{:x}", sha.finalize()),
    }
}
#[derive(Serialize)]
struct Raw<'a> {
    case: &'a str,
    kind: &'a str,
    scalar_bits: Option<u32>,
    regions: Vec<[usize; 2]>,
    before_words: Sidecar,
    after_words: Sidecar,
    expected_status: Vec<u32>,
    derived_arithmetic_oracle: bool,
    independent_block_witness: bool,
    learned_scalar_authenticated: bool,
}
fn report(
    dir: &Path,
    case: &str,
    kind: &str,
    scalar: Option<f32>,
    r: &[Region],
    words: (&[u32], &[u32]),
    status: &[u32],
) {
    let (before, after) = words;
    let raw = Raw {
        case,
        kind,
        scalar_bits: scalar.map(f32::to_bits),
        regions: r.iter().map(|v| [v.start, v.len]).collect(),
        before_words: capture(dir, case, "before", before),
        after_words: capture(dir, case, "after", after),
        expected_status: status.to_vec(),
        derived_arithmetic_oracle: true,
        independent_block_witness: false,
        learned_scalar_authenticated: false,
    };
    let stdout = std::io::stdout();
    let mut out = BufWriter::with_capacity(16384, stdout.lock());
    write!(out, "GEMMA_FP32_RESIDUAL_RAW ").unwrap();
    serde_json::to_writer(&mut out, &raw).unwrap();
    writeln!(out).unwrap();
    out.flush().unwrap();
}
fn immutable(before: &[u32], after: &[u32], writable: &[Region]) {
    assert_eq!(before.len(), after.len());
    for (i, (&a, &b)) in after.iter().zip(before).enumerate() {
        if !writable.iter().any(|r| (r.start..r.end()).contains(&i)) {
            assert_eq!(a, b, "immutable/guard{i}");
        }
    }
}
fn exact(actual: &[u32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a, e.to_bits(), "exact FP32 index{i}");
    }
}
fn reference_add(left: &[f32], right: &[f32]) -> Vec<f32> {
    let mut out = left.to_vec();
    crate::cpu::residual_add(&mut out, right).unwrap();
    for ((&a, &b), &result) in left.iter().zip(right).zip(&out) {
        // Independent exact-enough FP64 sum followed by one FP32 boundary.
        assert_eq!(
            result.to_bits(),
            ((f64::from(a) + f64::from(b)) as f32).to_bits()
        );
    }
    out
}
fn reference_scale(input: &[f32], scalar: f32) -> Vec<f32> {
    input
        .iter()
        .map(|&v| {
            let original = v * scalar;
            let oracle = (f64::from(v) * f64::from(scalar)) as f32;
            assert_eq!(original.to_bits(), oracle.to_bits());
            original
        })
        .collect()
}
#[test]
fn extent_status_and_scalar_boundary_checks_without_gpu() {
    for n in [1, 31, 32, 33, 1536, MAX_VALUES] {
        assert_eq!(geometry(n, n, 1, 32).unwrap(), n.div_ceil(32));
    }
    for (n, out, status, width) in [
        (0, 0, 1, 32),
        (MAX_VALUES + 1, MAX_VALUES + 1, 1, 32),
        (usize::MAX, usize::MAX, 1, 32),
        (1, 2, 1, 32),
        (1, 1, 0, 32),
        (1, 1, 2, 32),
        (1, 1, 1, 0),
        (2, 2, 1, usize::MAX),
    ] {
        assert!(geometry(n, out, status, width).is_err());
    }
    assert_eq!(decision(0).unwrap(), Decision::Accepted);
    assert_eq!(decision(2).unwrap(), Decision::ReferenceReplay);
    for flags in [1, 3] {
        assert!(matches!(decision(flags), Err(Error::Nonfinite)));
    }
    for flags in [4, 5, 6, 7, u32::MAX] {
        assert!(matches!(decision(flags), Err(Error::Options(_))));
    }
    assert_eq!(
        reference_add(&[0., 0., -0., -0.], &[0., -0., 0., -0.])
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        [0, 0, 0, 0x80000000]
    );
    for x in [0., -0., 1., -1., f32::from_bits(1), -f32::from_bits(1)] {
        for scalar in [0_f32, -0., 1., -1.] {
            assert_eq!(
                reference_scale(&[x], scalar)[0].to_bits(),
                (x * scalar).to_bits()
            );
        }
    }
}
#[derive(Clone, Copy)]
enum Operation {
    Add,
    Scale(f32),
    AddScale(f32),
}
fn exercise(
    gpu: &Gpu,
    p: &Pipelines,
    dir: &Path,
    name: &str,
    operands: (&[f32], &[f32]),
    operation: Operation,
    flags: [u32; 2],
) {
    let (left, right) = operands;
    let n = left.len();
    assert_eq!(right.len(), n);
    let (r, count) = regions(&[n, n, n, n, 1, 1]);
    let mut arena = Buffer::<u32>::new(gpu, count).unwrap();
    arena.with_mut(|s| {
        s.fill(SENTINEL);
        for (range, values) in [(r[0], left), (r[1], right)] {
            for (out, value) in s[range.start..range.end()].iter_mut().zip(values) {
                *out = value.to_bits();
            }
        }
        s[r[4].start] = 0;
        s[r[5].start] = 0;
    });
    let before = arena.with_slice(|s| s.to_vec());
    let command = gpu.command().unwrap();
    let (kind, scalar) = match operation {
        Operation::Add => {
            p.encode_add(
                &command,
                r[0].range(&arena),
                r[1].range(&arena),
                r[3].range(&arena),
                r[4].range(&arena),
            )
            .unwrap();
            ("add", None)
        }
        Operation::Scale(s) => {
            p.encode_scale(
                &command,
                r[0].range(&arena),
                r[3].range(&arena),
                r[5].range(&arena),
                s,
            )
            .unwrap();
            ("scale", Some(s))
        }
        Operation::AddScale(s) => {
            p.encode_add(
                &command,
                r[0].range(&arena),
                r[1].range(&arena),
                r[2].range(&arena),
                r[4].range(&arena),
            )
            .unwrap();
            p.encode_scale(
                &command,
                r[2].range(&arena),
                r[3].range(&arena),
                r[5].range(&arena),
                s,
            )
            .unwrap();
            ("separate-add-scale", Some(s))
        }
    };
    command.commit();
    command.waitUntilCompleted();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    arena.with_slice(|after| {
        report(dir, name, kind, scalar, &r, (&before, after), &flags);
        assert!(command.error().is_none());
        let writable = match operation {
            Operation::Add => vec![r[3], r[4]],
            Operation::Scale(_) => vec![r[3], r[5]],
            Operation::AddScale(_) => r[2..].to_vec(),
        };
        immutable(&before, after, &writable);
        assert_eq!([after[r[4].start], after[r[5].start]], flags);
        let selected = match decision(flags[0]) {
            Ok(Decision::Accepted) => decision(flags[1]),
            other => other,
        };
        let accepted = matches!(&selected, Ok(Decision::Accepted));
        if accepted {
            let expected = match operation {
                Operation::Add => reference_add(left, right),
                Operation::Scale(s) => reference_scale(left, s),
                Operation::AddScale(s) => {
                    let add = reference_add(left, right);
                    exact(&after[r[2].start..r[2].end()], &add);
                    reference_scale(&add, s)
                }
            };
            assert!(expected.iter().all(|v| v.is_finite()));
            exact(&after[r[3].start..r[3].end()], &expected);
        } else if matches!(&selected, Ok(Decision::ReferenceReplay)) {
            // An earlier replay is selected before a later fatal descendant.
            let published = vec![99.; n];
            assert!(published.iter().all(|&v| v == 99.));
        }
    });
}
#[test]
#[ignore = "explicit GPU FP32 residual/scalar arithmetic, signed-zero and replay controls"]
fn exact_arithmetic_zero_identities_and_ordered_replay() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let p = Pipelines::new(&gpu).unwrap();
        let dir = raw_dir();
        let add = [
            (
                "all-zero-add-signs",
                vec![0., 0., -0., -0.],
                vec![0., -0., 0., -0.],
                [0, 0],
            ),
            (
                "zero-left-subnormal-copy",
                vec![0., -0., 0., -0.],
                vec![
                    f32::from_bits(1),
                    -f32::from_bits(1),
                    f32::from_bits(0x007fffff),
                    -f32::from_bits(0x007fffff),
                ],
                [0, 0],
            ),
            (
                "zero-right-subnormal-copy",
                vec![f32::from_bits(1), -f32::from_bits(1)],
                vec![0., -0.],
                [0, 0],
            ),
            (
                "exact-normal-cancellation",
                vec![1., -1., f32::MIN_POSITIVE, -f32::MIN_POSITIVE],
                vec![-1., 1., -f32::MIN_POSITIVE, f32::MIN_POSITIVE],
                [0, 0],
            ),
            (
                "subnormal-nonidentity-replay",
                vec![f32::from_bits(1), -f32::from_bits(1)],
                vec![1., -1.],
                [2, 0],
            ),
            (
                "normal-cancellation-underflow-replay",
                vec![f32::MIN_POSITIVE.next_up(), -f32::MIN_POSITIVE.next_up()],
                vec![-f32::MIN_POSITIVE, f32::MIN_POSITIVE],
                [2, 0],
            ),
            ("add-overflow-fatal", vec![f32::MAX], vec![f32::MAX], [1, 0]),
            (
                "nonfinite-before-right-zero",
                vec![f32::INFINITY, f32::NAN, f32::NEG_INFINITY],
                vec![0., -0., 0.],
                [1, 0],
            ),
            (
                "nonfinite-before-left-zero",
                vec![0., -0.],
                vec![f32::NAN, f32::INFINITY],
                [1, 0],
            ),
            (
                "same-operation-fatal-dominates-replay",
                vec![f32::from_bits(1), f32::MAX],
                vec![1., f32::MAX],
                [3, 0],
            ),
        ];
        for (name, a, b, flags) in add {
            exercise(&gpu, &p, &dir, name, (&a, &b), Operation::Add, flags);
        }
        let identities = [
            0.,
            -0.,
            1.,
            -1.,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MAX,
            -f32::MAX,
        ];
        for scalar in [0_f32, -0., 1., -1.] {
            exercise(
                &gpu,
                &p,
                &dir,
                &format!("scalar-identity-{:08x}", scalar.to_bits()),
                (&identities, &[0.; 8]),
                Operation::Scale(scalar),
                [0, 0],
            );
            exercise(
                &gpu,
                &p,
                &dir,
                &format!("scalar-nonfinite-before-identity-{:08x}", scalar.to_bits()),
                (&[f32::INFINITY, f32::NAN], &[0.; 2]),
                Operation::Scale(scalar),
                [0, 1],
            );
        }
        for scalar in [0.5_f32, -0.5, 2., -2., 0.9921875] {
            exercise(
                &gpu,
                &p,
                &dir,
                &format!("normal-scale-{:08x}", scalar.to_bits()),
                (&[0.25, -3., 1_f32.next_up(), f32::MAX / 4.], &[0.; 4]),
                Operation::Scale(scalar),
                [0, 0],
            );
        }
        for (name, input, scalar, flags) in [
            (
                "subnormal-amplification-replay",
                f32::from_bits(0x007fffff),
                f32::MAX,
                2,
            ),
            (
                "subnormal-scalar-amplification-replay",
                f32::MAX,
                f32::from_bits(1),
                2,
            ),
            ("normal-product-subnormal-replay", f32::MIN_POSITIVE, 0.5, 2),
            (
                "normal-product-zero-underflow-replay",
                f32::MIN_POSITIVE,
                f32::MIN_POSITIVE,
                2,
            ),
            ("scale-overflow-fatal", f32::MAX, 2., 1),
        ] {
            exercise(
                &gpu,
                &p,
                &dir,
                name,
                (&[input], &[0.]),
                Operation::Scale(scalar),
                [0, flags],
            );
        }
        // Rounding the add before multiply differs from an unrounded/fused expression.
        let a = 1.;
        let b = 2_f32.powi(-24);
        let scalar = 1_f32.next_up();
        let separate = reference_scale(&reference_add(&[a], &[b]), scalar)[0];
        let fused = ((f64::from(a) + f64::from(b)) * f64::from(scalar)) as f32;
        assert_ne!(separate.to_bits(), fused.to_bits());
        exercise(
            &gpu,
            &p,
            &dir,
            "separate-rounding-witness",
            (&[a], &[b]),
            Operation::AddScale(scalar),
            [0, 0],
        );
        exercise(
            &gpu,
            &p,
            &dir,
            "earlier-replay-discards-later-fatal",
            (&[f32::from_bits(1), f32::MAX], &[1., 0.]),
            Operation::AddScale(2.),
            [2, 1],
        );
        for n in [31, 33, 257, MAX_VALUES] {
            let a: Vec<_> = (0..n).map(|i| ((i % 17) as f32 - 8.) / 16.).collect();
            let b: Vec<_> = (0..n).map(|i| ((i % 13) as f32 - 6.) / 32.).collect();
            exercise(
                &gpu,
                &p,
                &dir,
                &format!("extent-{n}"),
                (&a, &b),
                Operation::AddScale(0.9921875),
                [0, 0],
            );
        }
    });
}

fn bounded_read(path: &Path, bytes: usize) -> Vec<u8> {
    assert_eq!(std::fs::metadata(path).unwrap().len(), bytes as u64);
    let mut out = Vec::with_capacity(bytes);
    std::fs::File::open(path)
        .unwrap()
        .take(bytes as u64 + 1)
        .read_to_end(&mut out)
        .unwrap();
    assert_eq!(out.len(), bytes);
    out
}
#[test]
#[ignore = "explicit saved original/future producer operand arithmetic; derived residual oracle only"]
fn saved_producer_operands_preserve_three_adds_then_separate_synthetic_scale() {
    autoreleasepool(|_| {
        let fixture = PathBuf::from(std::env::var_os("TOKLET_GEMMA_PRODUCER_FIXTURE").unwrap());
        let protocol = bounded_read(&fixture.join("protocol.json"), 301270);
        assert_eq!(
            format!("{:x}", Sha256::digest(&protocol)),
            "53de539d90152909feeb08e0ddf54700cbf6b992bb9dabaa9467d2eb7bc06fc2"
        );
        let protocol: Value = serde_json::from_slice(&protocol).unwrap();
        let states = bounded_read(&fixture.join("states.bin"), 2711552);
        assert_eq!(
            format!("{:x}", Sha256::digest(&states)),
            "29cdaff7bbcb8b1b71367594c01308932b4f1b78fcf57b9003a063d3bd1875ea"
        );
        assert_eq!(
            protocol["missing_independent_states"]
                .as_array()
                .unwrap()
                .len(),
            7
        );
        let gpu = Gpu::new().unwrap();
        let p = Pipelines::new(&gpu).unwrap();
        let dir = raw_dir();
        for case in protocol["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let rows = case["queries"].as_u64().unwrap() as usize;
            let n = rows * 1536;
            let get = |key: &str| -> Vec<f32> {
                let desc = &case["expected_states"][key];
                assert_eq!(desc["contiguous"], true);
                assert_eq!(desc["shape"], serde_json::json!([rows, 1536]));
                assert_eq!(desc["byte_strides"], serde_json::json!([6144, 4]));
                let start = desc["offset"].as_u64().unwrap() as usize;
                states[start..start + n * 4]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            };
            let input = get("input");
            let attention = get("post_attention_layernorm");
            let ff = get("post_feedforward_layernorm");
            let ple = get("post_per_layer_input_norm");
            let first = reference_add(&attention, &input);
            let second = reference_add(&ff, &first);
            let third = reference_add(&second, &ple);
            // The snapshot contains these four actual operands, but none of the
            // three residual states or learned scalar payload. Expectations here
            // are independently derived arithmetic, never invented block witnesses.
            for scalar in [0.5_f32, -1., 1.0078125] {
                let label = format!("saved-{name}-synthetic-{:08x}", scalar.to_bits());
                let scaled = reference_scale(&third, scalar);
                assert!(scaled.iter().all(|v| v.is_finite()));
                let (r, count) = regions(&[n, n, n, n, n, n, n, n, 1, 1, 1, 1]);
                let mut arena = Buffer::<u32>::new(&gpu, count).unwrap();
                arena.with_mut(|s| {
                    s.fill(SENTINEL);
                    for (range, source) in [
                        (r[0], &attention),
                        (r[1], &input),
                        (r[2], &ff),
                        (r[3], &ple),
                    ] {
                        for (word, value) in s[range.start..range.end()].iter_mut().zip(source) {
                            *word = value.to_bits();
                        }
                    }
                    for slot in &r[8..] {
                        s[slot.start] = 0;
                    }
                });
                let before = arena.with_slice(|s| s.to_vec());
                let command = gpu.command().unwrap();
                p.encode_add(
                    &command,
                    r[0].range(&arena),
                    r[1].range(&arena),
                    r[4].range(&arena),
                    r[8].range(&arena),
                )
                .unwrap();
                p.encode_add(
                    &command,
                    r[2].range(&arena),
                    r[4].range(&arena),
                    r[5].range(&arena),
                    r[9].range(&arena),
                )
                .unwrap();
                p.encode_add(
                    &command,
                    r[5].range(&arena),
                    r[3].range(&arena),
                    r[6].range(&arena),
                    r[10].range(&arena),
                )
                .unwrap();
                p.encode_scale(
                    &command,
                    r[6].range(&arena),
                    r[7].range(&arena),
                    r[11].range(&arena),
                    scalar,
                )
                .unwrap();
                command.commit();
                command.waitUntilCompleted();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                arena.with_slice(|after| {
                    report(
                        &dir,
                        &label,
                        "saved-three-residuals-separate-synthetic-scale",
                        Some(scalar),
                        &r,
                        (&before, after),
                        &[0; 4],
                    );
                    assert!(command.error().is_none());
                    immutable(&before, after, &r[4..]);
                    for slot in &r[8..] {
                        assert_eq!(after[slot.start], 0);
                        assert_eq!(decision(after[slot.start]).unwrap(), Decision::Accepted);
                    }
                    for (range, expected) in [
                        (r[4], &first),
                        (r[5], &second),
                        (r[6], &third),
                        (r[7], &scaled),
                    ] {
                        exact(&after[range.start..range.end()], expected);
                    }
                });
            }
        }
    });
}

#[test]
#[ignore = "explicit FP32 residual/scale range, scalar and command preflight rejection"]
fn invalid_bindings_and_scalars_reject_before_encoding() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let p = Pipelines::new(&gpu).unwrap();
        let dir = raw_dir();
        let mut arena = Buffer::<u32>::new(&gpu, 100).unwrap();
        arena.with_mut(|s| {
            s.fill(SENTINEL);
            s[7..15].fill(1_f32.to_bits());
            s[80] = 0;
        });
        let before = arena.with_slice(|s| s.to_vec());
        let range = |o, n| BufferRange::new(&arena, o, n).unwrap();
        let command = gpu.command().unwrap();
        for n in [0, 3, 5] {
            assert!(
                p.encode_add(
                    &command,
                    range(7, n),
                    range(11, 4),
                    range(30, 4),
                    range(80, 1)
                )
                .is_err()
            );
            assert!(
                p.encode_scale(&command, range(7, n), range(30, 4), range(80, 1), 1.)
                    .is_err()
            );
        }
        for scalar in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                p.encode_scale(&command, range(7, 4), range(30, 4), range(80, 1), scalar)
                    .is_err()
            );
        }
        for (a, b, out, status) in [
            (range(7, 4), range(11, 3), range(30, 4), range(80, 1)),
            (range(7, 4), range(11, 4), range(30, 3), range(80, 1)),
            (range(7, 4), range(11, 4), range(30, 4), range(80, 0)),
            (range(7, 4), range(11, 4), range(8, 4), range(80, 1)),
            (range(7, 4), range(11, 4), range(30, 4), range(31, 1)),
            (range(7, 4), range(11, 4), range(30, 4), range(7, 1)),
        ] {
            assert!(p.encode_add(&command, a, b, out, status).is_err());
        }
        for (input, out, status) in [
            (range(7, 4), range(30, 3), range(80, 1)),
            (range(7, 4), range(30, 4), range(80, 0)),
            (range(7, 4), range(8, 4), range(80, 1)),
            (range(7, 4), range(30, 4), range(31, 1)),
            (range(7, 4), range(30, 4), range(7, 1)),
        ] {
            assert!(p.encode_scale(&command, input, out, status, 1.).is_err());
        }
        let unretained = gpu.queue.commandBufferWithUnretainedReferences().unwrap();
        assert!(
            p.encode_add(
                &unretained,
                range(7, 4),
                range(11, 4),
                range(30, 4),
                range(80, 1)
            )
            .is_err()
        );
        assert!(
            p.encode_scale(&unretained, range(7, 4), range(30, 4), range(80, 1), 1.)
                .is_err()
        );
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        arena.with_slice(|after| {
            report(
                &dir,
                "preflight-rejections",
                "rejected",
                None,
                &[],
                (&before, after),
                &[],
            );
            assert_eq!(before, after);
        });
        assert!(
            p.encode_add(
                &command,
                range(7, 4),
                range(11, 4),
                range(30, 4),
                range(80, 1)
            )
            .is_err()
        );
        assert!(
            p.encode_scale(&command, range(7, 4), range(30, 4), range(80, 1), 1.)
                .is_err()
        );
        // Same range bound twice is legal because both operands are read-only.
        let valid = gpu.command().unwrap();
        p.encode_add(&valid, range(7, 4), range(7, 4), range(30, 4), range(80, 1))
            .unwrap();
        valid.commit();
        valid.waitUntilCompleted();
        assert_eq!(valid.status(), MTLCommandBufferStatus::Completed);
        arena.with_slice(|after| {
            report(
                &dir,
                "read-only-alias",
                "add-read-alias",
                None,
                &[],
                (&before, after),
                &[0],
            );
            assert_eq!(after[80], 0);
            assert_eq!(&after[30..34], &[2_f32.to_bits(); 4]);
            for i in 0..after.len() {
                if !(30..34).contains(&i) {
                    assert_eq!(before[i], after[i]);
                }
            }
        });
    });
}
