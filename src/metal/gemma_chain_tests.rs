//! Explicit resident composition qualification; the ordinary graph is unchanged.
use super::{
    gemma_attention as attention, gemma_rms as rms, gemma_rotary as rotary,
    memory::{Buffer, BufferRange, Gpu},
};
use crate::cpu::{self, AttentionKvSegment, ProportionalRotaryShape, SpanAttentionShape};
use objc2::rc::autoreleasepool;
use objc2_metal::*;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

const GUARD: usize = 7;
const SENTINEL: u32 = 0x7fc1_2345;

#[derive(Clone, Copy)]
struct Region {
    offset: usize,
    len: usize,
}
impl Region {
    fn range(self, b: &Buffer<u32>) -> BufferRange<'_> {
        BufferRange::new(b, self.offset, self.len).unwrap()
    }
    fn end(self) -> usize {
        self.offset + self.len
    }
}
fn check(actual: &[u32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&word, &want)) in actual.iter().zip(expected).enumerate() {
        let got = f32::from_bits(word);
        assert!(got.is_finite());
        assert!(
            (f64::from(got) - f64::from(want)).abs() <= 1e-5 + 1e-4 * f64::from(want).abs(),
            "index {i}: {got} != {want}"
        );
    }
}
#[derive(Serialize)]
struct ChainRaw<'a> {
    name: &'a str,
    device: &'a str,
    queries: usize,
    heads: usize,
    dim: usize,
    rotated_dims: usize,
    offset: usize,
    window: Option<usize>,
    regions: [[usize; 2]; 10],
    before_words: RawFile,
    after_words: RawFile,
    commands: usize,
    waits: usize,
    intermediate_cpu_staging: bool,
    graph_qualified: bool,
}
#[derive(Serialize)]
struct RawWords {
    before_words: RawFile,
    after_words: RawFile,
}
#[derive(Serialize)]
struct RawFile {
    path: String,
    words: usize,
    bytes: usize,
    sha256: String,
}
fn raw_directory() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let directory = if let Some(path) = std::env::var_os("TOKLET_CHAIN_RAW_DIR") {
        let path = std::path::PathBuf::from(path);
        std::fs::create_dir_all(&path).unwrap();
        path
    } else {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "toklet-gemma-chain-{}-{nanos}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        path
    };
    directory.canonicalize().unwrap()
}
fn capture_words(directory: &std::path::Path, case: &str, phase: &str, words: &[u32]) -> RawFile {
    use std::io::{BufWriter, Write};
    let path = directory.join(format!("{case}-{phase}.u32le"));
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    let mut writer = BufWriter::with_capacity(16 * 1024, file);
    let mut digest = Sha256::new();
    let mut block = [0_u8; 4096];
    for chunk in words.chunks(block.len() / 4) {
        for (word, bytes) in chunk.iter().zip(block.chunks_exact_mut(4)) {
            bytes.copy_from_slice(&word.to_le_bytes());
        }
        let active = &block[..chunk.len() * 4];
        writer.write_all(active).unwrap();
        digest.update(active);
    }
    writer.flush().unwrap();
    let bytes = words.len().checked_mul(4).unwrap();
    assert_eq!(writer.get_ref().metadata().unwrap().len(), bytes as u64);
    RawFile {
        path: path.to_str().unwrap().to_owned(),
        words: words.len(),
        bytes,
        sha256: format!("{:x}", digest.finalize()),
    }
}
fn report(label: &str, value: &impl Serialize) {
    use std::io::{BufWriter, Write};
    let stdout = std::io::stdout();
    let mut writer = BufWriter::with_capacity(16 * 1024, stdout.lock());
    write!(writer, "{label} ").unwrap();
    serde_json::to_writer(&mut writer, value).unwrap();
    writeln!(writer).unwrap();
    writer.flush().unwrap();
}

#[test]
#[ignore = "explicit Apple GPU shared RMS/rotary/attention single-command chain"]
fn resident_chain_frozen_small_and_32_query_actual_geometry() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let rms = rms::Pipelines::new(&gpu).unwrap();
        let rotary = rotary::Pipelines::new(&gpu).unwrap();
        let attention = attention::Pipelines::new(&gpu).unwrap();
        let raw_directory = raw_directory();
        let raw = include_bytes!("../../tests/fixtures/gemma-metal-rms-v1/math.json");
        assert_eq!(
            format!("{:x}", Sha256::digest(raw)),
            "935e1ecac3d8418cd4f668b5200e3e58730c0798391561bebe1b945391e859bb"
        );
        let fixture: Value = serde_json::from_slice(raw).unwrap();
        let small = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["name"] == "hand-direct-scales")
            .unwrap();
        for (name, queries, heads, dim, rotated, offset) in [
            ("frozen-small", 2, 1, 4, 2, 17),
            ("producer32-global", 32, 8, 512, 128, 131040),
            ("producer32-local", 32, 8, 256, 256, 497),
        ] {
            let values = queries * heads * dim;
            let keys = queries;
            let score_count = heads * queries * keys;
            let input: Vec<f32> = if dim == 4 {
                small["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect()
            } else {
                (0..values)
                    .map(|i| ((i * 13 % 101) as f32 - 50.) / 128.)
                    .collect()
            };
            let weights: Vec<f32> = if dim == 4 {
                small["weights"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect()
            } else {
                (0..dim).map(|i| 0.5 + (i % 17) as f32 / 32.).collect()
            };
            let kv: Vec<f32> = (0..keys * dim)
                .map(|i| ((i * 7 % 41) as f32 - 20.) / 256.)
                .collect();
            let rshape = ProportionalRotaryShape {
                tokens: queries,
                heads,
                head_dim: dim,
                rotated_dims: rotated,
                offset,
                base: 10000.,
                factor: 1.,
            };
            let ashape = SpanAttentionShape {
                queries,
                keys,
                query_heads: heads,
                kv_heads: 1,
                head_dim: dim,
                query_start: offset,
                key_start: offset,
                window: (dim == 256).then_some(16),
                scale: 1.,
            };
            let mut reference_rms = vec![0.; values];
            cpu::rmsnorm(
                &input,
                &weights,
                &mut reference_rms,
                queries * heads,
                dim,
                1e-6,
            )
            .unwrap();
            if dim == 4 {
                let frozen: Vec<_> = small["rounded_fp32"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect();
                assert_eq!(reference_rms, frozen);
            }
            let mut reference_rotary = vec![0.; values];
            cpu::proportional_rotary(&reference_rms, &mut reference_rotary, &rshape).unwrap();
            let empty = AttentionKvSegment {
                k: &[],
                v: &[],
                keys: 0,
                head_stride: 0,
                key_start: 0,
            };
            let spans = [
                AttentionKvSegment {
                    k: &kv,
                    v: &kv,
                    keys,
                    head_stride: keys * dim,
                    key_start: offset,
                },
                empty,
                empty,
            ];
            let mut reference_context = vec![0.; values];
            let mut reference_scores = vec![0.; score_count];
            let mut reference_probabilities = vec![0.; score_count];
            cpu::span_attention(
                &reference_rotary,
                &spans,
                &mut reference_context,
                &mut reference_scores,
                &mut reference_probabilities,
                &ashape,
            )
            .unwrap();

            // One shared allocation with nonzero offsets and guarded gaps. K/V
            // deliberately alias read-only; all outputs/scratch/status are disjoint.
            let mut cursor = GUARD;
            let mut region = |len| {
                let r = Region {
                    offset: cursor,
                    len,
                };
                cursor = r.end() + GUARD;
                r
            };
            let input_r = region(values);
            let weights_r = region(dim);
            let coefficients_r = region(queries * rotated);
            let kv_r = region(kv.len());
            let rms_r = region(values);
            let rotary_r = region(values);
            let scores_r = region(score_count);
            let probabilities_r = region(score_count);
            let context_r = region(values);
            let status_r = region(1);
            let mut arena = Buffer::<u32>::new(&gpu, cursor).unwrap();
            arena.with_mut(|words| {
                words.fill(SENTINEL);
                for (r, data) in [(input_r, &input), (weights_r, &weights), (kv_r, &kv)] {
                    for (out, value) in words[r.offset..r.end()].iter_mut().zip(data) {
                        *out = value.to_bits();
                    }
                }
                rotary::prepare_coefficients(
                    &rshape,
                    &mut words[coefficients_r.offset..coefficients_r.end()],
                )
                .unwrap();
                words[status_r.offset] = 0;
            });
            let before = arena.with_slice(|w| w.to_vec());
            let command = gpu.command().unwrap();
            rms.encode(
                &command,
                &rms::DeviceBuffers {
                    input: input_r.range(&arena),
                    weights: weights_r.range(&arena),
                    output: rms_r.range(&arena),
                    status: status_r.range(&arena),
                },
                queries * heads,
                dim,
                1e-6,
                true,
            )
            .unwrap();
            rotary
                .encode(
                    &command,
                    &rotary::DeviceBuffers {
                        input: rms_r.range(&arena),
                        coefficients: coefficients_r.range(&arena),
                        output: rotary_r.range(&arena),
                        status: status_r.range(&arena),
                    },
                    &rshape,
                )
                .unwrap();
            let empty_device = || attention::DeviceSegment {
                k: BufferRange::new(&arena, 0, 0).unwrap(),
                v: BufferRange::new(&arena, 0, 0).unwrap(),
                keys: 0,
                head_stride: 0,
                key_start: 0,
            };
            attention
                .encode(
                    &command,
                    &attention::DeviceBuffers {
                        q: rotary_r.range(&arena),
                        segments: [
                            attention::DeviceSegment {
                                k: kv_r.range(&arena),
                                v: kv_r.range(&arena),
                                keys,
                                head_stride: keys * dim,
                                key_start: offset,
                            },
                            empty_device(),
                            empty_device(),
                        ],
                        scores: scores_r.range(&arena),
                        probabilities: probabilities_r.range(&arena),
                        output: context_r.range(&arena),
                        status: status_r.range(&arena),
                    },
                    &ashape,
                )
                .unwrap();
            // Exactly one submission/wait after every dependent encoder. No CPU
            // staging/readback of RMS or rotary output occurs in between.
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            // Stream exact binary words from the completed allocation and
            // pre-submit snapshot, then publish small authenticated descriptors.
            // Capture all words before numerical/status/guard gates.
            arena.with_slice(|after| {
                // Preserve bounded raw words/layouts before numerical/status/guard gates.
                let regions = [
                    input_r,
                    weights_r,
                    coefficients_r,
                    kv_r,
                    rms_r,
                    rotary_r,
                    scores_r,
                    probabilities_r,
                    context_r,
                    status_r,
                ]
                .map(|r| [r.offset, r.len]);
                let device = gpu.device.name().to_string();
                report(
                    "GEMMA_RESIDENT_CHAIN_RAW",
                    &ChainRaw {
                        name,
                        device: &device,
                        queries,
                        heads,
                        dim,
                        rotated_dims: rotated,
                        offset,
                        window: ashape.window,
                        regions,
                        before_words: capture_words(&raw_directory, name, "before", &before),
                        after_words: capture_words(&raw_directory, name, "after", after),
                        commands: 1,
                        waits: 1,
                        intermediate_cpu_staging: false,
                        graph_qualified: false,
                    },
                );
                assert!(command.error().is_none());
                assert_eq!(after[status_r.offset], 0);
                for (i, (&a, &b)) in after.iter().zip(&before).enumerate() {
                    if ![
                        rms_r,
                        rotary_r,
                        scores_r,
                        probabilities_r,
                        context_r,
                        status_r,
                    ]
                    .iter()
                    .any(|r| (r.offset..r.end()).contains(&i))
                    {
                        assert_eq!(a, b, "{name}: immutable/guard word {i}");
                    }
                }
                check(&after[rms_r.offset..rms_r.end()], &reference_rms);
                check(&after[rotary_r.offset..rotary_r.end()], &reference_rotary);
                check(&after[scores_r.offset..scores_r.end()], &reference_scores);
                check(
                    &after[probabilities_r.offset..probabilities_r.end()],
                    &reference_probabilities,
                );
                check(
                    &after[context_r.offset..context_r.end()],
                    &reference_context,
                );
                for (i, &expected) in reference_probabilities.iter().enumerate() {
                    if expected == 0. {
                        assert_eq!(after[probabilities_r.offset + i], 0);
                    }
                }
            });
        }
    });
}

#[test]
#[ignore = "explicit Apple GPU borrowed unary shape/range/command admission"]
fn borrowed_unary_rejections_precede_encoding() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let rms = rms::Pipelines::new(&gpu).unwrap();
        let rotary = rotary::Pipelines::new(&gpu).unwrap();
        let raw_directory = raw_directory();
        let mut arena = Buffer::<u32>::new(&gpu, 40).unwrap();
        arena.with_mut(|s| {
            s.fill(1_f32.to_bits());
            s[32] = 0;
        });
        let before = arena.with_slice(|s| s.to_vec());
        let range = |offset, len| BufferRange::new(&arena, offset, len).unwrap();
        for (offset, len) in [(usize::MAX, 1), (1, usize::MAX), (40, 0), (39, 2)] {
            assert!(BufferRange::new(&arena, offset, len).is_err());
        }
        let rshape = ProportionalRotaryShape {
            tokens: 1,
            heads: 1,
            head_dim: 4,
            rotated_dims: 4,
            offset: 0,
            base: 10000.,
            factor: 1.,
        };
        let rb = || rms::DeviceBuffers {
            input: range(4, 4),
            weights: range(4, 4),
            output: range(16, 4),
            status: range(32, 1),
        };
        let tb = || rotary::DeviceBuffers {
            input: range(4, 4),
            coefficients: range(4, 4),
            output: range(24, 4),
            status: range(32, 1),
        };
        let command = gpu.command().unwrap();
        for row in [0, 257, usize::MAX] {
            assert!(rms.encode(&command, &rb(), row, 4, 1e-6, true).is_err());
        }
        for width in [0, 1537, usize::MAX] {
            assert!(rms.encode(&command, &rb(), 1, width, 1e-6, true).is_err());
        }
        for epsilon in [f32::NAN, 0., f32::from_bits(1e-6_f32.to_bits() + 1)] {
            assert!(rms.encode(&command, &rb(), 1, 4, epsilon, true).is_err());
        }
        for field in 0..4 {
            let mut b = rb();
            match field {
                0 => b.input = range(4, 3),
                1 => b.weights = range(4, 3),
                2 => b.output = range(16, 3),
                _ => b.status = range(32, 0),
            }
            assert!(rms.encode(&command, &b, 1, 4, 1e-6, true).is_err());
            let mut b = tb();
            match field {
                0 => b.input = range(4, 3),
                1 => b.coefficients = range(4, 3),
                2 => b.output = range(24, 3),
                _ => b.status = range(32, 0),
            }
            assert!(rotary.encode(&command, &b, &rshape).is_err());
        }
        for field in 0..3 {
            let mut b = rb();
            match field {
                0 => b.output = range(5, 4),
                1 => b.status = range(16, 1),
                _ => b.status = range(4, 1),
            }
            assert!(rms.encode(&command, &b, 1, 4, 1e-6, true).is_err());
            let mut b = tb();
            match field {
                0 => b.output = range(5, 4),
                1 => b.status = range(24, 1),
                _ => b.status = range(4, 1),
            }
            assert!(rotary.encode(&command, &b, &rshape).is_err());
        }
        for tokens in [0, 33, usize::MAX] {
            let mut s = rshape;
            s.tokens = tokens;
            assert!(rotary.encode(&command, &tb(), &s).is_err());
        }
        let unretained = gpu.queue.commandBufferWithUnretainedReferences().unwrap();
        assert!(!unretained.retainedReferences());
        assert!(rms.encode(&unretained, &rb(), 1, 4, 1e-6, true).is_err());
        assert!(rotary.encode(&unretained, &tb(), &rshape).is_err());
        assert_eq!(command.status(), MTLCommandBufferStatus::NotEnqueued);
        // Only rejected requests touched this command. Completing it must leave
        // every word untouched, including every proposed output and status.
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let after = arena.with_slice(|s| s.to_vec());
        report(
            "GEMMA_UNARY_REJECTION_RAW",
            &RawWords {
                before_words: capture_words(&raw_directory, "unary-rejection", "before", &before),
                after_words: capture_words(&raw_directory, "unary-rejection", "after", &after),
            },
        );
        assert_eq!(before, after);
        assert!(rms.encode(&command, &rb(), 1, 4, 1e-6, true).is_err());
        assert!(rotary.encode(&command, &tb(), &rshape).is_err());
        // Valid read-only input/coefficient aliases are admitted. Both encoders
        // share status and retain the same command/resource, then finish once.
        let valid = gpu.command().unwrap();
        rms.encode(&valid, &rb(), 1, 4, 1e-6, true).unwrap();
        rotary.encode(&valid, &tb(), &rshape).unwrap();
        valid.commit();
        valid.waitUntilCompleted();
        assert_eq!(valid.status(), MTLCommandBufferStatus::Completed);
        assert!(valid.error().is_none());
        let after = arena.with_slice(|s| s.to_vec());
        report(
            "GEMMA_UNARY_READ_ALIAS_RAW",
            &RawWords {
                before_words: capture_words(&raw_directory, "unary-read-alias", "before", &before),
                after_words: capture_words(&raw_directory, "unary-read-alias", "after", &after),
            },
        );
        assert_eq!(after[32], 0);
        assert_eq!(&after[4..8], &before[4..8]);
        for i in 0..40 {
            if !(16..20).contains(&i) && !(24..28).contains(&i) {
                assert_eq!(after[i], before[i]);
            }
        }
    });
}
