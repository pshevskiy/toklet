use super::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Seek, Write},
    path::{Path, PathBuf},
};
const PROTOCOL_SHA: &str = "d39ff3bcd413a987b6425723e2815e944de66406f829b431e93918f027a03d5f";

#[test]
fn strided_suffix_preserves_original_bf16_body_except_name_and_store() {
    assert_eq!(
        format!("{:x}", Sha256::digest(ORIGINAL)),
        "27de2af5e64cdc2ac086d93f64cbf9667c435e332fedb97725377fd1a8ef3efd"
    );
    let old = &ORIGINAL[ORIGINAL.find("kernel void gemma_bf16_product(").unwrap()..];
    let new = &SUFFIX[SUFFIX
        .find("kernel void gemma_bf16_product_strided(")
        .unwrap()..];
    assert_eq!(
        old.replace(
            "kernel void gemma_bf16_product(",
            "kernel void gemma_bf16_product_strided("
        )
        .replace(
            "output[token * shape[2] + row]",
            "output[token * shape[3] + shape[4] + row]"
        ),
        new
    );
    assert!(ORIGINAL.contains("#pragma clang fp contract(off)"));
}

#[test]
fn full_output_geometry_covers_every_row_and_checked_abi() {
    for rows in [1, 8191, 8192, 8193, 8960, 12288, 262144] {
        for columns in [64, 1536, 12288] {
            for tokens in [1, 2, 4, 32] {
                let g = Geometry::new(rows, columns, tokens).unwrap();
                assert_eq!(g.input, tokens * columns);
                assert_eq!(g.output, tokens * rows);
                let mut next = 0;
                for index in 0..g.tiles {
                    let t = g.tile(index).unwrap();
                    assert_eq!(
                        t.words,
                        [
                            tokens as u32,
                            columns as u32,
                            (rows - next).min(8192) as u32,
                            rows as u32,
                            next as u32
                        ]
                    );
                    assert_eq!(t.groups, tokens * t.words[2] as usize);
                    assert_eq!(t.coefficients, t.words[2] as usize * columns / 2);
                    for token in 0..tokens {
                        let first = token * rows + next;
                        let last = first + t.words[2] as usize - 1;
                        assert!(first <= last && last < g.output);
                    }
                    next += t.words[2] as usize;
                }
                assert_eq!(next, rows);
                assert!(g.tile(g.tiles).is_err());
            }
        }
    }
    for (rows, columns, tokens) in [
        (0, 64, 1),
        (262145, 64, 1),
        (usize::MAX, 64, 1),
        (1, 0, 1),
        (1, 63, 1),
        (1, 65, 1),
        (1, 12352, 1),
        (1, usize::MAX, 1),
        (1, 64, 0),
        (1, 64, 33),
        (1, 64, usize::MAX),
    ] {
        assert!(Geometry::new(rows, columns, tokens).is_err());
    }
    assert!(checked_product(usize::MAX, 4).is_err());
    assert!(validate_width(32, 32, 32).is_ok());
    for (original, got, max) in [
        (32, 0, 1024),
        (32, 16, 1024),
        (32, 64, 1024),
        (32, 32, 16),
        (65536, 65536, 65536),
    ] {
        assert!(validate_width(original, got, max).is_err());
    }
}

fn sha(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}
fn hash_words(words: &[u32]) -> String {
    let mut digest = Sha256::new();
    let mut bytes = [0; 4096];
    for chunk in words.chunks(1024) {
        for (word, out) in chunk.iter().zip(bytes.chunks_exact_mut(4)) {
            out.copy_from_slice(&word.to_le_bytes());
        }
        digest.update(&bytes[..chunk.len() * 4]);
    }
    format!("{:x}", digest.finalize())
}
fn bounded(path: &Path, cap: usize) -> Vec<u8> {
    assert!(path.metadata().unwrap().len() <= cap as u64);
    std::fs::read(path).unwrap()
}
fn read_fixture() -> (Value, Vec<u8>, Vec<u8>) {
    let root = PathBuf::from(
        std::env::var_os("TOKLET_BF16_STRIDED_FIXTURE").expect("explicit frozen fixture directory"),
    );
    let raw = bounded(&root.join("protocol.json"), 256 * 1024);
    assert_eq!(sha(&raw), PROTOCOL_SHA);
    let p: Value = serde_json::from_slice(&raw).unwrap();
    assert_eq!(p["case_count"], 29);
    assert_eq!(p["original_case_count"], 19);
    let gate = p["numerical_gate"].as_object().unwrap();
    assert_eq!(gate.len(), 3);
    assert_eq!(
        gate["absolute"].as_f64().unwrap().to_bits(),
        1e-5_f64.to_bits()
    );
    assert_eq!(
        gate["relative"].as_f64().unwrap().to_bits(),
        1e-4_f64.to_bits()
    );
    assert_eq!(gate["relaxed"], false);
    let read = |name: &str| {
        let desc = &p[name];
        let bytes = bounded(&root.join(format!("{name}.bin")), 16 * 1024 * 1024);
        assert_eq!(bytes.len() as u64, desc["bytes"].as_u64().unwrap());
        assert_eq!(sha(&bytes), desc["sha256"]);
        bytes
    };
    let input = read("inputs");
    let expected = read("expected");
    (p, input, expected)
}
fn array<'a>(raw: &'a [u8], desc: &Value) -> &'a [u8] {
    let start = desc["offset"].as_u64().unwrap() as usize;
    let len = desc["bytes"].as_u64().unwrap() as usize;
    let bytes = &raw[start..start + len];
    assert_eq!(sha(bytes), desc["sha256"]);
    bytes
}
fn structured_weights(row: usize) -> [f32; 3] {
    let [a, b] = match row % 3 {
        0 => [1., -1.],
        1 => [0.5, -0.75],
        _ => [1.0078125, -1.],
    };
    let mut c = if row.is_multiple_of(3) {
        ((row / 8192) % 7 + 1) as f32 / 16.
    } else {
        (row % 17) as f32 / 32. - 0.25
    };
    if row.is_multiple_of(11) && c == 0. {
        c = -0.;
    }
    [a, b, c]
}
fn coefficients(gpu: &Gpu, case: &Value) -> Vec<Buffer<u32>> {
    let rows = case["rows"].as_u64().unwrap() as usize;
    let columns = case["columns"].as_u64().unwrap() as usize;
    let mut maximum = 0_f32;
    let mut tiles = Vec::with_capacity(rows.div_ceil(8192));
    for first in (0..rows).step_by(8192) {
        let count = (rows - first).min(8192);
        let mut tile = Buffer::<u32>::new(gpu, count * columns / 2 + GUARD).unwrap();
        tile.with_mut(|s| {
            s.fill(SENTINEL);
            for (local, out) in s[..count * columns / 2]
                .chunks_exact_mut(columns / 2)
                .enumerate()
            {
                let row = first + local;
                if case["mode"] == "structured" {
                    out.fill(0);
                    for (col, value) in [0, columns / 2, columns - 1]
                        .into_iter()
                        .zip(structured_weights(row))
                    {
                        assert_eq!(value.to_bits() & 0xffff, 0);
                        maximum = maximum.max(value.abs());
                        out[col / 2] |= (value.to_bits() >> 16) << (16 * (col % 2));
                    }
                } else {
                    let values = case["constant_weights"].as_array().unwrap();
                    let value = values[row % values.len()].as_f64().unwrap() as f32;
                    assert_eq!(value.to_bits() & 0xffff, 0);
                    maximum = maximum.max(value.abs());
                    let half = value.to_bits() >> 16;
                    out.fill(half | half << 16);
                }
            }
        });
        let desc = &case["tiles"][tiles.len()];
        assert_eq!(tile.with_slice(hash_words), desc["guarded_sha256"]);
        tiles.push(tile);
    }
    assert_eq!(f64::from(maximum), case["maximum"].as_f64().unwrap());
    tiles
}
#[derive(Clone, Copy)]
struct Region {
    start: usize,
    len: usize,
}
impl Region {
    fn end(self) -> usize {
        self.start + self.len
    }
    fn range(self, b: &Buffer<u32>) -> BufferRange<'_> {
        BufferRange::new(b, self.start, self.len).unwrap()
    }
}
fn regions(lengths: &[usize]) -> (Vec<Region>, usize) {
    let mut cursor = GUARD;
    let regions = lengths
        .iter()
        .map(|&len| {
            let r = Region { start: cursor, len };
            cursor = r.end() + GUARD;
            r
        })
        .collect();
    (regions, cursor)
}
struct Capture {
    file: File,
    root: PathBuf,
    cases: Vec<Value>,
}
impl Capture {
    fn new() -> Self {
        let root = PathBuf::from(
            std::env::var_os("TOKLET_BF16_STRIDED_CAPTURE")
                .expect("new explicit capture directory"),
        );
        std::fs::create_dir(&root).unwrap();
        let file = File::create_new(root.join("arrays.bin")).unwrap();
        Self {
            file,
            root,
            cases: Vec::new(),
        }
    }
    fn record(&mut self, words: &[u32]) -> Value {
        let offset = self.file.stream_position().unwrap();
        assert!(offset + words.len() as u64 * 4 <= 16 * 1024 * 1024);
        let mut bytes = [0; 4096];
        for chunk in words.chunks(1024) {
            for (word, out) in chunk.iter().zip(bytes.chunks_exact_mut(4)) {
                out.copy_from_slice(&word.to_le_bytes());
            }
            self.file.write_all(&bytes[..chunk.len() * 4]).unwrap();
        }
        json!({"offset":offset,"bytes":words.len()*4,"sha256":hash_words(words)})
    }
    fn finish(mut self, format: &str) {
        self.file.flush().unwrap();
        drop(self.file);
        let raw = bounded(&self.root.join("arrays.bin"), 16 * 1024 * 1024);
        let report = json!({"format":format,"passed":true,
            "protocol_sha256":PROTOCOL_SHA,"cases":self.cases,
            "raw":{"path":self.root.join("arrays.bin"),"bytes":raw.len(),"sha256":sha(&raw)},
            "precision":"original compensated BF16 coefficients / FP32 outputs; Metal2.3 Safe Precise contract-off",
            "normal_library_selected":false,"whole_model_qualified":false,"performance_qualified":false});
        let mut out = File::create_new(self.root.join("result.json")).unwrap();
        serde_json::to_writer_pretty(&mut out, &report).unwrap();
        out.write_all(b"\n").unwrap();
    }
}
fn numeric(got: u32, expected: u32) {
    let actual = f64::from(f32::from_bits(got));
    let expected = f64::from(f32::from_bits(expected));
    assert!(actual.is_finite() && expected.is_finite());
    assert!((actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs());
}

#[test]
#[ignore = "explicit bounded BF16 strided all-tiles original/scalar qualification"]
fn complete_tiled_outputs_match_original_per_token_and_frozen_references() {
    autoreleasepool(|_| {
        use super::super::bf16_encode::{AdmissionDecision, AdmissionPipeline, admission_decision};
        let (p, inputs, expected) = read_fixture();
        let context = Context::with_projection_pipeline(CONTEXT_ALLOWANCE_BYTES, 16).unwrap();
        let gpu = &context.gpu;
        let pipeline =
            StridedBf16Pipelines::new(&context, StridedBf16Pipelines::policy_bytes()).unwrap();
        assert_eq!(pipeline.width, context.width);
        let admission = AdmissionPipeline::new(gpu).unwrap();
        let mut capture = Capture::new();
        for case in p["cases"].as_array().unwrap() {
            autoreleasepool(|_| {
                let rows = case["rows"].as_u64().unwrap() as usize;
                let columns = case["columns"].as_u64().unwrap() as usize;
                let tokens = case["tokens"].as_u64().unwrap() as usize;
                let tiles = coefficients(gpu, case);
                let matrix = Bf16MatrixView::new(
                    &context,
                    &tiles,
                    rows,
                    columns,
                    case["maximum"].as_f64().unwrap(),
                )
                .unwrap();
                let controls: Vec<_> = tiles
                    .iter()
                    .enumerate()
                    .map(|(i, b)| {
                        Bf16Tile::new(
                            gpu,
                            (rows - i * 8192).min(8192),
                            columns,
                            b,
                            case["maximum"].as_f64().unwrap(),
                        )
                        .unwrap()
                    })
                    .collect();
                let (r, count) =
                    regions(&[tokens * columns, tokens * rows, tokens * rows, 1, 1, 1]);
                let mut arena = Buffer::<u32>::new(gpu, count).unwrap();
                arena.with_mut(|s| {
                    s.fill(SENTINEL);
                    for (out, raw) in s[r[0].start..r[0].end()]
                        .iter_mut()
                        .zip(array(&inputs, &case["input"]).chunks_exact(4))
                    {
                        *out = u32::from_le_bytes(raw.try_into().unwrap());
                    }
                    for region in &r[3..] {
                        s[region.start] = 0;
                    }
                });
                let command = gpu.command().unwrap();
                admission
                    .encode_with_l1(
                        &command,
                        matrix.admission_tile(),
                        r[0].range(&arena),
                        r[3].range(&arena),
                        tokens,
                    )
                    .unwrap();
                pipeline
                    .encode(
                        &command,
                        matrix,
                        DeviceBuffers {
                            input: r[0].range(&arena),
                            output: r[1].range(&arena),
                            status: r[4].range(&arena),
                        },
                        tokens,
                    )
                    .unwrap();
                // Independent layout control: unchanged original shader with T1,
                // correctly offset input/output for every actual token and tile.
                for token in 0..tokens {
                    for (index, &tile) in controls.iter().enumerate() {
                        let tile_rows = (rows - index * 8192).min(8192);
                        super::super::bf16_encode::encode_bf16(
                            &context,
                            &command,
                            tile,
                            BufferRange::new(&arena, r[0].start + token * columns, columns)
                                .unwrap(),
                            BufferRange::new(
                                &arena,
                                r[2].start + token * rows + index * 8192,
                                tile_rows,
                            )
                            .unwrap(),
                            r[5].range(&arena),
                            1,
                        )
                        .unwrap();
                    }
                }
                command.commit();
                command.waitUntilCompleted();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                assert!(command.error().is_none());
                let expected = array(&expected, &case["expected"]);
                let after = arena.with_slice(|s| s.to_vec());
                assert_eq!(
                    &after[r[1].start..r[1].end()],
                    &after[r[2].start..r[2].end()]
                );
                assert_eq!(after[r[4].start], after[r[5].start]);
                let host_admitted = case["original_host_admitted"].as_bool().unwrap();
                let decision = admission_decision(after[r[3].start]).unwrap();
                if case["mode"] == "structured" {
                    assert_eq!(decision, AdmissionDecision::Admitted);
                }
                if !host_admitted {
                    assert_eq!(decision, AdmissionDecision::ReferenceReplay);
                }
                if host_admitted {
                    assert_eq!(after[r[4].start], 0);
                    for (&got, word) in after[r[1].start..r[1].end()]
                        .iter()
                        .zip(expected.chunks_exact(4))
                    {
                        numeric(got, u32::from_le_bytes(word.try_into().unwrap()));
                    }
                }
                for (index, &word) in after.iter().enumerate() {
                    if !r
                        .iter()
                        .any(|region| (region.start..region.end()).contains(&index))
                    {
                        assert_eq!(word, SENTINEL);
                    }
                }
                assert_eq!(
                    hash_words(&after[r[0].start..r[0].end()]),
                    case["input"]["sha256"]
                );
                let hashes: Vec<_> = tiles
                    .iter()
                    .enumerate()
                    .map(|(i, t)| {
                        let digest = t.with_slice(hash_words);
                        assert_eq!(digest, case["tiles"][i]["guarded_sha256"]);
                        digest
                    })
                    .collect();
                let arena_desc = capture.record(&after);
                capture.cases.push(json!({"name":case["name"],"rows":rows,"columns":columns,"tokens":tokens,
                "arena":arena_desc,"regions":r.iter().map(|x|json!([x.start,x.len])).collect::<Vec<_>>(),
                "tile_guarded_sha256":hashes,"tile_count":matrix.tile_count(),
                "max_input_bits":matrix.admission_tile().max_input_bits(),"l1_input_bits":matrix.admission_tile().l1_input_bits(),
                "maximum_bits":matrix.admission_tile().maximum_bits(),
                "admission_dispatches":1,"candidate_dispatches":matrix.tile_count(),
                "original_control_dispatches":tokens*matrix.tile_count(),"submitted_commands":1,"waits":1,
                "native_completed":true,"native_error":false,
                "candidate_ready":after[r[3].start]==0 && after[r[4].start]==0,
                "original_host_admitted":host_admitted}));
            });
        }
        capture.finish("toklet-bf16-strided-capture-v1");
    });
}

fn constant_tiles(gpu: &Gpu, rows: usize, columns: usize, value: f32) -> Vec<Buffer<u32>> {
    assert_eq!(value.to_bits() & 0xffff, 0);
    (0..rows)
        .step_by(8192)
        .map(|first| {
            let count = (rows - first).min(8192) * columns / 2;
            let mut b = Buffer::<u32>::new(gpu, count + GUARD).unwrap();
            b.with_mut(|s| {
                s.fill(SENTINEL);
                let half = value.to_bits() >> 16;
                s[..count].fill(half | half << 16);
            });
            b
        })
        .collect()
}

#[test]
#[ignore = "explicit complete preflight / late-encoding discard / sticky product status controls"]
fn aggregate_preflight_finishes_before_any_tile_encoder_and_failures_stay_private() {
    autoreleasepool(|_| {
        let context = Context::with_projection_pipeline(CONTEXT_ALLOWANCE_BYTES, 16).unwrap();
        let gpu = &context.gpu;
        assert!(matches!(
            StridedBf16Pipelines::new(&context, StridedBf16Pipelines::policy_bytes() - 1),
            Err(Error::Budget { .. })
        ));
        let pipeline =
            StridedBf16Pipelines::new(&context, StridedBf16Pipelines::policy_bytes()).unwrap();
        let tiles = constant_tiles(gpu, 16384, 64, 1.);
        let matrix = Bf16MatrixView::new(&context, &tiles, 16384, 64, 1.).unwrap();
        let (r, count) = regions(&[64, 16384, 1, 256]);
        let mut arena = Buffer::<u32>::new(gpu, count).unwrap();
        arena.with_mut(|s| {
            s.fill(SENTINEL);
            s[r[0].start..r[0].end()].fill(0.5_f32.to_bits());
            s[r[2].start] = 0;
            s[r[3].start..r[3].end()].fill(88_f32.to_bits());
        });
        let before = arena.with_slice(|s| s.to_vec());
        let initial_weights: Vec<_> = tiles.iter().map(|t| t.with_slice(hash_words)).collect();
        let buffers = DeviceBuffers {
            input: r[0].range(&arena),
            output: r[1].range(&arena),
            status: r[2].range(&arena),
        };
        let command = gpu.command().unwrap();
        let mut rejected = 1; // cold budget rejection above
        for tokens in [0, 33, usize::MAX] {
            assert!(pipeline.encode(&command, matrix, buffers, tokens).is_err());
            rejected += 1;
        }
        for bad in [
            DeviceBuffers {
                input: BufferRange {
                    len: 63,
                    ..buffers.input
                },
                ..buffers
            },
            DeviceBuffers {
                output: BufferRange {
                    len: 16383,
                    ..buffers.output
                },
                ..buffers
            },
            DeviceBuffers {
                status: BufferRange {
                    len: 0,
                    ..buffers.status
                },
                ..buffers
            },
            DeviceBuffers {
                input: BufferRange {
                    offset: usize::MAX,
                    ..buffers.input
                },
                ..buffers
            },
            DeviceBuffers {
                output: BufferRange {
                    offset: count - 10,
                    ..buffers.output
                },
                ..buffers
            },
            DeviceBuffers {
                output: BufferRange::new(&arena, r[0].start + 1, 16384).unwrap(),
                ..buffers
            },
            DeviceBuffers {
                status: BufferRange::new(&arena, r[1].start + 8193, 1).unwrap(),
                ..buffers
            },
            DeviceBuffers {
                status: BufferRange::new(&arena, r[0].start, 1).unwrap(),
                ..buffers
            },
            // Only the LAST tile aliases this full output. First-pass validation
            // must discover it before an earlier tile could write the arena.
            DeviceBuffers {
                output: BufferRange::new(&tiles[1], 17, 16384).unwrap(),
                ..buffers
            },
            DeviceBuffers {
                status: BufferRange::new(&tiles[1], 5, 1).unwrap(),
                ..buffers
            },
            DeviceBuffers {
                status: BufferRange::new(&tiles[1], tiles[1].len - GUARD, 1).unwrap(),
                ..buffers
            },
        ] {
            assert!(pipeline.encode(&command, matrix, bad, 1).is_err());
            rejected += 1;
        }
        for bad in [
            Bf16MatrixView {
                rows: usize::MAX,
                ..matrix
            },
            Bf16MatrixView {
                tiles: &tiles[..1],
                ..matrix
            },
        ] {
            assert!(pipeline.encode(&command, bad, buffers, 1).is_err());
            rejected += 1;
        }
        let mut short = constant_tiles(gpu, 8193, 64, 1.);
        short[1].len -= 1;
        assert!(Bf16MatrixView::new(&context, &short, 8193, 64, 1.).is_err());
        rejected += 1;
        // Deliberately bypass the cold constructor inside this private test.
        // A malformed last native allocation must also fail hot preflight.
        let malformed = Bf16MatrixView {
            tiles: &short,
            rows: 8193,
            columns: 64,
            admission: matrix.admission,
        };
        assert!(
            pipeline
                .encode(
                    &command,
                    malformed,
                    DeviceBuffers {
                        output: BufferRange::new(&arena, r[1].start, 8193).unwrap(),
                        ..buffers
                    },
                    1
                )
                .is_err()
        );
        rejected += 1;
        for bad_tiles in [&tiles[..1], &tiles[..]] {
            assert!(Bf16MatrixView::new(&context, bad_tiles, 8193, 64, 1.).is_err());
            rejected += 1;
        }
        for maximum in [
            f64::NAN,
            f64::INFINITY,
            -1.,
            f64::from(f32::from_bits(1)),
            1.00000001,
        ] {
            assert!(Bf16MatrixView::new(&context, &tiles, 16384, 64, maximum).is_err());
            rejected += 1;
        }
        let unretained = gpu.queue.commandBufferWithUnretainedReferences().unwrap();
        assert!(pipeline.encode(&unretained, matrix, buffers, 1).is_err());
        rejected += 1;
        let queued = gpu.command().unwrap();
        queued.enqueue();
        assert!(pipeline.encode(&queued, matrix, buffers, 1).is_err());
        rejected += 1;
        queued.commit();
        queued.waitUntilCompleted();
        assert_eq!(queued.status(), MTLCommandBufferStatus::Completed);
        let untracked = gpu
            .device
            .newBufferWithLength_options(
                64 * 4,
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::HazardTrackingModeUntracked,
            )
            .unwrap();
        assert!(
            pipeline
                .encode(
                    &command,
                    matrix,
                    DeviceBuffers {
                        input: BufferRange {
                            raw: &untracked,
                            offset: 0,
                            len: 64
                        },
                        ..buffers
                    },
                    1
                )
                .is_err()
        );
        rejected += 1;
        let private = gpu
            .device
            .newBufferWithLength_options(
                64 * 4,
                MTLResourceOptions::StorageModePrivate
                    | MTLResourceOptions::HazardTrackingModeTracked,
            )
            .unwrap();
        assert!(
            pipeline
                .encode(
                    &command,
                    matrix,
                    DeviceBuffers {
                        input: BufferRange {
                            raw: &private,
                            offset: 0,
                            len: 64
                        },
                        ..buffers
                    },
                    1
                )
                .is_err()
        );
        rejected += 1;
        assert_eq!(command.status(), MTLCommandBufferStatus::NotEnqueued);
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(command.error().is_none());
        let unchanged = arena.with_slice(|s| s.to_vec());
        assert_eq!(before, unchanged);
        for (t, h) in tiles.iter().zip(&initial_weights) {
            assert_eq!(t.with_slice(hash_words), *h);
        }
        assert!(pipeline.encode(&command, matrix, buffers, 1).is_err());
        rejected += 1;
        let mut capture = Capture::new();
        let initial = capture.record(&before);
        let after = capture.record(&unchanged);
        capture.cases.push(json!({"name":"all-preflight-rejections","rejections":rejected,"before":initial,"after":after,
            "regions":r.iter().map(|x|json!([x.start,x.len])).collect::<Vec<_>>(),"native_completed":true,"native_error":false,
            "foreign_device_control":"requires another physical Metal device; not fabricated on single-device host"}));
        let mut alias_output = Buffer::<u32>::new(gpu, 16384 + 1 + 3 * GUARD).unwrap();
        alias_output.with_mut(|s| {
            s.fill(SENTINEL);
            s[GUARD + 16384 + GUARD] = 0;
        });
        let command = gpu.command().unwrap();
        pipeline
            .encode(
                &command,
                matrix,
                DeviceBuffers {
                    input: BufferRange::new(&tiles[0], 31, 64).unwrap(),
                    output: BufferRange::new(&alias_output, GUARD, 16384).unwrap(),
                    status: BufferRange::new(&alias_output, GUARD + 16384 + GUARD, 1).unwrap(),
                },
                1,
            )
            .unwrap();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(command.error().is_none());
        let expected = (f32::from_bits(0x3f803f80) * 64.).to_bits();
        alias_output
            .with_slice(|s| assert!(s[GUARD..GUARD + 16384].iter().all(|&x| x == expected)));
        let raw = alias_output.with_slice(|s| capture.record(s));
        capture.cases.push(json!({"name":"readonly-weight-input-alias-is-legal","after":raw,"expected_output_bits":expected,
            "native_completed":true,"native_error":false}));
        // Late native-encoder absence is injected after one genuine encoded
        // tile. The sole transaction owner drains but never publishes partial
        // scratch. Synthetic committed-KV guard span is never bound writable.
        pipeline.fail_before_tile.set(Some(1));
        let command = gpu.command().unwrap();
        assert!(pipeline.encode(&command, matrix, buffers, 1).is_err());
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(command.error().is_none());
        let partial = arena.with_slice(|s| s.to_vec());
        assert!(
            partial[r[1].start..r[1].start + 8192]
                .iter()
                .all(|&v| v == 32_f32.to_bits())
        );
        assert!(
            partial[r[1].start + 8192..r[1].end()]
                .iter()
                .all(|&v| v == SENTINEL)
        );
        assert_eq!(
            &partial[r[3].start..r[3].end()],
            &before[r[3].start..r[3].end()]
        );
        let raw = capture.record(&partial);
        capture.cases.push(json!({"name":"injected-late-encoder-absence","after":raw,"encoded_tiles":1,"published":false,
            "native_completed":true,"native_error":false,"fault_is_test_injection":true}));
        pipeline.fail_before_tile.set(None);
        for (name, input, initial_status, wanted_status) in [
            ("product-status-is-never-reset", 0.5_f32.to_bits(), 16, 16),
            ("product-nonfinite-is-fatal", f32::INFINITY.to_bits(), 0, 1),
        ] {
            arena.with_mut(|s| {
                s.copy_from_slice(&before);
                s[r[0].start..r[0].end()].fill(input);
                s[r[2].start] = initial_status;
            });
            let command = gpu.command().unwrap();
            pipeline
                .encode(
                    &command,
                    matrix,
                    DeviceBuffers {
                        input: r[0].range(&arena),
                        output: r[1].range(&arena),
                        status: r[2].range(&arena),
                    },
                    1,
                )
                .unwrap();
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            assert!(command.error().is_none());
            let after = arena.with_slice(|s| s.to_vec());
            assert_eq!(after[r[2].start], wanted_status);
            assert_eq!(
                &after[r[3].start..r[3].end()],
                &before[r[3].start..r[3].end()]
            );
            let raw = capture.record(&after);
            capture.cases.push(json!({"name":name,"after":raw,"wanted_product_status":wanted_status,"published":false,"native_completed":true,"native_error":false}));
        }
        for (t, h) in tiles.iter().zip(initial_weights) {
            assert_eq!(t.with_slice(hash_words), h);
        }
        capture.finish("toklet-bf16-strided-rejection-v1");
    });
}

#[test]
#[ignore = "explicit warmed borrowed multi-tile Rust allocation count; native allocations are separate"]
fn warmed_tiled_encoding_keeps_zero_rust_allocations() {
    autoreleasepool(|_| {
        use super::super::bf16_encode::AdmissionPipeline;
        use crate::model::gemma_reference::tests::Measurement;
        let fixture = PathBuf::from(
            std::env::var_os("TOKLET_BF16_STRIDED_FIXTURE").expect("explicit fixture directory"),
        );
        let raw = bounded(&fixture.join("warm-chain-protocol.json"), 8192);
        assert_eq!(
            sha(&raw),
            "9fd70d177d5ef43335c2863973af7cc4f7ecced983f52de087f9d4d64c99ec0e"
        );
        let context = Context::with_projection_pipeline(CONTEXT_ALLOWANCE_BYTES, 16).unwrap();
        let gpu = &context.gpu;
        let pipelines =
            StridedBf16Pipelines::new(&context, StridedBf16Pipelines::policy_bytes()).unwrap();
        let admission = AdmissionPipeline::new(gpu).unwrap();
        let first_tiles = constant_tiles(gpu, 8960, 64, 1. / 1024.);
        let second_tiles = constant_tiles(gpu, 3, 8960, 0.5);
        let first = Bf16MatrixView::new(&context, &first_tiles, 8960, 64, 1. / 1024.).unwrap();
        let second = Bf16MatrixView::new(&context, &second_tiles, 3, 8960, 0.5).unwrap();
        let first_hashes: Vec<_> = first_tiles
            .iter()
            .map(|t| t.with_slice(hash_words))
            .collect();
        let second_hashes: Vec<_> = second_tiles
            .iter()
            .map(|t| t.with_slice(hash_words))
            .collect();
        let (r, count) = regions(&[64, 8960, 3, 1, 1, 1, 1]);
        let mut arena = Buffer::<u32>::new(gpu, count).unwrap();
        arena.with_mut(|s| {
            s.fill(SENTINEL);
            s[r[0].start..r[0].end()].fill(0.25_f32.to_bits());
            for reg in &r[3..] {
                s[reg.start] = 0;
            }
        });
        let run = |arena: &Buffer<u32>| {
            autoreleasepool(|_| {
                let command = gpu.command().unwrap();
                admission
                    .encode_with_l1(
                        &command,
                        first.admission_tile(),
                        r[0].range(arena),
                        r[3].range(arena),
                        1,
                    )
                    .unwrap();
                pipelines
                    .encode(
                        &command,
                        first,
                        DeviceBuffers {
                            input: r[0].range(arena),
                            output: r[1].range(arena),
                            status: r[4].range(arena),
                        },
                        1,
                    )
                    .unwrap();
                // Genuine device dependency: the second admission/projection read
                // the first full strided output on the SAME retained command.
                admission
                    .encode_with_l1(
                        &command,
                        second.admission_tile(),
                        r[1].range(arena),
                        r[5].range(arena),
                        1,
                    )
                    .unwrap();
                pipelines
                    .encode(
                        &command,
                        second,
                        DeviceBuffers {
                            input: r[1].range(arena),
                            output: r[2].range(arena),
                            status: r[6].range(arena),
                        },
                        1,
                    )
                    .unwrap();
                command.commit();
                command.waitUntilCompleted();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                assert!(command.error().is_none());
            });
        };
        run(&arena);
        let measurement = Measurement::start();
        for _ in 0..32 {
            run(&arena);
        }
        let allocations = measurement.finish();
        assert_eq!(allocations, 0);
        arena.with_slice(|s| {
            assert!(
                s[r[1].start..r[1].end()]
                    .iter()
                    .all(|&x| x == (1_f32 / 64.).to_bits())
            );
            assert!(
                s[r[2].start..r[2].end()]
                    .iter()
                    .all(|&x| x == 70_f32.to_bits())
            );
            for reg in &r[3..] {
                assert_eq!(s[reg.start], 0);
            }
        });
        for (tile, before) in first_tiles.iter().zip(&first_hashes) {
            assert_eq!(tile.with_slice(hash_words), *before);
        }
        for (tile, before) in second_tiles.iter().zip(&second_hashes) {
            assert_eq!(tile.with_slice(hash_words), *before);
        }
        let mut capture = Capture::new();
        let after = arena.with_slice(|s| capture.record(s));
        capture.cases.push(json!({"name":"warm-32-dependent-chain","iterations":32,"rust_allocations":allocations,"native_allocations_measured":false,
            "warm_protocol_sha256":"9fd70d177d5ef43335c2863973af7cc4f7ecced983f52de087f9d4d64c99ec0e",
            "after":after,"regions":r.iter().map(|x|json!([x.start,x.len])).collect::<Vec<_>>(),
            "dispatches_per_iteration":5,"commands_per_iteration":1,"waits_per_iteration":1,"native_completed":true,"native_error":false}));
        capture.finish("toklet-bf16-strided-warm-v1");
    });
}
