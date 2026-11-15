use super::*;
use crate::metal::gemma_affine::tests::l1_fixture;
use crate::metal::{gemma_rms, memory::BufferRange};
use project::bf16_encode::{AdmissionDecision, AdmissionPipeline, admission_decision};
use serde_json::{Value, json};
use std::io::Cursor;

fn frozen_sparse() -> l1_fixture::Case {
    let case = l1_fixture::cases()
        .into_iter()
        .find(|v| v.reference["name"] == "shape-sparse-64")
        .unwrap();
    // Authenticate the unchanged frozen fixture representation as well as its
    // input/reference arrays. The BF16 matrix below has these same constant rows.
    crate::mlx::affine::Matrix::from_le_bytes(case.layout, &case.codes, &case.scales, &case.biases)
        .unwrap();
    assert_eq!(case.layout.columns(), 64);
    assert_eq!(case.reference["weights"], json!([1., 0.5, -0.75]));
    case
}
fn report(label: &str, raw: Value) {
    use std::io::Write;
    println!("{label} {raw}");
    std::io::stdout().flush().unwrap();
}
fn packed(rows: usize, columns: usize, weight: impl Fn(usize, usize) -> f32) -> Vec<u8> {
    let mut raw = Vec::with_capacity(rows * columns * 2);
    for row in 0..rows {
        for column in 0..columns {
            let w = weight(row, column);
            assert_eq!(w.to_bits() & 0xffff, 0);
            raw.extend_from_slice(&((w.to_bits() >> 16) as u16).to_le_bytes());
        }
    }
    raw
}
fn matrix(context: &Bf16Context, rows: usize, columns: usize, raw: &[u8]) -> Matrix {
    let mut storage = 0;
    upload_matrix(
        &context.0.gpu,
        &mut Cursor::new(raw),
        rows,
        columns,
        &mut storage,
        usize::MAX,
    )
    .unwrap()
}
fn check(raw: &[u32], expected: &[f32]) {
    assert_eq!(raw.len(), expected.len());
    for (&bits, &reference) in raw.iter().zip(expected) {
        let actual = f32::from_bits(bits);
        assert!(actual.is_finite());
        assert!(
            (f64::from(actual) - f64::from(reference)).abs()
                <= 1e-5 + 1e-4 * f64::from(reference).abs()
        );
    }
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
    fn range(self, arena: &Buffer<u32>) -> BufferRange<'_> {
        BufferRange::new(arena, self.start, self.len).unwrap()
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
fn guard_and_immutable(before: &[u32], after: &[u32], writable: &[Region]) {
    assert_eq!(before.len(), after.len());
    for (i, (&a, &b)) in after.iter().zip(before).enumerate() {
        if !writable.iter().any(|r| (r.start..r.end()).contains(&i)) {
            assert_eq!(a, b, "immutable/guard {i}");
        }
    }
}

#[test]
#[ignore = "explicit BF16 shared RMS/admission/two-projection single-command fixture"]
fn borrowed_single_tile_chain_keeps_weights_ranges_and_statuses_private() {
    autoreleasepool(|_| {
        let case = frozen_sparse();
        let context = Bf16Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let gpu = &context.0.gpu;
        let rms = gemma_rms::Pipelines::new(gpu).unwrap();
        let admission = AdmissionPipeline::new(gpu).unwrap();
        let first = matrix(
            &context,
            64,
            64,
            &packed(64, 64, |row, column| if row == column { 1. } else { 0. }),
        );
        let second = matrix(
            &context,
            3,
            64,
            &packed(3, 64, |row, _| [1., 0.5, -0.75][row]),
        );
        let first_view = first.single_tile(&context).unwrap();
        let second_view = second.single_tile(&context).unwrap();
        let original_weights = [
            first.tiles[0].with_slice(|s| s.to_vec()),
            second.tiles[0].with_slice(|s| s.to_vec()),
        ];
        for tokens in [1, 32] {
            let input = case.input.repeat(tokens);
            let mut reference_rms = vec![0.; input.len()];
            crate::cpu::rmsnorm(&input, &[1.; 64], &mut reference_rms, tokens, 64, 1e-6).unwrap();
            let reference: Vec<f32> = reference_rms
                .chunks_exact(64)
                .flat_map(|row| {
                    [1., 0.5, -0.75]
                        .map(|w| row.iter().map(|&x| f64::from(x) * w).sum::<f64>() as f32)
                })
                .collect();
            // Input, learned RMS vector, norm, first output, final output and
            // five distinct status slots. Seven-word gaps and nonzero offsets.
            let (r, count) = regions(&[
                input.len(),
                64,
                input.len(),
                input.len(),
                tokens * 3,
                1,
                1,
                1,
                1,
                1,
            ]);
            let mut arena = Buffer::<u32>::new(gpu, count).unwrap();
            arena.with_mut(|s| {
                s.fill(SENTINEL);
                for (out, value) in s[r[0].start..r[0].end()].iter_mut().zip(&input) {
                    *out = value.to_bits();
                }
                s[r[1].start..r[1].end()].fill(1_f32.to_bits());
                for status in &r[5..] {
                    s[status.start] = 0;
                }
            });
            let before = arena.with_slice(|s| s.to_vec());
            let command = gpu.command().unwrap();
            rms.encode(
                &command,
                &gemma_rms::DeviceBuffers {
                    input: r[0].range(&arena),
                    weights: r[1].range(&arena),
                    output: r[2].range(&arena),
                    status: r[5].range(&arena),
                },
                tokens,
                64,
                1e-6,
                true,
            )
            .unwrap();
            admission
                .encode(
                    &command,
                    first_view,
                    r[2].range(&arena),
                    r[6].range(&arena),
                    tokens,
                )
                .unwrap();
            project::encode_bf16(
                &context.0,
                &command,
                first_view,
                r[2].range(&arena),
                r[3].range(&arena),
                r[7].range(&arena),
                tokens,
            )
            .unwrap();
            admission
                .encode(
                    &command,
                    second_view,
                    r[3].range(&arena),
                    r[8].range(&arena),
                    tokens,
                )
                .unwrap();
            project::encode_bf16(
                &context.0,
                &command,
                second_view,
                r[3].range(&arena),
                r[4].range(&arena),
                r[9].range(&arena),
                tokens,
            )
            .unwrap();
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            let after = arena.with_slice(|s| s.to_vec());
            let current_weights = [
                first.tiles[0].with_slice(|s| s.to_vec()),
                second.tiles[0].with_slice(|s| s.to_vec()),
            ];
            report(
                "BF16_BORROWED_CHAIN_RAW",
                json!({"tokens":tokens,"columns":64,"first_rows":64,
                "last_rows":3,"source_fixture":"gemma-metal-affine-l1-v1/shape-sparse-64",
                "device":context.device_name(),"regions":r.iter().map(|v|[v.start,v.len]).collect::<Vec<_>>(),
                "before_words":before,"after_words":after,"original_weight_words":original_weights,
                "current_weight_words":current_weights,"threshold_bits":[first_view.max_input_bits(),second_view.max_input_bits()],
                "chain_commands":1,"chain_waits":1,"graph_qualified":false}),
            );
            assert!(command.error().is_none());
            assert_eq!(current_weights, original_weights);
            guard_and_immutable(&before, &after, &r[2..]);
            for slot in &r[5..] {
                assert_eq!(after[slot.start], 0);
            }
            assert_eq!(
                admission_decision(after[r[6].start]).unwrap(),
                AdmissionDecision::Admitted
            );
            assert_eq!(
                admission_decision(after[r[8].start]).unwrap(),
                AdmissionDecision::Admitted
            );
            check(&after[r[2].start..r[2].end()], &reference_rms);
            // Identity BF16 projection preserves the admitted normal FP32 input exactly.
            assert_eq!(
                &after[r[3].start..r[3].end()],
                &after[r[2].start..r[2].end()]
            );
            check(&after[r[4].start..r[4].end()], &reference);
        }
    });
}

#[test]
#[ignore = "explicit BF16 integer max admission, sparse L1 replay and nonfinite status ordering"]
fn max_miss_requests_unchanged_reference_l1_instead_of_rejecting() {
    autoreleasepool(|_| {
        let case = frozen_sparse();
        let context = Bf16Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let gpu = &context.0.gpu;
        let admission = AdmissionPipeline::new(gpu).unwrap();
        let weights = matrix(
            &context,
            3,
            64,
            &packed(3, 64, |row, _| [1., 0.5, -0.75][row]),
        );
        let tile = weights.single_tile(&context).unwrap();
        let initial_weights = weights.tiles[0].with_slice(|s| s.to_vec());
        let edge = f32::from_bits(tile.max_input_bits());
        let next = f32::from_bits(tile.max_input_bits() + 1);
        let mut sparse_next = vec![0.; 64];
        sparse_next[63] = -next;
        let mut both = vec![next; 128];
        both[127] = f32::NAN;
        let cases = [
            ("frozen-sparse-L1-pass", case.input.clone(), 2),
            ("threshold-inclusive", vec![-edge; 64], 0),
            ("threshold-next-dense", vec![next; 64], 2),
            ("finite-absolute-reject", vec![f32::MAX; 64], 2),
            ("threshold-next-sparse-L1-pass", sparse_next, 2),
            (
                "subnormal-signs",
                (0..64)
                    .map(|i| f32::from_bits(if i % 2 == 0 { 1 } else { 0x80000001 }))
                    .collect(),
                0,
            ),
            ("positive-infinity", vec![f32::INFINITY; 64], 1),
            ("negative-infinity", vec![f32::NEG_INFINITY; 64], 1),
            ("finite-miss-and-nan-rows", both, 3),
        ];
        for (name, input, expected_status) in cases {
            let tokens = input.len() / 64;
            let (r, count) = regions(&[input.len(), tokens * 3, 1, 1]);
            let mut arena = Buffer::<u32>::new(gpu, count).unwrap();
            arena.with_mut(|s| {
                s.fill(SENTINEL);
                for (out, value) in s[r[0].start..r[0].end()].iter_mut().zip(&input) {
                    *out = value.to_bits();
                }
                s[r[2].start] = 0;
                s[r[3].start] = 0;
            });
            let before = arena.with_slice(|s| s.to_vec());
            let command = gpu.command().unwrap();
            admission
                .encode(
                    &command,
                    tile,
                    r[0].range(&arena),
                    r[2].range(&arena),
                    tokens,
                )
                .unwrap();
            // Deliberate speculation demonstrates that a proof miss is decoded
            // before any dependent product nonfinite status can change its meaning.
            project::encode_bf16(
                &context.0,
                &command,
                tile,
                r[0].range(&arena),
                r[1].range(&arena),
                r[3].range(&arena),
                tokens,
            )
            .unwrap();
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            let after = arena.with_slice(|s| s.to_vec());
            report(
                "BF16_ADMISSION_RAW",
                json!({"case":name,"tokens":tokens,"threshold_bits":tile.max_input_bits(),
                "regions":r.iter().map(|v|[v.start,v.len]).collect::<Vec<_>>(),
                "before_words":before,"after_words":after,"expected_status":expected_status}),
            );
            assert!(command.error().is_none());
            guard_and_immutable(&before, &after, &r[1..]);
            assert_eq!(after[r[2].start], expected_status);
            let mut published = vec![99.; tokens * 3];
            match admission_decision(after[r[2].start]) {
                Ok(AdmissionDecision::Admitted) => {
                    assert_eq!(after[r[3].start], 0);
                    assert!(input.iter().all(|v| v.is_finite()));
                    admit(64, weights.maximum, &input).unwrap();
                }
                Ok(AdmissionDecision::ReferenceReplay) => {
                    assert!(published.iter().all(|&v| v == 99.));
                    // Unchanged host proof decides whether this request passes
                    // L1 or fails with the original absolute-before-ordered error.
                    let result = admit(64, weights.maximum, &input);
                    if name.contains("L1-pass") {
                        result.unwrap();
                        let mut a = Buffer::new(gpu, input.len() + GUARD).unwrap();
                        let mut b = Buffer::new(gpu, published.len() + GUARD).unwrap();
                        let mut status = Buffer::new(gpu, 1 + GUARD).unwrap();
                        let mut pending = None;
                        project::run(
                            &context.0,
                            project::Matrix::bf16(3, 64, &weights.tiles[0]),
                            project::Buffers {
                                input: &mut a,
                                output: &mut b,
                                status: &mut status,
                                pending: &mut pending,
                            },
                            &input,
                            &mut published,
                            tokens,
                        )
                        .unwrap();
                        let expected: Vec<f32> = input
                            .chunks_exact(64)
                            .flat_map(|row| {
                                [1., 0.5, -0.75].map(|w| {
                                    row.iter().map(|&x| f64::from(x) * w).sum::<f64>() as f32
                                })
                            })
                            .collect();
                        assert_eq!(published, expected);
                        if name == "frozen-sparse-L1-pass" {
                            assert!(
                                published
                                    .iter()
                                    .zip(case.reference["ordered_fp64"].as_array().unwrap())
                                    .all(|(&a, b)| f64::from(a) == b.as_f64().unwrap())
                            );
                        }
                    } else {
                        if name == "finite-absolute-reject" {
                            assert_eq!(after[r[3].start], 1);
                            assert!(matches!(
                                result,
                                Err(Error::Options("Gemma affine input absolute-product bound"))
                            ));
                        } else {
                            assert!(matches!(
                                result,
                                Err(Error::Options(
                                    "Gemma affine input ordered-sum accuracy bound"
                                ))
                            ));
                        }
                        assert!(published.iter().all(|&v| v == 99.));
                    }
                }
                Err(Error::Nonfinite) => {
                    assert!(expected_status & 1 != 0);
                    assert!(published.iter().all(|&v| v == 99.));
                }
                other => panic!("unexpected admission result: {other:?}"),
            }
        }
        assert_eq!(weights.tiles[0].with_slice(|s| s.to_vec()), initial_weights);
    });
}

#[test]
#[ignore = "explicit BF16 one-tile/range/command rejection and finite zero-weight admission"]
fn borrowed_projection_preflight_rejects_before_any_encoding() {
    autoreleasepool(|_| {
        let context = Bf16Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let gpu = &context.0.gpu;
        let admission = AdmissionPipeline::new(gpu).unwrap();
        let mut weights = matrix(&context, 3, 64, &packed(3, 64, |_, _| 1.));
        weights.rows = 8193;
        assert!(weights.single_tile(&context).is_err());
        weights.rows = 3;
        weights.columns = 65;
        assert!(weights.single_tile(&context).is_err());
        weights.columns = 64;
        weights.maximum = f64::NAN;
        assert!(weights.single_tile(&context).is_err());
        weights.maximum = 1.;
        weights.tiles[0].len -= 1;
        assert!(weights.single_tile(&context).is_err());
        weights.tiles[0].len += 1;
        let one = weights.tiles.pop().unwrap();
        assert!(weights.single_tile(&context).is_err());
        weights.tiles.push(one);
        weights
            .tiles
            .push(Buffer::new(gpu, 3 * 64 / 2 + GUARD).unwrap());
        assert!(weights.single_tile(&context).is_err());
        drop(weights.tiles.pop());
        let tile = weights.single_tile(&context).unwrap();
        let mut arena = Buffer::<u32>::new(gpu, 128).unwrap();
        arena.with_mut(|s| {
            s.fill(SENTINEL);
            s[7..71].fill(1_f32.to_bits());
            s[100] = 0;
        });
        let before = arena.with_slice(|s| s.to_vec());
        let range = |start, len| BufferRange::new(&arena, start, len).unwrap();
        let command = gpu.command().unwrap();
        for tokens in [0, 33, usize::MAX] {
            assert!(
                project::encode_bf16(
                    &context.0,
                    &command,
                    tile,
                    range(7, 64),
                    range(80, 3),
                    range(100, 1),
                    tokens
                )
                .is_err()
            );
            assert!(
                admission
                    .encode(&command, tile, range(7, 64), range(100, 1), tokens)
                    .is_err()
            );
        }
        for field in 0..3 {
            let (input, output, status) = match field {
                0 => (range(7, 63), range(80, 3), range(100, 1)),
                1 => (range(7, 64), range(80, 2), range(100, 1)),
                _ => (range(7, 64), range(80, 3), range(100, 0)),
            };
            assert!(
                project::encode_bf16(&context.0, &command, tile, input, output, status, 1).is_err()
            );
        }
        assert!(
            admission
                .encode(&command, tile, range(7, 63), range(100, 1), 1)
                .is_err()
        );
        assert!(
            admission
                .encode(&command, tile, range(7, 64), range(100, 0), 1)
                .is_err()
        );
        for (output, status) in [
            (range(8, 3), range(100, 1)),
            (range(80, 3), range(81, 1)),
            (range(80, 3), range(7, 1)),
        ] {
            assert!(
                project::encode_bf16(&context.0, &command, tile, range(7, 64), output, status, 1)
                    .is_err()
            );
        }
        assert!(
            admission
                .encode(&command, tile, range(7, 64), range(8, 1), 1)
                .is_err()
        );
        let coefficient_status = BufferRange::new(&weights.tiles[0], 1, 1).unwrap();
        assert!(
            project::encode_bf16(
                &context.0,
                &command,
                tile,
                range(7, 64),
                range(80, 3),
                coefficient_status,
                1
            )
            .is_err()
        );
        assert!(
            admission
                .encode(&command, tile, range(7, 64), coefficient_status, 1)
                .is_err()
        );
        let unretained = gpu.queue.commandBufferWithUnretainedReferences().unwrap();
        assert!(
            project::encode_bf16(
                &context.0,
                &unretained,
                tile,
                range(7, 64),
                range(80, 3),
                range(100, 1),
                1
            )
            .is_err()
        );
        assert!(
            admission
                .encode(&unretained, tile, range(7, 64), range(100, 1), 1)
                .is_err()
        );
        assert_eq!(command.status(), MTLCommandBufferStatus::NotEnqueued);
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let after = arena.with_slice(|s| s.to_vec());
        report(
            "BF16_BORROWED_REJECTION_RAW",
            json!({"before_words":before,"after_words":after}),
        );
        assert_eq!(before, after);
        assert!(
            project::encode_bf16(
                &context.0,
                &command,
                tile,
                range(7, 64),
                range(80, 3),
                range(100, 1),
                1
            )
            .is_err()
        );
        assert!(
            admission
                .encode(&command, tile, range(7, 64), range(100, 1), 1)
                .is_err()
        );

        // A zero matrix's original bounds admit every finite FP32 input.
        // Exponent-255 still rejects even though its weight maximum is zero.
        let zero = matrix(&context, 1, 64, &packed(1, 64, |_, _| 0.));
        let zero_tile = zero.single_tile(&context).unwrap();
        assert_eq!(zero_tile.max_input_bits(), f32::MAX.to_bits());
        for (bits, flags) in [
            (f32::MAX.to_bits(), 0),
            ((-f32::MAX).to_bits(), 0),
            (0x80000000, 0),
            (0x00000001, 0),
            (0x80000001, 0),
            (0x7f800000, 1),
            (0xff800000, 1),
            (0x7fc12345, 1),
        ] {
            let mut input = Buffer::<u32>::new(gpu, 64 + GUARD).unwrap();
            let mut status = Buffer::<u32>::new(gpu, 1 + GUARD).unwrap();
            input.with_mut(|s| {
                s.fill(SENTINEL);
                s[..64].fill(bits);
            });
            status.with_mut(|s| {
                s.fill(SENTINEL);
                s[0] = 0;
            });
            let command = gpu.command().unwrap();
            admission
                .encode(
                    &command,
                    zero_tile,
                    BufferRange::new(&input, 0, 64).unwrap(),
                    BufferRange::new(&status, 0, 1).unwrap(),
                    1,
                )
                .unwrap();
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            let raw_input = input.with_slice(|s| s.to_vec());
            let raw_status = status.with_slice(|s| s.to_vec());
            report(
                "BF16_ZERO_MAX_ADMISSION_RAW",
                json!({"input_bits":bits,"input_words":raw_input,
                "status_words":raw_status,"expected_status":flags,"threshold_bits":zero_tile.max_input_bits()}),
            );
            assert!(command.error().is_none());
            assert_eq!(&raw_input[..64], &[bits; 64]);
            assert!(raw_input[64..].iter().all(|&v| v == SENTINEL));
            assert_eq!(raw_status[0], flags);
            assert!(raw_status[1..].iter().all(|&v| v == SENTINEL));
        }
    });
}

#[test]
#[ignore = "explicit GPU BF16 max+conservative L1 source-order admission/projection controls"]
fn borrowed_l1_supplement_accepts_sparse_max_misses_without_host_replay() {
    autoreleasepool(|_| {
        let context = Bf16Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let gpu = &context.0.gpu;
        let admission = AdmissionPipeline::new(gpu).unwrap();
        let second = matrix(
            &context,
            3,
            64,
            &packed(3, 64, |row, _| [1., 0.5, -0.75][row]),
        );
        let tile = second.single_tile(&context).unwrap();
        // Actual BF16 maxima, including real zero/capped ties and order changes.
        // No hypothetical rounded-threshold witness is passed off as a model.
        let small = matrix(&context, 1, 64, &packed(1, 64, |_, _| 0.5));
        let medium = matrix(&context, 1, 64, &packed(1, 64, |_, _| 0.75));
        let zero = matrix(&context, 3, 64, &packed(3, 64, |_, _| 0.));
        let tiny = matrix(&context, 1, 64, &packed(1, 64, |_, _| f32::MIN_POSITIVE));
        let a = small.single_tile(&context).unwrap();
        let b = medium.single_tile(&context).unwrap();
        let z = zero.single_tile(&context).unwrap();
        let n = tiny.single_tile(&context).unwrap();
        for group in [
            [a, tile, b],
            [tile, b, a],
            [b, a, tile],
            [z, n, z],
            [n, z, n],
        ] {
            let selected = super::producer::strongest_admission_tile(&group).unwrap();
            assert_eq!(
                selected.maximum_bits(),
                group.iter().map(|t| t.maximum_bits()).max().unwrap()
            );
            for member in group {
                assert!(selected.max_input_bits() <= member.max_input_bits());
                assert!(selected.l1_input_bits() <= member.l1_input_bits());
            }
        }
        let max_next = f32::from_bits(tile.max_input_bits() + 1);
        let l1 = f32::from_bits(tile.l1_input_bits());
        let mut sparse = vec![-0.; 64];
        sparse[63] = max_next;
        let mut subnormal = sparse.clone();
        subnormal[0] = f32::from_bits(1);
        subnormal[1] = -f32::from_bits(0x007f_ffff);
        let mut endpoint = vec![0.; 64];
        endpoint[63] = l1;
        let mut over = endpoint.clone();
        over[63] = f32::from_bits(tile.l1_input_bits() + 1);
        let mut nonfinite = sparse.clone();
        nonfinite[31] = f32::INFINITY;
        let mut nan = sparse.clone();
        nan[17] = f32::from_bits(0x7fc0_0031);
        let mut overflow = vec![0.; 64];
        overflow[0] = f32::MAX;
        overflow[1] = 1.;
        // Deliberate conservative miss: source max fails; old FP64 L1 passes,
        // while two-up FP32 inflation near its cutoff requests reference replay.
        let rest = (((f64::from(l1) - f64::from(max_next)) / 63.) as f32).next_down();
        let mut conservative = vec![rest; 64];
        conservative[0] = max_next;
        admit(64, 1., &conservative).unwrap();
        let frozen = frozen_sparse();
        let mut cases = vec![
            ("frozen-independent-sparse", frozen.input, 0, 0),
            (
                "max-endpoint",
                vec![f32::from_bits(tile.max_input_bits()); 64],
                0,
                0,
            ),
            ("max-next-sparse", sparse.clone(), 0, 0),
            ("max-next-subnormal-inflation", subnormal, 0, 0),
            ("l1-endpoint-sparse", endpoint, 0, 0),
            ("l1-next-sparse-replay", over, 0, 2),
            ("conservative-miss-host-l1-pass", conservative, 0, 2),
            ("dense-max-next-replay", vec![max_next; 64], 0, 2),
            ("positive-overflow-replay", overflow, 0, 2),
            ("nonfinite-is-fatal", nonfinite, 0, 1),
            ("nan-is-fatal", nan, 0, 1),
            ("status-is-never-cleared", vec![0.; 64], 2, 2),
            ("rows32-sparse", sparse.repeat(32), 0, 0),
        ];
        // A fatal row must dominate another row's proof miss in the dedicated
        // status. Publication still uses the unchanged graph-order classifier.
        let mut mixed = vec![max_next; 64 * 32];
        mixed[64 * 31] = f32::NEG_INFINITY;
        cases.push(("rows32-fatal-and-replay", mixed, 0, 3));
        let mut cases: Vec<_> = cases
            .into_iter()
            .map(|(name, input, initial, expected)| (name, input, initial, expected, false))
            .collect();
        cases.push((
            "zero-weight-all-finite-max-pass",
            vec![f32::MAX; 64],
            0,
            0,
            true,
        ));
        for (name, input, initial_status, expected_status, is_zero) in cases {
            let source = if is_zero { &zero } else { &second };
            let tile = source.single_tile(&context).unwrap();
            let maximum = if is_zero { 0. } else { 1. };
            let tokens = input.len() / 64;
            let (r, count) = regions(&[input.len(), tokens * 3, 1, 1, tokens * 4]);
            let mut arena = Buffer::<u32>::new(gpu, count).unwrap();
            arena.with_mut(|s| {
                s.fill(SENTINEL);
                for (dst, value) in s[r[0].start..r[0].end()].iter_mut().zip(&input) {
                    *dst = value.to_bits();
                }
                s[r[2].start] = initial_status;
                s[r[3].start] = 0;
            });
            let before = arena.with_slice(|s| s.to_vec());
            let weights_before = source.tiles[0].with_slice(|s| s.to_vec());
            let command = gpu.command().unwrap();
            assert!(
                admission
                    .encode_with_l1(
                        &command,
                        tile,
                        BufferRange::new(&arena, r[0].start, input.len() - 1).unwrap(),
                        r[2].range(&arena),
                        tokens
                    )
                    .is_err()
            );
            assert!(
                admission
                    .encode_with_l1(
                        &command,
                        tile,
                        r[0].range(&arena),
                        BufferRange::new(&arena, r[0].start, 1).unwrap(),
                        tokens
                    )
                    .is_err()
            );
            assert_eq!(command.status(), MTLCommandBufferStatus::NotEnqueued);
            admission
                .encode_with_l1(
                    &command,
                    tile,
                    r[0].range(&arena),
                    r[2].range(&arena),
                    tokens,
                )
                .unwrap();
            project::encode_bf16(
                &context.0,
                &command,
                tile,
                r[0].range(&arena),
                r[1].range(&arena),
                r[3].range(&arena),
                tokens,
            )
            .unwrap();
            admission
                .encode_l1_diagnostic(
                    &command,
                    tile,
                    r[0].range(&arena),
                    r[4].range(&arena),
                    tokens,
                )
                .unwrap();
            command.commit();
            command.waitUntilCompleted();
            let after = arena.with_slice(|s| s.to_vec());
            let weights_after = source.tiles[0].with_slice(|s| s.to_vec());
            report(
                "BF16_L1_CHAIN_RAW",
                json!({"name":name,"tokens":tokens,"columns":64,"rows":3,
                "maximum":maximum,"max_input_bits":tile.max_input_bits(),"l1_input_bits":tile.l1_input_bits(),
                "regions":r.iter().map(|r|vec![r.start,r.len]).collect::<Vec<_>>(),
                "before_words":before,"after_words":after,"weights_before":weights_before,"weights_after":weights_after,
                "encoded_dispatches":3,"diagnostic_dispatches":1,"submitted_commands":1,"waits":1,
                "command_completed":command.status()==MTLCommandBufferStatus::Completed,"native_error":command.error().is_some()}),
            );
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            assert!(command.error().is_none());
            assert!(
                admission
                    .encode_with_l1(
                        &command,
                        tile,
                        r[0].range(&arena),
                        r[2].range(&arena),
                        tokens
                    )
                    .is_err()
            );
            guard_and_immutable(&before, &after, &[r[1], r[2], r[3], r[4]]);
            assert_eq!(weights_before, weights_after);
            assert_eq!(after[r[2].start], expected_status);
            if expected_status == 0 {
                admit(64, maximum, &input).unwrap();
                let expected: Vec<_> = input
                    .chunks_exact(64)
                    .flat_map(|row| {
                        let sum = row.iter().map(|&v| f64::from(v)).sum::<f64>();
                        if is_zero {
                            [0.; 3]
                        } else {
                            [sum as f32, (sum * 0.5) as f32, (sum * (-0.75)) as f32]
                        }
                    })
                    .collect();
                check(&after[r[1].start..r[1].end()], &expected);
                assert_eq!(after[r[3].start], 0);
            } else if expected_status & 1 != 0 {
                assert!(matches!(
                    admission_decision(after[r[2].start]),
                    Err(Error::Nonfinite)
                ));
            } else {
                assert_eq!(
                    admission_decision(after[r[2].start]).unwrap(),
                    AdmissionDecision::ReferenceReplay
                );
            }
        }
    });
}
