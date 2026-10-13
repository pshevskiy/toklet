mod support;

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
use support::Fixture;
use toklet::{
    cpu,
    model::{Config, LayerWeights, Model, Projection, Weights},
    sampling,
    session::{Error, Options, Session},
};

struct CountAllocations;
thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn count_allocation() {
    let _ = TRACK.try_with(|track| {
        if track.get() {
            ALLOCATIONS.with(|count| count.set(count.get() + 1));
        }
    });
}

// SAFETY: This test-only shim delegates all allocation/deallocation and layout
// contracts unchanged to System. TLS counting touches no allocated memory and
// records only the measured test thread, excluding concurrent harness activity.
unsafe impl GlobalAlloc for CountAllocations {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count_allocation();
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountAllocations = CountAllocations;

fn tiny_parts(f: &Fixture) -> (Config, Weights) {
    let m = &f.manifest["model"];
    let integer = |name: &str| m[name].as_u64().unwrap() as usize;
    let config = Config {
        vocab_size: integer("vocab_size"),
        hidden_size: integer("hidden_size"),
        intermediate_size: integer("intermediate_size"),
        num_layers: integer("num_layers"),
        query_heads: integer("num_query_heads"),
        kv_heads: integer("num_kv_heads"),
        head_dim: integer("head_dim"),
        max_positions: integer("capacity"),
        rms_norm_eps: m["rms_norm_eps"].as_f64().unwrap() as f32,
        rope_theta: m["rope_theta"].as_f64().unwrap() as f32,
    };
    let layers = (0..config.num_layers)
        .map(|index| {
            let prefix = format!("weights.layers.{index}");
            let projection = |name| Projection {
                weight: f.f32(&format!("{prefix}.{name}_proj.weight")),
                bias: f.f32(&format!("{prefix}.{name}_proj.bias")),
            };
            LayerWeights {
                input_norm: f.f32(&format!("{prefix}.input_layernorm")),
                post_attention_norm: f.f32(&format!("{prefix}.post_attention_layernorm")),
                q: projection("q"),
                k: projection("k"),
                v: projection("v"),
                o: f.f32(&format!("{prefix}.o_proj.weight")),
                gate: f.f32(&format!("{prefix}.gate_proj.weight")),
                up: f.f32(&format!("{prefix}.up_proj.weight")),
                down: f.f32(&format!("{prefix}.down_proj.weight")),
            }
        })
        .collect();
    (
        config,
        Weights {
            embedding: f.f32("weights.embed_tokens"),
            layers,
            final_norm: f.f32("weights.norm"),
        },
    )
}

fn options(capacity: usize, chunk: usize, offset: usize) -> Options {
    Options {
        capacity,
        prefill_chunk_size: chunk,
        position_offset: offset,
        ..Options::default()
    }
}

#[test]
fn explicit_all_row_cached_logits_match_each_independent_prefix_without_allocations() {
    let tokens = [1, 7, 13, 3, 47, 11, 2];
    for kernel in [cpu::ProjectionKernel::Scalar, cpu::ProjectionKernel::Tiled] {
        let f = Fixture::tiny();
        let (config, weights) = tiny_parts(&f);
        let model = Model::new(config, weights)
            .unwrap()
            .with_projection_kernel(kernel);
        for offset in [0, 5] {
            let mut scratch = model.scratch(tokens.len() + 1).unwrap();
            let mut expected = vec![0.; model.config().vocab_size];
            for chunk in [1, 3, 8] {
                let mut session = Session::new(&model, options(27, chunk, offset)).unwrap();
                let memory = session.memory();
                let width = session.vocab_size();
                let mut output = vec![0.; chunk * width]; // Separately owned/admitted diagnostic storage.
                session.append(&tokens).unwrap();
                let before = session.history().to_vec();
                assert!(
                    session
                        .evaluate_next_chunk_with_logits(&mut output[..width - 1])
                        .is_err()
                );
                assert_eq!(session.history(), before);
                assert_eq!(session.evaluated_len(), 0);
                while session.pending_len() != 0 {
                    let start = session.evaluated_len();
                    let rows = session.pending_len().min(chunk);
                    TRACK.with(|track| track.set(true));
                    ALLOCATIONS.with(|n| n.set(0));
                    let result =
                        session.evaluate_next_chunk_with_logits(&mut output[..rows * width]);
                    TRACK.with(|track| track.set(false));
                    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
                    assert_eq!(result.unwrap(), rows);
                    for row in 0..rows {
                        model
                            .forward(
                                &tokens[..start + row + 1],
                                offset,
                                &mut scratch,
                                &mut expected,
                            )
                            .unwrap();
                        assert_eq!(
                            &output[row * width..(row + 1) * width],
                            expected,
                            "{kernel:?} offset{offset} chunk{chunk} row{row}"
                        );
                    }
                    assert_eq!(session.memory(), memory);
                    if session.pending_len() != 0 {
                        assert!(session.logits().is_none());
                    }
                }
                assert_eq!(session.logits().unwrap(), expected);
                assert_eq!(session.evaluate_next_chunk_with_logits(&mut []).unwrap(), 0);
                assert!(
                    session
                        .evaluate_next_chunk_with_logits(&mut output[..width])
                        .is_err()
                );
                assert_eq!(session.logits().unwrap(), expected);
                // The original final-row method remains available in the same session.
                session.append(&[13]).unwrap();
                session.evaluate_next_chunk().unwrap();
                model
                    .forward(session.history(), offset, &mut scratch, &mut expected)
                    .unwrap();
                assert_eq!(session.logits().unwrap(), expected);
            }
        }
    }
}

#[test]
fn teacher_forced_timing_keeps_exact_prefix_and_pending_state() {
    for kernel in [cpu::ProjectionKernel::Scalar, cpu::ProjectionKernel::Tiled] {
        use toklet::{benchmark::measure_teacher_forced, generation::Input};
        let f = Fixture::tiny();
        let (c, w) = tiny_parts(&f);
        let model = Model::new(c, w).unwrap().with_projection_kernel(kernel);
        let mut session = Session::new(&model, options(32, 3, 0)).unwrap();
        let prompt = f.u32("cases.prompt.tokens");
        let fixed = [47, 11, 3, 7];
        let sample = measure_teacher_forced(&mut session, Input::Reset(&prompt), &fixed).unwrap();
        assert_eq!(
            (
                sample.reused_tokens,
                sample.prefill_tokens,
                sample.evaluated_tokens,
                sample.pending_tokens
            ),
            (0, 4, 7, 1)
        );
        assert_eq!(sample.token_event_seconds.len(), 4);
        assert_eq!(sample.ttft_seconds, sample.token_event_seconds[0]);
        assert_eq!(
            sample.decode_seconds,
            sample.token_event_seconds[3] - sample.ttft_seconds
        );
        assert!(sample.prefill_seconds <= sample.ttft_seconds);
        assert!(sample.token_event_seconds.windows(2).all(|w| w[0] <= w[1]));
        assert!(sample.total_seconds >= sample.token_event_seconds[3]);
        let mut history = prompt;
        history.extend(fixed);
        assert_eq!(session.history(), history);
        let sample = measure_teacher_forced(&mut session, Input::Continue, &[13]).unwrap();
        assert_eq!(
            (
                sample.reused_tokens,
                sample.prefill_tokens,
                sample.evaluated_tokens,
                sample.pending_tokens
            ),
            (7, 1, 8, 1)
        );
        assert_eq!(sample.decode_seconds, 0.0);
        session.discard_pending();
        let mut reference = [0.0; 64];
        model
            .forward(&history, 0, &mut model.scratch(32).unwrap(), &mut reference)
            .unwrap();
        assert_eq!(session.logits().unwrap(), reference);
        let sample =
            measure_teacher_forced(&mut session, Input::Prompt(&history), &[17, 1]).unwrap();
        assert_eq!((sample.reused_tokens, sample.prefill_tokens), (8, 0));
        session.discard_pending();
        let before = session.history().to_vec();
        let logits = session.logits().unwrap().to_vec();
        for (input, output) in [
            (Input::Reset(&[1]), &[][..]),
            (Input::Continue, &[64][..]),
            (Input::Reset(&[1; 32]), &[1][..]),
            (Input::Append(&[64]), &[1][..]),
        ] {
            assert!(measure_teacher_forced(&mut session, input, output).is_err());
            assert_eq!(session.history(), before);
            assert_eq!(session.logits().unwrap(), logits);
        }
    }
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && (a - e).abs() <= 1e-5 + 1e-4 * e.abs(),
            "{a} != {e}"
        );
    }
}

#[test]
fn one_shot_irregular_chunks_and_decode_match_full_prefix_and_frozen_oracle() {
    for kernel in [cpu::ProjectionKernel::Scalar, cpu::ProjectionKernel::Tiled] {
        let fixture = Fixture::tiny();
        let (c, w) = tiny_parts(&fixture);
        let model = Model::new(c, w).unwrap().with_projection_kernel(kernel);
        let mut scratch = model.scratch(32).unwrap();
        let mut reference = [0.0; 64];
        for case in ["prompt", "future_changed", "offset", "single", "capacity"] {
            let tokens = fixture.u32(&format!("cases.{case}.tokens"));
            let expected = fixture.f32(&format!("cases.{case}.logits"));
            let offset = fixture.manifest["cases"][case]["position_offset"]
                .as_u64()
                .unwrap() as usize;
            for chunk in [tokens.len(), 3, 1] {
                let mut session =
                    Session::new(&model, options(32 - offset, chunk, offset)).unwrap();
                let memory = session.memory();
                for part in tokens.chunks(chunk) {
                    session.append(part).unwrap();
                    assert!(session.logits().is_none());
                    session.evaluate().unwrap();
                    let len = session.evaluated_len();
                    assert_eq!(session.pending_len(), 0);
                    model
                        .forward(&tokens[..len], offset, &mut scratch, &mut reference)
                        .unwrap();
                    assert_eq!(
                        session.logits().unwrap(),
                        reference,
                        "{case}, chunk {chunk}, position {len}"
                    );
                    close(
                        session.logits().unwrap(),
                        &expected[(len - 1) * 64..len * 64],
                    );
                    assert_eq!(session.memory(), memory);
                }
                assert_eq!(session.history(), tokens);
                assert_eq!(session.evaluate_next_chunk().unwrap(), 0);
                assert_eq!(session.logits().unwrap(), reference);
            }
        }
    }
}

#[test]
fn sampled_history_is_pending_until_evaluated_and_greedy_trace_matches() {
    for kernel in [cpu::ProjectionKernel::Scalar, cpu::ProjectionKernel::Tiled] {
        let fixture = Fixture::tiny();
        let (c, w) = tiny_parts(&fixture);
        let model = Model::new(c, w).unwrap().with_projection_kernel(kernel);
        let mut session = Session::new(&model, options(32, 3, 0)).unwrap();
        session.append(&fixture.u32("cases.prompt.tokens")).unwrap();
        session.evaluate().unwrap();
        for (step, expected) in fixture.manifest["greedy"]["generated_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            close(
                session.logits().unwrap(),
                &fixture.f32(&format!("greedy.step_{step}.logits")),
            );
            let token = sampling::greedy(session.logits().unwrap()).unwrap();
            assert_eq!(token as u64, expected.as_u64().unwrap());
            let committed = session.evaluated_len();
            session.append(&[token]).unwrap();
            assert_eq!(session.evaluated_len(), committed);
            assert_eq!(session.pending_len(), 1);
            assert!(session.logits().is_none());
            // The entire history, including the sampled pending token, matches.
            let same = session.history().to_vec();
            session.set_tokens(&same).unwrap();
            assert_eq!(session.evaluated_len(), committed);
            assert_eq!(session.evaluate_next_chunk().unwrap(), 1);
        }
    }
}

#[test]
fn longest_committed_prefix_edits_shortening_pending_and_discard_match_full_replay() {
    for kernel in [cpu::ProjectionKernel::Scalar, cpu::ProjectionKernel::Tiled] {
        let fixture = Fixture::tiny();
        let (c, w) = tiny_parts(&fixture);
        let model = Model::new(c, w).unwrap().with_projection_kernel(kernel);
        let base = [1, 7, 13, 3, 47, 11];
        for offset in [0, 5] {
            let mut s = Session::new(&model, options(27, 3, offset)).unwrap();
            let mut scratch = model.scratch(27).unwrap();
            let mut expected = [0.0; 64];
            for (tokens, keep) in [
                (&base[..], 6),
                (&[1, 7, 13, 3, 47, 12][..], 5),
                (&[1, 8, 13, 3][..], 1),
                (&[2, 7, 13][..], 0),
                (&base[..4], 3),
                (&base[..1], 0),
                (&[][..], 0),
            ] {
                s.reset();
                s.set_tokens(&base).unwrap();
                s.evaluate().unwrap();
                let memory = s.memory();
                s.set_tokens(tokens).unwrap();
                assert_eq!(s.evaluated_len(), keep);
                assert_eq!(s.history(), tokens);
                if keep < tokens.len() {
                    assert!(s.logits().is_none());
                }
                s.evaluate().unwrap();
                assert_eq!(s.evaluated_len(), tokens.len());
                assert_eq!(s.memory(), memory);
                if tokens.is_empty() {
                    assert!(s.logits().is_none());
                } else {
                    model
                        .forward(tokens, offset, &mut scratch, &mut expected)
                        .unwrap();
                    assert_eq!(s.logits().unwrap(), expected);
                }
            }
            // Pending sampled history cannot masquerade as an extra committed KV row.
            for pending in [47, 48] {
                s.reset();
                s.set_tokens(&base).unwrap();
                s.evaluate().unwrap();
                s.append(&[47]).unwrap();
                let mut changed = base.to_vec();
                changed.push(pending);
                s.set_tokens(&changed).unwrap();
                assert_eq!((s.evaluated_len(), s.pending_len()), (6, 1));
                s.evaluate().unwrap();
                model
                    .forward(&changed, offset, &mut scratch, &mut expected)
                    .unwrap();
                assert_eq!(s.logits().unwrap(), expected);
            }
            s.reset();
            s.set_tokens(&base).unwrap();
            s.evaluate().unwrap();
            s.append(&[47]).unwrap();
            s.set_tokens(&base).unwrap();
            assert_eq!((s.evaluated_len(), s.pending_len()), (6, 0));
            assert!(s.logits().is_some());
            // A cancelled edited request retains KV, but its earlier logits are
            // unavailable. Discard stages one retained token rather than exposing
            // the old distribution belonging to the longer prompt.
            s.set_tokens(&[1, 7, 13, 3, 47, 12]).unwrap();
            assert_eq!(s.evaluated_len(), 5);
            assert!(s.logits().is_none());
            s.discard_pending();
            assert_eq!(s.history(), &base[..5]);
            assert_eq!((s.evaluated_len(), s.pending_len()), (4, 1));
            assert!(s.logits().is_none());
            s.discard_pending();
            assert_eq!(s.history(), &base[..5]);
            assert_eq!((s.evaluated_len(), s.pending_len()), (4, 1));
            s.append(&[12, 13]).unwrap();
            s.discard_pending();
            s.discard_pending();
            assert_eq!(s.history(), &base[..5]);
            assert_eq!((s.evaluated_len(), s.pending_len()), (4, 1));
            let retained = s.history().to_vec();
            s.set_tokens(&retained).unwrap();
            s.discard_pending();
            assert_eq!(s.history(), &base[..5]);
            assert_eq!((s.evaluated_len(), s.pending_len()), (4, 1));
            s.evaluate().unwrap();
            model
                .forward(&base[..5], offset, &mut scratch, &mut expected)
                .unwrap();
            assert_eq!(s.logits().unwrap(), expected);
            // A fully cached request at the current truncated boundary still
            // requires replay when its distribution has not yet been computed.
            s.set_tokens(&[1, 7, 13, 3, 48]).unwrap();
            assert_eq!(s.evaluated_len(), 4);
            s.set_tokens(&base[..4]).unwrap();
            assert_eq!((s.evaluated_len(), s.pending_len()), (3, 1));
            s.discard_pending();
            s.discard_pending();
            assert_eq!(s.history(), &base[..4]);
            assert_eq!((s.evaluated_len(), s.pending_len()), (3, 1));
            s.evaluate().unwrap();
            model
                .forward(&base[..4], offset, &mut scratch, &mut expected)
                .unwrap();
            assert_eq!(s.logits().unwrap(), expected);
        }
    }
}

#[test]
fn exact_append_cached_prompt_changed_prefix_reset_and_chunk_boundaries() {
    for kernel in [cpu::ProjectionKernel::Scalar, cpu::ProjectionKernel::Tiled] {
        let fixture = Fixture::tiny();
        let (c, w) = tiny_parts(&fixture);
        let model = Model::new(c, w).unwrap().with_projection_kernel(kernel);
        let mut session = Session::new(&model, options(32, 2, 0)).unwrap();
        session.set_tokens(&[1, 7, 13, 3]).unwrap();
        assert_eq!(session.evaluate_next_chunk().unwrap(), 2);
        assert_eq!(session.evaluated_len(), 2);
        assert_eq!(session.pending_len(), 2);
        // A caller may stop here without submitting another chunk, then resume.
        session.evaluate().unwrap();
        let last = session.logits().unwrap().to_vec();
        session.set_tokens(&[1, 7, 13, 3]).unwrap();
        assert_eq!(session.evaluated_len(), 4);
        assert_eq!(session.evaluate_next_chunk().unwrap(), 0);
        assert_eq!(session.logits().unwrap(), last);
        session.set_tokens(&[1, 7, 13, 3, 47, 11]).unwrap();
        assert_eq!(session.evaluated_len(), 4);
        session.discard_pending();
        assert_eq!(session.logits().unwrap(), last);
        session.set_tokens(&[1, 7, 13, 17]).unwrap();
        assert_eq!(session.evaluated_len(), 3);
        assert!(session.logits().is_none());
        session.evaluate().unwrap();
        close(
            session.logits().unwrap(),
            &fixture.f32("cases.future_changed.logits")[3 * 64..],
        );
        // Shortening retains all but the final shared token, whose logits need replay.
        session.set_tokens(&[1, 7]).unwrap();
        assert_eq!(session.evaluated_len(), 1);
        session.evaluate().unwrap();
        session.reset();
        assert!(session.history().is_empty());
        assert!(session.logits().is_none());
        assert_eq!(session.evaluate_next_chunk().unwrap(), 0);
        session.set_tokens(&[11]).unwrap();
        session.evaluate().unwrap();
        close(
            session.logits().unwrap(),
            &fixture.f32("cases.single.logits"),
        );
        session.set_tokens(&[]).unwrap();
        assert!(session.logits().is_none());
    }
}

#[test]
fn requests_and_memory_are_admitted_before_mutation() {
    for kernel in [cpu::ProjectionKernel::Scalar, cpu::ProjectionKernel::Tiled] {
        let fixture = Fixture::tiny();
        let (c, w) = tiny_parts(&fixture);
        let model = Model::new(c, w).unwrap().with_projection_kernel(kernel);
        for opts in [
            options(0, 1, 0),
            options(2049, 1, 0),
            options(33, 1, 0),
            options(4, 0, 0),
            options(4, 5, 0),
            options(4, 1, 29),
            options(4, 1, usize::MAX),
            Options {
                memory_limit_bytes: 0,
                ..options(4, 1, 0)
            },
            Options {
                memory_limit_bytes: usize::MAX,
                ..options(4, 1, 0)
            },
        ] {
            assert!(Session::new(&model, opts).is_err());
        }
        let mut session = Session::new(&model, options(4, 2, 0)).unwrap();
        let m = session.memory();
        assert_eq!(m.weights_bytes, model.allocated_weights_bytes().unwrap());
        assert_eq!(m.kv_bytes, 2 * 2 * 2 * 4 * 8 * 4);
        assert_eq!(m.history_bytes, 4 * 4);
        assert_eq!(m.logits_bytes, 2 * 64 * 4);
        assert_eq!(
            m.admitted_bytes,
            m.weights_bytes
                + m.kv_bytes
                + m.scratch_bytes
                + m.history_bytes
                + m.logits_bytes
                + m.metadata_bytes
                + m.metadata_allowance_bytes
        );
        assert!(matches!(
            Session::new(
                &model,
                Options {
                    memory_limit_bytes: m.admitted_bytes - 1,
                    ..options(4, 2, 0)
                }
            ),
            Err(Error::MemoryBudget { .. })
        ));
        assert!(
            Session::new(
                &model,
                Options {
                    memory_limit_bytes: m.admitted_bytes,
                    ..options(4, 2, 0)
                }
            )
            .is_ok()
        );
        session.append(&[1, 7]).unwrap();
        session.evaluate().unwrap();
        let last = session.logits().unwrap().to_vec();
        for invalid in [&[1, 64][..], &[1, 2, 3][..]] {
            assert!(session.append(invalid).is_err());
            assert_eq!(session.history(), [1, 7]);
            assert_eq!(session.evaluated_len(), 2);
            assert_eq!(session.logits().unwrap(), last);
        }
        assert!(session.set_tokens(&[64]).is_err());
        assert!(session.set_tokens(&[1, 2, 3, 4, 5]).is_err());
        assert_eq!(session.history(), [1, 7]);
        assert_eq!(session.logits().unwrap(), last);
        session.append(&[13, 3]).unwrap();
        session.evaluate().unwrap();
        assert_eq!(session.evaluated_len(), 4);
        assert!(session.append(&[1]).is_err());
        assert_eq!(session.memory(), m);
    }
}

fn small_parts(max_positions: usize) -> (Config, Weights) {
    let c = Config {
        vocab_size: 3,
        hidden_size: 2,
        intermediate_size: 2,
        num_layers: 2,
        query_heads: 1,
        kv_heads: 1,
        head_dim: 2,
        max_positions,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
    };
    let layers = (0..2)
        .map(|_| LayerWeights {
            input_norm: vec![1.0; 2],
            post_attention_norm: vec![1.0; 2],
            q: Projection {
                weight: vec![0.1, 0.05, 0.025, 0.1],
                bias: vec![0.0; 2],
            },
            k: Projection {
                weight: vec![0.1, 0.05, 0.025, 0.1],
                bias: vec![0.0; 2],
            },
            v: Projection {
                weight: vec![0.1, 0.05, 0.025, 0.1],
                bias: vec![0.0; 2],
            },
            o: vec![0.03; 4],
            gate: vec![0.1; 4],
            up: vec![0.1; 4],
            down: vec![0.01; 4],
        })
        .collect();
    let w = Weights {
        embedding: vec![1.0, 0.0, 0.0, 1.0, 0.1, -0.2],
        layers,
        final_norm: vec![1.0; 2],
    };
    (c, w)
}

#[test]
fn layer_or_final_projection_failure_never_commits_a_staged_chunk() {
    for kernel in [cpu::ProjectionKernel::Scalar, cpu::ProjectionKernel::Tiled] {
        for late_projection in [false, true] {
            let (c, mut w) = small_parts(32);
            w.embedding[..2].fill(0.0);
            w.embedding[2..4].fill(1.0);
            if late_projection {
                // Fail only in the tied output projection, after all cache writes.
                w.embedding[4..6].fill(f32::MAX);
            } else {
                // Layer zero's staged cache is written before layer one fails.
                w.layers[1].q.weight.fill(f32::MAX);
            }
            let model = Model::new(c, w).unwrap().with_projection_kernel(kernel);
            for all_rows in [false, true] {
                let mut session = Session::new(&model, options(8, 2, 0)).unwrap();
                session.append(&[0]).unwrap();
                session.evaluate().unwrap();
                let last = session.logits().unwrap().to_vec();
                session.append(&[0, 1]).unwrap();
                let mut output = [0.; 6];
                let result = if all_rows {
                    session.evaluate_next_chunk_with_logits(&mut output)
                } else {
                    session.evaluate_next_chunk()
                };
                assert!(matches!(
                    result,
                    Err(Error::Model(toklet::model::Error::Kernel(
                        cpu::Error::NonFinite { .. }
                    )))
                ));
                assert_eq!(session.evaluated_len(), 1);
                assert_eq!(session.history(), [0, 0, 1]);
                assert_eq!(session.pending_len(), 2);
                assert!(session.logits().is_none());
                assert!(session.evaluate_next_chunk().is_err());
                assert_eq!(session.evaluated_len(), 1);
                session.discard_pending();
                assert_eq!(session.logits().unwrap(), last);
                session.append(&[0]).unwrap();
                session.evaluate().unwrap();
                assert_eq!(session.evaluated_len(), 2);
                let mut scratch = model.scratch(2).unwrap();
                let mut reference = [0.0; 3];
                model
                    .forward(&[0, 0], 0, &mut scratch, &mut reference)
                    .unwrap();
                assert_eq!(session.logits().unwrap(), reference);
            }
        }
    }
}

#[test]
fn all_2048_tokens_and_nonzero_absolute_boundary_match_full_prefix() {
    for kernel in [cpu::ProjectionKernel::Scalar, cpu::ProjectionKernel::Tiled] {
        let (c, w) = small_parts(2048);
        let model = Model::new(c, w).unwrap().with_projection_kernel(kernel);
        let tokens: Vec<u32> = (0..2048).map(|i| (i % 3) as u32).collect();
        let mut session = Session::new(&model, options(2048, 31, 0)).unwrap();
        session.append(&tokens).unwrap();
        session.evaluate().unwrap();
        assert_eq!(session.evaluated_len(), 2048);
        assert_eq!(session.history(), tokens);
        let mut scratch = model.scratch(2048).unwrap();
        let mut reference = [0.0; 3];
        model
            .forward(&tokens, 0, &mut scratch, &mut reference)
            .unwrap();
        assert_eq!(session.logits().unwrap(), reference);
        assert!(session.append(&[1]).is_err());
        assert_eq!(session.evaluated_len(), 2048);
        assert_eq!(session.logits().unwrap(), reference);
        let mut offset = Session::new(&model, options(2, 1, 2046)).unwrap();
        offset.append(&[1, 2]).unwrap();
        offset.evaluate().unwrap();
        model
            .forward(&[1, 2], 2046, &mut scratch, &mut reference)
            .unwrap();
        assert_eq!(offset.logits().unwrap(), reference);
        assert!(Session::new(&model, options(2, 1, 2047)).is_err());
    }
}

#[test]
fn admitted_prefill_decode_and_reuse_allocate_no_buffers() {
    for kernel in [cpu::ProjectionKernel::Scalar, cpu::ProjectionKernel::Tiled] {
        let fixture = Fixture::tiny();
        let (c, w) = tiny_parts(&fixture);
        let model = Model::new(c, w).unwrap().with_projection_kernel(kernel);
        let mut session = Session::new(&model, options(32, 3, 0)).unwrap();
        let memory = session.memory();
        ALLOCATIONS.with(|count| count.set(0));
        TRACK.with(|track| track.set(true));
        session.set_tokens(&[1, 7, 13, 3]).unwrap();
        session.evaluate().unwrap();
        session.set_tokens(&[1, 7, 13, 3]).unwrap();
        session.evaluate().unwrap();
        for _ in 0..8 {
            let token = sampling::greedy(session.logits().unwrap()).unwrap();
            session.append(&[token]).unwrap();
            session.evaluate().unwrap();
        }
        session.append(&[1, 7]).unwrap();
        session.discard_pending();
        session.set_tokens(&[11]).unwrap();
        session.evaluate().unwrap();
        session.reset();
        session.append(&[7, 13, 3]).unwrap();
        session.evaluate().unwrap();
        TRACK.with(|track| track.set(false));
        assert_eq!(ALLOCATIONS.with(Cell::get), 0);
        assert_eq!(session.memory(), memory);
    }
}

#[test]
fn bounded_generation_matches_greedy_oracle_and_continues_pending_token() {
    for kernel in [cpu::ProjectionKernel::Scalar, cpu::ProjectionKernel::Tiled] {
        use toklet::generation::{self, Input, Request, StopReason};
        let f = Fixture::tiny();
        let (c, w) = tiny_parts(&f);
        let model = Model::new(c, w).unwrap().with_projection_kernel(kernel);
        let mut session = Session::new(&model, options(32, 3, 0)).unwrap();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let mut events = Vec::new();
        let report = generation::generate(
            &mut session,
            Request {
                input: Input::Prompt(&[1, 7, 13, 3]),
                max_new_tokens: 4,
                stop_tokens: &[],
            },
            &cancel,
            |e| {
                events.push(e);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            events.iter().map(|e| e.token_id).collect::<Vec<_>>(),
            [47; 4]
        );
        assert_eq!(report.reason, StopReason::OutputLimit);
        assert_eq!(report.sampled_tokens, 4);
        assert_eq!(report.emitted_tokens, 4);
        assert_eq!(report.prefill_tokens, 4);
        assert_eq!(report.history_tokens, 8);
        assert_eq!(report.evaluated_tokens, 7);
        assert_eq!(report.pending_tokens, 1);
        assert_eq!(report.first_token_seconds, Some(events[0].elapsed_seconds));
        assert!(report.total_seconds >= events[3].elapsed_seconds);
        let mut history = session.history().to_vec();
        let mut scratch = model.scratch(32).unwrap();
        let mut logits = [0.0; 64];
        let mut expected = Vec::new();
        for _ in 0..2 {
            model
                .forward(&history, 0, &mut scratch, &mut logits)
                .unwrap();
            let id = sampling::greedy(&logits).unwrap();
            history.push(id);
            expected.push(id);
        }
        events.clear();
        let report = generation::generate(
            &mut session,
            Request {
                input: Input::Continue,
                max_new_tokens: 2,
                stop_tokens: &[],
            },
            &cancel,
            |e| {
                events.push(e);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            events.iter().map(|e| e.token_id).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(report.reused_tokens, 7);
        assert_eq!(report.prefill_tokens, 1);
        assert_eq!(session.history(), history);
        assert_eq!(session.pending_len(), 1);
    }
}

#[test]
fn stop_ids_are_pending_and_hidden_and_zero_output_retains_cached_logits() {
    for kernel in [cpu::ProjectionKernel::Scalar, cpu::ProjectionKernel::Tiled] {
        use toklet::generation::{self, Input, Request, StopReason};
        let f = Fixture::tiny();
        let (c, w) = tiny_parts(&f);
        let model = Model::new(c, w).unwrap().with_projection_kernel(kernel);
        let mut s = Session::new(&model, options(32, 2, 0)).unwrap();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let report = generation::generate(
            &mut s,
            Request {
                input: Input::Prompt(&[1, 7, 13, 3]),
                max_new_tokens: 4,
                stop_tokens: &[47],
            },
            &cancel,
            |_| panic!("stop token must not be visible"),
        )
        .unwrap();
        assert_eq!(report.reason, StopReason::StopToken { token_id: 47 });
        assert_eq!(report.sampled_tokens, 1);
        assert_eq!(report.emitted_tokens, 0);
        assert_eq!(s.history(), [1, 7, 13, 3, 47]);
        assert_eq!(s.evaluated_len(), 4);
        let report = generation::generate(
            &mut s,
            Request {
                input: Input::Reset(&[1, 7, 13, 3]),
                max_new_tokens: 0,
                stop_tokens: &[],
            },
            &cancel,
            |_| panic!(),
        )
        .unwrap();
        assert_eq!(report.prefill_tokens, 4);
        assert_eq!(report.first_token_seconds, None);
        assert_eq!(s.pending_len(), 0);
        let report = generation::generate(
            &mut s,
            Request {
                input: Input::Prompt(&[1, 7, 13, 3]),
                max_new_tokens: 1,
                stop_tokens: &[],
            },
            &cancel,
            |e| {
                assert_eq!(e.token_id, 47);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(report.reused_tokens, 4);
        assert_eq!(report.prefill_tokens, 0);
    }
}

#[test]
fn generation_admission_rejects_all_invalid_inputs_before_state_changes() {
    for kernel in [cpu::ProjectionKernel::Scalar, cpu::ProjectionKernel::Tiled] {
        use toklet::generation::{self, Input, Request};
        let f = Fixture::tiny();
        let (c, w) = tiny_parts(&f);
        let model = Model::new(c, w).unwrap().with_projection_kernel(kernel);
        let mut s = Session::new(&model, options(8, 2, 0)).unwrap();
        s.append(&[1, 7]).unwrap();
        s.evaluate().unwrap();
        let logits = s.logits().unwrap().to_vec();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        for request in [
            Request {
                input: Input::Reset(&[]),
                max_new_tokens: 1,
                stop_tokens: &[],
            },
            Request {
                input: Input::Reset(&[64]),
                max_new_tokens: 1,
                stop_tokens: &[],
            },
            Request {
                input: Input::Prompt(&[1, 2, 3, 4]),
                max_new_tokens: 5,
                stop_tokens: &[],
            },
            Request {
                input: Input::Append(&[1, 2]),
                max_new_tokens: 5,
                stop_tokens: &[],
            },
            Request {
                input: Input::Continue,
                max_new_tokens: usize::MAX,
                stop_tokens: &[],
            },
            Request {
                input: Input::Continue,
                max_new_tokens: 1,
                stop_tokens: &[64],
            },
        ] {
            assert!(generation::generate(&mut s, request, &cancel, |_| panic!()).is_err());
            assert_eq!(s.history(), [1, 7]);
            assert_eq!(s.evaluated_len(), 2);
            assert_eq!(s.logits().unwrap(), logits);
        }
    }
}

#[test]
fn cancellation_and_output_failure_preserve_recoverable_prefixes() {
    for kernel in [cpu::ProjectionKernel::Scalar, cpu::ProjectionKernel::Tiled] {
        use std::sync::atomic::{AtomicBool, Ordering};
        use toklet::generation::{self, Input, Request, StopReason};
        let f = Fixture::tiny();
        let (c, w) = tiny_parts(&f);
        let model = Model::new(c, w).unwrap().with_projection_kernel(kernel);
        let mut s = Session::new(&model, options(32, 2, 0)).unwrap();
        let cancel = AtomicBool::new(true);
        let report = generation::generate(
            &mut s,
            Request {
                input: Input::Prompt(&[1, 7, 13, 3]),
                max_new_tokens: 4,
                stop_tokens: &[],
            },
            &cancel,
            |_| panic!(),
        )
        .unwrap();
        assert_eq!(report.reason, StopReason::Cancelled);
        assert!(s.history().is_empty());
        cancel.store(false, Ordering::Relaxed);
        let report = generation::generate(
            &mut s,
            Request {
                input: Input::Prompt(&[1, 7, 13, 3]),
                max_new_tokens: 4,
                stop_tokens: &[],
            },
            &cancel,
            |_| {
                cancel.store(true, Ordering::Relaxed);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(report.reason, StopReason::Cancelled);
        assert_eq!(report.emitted_tokens, 1);
        assert_eq!(s.evaluated_len(), 4);
        assert_eq!(s.pending_len(), 1);
        cancel.store(false, Ordering::Relaxed);
        let report = generation::generate(
            &mut s,
            Request {
                input: Input::Continue,
                max_new_tokens: 1,
                stop_tokens: &[],
            },
            &cancel,
            |e| {
                assert_eq!(e.token_id, 47);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(report.prefill_tokens, 1);
        assert_eq!(report.reused_tokens, 4);
        s.discard_pending();
        let before = s.history().to_vec();
        let len = s.evaluated_len();
        let result = generation::generate(
            &mut s,
            Request {
                input: Input::Continue,
                max_new_tokens: 1,
                stop_tokens: &[],
            },
            &cancel,
            |_| {
                Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "closed sink",
                ))
            },
        );
        assert!(matches!(result, Err(generation::Error::Output(_))));
        assert_eq!(s.evaluated_len(), len);
        assert_eq!(s.pending_len(), 1);
        s.discard_pending();
        assert_eq!(s.history(), before);
        assert!(s.logits().is_some());
    }
}

#[test]
fn generation_events_and_statistics_allocate_no_numerical_buffers() {
    for kernel in [cpu::ProjectionKernel::Scalar, cpu::ProjectionKernel::Tiled] {
        use toklet::generation::{self, Input, Request};
        let f = Fixture::tiny();
        let (c, w) = tiny_parts(&f);
        let model = Model::new(c, w).unwrap().with_projection_kernel(kernel);
        let mut s = Session::new(&model, options(32, 3, 0)).unwrap();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let mut ids = [0; 4];
        ALLOCATIONS.with(|count| count.set(0));
        TRACK.with(|track| track.set(true));
        let report = generation::generate(
            &mut s,
            Request {
                input: Input::Prompt(&[1, 7, 13, 3]),
                max_new_tokens: 4,
                stop_tokens: &[],
            },
            &cancel,
            |e| {
                ids[e.index] = e.token_id;
                Ok(())
            },
        )
        .unwrap();
        TRACK.with(|track| track.set(false));
        assert_eq!(ALLOCATIONS.with(Cell::get), 0);
        assert_eq!(ids, [47; 4]);
        assert_eq!(report.emitted_tokens, 4);
    }
}
