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
    let fixture = Fixture::tiny();
    let (c, w) = tiny_parts(&fixture);
    let model = Model::new(c, w).unwrap();
    let mut scratch = model.scratch(32).unwrap();
    let mut reference = [0.0; 64];
    for case in ["prompt", "future_changed", "offset", "single", "capacity"] {
        let tokens = fixture.u32(&format!("cases.{case}.tokens"));
        let expected = fixture.f32(&format!("cases.{case}.logits"));
        let offset = fixture.manifest["cases"][case]["position_offset"]
            .as_u64()
            .unwrap() as usize;
        for chunk in [tokens.len(), 3, 1] {
            let mut session = Session::new(&model, options(32 - offset, chunk, offset)).unwrap();
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

#[test]
fn sampled_history_is_pending_until_evaluated_and_greedy_trace_matches() {
    let fixture = Fixture::tiny();
    let (c, w) = tiny_parts(&fixture);
    let model = Model::new(c, w).unwrap();
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

#[test]
fn exact_append_cached_prompt_changed_prefix_reset_and_chunk_boundaries() {
    let fixture = Fixture::tiny();
    let (c, w) = tiny_parts(&fixture);
    let model = Model::new(c, w).unwrap();
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
    assert_eq!(session.evaluated_len(), 0);
    assert!(session.logits().is_none());
    session.evaluate().unwrap();
    close(
        session.logits().unwrap(),
        &fixture.f32("cases.future_changed.logits")[3 * 64..],
    );
    // Shortening also resets; longest-prefix truncation is deliberately deferred.
    session.set_tokens(&[1, 7]).unwrap();
    assert_eq!(session.evaluated_len(), 0);
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

#[test]
fn requests_and_memory_are_admitted_before_mutation() {
    let fixture = Fixture::tiny();
    let (c, w) = tiny_parts(&fixture);
    let model = Model::new(c, w).unwrap();
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
        let model = Model::new(c, w).unwrap();
        let mut session = Session::new(&model, options(8, 2, 0)).unwrap();
        session.append(&[0]).unwrap();
        session.evaluate().unwrap();
        let last = session.logits().unwrap().to_vec();
        session.append(&[0, 1]).unwrap();
        assert!(matches!(
            session.evaluate_next_chunk(),
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

#[test]
fn all_2048_tokens_and_nonzero_absolute_boundary_match_full_prefix() {
    let (c, w) = small_parts(2048);
    let model = Model::new(c, w).unwrap();
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

#[test]
fn admitted_prefill_decode_and_reuse_allocate_no_buffers() {
    let fixture = Fixture::tiny();
    let (c, w) = tiny_parts(&fixture);
    let model = Model::new(c, w).unwrap();
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
