//! Physical head-stride transactions. Seeded source prefixes are not generated
//! model prefixes; attention is not encoded by these cache-only tests.
use super::*;
use crate::{metal::staged_gqa_fixture as f, model::gemma_reference::tests::Measurement};
use half::f16;
use objc2::rc::autoreleasepool;
use serde_json::{Value, json};
use std::io::Write;

fn config(c: &Value) -> OwnerConfig {
    OwnerConfig {
        kv_heads: f::number(&c["kv_heads"]),
        dim: 64,
        window: c["window"].as_u64().map(|v| v as usize),
    }
}
#[test]
fn gqa_head_intervals_bytes_and_six_layer_owners() {
    let m = f::fixture();
    for c in m["cases"].as_array().unwrap() {
        let p = Plan::new(f::number(&c["capacity"]), &[config(c)]).unwrap();
        let owner = p.owners[0];
        let n = f::number(&c["tokens"]);
        assert_eq!(owner.staged_len, owner.config.kv_heads * 32 * 64 + 7);
        assert_eq!(
            owner.old_len,
            owner.config.kv_heads * owner.retained * 64 + 7
        );
        assert_eq!(p.buffer_bytes, 4 * (owner.old_len + owner.staged_len) + 32);
        for r in c["intervals"]["active_stage_fill_and_validation"]
            .as_array()
            .unwrap()
        {
            assert_eq!(
                active_range(owner, f::number(&r["head"]), n).unwrap(),
                (f::number(&r["byte_offset"]), f::number(&r["bytes"]))
            );
        }
        let runs = window_rows::copy_runs(
            owner.retained,
            f::number(&c["committed"]),
            n,
            owner.config.window.is_some(),
        );
        let mut actual = Vec::new();
        for kind in ["k", "v"] {
            for head in 0..owner.config.kv_heads {
                for run in runs {
                    if run.rows > 0 {
                        actual.push(json!({"kind":kind,"head":head,"source_byte_offset":(head*32+run.source_row)*128,"destination_byte_offset":(head*owner.retained+run.slot)*128,"bytes":run.rows*128}));
                    }
                }
            }
        }
        assert_eq!(Value::Array(actual), c["intervals"]["phase_b_copy"]);
        assert!(active_range(owner, owner.config.kv_heads, n).is_err());
        assert!(active_range(owner, 0, 0).is_err());
        assert!(active_range(owner, 0, 33).is_err());
    }
    let c = OwnerConfig {
        kv_heads: 8,
        dim: 64,
        window: None,
    };
    let p = Plan::new(544, &[c; 6]).unwrap();
    assert_eq!(p.count, 6);
    assert_eq!(p.buffer_bytes, 6 * 4 * (8 * (544 + 32) * 64 + 14) + 32);
    assert_eq!(
        p.session_bytes().unwrap(),
        p.buffer_bytes + size_of::<Set<'_>>() + TRANSACTION_METADATA_BYTES
    );
    for (kv, dim) in [
        (0, 64),
        (1, 64),
        (2, 32),
        (4, 64),
        (8, 256),
        (usize::MAX, 64),
    ] {
        assert!(
            Plan::new(
                64,
                &[OwnerConfig {
                    kv_heads: kv,
                    dim,
                    window: None
                }]
            )
            .is_err()
        );
    }
    for dim in [32, 256] {
        assert!(
            Plan::new(
                544,
                &[OwnerConfig {
                    kv_heads: 1,
                    dim,
                    window: None
                }]
            )
            .is_ok()
        );
    }
}
struct Input {
    k: Buffer<u16>,
    v: Buffer<u16>,
    before: [String; 2],
}
fn input(gpu: &Gpu, k: &[u16], v: &[u16]) -> Input {
    let mut kb = Buffer::new(gpu, k.len()).unwrap();
    kb.with_mut(|x| x.copy_from_slice(k));
    let mut vb = Buffer::new(gpu, v.len()).unwrap();
    vb.with_mut(|x| x.copy_from_slice(v));
    Input {
        k: kb,
        v: vb,
        before: [f::hash16(k), f::hash16(v)],
    }
}
#[derive(Clone, PartialEq, Eq)]
struct Words {
    old: [Vec<u16>; 2],
    stage: [Vec<u16>; 2],
}
struct Capture {
    words: Vec<Words>,
    status: Vec<u32>,
    offset: usize,
    generation: u64,
    poisoned: bool,
    ids: Vec<[usize; 4]>,
}
fn record(set: &Set<'_>, label: &str, inputs: &[Input], full_old: bool) -> Capture {
    // Refuse all mapped access while any owned command remains in flight.
    set.test_safe_read().unwrap();
    let words: Vec<_> = set.owners[..set.operation.plan.count]
        .iter()
        .map(|o| {
            let o = o.as_ref().unwrap();
            Words {
                old: [
                    o.old_k.with_slice(|v| v.to_vec()),
                    o.old_v.with_slice(|v| v.to_vec()),
                ],
                stage: [
                    o.new_k.with_slice(|v| v.to_vec()),
                    o.new_v.with_slice(|v| v.to_vec()),
                ],
            }
        })
        .collect();
    let ids: Vec<_> = set.owners[..set.operation.plan.count]
        .iter()
        .map(|o| {
            let o = o.as_ref().unwrap();
            [&o.old_k, &o.old_v, &o.new_k, &o.new_v].map(|b| b.raw.contents().as_ptr() as usize)
        })
        .collect();
    let status = set.status.with_slice(|v| v.to_vec());
    let failure_stage = status[0] != 0
        || set
            .command
            .as_ref()
            .is_some_and(|c| terminal(c) && !completed(c));
    let owners:Vec<_>=words.iter().zip(&set.operation.plan.owners[..set.operation.plan.count]).zip(&ids).map(|((w,p),id)|json!({"kv_heads":p.config.kv_heads,"retained":p.retained,"dim":p.config.dim,"old_lengths":[w.old[0].len(),w.old[1].len()],"stage_lengths":[w.stage[0].len(),w.stage[1].len()],"old_full_u16le_hex":if full_old{Some(w.old.each_ref().map(|v|f::hex16(v)))}else{None},"old_full_sha256":w.old.each_ref().map(|v|f::hash16(v)),"stage_full_sha256":w.stage.each_ref().map(|v|f::hash16(v)),"failure_stage_full_u16le_hex":if failure_stage{Some(w.stage.each_ref().map(|v|f::hex16(v)))}else{None},"buffer_addresses":id})).collect();
    let hashes:Vec<_>=inputs.iter().map(|i|json!({"before":i.before,"after":[i.k.with_slice(f::hash16),i.v.with_slice(f::hash16)]})).collect();
    let command=set.command.as_ref().map(|c|json!({"status":c.status().0,"error":c.error().map(|e|e.to_string()),"terminal":terminal(c),"completed":completed(c)}));
    println!(
        "GQA_CACHE_RAW {}",
        json!({"label":label,"owners":owners,"status_words":status,"state":format!("{:?}",set.state),"offset":set.offset,"generation":set.generation,"epoch":set.epoch,"identity":set.identity,"poisoned":set.is_poisoned(),"command":command,"fixture_input_hashes":hashes,"buffer_bytes":set.operation.plan.buffer_bytes,"session_bytes":set.operation.plan.session_bytes().unwrap(),"raw_scope":"full old words only where present; other physical words validated by actual full hashes against retained known input/state"})
    );
    std::io::stdout().flush().unwrap();
    Capture {
        words,
        status,
        offset: set.offset,
        generation: set.generation,
        poisoned: set.is_poisoned(),
        ids,
    }
}
fn validate(set: &Set<'_>, s: &Capture, inputs: &[Input]) {
    assert!(s.status[1..].iter().all(|&v| v == STATUS_PAD));
    if let Some(c) = &set.command {
        assert!(c.status() == MTLCommandBufferStatus::NotEnqueued || completed(c));
    }
    for (w, p) in s
        .words
        .iter()
        .zip(&set.operation.plan.owners[..set.operation.plan.count])
    {
        for old in &w.old {
            assert!(
                old[p.config.kv_heads * p.retained * p.config.dim..]
                    .iter()
                    .all(|&v| v == PAD)
            );
        }
        for stage in &w.stage {
            assert!(
                stage[p.config.kv_heads * CHUNK * p.config.dim..]
                    .iter()
                    .all(|&v| v == PAD)
            );
        }
    }
    for i in inputs {
        assert_eq!(
            [i.k.with_slice(f::hash16), i.v.with_slice(f::hash16)],
            i.before
        );
    }
}
fn same_old(a: &Capture, b: &Capture) {
    assert_eq!(a.offset, b.offset);
    assert_eq!(a.generation, b.generation);
    assert_eq!(a.ids, b.ids);
    for (a, b) in a.words.iter().zip(&b.words) {
        assert_eq!(a.old, b.old);
    }
}
// Source sources are test-owned and remain live through terminal completion.
// Every transfer uses a real per-head blit after begin's active NaN stamp.
fn produce(set: &Set<'_>, batch: &Batch, inputs: &[Input], fault: Option<usize>) {
    assert_eq!(inputs.len(), batch.len);
    for (i, source) in inputs.iter().enumerate() {
        let view = set.view(&batch.frames()[i]).unwrap();
        let encoder = view.command.blitCommandEncoder().unwrap();
        for head in 0..view.config.kv_heads {
            let (offset, bytes) =
                active_range(set.operation.plan.owners[i], head, view.tokens).unwrap();
            for (kind, (src, dst)) in [(&source.k, &view.new_k), (&source.v, &view.new_v)]
                .into_iter()
                .enumerate()
            {
                let last = i + 1 == inputs.len() && head + 1 == view.config.kv_heads && kind == 1;
                if last && fault == Some(0) {
                    continue;
                }
                let count = if last && fault == Some(1) {
                    bytes - 4
                } else {
                    bytes
                };
                assert!(offset + count <= src.raw.length() && offset + count <= dst.raw.length());
                // SAFETY: checked aligned per-head slices of distinct initialized
                // same-device buffers, retained until Set waits for completion.
                unsafe {
                    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &src.raw, offset, &dst.raw, offset, count,
                    );
                }
            }
        }
        encoder.endEncoding();
    }
}
fn measured<T>(label: &str, f: impl FnOnce() -> Result<T>) -> (Result<T>, usize) {
    let m = Measurement::start();
    let r = f();
    let n = m.finish();
    println!(
        "GQA_CACHE_ALLOCATION {}",
        json!({"label":label,"rust_allocations":n,"result_ok":r.is_ok(),"includes_command_creation_and_waits":true,"release_required_zero":!cfg!(debug_assertions)})
    );
    std::io::stdout().flush().unwrap();
    (r, n)
}
fn zero_allocations(counts: [usize; 3]) {
    if !cfg!(debug_assertions) {
        assert_eq!(counts, [0; 3]);
    }
}
fn begin_measured(
    set: &mut Set<'_>,
    gpu: &Gpu,
    n: usize,
    inputs: &[Input],
    label: &str,
) -> (Batch, usize) {
    let (result, count) = measured(label, || set.begin(gpu, n));
    if result.is_err() {
        record(set, "begin_failure", inputs, false);
    }
    (result.unwrap(), count)
}
fn same_stage_overwrite(before: &Capture, after: &Capture, inputs: &[Input], n: usize) {
    for ((a, b), source) in before.words.iter().zip(&after.words).zip(inputs) {
        for (kind, src) in [&source.k, &source.v].into_iter().enumerate() {
            let mut expected = a.stage[kind].clone();
            src.with_slice(|v| {
                for h in 0..8 {
                    expected[h * 32 * 64..(h * 32 + n) * 64]
                        .copy_from_slice(&v[h * 32 * 64..(h * 32 + n) * 64]);
                }
            });
            assert_eq!(b.stage[kind], expected);
        }
    }
}
fn operation(gpu: &Gpu, capacity: usize, configs: &[OwnerConfig]) -> Operation {
    let amount = PIPELINE_ALLOWANCE + size_of::<Operation>();
    assert!(
        matches!(Operation::new(gpu,Plan::new(capacity,configs).unwrap(),amount-1),Err(Error::Budget{required,limit})if required==amount&&limit==amount-1)
    );
    Operation::new(gpu, Plan::new(capacity, configs).unwrap(), amount).unwrap()
}
fn new_set<'a>(gpu: &Gpu, op: &'a Operation) -> Set<'a> {
    let amount = op.plan.session_bytes().unwrap();
    assert!(
        matches!(Set::new(gpu,op,amount-1),Err(Error::Budget{required,limit})if required==amount&&limit==amount-1)
    );
    let set = Set::new(gpu, op, amount).unwrap();
    let bytes = set.owners[..op.plan.count]
        .iter()
        .map(|o| {
            let o = o.as_ref().unwrap();
            [&o.old_k, &o.old_v, &o.new_k, &o.new_v]
                .iter()
                .map(|b| b.raw.length())
                .sum::<usize>()
        })
        .sum::<usize>()
        + set.status.raw.length();
    assert_eq!(bytes, op.plan.buffer_bytes);
    set
}
#[test]
#[ignore = "explicit Apple GPU GQA per-head source copy and complete raw physical state"]
fn gpu_source_per_head_copies() {
    autoreleasepool(|_| {
        let m = f::fixture();
        let gpu = Gpu::new().unwrap();
        let mut words = 0;
        for first in m["cases"].as_array().unwrap() {
            if first["name"] == "real_513" {
                continue;
            }
            let op = operation(&gpu, f::number(&first["capacity"]), &[config(first)]);
            let mut set = new_set(&gpu, &op);
            let cases = if first["name"] == "real_512" {
                vec![first, &m["cases"][8]]
            } else {
                vec![first]
            };
            for (step, c) in cases.into_iter().enumerate() {
                let old = [
                    f::words(&c["inputs"]["old_k"]),
                    f::words(&c["inputs"]["old_v"]),
                ];
                let staged = [
                    f::words(&c["inputs"]["new_k"]),
                    f::words(&c["inputs"]["new_v"]),
                ];
                let sources = [input(&gpu, &staged[0], &staged[1])];
                if step == 0 {
                    let owner = set.owners[0].as_mut().unwrap();
                    owner.old_k.with_mut(|v| v.copy_from_slice(&old[0]));
                    owner.old_v.with_mut(|v| v.copy_from_slice(&old[1]));
                    set.offset = f::number(&c["committed"]);
                } else {
                    assert_eq!(c["name"], "real_513");
                    assert_eq!(set.offset, f::number(&c["committed"]));
                    let owner = set.owners[0].as_ref().unwrap();
                    assert_eq!(owner.old_k.with_slice(|v| v.to_vec()), old[0]);
                    assert_eq!(owner.old_v.with_slice(|v| v.to_vec()), old[1]);
                }
                let before = record(
                    &set,
                    &format!("{}/seeded", c["name"].as_str().unwrap()),
                    &sources,
                    false,
                );
                validate(&set, &before, &sources);
                let (batch, bn) = begin_measured(
                    &mut set,
                    &gpu,
                    f::number(&c["tokens"]),
                    &sources,
                    "source_begin",
                );
                produce(&set, &batch, &sources, None);
                let (ready, fn_count) =
                    measured("source_finish", || set.finish_math(batch.frames()));
                let math = record(
                    &set,
                    &format!("{}/math", c["name"].as_str().unwrap()),
                    &sources,
                    false,
                );
                let ready = ready.unwrap();
                validate(&set, &math, &sources);
                same_old(&before, &math);
                let (info, cn) = measured("source_commit", || set.commit(&gpu, &ready));
                let after = record(&set, c["name"].as_str().unwrap(), &sources, true);
                let info = info.unwrap();
                validate(&set, &after, &sources);
                zero_allocations([bn, fn_count, cn]);
                assert_eq!(after.status[0], 0);
                assert_eq!(after.offset, before.offset + f::number(&c["tokens"]));
                assert_eq!(after.generation, before.generation + 1);
                assert_eq!(after.ids, before.ids);
                assert_eq!(info.owners, 1);
                assert_eq!(info.start, before.offset);
                assert_eq!(info.end, after.offset);
                assert_eq!(info.generation, after.generation);
                // These consecutive cases both write one row per head. No CPU
                // cache copy, reseeding or Set replacement between 512 and 513.
                assert_eq!(after.words[0].stage, staged);
                for (kind, key) in ["expected_after_k", "expected_after_v"]
                    .into_iter()
                    .enumerate()
                {
                    let expected = f::words(&c["inputs"][key]);
                    assert_eq!(after.words[0].old[kind], expected);
                    words += expected.len();
                }
            }
        }
        println!(
            "GQA_CACHE_SOURCE_RESULT {}",
            json!({"cases":11,"phase_a_commands":11,"phase_b_commands":11,"required_zero_boundaries":33,"physical_after_words":words,"seeded_prefixes":true,"resident_512_to_513_same_set":true,"real_model_executed":false})
        );
    });
}
fn patterned(gpu: &Gpu, offset: usize, n: usize) -> Vec<Input> {
    (0..6)
        .map(|owner| {
            let arrays: [Vec<u16>; 2] = std::array::from_fn(|kind| {
                let mut words = vec![PAD; 8 * 32 * 64 + 7];
                for head in 0..8 {
                    for t in 0..n {
                        for j in 0..64 {
                            let v =
                                ((owner * 29 + head * 17 + (offset + t) * 11 + j * 3 + kind * 7)
                                    % 251) as f32
                                    - 125.;
                            words[(head * 32 + t) * 64 + j] = f16::from_f32(v).to_bits();
                        }
                    }
                }
                words
            });
            input(gpu, &arrays[0], &arrays[1])
        })
        .collect()
}
fn expected_copy(
    before: &Capture,
    inputs: &[Input],
    offset: usize,
    n: usize,
) -> Vec<[Vec<u16>; 2]> {
    before
        .words
        .iter()
        .zip(inputs)
        .map(|(old, src)| {
            let mut result = old.old.clone();
            for (kind, b) in [&src.k, &src.v].into_iter().enumerate() {
                b.with_slice(|v| {
                    for h in 0..8 {
                        result[kind][(h * 64 + offset) * 64..(h * 64 + offset + n) * 64]
                            .copy_from_slice(&v[h * 32 * 64..h * 32 * 64 + n * 64]);
                    }
                });
            }
            result
        })
        .collect()
}
#[test]
#[ignore = "explicit Apple GPU six GQA owners, last-head validation, poisoned partial copy and replay"]
fn gpu_six_owners_last_head_failures_and_replay() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let op = operation(
            &gpu,
            64,
            &[OwnerConfig {
                kv_heads: 8,
                dim: 64,
                window: None,
            }; 6],
        );
        let mut set = new_set(&gpu, &op);
        assert_eq!(set.operation.plan.count, 6);
        let initial_inputs = patterned(&gpu, 0, 3);
        let initial = record(&set, "initial", &initial_inputs, false);
        validate(&set, &initial, &initial_inputs);
        for o in &initial.words {
            assert!(o.old.iter().flatten().all(|&w| w == PAD));
        }
        // Missing last head, partial last row, and late NaN/Inf all fail Phase A
        // without any attention encoder. Every physical old owner stays untouched.
        for fault in 0..4 {
            let mut inputs = patterned(&gpu, 0, 3);
            if fault >= 2 {
                inputs[5].v.with_mut(|v| {
                    v[(7 * 32 + 2) * 64 + 63] = if fault == 2 { 0x7e01 } else { 0xfc00 }
                });
                inputs[5].before[1] = inputs[5].v.with_slice(f::hash16);
            }
            let batch = set.begin(&gpu, 3).unwrap();
            produce(&set, &batch, &inputs, (fault < 2).then_some(fault));
            let outcome = set.finish_math(batch.frames());
            let after = record(
                &set,
                &format!("invalid_last_owner_head_{fault}"),
                &inputs,
                false,
            );
            validate(&set, &after, &inputs);
            assert!(matches!(outcome, Err(Error::Nonfinite)));
            assert_ne!(after.status[0], 0);
            same_old(&initial, &after);
            assert!(!after.poisoned);
            for (owner, w) in after.words.iter().enumerate() {
                for kind in 0..2 {
                    let mut expected = if kind == 0 {
                        inputs[owner].k.with_slice(|v| v.to_vec())
                    } else {
                        inputs[owner].v.with_slice(|v| v.to_vec())
                    };
                    if owner == 5 && kind == 1 && fault < 2 {
                        let end = (7 * 32 + 3) * 64;
                        let start = if fault == 0 { 7 * 32 * 64 } else { end - 2 };
                        expected[start..end].fill(0x7e7e);
                    }
                    assert_eq!(w.stage[kind], expected);
                }
            }
        }
        // No command is submitted by bad frame metadata. Correct metadata remains
        // usable until explicit discard; this is not a simulated hardware fault.
        for fault in 0..5 {
            let inputs = patterned(&gpu, 0, 3);
            let batch = set.begin(&gpu, 3).unwrap();
            let mut frames = batch.frames;
            let len = if fault == 0 { batch.len - 1 } else { batch.len };
            match fault {
                0 => {}
                1 => frames[5] = frames[4],
                2 => frames.swap(0, 5),
                3 => frames[5].set += 1,
                4 => frames[5].epoch -= 1,
                _ => unreachable!(),
            };
            let outcome = set.finish_math(&frames[..len]);
            let after = record(&set, &format!("metadata_reject_{fault}"), &inputs, false);
            assert!(outcome.is_err());
            assert_eq!(
                set.command.as_ref().unwrap().status(),
                MTLCommandBufferStatus::NotEnqueued
            );
            same_old(&initial, &after);
            set.discard().unwrap();
        }
        let inputs = patterned(&gpu, 0, 3);
        let (batch, bn) = begin_measured(&mut set, &gpu, 3, &inputs, "six_begin");
        produce(&set, &batch, &inputs, None);
        let (ready, fn_count) = measured("six_finish", || set.finish_math(batch.frames()));
        let math = record(&set, "six_math", &inputs, false);
        let ready = ready.unwrap();
        validate(&set, &math, &inputs);
        same_old(&initial, &math);
        let (info, cn) = measured("six_commit", || set.commit(&gpu, &ready));
        let good = record(&set, "six_committed", &inputs, true);
        let info = info.unwrap();
        validate(&set, &good, &inputs);
        zero_allocations([bn, fn_count, cn]);
        assert_eq!(info.owners, 6);
        for (w, source) in good.words.iter().zip(&inputs) {
            assert_eq!(
                w.stage,
                [
                    source.k.with_slice(|v| v.to_vec()),
                    source.v.with_slice(|v| v.to_vec())
                ]
            );
        }
        assert_eq!(good.status[0], 0);
        assert_eq!(good.offset, 3);
        assert_eq!(good.generation, 1);
        assert_eq!(good.ids, initial.ids);
        for (w, expected) in good
            .words
            .iter()
            .zip(expected_copy(&initial, &inputs, 0, 3))
        {
            assert_eq!(w.old, expected);
        }
        let next = patterned(&gpu, 3, 2);
        let batch = set.begin(&gpu, 2).unwrap();
        produce(&set, &batch, &next, None);
        let ready = set.finish_math(batch.frames());
        let math = record(&set, "partial_copy_math", &next, false);
        let ready = ready.unwrap();
        validate(&set, &math, &next);
        same_old(&good, &math);
        same_stage_overwrite(&good, &math, &next, 2);
        let outcome = set.interrupt_copy_after_first_run(&gpu, &ready);
        let partial = record(
            &set,
            "first_owner_head0_k_real_copy_then_protocol_failure",
            &next,
            true,
        );
        validate(&set, &partial, &next);
        assert!(matches!(outcome, Err(Error::ReplayRequired)));
        assert!(partial.poisoned);
        assert_eq!(partial.offset, 3);
        assert_eq!(partial.generation, 1);
        assert_eq!(partial.ids, good.ids);
        for (i, (before, after)) in good.words.iter().zip(&partial.words).enumerate() {
            let mut expected = before.old.clone();
            if i == 0 {
                next[0]
                    .k
                    .with_slice(|v| expected[0][3 * 64..5 * 64].copy_from_slice(&v[..2 * 64]));
                assert_ne!(expected, before.old);
            }
            assert_eq!(after.old, expected);
        }
        assert!(matches!(set.begin(&gpu, 1), Err(Error::ReplayRequired)));
        assert!(matches!(
            set.view(&batch.frames()[0]),
            Err(Error::ReplayRequired)
        ));
        assert!(matches!(set.discard(), Err(Error::ReplayRequired)));
        set.reset().unwrap();
        let reset = record(
            &set,
            "reset_logical_prefix_zero_physical_tail_retained",
            &next,
            false,
        );
        validate(&set, &reset, &next);
        assert_eq!(reset.offset, 0);
        assert!(!reset.poisoned);
        for (a, b) in reset.words.iter().zip(&partial.words) {
            assert_eq!(a.old, b.old);
        }
        for n in [1, 2] {
            let offset = set.offset;
            let inputs = patterned(&gpu, offset, n);
            let before = record(&set, "replay_before", &inputs, false);
            let expected = expected_copy(&before, &inputs, offset, n);
            let (batch, bn) = begin_measured(&mut set, &gpu, n, &inputs, "replay_begin");
            produce(&set, &batch, &inputs, None);
            let (ready, fn_count) = measured("replay_finish", || set.finish_math(batch.frames()));
            let math = record(&set, "replay_math", &inputs, false);
            let ready = ready.unwrap();
            validate(&set, &math, &inputs);
            same_old(&before, &math);
            same_stage_overwrite(&before, &math, &inputs, n);
            let (info, cn) = measured("replay_commit", || set.commit(&gpu, &ready));
            let after = record(&set, "replay_after", &inputs, n == 2);
            info.unwrap();
            validate(&set, &after, &inputs);
            zero_allocations([bn, fn_count, cn]);
            assert_eq!(after.status[0], 0);
            for (w, expected) in after.words.iter().zip(expected) {
                assert_eq!(w.old, expected);
            }
            assert_eq!(after.ids, initial.ids);
        }
        let final_state = record(&set, "replay_final_hash_only", &[], false);
        assert_eq!(final_state.offset, 3);
        for (a, b) in final_state.words.iter().zip(&good.words) {
            for kind in 0..2 {
                for h in 0..8 {
                    assert_eq!(
                        &a.old[kind][h * 64 * 64..(h * 64 + 3) * 64],
                        &b.old[kind][h * 64 * 64..(h * 64 + 3) * 64]
                    );
                }
            }
        }
        println!(
            "GQA_CACHE_FAILURE_RESULT {}",
            json!({"owners":6,"heads_per_owner":8,"phase_a_commands":8,"phase_b_commands":4,"numerical_last_head_rejections":4,"unsubmitted_metadata_rejections":5,"required_zero_boundaries":9,"real_partial_copy":true,"forced_protocol_failure_not_hardware_error":true,"replay_chunks":[1,2],"replay_end":3,"inactive_tail_cleared":false})
        );
    });
}
