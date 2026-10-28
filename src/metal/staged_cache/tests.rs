use super::*;
use crate::model::gemma_reference::tests::Measurement;
use objc2::rc::autoreleasepool;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::Write;
const RAW: &[u8] = include_bytes!("../../../tests/fixtures/gemma3-staged-attention-v1/arrays.bin");
fn number(v: &Value) -> usize {
    usize::try_from(v.as_u64().unwrap()).unwrap()
}
fn fixture() -> Value {
    let bytes = include_bytes!("../../../tests/fixtures/gemma3-staged-attention-v1/manifest.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "90656e1a724b1bfb8d3c14fd742166348d5abf64dedbaf3f8d83645e5e5ca7cb"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(RAW)),
        "12339d584e5d67602d386b75fcdca2dcfe62087ef1f4ac13eb0a5109525c150e"
    );
    let m: Value = serde_json::from_slice(bytes).unwrap();
    assert_eq!(m["format"], "toklet-gemma3-staged-attention-v1");
    assert_eq!(m["stored_arrays"], 104);
    m
}
fn descriptor(d: &Value) -> &[u8] {
    let start = number(&d["offset"]);
    let bytes = &RAW[start..start + number(&d["bytes"])];
    assert_eq!(d["file"], "arrays.bin");
    assert_eq!(d["sha256"], format!("{:x}", Sha256::digest(bytes)));
    bytes
}
fn words(d: &Value) -> Vec<u16> {
    assert_eq!(d["dtype"], "u16le");
    descriptor(d)
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn hash(v: &[u16]) -> String {
    let mut h = Sha256::new();
    for word in v {
        h.update(word.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
#[test]
fn bounded_plans_bytes_and_copy_ranges() {
    let local = OwnerConfig {
        dim: 256,
        window: Some(512),
    };
    let full = OwnerConfig {
        dim: 256,
        window: None,
    };
    assert_eq!(Plan::new(544, &[local]).unwrap().buffer_bytes, 557112 + 32);
    assert_eq!(Plan::new(544, &[full]).unwrap().buffer_bytes, 589880 + 32);
    assert_eq!(
        Plan::new(544, &[full; 32]).unwrap().buffer_bytes,
        18876160 + 32
    );
    let mut configs = [local; 18];
    configs[15..].fill(full);
    assert_eq!(
        Plan::new(544, &configs).unwrap().buffer_bytes,
        10126320 + 32
    );
    for (capacity, configs) in [
        (0, vec![full]),
        (545, vec![full]),
        (12, vec![]),
        (12, vec![full; 33]),
        (
            12,
            vec![OwnerConfig {
                dim: 64,
                window: None,
            }],
        ),
        (
            12,
            vec![OwnerConfig {
                dim: 32,
                window: Some(0),
            }],
        ),
        (
            12,
            vec![OwnerConfig {
                dim: 32,
                window: Some(13),
            }],
        ),
    ] {
        assert!(Plan::new(capacity, &configs).is_err());
    }
    for capacity in [1, 12, 544] {
        for dim in [32, 256] {
            for window in [None, Some(1), Some(capacity)] {
                let p = Plan::new(capacity, &[OwnerConfig { dim, window }]).unwrap();
                assert_eq!(
                    p.session_bytes().unwrap(),
                    p.buffer_bytes + size_of::<Set<'_>>() + TRANSACTION_METADATA_BYTES
                );
                let retained = window.unwrap_or(capacity);
                for offset in [0, capacity / 2, capacity - 1] {
                    let tokens = (capacity - offset).min(CHUNK);
                    let runs = window_rows::copy_runs(retained, offset, tokens, window.is_some());
                    let actual: Vec<_> = runs
                        .into_iter()
                        .flat_map(|r| (0..r.rows).map(move |i| (r.source_row + i, r.slot + i)))
                        .collect();
                    let skip = window.map_or(0, |w| tokens.saturating_sub(w));
                    let expected: Vec<_> = (skip..tokens)
                        .map(|r| (r, window.map_or(offset + r, |w| (offset + r) % w)))
                        .collect();
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}
#[test]
fn opaque_frame_identity_controls_and_frozen_copy_inputs() {
    let mut batch = Batch {
        frames: [EMPTY_FRAME; MAX],
        len: 2,
    };
    for (i, frame) in batch.frames.iter_mut().take(2).enumerate() {
        *frame = Frame {
            set: 7,
            owner: i,
            generation: 3,
            epoch: 9,
            offset: 5,
            tokens: 7,
        };
    }
    assert!(check_frames(batch.frames(), &batch).is_ok());
    assert!(check_frames(&batch.frames()[..1], &batch).is_err());
    assert!(check_frames(&[], &batch).is_err());
    for field in 0..8 {
        let mut wrong = batch.frames;
        match field {
            0 => wrong.swap(0, 1),
            1 => wrong[1] = wrong[0],
            2 => wrong[0].set += 1,
            3 => wrong[0].generation += 1,
            4 => wrong[0].epoch += 1,
            5 => wrong[0].offset += 1,
            6 => wrong[0].tokens += 1,
            7 => wrong[0].owner = MAX,
            _ => unreachable!(),
        }
        assert!(check_frames(&wrong[..2], &batch).is_err());
    }
    let m = fixture();
    let mut extents = Vec::new();
    for c in m["cases"].as_array().unwrap() {
        for d in c["inputs"].as_object().unwrap().values().chain(
            c["arrays"]
                .as_object()
                .unwrap()
                .values()
                .flat_map(|a| [&a["main"], &a["wide"]]),
        ) {
            descriptor(d);
            extents.push((number(&d["offset"]), number(&d["bytes"])));
        }
        let dim = number(&c["dim"]);
        for kind in ["k", "v"] {
            let mut before = words(&c["inputs"][format!("old_{kind}")]);
            let new = words(&c["inputs"][format!("new_{kind}")]);
            for run in c["copy_runs"].as_array().unwrap() {
                let src = number(&run["source_row"]) * dim;
                let dst = number(&run["destination_row"]) * dim;
                let len = number(&run["rows"]) * dim;
                before[dst..dst + len].copy_from_slice(&new[src..src + len]);
            }
            assert_eq!(
                before,
                words(&c["inputs"][format!("expected_after_{kind}")])
            );
        }
    }
    extents.sort_unstable();
    assert_eq!(extents.len(), 104);
    let mut end = 0;
    for (start, count) in extents {
        assert_eq!(start, end);
        end += count;
    }
    assert_eq!(end, RAW.len());
}
struct Input {
    k: Buffer<u16>,
    v: Buffer<u16>,
    before: [String; 2],
}
fn input(gpu: &Gpu, k: &[u16], v: &[u16]) -> Input {
    let mut kb = Buffer::new(gpu, k.len()).unwrap();
    kb.with_mut(|b| b.copy_from_slice(k));
    let mut vb = Buffer::new(gpu, v.len()).unwrap();
    vb.with_mut(|b| b.copy_from_slice(v));
    Input {
        k: kb,
        v: vb,
        before: [hash(k), hash(v)],
    }
}
fn inputs_for(gpu: &Gpu, cases: &[Value]) -> Vec<Input> {
    cases
        .iter()
        .map(|c| {
            input(
                gpu,
                &words(&c["inputs"]["new_k"]),
                &words(&c["inputs"]["new_v"]),
            )
        })
        .collect()
}
fn patterned(gpu: &Gpu, count: usize, dim: usize, pattern: &[u16]) -> Vec<Input> {
    (0..count)
        .map(|owner| {
            let mut v = vec![PAD; CHUNK * dim + GUARD];
            for (i, w) in v[..dim].iter_mut().enumerate() {
                *w = pattern[(i + owner) % pattern.len()];
            }
            input(gpu, &v, &v)
        })
        .collect()
}
// Fixture inputs are test-owned. This real GPU blit follows begin's NaN fill,
// never a CPU upload that a queued fill would erase. No attention is encoded.
fn write(set: &Set<'_>, batch: &Batch, inputs: &[Input], owners: usize) {
    assert_eq!(inputs.len(), batch.len);
    for (i, source) in inputs.iter().enumerate().take(owners) {
        let view = set.view(&batch.frames()[i]).unwrap();
        assert_eq!(view.capacity, set.operation.plan.capacity);
        assert_eq!(view.offset, set.offset());
        assert_eq!(view.old_k.len, view.old_v.len);
        assert_eq!(view.status.len, 1 + GUARD);
        let count = view.tokens * view.config.dim * 2;
        assert!(count <= source.k.raw.length() && count <= source.v.raw.length());
        let e = view.command.blitCommandEncoder().unwrap();
        // SAFETY: same-device fresh fixture sources, checked active byte count,
        // aligned tight rows, distinct destinations retained by the command.
        unsafe {
            e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &source.k.raw,
                0,
                &view.new_k.raw,
                0,
                count,
            );
            e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &source.v.raw,
                0,
                &view.new_v.raw,
                0,
                count,
            );
        }
        e.endEncoding();
    }
}
#[derive(Clone, PartialEq, Eq)]
struct OwnerWords {
    old_k: Vec<u16>,
    old_v: Vec<u16>,
    new_k: Vec<u16>,
    new_v: Vec<u16>,
}
struct Snapshot {
    owners: Vec<OwnerWords>,
    status: Vec<u32>,
    offset: usize,
    generation: u64,
    poisoned: bool,
}
fn snapshot(set: &Set<'_>, label: &str, inputs: &[Input]) -> Snapshot {
    // Safety precedes readback: report only metadata and stop if a command is
    // enqueued/in-flight. No mapped slice may be made in that state.
    if let Some(command) = &set.command
        && command.status() != MTLCommandBufferStatus::NotEnqueued
        && !terminal(command)
    {
        println!(
            "STAGED_CACHE_INFLIGHT_READBACK_REJECTED {}",
            json!({"label":label,"status":format!("{:?}",command.status()),"error":command.error().map(|e|e.to_string())})
        );
        std::io::stdout().flush().unwrap();
        panic!("in-flight cache readback forbidden");
    }
    let owners: Vec<_> = set.owners[..set.operation.plan.count]
        .iter()
        .map(|o| {
            let o = o.as_ref().unwrap();
            OwnerWords {
                old_k: o.old_k.with_slice(|v| v.to_vec()),
                old_v: o.old_v.with_slice(|v| v.to_vec()),
                new_k: o.new_k.with_slice(|v| v.to_vec()),
                new_v: o.new_v.with_slice(|v| v.to_vec()),
            }
        })
        .collect();
    let status = set.status.with_slice(|v| v.to_vec());
    let command=set.command.as_ref().map(|c|json!({"status":format!("{:?}",c.status()),"error":c.error().map(|e|e.to_string()),"terminal":terminal(c),"completed_without_error":completed(c)}));
    let sources: Vec<_> = inputs
        .iter()
        .map(|i| json!({"before":i.before,"after":[i.k.with_slice(hash),i.v.with_slice(hash)]}))
        .collect();
    let raw: Vec<_> = owners
        .iter()
        .map(|o| json!({"old_k":o.old_k,"old_v":o.old_v,"new_k":o.new_k,"new_v":o.new_v}))
        .collect();
    println!(
        "STAGED_CACHE_RAW {}",
        json!({"label":label,"state":format!("{:?}",set.state),"offset":set.offset,"generation":set.generation,"epoch":set.epoch,"command":command,"owners":raw,"status":status,"fixture_inputs":sources,"session_bytes":set.operation.plan.session_bytes().unwrap(),"driver_bytes_measured":false})
    );
    std::io::stdout().flush().unwrap();
    Snapshot {
        owners,
        status,
        offset: set.offset,
        generation: set.generation,
        poisoned: set.is_poisoned(),
    }
}
fn guards(set: &Set<'_>, s: &Snapshot, inputs: &[Input]) {
    assert!(s.status[1..].iter().all(|&v| v == STATUS_PAD));
    for (o, p) in s
        .owners
        .iter()
        .zip(&set.operation.plan.owners[..set.operation.plan.count])
    {
        for v in [&o.old_k, &o.old_v] {
            assert!(v[p.retained * p.config.dim..].iter().all(|&w| w == PAD));
        }
        for v in [&o.new_k, &o.new_v] {
            assert!(v[CHUNK * p.config.dim..].iter().all(|&w| w == PAD));
        }
    }
    for input in inputs {
        assert_eq!(
            [input.k.with_slice(hash), input.v.with_slice(hash)],
            input.before
        );
    }
}
fn same_old(a: &Snapshot, b: &Snapshot) {
    assert_eq!(a.offset, b.offset);
    assert_eq!(a.generation, b.generation);
    for (a, b) in a.owners.iter().zip(&b.owners) {
        assert_eq!(a.old_k, b.old_k);
        assert_eq!(a.old_v, b.old_v);
    }
}
fn operation(gpu: &Gpu, capacity: usize, configs: &[OwnerConfig]) -> Operation {
    Operation::new(
        gpu,
        Plan::new(capacity, configs).unwrap(),
        PIPELINE_ALLOWANCE + size_of::<Operation>(),
    )
    .unwrap()
}
fn new_set<'a>(gpu: &Gpu, op: &'a Operation) -> Set<'a> {
    Set::new(gpu, op, op.plan.session_bytes().unwrap()).unwrap()
}
// Seeded source comparisons are explicitly different from reset/replay below.
fn seed(set: &mut Set<'_>, cases: &[Value]) {
    assert_eq!(set.state, State::Idle);
    assert!(set.command.is_none());
    for (owner, c) in set.owners.iter_mut().flatten().zip(cases) {
        owner
            .old_k
            .with_mut(|v| v.copy_from_slice(&words(&c["inputs"]["old_k"])));
        owner
            .old_v
            .with_mut(|v| v.copy_from_slice(&words(&c["inputs"]["old_v"])));
    }
    set.offset = number(&cases[0]["committed"]);
}
fn expected(snapshot: &Snapshot, cases: &[Value]) {
    for (o, c) in snapshot.owners.iter().zip(cases) {
        assert_eq!(o.old_k, words(&c["inputs"]["expected_after_k"]));
        assert_eq!(o.old_v, words(&c["inputs"]["expected_after_v"]));
    }
}
fn successful(
    gpu: &Gpu,
    set: &mut Set<'_>,
    tokens: usize,
    inputs: &[Input],
    label: &str,
    measure: bool,
) -> Snapshot {
    let before = snapshot(set, &format!("{label}_before"), inputs);
    let measured = Measurement::start();
    let batch = set.begin(gpu, tokens);
    let a = measured.finish();
    let batch = batch.unwrap();
    write(set, &batch, inputs, inputs.len());
    let measured = Measurement::start();
    let ready = set.finish_math(batch.frames());
    let b = measured.finish();
    let math = snapshot(set, &format!("{label}_math"), inputs);
    let ready = ready.unwrap();
    same_old(&before, &math);
    for ((before, after), plan) in before
        .owners
        .iter()
        .zip(&math.owners)
        .zip(&set.operation.plan.owners[..set.operation.plan.count])
    {
        let end = tokens * plan.config.dim;
        assert_eq!(&before.new_k[end..], &after.new_k[end..]);
        assert_eq!(&before.new_v[end..], &after.new_v[end..]);
    }
    guards(set, &math, inputs);
    assert_eq!(math.status[0], 0);
    let measured = Measurement::start();
    let result = set.commit(gpu, &ready);
    let c = measured.finish();
    let after = snapshot(set, &format!("{label}_copy"), inputs);
    println!(
        "STAGED_CACHE_ALLOCATIONS {}",
        json!({"label":label,"rust_boundary_allocations":[a,b,c],"release":!cfg!(debug_assertions),"driver_allocations_measured":false})
    );
    let info = result.unwrap();
    assert_eq!(info.start, before.offset);
    assert_eq!(info.end, before.offset + tokens);
    assert_eq!(info.owners, inputs.len());
    assert_eq!(info.generation, before.generation + 1);
    assert_eq!(after.offset, info.end);
    assert_eq!(after.generation, info.generation);
    assert!(!after.poisoned);
    guards(set, &after, inputs);
    assert!(set.commit(gpu, &ready).is_err());
    if measure {
        assert!(!cfg!(debug_assertions));
        assert_eq!([a, b, c], [0, 0, 0]);
    }
    after
}
#[test]
#[ignore = "explicit release Metal cache copies and source chain"]
fn gpu_source_copies_chain_and_owner_bounds() {
    autoreleasepool(|_| {
        let m = fixture();
        let cases = m["cases"].as_array().unwrap();
        let gpu = Gpu::new().unwrap();
        let tiny = operation(
            &gpu,
            12,
            &[
                OwnerConfig {
                    dim: 32,
                    window: Some(4),
                },
                OwnerConfig {
                    dim: 32,
                    window: None,
                },
            ],
        );
        let real = operation(
            &gpu,
            544,
            &[
                OwnerConfig {
                    dim: 256,
                    window: Some(512),
                },
                OwnerConfig {
                    dim: 256,
                    window: None,
                },
            ],
        );
        assert!(matches!(
            Set::new(&gpu, &tiny, tiny.plan.session_bytes().unwrap() - 1),
            Err(Error::Budget { .. })
        ));
        assert!(matches!(
            Operation::new(
                &gpu,
                Plan::new(
                    12,
                    &[OwnerConfig {
                        dim: 32,
                        window: None
                    }]
                )
                .unwrap(),
                PIPELINE_ALLOWANCE + size_of::<Operation>() - 1
            ),
            Err(Error::Budget { .. })
        ));
        let mut small = new_set(&gpu, &tiny);
        let mut chain = new_set(&gpu, &real);
        let mut chunk = new_set(&gpu, &real);
        for i in [0, 2, 4, 6] {
            let pair = &cases[i..i + 2];
            let set = match i {
                0 => &mut small,
                2 | 4 => &mut chain,
                _ => &mut chunk,
            };
            if i != 4 {
                seed(set, pair);
            } else {
                let s = snapshot(set, "chain_514_before", &[]);
                for (o, c) in s.owners.iter().zip(pair) {
                    assert_eq!(o.old_k, words(&c["inputs"]["old_k"]));
                    assert_eq!(o.old_v, words(&c["inputs"]["old_v"]));
                }
            }
            let inputs = inputs_for(&gpu, pair);
            let after = successful(
                &gpu,
                set,
                number(&pair[0]["tokens"]),
                &inputs,
                pair[0]["name"].as_str().unwrap(),
                true,
            );
            expected(&after, pair);
        }
        for count in [1, 32] {
            let op = operation(
                &gpu,
                1,
                &vec![
                    OwnerConfig {
                        dim: 32,
                        window: Some(1)
                    };
                    count
                ],
            );
            let mut set = new_set(&gpu, &op);
            let inputs = patterned(&gpu, count, 32, &[0x3c00, 0xbc00, 0x0001, 0x8001]);
            let after = successful(&gpu, &mut set, 1, &inputs, &format!("owners{count}"), false);
            for (o, input) in after.owners.iter().zip(&inputs) {
                assert_eq!(
                    o.old_k[..32].to_vec(),
                    input.k.with_slice(|v| v[..32].to_vec())
                );
                assert_eq!(
                    o.old_v[..32].to_vec(),
                    input.v.with_slice(|v| v[..32].to_vec())
                );
            }
            assert!(set.begin(&gpu, 1).is_err());
        }
        println!(
            "STAGED_CACHE_SOURCE_RESULT {}",
            json!({"fixture_cases":8,"paired_steps":4,"chained_513_514":true,"actual_gpu_copy":true,"release_boundary_calls":12,"extreme_owner_sets":[1,32],"graph_qualified":false,"performance_claim":false})
        );
    });
}
#[test]
#[ignore = "explicit release Metal all-owner Half validation and metadata rejection"]
fn gpu_half_words_omission_and_readiness() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let op = operation(
            &gpu,
            12,
            &[
                OwnerConfig {
                    dim: 32,
                    window: Some(4),
                },
                OwnerConfig {
                    dim: 32,
                    window: None,
                },
            ],
        );
        let mut set = new_set(&gpu, &op);
        let finite = [
            0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x7bff, 0xfbff,
        ];
        let inputs = patterned(&gpu, 2, 32, &finite);
        let valid = successful(&gpu, &mut set, 1, &inputs, "finite_half_classes", false);
        for (o, input) in valid.owners.iter().zip(&inputs) {
            assert_eq!(
                o.old_k[..32].to_vec(),
                input.k.with_slice(|v| v[..32].to_vec())
            );
            assert_eq!(
                o.old_v[..32].to_vec(),
                input.v.with_slice(|v| v[..32].to_vec())
            );
        }
        // Reusing already-finite staging without this owner's new writer must
        // fail: begin's active fill is encoded before the remaining producers.
        let before = snapshot(&set, "omitted_before", &inputs);
        let batch = set.begin(&gpu, 1).unwrap();
        write(&set, &batch, &inputs, 1);
        let result = set.finish_math(batch.frames());
        let after = snapshot(&set, "omitted_after", &inputs);
        assert!(matches!(result, Err(Error::Nonfinite)));
        same_old(&before, &after);
        guards(&set, &after, &inputs);
        assert_ne!(after.status[0], 0);
        assert!(after.owners[1].new_k[..32].iter().all(|&w| w == 0x7e7e));
        assert!(after.owners[1].new_v[..32].iter().all(|&w| w == 0x7e7e));
        // A partial producer must not escape validation at the last row/tail.
        // Offset1 + seven rows fits C12; leave only the last two K Half words
        // stamped. The truncated byte count444 is still Metal-blit aligned.
        let mut full_k = vec![PAD; CHUNK * 32 + GUARD];
        let mut full_v = full_k.clone();
        full_k[..224].fill(0x3c00);
        full_v[..224].fill(0xbc00);
        let partial_inputs: Vec<_> = (0..2).map(|_| input(&gpu, &full_k, &full_v)).collect();
        let before = snapshot(&set, "partial_producer_before", &partial_inputs);
        let batch = set.begin(&gpu, 7).unwrap();
        write(&set, &batch, &partial_inputs, 1);
        {
            let view = set.view(&batch.frames()[1]).unwrap();
            let encoder = view.command.blitCommandEncoder().unwrap();
            // SAFETY: same-device, distinct retained buffers, bounded active
            // tight ranges; both sizes are multiples of four. Intentional
            // missing final K words remain the begin-stamped NaN sentinel.
            unsafe {
                encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &partial_inputs[1].k.raw,
                    0,
                    &view.new_k.raw,
                    0,
                    222 * 2,
                );
                encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &partial_inputs[1].v.raw,
                    0,
                    &view.new_v.raw,
                    0,
                    224 * 2,
                );
            }
            encoder.endEncoding();
        }
        let result = set.finish_math(batch.frames());
        let after = snapshot(
            &set,
            "partial_producer_last_two_K_unwritten",
            &partial_inputs,
        );
        assert!(matches!(result, Err(Error::Nonfinite)));
        same_old(&before, &after);
        guards(&set, &after, &partial_inputs);
        assert_ne!(after.status[0], 0);
        assert!(after.owners[1].new_k[..222].iter().all(|&w| w == 0x3c00));
        assert_eq!(&after.owners[1].new_k[222..224], &[0x7e7e, 0x7e7e]);
        assert!(after.owners[1].new_v[..224].iter().all(|&w| w == 0xbc00));
        for (a, b) in before.owners.iter().zip(&after.owners) {
            assert_eq!(&a.new_k[224..], &b.new_k[224..]);
            assert_eq!(&a.new_v[224..], &b.new_v[224..]);
        }
        for (i, word) in [0x7c00, 0xfc00, 0x7c01, 0xfc01, 0x7e00, 0xfe00]
            .into_iter()
            .enumerate()
        {
            let mut poisoned = patterned(&gpu, 2, 32, &finite);
            let owner = i % 2;
            if (i / 2) % 2 == 0 {
                poisoned[owner].k.with_mut(|v| v[5] = word);
            } else {
                poisoned[owner].v.with_mut(|v| v[5] = word);
            }
            poisoned[owner].before = [
                poisoned[owner].k.with_slice(hash),
                poisoned[owner].v.with_slice(hash),
            ];
            let before = snapshot(&set, &format!("bad{i}_before"), &poisoned);
            let batch = set.begin(&gpu, 1).unwrap();
            write(&set, &batch, &poisoned, 2);
            let result = set.finish_math(batch.frames());
            let after = snapshot(&set, &format!("bad{i}_after"), &poisoned);
            assert!(matches!(result, Err(Error::Nonfinite)));
            same_old(&before, &after);
            guards(&set, &after, &poisoned);
            assert_ne!(after.status[0], 0);
        }
        // Fresh begin clears the completed prior error. Valid stages with a
        // shared graph-status error still cannot produce Ready.
        let batch = set.begin(&gpu, 1).unwrap();
        write(&set, &batch, &inputs, 2);
        set.status.with_mut(|v| v[0] = 4);
        let result = set.finish_math(batch.frames());
        let after = snapshot(&set, "sticky_graph_status", &inputs);
        assert!(matches!(result, Err(Error::Nonfinite)));
        assert_eq!(after.status[0], 4);
        same_old(&valid, &after);
        let batch = set.begin(&gpu, 1).unwrap();
        write(&set, &batch, &inputs, 2);
        let canonical = batch.frames;
        let mut foreign_set = new_set(&gpu, &op);
        let foreign_batch = foreign_set.begin(&gpu, 1).unwrap();
        let foreign_result = set.finish_math(foreign_batch.frames());
        let foreign_snapshot = snapshot(&set, "foreign_set_frames_rejected", &inputs);
        snapshot(&foreign_set, "foreign_set_unsubmitted", &[]);
        assert!(foreign_result.is_err());
        same_old(&valid, &foreign_snapshot);
        foreign_set.discard().unwrap();
        for field in 0..8 {
            let mut wrong = canonical;
            match field {
                0 => wrong.swap(0, 1),
                1 => wrong[1] = wrong[0],
                2 => wrong[0].set += 1,
                3 => wrong[0].generation += 1,
                4 => wrong[0].epoch += 1,
                5 => wrong[0].offset += 1,
                6 => wrong[0].tokens += 1,
                7 => wrong[0].owner = MAX,
                _ => unreachable!(),
            }
            let result = set.finish_math(&wrong[..2]);
            let s = snapshot(&set, &format!("bad_frames{field}"), &inputs);
            assert!(result.is_err());
            same_old(&valid, &s);
            assert_eq!(
                set.command.as_ref().unwrap().status(),
                MTLCommandBufferStatus::NotEnqueued
            );
        }
        let result = set.finish_math(&canonical[..1]);
        let s = snapshot(&set, "missing_frame", &inputs);
        assert!(result.is_err());
        same_old(&valid, &s);
        let ready = set.finish_math(batch.frames());
        let s = snapshot(&set, "ready_after_rejections", &inputs);
        let ready = ready.unwrap();
        assert_eq!(s.status[0], 0);
        same_old(&valid, &s);
        assert!(set.view(&batch.frames()[0]).is_err());
        let mut wrong = ready.batch.frames;
        wrong[1] = wrong[0];
        let bad_ready = Ready {
            batch: Batch {
                frames: wrong,
                len: 2,
            },
        };
        let result = set.commit(&gpu, &bad_ready);
        let s = snapshot(&set, "bad_ready_before_copy", &inputs);
        assert!(result.is_err());
        same_old(&valid, &s);
        assert_eq!(set.state, State::Ready);
        set.discard().unwrap();
        assert!(set.commit(&gpu, &ready).is_err());
        let stale = batch.frames[0];
        let next = set.begin(&gpu, 1).unwrap();
        assert!(set.view(&stale).is_err());
        write(&set, &next, &inputs, 2);
        // Deliberate violation of the internal submit contract cannot mint Ready,
        // even if this arbitrary command completed and status remains zero.
        set.command.as_ref().unwrap().enqueue();
        let malformed = set.finish_math(&[]);
        let premature_reset = set.reset();
        // Finish the already-enqueued command before any mapped capture/assert.
        let raw = set.command.as_ref().unwrap();
        raw.commit();
        raw.waitUntilCompleted();
        let result = set.finish_math(next.frames());
        let s = snapshot(&set, "premature_submission_rejected", &inputs);
        assert!(matches!(malformed, Err(Error::ReplayRequired)));
        assert!(premature_reset.is_err());
        assert!(matches!(result, Err(Error::ReplayRequired)));
        assert!(s.poisoned);
        assert!(set.begin(&gpu, 1).is_err());
        set.reset().unwrap();
        assert_eq!(set.offset(), 0);
        assert!(!set.is_poisoned());
        set.epoch = u64::MAX;
        let before = snapshot(&set, "epoch_exhausted_before", &[]);
        let result = set.begin(&gpu, 1);
        let after = snapshot(&set, "epoch_exhausted_after", &[]);
        assert!(result.is_err());
        same_old(&before, &after);
        set.epoch = 0;
        set.generation = u64::MAX;
        let batch = set.begin(&gpu, 1).unwrap();
        write(&set, &batch, &inputs, 2);
        let ready = set.finish_math(batch.frames());
        let before = snapshot(&set, "generation_exhausted_ready", &inputs);
        let ready = ready.unwrap();
        let result = set.commit(&gpu, &ready);
        let after = snapshot(&set, "generation_exhausted_copy_rejected", &inputs);
        assert!(result.is_err());
        same_old(&before, &after);
        assert!(set.reset().is_err());
        println!(
            "STAGED_CACHE_VALIDATION_RESULT {}",
            json!({"finite_half_patterns":finite,"nonfinite_patterns":6,"omitted_owner_after_finite_staging_detected":true,"partial_last_row_K_tail_detected":true,"attention_encoded":false,"shared_status_blocks_readiness":true,"frame_mutations":8,"missing_frame_rejected":true,"bad_ready_pre_copy_rejected":true,"premature_command_rejected":true,"epoch_generation_exhaustion_rejected":true,"fake_boolean_proof_api":false})
        );
    });
}
#[test]
#[ignore = "explicit release Metal partial-copy poisoning and zero-prefix replay"]
fn gpu_partial_copy_poison_reset_replay() {
    autoreleasepool(|_| {
        let m = fixture();
        let pair = &m["cases"].as_array().unwrap()[..2];
        let gpu = Gpu::new().unwrap();
        let op = operation(
            &gpu,
            12,
            &[
                OwnerConfig {
                    dim: 32,
                    window: Some(4),
                },
                OwnerConfig {
                    dim: 32,
                    window: None,
                },
            ],
        );
        let mut set = new_set(&gpu, &op);
        seed(&mut set, pair);
        let inputs = inputs_for(&gpu, pair);
        let before = snapshot(&set, "fault_before", &inputs);
        let batch = set.begin(&gpu, 7).unwrap();
        write(&set, &batch, &inputs, 2);
        let ready = set.finish_math(batch.frames());
        let math = snapshot(&set, "fault_math_ready", &inputs);
        let ready = ready.unwrap();
        same_old(&before, &math);
        let result = set.interrupt_copy_after_first_run(&gpu, &ready);
        let partial = snapshot(&set, "injected_first_K_run_only", &inputs);
        assert!(matches!(result, Err(Error::ReplayRequired)));
        assert!(partial.poisoned);
        assert_eq!(partial.offset, before.offset);
        assert_eq!(partial.generation, before.generation);
        assert_ne!(partial.owners[0].old_k, before.owners[0].old_k);
        assert_eq!(
            partial.owners[0].old_k,
            words(&pair[0]["inputs"]["expected_after_k"])
        );
        assert_eq!(partial.owners[0].old_v, before.owners[0].old_v);
        assert_eq!(partial.owners[1].old_k, before.owners[1].old_k);
        assert_eq!(partial.owners[1].old_v, before.owners[1].old_v);
        guards(&set, &partial, &inputs);
        assert!(set.begin(&gpu, 1).is_err());
        assert!(set.view(&batch.frames()[0]).is_err());
        assert!(set.commit(&gpu, &ready).is_err());
        assert!(set.discard().is_err());
        assert!(completed(set.command.as_ref().unwrap()));
        set.reset().unwrap();
        let reset = snapshot(&set, "terminal_zero_prefix_reset", &inputs);
        assert_eq!(reset.offset, 0);
        assert!(!reset.poisoned);
        assert!(set.commit(&gpu, &ready).is_err());
        // Replay positions0..4 from the full-owner original seed, followed by
        // the same original new positions5..11. No CPU cache repair/old rewind.
        let old_k = words(&pair[1]["inputs"]["old_k"]);
        let old_v = words(&pair[1]["inputs"]["old_v"]);
        let mut k = vec![PAD; CHUNK * 32 + GUARD];
        let mut v = k.clone();
        k[..5 * 32].copy_from_slice(&old_k[..5 * 32]);
        v[..5 * 32].copy_from_slice(&old_v[..5 * 32]);
        let prefix: Vec<_> = (0..2).map(|_| input(&gpu, &k, &v)).collect();
        successful(&gpu, &mut set, 5, &prefix, "replay_prefix5", false);
        let after = successful(&gpu, &mut set, 7, &inputs, "replay_tail7", false);
        expected(&after, pair);
        println!(
            "STAGED_CACHE_FAULT_RESULT {}",
            json!({"fault":"test-only command containing first K run; forced error after real terminal completion","real_metal_error_claim":false,"actual_partial_mutation_verified":true,"entire_set_poisoned":true,"old_offsets_not_published":true,"terminal_reset_prefix":0,"replay_chunks":[5,7],"all_final_owner_words_match_frozen_expected":true,"rollback_claim":false,"graph_qualified":false})
        );
    });
}
