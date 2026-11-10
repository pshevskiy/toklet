use super::*;
use half::bf16;
use serde_json::Value;
use sha2::{Digest, Sha256};
use smallvec::SmallVec;

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn digest(values: impl IntoIterator<Item = bf16>) -> String {
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 4096];
    let mut used = 0;
    for value in values {
        let bytes = value.to_bits().to_le_bytes();
        buffer[used] = bytes[0];
        buffer[used + 1] = bytes[1];
        used += 2;
        if used == buffer.len() {
            hash.update(buffer);
            used = 0;
        }
    }
    hash.update(&buffer[..used]);
    format!("{:x}", hash.finalize())
}
fn fixture() -> Value {
    let raw = include_bytes!("../../../tests/fixtures/lfm3-bf16-state-v1/state.json");
    assert_eq!(
        hash(raw),
        "df7aaec7b0e55c2029bd43162ee15801f30b87772ce75fd46f6bd0acbb42cf5c"
    );
    serde_json::from_slice(raw).unwrap()
}
fn word(owner: usize, field: usize, head: usize, position: usize, lane: usize) -> bf16 {
    let index = (owner * 8191 + field * 4099 + head * 2053 + position * 37 + lane * 17) % 65280;
    bf16::from_bits((if index < 32640 { index } else { index + 128 }) as u16)
}
fn rows(
    owner: usize,
    field: usize,
    heads: usize,
    dims: usize,
    start: usize,
    tokens: usize,
) -> Vec<bf16> {
    (0..heads)
        .flat_map(|h| {
            (start..start + tokens)
                .flat_map(move |p| (0..dims).map(move |lane| word(owner, field, h, p, lane)))
        })
        .collect()
}
fn identity() -> Vec<bf16> {
    let mut weights = vec![bf16::ZERO; 7 * 2048];
    for row in weights.chunks_exact_mut(7) {
        row[6] = bf16::ONE;
    }
    weights
}
fn positions(cache: &attention::WindowKvCache, capacity: usize) -> String {
    let mut hash = Sha256::new();
    for slot in 0..capacity {
        hash.update(
            cache
                .position_in_slot(slot)
                .map(|p| p as u64)
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
    }
    format!("{:x}", hash.finalize())
}
fn kv_snapshot(
    cache: &attention::WindowKvCache,
    capacity: usize,
) -> (usize, u64, String, String, String) {
    let (k, v) = cache.committed();
    (
        cache.offset(),
        cache.generation(),
        digest(k.iter().copied()),
        digest(v.iter().copied()),
        positions(cache, capacity),
    )
}

#[test]
fn bf16_views_physical_rings_and_visibility_match_all_frozen_schedules() {
    let fixture = fixture();
    let mut observations = 0;
    for case in fixture["kv"].as_array().unwrap() {
        let n = |key: &str| case[key].as_u64().unwrap() as usize;
        let capacity = n("capacity");
        let window = case["window"].as_u64().map(|v| v as usize);
        let mut cache = attention::WindowKvCache::new(4, 128, capacity, 2048, window).unwrap();
        assert_eq!(
            cache.allocated_bytes(),
            attention::WindowKvCache::planned_bytes(4, 128, capacity, 2048, window).unwrap()
        );
        let pointers = {
            let (k, v) = cache.committed();
            (k.as_ptr(), v.as_ptr())
        };
        for (step, tokens) in case["chunks"].as_array().unwrap().iter().enumerate() {
            let tokens = tokens.as_u64().unwrap() as usize;
            let offset = cache.offset();
            let k = rows(n("owner"), 0, 4, 128, offset, tokens);
            let v = rows(n("owner"), 1, 4, 128, offset, tokens);
            let selected = case["observations"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["step"].as_u64().unwrap() as usize == step);
            let before = selected.map(|_| kv_snapshot(&cache, capacity));
            let frame = cache.prepare(&k, &v, tokens).unwrap();
            if let Some(expected) = selected {
                let view = cache.view(&frame).unwrap();
                assert_eq!(view.query_start(), offset);
                assert_eq!(view.end(), offset + tokens);
                assert_eq!(
                    view.start(),
                    expected["view_start"].as_u64().unwrap() as usize
                );
                let mut keys = Vec::new();
                let mut values = Vec::new();
                for head in 0..4 {
                    for span in view.head(head).unwrap() {
                        keys.extend_from_slice(span.k);
                        values.extend_from_slice(span.v);
                    }
                }
                assert_eq!(digest(keys), expected["view_key_sha256"].as_str().unwrap());
                assert_eq!(
                    digest(values),
                    expected["view_value_sha256"].as_str().unwrap()
                );
                let segments = view.segments();
                let mut keys = Vec::new();
                let mut values = Vec::new();
                for head in 0..4 {
                    for segment in &segments {
                        if segment.keys == 0 {
                            continue;
                        }
                        let start = head * segment.head_stride;
                        let end = start + segment.keys * 128;
                        keys.extend_from_slice(&segment.k[start..end]);
                        values.extend_from_slice(&segment.v[start..end]);
                    }
                }
                assert_eq!(digest(keys), expected["view_key_sha256"].as_str().unwrap());
                assert_eq!(
                    digest(values),
                    expected["view_value_sha256"].as_str().unwrap()
                );
                let mut ranges = Sha256::new();
                let mut mask = Sha256::new();
                let mut bytes = [0_u8; 4096];
                let mut used = 0;
                for query in offset..offset + tokens {
                    let visible = view.visible_range(query).unwrap();
                    ranges.update((visible.start as u64).to_le_bytes());
                    ranges.update((visible.end as u64).to_le_bytes());
                    for key in view.start()..view.end() {
                        bytes[used] = u8::from(visible.contains(&key));
                        used += 1;
                        if used == bytes.len() {
                            mask.update(bytes);
                            used = 0;
                        }
                    }
                }
                mask.update(&bytes[..used]);
                assert_eq!(
                    format!("{:x}", ranges.finalize()),
                    expected["visible_ranges_sha256"].as_str().unwrap()
                );
                assert_eq!(
                    format!("{:x}", mask.finalize()),
                    expected["mask_sha256"].as_str().unwrap()
                );
                assert!(view.visible_range(view.end()).is_err());
                assert_eq!(kv_snapshot(&cache, capacity), before.clone().unwrap());
            }
            cache.commit(&frame).unwrap();
            if let Some(expected) = selected {
                let (k, v) = cache.committed();
                assert_eq!(
                    digest(k.iter().copied()),
                    expected["physical_key_sha256"].as_str().unwrap()
                );
                assert_eq!(
                    digest(v.iter().copied()),
                    expected["physical_value_sha256"].as_str().unwrap()
                );
                assert_eq!(
                    positions(&cache, capacity),
                    expected["physical_positions_sha256"].as_str().unwrap()
                );
                observations += 1;
            }
            let (k, v) = cache.committed();
            assert_eq!((k.as_ptr(), v.as_ptr()), pointers);
        }
    }
    assert_eq!(observations, 120);
}

#[test]
fn bf16_history_candidates_match_frozen_streams_without_work_banks() {
    let fixture = fixture();
    let weights = identity();
    let mut observations = 0;
    for case in fixture["history"].as_array().unwrap() {
        let mut owner = convolution::History::new(2048, 2048).unwrap();
        let mut candidate = vec![bf16::NAN; 6 * 2048];
        let pointer = owner.committed().as_ptr();
        for (step, tokens) in case["chunks"].as_array().unwrap().iter().enumerate() {
            let tokens = tokens.as_u64().unwrap() as usize;
            let input = rows(0, 2, 1, 2048, owner.offset(), tokens);
            let mut output = vec![bf16::NAN; input.len()];
            let selected = case["observations"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["step"].as_u64().unwrap() as usize == step);
            if let Some(expected) = selected {
                assert_eq!(
                    digest(owner.committed().iter().copied()),
                    expected["previous_sha256"].as_str().unwrap()
                );
                {
                    let _aborted = owner
                        .prepare_chunk(&input, &weights, &mut candidate, &mut output, tokens)
                        .unwrap();
                }
                assert_eq!(
                    digest(owner.committed().iter().copied()),
                    expected["previous_sha256"].as_str().unwrap()
                );
                assert_eq!(
                    digest(candidate.iter().copied()),
                    expected["candidate_sha256"].as_str().unwrap()
                );
            }
            let frame = owner
                .prepare_chunk(&input, &weights, &mut candidate, &mut output, tokens)
                .unwrap();
            owner.commit(&frame).unwrap();
            if let Some(expected) = selected {
                assert_eq!(
                    digest(owner.committed().iter().copied()),
                    expected["candidate_sha256"].as_str().unwrap()
                );
                observations += 1;
            }
            assert_eq!(owner.committed().as_ptr(), pointer);
        }
    }
    assert_eq!(observations, 60);
}

fn source_inputs(ordinal: usize, rows: usize) -> Vec<bf16> {
    let pool: Vec<_> = [0, 0x8000]
        .into_iter()
        .flat_map(|sign| (0..=0x4180).map(move |word| bf16::from_bits(word | sign)))
        .collect();
    (0..rows * 2048)
        .map(|index| {
            let selector = index * 37 + index / 2048 * 11 + ordinal * 3;
            if index.is_multiple_of(2) {
                pool[selector % pool.len()]
            } else {
                bf16::from_bits(((0x3d80 + selector % 1025) | ((selector / 1025 % 2) << 15)) as u16)
            }
        })
        .collect()
}

#[test]
fn all_thirteen_real_convolutions_preserve_canonical_output_across_history_commits() {
    let raw = include_bytes!("../../../tests/fixtures/lfm3-conv-bf16-v1/parent.json");
    assert_eq!(
        hash(raw),
        "c90abf0c6fed114ba8f3e1459d3a5e82a4fd94c0347249bde41cac0c1794493a"
    );
    let fixture: Value = serde_json::from_slice(raw).unwrap();
    let raw = include_bytes!("../../../tests/fixtures/lfm3-conv-bf16-v1/weights.bf16le");
    assert_eq!(
        hash(raw),
        "f305ef6e0054d9128004f7ac9c4eccb7dd5bf0e0f1bc3143ef39de3355ed9e62"
    );
    for ordinal in 0..13 {
        let bank = &fixture["convolution"]["banks"][ordinal];
        let start = bank["byte_offset"].as_u64().unwrap() as usize;
        let bytes = bank["byte_count"].as_u64().unwrap() as usize;
        let weights: Vec<_> = raw[start..start + bytes]
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let input = source_inputs(ordinal, 128);
        let mut output = vec![bf16::NAN; input.len()];
        let mut candidate = vec![bf16::NAN; 6 * 2048];
        let mut owner = convolution::History::new(2048, 128).unwrap();
        let mut offset = 0;
        for tokens in [1, 5, 1, 7, 17, 97] {
            let end = offset + tokens;
            let frame = owner
                .prepare_chunk(
                    &input[offset * 2048..end * 2048],
                    &weights,
                    &mut candidate,
                    &mut output[offset * 2048..end * 2048],
                    tokens,
                )
                .unwrap();
            owner.commit(&frame).unwrap();
            offset = end;
        }
        let expected = fixture["convolution"]["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| {
                case["ordinal"] == ordinal
                    && case["tokens"] == 128
                    && case["history_profile"] == "zero-history"
            })
            .unwrap();
        assert_eq!(digest(output), expected["output_sha256"].as_str().unwrap());
        assert_eq!(
            digest(owner.committed().iter().copied()),
            digest(input[122 * 2048..].iter().copied())
        );
    }
}

struct Buffers {
    input: Vec<Vec<bf16>>,
    weights: Vec<bf16>,
    candidates: Vec<Vec<bf16>>,
    output: Vec<bf16>,
    keys: Vec<Vec<bf16>>,
    values: Vec<Vec<bf16>>,
}
struct Frames<'a> {
    histories: SmallVec<[convolution::Prepared<'a>; 13]>,
    caches: SmallVec<[attention::PreparedKv<'a>; 11]>,
}
impl Buffers {
    fn new(tokens: usize, offset: usize) -> Self {
        Self {
            input: CONV_LAYERS
                .iter()
                .map(|&layer| rows(layer, 2, 1, 2048, offset, tokens))
                .collect(),
            weights: identity(),
            candidates: (0..13).map(|_| vec![bf16::NAN; 6 * 2048]).collect(),
            output: vec![bf16::NAN; tokens * 2048],
            keys: ATTENTION_LAYERS
                .iter()
                .map(|&layer| rows(layer, 0, 4, 128, offset, tokens))
                .collect(),
            values: ATTENTION_LAYERS
                .iter()
                .map(|&layer| rows(layer, 1, 4, 128, offset, tokens))
                .collect(),
        }
    }
    fn prepare<'a>(&'a mut self, state: &State, tokens: usize) -> Result<Frames<'a>, Error> {
        let mut histories = SmallVec::new();
        let mut caches = SmallVec::new();
        for (i, (owner, candidate)) in state
            .histories()
            .iter()
            .zip(&mut self.candidates)
            .enumerate()
        {
            histories.push(owner.prepare_chunk(
                &self.input[i],
                &self.weights,
                candidate,
                &mut self.output,
                tokens,
            )?);
        }
        for (i, owner) in state.caches().iter().enumerate() {
            caches.push(owner.prepare(&self.keys[i], &self.values[i], tokens)?);
        }
        Ok(Frames { histories, caches })
    }
}
fn step(state: &mut State, offset: usize, tokens: usize) {
    let mut buffers = Buffers::new(tokens, offset);
    let frames = buffers.prepare(state, tokens).unwrap();
    state
        .commit(&frames.histories, &frames.caches, tokens)
        .unwrap();
}
#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    offset: usize,
    generation: u64,
    histories: Vec<(usize, u64, String)>,
    caches: Vec<(usize, u64, String, String)>,
}
fn snapshot(state: &State) -> Snapshot {
    Snapshot {
        offset: state.offset(),
        generation: state.generation(),
        histories: state
            .histories()
            .iter()
            .map(|h| {
                (
                    h.offset(),
                    h.generation(),
                    digest(h.committed().iter().copied()),
                )
            })
            .collect(),
        caches: state
            .caches()
            .iter()
            .map(|c| {
                let (k, v) = c.committed();
                (
                    c.offset(),
                    c.generation(),
                    digest(k.iter().copied()),
                    digest(v.iter().copied()),
                )
            })
            .collect(),
    }
}

#[test]
fn fixed_topology_and_actual_storage_accounting_match_the_prospective_contract() {
    let raw = include_bytes!("../../../tests/fixtures/lfm3-bf16-state-v1/contract.json");
    assert_eq!(
        hash(raw),
        "ad7329ea7d02b63334b7229a78c414f16b5095076d96c62a528dfdd5e08e4953"
    );
    let contract: Value = serde_json::from_slice(raw).unwrap();
    let layers = contract["layer_types"].as_array().unwrap();
    assert_eq!(
        CONV_LAYERS.to_vec(),
        layers
            .iter()
            .enumerate()
            .filter_map(|(i, t)| (t == "conv").then_some(i))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        ATTENTION_LAYERS.to_vec(),
        layers
            .iter()
            .enumerate()
            .filter_map(|(i, t)| (t != "conv").then_some(i))
            .collect::<Vec<_>>()
    );
    let state = State::new(2048).unwrap();
    assert_eq!(state.histories().len(), 13);
    assert_eq!(state.caches().len(), 11);
    let mut history_data = 0;
    let mut kv_data = 0;
    for owner in state.histories() {
        assert_eq!(owner.committed().len(), 6 * 2048);
        history_data += owner.committed().len() * 2;
    }
    for (i, owner) in state.caches().iter().enumerate() {
        let expected_window = (layers[ATTENTION_LAYERS[i]] == "sliding_attention").then_some(256);
        assert_eq!(ATTENTION_WINDOWS[i], expected_window);
        let count = 4 * expected_window.unwrap_or(2048) * 128;
        let (k, v) = owner.committed();
        assert_eq!(k.len(), count);
        assert_eq!(v.len(), count);
        kv_data += (k.len() + v.len()) * 2;
    }
    assert_eq!(history_data, 319488);
    assert_eq!(kv_data, 27787264);
    let overhead = std::mem::size_of::<State>()
        + 13 * std::mem::size_of::<convolution::History>()
        + 11 * std::mem::size_of::<attention::WindowKvCache>();
    assert_eq!(state.allocated_bytes(), history_data + kv_data + overhead);
    assert_eq!(state.allocated_bytes(), State::planned_bytes(2048).unwrap());
    assert!(State::new(0).is_err());
    assert!(State::new(262145).is_err());
    assert!(State::planned_bytes(usize::MAX).is_err());
    assert_ne!(
        crate::capacity::KvStorage::Bf16,
        crate::capacity::KvStorage::F16
    );
    assert!(crate::model::short_conv::History::<3, bf16>::new(2, 8).is_err());
}

#[test]
fn joint_publication_preflights_both_families_and_abort_retry_reset_replay() {
    let mut state = State::new(8).unwrap();
    step(&mut state, 0, 1);
    let pointers = (
        state
            .histories()
            .iter()
            .map(|h| h.committed().as_ptr())
            .collect::<Vec<_>>(),
        state
            .caches()
            .iter()
            .map(|c| {
                let (k, v) = c.committed();
                (k.as_ptr(), v.as_ptr())
            })
            .collect::<Vec<_>>(),
    );
    let before = snapshot(&state);
    let mut buffers = Buffers::new(1, 1);
    {
        let _aborted = buffers.prepare(&state, 1).unwrap();
    }
    assert_eq!(snapshot(&state), before);
    let mut frames = buffers.prepare(&state, 1).unwrap();
    assert!(
        state
            .commit(&frames.histories[..12], &frames.caches, 1)
            .is_err()
    );
    assert!(
        state
            .commit(&frames.histories, &frames.caches[..10], 1)
            .is_err()
    );
    assert!(state.commit(&frames.histories, &frames.caches, 0).is_err());
    assert!(state.commit(&frames.histories, &frames.caches, 2).is_err());
    assert_eq!(snapshot(&state), before);
    frames.histories.swap(0, 12);
    assert!(state.commit(&frames.histories, &frames.caches, 1).is_err());
    frames.histories.swap(0, 12);
    frames.caches.swap(0, 10);
    assert!(state.commit(&frames.histories, &frames.caches, 1).is_err());
    frames.caches.swap(0, 10);
    assert_eq!(snapshot(&state), before);
    state.commit(&frames.histories, &frames.caches, 1).unwrap();
    assert_eq!(state.offset(), 2);
    assert_eq!(state.generation(), 2);
    let committed = snapshot(&state);
    assert!(state.commit(&frames.histories, &frames.caches, 1).is_err());
    assert_eq!(snapshot(&state), committed);
    let allocated = state.allocated_bytes();
    state.reset().unwrap();
    assert_eq!(state.offset(), 0);
    assert_eq!(state.generation(), 3);
    assert_eq!(state.allocated_bytes(), allocated);
    assert!(
        state
            .histories()
            .iter()
            .all(|h| h.committed().iter().all(|v| v.to_bits() == 0))
    );
    assert!(
        state
            .caches()
            .iter()
            .all(|c| (0..256).all(|slot| c.position_in_slot(slot).is_none()))
    );
    assert_eq!(
        (
            state
                .histories()
                .iter()
                .map(|h| h.committed().as_ptr())
                .collect::<Vec<_>>(),
            state
                .caches()
                .iter()
                .map(|c| {
                    let (k, v) = c.committed();
                    (k.as_ptr(), v.as_ptr())
                })
                .collect::<Vec<_>>()
        ),
        pointers
    );
    step(&mut state, 0, 2);
    let replay = snapshot(&state);
    assert_eq!(
        replay.histories.iter().map(|h| &h.2).collect::<Vec<_>>(),
        committed.histories.iter().map(|h| &h.2).collect::<Vec<_>>()
    );
    assert_eq!(
        replay
            .caches
            .iter()
            .map(|c| (&c.2, &c.3))
            .collect::<Vec<_>>(),
        committed
            .caches
            .iter()
            .map(|c| (&c.2, &c.3))
            .collect::<Vec<_>>()
    );
}

#[test]
fn foreign_late_family_and_generation_failures_leave_every_owner_unchanged() {
    let mut state = State::new(8).unwrap();
    let mut other = State::new(8).unwrap();
    step(&mut state, 0, 1);
    step(&mut other, 0, 1);
    let mut a = Buffers::new(1, 1);
    let mut b = Buffers::new(1, 1);
    let mut frames = a.prepare(&state, 1).unwrap();
    let mut foreign = b.prepare(&other, 1).unwrap();
    std::mem::swap(&mut frames.caches[10], &mut foreign.caches[10]);
    let before = snapshot(&state);
    assert!(state.commit(&frames.histories, &frames.caches, 1).is_err());
    assert_eq!(snapshot(&state), before);
    std::mem::swap(&mut frames.caches[10], &mut foreign.caches[10]);
    state.caches[10].set_generation_for_test(u64::MAX);
    let before = snapshot(&state);
    assert!(state.commit(&frames.histories, &frames.caches, 1).is_err());
    assert!(state.reset().is_err());
    assert_eq!(snapshot(&state), before);
    state.caches[10].set_generation_for_test(1);
    state.histories[12].set_generation_for_test(u64::MAX);
    let before = snapshot(&state);
    assert!(state.commit(&frames.histories, &frames.caches, 1).is_err());
    assert!(state.reset().is_err());
    assert_eq!(snapshot(&state), before);
    state.histories[12].set_generation_for_test(1);
    state.generation = u64::MAX;
    let before = snapshot(&state);
    assert!(state.commit(&frames.histories, &frames.caches, 1).is_err());
    assert!(state.reset().is_err());
    assert_eq!(snapshot(&state), before);
    state.generation = 1;
    state.commit(&frames.histories, &frames.caches, 1).unwrap();
    assert_eq!(state.offset(), 2);
}

#[test]
fn every_finite_bf16_pattern_survives_storage_and_numerical_failure_publishes_nothing() {
    let finite: Vec<_> = (0..=u16::MAX)
        .filter(|word| word & 0x7f80 != 0x7f80)
        .map(bf16::from_bits)
        .collect();
    assert_eq!(finite.len(), 65280);
    assert_eq!(
        digest(finite.iter().copied()),
        fixture()["finite_bf16"]["sha256"].as_str().unwrap()
    );
    let input: Vec<_> = finite
        .iter()
        .copied()
        .chain(std::iter::repeat_n(bf16::ZERO, 256))
        .collect();
    let mut owner = convolution::History::new(2048, 32).unwrap();
    let mut candidate = vec![bf16::NAN; 6 * 2048];
    let mut output = vec![bf16::NAN; input.len()];
    let frame = owner
        .prepare_chunk(&input, &identity(), &mut candidate, &mut output, 32)
        .unwrap();
    owner.commit(&frame).unwrap();
    assert_eq!(
        digest(owner.committed().iter().copied()),
        digest(input[26 * 2048..].iter().copied())
    );
    let mut cache = attention::WindowKvCache::new(4, 128, 128, 128, None).unwrap();
    let frame = cache.prepare(&input, &input, 128).unwrap();
    cache.commit(&frame).unwrap();
    let (k, v) = cache.committed();
    assert_eq!(digest(k.iter().copied()), digest(input.iter().copied()));
    assert_eq!(digest(v.iter().copied()), digest(input.iter().copied()));
    let state = State::new(8).unwrap();
    let before = snapshot(&state);
    let mut buffers = Buffers::new(1, 0);
    buffers.input[12][2047] = bf16::NAN;
    assert!(buffers.prepare(&state, 1).is_err());
    assert_eq!(snapshot(&state), before);
    buffers.input[12][2047] = bf16::MAX;
    buffers.weights[2047 * 7 + 6] = bf16::MAX;
    assert!(buffers.prepare(&state, 1).is_err());
    assert_eq!(snapshot(&state), before);
}

#[test]
fn nonfinite_admission_and_zero_offset_reset_reject_before_publication() {
    let cache = attention::WindowKvCache::new(1, 2, 2, 8, Some(2)).unwrap();
    let mut history = convolution::History::new(2, 8).unwrap();
    let weights = [bf16::ONE; 14];
    let sentinel = bf16::from_f32(-7.);
    let mut candidate = [sentinel; 12];
    let mut output = [sentinel; 4];
    let original = kv_snapshot(&cache, 2);
    for bits in (0..=u16::MAX).filter(|word| word & 0x7f80 == 0x7f80) {
        let mut bad = [bf16::ONE; 4];
        bad[3] = bf16::from_bits(bits);
        assert!(cache.prepare(&bad, &[bf16::ONE; 4], 2).is_err());
        assert!(cache.prepare(&[bf16::ONE; 4], &bad, 2).is_err());
        assert!(
            history
                .prepare_chunk(&bad, &weights, &mut candidate, &mut output, 2)
                .is_err()
        );
        assert_eq!(candidate, [sentinel; 12]);
        assert_eq!(output, [sentinel; 4]);
    }
    assert_eq!(kv_snapshot(&cache, 2), original);
    assert_eq!(history.offset(), 0);
    let good = [bf16::ONE; 4];
    assert!(cache.prepare(&[], &[], 0).is_err());
    assert!(cache.prepare(&good[..3], &good, 2).is_err());
    assert!(cache.prepare(&good, &good[..3], 2).is_err());
    assert!(cache.prepare(&good, &good, usize::MAX).is_err());
    assert!(
        history
            .prepare_chunk(&good, &weights[..13], &mut candidate, &mut output, 2)
            .is_err()
    );
    assert!(
        history
            .prepare_chunk(&good, &weights, &mut candidate[..11], &mut output, 2)
            .is_err()
    );
    assert!(
        history
            .prepare_chunk(&good, &weights, &mut candidate, &mut output[..3], 2)
            .is_err()
    );
    assert_eq!(candidate, [sentinel; 12]);
    assert_eq!(output, [sentinel; 4]);
    let frame = history
        .prepare_chunk(&good, &weights, &mut candidate, &mut output, 2)
        .unwrap();
    history.commit(&frame).unwrap();
    assert_eq!(history.offset(), 2);
    let mut state = State::new(1).unwrap();
    let mut buffers = Buffers::new(1, 0);
    let frames = buffers.prepare(&state, 1).unwrap();
    state.reset().unwrap();
    let after_reset = snapshot(&state);
    assert!(state.commit(&frames.histories, &frames.caches, 1).is_err());
    assert_eq!(snapshot(&state), after_reset);
    step(&mut state, 0, 1);
    let full = snapshot(&state);
    assert!(state.commit(&[], &[], 1).is_err());
    assert_eq!(snapshot(&state), full);
}
