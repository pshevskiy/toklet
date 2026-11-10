//! Prepared BF16 state transitions; not a complete LFM3 graph or model benchmark.
use half::bf16;
use sha2::{Digest, Sha256};
use smallvec::SmallVec;
use std::{hint::black_box, time::Instant};
use toklet::model::{Error, lfm3_state::State, short_conv7_bf16, window_cache_bf16};

struct Work {
    input: Vec<bf16>,
    keys: Vec<bf16>,
    values: Vec<bf16>,
    weights: Vec<bf16>,
    candidates: Vec<Vec<bf16>>,
    output: Vec<bf16>,
}
impl Work {
    fn new(rows: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let raw = include_bytes!("../../tests/fixtures/lfm3-conv-bf16-v1/weights.bf16le");
        if format!("{:x}", Sha256::digest(raw))
            != "f305ef6e0054d9128004f7ac9c4eccb7dd5bf0e0f1bc3143ef39de3355ed9e62"
        {
            return Err("Pinned source coefficient banks differ".into());
        }
        Ok(Self {
            input: (0..rows * 2048)
                .map(|i| bf16::from_f32(((i * 13 % 257) as f32 - 128.) / 32.))
                .collect(),
            keys: (0..4 * rows * 128)
                .map(|i| bf16::from_f32(((i * 17 % 257) as f32 - 128.) / 32.))
                .collect(),
            values: (0..4 * rows * 128)
                .map(|i| bf16::from_f32(((i * 19 % 257) as f32 - 128.) / 32.))
                .collect(),
            weights: raw
                .chunks_exact(2)
                .map(|b| bf16::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            candidates: (0..13).map(|_| vec![bf16::ZERO; 6 * 2048]).collect(),
            output: vec![bf16::ZERO; rows * 2048],
        })
    }
    fn step(&mut self, state: &mut State, tokens: usize) -> Result<[u128; 4], Error> {
        let begin = Instant::now();
        let mut histories: SmallVec<[short_conv7_bf16::Prepared<'_>; 13]> = SmallVec::new();
        let mut caches: SmallVec<[window_cache_bf16::PreparedKv<'_>; 11]> = SmallVec::new();
        for (i, (owner, candidate)) in state
            .histories()
            .iter()
            .zip(&mut self.candidates)
            .enumerate()
        {
            histories.push(owner.prepare_chunk(
                black_box(&self.input[..tokens * 2048]),
                &self.weights[i * 2048 * 7..(i + 1) * 2048 * 7],
                candidate,
                &mut self.output[..tokens * 2048],
                tokens,
            )?);
        }
        let history_ns = begin.elapsed().as_nanos();
        let phase = Instant::now();
        for cache in state.caches() {
            let frame = cache.prepare(
                &self.keys[..4 * tokens * 128],
                &self.values[..4 * tokens * 128],
                tokens,
            )?;
            let view = cache.view(&frame)?;
            black_box(view.segments());
            black_box(view.attention_shape(16, 0.125));
            for head in 0..4 {
                black_box(view.head(head)?);
            }
            black_box(view.visible_range(state.offset() + tokens - 1)?);
            caches.push(frame);
        }
        let kv_ns = phase.elapsed().as_nanos();
        let phase = Instant::now();
        state.commit(&histories, &caches, tokens)?;
        let publication_ns = phase.elapsed().as_nanos();
        let total_ns = begin.elapsed().as_nanos();
        black_box(&self.output);
        Ok([history_ns, kv_ns, publication_ns, total_ns])
    }
    fn staging_bytes(&self) -> usize {
        (self.input.capacity()
            + self.keys.capacity()
            + self.values.capacity()
            + self.output.capacity())
            * 2
    }
}
fn fingerprint(values: &[bf16]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err("release build required".into());
    }
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 || args[0] != "--tokens" || args[2] != "--offset" {
        return Err("usage: lfm3_bf16_state_probe --tokens 1|32|128 --offset 0|511|1023".into());
    }
    let tokens: usize = args[1].parse()?;
    let offset: usize = args[3].parse()?;
    if !matches!(tokens, 1 | 32 | 128) || !matches!(offset, 0 | 511 | 1023) {
        return Err("unsupported tokens or offset".into());
    }
    let begin = Instant::now();
    let mut state = State::new(2048)?;
    let construction_ms = begin.elapsed().as_secs_f64() * 1000.;
    let mut work = Work::new(offset.max(tokens))?;
    let repeats = (32 / tokens).max(1);
    let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::with_capacity(10));
    let mut raw = Vec::with_capacity(10 * repeats);
    let mut first = [0.; 4];
    for trial in 0..14 {
        let repetitions = if trial == 0 { 1 } else { repeats };
        let mut total = [0_u128; 4];
        for _ in 0..repetitions {
            state.reset()?;
            if offset > 0 {
                work.step(&mut state, offset)?;
            }
            let observed = work.step(&mut state, tokens)?;
            assert_eq!(state.offset(), offset + tokens);
            for (sum, value) in total.iter_mut().zip(observed) {
                *sum += value;
            }
            if trial >= 4 {
                raw.push(observed);
            }
        }
        let measured = total.map(|v| v as f64 / 1e6 / repetitions as f64);
        if trial == 0 {
            first = measured;
        } else if trial >= 4 {
            for (sample, value) in samples.iter_mut().zip(measured) {
                sample.push(value);
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "format":"toklet-lfm3-bf16-state-micro-v1","storage":"BF16","capacity":2048,
            "history_owners":13,"full_kv_owners":6,"sliding_kv_owners":5,"window":256,
            "heads":4,"head_dim":128,"channels":2048,"tokens":tokens,"offset":offset,
            "construction_ms":construction_ms,"first_history_kv_publication_total_ms":first,
            "warm_history_ms":samples[0],"warm_kv_ms":samples[1],
            "warm_publication_ms":samples[2],"warm_state_total_ms":samples[3],
            "warmups":3,"measured_trials":10,"repetitions_per_trial":repeats,"raw_transition_ns":raw,
            "state_planned_bytes":State::planned_bytes(2048)?,"state_actual_bytes":state.allocated_bytes(),
            "coefficient_bytes":work.weights.capacity()*2,
            "candidate_bytes":work.candidates.iter().map(|c|c.capacity()*2).sum::<usize>(),
            "caller_input_output_kv_staging_bytes":work.staging_bytes(),
            "probe_descriptor_bytes":std::mem::size_of::<Work>()+work.candidates.capacity()*std::mem::size_of::<Vec<bf16>>(),
            "prepared_frame_inline_storage_bytes":std::mem::size_of::<SmallVec<[short_conv7_bf16::Prepared<'_>;13]>>()
                +std::mem::size_of::<SmallVec<[window_cache_bf16::PreparedKv<'_>;11]>>(),
            "embedded_fixture_bytes":372736,
            "input_sha256":fingerprint(&work.input),"keys_sha256":fingerprint(&work.keys),
            "values_sha256":fingerprint(&work.values),
            "scope":"Prepared13histories+11KVproposals/views and jointpublication. Sourcecoefficients, synthetic shared input/KVstaging; not graph order or model scratch. Reset/depthseed outside transition clocks; construction separately named. No projections/norms/attention/codec/sampling/complete model throughput.",
            "model_execution_qualified":false,"quality_adoption_qualified":false
        })
    );
    Ok(())
}
