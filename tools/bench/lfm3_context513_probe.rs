//! Explicit real-model numerical and state campaign. Ordinary suites need no weights.
use std::{
    fs::File,
    hint::black_box,
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
    time::Instant,
};

use half::bf16;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use toklet::{
    model::{
        Error,
        lfm3::{AttentionView, Model, Observation},
    },
    weights::{INITIAL_MEMORY_LIMIT_BYTES, load_lfm3},
};

const CONTRACT: &[u8] = include_bytes!("../../tests/fixtures/lfm3-context513-v1/contract.json");
const PRODUCER: &[u8] = include_bytes!("../reference/lfm3_context513_fixture.py");

const LOSS_CONTRACT: &[u8] =
    include_bytes!("../../tests/fixtures/lfm3-context1025-v1/contract.json");
const LOSS_PRODUCER: &[u8] = include_bytes!("../reference/lfm3_context1025_fixture.py");

#[derive(Clone, Copy, PartialEq, Eq)]
enum Campaign {
    Rollover,
    LossContext,
}

impl Campaign {
    fn fixture_inputs(self) -> (&'static [u8], &'static [u8], &'static str, &'static [usize]) {
        match self {
            Self::Rollover => (
                CONTRACT,
                PRODUCER,
                "toklet-lfm3-context513-source-v1",
                &[257],
            ),
            Self::LossContext => (
                LOSS_CONTRACT,
                LOSS_PRODUCER,
                "toklet-lfm3-context1025-source-v1",
                &[257, 1024],
            ),
        }
    }
}

#[derive(Deserialize)]
struct Fixture {
    format: String,
    tokens: Vec<u32>,
    schedules: Vec<Schedule>,
    source_bytes: u64,
    source_sha256: String,
    contract_sha256: String,
    producer_sha256: String,
    fixture_completed: bool,
}
#[derive(Deserialize)]
struct Schedule {
    chunks: Vec<usize>,
    records: Vec<Record>,
}
#[derive(Deserialize)]
struct Record {
    layer: Option<usize>,
    chunk: usize,
    stage: String,
    offset: u64,
    words: usize,
    sha256: String,
    key_start: Option<usize>,
    logical_keys: Option<usize>,
    window: Option<usize>,
}

fn sha(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}
fn bytes(values: &[bf16], output: &mut Vec<u8>) {
    output.clear();
    output.extend(values.iter().flat_map(|value| value.to_le_bytes()));
}
fn exact_logit_hash(values: &[bf16]) -> String {
    let mut hash = Sha256::new();
    for v in values {
        hash.update(v.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}
fn view_expected(record: &Record, offset: usize, tokens: usize) -> Option<AttentionView> {
    let key_start = record.key_start?;
    let old = offset.checked_sub(key_start).unwrap();
    let first = record
        .window
        .map_or(old, |window| old.min(window - key_start % window));
    Some(AttentionView {
        key_start,
        logical_keys: record.logical_keys.unwrap(),
        window: record.window,
        segment_rows: [first, old - first, tokens],
    })
}
fn argmax(values: impl Iterator<Item = f32>) -> usize {
    let mut best = (0, f32::NEG_INFINITY);
    for (index, value) in values.enumerate() {
        if value > best.1 {
            best = (index, value);
        }
    }
    best.0
}

fn compare_row(
    actual: Observation<'_>,
    expected: &Record,
    offset: usize,
    tokens: usize,
    source: &mut File,
    actual_bytes: &mut Vec<u8>,
    reference: &mut Vec<u8>,
) -> Result<Value, Box<dyn std::error::Error>> {
    assert_eq!(
        (actual.layer, actual.stage, actual.values.len()),
        (expected.layer, expected.stage.as_str(), expected.words)
    );
    let view = view_expected(expected, offset, tokens);
    assert_eq!(actual.attention_view, view);
    bytes(actual.values, actual_bytes);
    let digest = sha(actual_bytes);
    let mut differences = 0;
    let mut maximum_abs = 0.0_f32;
    let mut first = Vec::new();
    let mut argmax_reference = None;
    let mut argmax_native = None;
    if digest != expected.sha256 || actual.stage == "logits" {
        reference.resize(actual_bytes.len(), 0);
        source.seek(SeekFrom::Start(expected.offset))?;
        source.read_exact(reference)?;
        assert_eq!(sha(reference), expected.sha256);
        let words = || {
            reference
                .chunks_exact(2)
                .map(|v| bf16::from_le_bytes(v.try_into().unwrap()))
        };
        for (index, (got, wanted)) in actual.values.iter().zip(words()).enumerate() {
            if got.to_bits() != wanted.to_bits() {
                differences += 1;
                maximum_abs = maximum_abs.max((got.to_f32() - wanted.to_f32()).abs());
                if first.len() < 4 {
                    first.push(json!([index, got.to_bits(), wanted.to_bits()]));
                }
            }
        }
        if actual.stage == "logits" {
            argmax_reference = Some(argmax(words().map(bf16::to_f32)));
            argmax_native = Some(argmax(actual.values.iter().map(|v| v.to_f32())));
        }
    }
    Ok(json!({
        "layer": actual.layer, "stage": actual.stage, "words": expected.words,
        "source_sha256": expected.sha256, "native_sha256": digest,
        "word_differences": differences, "max_absolute_difference": maximum_abs,
        "first_differences": first, "argmax_source": argmax_reference, "argmax_native": argmax_native,
        "attention_view": view.map(|v| json!({"key_start": v.key_start,
            "logical_keys": v.logical_keys, "window": v.window, "segment_rows": v.segment_rows})),
    }))
}

fn verify(
    model: &mut Model,
    input: &Path,
    output: &Path,
    campaign: Campaign,
) -> Result<(), Box<dyn std::error::Error>> {
    let fixture: Fixture = serde_json::from_reader(File::open(input.join("fixture.json"))?)?;
    let (contract_bytes, producer, format, abort_offsets) = campaign.fixture_inputs();
    let contract: Value = serde_json::from_slice(contract_bytes)?;
    let length: u32 = contract["tokens"]["length"].as_u64().unwrap().try_into()?;
    assert!(fixture.fixture_completed);
    assert_eq!(fixture.format, format);
    assert_eq!(fixture.contract_sha256, sha(contract_bytes));
    assert_eq!(fixture.producer_sha256, sha(producer));
    assert_eq!(
        fixture.source_sha256,
        contract["checkpoint_sha256"].as_str().unwrap()
    );
    assert_eq!(
        fixture.tokens,
        std::iter::once(124894)
            .chain((1..length).map(|i| (i * 3253 + 41) % 124000))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        serde_json::to_value(
            fixture
                .schedules
                .iter()
                .map(|s| &s.chunks)
                .collect::<Vec<_>>()
        )?,
        contract["schedules"]
    );
    std::fs::create_dir(output)?;
    let mut source = File::open(input.join("source.bf16le"))?;
    assert_eq!(source.metadata()?.len(), fixture.source_bytes);
    let mut evidence = BufWriter::new(File::create(output.join("observations.jsonl"))?);
    let mut actual_bytes = Vec::new();
    let mut reference = Vec::new();
    let mut session = model.session()?;
    let mut records = 0;
    let mut words = 0;
    let mut differing_records = 0;
    let mut differing_words = 0;
    let mut two_old_segments = 0;
    let mut aborts = 0;
    for (schedule_id, schedule) in fixture.schedules.iter().enumerate() {
        session.reset()?;
        let mut logits_out = BufWriter::new(File::create(
            output.join(format!("logits-{schedule_id}.bf16le")),
        )?);
        let mut offset = 0;
        for (chunk_id, &count) in schedule.chunks.iter().enumerate() {
            let end = offset + count;
            let tokens = &fixture.tokens[offset..end];
            let aborted = if abort_offsets.contains(&offset) {
                let before = exact_logit_hash(session.logits().unwrap());
                let generation = session.generation();
                let mut candidate = None;
                let failure = session.forward_observed(tokens, |event| {
                    if event.stage == "logits" {
                        candidate = Some(exact_logit_hash(event.values));
                        Err(Error::InvalidConfig("context probe abort"))
                    } else {
                        Ok(())
                    }
                });
                assert!(matches!(
                    failure,
                    Err(Error::InvalidConfig("context probe abort"))
                ));
                assert_eq!(session.history(), &fixture.tokens[..offset]);
                assert_eq!(session.generation(), generation);
                assert_eq!(exact_logit_hash(session.logits().unwrap()), before);
                aborts += 1;
                candidate
            } else {
                None
            };
            let mut expected = schedule.records.iter().filter(|row| row.chunk == chunk_id);
            let mut outputs = Vec::with_capacity(61);
            let mut diagnostic_error = None;
            let start = Instant::now();
            let result = session.forward_observed(tokens, |event| {
                let record = expected.next().expect("missing source event");
                match compare_row(
                    event,
                    record,
                    offset,
                    count,
                    &mut source,
                    &mut actual_bytes,
                    &mut reference,
                ) {
                    Ok(value) => outputs.push(value),
                    Err(error) => {
                        diagnostic_error = Some(error);
                        return Err(Error::InvalidConfig("context reference read"));
                    }
                }
                Ok(())
            });
            let elapsed_ns = start.elapsed().as_nanos();
            if let Some(error) = diagnostic_error {
                return Err(error);
            }
            let logits = result?;
            assert!(expected.next().is_none());
            assert_eq!(outputs.len(), 61);
            if let Some(aborted) = aborted {
                assert_eq!(exact_logit_hash(logits), aborted);
            }
            bytes(logits, &mut actual_bytes);
            logits_out.write_all(&actual_bytes)?;
            assert_eq!(session.history(), &fixture.tokens[..end]);
            let mut chunk_differences = 0;
            for value in &outputs {
                let differences = value["word_differences"].as_u64().unwrap();
                records += 1;
                words += value["words"].as_u64().unwrap();
                differing_words += differences;
                differing_records += usize::from(differences != 0);
                chunk_differences += differences;
                two_old_segments += usize::from(
                    value["attention_view"]["segment_rows"][1]
                        .as_u64()
                        .unwrap_or(0)
                        != 0,
                );
            }
            let row = json!({
                "schedule": schedule_id, "chunk": chunk_id, "offset": offset, "end": end,
                "forward_observed_ns": elapsed_ns, "observations": outputs,
            });
            writeln!(evidence, "{row}")?;
            evidence.flush()?;
            println!(
                "{}",
                json!({"schedule": schedule_id, "end": end,
                "word_differences": chunk_differences, "forward_observed_ns": elapsed_ns})
            );
            offset = end;
        }
        logits_out.flush()?;
    }
    assert_eq!(aborts, 2);
    assert!(two_old_segments > 0);
    let result = json!({"observation_count": records, "words": words, "differing_words": differing_words,
        "differing_observations": differing_records, "two_old_segments": two_old_segments,
        "late_abort_and_retry": aborts, "exact_pass": differing_words == 0});
    println!("{result}");
    std::fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    if campaign == Campaign::LossContext && differing_words == 0 {
        std::fs::write(output.join("timing-start"), b"ordinary\n")?;
        let mut samples = Vec::with_capacity(14);
        for offset in length..length + 14 {
            let token = (offset * 3253 + 41) % 124000;
            let started = Instant::now();
            let logits = session.forward(black_box(std::slice::from_ref(&token)))?;
            let elapsed_ns = started.elapsed().as_nanos();
            samples.push(
                json!({"offset":offset,"token":token,"elapsed_ns":elapsed_ns,
                "logits_sha256":exact_logit_hash(logits)}),
            );
        }
        assert_eq!(session.history().len(), length as usize + 14);
        let timing = json!({"first":samples[0],"warmups":&samples[1..4],"samples":&samples[4..]});
        std::fs::write(
            output.join("timing.json"),
            serde_json::to_vec_pretty(&timing)?,
        )?;
    }
    Ok(())
}

fn benchmark(model: &mut Model, reverse: bool) -> Result<(), Box<dyn std::error::Error>> {
    let tokens = std::iter::once(124894)
        .chain((1..271).map(|i| (i * 3253 + 41) % 124000))
        .collect::<Vec<_>>();
    let mut session = model.session()?;
    let prefix_started = Instant::now();
    for chunk in tokens[..257].chunks(128) {
        session.forward(chunk)?;
    }
    let prefix_ns = prefix_started.elapsed().as_nanos();
    let mut modes = [false, true];
    if reverse {
        modes.reverse();
    }
    // A late observer failure leaves the fixed prefix in place. All timed
    // calls perform full graph/tied-head work, with publication aborted.
    let before = exact_logit_hash(session.logits().unwrap());
    let generation = session.generation();
    for detailed in modes {
        let mut samples = [0; 14];
        for duration in &mut samples {
            let started = Instant::now();
            let result = session.forward_observed(black_box(&tokens[257..258]), |event| {
                if detailed {
                    black_box(event.values);
                    black_box(event.attention_view);
                }
                if event.stage == "logits" {
                    Err(Error::InvalidConfig("prepared benchmark abort"))
                } else {
                    Ok(())
                }
            });
            *duration = started.elapsed().as_nanos();
            assert!(matches!(
                result,
                Err(Error::InvalidConfig("prepared benchmark abort"))
            ));
            assert_eq!(session.generation(), generation);
        }
        println!(
            "{}",
            json!({"offset":257, "prefix_ns":prefix_ns, "detailed":detailed,
            "first_ns":samples[0], "warmup_ns":&samples[1..4], "samples_ns":&samples[4..]})
        );
    }
    assert_eq!(session.history(), &tokens[..257]);
    assert_eq!(exact_logit_hash(session.logits().unwrap()), before);
    // Ordinary publication advances the prefix. Preserve exact depth and
    // input per sample and report it as a separate progression.
    let mut ordinary = Vec::with_capacity(14);
    for (offset, token) in tokens.iter().copied().enumerate().skip(257) {
        let started = Instant::now();
        let logits = session.forward(black_box(std::slice::from_ref(&token)))?;
        ordinary.push(
            json!({"offset":offset,"token":token,"elapsed_ns":started.elapsed().as_nanos(),
            "logits_sha256":exact_logit_hash(logits)}),
        );
    }
    assert_eq!(session.history(), tokens);
    println!(
        "{}",
        json!({"ordinary_progression":true,"prefix_ns":prefix_ns,
        "first":ordinary[0],"warmups":&ordinary[1..4],"samples":&ordinary[4..]})
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err("release build required".into());
    }
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let [model, mode, extra @ ..] = args.as_slice() else {
        return Err("usage: lfm3_context513_probe MODEL verify|verify-1025 SOURCE OUTPUT | MODEL forward | MODEL reverse".into());
    };
    let started = Instant::now();
    let loaded = load_lfm3(Path::new(model), 2048, 128, INITIAL_MEMORY_LIMIT_BYTES)?;
    let mut model = loaded.model;
    println!(
        "{}",
        json!({"load_ns":started.elapsed().as_nanos(),
        "checkpoint_sha256":loaded.report.source_sha256, "admitted_bytes":model.memory().total_bytes})
    );
    match (mode.as_str(), extra) {
        ("verify" | "verify-1025", [input, output]) => {
            let campaign = if mode == "verify" {
                Campaign::Rollover
            } else {
                Campaign::LossContext
            };
            verify(&mut model, Path::new(input), Path::new(output), campaign)
        }
        ("forward", []) => benchmark(&mut model, false),
        ("reverse", []) => benchmark(&mut model, true),
        _ => Err("invalid probe mode or argument count".into()),
    }
}
