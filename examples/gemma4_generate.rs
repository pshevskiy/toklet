//! Closed raw/basic-hello E2B diagnostics; no app capability or context extension.
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    error::Error,
    fs::{File, OpenOptions},
    io::{self, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};
use toklet::{
    app_chat::{
        gemma4::{self, Codec},
        stream::ResponseLimits,
    },
    generation::{self, Input, Request, StopReason},
    mlx::gemma4::{ModalityPolicy, payload},
    model::gemma_packed::Model,
    session::{self, InferenceSession, Memory, Options},
};

const PROMPT: [u32; 4] = [2, 100, 257, 100];
const VOCAB: usize = 262144;
const ROWS: usize = 16;

const HELLO_PROMPT: [u32; 10] = [2, 105, 2364, 107, 10979, 106, 107, 105, 4368, 107];
const HELLO_RENDERED: &str = "<bos><|turn>user\nHi<turn|>\n<|turn>model\n";
const HELLO_MESSAGES: &[u8] = br#"[{"role":"user","content":"Hi"}]"#;
// Resident reserve, separate from streamed diagnostic files: bounded ledgers,
// JSON serialization, BufWriter, input/render/IDs and both decoded Strings.
const DIAGNOSTIC_ALLOWANCE: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Profile {
    Raw,
    BasicHello,
    BasicHelloStream,
}
impl Profile {
    fn parse(value: &str) -> io::Result<Self> {
        match value {
            "raw" => Ok(Self::Raw),
            "basic-hello" => Ok(Self::BasicHello),
            "basic-hello-stream" => Ok(Self::BasicHelloStream),
            _ => Err(io::Error::other("unsupported fixed generation profile")),
        }
    }
    fn plan(self) -> (usize, usize, usize, usize) {
        match self {
            Self::Raw => (ROWS, 4, 4, 7),
            Self::BasicHello => (32, 16, 10, 25),
            Self::BasicHelloStream => (48, 16, 10, 25),
        }
    }
    fn numerical_budget(self, total: usize) -> io::Result<usize> {
        if total == 0 || total > toklet::weights::INITIAL_MEMORY_LIMIT_BYTES {
            return Err(io::Error::other("combined diagnostic memory limit differs"));
        }
        match self {
            Self::Raw => Ok(total),
            Self::BasicHello | Self::BasicHelloStream => total
                .checked_sub(
                    gemma4::ALLOWANCE_BYTES
                        + DIAGNOSTIC_ALLOWANCE
                        + if self == Self::BasicHelloStream {
                            gemma4::STREAM_ALLOWANCE_BYTES
                        } else {
                            0
                        },
                )
                .filter(|remaining| *remaining != 0)
                .ok_or_else(|| io::Error::other("combined codec/diagnostic reserve exceeds limit")),
        }
    }
}

fn admitted_codec(
    root: &Path,
    profile: Profile,
    total: usize,
) -> Result<(usize, Option<Codec>), Box<dyn Error>> {
    let numerical_budget = profile.numerical_budget(total)?;
    let codec = if profile != Profile::Raw {
        Some(Codec::load(root, gemma4::ALLOWANCE_BYTES)?)
    } else {
        None
    };
    Ok((numerical_budget, codec))
}

fn metadata(path: PathBuf, expected: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 1024 * 1024 || format!("{:x}", Sha256::digest(&bytes)) != expected {
        return Err("pinned Gemma4 metadata identity differs".into());
    }
    Ok(bytes)
}

// This example alone streams diagnostics at the coarse sampling boundary.
// Capture precedes append, while the distribution is still valid. No numerical
// session API or kernel is modified, and no full vocabulary copy is retained.
struct Capture {
    output: BufWriter<File>,
    arrays: Vec<serde_json::Value>,
    bytes: usize,
    scenario: usize,
    scenario_rows: usize,
    error: Option<String>,
    plan: (usize, usize, usize, usize),
}
impl Capture {
    fn new(root: &Path, profile: Profile) -> io::Result<Self> {
        let mut arrays = Vec::new();
        arrays
            .try_reserve_exact(profile.plan().0)
            .map_err(io::Error::other)?;
        Ok(Self {
            output: BufWriter::with_capacity(
                64 * 1024,
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(root.join("arrays.bin"))?,
            ),
            arrays,
            bytes: 0,
            scenario: 0,
            scenario_rows: 0,
            error: None,
            plan: profile.plan(),
        })
    }
    fn row(&mut self, prefix: &[u32], values: &[f32]) -> io::Result<()> {
        if self.arrays.len() >= self.plan.0
            || self.scenario_rows >= self.plan.1
            || values.len() != VOCAB
            || !(self.plan.2..=self.plan.3).contains(&prefix.len())
        {
            return Err(io::Error::other("fixed diagnostic capture bound differs"));
        }
        self.arrays.push(serde_json::json!({
            "scenario":self.scenario,"step":self.scenario_rows,"prefix":prefix,
            "offset":self.bytes,"dtype":"<f4","shape":[VOCAB],
            "bytes":VOCAB*4,"completed":false
        }));
        let mut sha = Sha256::new();
        let mut finite = true;
        for value in values {
            let bytes = value.to_le_bytes();
            self.output.write_all(&bytes)?;
            sha.update(bytes);
            finite &= value.is_finite();
        }
        self.output.flush()?;
        self.bytes += VOCAB * 4;
        let item = self
            .arrays
            .last_mut()
            .expect("descriptor was just appended");
        item["sha256"] = serde_json::json!(format!("{:x}", sha.finalize()));
        item["completed"] = true.into();
        self.scenario_rows += 1;
        if !finite {
            return Err(io::Error::other("nonfinite captured distribution"));
        }
        Ok(())
    }
}

struct Observed<'a, S> {
    inner: &'a mut S,
    capture: &'a RefCell<Capture>,
}
impl<S: InferenceSession> InferenceSession for Observed<'_, S> {
    fn history(&self) -> &[u32] {
        self.inner.history()
    }
    fn evaluated_len(&self) -> usize {
        self.inner.evaluated_len()
    }
    fn pending_len(&self) -> usize {
        self.inner.pending_len()
    }
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn memory(&self) -> Memory {
        self.inner.memory()
    }
    fn logits(&self) -> Option<&[f32]> {
        self.inner.logits()
    }
    fn greedy_token(&self) -> Option<Result<u32, toklet::cpu::Error>> {
        let values = self.inner.logits()?;
        let mut capture = self.capture.borrow_mut();
        if capture.error.is_some() {
            return None;
        }
        if let Err(error) = capture.row(self.inner.history(), values) {
            capture.error = Some(error.to_string());
            return None;
        }
        self.inner.greedy_token()
    }
    fn append(&mut self, tokens: &[u32]) -> Result<(), session::Error> {
        self.inner.append(tokens)
    }
    fn set_tokens(&mut self, tokens: &[u32]) -> Result<(), session::Error> {
        self.inner.set_tokens(tokens)
    }
    fn evaluate_next_chunk(&mut self) -> Result<usize, session::Error> {
        self.inner.evaluate_next_chunk()
    }
    fn discard_pending(&mut self) {
        self.inner.discard_pending();
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
}

fn scenarios(
    session: &mut impl InferenceSession,
    capture: &RefCell<Capture>,
    record: &mut serde_json::Value,
) -> Result<(), Box<dyn Error>> {
    let memory = session.memory();
    record["memory"] = serde_json::to_value(memory)?;
    for (index, name) in ["uninterrupted", "split", "cancel_resume", "caller_stop"]
        .iter()
        .enumerate()
    {
        {
            let mut saved = capture.borrow_mut();
            saved.scenario = index;
            saved.scenario_rows = 0;
        }
        let mut observed = Observed {
            inner: session,
            capture,
        };
        let cancelled = AtomicBool::new(false);
        let mut emitted = Vec::with_capacity(4);
        let mut reports = Vec::with_capacity(2);
        let result = (|| -> Result<(), Box<dyn Error>> {
            let first = generation::generate(
                &mut observed,
                Request {
                    input: Input::Reset(&PROMPT),
                    max_new_tokens: if index == 1 { 2 } else { 4 },
                    // Explicit caller stop, not a qualified model EOS policy.
                    stop_tokens: if index == 3 { &[107] } else { &[] },
                },
                &cancelled,
                |event| {
                    emitted.push(event.token_id);
                    if index == 2 && emitted.len() == 2 {
                        cancelled.store(true, Ordering::Relaxed);
                    }
                    Ok(())
                },
            )?;
            let reason = first.reason;
            reports.push(first);
            if reason
                != match index {
                    2 => StopReason::Cancelled,
                    3 => StopReason::StopToken { token_id: 107 },
                    _ => StopReason::OutputLimit,
                }
            {
                return Err("first generation stop reason differs".into());
            }
            if index != 0 {
                cancelled.store(false, Ordering::Relaxed);
                reports.push(generation::generate(
                    &mut observed,
                    Request {
                        input: Input::Continue,
                        max_new_tokens: if index == 3 { 3 } else { 2 },
                        stop_tokens: &[],
                    },
                    &cancelled,
                    |event| {
                        emitted.push(event.token_id);
                        Ok(())
                    },
                )?);
            }
            if capture.borrow().scenario_rows != 4
                || observed.history().len() != 8
                || observed.evaluated_len() != 7
                || observed.pending_len() != 1
                || observed.logits().is_some()
                || observed.memory() != memory
            {
                return Err("final generation state or storage differs".into());
            }
            Ok(())
        })();
        record["scenarios"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "name":name,"reports":reports,"emitted_ids":emitted,
                "history":observed.history(),"evaluated":observed.evaluated_len(),
                "pending":observed.pending_len(),"logits_available":observed.logits().is_some(),
                "completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string)
            }));
        result?;
    }
    Ok(())
}

// This is the actual two-call policy, shared with small scripted-session tests.
// A natural stop is terminal; only an output limit permits continuation.
fn basic_requests(
    session: &mut impl InferenceSession,
    split: bool,
    reports: &mut Vec<generation::Report>,
    mut emit: impl FnMut(generation::TokenEvent) -> io::Result<()>,
) -> Result<(), generation::Error> {
    let cancelled = AtomicBool::new(false);
    let first = generation::generate(
        session,
        Request {
            input: Input::Reset(&HELLO_PROMPT),
            max_new_tokens: if split { 2 } else { 16 },
            stop_tokens: &gemma4::STOP_IDS,
        },
        &cancelled,
        &mut emit,
    )?;
    let remaining = 16 - first.sampled_tokens;
    let resume = split && first.reason == StopReason::OutputLimit && remaining != 0;
    reports.push(first);
    if resume {
        reports.push(generation::generate(
            session,
            Request {
                input: Input::Continue,
                max_new_tokens: remaining,
                stop_tokens: &gemma4::STOP_IDS,
            },
            &cancelled,
            emit,
        )?);
    }
    Ok(())
}

// None is the single terminal flush. OutputLimit at the internal split is
// deliberately invisible to delivery, preserving a whole byte-fallback run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum StreamSchedule {
    Uninterrupted,
    Split,
    Cancel,
}
fn stream_requests(
    session: &mut impl InferenceSession,
    schedule: StreamSchedule,
    reports: &mut Vec<generation::Report>,
    mut deliver: impl FnMut(Option<generation::TokenEvent>) -> io::Result<()>,
) -> Result<(), generation::Error> {
    if schedule == StreamSchedule::Cancel {
        let cancelled = AtomicBool::new(false);
        let mut visible = 0;
        let report = generation::generate(
            session,
            Request {
                input: Input::Reset(&HELLO_PROMPT),
                max_new_tokens: 16,
                stop_tokens: &gemma4::STOP_IDS,
            },
            &cancelled,
            |event| {
                deliver(Some(event))?;
                visible += 1;
                if visible == 2 {
                    cancelled.store(true, Ordering::Relaxed);
                }
                Ok(())
            },
        )?;
        reports.push(report);
    } else {
        basic_requests(
            session,
            schedule == StreamSchedule::Split,
            reports,
            |event| deliver(Some(event)),
        )?;
    }
    deliver(None).map_err(generation::Error::Output)
}

fn basic_scenarios(
    session: &mut impl InferenceSession,
    codec: &Codec,
    numerical_budget: usize,
    capture: &RefCell<Capture>,
    record: &mut serde_json::Value,
    streaming: bool,
) -> Result<(), Box<dyn Error>> {
    let rendered = gemma4::render(
        HELLO_MESSAGES,
        gemma4::RenderOptions {
            enable_thinking: false,
            add_generation_prompt: true,
        },
    )?;
    let prompt = codec.encode_raw(&rendered, false)?;
    if rendered != HELLO_RENDERED || prompt != HELLO_PROMPT || codec.stop_ids() != [1, 106, 50] {
        return Err("frozen basic-hello prompt or stop policy differs".into());
    }
    record["rendered_prompt"] = rendered.into();
    record["stop_ids"] = serde_json::to_value(codec.stop_ids())?;
    let memory = session.memory();
    let combined = memory
        .admitted_bytes
        .checked_add(codec.allowance_bytes())
        .and_then(|n| n.checked_add(DIAGNOSTIC_ALLOWANCE))
        .and_then(|n| {
            n.checked_add(if streaming {
                gemma4::STREAM_ALLOWANCE_BYTES
            } else {
                0
            })
        })
        .ok_or("combined diagnostic admission overflow")?;
    if combined > toklet::weights::INITIAL_MEMORY_LIMIT_BYTES {
        return Err("combined diagnostic admission exceeds limit".into());
    }
    record["memory"] = serde_json::to_value(memory)?;
    record["numerical_budget_bytes"] = numerical_budget.into();
    record["combined_admitted_bytes"] = combined.into();
    let names: &[&str] = if streaming {
        &["uninterrupted", "split", "cancel"]
    } else {
        &["uninterrupted", "split"]
    };
    for (index, name) in names.iter().enumerate() {
        {
            let mut saved = capture.borrow_mut();
            saved.scenario = index;
            saved.scenario_rows = 0;
        }
        let mut observed = Observed {
            inner: session,
            capture,
        };
        let mut emitted = Vec::with_capacity(16);
        let mut reports = Vec::with_capacity(2);
        let mut chunks = Vec::with_capacity(17);
        let mut stream = if streaming {
            Some(codec.response_stream(
                ResponseLimits::new(16, gemma4::MAX_TEXT_BYTES)?,
                gemma4::STREAM_ALLOWANCE_BYTES,
            )?)
        } else {
            None
        };
        let owner_before = stream.as_ref().map(|s| s.allocated_bytes()).transpose()?;
        let result = if let Some(stream) = stream.as_mut() {
            stream_requests(
                &mut observed,
                match index {
                    0 => StreamSchedule::Uninterrupted,
                    1 => StreamSchedule::Split,
                    _ => StreamSchedule::Cancel,
                },
                &mut reports,
                |event| {
                    // Retain the delivered token even if its text mutation fails.
                    if let Some(event) = event {
                        emitted.push(event.token_id);
                    }
                    let chunk = match event {
                        Some(event) => stream.push(&[event.token_id])?,
                        None => stream.finish()?,
                    };
                    // Fixed at <=16 callbacks plus one terminal flush; copy the
                    // borrowed chunk before the next mutation or comparison.
                    chunks.push(serde_json::json!({"terminal":event.is_none(),"text":chunk}));
                    Ok(())
                },
            )
        } else {
            basic_requests(&mut observed, index == 1, &mut reports, |event| {
                emitted.push(event.token_id);
                Ok(())
            })
        };
        // Capture the valid state even when generation or diagnostic writes fail.
        let sampled = observed.history().get(HELLO_PROMPT.len()..).unwrap_or(&[]);
        let mut state = serde_json::json!({
            "name":name,"reports":reports,"emitted_ids":emitted,
            "sampled_ids":sampled,"history":observed.history(),
            "evaluated":observed.evaluated_len(),"pending":observed.pending_len(),
            "logits_available":observed.logits().is_some(),
            "completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string)
        });
        if let Some(stream) = stream.as_ref() {
            state["stream"] = serde_json::json!({
                "chunks":chunks,"consumed_tokens":stream.consumed_tokens(),
                "emitted_bytes":stream.emitted_bytes(),"finished":stream.is_finished(),
                "failed":stream.is_failed(),"owner_before_bytes":owner_before,
                "owner_after_bytes":stream.allocated_bytes().ok(),
            });
        }
        record["scenarios"].as_array_mut().unwrap().push(state);
        result?;
        let rows = capture.borrow().scenario_rows;
        let count: usize = reports.iter().map(|r| r.sampled_tokens).sum();
        let final_reason = reports
            .last()
            .ok_or("missing basic generation report")?
            .reason;
        if rows != count
            || !(1..=16).contains(&count)
            || sampled.len() != count
            || observed.history()[..10] != HELLO_PROMPT
            || observed.evaluated_len() + 1 != observed.history().len()
            || observed.pending_len() != 1
            || observed.logits().is_some()
            || observed.memory() != memory
            || emitted
                != sampled
                    [..count - usize::from(matches!(final_reason, StopReason::StopToken { .. }))]
        {
            return Err("basic generation state or diagnostic storage differs".into());
        }
        let row = record["scenarios"]
            .as_array_mut()
            .unwrap()
            .last_mut()
            .unwrap();
        row["visible_decode_keep_special"] = codec.decode(&emitted, false)?.into();
        row["visible_decode_skip_special"] = codec.decode(&emitted, true)?.into();
        row["finish_reason"] = serde_json::to_value(final_reason)?;
        if let Some(stream) = stream.as_ref() {
            let actual: String = chunks
                .iter()
                .map(|row| row["text"].as_str().unwrap())
                .collect();
            if actual != row["visible_decode_skip_special"].as_str().unwrap()
                || stream.consumed_tokens() != emitted.len()
                || stream.emitted_bytes() != actual.len()
                || !stream.is_finished()
                || stream.is_failed()
                || Some(stream.allocated_bytes()?) != owner_before
                || chunks.len() != emitted.len() + 1
                || chunks.last().unwrap()["terminal"] != true
            {
                return Err("stream delivery or diagnostic storage differs".into());
            }
        }
    }
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[path = "gemma4_generate/hybrid.rs"]
mod hybrid;
#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    Scalar,
    Hybrid,
}
impl Backend {
    fn parse(value: &str) -> io::Result<Self> {
        match value {
            "scalar" => Ok(Self::Scalar),
            "hybrid" => Ok(Self::Hybrid),
            _ => Err(io::Error::other("unsupported fixed backend")),
        }
    }
    fn check_available(self) -> io::Result<()> {
        if self == Self::Hybrid
            && !cfg!(all(
                feature = "metal",
                target_os = "macos",
                target_arch = "aarch64"
            ))
        {
            return Err(io::Error::other(
                "hybrid diagnostic requires Apple Silicon Mac and metal feature",
            ));
        }
        Ok(())
    }
}
fn run_session(
    session: &mut impl InferenceSession,
    codec: Option<&Codec>,
    budget: usize,
    capture: &RefCell<Capture>,
    record: &mut serde_json::Value,
    profile: Profile,
) -> Result<(), Box<dyn Error>> {
    if let Some(codec) = codec {
        basic_scenarios(
            session,
            codec,
            budget,
            capture,
            record,
            profile == Profile::BasicHelloStream,
        )
    } else {
        scenarios(session, capture, record)
    }
}

fn execute(
    root: &Path,
    capture: &RefCell<Capture>,
    record: &mut serde_json::Value,
    profile: Profile,
    backend: Backend,
) -> Result<(), Box<dyn Error>> {
    backend.check_available()?;
    let total = toklet::weights::INITIAL_MEMORY_LIMIT_BYTES;
    let (numerical_budget, codec) = admitted_codec(root, profile, total)?;
    let config = metadata(
        root.join("config.json"),
        "6397cb6eca41b911d1dcab74e17941351057bd759284052a2331918ff6f9246c",
    )?;
    let index = metadata(
        root.join("model.safetensors.index.json"),
        "edb157dbf495e23f37377af4a628a9ad13c4ee7937f93ccb36ec9e9a19940f16",
    )?;
    // Independently frozen LFS digest, never supplied by the consumed container.
    let digest = [
        0x03, 0x8e, 0x39, 0xa3, 0x7a, 0x76, 0x67, 0x37, 0x3d, 0x2c, 0x39, 0x91, 0x37, 0x54, 0x46,
        0xb1, 0x0c, 0x96, 0xae, 0x1d, 0x71, 0x7a, 0x68, 0x67, 0x48, 0x70, 0x34, 0x3d, 0xb3, 0x76,
        0xb7, 0x6e,
    ];
    if backend == Backend::Hybrid {
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        return hybrid::execute(
            root,
            capture,
            record,
            profile,
            numerical_budget,
            codec.as_ref(),
            (&config, &index, digest),
        );
        #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
        unreachable!("backend checked before reading assets");
    }
    let store = payload::load(
        &config,
        &index,
        File::open(root.join("model.safetensors"))?,
        payload::LoadOptions {
            container_bytes: 3550670554,
            container_sha256: digest,
            memory_limit_bytes: numerical_budget,
            modality: ModalityPolicy::TextOnly,
        },
    )?;
    let model = Model::from_e2b(&store)?;
    record["model_bytes_including_store"] = model.model_bytes().into();
    if model.model_bytes() > numerical_budget {
        return Err("model exceeds remaining combined numerical allowance".into());
    }
    let mut session = model.new_generation_session(Options {
        capacity: 32,
        prefill_chunk_size: 4,
        position_offset: 0,
        memory_limit_bytes: numerical_budget,
    })?;
    run_session(
        &mut session,
        codec.as_ref(),
        numerical_budget,
        capture,
        record,
        profile,
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1).peekable();
    let usage = "usage: gemma4_generate [--backend scalar|hybrid] [--profile raw|basic-hello|basic-hello-stream] CHECKPOINT_DIRECTORY NEW_OUTPUT_DIRECTORY";
    let mut profile = Profile::Raw;
    let mut backend = Backend::Scalar;
    let mut seen_profile = false;
    let mut seen_backend = false;
    while let Some(value) = args.peek().and_then(|v| v.to_str()) {
        match value {
            "--profile" if !seen_profile => {
                args.next();
                profile = Profile::parse(args.next().ok_or(usage)?.to_str().ok_or(usage)?)?;
                seen_profile = true;
            }
            "--backend" if !seen_backend => {
                args.next();
                backend = Backend::parse(args.next().ok_or(usage)?.to_str().ok_or(usage)?)?;
                seen_backend = true;
            }
            value if value.starts_with("--") => return Err(usage.into()),
            _ => break,
        }
    }
    let root = PathBuf::from(args.next().ok_or(usage)?);
    let output = PathBuf::from(args.next().ok_or(usage)?);
    if args.next().is_some() {
        return Err(usage.into());
    }
    std::fs::create_dir(&output)?;
    let capture = RefCell::new(Capture::new(&output, profile)?);
    let mut record = serde_json::json!({
        "format":"toklet-gemma4-raw-generation-probe-v1","passed":false,
        "repository":"mlx-community/gemma-4-e2b-it-4bit",
        "revision":"238767527555cb75a05732a84dff5d6ba0dd6809",
        "prompt":PROMPT,"scenarios":[],"primary_error":null,"endpoint_errors":[],
        "max_executable_positions":32,"local_cache_slots":512,"global_cache_slots":32,
        "memory_limit_bytes":toklet::weights::INITIAL_MEMORY_LIMIT_BYTES,
        "diagnostic_byte_limit":ROWS*VOCAB*4,"numerical_comparison_complete":false,
        "original_bf16_qualified":false,"tokenizer_chat_or_model_stop_policy_qualified":false,
        "metal_qualified":false,"performance_qualified":false,"quality_qualified":false,
        "device_fit_qualified":false,"app_eligibility_changed":false
    });
    if profile != Profile::Raw {
        record["format"] = "toklet-gemma4-basic-chat-generation-probe-v1".into();
        record["profile"] = "basic-hello".into();
        record["prompt"] = serde_json::to_value(HELLO_PROMPT)?;
        record["diagnostic_byte_limit"] = (profile.plan().0 * VOCAB * 4).into();
        record["tokenizer_allowance_bytes"] = gemma4::ALLOWANCE_BYTES.into();
        record["diagnostic_allowance_bytes"] = DIAGNOSTIC_ALLOWANCE.into();
        record["formatted_fixture_sha256"] =
            "dab866a1e86e8bed3179e39383fa2f826222f9b1708aa2c2e6df66e78348627e".into();
        record["text_streaming_qualified"] = false.into();
    }
    if profile == Profile::BasicHelloStream {
        record["format"] = "toklet-gemma4-basic-chat-stream-probe-v1".into();
        record["profile"] = "basic-hello-stream".into();
        record["stream_allowance_bytes"] = gemma4::STREAM_ALLOWANCE_BYTES.into();
    }
    if backend == Backend::Hybrid {
        record["execution_backend"] = "hybrid".into();
    }
    if let Err(error) = execute(&root, &capture, &mut record, profile, backend) {
        record["primary_error"] = capture
            .borrow()
            .error
            .as_deref()
            .map_or_else(|| error.to_string(), str::to_owned)
            .into();
    }
    let mut capture = capture.into_inner();
    record["capture_error"] = serde_json::to_value(&capture.error)?;
    if let Err(error) = capture.output.flush() {
        record["endpoint_errors"]
            .as_array_mut()
            .unwrap()
            .push(error.to_string().into());
    }
    record["arrays"] = serde_json::to_value(&capture.arrays)?;
    record["completed_array_bytes"] = capture.bytes.into();
    record["passed"] = (record["primary_error"].is_null()
        && capture.error.is_none()
        && record["endpoint_errors"].as_array().unwrap().is_empty()
        && match profile {
            Profile::Raw => capture.arrays.len() == ROWS,
            Profile::BasicHello | Profile::BasicHelloStream => {
                !capture.arrays.is_empty()
                    && capture.arrays.len() <= profile.plan().0
                    && record["scenarios"].as_array().unwrap().len()
                        == if profile == Profile::BasicHelloStream {
                            3
                        } else {
                            2
                        }
            }
        })
    .into();
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join("observation.json"))?;
    serde_json::to_writer_pretty(&mut file, &record)?;
    file.write_all(b"\n")?;
    file.flush()?;
    println!(
        "{}",
        serde_json::json!({"passed":record["passed"],"arrays":capture.arrays.len(),"primary_error":record["primary_error"],"capture_error":record["capture_error"]})
    );
    if record["passed"] != true {
        return Err("raw E2B generation probe failed; observations retained".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct Temporary(PathBuf);
    impl Temporary {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "toklet-gemma-generate-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temporary {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn finite_rows_preserve_exact_bytes_offsets_and_prewrite_bounds() {
        let temp = Temporary::new();
        let mut capture = Capture::new(&temp.0, Profile::Raw).unwrap();
        let mut values = vec![0.0; VOCAB];
        values[..4].copy_from_slice(&[-0.0, 1.25, f32::MIN_POSITIVE, f32::from_bits(1)]);
        capture.row(&PROMPT, &values).unwrap();
        capture.row(&[2, 100, 257, 100, 107], &values).unwrap();
        let raw = std::fs::read(temp.0.join("arrays.bin")).unwrap();
        assert_eq!(raw.len(), 2 * VOCAB * 4);
        for (index, item) in capture.arrays.iter().enumerate() {
            let row = &raw[index * VOCAB * 4..(index + 1) * VOCAB * 4];
            assert_eq!(item["offset"], index * VOCAB * 4);
            assert_eq!(item["completed"], true);
            assert_eq!(item["sha256"], format!("{:x}", Sha256::digest(row)));
            for (bytes, value) in row.chunks_exact(4).zip(&values) {
                assert_eq!(bytes, value.to_le_bytes());
            }
        }
        assert!(capture.row(&[2; 3], &values).is_err());
        assert!(capture.row(&[2; 8], &values).is_err());
        capture.scenario_rows = 4;
        assert!(capture.row(&PROMPT, &values).is_err());
        assert_eq!(capture.arrays.len(), 2);
        capture.scenario_rows = 0;
        capture.arrays.resize(ROWS, serde_json::Value::Null);
        assert!(capture.row(&PROMPT, &values).is_err());
        assert_eq!(capture.bytes, raw.len());
        assert_eq!(std::fs::read(temp.0.join("arrays.bin")).unwrap(), raw);
    }

    #[test]
    fn invalid_shape_rejects_before_capture_nonfinite_retains_complete_raw() {
        let temp = Temporary::new();
        let mut capture = Capture::new(&temp.0, Profile::Raw).unwrap();
        assert!(capture.row(&PROMPT, &[0.0]).is_err());
        assert!(capture.arrays.is_empty());
        let mut values = vec![0.0; VOCAB];
        values[42] = f32::from_bits(0x7fc00042);
        assert!(capture.row(&PROMPT, &values).is_err());
        assert_eq!(capture.arrays.len(), 1);
        assert_eq!(capture.arrays[0]["completed"], true);
        let raw = std::fs::read(temp.0.join("arrays.bin")).unwrap();
        assert_eq!(raw.len(), VOCAB * 4);
        assert_eq!(&raw[42 * 4..43 * 4], &values[42].to_le_bytes());
        assert_eq!(
            capture.arrays[0]["sha256"],
            format!("{:x}", Sha256::digest(&raw))
        );
    }

    #[test]
    fn actual_write_failure_retains_incomplete_descriptor() {
        let temp = Temporary::new();
        let path = temp.0.join("readonly.bin");
        File::create_new(&path).unwrap();
        let mut capture = Capture {
            output: BufWriter::with_capacity(8, File::open(&path).unwrap()),
            arrays: Vec::new(),
            bytes: 0,
            scenario: 0,
            scenario_rows: 0,
            error: None,
            plan: Profile::Raw.plan(),
        };
        let error = capture.row(&PROMPT, &vec![0.0; VOCAB]).unwrap_err();
        assert!(error.raw_os_error().is_some());
        assert_eq!(capture.arrays.len(), 1);
        assert_eq!(capture.arrays[0]["completed"], false);
        assert!(capture.arrays[0].get("sha256").is_none());
        assert_eq!(capture.bytes, 0);
        assert_eq!(capture.scenario_rows, 0);
        assert_eq!(std::fs::metadata(path).unwrap().len(), 0);
    }

    struct Scripted {
        history: Vec<u32>,
        evaluated: usize,
        selected: [u32; 16],
    }
    impl InferenceSession for Scripted {
        fn history(&self) -> &[u32] {
            &self.history
        }
        fn evaluated_len(&self) -> usize {
            self.evaluated
        }
        fn pending_len(&self) -> usize {
            self.history.len() - self.evaluated
        }
        fn capacity(&self) -> usize {
            32
        }
        fn vocab_size(&self) -> usize {
            VOCAB
        }
        fn memory(&self) -> Memory {
            Memory {
                weights_bytes: 0,
                kv_bytes: 0,
                scratch_bytes: 0,
                history_bytes: 128,
                logits_bytes: 0,
                metadata_bytes: 0,
                metadata_allowance_bytes: 0,
                admitted_bytes: 128,
            }
        }
        fn logits(&self) -> Option<&[f32]> {
            None
        }
        fn greedy_token(&self) -> Option<Result<u32, toklet::cpu::Error>> {
            (self.evaluated == self.history.len())
                .then(|| Ok(self.selected[self.history.len() - 10]))
        }
        fn append(&mut self, tokens: &[u32]) -> Result<(), session::Error> {
            self.history.extend_from_slice(tokens);
            Ok(())
        }
        fn set_tokens(&mut self, tokens: &[u32]) -> Result<(), session::Error> {
            self.reset();
            self.append(tokens)
        }
        fn evaluate_next_chunk(&mut self) -> Result<usize, session::Error> {
            let count = self.pending_len();
            self.evaluated = self.history.len();
            Ok(count)
        }
        fn discard_pending(&mut self) {
            self.history.truncate(self.evaluated);
        }
        fn reset(&mut self) {
            self.history.clear();
            self.evaluated = 0;
        }
    }

    #[test]
    fn basic_split_preserves_early_stop_final_stop_and_sampled_budget() {
        for split in [false, true] {
            for stop in [Some((0, 106)), Some((1, 50)), Some((15, 1)), None] {
                let mut selected = [7; 16];
                if let Some((index, id)) = stop {
                    selected[index] = id;
                }
                let mut session = Scripted {
                    history: Vec::with_capacity(32),
                    evaluated: 0,
                    selected,
                };
                let mut reports = Vec::with_capacity(2);
                let mut emitted = Vec::with_capacity(16);
                basic_requests(&mut session, split, &mut reports, |event| {
                    emitted.push(event.token_id);
                    Ok(())
                })
                .unwrap();
                let count = stop.map_or(16, |(index, _)| index + 1);
                assert_eq!(reports.len(), if split && count > 2 { 2 } else { 1 });
                assert_eq!(
                    reports.iter().map(|r| r.sampled_tokens).sum::<usize>(),
                    count
                );
                assert_eq!(emitted, selected[..count - usize::from(stop.is_some())]);
                assert_eq!(&session.history[..10], &HELLO_PROMPT);
                assert_eq!(&session.history[10..], &selected[..count]);
                assert_eq!(session.evaluated, 10 + count - 1);
                assert_eq!(session.pending_len(), 1);
                assert_eq!(
                    reports.last().unwrap().reason,
                    stop.map_or(StopReason::OutputLimit, |(_, token_id)| {
                        StopReason::StopToken { token_id }
                    })
                );
            }
        }
    }

    #[test]
    fn basic_failure_retains_completed_report_and_pending_sample() {
        let mut session = Scripted {
            history: Vec::with_capacity(32),
            evaluated: 0,
            selected: [7; 16],
        };
        let mut reports = Vec::with_capacity(2);
        let mut emitted = 0;
        let error = basic_requests(&mut session, true, &mut reports, |_| {
            emitted += 1;
            if emitted == 3 {
                Err(io::Error::other("synthetic output failure"))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
        assert!(matches!(error, generation::Error::Output(_)));
        assert!(error.to_string().contains("synthetic output failure"));
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].sampled_tokens, 2);
        assert_eq!(session.history.len(), 13);
        assert_eq!(session.evaluated, 12);
        assert_eq!(session.pending_len(), 1);
    }

    #[test]
    fn closed_profile_and_combined_reserve_reject_before_asset_access() {
        assert!(Profile::parse("other").is_err());
        assert!(Profile::parse("basic-hello ").is_err());
        assert!(Profile::parse("basic-hello").is_ok());
        assert!(Profile::parse("raw").is_ok());
        let temp = Temporary::new();
        let reserve = gemma4::ALLOWANCE_BYTES + DIAGNOSTIC_ALLOWANCE;
        for total in [
            0,
            reserve - 1,
            reserve,
            toklet::weights::INITIAL_MEMORY_LIMIT_BYTES + 1,
        ] {
            let error = match admitted_codec(&temp.0, Profile::BasicHello, total) {
                Err(error) => error,
                Ok(_) => panic!("invalid reserve admitted"),
            };
            assert!(error.to_string().contains("combined"));
        }
        assert_eq!(
            Profile::BasicHello.numerical_budget(reserve + 1).unwrap(),
            1
        );
        assert_eq!(Profile::Raw.numerical_budget(reserve).unwrap(), reserve);
        let fixture = include_bytes!("../tests/fixtures/gemma4-basic-chat-math-v1/math.json");
        assert_eq!(
            format!("{:x}", Sha256::digest(fixture)),
            "dab866a1e86e8bed3179e39383fa2f826222f9b1708aa2c2e6df66e78348627e"
        );
        let witness: serde_json::Value = serde_json::from_slice(fixture).unwrap();
        assert_eq!(witness["prompt"], serde_json::json!(HELLO_PROMPT));
        assert_eq!(witness["rendered_prompt"], HELLO_RENDERED);
        assert_eq!(witness["stop_ids"], serde_json::json!(gemma4::STOP_IDS));
    }

    #[test]
    fn stream_policy_preserves_fallback_run_across_split_until_one_flush() {
        let tokenizer = tokenizers::Tokenizer::from_bytes(include_bytes!(
            "../tests/fixtures/gemma4-stream-v1/tokenizer.json"
        ))
        .unwrap();
        let a = tokenizer.token_to_id("<0x41>").unwrap();
        let ff = tokenizer.token_to_id("<0xFF>").unwrap();
        for schedule in [StreamSchedule::Uninterrupted, StreamSchedule::Split] {
            let mut selected = [106; 16];
            selected[..4].copy_from_slice(&[a, a, ff, 106]);
            let mut session = Scripted {
                history: Vec::with_capacity(32),
                evaluated: 0,
                selected,
            };
            let mut reports = Vec::with_capacity(2);
            let mut pending = Vec::new();
            let mut flushed = false;
            let mut text = String::new();
            stream_requests(&mut session, schedule, &mut reports, |event| {
                assert!(!flushed, "delivery after terminal flush");
                if let Some(event) = event {
                    pending.push(event.token_id);
                } else {
                    text = tokenizer.decode(&pending, true).unwrap();
                    flushed = true;
                }
                Ok(())
            })
            .unwrap();
            assert!(flushed);
            assert_eq!(pending, [a, a, ff]);
            assert_eq!(text, "���");
            assert_eq!(
                reports.len(),
                if schedule == StreamSchedule::Split {
                    2
                } else {
                    1
                }
            );
            assert_eq!(
                reports.last().unwrap().reason,
                StopReason::StopToken { token_id: 106 }
            );
            assert_eq!(session.evaluated, 13);
            assert_eq!(session.pending_len(), 1);
        }
    }

    #[test]
    fn stream_cancel_finishes_visible_prefix_once_without_continuation() {
        let mut session = Scripted {
            history: Vec::with_capacity(32),
            evaluated: 0,
            selected: [7; 16],
        };
        let mut reports = Vec::with_capacity(2);
        let mut delivery = Vec::new();
        stream_requests(
            &mut session,
            StreamSchedule::Cancel,
            &mut reports,
            |event| {
                delivery.push(event.map(|e| e.token_id));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(delivery, [Some(7), Some(7), None]);
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].reason, StopReason::Cancelled);
        assert_eq!(
            (reports[0].sampled_tokens, reports[0].emitted_tokens),
            (2, 2)
        );
        assert_eq!(
            (
                session.history.len(),
                session.evaluated,
                session.pending_len()
            ),
            (12, 11, 1)
        );
    }

    #[test]
    fn stream_callback_failure_preserves_prefix_pending_and_completed_report() {
        let mut session = Scripted {
            history: Vec::with_capacity(32),
            evaluated: 0,
            selected: [7; 16],
        };
        let mut reports = Vec::with_capacity(2);
        let mut retained = Vec::new();
        let error = stream_requests(&mut session, StreamSchedule::Split, &mut reports, |event| {
            retained.push(event.map(|e| e.token_id));
            if retained.len() == 3 {
                return Err(io::Error::other("retained publication failure"));
            }
            Ok(())
        })
        .unwrap_err();
        assert!(error.to_string().contains("retained publication failure"));
        assert_eq!(retained, [Some(7), Some(7), Some(7)]);
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].sampled_tokens, 2);
        assert_eq!(
            (
                session.history.len(),
                session.evaluated,
                session.pending_len()
            ),
            (13, 12, 1)
        );
    }

    #[test]
    fn stream_terminal_failure_is_primary_and_keeps_completed_reports() {
        let mut selected = [7; 16];
        selected[3] = 106;
        let mut session = Scripted {
            history: Vec::with_capacity(32),
            evaluated: 0,
            selected,
        };
        let mut reports = Vec::with_capacity(2);
        let mut retained = Vec::new();
        let error = stream_requests(&mut session, StreamSchedule::Split, &mut reports, |event| {
            retained.push(event.map(|e| e.token_id));
            if event.is_none() {
                return Err(io::Error::other("terminal sink already sealed"));
            }
            Ok(())
        })
        .unwrap_err();
        assert!(error.to_string().contains("terminal sink already sealed"));
        assert_eq!(retained, [Some(7), Some(7), Some(7), None]);
        assert_eq!(reports.len(), 2);
        assert_eq!(
            reports.last().unwrap().reason,
            StopReason::StopToken { token_id: 106 }
        );
        assert_eq!(
            (
                session.history.len(),
                session.evaluated,
                session.pending_len()
            ),
            (14, 13, 1)
        );
    }

    #[test]
    fn stream_reserve_rejects_before_asset_access_and_preserves_old_budgets() {
        let missing = Path::new("/nonexistent-toklet-stream-generation-admission-test");
        let reserve =
            gemma4::ALLOWANCE_BYTES + DIAGNOSTIC_ALLOWANCE + gemma4::STREAM_ALLOWANCE_BYTES;
        assert!(Profile::parse("basic-hello-stream").is_ok());
        for total in [
            0,
            reserve - 1,
            reserve,
            toklet::weights::INITIAL_MEMORY_LIMIT_BYTES + 1,
        ] {
            let error = admitted_codec(missing, Profile::BasicHelloStream, total)
                .err()
                .unwrap();
            assert!(error.to_string().contains("combined"));
        }
        assert_eq!(
            Profile::BasicHelloStream
                .numerical_budget(reserve + 1)
                .unwrap(),
            1
        );
        assert_eq!(
            Profile::BasicHello.numerical_budget(reserve + 1).unwrap(),
            gemma4::STREAM_ALLOWANCE_BYTES + 1
        );
        assert_eq!(
            Profile::Raw.numerical_budget(reserve + 1).unwrap(),
            reserve + 1
        );
    }

    #[test]
    fn basic_capture_plan_rejects_outside_prefix_and_row_bounds_before_write() {
        let temp = Temporary::new();
        let mut capture = Capture::new(&temp.0, Profile::BasicHello).unwrap();
        let values = vec![0.0; VOCAB];
        capture.row(&[2; 10], &values).unwrap();
        capture.row(&[2; 25], &values).unwrap();
        let bytes = capture.bytes;
        for prefix in [9, 26] {
            assert!(capture.row(&vec![2; prefix], &values).is_err());
            assert_eq!(capture.arrays.len(), 2);
            assert_eq!(capture.bytes, bytes);
        }
        capture.scenario_rows = 16;
        assert!(capture.row(&[2; 10], &values).is_err());
        assert_eq!(capture.arrays.len(), 2);
        assert_eq!(capture.bytes, bytes);
        capture.scenario_rows = 0;
        capture.arrays.resize(32, serde_json::Value::Null);
        assert!(capture.row(&[2; 10], &values).is_err());
        assert_eq!(capture.arrays.len(), 32);
        assert_eq!(capture.bytes, bytes);
        assert_eq!(
            std::fs::metadata(temp.0.join("arrays.bin")).unwrap().len(),
            bytes as u64
        );
    }
}
