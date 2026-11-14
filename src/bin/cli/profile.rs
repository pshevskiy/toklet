//! Explicit bounded diagnostic over the existing official BF16/G2 profiler.
use super::{invalid, utf8};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    ffi::OsString,
    fs::File,
    io::{self, Read},
    path::{Path, PathBuf},
};
use toklet::app_chat::gemma4;

const WORKLOAD_LIMIT: usize = 16 * 1024;
const HELP: &str =
    "Usage: toklet profile --model PATH --workload PATH --evaluation ORD --output NEW_DIR
       [--submission per-tile|batched-single-row]

Explicit official Gemma BF16/G2 diagnostic on macOS arm64 with the metal feature.
The frozen O1 workload must include expected output; capacity512/chunk32 are fixed.
ORD selects one completed evaluation in the expected request (1..=512).
Runs ordinary then profiled generation sequentially on one authenticated owner,
checks exact outputs/state and every selected logit bit, and writes bounded
profile.json, inventory.json and a final manifest.json in a NEW directory.
The selected profiler reserves64MiB inside the whole6GiB engine budget.
Submission defaults to per-tile. Explicit batching captures schema2 groups and
compares the candidate against ordinary G2; it is not default adoption.
Host times include diagnostic witness work; this is not acceptance timing,
ordinary generation adoption, or UI/device/quality qualification.
Exit0 complete verified artifacts,2 malformed arguments/workload,1 runtime or
export failure. No fallback, implicit download, or overwrite.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Submission {
    PerTile,
    BatchedSingleRow,
}

#[derive(Debug)]
struct Args {
    model: PathBuf,
    workload: PathBuf,
    evaluation: usize,
    output: PathBuf,
    submission: Submission,
}

fn parse(values: Vec<OsString>) -> io::Result<Args> {
    let (mut model, mut workload, mut evaluation, mut output) = (None, None, None, None);
    let mut submission = Submission::PerTile;
    let mut seen = HashSet::new();
    let mut values = values.into_iter();
    while let Some(flag) = values.next() {
        let flag = utf8(flag)?;
        if !seen.insert(flag.clone()) {
            return Err(invalid(format!("duplicate {flag}")));
        }
        let value = values
            .next()
            .ok_or_else(|| invalid(format!("missing value for {flag}")))?;
        match flag.as_str() {
            "--model" if !value.is_empty() => model = Some(PathBuf::from(value)),
            "--workload" if !value.is_empty() => workload = Some(PathBuf::from(value)),
            "--output" if !value.is_empty() => output = Some(PathBuf::from(value)),
            "--submission" => {
                submission = match utf8(value)?.as_str() {
                    "per-tile" => Submission::PerTile,
                    "batched-single-row" => Submission::BatchedSingleRow,
                    _ => return Err(invalid("submission must be per-tile or batched-single-row")),
                };
            }
            "--evaluation" => {
                let text = utf8(value)?;
                if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
                    return Err(invalid("evaluation must be ASCII digits"));
                }
                let ordinal: usize = text.parse().map_err(invalid)?;
                if !(1..=512).contains(&ordinal) {
                    return Err(invalid("evaluation must be within1..=512"));
                }
                evaluation = Some(ordinal);
            }
            _ => return Err(invalid(format!("unknown or empty profile option {flag}"))),
        }
    }
    Ok(Args {
        model: model.ok_or_else(|| invalid("profile requires --model PATH"))?,
        workload: workload.ok_or_else(|| invalid("profile requires --workload PATH"))?,
        evaluation: evaluation.ok_or_else(|| invalid("profile requires --evaluation ORD"))?,
        output: output.ok_or_else(|| invalid("profile requires --output NEW_DIR"))?,
        submission,
    })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedOutput {
    tokens: Vec<u32>,
    history: usize,
    committed: usize,
    pending: usize,
    reason: u32,
    stop_token: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkloadDocument {
    schema_version: u32,
    workload_id: String,
    conversation_json: String,
    conversation_sha256: String,
    input_token_count: usize,
    input_tokens_sha256: String,
    maximum_output_tokens: usize,
    capacity: usize,
    chunk: usize,
    cache_policy: String,
    expected_output: ExpectedOutput,
}

impl WorkloadDocument {
    fn sampled_tokens(&self) -> usize {
        self.expected_output.tokens.len() + usize::from(self.expected_output.reason == 2)
    }
    fn evaluations(&self) -> usize {
        self.input_token_count.div_ceil(32) + self.sampled_tokens() - 1
    }
    fn validate(&self) -> io::Result<()> {
        let id = self.workload_id.as_bytes();
        if self.schema_version != 1
            || id.is_empty()
            || id.len() > 64
            || !(id[0].is_ascii_lowercase() || id[0].is_ascii_digit())
            || !id
                .iter()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'_' | b'-'))
            || !digest(&self.conversation_sha256)
            || !digest(&self.input_tokens_sha256)
            || self.conversation_json.is_empty()
            || self.conversation_json.len() > 8192
            || sha(self.conversation_json.as_bytes()) != self.conversation_sha256
            || !(1..=512).contains(&self.input_token_count)
            || !(1..=256).contains(&self.maximum_output_tokens)
            || self.input_token_count + self.maximum_output_tokens > 512
            || self.capacity != 512
            || self.chunk != 32
            || self.cache_policy != "reset_chat_each_request"
        {
            return Err(invalid("invalid frozen profile workload"));
        }
        let expected = &self.expected_output;
        let emitted = expected.tokens.len();
        if emitted == 0
            || emitted > self.maximum_output_tokens
            || !match expected.reason {
                1 => emitted == self.maximum_output_tokens && expected.stop_token == u32::MAX,
                2 => {
                    emitted < self.maximum_output_tokens
                        && gemma4::STOP_IDS.contains(&expected.stop_token)
                }
                _ => false,
            }
            || expected.history != self.input_token_count + self.sampled_tokens()
            || expected.history > 512
            || expected.pending != 1
            || expected.committed.checked_add(1) != Some(expected.history)
        {
            return Err(invalid("invalid frozen profile output/state"));
        }
        // The ordinary codec's typed parser rejects duplicate/unknown message
        // fields and nested/non-string values before any model is loaded.
        gemma4::render(
            self.conversation_json.as_bytes(),
            gemma4::RenderOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            },
        )?;
        Ok(())
    }
}

#[derive(Debug)]
struct Workload {
    document: WorkloadDocument,
    sha256: String,
}

fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn load_workload(path: &Path) -> io::Result<Workload> {
    let mut raw = Vec::new();
    File::open(path)?
        .take(WORKLOAD_LIMIT as u64 + 1)
        .read_to_end(&mut raw)?;
    if raw.is_empty() || raw.len() > WORKLOAD_LIMIT {
        return Err(invalid("profile workload exceeds16KiB"));
    }
    let document: WorkloadDocument = serde_json::from_slice(&raw).map_err(invalid)?;
    document.validate()?;
    Ok(Workload {
        document,
        sha256: sha(&raw),
    })
}

#[cfg(any(
    test,
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
use toklet::session::{self, InferenceSession, Memory};

/// Diagnostic-only static delegate; it never selects or reproduces arithmetic.
#[cfg(any(
    test,
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
struct Witness<'a, S> {
    inner: &'a mut S,
    ordinal: usize,
    evaluations: usize,
    selected: &'a mut [f32],
    captured: bool,
}

#[cfg(any(
    test,
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
impl<'a, S: InferenceSession> Witness<'a, S> {
    fn new(inner: &'a mut S, ordinal: usize, selected: &'a mut [f32]) -> io::Result<Self> {
        if !(1..=512).contains(&ordinal) || selected.len() != inner.vocab_size() {
            return Err(invalid("invalid selected-logit witness bounds"));
        }
        Ok(Self {
            inner,
            ordinal,
            evaluations: 0,
            selected,
            captured: false,
        })
    }
}

#[cfg(any(
    test,
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
impl<S: InferenceSession> InferenceSession for Witness<'_, S> {
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
        self.inner.greedy_token()
    }
    fn append(&mut self, tokens: &[u32]) -> Result<(), session::Error> {
        self.inner.append(tokens)
    }
    fn set_tokens(&mut self, tokens: &[u32]) -> Result<(), session::Error> {
        self.inner.set_tokens(tokens)
    }
    fn evaluate_next_chunk(&mut self) -> Result<usize, session::Error> {
        let rows = self.inner.evaluate_next_chunk()?;
        self.evaluations += 1;
        if self.evaluations == self.ordinal {
            let values = self
                .inner
                .logits()
                .ok_or_else(|| session::Error::Backend("selected logits unavailable".into()))?;
            if values.len() != self.selected.len() || values.iter().any(|v| !v.is_finite()) {
                return Err(session::Error::Backend("invalid selected logits".into()));
            }
            self.selected.copy_from_slice(values);
            self.captured = true;
        }
        Ok(rows)
    }
    fn discard_pending(&mut self) {
        self.inner.discard_pending();
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn try_reset(&mut self) -> Result<(), session::Error> {
        self.inner.try_reset()
    }
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod execution {
    use super::{super::CANCELLED, *};
    use serde::Serialize;
    use serde_json::{Value, json};
    use std::{
        fs,
        io::Write,
        mem::{size_of, size_of_val},
        sync::atomic::Ordering,
    };
    use toklet::{
        generation::{self, Input, Request, StopReason},
        metal::gemma_affine::{Bf16CaptureState, Bf16ProfileOptions, OfficialBf16},
        session::Options,
    };

    const LIMIT: usize = 6 * 1024 * 1024 * 1024;
    const CALLER: usize = 16 * 1024 * 1024;
    const HELPER: usize = 32 * 1024 * 1024;
    const DIAGNOSTIC: usize = 64 * 1024 * 1024;
    const OWNER_LIMIT: usize = LIMIT - CALLER - HELPER;
    const JSON_LIMIT: usize = 1024 * 1024;

    fn require(ok: bool, message: &str) -> io::Result<()> {
        if ok {
            Ok(())
        } else {
            Err(io::Error::other(message))
        }
    }
    fn reserve<T>(count: usize) -> io::Result<Vec<T>> {
        let mut result = Vec::new();
        result.try_reserve_exact(count).map_err(io::Error::other)?;
        require(result.capacity() <= count, "caller buffer overreservation")?;
        Ok(result)
    }
    fn zeros(count: usize) -> io::Result<Vec<f32>> {
        let mut result = reserve(count)?;
        result.resize(count, 0.);
        Ok(result)
    }

    /// The bound is checked before every extension, including JSON escaping.
    struct BoundedBuffer {
        bytes: Vec<u8>,
        limit: usize,
    }
    impl BoundedBuffer {
        fn new(limit: usize) -> io::Result<Self> {
            Ok(Self {
                bytes: reserve(limit)?,
                limit,
            })
        }
        fn encode(&mut self, value: &impl Serialize) -> io::Result<()> {
            self.bytes.clear();
            serde_json::to_writer(&mut *self, value).map_err(io::Error::other)?;
            self.write_all(b"\n")
        }
        fn publish(&self, folder: &Path, name: &str) -> io::Result<Value> {
            let mut file = File::create_new(folder.join(name))?;
            file.write_all(&self.bytes)?;
            file.flush()?;
            file.sync_all()?;
            require(
                file.metadata()?.len() == self.bytes.len() as u64,
                "export length differs",
            )?;
            Ok(json!({"file":name,"sha256":sha(&self.bytes),"bytes":self.bytes.len()}))
        }
    }
    impl Write for BoundedBuffer {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self
                .bytes
                .len()
                .checked_add(bytes.len())
                .is_none_or(|length| length > self.limit)
            {
                return Err(io::Error::other("diagnostic JSON byte limit exceeded"));
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn combined(admitted: usize, reserved: usize) -> io::Result<usize> {
        let owner = admitted
            .checked_add(reserved)
            .ok_or_else(|| io::Error::other("owner admission overflow"))?;
        require(owner <= OWNER_LIMIT, "owner admission exceeds budget")?;
        let total = owner
            .checked_add(CALLER + HELPER)
            .ok_or_else(|| io::Error::other("combined admission overflow"))?;
        require(total <= LIMIT, "combined admission exceeds6GiB")?;
        Ok(total)
    }

    fn check_report(
        report: &generation::Report,
        work: &WorkloadDocument,
        input: &[u32],
        output: &[u32],
        history: &[u32],
        evaluated: usize,
        pending: usize,
    ) -> io::Result<()> {
        let expected = &work.expected_output;
        let reason_matches = match report.reason {
            StopReason::OutputLimit => expected.reason == 1,
            StopReason::StopToken { token_id } => {
                expected.reason == 2 && expected.stop_token == token_id
            }
            StopReason::Cancelled => false,
        };
        require(
            reason_matches
                && output == expected.tokens
                && history.len() == expected.history
                && evaluated == expected.committed
                && pending == expected.pending
                && history.starts_with(input)
                && history[input.len()..input.len() + output.len()] == *output
                && (expected.reason != 2 || history.last() == Some(&expected.stop_token)),
            "generation output/history/state differs from frozen workload",
        )?;
        require(
            report.reused_tokens == 0
                && report.prefill_tokens == input.len()
                && report.decode_evaluations == work.sampled_tokens() - 1
                && report.sampled_tokens == work.sampled_tokens()
                && report.emitted_tokens == output.len()
                && report.history_tokens == history.len()
                && report.evaluated_tokens == evaluated
                && report.pending_tokens == pending,
            "completed generation work counts differ",
        )?;
        require(
            [
                report.prefill_seconds,
                report.decode_seconds,
                report.sampling_seconds,
                report.total_seconds,
            ]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.)
                && report
                    .first_token_seconds
                    .is_some_and(|v| v.is_finite() && v >= 0. && v <= report.total_seconds),
            "invalid completed generation clocks",
        )
    }

    fn report_value(report: &generation::Report) -> io::Result<Value> {
        let mut value = serde_json::to_value(report).map_err(io::Error::other)?;
        // Core Report intentionally skips this field in its existing Serde API.
        value["decode_evaluations"] = json!(report.decode_evaluations);
        Ok(value)
    }
    fn token_sha(ids: &[u32]) -> String {
        let mut hash = Sha256::new();
        for id in ids {
            hash.update(id.to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }

    pub(super) fn run(args: &Args, workload: &Workload) -> io::Result<Value> {
        // Exclusive directory ownership permits removal of our own unsuccessful
        // final marker; pre-existing paths are never changed.
        fs::create_dir(&args.output)?;
        let result = execute(args, workload);
        if let Err(error) = &result {
            let _ = fs::remove_file(args.output.join("manifest.json"));
            if let Ok(mut buffer) = BoundedBuffer::new(4096) {
                let message: String = error.to_string().chars().take(256).collect();
                if buffer
                    .encode(&json!({"format":"toklet-gemma-profile-failure-v1",
                        "passed":false,"diagnostic_only":true,"error":message}))
                    .is_ok()
                {
                    let _ = buffer.publish(&args.output, "failure.json");
                }
            }
        }
        result
    }

    fn execute(args: &Args, workload: &Workload) -> io::Result<Value> {
        let work = &workload.document;
        // All diagnostic checkpoints and serialization buffers are admitted
        // outside generation. No full logit trace or second model is retained.
        let mut plain_logits = zeros(gemma4::VOCAB)?;
        let mut profiled_logits = zeros(gemma4::VOCAB)?;
        let mut plain_output = reserve(work.maximum_output_tokens)?;
        let mut profiled_output = reserve(work.maximum_output_tokens)?;
        let mut plain_history = reserve(512)?;
        let mut profile_json = BoundedBuffer::new(JSON_LIMIT)?;
        let mut inventory_json = BoundedBuffer::new(JSON_LIMIT)?;
        let mut manifest_json = BoundedBuffer::new(64 * 1024)?;
        ctrlc::set_handler(|| CANCELLED.store(true, Ordering::Relaxed))
            .map_err(io::Error::other)?;
        let owner = OfficialBf16::load(&args.model, 512, OWNER_LIMIT).map_err(io::Error::other)?;
        require(
            owner.memory().load_admitted_bytes <= OWNER_LIMIT,
            "load admission exceeds owner budget",
        )?;
        let model = owner.model().map_err(io::Error::other)?;
        let input = owner.codec().encode_chat(
            work.conversation_json.as_bytes(),
            gemma4::RenderOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            },
        )?;
        require(
            input.len() == work.input_token_count && token_sha(&input) == work.input_tokens_sha256,
            "actual encoded input count/hash differs",
        )?;
        let options = Options {
            capacity: 512,
            prefill_chunk_size: 32,
            position_offset: 0,
            memory_limit_bytes: OWNER_LIMIT,
        };
        let reserved = owner.memory().session_reserved_bytes;
        let request = || Request {
            input: Input::Reset(&input),
            max_new_tokens: work.maximum_output_tokens,
            stop_tokens: owner.codec().stop_ids(),
        };
        let (plain_report, plain_admitted) = {
            let mut session = owner
                .new_g2_session(&model, options)
                .map_err(io::Error::other)?;
            let admitted = session.memory().admitted_bytes;
            combined(admitted, reserved)?;
            let mut witness = Witness::new(&mut session, args.evaluation, &mut plain_logits)?;
            let report = generation::generate(&mut witness, request(), &CANCELLED, |event| {
                require(
                    plain_output.len() < work.maximum_output_tokens,
                    "plain output bound exceeded",
                )?;
                plain_output.push(event.token_id);
                Ok(())
            })
            .map_err(io::Error::other)?;
            require(
                witness.captured && witness.evaluations == work.evaluations(),
                "plain selected evaluation unavailable",
            )?;
            check_report(
                &report,
                work,
                &input,
                &plain_output,
                witness.history(),
                witness.evaluated_len(),
                witness.pending_len(),
            )?;
            plain_history.extend_from_slice(witness.history());
            (report, admitted)
        };
        // The plain session has been dropped before profiled KV/scratch exists.
        let mut hash = Sha256::new();
        hash.update(workload.sha256.as_bytes());
        hash.update((args.evaluation as u64).to_le_bytes());
        let request_id: [u8; 16] = hash.finalize()[..16].try_into().map_err(io::Error::other)?;
        let profile_options = Bf16ProfileOptions {
            request_id,
            evaluation_ordinal: args.evaluation,
        };
        let mut profiled = match args.submission {
            Submission::PerTile => owner.new_profiled_session(&model, options, profile_options),
            Submission::BatchedSingleRow => {
                owner.new_batched_profiled_session(&model, options, profile_options)
            }
        }
        .map_err(io::Error::other)?;
        let profiled_admitted = profiled.memory().admitted_bytes;
        let total = combined(profiled_admitted, reserved)?;
        require(
            plain_admitted
                .checked_add(DIAGNOSTIC)
                .is_some_and(|minimum| profiled_admitted >= minimum),
            "profile diagnostic allowance missing",
        )?;
        let profiled_report = {
            let mut witness = Witness::new(&mut profiled, args.evaluation, &mut profiled_logits)?;
            let report = generation::generate(&mut witness, request(), &CANCELLED, |event| {
                require(
                    profiled_output.len() < work.maximum_output_tokens,
                    "profiled output bound exceeded",
                )?;
                profiled_output.push(event.token_id);
                Ok(())
            })
            .map_err(io::Error::other)?;
            require(
                witness.captured && witness.evaluations == work.evaluations(),
                "profiled selected evaluation unavailable",
            )?;
            check_report(
                &report,
                work,
                &input,
                &profiled_output,
                witness.history(),
                witness.evaluated_len(),
                witness.pending_len(),
            )?;
            require(
                witness.history() == plain_history,
                "plain/profiled histories differ",
            )?;
            report
        };
        let mut first_mismatch = None;
        let mut selected_hash = Sha256::new();
        for (index, (plain, profiled)) in plain_logits.iter().zip(&profiled_logits).enumerate() {
            if plain.to_bits() != profiled.to_bits() && first_mismatch.is_none() {
                first_mismatch = Some(index);
            }
            selected_hash.update(plain.to_le_bytes());
        }
        if let Some(index) = first_mismatch {
            return Err(io::Error::other(format!(
                "selected logit bit mismatch at{index}: plain={:08x} profiled={:08x}",
                plain_logits[index].to_bits(),
                profiled_logits[index].to_bits()
            )));
        }
        let selected_sha = format!("{:x}", selected_hash.finalize());
        let frame = profiled.bf16_profile().map_err(io::Error::other)?;
        let inventory = match args.submission {
            Submission::PerTile => owner.runtime_inventory(32),
            Submission::BatchedSingleRow => owner.runtime_inventory_batched_single_row(32),
        }
        .map_err(io::Error::other)?;
        let prefill = input.len().div_ceil(32);
        let (position, rows) = if args.evaluation <= prefill {
            let position = (args.evaluation - 1) * 32;
            (position, (input.len() - position).min(32))
        } else {
            (input.len() + args.evaluation - prefill - 1, 1)
        };
        let tiles: usize = inventory
            .operations
            .iter()
            .map(|op| op.projection_tiles)
            .sum();
        require(
            inventory.payload_authenticated
                && frame.schema_version
                    == if args.submission == Submission::PerTile {
                        1
                    } else {
                        2
                    }
                && frame.evaluation_ordinal == args.evaluation
                && frame.request_id == request_id
                && frame.status.state == Bf16CaptureState::Complete
                && !frame.status.overflowed
                && frame.status.tiles_recorded == tiles
                && frame.tiles.len() == tiles
                && frame.clocks.events.len() == tiles
                && frame.status.gathers_recorded == 2
                && frame.diagnostic_allowance_bytes == DIAGNOSTIC
                && frame.evaluation.position == position
                && frame.evaluation.input_rows == rows
                && frame.evaluation.output_rows == 1
                && frame.evaluation.committed_positions == position + rows,
            "selected frame/inventory geometry or coverage differs",
        )?;
        // Include actual retained Vec/String capacities, plus a separate1MiB
        // margin for the small manifest/Serde values and cold stack metadata.
        let caller_bytes = (plain_logits.capacity() + profiled_logits.capacity())
            * size_of::<f32>()
            + (input.capacity()
                + plain_output.capacity()
                + profiled_output.capacity()
                + plain_history.capacity())
                * size_of::<u32>()
            + profile_json.bytes.capacity()
            + inventory_json.bytes.capacity()
            + manifest_json.bytes.capacity()
            + WORKLOAD_LIMIT
            + frame.tiles.capacity() * frame.tiles.first().map_or(0, size_of_val)
            + frame.clocks.events.capacity() * frame.clocks.events.first().map_or(0, size_of_val)
            + frame.submission_groups.as_ref().map_or(0, |groups| {
                groups.capacity() * groups.first().map_or(0, size_of_val)
            })
            + inventory.operations.capacity() * inventory.operations.first().map_or(0, size_of_val)
            + inventory
                .operations
                .iter()
                .map(|op| op.logical.source_tensor.capacity())
                .sum::<usize>()
            + JSON_LIMIT;
        require(
            caller_bytes <= CALLER,
            "diagnostic caller allowance exceeded",
        )?;
        require(
            !CANCELLED.load(Ordering::Relaxed),
            "profile export cancelled",
        )?;
        profile_json.encode(&frame)?;
        inventory_json.encode(&inventory)?;
        let profile_pin = profile_json.publish(&args.output, "profile.json")?;
        let inventory_pin = inventory_json.publish(&args.output, "inventory.json")?;
        let mut manifest = json!({
            "format":"toklet-gemma-profile-v1","passed":true,
            "diagnostic_only":true,"timing_qualified":false,
            "model":{"repository":OfficialBf16::REPOSITORY,"revision":OfficialBf16::REVISION},
            "workload":{"sha256":workload.sha256,"conversation_sha256":work.conversation_sha256,
                "input_tokens_sha256":work.input_tokens_sha256,"input_tokens":input.len(),
                "maximum_output_tokens":work.maximum_output_tokens,"capacity":512,"chunk":32},
            "capture":{"evaluation_ordinal":args.evaluation,"request_id":request_id,
                "position":position,"input_rows":rows,"output_rows":1,
                "committed_positions":position+rows},
            "parity":{"selected_logits_bitwise":true,"selected_logits_count":plain_logits.len(),
                "selected_logits_sha256":selected_sha,"output_ids":true,"history":true,"report_counts":true},
            "plain":report_value(&plain_report)?,"profiled":report_value(&profiled_report)?,
            "inventory":inventory_pin,"profile":profile_pin,
            "memory":{"limit_bytes":LIMIT,"owner_limit_bytes":OWNER_LIMIT,
                "caller_allowance_bytes":CALLER,"helper_allowance_bytes":HELPER,
                "diagnostic_allowance_bytes":DIAGNOSTIC,"owner_reserved_bytes":reserved,
                "plain_admitted_bytes":plain_admitted,"profiled_admitted_bytes":profiled_admitted,
                "combined_profiled_admitted_bytes":total},
            "clock_scope":"Completed core host times include selected diagnostic witness work; frame clocks have separate scopes. No acceptance timing or CPU/GPU origin subtraction. Numerical session admissions exclude the separately reported owner reserve."
        });
        if args.submission == Submission::BatchedSingleRow {
            manifest["format"] = json!("toklet-gemma-profile-v2");
            manifest["submission_policy"] = json!("batched_single_row");
        }
        require(
            !CANCELLED.load(Ordering::Relaxed),
            "profile publication cancelled",
        )?;
        manifest_json.encode(&manifest)?;
        manifest_json.publish(&args.output, "manifest.json")?;
        Ok(manifest)
    }
}

pub(super) fn main(values: Vec<OsString>) -> u8 {
    if values.len() == 1 && (values[0] == "--help" || values[0] == "-h") {
        println!("{HELP}");
        return 0;
    }
    let input = (|| {
        let args = parse(values)?;
        let workload = load_workload(&args.workload)?;
        if args.evaluation > workload.document.evaluations() {
            return Err(invalid("evaluation exceeds the frozen request"));
        }
        Ok((args, workload))
    })();
    let (args, workload) = match input {
        Ok(input) => input,
        Err(error) => {
            eprintln!("error: {error}\nUse 'toklet profile --help' for usage.");
            return 2;
        }
    };
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    {
        match execution::run(&args, &workload) {
            Ok(manifest) => super::inspect::write_report(&manifest),
            Err(error) => {
                eprintln!("error: {error}");
                1
            }
        }
    }
    #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
    {
        let Args {
            model,
            output,
            submission,
            ..
        } = args;
        let _ = (model, output, submission, workload.sha256);
        eprintln!("error: official BF16/G2 profiling requires macOS arm64 and the metal feature");
        1
    }
}

#[cfg(test)]
mod tests;
