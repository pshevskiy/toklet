//! Fixed request-boundary summaries; collection never changes inference state.
use serde::Serialize;
use std::{io, path::Path};

pub(super) const COLLECTOR_LIMIT: usize = 16 * 1024;
pub(super) const RING_LIMIT: usize = 1024 * 1024;
pub(super) const RECORD_LIMIT: usize = 4096;
pub(super) const ALLOWANCE_BYTES: usize = COLLECTOR_LIMIT + RING_LIMIT;

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(super) struct State {
    pub history_tokens: usize,
    pub committed_tokens: usize,
    pub pending_tokens: usize,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(super) struct Work {
    pub prompt_tokens: Option<usize>,
    pub reused_prefix_tokens: Option<usize>,
    pub prefill_rows: Option<usize>,
    pub decode_evaluations: Option<usize>,
    pub sampled_tokens: Option<usize>,
    pub emitted_tokens: Option<usize>,
    pub callback_tokens: usize,
    pub replayed_rows: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(super) struct Timing {
    pub request_total_ms: f64,
    pub native_total_ms: Option<f64>,
    pub tokenization_ms: Option<f64>,
    pub prefill_ms: Option<f64>,
    pub decode_ms: Option<f64>,
    pub sampling_ms: Option<f64>,
    pub ttft_event_ms: Option<f64>,
    pub first_text_publication_ms: Option<f64>,
    pub last_token_event_ms: Option<f64>,
    pub token_interval_count: usize,
    pub token_interval_min_ms: Option<f64>,
    pub token_interval_max_ms: Option<f64>,
    pub token_interval_sum_ms: f64,
    pub output_decode_ms: f64,
    pub output_publication_ms: f64,
    pub cancel_to_terminal_ms: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(super) struct Memory {
    pub weights_owned_bytes: Option<usize>,
    pub kv_allocated_bytes: Option<usize>,
    pub scratch_bytes: Option<usize>,
    pub kv_valid_positions: usize,
    pub owner_reserved_bytes: usize,
    pub load_admitted_bytes: usize,
    pub session_admitted_bytes: usize,
    pub app_allowance_bytes: usize,
    pub stream_allowance_bytes: usize,
    pub telemetry_allowance_bytes: usize,
    pub helper_reserve_bytes: usize,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Kind {
    Generate,
    Append,
    Continue,
    Reset,
    Invalid,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Outcome {
    Success,
    Stop,
    Cancelled,
    Rejected,
    ExecutionError,
    OutputError,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Phase {
    Parse,
    Tokenization,
    Admission,
    Execution,
    Output,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ErrorCategory {
    InvalidInput,
    Session,
    Sampling,
    Output,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum CacheAction {
    Fresh,
    ExactPrefix,
    Append,
    FullyCached,
    Pending,
    Replay,
    Reset,
    Unchanged,
}

#[derive(Debug, Serialize)]
pub(super) struct Record<'a> {
    pub session_id: &'a str,
    pub request_id: u64,
    pub attempt_id: u64,
    pub owner_generation: u64,
    pub request_kind: Kind,
    pub terminal_status: Outcome,
    pub failed_phase: Option<Phase>,
    pub error_category: Option<ErrorCategory>,
    pub cache_action: CacheAction,
    pub replay_required: bool,
    pub before: State,
    pub after: State,
    pub work: Work,
    pub timing: Timing,
    pub memory: Memory,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(super) struct ExportStatus {
    pub retained: u64,
    pub telemetry_dropped: u64,
    pub export_errors: u64,
    pub invalid_records: u64,
    pub counter_overflow: bool,
}

#[derive(Debug, Serialize)]
pub(super) struct Manifest<'a> {
    pub session_id: &'a str,
    pub capacity: usize,
    pub prefill_chunk_size: usize,
    pub submission: &'static str,
    pub load_ms: f64,
    pub route_start_to_ready_ms: f64,
    pub load_admitted_bytes: usize,
    /// Whole session admission, including the immutable owner's reserve and
    /// separately admitted application/stream/telemetry/helper allowances.
    pub session_admitted_bytes: usize,
    /// Numerical ownership only. Its component sum equals admitted_bytes;
    /// it is a partition of session_admitted_bytes, not an additional owner.
    pub logical_memory: toklet::session::Memory,
}

const SLOTS: usize = (RING_LIMIT - COLLECTOR_LIMIT) / RECORD_LIMIT;
const RING_BYTES: usize = SLOTS * RECORD_LIMIT;
const MEMORY_LIMIT: usize = 6 * 1024 * 1024 * 1024;
// Leave room for a terminal output-error correction without losing a retained row.
const INITIAL_LINE_LIMIT: usize = RECORD_LIMIT - 128;

// The active Apple route reads its identity directly from the verified owner.
// Portable CPU formatter controls use the same source-audited canonical literals;
// that build cannot start the Metal-only Gemma route.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
const MODEL_REPOSITORY: &str = toklet::metal::gemma_affine::OfficialBf16::REPOSITORY;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
const MODEL_REVISION: &str = toklet::metal::gemma_affine::OfficialBf16::REVISION;
#[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
const MODEL_REPOSITORY: &str = "google/gemma-4-E2B-it";
#[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
const MODEL_REVISION: &str = "70af34e20bd4b7a91f0de6b22675850c43922a03";

#[derive(Serialize)]
struct RouteMetadata {
    model_profile: &'static str,
    backend: &'static str,
    model_repository: &'static str,
    model_revision: &'static str,
    weight_dtype: &'static str,
    weight_dtype_scope: &'static str,
    kv_dtype: &'static str,
    sampling: &'static str,
    thinking_enabled: bool,
    instrumentation: &'static str,
    engine_crate_version: &'static str,
    engine_revision: Option<&'static str>,
    build_flags: Option<&'static str>,
    device_identity: Option<&'static str>,
    device_power: Option<&'static str>,
    model_content_sha256: Option<&'static str>,
    tokenizer_content_sha256: Option<&'static str>,
    external_provenance_reason: &'static str,
    conversion_ms: Option<f64>,
    compile_ms: Option<f64>,
    process_start_to_ready_ms: Option<f64>,
    conversion_availability_reason: &'static str,
    compile_availability_reason: &'static str,
    process_start_to_ready_availability_reason: &'static str,
    cancel_to_terminal_availability_reason: &'static str,
    route_start_to_ready_scope: &'static str,
    load_timing_scope: &'static str,
}
fn route_metadata() -> RouteMetadata {
    RouteMetadata {
        model_profile: "gemma4-bf16",
        backend: "metal",
        model_repository: MODEL_REPOSITORY,
        model_revision: MODEL_REVISION,
        // Bf16Weights keeps original BF16 matrices/PLE and decoded f32 vectors.
        // Bf16Model uses gemma_packed::Session's WindowKvCache = Cache<f32>.
        weight_dtype: "bf16",
        weight_dtype_scope: "Original BF16 matrix coefficients and file-backed PLE; resident learned vectors are FP32.",
        kv_dtype: "fp32",
        sampling: "greedy",
        thinking_enabled: false,
        instrumentation: "summary",
        engine_crate_version: env!("CARGO_PKG_VERSION"),
        engine_revision: None,
        build_flags: None,
        device_identity: None,
        device_power: None,
        model_content_sha256: None,
        tokenizer_content_sha256: None,
        external_provenance_reason: "external_capture_manifest_required",
        conversion_ms: None,
        compile_ms: None,
        process_start_to_ready_ms: None,
        conversion_availability_reason: "not_separately_timed_within_owner_load",
        compile_availability_reason: "context_compilation_overlaps_owner_load; phase_not_separately_timed",
        process_start_to_ready_availability_reason: "process_start_not_observed; route_start_to_ready_ms_is_handled_route_clock",
        cancel_to_terminal_availability_reason: "signal_timestamp_not_observed; cancellation_outcome_and_state_do_not_measure_signal_to_ack_latency",
        route_start_to_ready_scope: "Handled Gemma route entry to ready; process startup and stdout/UI receipt excluded.",
        load_timing_scope: "Completed OfficialBf16::load owner construction; context/pipeline and authenticated asset preparation overlap; conversion/compile components unavailable.",
    }
}

#[derive(Serialize)]
struct Header<'a, 'b> {
    event: &'static str,
    schema_version: u32,
    #[serde(flatten)]
    manifest: &'a Manifest<'b>,
    numerical_session_admitted_bytes: usize,
    additional_owner_and_external_admitted_bytes: usize,
    logical_memory_scope: &'static str,
    collector_fixed_bytes: usize,
    retained_ring_capacity_bytes: usize,
    collector_peak_bytes: usize,
    telemetry_admitted_bytes: usize,
    collector_memory_scope: &'static str,
    #[serde(flatten)]
    route: RouteMetadata,
}

#[derive(Serialize)]
struct OrdinaryObservations {
    gpu_ms: Option<f64>,
    host_submit_ms: Option<f64>,
    host_wait_ms: Option<f64>,
    command_count: Option<u64>,
    dispatch_count: Option<u64>,
    wait_count: Option<u64>,
    readback_bytes: Option<u64>,
    buffer_growth_count: Option<u64>,
    pipeline_compile_count: Option<u64>,
    allocation_count: Option<u64>,
    process_footprint_bytes: Option<u64>,
    ui_first_visible_ms: Option<f64>,
    energy_joules: Option<f64>,
}

#[derive(Serialize)]
struct Availability {
    ordinary_observations: &'static str,
    process_footprint_bytes: &'static str,
    ui_first_visible_ms: &'static str,
    energy_joules: &'static str,
}

#[derive(Serialize)]
struct Summary<'a, 'b> {
    event: &'static str,
    schema_version: u32,
    #[serde(flatten)]
    record: &'a Record<'b>,
    prefill_tokens_per_s: Option<f64>,
    decode_tokens_per_s: Option<f64>,
    decode_evaluations_per_s: Option<f64>,
    token_interval_mean_ms: Option<f64>,
    ordinary_observations: OrdinaryObservations,
    availability: Availability,
}

#[derive(Serialize)]
struct Footer {
    event: &'static str,
    schema_version: u32,
    complete: bool,
    #[serde(flatten)]
    status: ExportStatus,
}

struct FixedWriter<'a> {
    bytes: &'a mut [u8],
    len: usize,
}
impl io::Write for FixedWriter<'_> {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        let end = self
            .len
            .checked_add(input.len())
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| io::Error::other("request metrics line capacity"))?;
        self.bytes[self.len..end].copy_from_slice(input);
        self.len = end;
        Ok(input.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn line(bytes: &mut [u8], value: &impl Serialize) -> io::Result<usize> {
    use io::Write;
    let mut writer = FixedWriter { bytes, len: 0 };
    serde_json::to_writer(&mut writer, value).map_err(io::Error::other)?;
    writer.write_all(b"\n")?;
    Ok(writer.len)
}

fn increment(value: &mut u64, overflow: &mut bool) {
    match value.checked_add(1) {
        Some(next) => *value = next,
        None => *overflow = true,
    }
}
fn finite(value: f64) -> bool {
    value.is_finite() && value >= 0.
}
fn optional(value: Option<f64>) -> bool {
    value.is_none_or(finite)
}
fn identity(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.:".contains(&b))
}
fn rate(count: Option<usize>, milliseconds: Option<f64>) -> Option<f64> {
    let count = count.filter(|n| *n > 0)?;
    let milliseconds = milliseconds.filter(|ms| *ms > 0.)?;
    let value = count as f64 * 1000. / milliseconds;
    value.is_finite().then_some(value)
}
fn summary<'a, 'b>(record: &'a Record<'b>) -> Summary<'a, 'b> {
    let t = record.timing;
    let interval = t
        .last_token_event_ms
        .zip(t.ttft_event_ms)
        .and_then(|(last, first)| (last > first).then_some(last - first));
    Summary {
        event: "request_summary",
        schema_version: 1,
        record,
        prefill_tokens_per_s: rate(record.work.prefill_rows, t.prefill_ms),
        decode_tokens_per_s: rate(record.work.callback_tokens.checked_sub(1), interval),
        decode_evaluations_per_s: rate(record.work.decode_evaluations, t.decode_ms),
        token_interval_mean_ms: (t.token_interval_count > 0)
            .then(|| t.token_interval_sum_ms / t.token_interval_count as f64),
        ordinary_observations: OrdinaryObservations {
            gpu_ms: None,
            host_submit_ms: None,
            host_wait_ms: None,
            command_count: None,
            dispatch_count: None,
            wait_count: None,
            readback_bytes: None,
            buffer_growth_count: None,
            pipeline_compile_count: None,
            allocation_count: None,
            process_footprint_bytes: None,
            ui_first_visible_ms: None,
            energy_joules: None,
        },
        availability: Availability {
            ordinary_observations: "not_observed_in_ordinary_request",
            process_footprint_bytes: "admission_is_not_sampled_process_footprint",
            ui_first_visible_ms: "stdout_publication_is_not_ui_visibility",
            energy_joules: "not_measured",
        },
    }
}
fn numerical_admission(memory: toklet::session::Memory) -> Option<usize> {
    let total = [
        memory.weights_bytes,
        memory.positional_bytes,
        memory.kv_bytes,
        memory.scratch_bytes,
        memory.history_bytes,
        memory.logits_bytes,
        memory.metadata_bytes,
        memory.metadata_allowance_bytes,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)?;
    (total == memory.admitted_bytes && total <= MEMORY_LIMIT).then_some(total)
}
fn valid_state(s: State, capacity: usize) -> bool {
    s.history_tokens <= capacity
        && s.pending_tokens <= s.history_tokens
        && s.committed_tokens.checked_add(s.pending_tokens) == Some(s.history_tokens)
}
fn valid_record(r: &Record<'_>, capacity: usize) -> bool {
    let w = r.work;
    let t = r.timing;
    if !identity(r.session_id)
        || r.owner_generation != 1
        || !valid_state(r.before, capacity)
        || !valid_state(r.after, capacity)
        || w.callback_tokens > capacity
        || t.cancel_to_terminal_ms.is_some()
        || t.token_interval_count != w.callback_tokens.saturating_sub(1)
        || [
            w.prompt_tokens,
            w.reused_prefix_tokens,
            w.prefill_rows,
            w.decode_evaluations,
            w.sampled_tokens,
            w.emitted_tokens,
            w.replayed_rows,
        ]
        .into_iter()
        .flatten()
        .any(|n| n > capacity)
        || ![
            t.request_total_ms,
            t.token_interval_sum_ms,
            t.output_decode_ms,
            t.output_publication_ms,
        ]
        .into_iter()
        .all(finite)
        || ![
            t.native_total_ms,
            t.tokenization_ms,
            t.prefill_ms,
            t.decode_ms,
            t.sampling_ms,
            t.ttft_event_ms,
            t.first_text_publication_ms,
            t.last_token_event_ms,
            t.token_interval_min_ms,
            t.token_interval_max_ms,
        ]
        .into_iter()
        .all(optional)
    {
        return false;
    }
    let errors_match = match r.terminal_status {
        Outcome::Success | Outcome::Stop | Outcome::Cancelled => {
            r.failed_phase.is_none() && r.error_category.is_none()
        }
        Outcome::Rejected | Outcome::ExecutionError => {
            r.failed_phase.is_some() && r.error_category.is_some()
        }
        Outcome::OutputError => {
            matches!(r.failed_phase, Some(Phase::Output))
                && matches!(r.error_category, Some(ErrorCategory::Output))
        }
    };
    if !errors_match {
        return false;
    }
    let counts = [
        w.reused_prefix_tokens,
        w.prefill_rows,
        w.decode_evaluations,
        w.sampled_tokens,
        w.emitted_tokens,
    ];
    let times = [t.native_total_ms, t.prefill_ms, t.decode_ms, t.sampling_ms];
    let complete = counts.iter().all(Option::is_some) && times.iter().all(Option::is_some);
    if !complete && (counts.iter().any(Option::is_some) || times.iter().any(Option::is_some)) {
        return false;
    }
    if complete
        && (w.emitted_tokens > w.sampled_tokens
            || w.decode_evaluations > w.sampled_tokens
            || w.sampled_tokens.is_some_and(|n| w.callback_tokens > n))
    {
        return false;
    }
    if w.replayed_rows
        .is_some_and(|rows| w.prefill_rows.is_none_or(|prefill| rows > prefill))
    {
        return false;
    }
    if complete {
        let (Some(prompt), Some(reused), Some(prefill), Some(sampled), Some(emitted), Some(decode)) = (
            w.prompt_tokens,
            w.reused_prefix_tokens,
            w.prefill_rows,
            w.sampled_tokens,
            w.emitted_tokens,
            w.decode_evaluations,
        ) else {
            return false;
        };
        let Some(evaluated_prompt) = reused.checked_add(prefill) else {
            return false;
        };
        if evaluated_prompt > prompt {
            return false;
        }
        // Completed cancellation/output-error reports can describe partial input
        // evaluation. Successful terminal reports cover the full logical prompt.
        if matches!(r.terminal_status, Outcome::Success | Outcome::Stop)
            && (evaluated_prompt != prompt
                || prompt.checked_add(sampled) != Some(r.after.history_tokens)
                || r.after.pending_tokens != usize::from(sampled > 0)
                || emitted != w.callback_tokens
                || decode != sampled.saturating_sub(1))
        {
            return false;
        }
    }
    if w.callback_tokens == 0 {
        if t.ttft_event_ms.is_some() || t.last_token_event_ms.is_some() {
            return false;
        }
    } else {
        match (t.ttft_event_ms, t.last_token_event_ms) {
            (Some(first), Some(last)) if first <= last && last <= t.request_total_ms => {}
            _ => return false,
        }
    }
    if t.token_interval_count == 0 {
        if t.token_interval_min_ms.is_some()
            || t.token_interval_max_ms.is_some()
            || t.token_interval_sum_ms != 0.
        {
            return false;
        }
    } else {
        match (t.token_interval_min_ms, t.token_interval_max_ms) {
            (Some(min), Some(max))
                if min <= max
                    && min <= t.token_interval_sum_ms
                    && max <= t.token_interval_sum_ms => {}
            _ => return false,
        }
    }
    if t.first_text_publication_ms
        .is_some_and(|v| v > t.request_total_ms)
    {
        return false;
    }
    let m = r.memory;
    m.kv_valid_positions <= r.after.committed_tokens
        && [m.weights_owned_bytes, m.kv_allocated_bytes, m.scratch_bytes]
            .into_iter()
            .flatten()
            .all(|n| n <= MEMORY_LIMIT)
        && [
            m.owner_reserved_bytes,
            m.load_admitted_bytes,
            m.session_admitted_bytes,
            m.app_allowance_bytes,
            m.stream_allowance_bytes,
            m.telemetry_allowance_bytes,
            m.helper_reserve_bytes,
        ]
        .into_iter()
        .all(|n| n <= MEMORY_LIMIT)
}

/// Fixed metadata buffers plus a fallible byte ring. No stored inference owners,
/// borrowed request text, token arrays, GPU objects, or per-record heap ownership.
pub(super) struct Collector {
    enabled: bool,
    ring: Vec<u8>,
    lengths: [u16; SLOTS],
    first: usize,
    retained: usize,
    header: [u8; RECORD_LIMIT],
    header_len: usize,
    scratch: [u8; RECORD_LIMIT],
    session_id: [u8; 64],
    session_len: usize,
    capacity: usize,
    logical_memory: Option<toklet::session::Memory>,
    additional_admitted_bytes: usize,
    load_admitted_bytes: usize,
    last_ids: Option<(u64, u64, u64)>,
    counters: ExportStatus,
}
impl Collector {
    pub(super) fn new(enabled: bool) -> Self {
        Self::with_reservation(enabled, RING_BYTES)
    }
    fn with_reservation(enabled: bool, requested: usize) -> Self {
        let mut result = Self {
            enabled,
            ring: Vec::new(),
            lengths: [0; SLOTS],
            first: 0,
            retained: 0,
            header: [0; RECORD_LIMIT],
            header_len: 0,
            scratch: [0; RECORD_LIMIT],
            session_id: [0; 64],
            session_len: 0,
            capacity: 0,
            logical_memory: None,
            additional_admitted_bytes: 0,
            load_admitted_bytes: 0,
            last_ids: None,
            counters: ExportStatus::default(),
        };
        if enabled {
            if result.ring.try_reserve_exact(requested).is_ok()
                && (RING_BYTES..=RING_LIMIT - COLLECTOR_LIMIT).contains(&result.ring.capacity())
            {
                result.ring.resize(RING_BYTES, 0);
            } else {
                result.ring = Vec::new();
                result.drop_record(false);
            }
        }
        result
    }
    fn drop_record(&mut self, invalid: bool) {
        increment(
            &mut self.counters.telemetry_dropped,
            &mut self.counters.counter_overflow,
        );
        if invalid {
            self.invalid();
        }
    }
    /// The next handled invocation has no representable identity. Mark the
    /// missing metadata explicitly without fabricating a record or session work.
    pub(super) fn note_counter_overflow(&mut self) {
        self.counters.counter_overflow = true;
        self.drop_record(false);
    }
    fn invalid(&mut self) {
        increment(
            &mut self.counters.invalid_records,
            &mut self.counters.counter_overflow,
        );
    }
    fn export_error(&mut self) {
        increment(
            &mut self.counters.export_errors,
            &mut self.counters.counter_overflow,
        );
    }
    pub(super) fn set_manifest(&mut self, manifest: &Manifest<'_>) {
        if !self.enabled {
            return;
        }
        if self.header_len != 0
            || !identity(manifest.session_id)
            || !(1..=512).contains(&manifest.capacity)
            || !(1..=32).contains(&manifest.prefill_chunk_size)
            || !matches!(manifest.submission, "per-tile" | "batched-single-row")
            || !finite(manifest.load_ms)
            || !finite(manifest.route_start_to_ready_ms)
            || manifest.load_admitted_bytes > MEMORY_LIMIT
            || manifest.session_admitted_bytes > MEMORY_LIMIT
        {
            self.drop_record(true);
            return;
        }
        let Some(numerical) = numerical_admission(manifest.logical_memory) else {
            self.drop_record(true);
            return;
        };
        let Some(additional) = manifest.session_admitted_bytes.checked_sub(numerical) else {
            self.drop_record(true);
            return;
        };
        // These are invariant logical owned bytes after construction, not a
        // sampled process peak or constructor/temporary allocation observation.
        let collector_fixed_bytes = std::mem::size_of::<Self>();
        let retained_ring_capacity_bytes = self.ring.capacity();
        let collector_peak_bytes = self.allocated_bytes();
        match line(
            &mut self.header,
            &Header {
                event: "request_metrics_manifest",
                schema_version: 1,
                manifest,
                numerical_session_admitted_bytes: numerical,
                additional_owner_and_external_admitted_bytes: additional,
                logical_memory_scope: "Numerical component sum equals logical_memory.admitted_bytes. Whole session admission is numerical plus additional owner/external admission; do not add logical_memory twice. Admission is not measured process footprint.",
                collector_fixed_bytes,
                retained_ring_capacity_bytes,
                collector_peak_bytes,
                telemetry_admitted_bytes: ALLOWANCE_BYTES,
                collector_memory_scope: "Logical Collector value plus retained ring capacity after construction, fixed until drop. Excludes allocator overhead, temporary stack/copy/export storage and native/OS memory; not RSS or process peak.",
                route: route_metadata(),
            },
        ) {
            Ok(len) => {
                self.header_len = len;
                self.capacity = manifest.capacity;
                self.logical_memory = Some(manifest.logical_memory);
                self.additional_admitted_bytes = additional;
                self.load_admitted_bytes = manifest.load_admitted_bytes;
                self.session_len = manifest.session_id.len();
                self.session_id[..self.session_len].copy_from_slice(manifest.session_id.as_bytes());
            }
            Err(_) => self.drop_record(false),
        }
    }
    fn same_session(&self, record: &Record<'_>) -> bool {
        record.session_id.as_bytes() == &self.session_id[..self.session_len]
    }
    fn same_memory_partition(&self, record: &Record<'_>) -> bool {
        let Some(logical) = self.logical_memory else {
            return false;
        };
        let m = record.memory;
        let additional = [
            m.owner_reserved_bytes,
            m.app_allowance_bytes,
            m.stream_allowance_bytes,
            m.telemetry_allowance_bytes,
            m.helper_reserve_bytes,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add);
        additional == Some(self.additional_admitted_bytes)
            && logical
                .admitted_bytes
                .checked_add(self.additional_admitted_bytes)
                == Some(m.session_admitted_bytes)
            && m.load_admitted_bytes == self.load_admitted_bytes
            && m.weights_owned_bytes
                .is_none_or(|v| v == logical.weights_bytes)
            && m.kv_allocated_bytes.is_none_or(|v| v == logical.kv_bytes)
            && m.scratch_bytes.is_none_or(|v| v == logical.scratch_bytes)
    }
    pub(super) fn record(&mut self, record: &Record<'_>) {
        if !self.enabled {
            return;
        }
        if self.header_len == 0
            || !self.same_session(record)
            || !valid_record(record, self.capacity)
            || !self.same_memory_partition(record)
            || self.last_ids.is_some_and(|(request, attempt, generation)| {
                record.request_id <= request
                    || record.attempt_id <= attempt
                    || record.owner_generation < generation
            })
        {
            self.drop_record(true);
            return;
        }
        self.last_ids = Some((
            record.request_id,
            record.attempt_id,
            record.owner_generation,
        ));
        if self.ring.is_empty() {
            self.drop_record(false);
            return;
        }
        let len = match line(&mut self.scratch[..INITIAL_LINE_LIMIT], &summary(record)) {
            Ok(len) => len,
            Err(_) => {
                self.drop_record(false);
                return;
            }
        };
        let slot = if self.retained == SLOTS {
            let slot = self.first;
            self.first = (self.first + 1) % SLOTS;
            self.drop_record(false);
            slot
        } else {
            let slot = (self.first + self.retained) % SLOTS;
            self.retained += 1;
            slot
        };
        let start = slot * RECORD_LIMIT;
        self.ring[start..start + len].copy_from_slice(&self.scratch[..len]);
        self.lengths[slot] = len as u16;
    }
    pub(super) fn correct_last_terminal(&mut self, record: &Record<'_>) {
        if !self.enabled
            || self.retained == 0
            || !self.same_session(record)
            || !self.last_ids.is_some_and(|(request, attempt, _)| {
                request == record.request_id && attempt == record.attempt_id
            })
        {
            return;
        }
        if !matches!(record.terminal_status, Outcome::OutputError)
            || !matches!(record.failed_phase, Some(Phase::Output))
            || !matches!(record.error_category, Some(ErrorCategory::Output))
            || !record.replay_required
            || !valid_record(record, self.capacity)
            || !self.same_memory_partition(record)
        {
            self.invalid();
            return;
        }
        let len = match line(&mut self.scratch, &summary(record)) {
            Ok(len) => len,
            Err(_) => {
                self.invalid();
                return;
            }
        };
        let slot = (self.first + self.retained - 1) % SLOTS;
        let start = slot * RECORD_LIMIT;
        let old = &self.ring[start..start + usize::from(self.lengths[slot])];
        let new = &self.scratch[..len];
        // Fixed serde field order and restricted IDs make these unique structural
        // markers. Only the three terminal enums and false->true replay marking
        // may change; an already true replay marker must remain true.
        let terminal = b",\"terminal_status\":";
        let cache = b",\"cache_action\":";
        let replay = b",\"replay_required\":";
        let before = b",\"before\":";
        let find =
            |bytes: &[u8], needle: &[u8]| bytes.windows(needle.len()).position(|x| x == needle);
        let matches = match (
            find(old, terminal),
            find(new, terminal),
            find(old, cache),
            find(new, cache),
            find(old, replay),
            find(new, replay),
            find(old, before),
            find(new, before),
        ) {
            (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f), Some(g), Some(h)) => {
                a < c
                    && c < e
                    && e < g
                    && b < d
                    && d < f
                    && f < h
                    && old[..a] == new[..b]
                    && old[c..e] == new[d..f]
                    && (&old[e + replay.len()..g] == b"false"
                        || &old[e + replay.len()..g] == b"true")
                    && &new[f + replay.len()..h] == b"true"
                    && old[g..] == new[h..]
            }
            _ => false,
        };
        if !matches {
            self.invalid();
            return;
        }
        self.ring[start..start + len].copy_from_slice(&self.scratch[..len]);
        self.lengths[slot] = len as u16;
    }
    pub(super) fn export(&mut self, path: &Path) -> io::Result<ExportStatus> {
        // create_new is atomic and refuses every existing path, including a
        // symlink or FIFO, before an existing FIFO could be opened for writing.
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
        {
            Ok(file) => self.export_to(file),
            Err(error) => {
                self.export_error();
                Err(error)
            }
        }
    }
    pub(super) fn status(&self) -> ExportStatus {
        ExportStatus {
            retained: self.retained as u64,
            ..self.counters
        }
    }
    pub(super) fn allocated_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.ring.capacity()
    }
    /// Caller invokes this only after active inference. Export never touches the
    /// model/session; a failed sink leaves no successful exporter return value.
    pub(super) fn export_to(&mut self, mut sink: impl io::Write) -> io::Result<ExportStatus> {
        let result = (|| {
            if !self.enabled || self.header_len == 0 {
                return Err(io::Error::other("request metrics manifest unavailable"));
            }
            sink.write_all(&self.header[..self.header_len])?;
            for index in 0..self.retained {
                let slot = (self.first + index) % SLOTS;
                let start = slot * RECORD_LIMIT;
                sink.write_all(&self.ring[start..start + usize::from(self.lengths[slot])])?;
            }
            sink.flush()?;
            let status = self.status();
            let len = line(
                &mut self.scratch,
                &Footer {
                    event: "request_metrics_export",
                    schema_version: 1,
                    complete: true,
                    status,
                },
            )?;
            sink.write_all(&self.scratch[..len])?;
            sink.flush()?;
            Ok(status)
        })();
        if result.is_err() {
            self.export_error();
            // Best effort only. A permanently failed sink instead has a missing
            // or partial footer, which is also explicitly incomplete framing.
            let status = self.status();
            if let Ok(len) = line(
                &mut self.scratch,
                &Footer {
                    event: "request_metrics_export",
                    schema_version: 1,
                    complete: false,
                    status,
                },
            ) {
                let _ = sink.write_all(b"\n");
                let _ = sink.write_all(&self.scratch[..len]);
                let _ = sink.flush();
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn manifest() -> Manifest<'static> {
        Manifest {
            session_id: "test-1-0",
            capacity: 512,
            prefill_chunk_size: 32,
            submission: "per-tile",
            load_ms: 1.,
            route_start_to_ready_ms: 2.,
            load_admitted_bytes: 0,
            session_admitted_bytes: 0,
            logical_memory: toklet::session::Memory {
                weights_bytes: 0,
                positional_bytes: 0,
                kv_bytes: 0,
                scratch_bytes: 0,
                history_bytes: 0,
                logits_bytes: 0,
                metadata_bytes: 0,
                metadata_allowance_bytes: 0,
                admitted_bytes: 0,
            },
        }
    }
    fn record(id: u64) -> Record<'static> {
        Record {
            session_id: "test-1-0",
            request_id: id,
            attempt_id: id,
            owner_generation: 1,
            request_kind: Kind::Invalid,
            terminal_status: Outcome::Rejected,
            failed_phase: Some(Phase::Parse),
            error_category: Some(ErrorCategory::InvalidInput),
            cache_action: CacheAction::Unchanged,
            replay_required: false,
            before: State::default(),
            after: State::default(),
            work: Work::default(),
            timing: Timing::default(),
            memory: Memory::default(),
        }
    }
    fn collector() -> Collector {
        let mut c = Collector::new(true);
        c.set_manifest(&manifest());
        c
    }
    fn rows(c: &mut Collector) -> Vec<serde_json::Value> {
        let mut bytes = Vec::new();
        c.export_to(&mut bytes).unwrap();
        bytes
            .split(|b| *b == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| {
                assert!(line.len() < RECORD_LIMIT);
                serde_json::from_slice(line).unwrap()
            })
            .collect()
    }
    struct Broken;
    impl Write for Broken {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("controlled sink failure"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("controlled sink failure"))
        }
    }
    #[test]
    fn bounded_storage_wrap_and_disabled_no_ring() {
        let mut disabled = Collector::new(false);
        disabled.set_manifest(&manifest());
        disabled.record(&record(1));
        assert!(disabled.ring.is_empty());
        assert_eq!(disabled.status().retained, 0);
        assert!(std::mem::size_of::<Collector>() <= COLLECTOR_LIMIT);
        let mut c = collector();
        assert!(c.allocated_bytes() <= RING_LIMIT);
        for id in 1..=SLOTS as u64 + 7 {
            c.record(&record(id));
        }
        let status = c.status();
        assert_eq!(status.retained, SLOTS as u64);
        assert_eq!(status.telemetry_dropped, 7);
        let output = rows(&mut c);
        assert_eq!(output.len(), SLOTS + 2);
        assert_eq!(output[1]["request_id"], 8);
        assert_eq!(output[SLOTS]["request_id"], SLOTS + 7);
        assert_eq!(output.last().unwrap()["complete"], true);
    }
    #[test]
    fn resource_header_is_exact_and_stable_through_wrap_correction_and_export_failure() {
        let mut c = collector();
        let owned_bytes = c.allocated_bytes();
        let initial = rows(&mut c)[0].clone();
        assert_eq!(
            initial["collector_fixed_bytes"],
            std::mem::size_of::<Collector>()
        );
        assert_eq!(initial["retained_ring_capacity_bytes"], RING_BYTES);
        assert_eq!(initial["collector_peak_bytes"], owned_bytes);
        assert_eq!(initial["telemetry_admitted_bytes"], ALLOWANCE_BYTES);
        assert_eq!(
            initial["collector_fixed_bytes"].as_u64().unwrap()
                + initial["retained_ring_capacity_bytes"].as_u64().unwrap(),
            initial["collector_peak_bytes"].as_u64().unwrap()
        );
        assert!(std::mem::size_of::<Collector>() <= COLLECTOR_LIMIT);
        assert!(owned_bytes <= RING_LIMIT);
        assert!(owned_bytes <= ALLOWANCE_BYTES);
        assert_eq!(
            initial["collector_memory_scope"],
            "Logical Collector value plus retained ring capacity after construction, fixed until drop. Excludes allocator overhead, temporary stack/copy/export storage and native/OS memory; not RSS or process peak."
        );
        for id in 1..=SLOTS as u64 + 3 {
            c.record(&record(id));
        }
        let mut last = record(SLOTS as u64 + 3);
        last.terminal_status = Outcome::OutputError;
        last.failed_phase = Some(Phase::Output);
        last.error_category = Some(ErrorCategory::Output);
        last.replay_required = true;
        let counters = serde_json::to_value(c.status()).unwrap();
        c.correct_last_terminal(&last);
        assert_eq!(serde_json::to_value(c.status()).unwrap(), counters);
        assert!(c.export_to(Broken).is_err());
        let output = rows(&mut c);
        assert_eq!(output[0], initial);
        assert_eq!(c.allocated_bytes(), owned_bytes);
        assert_eq!(c.status().retained, SLOTS as u64);
        assert_eq!(c.status().telemetry_dropped, 3);
        assert_eq!(c.status().export_errors, 1);
        assert_eq!(output[SLOTS]["terminal_status"], "output_error");

        let mut wide = Collector::new(true);
        let mut m = manifest();
        m.session_id = "1234567890123456789012345678901234567890123456789012345678901234";
        assert_eq!(m.session_id.len(), 64);
        wide.set_manifest(&m);
        assert_ne!(wide.header_len, 0);
        assert!(wide.header_len <= RECORD_LIMIT);
        assert_eq!(rows(&mut wide).len(), 2);
    }
    #[test]
    fn cancellation_signal_latency_is_unavailable_and_cannot_be_invented() {
        let mut c = collector();
        let mut r = completed_record();
        r.terminal_status = Outcome::Cancelled;
        assert!(valid_record(&r, 512));
        c.record(&r);
        r.request_id = 2;
        r.attempt_id = 2;
        for value in [0., 1., f64::NAN, f64::INFINITY] {
            r.timing.cancel_to_terminal_ms = Some(value);
            assert!(!valid_record(&r, 512));
            c.record(&r);
        }
        let output = rows(&mut c);
        assert_eq!(
            output[0]["cancel_to_terminal_availability_reason"],
            "signal_timestamp_not_observed; cancellation_outcome_and_state_do_not_measure_signal_to_ack_latency"
        );
        assert_eq!(output[1]["terminal_status"], "cancelled");
        assert!(output[1]["timing"]["cancel_to_terminal_ms"].is_null());
        assert_eq!(c.status().retained, 1);
        assert_eq!(c.status().invalid_records, 4);
        assert_eq!(c.status().telemetry_dropped, 4);
    }
    #[test]
    fn manifest_route_and_unavailable_components_are_explicit_and_bounded() {
        let mut c = collector();
        let output = rows(&mut c);
        let header = &output[0];
        assert_eq!(header["model_profile"], "gemma4-bf16");
        assert_eq!(header["backend"], "metal");
        assert_eq!(header["model_repository"], "google/gemma-4-E2B-it");
        assert_eq!(
            header["model_revision"],
            "70af34e20bd4b7a91f0de6b22675850c43922a03"
        );
        assert_eq!(header["weight_dtype"], "bf16");
        assert_eq!(header["kv_dtype"], "fp32");
        assert_eq!(header["sampling"], "greedy");
        assert_eq!(header["thinking_enabled"], false);
        assert_eq!(header["instrumentation"], "summary");
        assert_eq!(header["engine_crate_version"], env!("CARGO_PKG_VERSION"));
        for field in [
            "engine_revision",
            "build_flags",
            "device_identity",
            "device_power",
            "model_content_sha256",
            "tokenizer_content_sha256",
            "conversion_ms",
            "compile_ms",
            "process_start_to_ready_ms",
        ] {
            assert!(header.get(field).is_some_and(serde_json::Value::is_null));
        }
        assert_eq!(
            header["external_provenance_reason"],
            "external_capture_manifest_required"
        );
        assert_eq!(
            header["conversion_availability_reason"],
            "not_separately_timed_within_owner_load"
        );
        assert!(
            header["route_start_to_ready_scope"]
                .as_str()
                .unwrap()
                .contains("process startup")
        );
        assert!(c.header_len <= RECORD_LIMIT);
    }
    #[test]
    fn numerical_memory_partition_is_checked_without_double_counting() {
        let mut m = manifest();
        m.logical_memory.positional_bytes = 7;
        m.logical_memory.history_bytes = 11;
        m.logical_memory.logits_bytes = 13;
        m.logical_memory.metadata_bytes = 17;
        m.logical_memory.metadata_allowance_bytes = 19;
        m.logical_memory.admitted_bytes = 67;
        m.session_admitted_bytes = 90;
        let mut c = Collector::new(true);
        c.set_manifest(&m);
        let mut r = record(1);
        r.memory.owner_reserved_bytes = 23;
        r.memory.session_admitted_bytes = 90;
        c.record(&r);
        let output = rows(&mut c);
        assert_eq!(output[0]["logical_memory"]["positional_bytes"], 7);
        assert_eq!(output[0]["numerical_session_admitted_bytes"], 67);
        assert_eq!(
            output[0]["additional_owner_and_external_admitted_bytes"],
            23
        );
        assert_eq!(output[2]["retained"], 1);
        m.logical_memory.weights_bytes = usize::MAX;
        let mut invalid = Collector::new(true);
        invalid.set_manifest(&m);
        assert_eq!(invalid.status().invalid_records, 1);
        assert_eq!(invalid.header_len, 0);
        let mut mismatch = record(2);
        mismatch.memory.session_admitted_bytes = 90;
        c.record(&mismatch);
        assert_eq!(c.status().invalid_records, 1);
    }
    #[test]
    fn reservation_failure_drops_only_metadata() {
        // try_reserve_exact rejects this allocation-size overflow without
        // attempting a giant allocation. It uses the production failure branch.
        let mut c = Collector::with_reservation(true, usize::MAX);
        assert!(c.ring.is_empty());
        assert_eq!(c.status().telemetry_dropped, 1);
        c.set_manifest(&manifest());
        c.record(&record(1));
        assert_eq!(c.status().telemetry_dropped, 2);
        assert_eq!(c.status().retained, 0);
        let output = rows(&mut c);
        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["retained_ring_capacity_bytes"], 0);
        assert_eq!(
            output[0]["collector_fixed_bytes"],
            std::mem::size_of::<Collector>()
        );
        assert_eq!(
            output[0]["collector_peak_bytes"],
            std::mem::size_of::<Collector>()
        );
        assert_eq!(output[0]["telemetry_admitted_bytes"], ALLOWANCE_BYTES);
        assert_eq!(output[1]["telemetry_dropped"], 2);
    }
    #[test]
    fn invalid_metadata_and_counter_saturation_are_infallible() {
        let mut c = collector();
        c.counters.telemetry_dropped = u64::MAX;
        c.counters.invalid_records = u64::MAX;
        let mut r = record(1);
        r.timing.request_total_ms = f64::NAN;
        c.record(&r);
        assert_eq!(c.status().telemetry_dropped, u64::MAX);
        assert_eq!(c.status().invalid_records, u64::MAX);
        assert!(c.status().counter_overflow);
        c.counters.export_errors = u64::MAX;
        assert!(c.export_to(Broken).is_err());
        assert_eq!(c.status().export_errors, u64::MAX);
        assert!(c.status().counter_overflow);
        assert_eq!(c.status().retained, 0);
    }
    #[test]
    fn failed_export_counts_once_and_retains_metadata() {
        let mut c = collector();
        c.record(&record(1));
        assert!(c.export_to(Broken).is_err());
        assert_eq!(c.status().export_errors, 1);
        let output = rows(&mut c);
        assert_eq!(output.last().unwrap()["export_errors"], 1);
        assert_eq!(c.status().retained, 1);
    }
    #[test]
    fn terminal_output_correction_preserves_payload_and_marks_replay() {
        let mut c = collector();
        let mut r = record(1);
        c.record(&r);
        let before_status = serde_json::to_value(c.status()).unwrap();
        r.terminal_status = Outcome::OutputError;
        r.failed_phase = Some(Phase::Output);
        r.error_category = Some(ErrorCategory::Output);
        r.replay_required = true;
        c.correct_last_terminal(&r);
        assert_eq!(serde_json::to_value(c.status()).unwrap(), before_status);
        let output = rows(&mut c);
        assert_eq!(output[1]["terminal_status"], "output_error");
        assert_eq!(output[1]["replay_required"], true);
        // Reapplying true->true remains an in-place correction with no counters.
        c.correct_last_terminal(&r);
        assert_eq!(serde_json::to_value(c.status()).unwrap(), before_status);
        // Clearing replay is prohibited even when every other byte matches.
        r.replay_required = false;
        c.correct_last_terminal(&r);
        assert_eq!(c.status().invalid_records, 1);
        assert_eq!(rows(&mut c)[1]["replay_required"], true);
        // A separately valid clock mutation is also forbidden by byte equality.
        r.replay_required = true;
        r.timing.request_total_ms = 1.;
        c.correct_last_terminal(&r);
        assert_eq!(c.status().invalid_records, 2);
        assert_eq!(rows(&mut c)[1]["timing"]["request_total_ms"], 0.);
        assert_eq!(c.status().retained, 1);
        assert_eq!(c.status().telemetry_dropped, 0);
    }
    fn completed_record() -> Record<'static> {
        let mut r = record(1);
        r.request_kind = Kind::Generate;
        r.terminal_status = Outcome::Success;
        r.failed_phase = None;
        r.error_category = None;
        r.cache_action = CacheAction::Fresh;
        r.after = State {
            history_tokens: 5,
            committed_tokens: 4,
            pending_tokens: 1,
        };
        r.work = Work {
            prompt_tokens: Some(3),
            reused_prefix_tokens: Some(0),
            prefill_rows: Some(3),
            decode_evaluations: Some(1),
            sampled_tokens: Some(2),
            emitted_tokens: Some(2),
            callback_tokens: 2,
            replayed_rows: Some(0),
        };
        r.timing = Timing {
            request_total_ms: 10.,
            native_total_ms: Some(9.),
            tokenization_ms: Some(0.5),
            prefill_ms: Some(3.),
            decode_ms: Some(4.),
            sampling_ms: Some(1.),
            ttft_event_ms: Some(4.),
            last_token_event_ms: Some(9.),
            token_interval_count: 1,
            token_interval_min_ms: Some(5.),
            token_interval_max_ms: Some(5.),
            token_interval_sum_ms: 5.,
            ..Timing::default()
        };
        r
    }
    #[test]
    fn completed_report_count_and_state_inconsistencies_drop_metadata() {
        assert!(valid_record(&completed_record(), 512));
        let mut c = collector();
        let mut r = completed_record();
        r.work.prompt_tokens = None;
        c.record(&r);
        let mut r = completed_record();
        r.work.prefill_rows = Some(4);
        c.record(&r);
        let mut r = completed_record();
        r.work.prefill_rows = Some(2);
        c.record(&r);
        let mut r = completed_record();
        r.work.replayed_rows = Some(4);
        c.record(&r);
        let mut r = completed_record();
        r.after.history_tokens = 6;
        r.after.committed_tokens = 5;
        c.record(&r);
        let mut r = completed_record();
        r.after.pending_tokens = 0;
        r.after.committed_tokens = 5;
        c.record(&r);
        let mut r = completed_record();
        r.work.emitted_tokens = Some(1);
        c.record(&r);
        let mut r = completed_record();
        r.work.decode_evaluations = Some(2);
        c.record(&r);
        assert_eq!(c.status().invalid_records, 8);
        assert_eq!(c.status().telemetry_dropped, 8);
        assert_eq!(c.status().retained, 0);
        c.record(&completed_record());
        assert_eq!(c.status().retained, 1);
    }
    #[test]
    fn partial_cancelled_and_output_error_reports_keep_partial_prefill() {
        let mut r = completed_record();
        r.terminal_status = Outcome::Cancelled;
        r.after = State {
            history_tokens: 3,
            committed_tokens: 1,
            pending_tokens: 2,
        };
        r.work.prefill_rows = Some(1);
        r.work.decode_evaluations = Some(0);
        r.work.sampled_tokens = Some(0);
        r.work.emitted_tokens = Some(0);
        r.work.callback_tokens = 0;
        r.timing.native_total_ms = Some(3.);
        r.timing.decode_ms = Some(0.);
        r.timing.sampling_ms = Some(0.);
        r.timing.ttft_event_ms = None;
        r.timing.last_token_event_ms = None;
        r.timing.token_interval_count = 0;
        r.timing.token_interval_min_ms = None;
        r.timing.token_interval_max_ms = None;
        r.timing.token_interval_sum_ms = 0.;
        assert!(valid_record(&r, 512));
        r.terminal_status = Outcome::OutputError;
        r.failed_phase = Some(Phase::Output);
        r.error_category = Some(ErrorCategory::Output);
        r.replay_required = true;
        assert!(valid_record(&r, 512));
        r.work.replayed_rows = Some(2);
        assert!(!valid_record(&r, 512));
    }
    #[test]
    fn unassignable_identity_marks_missing_metadata_without_a_record() {
        let mut c = collector();
        c.note_counter_overflow();
        c.note_counter_overflow();
        let status = c.status();
        assert!(status.counter_overflow);
        assert_eq!(status.telemetry_dropped, 2);
        assert_eq!(status.invalid_records, 0);
        assert_eq!(status.retained, 0);
        let output = rows(&mut c);
        assert_eq!(output.len(), 2);
        assert_eq!(output[1]["counter_overflow"], true);
        assert_eq!(output[1]["telemetry_dropped"], 2);
    }
    #[test]
    fn callback_and_completed_rates_stay_distinct() {
        let mut r = completed_record();
        r.work.prompt_tokens = Some(4);
        r.work.prefill_rows = Some(4);
        r.work.callback_tokens = 3;
        r.work.sampled_tokens = Some(3);
        r.work.emitted_tokens = Some(3);
        r.work.decode_evaluations = Some(2);
        r.after = State {
            history_tokens: 7,
            committed_tokens: 6,
            pending_tokens: 1,
        };
        r.timing.request_total_ms = 40.;
        r.timing.prefill_ms = Some(2.);
        r.timing.decode_ms = Some(5.);
        r.timing.ttft_event_ms = Some(10.);
        r.timing.last_token_event_ms = Some(30.);
        r.timing.token_interval_count = 2;
        r.timing.token_interval_min_ms = Some(10.);
        r.timing.token_interval_max_ms = Some(10.);
        r.timing.token_interval_sum_ms = 20.;
        assert!(valid_record(&r, 512));
        let value = serde_json::to_value(summary(&r)).unwrap();
        assert_eq!(value["prefill_tokens_per_s"], 2000.);
        assert_eq!(value["decode_tokens_per_s"], 100.);
        assert_eq!(value["decode_evaluations_per_s"], 400.);
        r.work.callback_tokens = 1;
        r.work.sampled_tokens = Some(1);
        r.work.emitted_tokens = Some(1);
        r.work.decode_evaluations = Some(0);
        r.after = State {
            history_tokens: 5,
            committed_tokens: 4,
            pending_tokens: 1,
        };
        r.timing.decode_ms = Some(0.);
        r.timing.last_token_event_ms = Some(10.);
        r.timing.token_interval_count = 0;
        r.timing.token_interval_min_ms = None;
        r.timing.token_interval_max_ms = None;
        r.timing.token_interval_sum_ms = 0.;
        assert!(valid_record(&r, 512));
        assert!(summary(&r).decode_tokens_per_s.is_none());
        assert!(summary(&r).decode_evaluations_per_s.is_none());
        r.work.callback_tokens = 0;
        r.work.sampled_tokens = Some(0);
        r.work.emitted_tokens = Some(0);
        r.timing.sampling_ms = Some(0.);
        r.after = State {
            history_tokens: 4,
            committed_tokens: 4,
            pending_tokens: 0,
        };
        r.timing.ttft_event_ms = None;
        r.timing.last_token_event_ms = None;
        assert!(valid_record(&r, 512));
        assert!(summary(&r).decode_tokens_per_s.is_none());
        assert!(value["ordinary_observations"]["gpu_ms"].is_null());
    }
}
