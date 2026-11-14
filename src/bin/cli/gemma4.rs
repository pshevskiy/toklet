//! Ordinary pinned Gemma BF16 CLI; numerical execution stays in the owner.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod summary;

#[cfg(any(
    test,
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
fn write_line(out: &mut impl std::io::Write, value: impl std::fmt::Display) -> std::io::Result<()> {
    writeln!(out, "{value}")?;
    out.flush()
}

pub(super) fn run(args: super::Args) -> Result<u8, Box<dyn std::error::Error>> {
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    {
        mac::run(args)
    }
    #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = args;
        Err(super::invalid("Gemma BF16 requires the metal feature on Apple Silicon macOS").into())
    }
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod mac {
    use super::super::{Action, Args, CANCELLED, chat, invalid, publish_terminal, token_list};
    use super::summary::{self, CacheAction, ErrorCategory, Kind, Outcome, Phase};
    use serde::{Deserialize, Serialize};
    use serde_json::json;
    use std::{
        error::Error,
        fs::OpenOptions,
        io::{self, BufRead, Read, Write},
        os::unix::fs::OpenOptionsExt,
        path::Path,
        sync::{atomic::Ordering, mpsc},
        time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    };
    use toklet::{
        app_chat::{
            gemma4,
            stream::{ResponseLimits, ResponseStream},
        },
        generation::{self, Input, Request, StopReason},
        metal::gemma_affine::OfficialBf16,
        session::{InferenceSession, Options},
        weights::INITIAL_MEMORY_LIMIT_BYTES,
    };

    const APP: usize = 128 * 1024 * 1024;
    const HELPER: usize = 32 * 1024 * 1024;
    type Result<T> = std::result::Result<T, Box<dyn Error>>;

    // Typed messages preserve the pinned codec's duplicate-field rejection.
    // Parsing through serde_json::Value first would silently discard duplicates.
    #[derive(Deserialize, Serialize)]
    #[serde(rename_all = "lowercase")]
    enum Role {
        User,
        Assistant,
    }
    #[derive(Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct Message {
        role: Role,
        content: String,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct UserRequest {
        #[serde(default)]
        action: Action,
        tokens: Option<Vec<u32>>,
        prompt: Option<String>,
        messages: Option<Vec<Message>>,
        tools: Option<serde_json::Value>,
        raw: Option<bool>,
        max_new_tokens: Option<usize>,
    }

    fn add(a: usize, b: usize) -> io::Result<usize> {
        a.checked_add(b)
            .ok_or_else(|| invalid("Gemma CLI memory accounting overflow"))
    }
    fn read_regular(path: &Path, limit: usize) -> io::Result<Vec<u8>> {
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path)?;
        if !file.metadata()?.is_file() {
            return Err(invalid("input must be a regular file"));
        }
        let mut bytes = Vec::new();
        (&mut file)
            .take(
                limit
                    .checked_add(1)
                    .ok_or_else(|| invalid("input size overflow"))? as u64,
            )
            .read_to_end(&mut bytes)?;
        if bytes.len() > limit {
            return Err(invalid("input exceeds its byte bound"));
        }
        Ok(bytes)
    }
    fn snapshot(session: &impl InferenceSession) -> summary::State {
        summary::State {
            history_tokens: session.history().len(),
            committed_tokens: session.evaluated_len(),
            pending_tokens: session.pending_len(),
        }
    }

    struct Failure {
        phase: Phase,
        category: ErrorCategory,
        outcome: Outcome,
        message: String,
    }
    impl Failure {
        fn rejected(phase: Phase, error: impl std::fmt::Display) -> Self {
            Self {
                phase,
                category: ErrorCategory::InvalidInput,
                outcome: Outcome::Rejected,
                message: error.to_string(),
            }
        }
        fn output(error: impl std::fmt::Display) -> Self {
            Self {
                phase: Phase::Output,
                category: ErrorCategory::Output,
                outcome: Outcome::OutputError,
                message: error.to_string(),
            }
        }
        fn generation(error: generation::Error) -> Self {
            let (category, outcome, phase) = match &error {
                generation::Error::Output(_) => {
                    (ErrorCategory::Output, Outcome::OutputError, Phase::Output)
                }
                generation::Error::Sampling(_) => (
                    ErrorCategory::Sampling,
                    Outcome::ExecutionError,
                    Phase::Execution,
                ),
                generation::Error::Session(_) => (
                    ErrorCategory::Session,
                    Outcome::ExecutionError,
                    Phase::Execution,
                ),
                generation::Error::InvalidRequest(_) => (
                    ErrorCategory::InvalidInput,
                    Outcome::ExecutionError,
                    Phase::Execution,
                ),
            };
            Self {
                phase,
                category,
                outcome,
                message: error.to_string(),
            }
        }
    }
    struct Attempt {
        kind: Kind,
        cache: CacheAction,
        work: summary::Work,
        timing: summary::Timing,
        report: Option<generation::Report>,
        failure: Option<Failure>,
        cancelled: bool,
        replay_required: bool,
    }
    impl Default for Attempt {
        fn default() -> Self {
            Self {
                kind: Kind::Invalid,
                cache: CacheAction::Unchanged,
                work: summary::Work::default(),
                timing: summary::Timing::default(),
                report: None,
                failure: None,
                cancelled: false,
                replay_required: false,
            }
        }
    }
    struct Runner<'session, 'codec, S> {
        session: &'session mut S,
        codec: &'codec gemma4::Codec,
        stream: Option<ResponseStream<'codec>>,
        collector: summary::Collector,
        session_id: String,
        next_id: u64,
        memory: summary::Memory,
        logical_memory: toklet::session::Memory,
        stops: Vec<u32>,
        max: usize,
        raw: bool,
        json: bool,
        stats: bool,
        interactive: bool,
        metrics: bool,
        output_failed: bool,
        submission: &'static str,
    }

    fn write_content(
        out: &mut impl Write,
        text: &str,
        event: Option<generation::TokenEvent>,
        json_output: bool,
        metrics: bool,
        started: Instant,
        timing: &mut summary::Timing,
    ) -> io::Result<()> {
        let clock = metrics.then(Instant::now);
        let result = (|| {
            if json_output {
                if let Some(event) = event {
                    writeln!(
                        out,
                        "{}",
                        json!({"event":"token", "token_id":event.token_id,
                        "index":event.index, "elapsed_seconds":event.elapsed_seconds, "text":text})
                    )?;
                } else if !text.is_empty() {
                    writeln!(out, "{}", json!({"event":"text", "text":text}))?;
                }
            } else {
                out.write_all(text.as_bytes())?;
            }
            out.flush()
        })();
        if let Some(clock) = clock {
            timing.output_publication_ms += clock.elapsed().as_secs_f64() * 1000.;
            if result.is_ok() && !text.is_empty() && timing.first_text_publication_ms.is_none() {
                timing.first_text_publication_ms = Some(started.elapsed().as_secs_f64() * 1000.);
            }
        }
        result
    }

    impl<S: InferenceSession> Runner<'_, '_, S> {
        fn finish_response(
            &mut self,
            out: &mut impl Write,
            started: Instant,
            timing: &mut summary::Timing,
        ) -> io::Result<()> {
            if let Some(stream) = &mut self.stream
                && !stream.is_finished()
                && !stream.is_failed()
            {
                let clock = self.metrics.then(Instant::now);
                let result = stream.finish();
                if let Some(clock) = clock {
                    timing.output_decode_ms += clock.elapsed().as_secs_f64() * 1000.;
                }
                write_content(out, result?, None, self.json, self.metrics, started, timing)?;
            }
            Ok(())
        }

        fn execute(
            &mut self,
            request: UserRequest,
            out: &mut impl Write,
            started: Instant,
            attempt: &mut Attempt,
        ) -> std::result::Result<(), Failure> {
            attempt.kind = match request.action {
                Action::Generate => Kind::Generate,
                Action::Append => Kind::Append,
                Action::Continue => Kind::Continue,
                Action::Reset => Kind::Reset,
            };
            let count = usize::from(request.tokens.is_some())
                + usize::from(request.prompt.is_some())
                + usize::from(request.messages.is_some());
            if request.tools.is_some() {
                return Err(Failure::rejected(
                    Phase::Admission,
                    "Gemma basic text does not accept tools",
                ));
            }
            if request.action == Action::Reset {
                if count != 0 || request.raw.is_some() || request.max_new_tokens.is_some() {
                    return Err(Failure::rejected(
                        Phase::Admission,
                        "reset accepts no other fields",
                    ));
                }
                if CANCELLED.load(Ordering::Relaxed) {
                    attempt.cancelled = true;
                    return Ok(());
                }
                if self.output_failed {
                    self.stream = None;
                }
                self.finish_response(out, started, &mut attempt.timing)
                    .map_err(Failure::output)?;
                self.session.try_reset().map_err(|error| Failure {
                    phase: Phase::Execution,
                    category: ErrorCategory::Session,
                    outcome: Outcome::ExecutionError,
                    message: error.to_string(),
                })?;
                self.stream = None;
                self.output_failed = false;
                attempt.replay_required = false;
                attempt.cache = CacheAction::Reset;
                return Ok(());
            }
            if request.action == Action::Continue {
                if count != 0 || request.raw.is_some() {
                    return Err(Failure::rejected(
                        Phase::Admission,
                        "continue accepts only max_new_tokens",
                    ));
                }
            } else if count != 1 {
                return Err(Failure::rejected(
                    Phase::Admission,
                    "request needs exactly one tokens/prompt/messages field",
                ));
            }
            if request.tokens.is_some() && request.raw.is_some() {
                return Err(Failure::rejected(
                    Phase::Admission,
                    "token input cannot use raw",
                ));
            }
            if request.action == Action::Append && request.messages.is_some() {
                return Err(Failure::rejected(
                    Phase::Admission,
                    "append accepts raw prompt text or token IDs",
                ));
            }
            let tokens = if let Some(tokens) = request.tokens {
                tokens
            } else if count != 0 {
                let clock = Instant::now();
                let encoded = (|| {
                    if let Some(text) = request.prompt {
                        if request.action == Action::Append || request.raw.unwrap_or(self.raw) {
                            self.codec.encode_raw(&text, false)
                        } else {
                            self.codec.encode_chat(
                                &serde_json::to_vec(&json!([
                                {"role":"user", "content":text}]))
                                .map_err(invalid)?,
                                gemma4::RenderOptions {
                                    enable_thinking: false,
                                    add_generation_prompt: true,
                                },
                            )
                        }
                    } else {
                        if request.raw.unwrap_or(self.raw) {
                            return Err(invalid("raw mode cannot use messages"));
                        }
                        self.codec.encode_chat(
                            &serde_json::to_vec(&request.messages.unwrap()).map_err(invalid)?,
                            gemma4::RenderOptions {
                                enable_thinking: false,
                                add_generation_prompt: true,
                            },
                        )
                    }
                })();
                attempt.timing.tokenization_ms = Some(clock.elapsed().as_secs_f64() * 1000.);
                encoded.map_err(|error| Failure::rejected(Phase::Tokenization, error))?
            } else {
                Vec::new()
            };
            let max = request.max_new_tokens.unwrap_or(self.max);
            let input = match request.action {
                Action::Append => Input::Append(&tokens),
                Action::Continue => Input::Continue,
                _ => Input::Prompt(&tokens),
            };
            let admission = generation::request_admission(
                self.session,
                &Request {
                    input,
                    max_new_tokens: max,
                    stop_tokens: &self.stops,
                },
            )
            .map_err(|error| Failure::rejected(Phase::Admission, error))?;
            if request.action == Action::Continue && self.output_failed {
                return Err(Failure::rejected(
                    Phase::Admission,
                    "output failed; start a new response or reset before continuing",
                ));
            }
            let committed = self.session.evaluated_len();
            let replacing = request.action == Action::Generate
                && committed != 0
                && !tokens.starts_with(&self.session.history()[..committed]);
            attempt.cache = match request.action {
                Action::Append => CacheAction::Append,
                Action::Continue if self.session.pending_len() != 0 => CacheAction::Pending,
                Action::Continue => CacheAction::FullyCached,
                _ if self.session.history().is_empty() => CacheAction::Fresh,
                _ if replacing => CacheAction::Replay,
                _ if tokens.len() == committed => CacheAction::FullyCached,
                _ => CacheAction::ExactPrefix,
            };
            if !CANCELLED.load(Ordering::Relaxed) {
                if request.action != Action::Continue {
                    if self.output_failed {
                        self.stream = None;
                    }
                    self.finish_response(out, started, &mut attempt.timing)
                        .map_err(Failure::output)?;
                    self.stream = None;
                    self.output_failed = false;
                    attempt.replay_required = false;
                }
                if self.stream.as_ref().is_some_and(ResponseStream::is_failed) {
                    return Err(Failure::rejected(
                        Phase::Admission,
                        "response stream failed; generate or reset before continuing",
                    ));
                }
                if self.stream.as_ref().is_none_or(ResponseStream::is_finished) {
                    self.stream = Some(
                        self.codec
                            .response_stream(
                                ResponseLimits::new(
                                    self.session.capacity(),
                                    gemma4::MAX_TEXT_BYTES,
                                )
                                .map_err(Failure::output)?,
                                gemma4::STREAM_ALLOWANCE_BYTES,
                            )
                            .map_err(Failure::output)?,
                    );
                }
            }
            let stream = &mut self.stream;
            let json_output = self.json;
            let metrics = self.metrics;
            let work = &mut attempt.work;
            let timing = &mut attempt.timing;
            let generated = generation::generate(
                self.session,
                Request {
                    input,
                    max_new_tokens: max,
                    stop_tokens: &self.stops,
                },
                &CANCELLED,
                |event| {
                    if metrics {
                        let now = started.elapsed().as_secs_f64() * 1000.;
                        work.callback_tokens = work
                            .callback_tokens
                            .checked_add(1)
                            .ok_or_else(|| invalid("callback counter overflow"))?;
                        if let Some(previous) = timing.last_token_event_ms {
                            let interval = now - previous;
                            timing.token_interval_count = timing
                                .token_interval_count
                                .checked_add(1)
                                .ok_or_else(|| invalid("token interval counter overflow"))?;
                            timing.token_interval_min_ms = Some(
                                timing
                                    .token_interval_min_ms
                                    .map_or(interval, |old| old.min(interval)),
                            );
                            timing.token_interval_max_ms = Some(
                                timing
                                    .token_interval_max_ms
                                    .map_or(interval, |old| old.max(interval)),
                            );
                            timing.token_interval_sum_ms += interval;
                        }
                        timing.ttft_event_ms.get_or_insert(now);
                        timing.last_token_event_ms = Some(now);
                    }
                    let decoder = stream
                        .as_mut()
                        .ok_or_else(|| invalid("response stream unavailable"))?;
                    let clock = metrics.then(Instant::now);
                    let decoded = decoder.push(&[event.token_id]);
                    if let Some(clock) = clock {
                        timing.output_decode_ms += clock.elapsed().as_secs_f64() * 1000.;
                    }
                    write_content(
                        out,
                        decoded?,
                        Some(event),
                        json_output,
                        metrics,
                        started,
                        timing,
                    )
                },
            );
            match generated {
                Ok(report) => {
                    // Append and Continue include their existing history in
                    // the logical prompt; supplied tokens alone omit reuse.
                    attempt.work.prompt_tokens = Some(admission.history_after_input_tokens);
                    attempt.work.reused_prefix_tokens = Some(report.reused_tokens);
                    attempt.work.prefill_rows = Some(report.prefill_tokens);
                    attempt.work.decode_evaluations = Some(report.decode_evaluations);
                    attempt.work.sampled_tokens = Some(report.sampled_tokens);
                    attempt.work.emitted_tokens = Some(report.emitted_tokens);
                    // Only completed prefill rows overlapping the previously
                    // committed positions are replayed. Pending/new rows are
                    // evaluated for the first time, including after cancel.
                    attempt.work.replayed_rows = Some(
                        report.prefill_tokens.min(
                            committed
                                .min(admission.history_after_input_tokens)
                                .saturating_sub(report.reused_tokens),
                        ),
                    );
                    attempt.timing.native_total_ms = Some(report.total_seconds * 1000.);
                    attempt.timing.prefill_ms = Some(report.prefill_seconds * 1000.);
                    attempt.timing.decode_ms = Some(report.decode_seconds * 1000.);
                    attempt.timing.sampling_ms = Some(report.sampling_seconds * 1000.);
                    attempt.cancelled = report.reason == StopReason::Cancelled;
                    let finish = !self.interactive || report.reason != StopReason::OutputLimit;
                    attempt.report = Some(report);
                    if finish {
                        self.finish_response(out, started, &mut attempt.timing)
                            .map_err(Failure::output)?;
                    }
                    Ok(())
                }
                Err(error) => {
                    attempt.replay_required =
                        matches!(
                            error,
                            generation::Error::Session(_)
                                | generation::Error::Sampling(_)
                                | generation::Error::Output(_)
                        ) || self.stream.as_ref().is_some_and(ResponseStream::is_failed);
                    Err(Failure::generation(error))
                }
            }
        }

        fn handle(
            &mut self,
            out: &mut impl Write,
            parse: impl FnOnce() -> io::Result<UserRequest>,
        ) -> Result<(bool, bool)> {
            let started = Instant::now();
            let id = self.next_id;
            self.next_id = match self.next_id.checked_add(1) {
                Some(next) => next,
                None => {
                    self.collector.note_counter_overflow();
                    return Err(invalid("Gemma request identity overflow").into());
                }
            };
            let before = snapshot(self.session);
            let mut attempt = Attempt {
                replay_required: self.output_failed,
                ..Attempt::default()
            };
            let result = parse()
                .map_err(|error| Failure::rejected(Phase::Parse, error))
                .and_then(|request| self.execute(request, out, started, &mut attempt));
            attempt.failure = result.err();
            if attempt
                .failure
                .as_ref()
                .is_some_and(|failure| matches!(failure.outcome, Outcome::OutputError))
            {
                self.output_failed = true;
                attempt.replay_required = true;
            }
            publish_terminal(&CANCELLED, |active_cancelled| -> Result<(bool, bool)> {
                let completed_stop = attempt
                    .report
                    .as_ref()
                    .is_some_and(|report| matches!(report.reason, StopReason::StopToken { .. }));
                if active_cancelled && attempt.failure.is_none() && !completed_stop {
                    attempt.cancelled = true;
                    if let Some(report) = &mut attempt.report
                        && report.reason == StopReason::OutputLimit
                    {
                        report.reason = StopReason::Cancelled;
                    }
                    if let Err(error) = self.finish_response(out, started, &mut attempt.timing) {
                        attempt.failure = Some(Failure::output(error));
                        self.output_failed = true;
                        attempt.replay_required = true;
                    }
                }
                let after = snapshot(self.session);
                attempt.timing.request_total_ms = started.elapsed().as_secs_f64() * 1000.;
                let mut memory = self.memory;
                memory.kv_valid_positions = after.committed_tokens;
                let outcome = if let Some(failure) = &attempt.failure {
                    failure.outcome
                } else if attempt.cancelled {
                    Outcome::Cancelled
                } else if attempt
                    .report
                    .as_ref()
                    .is_some_and(|report| matches!(report.reason, StopReason::StopToken { .. }))
                {
                    Outcome::Stop
                } else {
                    Outcome::Success
                };
                let mut record = summary::Record {
                    session_id: &self.session_id,
                    request_id: id,
                    attempt_id: id,
                    owner_generation: 1,
                    request_kind: attempt.kind,
                    terminal_status: outcome,
                    failed_phase: attempt.failure.as_ref().map(|failure| failure.phase),
                    error_category: attempt.failure.as_ref().map(|failure| failure.category),
                    cache_action: attempt.cache,
                    replay_required: attempt.replay_required,
                    before,
                    after,
                    work: attempt.work,
                    timing: attempt.timing,
                    memory,
                };
                self.collector.record(&record);
                let terminal = (|| -> io::Result<()> {
                    if let Some(failure) = &attempt.failure {
                        if self.json {
                            writeln!(
                                out,
                                "{}",
                                json!({"event":"error", "request_id":id,
                                "message":failure.message, "state":after})
                            )?;
                        } else {
                            super::write_line(
                                &mut io::stderr().lock(),
                                format_args!("error: {}", failure.message),
                            )?;
                        }
                    } else if self.json {
                        if matches!(attempt.kind, Kind::Reset) && !attempt.cancelled {
                            writeln!(
                                out,
                                "{}",
                                json!({"event":"reset", "request_id":id, "state":after})
                            )?;
                        } else {
                            writeln!(
                                out,
                                "{}",
                                json!({"event":"done", "request_id":id,
                                "statistics": {"model_profile":"gemma4-bf16", "backend":"metal",
                                "submission":self.submission, "generation":attempt.report,
                                "decode_evaluations":record.work.decode_evaluations,
                                "state":after, "memory_admitted_bytes":self.memory.session_admitted_bytes,
                                "logical_memory":self.logical_memory}})
                            )?;
                        }
                    } else {
                        writeln!(out)?;
                        if self.stats {
                            super::write_line(
                                &mut io::stderr().lock(),
                                json!({"generation":attempt.report, "state":after,
                                    "memory_admitted_bytes":self.memory.session_admitted_bytes}),
                            )?;
                        }
                    }
                    out.flush()
                })();
                if let Err(error) = terminal {
                    record.terminal_status = Outcome::OutputError;
                    record.failed_phase = Some(Phase::Output);
                    record.error_category = Some(ErrorCategory::Output);
                    record.replay_required = true;
                    self.output_failed = true;
                    self.collector.correct_last_terminal(&record);
                    return Err(error.into());
                }
                Ok((attempt.cancelled, attempt.failure.is_some()))
            })
        }

        fn serve(&mut self, args: &mut Args) -> Result<u8> {
            let mut out = io::stdout().lock();
            if !self.interactive {
                let (cancelled, failed) = self.handle(&mut out, || initial(args))?;
                return Ok(if cancelled {
                    130
                } else if failed {
                    1
                } else {
                    0
                });
            }
            writeln!(
                out,
                "{}",
                json!({"event":"ready", "model_profile":"gemma4-bf16",
                "backend":"metal", "context_capacity":self.session.capacity(),
                "submission":self.submission, "memory_admitted_bytes":self.memory.session_admitted_bytes,
                "logical_memory":self.logical_memory})
            )?;
            out.flush()?;
            let (sender, receiver) = mpsc::sync_channel(1);
            std::thread::spawn(move || {
                let mut input = io::BufReader::new(io::stdin());
                loop {
                    let mut line = Vec::new();
                    let read = input
                        .by_ref()
                        .take((chat::MAX_INPUT_BYTES + 1) as u64)
                        .read_until(b'\n', &mut line);
                    match read {
                        Ok(0) => break,
                        Ok(_) if line.len() <= chat::MAX_INPUT_BYTES => {
                            if sender.send(Ok(line)).is_err() {
                                break;
                            }
                        }
                        Ok(_) => {
                            let _ = sender.send(Err(invalid("interactive request exceeds1MiB")));
                            break;
                        }
                        Err(error) => {
                            let _ = sender.send(Err(error));
                            break;
                        }
                    }
                }
            });
            loop {
                if CANCELLED.load(Ordering::Relaxed) {
                    self.finish_response(
                        &mut out,
                        Instant::now(),
                        &mut summary::Timing::default(),
                    )?;
                    return Ok(130);
                }
                let line = match receiver.recv_timeout(Duration::from_millis(50)) {
                    Ok(line) => line,
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        self.finish_response(
                            &mut out,
                            Instant::now(),
                            &mut summary::Timing::default(),
                        )?;
                        return Ok(0);
                    }
                };
                if line
                    .as_ref()
                    .is_ok_and(|line| line.iter().all(u8::is_ascii_whitespace))
                {
                    continue;
                }
                // Active cancellation consumes only this handled request's
                // flag. Keep its valid prefix available to continue/reset.
                // A new signal during publication survives the boundary and
                // is handled by the idle flag check at the next loop entry.
                self.handle(&mut out, || serde_json::from_slice(&line?).map_err(invalid))?;
            }
        }
    }

    fn initial(args: &mut Args) -> io::Result<UserRequest> {
        let tokens = if let Some(text) = args.tokens.take() {
            Some(token_list(&text)?)
        } else if let Some(path) = args.token_file.take() {
            let bytes = read_regular(
                &path,
                args.capacity
                    .checked_mul(4)
                    .ok_or_else(|| invalid("token input bound overflow"))?,
            )?;
            if !bytes.len().is_multiple_of(4) {
                return Err(invalid("token file must contain complete u32 values"));
            }
            Some(
                bytes
                    .chunks_exact(4)
                    .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                    .collect(),
            )
        } else {
            None
        };
        let messages = args
            .messages
            .take()
            .map(|path| {
                serde_json::from_slice(&read_regular(&path, gemma4::MAX_TEXT_BYTES)?)
                    .map_err(invalid)
            })
            .transpose()?;
        Ok(UserRequest {
            action: Action::Generate,
            tokens,
            prompt: args.prompt.take(),
            messages,
            tools: None,
            raw: None,
            max_new_tokens: None,
        })
    }

    pub(super) fn run(mut args: Args) -> Result<u8> {
        let route_start = Instant::now();
        ctrlc::try_set_handler(|| CANCELLED.store(true, Ordering::Relaxed))?;
        let telemetry = if args.request_metrics {
            summary::ALLOWANCE_BYTES
        } else {
            0
        };
        let external = add(
            add(add(APP, gemma4::STREAM_ALLOWANCE_BYTES)?, telemetry)?,
            HELPER,
        )?;
        let owner_budget = INITIAL_MEMORY_LIMIT_BYTES
            .checked_sub(external)
            .ok_or_else(|| invalid("Gemma CLI allowances exceed6GiB"))?;
        let load_start = Instant::now();
        let owner = OfficialBf16::load(&args.model, args.capacity, owner_budget)?;
        let load_ms = load_start.elapsed().as_secs_f64() * 1000.;
        let model = owner.model()?;
        let options = Options {
            capacity: args.capacity,
            prefill_chunk_size: args.chunk,
            position_offset: 0,
            memory_limit_bytes: owner_budget,
        };
        let mut session = if args.submission == "batched-single-row" {
            owner.new_batched_g2_session(&model, options)?
        } else {
            owner.new_g2_session(&model, options)?
        };
        let logical = session.memory();
        let owner_memory = owner.memory();
        let load_admitted = add(owner_memory.load_admitted_bytes, external)?;
        let session_admitted = add(
            add(logical.admitted_bytes, owner_memory.session_reserved_bytes)?,
            external,
        )?;
        if load_admitted.max(session_admitted) > INITIAL_MEMORY_LIMIT_BYTES {
            return Err(invalid("Gemma CLI whole-owner admission exceeds6GiB").into());
        }
        let mut collector = summary::Collector::new(args.request_metrics);
        // Disabled collection has no ring; its fixed facade is covered by APP.
        if args.request_metrics && collector.allocated_bytes() > telemetry {
            return Err(invalid("Gemma CLI collector exceeds its admitted allowance").into());
        }
        let epoch = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let session_id = format!("gemma-{}-{epoch:x}", std::process::id());
        if session_id.len() > 64 {
            return Err(invalid("session identity exceeds bound").into());
        }
        collector.set_manifest(&summary::Manifest {
            session_id: &session_id,
            capacity: args.capacity,
            prefill_chunk_size: args.chunk,
            submission: args.submission,
            load_ms,
            route_start_to_ready_ms: route_start.elapsed().as_secs_f64() * 1000.,
            load_admitted_bytes: load_admitted,
            session_admitted_bytes: session_admitted,
            logical_memory: logical,
        });
        let mut stops = if args.no_eos {
            Vec::new()
        } else {
            owner.codec().stop_ids().to_vec()
        };
        stops.append(&mut args.stop);
        let mut runner = Runner {
            session: &mut session,
            codec: owner.codec(),
            stream: None,
            collector,
            session_id,
            next_id: 1,
            logical_memory: logical,
            memory: summary::Memory {
                weights_owned_bytes: Some(logical.weights_bytes),
                kv_allocated_bytes: Some(logical.kv_bytes),
                scratch_bytes: Some(logical.scratch_bytes),
                kv_valid_positions: 0,
                owner_reserved_bytes: owner_memory.session_reserved_bytes,
                load_admitted_bytes: load_admitted,
                session_admitted_bytes: session_admitted,
                app_allowance_bytes: APP,
                stream_allowance_bytes: gemma4::STREAM_ALLOWANCE_BYTES,
                telemetry_allowance_bytes: telemetry,
                helper_reserve_bytes: HELPER,
            },
            stops,
            max: args.max,
            raw: args.raw,
            json: args.json || args.interactive,
            stats: args.stats,
            interactive: args.interactive,
            metrics: args.request_metrics,
            output_failed: false,
            submission: args.submission,
        };
        let inference_result = runner.serve(&mut args);
        // Telemetry publication follows active inference, including a failed
        // output sink. Its error never replaces the inference/session result.
        if let Some(path) = &args.metrics_output
            && let Err(error) = runner.collector.export(path)
        {
            // Fixed text plus a bounded enum; no panic if stderr is closed.
            let _ = super::write_line(
                &mut io::stderr().lock(),
                format_args!("request metrics export failed ({:?})", error.kind()),
            );
        }
        inference_result
    }
}

#[cfg(test)]
mod tests {
    use super::write_line;
    use std::io::{self, Write};

    struct FailingDiagnostic {
        fail_at_flush: bool,
    }
    impl Write for FailingDiagnostic {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.fail_at_flush {
                Ok(bytes.len())
            } else {
                Err(io::Error::from(io::ErrorKind::BrokenPipe))
            }
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::from(io::ErrorKind::BrokenPipe))
        }
    }

    #[test]
    fn requested_diagnostic_write_and_flush_errors_are_returned() {
        for fail_at_flush in [false, true] {
            let error =
                write_line(&mut FailingDiagnostic { fail_at_flush }, "statistics").unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
        }
        let mut bytes = Vec::new();
        write_line(&mut bytes, format_args!("error: {}", "invalid input")).unwrap();
        assert_eq!(bytes, b"error: invalid input\n");
    }
}
