//! Fixed ordinary continuation/cache matrix; no profiling or primary-bench edits.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use serde::{Deserialize, Serialize};
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::{
        fs::{self, File, OpenOptions},
        io::{self, Read, Seek, SeekFrom, Write},
        os::unix::fs::OpenOptionsExt,
        path::{Path, PathBuf},
        sync::atomic::{AtomicBool, Ordering},
        time::Instant,
    };
    use toklet::{
        app_chat::gemma4,
        generation::{self, Input, Request, StopReason},
        metal::gemma_affine::{Bf16Model, OfficialBf16},
        session::{InferenceSession, Options},
    };
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    const MIB: usize = 1024 * 1024;
    const LIMIT: usize = 6 * 1024 * MIB;
    const CALLER: usize = 16 * MIB;
    const HELPER: usize = 32 * MIB;
    const OWNER: usize = LIMIT - CALLER - HELPER;
    const LINE: usize = 32 * 1024;
    const LOG: usize = 8 * MIB;
    const TOKENS_SHA: &str = "20e4789c70ba16a364591f621b274e3532f09ccb1859a51b227cc5b9c14b2503";
    const CODEC: &str = "654dc6dfd66fd893a24f0faa39355b71d5e7ffc3f53c2dac394dfcfd9ed95298";
    const P: [u32; 15] = [
        2, 105, 2364, 107, 1567, 506, 5279, 529, 7001, 236761, 106, 107, 105, 4368, 107,
    ];
    const E: [u32; 15] = [
        2, 105, 2364, 107, 1567, 506, 5279, 529, 7001, 236761, 106, 107, 105, 4368, 2364,
    ];
    const Q: [u32; 4] = [818, 5279, 529, 7001];
    fn check(ok: bool, message: &str) -> io::Result<()> {
        if ok {
            Ok(())
        } else {
            Err(io::Error::other(message))
        }
    }
    fn add(a: usize, b: usize) -> io::Result<usize> {
        a.checked_add(b)
            .ok_or_else(|| io::Error::other("byte overflow"))
    }
    fn sha(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }
    fn ids_sha(ids: &[u32]) -> String {
        let mut hash = Sha256::new();
        for id in ids {
            hash.update(id.to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }
    fn read(path: &Path, bound: usize) -> Result<Vec<u8>> {
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW)
            .open(path)?;
        let meta = file.metadata()?;
        check(
            meta.is_file() && meta.len() <= bound as u64,
            "bounded regular input",
        )?;
        let mut bytes = Vec::with_capacity(bound + 1);
        Read::by_ref(&mut file)
            .take((bound + 1) as u64)
            .read_to_end(&mut bytes)?;
        check(
            bytes.len() as u64 == meta.len() && file.metadata()?.len() == meta.len(),
            "input EOF",
        )?;
        Ok(bytes)
    }
    struct Bytes {
        bytes: Vec<u8>,
        bound: usize,
    }
    impl Bytes {
        fn new(bound: usize) -> Self {
            Self {
                bytes: Vec::with_capacity(bound),
                bound,
            }
        }
    }
    impl Write for Bytes {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            check(
                add(self.bytes.len(), bytes.len())? <= self.bound,
                "JSON bound",
            )?;
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    fn publish(folder: &Path, name: &str, value: &impl Serialize, bound: usize) -> Result<()> {
        let mut bytes = Bytes::new(bound);
        serde_json::to_writer_pretty(&mut bytes, value)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(folder.join(name))?;
        file.write_all(&bytes.bytes)?;
        file.flush()?;
        Ok(())
    }
    struct Journal {
        file: File,
        buffer: Bytes,
        bytes: usize,
        records: usize,
    }
    impl Journal {
        fn append(&mut self, value: &Value) -> Result<()> {
            self.buffer.bytes.clear();
            serde_json::to_writer(&mut self.buffer, value)?;
            check(
                add(self.bytes, add(self.buffer.bytes.len(), 1)?)? <= LOG,
                "journal byte bound",
            )?;
            check(self.records < 256, "journal record bound")?;
            self.file.write_all(&self.buffer.bytes)?;
            self.file.write_all(b"\n")?;
            self.file.flush()?;
            self.bytes += self.buffer.bytes.len() + 1;
            self.records += 1;
            Ok(())
        }
        fn admit(&self) -> io::Result<()> {
            check(
                add(self.bytes, LINE + 1)? <= LOG && self.records < 256,
                "journal admission",
            )
        }
        fn digest(&mut self) -> Result<String> {
            self.file.flush()?;
            check(
                self.file.metadata()?.len() == self.bytes as u64,
                "journal EOF",
            )?;
            self.file.seek(SeekFrom::Start(0))?;
            let mut hash = Sha256::new();
            let mut block = [0; 16384];
            loop {
                let count = self.file.read(&mut block)?;
                if count == 0 {
                    break;
                }
                hash.update(&block[..count]);
            }
            Ok(format!("{:x}", hash.finalize()))
        }
    }
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Arm {
        A,
        B,
    }
    impl Arm {
        fn name(self) -> &'static str {
            if self == Self::A {
                "per-tile"
            } else {
                "batched-single-row"
            }
        }
    }
    #[derive(Clone, Copy)]
    enum Case {
        Reset,
        Append,
        Cached,
        Pending,
        Edited,
        Short,
        Boundary,
        Teacher128,
        Teacher384,
    }
    const CASES: [Case; 9] = [
        Case::Reset,
        Case::Append,
        Case::Cached,
        Case::Pending,
        Case::Edited,
        Case::Short,
        Case::Boundary,
        Case::Teacher128,
        Case::Teacher384,
    ];
    impl Case {
        fn name(self) -> &'static str {
            match self {
                Self::Reset => "reset",
                Self::Append => "append",
                Self::Cached => "fully_cached",
                Self::Pending => "pending",
                Self::Edited => "edited_prefix",
                Self::Short => "short_128_128",
                Self::Boundary => "boundary_prefill_512_0",
                Self::Teacher128 => "teacher_128_128",
                Self::Teacher384 => "teacher_384_128",
            }
        }
    }
    struct Args {
        model: PathBuf,
        tokens: PathBuf,
        output: PathBuf,
        prepare: bool,
        golden: Option<PathBuf>,
        pin: Option<String>,
        cohort: Option<usize>,
    }
    fn args() -> Result<Args> {
        let (mut model, mut tokens, mut output, mut golden, mut pin, mut cohort) =
            (None, None, None, None, None, None);
        let mut prepare = false;
        let mut input = std::env::args_os().skip(1);
        while let Some(flag) = input.next() {
            let flag = flag
                .to_str()
                .ok_or_else(|| io::Error::other("nonUTF8 flag"))?;
            if flag == "--prepare-goldens" {
                check(!prepare, "duplicate prepare")?;
                prepare = true;
                continue;
            }
            let value = input
                .next()
                .ok_or_else(|| io::Error::other("missing flag value"))?;
            match flag {
                "--model" => check(
                    model.replace(PathBuf::from(value)).is_none(),
                    "duplicate model",
                )?,
                "--tokens" => check(
                    tokens.replace(PathBuf::from(value)).is_none(),
                    "duplicate tokens",
                )?,
                "--output" => check(
                    output.replace(PathBuf::from(value)).is_none(),
                    "duplicate output",
                )?,
                "--goldens" => check(
                    golden.replace(PathBuf::from(value)).is_none(),
                    "duplicate goldens",
                )?,
                "--goldens-sha256" => check(
                    pin.replace(
                        value
                            .into_string()
                            .map_err(|_| io::Error::other("nonUTF8 pin"))?,
                    )
                    .is_none(),
                    "duplicate pin",
                )?,
                "--cohort" => {
                    let n = match value.to_str() {
                        Some("1") => 1,
                        Some("2") => 2,
                        _ => return Err(io::Error::other("cohort1or2").into()),
                    };
                    check(cohort.replace(n).is_none(), "duplicate cohort")?;
                }
                _ => return Err(io::Error::other("unknown option").into()),
            }
        }
        check(
            if prepare {
                golden.is_none() && pin.is_none() && cohort.is_none()
            } else {
                golden.is_some() && pin.is_some() && cohort.is_some()
            },
            "choose prepare OR pinned goldens+cohort",
        )?;
        if let Some(pin) = &pin {
            check(
                pin.len() == 64
                    && pin
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                "lowerhex SHA256",
            )?;
        }
        let missing = || io::Error::other("need --model --tokens --output");
        Ok(Args {
            model: model.ok_or_else(missing)?,
            tokens: tokens.ok_or_else(missing)?,
            output: output.ok_or_else(missing)?,
            prepare,
            golden,
            pin,
            cohort,
        })
    }
    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Expected {
        input_tokens: usize,
        maximum: usize,
        ids: Vec<u32>,
        output_sha256: String,
        history: usize,
        committed: usize,
        pending: usize,
    }
    impl Expected {
        fn new(input: usize, maximum: usize, ids: Vec<u32>) -> Self {
            Self {
                input_tokens: input,
                maximum,
                output_sha256: ids_sha(&ids),
                ids,
                history: input + maximum,
                committed: input + maximum - 1,
                pending: 1,
            }
        }
        fn validate(&self, input: usize, maximum: usize) -> io::Result<()> {
            check(
                self.input_tokens == input
                    && self.maximum == maximum
                    && self.ids.len() == maximum
                    && self.ids.iter().all(|id| *id < 262144)
                    && self.output_sha256 == ids_sha(&self.ids)
                    && (self.history, self.committed, self.pending)
                        == (input + maximum, input + maximum - 1, 1),
                "golden case identity/counts",
            )
        }
    }
    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Goldens {
        format: String,
        repository: String,
        revision: String,
        model_and_codec_manifest_sha256: String,
        tokens_sha256: String,
        prompt_sha256: String,
        edited_input_sha256: String,
        short_input_sha256: String,
        edited: Expected,
        short: Expected,
        baseline_candidate_parity: bool,
    }
    impl Goldens {
        fn validate(&self, tokens: &[u32]) -> Result<()> {
            check(
                self.format == "toklet-gemma-submission-matrix-goldens-v1"
                    && self.repository == OfficialBf16::REPOSITORY
                    && self.revision == OfficialBf16::REVISION
                    && self.model_and_codec_manifest_sha256 == CODEC
                    && self.tokens_sha256 == TOKENS_SHA
                    && self.prompt_sha256 == ids_sha(&P)
                    && self.edited_input_sha256 == ids_sha(&E)
                    && self.short_input_sha256 == ids_sha(&tokens[..128])
                    && self.baseline_candidate_parity,
                "golden source/inputs/parity",
            )?;
            self.edited.validate(15, 4)?;
            self.short.validate(128, 128)?;
            Ok(())
        }
    }
    fn state(session: &impl InferenceSession) -> Value {
        json!({"history":session.history().len(),"committed":session.evaluated_len(),"pending":session.pending_len(),"history_sha256":ids_sha(session.history())})
    }
    fn core(report: &generation::Report) -> Result<Value> {
        let phases = [
            report.prefill_seconds,
            report.decode_seconds,
            report.sampling_seconds,
            report.total_seconds,
        ];
        check(
            phases.iter().all(|v| v.is_finite() && *v >= 0.)
                && phases[..3].iter().sum::<f64>() <= report.total_seconds + 1e-9
                && (report.emitted_tokens > 0) == report.first_token_seconds.is_some()
                && report
                    .first_token_seconds
                    .is_none_or(|v| v.is_finite() && v >= 0. && v <= report.total_seconds),
            "completed report clocks",
        )?;
        let mut value = serde_json::to_value(report)?;
        value["decode_evaluations"] = json!(report.decode_evaluations);
        Ok(value)
    }
    fn call(
        session: &mut impl InferenceSession,
        request: Request<'_>,
        cancel: &AtomicBool,
        cancel_after: usize,
    ) -> Result<(generation::Report, Vec<u32>, Vec<f64>, f64)> {
        let mut ids = Vec::with_capacity(128);
        let mut events = Vec::with_capacity(128);
        let clock = Instant::now();
        let report = generation::generate(session, request, cancel, |event| {
            if ids.len() >= 128 || event.index != ids.len() {
                return Err(io::Error::other("callback bound/index"));
            }
            let now = clock.elapsed().as_secs_f64();
            ids.push(event.token_id);
            events.push(now);
            if cancel_after > 0 && ids.len() == cancel_after {
                cancel.store(true, Ordering::Relaxed)
            }
            Ok(())
        })?;
        let seconds = clock.elapsed().as_secs_f64();
        Ok((report, ids, events, seconds))
    }
    fn seed(session: &mut impl InferenceSession, case: Case) -> Result<Option<Value>> {
        let (tokens, count, expected): (&[u32], usize, &[u32]) = match case {
            Case::Reset | Case::Edited => (&P, 4, &Q),
            Case::Append => (&P[..14], 0, &[]),
            Case::Cached => (&P, 0, &[]),
            Case::Pending => (&P, 1, &Q[..1]),
            _ => return Ok(None),
        };
        let (report, ids, _, _) = call(
            session,
            Request {
                input: Input::Reset(tokens),
                max_new_tokens: count,
                stop_tokens: &[],
            },
            &AtomicBool::new(false),
            0,
        )?;
        let history = tokens.len() + count;
        let pending = usize::from(count > 0);
        check(
            ids == expected
                && report.reason == StopReason::OutputLimit
                && (
                    report.reused_tokens,
                    report.prefill_tokens,
                    report.decode_evaluations,
                ) == (0, tokens.len(), count.saturating_sub(1))
                && (
                    session.history().len(),
                    session.evaluated_len(),
                    session.pending_len(),
                ) == (history, history - pending, pending)
                && session
                    .history()
                    .iter()
                    .copied()
                    .eq(tokens.iter().chain(ids.iter()).copied()),
            "seed output/cache/state",
        )?;
        Ok(Some(core(&report)?))
    }
    fn generate(
        session: &mut impl InferenceSession,
        case: Case,
        tokens: &[u32],
        expected: Option<&[u32]>,
    ) -> Result<Value> {
        let (input, maximum, reused, prefill, decode, history, pending) = match case {
            Case::Reset => (Input::Reset(&P), 4, 0, 15, 3, 19, 1),
            Case::Append => (Input::Append(&P[14..]), 4, 14, 1, 3, 19, 1),
            Case::Cached => (Input::Prompt(&P), 4, 15, 0, 3, 19, 1),
            Case::Pending => (Input::Continue, 3, 15, 1, 2, 19, 1),
            Case::Edited => (Input::Prompt(&E), 4, 0, 15, 3, 19, 1),
            Case::Short => (Input::Reset(&tokens[..128]), 128, 0, 128, 127, 256, 1),
            Case::Boundary => (Input::Reset(&tokens[..512]), 0, 0, 512, 0, 512, 0),
            _ => return Err(io::Error::other("generation case").into()),
        };
        let (r, ids, events, elapsed) = call(
            session,
            Request {
                input,
                max_new_tokens: maximum,
                stop_tokens: &[],
            },
            &AtomicBool::new(false),
            0,
        )?;
        check(
            r.reason == StopReason::OutputLimit
                && (r.reused_tokens, r.prefill_tokens, r.decode_evaluations)
                    == (reused, prefill, decode)
                && r.sampled_tokens == maximum
                && r.emitted_tokens == maximum
                && ids.len() == maximum
                && (
                    session.history().len(),
                    session.evaluated_len(),
                    session.pending_len(),
                ) == (history, history - pending, pending)
                && (r.history_tokens, r.evaluated_tokens, r.pending_tokens)
                    == (history, history - pending, pending)
                && expected.is_none_or(|wanted| ids == wanted),
            "generation output/state/counts",
        )?;
        let base = match case {
            Case::Edited => &E[..],
            Case::Short => &tokens[..128],
            Case::Boundary => &tokens[..512],
            _ => &P[..],
        };
        let mut full = Vec::with_capacity(512);
        full.extend_from_slice(base);
        if matches!(case, Case::Pending) {
            full.push(818)
        }
        full.extend_from_slice(&ids);
        check(
            session.history() == full.as_slice(),
            "exact generation history",
        )?;
        check(
            events.windows(2).all(|p| p[0] <= p[1])
                && events
                    .iter()
                    .all(|v| v.is_finite() && *v >= 0. && *v <= elapsed),
            "callback clocks",
        )?;
        Ok(
            json!({"timing_kind":"generation","complete_request_seconds":elapsed,"token_event_seconds":events,
            "first_callback_seconds":events.first(),"last_callback_seconds":events.last(),"report":core(&r)?,
            "output_ids":ids,"output_sha256":ids_sha(&ids),"after":state(session)}),
        )
    }
    fn logits(session: &impl InferenceSession, bytes: &mut Vec<u8>) -> Result<(String, u32)> {
        let values = session
            .logits()
            .ok_or_else(|| io::Error::other("missing completed logits"))?;
        check(values.len() == 262144, "full logits")?;
        bytes.clear();
        for value in values {
            check(value.is_finite(), "nonfinite logits")?;
            bytes.extend_from_slice(&value.to_bits().to_le_bytes())
        }
        Ok((
            sha(bytes),
            session
                .greedy_token()
                .ok_or_else(|| io::Error::other("greedy unavailable"))??,
        ))
    }
    fn teacher(session: &mut impl InferenceSession, tokens: &[u32], input: usize) -> Result<Value> {
        check(matches!(input, 128 | 384), "fixed teacher input")?;
        let prefill_evaluations = input / 32;
        let mut prefill = Vec::with_capacity(16);
        let mut decode = Vec::with_capacity(128);
        let mut bytes = Vec::with_capacity(MIB);
        session.append(&tokens[..input])?;
        let wall = Instant::now();
        let mut numerical = 0.;
        while session.pending_len() > 0 {
            check(prefill.len() < prefill_evaluations, "prefill bound")?;
            let clock = Instant::now();
            let rows = session.evaluate_next_chunk()?;
            let elapsed = clock.elapsed().as_secs_f64();
            numerical += elapsed;
            check(
                rows == 32 && session.evaluated_len() == 32 * (prefill.len() + 1),
                "fixed teacher prefill rows/position",
            )?;
            prefill.push(json!({"rows":rows,"position":session.evaluated_len(),"seconds":elapsed}));
        }
        check(
            prefill.len() == prefill_evaluations
                && (
                    session.history().len(),
                    session.evaluated_len(),
                    session.pending_len(),
                ) == (input, input, 0)
                && session.history() == &tokens[..input],
            "exact teacher predecode coverage/state",
        )?;
        let initial = logits(session, &mut bytes)?;
        for (step, &id) in tokens[input..input + 128].iter().enumerate() {
            session.append(&[id])?;
            let clock = Instant::now();
            let rows = session.evaluate_next_chunk()?;
            let elapsed = clock.elapsed().as_secs_f64();
            numerical += elapsed;
            check(rows == 1, "teacher decode rows")?;
            let (hash, greedy) = logits(session, &mut bytes)?;
            decode.push(json!({"step":step,"token_id":id,"position":session.evaluated_len(),"seconds":elapsed,"logits_sha256":hash,"greedy_id":greedy}));
        }
        check(
            (
                session.history().len(),
                session.evaluated_len(),
                session.pending_len(),
            ) == (input + 128, input + 128, 0)
                && session.history() == &tokens[..input + 128],
            "teacher history/state",
        )?;
        Ok(
            json!({"timing_kind":"teacher_forced_evaluations","completed_numerical_seconds":numerical,
            "wall_with_posttimer_hash_seconds":wall.elapsed().as_secs_f64(),"prefill":prefill,
            "prefill_logits_sha256":initial.0,"prefill_greedy_id":initial.1,"decode":decode,"after":state(session),
            "callback_seconds":null,"scope":"Completed evaluation clocks only; append/hash/comparison excluded. No generation TTFT."}),
        )
    }
    struct Runner<'model, 'owner> {
        owner: &'owner OfficialBf16,
        model: &'model Bf16Model<'owner>,
        tokens: &'model [u32],
    }
    impl Runner<'_, '_> {
        fn execute(&self, arm: Arm, case: Case, expected: Option<&[u32]>) -> Result<Value> {
            let setup = Instant::now();
            let options = Options {
                capacity: 512,
                prefill_chunk_size: 32,
                position_offset: 0,
                memory_limit_bytes: OWNER,
            };
            let mut session = match arm {
                Arm::A => self.owner.new_g2_session(self.model, options)?,
                Arm::B => self.owner.new_batched_g2_session(self.model, options)?,
            };
            check(
                session.bf16_profile_status().is_none()
                    && session.history().is_empty()
                    && session.evaluated_len() == 0
                    && session.pending_len() == 0,
                "fresh ordinary session required",
            )?;
            let admitted = session.memory().admitted_bytes;
            let reserved = self.owner.memory().session_reserved_bytes;
            let combined = add(add(admitted, reserved)?, CALLER + HELPER)?;
            let maximum = combined.max(add(
                self.owner.memory().load_admitted_bytes,
                CALLER + HELPER,
            )?);
            check(maximum <= LIMIT, "whole admission")?;
            let seed = seed(&mut session, case)?;
            let before = state(&session);
            let setup_seconds = setup.elapsed().as_secs_f64();
            let mut row = match case {
                Case::Teacher128 => teacher(&mut session, self.tokens, 128)?,
                Case::Teacher384 => teacher(&mut session, self.tokens, 384)?,
                _ => generate(&mut session, case, self.tokens, expected)?,
            };
            drop(session);
            row["event"] = json!("request");
            row["case"] = json!(case.name());
            row["arm"] = json!(arm.name());
            row["passed"] = json!(true);
            row["setup_seconds"] = json!(setup_seconds);
            row["before"] = before;
            row["seed_report"] = json!(seed);
            row["admission"] = json!({"session_admitted_bytes":admitted,"owner_reserved_bytes":reserved,"caller_allowance_bytes":CALLER,
                "helper_allowance_bytes":HELPER,"combined_session_admitted_bytes":combined,"maximum_with_load_admitted_bytes":maximum,"limit_bytes":LIMIT});
            Ok(row)
        }
        fn controls(&self, arm: Arm, journal: &mut Journal) -> Result<()> {
            let options = Options {
                capacity: 512,
                prefill_chunk_size: 32,
                position_offset: 0,
                memory_limit_bytes: OWNER,
            };
            let mut s = match arm {
                Arm::A => self.owner.new_g2_session(self.model, options)?,
                Arm::B => self.owner.new_batched_g2_session(self.model, options)?,
            };
            check(
                add(
                    add(
                        s.memory().admitted_bytes,
                        self.owner.memory().session_reserved_bytes,
                    )?,
                    CALLER + HELPER,
                )? <= LIMIT,
                "control admission",
            )?;
            seed(&mut s, Case::Reset)?;
            let before = state(&s);
            let long = vec![P[0]; 385];
            let rejected = generation::generate(
                &mut s,
                Request {
                    input: Input::Reset(&long),
                    max_new_tokens: 128,
                    stop_tokens: &[],
                },
                &AtomicBool::new(false),
                |_| Err(io::Error::other("rejected request emitted")),
            );
            check(
                matches!(rejected, Err(generation::Error::InvalidRequest(_)))
                    && state(&s) == before,
                "capacity preserves prefix",
            )?;
            journal.append(&json!({"event":"control","case":"capacity_385_128","arm":arm.name(),"passed":true,"before":before,"after":state(&s)}))?;
            let (r, ids, _, _) = call(
                &mut s,
                Request {
                    input: Input::Reset(&P),
                    max_new_tokens: 4,
                    stop_tokens: &[],
                },
                &AtomicBool::new(true),
                0,
            )?;
            check(
                r.reason == StopReason::Cancelled
                    && ids.is_empty()
                    && state(&s) == before
                    && r.prefill_tokens == 0
                    && r.decode_evaluations == 0,
                "pre-cancel prefix",
            )?;
            journal.append(&json!({"event":"control","case":"pre_cancel","arm":arm.name(),"passed":true,"report":core(&r)?,"after":state(&s)}))?;
            let (r, ids, _, _) = call(
                &mut s,
                Request {
                    input: Input::Reset(&P),
                    max_new_tokens: 4,
                    stop_tokens: &[],
                },
                &AtomicBool::new(false),
                1,
            )?;
            check(
                r.reason == StopReason::Cancelled
                    && ids == Q[..1]
                    && (s.history().len(), s.evaluated_len(), s.pending_len()) == (16, 15, 1)
                    && r.decode_evaluations == 0,
                "callback cancel prefix",
            )?;
            journal.append(&json!({"event":"control","case":"callback_cancel","arm":arm.name(),"passed":true,"report":core(&r)?,"after":state(&s)}))?;
            let (r, ids, _, _) = call(
                &mut s,
                Request {
                    input: Input::Reset(&P),
                    max_new_tokens: 4,
                    stop_tokens: &[818],
                },
                &AtomicBool::new(false),
                0,
            )?;
            check(
                r.reason == StopReason::StopToken { token_id: 818 }
                    && ids.is_empty()
                    && r.sampled_tokens == 1
                    && r.emitted_tokens == 0
                    && r.first_token_seconds.is_none()
                    && (s.history().len(), s.evaluated_len(), s.pending_len()) == (16, 15, 1),
                "hidden stop prefix",
            )?;
            journal.append(&json!({"event":"control","case":"hidden_stop","arm":arm.name(),"passed":true,"report":core(&r)?,"after":state(&s)}))?;
            Ok(())
        }
    }
    fn same_pair(a: &Value, b: &Value, case: Case) -> io::Result<()> {
        check(
            a["before"] == b["before"] && a["after"] == b["after"],
            "paired state",
        )?;
        if matches!(case, Case::Teacher128 | Case::Teacher384) {
            check(
                a["prefill_logits_sha256"] == b["prefill_logits_sha256"]
                    && a["prefill_greedy_id"] == b["prefill_greedy_id"],
                "teacher prefill parity",
            )?;
            let left = a["decode"]
                .as_array()
                .ok_or_else(|| io::Error::other("decode array"))?;
            let right = b["decode"]
                .as_array()
                .ok_or_else(|| io::Error::other("decode array"))?;
            check(left.len() == 128 && right.len() == 128, "decode coverage")?;
            for (x, y) in left.iter().zip(right) {
                for key in ["step", "token_id", "position", "logits_sha256", "greedy_id"] {
                    check(x[key] == y[key], "teacher full-logit parity")?;
                }
            }
        } else {
            check(
                a["output_ids"] == b["output_ids"] && a["output_sha256"] == b["output_sha256"],
                "generation parity",
            )?;
        }
        Ok(())
    }
    fn capture(
        runner: &Runner<'_, '_>,
        journal: &mut Journal,
        arm: Arm,
        case: Case,
        expected: Option<&[u32]>,
    ) -> Result<Value> {
        journal.admit()?;
        match runner.execute(arm, case, expected) {
            Ok(row) => Ok(row),
            Err(error) => {
                journal.append(&json!({"event":"request_failed","case":case.name(),"arm":arm.name(),"passed":false,"error":error.to_string()}))?;
                Err(error)
            }
        }
    }
    pub fn run() -> Result<()> {
        let args = args()?;
        let raw = read(&args.tokens, 4100)?;
        check(
            raw.len() == 4100 && sha(&raw) == TOKENS_SHA,
            "fixed1025-token fixture",
        )?;
        let tokens = raw
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        check(
            tokens.len() == 1025 && tokens.iter().all(|t| *t < 262144),
            "fixture ID range",
        )?;
        let golden = if let Some(path) = &args.golden {
            let raw = read(path, 65536)?;
            check(Some(sha(&raw)).as_ref() == args.pin.as_ref(), "golden SHA")?;
            let g: Goldens = serde_json::from_slice(&raw)?;
            g.validate(&tokens)?;
            Some(g)
        } else {
            None
        };
        fs::create_dir(&args.output)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(args.output.join("events.jsonl"))?;
        let mut journal = Journal {
            file,
            buffer: Bytes::new(LINE),
            bytes: 0,
            records: 0,
        };
        let load = Instant::now();
        let owner = OfficialBf16::load(&args.model, 512, OWNER)?;
        let load_seconds = load.elapsed().as_secs_f64();
        check(
            add(owner.memory().load_admitted_bytes, CALLER + HELPER)? <= LIMIT,
            "load admission",
        )?;
        let model = owner.model()?;
        let tokenize = Instant::now();
        let encoded = owner.codec().encode_chat(
            br#"[{"role":"user","content":"Name the capital of France."}]"#,
            gemma4::RenderOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            },
        )?;
        check(encoded == P, "pinned codec prompt")?;
        let tokenization_seconds = tokenize.elapsed().as_secs_f64();
        journal.append(&json!({"event":"ready","format":"toklet-gemma-submission-matrix-v1","prepare_goldens":args.prepare,"cohort":args.cohort,
            "repository":OfficialBf16::REPOSITORY,"revision":OfficialBf16::REVISION,"model_and_codec_manifest_sha256":CODEC,"tokens_sha256":TOKENS_SHA,
            "prompt_ids":P,"prompt_sha256":ids_sha(&P),"edited_input_sha256":ids_sha(&E),"short_input_sha256":ids_sha(&tokens[..128]),"goldens_sha256":args.pin,
            "capacity":512,"chunk":32,"stop_policy":"empty for fixed numerical cases; hidden-stop control818 only","load_seconds":load_seconds,"tokenization_seconds":tokenization_seconds,
            "owner_load_admitted_bytes":owner.memory().load_admitted_bytes,"caller_allowance_bytes":CALLER,"helper_allowance_bytes":HELPER,"limit_bytes":LIMIT,
            "profiling":false,"selected_logit_witness":false,"scope":"Callbacks time completed generation; teacher-forced clocks time completed evaluations with hash/check outside. No UI/tails/adoption; ROOT provenance/physical/power guards required."}))?;
        let runner = Runner {
            owner: &owner,
            model: &model,
            tokens: &tokens,
        };
        let mut requests = 0;
        if args.prepare {
            let mut results = Vec::with_capacity(2);
            for case in [Case::Edited, Case::Short] {
                let first = capture(&runner, &mut journal, Arm::A, case, None)?;
                journal.append(&first)?;
                let expected = first["output_ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|id| {
                        id.as_u64()
                            .and_then(|n| u32::try_from(n).ok())
                            .ok_or_else(|| io::Error::other("reference ID"))
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                let second = capture(&runner, &mut journal, Arm::B, case, Some(&expected))?;
                journal.append(&second)?;
                same_pair(&first, &second, case)?;
                results.push(expected);
                requests += 2;
            }
            let short = results.pop().unwrap();
            let edited = results.pop().unwrap();
            let g = Goldens {
                format: "toklet-gemma-submission-matrix-goldens-v1".into(),
                repository: OfficialBf16::REPOSITORY.into(),
                revision: OfficialBf16::REVISION.into(),
                model_and_codec_manifest_sha256: CODEC.into(),
                tokens_sha256: TOKENS_SHA.into(),
                prompt_sha256: ids_sha(&P),
                edited_input_sha256: ids_sha(&E),
                short_input_sha256: ids_sha(&tokens[..128]),
                edited: Expected::new(15, 4, edited),
                short: Expected::new(128, 128, short),
                baseline_candidate_parity: true,
            };
            g.validate(&tokens)?;
            publish(&args.output, "goldens.json", &g, 65536)?;
        } else {
            let g = golden.as_ref().unwrap();
            for arm in [Arm::A, Arm::B] {
                runner.controls(arm, &mut journal)?;
            }
            for (warmup, count) in [(true, 3), (false, 10)] {
                for trial in 0..count {
                    for case in CASES {
                        let expected = match case {
                            Case::Reset | Case::Append | Case::Cached => Some(&Q[..]),
                            Case::Pending => Some(&Q[1..]),
                            Case::Edited => Some(&g.edited.ids[..]),
                            Case::Short => Some(&g.short.ids[..]),
                            Case::Boundary => Some(&[][..]),
                            _ => None,
                        };
                        let arms = if (trial + args.cohort.unwrap()) % 2 == 1 {
                            [Arm::A, Arm::B]
                        } else {
                            [Arm::B, Arm::A]
                        };
                        let mut pair = Vec::with_capacity(2);
                        for (within, arm) in arms.into_iter().enumerate() {
                            let mut row = capture(&runner, &mut journal, arm, case, expected)?;
                            row["warmup"] = json!(warmup);
                            row["trial"] = json!(trial);
                            row["within_pair"] = json!(within);
                            row["order"] = json!(if arms[0] == Arm::A { "AB" } else { "BA" });
                            journal.append(&row)?;
                            println!(
                                "MATRIX_REQUEST {}",
                                json!({"case":case.name(),"arm":arm.name(),"warmup":warmup,"trial":trial})
                            );
                            io::stdout().flush()?;
                            pair.push(row);
                            requests += 1;
                        }
                        same_pair(&pair[0], &pair[1], case)?;
                    }
                }
            }
            check(requests == 234, "fixed9case/3+10pair completion")?;
        }
        check(
            sha(&read(&args.tokens, 4100)?) == TOKENS_SHA,
            "fixture endpoint changed",
        )?;
        if let Some(path) = &args.golden {
            check(
                Some(sha(&read(path, 65536)?)).as_ref() == args.pin.as_ref(),
                "golden endpoint changed",
            )?;
        }
        let events_sha256 = journal.digest()?;
        publish(
            &args.output,
            "manifest.json",
            &json!({"format":"toklet-gemma-submission-matrix-manifest-v1","passed":true,"complete":true,
            "prepare_goldens":args.prepare,"cohort":args.cohort,"requests":requests,"cases":CASES.map(Case::name),"warmup_pairs":if args.prepare {0}else{3},"measured_pairs":if args.prepare {0}else{10},
            "events":{"file":"events.jsonl","bytes":journal.bytes,"sha256":events_sha256},"goldens_sha256":args.pin,
            "performance_qualified":false,"engine_adoption_qualified":false,"scope":"Fixed matrix only; primary/sustained separate. Unsupported512/1024/1920+128 admission not truncated or waived; ROOT retains full original quality/performance gates."}),
            MIB,
        )?;
        Ok(())
    }
}
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
fn main() {
    if let Err(error) = apple::run() {
        eprintln!("gemma_submission_matrix: {error}");
        std::process::exit(1);
    }
}
#[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("gemma_submission_matrix requires metal on macOS arm64");
    std::process::exit(2);
}
