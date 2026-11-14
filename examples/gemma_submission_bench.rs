//! Explicit ordinary numerical-track benchmark; never enables profiling.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod apple {
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
        session::Options,
    };
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    const LIMIT: usize = 6 * 1024 * 1024 * 1024;
    const CALLER: usize = 16 * 1024 * 1024;
    const HELPER: usize = 32 * 1024 * 1024;
    const OWNER: usize = LIMIT - CALLER - HELPER;
    const WORKLOAD_SHA: &str = "de4ebba10f7bcf6a721c973da65ca8c0a246d04cf7471e159a18e460b4bcc4fd";
    const LINE: usize = 7 * 1024;
    const JOURNAL: usize = 2 * 1024 * 1024;
    static CANCELLED: AtomicBool = AtomicBool::new(false);

    fn require(ok: bool, text: &str) -> io::Result<()> {
        if ok {
            Ok(())
        } else {
            Err(io::Error::other(text))
        }
    }
    fn add(a: usize, b: usize) -> io::Result<usize> {
        a.checked_add(b)
            .ok_or_else(|| io::Error::other("admission overflow"))
    }
    fn token_sha(ids: &[u32]) -> String {
        let mut hash = Sha256::new();
        for id in ids {
            hash.update(id.to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Arm {
        PerTile,
        Batched,
    }
    impl Arm {
        fn name(self) -> &'static str {
            match self {
                Self::PerTile => "per-tile",
                Self::Batched => "batched-single-row",
            }
        }
        fn parse(value: &str) -> io::Result<Self> {
            match value {
                "per-tile" => Ok(Self::PerTile),
                "batched-single-row" => Ok(Self::Batched),
                _ => Err(io::Error::other("invalid sustained arm")),
            }
        }
    }
    #[derive(Clone, Copy)]
    enum Mode {
        Paired(usize),
        Sustained(Arm),
    }
    struct Args {
        model: PathBuf,
        workload: PathBuf,
        output: PathBuf,
        mode: Mode,
    }
    fn args() -> Result<Args> {
        let mut model = None;
        let mut workload = None;
        let mut output = None;
        let mut mode = None;
        let mut input = std::env::args_os().skip(1);
        while let Some(flag) = input.next() {
            let flag = flag
                .to_str()
                .ok_or_else(|| io::Error::other("non-UTF8 option"))?;
            let value = input
                .next()
                .ok_or_else(|| io::Error::other("missing option value"))?;
            match flag {
                "--model" => require(
                    model.replace(PathBuf::from(value)).is_none(),
                    "duplicate model",
                )?,
                "--workload" => require(
                    workload.replace(PathBuf::from(value)).is_none(),
                    "duplicate workload",
                )?,
                "--output" => require(
                    output.replace(PathBuf::from(value)).is_none(),
                    "duplicate output",
                )?,
                "--cohort" => {
                    let cohort = match value.to_str() {
                        Some("1") => 1,
                        Some("2") => 2,
                        _ => return Err(io::Error::other("cohort must be1 or2").into()),
                    };
                    require(
                        mode.replace(Mode::Paired(cohort)).is_none(),
                        "choose one mode",
                    )?;
                }
                "--sustained-arm" => {
                    let arm = Arm::parse(
                        value
                            .to_str()
                            .ok_or_else(|| io::Error::other("non-UTF8 arm"))?,
                    )?;
                    require(
                        mode.replace(Mode::Sustained(arm)).is_none(),
                        "choose one mode",
                    )?;
                }
                _ => return Err(io::Error::other("unknown benchmark option").into()),
            }
        }
        let missing = || {
            io::Error::other("need --model --workload --output and --cohort1|2 or --sustained-arm")
        };
        Ok(Args {
            model: model.ok_or_else(missing)?,
            workload: workload.ok_or_else(missing)?,
            output: output.ok_or_else(missing)?,
            mode: mode.ok_or_else(missing)?,
        })
    }
    fn workload(path: &Path) -> Result<Value> {
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW)
            .open(path)?;
        let metadata = file.metadata()?;
        require(
            metadata.is_file() && metadata.len() <= 16 * 1024,
            "workload regular byte bound",
        )?;
        let mut raw = Vec::with_capacity(16 * 1024 + 1);
        Read::by_ref(&mut file)
            .take(16 * 1024 + 1)
            .read_to_end(&mut raw)?;
        require(
            raw.len() as u64 == metadata.len() && file.metadata()?.len() == metadata.len(),
            "workload EOF",
        )?;
        require(
            format!("{:x}", Sha256::digest(&raw)) == WORKLOAD_SHA,
            "fixed workload SHA differs",
        )?;
        Ok(serde_json::from_slice(&raw)?)
    }
    struct Bytes {
        data: Vec<u8>,
        limit: usize,
    }
    impl Bytes {
        fn new(limit: usize) -> Self {
            Self {
                data: Vec::with_capacity(limit),
                limit,
            }
        }
    }
    impl Write for Bytes {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            require(
                add(self.data.len(), bytes.len())? <= self.limit,
                "JSON byte bound",
            )?;
            self.data.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    struct JournalWriter {
        file: File,
        buffer: Bytes,
        bytes: usize,
    }
    impl JournalWriter {
        fn append(&mut self, value: &Value) -> Result<()> {
            self.buffer.data.clear();
            serde_json::to_writer(&mut self.buffer, value)?;
            require(
                add(self.bytes, add(self.buffer.data.len(), 1)?)? <= JOURNAL,
                "journal byte bound",
            )?;
            self.file.write_all(&self.buffer.data)?;
            self.file.write_all(b"\n")?;
            self.file.flush()?;
            self.bytes += self.buffer.data.len() + 1;
            Ok(())
        }
        fn admit_request(&self) -> io::Result<()> {
            require(
                add(self.bytes, LINE + 1)? <= JOURNAL,
                "journal request admission",
            )
        }
        fn sha(&mut self) -> Result<String> {
            self.file.flush()?;
            require(
                self.file.metadata()?.len() == self.bytes as u64,
                "journal EOF",
            )?;
            self.file.seek(SeekFrom::Start(0))?;
            let mut hash = Sha256::new();
            let mut block = [0; 16 * 1024];
            loop {
                let n = self.file.read(&mut block)?;
                if n == 0 {
                    break;
                }
                hash.update(&block[..n]);
            }
            Ok(format!("{:x}", hash.finalize()))
        }
    }
    struct Runner<'model, 'owner> {
        owner: &'owner OfficialBf16,
        model: &'model Bf16Model<'owner>,
        input: &'model [u32],
        expected: &'model [u32],
    }
    impl Runner<'_, '_> {
        fn measure(&self, arm: Arm) -> Result<Value> {
            require(
                !CANCELLED.load(Ordering::Relaxed),
                "benchmark cancelled before request",
            )?;
            let setup = Instant::now();
            let options = Options {
                capacity: 512,
                prefill_chunk_size: 32,
                position_offset: 0,
                memory_limit_bytes: OWNER,
            };
            let mut ids = Vec::with_capacity(128);
            let mut events = Vec::with_capacity(128);
            let mut session = match arm {
                Arm::PerTile => self.owner.new_g2_session(self.model, options)?,
                Arm::Batched => self.owner.new_batched_g2_session(self.model, options)?,
            };
            require(
                session.history().is_empty()
                    && session.evaluated_len() == 0
                    && session.pending_len() == 0
                    && session.bf16_profile_status().is_none(),
                "fresh ordinary session required",
            )?;
            let admitted = session.memory().admitted_bytes;
            let reserve = self.owner.memory().session_reserved_bytes;
            let combined = add(add(admitted, reserve)?, CALLER + HELPER)?;
            let maximum = combined.max(add(
                self.owner.memory().load_admitted_bytes,
                CALLER + HELPER,
            )?);
            require(maximum <= LIMIT, "whole owner/request budget")?;
            let setup_seconds = setup.elapsed().as_secs_f64();
            let clock = Instant::now();
            let generated = generation::generate(
                &mut session,
                Request {
                    input: Input::Reset(self.input),
                    max_new_tokens: 128,
                    stop_tokens: self.owner.codec().stop_ids(),
                },
                &CANCELLED,
                |event| {
                    if ids.len() >= 128 || event.index != ids.len() {
                        return Err(io::Error::other("bounded token callback"));
                    }
                    let now = clock.elapsed().as_secs_f64();
                    ids.push(event.token_id);
                    events.push(now);
                    Ok(())
                },
            );
            let complete_request_seconds = clock.elapsed().as_secs_f64();
            let check_clock = Instant::now();
            let history = session.history().len();
            let committed = session.evaluated_len();
            let pending = session.pending_len();
            let state_ok = history == 512
                && committed == 511
                && pending == 1
                && session.history().iter().copied().eq(self
                    .input
                    .iter()
                    .chain(ids.iter())
                    .copied());
            let history_sha256 = token_sha(session.history());
            let (report, error, report_ok) = match generated {
                Ok(report) => {
                    let times = [
                        report.prefill_seconds,
                        report.decode_seconds,
                        report.sampling_seconds,
                        report.total_seconds,
                    ];
                    let ok = report.reason == StopReason::OutputLimit
                        && report.reused_tokens == 0
                        && report.prefill_tokens == 384
                        && report.sampled_tokens == 128
                        && report.emitted_tokens == 128
                        && report.decode_evaluations == 127
                        && report.history_tokens == 512
                        && report.evaluated_tokens == 511
                        && report.pending_tokens == 1
                        && times.iter().all(|v| v.is_finite() && *v >= 0.)
                        && report.total_seconds <= complete_request_seconds
                        && report.prefill_seconds + report.decode_seconds + report.sampling_seconds
                            <= report.total_seconds + 1e-9
                        && report
                            .first_token_seconds
                            .is_some_and(|v| v.is_finite() && v >= 0. && v <= report.total_seconds);
                    let mut value = serde_json::to_value(&report)?;
                    value["decode_evaluations"] = json!(report.decode_evaluations);
                    (value, None, ok)
                }
                Err(error) => (Value::Null, Some(error.to_string()), false),
            };
            let output_ok = ids == self.expected;
            let output_sha256 = token_sha(&ids);
            let clocks_ok = events.len() == 128
                && events.windows(2).all(|p| p[0] <= p[1])
                && events
                    .iter()
                    .all(|v| v.is_finite() && *v >= 0. && *v <= complete_request_seconds);
            let first = events.first().copied();
            let last = events.last().copied();
            drop(session);
            Ok(
                json!({"event":"request","arm":arm.name(),"passed":state_ok&&report_ok&&output_ok&&clocks_ok,
                "generation_error":error,"setup_seconds":setup_seconds,"complete_request_seconds":complete_request_seconds,
                "first_callback_seconds":first,"last_callback_seconds":last,"token_event_seconds":events,
                "report":report,"output_ids":ids,"output_sha256":output_sha256,"history_sha256":history_sha256,
                "state":{"history":history,"committed":committed,"pending":pending},"state_matches":state_ok,"output_matches":output_ok,
                "postcheck_seconds":check_clock.elapsed().as_secs_f64(),"session_admitted_bytes":admitted,"owner_reserved_bytes":reserve,
                "combined_session_admitted_bytes":combined,"maximum_with_load_admitted_bytes":maximum,
                "caller_allowance_bytes":CALLER,"helper_allowance_bytes":HELPER,"limit_bytes":LIMIT}),
            )
        }
    }
    struct Position {
        warmup: bool,
        pair: Option<usize>,
        within: Option<usize>,
        order: Option<&'static str>,
    }
    struct Totals {
        requests: usize,
        measured: usize,
        measured_seconds: f64,
        maximum: u64,
    }
    impl Totals {
        fn request(
            &mut self,
            runner: &Runner<'_, '_>,
            journal: &mut JournalWriter,
            arm: Arm,
            position: Position,
        ) -> Result<()> {
            require(self.requests < 256, "total request cap")?;
            journal.admit_request()?;
            let mut row = runner.measure(arm)?;
            row["sequence"] = json!(self.requests);
            row["warmup"] = json!(position.warmup);
            row["pair_index"] = json!(position.pair);
            row["within_pair"] = json!(position.within);
            row["order"] = json!(position.order);
            journal.append(&row)?;
            println!(
                "BENCH_REQUEST {}",
                json!({"sequence":self.requests,"arm":arm.name(),"warmup":position.warmup,
                "complete_request_seconds":row["complete_request_seconds"],"passed":row["passed"]})
            );
            io::stdout().flush()?;
            require(
                row["passed"] == true,
                "request correctness failed; partial journal retained",
            )?;
            self.requests += 1;
            self.maximum = self
                .maximum
                .max(row["maximum_with_load_admitted_bytes"].as_u64().unwrap());
            if !position.warmup {
                self.measured += 1;
                self.measured_seconds += row["complete_request_seconds"].as_f64().unwrap();
            }
            Ok(())
        }
    }
    pub fn run() -> Result<()> {
        let args = args()?;
        let frozen = workload(&args.workload)?;
        let conversation = frozen["conversation_json"]
            .as_str()
            .ok_or_else(|| io::Error::other("pinned conversation"))?;
        let expected = frozen["expected_output"]["tokens"]
            .as_array()
            .ok_or_else(|| io::Error::other("pinned IDs"))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| io::Error::other("pinned ID type"))
            })
            .collect::<io::Result<Vec<_>>>()?;
        require(expected.len() == 128, "pinned output count")?;
        fs::create_dir(&args.output)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(args.output.join("events.jsonl"))?;
        let mut journal = JournalWriter {
            file,
            buffer: Bytes::new(LINE),
            bytes: 0,
        };
        ctrlc::set_handler(|| CANCELLED.store(true, Ordering::Relaxed))?;
        let load = Instant::now();
        let owner = OfficialBf16::load(&args.model, 512, OWNER)?;
        let load_seconds = load.elapsed().as_secs_f64();
        require(
            add(owner.memory().load_admitted_bytes, CALLER + HELPER)? <= LIMIT,
            "load admission",
        )?;
        let setup = Instant::now();
        let model = owner.model()?;
        let model_setup_seconds = setup.elapsed().as_secs_f64();
        let tokenize = Instant::now();
        let input = owner.codec().encode_chat(
            conversation.as_bytes(),
            gemma4::RenderOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            },
        )?;
        let tokenization_seconds = tokenize.elapsed().as_secs_f64();
        require(
            input.len() == 384 && frozen["input_tokens_sha256"] == token_sha(&input),
            "actual input identity",
        )?;
        // Cold export of the authenticated codec output for independent replay.
        let mut input_bytes = [0_u8; 384 * 4];
        for (bytes, id) in input_bytes.chunks_exact_mut(4).zip(&input) {
            bytes.copy_from_slice(&id.to_le_bytes());
        }
        let mut input_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(args.output.join("input_ids.u32le"))?;
        input_file.write_all(&input_bytes)?;
        input_file.flush()?;
        require(input_file.metadata()?.len() == 1536, "input bridge EOF")?;
        let input_bridge = json!({"file":"input_ids.u32le","bytes":1536,
            "sha256":format!("{:x}", Sha256::digest(input_bytes))});
        let mode = match args.mode {
            Mode::Paired(cohort) => json!({"kind":"paired","cohort":cohort}),
            Mode::Sustained(arm) => json!({"kind":"sustained","arm":arm.name()}),
        };
        let header = json!({"event":"ready","format":"toklet-gemma-submission-bench-v1","mode":mode,
            "repository":OfficialBf16::REPOSITORY,"revision":OfficialBf16::REVISION,"workload_sha256":WORKLOAD_SHA,
            "input_sha256":token_sha(&input),"input_ids":input_bridge,"expected_output_sha256":token_sha(&expected),"input_tokens":384,"maximum_output_tokens":128,
            "capacity":512,"chunk":32,"cache_policy":"new_session_reset_each_request","profiling":false,"selected_logit_witness":false,
            "load_seconds":load_seconds,"model_setup_seconds":model_setup_seconds,"tokenization_seconds":tokenization_seconds,
            "owner_load_admitted_bytes":owner.memory().load_admitted_bytes,"owner_reserved_bytes":owner.memory().session_reserved_bytes,
            "caller_allowance_bytes":CALLER,"helper_allowance_bytes":HELPER,"limit_bytes":LIMIT,
            "clock_scope":"Instant around completed generate; callbacks same caller origin. Core Report has its own origin. Setup/tokenization/check/hash/export excluded.",
            "source_build_hardware_guard_provenance":"ROOT external seals required; no UI/tails/energy/first-visible or adoption claim"});
        journal.append(&header)?;
        let runner = Runner {
            owner: &owner,
            model: &model,
            input: &input,
            expected: &expected,
        };
        let mut totals = Totals {
            requests: 0,
            measured: 0,
            measured_seconds: 0.,
            maximum: 0,
        };
        match args.mode {
            Mode::Paired(cohort) => {
                for (warmup, count) in [(true, 3), (false, 10)] {
                    for pair in 0..count {
                        let arms = if (pair + cohort) % 2 == 1 {
                            [Arm::PerTile, Arm::Batched]
                        } else {
                            [Arm::Batched, Arm::PerTile]
                        };
                        let order = if arms[0] == Arm::PerTile { "AB" } else { "BA" };
                        for (within, arm) in arms.into_iter().enumerate() {
                            totals.request(
                                &runner,
                                &mut journal,
                                arm,
                                Position {
                                    warmup,
                                    pair: Some(pair),
                                    within: Some(within),
                                    order: Some(order),
                                },
                            )?;
                        }
                    }
                }
                require(
                    totals.requests == 26 && totals.measured == 20,
                    "fixed pair completion",
                )?;
            }
            Mode::Sustained(arm) => {
                while totals.requests < 256
                    && (totals.requests < 3 || totals.measured_seconds < 600.)
                {
                    let warmup = totals.requests < 3;
                    totals.request(
                        &runner,
                        &mut journal,
                        arm,
                        Position {
                            warmup,
                            pair: None,
                            within: None,
                            order: None,
                        },
                    )?;
                }
                require(
                    totals.measured_seconds >= 600.,
                    "sustained600seconds unavailable at request cap",
                )?;
            }
        }
        require(
            !CANCELLED.load(Ordering::Relaxed),
            "cancelled before final publication",
        )?;
        let _ = workload(&args.workload)?;
        let summary = json!({"format":"toklet-gemma-submission-bench-manifest-v1","passed":true,"complete":true,
            "configuration":header,"requests":totals.requests,"measured_requests":totals.measured,
            "measured_complete_request_seconds":totals.measured_seconds,"maximum_with_load_admitted_bytes":totals.maximum,
            "events":{"file":"events.jsonl","bytes":journal.bytes,"sha256":journal.sha()?},
            "performance_qualified":false,"engine_adoption_qualified":false,
            "scope":"Ordinary numerical requests only; paired/sustained arms never pool. Root validates original performance/quality/continuation/observer gates."});
        let mut bytes = Bytes::new(1024 * 1024);
        serde_json::to_writer_pretty(&mut bytes, &summary)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(args.output.join("manifest.json"))?;
        file.write_all(&bytes.data)?;
        file.flush()?;
        println!(
            "BENCH_COMPLETE {}",
            json!({"requests":totals.requests,"manifest":"manifest.json"})
        );
        Ok(())
    }
}
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() {
    if let Err(error) = apple::run() {
        eprintln!("gemma_submission_bench: {error}");
        std::process::exit(1);
    }
}
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("gemma_submission_bench requires macOS arm64 and metal");
    std::process::exit(2);
}
