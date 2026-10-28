//! Explicit paired Reference scheduling benchmark; qualification is external.
//! Requires the separately reviewed ReferenceSchedule diagnostic constructor.
#[cfg(any(
    test,
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
mod protocol {
    use serde::{Deserialize, Serialize};
    use std::path::PathBuf;
    pub const WARMUPS: usize = 3;
    pub const SAMPLES: usize = 10;
    pub const TRIALS: usize = WARMUPS + SAMPLES;
    pub const VOCAB: usize = 151936;
    pub const LIMIT: usize = 6 * 1024 * 1024 * 1024;
    pub const APP: usize = 256 * 1024 * 1024;
    pub const INPUT_SHA: &str = "45f734c5fca32780b251eb14bf777bd4c512ba36a9a655c256d064aeea8b5c0b";

    #[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
    #[serde(rename_all = "snake_case")]
    pub enum Mode {
        DeepPair,
        Short128,
        Short480,
    }
    impl Mode {
        pub fn capacity(self) -> usize {
            if self == Self::DeepPair { 4096 } else { 512 }
        }
        pub fn prefix(self) -> usize {
            match self {
                Self::DeepPair => 4064,
                Self::Short128 => 128,
                Self::Short480 => 480,
            }
        }
        pub fn tokens(self) -> usize {
            match self {
                Self::DeepPair => {
                    2 * (4064 + TRIALS * 32 + (TRIALS - 1) + 1 + TRIALS + (TRIALS - 1))
                }
                _ => 2 * TRIALS * (self.prefix() + 32),
            }
        }
        pub fn calls_per_request(self) -> usize {
            self.prefix() / 32 + 32
        }
    }
    #[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
    #[serde(rename_all = "snake_case")]
    pub enum Order {
        SerialFirst,
        ParallelFirst,
    }
    impl Order {
        pub fn labels(self) -> [&'static str; 2] {
            match self {
                Self::SerialFirst => ["Serial", "ParallelOutputsExperimental"],
                Self::ParallelFirst => ["ParallelOutputsExperimental", "Serial"],
            }
        }
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct Binding {
        pub path: PathBuf,
        pub sha256: String,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct Config {
        pub format: String,
        pub mode: Mode,
        pub order: Order,
        pub warmups: usize,
        pub samples: usize,
        pub input: Binding,
    }
    pub fn digest(s: &str) -> bool {
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }
    impl Config {
        pub fn validate(&self) -> Result<(), &'static str> {
            if self.format != "toklet-qwen3-reference-schedule-bench-v1"
                || self.warmups != WARMUPS
                || self.samples != SAMPLES
                || !self.input.path.is_absolute()
                || self.input.sha256 != INPUT_SHA
            {
                return Err("fixed paired benchmark protocol differs");
            }
            Ok(())
        }
    }
    pub fn second_budget(first: usize, shared: usize) -> Option<usize> {
        let available = LIMIT.checked_sub(APP)?;
        if shared > first || first > available {
            return None;
        }
        available.checked_sub(first)?.checked_add(shared)
    }
    pub fn joint(first: usize, second: usize, shared: usize) -> Option<usize> {
        if shared > first || shared > second {
            return None;
        }
        let bytes = first
            .checked_add(second)?
            .checked_sub(shared)?
            .checked_add(APP)?;
        (bytes <= LIMIT).then_some(bytes)
    }
    pub fn shortened_state(prefix: usize) -> (usize, usize) {
        (prefix - 1, 1)
    }
    pub fn snapshot_bytes(calls: usize, capacity: usize) -> Option<usize> {
        calls
            .checked_mul(2)?
            .checked_mul(VOCAB.checked_add(capacity)?)?
            .checked_mul(4)
    }
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use super::protocol::*;
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::{
        fs::{File, OpenOptions},
        io::{self, Read, Seek, SeekFrom, Write},
        os::unix::fs::{MetadataExt, OpenOptionsExt},
        path::Path,
        time::Instant,
    };
    use toklet::{
        metal::{self, ReadbackMode, ReferenceSchedule},
        session, weights,
    };
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    type Identity = (u64, u64, u64, i64, i64, i64, i64);
    fn require(ok: bool, message: &'static str) -> Result<()> {
        if ok { Ok(()) } else { Err(message.into()) }
    }
    fn emit(out: &mut impl Write, value: Value) -> Result<()> {
        serde_json::to_writer(&mut *out, &value)?;
        out.write_all(b"\n")?;
        out.flush()?;
        Ok(())
    }
    fn identity(file: &File) -> io::Result<Identity> {
        let m = file.metadata()?;
        Ok((
            m.dev(),
            m.ino(),
            m.len(),
            m.mtime(),
            m.mtime_nsec(),
            m.ctime(),
            m.ctime_nsec(),
        ))
    }
    fn hash(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }
    fn bits_hash(bits: &[u32]) -> String {
        let mut digest = Sha256::new();
        for value in bits {
            digest.update(value.to_le_bytes());
        }
        format!("{:x}", digest.finalize())
    }
    fn open(path: &Path) -> Result<File> {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path)?;
        require(file.metadata()?.is_file(), "regular input required")?;
        Ok(file)
    }
    struct Owner {
        binding: Binding,
        file: File,
        identity: Identity,
    }
    impl Owner {
        fn read(binding: Binding, cap: usize) -> Result<(Self, Vec<u8>)> {
            let mut file = open(&binding.path)?;
            let before = identity(&file)?;
            require(
                before.2 > 0 && before.2 <= cap as u64,
                "bounded input required",
            )?;
            let mut body = vec![0; before.2 as usize];
            file.read_exact(&mut body)?;
            require(
                file.read(&mut [0])? == 0
                    && identity(&file)? == before
                    && hash(&body) == binding.sha256,
                "input identity/hash differs",
            )?;
            Ok((
                Self {
                    binding,
                    file,
                    identity: before,
                },
                body,
            ))
        }
        fn endpoint(&mut self) -> Result<()> {
            self.file.seek(SeekFrom::Start(0))?;
            let mut body = vec![0; self.identity.2 as usize];
            self.file.read_exact(&mut body)?;
            require(
                self.file.read(&mut [0])? == 0
                    && identity(&self.file)? == self.identity
                    && identity(&open(&self.binding.path)?)? == self.identity
                    && hash(&body) == self.binding.sha256,
                "input endpoint differs",
            )
        }
    }

    // No hashes or JSON are computed inside either elapsed engine calls or requests.
    // Full-logit witness vectors are reserved before request clocks start.
    struct Record {
        depth: usize,
        pending_before: usize,
        rows: Option<usize>,
        error: Option<String>,
        seconds: f64,
        timing: Option<metal::Timing>,
        witness_seconds: f64,
        evaluated: usize,
        pending: usize,
        history: Vec<u32>,
        bits: Vec<u32>,
    }
    impl Record {
        fn new(capacity: usize) -> Self {
            Self {
                depth: 0,
                pending_before: 0,
                rows: None,
                error: None,
                seconds: 0.,
                timing: None,
                witness_seconds: 0.,
                evaluated: 0,
                pending: 0,
                history: Vec::with_capacity(capacity),
                bits: Vec::with_capacity(VOCAB),
            }
        }
        fn call(&mut self, s: &mut metal::Session<'_>, total: &mut usize) -> Result<()> {
            self.depth = s.evaluated_len();
            self.pending_before = s.pending_len();
            self.history.clear();
            self.bits.clear();
            self.error = None;
            let clock = Instant::now();
            let result = s.evaluate_next_chunk();
            self.seconds = clock.elapsed().as_secs_f64();
            self.rows = result.as_ref().ok().copied();
            self.timing = s.timing();
            self.error = result.as_ref().err().map(ToString::to_string);
            let witness = Instant::now();
            self.evaluated = s.evaluated_len();
            self.pending = s.pending_len();
            self.history.extend_from_slice(s.history());
            if let Some(row) = s.logits() {
                self.bits.extend(row.iter().map(|v| v.to_bits()));
            }
            self.witness_seconds = witness.elapsed().as_secs_f64();
            let n = result?;
            *total = total.checked_add(n).ok_or("token accounting overflow")?;
            require(
                n == self.pending_before
                    && (1..=32).contains(&n)
                    && self.evaluated == self.depth + n
                    && self.pending == 0
                    && self.bits.len() == VOCAB
                    && self.timing.is_some(),
                "completed call shape/state differs",
            )?;
            require(
                self.bits.iter().all(|v| f32::from_bits(*v).is_finite()),
                "nonfinite completed logits",
            )
        }
        fn value(&self) -> Value {
            json!({"pre_call_depth":self.depth,"pre_call_pending":self.pending_before,"rows":self.rows,
                "error":self.error,"completed_call_seconds":self.seconds,"timing":self.timing,
                "witness_copy_seconds":self.witness_seconds,"evaluated":self.evaluated,"pending":self.pending,
                "history":self.history,"logits_sha256":bits_hash(&self.bits)})
        }
        fn exact(&self, other: &Self) -> bool {
            self.depth == other.depth
                && self.pending_before == other.pending_before
                && self.rows == other.rows
                && self.error == other.error
                && self.evaluated == other.evaluated
                && self.pending == other.pending
                && self.history == other.history
                && self.bits == other.bits
        }
    }
    fn witness_failure(
        out: &mut impl Write,
        label: &str,
        a: &Record,
        b: Option<&Record>,
    ) -> Result<()> {
        emit(
            out,
            json!({"event":"failure_witness","label":label,"first":a.value(),"first_fp32_bits":a.bits,
            "second":b.map(Record::value),"second_fp32_bits":b.map(|r| &r.bits)}),
        )
    }

    struct Pair<'m> {
        sessions: [metal::Session<'m>; 2],
        labels: [&'static str; 2],
        totals: [usize; 2],
    }
    fn sum_memory(m: session::Memory) -> Option<usize> {
        [
            m.weights_bytes,
            m.kv_bytes,
            m.scratch_bytes,
            m.history_bytes,
            m.logits_bytes,
            m.metadata_bytes,
            m.metadata_allowance_bytes,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
    impl<'m> Pair<'m> {
        fn new(
            model: &'m metal::Model,
            mode: Mode,
            order: Order,
            out: &mut impl Write,
        ) -> Result<Self> {
            let labels = order.labels();
            let schedules = match order {
                Order::SerialFirst => [
                    ReferenceSchedule::Serial,
                    ReferenceSchedule::ParallelOutputsExperimental,
                ],
                Order::ParallelFirst => [
                    ReferenceSchedule::ParallelOutputsExperimental,
                    ReferenceSchedule::Serial,
                ],
            };
            let options = session::Options {
                capacity: mode.capacity(),
                prefill_chunk_size: 32,
                position_offset: 0,
                memory_limit_bytes: LIMIT - APP,
            };
            let clock = Instant::now();
            let first = metal::Session::with_qwen3_reference_schedule_diagnostic(
                model,
                options,
                ReadbackMode::FullLogits,
                schedules[0],
            )?;
            let first_seconds = clock.elapsed().as_secs_f64();
            let a = first.memory();
            require(
                sum_memory(a) == Some(a.admitted_bytes),
                "first memory ledger differs",
            )?;
            let budget = second_budget(a.admitted_bytes, a.weights_bytes)
                .ok_or("second-session joint admission failed")?;
            emit(
                out,
                json!({"event":"second_admission","first_memory":a,"second_budget":budget,
                "shared_weight_bytes":a.weights_bytes,"application_reserve":APP,"first_schedule":labels[0]}),
            )?;
            let clock = Instant::now();
            let second = metal::Session::with_qwen3_reference_schedule_diagnostic(
                model,
                session::Options {
                    memory_limit_bytes: budget,
                    ..options
                },
                ReadbackMode::FullLogits,
                schedules[1],
            )?;
            let second_seconds = clock.elapsed().as_secs_f64();
            let b = second.memory();
            let admitted = joint(a.admitted_bytes, b.admitted_bytes, a.weights_bytes)
                .ok_or("joint plan exceeds6GiB")?;
            emit(
                out,
                json!({"event":"pair_ready","mode":mode,"order":order,"first_memory":a,"second_memory":b,
                "joint_admitted_bytes":admitted,"application_reserve":APP,"first_creation_seconds":first_seconds,
                "second_creation_seconds":second_seconds,"diagnostics":false,"readback":"FullLogits",
                "host_snapshot_bytes":snapshot_bytes(if mode==Mode::DeepPair {1} else {mode.calls_per_request()},mode.capacity())}),
            )?;
            require(
                a == b && sum_memory(b) == Some(b.admitted_bytes) && b.admitted_bytes <= budget,
                "shared model or numerical schedule ledger differs",
            )?;
            require(
                first.reference_schedule() == schedules[0]
                    && second.reference_schedule() == schedules[1],
                "explicit schedule selection differs",
            )?;
            Ok(Self {
                sessions: [first, second],
                labels,
                totals: [0, 0],
            })
        }
        fn paired_call(
            &mut self,
            out: &mut impl Write,
            label: &str,
            trial: Option<usize>,
        ) -> Result<()> {
            let mut records = [
                Record::new(self.sessions[0].capacity()),
                Record::new(self.sessions[1].capacity()),
            ];
            for (i, record) in records.iter_mut().enumerate() {
                let result = record.call(&mut self.sessions[i], &mut self.totals[i]);
                let saved = emit(
                    out,
                    json!({"event":"call","label":label,"trial":trial,"warmup":trial.map(|t|t<WARMUPS),
                    "schedule":self.labels[i],"record":record.value()}),
                );
                if let Err(error) = result {
                    if let Err(secondary) = saved {
                        eprintln!("secondary call journal error: {secondary}");
                    }
                    if let Err(secondary) = witness_failure(out, label, record, None) {
                        eprintln!("secondary witness error: {secondary}");
                    }
                    return Err(error);
                }
                saved?;
            }
            if !records[0].exact(&records[1]) {
                witness_failure(out, label, &records[0], Some(&records[1]))?;
                return Err("paired completed call bits/state differ".into());
            }
            Ok(())
        }
        fn restore(
            &mut self,
            input: &[u32],
            prefix: usize,
            label: &str,
            out: &mut impl Write,
        ) -> Result<()> {
            for (i, s) in self.sessions.iter_mut().enumerate() {
                let before = (s.history().len(), s.evaluated_len(), s.pending_len());
                let clock = Instant::now();
                let result = s.set_tokens(&input[..prefix]);
                let seconds = clock.elapsed().as_secs_f64();
                emit(
                    out,
                    json!({"event":"restoration_set_tokens","label":label,"schedule":self.labels[i],
                    "prefix":prefix,"before":before,"after":[s.history().len(),s.evaluated_len(),s.pending_len()],
                    "set_tokens_seconds":seconds,"error":result.as_ref().err().map(ToString::to_string)}),
                )?;
                result?;
                require(
                    s.history() == &input[..prefix]
                        && (s.evaluated_len(), s.pending_len()) == shortened_state(prefix),
                    "shortening must expose exactly one replay token",
                )?;
            }
            self.paired_call(out, label, None)?;
            for s in &self.sessions {
                require(
                    s.evaluated_len() == prefix && s.pending_len() == 0,
                    "restored prefix state differs",
                )?;
            }
            Ok(())
        }
        fn deep(&mut self, input: &[u32], out: &mut impl Write) -> Result<()> {
            let mut final_bits: Option<Vec<u32>> = None;
            for start in (0..4064).step_by(32) {
                for s in &mut self.sessions {
                    s.append(&input[start..start + 32])?;
                }
                self.paired_call(out, "initial_prefix", None)?;
            }
            for trial in 0..TRIALS {
                if trial > 0 {
                    self.restore(input, 4064, "primary_restoration", out)?;
                }
                for s in &mut self.sessions {
                    require(
                        s.history() == &input[..4064]
                            && s.evaluated_len() == 4064
                            && s.pending_len() == 0,
                        "primary prefix differs",
                    )?;
                    s.append(&input[4064..4096])?;
                }
                self.paired_call(out, "primary_deep32", Some(trial))?;
                self.repeat_final(out, "primary_deep32", trial, &mut final_bits)?;
            }
            self.restore(input, 4095, "decode_initial_restoration", out)?;
            for trial in 0..TRIALS {
                if trial > 0 {
                    self.restore(input, 4095, "decode_restoration", out)?;
                }
                for s in &mut self.sessions {
                    require(
                        s.history() == &input[..4095]
                            && s.evaluated_len() == 4095
                            && s.pending_len() == 0,
                        "decode prefix differs",
                    )?;
                    s.append(&input[4095..4096])?;
                }
                self.paired_call(out, "deep1", Some(trial))?;
                self.repeat_final(out, "deep1", trial, &mut final_bits)?;
            }
            for s in &self.sessions {
                require(
                    s.history() == input && s.evaluated_len() == 4096 && s.pending_len() == 0,
                    "final deep history differs",
                )?;
            }
            Ok(())
        }
        fn repeat_final(
            &self,
            out: &mut impl Write,
            label: &str,
            trial: usize,
            expected: &mut Option<Vec<u32>>,
        ) -> Result<()> {
            let actual: Vec<u32> = self.sessions[0]
                .logits()
                .ok_or("final distribution absent")?
                .iter()
                .map(|v| v.to_bits())
                .collect();
            if let Some(previous) = expected {
                let same = *previous == actual;
                emit(
                    out,
                    json!({"event":"repeat_final","label":label,"trial":trial,"exact":same,
                    "expected_sha256":bits_hash(previous),"actual_sha256":bits_hash(&actual)}),
                )?;
                if !same {
                    emit(
                        out,
                        json!({"event":"failure_witness","label":label,"trial":trial,
                        "expected_fp32_bits":previous,"actual_fp32_bits":actual}),
                    )?;
                    return Err("repeated deep distribution differs".into());
                }
            } else {
                *expected = Some(actual);
            }
            Ok(())
        }
    }

    struct Request {
        calls: Vec<Record>,
        used: usize,
        ids: Vec<u32>,
        visible_seconds: [f64; 32],
        sampling_seconds: [f64; 32],
        pending_history: Vec<u32>,
        pending_evaluated: usize,
        pending_count: usize,
        prefix: usize,
        reset_seconds: f64,
        prefill_seconds: f64,
        wall_seconds: f64,
        error: Option<String>,
    }
    impl Request {
        fn new(mode: Mode) -> Self {
            Self {
                calls: (0..mode.calls_per_request())
                    .map(|_| Record::new(512))
                    .collect(),
                used: 0,
                ids: Vec::with_capacity(32),
                visible_seconds: [0.; 32],
                sampling_seconds: [0.; 32],
                pending_history: Vec::with_capacity(512),
                pending_evaluated: 0,
                pending_count: 0,
                prefix: mode.prefix(),
                reset_seconds: 0.,
                prefill_seconds: 0.,
                wall_seconds: 0.,
                error: None,
            }
        }
        fn call(&mut self, s: &mut metal::Session<'_>, total: &mut usize) -> Result<()> {
            let record = self
                .calls
                .get_mut(self.used)
                .ok_or("too many completed request calls")?;
            self.used += 1;
            record.call(s, total)
        }
        fn run(
            &mut self,
            s: &mut metal::Session<'_>,
            input: &[u32],
            prefix: usize,
            total: &mut usize,
        ) -> Result<()> {
            self.used = 0;
            self.ids.clear();
            self.pending_history.clear();
            self.error = None;
            self.wall_seconds = 0.;
            self.prefill_seconds = 0.;
            self.pending_evaluated = 0;
            self.pending_count = 0;
            self.visible_seconds.fill(0.);
            self.sampling_seconds.fill(0.);
            let reset = Instant::now();
            s.reset();
            self.reset_seconds = reset.elapsed().as_secs_f64();
            require(
                s.history().is_empty()
                    && s.evaluated_len() == 0
                    && s.pending_len() == 0
                    && s.logits().is_none(),
                "request reset differs",
            )?;
            // All witness buffers already exist. Raw request wall includes their copy work.
            let request = Instant::now();
            let result = (|| -> Result<()> {
                for start in (0..prefix).step_by(32) {
                    s.append(&input[start..start + 32])?;
                    self.call(s, total)?;
                }
                self.prefill_seconds = request.elapsed().as_secs_f64();
                for index in 0..32 {
                    let sample = Instant::now();
                    let token = s.greedy_token().ok_or("missing greedy distribution")??;
                    self.sampling_seconds[index] = sample.elapsed().as_secs_f64();
                    self.ids.push(token);
                    self.visible_seconds[index] = request.elapsed().as_secs_f64();
                    // Explicit empty stop list: EOS, if selected, is visible in this raw request.
                    s.append(&[token])?;
                    if index < 31 {
                        self.call(s, total)?;
                    }
                }
                self.pending_history.extend_from_slice(s.history());
                self.pending_evaluated = s.evaluated_len();
                self.pending_count = s.pending_len();
                require(
                    self.pending_history.len() == prefix + 32
                        && self.pending_evaluated == prefix + 31
                        && self.pending_count == 1,
                    "final sample must be pending before flush",
                )?;
                self.call(s, total)?;
                require(
                    self.used == self.calls.len()
                        && s.history() == self.pending_history.as_slice()
                        && s.evaluated_len() == prefix + 32
                        && s.pending_len() == 0,
                    "request completion differs",
                )
            })();
            self.wall_seconds = request.elapsed().as_secs_f64();
            self.error = result.as_ref().err().map(ToString::to_string);
            result
        }
        fn value(&self) -> Value {
            let dt = self.visible_seconds[31] - self.visible_seconds[0];
            json!({"raw_request_wall_seconds":self.wall_seconds,"retention_metric":"raw_request_wall_seconds",
                "setup_reset_seconds":self.reset_seconds,"raw_numerical_prefill_seconds":self.prefill_seconds,
                "witness_instrumented":true,"detokenization_or_ui":false,
                "witness_copy_seconds":self.calls[..self.used].iter().map(|r|r.witness_seconds).sum::<f64>(),
                "first_visible_seconds":self.visible_seconds[0],"visible_seconds":self.visible_seconds,
                "sampling_seconds":self.sampling_seconds,"visible_decode_tokens_per_second":if self.ids.len()==32 && dt>0. {Some(31./dt)} else {None},
                "visible_ids":self.ids,"stop_ids":[],"stop_reason":if self.ids.len()==32 && self.error.is_none(){"output_limit"}else{"incomplete"},
                "pending_history":self.pending_history,"pending_evaluated":self.pending_evaluated,"pending_count":self.pending_count,
                "error":self.error,"calls":self.calls[..self.used].iter().enumerate().map(|(i,r)|json!({
                    "phase":if i<self.prefix/32{"prefill"}else if i==self.calls.len()-1{"final_flush"}else{"decode"},
                    "record":r.value()})).collect::<Vec<_>>()})
        }
        fn exact(&self, other: &Self) -> bool {
            self.used == other.used
                && self.ids == other.ids
                && self.pending_history == other.pending_history
                && self.pending_evaluated == other.pending_evaluated
                && self.pending_count == other.pending_count
                && self.error == other.error
                && self.calls[..self.used]
                    .iter()
                    .zip(&other.calls[..other.used])
                    .all(|(a, b)| a.exact(b))
        }
    }
    fn short(pair: &mut Pair<'_>, mode: Mode, input: &[u32], out: &mut impl Write) -> Result<()> {
        let mut requests = [Request::new(mode), Request::new(mode)];
        let mut initial_identity: Option<Value> = None;
        for trial in 0..TRIALS {
            for (i, request) in requests.iter_mut().enumerate() {
                let result = request.run(
                    &mut pair.sessions[i],
                    input,
                    mode.prefix(),
                    &mut pair.totals[i],
                );
                let saved = emit(
                    out,
                    json!({"event":"request","mode":mode,"trial":trial,"warmup":trial<WARMUPS,
                    "schedule":pair.labels[i],"request":request.value()}),
                );
                if let Err(error) = result {
                    if let Err(secondary) = saved {
                        eprintln!("secondary request journal error: {secondary}");
                    }
                    if let Some(record) = request
                        .used
                        .checked_sub(1)
                        .and_then(|i| request.calls.get(i))
                        && let Err(secondary) = witness_failure(out, "request_error", record, None)
                    {
                        eprintln!("secondary witness error: {secondary}");
                    }
                    return Err(error);
                }
                saved?;
            }
            if !requests[0].exact(&requests[1]) {
                for (a, b) in requests[0].calls[..requests[0].used]
                    .iter()
                    .zip(&requests[1].calls[..requests[1].used])
                {
                    if !a.exact(b) {
                        witness_failure(out, "request_pair_mismatch", a, Some(b))?;
                        break;
                    }
                }
                return Err("paired request IDs/distributions/state differ".into());
            }
            let current = json!({"ids":requests[0].ids,"pending_history":requests[0].pending_history,
                "row_hashes":requests[0].calls[..requests[0].used].iter().map(|r|bits_hash(&r.bits)).collect::<Vec<_>>()});
            if let Some(first) = &initial_identity {
                emit(
                    out,
                    json!({"event":"repeat_identity","trial":trial,"matches_first":*first==current}),
                )?;
                if *first != current {
                    let request = &requests[0];
                    let old_hashes = first["row_hashes"]
                        .as_array()
                        .ok_or("first repeat hashes absent")?;
                    let new_hashes = current["row_hashes"]
                        .as_array()
                        .ok_or("current repeat hashes absent")?;
                    let old_history = first["pending_history"]
                        .as_array()
                        .ok_or("first repeat history absent")?;
                    let first_call = (0..old_hashes.len().max(new_hashes.len())).find(|&i| {
                        if old_hashes.get(i) != new_hashes.get(i) {
                            return true;
                        }
                        let Some(record) = request.calls.get(i).filter(|_| i < request.used) else {
                            return true;
                        };
                        let expected_len = if i < mode.prefix() / 32 {
                            (i + 1) * 32
                        } else {
                            mode.prefix() + i - mode.prefix() / 32 + 1
                        };
                        record.evaluated != expected_len
                            || record.pending != 0
                            || record.history.len() != expected_len
                            || record.history.iter().enumerate().any(|(p, token)| {
                                old_history.get(p).and_then(Value::as_u64)
                                    != Some(u64::from(*token))
                            })
                    });
                    let record = first_call
                        .and_then(|i| request.calls.get(i))
                        .filter(|_| first_call.is_some_and(|i| i < request.used));
                    emit(
                        out,
                        json!({"event":"repeat_identity_failure_witness","first_trial":0,"trial":trial,
                        "schedule":pair.labels[0],"first_differing_call_index":first_call,
                        "expected_logit_sha256":first_call.and_then(|i|old_hashes.get(i)),
                        "current_logit_sha256":first_call.and_then(|i|new_hashes.get(i)),
                        "current_fp32_bits":record.map(|r|&r.bits),"current_call_state":record.map(Record::value),
                        "expected_raw_row_retained":false,
                        "expected_row_evidence":"preserved first-trial hash in the earlier authenticated request journal; historical raw row bytes are not retained",
                        "expected_visible_ids":first["ids"],"current_visible_ids":current["ids"],
                        "expected_pending_history":first["pending_history"],"current_pending_history":current["pending_history"],
                        "current_pending_evaluated":request.pending_evaluated,"current_pending_count":request.pending_count}),
                    )?;
                }
                require(*first == current, "request repeat identity differs")?;
            } else {
                initial_identity = Some(current);
            }
        }
        Ok(())
    }
    pub fn run() -> Result<()> {
        let mut args = std::env::args_os().skip(1);
        let model_path = args.next().ok_or("missing model path")?;
        let config_path = args.next().ok_or("missing pinned config path")?;
        let config_sha = args
            .next()
            .ok_or("missing config SHA")?
            .into_string()
            .map_err(|_| "config digest encoding")?;
        require(
            args.next().is_none() && digest(&config_sha),
            "exact model/config/SHA arguments required",
        )?;
        let (mut config_owner, body) = Owner::read(
            Binding {
                path: config_path.into(),
                sha256: config_sha.clone(),
            },
            16384,
        )?;
        let config: Config = serde_json::from_slice(&body)?;
        config.validate()?;
        let (mut input_owner, body) = Owner::read(config.input, 16384)?;
        require(body.len() == 16384, "complete fixed A4096 required")?;
        let input: Vec<u32> = body
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        require(
            input.len() == 4096 && input.iter().all(|t| (*t as usize) < VOCAB),
            "input count/vocabulary differs",
        )?;
        let mut out = io::stdout().lock();
        let primary = (|| -> Result<()> {
            require(
                !cfg!(debug_assertions),
                "benchmark requires optimized release build",
            )?;
            emit(
                &mut out,
                json!({"event":"start","format":"toklet-qwen3-reference-schedule-bench-events-v1",
                "status":"experimental_unqualified","mode":config.mode,"order":config.order,"warmups":WARMUPS,"samples":SAMPLES,
                "config_sha256":config_sha,"input_sha256":INPUT_SHA,"expected_committed_tokens":config.mode.tokens(),
                "external_model_source_runtime_power_thermal_memory_endpoints_required":true}),
            )?;
            let clock = Instant::now();
            let model = metal::Model::load_mlx_qwen3_context_diagnostic(
                Path::new(&model_path),
                weights::LoadOptions {
                    context_limit: config.mode.capacity(),
                    memory_limit_bytes: LIMIT - APP,
                },
            )?;
            emit(
                &mut out,
                json!({"event":"model_ready","device":model.device_name(),"load_compile_seconds":clock.elapsed().as_secs_f64()}),
            )?;
            let mut pair = Pair::new(&model, config.mode, config.order, &mut out)?;
            match config.mode {
                Mode::DeepPair => pair.deep(&input, &mut out)?,
                _ => short(&mut pair, config.mode, &input, &mut out)?,
            }
            emit(
                &mut out,
                json!({"event":"completed_work","mode":config.mode,"schedule_order":pair.labels,
                "committed_by_schedule":pair.totals,"expected_total":config.mode.tokens()}),
            )?;
            require(
                pair.totals[0] == config.mode.tokens() / 2
                    && pair.totals[1] == config.mode.tokens() / 2,
                "fixed workload accounting differs",
            )?;
            Ok(())
        })();
        let mut endpoint_errors = Vec::new();
        for owner in [&mut config_owner, &mut input_owner] {
            if let Err(error) = owner.endpoint() {
                endpoint_errors.push(error.to_string());
            }
        }
        let saved = emit(
            &mut out,
            json!({"event":"done","complete":primary.is_ok() && endpoint_errors.is_empty(),
            "primary_error":primary.as_ref().err().map(ToString::to_string),"endpoint_errors":endpoint_errors,
            "performance_qualified":false,"external_process_memory_and_cleanup_required":true}),
        );
        if let Err(error) = primary {
            if let Err(secondary) = saved {
                eprintln!("secondary final journal error: {secondary}");
            }
            return Err(error);
        }
        saved?;
        require(endpoint_errors.is_empty(), "input endpoint failure")
    }
}

fn main() {
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    if let Err(error) = apple::run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
    #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
    {
        eprintln!(
            "This finite diagnostic benchmark requires Apple Silicon macOS and --features metal."
        );
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::protocol::*;
    #[test]
    fn fixed_workload_includes_every_restoration_and_flush() {
        assert_eq!(TRIALS, 13);
        assert_eq!(2 * (4064 + 13 * 32 + 12), 8984);
        assert_eq!(2 * (1 + 13 + 12), 52);
        assert_eq!(Mode::DeepPair.tokens(), 9036);
        assert_eq!(Mode::Short128.tokens(), 4160);
        assert_eq!(Mode::Short480.tokens(), 13312);
        assert_eq!(
            [Mode::DeepPair, Mode::Short128, Mode::Short480]
                .iter()
                .map(|m| m.tokens())
                .sum::<usize>(),
            26508
        );
        assert_eq!(shortened_state(4064), (4063, 1));
        assert_eq!(shortened_state(4095), (4094, 1));
        assert_eq!(Mode::Short128.calls_per_request(), 36);
        assert_eq!(Mode::Short480.calls_per_request(), 47);
        assert_eq!(snapshot_bytes(47, 512), Some(57_320_448));
        assert!(snapshot_bytes(47, 512).unwrap() + 128 * 1024 * 1024 < APP);
    }
    #[test]
    fn shared_model_joint_budget_checks_before_second_allocation() {
        let available = LIMIT - APP;
        let (first, shared) = (1_209_576_594, 633_366_062);
        let second = second_budget(first, shared).unwrap();
        assert_eq!(joint(first, second, shared), Some(LIMIT));
        assert_eq!(joint(first, second + 1, shared), None);
        assert_eq!(second_budget(available + 1, shared), None);
        assert_eq!(second_budget(10, 11), None);
        assert_eq!(joint(usize::MAX, usize::MAX, 0), None);
        assert_eq!(snapshot_bytes(usize::MAX, 4096), None);
    }
    #[test]
    fn config_requires_fixed_counts_mode_order_and_approved_A() {
        let value = serde_json::json!({"format":"toklet-qwen3-reference-schedule-bench-v1","mode":"deep_pair",
            "order":"serial_first","warmups":3,"samples":10,"input":{"path":"/tmp/A.u32le","sha256":INPUT_SHA}});
        let mut c: Config = serde_json::from_value(value.clone()).unwrap();
        c.validate().unwrap();
        assert_eq!(c.mode, Mode::DeepPair);
        assert_eq!(c.mode.capacity(), 4096);
        assert_eq!(c.order.labels(), ["Serial", "ParallelOutputsExperimental"]);
        c.order = Order::ParallelFirst;
        assert_eq!(c.order.labels(), ["ParallelOutputsExperimental", "Serial"]);
        c.warmups = 2;
        assert!(c.validate().is_err());
        c.warmups = 3;
        c.samples = 9;
        assert!(c.validate().is_err());
        c.samples = 10;
        c.input.sha256 = "a".repeat(64);
        assert!(c.validate().is_err());
        for (field, bad) in [
            ("mode", serde_json::json!("all")),
            ("order", serde_json::json!("automatic")),
            ("capacity", serde_json::json!(8192)),
        ] {
            let mut altered = value.clone();
            altered[field] = bad;
            assert!(serde_json::from_value::<Config>(altered).is_err());
        }
        assert!(!digest(&"A".repeat(64)));
        assert!(digest(INPUT_SHA));
    }
}
