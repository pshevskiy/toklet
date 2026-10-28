//! Frozen finite-context experiment. No production admission or arithmetic changes.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[allow(dead_code)]
#[path = "loss.rs"]
mod loss;

#[cfg(any(
    test,
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
mod protocol {
    use serde::{Deserialize, Serialize};
    pub const CAPACITY: usize = 4096;
    pub const VOCAB: usize = 151936;
    #[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
    #[serde(rename_all = "snake_case")]
    pub enum Phase {
        Anchor,
        A32,
        A128,
        B128,
        Timing32Full,
        Timing128Full,
        Timing32Greedy,
        SampledReplay,
        Quality,
    }
    impl Phase {
        pub fn chunk(self) -> usize {
            if matches!(self, Self::A128 | Self::B128 | Self::Timing128Full) {
                128
            } else {
                32
            }
        }
        pub fn rows(self) -> usize {
            if self == Self::Anchor { 512 } else { CAPACITY }
        }
        pub fn committed_tokens(self) -> usize {
            if self == Self::A32 { 8192 } else { self.rows() }
        }
        pub fn expected_rows(self) -> usize {
            match self {
                Self::Anchor | Self::A32 => 512,
                Self::A128 => CAPACITY,
                Self::B128 => CAPACITY - 1,
                _ => 0,
            }
        }
        pub fn diagnostic(self) -> bool {
            matches!(
                self,
                Self::Anchor | Self::A32 | Self::A128 | Self::B128 | Self::Quality
            )
        }
        pub fn timing(self) -> bool {
            matches!(
                self,
                Self::Timing32Full | Self::Timing128Full | Self::Timing32Greedy
            )
        }
        pub fn historical_rows(self) -> bool {
            matches!(self, Self::A32 | Self::A128 | Self::B128 | Self::Quality)
        }
    }
    #[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
    #[serde(rename_all = "snake_case")]
    pub enum Schedule {
        Serial,
        ParallelOutputsExperimental,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct Binding {
        pub path: std::path::PathBuf,
        pub sha256: String,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct Config {
        pub format: String,
        pub phase: Phase,
        pub reference_schedule: Option<Schedule>,
        pub historical_rows: Option<Binding>,
        pub input: Binding,
        pub expected_rows: Option<Binding>,
        pub expected_visible_tokens: Option<Binding>,
        pub expected_final_logit_sha256: Option<String>,
        pub expected_final_greedy_token: Option<u32>,
    }
    pub fn digest(value: &str) -> bool {
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }
    impl Config {
        pub fn validate(&self) -> Result<(), &'static str> {
            let format_ok = match self.reference_schedule {
                None => self.format == "toklet-qwen3-context4096-native-phase-v1",
                Some(_) => self.format == "toklet-qwen3-parallel-reference-phase-v1",
            };
            if !format_ok
                || self.historical_rows.is_some()
                    != (self.reference_schedule.is_some() && self.phase.historical_rows())
                || self.expected_rows.is_some() != (self.phase.expected_rows() > 0)
                || self.expected_visible_tokens.is_some()
                    != matches!(self.phase, Phase::Timing128Full | Phase::Timing32Greedy)
                || self.expected_final_logit_sha256.is_some()
                    != matches!(self.phase, Phase::Timing128Full | Phase::SampledReplay)
                || self.expected_final_greedy_token.is_some()
                    != matches!(self.phase, Phase::Timing128Full | Phase::Timing32Greedy)
                || self
                    .expected_final_greedy_token
                    .is_some_and(|t| t as usize >= VOCAB)
                || self
                    .expected_final_logit_sha256
                    .as_ref()
                    .is_some_and(|s| !digest(s))
                || [
                    Some(&self.input),
                    self.expected_rows.as_ref(),
                    self.expected_visible_tokens.as_ref(),
                    self.historical_rows.as_ref(),
                ]
                .iter()
                .flatten()
                .any(|b| !b.path.is_absolute() || !digest(&b.sha256))
            {
                return Err("phase protocol, comparison bindings or digest differs");
            }
            Ok(())
        }
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct HistoricalRows {
        pub format: String,
        pub phase: Phase,
        pub input_sha256: String,
        pub parent_result_sha256: String,
        pub parent_journal_sha256: String,
        pub rows: Vec<String>,
    }
    impl HistoricalRows {
        pub fn validate(&self, phase: Phase, input_sha256: &str) -> Result<(), &'static str> {
            if self.format != "toklet-qwen3-f2-historical-rows-v1"
                || !phase.historical_rows()
                || self.phase != phase
                || !digest(input_sha256)
                || self.input_sha256 != input_sha256
                || self.parent_result_sha256
                    != "2961d4baef9c1fc488ea2497efda705ea5b8e96f3d4a021bebc1f889235cc9a6"
                || !digest(&self.parent_journal_sha256)
                || self.rows.len() != CAPACITY
                || self.rows.iter().any(|value| !digest(value))
            {
                return Err("complete phase/input/parent-bound historical rows required");
            }
            Ok(())
        }
    }
    pub fn validate_ids(ids: &[u32], count: usize) -> Result<(), &'static str> {
        if ids.len() != count || ids.iter().any(|&id| id as usize >= VOCAB) {
            return Err("complete token count or vocabulary differs");
        }
        Ok(())
    }
    pub fn target(position: usize) -> Option<usize> {
        (3071..4095).contains(&position).then_some(position + 1)
    }
}

#[cfg(any(
    test,
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
mod protocol8192 {
    use super::protocol::{Binding, Phase, Schedule, VOCAB, digest};
    use serde::Deserialize;
    pub const CAPACITY: usize = 8192;
    pub const SOURCE_A_SHA: &str =
        "45f734c5fca32780b251eb14bf777bd4c512ba36a9a655c256d064aeea8b5c0b";
    const F2_RESULT_SHA: &str = "2961d4baef9c1fc488ea2497efda705ea5b8e96f3d4a021bebc1f889235cc9a6";
    const A32_JOURNAL_SHA: &str =
        "3f53bed75c72f631ed04dfb38584fb95b610adab8b14f93f5c0c0b404c86cd66";

    pub fn numerical(phase: Phase) -> bool {
        matches!(phase, Phase::A32 | Phase::A128 | Phase::B128)
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct Config {
        pub format: String,
        pub phase: Phase,
        pub reference_schedule: Schedule,
        pub retained_parallel_result_sha256: String,
        pub prefix_rows: Option<Binding>,
        pub input: Binding,
        pub expected_rows: Option<Binding>,
        pub expected_visible_tokens: Option<Binding>,
        pub expected_final_logit_sha256: Option<String>,
        pub expected_final_greedy_token: Option<u32>,
    }
    impl Config {
        pub fn validate(&self) -> Result<(), &'static str> {
            if self.format != "toklet-qwen3-context8192-native-phase-v1"
                || self.reference_schedule != Schedule::ParallelOutputsExperimental
                || !digest(&self.retained_parallel_result_sha256)
                || self.prefix_rows.is_some() != numerical(self.phase)
                || self.expected_rows.is_some() != (self.phase.expected_rows() > 0)
                || self.expected_visible_tokens.is_some()
                    != matches!(self.phase, Phase::Timing128Full | Phase::Timing32Greedy)
                || self.expected_final_logit_sha256.is_some()
                    != matches!(self.phase, Phase::Timing128Full | Phase::SampledReplay)
                || self.expected_final_greedy_token.is_some()
                    != matches!(self.phase, Phase::Timing128Full | Phase::Timing32Greedy)
                || self
                    .expected_final_greedy_token
                    .is_some_and(|id| id as usize >= VOCAB)
                || self
                    .expected_final_logit_sha256
                    .as_ref()
                    .is_some_and(|s| !digest(s))
                || [
                    Some(&self.input),
                    self.expected_rows.as_ref(),
                    self.expected_visible_tokens.as_ref(),
                    self.prefix_rows.as_ref(),
                ]
                .iter()
                .flatten()
                .any(|b| !b.path.is_absolute() || !digest(&b.sha256))
            {
                return Err("8192 phase protocol or comparison bindings differ");
            }
            Ok(())
        }
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct PrefixRows {
        pub format: String,
        pub phase: Phase,
        pub source_input_sha256: String,
        pub source_result_sha256: String,
        pub source_journal_sha256: String,
        pub retained_parallel_result_sha256: String,
        pub destination_input_sha256: String,
        pub destination_prefix_sha256: String,
        pub rows: Vec<String>,
    }
    impl PrefixRows {
        pub fn validate(
            &self,
            phase: Phase,
            destination: &str,
            prefix: &str,
            retained: &str,
        ) -> Result<(), &'static str> {
            if self.format != "toklet-qwen3-context8192-prefix-rows-v1"
                || !numerical(phase)
                || self.phase != phase
                || self.source_input_sha256 != SOURCE_A_SHA
                || self.source_result_sha256 != F2_RESULT_SHA
                || self.source_journal_sha256 != A32_JOURNAL_SHA
                || !digest(retained)
                || self.retained_parallel_result_sha256 != retained
                || !digest(destination)
                || self.destination_input_sha256 != destination
                || prefix != SOURCE_A_SHA
                || self.destination_prefix_sha256 != prefix
                || self.rows.len() != 4096
                || self.rows.iter().any(|s| !digest(s))
            {
                return Err("8192 requires the complete authenticated historical A4096 prefix");
            }
            Ok(())
        }
    }
}

/// Two closed protocols; never a caller-supplied capacity or scoring interval.
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Endpoint {
    Context4096,
    Context8192,
}
#[cfg(any(
    test,
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
impl Endpoint {
    fn capacity(self) -> usize {
        match self {
            Self::Context4096 => protocol::CAPACITY,
            Self::Context8192 => protocol8192::CAPACITY,
        }
    }
    fn rows(self, phase: protocol::Phase) -> usize {
        match self {
            Self::Context4096 => phase.rows(),
            Self::Context8192 if phase == protocol::Phase::Anchor => 512,
            Self::Context8192 => self.capacity(),
        }
    }
    fn expected_rows(self, phase: protocol::Phase) -> usize {
        match self {
            Self::Context4096 => phase.expected_rows(),
            Self::Context8192 => match phase {
                protocol::Phase::Anchor | protocol::Phase::A32 => 512,
                protocol::Phase::A128 => 8192,
                protocol::Phase::B128 => 8191,
                _ => 0,
            },
        }
    }
    fn committed_tokens(self, phase: protocol::Phase) -> usize {
        match self {
            Self::Context4096 => phase.committed_tokens(),
            Self::Context8192 if phase == protocol::Phase::A32 => 16384,
            Self::Context8192 => self.rows(phase),
        }
    }
    fn target(self, position: usize) -> Option<usize> {
        match self {
            Self::Context4096 => protocol::target(position),
            Self::Context8192 => (7167..8191).contains(&position).then_some(position + 1),
        }
    }
    fn events(self) -> &'static str {
        match self {
            Self::Context4096 => "toklet-qwen3-context4096-native-events-v1",
            Self::Context8192 => "toklet-qwen3-context8192-native-events-v1",
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use super::{Endpoint, loss, protocol::*, protocol8192};
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
        metal::{self, AttentionMode, NormalizationMode, ProjectionMode, ReadbackMode},
        session, weights,
    };
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    const APP_BYTES: usize = 64 * 1024 * 1024;
    type Identity = (u64, u64, u64, i64, i64, i64, i64);
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
    fn open(path: &Path) -> Result<File> {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path)?;
        if !file.metadata()?.is_file() {
            return Err("regular input file required".into());
        }
        Ok(file)
    }
    struct Owner {
        binding: Binding,
        file: File,
        identity: Identity,
    }
    impl Owner {
        fn read(binding: Binding, limit: usize) -> Result<(Self, Vec<u8>)> {
            let mut file = open(&binding.path)?;
            let before = identity(&file)?;
            if before.2 == 0 || before.2 > limit as u64 {
                return Err("bounded nonempty artifact required".into());
            }
            let mut bytes = vec![0; before.2 as usize];
            file.read_exact(&mut bytes)?;
            if file.read(&mut [0])? != 0
                || identity(&file)? != before
                || hash(&bytes) != binding.sha256
            {
                return Err("artifact identity/digest changed".into());
            }
            Ok((
                Self {
                    binding,
                    file,
                    identity: before,
                },
                bytes,
            ))
        }
        fn close_check(&mut self) -> Result<()> {
            self.file.seek(SeekFrom::Start(0))?;
            let mut bytes = vec![0; self.identity.2 as usize];
            self.file.read_exact(&mut bytes)?;
            if identity(&self.file)? != self.identity
                || identity(&open(&self.binding.path)?)? != self.identity
                || hash(&bytes) != self.binding.sha256
                || self.file.read(&mut [0])? != 0
            {
                return Err("input endpoint changed".into());
            }
            Ok(())
        }
    }
    fn hash(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }
    fn row_hash(row: &[f32]) -> String {
        let mut h = Sha256::new();
        for x in row {
            h.update(x.to_bits().to_le_bytes());
        }
        format!("{:x}", h.finalize())
    }
    fn ids(bytes: &[u8]) -> Result<Vec<u32>> {
        if !bytes.len().is_multiple_of(4) {
            return Err("UInt32 input extent differs".into());
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
    struct Journal<W> {
        writer: W,
        chunk: usize,
        stage: &'static str,
        capacity: usize,
    }
    fn emit(out: &mut Journal<impl Write>, mut value: Value) -> Result<()> {
        value
            .as_object_mut()
            .ok_or("journal object required")?
            .insert("stage".into(), json!(out.stage));
        serde_json::to_writer(&mut out.writer, &value)?;
        out.writer.write_all(b"\n")?;
        out.writer.flush()?;
        Ok(())
    }
    fn state(s: &metal::Session<'_>) -> Value {
        json!({"history":s.history(),"evaluated":s.evaluated_len(),"pending":s.pending_len(),
            "logits_sha256":s.logits().map(row_hash),"greedy_token":s.greedy_token().map(|r|r.unwrap())})
    }
    fn require(value: bool, error: &'static str) -> Result<()> {
        if value { Ok(()) } else { Err(error.into()) }
    }
    fn reject(s: &mut metal::Session<'_>, out: &mut Journal<impl Write>) -> Result<()> {
        let before = state(s);
        for operation in ["append", "set_tokens"] {
            let result = if operation == "append" {
                s.append(&[0])
            } else {
                s.set_tokens(&vec![0; out.capacity + 1])
            };
            let after = state(s);
            emit(
                out,
                json!({"event":"capacity_rejection","operation":operation,"requested_history":out.capacity+1,
                "result":format!("{result:?}"),"before":before,"after":after}),
            )?;
            require(
                matches!(
                    result,
                    Err(metal::Error::Options(
                        "token ID/count outside session capacity"
                    ))
                ),
                "capacity rejection differs",
            )?;
            require(before == after, "capacity rejection changed state")?;
        }
        Ok(())
    }
    fn call(
        s: &mut metal::Session<'_>,
        out: &mut Journal<impl Write>,
        label: &str,
    ) -> Result<usize> {
        let depth = s.evaluated_len();
        let pending = s.pending_len();
        let start = Instant::now();
        let result = s.evaluate_next_chunk();
        let wall = start.elapsed().as_secs_f64();
        emit(
            out,
            json!({"event":"call","label":label,"pre_call_depth":depth,"pre_call_pending":pending,
            "rows":result.as_ref().ok(),"error":result.as_ref().err().map(ToString::to_string),
            "outer_completed_call_seconds":wall,"timing":result.as_ref().ok().filter(|n| **n > 0).and_then(|_|s.timing()),"evaluated":s.evaluated_len(),"pending":s.pending_len()}),
        )?;
        let rows = result?;
        require(
            rows == pending.min(out.chunk)
                && s.evaluated_len() == depth + rows
                && s.pending_len() == pending - rows,
            "completed call rows or state differs",
        )?;
        Ok(rows)
    }
    fn evaluate(
        s: &mut metal::Session<'_>,
        out: &mut Journal<impl Write>,
        label: &str,
    ) -> Result<()> {
        while s.pending_len() > 0 {
            require(
                call(s, out, label)? > 0,
                "empty evaluation with pending history",
            )?;
        }
        Ok(())
    }
    fn final_row(
        s: &metal::Session<'_>,
        expected: &str,
        out: &mut Journal<impl Write>,
    ) -> Result<()> {
        let row = s.logits().ok_or("final full distribution absent")?;
        let actual = row_hash(row);
        let finite = row.iter().all(|v| v.is_finite());
        emit(
            out,
            json!({"event":"final_distribution","sha256":actual,"expected_sha256":expected,"finite":finite}),
        )?;
        if actual != expected || !finite {
            emit(
                out,
                json!({"event":"failing_row","position":s.evaluated_len()-1,"expected_sha256":expected,
                "actual_fp32_bits":row.iter().map(|v|v.to_bits()).collect::<Vec<_>>()}),
            )?;
        }
        require(actual == expected && finite, "final distribution mismatch")
    }
    // Configuration is converted only after its distinct wire validator passes.
    struct RunConfig {
        phase: Phase,
        reference_schedule: Option<Schedule>,
        historical_rows: Option<Binding>,
        prefix_rows: Option<Binding>,
        retained_parallel_result_sha256: Option<String>,
        input: Binding,
        expected_rows: Option<Binding>,
        expected_visible_tokens: Option<Binding>,
        expected_final_logit_sha256: Option<String>,
        expected_final_greedy_token: Option<u32>,
    }
    fn parse_config(endpoint: Endpoint, bytes: &[u8]) -> Result<RunConfig> {
        Ok(match endpoint {
            Endpoint::Context4096 => {
                let c: Config = serde_json::from_slice(bytes)?;
                c.validate()?;
                RunConfig {
                    phase: c.phase,
                    reference_schedule: c.reference_schedule,
                    historical_rows: c.historical_rows,
                    prefix_rows: None,
                    retained_parallel_result_sha256: None,
                    input: c.input,
                    expected_rows: c.expected_rows,
                    expected_visible_tokens: c.expected_visible_tokens,
                    expected_final_logit_sha256: c.expected_final_logit_sha256,
                    expected_final_greedy_token: c.expected_final_greedy_token,
                }
            }
            Endpoint::Context8192 => {
                let c: protocol8192::Config = serde_json::from_slice(bytes)?;
                c.validate()?;
                RunConfig {
                    phase: c.phase,
                    reference_schedule: Some(c.reference_schedule),
                    historical_rows: None,
                    prefix_rows: c.prefix_rows,
                    retained_parallel_result_sha256: Some(c.retained_parallel_result_sha256),
                    input: c.input,
                    expected_rows: c.expected_rows,
                    expected_visible_tokens: c.expected_visible_tokens,
                    expected_final_logit_sha256: c.expected_final_logit_sha256,
                    expected_final_greedy_token: c.expected_final_greedy_token,
                }
            }
        })
    }
    pub fn run(endpoint: Endpoint) -> Result<()> {
        let mut args = std::env::args_os().skip(1);
        let model_path = args.next().ok_or("missing model directory")?;
        let path = args.next().ok_or("missing phase config")?.into();
        let sha256 = args
            .next()
            .ok_or("missing config SHA256")?
            .into_string()
            .map_err(|_| "invalid digest")?;
        if args.next().is_some() || !digest(&sha256) {
            return Err("invalid phase arguments".into());
        }
        let (owner, body) = Owner::read(Binding { path, sha256 }, 16384)?;
        let config = parse_config(endpoint, &body)?;
        let capacity = endpoint.capacity();
        let config_sha = owner.binding.sha256.clone();
        let mut owners = vec![owner];
        let phase = config.phase;
        let explicit_schedule = config.reference_schedule;
        let selected_schedule = explicit_schedule.unwrap_or(Schedule::Serial);
        let input_sha = config.input.sha256.clone();
        let (owner, body) = Owner::read(config.input, capacity * 4)?;
        owners.push(owner);
        let input = ids(&body)?;
        validate_ids(&input, capacity)?;
        let prefix_sha = if endpoint == Endpoint::Context8192 {
            let prefix = hash(&body[..4096 * 4]);
            if !matches!(phase, Phase::Quality | Phase::SampledReplay) {
                require(
                    prefix == protocol8192::SOURCE_A_SHA,
                    "8192 numerical prefix must be historical A4096",
                )?;
                require(
                    input.iter().enumerate().all(|(i, &token)| {
                        let original = input[i % 4096];
                        token
                            == if phase == Phase::B128 && i == 8191 {
                                (original + 1) % VOCAB as u32
                            } else {
                                original
                            }
                    }),
                    "8192 A/B repeated-prefix recipe differs",
                )?;
            }
            Some(prefix)
        } else {
            None
        };
        let expected = if let Some(binding) = config.expected_rows {
            let (owner, body) = Owner::read(
                binding,
                if endpoint == Endpoint::Context4096 {
                    512 * 1024
                } else {
                    1024 * 1024
                },
            )?;
            owners.push(owner);
            let rows: Vec<String> = serde_json::from_slice(&body)?;
            require(
                rows.len() == endpoint.expected_rows(phase) && rows.iter().all(|s| digest(s)),
                "expected row coverage differs",
            )?;
            rows
        } else {
            Vec::new()
        };
        let historical = if let Some(binding) = config.historical_rows {
            let (owner, body) = Owner::read(binding, 512 * 1024)?;
            owners.push(owner);
            let rows: HistoricalRows = serde_json::from_slice(&body)?;
            rows.validate(phase, &input_sha)?;
            rows.rows
        } else {
            Vec::new()
        };
        let prefix_binding_sha = config.prefix_rows.as_ref().map(|b| b.sha256.clone());
        let prefix_rows = if let Some(binding) = config.prefix_rows {
            let (owner, body) = Owner::read(binding, 512 * 1024)?;
            owners.push(owner);
            let rows: protocol8192::PrefixRows = serde_json::from_slice(&body)?;
            rows.validate(
                phase,
                &input_sha,
                prefix_sha.as_deref().ok_or("missing8192 prefix")?,
                config
                    .retained_parallel_result_sha256
                    .as_deref()
                    .ok_or("missing retained schedule parent")?,
            )?;
            rows.rows
        } else {
            Vec::new()
        };
        let visible_expected = if let Some(binding) = config.expected_visible_tokens {
            let (owner, body) = Owner::read(binding, 128)?;
            owners.push(owner);
            let tokens = ids(&body)?;
            validate_ids(&tokens, 32)?;
            tokens
        } else {
            Vec::new()
        };
        let mut out = Journal {
            writer: io::stdout().lock(),
            chunk: phase.chunk(),
            stage: "main",
            capacity,
        };
        emit(
            &mut out,
            json!({"event":"start","format":endpoint.events(),"phase":phase,
            "config_sha256":config_sha,"input_sha256":input_sha,"input_tokens":input.len(),
            "expected_rows":expected.len(),"profile":"MlxAffine/Compensated/Reference/FP16",
            "reference_schedule":selected_schedule,
            "diagnostic_only":true,"external_source_identity_and_resource_endpoints_required":true}),
        )?;
        if endpoint == Endpoint::Context8192 {
            emit(
                &mut out,
                json!({"event":"depth_parent_bindings",
                "retained_parallel_result_sha256":config.retained_parallel_result_sha256,
                "historical_A4096_prefix_sha256":prefix_sha,"prefix_receipt_sha256":prefix_binding_sha,
                "prefix_rows":prefix_rows.len(),"new_quality_input_has_no_historical_Q_comparison":phase==Phase::Quality}),
            )?;
        }
        let budget = weights::INITIAL_MEMORY_LIMIT_BYTES - APP_BYTES;
        let start = Instant::now();
        let options = weights::LoadOptions {
            context_limit: endpoint.rows(phase),
            memory_limit_bytes: budget,
        };
        let model = if phase == Phase::Anchor && explicit_schedule.is_none() {
            metal::Model::load_mlx(Path::new(&model_path), options)?
        } else {
            metal::Model::load_mlx_qwen3_context_diagnostic(Path::new(&model_path), options)?
        };
        let load_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let readback = if phase == Phase::Timing32Greedy {
            ReadbackMode::GreedyToken
        } else {
            ReadbackMode::FullLogits
        };
        let session_options = session::Options {
            capacity: endpoint.rows(phase),
            prefill_chunk_size: phase.chunk(),
            position_offset: 0,
            memory_limit_bytes: budget,
        };
        let mut s = if endpoint == Endpoint::Context8192 {
            metal::Session::with_qwen3_reference_schedule8192_diagnostic(
                &model,
                session_options,
                readback,
                metal::ReferenceSchedule::ParallelOutputsExperimental,
            )?
        } else if let Some(schedule) = explicit_schedule {
            metal::Session::with_qwen3_reference_schedule_diagnostic(
                &model,
                session_options,
                readback,
                match schedule {
                    Schedule::Serial => metal::ReferenceSchedule::Serial,
                    Schedule::ParallelOutputsExperimental => {
                        metal::ReferenceSchedule::ParallelOutputsExperimental
                    }
                },
            )?
        } else {
            metal::Session::with_attention_mode(
                &model,
                session_options,
                ProjectionMode::MlxAffine,
                NormalizationMode::Compensated,
                readback,
                AttentionMode::Reference,
            )?
        };
        if phase.diagnostic() {
            s.enable_diagnostics(&[])?;
        }
        let session_seconds = start.elapsed().as_secs_f64();
        let admitted = s
            .memory()
            .admitted_bytes
            .checked_add(APP_BYTES)
            .ok_or("memory overflow")?;
        emit(
            &mut out,
            json!({"event":"ready","device":model.device_name(),"debug_assertions":cfg!(debug_assertions),
            "capacity":endpoint.rows(phase),"chunk":phase.chunk(),"readback":readback,"diagnostics":phase.diagnostic(),
            "reference_schedule":format!("{:?}",s.reference_schedule()),
            "load_compile_seconds":load_seconds,"session_creation_seconds":session_seconds,"memory":s.memory(),
            "application_allowance_bytes":APP_BYTES,"admitted_bytes":admitted,"state":state(&s)}),
        )?;
        require(
            admitted <= weights::INITIAL_MEMORY_LIMIT_BYTES,
            "process allowance exceeds6GiB",
        )?;
        let mut committed = 0;
        if phase.diagnostic() {
            let mut total = loss::Sum::default();
            let mut targets = 0;
            let mut observed = 0;
            let mut row_hashes = Vec::with_capacity(endpoint.rows(phase));
            s.append(&input[..endpoint.rows(phase)])?;
            while s.pending_len() > 0 {
                let position = s.evaluated_len();
                let n = call(&mut s, &mut out, "diagnostic")?;
                committed += n;
                let snapshots = s.diagnostics()?;
                require(
                    snapshots.iter().map(|d| d.stage).collect::<Vec<_>>()
                        == ["embedding", "final_norm", "logits"],
                    "snapshot stages differ",
                )?;
                let logits = snapshots
                    .iter()
                    .find(|d| d.stage == "logits")
                    .ok_or("missing all-row logits")?;
                require(
                    logits.rows == n && logits.columns == VOCAB && logits.values.len() == n * VOCAB,
                    "row extent differs",
                )?;
                for (r, row) in logits.values.chunks_exact(VOCAB).enumerate() {
                    let position = position + r;
                    let hash = row_hash(row);
                    let finite = row.iter().all(|v| v.is_finite());
                    let expected_hash = expected.get(position);
                    let historical_hash = historical
                        .get(position)
                        .or_else(|| prefix_rows.get(position));
                    let matches = expected_hash.is_none_or(|h| *h == hash)
                        && historical_hash.is_none_or(|h| *h == hash);
                    emit(
                        &mut out,
                        json!({"event":"row","position":position,"sha256":hash,"finite":finite,
                        "expected_sha256":expected_hash,"matches":matches,"evaluated":s.evaluated_len(),"pending":s.pending_len()}),
                    )?;
                    if !finite || !matches {
                        let mut failure = json!({"event":"failing_row","position":position,"expected_sha256":expected_hash,
                            "actual_fp32_bits":row.iter().map(|v|v.to_bits()).collect::<Vec<_>>()});
                        if let Some(historical_hash) = historical_hash {
                            failure
                                .as_object_mut()
                                .unwrap()
                                .insert("historical_sha256".into(), json!(historical_hash));
                        }
                        emit(&mut out, failure)?;
                    }
                    require(finite && matches, "nonfinite or unequal row")?;
                    observed += 1;
                    row_hashes.push(hash.clone());
                    if phase == Phase::Quality
                        && let Some(index) = endpoint.target(position)
                    {
                        let token = input[index];
                        let nll = loss::loss(row, token as usize)?;
                        total.add(nll);
                        targets += 1;
                        emit(
                            &mut out,
                            json!({"event":"loss","position":position,"target_index":index,"target_token":token,
                            "negative_log_likelihood":nll,"target_logit_f32_bits":row[token as usize].to_bits(),"logits_sha256":hash}),
                        )?;
                    }
                }
            }
            emit(
                &mut out,
                json!({"event":"rows_done","rows":observed,"state":state(&s)}),
            )?;
            require(
                observed == endpoint.rows(phase)
                    && s.evaluated_len() == endpoint.rows(phase)
                    && s.history() == &input[..endpoint.rows(phase)],
                "coverage differs",
            )?;
            if phase == Phase::Quality {
                emit(
                    &mut out,
                    json!({"event":"quality_done","targets":targets,"first_target_index":capacity-1024,"last_target_index":capacity-1,
                    "total_negative_log_likelihood":total.value,"mean_negative_log_likelihood":total.value/targets as f64}),
                )?;
                require(targets == 1024, "quality target count differs")?;
            }
            if phase == Phase::A32 {
                let before = state(&s);
                s.set_tokens(&input)?;
                let rows = call(&mut s, &mut out, "fully_cached")?;
                emit(
                    &mut out,
                    json!({"event":"cached","rows":rows,"before":before,"after":state(&s)}),
                )?;
                require(
                    rows == 0 && before == state(&s),
                    "fully cached evaluation changed state",
                )?;
                reject(&mut s, &mut out)?;
                out.stage = "prescribed_replay";
                let complete = state(&s);
                s.reset();
                emit(&mut out, json!({"event":"reset","state":state(&s)}))?;
                require(
                    s.history().is_empty()
                        && s.evaluated_len() == 0
                        && s.pending_len() == 0
                        && s.logits().is_none(),
                    "reset differs",
                )?;
                s.append(&input[..capacity - 1])?;
                evaluate(&mut s, &mut out, "prescribed_prefix")?;
                committed += capacity - 1;
                final_row(&s, &row_hashes[capacity - 2], &mut out)?;
                let prefix = state(&s);
                s.append(&input[capacity - 1..])?;
                emit(
                    &mut out,
                    json!({"event":"prescribed_pending","state":state(&s)}),
                )?;
                require(
                    s.evaluated_len() == capacity - 1
                        && s.pending_len() == 1
                        && s.history() == input,
                    "pending prescribed token differs",
                )?;
                reject(&mut s, &mut out)?;
                s.discard_pending();
                emit(
                    &mut out,
                    json!({"event":"discard","before":prefix,"after":state(&s)}),
                )?;
                require(state(&s) == prefix, "discard changed prefix")?;
                s.append(&input[capacity - 1..])?;
                committed += call(&mut s, &mut out, "prescribed_flush")?;
                final_row(&s, &row_hashes[capacity - 1], &mut out)?;
                reject(&mut s, &mut out)?;

                emit(
                    &mut out,
                    json!({"event":"replay_done","before":complete,"after":state(&s)}),
                )?;
                require(complete == state(&s), "reset replay changed full state")?;
            }
        } else if phase.timing() {
            // Raw greedy continuation with explicitly empty stop IDs and output limit32.
            // Calls are made explicitly so every prefill chunk and decode has one Timing record.
            s.append(&input[..capacity - 32])?;
            evaluate(&mut s, &mut out, "ordinary_prefill")?;
            committed += capacity - 32;
            let mut visible = Vec::with_capacity(32);
            for index in 0..32 {
                let token = s.greedy_token().ok_or("missing greedy distribution")??;
                visible.push(token);
                s.append(&[token])?;
                emit(
                    &mut out,
                    json!({"event":"token","index":index,"token_id":token,"evaluated":s.evaluated_len(),"pending":s.pending_len()}),
                )?;
                if index < 31 {
                    committed += call(&mut s, &mut out, "ordinary_decode")?;
                }
            }
            emit(
                &mut out,
                json!({"event":"generation_done","visible_token_ids":visible,"stop_tokens":[],"stop_reason":"output_limit",
                "sampled_tokens":32,"emitted_tokens":32,"state":state(&s)}),
            )?;
            require(
                s.history().len() == capacity
                    && s.evaluated_len() == capacity - 1
                    && s.pending_len() == 1,
                "pending final generation state differs",
            )?;
            require(
                visible_expected.is_empty() || visible_expected == visible,
                "ordinary visible tokens differ",
            )?;
            committed += call(&mut s, &mut out, "ordinary_final_flush")?;
            if let Some(expected) = config.expected_final_logit_sha256 {
                final_row(&s, &expected, &mut out)?;
            }
            if let Some(expected) = config.expected_final_greedy_token {
                let actual = s.greedy_token().ok_or("final selection missing")??;
                emit(
                    &mut out,
                    json!({"event":"final_selection", "actual":actual, "expected":expected}),
                )?;
                require(actual == expected, "final selection differs")?;
            }
            reject(&mut s, &mut out)?;
        } else {
            s.append(&input)?;
            evaluate(&mut s, &mut out, "sampled_history_replay")?;
            committed += capacity;
            final_row(
                &s,
                config
                    .expected_final_logit_sha256
                    .as_deref()
                    .ok_or("missing sampled final hash")?,
                &mut out,
            )?;
            reject(&mut s, &mut out)?;
        }
        emit(
            &mut out,
            json!({"event":"final_state","state":state(&s),"committed_tokens":committed,"memory":s.memory()}),
        )?;
        require(
            committed == endpoint.committed_tokens(phase)
                && s.evaluated_len() == endpoint.rows(phase)
                && s.pending_len() == 0,
            "final committed workload differs",
        )?;
        drop(s);
        drop(model);
        for owner in &mut owners {
            owner.close_check()?;
        }
        emit(
            &mut out,
            json!({"event":"done","phase":phase,"committed_tokens":committed,"input_endpoints_checked":true,
            "passed":true,"performance_qualified":false,"full_window_qualified":false}),
        )
    }
}
pub fn main(endpoint: Endpoint) {
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    if let Err(error) = apple::run(endpoint) {
        eprintln!("{error}");
        std::process::exit(1);
    }
    #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = endpoint;
        eprintln!("Qwen3 context diagnostic requires Apple Silicon macOS and --features metal");
        std::process::exit(1);
    }
}
#[cfg(test)]
mod tests {
    use super::protocol::*;
    #[test]
    fn exact_quality_offsets_and_workload() {
        let targets: Vec<_> = (0..4096).filter_map(target).collect();
        assert_eq!(targets, (3072..4096).collect::<Vec<_>>());
        assert_eq!(targets.len(), 1024);
        assert_eq!(
            [
                Phase::Anchor,
                Phase::A32,
                Phase::A128,
                Phase::B128,
                Phase::Timing32Full,
                Phase::Timing128Full,
                Phase::Timing32Greedy,
                Phase::SampledReplay,
                Phase::Quality
            ]
            .iter()
            .map(|p| p.committed_tokens())
            .sum::<usize>(),
            37376
        );
        assert!(Phase::A128.diagnostic());
        assert!(Phase::Timing128Full.timing());
        assert!(!Phase::Timing32Full.diagnostic());
        assert_eq!(Phase::A128.chunk(), 128);
        assert_eq!(Phase::Quality.chunk(), 32);
    }
    #[test]
    fn cold_protocol_rejects_missing_or_extraneous_comparisons_and_corrupt_ids() {
        let mut c: Config = serde_json::from_value(
            serde_json::json!({"format":"toklet-qwen3-context4096-native-phase-v1",
            "phase":"a32","input":{"path":"/tmp/a","sha256":"a".repeat(64)},
            "expected_rows":{"path":"/tmp/anchor","sha256":"b".repeat(64)}}),
        )
        .unwrap();
        c.validate().unwrap();
        c.expected_rows = None;
        assert!(c.validate().is_err());
        c.phase = Phase::Quality;
        c.validate().unwrap();
        c.historical_rows = Some(Binding {
            path: "/tmp/historical".into(),
            sha256: "b".repeat(64),
        });
        assert!(c.validate().is_err());
        c.historical_rows = None;
        c.input.sha256 = "A".repeat(64);
        assert!(c.validate().is_err());
        c.input.sha256 = "a".repeat(64);
        c.expected_final_logit_sha256 = Some("b".repeat(64));
        assert!(c.validate().is_err());
        validate_ids(&vec![0; 4096], 4096).unwrap();
        for n in [0, 4095, 4097] {
            assert!(validate_ids(&vec![0; n], 4096).is_err());
        }
        let mut ids = vec![0; 4096];
        ids[4095] = 151936;
        assert!(validate_ids(&ids, 4096).is_err());
        assert!(serde_json::from_str::<Config>(r#"{"format":"x","phase":"quality","input":{"path":"/tmp/a","sha256":"x"},"capacity":8192}"#).is_err());
    }
    #[test]
    fn experimental_schedule_requires_explicit_versioned_protocol() {
        let value = serde_json::json!({"format":"toklet-qwen3-context4096-native-phase-v1",
            "phase":"quality","input":{"path":"/tmp/q","sha256":"a".repeat(64)}});
        let mut config: Config = serde_json::from_value(value.clone()).unwrap();
        config.validate().unwrap();
        config.reference_schedule = Some(Schedule::Serial);
        assert!(config.validate().is_err());
        config.format = "toklet-qwen3-parallel-reference-phase-v1".into();
        assert!(config.validate().is_err());
        config.historical_rows = Some(Binding {
            path: "/tmp/historical".into(),
            sha256: "b".repeat(64),
        });
        config.validate().unwrap();
        config.reference_schedule = Some(Schedule::ParallelOutputsExperimental);
        config.validate().unwrap();
        config.reference_schedule = None;
        assert!(config.validate().is_err());
        let mut corrupt = value;
        corrupt["reference_schedule"] = serde_json::json!("unknown");
        assert!(serde_json::from_value::<Config>(corrupt).is_err());
    }
    #[test]
    fn complete_historical_rows_require_the_frozen_phase_input_and_parent() {
        let mut rows = HistoricalRows {
            format: "toklet-qwen3-f2-historical-rows-v1".into(),
            phase: Phase::Quality,
            input_sha256: "a".repeat(64),
            parent_result_sha256:
                "2961d4baef9c1fc488ea2497efda705ea5b8e96f3d4a021bebc1f889235cc9a6".into(),
            parent_journal_sha256: "b".repeat(64),
            rows: vec!["c".repeat(64); CAPACITY],
        };
        rows.validate(Phase::Quality, &"a".repeat(64)).unwrap();
        assert!(rows.validate(Phase::A32, &"a".repeat(64)).is_err());
        assert!(rows.validate(Phase::Quality, &"b".repeat(64)).is_err());
        rows.rows.pop();
        assert!(rows.validate(Phase::Quality, &"a".repeat(64)).is_err());
        rows.rows.push("C".repeat(64));
        assert!(rows.validate(Phase::Quality, &"a".repeat(64)).is_err());
        rows.rows[4095] = "c".repeat(64);
        rows.rows.push("c".repeat(64));
        assert!(rows.validate(Phase::Quality, &"a".repeat(64)).is_err());
        rows.rows.pop();
        rows.parent_journal_sha256 = "x".repeat(64);
        assert!(rows.validate(Phase::Quality, &"a".repeat(64)).is_err());
        rows.parent_journal_sha256 = "b".repeat(64);
        rows.parent_result_sha256 = "d".repeat(64);
        assert!(rows.validate(Phase::Quality, &"a".repeat(64)).is_err());
    }
}

#[cfg(test)]
mod depth8192_tests {
    use super::{
        Endpoint,
        protocol::{self, Phase, Schedule},
        protocol8192,
    };
    use serde_json::json;

    fn config(phase: &str) -> serde_json::Value {
        let mut value = json!({"format":"toklet-qwen3-context8192-native-phase-v1",
            "phase":phase,"reference_schedule":"parallel_outputs_experimental",
            "retained_parallel_result_sha256":"d".repeat(64),
            "input":{"path":"/tmp/input8192","sha256":"a".repeat(64)}});
        if matches!(phase, "a32" | "a128" | "b128") {
            value["prefix_rows"] = json!({"path":"/tmp/A4096-prefix","sha256":"b".repeat(64)});
        }
        if matches!(phase, "anchor" | "a32" | "a128" | "b128") {
            value["expected_rows"] = json!({"path":"/tmp/rows","sha256":"c".repeat(64)});
        }
        value
    }
    #[test]
    fn two_fixed_endpoints_keep_literal_work_and_quality_coverage() {
        let phases = [
            Phase::Anchor,
            Phase::A32,
            Phase::A128,
            Phase::B128,
            Phase::Timing32Full,
            Phase::Timing128Full,
            Phase::Timing32Greedy,
            Phase::SampledReplay,
            Phase::Quality,
        ];
        for (endpoint, capacity, work, first, last) in [
            (Endpoint::Context4096, 4096, 37376, 3072, 4095),
            (Endpoint::Context8192, 8192, 74240, 7168, 8191),
        ] {
            assert_eq!(endpoint.capacity(), capacity);
            assert_eq!(
                phases
                    .iter()
                    .map(|&p| endpoint.committed_tokens(p))
                    .sum::<usize>(),
                work
            );
            let targets: Vec<_> = (0..capacity).filter_map(|p| endpoint.target(p)).collect();
            assert_eq!(targets, (first..=last).collect::<Vec<_>>());
            assert_eq!(targets.len(), 1024);
            assert_eq!(endpoint.rows(Phase::Anchor), 512);
            assert_eq!(endpoint.rows(Phase::Quality), capacity);
            assert_eq!(endpoint.expected_rows(Phase::A32), 512);
            assert_eq!(endpoint.expected_rows(Phase::A128), capacity);
            assert_eq!(endpoint.expected_rows(Phase::B128), capacity - 1);
            assert_eq!(endpoint.expected_rows(Phase::Quality), 0);
        }
        assert_ne!(
            Endpoint::Context4096.events(),
            Endpoint::Context8192.events()
        );
        assert_eq!(Endpoint::Context8192.rows(Phase::Timing32Full) - 32, 8160);
    }
    #[test]
    fn distinct8192_config_cannot_bypass4096_historical_policy() {
        for phase in ["anchor", "a32", "a128", "b128", "quality"] {
            let value = config(phase);
            let c: protocol8192::Config = serde_json::from_value(value.clone()).unwrap();
            c.validate().unwrap();
            assert!(serde_json::from_value::<protocol::Config>(value).is_err());
        }
        let mut c: protocol8192::Config = serde_json::from_value(config("a32")).unwrap();
        c.prefix_rows = None;
        assert!(c.validate().is_err());
        c.phase = Phase::Quality;
        c.expected_rows = None;
        c.validate().unwrap();
        c.reference_schedule = Schedule::Serial;
        assert!(c.validate().is_err());
        c.reference_schedule = Schedule::ParallelOutputsExperimental;
        c.format = "toklet-qwen3-parallel-reference-phase-v1".into();
        assert!(c.validate().is_err());
        let mut value = config("quality");
        value["historical_rows"] = json!({"path":"/tmp/oldQ","sha256":"a".repeat(64)});
        assert!(serde_json::from_value::<protocol8192::Config>(value).is_err());
        let mut c: protocol8192::Config = serde_json::from_value(config("quality")).unwrap();
        c.prefix_rows = Some(protocol::Binding {
            path: "/tmp/A".into(),
            sha256: "a".repeat(64),
        });
        assert!(c.validate().is_err());
        let old = json!({"format":"toklet-qwen3-parallel-reference-phase-v1", "phase":"quality",
            "reference_schedule":"parallel_outputs_experimental", "input":{"path":"/tmp/oldQ","sha256":"a".repeat(64)}});
        let c: protocol::Config = serde_json::from_value(old.clone()).unwrap();
        assert!(c.validate().is_err()); // Original4K still requires complete oldQ history.
        assert!(serde_json::from_value::<protocol8192::Config>(old).is_err());
    }
    #[test]
    fn prefix_receipt_requires_historical_A_and_exact_destination_parent() {
        let retained = "d".repeat(64);
        let destination = "e".repeat(64);
        let prefix = protocol8192::SOURCE_A_SHA;
        let value = json!({"format":"toklet-qwen3-context8192-prefix-rows-v1", "phase":"b128",
            "source_input_sha256":prefix,
            "source_result_sha256":"2961d4baef9c1fc488ea2497efda705ea5b8e96f3d4a021bebc1f889235cc9a6",
            "source_journal_sha256":"3f53bed75c72f631ed04dfb38584fb95b610adab8b14f93f5c0c0b404c86cd66",
            "retained_parallel_result_sha256":retained,"destination_input_sha256":destination,
            "destination_prefix_sha256":prefix,"rows":vec!["c".repeat(64);4096]});
        let mut rows: protocol8192::PrefixRows = serde_json::from_value(value).unwrap();
        rows.validate(Phase::B128, &destination, prefix, &retained)
            .unwrap();
        assert!(
            rows.validate(Phase::A32, &destination, prefix, &retained)
                .is_err()
        );
        assert!(
            rows.validate(Phase::Quality, &destination, prefix, &retained)
                .is_err()
        );
        assert!(
            rows.validate(Phase::B128, &"f".repeat(64), prefix, &retained)
                .is_err()
        );
        assert!(
            rows.validate(Phase::B128, &destination, prefix, &"f".repeat(64))
                .is_err()
        );
        assert!(
            rows.validate(Phase::B128, &destination, &"f".repeat(64), &retained)
                .is_err()
        );
        rows.source_input_sha256 =
            "acef763a2ca5fa7fd19bc6bf05098689285b84bfded5f2ff6748584ae3a0dd9c".into();
        assert!(
            rows.validate(Phase::B128, &destination, prefix, &retained)
                .is_err()
        ); // OldB4096 is forbidden.
        rows.source_input_sha256 = prefix.into();
        rows.rows.pop();
        assert!(
            rows.validate(Phase::B128, &destination, prefix, &retained)
                .is_err()
        );
        rows.rows.push("C".repeat(64));
        assert!(
            rows.validate(Phase::B128, &destination, prefix, &retained)
                .is_err()
        );
    }
}
