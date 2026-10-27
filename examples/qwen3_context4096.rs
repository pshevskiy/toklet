//! Frozen finite-context experiment. No production admission or arithmetic changes.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[allow(dead_code)]
#[path = "support/loss.rs"]
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
            if self.format != "toklet-qwen3-context4096-native-phase-v1"
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

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use super::{loss, protocol::*};
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
                s.set_tokens(&vec![0; CAPACITY + 1])
            };
            let after = state(s);
            emit(
                out,
                json!({"event":"capacity_rejection","operation":operation,"requested_history":4097,
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
    pub fn run() -> Result<()> {
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
        let config: Config = serde_json::from_slice(&body)?;
        config.validate()?;
        let config_sha = owner.binding.sha256.clone();
        let mut owners = vec![owner];
        let phase = config.phase;
        let input_sha = config.input.sha256.clone();
        let (owner, body) = Owner::read(config.input, CAPACITY * 4)?;
        owners.push(owner);
        let input = ids(&body)?;
        validate_ids(&input, CAPACITY)?;
        let expected = if let Some(binding) = config.expected_rows {
            let (owner, body) = Owner::read(binding, 512 * 1024)?;
            owners.push(owner);
            let rows: Vec<String> = serde_json::from_slice(&body)?;
            require(
                rows.len() == phase.expected_rows() && rows.iter().all(|s| digest(s)),
                "expected row coverage differs",
            )?;
            rows
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
        };
        emit(
            &mut out,
            json!({"event":"start","format":"toklet-qwen3-context4096-native-events-v1","phase":phase,
            "config_sha256":config_sha,"input_sha256":input_sha,"input_tokens":input.len(),
            "expected_rows":expected.len(),"profile":"MlxAffine/Compensated/Reference/FP16",
            "diagnostic_only":true,"external_source_identity_and_resource_endpoints_required":true}),
        )?;
        let budget = weights::INITIAL_MEMORY_LIMIT_BYTES - APP_BYTES;
        let start = Instant::now();
        let options = weights::LoadOptions {
            context_limit: phase.rows(),
            memory_limit_bytes: budget,
        };
        let model = if phase == Phase::Anchor {
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
        let mut s = metal::Session::with_attention_mode(
            &model,
            session::Options {
                capacity: phase.rows(),
                prefill_chunk_size: phase.chunk(),
                position_offset: 0,
                memory_limit_bytes: budget,
            },
            ProjectionMode::MlxAffine,
            NormalizationMode::Compensated,
            readback,
            AttentionMode::Reference,
        )?;
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
            "capacity":phase.rows(),"chunk":phase.chunk(),"readback":readback,"diagnostics":phase.diagnostic(),
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
            let mut row_hashes = Vec::with_capacity(phase.rows());
            s.append(&input[..phase.rows()])?;
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
                    let matches = expected_hash.is_none_or(|h| *h == hash);
                    emit(
                        &mut out,
                        json!({"event":"row","position":position,"sha256":hash,"finite":finite,
                        "expected_sha256":expected_hash,"matches":matches,"evaluated":s.evaluated_len(),"pending":s.pending_len()}),
                    )?;
                    if !finite || !matches {
                        emit(
                            &mut out,
                            json!({"event":"failing_row","position":position,"expected_sha256":expected_hash,
                            "actual_fp32_bits":row.iter().map(|v|v.to_bits()).collect::<Vec<_>>()}),
                        )?;
                    }
                    require(finite && matches, "nonfinite or unequal row")?;
                    observed += 1;
                    row_hashes.push(hash.clone());
                    if phase == Phase::Quality
                        && let Some(index) = target(position)
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
                observed == phase.rows()
                    && s.evaluated_len() == phase.rows()
                    && s.history() == &input[..phase.rows()],
                "coverage differs",
            )?;
            if phase == Phase::Quality {
                emit(
                    &mut out,
                    json!({"event":"quality_done","targets":targets,"first_target_index":3072,"last_target_index":4095,
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
                s.append(&input[..4095])?;
                evaluate(&mut s, &mut out, "prescribed_prefix")?;
                committed += 4095;
                final_row(&s, &row_hashes[4094], &mut out)?;
                let prefix = state(&s);
                s.append(&input[4095..])?;
                emit(
                    &mut out,
                    json!({"event":"prescribed_pending","state":state(&s)}),
                )?;
                require(
                    s.evaluated_len() == 4095 && s.pending_len() == 1 && s.history() == input,
                    "pending prescribed token differs",
                )?;
                reject(&mut s, &mut out)?;
                s.discard_pending();
                emit(
                    &mut out,
                    json!({"event":"discard","before":prefix,"after":state(&s)}),
                )?;
                require(state(&s) == prefix, "discard changed prefix")?;
                s.append(&input[4095..])?;
                committed += call(&mut s, &mut out, "prescribed_flush")?;
                final_row(&s, &row_hashes[4095], &mut out)?;
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
            s.append(&input[..4064])?;
            evaluate(&mut s, &mut out, "ordinary_prefill")?;
            committed += 4064;
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
                s.history().len() == 4096 && s.evaluated_len() == 4095 && s.pending_len() == 1,
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
            committed += 4096;
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
            committed == phase.committed_tokens()
                && s.evaluated_len() == phase.rows()
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
fn main() {
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    if let Err(error) = apple::run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
    #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
    {
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
}
