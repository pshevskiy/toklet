//! Explicit full-coordinate hybrid diagnostic, with bounded optional streaming.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod native {
    use serde::Serialize;
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::{
        collections::BTreeMap,
        fs::{self, File, Metadata, OpenOptions},
        io::{self, BufWriter, Read, Seek, SeekFrom, Write},
        os::unix::fs::{MetadataExt, OpenOptionsExt},
        path::{Path, PathBuf},
        sync::atomic::AtomicBool,
        time::Instant,
    };
    use toklet::{
        metal::gemma_affine::{CONTEXT_ALLOWANCE_BYTES, Context},
        mlx::qwen35::{ModalityPolicy, two_b},
        model::qwen35::{self as graph, MetalModel, MetalSession, Observation},
    };
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    const PROTOCOL: &[u8] = include_bytes!("../tools/reference/qwen35_2b_graph_protocol.py");
    const LIMIT: usize = 6 * 1024 * 1024 * 1024;
    const HELPER: usize = 32 * 1024 * 1024;
    const APPLICATION: usize = 128 * 1024 * 1024;
    const FRAME: usize = 2 * 1024 * 1024;
    const TOKENS: [u32; 17] = [1, 7, 13, 3, 9, 5, 11, 2, 17, 4, 6, 10, 19, 8, 21, 23, 29];
    fn need(ok: bool, why: &str) -> Result<()> {
        if ok { Ok(()) } else { Err(why.into()) }
    }
    fn hash(raw: &[u8]) -> String {
        format!("{:x}", Sha256::digest(raw))
    }
    fn hex(raw: &[u8]) -> String {
        raw.iter().map(|v| format!("{v:02x}")).collect()
    }
    fn digest(text: &str) -> Result<[u8; 32]> {
        need(
            text.len() == 64
                && text
                    .bytes()
                    .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v)),
            "lowercase SHA256 required",
        )?;
        let mut out = [0; 32];
        for (i, v) in out.iter_mut().enumerate() {
            *v = u8::from_str_radix(&text[2 * i..2 * i + 2], 16)?;
        }
        Ok(out)
    }
    #[derive(Clone, Serialize)]
    struct Case {
        name: String,
        tokens: Vec<u32>,
        position_offset: usize,
    }
    fn cases(phase: &str) -> Result<Vec<Case>> {
        let c = |name: &str, tokens: Vec<u32>, position_offset| Case {
            name: name.into(),
            tokens,
            position_offset,
        };
        Ok(match phase {
            "initial_short" => vec![
                c("single0", vec![1], 0),
                c("prefix4", vec![1, 7, 13, 3], 0),
                c("future_changed4", vec![1, 7, 13, 5], 0),
                c("offset3", vec![2, 11, 9], 37),
            ],
            "context17" => {
                let mut changed = TOKENS.to_vec();
                changed[16] = 31;
                vec![
                    c("context17", TOKENS.to_vec(), 0),
                    c("offset17", TOKENS.to_vec(), 37),
                    c("future_changed17", changed, 0),
                ]
            }
            "fixed255" | "context512" => {
                let n = if phase == "fixed255" { 255 } else { 512 };
                vec![c(phase, (0..n).map(|i| TOKENS[i % 17]).collect(), 0)]
            }
            _ => return Err("unsupported phase; capacity remains512".into()),
        })
    }
    struct Args {
        model: PathBuf,
        manifest: PathBuf,
        manifest_sha: [u8; 32],
        protocol: PathBuf,
        protocol_sha: [u8; 32],
        phase: String,
        stream: bool,
        output: PathBuf,
    }
    fn args() -> Result<Args> {
        let mut map = BTreeMap::new();
        let mut it = std::env::args_os().skip(1);
        let mut stream = false;
        while let Some(key) = it.next() {
            let key = key.into_string().map_err(|_| "non-UTF8 flag")?;
            if key == "--stream" {
                need(!stream, "duplicate stream flag")?;
                stream = true;
                continue;
            }
            need(
                [
                    "--model",
                    "--artifact-manifest",
                    "--artifact-manifest-sha256",
                    "--protocol",
                    "--protocol-sha256",
                    "--phase",
                    "--fresh-output",
                ]
                .contains(&key.as_str()),
                "unknown flag",
            )?;
            let value = it.next().ok_or("missing flag value")?;
            need(map.insert(key, value).is_none(), "duplicate flag")?;
        }
        let mut take = |key: &str| map.remove(key).ok_or_else(|| format!("missing {key}"));
        Ok(Args {
            model: take("--model")?.into(),
            manifest: take("--artifact-manifest")?.into(),
            manifest_sha: digest(
                &take("--artifact-manifest-sha256")?
                    .into_string()
                    .map_err(|_| "non-UTF8 digest")?,
            )?,
            protocol: take("--protocol")?.into(),
            protocol_sha: digest(
                &take("--protocol-sha256")?
                    .into_string()
                    .map_err(|_| "non-UTF8 digest")?,
            )?,
            phase: take("--phase")?
                .into_string()
                .map_err(|_| "non-UTF8 phase")?,
            stream,
            output: take("--fresh-output")?.into(),
        })
    }
    type Identity = (u64, u64, u64, i64, i64, i64, i64, u32, u64);
    fn identity(m: &Metadata) -> Identity {
        (
            m.dev(),
            m.ino(),
            m.len(),
            m.mtime(),
            m.mtime_nsec(),
            m.ctime(),
            m.ctime_nsec(),
            m.mode(),
            m.nlink(),
        )
    }
    struct Held {
        path: PathBuf,
        file: File,
        before: Metadata,
    }
    impl Held {
        fn open(path: &Path, max: u64) -> Result<Self> {
            let before = fs::symlink_metadata(path)?;
            need(
                before.is_file() && before.len() > 0 && before.len() <= max,
                "bounded regular input required",
            )?;
            let file = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                .open(path)?;
            let held = Self {
                path: path.into(),
                file,
                before,
            };
            held.verify()?;
            Ok(held)
        }
        fn verify(&self) -> Result<()> {
            need(
                identity(&fs::symlink_metadata(&self.path)?) == identity(&self.before)
                    && identity(&self.file.metadata()?) == identity(&self.before),
                "input descriptor/path changed",
            )
        }
        fn read(&mut self, max: usize) -> Result<Vec<u8>> {
            self.verify()?;
            self.file.seek(SeekFrom::Start(0))?;
            let mut raw = Vec::new();
            Read::by_ref(&mut self.file)
                .take((max + 1) as u64)
                .read_to_end(&mut raw)?;
            self.verify()?;
            need(
                raw.len() as u64 == self.before.len() && raw.len() <= max,
                "input length changed",
            )?;
            Ok(raw)
        }
        fn sha(&mut self) -> Result<String> {
            self.verify()?;
            self.file.seek(SeekFrom::Start(0))?;
            let mut h = Sha256::new();
            let mut raw = [0; 65536];
            let mut n = 0;
            loop {
                let got = self.file.read(&mut raw)?;
                if got == 0 {
                    break;
                }
                n += got as u64;
                need(n <= self.before.len(), "input grew")?;
                h.update(&raw[..got]);
            }
            need(n == self.before.len(), "input shrank")?;
            self.verify()?;
            Ok(format!("{:x}", h.finalize()))
        }
    }
    fn shape(o: &Observation<'_>) -> Result<Vec<usize>> {
        let shape = match o.stage {
            "embedding" | "final_norm" | "input_norm" | "operator_output" | "operator_residual"
            | "post_norm" | "mlp_output" | "layer_output" => vec![2048],
            "logits" => vec![248320],
            "mlp_gate" | "mlp_up" | "mlp_activated" => vec![6144],
            "delta_qkv" | "delta_convolved" => vec![6144],
            "delta_z" | "delta_v" | "delta_output" | "delta_norm" | "delta_gated" => vec![16, 128],
            "delta_q" | "delta_k" => vec![16, 128],
            "delta_a" | "delta_b" | "delta_decay" | "delta_beta" => vec![16],
            "delta_state" => vec![16, 128, 128],
            "delta_history" => vec![3, 6144],
            "attention_q_gate" => vec![8, 2, 256],
            "attention_q" | "attention_q_norm" | "attention_q_rot" | "attention_context"
            | "attention_gated" => vec![8, 256],
            "attention_k" | "attention_v" | "attention_k_norm" | "attention_k_rot" => vec![2, 256],
            "attention_k_cache" | "attention_v_cache" => vec![o.token + 1, 256],
            "attention_scores" | "attention_probabilities" => vec![o.token + 1],
            _ => return Err("unrecognized graph stage".into()),
        };
        need(
            shape.iter().product::<usize>() == o.values.len(),
            "stage shape/count differs",
        )?;
        Ok(shape)
    }
    struct Output {
        rows: Option<BufWriter<File>>,
        stream: Option<BufWriter<io::Stdout>>,
        journal: BufWriter<File>,
        row_hash: Sha256,
        journal_hash: Sha256,
        row_bytes: u64,
        journal_bytes: u64,
        arrays: usize,
        raw: Vec<u8>,
    }
    impl Output {
        fn new(path: &Path, stream: bool) -> Result<Self> {
            Ok(Self {
                rows: if stream {
                    None
                } else {
                    Some(BufWriter::new(File::create_new(path.join("rows.bin"))?))
                },
                stream: stream.then(|| BufWriter::new(io::stdout())),
                journal: BufWriter::new(File::create_new(path.join("arrays.jsonl"))?),
                row_hash: Sha256::new(),
                journal_hash: Sha256::new(),
                row_bytes: 0,
                journal_bytes: 0,
                arrays: 0,
                raw: Vec::with_capacity(FRAME),
            })
        }
        fn emit(&mut self, case: &Case, o: &Observation<'_>) -> Result<()> {
            need(
                o.values.len().checked_mul(4).is_some_and(|n| n <= FRAME),
                "bounded frame exceeded",
            )?;
            self.raw.clear();
            for v in o.values {
                need(v.is_finite(), "nonfinite native value")?;
                self.raw.extend_from_slice(&v.to_le_bytes());
            }
            let row = json!({"profile":"native_fp32","case":case.name,"token":o.token,"position":o.position,"layer":o.layer,"stage":o.stage,"head":o.head,"shape":shape(o)?,"dtype":"F32","count":o.values.len(),"offset":self.row_bytes,"bytes":self.raw.len(),"sha256":hash(&self.raw)});
            let encoded = serde_json::to_vec(&row)?;
            need(encoded.len() <= 2048, "frame descriptor exceeded bound")?;
            if let Some(rows) = &mut self.rows {
                rows.write_all(&self.raw)?;
            }
            if let Some(out) = &mut self.stream {
                out.write_all(&(encoded.len() as u32).to_le_bytes())?;
                out.write_all(&encoded)?;
                out.write_all(&self.raw)?;
                out.flush()?;
            }
            self.journal.write_all(&encoded)?;
            self.journal.write_all(b"\n")?;
            self.row_hash.update(&self.raw);
            self.journal_hash.update(&encoded);
            self.journal_hash.update(b"\n");
            self.row_bytes += self.raw.len() as u64;
            self.journal_bytes += encoded.len() as u64 + 1;
            self.arrays += 1;
            need(
                self.journal_bytes <= 1024 * 1024 * 1024,
                "journal limit exceeded",
            )?;
            Ok(())
        }
        fn finish(&mut self) -> Result<()> {
            self.journal.flush()?;
            self.journal.get_ref().sync_all()?;
            if let Some(rows) = &mut self.rows {
                rows.flush()?;
                rows.get_ref().sync_all()?;
            }
            Ok(())
        }
    }
    #[derive(Serialize, PartialEq, Debug)]
    struct Trace {
        arrays: usize,
        values: usize,
        trace_sha256: String,
        state_sha256: String,
        logits_sha256: String,
        history: Vec<u32>,
    }
    fn execute(
        session: &mut MetalSession<'_, '_, '_>,
        case: &Case,
        chunk: usize,
        mut output: Option<&mut Output>,
    ) -> Result<Trace> {
        let mut trace = Sha256::new();
        let mut state = Sha256::new();
        let mut logits = Sha256::new();
        let mut arrays = 0;
        let mut values = 0;
        for input in case.tokens.chunks(chunk) {
            session.forward_observed(input, &AtomicBool::new(false), &mut |o| {
                if o.stage == "before_commit" {
                    return Ok(());
                }
                if let Some(writer) = &mut output {
                    writer
                        .emit(case, &o)
                        .map_err(|_| graph::Error::Observation("capture output or shape failed"))?;
                }
                for value in o.values {
                    if !value.is_finite() {
                        return Err(graph::Error::Observation("nonfinite control value"));
                    }
                    let raw = value.to_le_bytes();
                    trace.update(raw);
                    if o.token + 1 == case.tokens.len() {
                        if [
                            "delta_state",
                            "delta_history",
                            "attention_k_cache",
                            "attention_v_cache",
                        ]
                        .contains(&o.stage)
                        {
                            state.update(raw);
                        }
                        if o.stage == "logits" {
                            logits.update(raw);
                        }
                    }
                }
                arrays += 1;
                values += o.values.len();
                Ok(())
            })?;
        }
        need(
            arrays == 669 * case.tokens.len() && session.history() == case.tokens,
            "incomplete native trace/history",
        )?;
        Ok(Trace {
            arrays,
            values,
            trace_sha256: format!("{:x}", trace.finalize()),
            state_sha256: format!("{:x}", state.finalize()),
            logits_sha256: format!("{:x}", logits.finalize()),
            history: session.history().to_vec(),
        })
    }
    fn pins_json(p: &two_b::ArtifactPins) -> Value {
        json!({"config_bytes":p.config_bytes,"config_sha256":hex(&p.config_sha256),"header_bytes":p.header_bytes,"header_sha256":hex(&p.header_sha256),"container_bytes":p.container_bytes,"container_sha256":hex(&p.container_sha256)})
    }
    fn geometry() -> Value {
        json!({"hidden":2048,"intermediate":6144,"vocab":248320,"layers":24,"query_heads":8,"kv_heads":2,"head_dim":256,"rotated_dims":64,"key_heads":16,"value_heads":16,"key_dim":128,"value_dim":128,"max_positions":262144,"rope_theta":10000000.0,"rms_epsilon":1e-6})
    }
    pub fn run() -> Result<()> {
        let a = args()?;
        let cases = cases(&a.phase)?;
        need(
            a.stream || a.phase == "initial_short",
            "deeper phases require bounded streaming",
        )?;
        let mut protocol = Held::open(&a.protocol, 1024 * 1024)?;
        let protocol_raw = protocol.read(1024 * 1024)?;
        need(
            protocol_raw == PROTOCOL && Sha256::digest(&protocol_raw)[..] == a.protocol_sha,
            "compiled frozen protocol differs",
        )?;
        let mut manifest = Held::open(&a.manifest, 64 * 1024)?;
        let manifest_raw = manifest.read(64 * 1024)?;
        let pins = two_b::artifact::pins(&manifest_raw, a.manifest_sha)?;
        let mut config = Held::open(&a.model.join("config.json"), pins.config_bytes as u64)?;
        let config_raw = config.read(pins.config_bytes)?;
        let mut source = Held::open(&a.model.join("model.safetensors"), pins.container_bytes)?;
        need(
            source.before.len() == pins.container_bytes,
            "source length differs",
        )?;
        fs::create_dir(&a.output)?;
        let budget = LIMIT - APPLICATION - HELPER;
        let total = Instant::now();
        let start = Instant::now();
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES)?;
        let context_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let cpu = two_b::payload::load(
            &config_raw,
            &mut source.file,
            two_b::payload::LoadOptions {
                memory_limit_bytes: budget - CONTEXT_ALLOWANCE_BYTES,
                modality: ModalityPolicy::TextOnly,
            },
            &pins,
        )?;
        let load_seconds = start.elapsed().as_secs_f64();
        source.verify()?;
        let start = Instant::now();
        let store = cpu.into_metal(&context, budget)?;
        let upload_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let model = MetalModel::from_two_b_metal_store(&store)?;
        let binding_seconds = start.elapsed().as_secs_f64();
        let capacity = if cases.iter().any(|c| c.tokens.len() > 128) {
            512
        } else {
            128
        };
        let mut output = Output::new(&a.output, a.stream)?;
        let mut controls = BufWriter::new(File::create_new(a.output.join("controls.jsonl"))?);
        let mut control_hash = Sha256::new();
        let mut control_bytes = 0;
        let mut values = 0;
        let mut reports = Vec::new();
        for case in &cases {
            let options = graph::Options {
                capacity,
                chunk: 32,
                position_offset: case.position_offset,
                memory_limit_bytes: budget,
            };
            let start = Instant::now();
            let mut session = model.session(&context, options)?;
            let session_initialization_seconds = start.elapsed().as_secs_f64();
            let canonical = execute(&mut session, case, 32, Some(&mut output))?;
            values += canonical.values;
            reports.push(json!({"case":case.name,"memory":session.memory(),"work":session.work(),
                "work_scope":"last numerical chunk only","session_initialization_seconds":session_initialization_seconds}));
            for (schedule, chunk) in [("chunk1", 1), ("chunk3", 3), ("reset", 32)] {
                session.reset();
                if schedule == "reset" {
                    session.forward(&case.tokens[..1])?;
                    session.reset();
                }
                let replay = execute(&mut session, case, chunk, None)?;
                need(replay == canonical, "schedule/reset trace or state differs")?;
                let mut record = serde_json::to_value(replay)?;
                record["case"] = json!(case.name);
                record["schedule"] = json!(schedule);
                let raw = serde_json::to_vec(&record)?;
                controls.write_all(&raw)?;
                controls.write_all(b"\n")?;
                control_hash.update(&raw);
                control_hash.update(b"\n");
                control_bytes += raw.len() + 1;
            }
        }
        output.finish()?;
        controls.flush()?;
        controls.get_ref().sync_all()?;
        need(
            source.sha()? == hex(&pins.container_sha256),
            "source endpoint SHA differs",
        )?;
        need(
            config.sha()? == hex(&pins.config_sha256)
                && manifest.sha()? == hex(&a.manifest_sha)
                && protocol.sha()? == hex(&a.protocol_sha),
            "metadata endpoint SHA differs",
        )?;
        let result = json!({"format":"toklet-qwen35-2b-graph-capture-v1","role":"native","phase":a.phase,"capture_mode":if a.stream{"stream"}else{"raw"},"protocol_sha256":hex(&a.protocol_sha),"source_manifest_sha256":hex(&a.manifest_sha),"artifact_pins":pins_json(&pins),"cases":cases,"geometry":geometry(),"gate":{"atol":1e-5,"rtol":1e-4},"profiles":["native_fp32"],"arrays":output.arrays,"values_per_profile":values,"rows_bytes":output.row_bytes,"rows_sha256":format!("{:x}",output.row_hash.clone().finalize()),"journal_bytes":output.journal_bytes,"journal_sha256":format!("{:x}",output.journal_hash.clone().finalize()),"controls":{"bytes":control_bytes,"sha256":format!("{:x}",control_hash.finalize())},"completed":true,"qualifications":{"full_model_graph":false,"representative_context":false,"quality":false,"performance":false,"adoption":false}});
        let raw = serde_json::to_vec(&result)?;
        need(raw.len() <= 65536, "trailer exceeded bound")?;
        let mut file = File::create_new(a.output.join("result.json"))?;
        file.write_all(&raw)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        let runtime = json!({"phase":a.phase,"capacity":capacity,"chunk":32,"application_allowance_bytes":APPLICATION,"helper_allowance_bytes":HELPER,"context_initialization_seconds":context_seconds,"authenticated_load_seconds":load_seconds,"upload_seconds":upload_seconds,"binding_seconds":binding_seconds,"whole_diagnostic_seconds":total.elapsed().as_secs_f64(),"upload_report":store.report(),"cases":reports,"timing_scope":"diagnostic capture/replay, not a controlled performance cohort","capture_source_sha256":hash(include_bytes!("qwen35_2b_graph_capture.rs")),"compilation_seconds":Value::Null});
        serde_json::to_writer_pretty(File::create_new(a.output.join("runtime.json"))?, &runtime)?;
        if let Some(out) = &mut output.stream {
            out.write_all(&0_u32.to_le_bytes())?;
            out.write_all(&(raw.len() as u32).to_le_bytes())?;
            out.write_all(&raw)?;
            out.flush()?;
        }
        eprintln!("Qwen3.5-2B diagnostic complete: {} arrays", output.arrays);
        Ok(())
    }
}
fn main() -> std::process::ExitCode {
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    {
        if let Err(e) = native::run() {
            eprintln!("{e}");
            return std::process::ExitCode::FAILURE;
        }
        std::process::ExitCode::SUCCESS
    }
    #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
    {
        eprintln!("Requires the metal feature on Apple Silicon macOS");
        std::process::ExitCode::FAILURE
    }
}
