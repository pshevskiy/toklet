mod chat;
mod inspect;
mod plan;
mod qwen35_2b;
use chat::{Assets, OutputDecoder, invalid};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    env,
    ffi::OsString,
    io::{self, BufRead, Read, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};
use toklet::{
    cpu::ProjectionKernel,
    generation::{self, Input, Request, StopReason},
    session::{InferenceSession, Options, Session},
    weights::{self, LoadOptions},
};
const APPLICATION_ALLOWANCE: usize = 128 * 1024 * 1024;
static CANCELLED: AtomicBool = AtomicBool::new(false);
// The swap is the active/idle boundary. A signal raised during or after
// publication belongs to idle state and must survive until the next loop.
fn publish_terminal<T>(flag: &AtomicBool, publish: impl FnOnce(bool) -> T) -> T {
    let active_cancelled = flag.swap(false, Ordering::Relaxed);
    publish(active_cancelled)
}
const HELP: &str = "Toklet — a small Rust inference engine for Apple Silicon

Usage: toklet [OPTIONS]
       toklet inspect --help
       toklet plan --help

  --model PATH              Local model directory
  --model-profile PROFILE   qwen (default), lfm3-bf16, ministral3-diagnostic, qwen35-2b-diagnostic
  --artifact-manifest PATH  Explicit2B child manifest (diagnostic profile only)
  --artifact-manifest-sha256 SHA  Caller-selected manifest SHA256 (64 lower hex)
  --prompt TEXT             Chat-formatted user prompt
  --messages PATH           JSON array of chat messages
  --tools PATH              JSON tool schemas
  --raw                     Encode prompt without chat formatting
  --tokens IDS              Comma-separated raw token IDs
  --token-file PATH         Little-endian u32 token IDs
  --max-new-tokens N         Sampled output limit, including stop token (32)
  --context-capacity N      Prompt plus output limit (2048; Metal up to32768)
  --prefill-chunk N          Prefill cancellation boundary (128)
  --backend cpu|metal       CPU (default), or opt-in Metal feature
  --cpu-projection MODE     scalar (default) or opt-in tiled projections
  --metal-projection MODE   original (default), half-compensated, q4-v2-compensated-tiled, mlx-affine
  --metal-attention MODE    reference (default) or experimental online
  --q4-weights PATH         Explicit Q4 v2 packed artifact for the v2 mode
  --stop-token ID           Additional stop ID; repeatable
  --no-eos                  Ignore configured EOS IDs
  --json                    JSON token events and statistics
  --stats                   Statistics to stderr with text output
  --interactive             Persistent JSON-lines requests on stdin
  -h, --help                Print help
  -V, --version             Print version

Generation is greedy without penalties. Text requires local tokenizer assets.
Ministral3 diagnostic uses Metal mlx-affine/reference, token IDs and JSON only;
its default capacity is 128 and its maximum capacity is 512.
Qwen3.5-2B diagnostic uses GPU projections with CPU FP32 state, token IDs and
JSON only; default capacity128/chunk32, maximum capacity512/chunk32.
Interactive actions: generate, append (raw text or IDs), continue, reset.
Ctrl-C cancels between backend chunks/steps; idle interactive Ctrl-C exits.";
#[derive(Clone, Copy, PartialEq, Eq)]
enum ModelProfile {
    Qwen,
    Lfm3Bf16,
    Ministral3Diagnostic,
    Qwen35TwoBDiagnostic,
}
struct Args {
    model: PathBuf,
    model_profile: ModelProfile,
    prompt: Option<String>,
    messages: Option<PathBuf>,
    tools: Option<PathBuf>,
    tokens: Option<String>,
    token_file: Option<PathBuf>,
    raw: bool,
    json: bool,
    stats: bool,
    interactive: bool,
    no_eos: bool,
    stop: Vec<u32>,
    max: usize,
    capacity: usize,
    chunk: usize,
    projection: ProjectionKernel,
    backend: &'static str,
    metal_projection: &'static str,
    metal_attention: &'static str,
    q4_weights: Option<PathBuf>,
    artifact_manifest: Option<PathBuf>,
    artifact_manifest_sha256: Option<[u8; 32]>,
}
fn utf8(v: OsString) -> io::Result<String> {
    v.into_string()
        .map_err(|_| invalid("argument must be UTF-8"))
}
fn number<T: std::str::FromStr>(v: OsString) -> io::Result<T> {
    utf8(v)?
        .parse()
        .map_err(|_| invalid("invalid unsigned integer"))
}
fn parse(values: Vec<OsString>) -> io::Result<Args> {
    let mut a = Args {
        model: PathBuf::new(),
        model_profile: ModelProfile::Qwen,
        prompt: None,
        messages: None,
        tools: None,
        tokens: None,
        token_file: None,
        raw: false,
        json: false,
        stats: false,
        interactive: false,
        no_eos: false,
        stop: Vec::new(),
        max: 32,
        capacity: 2048,
        chunk: 128,
        projection: ProjectionKernel::Scalar,
        backend: "cpu",
        metal_projection: "original",
        metal_attention: "reference",
        q4_weights: None,
        artifact_manifest: None,
        artifact_manifest_sha256: None,
    };
    let mut seen = HashSet::new();
    let mut values = values.into_iter().peekable();
    if values.peek().is_some_and(|v| v == "generate") {
        values.next();
    }
    while let Some(flag) = values.next() {
        let flag = utf8(flag)?;
        if flag != "--stop-token" && !seen.insert(flag.clone()) {
            return Err(invalid(format!("duplicate {flag}")));
        }
        let mut value = || {
            values
                .next()
                .ok_or_else(|| invalid(format!("missing value for {flag}")))
        };
        match flag.as_str() {
            "--metal-projection" => {
                a.metal_projection = match utf8(value()?)?.as_str() {
                    "original" => "original",
                    "mlx-affine" => "mlx-affine",
                    "half-compensated" => "half-compensated",
                    "q4-v2-compensated-tiled" => "q4-v2-compensated-tiled",
                    _ => {
                        return Err(invalid(
                            "--metal-projection must be original, half-compensated, q4-v2-compensated-tiled or mlx-affine",
                        ));
                    }
                };
            }
            "--metal-attention" => {
                a.metal_attention = match utf8(value()?)?.as_str() {
                    "reference" => "reference",
                    "online" => "online",
                    _ => return Err(invalid("--metal-attention must be reference or online")),
                };
            }
            "--q4-weights" => a.q4_weights = Some(value()?.into()),
            "--artifact-manifest" => a.artifact_manifest = Some(value()?.into()),
            "--artifact-manifest-sha256" => {
                a.artifact_manifest_sha256 = Some(qwen35_2b::sha256(&utf8(value()?)?)?);
            }
            "--model" => a.model = value()?.into(),
            "--model-profile" => {
                a.model_profile = match utf8(value()?)?.as_str() {
                    "qwen" => ModelProfile::Qwen,
                    "lfm3-bf16" => ModelProfile::Lfm3Bf16,
                    "ministral3-diagnostic" => ModelProfile::Ministral3Diagnostic,
                    "qwen35-2b-diagnostic" => ModelProfile::Qwen35TwoBDiagnostic,
                    _ => {
                        return Err(invalid(
                            "model profile must be qwen, lfm3-bf16, ministral3-diagnostic or qwen35-2b-diagnostic",
                        ));
                    }
                };
            }
            "--prompt" => a.prompt = Some(utf8(value()?)?),
            "--messages" => a.messages = Some(value()?.into()),
            "--tools" => a.tools = Some(value()?.into()),
            "--tokens" => a.tokens = Some(utf8(value()?)?),
            "--token-file" => a.token_file = Some(value()?.into()),
            "--max-new-tokens" => a.max = number(value()?)?,
            "--context-capacity" => a.capacity = number(value()?)?,
            "--prefill-chunk" => a.chunk = number(value()?)?,
            "--stop-token" => a.stop.push(number(value()?)?),
            "--cpu-projection" => {
                a.projection = match utf8(value()?)?.as_str() {
                    "scalar" => ProjectionKernel::Scalar,
                    "tiled" => ProjectionKernel::Tiled,
                    _ => return Err(invalid("--cpu-projection must be scalar or tiled")),
                };
            }
            "--backend" => {
                a.backend = match utf8(value()?)?.as_str() {
                    "cpu" => "cpu",
                    "metal" => "metal",
                    _ => return Err(invalid("backend must be cpu or metal")),
                };
                #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
                if a.backend == "metal" {
                    return Err(invalid(
                        "Metal requires the metal feature on Apple Silicon macOS",
                    ));
                }
            }
            "--raw" => a.raw = true,
            "--json" => a.json = true,
            "--stats" => a.stats = true,
            "--interactive" => a.interactive = true,
            "--no-eos" => a.no_eos = true,
            _ => return Err(invalid(format!("unexpected argument {flag}"))),
        }
    }
    if a.model_profile == ModelProfile::Lfm3Bf16 {
        if a.backend != "cpu" || seen.contains("--cpu-projection") || a.tools.is_some() {
            return Err(invalid(
                "LFM3 uses the fixed BF16 CPU graph and plain-text chat",
            ));
        }
        if !seen.contains("--prefill-chunk") {
            a.chunk = 16;
        }
    }
    if a.model_profile == ModelProfile::Ministral3Diagnostic {
        if a.backend != "metal"
            || a.metal_projection != "mlx-affine"
            || a.metal_attention != "reference"
            || a.q4_weights.is_some()
            || seen.contains("--cpu-projection")
        {
            return Err(invalid(
                "Ministral3 diagnostic requires Metal mlx-affine/reference",
            ));
        }
        if a.prompt.is_some() || a.messages.is_some() || a.tools.is_some() || a.raw {
            return Err(invalid("Diagnostic profile accepts only token IDs"));
        }
        if !a.json && !a.interactive {
            return Err(invalid("Ministral3 diagnostic requires JSON output"));
        }
        if !seen.contains("--context-capacity") {
            a.capacity = 128;
        }
        if a.capacity > 512 {
            return Err(invalid(
                "Ministral3 diagnostic supports at most 512 positions",
            ));
        }
    }
    if a.model_profile == ModelProfile::Qwen35TwoBDiagnostic {
        if a.backend != "metal"
            || a.metal_projection != "mlx-affine"
            || a.metal_attention != "reference"
            || a.q4_weights.is_some()
            || seen.contains("--cpu-projection")
        {
            return Err(invalid(
                "Qwen3.5-2B diagnostic requires Metal mlx-affine/reference",
            ));
        }
        if a.prompt.is_some() || a.messages.is_some() || a.tools.is_some() || a.raw {
            return Err(invalid("Qwen3.5-2B diagnostic accepts only token IDs"));
        }
        if !a.json && !a.interactive {
            return Err(invalid("Qwen3.5-2B diagnostic requires JSON output"));
        }
        if a.artifact_manifest.is_none() || a.artifact_manifest_sha256.is_none() {
            return Err(invalid(
                "Qwen3.5-2B diagnostic requires an artifact manifest and its SHA256",
            ));
        }
        if !seen.contains("--context-capacity") {
            a.capacity = 128;
        }
        if !seen.contains("--prefill-chunk") {
            a.chunk = 32;
        }
        if a.capacity > 512 || a.chunk > 32 {
            return Err(invalid(
                "Qwen3.5-2B diagnostic supports capacity512/chunk32 at most",
            ));
        }
    } else if a.artifact_manifest.is_some() || a.artifact_manifest_sha256.is_some() {
        return Err(invalid(
            "artifact manifest flags require qwen35-2b-diagnostic",
        ));
    }
    if a.backend == "metal" && seen.contains("--cpu-projection") {
        return Err(invalid("--cpu-projection requires cpu backend"));
    }
    if a.backend != "metal" && seen.contains("--metal-projection") {
        return Err(invalid("--metal-projection requires metal backend"));
    }
    if a.backend != "metal" && seen.contains("--metal-attention") {
        return Err(invalid("--metal-attention requires metal backend"));
    }
    if a.q4_weights.is_some() && a.backend != "metal" {
        return Err(invalid("--q4-weights requires metal backend"));
    }
    if a.q4_weights.is_some() != (a.metal_projection == "q4-v2-compensated-tiled") {
        return Err(invalid(
            "--q4-weights and --metal-projection q4-v2-compensated-tiled are required together",
        ));
    }
    if a.model.as_os_str().is_empty() {
        return Err(invalid("--model is required"));
    }
    let count = [
        a.prompt.is_some(),
        a.messages.is_some(),
        a.tokens.is_some(),
        a.token_file.is_some(),
    ]
    .into_iter()
    .filter(|&v| v)
    .count();
    if count > 1 || (count == 0 && !a.interactive) || (count != 0 && a.interactive) {
        return Err(invalid(
            "choose one prompt/messages/token input, or --interactive",
        ));
    }
    if a.raw && (a.messages.is_some() || a.tools.is_some()) {
        return Err(invalid("--raw cannot use chat messages/tools"));
    }
    if a.tools.is_some() && a.prompt.is_none() && a.messages.is_none() {
        return Err(invalid("--tools needs chat input"));
    }
    let context_limit = if a.backend == "metal" {
        weights::MAX_CONTEXT_LIMIT
    } else {
        weights::INITIAL_CONTEXT_LIMIT
    };
    if a.capacity == 0 || a.capacity > context_limit || a.max > 2048 || a.chunk == 0 {
        return Err(invalid("invalid output/context/chunk limit"));
    }
    if a.prompt
        .as_ref()
        .is_some_and(|v| v.len() > chat::MAX_TEXT_BYTES)
        || a.tokens
            .as_ref()
            .is_some_and(|v| v.len() > chat::MAX_INPUT_BYTES)
    {
        return Err(invalid("input is too large"));
    }
    if seen.contains("--prefill-chunk") && a.chunk > a.capacity {
        return Err(invalid("chunk exceeds context capacity"));
    }
    a.chunk = a.chunk.min(a.capacity);
    Ok(a)
}
#[derive(Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Action {
    #[default]
    Generate,
    Append,
    Continue,
    Reset,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UserRequest {
    #[serde(default)]
    action: Action,
    tokens: Option<Vec<u32>>,
    prompt: Option<String>,
    messages: Option<Vec<Value>>,
    tools: Option<Vec<Value>>,
    raw: Option<bool>,
    max_new_tokens: Option<usize>,
}
#[derive(Clone, Copy, Serialize)]
struct ExecutionMetadata {
    weight_dtype: &'static str,
    kv_dtype: &'static str,
    projection_mode: &'static str,
    normalization_mode: &'static str,
    readback_mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    attention_mode: Option<&'static str>,
    prefill_chunk_size: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    packed_format_version: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quantization_group_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quantizer: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    state_backend: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    state_dtype: Option<&'static str>,
}
struct Frontend {
    assets: Option<Assets>,
    model_profile: ModelProfile,
    lfm3_load: Option<weights::Lfm3LoadReport>,
    decoder: OutputDecoder,
    interactive: bool,
    stops: Vec<u32>,
    json: bool,
    stats: bool,
    raw: bool,
    max: usize,
    load_seconds: f64,
    conversion_seconds: f64,
    compilation_seconds: Option<f64>,
    tokenizer_load_seconds: f64,
    memory_admitted_bytes: usize,
    requests: usize,
    cpu_projection: &'static str,
    backend: &'static str,
    execution: ExecutionMetadata,
    external_bytes: usize,
    qwen35_2b_load: Option<Value>,
}
#[derive(Default)]
struct OutputTiming {
    detokenization_seconds: f64,
    publication_seconds: f64,
}
impl Frontend {
    fn render(&self, messages: &[Value], tools: Option<&[Value]>) -> io::Result<String> {
        match self.model_profile {
            ModelProfile::Qwen => chat::render(messages, tools.unwrap_or(&[]), true),
            ModelProfile::Lfm3Bf16 => {
                if tools.is_some() {
                    return Err(invalid("LFM3 text chat does not accept tools"));
                }
                toklet::app_chat::lfm3::render(messages)
            }
            ModelProfile::Ministral3Diagnostic | ModelProfile::Qwen35TwoBDiagnostic => {
                Err(invalid("Diagnostic profile accepts only token IDs"))
            }
        }
    }

    fn finish_output(&mut self, out: &mut impl Write) -> io::Result<OutputTiming> {
        let mut timing = OutputTiming::default();
        if let Some(assets) = &self.assets {
            let clock = Instant::now();
            let tail = self.decoder.finish(assets)?;
            timing.detokenization_seconds = clock.elapsed().as_secs_f64();
            let clock = Instant::now();
            if !tail.is_empty() {
                if self.json {
                    writeln!(out, "{}", json!({"event":"text","text":tail}))?;
                } else {
                    out.write_all(tail.as_bytes())?;
                }
            }
            out.flush()?;
            timing.publication_seconds = clock.elapsed().as_secs_f64();
        } else {
            let clock = Instant::now();
            out.flush()?;
            timing.publication_seconds = clock.elapsed().as_secs_f64();
        }
        Ok(timing)
    }

    fn execute<S: InferenceSession>(
        &mut self,
        session: &mut S,
        r: UserRequest,
        diagnostic: &impl Fn(&S) -> Value,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        validate_profile_input(self.model_profile, &r)?;
        let count = [r.tokens.is_some(), r.prompt.is_some(), r.messages.is_some()]
            .into_iter()
            .filter(|&v| v)
            .count();
        let mut out = io::stdout().lock();
        if r.action == Action::Reset {
            if count != 0 || r.tools.is_some() || r.raw.is_some() || r.max_new_tokens.is_some() {
                return Err(invalid("reset accepts no other fields").into());
            }
            self.finish_output(&mut out)?;
            session.try_reset()?;
            publish_terminal(&CANCELLED, |_| -> io::Result<()> {
                writeln!(
                    out,
                    "{}",
                    if self.model_profile == ModelProfile::Qwen35TwoBDiagnostic {
                        json!({"event":"reset","hybrid":diagnostic(session)})
                    } else {
                        json!({"event":"reset"})
                    }
                )?;
                out.flush()
            })?;
            return Ok(false);
        }
        if r.action == Action::Continue {
            if count != 0 || r.tools.is_some() || r.raw.is_some() {
                return Err(invalid("continue accepts only max_new_tokens").into());
            }
        } else if count != 1 {
            return Err(invalid("request needs exactly one tokens/prompt/messages field").into());
        }
        let clock = Instant::now();
        let mut input_formatting_seconds = 0.0;
        let mut tokenization_seconds = 0.0;
        let tokens = if let Some(tokens) = r.tokens {
            if r.tools.is_some() || r.raw.is_some() {
                return Err(invalid("token input cannot use tools/raw").into());
            }
            tokens
        } else if count > 0 {
            let assets = self
                .assets
                .as_ref()
                .ok_or_else(|| invalid("text input requires local tokenizer assets"))?;
            let formatting_clock = Instant::now();
            let raw = r.action == Action::Append || r.raw.unwrap_or(self.raw);
            let text = if let Some(prompt) = r.prompt {
                if raw {
                    if r.tools.is_some() {
                        return Err(invalid("raw append cannot use tools").into());
                    }
                    prompt
                } else {
                    self.render(
                        &[json!({"role":"user","content":prompt})],
                        r.tools.as_deref(),
                    )?
                }
            } else {
                if raw {
                    return Err(invalid("append/raw cannot use formatted messages").into());
                }
                self.render(r.messages.as_deref().unwrap(), r.tools.as_deref())?
            };
            input_formatting_seconds = formatting_clock.elapsed().as_secs_f64();
            let encoding_clock = Instant::now();
            let tokens = assets.encode(&text)?;
            tokenization_seconds = encoding_clock.elapsed().as_secs_f64();
            tokens
        } else {
            Vec::new()
        };
        let input_seconds = clock.elapsed().as_secs_f64();
        let max = r.max_new_tokens.unwrap_or(self.max);
        if max > 2048 {
            return Err(invalid("max_new_tokens exceeds 2048").into());
        }
        let input = match r.action {
            Action::Append => Input::Append(&tokens),
            Action::Continue => Input::Continue,
            _ => Input::Prompt(&tokens),
        };
        generation::validate_request(
            session,
            &Request {
                input,
                max_new_tokens: max,
                stop_tokens: &self.stops,
            },
        )?;
        // A new prompt/append starts a new response; continue retains pending
        // UTF-8. Rejected requests never change decoder state.
        if r.action != Action::Continue && !CANCELLED.load(Ordering::Relaxed) {
            self.finish_output(&mut out)?;
        }
        let request = Request {
            input: match r.action {
                Action::Append => Input::Append(&tokens),
                Action::Continue => Input::Continue,
                _ => Input::Prompt(&tokens),
            },
            max_new_tokens: max,
            stop_tokens: &self.stops,
        };
        let mut output_timing = OutputTiming::default();
        let mut report = generation::generate(session, request, &CANCELLED, |event| {
            let text = match &self.assets {
                Some(assets) => {
                    let clock = Instant::now();
                    let text = self.decoder.step(assets, event.token_id)?;
                    output_timing.detokenization_seconds += clock.elapsed().as_secs_f64();
                    text
                }
                None => String::new(),
            };
            let clock = Instant::now();
            if self.json {
                writeln!(
                    out,
                    "{}",
                    json!({"event":"token","token_id":event.token_id,"index":event.index,"elapsed_seconds":event.elapsed_seconds,"text":text})
                )?;
            } else {
                out.write_all(text.as_bytes())?;
            }
            out.flush()?;
            output_timing.publication_seconds += clock.elapsed().as_secs_f64();
            Ok(())
        })?;
        if !self.interactive || matches!(report.reason, StopReason::StopToken { .. }) {
            let tail = self.finish_output(&mut out)?;
            output_timing.detokenization_seconds += tail.detokenization_seconds;
            output_timing.publication_seconds += tail.publication_seconds;
        }
        publish_terminal(&CANCELLED, |active_cancelled| {
            if active_cancelled && report.reason == StopReason::OutputLimit {
                report.reason = StopReason::Cancelled;
            }
            let cancelled = report.reason == StopReason::Cancelled;
            let mut stats = json!({"generation":report,"backend":self.backend,"cpu_projection":self.cpu_projection,"execution":self.execution,"logical_memory":logical_memory(session.memory(),self.external_bytes),"input_preparation_seconds":input_seconds,
                "input_formatting_seconds":input_formatting_seconds,"tokenization_seconds":tokenization_seconds,
                "detokenization_seconds":output_timing.detokenization_seconds,"publication_seconds":output_timing.publication_seconds,
                "load_seconds":if self.requests==0{self.load_seconds}else{0.0},"conversion_seconds":if self.requests==0{self.conversion_seconds}else{0.0},"tokenizer_load_seconds":if self.requests==0{self.tokenizer_load_seconds}else{0.0},"memory_admitted_bytes":self.memory_admitted_bytes});
            if self.model_profile == ModelProfile::Ministral3Diagnostic {
                stats["model_profile"] = json!("ministral3-diagnostic");
            }
            if self.model_profile == ModelProfile::Qwen35TwoBDiagnostic {
                stats["model_profile"] = json!("qwen35-2b-diagnostic");
                stats["hybrid"] = diagnostic(session);
                stats["compilation_seconds"] = Value::Null;
                stats["compilation_timing_status"] =
                    json!("unavailable; context initialization includes pipeline creation");
                if self.requests == 0 {
                    stats["checkpoint_load"] = self.qwen35_2b_load.clone().unwrap_or(Value::Null);
                }
            }
            if let Some(seconds) = self.compilation_seconds {
                stats["compilation_seconds"] =
                    json!(if self.requests == 0 { seconds } else { 0.0 });
            }
            if let Some(load) = self.lfm3_load {
                stats["model_profile"] = json!("lfm3-bf16");
                stats["decode_evaluations"] = json!(report.decode_evaluations);
                if self.requests == 0 {
                    stats["checkpoint_load"] = lfm3_load_json(load);
                }
            }
            self.requests += 1;
            if self.json {
                writeln!(out, "{}", json!({"event":"done","statistics":stats}))?;
            } else {
                writeln!(out)?;
                if self.stats {
                    eprintln!("{stats}");
                }
            }
            out.flush()?;
            Ok(cancelled)
        })
    }
}
fn validate_profile_input(profile: ModelProfile, request: &UserRequest) -> io::Result<()> {
    if matches!(
        profile,
        ModelProfile::Ministral3Diagnostic | ModelProfile::Qwen35TwoBDiagnostic
    ) && (request.prompt.is_some()
        || request.messages.is_some()
        || request.tools.is_some()
        || request.raw.is_some())
    {
        return Err(invalid("Diagnostic profile accepts only token IDs"));
    }
    Ok(())
}
fn token_list(text: &str) -> io::Result<Vec<u32>> {
    if text.is_empty() {
        return Err(invalid("token list is empty"));
    }
    let tokens: Vec<u32> = text
        .split(',')
        .map(|w| w.trim().parse().map_err(|_| invalid("invalid token ID")))
        .collect::<Result<_, _>>()?;
    if tokens.len() > weights::MAX_CONTEXT_LIMIT {
        return Err(invalid("token list exceeds32768 IDs"));
    }
    Ok(tokens)
}
fn array_file(path: &std::path::Path) -> io::Result<Vec<Value>> {
    chat::json_file(path, chat::MAX_INPUT_BYTES)?
        .as_array()
        .cloned()
        .ok_or_else(|| invalid("expected a JSON array"))
}
fn run(a: Args) -> Result<u8, Box<dyn std::error::Error>> {
    ctrlc::try_set_handler(|| CANCELLED.store(true, Ordering::Relaxed))?;
    let lfm3 = a.model_profile == ModelProfile::Lfm3Bf16;
    let ministral3 = a.model_profile == ModelProfile::Ministral3Diagnostic;
    let qwen35_2b = a.model_profile == ModelProfile::Qwen35TwoBDiagnostic;
    let vocab = if lfm3 {
        toklet::model::lfm3::VOCAB
    } else if ministral3 {
        // The explicit source opener independently validates this selected
        // geometry. No generic top-level config or tokenizer is inferred.
        131_072
    } else if qwen35_2b {
        248_320
    } else {
        let config = chat::json_file(&a.model.join("config.json"), chat::MAX_INPUT_BYTES)?;
        config["vocab_size"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| invalid("invalid model vocabulary"))?
    };
    let clock = Instant::now();
    let needs_text = a.prompt.is_some() || a.messages.is_some() || !a.json && !a.interactive;
    let assets = if !ministral3
        && !qwen35_2b
        && (needs_text || a.interactive && a.model.join("tokenizer.json").exists())
    {
        Some(if lfm3 {
            let admitted_bytes = toklet::app_chat::lfm3::ALLOWANCE_BYTES;
            Assets {
                tokenizer: toklet::app_chat::lfm3::load_tokenizer(&a.model, admitted_bytes)?,
                admitted_bytes,
            }
        } else {
            Assets::load(&a.model, vocab)?
        })
    } else {
        None
    };
    let tokenizer_load_seconds = clock.elapsed().as_secs_f64();
    let external_bytes = assets
        .as_ref()
        .map_or(0, |a| a.admitted_bytes)
        .checked_add(if lfm3 {
            32 * 1024 * 1024
        } else {
            APPLICATION_ALLOWANCE
        })
        .ok_or_else(|| invalid("application admission overflow"))?;
    let budget = weights::INITIAL_MEMORY_LIMIT_BYTES
        .checked_sub(external_bytes)
        .filter(|&n| n > 0)
        .ok_or_else(|| invalid("tokenizer/application exceed memory budget"))?;
    let mut stops = if a.no_eos {
        Vec::new()
    } else if lfm3 {
        vec![toklet::app_chat::lfm3::EOS]
    } else if ministral3 {
        vec![2]
    } else if qwen35_2b {
        // Selected2B nested text_config EOS, without parent/codec stop inference.
        vec![248_044]
    } else {
        chat::stop_ids(&a.model, vocab)?
    };
    stops.extend(a.stop);
    let initial = if a.interactive {
        None
    } else {
        let tokens = if let Some(text) = &a.tokens {
            Some(token_list(text)?)
        } else if let Some(path) = &a.token_file {
            let b = chat::read_bounded(path, a.capacity * 4)?;
            if !b.len().is_multiple_of(4) {
                return Err(invalid("token file must contain complete u32 values").into());
            }
            Some(
                b.chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
            )
        } else {
            None
        };
        Some(UserRequest {
            action: Action::Generate,
            tokens,
            prompt: a.prompt,
            messages: a.messages.as_deref().map(array_file).transpose()?,
            tools: a.tools.as_deref().map(array_file).transpose()?,
            raw: None,
            max_new_tokens: None,
        })
    };
    let mut frontend = Frontend {
        assets,
        model_profile: a.model_profile,
        lfm3_load: None,
        decoder: OutputDecoder::default(),
        interactive: a.interactive,
        stops,
        json: a.json || a.interactive,
        stats: a.stats,
        raw: a.raw,
        max: a.max,
        load_seconds: 0.,
        conversion_seconds: 0.,
        compilation_seconds: None,
        tokenizer_load_seconds,
        memory_admitted_bytes: 0,
        requests: 0,
        cpu_projection: a.projection.name(),
        backend: a.backend,
        execution: ExecutionMetadata {
            weight_dtype: "FP32",
            kv_dtype: "FP32",
            projection_mode: a.projection.name(),
            normalization_mode: "OrderedFP64",
            readback_mode: "FullLogits",
            attention_mode: None,
            prefill_chunk_size: a.chunk,
            packed_format_version: None,
            quantization_group_size: None,
            quantizer: None,
            state_backend: None,
            state_dtype: None,
        },
        external_bytes,
        qwen35_2b_load: None,
    };

    if lfm3 {
        use toklet::model::lfm3::GenerationSession;
        let extra = GenerationSession::planned_extra_bytes(a.capacity)?;
        let load_budget = budget
            .checked_sub(extra)
            .ok_or_else(|| invalid("LFM3 generation owners exceed memory budget"))?;
        let started = Instant::now();
        let loaded = weights::load_lfm3(&a.model, a.capacity, a.chunk, load_budget)?;
        frontend.load_seconds = started.elapsed().as_secs_f64();
        frontend.lfm3_load = Some(loaded.report);
        let mut model = loaded.model;
        let mut session = GenerationSession::new(model.session()?, budget)?;
        frontend.cpu_projection = "Lfm3Bf16Source";
        frontend.execution = ExecutionMetadata {
            weight_dtype: "BF16",
            kv_dtype: "BF16",
            projection_mode: "Lfm3Bf16Source",
            normalization_mode: "Lfm3Bf16Source",
            readback_mode: "WidenedBf16Logits",
            attention_mode: Some("Lfm3EagerCausalOrWindow256"),
            ..frontend.execution
        };
        return run_session(
            frontend,
            &mut session,
            initial,
            a.capacity,
            external_bytes,
            |_| Value::Null,
        );
    }

    let options = Options {
        capacity: a.capacity,
        prefill_chunk_size: a.chunk,
        memory_limit_bytes: budget,
        ..Options::default()
    };
    let load_options = LoadOptions {
        context_limit: a.capacity,
        memory_limit_bytes: budget,
    };
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    if qwen35_2b {
        return qwen35_2b::run(
            qwen35_2b::Inputs {
                directory: &a.model,
                manifest: a.artifact_manifest.as_deref().unwrap(),
                manifest_sha256: a.artifact_manifest_sha256.unwrap(),
                capacity: a.capacity,
                chunk: a.chunk,
                budget,
            },
            frontend,
            initial,
        );
    }
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    if a.backend == "metal" {
        let q4_v2 = a.q4_weights.is_some();
        let mlx = a.metal_projection == "mlx-affine";
        let model = if ministral3 {
            toklet::metal::Model::load_ministral3_diagnostic(&a.model, load_options)?
        } else if mlx {
            toklet::metal::Model::load_mlx(&a.model, load_options)?
        } else if let Some(packed) = &a.q4_weights {
            toklet::metal::Model::load_q4(&a.model, packed, load_options)?
        } else {
            toklet::metal::Model::load(&a.model, load_options)?
        };
        if q4_v2 && model.q4_format() != Some(toklet::q4::Format::V2) {
            return Err(invalid("--q4-weights requires a Q4 v2 artifact").into());
        }
        let projection = match a.metal_projection {
            "q4-v2-compensated-tiled" => toklet::metal::ProjectionMode::Q4CompensatedTiled,
            "half-compensated" => toklet::metal::ProjectionMode::HalfCompensated,
            "mlx-affine" => toklet::metal::ProjectionMode::MlxAffine,
            _ => toklet::metal::ProjectionMode::Original,
        };
        let normalization = if q4_v2 || mlx {
            toklet::metal::NormalizationMode::Compensated
        } else {
            toklet::metal::NormalizationMode::Original
        };
        let attention = if a.metal_attention == "online" {
            toklet::metal::AttentionMode::Online
        } else {
            toklet::metal::AttentionMode::Reference
        };
        let mut session = toklet::metal::Session::with_attention_mode(
            &model,
            options,
            projection,
            normalization,
            toklet::metal::ReadbackMode::FullLogits,
            attention,
        )?;
        frontend.execution = ExecutionMetadata {
            weight_dtype: if ministral3 {
                "MLX affine4/8 + FP16 norms"
            } else if mlx {
                "MLX affine + FP16 exclusions"
            } else if q4_v2 {
                "Q4 v2 + FP16 exclusions"
            } else {
                "FP16"
            },
            kv_dtype: "FP16",
            projection_mode: match projection {
                toklet::metal::ProjectionMode::Q4CompensatedTiled => "Q4CompensatedTiled",
                toklet::metal::ProjectionMode::HalfCompensated => "HalfCompensated",
                toklet::metal::ProjectionMode::MlxAffine => "MlxAffine",
                _ => "Original",
            },
            normalization_mode: if q4_v2 || mlx {
                "Compensated"
            } else {
                "Original"
            },
            readback_mode: "FullLogits",
            attention_mode: if a.metal_attention == "online" {
                Some("Online")
            } else if ministral3 {
                Some("Reference")
            } else {
                None
            },
            prefill_chunk_size: a.chunk,
            packed_format_version: q4_v2.then_some(2),
            quantization_group_size: if ministral3 {
                Some(64)
            } else {
                model.q4_format().map(|f| f.group())
            },
            quantizer: if mlx {
                Some("MLX affine (per-tensor bits/groups)")
            } else {
                model.q4_format().map(|f| f.quantizer())
            },
            state_backend: None,
            state_dtype: None,
        };
        frontend.load_seconds = model.load_timing.load_seconds;
        frontend.conversion_seconds = model.load_timing.conversion_seconds;
        frontend.compilation_seconds = Some(model.load_timing.compilation_seconds);
        frontend.cpu_projection = "none";
        return run_session(
            frontend,
            &mut session,
            initial,
            a.capacity,
            external_bytes,
            |_| Value::Null,
        );
    }
    let loaded = weights::load_qwen(&a.model, load_options)?;
    let model = loaded.model.with_projection_kernel(a.projection);
    let mut session = Session::new(&model, options)?;
    frontend.load_seconds = loaded.report.load_seconds;
    frontend.conversion_seconds = loaded.report.conversion_seconds;
    run_session(
        frontend,
        &mut session,
        initial,
        a.capacity,
        external_bytes,
        |_| Value::Null,
    )
}
fn lfm3_load_json(report: weights::Lfm3LoadReport) -> Value {
    let hash: String = report
        .source_sha256
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    json!({
        "sha256": hash,
        "source_bytes": report.source_bytes,
        "temporary_allowance_bytes": report.temporary_allowance_bytes,
        "authentication_ns": report.authenticated_ns,
        "layout_ns": report.layout_ns,
        "verified_delivery_ns": report.delivery_ns,
        "rotary_ns": report.rotary_ns,
    })
}

fn logical_memory(memory: toklet::session::Memory, external_bytes: usize) -> Value {
    let mut value = json!({"weights_bytes":memory.weights_bytes,"kv_bytes":memory.kv_bytes,"scratch_bytes":memory.scratch_bytes,
        "history_bytes":memory.history_bytes,"logits_bytes":memory.logits_bytes,"metadata_bytes":memory.metadata_bytes,
        "metadata_allowance_bytes":memory.metadata_allowance_bytes,"tokenizer_application_allowance_bytes":external_bytes});
    if memory.positional_bytes != 0 {
        value["positional_bytes"] = memory.positional_bytes.into();
    }
    value
}
fn run_session<S: InferenceSession>(
    mut frontend: Frontend,
    session: &mut S,
    initial: Option<UserRequest>,
    capacity: usize,
    external_bytes: usize,
    diagnostic: impl Fn(&S) -> Value,
) -> Result<u8, Box<dyn std::error::Error>> {
    frontend.memory_admitted_bytes = session
        .memory()
        .admitted_bytes
        .checked_add(external_bytes)
        .ok_or_else(|| invalid("memory accounting overflow"))?;
    if let Some(r) = initial {
        return Ok(if frontend.execute(session, r, &diagnostic)? {
            130
        } else {
            0
        });
    }
    let memory = session.memory();
    let mut ready = json!({"event":"ready", "backend":frontend.backend, "context_capacity":capacity,
        "cpu_projection":frontend.cpu_projection, "memory_admitted_bytes":frontend.memory_admitted_bytes,
        "execution":frontend.execution,"logical_memory":logical_memory(memory,external_bytes)});
    if frontend.model_profile == ModelProfile::Ministral3Diagnostic {
        ready["model_profile"] = json!("ministral3-diagnostic");
    }
    if frontend.model_profile == ModelProfile::Qwen35TwoBDiagnostic {
        ready["model_profile"] = json!("qwen35-2b-diagnostic");
        ready["hybrid"] = diagnostic(session);
        ready["checkpoint_load"] = frontend.qwen35_2b_load.clone().unwrap_or(Value::Null);
    }
    if let Some(load) = frontend.lfm3_load {
        ready["model_profile"] = json!("lfm3-bf16");
        ready["checkpoint_load"] = lfm3_load_json(load);
    }
    writeln!(io::stdout().lock(), "{ready}")?;
    io::stdout().flush()?;
    let (sender, receiver) = mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let mut input = io::BufReader::new(io::stdin());
        loop {
            let mut line = Vec::new();
            let result = input
                .by_ref()
                .take((chat::MAX_INPUT_BYTES + 1) as u64)
                .read_until(b'\n', &mut line);
            match result {
                Ok(0) => break,
                Ok(_) if line.len() <= chat::MAX_INPUT_BYTES => {
                    if sender.send(Ok(line)).is_err() {
                        break;
                    }
                }
                Ok(_) => {
                    let _ = sender.send(Err(invalid("interactive request exceeds 1 MiB")));
                    break;
                }
                Err(e) => {
                    let _ = sender.send(Err(e));
                    break;
                }
            }
        }
    });
    loop {
        if CANCELLED.load(Ordering::Relaxed) {
            frontend.finish_output(&mut io::stdout().lock())?;
            return Ok(130);
        }
        let line = match receiver.recv_timeout(Duration::from_millis(50)) {
            Ok(line) => line?,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                frontend.finish_output(&mut io::stdout().lock())?;
                return Ok(0);
            }
        };
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let result = serde_json::from_slice::<UserRequest>(&line)
            .map_err(invalid)
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })
            .and_then(|r| frontend.execute(session, r, &diagnostic));
        if let Err(e) = result {
            publish_terminal(&CANCELLED, |_| -> io::Result<()> {
                writeln!(
                    io::stdout().lock(),
                    "{}",
                    if frontend.model_profile == ModelProfile::Qwen35TwoBDiagnostic {
                        json!({"event":"error","message":e.to_string(),"hybrid":diagnostic(session)})
                    } else {
                        json!({"event":"error","message":e.to_string()})
                    }
                )?;
                io::stdout().flush()
            })?;
        }
    }
}
pub fn main() -> u8 {
    let a: Vec<_> = env::args_os().skip(1).collect();
    if a.first().is_some_and(|value| value == "inspect") {
        return inspect::main(a.into_iter().skip(1).collect());
    }
    if a.first().is_some_and(|value| value == "plan") {
        return plan::main(a.into_iter().skip(1).collect());
    }
    if a.is_empty() || a.len() == 1 && (a[0] == "--help" || a[0] == "-h") {
        println!("{HELP}");
        return 0;
    }
    if a.len() == 1 && (a[0] == "--version" || a[0] == "-V") {
        println!("toklet {}", env!("CARGO_PKG_VERSION"));
        return 0;
    }
    let a = match parse(a) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\nUse 'toklet --help' for usage.");
            return 2;
        }
    };
    match run(a) {
        Ok(code) => code,
        Err(e)
            if e.downcast_ref::<io::Error>()
                .is_some_and(|e| e.kind() == io::ErrorKind::BrokenPipe)
                || e.downcast_ref::<generation::Error>().is_some_and(|e| {
                    matches!(e,
                    generation::Error::Output(e) if e.kind() == io::ErrorKind::BrokenPipe)
                }) =>
        {
            0
        }
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }
    #[test]
    fn two_b_interactive_inputs_remain_id_only_and_manifest_flags_are_scoped() {
        for value in [
            json!({"tokens":[1,2]}),
            json!({"action":"append","tokens":[3]}),
            json!({"action":"continue","max_new_tokens":2}),
            json!({"action":"reset"}),
        ] {
            let request = serde_json::from_value(value).unwrap();
            assert!(validate_profile_input(ModelProfile::Qwen35TwoBDiagnostic, &request).is_ok());
        }
        for value in [
            json!({"prompt":"text"}),
            json!({"messages":[]}),
            json!({"tokens":[1],"tools":[]}),
            json!({"tokens":[1],"raw":false}),
        ] {
            let request = serde_json::from_value(value).unwrap();
            assert!(validate_profile_input(ModelProfile::Qwen35TwoBDiagnostic, &request).is_err());
        }
        for extra in [
            vec!["--artifact-manifest", "missing"],
            vec![
                "--artifact-manifest-sha256",
                "0000000000000000000000000000000000000000000000000000000000000000",
            ],
        ] {
            let mut values = args(&["--model", "missing", "--tokens", "1", "--json"]);
            values.extend(args(&extra));
            assert!(parse(values).is_err());
        }
    }
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn two_b_diagnostic_flags_bound_modes_and_require_manifest_before_io() {
        let base = [
            "--model",
            "missing",
            "--model-profile",
            "qwen35-2b-diagnostic",
            "--backend",
            "metal",
            "--metal-projection",
            "mlx-affine",
            "--artifact-manifest",
            "missing",
            "--artifact-manifest-sha256",
            "abababababababababababababababababababababababababababababababab",
            "--tokens",
            "1",
            "--json",
        ];
        let parsed = parse(args(&base)).unwrap();
        assert_eq!((parsed.capacity, parsed.chunk), (128, 32));
        for extra in [
            vec!["--context-capacity", "0"],
            vec!["--context-capacity", "513"],
            vec!["--prefill-chunk", "33"],
            vec!["--metal-attention", "online"],
            vec!["--prompt", "text"],
            vec!["--messages", "missing"],
            vec!["--tools", "missing"],
            vec!["--raw"],
            vec!["--q4-weights", "missing"],
            vec!["--cpu-projection", "scalar"],
        ] {
            let mut values = args(&base);
            values.extend(args(&extra));
            assert!(parse(values).is_err());
        }
        for flag in ["--artifact-manifest", "--artifact-manifest-sha256"] {
            let mut values = base.to_vec();
            let index = values.iter().position(|v| *v == flag).unwrap();
            values.drain(index..index + 2);
            assert!(parse(args(&values)).is_err());
        }
        for (from, to) in [("metal", "cpu"), ("mlx-affine", "original")] {
            let mut values = base.to_vec();
            *values.iter_mut().find(|v| **v == from).unwrap() = to;
            assert!(parse(args(&values)).is_err());
        }
        for capacity in [1, 128, 512] {
            let mut values = args(&base);
            values.extend(args(&["--context-capacity", &capacity.to_string()]));
            assert_eq!(parse(values).unwrap().capacity, capacity);
        }
    }
    #[test]
    fn ministral3_diagnostic_interactive_fields_reject_before_session_changes() {
        for value in [
            json!({"tokens":[1,2]}),
            json!({"action":"append","tokens":[3]}),
            json!({"action":"continue","max_new_tokens":2}),
            json!({"action":"reset"}),
        ] {
            let request = serde_json::from_value(value).unwrap();
            assert!(validate_profile_input(ModelProfile::Ministral3Diagnostic, &request).is_ok());
        }
        for value in [
            json!({"prompt":"text"}),
            json!({"messages":[]}),
            json!({"tokens":[1],"tools":[]}),
            json!({"tokens":[1],"raw":false}),
            json!({"action":"continue","raw":true}),
        ] {
            let request = serde_json::from_value(value).unwrap();
            assert!(validate_profile_input(ModelProfile::Ministral3Diagnostic, &request).is_err());
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn ministral3_diagnostic_flags_reject_unqualified_routes_before_io() {
        let base = [
            "--model",
            "missing",
            "--model-profile",
            "ministral3-diagnostic",
            "--backend",
            "metal",
            "--metal-projection",
            "mlx-affine",
            "--tokens",
            "1",
            "--json",
        ];
        let parsed = parse(args(&base)).unwrap();
        assert_eq!((parsed.capacity, parsed.chunk), (128, 128));
        assert!(parsed.model_profile == ModelProfile::Ministral3Diagnostic);
        for extra in [
            vec!["--context-capacity", "0"],
            vec!["--context-capacity", "513"],
            vec!["--metal-attention", "online"],
            vec!["--prompt", "text"],
            vec!["--tools", "missing"],
            vec!["--raw"],
        ] {
            let mut values = args(&base);
            values.extend(args(&extra));
            assert!(parse(values).is_err());
        }
        for capacity in [1, 128, 512] {
            let mut values = args(&base);
            values.extend(args(&["--context-capacity", &capacity.to_string()]));
            assert_eq!(parse(values).unwrap().capacity, capacity);
        }
        for (from, to) in [("metal", "cpu"), ("mlx-affine", "original")] {
            let mut values = base.to_vec();
            *values.iter_mut().find(|value| **value == from).unwrap() = to;
            assert!(parse(args(&values)).is_err());
        }
        assert!(parse(args(&base[..base.len() - 1])).is_err());
        let mut interactive = base[..8].to_vec();
        interactive.push("--interactive");
        assert!(parse(args(&interactive)).unwrap().interactive);
    }
    #[test]
    fn lfm3_profile_rejects_other_graphs_and_unqualified_tools_before_io() {
        let base = &[
            "--model",
            "missing",
            "--model-profile",
            "lfm3-bf16",
            "--tokens",
            "1",
        ];
        let request = parse(args(base)).unwrap();
        assert!(request.model_profile == ModelProfile::Lfm3Bf16);
        assert_eq!(request.chunk, 16);
        for options in [
            ["--cpu-projection", "scalar"],
            ["--backend", "metal"],
            ["--tools", "missing"],
            ["--prefill-chunk", "2049"],
        ] {
            let mut values = args(base);
            values.extend(args(&options));
            assert!(parse(values).is_err());
        }
    }

    #[test]
    #[ignore = "explicit pinned LFM3 tokenizer assets; no checkpoint execution"]
    fn lfm3_source_render_tokenization_and_streaming() {
        use sha2::{Digest, Sha256};
        let directory = PathBuf::from(env::var_os("TOKLET_LFM3_MODEL").unwrap());
        let bytes = include_bytes!("../../../tests/fixtures/lfm3-cli-text-v1/fixture.json");
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            "2f35654790add9749a7a165ee04a535f8e7c437351d6b788b1765a642376858e"
        );
        let source: Value = serde_json::from_slice(bytes).unwrap();
        let limit = toklet::app_chat::lfm3::ALLOWANCE_BYTES;
        let assets = Assets {
            tokenizer: toklet::app_chat::lfm3::load_tokenizer(&directory, limit).unwrap(),
            admitted_bytes: limit,
        };
        for row in source["chats"].as_array().unwrap() {
            let text = toklet::app_chat::lfm3::render(row["messages"].as_array().unwrap()).unwrap();
            assert_eq!(text, row["text"].as_str().unwrap());
            assert_eq!(json!(assets.encode(&text).unwrap()), row["ids"]);
        }
        for row in source["raw"].as_array().unwrap() {
            let ids = assets.encode(row["text"].as_str().unwrap()).unwrap();
            assert_eq!(json!(ids), row["ids"]);
            let mut stream = OutputDecoder::default();
            let mut decoded = String::new();
            for id in ids {
                decoded.push_str(&stream.step(&assets, id).unwrap());
            }
            decoded.push_str(&stream.finish(&assets).unwrap());
            assert_eq!(decoded, row["decoded"].as_str().unwrap());
        }
    }

    #[test]
    fn attention_selection_and_context_bounds_reject_before_model_access() {
        for tail in [
            vec!["--metal-attention", "online"],
            vec!["--metal-attention", "reference"],
            vec!["--metal-attention", "bad"],
            vec!["--metal-attention", "online", "--metal-attention", "online"],
            vec!["--context-capacity", "2049"],
        ] {
            let mut values = args(&["--model", "missing", "--tokens", "1"]);
            values.extend(args(&tail));
            assert!(parse(values).is_err());
        }
        let base = parse(args(&["--model", "missing", "--tokens", "1"])).unwrap();
        assert_eq!(
            (base.capacity, base.chunk, base.metal_attention),
            (2048, 128, "reference")
        );
    }
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn explicit_metal_context_and_attention_parse_with_default_compatibility() {
        for mode in ["reference", "online"] {
            for capacity in ["4096", "8192", "16384", "32768"] {
                let values = args(&[
                    "--model",
                    "missing",
                    "--tokens",
                    "1",
                    "--backend",
                    "metal",
                    "--context-capacity",
                    capacity,
                    "--metal-attention",
                    mode,
                ]);
                let parsed = parse(values).unwrap();
                assert_eq!(parsed.capacity, capacity.parse::<usize>().unwrap());
                assert_eq!(parsed.metal_attention, mode);
            }
        }
        assert!(
            parse(args(&[
                "--model",
                "missing",
                "--tokens",
                "1",
                "--backend",
                "metal",
                "--context-capacity",
                "32769"
            ]))
            .is_err()
        );
    }
    #[test]
    fn native_mlx_selection_is_explicit_and_conflicts_reject_before_loading() {
        let mut base = args(&["--model", "missing", "--tokens", "1"]);
        base.extend(args(&["--metal-projection", "mlx-affine"]));
        assert!(parse(base.clone()).is_err());
        base.extend(args(&["--backend", "metal"]));
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        {
            assert_eq!(parse(base.clone()).unwrap().metal_projection, "mlx-affine");
            let mut duplicate = base.clone();
            duplicate.extend(args(&["--metal-projection", "mlx-affine"]));
            assert!(parse(duplicate).is_err());
        }
        #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
        assert!(parse(base.clone()).is_err());
        base.extend(args(&["--q4-weights", "missing-packed"]));
        assert!(parse(base).is_err());
    }
    #[test]
    fn idle_interrupt_after_terminal_boundary_is_preserved() {
        let flag = AtomicBool::new(true);
        publish_terminal(&flag, |active_cancelled| {
            assert!(active_cancelled);
            assert!(!flag.load(Ordering::Relaxed));
            // A signal while publishing done/error belongs to idle state.
            flag.store(true, Ordering::Relaxed);
        });
        assert!(flag.load(Ordering::Relaxed));
    }
}
