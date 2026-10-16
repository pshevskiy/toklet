//! Bounded full-distribution diagnostic derived from the reviewed M6 paired
//! worker. The numerical execution order and kernels are unchanged. No timings
//! from this diagnostic qualify latency. Writes only already completed logits.
#[cfg(any(
    test,
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
mod protocol {
    use serde::Deserialize;
    #[derive(Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub(super) enum Action {
        Compare,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(super) struct Request {
        pub(super) action: Action,
        pub(super) mode: Option<()>,
        pub(super) tokens: Vec<u32>,
        pub(super) continuation: Vec<u32>,
        pub(super) captures: Vec<usize>,
    }
    impl Request {
        pub(super) fn validate(&self, capacity: usize) -> Result<(), &'static str> {
            if !matches!(self.action, Action::Compare)
                || self.mode.is_some()
                || self.tokens.is_empty()
                || self.continuation.len() != 128
                || self
                    .tokens
                    .len()
                    .checked_add(128)
                    .ok_or("capacity overflow")?
                    != capacity
                || self.captures.is_empty()
                || self.captures.len() > 8
                || self.captures.iter().any(|i| *i >= 128)
                || self.captures.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(
                    "capture requires one exact-capacity compare and 1..8 sorted unique target indices",
                );
            }
            Ok(())
        }
    }
}

#[cfg(any(
    test,
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
mod capture {
    use std::{
        fs::OpenOptions,
        io::{BufWriter, Write},
        path::Path,
    };
    // Fixed filenames and exclusive files; no caller-provided filename/path.
    // Caller admits at most eight pairs, 151936 f32 values per real-model file.
    pub(super) fn write(
        dir: &Path,
        index: usize,
        mode: &str,
        values: &[f32],
    ) -> std::io::Result<String> {
        let name = format!("{mode}-target{index:04}.fp32le");
        let mut out = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(dir.join(&name))?,
        );
        for value in values {
            out.write_all(&value.to_le_bytes())?;
        }
        out.flush()?;
        Ok(name)
    }
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use super::protocol::*;
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::{
        io::{self, BufRead, Read, Write},
        path::{Path, PathBuf},
    };
    use toklet::{
        generation::{self, Input, Request as GenerationRequest},
        metal::{self, Session},
        session, weights,
    };
    const APP: usize = 64 * 1024 * 1024;
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    fn history_sha(s: &Session<'_>) -> String {
        let mut h = Sha256::new();
        for id in s.history() {
            h.update(id.to_le_bytes());
        }
        format!("{:x}", h.finalize())
    }
    fn logits_sha(values: &[f32]) -> String {
        let mut h = Sha256::new();
        for v in values {
            h.update(v.to_le_bytes());
        }
        format!("{:x}", h.finalize())
    }
    fn state(s: &Session<'_>) -> Value {
        json!({"history_tokens":s.history().len(),"evaluated_tokens":s.evaluated_len(),"pending_tokens":s.pending_len(),"history_sha256":history_sha(s)})
    }
    fn admit(s: &Session<'_>, r: &Request) -> Result<()> {
        generation::validate_request(
            s,
            &GenerationRequest {
                input: Input::Reset(&r.tokens),
                max_new_tokens: r.continuation.len(),
                stop_tokens: &[],
            },
        )?;
        if r.continuation
            .iter()
            .any(|id| *id as usize >= s.vocab_size())
        {
            return Err("fixed output ID outside vocabulary".into());
        }
        Ok(())
    }
    fn apply(s: &mut Session<'_>, r: &Request) -> Result<()> {
        s.reset();
        s.append(&r.tokens)?;
        Ok(())
    }
    fn nll(values: &[f32], target: usize) -> Result<f64> {
        if target >= values.len() || values.iter().any(|x| !x.is_finite()) {
            return Err("invalid quality distribution/target".into());
        }
        let maximum = values
            .iter()
            .map(|x| f64::from(*x))
            .fold(f64::NEG_INFINITY, f64::max);
        let (mut sum, mut correction) = (0., 0.);
        for value in values {
            let adjusted = (f64::from(*value) - maximum).exp() - correction;
            let next = sum + adjusted;
            correction = (next - sum) - adjusted;
            sum = next;
        }
        let value = (maximum - f64::from(values[target])) + sum.ln();
        if !value.is_finite() || value < 0. {
            return Err("invalid quality loss".into());
        }
        Ok(value)
    }
    fn compare(
        sessions: &mut [Session<'_>; 2],
        r: &Request,
        capture_dir: &Path,
    ) -> Result<Vec<Value>> {
        for s in sessions.iter() {
            admit(s, r)?;
        }
        for s in sessions.iter_mut() {
            apply(s, r)?;
            s.evaluate()?;
        }
        let initial_prefix = sessions[0].evaluated_len();
        let mut rows = Vec::with_capacity(r.continuation.len());
        let mut captures = Vec::with_capacity(r.captures.len());
        for (i, id) in r.continuation.iter().enumerate() {
            if i > 0 {
                for s in sessions.iter_mut() {
                    s.evaluate_next_chunk()?;
                }
            }
            let a = sessions[0].logits().ok_or("reference logits unavailable")?;
            let b = sessions[1].logits().ok_or("Online logits unavailable")?;
            if a.len() != b.len() {
                return Err("paired vocabulary differs".into());
            }
            let (mut delta, mut square) = (0f64, 0f64);
            for (a, b) in a.iter().zip(b) {
                let difference = (f64::from(*a) - f64::from(*b)).abs();
                if !difference.is_finite() {
                    return Err("nonfinite paired logits".into());
                }
                delta = delta.max(difference);
                square += difference * difference;
            }
            let ar = toklet::sampling::greedy(a)?;
            let br = toklet::sampling::greedy(b)?;
            let second = a
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != ar as usize)
                .map(|(_, x)| *x)
                .fold(f32::NEG_INFINITY, f32::max);
            rows.push(json!({"target_index":i,"target_token":id,"prefix_tokens":sessions[0].evaluated_len(),
                "reference_nll":nll(a,*id as usize)?,"online_nll":nll(b,*id as usize)?,
                "reference_greedy":ar,"online_greedy":br,"reference_margin":f64::from(a[ar as usize])-f64::from(second),
                "reference_winner_logits":[a[ar as usize],b[ar as usize]],"online_winner_logits":[a[br as usize],b[br as usize]],
                "max_absolute_logit_delta":delta,"rms_logit_delta":(square/a.len() as f64).sqrt(),
                "vocabulary_values_compared":a.len(),"reference_logits_sha256":logits_sha(a),"online_logits_sha256":logits_sha(b)}));
            if r.captures.binary_search(&i).is_ok() {
                let reference_file = super::capture::write(capture_dir, i, "reference", a)?;
                let online_file = super::capture::write(capture_dir, i, "online", b)?;
                captures.push(json!({"event":"capture", "target_index":i,
                    "prefix_tokens":sessions[0].evaluated_len(), "prefix_sha256":history_sha(&sessions[0]),
                    "vocabulary_values":a.len(), "reference_file":reference_file, "online_file":online_file,
                    "reference_logits_sha256":logits_sha(a), "online_logits_sha256":logits_sha(b)}));
            }
            for s in sessions.iter_mut() {
                s.append(&[*id])?;
            }
        }
        if sessions[0].history() != sessions[1].history()
            || sessions[0].evaluated_len() != sessions[1].evaluated_len()
        {
            return Err("paired quality state differs".into());
        }
        let mut events = captures;
        events.push(json!({"event":"comparison_begin","scored_rows":rows.len(),"initial_prefix_tokens":initial_prefix}));
        for mut row in rows {
            row["event"] = json!("comparison_row");
            events.push(row);
        }
        events.push(json!({"event":"comparison_done","scored_rows":r.continuation.len(),"states":sessions.iter().map(state).collect::<Vec<_>>(),"memories":sessions.iter().map(Session::memory).collect::<Vec<_>>()}));
        Ok(events)
    }
    pub fn run() -> Result<()> {
        let mut args = std::env::args_os().skip(1);
        let source = PathBuf::from(
            args.next()
                .ok_or("usage: metal_context_capture MODEL PACKED|- CAPACITY CHUNK CAPTURE_DIR")?,
        );
        let packed = args.next().ok_or("missing packed path or -")?;
        let capacity = args
            .next()
            .ok_or("missing capacity")?
            .to_string_lossy()
            .parse::<usize>()?;
        let chunk = args
            .next()
            .ok_or("missing chunk")?
            .to_string_lossy()
            .parse::<usize>()?;
        let capture_dir = PathBuf::from(args.next().ok_or("missing fresh capture directory")?);
        let profiling = false;
        if args.next().is_some()
            || capacity == 0
            || capacity > weights::MAX_CONTEXT_LIMIT
            || chunk == 0
            || chunk > 1024
            || chunk > capacity
        {
            return Err("unsupported context worker arguments".into());
        }
        std::fs::create_dir(&capture_dir)?;
        let budget = weights::INITIAL_MEMORY_LIMIT_BYTES - APP;
        let load = weights::LoadOptions {
            context_limit: capacity,
            memory_limit_bytes: budget,
        };
        let quantized = packed != "-";
        let model = if quantized {
            metal::Model::load_q4(&source, &PathBuf::from(packed), load)?
        } else {
            metal::Model::load(&source, load)?
        };
        let c = model.config();
        if !matches!(
            (c.hidden_size, c.num_layers, model.q4_format()),
            (32, 2, None) | (896, 24, None) | (1536, 28, Some(toklet::q4::Format::V2))
        ) {
            return Err("only tiny Half, selected small Half or larger Q4 v2 supported".into());
        }
        let projection = if quantized {
            metal::ProjectionMode::Q4CompensatedTiled
        } else {
            metal::ProjectionMode::HalfCompensated
        };
        let normalization = if quantized {
            metal::NormalizationMode::Compensated
        } else {
            metal::NormalizationMode::Original
        };
        let create = |limit, attention| {
            Session::with_attention_mode(
                &model,
                session::Options {
                    capacity,
                    prefill_chunk_size: chunk,
                    memory_limit_bytes: limit,
                    ..Default::default()
                },
                projection,
                normalization,
                metal::ReadbackMode::FullLogits,
                attention,
            )
        };
        let mut reference = create(budget, metal::AttentionMode::Reference)?;
        if profiling {
            reference.enable_profile()?;
        }
        // Model weights are shared once. Reserve all reference-owned resources
        // before admitting any Online allocations, including profile allowance.
        let reference_owned = reference
            .memory()
            .admitted_bytes
            .checked_sub(model.weights_bytes())
            .ok_or("paired memory subtraction")?;
        let remaining = budget
            .checked_sub(reference_owned)
            .ok_or("paired memory budget exhausted")?;
        let mut online = create(remaining, metal::AttentionMode::Online)?;
        if profiling {
            online.enable_profile()?;
        }
        let admitted = reference
            .memory()
            .admitted_bytes
            .checked_add(online.memory().admitted_bytes)
            .and_then(|n| n.checked_sub(model.weights_bytes()))
            .and_then(|n| n.checked_add(APP))
            .ok_or("paired memory overflow")?;
        if admitted > weights::INITIAL_MEMORY_LIMIT_BYTES {
            return Err("paired admission exceeds6GiB".into());
        }
        let mut sessions = [reference, online];
        let mut out = io::stdout().lock();
        writeln!(
            out,
            "{}",
            json!({"event":"ready","format":"toklet-m6-winner-capture-v1","debug_assertions":cfg!(debug_assertions),
            "device":model.device_name(),"context_capacity":capacity,"chunk_size":chunk,"profiling":profiling,
            "hidden_size":c.hidden_size,"num_layers":c.num_layers,"vocab_size":c.vocab_size,"projection_mode":projection,
            "normalization_mode":normalization,"readback_mode":"FullLogits","kv_dtype":"FP16",
            "weight_dtype":if quantized{"Q4 v2 + FP16 exclusions"}else{"FP16"},"packed_format_version":model.q4_format().map(|_|2),"group_size":model.q4_format().map(|f|f.group()),"quantizer":model.q4_format().map(|f|f.quantizer()),
            "q4_execution_widths":model.q4_compensated_execution_widths(),"half_execution_widths":model.half_compensated_execution_widths(),
            "reference_memory":sessions[0].memory(),"online_memory":sessions[1].memory(),"shared_weights_bytes":model.weights_bytes(),
            "application_bytes":APP,"unique_memory_admitted_bytes":admitted,"load_seconds":model.load_timing.load_seconds})
        )?;
        out.flush()?;
        // Exactly one compare request per process; output files cannot be replaced.
        let mut input = io::BufReader::new(io::stdin());
        let mut line = Vec::new();
        let n = input
            .by_ref()
            .take(1024 * 1024 + 1)
            .read_until(b'\n', &mut line)?;
        if n == 0 || n > 1024 * 1024 {
            return Err("capture request missing or exceeds1MiB".into());
        }
        let r: Request = serde_json::from_slice(&line)?;
        r.validate(capacity)?;
        for event in compare(&mut sessions, &r, &capture_dir)? {
            writeln!(out, "{event}")?;
        }
        out.flush()?;
        let mut extra = [0u8; 1];
        if input.read(&mut extra)? != 0 {
            return Err("capture accepts one request only".into());
        }
        Ok(())
    }
}
fn main() {
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    if let Err(e) = apple::run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
    #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
    {
        eprintln!("capture requires Metal on Apple Silicon");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{capture, protocol::Request};
    use serde_json::json;
    fn request() -> Request {
        serde_json::from_value(json!({"action":"compare", "tokens":[1],
            "continuation":vec![2;128], "captures":[90,127]}))
        .unwrap()
    }
    #[test]
    fn bounded_exact_capacity_capture_protocol() {
        let mut r = request();
        assert!(r.validate(129).is_ok());
        for indices in [
            vec![],
            vec![128],
            vec![90, 90],
            vec![127, 90],
            (0..9).collect(),
        ] {
            r.captures = indices;
            assert!(r.validate(129).is_err());
        }
        let mut r = request();
        r.continuation.pop();
        assert!(r.validate(129).is_err());
        let mut r = request();
        r.tokens.clear();
        assert!(r.validate(128).is_err());
        assert!(request().validate(128).is_err());
        assert!(request().validate(130).is_err());
        for value in [
            json!({"action":"profile"}),
            json!({"action":"compare","captures":[-1]}),
            json!({"action":"compare","tokens":[1],"continuation":vec![2;128],"captures":[90],"other":true}),
        ] {
            assert!(serde_json::from_value::<Request>(value).is_err());
        }
    }
    #[test]
    fn exact_fp32_file_bits_and_exclusive_creation() {
        let dir = std::env::temp_dir().join(format!("toklet-m6-capture-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        {
            let values = [1.0f32, -0.0, f32::from_bits(1), 17.63076];
            let name = capture::write(&dir, 127, "reference", &values).unwrap();
            assert_eq!(name, "reference-target0127.fp32le");
            let bytes = std::fs::read(dir.join(&name)).unwrap();
            assert_eq!(
                bytes,
                values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>()
            );
            assert!(capture::write(&dir, 127, "reference", &values).is_err());
            assert_eq!(std::fs::read(dir.join(name)).unwrap(), bytes);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
