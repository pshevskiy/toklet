//! Paired Reference/Online sessions for explicit M6 qualification. Portable
//! protocol tests remain available without Metal; native execution is opt-in.
#[cfg(any(
    test,
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
mod protocol {
    use serde::Deserialize;

    #[derive(Clone, Copy, Deserialize, serde::Serialize)]
    #[serde(rename_all = "snake_case")]
    pub(super) enum Mode {
        Reference,
        Online,
    }
    #[derive(Clone, Copy, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub(super) enum Action {
        Reset,
        Prompt,
        Append,
        Continue,
        Prepare,
        Stage,
        Compare,
        Profile,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(super) struct Request {
        pub(super) action: Action,
        pub(super) mode: Option<Mode>,
        #[serde(default)]
        pub(super) tokens: Vec<u32>,
        #[serde(default)]
        pub(super) continuation: Vec<u32>,
    }
    impl Request {
        pub(super) fn validate(
            &self,
            capacity: usize,
            profiling: bool,
        ) -> Result<(), &'static str> {
            if self.tokens.len() > capacity.checked_add(1).ok_or("capacity overflow")?
                || self.continuation.len() > 128
            {
                return Err("bounded context request sizes exceeded");
            }
            if matches!(self.action, Action::Continue) && !self.tokens.is_empty() {
                return Err("continue accepts no tokens");
            }
            if matches!(self.action, Action::Compare) != self.mode.is_none() {
                return Err("compare requires paired mode; all other actions need explicit mode");
            }
            let preparation = matches!(self.action, Action::Prepare | Action::Stage);
            if preparation != self.continuation.is_empty() {
                return Err("preparation accepts no continuation; other actions need output IDs");
            }
            if matches!(self.action, Action::Profile) != profiling && !preparation {
                return Err("profiled and ordinary requests must use separate workers");
            }
            if matches!(self.action, Action::Profile) && self.continuation.len() != 1 {
                return Err("profile needs exactly one committed decode input");
            }
            Ok(())
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use super::protocol::*;
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::{
        hint::black_box,
        io::{self, BufRead, Read, Write},
        path::PathBuf,
        time::Instant,
    };
    use toklet::{
        benchmark::Sample,
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
    fn input(r: &Request) -> Input<'_> {
        match r.action {
            Action::Prompt | Action::Profile => Input::Prompt(&r.tokens),
            Action::Append => Input::Append(&r.tokens),
            Action::Continue => Input::Continue,
            _ => Input::Reset(&r.tokens),
        }
    }
    fn admit(s: &Session<'_>, r: &Request) -> Result<()> {
        generation::validate_request(
            s,
            &GenerationRequest {
                input: input(r),
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
        match input(r) {
            Input::Prompt(t) => s.set_tokens(t)?,
            Input::Append(t) => s.append(t)?,
            Input::Continue => {}
            Input::Reset(t) => {
                s.reset();
                s.append(t)?;
            }
        }
        Ok(())
    }
    fn measure(s: &mut Session<'_>, r: &Request) -> Result<Sample> {
        admit(s, r)?;
        let mut events = Vec::with_capacity(r.continuation.len());
        let clock = Instant::now();
        apply(s, r)?;
        let reused = s.evaluated_len();
        let pending = s.pending_len();
        let prefill = Instant::now();
        s.evaluate()?;
        let prefill_seconds = prefill.elapsed().as_secs_f64();
        black_box(s.logits().ok_or("no completed benchmark distribution")?);
        events.push(clock.elapsed().as_secs_f64());
        s.append(&r.continuation[..1])?;
        for id in &r.continuation[1..] {
            s.evaluate_next_chunk()?;
            black_box(s.logits().ok_or("no completed decode distribution")?);
            events.push(clock.elapsed().as_secs_f64());
            s.append(&[*id])?;
        }
        Ok(Sample {
            prefill_seconds,
            ttft_seconds: events[0],
            decode_seconds: events[events.len() - 1] - events[0],
            total_seconds: clock.elapsed().as_secs_f64(),
            emitted_tokens: r.continuation.len(),
            reused_tokens: reused,
            prefill_tokens: pending,
            evaluated_tokens: s.evaluated_len(),
            pending_tokens: s.pending_len(),
            token_event_seconds: events,
        })
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
    fn compare(sessions: &mut [Session<'_>; 2], r: &Request) -> Result<Vec<Value>> {
        for s in sessions.iter() {
            admit(s, r)?;
        }
        for s in sessions.iter_mut() {
            apply(s, r)?;
            s.evaluate()?;
        }
        let initial_prefix = sessions[0].evaluated_len();
        let mut rows = Vec::with_capacity(r.continuation.len());
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
            for s in sessions.iter_mut() {
                s.append(&[*id])?;
            }
        }
        if sessions[0].history() != sessions[1].history()
            || sessions[0].evaluated_len() != sessions[1].evaluated_len()
        {
            return Err("paired quality state differs".into());
        }
        let mut events = vec![
            json!({"event":"comparison_begin","scored_rows":rows.len(),"initial_prefix_tokens":initial_prefix}),
        ];
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
                .ok_or("usage: metal_context MODEL PACKED|- CAPACITY CHUNK [profile]")?,
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
        let profiling = match args.next() {
            None => false,
            Some(v) if v == "profile" => true,
            _ => return Err("optional mode must be profile".into()),
        };
        if args.next().is_some()
            || capacity == 0
            || capacity > weights::MAX_CONTEXT_LIMIT
            || chunk == 0
            || chunk > 1024
            || chunk > capacity
        {
            return Err("unsupported context worker arguments".into());
        }
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
            json!({"event":"ready","format":"toklet-m6-context-v1","debug_assertions":cfg!(debug_assertions),
            "device":model.device_name(),"context_capacity":capacity,"chunk_size":chunk,"profiling":profiling,
            "hidden_size":c.hidden_size,"num_layers":c.num_layers,"vocab_size":c.vocab_size,"projection_mode":projection,
            "normalization_mode":normalization,"readback_mode":"FullLogits","kv_dtype":"FP16",
            "weight_dtype":if quantized{"Q4 v2 + FP16 exclusions"}else{"FP16"},"packed_format_version":model.q4_format().map(|_|2),"group_size":model.q4_format().map(|f|f.group()),"quantizer":model.q4_format().map(|f|f.quantizer()),
            "q4_execution_widths":model.q4_compensated_execution_widths(),"half_execution_widths":model.half_compensated_execution_widths(),
            "reference_memory":sessions[0].memory(),"online_memory":sessions[1].memory(),"shared_weights_bytes":model.weights_bytes(),
            "application_bytes":APP,"unique_memory_admitted_bytes":admitted,"load_seconds":model.load_timing.load_seconds})
        )?;
        out.flush()?;
        let mut input = io::BufReader::new(io::stdin());
        let mut count = 0;
        loop {
            let mut line = Vec::new();
            let n = input
                .by_ref()
                .take(1024 * 1024 + 1)
                .read_until(b'\n', &mut line)?;
            if n == 0 {
                break;
            }
            if n > 1024 * 1024 {
                return Err("request exceeds1MiB".into());
            }
            count += 1;
            if count > 512 {
                return Err("request count exceeds512".into());
            }
            let result = (|| -> Result<Vec<Value>> {
                let r: Request = serde_json::from_slice(&line)?;
                r.validate(capacity, profiling)?;
                if matches!(r.action, Action::Compare) {
                    return compare(&mut sessions, &r);
                }
                let mode = r.mode.ok_or("mode required")?;
                let index = match mode {
                    Mode::Reference => 0,
                    Mode::Online => 1,
                };
                let s = &mut sessions[index];
                if matches!(r.action, Action::Prepare | Action::Stage) {
                    if matches!(r.action, Action::Prepare) {
                        admit(s, &r)?;
                        apply(s, &r)?;
                        s.evaluate()?;
                    } else {
                        s.append(&r.tokens)?;
                    }
                    return Ok(vec![
                        json!({"event":"prepared","mode":mode,"state":state(s),"memory":s.memory()}),
                    ]);
                }
                if matches!(r.action, Action::Profile) {
                    admit(s, &r)?;
                    apply(s, &r)?;
                    let reused = s.evaluated_len();
                    let prefill = s.pending_len();
                    s.evaluate()?;
                    s.append(&r.continuation)?;
                    if s.evaluate_next_chunk()? != 1 {
                        return Err("profile must commit one decode row".into());
                    }
                    let profile = s.profile()?;
                    let mut events = vec![
                        json!({"event":"profile_begin","mode":mode,"start":profile.start,"end":profile.end,
                        "host_clock_sample_bracket_seconds":profile.host_clock_sample_bracket_seconds,"reused_tokens":reused,"prefill_tokens":prefill}),
                    ];
                    for (i, event) in profile.events.iter().enumerate() {
                        events.push(json!({"event":"profile_operator","index":i,"profile":event}));
                    }
                    events.push(json!({"event":"profile_done","mode":mode,"operator_count":profile.events.len(),"state":state(s),"memory":s.memory(),"timing":s.timing()}));
                    return Ok(events);
                }
                let sample = measure(s, &r)?;
                Ok(vec![
                    json!({"event":"sample","mode":mode,"sample":sample,"state":state(s),"memory":s.memory(),"last_completed_timing":s.timing()}),
                ])
            })();
            match result {
                Ok(events) => {
                    for event in events {
                        writeln!(out, "{event}")?;
                    }
                }
                Err(error) => writeln!(
                    out,
                    "{}",
                    json!({"event":"error","message":error.to_string(),"states":sessions.iter().map(state).collect::<Vec<_>>(),"memories":sessions.iter().map(Session::memory).collect::<Vec<_>>()})
                )?,
            }
            out.flush()?;
        }
        Ok(())
    }
}
fn main() {
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    if let Err(error) = apple::run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
    #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
    {
        eprintln!("error: native execution requires Metal on Apple Silicon macOS");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::protocol::*;
    use serde_json::json;
    fn request(value: serde_json::Value) -> Request {
        serde_json::from_value(value).unwrap()
    }
    #[test]
    fn ordinary_and_profile_modes_cannot_be_mixed() {
        let r = request(json!({"action":"reset","mode":"online","tokens":[1],"continuation":[2]}));
        assert!(r.validate(32, false).is_ok());
        assert!(r.validate(32, true).is_err());
        let r =
            request(json!({"action":"profile","mode":"reference","tokens":[1],"continuation":[2]}));
        assert!(r.validate(32, true).is_ok());
        assert!(r.validate(32, false).is_err());
        assert!(
            request(
                json!({"action":"profile","mode":"reference","tokens":[1],"continuation":[2,3]})
            )
            .validate(32, true)
            .is_err()
        );
    }
    #[test]
    fn compare_requires_paired_mode_and_nonempty_fixed_targets() {
        assert!(
            request(json!({"action":"compare","tokens":[1],"continuation":[2]}))
                .validate(32, false)
                .is_ok()
        );
        assert!(
            request(json!({"action":"compare","mode":"online","tokens":[1],"continuation":[2]}))
                .validate(32, false)
                .is_err()
        );
        assert!(
            request(json!({"action":"compare","tokens":[1]}))
                .validate(32, false)
                .is_err()
        );
        assert!(
            request(json!({"action":"reset","tokens":[1],"continuation":[2]}))
                .validate(32, false)
                .is_err()
        );
    }
    #[test]
    fn preparation_and_continuation_reject_conflicting_payloads() {
        for action in ["prepare", "stage"] {
            assert!(
                request(json!({"action":action,"mode":"online","tokens":[1]}))
                    .validate(32, false)
                    .is_ok()
            );
            assert!(
                request(json!({"action":action,"mode":"online","tokens":[1],"continuation":[2]}))
                    .validate(32, false)
                    .is_err()
            );
        }
        assert!(
            request(
                json!({"action":"continue","mode":"reference","tokens":[1],"continuation":[2]})
            )
            .validate(32, false)
            .is_err()
        );
        assert!(
            request(json!({"action":"continue","mode":"reference","continuation":[2]}))
                .validate(32, false)
                .is_ok()
        );
    }
    #[test]
    fn fixed_output_and_input_bounds_do_not_overflow() {
        let mut r =
            request(json!({"action":"reset","mode":"online","tokens":[1],"continuation":[2]}));
        r.continuation = vec![2; 128];
        assert!(r.validate(32, false).is_ok());
        r.continuation.push(2);
        assert!(r.validate(32, false).is_err());
        r.continuation = vec![2];
        r.tokens = vec![1; 34];
        assert!(r.validate(32, false).is_err());
        r.tokens = vec![1; 33];
        assert!(r.validate(32, false).is_ok()); // backend checks one-over before mutation
        assert!(r.validate(usize::MAX, false).is_err());
    }
    #[test]
    fn unknown_modes_actions_and_fields_fail_parsing() {
        for value in [
            json!({"action":"reset","mode":"automatic"}),
            json!({"action":"magic","mode":"online"}),
            json!({"action":"reset","mode":"online","ignore_errors":true}),
            json!({"action":"reset","mode":"online","tokens":[-1]}),
        ] {
            assert!(serde_json::from_value::<Request>(value).is_err());
        }
    }
}
