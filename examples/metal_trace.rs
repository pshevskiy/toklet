//! Explicit diagnostic worker. A single immutable model is shared by reference
//! and profiled GPU sessions. Counter exports and idle handshakes are outside GPU work.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{
        io::{self, BufRead, Read, Write},
        path::PathBuf,
        time::Instant,
    };
    use toklet::{metal, sampling, session, weights};
    fn handshake() -> Result<(), Box<dyn std::error::Error>> {
        let mut line = String::new();
        if io::stdin().lock().take(16).read_line(&mut line)? == 0 || line != "continue\n" {
            return Err("trace worker requires bounded idle continue handshake".into());
        }
        Ok(())
    }
    fn same(
        a: &metal::Session<'_>,
        b: &metal::Session<'_>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let left = a.logits().ok_or("reference distribution missing")?;
        let right = b.logits().ok_or("profile distribution missing")?;
        if left.len() != right.len()
            || left
                .iter()
                .zip(right)
                .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err("profiling changed logits bits".into());
        }
        let mut hash = Sha256::new();
        for value in right {
            hash.update(value.to_le_bytes());
        }
        Ok(format!("{:x}", hash.finalize()))
    }
    fn export(
        out: &mut impl Write,
        run: usize,
        operation: &str,
        base: &metal::Session<'_>,
        profiled: &metal::Session<'_>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let hash = same(base, profiled)?;
        let profile = profiled.profile()?;
        writeln!(
            out,
            "{}",
            json!({"event":"profile","run":run,"operation":operation,"baseline_timing":base.timing().unwrap(),"profile_timing":profiled.timing().unwrap(),"start":profile.start,"end":profile.end,"host_clock_sample_bracket_seconds":profile.host_clock_sample_bracket_seconds,"event_count":profile.events.len(),"logits_sha256":hash,"greedy_id":sampling::greedy(profiled.logits().unwrap())?})
        )?;
        for (index, event) in profile.events.iter().enumerate() {
            writeln!(
                out,
                "{}",
                json!({"event":"stage","run":run,"operation":operation,"index":index,"sample":event})
            )?;
        }
        out.flush()?;
        Ok(())
    }
    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let mut args = std::env::args_os().skip(1);
        let model_path = PathBuf::from(args.next().ok_or("missing model directory")?);
        let capture_path = args.next().map(PathBuf::from);
        if args.next().is_some() {
            return Err("too many trace arguments".into());
        }
        let model = metal::Model::load(
            &model_path,
            weights::LoadOptions {
                context_limit: 32,
                ..Default::default()
            },
        )?;
        let tiny = model.config().vocab_size == 64;
        let prompt = if tiny {
            vec![1, 7, 13, 3]
        } else {
            vec![9707, 11, 358, 2776]
        };
        let clock = Instant::now();
        let options = session::Options {
            capacity: 32,
            prefill_chunk_size: 4,
            ..Default::default()
        };
        let mut base = metal::Session::new(&model, options)?;
        // Account the shared model once while admitting a second session plus
        // counter allowance. Reference/session temporaries do not share KV/scratch.
        let remaining = weights::INITIAL_MEMORY_LIMIT_BYTES
            .checked_sub(base.memory().admitted_bytes - model.weights_bytes())
            .ok_or("combined admission overflow")?;
        let mut profiled = metal::Session::new(
            &model,
            session::Options {
                memory_limit_bytes: remaining,
                ..options
            },
        )?;
        profiled.enable_profile()?;
        let admitted = base
            .memory()
            .admitted_bytes
            .checked_add(profiled.memory().admitted_bytes)
            .and_then(|n| n.checked_sub(model.weights_bytes()))
            .ok_or("combined admission overflow")?;
        if admitted > weights::INITIAL_MEMORY_LIMIT_BYTES {
            return Err("combined trace admission exceeded".into());
        }
        // Four captured commands can snapshot their shared resources separately.
        // Reserve those four versions plus one working copy and native overhead.
        let capture_memory_allowance = if capture_path.is_some() {
            model
                .weights_bytes()
                .checked_add(base.memory().kv_bytes)
                .and_then(|n| n.checked_add(base.memory().scratch_bytes))
                .and_then(|n| n.checked_mul(5))
                .and_then(|n| n.checked_add(64 * 1024 * 1024))
                .ok_or("capture allowance overflow")?
        } else {
            0
        };
        let capture_admitted_bytes = admitted
            .checked_add(capture_memory_allowance)
            .ok_or("capture admission overflow")?;
        if capture_admitted_bytes > weights::INITIAL_MEMORY_LIMIT_BYTES {
            return Err("capture diagnostic admission exceeded".into());
        }
        let allocation_seconds = clock.elapsed().as_secs_f64();
        let mut out = io::stdout().lock();
        writeln!(
            out,
            "{}",
            json!({"event":"ready","model":if tiny {"tiny"} else {"qwen"},"device":model.device_name(),"load_seconds":model.load_timing.load_seconds,"conversion_seconds":model.load_timing.conversion_seconds,"compilation_seconds":model.load_timing.compilation_seconds,"session_allocation_seconds":allocation_seconds,"combined_admitted_bytes":admitted,"capture_memory_allowance_bytes":capture_memory_allowance,"capture_admitted_bytes":capture_admitted_bytes,"debug_assertions":cfg!(debug_assertions),"shared_weight_bytes":model.weights_bytes(),"baseline_memory":base.memory(),"profile_memory":profiled.memory(),"runs":14,"operations":["prefill","decode"],"events_per_operation":38*model.config().num_layers+6})
        )?;
        out.flush()?;
        handshake()?;
        for run in 0..14 {
            let capture = if run == 13 {
                capture_path
                    .as_ref()
                    .map(|p| model.capture(p))
                    .transpose()?
            } else {
                None
            };
            base.reset();
            profiled.reset();
            base.append(&prompt)?;
            profiled.append(&prompt)?;
            base.evaluate()?;
            profiled.evaluate()?;
            let token = sampling::greedy(base.logits().unwrap())?;
            if token != if tiny { 47 } else { 4460 } {
                return Err("frozen first greedy ID differs".into());
            }
            export(&mut out, run, "prefill", &base, &profiled)?;
            base.append(&[token])?;
            profiled.append(&[token])?;
            base.evaluate()?;
            profiled.evaluate()?;
            export(&mut out, run, "decode", &base, &profiled)?;
            drop(capture);
            writeln!(
                out,
                "{}",
                json!({"event":"idle","run":run,"phase":if run<3 {"warmup"} else if run<13 {"sample"} else {"capture"},"baseline_memory":base.memory(),"profile_memory":profiled.memory()})
            )?;
            out.flush()?;
            handshake()?;
        }
        writeln!(out, "{}", json!({"event":"done"}))?;
        out.flush()?;
        Ok(())
    }
}
fn main() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Err(error) = apple::run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        eprintln!("Metal trace requires Apple Silicon macOS");
        std::process::exit(1);
    }
}
