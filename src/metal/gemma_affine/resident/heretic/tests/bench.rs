//! Same-source completed console requests and predeclared All/G2 comparisons.
use super::*;
use std::time::Instant;

fn availability() -> serde_json::Value {
    serde_json::json!({"operator_profile":null,"operator_profile_reason":"unsupported for this mixed backend",
        "command_buffers":null,"host_waits":null,"copy_bytes":null,"counter_reason":"not instrumented; no zero-filled counters",
        "steady_state_allocations":null,"allocation_reason":"not instrumented",
        "ttft_visible":null,"ui_reason":"bare token engine, no renderer",
        "energy":null,"energy_reason":"no qualified energy instrument"})
}

fn mode() -> String {
    let m = std::env::var("TOKLET_HERETIC_PREFILL").unwrap();
    assert!(matches!(m.as_str(), "all" | "last"));
    m
}
fn session<'m, 'a, 'w, 'c>(
    norms: &'a Norms<'w>,
    model: &'m Model<'a, 'w>,
    context: &'c HereticContext,
    mode: &str,
) -> GenerationSession<'m, 'a, 'w, 'c> {
    if mode == "all" {
        norms
            .reference_generation_session(model, context, opts())
            .unwrap()
    } else {
        norms
            .new_generation_session(model, context, opts())
            .unwrap()
    }
}
fn thermal() -> isize {
    objc2::rc::autoreleasepool(|_| {
        objc2_foundation::NSProcessInfo::processInfo()
            .thermalState()
            .0
    })
}
fn request(s: &mut impl InferenceSession, trial: usize, warmup: bool) -> serde_json::Value {
    let before = thermal();
    assert_eq!(before, 0);
    let mut ids = Vec::with_capacity(12);
    let mut last_token_event_seconds = None;
    let clock = Instant::now();
    let r = generation::generate(
        s,
        Request {
            input: Input::Reset(&PROMPT),
            max_new_tokens: 12,
            stop_tokens: &[106],
        },
        &AtomicBool::new(false),
        |e| {
            ids.push(e.token_id);
            last_token_event_seconds = Some(e.elapsed_seconds);
            Ok(())
        },
    )
    .unwrap();
    let caller_seconds = clock.elapsed().as_secs_f64();
    let after = thermal();
    assert_eq!(after, 0);
    assert_eq!((r.reused_tokens, r.prefill_tokens), (0, 15));
    assert_eq!(r.emitted_tokens, ids.len());
    assert_eq!(r.history_tokens, 15 + r.sampled_tokens);
    assert_eq!(r.pending_tokens, 1);
    assert_eq!(r.evaluated_tokens + 1, r.history_tokens);
    assert_eq!(
        (s.history().len(), s.evaluated_len(), s.pending_len()),
        (r.history_tokens, r.evaluated_tokens, 1)
    );
    let event_decode_tokens_per_second = match (r.first_token_seconds, last_token_event_seconds) {
        (Some(first), Some(last)) if ids.len() > 1 && last > first => {
            Some((ids.len() - 1) as f64 / (last - first))
        }
        _ => None,
    };
    let decode_evaluations = r.evaluated_tokens - r.reused_tokens - r.prefill_tokens;
    let numerical_decode_evaluations_per_second = (decode_evaluations > 0 && r.decode_seconds > 0.)
        .then(|| decode_evaluations as f64 / r.decode_seconds);
    let prefill_tokens_per_second = (r.prefill_tokens > 0 && r.prefill_seconds > 0.)
        .then(|| r.prefill_tokens as f64 / r.prefill_seconds);
    serde_json::json!({"trial":trial,"warmup":warmup,"ids":ids,"report":r,"caller_seconds":caller_seconds,"thermal_before":before,"thermal_after":after,
        "last_token_event_seconds":last_token_event_seconds,"event_decode_tokens_per_second":event_decode_tokens_per_second,
        "event_decode_rate_reason":if event_decode_tokens_per_second.is_none() {Some("fewer than two events or zero event interval")} else {None},
        "decode_evaluations":decode_evaluations,"numerical_decode_evaluations_per_second":numerical_decode_evaluations_per_second,
        "prefill_tokens_per_second":prefill_tokens_per_second})
}

#[test]
#[ignore = "explicit matched3warm+10 release native Heretic All/G2 console requests"]
fn matched_console_requests() {
    let mode = mode();
    let context = HereticContext::new(CONTEXT_BYTES).unwrap();
    let memory_before_load = footprint();
    let source = load(&context);
    let memory_after_load = footprint();
    let norms = Norms::new(&source, LIMIT - CALLER).unwrap();
    let model = norms.model(512).unwrap();
    let start = Instant::now();
    let mut s = session(&norms, &model, &context, &mode);
    let session_create_seconds = start.elapsed().as_secs_f64();
    assert!(s.memory().admitted_bytes + CALLER <= LIMIT);
    let samples: Vec<_> = (0..13).map(|i| request(&mut s, i, i < 3)).collect();
    for v in &samples {
        assert_eq!(v["ids"], samples[0]["ids"]);
        assert_eq!(v["report"]["reason"], samples[0]["report"]["reason"]);
    }
    println!(
        "HERETIC_PRIMARY {}",
        serde_json::json!({"format":"toklet-heretic-primary-v2","passed":true,"mode":mode,"availability":availability(),"memory_before_load":memory_before_load,"memory_after_load":memory_after_load,"warmups":3,"measured":10,"samples":samples,"source":source.identity(),"load":source.report(),"context_creation_seconds":context.creation_seconds(),"session_create_seconds":session_create_seconds,"generation_memory":s.memory(),"caller_reserve_bytes":CALLER,"memory":footprint(),"scope":"Bare numerical token requests; no tokenizer/UI/quality/model-adoption qualification"})
    );
}

fn observed(s: &impl InferenceSession) -> serde_json::Value {
    let row = s.logits().unwrap();
    serde_json::json!({"sha256":digest(row),"greedy_id":greedy(row)})
}
#[test]
#[ignore = "explicit128/384input+128teacherforced native Heretic All/G2 full-logit trajectories"]
fn matched_trajectories() {
    use std::io::Read;
    let mode = mode();
    let context = HereticContext::new(CONTEXT_BYTES).unwrap();
    let memory_before_load = footprint();
    let source = load(&context);
    let memory_after_load = footprint();
    let norms = Norms::new(&source, LIMIT - CALLER).unwrap();
    let model = norms.model(512).unwrap();
    {
        let mut warm = session(&norms, &model, &context, &mode);
        for i in 0..3 {
            request(&mut warm, i, true);
        }
    }
    let p = PathBuf::from(std::env::var_os("TOKLET_HERETIC_TRAJECTORY_TOKENS").unwrap());
    let mut raw = Vec::new();
    std::fs::File::open(p)
        .unwrap()
        .take(2049)
        .read_to_end(&mut raw)
        .unwrap();
    assert_eq!(raw.len(), 2048);
    let expected = "a26a7e3d57f7926ca46ad11e782a76a32b534d7d8951bd6081ad360be87bf7f4";
    assert_eq!(format!("{:x}", Sha256::digest(&raw)), expected);
    let tokens: Vec<u32> = raw
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut cases = Vec::new();
    for input in [128, 384] {
        let mut s = session(&norms, &model, &context, &mode);
        assert!(s.memory().admitted_bytes + CALLER <= LIMIT);
        s.append(&tokens[..input]).unwrap();
        let mut prefill = Vec::new();
        while s.pending_len() > 0 {
            assert_eq!(thermal(), 0);
            let clock = Instant::now();
            let rows = s.evaluate_next_chunk().unwrap();
            let seconds = clock.elapsed().as_secs_f64();
            assert_eq!(thermal(), 0);
            prefill.push(
                serde_json::json!({"rows":rows,"position":s.evaluated_len(),"seconds":seconds}),
            );
        }
        let prefill_logits = observed(&s);
        let mut decode = Vec::new();
        for (step, &token) in tokens[input..input + 128].iter().enumerate() {
            s.append(&[token]).unwrap();
            assert_eq!(thermal(), 0);
            let clock = Instant::now();
            assert_eq!(s.evaluate_next_chunk().unwrap(), 1);
            let seconds = clock.elapsed().as_secs_f64();
            assert_eq!(thermal(), 0);
            decode.push(serde_json::json!({"step":step,"token_id":token,"position":s.evaluated_len(),"seconds":seconds,"logits":observed(&s)}));
        }
        assert_eq!(
            (s.history().len(), s.evaluated_len(), s.pending_len()),
            (input + 128, input + 128, 0)
        );
        cases.push(serde_json::json!({"input":input,"output_evaluations":128,"prefill":prefill,"prefill_logits":prefill_logits,"decode":decode,"committed":s.evaluated_len()}));
    }
    println!(
        "HERETIC_TRAJECTORIES {}",
        serde_json::json!({"format":"toklet-heretic-trajectories-v2","passed":true,"mode":mode,"availability":availability(),"memory_before_load":memory_before_load,"memory_after_load":memory_after_load,"warmup_requests":3,"tokens":tokens,"cases":cases,"source":source.identity(),"memory":footprint(),"scope":"Completed numerical timers exclude digest collection; public document, not held-out quality."})
    );
}

#[test]
#[ignore = "explicit>=600second bounded native Heretic All/G2 Reset stability run"]
fn sustained_console_requests() {
    let mode = mode();
    let context = HereticContext::new(CONTEXT_BYTES).unwrap();
    let memory_before_load = footprint();
    let source = load(&context);
    let memory_after_load = footprint();
    let norms = Norms::new(&source, LIMIT - CALLER).unwrap();
    let model = norms.model(512).unwrap();
    let mut s = session(&norms, &model, &context, &mode);
    assert!(s.memory().admitted_bytes + CALLER <= LIMIT);
    for i in 0..3 {
        request(&mut s, i, true);
    }
    let mut samples = Vec::with_capacity(2048);
    let mut memory = Vec::with_capacity(129);
    let clock = Instant::now();
    while clock.elapsed().as_secs_f64() < 600. {
        assert!(samples.len() < 2048);
        let sample = request(&mut s, samples.len(), false);
        if let Some(first) = samples.first() {
            let first: &serde_json::Value = first;
            assert_eq!(sample["ids"], first["ids"]);
            assert_eq!(sample["report"]["reason"], first["report"]["reason"]);
        }
        samples.push(sample);
        if samples.len().is_multiple_of(16) {
            memory.push(serde_json::json!({"requests":samples.len(),"memory":footprint()}));
        }
    }
    println!(
        "HERETIC_SUSTAINED {}",
        serde_json::json!({"format":"toklet-heretic-sustained-v2","passed":true,"mode":mode,"availability":availability(),"memory_before_load":memory_before_load,"memory_after_load":memory_after_load,"warmup_requests":3,"wall_seconds":clock.elapsed().as_secs_f64(),"samples":samples,"memory_samples":memory,"source":source.identity(),"generation_memory":s.memory(),"caller_reserve_bytes":CALLER,"memory":footprint(),"scope":"At least600s Reset stability only; no tail/energy/UI/full-quality claim."})
    );
}
