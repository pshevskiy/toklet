//! Frozen measured requests and teacher-forced/sustained guards, profiling off.
use crate::{
    app_chat::gemma4,
    generation::{self, Input, Request, StopReason},
    metal::gemma_affine::OfficialBf16,
    session::{InferenceSession, Options},
};
use sha2::{Digest, Sha256};
use std::{path::PathBuf, sync::atomic::AtomicBool, time::Instant};
const LIMIT: usize = crate::weights::INITIAL_MEMORY_LIMIT_BYTES;
const CALLER: usize = 16 * 1024 * 1024;
fn thermal() -> isize {
    objc2::rc::autoreleasepool(|_| {
        objc2_foundation::NSProcessInfo::processInfo()
            .thermalState()
            .0
    })
}
fn footprint() -> serde_json::Value {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: flavor4 matches this owned zeroed writable Apple structure. The
    // successful FFI copy initializes fields without aliases during the call.
    assert_eq!(
        unsafe {
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V4,
                info.as_mut_ptr().cast(),
            )
        },
        0
    );
    // SAFETY: successful flavor4 copy populated the zeroed structure above.
    let info = unsafe { info.assume_init() };
    assert!(
        info.ri_phys_footprint > 0
            && info.ri_phys_footprint <= info.ri_lifetime_max_phys_footprint
            && info.ri_lifetime_max_phys_footprint <= LIMIT as u64
    );
    serde_json::json!({"physical_bytes":info.ri_phys_footprint,"lifetime_peak_bytes":info.ri_lifetime_max_phys_footprint})
}
fn load() -> OfficialBf16 {
    let dir = PathBuf::from(std::env::var_os("TOKLET_GEMMA_OFFICIAL_DIRECTORY").unwrap());
    OfficialBf16::load(&dir, 512, LIMIT - CALLER).unwrap()
}
fn options() -> Options {
    Options {
        capacity: 512,
        prefill_chunk_size: 32,
        position_offset: 0,
        memory_limit_bytes: LIMIT - CALLER,
    }
}
fn prompt(codec: &gemma4::Codec) -> Vec<u32> {
    let p = codec
        .encode_chat(
            br#"[{"role":"user","content":"Name the capital of France."}]"#,
            gemma4::RenderOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            },
        )
        .unwrap();
    assert_eq!(
        p,
        [
            2, 105, 2364, 107, 1567, 506, 5279, 529, 7001, 236761, 106, 107, 105, 4368, 107
        ]
    );
    p
}
fn request(
    session: &mut impl InferenceSession,
    codec: &gemma4::Codec,
    prompt: &[u32],
    trial: usize,
    warmup: bool,
) -> serde_json::Value {
    let before = thermal();
    assert_eq!(before, 0);
    let mut ids = Vec::with_capacity(12);
    let clock = Instant::now();
    let report = generation::generate(
        session,
        Request {
            input: Input::Reset(prompt),
            max_new_tokens: 12,
            stop_tokens: codec.stop_ids(),
        },
        &AtomicBool::new(false),
        |e| {
            ids.push(e.token_id);
            Ok(())
        },
    )
    .unwrap();
    let caller_seconds = clock.elapsed().as_secs_f64();
    let after = thermal();
    assert_eq!(after, 0);
    assert_eq!(ids, [818, 5279, 529, 7001, 563, 5213, 50429, 84750]);
    assert_eq!(report.reason, StopReason::StopToken { token_id: 106 });
    assert_eq!(
        (
            report.reused_tokens,
            report.prefill_tokens,
            report.sampled_tokens,
            report.emitted_tokens
        ),
        (0, 15, 9, 8)
    );
    assert_eq!(
        (
            session.history().len(),
            session.evaluated_len(),
            session.pending_len()
        ),
        (24, 23, 1)
    );
    assert!(session.logits().is_none());
    let text = codec.decode(&ids, true).unwrap();
    assert_eq!(text, "The capital of France is **Paris**.");
    serde_json::json!({"trial":trial,"warmup":warmup,"report":report,"caller_seconds":caller_seconds,"ids":ids,"text":text,"thermal_before":before,"thermal_after":after})
}
fn observed_logits(session: &impl InferenceSession, bytes: &mut Vec<u8>) -> serde_json::Value {
    bytes.clear();
    for value in session.logits().unwrap() {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    let sha = format!("{:x}", Sha256::digest(&*bytes));
    let greedy = session.greedy_token().unwrap().unwrap();
    serde_json::json!({"sha256":sha,"greedy_id":greedy,"logit_bytes":bytes.len()})
}
#[test]
#[ignore = "Explicit frozen official BF16/G2 primary end-to-end guard candidate comparison"]
fn guard_short_requests() {
    let owner = load();
    let model = owner.model().unwrap();
    let p = prompt(owner.codec());
    let mut session = owner.new_session(&model, options()).unwrap();
    session.use_last_row_prefill();
    assert!(session.bf16_profile_status().is_none());
    let admission =
        session.memory().admitted_bytes + owner.memory().session_reserved_bytes + CALLER;
    assert!(admission <= LIMIT);
    let samples: Vec<_> = (0..13)
        .map(|trial| request(&mut session, owner.codec(), &p, trial, trial < 3))
        .collect();
    println!(
        "GUARD_PRIMARY {}",
        serde_json::json!({"passed":true,"warmups":3,"measured":10,"prompt_ids":p,"samples":samples,"admission_with_caller_bytes":admission,"memory":footprint()})
    );
}
#[test]
#[ignore = "Explicit admitted deep-context teacher-forced numerical/state/full-logit guard trajectories"]
fn guard_teacher_forced_trajectories() {
    let owner = load();
    let model = owner.model().unwrap();
    let p = prompt(owner.codec());
    {
        let mut warm = owner.new_session(&model, options()).unwrap();
        warm.use_last_row_prefill();
        for trial in 0..3 {
            request(&mut warm, owner.codec(), &p, trial, true);
        }
    }
    let text="// Public benchmark document.\nfn sum(values: &[u32]) -> u32 { values.iter().copied().sum() }\nlet answer = sum(&[1, 2, 3, 4]); assert_eq!(answer, 10);\n".repeat(64);
    let raw = owner.codec().encode_raw(&text, false).unwrap();
    assert!(raw.len() >= 511);
    let mut tokens = vec![2];
    tokens.extend_from_slice(&raw[..511]);
    let mut cases = Vec::new();
    let mut bytes = Vec::with_capacity(262_144 * 4);
    for input in [128, 384] {
        let mut session = owner.new_session(&model, options()).unwrap();
        session.use_last_row_prefill();
        assert!(session.bf16_profile_status().is_none());
        assert!(
            session.memory().admitted_bytes + owner.memory().session_reserved_bytes + CALLER
                <= LIMIT
        );
        session.append(&tokens[..input]).unwrap();
        let mut prefill = Vec::new();
        while session.pending_len() > 0 {
            let before = thermal();
            assert_eq!(before, 0);
            let clock = Instant::now();
            let rows = session.evaluate_next_chunk().unwrap();
            let seconds = clock.elapsed().as_secs_f64();
            let after = thermal();
            assert_eq!(after, 0);
            prefill.push(serde_json::json!({"rows":rows,"position":session.evaluated_len(),"seconds":seconds,"thermal_before":before,"thermal_after":after}));
        }
        // Pending prompt rows make logits unavailable; inspect only after the
        // complete prompt is committed, then after each one-token evaluation.
        let prefill_logits = observed_logits(&session, &mut bytes);
        let mut decode = Vec::new();
        for (step, &id) in tokens[input..input + 128].iter().enumerate() {
            session.append(&[id]).unwrap();
            let before = thermal();
            assert_eq!(before, 0);
            let clock = Instant::now();
            assert_eq!(session.evaluate_next_chunk().unwrap(), 1);
            let seconds = clock.elapsed().as_secs_f64();
            let after = thermal();
            assert_eq!(after, 0);
            let logits = observed_logits(&session, &mut bytes);
            decode.push(serde_json::json!({"step":step,"token_id":id,"position":session.evaluated_len(),"seconds":seconds,"logits":logits,"thermal_before":before,"thermal_after":after}));
        }
        assert_eq!(
            (
                session.history().len(),
                session.evaluated_len(),
                session.pending_len()
            ),
            (input + 128, input + 128, 0)
        );
        cases.push(serde_json::json!({"input":input,"output_evaluations":128,"prefill":prefill,"prefill_logits":prefill_logits,"decode":decode,"committed":session.evaluated_len()}));
    }
    println!(
        "GUARD_TRAJECTORIES {}",
        serde_json::json!({"passed":true,"warmup_requests":3,"tokens":tokens,"cases":cases,"memory":footprint(),"scope":"Teacher-forced raw public document, not chat quality or end-to-end generation. Completed numerical timers exclude full-logit digest collection."})
    );
}
#[test]
#[ignore = "Explicit at least ten-minute repeated exact BF16/G2 generation and memory/thermal guard"]
fn guard_sustained_requests() {
    let owner = load();
    let model = owner.model().unwrap();
    let p = prompt(owner.codec());
    let mut session = owner.new_session(&model, options()).unwrap();
    session.use_last_row_prefill();
    assert!(session.bf16_profile_status().is_none());
    assert!(
        session.memory().admitted_bytes + owner.memory().session_reserved_bytes + CALLER <= LIMIT
    );
    for trial in 0..3 {
        request(&mut session, owner.codec(), &p, trial, true);
    }
    let mut samples = Vec::with_capacity(1024);
    let mut memory = Vec::new();
    let clock = Instant::now();
    while clock.elapsed().as_secs_f64() < 600. {
        assert!(
            samples.len() < 2048,
            "bounded sustained sample capacity exhausted"
        );
        let trial = samples.len();
        samples.push(request(&mut session, owner.codec(), &p, trial, false));
        if trial % 16 == 0 {
            memory.push(serde_json::json!({"trial":trial,"memory":footprint()}));
        }
    }
    let wall_seconds = clock.elapsed().as_secs_f64();
    assert!(wall_seconds >= 600.);
    println!(
        "GUARD_SUSTAINED {}",
        serde_json::json!({"passed":true,"warmups":3,"samples":samples,"wall_seconds":wall_seconds,"memory_samples":memory,"memory":footprint(),"scope":"Repeated complete Reset generation with exact output/state; interval includes outside-timer verification/recording. No tail/energy/UI qualification."})
    );
}
