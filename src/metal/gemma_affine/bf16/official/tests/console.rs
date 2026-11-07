//! Explicit real-model console smoke, independent of Swift, UI and capture.
use super::*;
use std::time::Instant;

#[test]
#[ignore = "Explicit offline official BF16/G2 console generation and state witness"]
fn official_bf16_console_smoke() {
    let directory = std::path::PathBuf::from(
        std::env::var_os("TOKLET_GEMMA_OFFICIAL_DIRECTORY").expect("pinned offline source"),
    );
    let start = Instant::now();
    let owner = OfficialBf16::load(&directory, 512, INITIAL_MEMORY_LIMIT_BYTES).unwrap();
    let load_seconds = start.elapsed().as_secs_f64();
    let model = owner.model().unwrap();
    let options = Options {
        capacity: 512,
        prefill_chunk_size: 32,
        position_offset: 0,
        memory_limit_bytes: INITIAL_MEMORY_LIMIT_BYTES,
    };
    let mut session = owner.new_session(&model, options).unwrap();
    session.use_last_row_prefill();
    let admission = session.memory().admitted_bytes + owner.memory().session_reserved_bytes;
    assert!(admission <= INITIAL_MEMORY_LIMIT_BYTES);
    let start = Instant::now();
    let prompt = owner
        .codec()
        .encode_chat(
            br#"[{"role":"user","content":"Name the capital of France."}]"#,
            gemma4::RenderOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            },
        )
        .unwrap();
    let tokenization_seconds = start.elapsed().as_secs_f64();
    assert_eq!(prompt.len(), 15);
    let cohort = std::env::var("TOKLET_GEMMA_CONSOLE_COHORT").ok();
    if let Some(cohort) = &cohort {
        assert!(
            matches!(cohort.as_str(), "1" | "2"),
            "explicit benchmark cohort"
        );
    }
    let warmups = if cohort.is_some() { 3 } else { 0 };
    let trials = if cohort.is_some() { 13 } else { 2 };
    let expected = [818, 5279, 529, 7001, 563, 5213, 50429, 84750];
    let mut samples = Vec::with_capacity(trials);
    for trial in 0..trials {
        let mut ids = Vec::with_capacity(12);
        let result = generation::generate(
            &mut session,
            Request {
                input: Input::Reset(&prompt),
                max_new_tokens: 12,
                stop_tokens: owner.codec().stop_ids(),
            },
            &AtomicBool::new(false),
            |event| {
                ids.push(event.token_id);
                Ok(())
            },
        )
        .unwrap();
        let text = owner.codec().decode(&ids, true).unwrap();
        println!("CONSOLE_ANSWER trial={trial}: {text}");
        assert_eq!(ids, expected);
        assert_eq!(text, "The capital of France is **Paris**.");
        assert!(
            matches!(result.reason,StopReason::StopToken {token_id} if owner.codec().stop_ids().contains(&token_id))
        );
        assert_eq!(
            (
                result.prefill_tokens,
                result.reused_tokens,
                result.sampled_tokens,
                result.emitted_tokens
            ),
            (15, 0, 9, 8)
        );
        assert_eq!(
            (
                session.history().len(),
                session.evaluated_len(),
                session.pending_len()
            ),
            (24, 23, 1)
        );
        samples.push(serde_json::json!({"trial":trial,"warmup":trial<warmups,
            "metrics":result,"emitted_ids":ids,"text":text,
            "thermal_state":objc2_foundation::NSProcessInfo::processInfo().thermalState().0}));
    }
    let before = session.history().to_vec();
    let cancelled = generation::generate(
        &mut session,
        Request {
            input: Input::Reset(&prompt),
            max_new_tokens: 12,
            stop_tokens: owner.codec().stop_ids(),
        },
        &AtomicBool::new(true),
        |_| panic!("pre-start cancellation emitted output"),
    )
    .unwrap();
    assert_eq!(cancelled.reason, StopReason::Cancelled);
    assert_eq!(session.history(), before);
    let full = vec![2; 512];
    assert!(
        generation::generate(
            &mut session,
            Request {
                input: Input::Reset(&full),
                max_new_tokens: 1,
                stop_tokens: owner.codec().stop_ids()
            },
            &AtomicBool::new(false),
            |_| panic!("over-capacity request emitted output")
        )
        .is_err()
    );
    assert_eq!(session.history(), before);
    assert_eq!((session.evaluated_len(), session.pending_len()), (23, 1));
    let mut footprint = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: flavor4 matches the owned, zeroed, writable libc/Apple structure;
    // the successful copy initializes its fields, with no aliases during FFI.
    assert_eq!(
        unsafe {
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V4,
                footprint.as_mut_ptr().cast(),
            )
        },
        0
    );
    // SAFETY: successful flavor4 call above populated the zeroed structure.
    let footprint = unsafe { footprint.assume_init() };
    assert!(
        footprint.ri_phys_footprint > 0
            && footprint.ri_phys_footprint <= footprint.ri_lifetime_max_phys_footprint
            && footprint.ri_lifetime_max_phys_footprint <= INITIAL_MEMORY_LIMIT_BYTES as u64
    );
    println!(
        "GEMMA_CONSOLE_SMOKE {}",
        serde_json::json!({"route":"official BF16/G2 Mac","repository":OfficialBf16::REPOSITORY,
        "revision":OfficialBf16::REVISION,"capacity":512,"chunk":32,"prompt_tokens":prompt.len(),"load_seconds":load_seconds,
        "tokenization_seconds":tokenization_seconds,"samples":samples,"session_and_owner_admitted_bytes":admission,
        "cohort":cohort,"warmups":warmups,"measured_samples":trials-warmups,"prompt_ids":prompt,
        "rendered_prompt":gemma4::render(br#"[{"role":"user","content":"Name the capital of France."}]"#,
            gemma4::RenderOptions {enable_thinking:false,add_generation_prompt:true}).unwrap(),
        "physical_bytes":footprint.ri_phys_footprint,"process_lifetime_peak_bytes":footprint.ri_lifetime_max_phys_footprint,
        "prestart_cancellation_preserves_state":true,"capacity_rejection_preserves_state":true,
        "timing_scope":if cohort.is_some(){"Frozen console cohort; compare only matched precision/work/cache scopes"}
            else{"Two smoke samples only; not benchmark or performance acceptance"}})
    );
}
