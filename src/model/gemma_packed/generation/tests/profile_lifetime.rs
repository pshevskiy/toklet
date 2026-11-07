//! Repeated native diagnostic calls on a Rust worker without an outer pool.
use crate::{
    app_chat::gemma4,
    metal::gemma_affine::{Bf16CaptureState, Bf16ProfileOptions, OfficialBf16},
    session::Options,
};
use std::path::PathBuf;
const LIMIT: usize = crate::weights::INITIAL_MEMORY_LIMIT_BYTES;
const CALLER: usize = 16 * 1024 * 1024;
const ALLOWANCE: u64 = 64 * 1024 * 1024;
fn footprint() -> (u64, u64) {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: flavor4 matches this owned zeroed writable Apple structure.
    // The successful FFI copy initializes fields without aliases during the call.
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
    (info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint)
}
#[test]
#[ignore = "Explicit repeated BF16 diagnostic constructor/export native lifetime check on a Rust worker"]
fn official_bf16_profile_temporaries_stay_bounded_without_outer_pool() {
    let directory = PathBuf::from(std::env::var_os("TOKLET_GEMMA_OFFICIAL_DIRECTORY").unwrap());
    // Create/retain the owner inside the worker; neither the test harness nor
    // an app can provide a persistent outer autorelease pool for these calls.
    let result=std::thread::spawn(move || {
        let owner=OfficialBf16::load(&directory,512,LIMIT-CALLER).unwrap();
        let model=owner.model().unwrap();
        let options=Options {capacity:512,prefill_chunk_size:32,position_offset:0,memory_limit_bytes:LIMIT-CALLER};
        let profile=Bf16ProfileOptions {request_id:[7;16],evaluation_ordinal:1};
        for _ in 0..32 {
            let session=owner.new_profiled_session(&model,options,profile).unwrap();
            assert_eq!(session.bf16_profile_status().unwrap().state,Bf16CaptureState::Armed);
        }
        let constructor_before=footprint().0;let mut constructor_samples=Vec::new();
        for iteration in 1..=512 {
            {
                let session=owner.new_profiled_session(&model,options,profile).unwrap();
                assert!(session.memory().admitted_bytes+owner.memory().session_reserved_bytes+CALLER<=LIMIT);
                assert_eq!(session.bf16_profile_status().unwrap().state,Bf16CaptureState::Armed);
            }
            if iteration%32==0 {constructor_samples.push(footprint().0);}
        }
        let constructor_max=*constructor_samples.iter().max().unwrap();
        assert!(constructor_max<=constructor_before+ALLOWANCE,"constructor native footprint grows beyond diagnostic allowance");
        let mut session=owner.new_profiled_session(&model,options,profile).unwrap();
        let prompt=owner.codec().encode_chat(br#"[{"role":"user","content":"Name the capital of France."}]"#,
            gemma4::RenderOptions {enable_thinking:false,add_generation_prompt:true}).unwrap();
        session.append(&prompt).unwrap();assert_eq!(session.evaluate_next_chunk().unwrap(),15);
        assert_eq!(session.evaluated_len(),15);
        for _ in 0..64 {let frame=session.bf16_profile().unwrap();assert_eq!(frame.tiles.len(),349);}
        let export_before=footprint().0;let mut export_samples=Vec::new();
        for iteration in 1..=16_384 {
            {
                let frame=session.bf16_profile().unwrap();
                assert_eq!(frame.request_id,[7;16]);
                assert_eq!(frame.status.state,Bf16CaptureState::Complete);
                assert_eq!((frame.tiles.len(),frame.clocks.events.len(),frame.status.gathers_recorded),(349,349,2));
            }
            if iteration%1024==0 {export_samples.push(footprint().0);}
        }
        let export_max=*export_samples.iter().max().unwrap();
        assert!(export_max<=export_before+ALLOWANCE,"export native footprint grows beyond diagnostic allowance");
        assert_eq!((session.evaluated_len(),session.pending_len()),(15,0));
        let (_,peak)=footprint();
        serde_json::json!({"passed":true,"outer_autorelease_pool":false,"constructor_warmups":32,
            "constructor_repetitions":512,"constructor_before_bytes":constructor_before,
            "constructor_samples_bytes":constructor_samples,"constructor_growth_bytes":constructor_max.saturating_sub(constructor_before),
            "export_warmups":64,"export_repetitions":16384,"export_before_bytes":export_before,
            "export_samples_bytes":export_samples,"export_growth_bytes":export_max.saturating_sub(export_before),
            "growth_limit_bytes":ALLOWANCE,"process_lifetime_peak_bytes":peak,"timing_qualified":false,
            "scope":"Observed repeated native lifetime bound on this pinned Mac; no continuous allocator or general driver-memory guarantee."})
    }).join().unwrap();
    println!("BF16_PROFILE_LIFETIME {result}");
}
