//! Explicit authenticated2B hybrid transactions; never an ordinary CPU test.
#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
use super::*;
use crate::session::InferenceSession;

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "explicit authenticated2B hybrid state/failure/repair diagnostic"]
fn actual_source_transactions_and_repair() {
    use crate::{
        metal::gemma_affine::{CONTEXT_ALLOWANCE_BYTES, Context, with_projection_failure},
        mlx::qwen35::{ModalityPolicy, two_b},
    };
    use std::{fs::File, io::Read, path::PathBuf};
    fn captured(path: &std::path::Path, limit: usize) -> Vec<u8> {
        let mut file = crate::weights::audit::open_regular(path).unwrap();
        let size = file.metadata().unwrap().len();
        assert!(size > 0 && size <= limit as u64);
        let mut raw = Vec::new();
        Read::by_ref(&mut file)
            .take((limit + 1) as u64)
            .read_to_end(&mut raw)
            .unwrap();
        assert_eq!(raw.len() as u64, size);
        assert_eq!(file.metadata().unwrap().len(), size);
        raw
    }
    let directory = PathBuf::from(std::env::var_os("TOKLET_QWEN35_2B_MODEL_DIRECTORY").unwrap());
    let manifest_path =
        PathBuf::from(std::env::var_os("TOKLET_QWEN35_2B_ARTIFACT_MANIFEST").unwrap());
    let output = PathBuf::from(std::env::var_os("TOKLET_QWEN35_2B_TRANSACTION_CAPTURE").unwrap());
    let expected = std::env::var("TOKLET_QWEN35_2B_ARTIFACT_MANIFEST_SHA256").unwrap();
    assert_eq!(expected.len(), 64);
    assert!(
        expected
            .bytes()
            .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v))
    );
    let mut digest = [0; 32];
    for (n, item) in digest.iter_mut().enumerate() {
        *item = u8::from_str_radix(&expected[2 * n..2 * n + 2], 16).unwrap();
    }
    let raw = captured(&manifest_path, 64 * 1024);
    let pins = two_b::artifact::pins(&raw, digest).unwrap();
    let config = captured(&directory.join("config.json"), pins.config_bytes);
    let budget = MEMORY_LIMIT - 128 * 1024 * 1024;
    let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let store = two_b::payload::load(
        &config,
        crate::weights::audit::open_regular(&directory.join("model.safetensors")).unwrap(),
        two_b::payload::LoadOptions {
            memory_limit_bytes: budget - CONTEXT_ALLOWANCE_BYTES,
            modality: ModalityPolicy::TextOnly,
        },
        &pins,
    )
    .unwrap();
    let store = store.into_metal(&context, budget).unwrap();
    assert_eq!(store.report().retained_cpu_matrix_bytes, 0);
    let model = MetalModel::from_two_b_metal_store(&store).unwrap();
    assert!(model.output_head.is_none());
    let opts = Options {
        capacity: 128,
        chunk: 32,
        position_offset: 0,
        memory_limit_bytes: budget,
    };
    let planned = model.planned_session_memory(opts).unwrap();
    assert!(matches!(
        model.session(
            &context,
            Options {
                memory_limit_bytes: planned.admitted_bytes - 1,
                ..opts
            }
        ),
        Err(Error::MemoryBudget { .. })
    ));
    let mut clean = model.session(&context, opts).unwrap();
    clean.forward(&[1]).unwrap();
    let prefix = snapshot(&clean);
    let prefix_logits = clean.logits().unwrap().to_vec();
    clean.forward(&[7, 13]).unwrap();
    let target = snapshot(&clean);
    let mut controls = Vec::new();
    for stage in ["delta_state", "attention_scores", "logits", "before_commit"] {
        let mut session = model.session(&context, opts).unwrap();
        session.forward(&[1]).unwrap();
        let memory = session.memory();
        let result = session.forward_observed(&[7, 13], &AtomicBool::new(false), &mut |o| {
            if o.stage == stage {
                Err(Error::Observation("2B transaction control"))
            } else {
                Ok(())
            }
        });
        assert!(matches!(
            result,
            Err(Error::Observation("2B transaction control"))
        ));
        assert_eq!(snapshot(&session), prefix);
        assert_float_bits(session.logits().unwrap(), &prefix_logits);
        assert_eq!(session.work().tokens_committed, 0);
        assert_eq!(session.memory(), memory);
        let failed_work = session.work();
        session.forward(&[7, 13]).unwrap();
        assert_eq!(snapshot(&session), target);
        controls.push(serde_json::json!({"control":"late_observer_failure","stage":stage,"prefix_sha256":prefix,"retry_sha256":target,"failed_work":failed_work}));
    }
    let mut session = model.session(&context, opts).unwrap();
    session.forward(&[1]).unwrap();
    assert!(matches!(
        session.forward_observed(&[7], &AtomicBool::new(true), &mut |_| Ok(())),
        Err(Error::Cancelled)
    ));
    assert_eq!(snapshot(&session), prefix);
    assert_eq!(session.work().tokens_committed, 0);
    let cancel = AtomicBool::new(false);
    assert!(matches!(
        session.forward_observed(&[7, 13], &cancel, &mut |o| {
            if o.stage == "before_commit" {
                cancel.store(true, Ordering::Relaxed);
            }
            Ok(())
        }),
        Err(Error::Cancelled)
    ));
    assert_eq!(snapshot(&session), prefix);
    assert_eq!(session.work().tokens_committed, 0);
    session.forward(&[7, 13]).unwrap();
    assert_eq!(snapshot(&session), target);
    session.reset();
    session.forward(&[1]).unwrap();
    with_projection_failure(2, || assert!(session.forward(&[7]).is_err()));
    assert!(session.requires_repair());
    assert_eq!(snapshot(&session), prefix);
    let poisoned_work = session.work();
    assert!(matches!(
        session.forward(&[7]),
        Err(Error::ExecutionPoisoned)
    ));
    let execution_bytes = session.execution.owned_bytes().unwrap();
    let overlap = session
        .memory()
        .admitted_bytes
        .checked_add(execution_bytes)
        .unwrap();
    session.options.memory_limit_bytes = overlap - 1;
    assert!(
        matches!(session.repair_execution(&context), Err(Error::MemoryBudget{required,limit}) if required==overlap && limit+1==overlap)
    );
    assert!(session.requires_repair());
    assert_eq!(snapshot(&session), prefix);
    session.options.memory_limit_bytes = budget;
    session.repair_execution(&context).unwrap();
    assert!(!session.requires_repair());
    assert_eq!(snapshot(&session), prefix);
    session.forward(&[7, 13]).unwrap();
    assert_eq!(snapshot(&session), target);
    let mut generation =
        GenerationSession::new(model.session(&context, opts).unwrap(), budget).unwrap();
    generation.set_tokens(&[1]).unwrap();
    generation.evaluate_next_chunk().unwrap();
    generation.append(&[7]).unwrap();
    assert_eq!(
        (
            generation.state().committed_tokens,
            generation.state().pending_tokens
        ),
        (1, 1)
    );
    generation.set_tokens(&[1]).unwrap();
    assert_eq!(generation.state().prefix_action, PrefixAction::FullyCached);
    generation.set_tokens(&[1, 13]).unwrap();
    generation.evaluate_next_chunk().unwrap();
    assert_eq!(generation.state().prefix_action, PrefixAction::ExactPrefix);
    generation.set_tokens(&[2]).unwrap();
    assert_eq!(
        generation.state().prefix_action,
        PrefixAction::ChangedReplay
    );
    assert_eq!(
        (
            generation.state().committed_tokens,
            generation.state().pending_tokens
        ),
        (0, 1)
    );
    generation.evaluate_next_chunk().unwrap();
    let report = serde_json::json!({"format":"toklet-qwen35-2b-transactions-v1","artifact_manifest_sha256":expected,
        "capacity":128,"chunk":32,"memory":session.memory(),"upload_report":store.report(),
        "controls":controls,"cancel_before_and_late_prefix_preserved":true,"poisoned_work":poisoned_work,
        "poisoned_prefix_preserved":true,"repair_overlap_minus_one_rejected":true,"repair_retry_exact":true,
        "prefix_sha256":prefix,"retry_sha256":target,"generation_memory":generation.memory_report(),
        "generation_state":generation.state(),"full_model_numerical_quality_performance_adoption_qualified":false});
    std::fs::create_dir(&output).unwrap();
    serde_json::to_writer_pretty(File::create(output.join("result.json")).unwrap(), &report)
        .unwrap();
    println!("QWEN35_2B_TRANSACTION_REPORT {report}");
}
