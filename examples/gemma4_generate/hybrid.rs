//! Explicit Mac diagnostic backend; shared generation/codec schedules stay above.
use super::*;
use std::os::unix::fs::OpenOptionsExt;
use toklet::metal::gemma_affine::{
    CONTEXT_ALLOWANCE_BYTES, Context,
    resident::{E2bWeights, LoadOptions, hybrid::Norms},
};

// Cold integration receives the same bounded metadata, codec and exact source
// identity as the scalar example. No generic kernel framework or app profile.
pub(super) fn execute(
    root: &Path,
    capture: &RefCell<Capture>,
    record: &mut serde_json::Value,
    profile: Profile,
    numerical_budget: usize,
    codec: Option<&Codec>,
    metadata: (&[u8], &[u8], [u8; 32]),
) -> Result<(), Box<dyn Error>> {
    let (config, index, digest) = metadata;
    // Scalar's historical raw protocol is unchanged. This new backend reserves
    // its diagnostic ownership explicitly before loading or session allocation.
    let budget = if profile == Profile::Raw {
        numerical_budget
            .checked_sub(DIAGNOSTIC_ALLOWANCE)
            .ok_or("hybrid diagnostic reserve overflow")?
    } else {
        numerical_budget
    };
    let source = objc2::rc::autoreleasepool(|_| {
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES)?;
        E2bWeights::load(
            &context,
            config,
            index,
            OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                .open(root.join("model.safetensors"))?,
            LoadOptions {
                container_bytes: 3550670554,
                container_sha256: digest,
                available_bytes: budget,
                modality: ModalityPolicy::TextOnly,
            },
        )
        .map_err(|e| Box::new(e) as Box<dyn Error>)
    })?;
    let norms = Norms::new(&source, budget)?;
    let model = norms.model()?;
    record["model_bytes_including_store"] = model.model_bytes().into();
    record["hybrid_norm_copy_bytes"] = norms.owned_bytes().into();
    record["construction_context_dropped"] = true.into();
    let context = Context::new(CONTEXT_ALLOWANCE_BYTES)?;
    let mut options = Options {
        capacity: 32,
        prefill_chunk_size: 4,
        position_offset: 0,
        memory_limit_bytes: budget,
    };
    let warm = norms.new_generation_session(&model, &context, options)?;
    let memory = warm.memory();
    drop(warm);
    options.memory_limit_bytes = memory.admitted_bytes - 1;
    let rejected = norms.new_generation_session(&model, &context, options);
    let intended = matches!(&rejected, Err(session::Error::MemoryBudget {required,limit})
        if *required == memory.admitted_bytes && *limit == memory.admitted_bytes - 1);
    record["generation_admission_probe"] = serde_json::json!({
        "required":memory.admitted_bytes,"limit":options.memory_limit_bytes,
        "minus_one_expected_budget_error":intended,"error":rejected.as_ref().err().map(|e|format!("{e:?}")),
        "exact_memory_unchanged":false});
    if !intended {
        return Err("hybrid generation admission did not return the expected budget error".into());
    }
    drop(rejected);
    options.memory_limit_bytes = memory.admitted_bytes;
    let mut session = norms.new_generation_session(&model, &context, options)?;
    if session.memory() != memory {
        return Err("hybrid exact-budget ownership differs".into());
    }
    record["generation_admission_probe"]["exact_memory_unchanged"] = true.into();
    if profile == Profile::Raw {
        record["numerical_budget_bytes"] = budget.into();
        record["diagnostic_allowance_bytes"] = DIAGNOSTIC_ALLOWANCE.into();
        record["combined_admitted_bytes"] =
            (session.memory().admitted_bytes + DIAGNOSTIC_ALLOWANCE).into();
    }
    run_session(&mut session, codec, budget, capture, record, profile)
}
