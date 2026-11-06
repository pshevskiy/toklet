//! Same-owner official BF16 candidate accounting, distinct from dense reports.
use super::*;

pub(super) fn plan(
    model: &ModelHandle,
    current: &SessionHandle,
    state: &OwnedSession,
    candidate: session::Options,
    request: TransitionPlanningOptions,
) -> Result<GemmaBf16TransitionInfo> {
    validate_transition_owners(model, current, state, request)?;
    let (NativeModel::OfficialGemma(model_owner), NativeSession::OfficialGemma(current_owner)) =
        (model.model.as_ref(), &state.session)
    else {
        return Err(fail(
            UNSUPPORTED,
            "official Gemma BF16 transition requires a paired model and session",
        ));
    };
    let shared = model_owner.shared_model_bytes()?;
    let current_combined = checked_sum(&[current_owner.admitted_bytes, BRIDGE_BYTES])?;
    let candidate_combined = checked_sum(&[
        model_owner.planned_session_admission(candidate)?,
        BRIDGE_BYTES,
    ])?;
    // The candidate still carries its full standalone context/driver allowance.
    // Subtract only the exact common model graph and already-reserved codec.
    let incremental = candidate_combined.checked_sub(shared).ok_or_else(|| {
        fail(
            ENGINE_ERROR,
            "Gemma candidate is smaller than its shared owner",
        )
    })?;
    let reserve = |value| {
        usize::try_from(value)
            .map_err(|_| fail(INVALID, "Gemma transition reserve exceeds address width"))
    };
    let aggregate = checked_sum(&[
        current_combined,
        incremental,
        reserve(request.replay_reserve_bytes)?,
        reserve(request.text_reserve_bytes)?,
        reserve(request.application_reserve_bytes)?,
    ])?;
    let kv = crate::model::gemma_packed::official_bf16_planned_kv(
        candidate.capacity,
        candidate.prefill_chunk_size,
    )
    .map_err(|error| fail(INVALID, error.to_string()))?;
    Ok(GemmaBf16TransitionInfo {
        abi_version: ABI_VERSION,
        struct_size: size_of::<GemmaBf16TransitionInfo>() as u32,
        model_instance: model.instance,
        session_instance: current.instance,
        candidate_capacity: candidate.capacity as u32,
        candidate_chunk: candidate.prefill_chunk_size as u32,
        current_combined_bytes: current_combined as u64,
        candidate_combined_bytes: candidate_combined as u64,
        candidate_incremental_bytes: incremental as u64,
        candidate_kv_bytes: kv as u64,
        aggregate_required_bytes: aggregate as u64,
    })
}

pub(super) fn readmit(plan: &GemmaBf16TransitionInfo, candidate: &SessionHandle) -> Result<()> {
    let state = candidate
        .state
        .try_borrow()
        .map_err(|_| fail(BUSY, "Gemma candidate is busy"))?;
    let NativeSession::OfficialGemma(candidate_owner) = &state.session else {
        return Err(fail(ENGINE_ERROR, "Gemma candidate backend differs"));
    };
    let actual = checked_sum(&[candidate_owner.admitted_bytes, BRIDGE_BYTES])? as u64;
    if state.model_instance != plan.model_instance
        || candidate_owner.inference.capacity() as u64 != u64::from(plan.candidate_capacity)
        || actual > plan.candidate_combined_bytes
        || candidate_owner.inference.memory().kv_bytes as u64 != plan.candidate_kv_bytes
    {
        return Err(fail(
            ENGINE_ERROR,
            "actual Gemma candidate differs from the approved plan",
        ));
    }
    Ok(())
}
