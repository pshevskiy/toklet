//! Model-bound, non-reserving old-plus-candidate session accounting.
use super::*;
use serde::Serialize;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TransitionPlanningOptions {
    pub abi_version: u32,
    pub struct_size: u32,
    pub expected_model_instance: u64,
    pub expected_session_instance: u64,
    /// Aggregate budget, including both sessions and all named reserves. Zero
    /// produces a false-fit report. No physical/device budget is inferred.
    pub transition_budget_bytes: u64,
    pub replay_reserve_bytes: u64,
    pub text_reserve_bytes: u64,
    pub application_reserve_bytes: u64,
    pub reserved: u64,
}
const _: () = assert!(size_of::<TransitionPlanningOptions>() == 64);

impl TransitionPlanningOptions {
    fn validate(self) -> Result<()> {
        if self.abi_version != ABI_VERSION
            || self.struct_size as usize != size_of::<Self>()
            || self.expected_model_instance == 0
            || self.expected_session_instance == 0
            || self.reserved != 0
            || [
                self.transition_budget_bytes,
                self.replay_reserve_bytes,
                self.text_reserve_bytes,
                self.application_reserve_bytes,
            ]
            .into_iter()
            .any(|n| n > weights::INITIAL_MEMORY_LIMIT_BYTES as u64)
        {
            return Err(fail(
                INVALID,
                "transition options outside ABI/engine bounds",
            ));
        }
        Ok(())
    }
}

fn checked_sum(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(0_usize, |sum, &n| {
        sum.checked_add(n)
            .ok_or_else(|| fail(INVALID, "transition size overflow"))
    })
}

#[derive(Serialize)]
struct Shared {
    learned_bytes: usize,
    positional_bytes: usize,
    model_table_bytes: usize,
    total_bytes: usize,
}
#[derive(Serialize)]
struct Owned {
    kv_bytes: usize,
    scratch_bytes: usize,
    history_bytes: usize,
    logits_bytes: usize,
    session_table_bytes: usize,
    total_bytes: usize,
}
impl Owned {
    fn from_memory(memory: session::Memory, shared_metadata: usize) -> Result<Self> {
        let metadata = memory
            .metadata_bytes
            .checked_sub(shared_metadata)
            .ok_or_else(|| fail(ENGINE_ERROR, "shared metadata exceeds session report"))?;
        Ok(Self {
            kv_bytes: memory.kv_bytes,
            scratch_bytes: memory.scratch_bytes,
            history_bytes: memory.history_bytes,
            logits_bytes: memory.logits_bytes,
            session_table_bytes: metadata,
            total_bytes: checked_sum(&[
                memory.kv_bytes,
                memory.scratch_bytes,
                memory.history_bytes,
                memory.logits_bytes,
                metadata,
            ])?,
        })
    }
}
#[derive(Serialize)]
struct Configuration {
    capacity: usize,
    chunk: usize,
    position_offset: usize,
    individual_numerical_budget_bytes: usize,
}
impl From<session::Options> for Configuration {
    fn from(value: session::Options) -> Self {
        Self {
            capacity: value.capacity,
            chunk: value.prefill_chunk_size,
            position_offset: value.position_offset,
            individual_numerical_budget_bytes: value.memory_limit_bytes,
        }
    }
}
#[derive(Serialize)]
struct Reserves {
    current_metadata_driver_bytes: usize,
    candidate_metadata_driver_bytes: usize,
    current_bridge_bytes: usize,
    candidate_bridge_bytes: usize,
    json_snapshot_bytes: usize,
    replay_bytes: usize,
    text_bytes: usize,
    application_bytes: usize,
    total_bytes: usize,
}
#[derive(Serialize)]
struct ExecutionModes {
    projection: String,
    normalization: String,
    attention: String,
    readback: &'static str,
}
#[derive(Serialize)]
struct TransitionReport {
    schema_version: u32,
    report_kind: &'static str,
    model_instance: u64,
    session_instance: u64,
    backend: &'static str,
    mode_selection: &'static str,
    execution_modes: ExecutionModes,
    loaded_position_limit: usize,
    current: Configuration,
    candidate: Configuration,
    history_tokens: u32,
    committed_tokens: u32,
    pending_tokens: u32,
    shared_model: Shared,
    current_allocated: Owned,
    candidate_planned: Owned,
    reserves: Reserves,
    aggregate_required_bytes: usize,
    transition_budget_bytes: u64,
    fits_transition_budget: bool,
    candidate_constructor_required_bytes: usize,
    candidate_whole_budget_bytes: u64,
    fits_candidate_individual_budget: bool,
    candidate_largest_device_buffer_bytes: Option<usize>,
    device_maximum_buffer_bytes: Option<usize>,
    candidate_allocated: bool,
    reserves_verified: bool,
    creation_revalidation_required: bool,
    context_qualification_evaluated: bool,
    whole_app_device_fit_evaluated: bool,
}

fn capture_transition(
    model: &ModelHandle,
    current: &SessionHandle,
    state: &OwnedSession,
    candidate: session::Options,
    whole_candidate_budget: u64,
    request: TransitionPlanningOptions,
) -> Result<TransitionReport> {
    if model.instance != request.expected_model_instance
        || current.instance != request.expected_session_instance
        || state.model_instance != model.instance
        || !Rc::ptr_eq(&model.model, &state._model)
    {
        return Err(fail(INVALID, "transition model/session identity mismatch"));
    }
    let (backend, loaded, metadata, metadata_in_report, actual, planned, old_options, buffer) =
        match (model.model.as_ref(), &state.session) {
            (NativeModel::Cpu(model), NativeSession::Cpu(session)) => {
                let metadata = model
                    .allocated_metadata_bytes()
                    .map_err(|e| fail(ENGINE_ERROR, e.to_string()))?;
                (
                    "cpu",
                    model.config().max_positions,
                    metadata,
                    metadata,
                    session
                        .owned_memory()
                        .map_err(|e| fail(ENGINE_ERROR, e.to_string()))?,
                    session::planned_memory(model, candidate)
                        .map_err(|e| fail(INVALID, e.to_string()))?,
                    session.options(),
                    None::<(usize, usize)>,
                )
            }
            #[cfg(all(
                feature = "metal",
                any(target_os = "macos", target_os = "ios"),
                target_arch = "aarch64"
            ))]
            (NativeModel::Metal(model), NativeSession::Metal(session)) => {
                let (planned, largest, maximum) =
                    model.native_session_plan(candidate).map_err(|e| {
                        let code = match e {
                            crate::metal::Error::Unsupported(_) => UNSUPPORTED,
                            crate::metal::Error::Options(_) => INVALID,
                            _ => ENGINE_ERROR,
                        };
                        fail(code, e.to_string())
                    })?;
                (
                    "metal",
                    model.config().max_positions,
                    model
                        .allocated_layer_metadata_bytes()
                        .map_err(|e| fail(ENGINE_ERROR, e.to_string()))?,
                    0,
                    session
                        .native_owned_memory()
                        .map_err(|e| fail(UNSUPPORTED, e.to_string()))?,
                    planned,
                    session.native_options(),
                    Some((largest, maximum)),
                )
            }
            #[cfg(all(
                feature = "metal",
                any(target_os = "macos", target_os = "ios"),
                target_arch = "aarch64"
            ))]
            _ => return Err(fail(INVALID, "transition backend identity mismatch")),
        };
    if actual.weights_bytes != planned.weights_bytes
        || actual.positional_bytes != planned.positional_bytes
    {
        return Err(fail(ENGINE_ERROR, "shared model accounting differs"));
    }
    let modes = match model.model.as_ref() {
        NativeModel::Cpu(_) => ExecutionModes {
            projection: "Scalar".into(),
            normalization: "Scalar".into(),
            attention: "Scalar".into(),
            readback: "FullLogits",
        },
        #[cfg(all(
            feature = "metal",
            any(target_os = "macos", target_os = "ios"),
            target_arch = "aarch64"
        ))]
        NativeModel::Metal(model) => {
            let (projection, normalization) = model.native_session_modes();
            ExecutionModes {
                projection: format!("{projection:?}"),
                normalization: format!("{normalization:?}"),
                attention: format!("{:?}", model.native_attention_mode()),
                readback: "FullLogits",
            }
        }
    };
    let shared = Shared {
        learned_bytes: actual.weights_bytes,
        positional_bytes: actual.positional_bytes,
        model_table_bytes: metadata,
        total_bytes: checked_sum(&[actual.weights_bytes, actual.positional_bytes, metadata])?,
    };
    let current_owned = Owned::from_memory(actual, metadata_in_report)?;
    let candidate_owned = Owned::from_memory(planned, metadata_in_report)?;
    let reserve =
        |n| usize::try_from(n).map_err(|_| fail(INVALID, "reserve exceeds address width"));
    let mut reserves = Reserves {
        current_metadata_driver_bytes: actual.metadata_allowance_bytes,
        candidate_metadata_driver_bytes: planned.metadata_allowance_bytes,
        current_bridge_bytes: BRIDGE_BYTES,
        candidate_bridge_bytes: BRIDGE_BYTES,
        json_snapshot_bytes: compatibility::COMPATIBILITY_JSON_ALLOWANCE_BYTES,
        replay_bytes: reserve(request.replay_reserve_bytes)?,
        text_bytes: reserve(request.text_reserve_bytes)?,
        application_bytes: reserve(request.application_reserve_bytes)?,
        total_bytes: 0,
    };
    reserves.total_bytes = checked_sum(&[
        reserves.current_metadata_driver_bytes,
        reserves.candidate_metadata_driver_bytes,
        reserves.current_bridge_bytes,
        reserves.candidate_bridge_bytes,
        reserves.json_snapshot_bytes,
        reserves.replay_bytes,
        reserves.text_bytes,
        reserves.application_bytes,
    ])?;
    let required = checked_sum(&[
        shared.total_bytes,
        current_owned.total_bytes,
        candidate_owned.total_bytes,
        reserves.total_bytes,
    ])?;
    let candidate_required = checked_sum(&[planned.admitted_bytes, BRIDGE_BYTES])?;
    let snapshot = report(&state.session);
    Ok(TransitionReport {
        schema_version: 1,
        report_kind: "same_model_session_transition",
        model_instance: model.instance,
        session_instance: current.instance,
        backend,
        mode_selection: "existing_native_factory",
        execution_modes: modes,
        loaded_position_limit: loaded,
        current: old_options.into(),
        candidate: candidate.into(),
        history_tokens: snapshot.history_tokens,
        committed_tokens: snapshot.evaluated_tokens,
        pending_tokens: snapshot.pending_tokens,
        shared_model: shared,
        current_allocated: current_owned,
        candidate_planned: candidate_owned,
        reserves,
        aggregate_required_bytes: required,
        transition_budget_bytes: request.transition_budget_bytes,
        fits_transition_budget: required as u64 <= request.transition_budget_bytes,
        candidate_constructor_required_bytes: candidate_required,
        candidate_whole_budget_bytes: whole_candidate_budget,
        fits_candidate_individual_budget: candidate_required as u64 <= whole_candidate_budget,
        candidate_largest_device_buffer_bytes: buffer.map(|b| b.0),
        device_maximum_buffer_bytes: buffer.map(|b| b.1),
        candidate_allocated: false,
        reserves_verified: false,
        creation_revalidation_required: true,
        context_qualification_evaluated: false,
        whole_app_device_fit_evaluated: false,
    })
}

/// Snapshot known shared model storage once and old-plus-candidate session costs.
/// No candidate, numerical execution, source-directory access or state mutation.
/// Candidate fit and aggregate fit are independent non-reserving estimates.
/// Actual allocation, caller reserves and identities must be rechecked by a
/// future creator. Copy/release use the existing compatibility JSON functions.
/// # Safety
/// Follow the parent pointer, creating-thread and disjoint-output contracts.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_session_transition_plan_create(
    model: *const ModelHandle,
    current: *const SessionHandle,
    candidate: *const Options,
    transition: *const TransitionPlanningOptions,
    out: *mut *mut CompatibilityHandle,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(out)?;
            out.write(ptr::null_mut());
            pointer(model)?;
            pointer(current)?;
            pointer(transition)?;
            owner((*model).owner)?;
            let request = *transition;
            request.validate()?;
            let options = options(candidate)?;
            (*current).access_owned(|state| {
                let report = capture_transition(
                    &*model,
                    &*current,
                    state,
                    options,
                    (*candidate).memory_limit_bytes,
                    request,
                )?;
                out.write(compatibility::capture(&report)?);
                Ok(())
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_components_never_wrap_or_subtract_allowances() {
        assert!(checked_sum(&[usize::MAX, 1]).is_err());
        assert_eq!(
            checked_sum(&[64 * 1024 * 1024; 2]).unwrap(),
            128 * 1024 * 1024
        );
    }
}
