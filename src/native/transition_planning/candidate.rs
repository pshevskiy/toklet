//! Construct an empty candidate without mutating or poisoning the old session.
use super::*;

fn cancelled(cancel: &CancelHandle) -> Result<()> {
    if cancel.flag.load(Ordering::Relaxed) {
        Err(fail(CANCELLED, "empty session preparation cancelled"))
    } else {
        Ok(())
    }
}

/// Actual individual admission follows the backend's existing Memory contract;
/// shared model-table bytes are removed only from the candidate-owned component.
fn readmit(plan: &TransitionReport, actual: session::Memory, shared_metadata: usize) -> Result<()> {
    if actual.weights_bytes != plan.shared_model.learned_bytes
        || actual.positional_bytes != plan.shared_model.positional_bytes
        || actual.metadata_allowance_bytes != plan.reserves.candidate_metadata_driver_bytes
    {
        return Err(fail(ENGINE_ERROR, "candidate shared ownership changed"));
    }
    let owned = Owned::from_memory(actual, shared_metadata)?;
    let individual = checked_sum(&[actual.admitted_bytes, BRIDGE_BYTES])?;
    let aggregate = checked_sum(&[
        plan.shared_model.total_bytes,
        plan.current_allocated.total_bytes,
        owned.total_bytes,
        plan.reserves.total_bytes,
    ])?;
    if individual as u64 > plan.candidate_whole_budget_bytes
        || aggregate as u64 > plan.transition_budget_bytes
    {
        return Err(fail(
            ENGINE_ERROR,
            "actual candidate exceeds individual or coexistence budget",
        ));
    }
    Ok(())
}

fn candidate_memory(candidate: &SessionHandle) -> Result<(session::Memory, usize)> {
    let state = candidate
        .state
        .try_borrow()
        .map_err(|_| fail(BUSY, "candidate is busy"))?;
    match &state.session {
        NativeSession::Cpu(session) => {
            let metadata = match state._model.as_ref() {
                NativeModel::Cpu(model) => model
                    .allocated_metadata_bytes()
                    .map_err(|e| fail(ENGINE_ERROR, e.to_string()))?,
                #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
                NativeModel::OfficialGemma(_) => {
                    return Err(fail(ENGINE_ERROR, "candidate backend differs"));
                }
                #[cfg(all(
                    feature = "metal",
                    any(target_os = "macos", target_os = "ios"),
                    target_arch = "aarch64"
                ))]
                NativeModel::Metal(_) => {
                    return Err(fail(ENGINE_ERROR, "candidate backend differs"));
                }
            };
            Ok((
                session
                    .owned_memory()
                    .map_err(|e| fail(ENGINE_ERROR, e.to_string()))?,
                metadata,
            ))
        }
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        NativeSession::OfficialGemma(_) => Err(fail(
            UNSUPPORTED,
            "Gemma BF16 live session replacement is unavailable",
        )),
        #[cfg(all(
            feature = "metal",
            any(target_os = "macos", target_os = "ios"),
            target_arch = "aarch64"
        ))]
        NativeSession::Metal(session) => Ok((
            session
                .native_owned_memory()
                .map_err(|e| fail(ENGINE_ERROR, e.to_string()))?,
            0,
        )),
    }
}

fn create_candidate(
    model: &ModelHandle,
    options: session::Options,
    cancel: &CancelHandle,
) -> Result<Box<SessionHandle>> {
    cancelled(cancel)?;
    let prepared = create_session(model, options)?;
    #[cfg(test)]
    fault_after(&prepared, cancel)?;
    Ok(prepared)
}

/// Prepare an explicitly empty session on a supported model owner. Current stays borrowed only
/// for inspection; candidate unwind never traverses access_owned's poison path.
/// # Safety
/// Follow the parent pointer/thread contract; cancel stays live through return.
/// Outputs/error must be writable and disjoint from handles and request inputs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_session_prepare_empty(
    model: *const ModelHandle,
    current: *const SessionHandle,
    candidate: *const Options,
    transition: *const TransitionPlanningOptions,
    cancel: *const CancelHandle,
    out: *mut *mut SessionHandle,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(out)?;
            out.write(ptr::null_mut());
            pointer(model)?;
            pointer(current)?;
            pointer(transition)?;
            pointer(cancel)?;
            owner((*model).owner)?;
            owner((*current).owner)?;
            if (*current).poisoned.get() {
                return Err(fail(POISONED, "session requires release and recreation"));
            }
            let state = (*current)
                .state
                .try_borrow()
                .map_err(|_| fail(BUSY, "current session is in another operation"))?;
            let request = *transition;
            request.validate()?;
            let opts = options(candidate)?;
            cancelled(&*cancel)?;
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            if matches!((*model).model.as_ref(), NativeModel::OfficialGemma(_)) {
                let plan = official::plan(&*model, &*current, &state, opts, request)?;
                if plan.candidate_combined_bytes > (*candidate).memory_limit_bytes
                    || plan.aggregate_required_bytes > request.transition_budget_bytes
                {
                    return Err(fail(
                        ENGINE_ERROR,
                        "planned Gemma candidate exceeds individual or coexistence budget",
                    ));
                }
                let prepared = create_candidate(&*model, opts, &*cancel)?;
                official::readmit(&plan, &prepared)?;
                cancelled(&*cancel)?;
                out.write(Box::into_raw(prepared));
                return Ok(());
            }
            let plan = capture_transition(
                &*model,
                &*current,
                &state,
                opts,
                (*candidate).memory_limit_bytes,
                request,
            )?;
            if !plan.fits_candidate_individual_budget || !plan.fits_transition_budget {
                return Err(fail(
                    ENGINE_ERROR,
                    "planned candidate exceeds individual or coexistence budget",
                ));
            }
            let prepared = create_candidate(&*model, opts, &*cancel)?;
            let (actual, shared_metadata) = candidate_memory(&prepared)?;
            readmit(&plan, actual, shared_metadata)?;
            cancelled(&*cancel)?;
            // No fallible work follows the final cancellation load. Current's
            // immutable borrow lasts through publication; prepared retains model.
            out.write(Box::into_raw(prepared));
            Ok(())
        })
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fault {
    None,
    Allocation,
    PanicConstruction,
    FailureAfter,
    PanicAfter,
    CancelAfter,
    ExcessHistory,
}
#[cfg(test)]
thread_local! {
    static FAULT: Cell<Fault> = const { Cell::new(Fault::None) };
}
#[cfg(test)]
pub(crate) fn inject(fault: Fault) {
    FAULT.set(fault);
}
#[cfg(test)]
pub(in crate::native) fn construction_fault() -> Result<()> {
    match FAULT.get() {
        Fault::Allocation => {
            FAULT.set(Fault::None);
            Err(fail(ENGINE_ERROR, "injected allocation failure"))
        }
        Fault::PanicConstruction => {
            FAULT.set(Fault::None);
            panic!("injected candidate construction panic")
        }
        _ => Ok(()),
    }
}
#[cfg(test)]
fn fault_after(candidate: &SessionHandle, cancel: &CancelHandle) -> Result<()> {
    match FAULT.replace(Fault::None) {
        Fault::FailureAfter => Err(fail(ENGINE_ERROR, "injected candidate failure")),
        Fault::PanicAfter => panic!("injected constructed candidate panic"),
        Fault::CancelAfter => {
            cancel.flag.store(true, Ordering::Relaxed);
            Ok(())
        }
        Fault::ExcessHistory => {
            match &mut candidate.state.borrow_mut().session {
                NativeSession::Cpu(s) => s.test_reserve_history(4096),
                #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
                NativeSession::OfficialGemma(_) => panic!("CPU-only allocation excess control"),
                #[cfg(all(
                    feature = "metal",
                    any(target_os = "macos", target_os = "ios"),
                    target_arch = "aarch64"
                ))]
                NativeSession::Metal(_) => panic!("CPU-only allocation excess control"),
            }
            Ok(())
        }
        _ => Ok(()),
    }
}
