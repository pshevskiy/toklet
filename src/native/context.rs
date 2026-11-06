//! Scalar execution facts and non-reserving request admission at the C boundary.
//! Instance IDs describe live objects in this process, never artifact provenance.
use super::*;
use std::sync::atomic::AtomicU64;

pub const CONTEXT_DECLARED_TOTAL_KNOWN: u32 = 1;
pub const CONTEXT_DECLARED_INPUT_KNOWN: u32 = 2;
pub const CONTEXT_DECLARED_OUTPUT_KNOWN: u32 = 4;
/// Neither numerical context qualification nor whole-app/device fit was evaluated.
pub const CONTEXT_NOT_EVALUATED: u32 = 0;

struct Instances(AtomicU64);
impl Instances {
    fn next(&self) -> Result<u64> {
        self.0
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                if n == 0 { None } else { n.checked_add(1) }
            })
            .map_err(|_| fail(ENGINE_ERROR, "native instance identity exhausted"))
    }
}
static INSTANCES: Instances = Instances(AtomicU64::new(1));
pub(super) fn next_instance() -> Result<u64> {
    INSTANCES.next()
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContextInfo {
    pub abi_version: u32,
    pub struct_size: u32,
    pub known_fields: u32,
    /// 0 scalar CPU, 1 Metal. This is execution selection, not qualification.
    pub backend: u32,
    pub context_qualification: u32,
    pub whole_app_device_fit: u32,
    pub reserved: [u32; 2],
    pub model_instance: u64,
    pub session_instance: u64,
    pub declared_total_positions: u64,
    pub declared_input_tokens: u64,
    pub declared_output_tokens: u64,
    /// Global loader ceiling, not a backend/model qualification or usable limit.
    pub engine_global_position_ceiling: u64,
    pub loaded_position_limit: u64,
    pub session_capacity: u64,
    pub absolute_position_offset: u64,
    pub history_tokens: u64,
    pub committed_tokens: u64,
    pub pending_tokens: u64,
    pub numerical_and_bridge_admitted_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RequestAdmission {
    pub context: ContextInfo,
    pub operation: u32,
    pub reserved: u32,
    pub base_history_tokens: u64,
    pub input_tokens: u64,
    pub history_after_input_tokens: u64,
    pub requested_output_tokens: u64,
    pub required_capacity: u64,
    /// Session capacity minus required_capacity, after requested continuation.
    pub remaining_tokens: u64,
}

pub(super) fn generation_input(operation: u32, tokens: &[u32]) -> Result<generation::Input<'_>> {
    match operation {
        0 => Ok(generation::Input::Prompt(tokens)),
        1 => Ok(generation::Input::Append(tokens)),
        2 if tokens.is_empty() => Ok(generation::Input::Continue),
        3 => Ok(generation::Input::Reset(tokens)),
        _ => Err(fail(INVALID, "invalid operation or continue input")),
    }
}

fn facts(state: &OwnedSession, session_instance: u64) -> ContextInfo {
    let (backend, loaded, declared): (u32, usize, Option<usize>) = match state._model.as_ref() {
        NativeModel::Cpu(model) => (0, model.config().max_positions, None),
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        NativeModel::OfficialGemma(model) => (
            1,
            model.loaded_positions(),
            Some(crate::metal::gemma_affine::OfficialBf16::DECLARED_POSITIONS),
        ),
        #[cfg(all(
            feature = "metal",
            any(target_os = "macos", target_os = "ios"),
            target_arch = "aarch64"
        ))]
        NativeModel::Metal(model) => (
            1,
            model.config().max_positions,
            model.mlx_declared_positions(),
        ),
    };
    let snapshot = report(&state.session);
    let capacity = match &state.session {
        NativeSession::Cpu(s) => s.capacity(),
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        NativeSession::OfficialGemma(s) => s.inference.capacity(),
        #[cfg(all(
            feature = "metal",
            any(target_os = "macos", target_os = "ios"),
            target_arch = "aarch64"
        ))]
        NativeSession::Metal(s) => s.capacity(),
    };
    ContextInfo {
        abi_version: ABI_VERSION,
        struct_size: size_of::<ContextInfo>() as u32,
        known_fields: if declared.is_some() {
            CONTEXT_DECLARED_TOTAL_KNOWN
        } else {
            0
        },
        backend,
        context_qualification: CONTEXT_NOT_EVALUATED,
        whole_app_device_fit: CONTEXT_NOT_EVALUATED,
        reserved: [0; 2],
        model_instance: state.model_instance,
        session_instance,
        declared_total_positions: declared.unwrap_or(0) as u64,
        declared_input_tokens: 0,
        declared_output_tokens: 0,
        engine_global_position_ceiling: weights::MAX_CONTEXT_LIMIT as u64,
        loaded_position_limit: loaded as u64,
        session_capacity: capacity as u64,
        // Every current native Options factory admits position_offset=0. Keep
        // this explicit if an additive offset-enabled factory is introduced.
        absolute_position_offset: 0,
        history_tokens: u64::from(snapshot.history_tokens),
        committed_tokens: u64::from(snapshot.evaluated_tokens),
        pending_tokens: u64::from(snapshot.pending_tokens),
        numerical_and_bridge_admitted_bytes: snapshot.admitted_bytes,
    }
}

impl NativeSession {
    fn request_admission(
        &self,
        request: &generation::Request<'_>,
    ) -> std::result::Result<generation::RequestAdmission, generation::Error> {
        match self {
            Self::Cpu(s) => generation::request_admission(s.as_ref(), request),
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            Self::OfficialGemma(s) => generation::request_admission(&s.inference, request),
            #[cfg(all(
                feature = "metal",
                any(target_os = "macos", target_os = "ios"),
                target_arch = "aarch64"
            ))]
            Self::Metal(s) => generation::request_admission(s.as_ref(), request),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GemmaBf16MemoryInfo {
    pub abi_version: u32,
    pub struct_size: u32,
    pub load_admitted_bytes: u64,
    pub session_admitted_bytes: u64,
    pub session_reserved_bytes: u64,
    pub kv_owned_bytes: u64,
    pub codec_allowance_bytes: u64,
}
const _: () = assert!(size_of::<GemmaBf16MemoryInfo>() == 48);

/// Read combined owner admission for a live official BF16 session. The two
/// totals include the codec and native bridge. Caller-owned app memory is separate.
/// # Safety
/// Follow the parent pointer/thread contract; output is untouched unless status is OK.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_session_gemma_bf16_memory(
    session: *const SessionHandle,
    out: *mut GemmaBf16MemoryInfo,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(session)?;
            pointer(out)?;
            (*session).access_owned(|state| {
                #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
                if let (NativeModel::OfficialGemma(model), NativeSession::OfficialGemma(session)) =
                    (state._model.as_ref(), &state.session)
                {
                    let owner = model.memory();
                    out.write(GemmaBf16MemoryInfo {
                        abi_version: ABI_VERSION,
                        struct_size: size_of::<GemmaBf16MemoryInfo>() as u32,
                        load_admitted_bytes: (owner.load_admitted_bytes + BRIDGE_BYTES) as u64,
                        session_admitted_bytes: (session.admitted_bytes + BRIDGE_BYTES) as u64,
                        session_reserved_bytes: owner.session_reserved_bytes as u64,
                        kv_owned_bytes: session.inference.memory().kv_bytes as u64,
                        codec_allowance_bytes: crate::app_chat::gemma4::ALLOWANCE_BYTES as u64,
                    });
                    return Ok(());
                }
                let _ = state;
                Err(fail(
                    UNSUPPORTED,
                    "Official Gemma BF16 memory is unavailable for this session",
                ))
            })
        })
    }
}

/// Query the live retained model/session without touching their source directory.
/// Unknown declared fields are zero with their known bit clear. Qualification
/// and whole-app/device fit remain not evaluated; allocation is numerical only.
/// # Safety
/// Follow the parent pointer/thread/disjoint-output contract. Output points to a
/// complete ContextInfo and is untouched unless status is OK.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_session_context(
    session: *const SessionHandle,
    out: *mut ContextInfo,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(session)?;
            pointer(out)?;
            (*session).access_owned(|state| {
                out.write(facts(state, (*session).instance));
                Ok(())
            })
        })
    }
}

/// Validate the exact raw/rendered token request without executing or mutating.
/// Prompt/Reset replace history; Append/Continue count all sampled history,
/// including pending tokens. Zero output retains existing prefill-only semantics.
/// This is arithmetic/token validation, not a reservation or guarantee execution
/// succeeds; generation revalidates independently.
/// # Safety
/// Follow the parent pointer/thread/disjoint-output contract. Tokens/stops obey
/// the generate API's32768-element bound. Output is untouched on any error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_session_preflight(
    session: *const SessionHandle,
    operation: u32,
    tokens: *const u32,
    token_count: usize,
    maximum: u32,
    stops: *const u32,
    stop_count: usize,
    out: *mut RequestAdmission,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(session)?;
            pointer(out)?;
            let input = slice(tokens, token_count, weights::MAX_CONTEXT_LIMIT)?;
            let stops = slice(stops, stop_count, weights::MAX_CONTEXT_LIMIT)?;
            let request = generation::Request {
                input: generation_input(operation, input)?,
                max_new_tokens: maximum as usize,
                stop_tokens: stops,
            };
            (*session).access_owned(|state| {
                let admitted = state
                    .session
                    .request_admission(&request)
                    .map_err(|e| fail(INVALID, e.to_string()))?;
                out.write(RequestAdmission {
                    context: facts(state, (*session).instance),
                    operation,
                    reserved: 0,
                    base_history_tokens: admitted.base_history_tokens as u64,
                    input_tokens: admitted.input_tokens as u64,
                    history_after_input_tokens: admitted.history_after_input_tokens as u64,
                    requested_output_tokens: admitted.requested_output_tokens as u64,
                    required_capacity: admitted.required_capacity as u64,
                    remaining_tokens: admitted.remaining_tokens as u64,
                });
                Ok(())
            })
        })
    }
}

#[cfg(test)]
mod tests;
