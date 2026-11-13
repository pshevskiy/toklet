//! Additive completed-generation observation; no session state or hot-loop timers.
use super::*;

pub const GENERATION_REPORT_KNOWN: u32 = 1;
pub const GENERATION_FIRST_TOKEN_KNOWN: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GenerationMetrics {
    pub abi_version: u32,
    pub struct_size: u32,
    pub known_fields: u32,
    pub reserved: u32,
    pub session_instance: u64,
    pub reused_tokens: u64,
    pub prefill_tokens: u64,
    pub sampled_tokens: u64,
    pub emitted_tokens: u64,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub sampling_seconds: f64,
    pub total_seconds: f64,
    pub first_token_seconds: f64,
}
impl GenerationMetrics {
    pub(super) fn empty() -> Self {
        Self {
            abi_version: ABI_VERSION,
            struct_size: size_of::<Self>() as u32,
            ..Self::default()
        }
    }
    pub(super) fn from_report(session_instance: u64, r: &generation::Report) -> Self {
        Self {
            known_fields: GENERATION_REPORT_KNOWN
                | if r.first_token_seconds.is_some() {
                    GENERATION_FIRST_TOKEN_KNOWN
                } else {
                    0
                },
            session_instance,
            reused_tokens: r.reused_tokens as u64,
            prefill_tokens: r.prefill_tokens as u64,
            sampled_tokens: r.sampled_tokens as u64,
            emitted_tokens: r.emitted_tokens as u64,
            prefill_seconds: r.prefill_seconds,
            decode_seconds: r.decode_seconds,
            sampling_seconds: r.sampling_seconds,
            total_seconds: r.total_seconds,
            first_token_seconds: r.first_token_seconds.unwrap_or(0.0),
            ..Self::empty()
        }
    }
}

/// Additive completed decode-row work, preserving the embedded legacy header.
/// `base.struct_size` describes `GenerationMetrics`, not this extension.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GenerationMetricsV2 {
    pub base: GenerationMetrics,
    pub decode_evaluations: u64,
}
impl GenerationMetricsV2 {
    pub(super) fn empty() -> Self {
        Self {
            base: GenerationMetrics::empty(),
            decode_evaluations: 0,
        }
    }
    pub(super) fn from_report(session_instance: u64, r: &generation::Report) -> Self {
        Self {
            base: GenerationMetrics::from_report(session_instance, r),
            decode_evaluations: r.decode_evaluations as u64,
        }
    }
}

/// Same generation/state/callback contract as toklet_session_generate, with a
/// separate completed-report observation. Failure clears known_fields; partial
/// native timing is unavailable, never a stale report from an earlier call.
/// Cancellation is a completed report, including zero-emission cancellation.
/// # Safety
/// Follow the foreign caller contract. metrics is required, aligned, live,
/// writable initialized storage, disjoint from every input/output/error record.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_session_generate_observed(
    session: *const SessionHandle,
    operation: u32,
    tokens: *const u32,
    token_count: usize,
    max_new_tokens: u32,
    stops: *const u32,
    stop_count: usize,
    cancel: *const CancelHandle,
    emit: Emit,
    context: *mut c_void,
    out: *mut Report,
    metrics: *mut GenerationMetrics,
    error: *mut NativeError,
) -> u32 {
    if metrics.is_null() {
        return unsafe {
            boundary(error, || {
                Err(fail(INVALID, "generation metrics output required"))
            })
        };
    }
    unsafe {
        generate_native(
            session,
            operation,
            tokens,
            token_count,
            max_new_tokens,
            stops,
            stop_count,
            cancel,
            emit,
            context,
            out,
            GenerationObservation::V1(metrics),
            error,
        )
    }
}

/// Same generation/state/callback rules as the legacy observed function, with
/// the existing completed Report's decode-row count. Availability is governed
/// by base.known_fields; an unknown zero is not an observed zero. Completed
/// cancellation may include a decode evaluation without another sampled token.
/// # Safety
/// Follow the foreign caller contract. metrics is required, aligned, live,
/// writable initialized storage for the complete V2 record, disjoint from every
/// input/output/error record. The embedded header remains the legacy96B layout.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_session_generate_observed_v2(
    session: *const SessionHandle,
    operation: u32,
    tokens: *const u32,
    token_count: usize,
    max_new_tokens: u32,
    stops: *const u32,
    stop_count: usize,
    cancel: *const CancelHandle,
    emit: Emit,
    context: *mut c_void,
    out: *mut Report,
    metrics: *mut GenerationMetricsV2,
    error: *mut NativeError,
) -> u32 {
    if metrics.is_null() {
        return unsafe {
            boundary(error, || {
                Err(fail(INVALID, "generation metrics v2 output required"))
            })
        };
    }
    unsafe {
        generate_native(
            session,
            operation,
            tokens,
            token_count,
            max_new_tokens,
            stops,
            stop_count,
            cancel,
            emit,
            context,
            out,
            GenerationObservation::V2(metrics),
            error,
        )
    }
}

#[cfg(test)]
mod tests;
