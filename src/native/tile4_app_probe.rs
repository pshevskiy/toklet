//! Mac-only diagnostic build hooks for the exact audited Chat configuration.
use super::*;
use crate::metal::{Model, Session};

pub(super) fn configure<'m>(
    model: &Model,
    mut session: Session<'m>,
    options: session::Options,
) -> Result<Session<'m>> {
    if options.capacity == 512
        && options.prefill_chunk_size == 32
        && model.mlx_qwen2_source_content_identity().is_some()
        && model.device_name() == "Apple M1 Pro"
    {
        session
            .enable_affine_prefill_tile4()
            .map_err(|error| fail(ENGINE_ERROR, error.to_string()))?;
        session
            .enable_profile()
            .map_err(|error| fail(ENGINE_ERROR, error.to_string()))?;
        eprintln!(
            "TOKLET_TILE4_APP {}",
            serde_json::json!({"event":"session","capacity":options.capacity,
                "chunk":options.prefill_chunk_size,"admitted_bytes":session.memory().admitted_bytes})
        );
    }
    Ok(session)
}

pub(super) fn generate(
    session: &mut Session<'_>,
    request: generation::Request<'_>,
    cancel: &AtomicBool,
    emit: impl FnMut(generation::TokenEvent) -> io::Result<()>,
) -> std::result::Result<generation::Report, generation::Error> {
    let result = generation::generate(session, request, cancel, emit);
    if let Ok(report) = &result
        && report.sampled_tokens != 0
        && session.profile().is_ok()
    {
        eprintln!(
            "TOKLET_TILE4_APP {}",
            serde_json::json!({"event":"reply","reused_tokens":report.reused_tokens,
                "prefill_tokens":report.prefill_tokens,"sampled":report.sampled_tokens})
        );
    }
    result
}
