//! Isolated Mac Chat probe. Replacement sessions use the ordinary factory.
use super::*;
use crate::weights::content::DenseProfile;

pub(super) fn configure(
    mut handle: Box<SessionHandle>,
    options: session::Options,
) -> Result<Box<SessionHandle>> {
    let state = handle.state.get_mut();
    if let (NativeModel::Metal(model), NativeSession::Metal(session)) =
        (state._model.as_ref(), &mut state.session)
        && options.capacity == 512
        && options.prefill_chunk_size == 32
        && options.position_offset == 0
        && model
            .mlx_dense_profile_content_identity(DenseProfile::Llama3)
            .is_some()
        && model.device_name() == "Apple M1 Pro"
    {
        // Profiling is session-owned. Admit it before installing the shared
        // staged pipeline so a profiling-budget rejection leaves the model alone.
        session
            .enable_profile()
            .and_then(|()| session.enable_llama3_staged_simd())
            .map_err(|error| fail(ENGINE_ERROR, error.to_string()))?;
        eprintln!(
            "TOKLET_LLAMA_STAGED_APP {}",
            serde_json::json!({"event":"session","capacity":options.capacity,
                "chunk":options.prefill_chunk_size,"admitted_bytes":session.memory().admitted_bytes})
        );
    }
    Ok(handle)
}
