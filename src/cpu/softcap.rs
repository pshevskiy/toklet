use super::{Error, finite, nonzero};

/// Apply `cap * tanh(logit / cap)` to a nonempty contiguous FP32 slice in place.
/// Cap must be finite and positive. The entire input is checked for finiteness
/// before any writes; every admission error leaves all input/output bits intact.
///
/// Division, tanh and multiplication use FP64, then round once to FP32. For every
/// finite FP32 logit and positive finite FP32 cap, the quotient fits FP64 and the
/// result magnitude is bounded by cap, so valid inputs cannot overflow output.
/// This avoids the possible infinite quotient in a staged FP32 implementation.
/// Signed zero is preserved. No allocation or alignment beyond f32 is required.
///
/// The caller skips this operation when its model disables capping; cap=0 is not
/// an identity mode. This operator does not qualify a model, quantized checkpoint
/// or Metal graph. Empty sequences make no numerical call.
pub fn logit_softcap(logits: &mut [f32], cap: f32) -> Result<(), Error> {
    nonzero(logits.len(), "logit softcap width")?;
    if !cap.is_finite() || cap <= 0. {
        return Err(Error::InvalidParameter("logit softcap"));
    }
    finite("logits", logits)?;
    let cap = f64::from(cap);
    for value in logits {
        *value = ((f64::from(*value) / cap).tanh() * cap) as f32;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
