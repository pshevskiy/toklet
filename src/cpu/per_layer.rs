use super::{Error, finite, length, nonzero, product, rmsnorm};

/// Scalar per-layer input composition. Tokens flatten any independent B/N rows;
/// projection/lookup/output are `[tokens,layers,width]`, with projection initially
/// the raw backend result `[tokens,layers*width]`. Norm weights are `[width]`.
/// Hidden is the main embedding width, not layers*width.
pub struct PerLayerInputShape {
    pub tokens: usize,
    pub hidden: usize,
    pub layers: usize,
    pub width: usize,
    pub epsilon: f32,
}

/// Copy raw `[tokens,hidden]` main embeddings, multiplying by FP32 sqrt(hidden).
/// The scale is computed in FP64 and cast before FP32 multiplication. Do not
/// cancel it against the later projection scale across rounding boundaries.
/// All dimensions, lengths and finite inputs are checked before writes. A
/// multiplication overflow leaves discardable output and unchanged input.
/// Empty sequences make no call; zero dimensions reject. No allocation.
pub fn prepare_per_layer_projection_input(
    raw: &[f32],
    scaled: &mut [f32],
    tokens: usize,
    hidden: usize,
) -> Result<(), Error> {
    nonzero(tokens, "per-layer tokens")?;
    nonzero(hidden, "per-layer hidden")?;
    let count = product(tokens, hidden)?;
    length("per-layer raw embedding", raw.len(), count)?;
    length("per-layer scaled embedding", scaled.len(), count)?;
    finite("per-layer raw embedding", raw)?;
    let scale = (hidden as f64).sqrt() as f32;
    for (index, (&value, out)) in raw.iter().zip(scaled).enumerate() {
        let result = value * scale;
        if !result.is_finite() {
            return Err(Error::NonFinite {
                tensor: "per-layer scaled embedding",
                index,
            });
        }
        *out = result;
    }
    Ok(())
}

/// Compose a raw backend projection with optional unscaled selected lookup rows.
/// Backend GEMM and table lookup stay outside this function; quantized weights
/// need no expanded FP32 copy. This FP32 mechanism does not qualify that backend.
///
/// Scale projection by FP32 hidden^-0.5 into caller scratch, normalize each width
/// row using FP64 RMS/direct learned scale to FP32 output, then (when present)
/// scale each lookup by FP32 sqrt(width), add in FP32 and scale by FP32 sqrt(0.5).
/// Constants are computed in FP64 then cast. Each multiplication/addition has its
/// own FP32 boundary. Projected-only output is the normalized projection.
///
/// All geometry/size/length/epsilon and finite projection/lookup/norm values are
/// checked before scratch/output writes. Arithmetic overflow can leave partial
/// scratch/output: discard both and replay from immutable inputs. Empty sequences
/// make no call; zero dimensions reject. No buffers are allocated.
pub fn finish_per_layer_inputs(
    projection: &[f32],
    lookup: Option<&[f32]>,
    norm: &[f32],
    scaled_projection: &mut [f32],
    output: &mut [f32],
    shape: &PerLayerInputShape,
) -> Result<(), Error> {
    let s = shape;
    for (value, name) in [
        (s.tokens, "per-layer tokens"),
        (s.hidden, "per-layer hidden"),
        (s.layers, "per-layer layers"),
        (s.width, "per-layer width"),
    ] {
        nonzero(value, name)?;
    }
    let features = product(s.layers, s.width)?;
    let count = product(s.tokens, features)?;
    let rows = product(s.tokens, s.layers)?;
    if !s.epsilon.is_finite() || s.epsilon <= 0. {
        return Err(Error::InvalidParameter("per-layer epsilon"));
    }
    for (name, actual, expected) in [
        ("per-layer projection", projection.len(), count),
        (
            "per-layer projection scratch",
            scaled_projection.len(),
            count,
        ),
        ("per-layer output", output.len(), count),
        ("per-layer norm", norm.len(), s.width),
    ] {
        length(name, actual, expected)?;
    }
    if let Some(lookup) = lookup {
        length("per-layer selected lookup", lookup.len(), count)?;
        finite("per-layer selected lookup", lookup)?;
    }
    finite("per-layer projection", projection)?;
    finite("per-layer norm", norm)?;
    let scale = (s.hidden as f64).powf(-0.5) as f32;
    for (index, (&value, out)) in projection
        .iter()
        .zip(scaled_projection.iter_mut())
        .enumerate()
    {
        let value = value * scale;
        if !value.is_finite() {
            return Err(Error::NonFinite {
                tensor: "per-layer scaled projection",
                index,
            });
        }
        *out = value;
    }
    rmsnorm(scaled_projection, norm, output, rows, s.width, s.epsilon)?;
    if let Some(lookup) = lookup {
        let lookup_scale = (s.width as f64).sqrt() as f32;
        let combination_scale = 2_f64.powf(-0.5) as f32;
        for (index, (out, &value)) in output.iter_mut().zip(lookup).enumerate() {
            let selected = value * lookup_scale;
            if !selected.is_finite() {
                return Err(Error::NonFinite {
                    tensor: "per-layer scaled lookup",
                    index,
                });
            }
            let combined = *out + selected;
            if !combined.is_finite() {
                return Err(Error::NonFinite {
                    tensor: "per-layer combined input",
                    index,
                });
            }
            let value = combined * combination_scale;
            if !value.is_finite() {
                return Err(Error::NonFinite {
                    tensor: "per-layer output",
                    index,
                });
            }
            *out = value;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
