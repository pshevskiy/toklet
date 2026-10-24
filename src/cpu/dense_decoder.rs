use super::{
    Error, finite, gated_gelu_tanh, length, matmul_f64, nonzero, product, residual_add, rmsnorm,
};

/// Independent token rows `[tokens,hidden]`; FFN has `intermediate` features.
/// Epsilon is finite and positive. Empty sequences make no numerical call.
pub struct DenseDecoderShape {
    pub tokens: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub epsilon: f32,
}

/// A complete optional PLE gate, with row-major `[width,hidden]` gate and
/// `[hidden,width]` projection weights, and direct learned `[hidden]` RMS scale.
/// Partially missing modules are not representable; adapters must reject them.
pub struct PerLayerGateWeights<'a> {
    pub width: usize,
    pub gate: &'a [f32],
    pub projection: &'a [f32],
    pub post_norm: &'a [f32],
}

/// Borrowed FP32 reference weights. Norms are direct learned `[hidden]` scales;
/// FFN gate/up/down are row-major `[intermediate,hidden]`,
/// `[intermediate,hidden]`, `[hidden,intermediate]`. Scalar is applied last.
/// These are dense FP32 reference slices, not packed quantized weights. Future
/// quantized integration must avoid retaining/expanding this reference weight set.
pub struct DenseDecoderWeights<'a> {
    pub post_attention_norm: &'a [f32],
    pub pre_ff_norm: &'a [f32],
    pub post_ff_norm: &'a [f32],
    pub ff_gate: &'a [f32],
    pub ff_up: &'a [f32],
    pub ff_down: &'a [f32],
    pub per_layer: Option<PerLayerGateWeights<'a>>,
    pub scalar: Option<f32>,
}

/// Reused caller-owned buffers. Hidden work/norm are each `[tokens,hidden]`;
/// gate/product are two distinct `[tokens,max(intermediate,PLE width)]` buffers;
/// up is `[tokens,intermediate]`. Exact lengths are required, including unused
/// PLE capacity when its weights exist but the request supplies no PLE input.
pub struct DenseDecoderScratch<'a> {
    pub hidden_work: &'a mut [f32],
    pub hidden_norm: &'a mut [f32],
    pub gate: &'a mut [f32],
    pub up: &'a mut [f32],
    pub product: &'a mut [f32],
}

/// Finish a dense layer from immutable hidden/residual and externally computed
/// attention output, both `[tokens,hidden]`. Input normalization and attention
/// happen outside this function. No cache state is owned or modified here.
///
/// Post-normalize attention, add hidden; pre-normalize that residual, apply
/// GeGLU FFN, post-normalize and add the saved residual. A complete PLE gate with
/// supplied `[tokens,width]` input then applies Linear/GELU/product/Linear/RMS,
/// adds its own residual, and the optional scalar multiplies the final sum.
/// PLE weights without input skip that branch. Input without weights rejects.
///
/// Linear products/accumulations, RMS and GELU use the existing FP64 reference
/// kernels with FP32 storage at operator boundaries. Products, residuals and
/// scalar multiply have separate FP32 boundaries. No allocations, copies of
/// weights, dynamic numerical dispatch or in-place GELU aliases are used.
///
/// Validate every weight/input (even a skipped PLE), size, epsilon and scalar
/// before any scratch/output write. Arithmetic overflow can leave partial
/// scratch/output; discard all of them and replay from unchanged borrowed inputs.
/// This standalone FP32 reference does not qualify a quantized/full model graph.
pub fn finish_dense_decoder(
    hidden: &[f32],
    attention_output: &[f32],
    per_layer_input: Option<&[f32]>,
    weights: &DenseDecoderWeights<'_>,
    scratch: &mut DenseDecoderScratch<'_>,
    output: &mut [f32],
    shape: &DenseDecoderShape,
) -> Result<(), Error> {
    let s = shape;
    for (value, name) in [
        (s.tokens, "dense tokens"),
        (s.hidden, "dense hidden"),
        (s.intermediate, "dense intermediate"),
    ] {
        nonzero(value, name)?;
    }
    if !s.epsilon.is_finite() || s.epsilon <= 0. {
        return Err(Error::InvalidParameter("dense epsilon"));
    }
    if let Some(scalar) = weights.scalar
        && !scalar.is_finite()
    {
        return Err(Error::InvalidParameter("dense scalar"));
    }
    let count = product(s.tokens, s.hidden)?;
    let ff_count = product(s.tokens, s.intermediate)?;
    let ff_weight_count = product(s.intermediate, s.hidden)?;
    let mut scratch_width = s.intermediate;
    let mut ple_count = 0;
    if let Some(ple) = &weights.per_layer {
        nonzero(ple.width, "dense PLE width")?;
        scratch_width = scratch_width.max(ple.width);
        ple_count = product(s.tokens, ple.width)?;
        let weight_count = product(s.hidden, ple.width)?;
        for (name, value, expected) in [
            ("dense PLE gate", ple.gate, weight_count),
            ("dense PLE projection", ple.projection, weight_count),
            ("dense PLE norm", ple.post_norm, s.hidden),
        ] {
            length(name, value.len(), expected)?;
            finite(name, value)?;
        }
        if let Some(input) = per_layer_input {
            length("dense PLE input", input.len(), ple_count)?;
            finite("dense PLE input", input)?;
        }
    } else if per_layer_input.is_some() {
        return Err(Error::InvalidParameter("dense PLE input without weights"));
    }
    let gate_count = product(s.tokens, scratch_width)?;
    for (name, value, expected) in [
        ("dense hidden", hidden, count),
        ("dense attention output", attention_output, count),
        (
            "dense post attention norm",
            weights.post_attention_norm,
            s.hidden,
        ),
        ("dense pre FF norm", weights.pre_ff_norm, s.hidden),
        ("dense post FF norm", weights.post_ff_norm, s.hidden),
        ("dense FF gate", weights.ff_gate, ff_weight_count),
        ("dense FF up", weights.ff_up, ff_weight_count),
        ("dense FF down", weights.ff_down, ff_weight_count),
    ] {
        length(name, value.len(), expected)?;
        finite(name, value)?;
    }
    for (name, actual, expected) in [
        ("dense hidden work", scratch.hidden_work.len(), count),
        ("dense hidden norm", scratch.hidden_norm.len(), count),
        ("dense gate work", scratch.gate.len(), gate_count),
        ("dense up work", scratch.up.len(), ff_count),
        ("dense product work", scratch.product.len(), gate_count),
        ("dense output", output.len(), count),
    ] {
        length(name, actual, expected)?;
    }

    rmsnorm(
        attention_output,
        weights.post_attention_norm,
        output,
        s.tokens,
        s.hidden,
        s.epsilon,
    )?;
    residual_add(output, hidden)?;
    rmsnorm(
        output,
        weights.pre_ff_norm,
        scratch.hidden_work,
        s.tokens,
        s.hidden,
        s.epsilon,
    )?;
    matmul_f64(
        scratch.hidden_work,
        weights.ff_gate,
        None,
        &mut scratch.gate[..ff_count],
        s.tokens,
        s.hidden,
        s.intermediate,
    )?;
    matmul_f64(
        scratch.hidden_work,
        weights.ff_up,
        None,
        scratch.up,
        s.tokens,
        s.hidden,
        s.intermediate,
    )?;
    gated_gelu_tanh(
        &scratch.gate[..ff_count],
        scratch.up,
        &mut scratch.product[..ff_count],
    )?;
    matmul_f64(
        &scratch.product[..ff_count],
        weights.ff_down,
        None,
        scratch.hidden_work,
        s.tokens,
        s.intermediate,
        s.hidden,
    )?;
    rmsnorm(
        scratch.hidden_work,
        weights.post_ff_norm,
        scratch.hidden_norm,
        s.tokens,
        s.hidden,
        s.epsilon,
    )?;
    residual_add(output, scratch.hidden_norm)?;
    if let (Some(ple), Some(input)) = (&weights.per_layer, per_layer_input) {
        matmul_f64(
            output,
            ple.gate,
            None,
            &mut scratch.gate[..ple_count],
            s.tokens,
            s.hidden,
            ple.width,
        )?;
        gated_gelu_tanh(
            &scratch.gate[..ple_count],
            input,
            &mut scratch.product[..ple_count],
        )?;
        matmul_f64(
            &scratch.product[..ple_count],
            ple.projection,
            None,
            scratch.hidden_work,
            s.tokens,
            ple.width,
            s.hidden,
        )?;
        rmsnorm(
            scratch.hidden_work,
            ple.post_norm,
            scratch.hidden_norm,
            s.tokens,
            s.hidden,
            s.epsilon,
        )?;
        residual_add(output, scratch.hidden_norm)?;
    }
    if let Some(scalar) = weights.scalar {
        for (index, value) in output.iter_mut().enumerate() {
            let scaled = *value * scalar;
            if !scaled.is_finite() {
                return Err(Error::NonFinite {
                    tensor: "dense scalar output",
                    index,
                });
            }
            *value = scaled;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
