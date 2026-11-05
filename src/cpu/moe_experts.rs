//! Borrowed scalar reference for fused sparse-expert GLU and weighted reduction.
use super::{Error, MlpWeights, finite, gated_mlp, length, nonzero, product};

#[derive(Clone, Copy, Debug)]
pub struct SelectedExpertShape {
    pub tokens: usize,
    pub experts: usize,
    pub selected: usize,
    pub hidden: usize,
    pub intermediate: usize,
}

/// Immutable fused gate/up `[experts, 2*intermediate, hidden]` and down
/// `[experts, hidden, intermediate]`, all row-major. Gate rows precede up
/// rows within each expert; no whole-table split or packing is assumed.
pub struct ExpertWeights<'a> {
    pub gate_up: &'a [f32],
    pub down: &'a [f32],
}

/// Project selected experts without changing supplied token/slot order.
/// Input is `[tokens, hidden]`, IDs `[tokens, selected]`, output
/// `[tokens, selected, hidden]`; both scratch buffers are
/// `[tokens, selected, intermediate]`. Scratch retains gated SiLU and up.
/// Uses the existing checked FP32 GLU reference, with no allocation/dispatch
/// abstraction. Validation failures preserve every buffer; arithmetic overflow
/// may leave partial output/scratch, which the caller must discard.
pub fn selected_expert_glu(
    input: &[f32],
    ids: &[u32],
    weights: &ExpertWeights<'_>,
    output: &mut [f32],
    gate_scratch: &mut [f32],
    up_scratch: &mut [f32],
    shape: SelectedExpertShape,
) -> Result<(), Error> {
    let SelectedExpertShape {
        tokens,
        experts,
        selected,
        hidden,
        intermediate,
    } = shape;
    for (value, name) in [
        (tokens, "expert tokens"),
        (experts, "experts"),
        (selected, "selected experts"),
        (hidden, "expert hidden"),
        (intermediate, "expert intermediate"),
    ] {
        nonzero(value, name)?;
    }
    if experts > 32 || selected > experts {
        return Err(Error::InvalidParameter(
            "selected <= experts <= 32 required",
        ));
    }
    let slots = product(tokens, selected)?;
    let matrix = product(hidden, intermediate)?;
    let fused_stride = product(2, matrix)?;
    length("expert input", input.len(), product(tokens, hidden)?)?;
    length("expert IDs", ids.len(), slots)?;
    length(
        "expert gate/up",
        weights.gate_up.len(),
        product(experts, fused_stride)?,
    )?;
    length("expert down", weights.down.len(), product(experts, matrix)?)?;
    length("expert output", output.len(), product(slots, hidden)?)?;
    let scratch = product(slots, intermediate)?;
    length("expert gate scratch", gate_scratch.len(), scratch)?;
    length("expert up scratch", up_scratch.len(), scratch)?;
    for row in ids.chunks_exact(selected) {
        for (slot, &id) in row.iter().enumerate() {
            if id as usize >= experts || row[..slot].contains(&id) {
                return Err(Error::InvalidParameter(
                    "expert IDs must be in range and distinct per token",
                ));
            }
        }
    }
    finite("expert input", input)?;
    finite("expert gate/up", weights.gate_up)?;
    finite("expert down", weights.down)?;
    for (token, row) in ids.chunks_exact(selected).enumerate() {
        let x = &input[token * hidden..(token + 1) * hidden];
        for (slot, &id) in row.iter().enumerate() {
            let expert = id as usize;
            let fused = expert * fused_stride;
            let index = token * selected + slot;
            let view = MlpWeights {
                gate: &weights.gate_up[fused..fused + matrix],
                up: &weights.gate_up[fused + matrix..fused + fused_stride],
                down: &weights.down[expert * matrix..(expert + 1) * matrix],
                hidden,
                intermediate,
            };
            gated_mlp(
                x,
                &view,
                &mut output[index * hidden..(index + 1) * hidden],
                &mut gate_scratch[index * intermediate..(index + 1) * intermediate],
                &mut up_scratch[index * intermediate..(index + 1) * intermediate],
                1,
            )?;
        }
    }
    Ok(())
}

/// Reduce `[tokens, selected, hidden]` with nonnegative finite FP32 gates
/// `[tokens, selected]` into `[tokens, hidden]`. Do not renormalize/reorder.
/// Each product rounds FP32 before ascending-slot FP32 accumulation. All-zero
/// gates are valid. Validation errors preserve output; discard partial output
/// after arithmetic overflow. No residual, Half store or allocation is added.
pub fn reduce_selected_experts(
    values: &[f32],
    gates: &[f32],
    output: &mut [f32],
    tokens: usize,
    selected: usize,
    hidden: usize,
) -> Result<(), Error> {
    nonzero(tokens, "reduction tokens")?;
    nonzero(selected, "reduction selected")?;
    nonzero(hidden, "reduction hidden")?;
    if selected > 32 {
        return Err(Error::InvalidParameter(
            "reduction supports at most 32 slots",
        ));
    }
    let slots = product(tokens, selected)?;
    length("expert values", values.len(), product(slots, hidden)?)?;
    length("expert gates", gates.len(), slots)?;
    length("expert reduction", output.len(), product(tokens, hidden)?)?;
    finite("expert values", values)?;
    finite("expert gates", gates)?;
    if gates.iter().any(|&g| g < 0.) {
        return Err(Error::InvalidParameter("expert gates must be nonnegative"));
    }
    for (token, out) in output.chunks_exact_mut(hidden).enumerate() {
        for (column, destination) in out.iter_mut().enumerate() {
            let mut sum = 0_f32;
            for slot in 0..selected {
                let index = token * selected + slot;
                let weighted = values[index * hidden + column] * gates[index];
                if !weighted.is_finite() {
                    return Err(Error::NonFinite {
                        tensor: "weighted expert product",
                        index: index * hidden + column,
                    });
                }
                sum += weighted;
                if !sum.is_finite() {
                    return Err(Error::NonFinite {
                        tensor: "weighted expert sum",
                        index: token * hidden + column,
                    });
                }
            }
            *destination = sum;
        }
    }
    Ok(())
}
