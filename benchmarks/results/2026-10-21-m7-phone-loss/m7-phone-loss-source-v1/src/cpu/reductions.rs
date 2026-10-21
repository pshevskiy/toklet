use super::{Error, finite, length, matmul, nonzero, product};

/// Normalize each row of `[tokens, hidden]` with a learned `[hidden]` weight.
///
/// Epsilon must be finite and positive. Squares, reduction, normalization and
/// scaling use FP64 to avoid overflow for every finite FP32 input magnitude;
/// final values are stored in FP32. No output is written for invalid inputs.
pub fn rmsnorm(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    tokens: usize,
    hidden: usize,
    epsilon: f32,
) -> Result<(), Error> {
    nonzero(tokens, "tokens")?;
    nonzero(hidden, "hidden")?;
    let count = product(tokens, hidden)?;
    length("input", input.len(), count)?;
    length("output", output.len(), count)?;
    length("norm weight", weight.len(), hidden)?;
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Error::InvalidParameter(
            "epsilon must be finite and positive",
        ));
    }
    finite("input", input)?;
    finite("norm weight", weight)?;
    for (row, (source, destination)) in input
        .chunks_exact(hidden)
        .zip(output.chunks_exact_mut(hidden))
        .enumerate()
    {
        let squares: f64 = source.iter().map(|&x| f64::from(x).powi(2)).sum();
        let denominator = (squares / hidden as f64 + f64::from(epsilon)).sqrt();
        for (column, ((&x, &scale), out)) in source.iter().zip(weight).zip(destination).enumerate()
        {
            let value = (f64::from(x) / denominator * f64::from(scale)) as f32;
            if !value.is_finite() {
                return Err(Error::NonFinite {
                    tensor: "output",
                    index: row * hidden + column,
                });
            }
            *out = value;
        }
    }
    Ok(())
}

/// Max-subtracted softmax over rows `[rows, columns]`.
///
/// An optional same-shaped mask uses `true` for included entries. Masked
/// probabilities are exactly zero. All scores must be finite, even masked
/// scores. Reject empty dimensions and entirely masked rows before writing.
/// Subtraction, exponentials and sums use FP64; probabilities are FP32.
pub fn softmax(
    scores: &[f32],
    mask: Option<&[bool]>,
    output: &mut [f32],
    rows: usize,
    columns: usize,
) -> Result<(), Error> {
    nonzero(rows, "rows")?;
    nonzero(columns, "columns")?;
    let count = product(rows, columns)?;
    length("scores", scores.len(), count)?;
    length("output", output.len(), count)?;
    finite("scores", scores)?;
    if let Some(mask) = mask {
        length("mask", mask.len(), count)?;
        for (row, values) in mask.chunks_exact(columns).enumerate() {
            if !values.iter().any(|&included| included) {
                return Err(Error::FullyMasked { row });
            }
        }
    }
    for (row, (source, destination)) in scores
        .chunks_exact(columns)
        .zip(output.chunks_exact_mut(columns))
        .enumerate()
    {
        let row_mask = mask.map(|mask| &mask[row * columns..(row + 1) * columns]);
        let included = |column: usize| row_mask.is_none_or(|mask| mask[column]);
        let maximum = source
            .iter()
            .enumerate()
            .filter(|(column, _)| included(*column))
            .map(|(_, &value)| f64::from(value))
            .fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for (column, &score) in source.iter().enumerate() {
            if included(column) {
                sum += (f64::from(score) - maximum).exp();
            }
        }
        for (column, (&score, out)) in source.iter().zip(destination).enumerate() {
            *out = if included(column) {
                ((f64::from(score) - maximum).exp() / sum) as f32
            } else {
                0.0
            };
        }
    }
    Ok(())
}

fn silu_value(value: f32) -> f32 {
    let exponential = (-value.abs()).exp();
    let sigmoid = if value >= 0.0 {
        1.0 / (1.0 + exponential)
    } else {
        exponential / (1.0 + exponential)
    };
    value * sigmoid
}

/// Stable FP32 `x * sigmoid(x)` with no positive exponential arguments.
pub fn silu(input: &[f32], output: &mut [f32]) -> Result<(), Error> {
    nonzero(input.len(), "activation length")?;
    length("output", output.len(), input.len())?;
    finite("input", input)?;
    for (&value, out) in input.iter().zip(output) {
        *out = silu_value(value);
    }
    Ok(())
}

/// Elementwise FP32 `silu(gate) * up`, with caller-owned output.
pub fn gated_silu(gate: &[f32], up: &[f32], output: &mut [f32]) -> Result<(), Error> {
    nonzero(gate.len(), "activation length")?;
    length("up", up.len(), gate.len())?;
    length("output", output.len(), gate.len())?;
    finite("gate", gate)?;
    finite("up", up)?;
    for (index, ((&gate, &up), out)) in gate.iter().zip(up).zip(output).enumerate() {
        let value = silu_value(gate) * up;
        if !value.is_finite() {
            return Err(Error::NonFinite {
                tensor: "output",
                index,
            });
        }
        *out = value;
    }
    Ok(())
}

/// Borrowed immutable weights for the bias-free gated MLP.
///
/// Gate and up are row-major `[intermediate, hidden]`; down is
/// `[hidden, intermediate]`. This view owns no storage and uses static calls.
pub struct MlpWeights<'a> {
    pub gate: &'a [f32],
    pub up: &'a [f32],
    pub down: &'a [f32],
    pub hidden: usize,
    pub intermediate: usize,
}

/// Compute `down(silu(gate(input)) * up(input))` without allocating.
///
/// Input/output are `[tokens, hidden]`. Two scratch slices are exactly
/// `[tokens, intermediate]`; on success they retain gated and up projections.
/// Validate all shapes and finite input/weights before writing any buffer.
/// Arithmetic failure may leave partial output/scratch, which must be discarded.
pub fn gated_mlp(
    input: &[f32],
    weights: &MlpWeights<'_>,
    output: &mut [f32],
    gate_scratch: &mut [f32],
    up_scratch: &mut [f32],
    tokens: usize,
) -> Result<(), Error> {
    nonzero(tokens, "tokens")?;
    nonzero(weights.hidden, "hidden")?;
    nonzero(weights.intermediate, "intermediate")?;
    let activation_count = product(tokens, weights.hidden)?;
    let scratch_count = product(tokens, weights.intermediate)?;
    let weight_count = product(weights.hidden, weights.intermediate)?;
    length("input", input.len(), activation_count)?;
    length("output", output.len(), activation_count)?;
    length("gate scratch", gate_scratch.len(), scratch_count)?;
    length("up scratch", up_scratch.len(), scratch_count)?;
    for (name, values) in [
        ("gate weight", weights.gate),
        ("up weight", weights.up),
        ("down weight", weights.down),
    ] {
        length(name, values.len(), weight_count)?;
        finite(name, values)?;
    }
    finite("input", input)?;
    matmul(
        input,
        weights.gate,
        None,
        gate_scratch,
        tokens,
        weights.hidden,
        weights.intermediate,
    )?;
    matmul(
        input,
        weights.up,
        None,
        up_scratch,
        tokens,
        weights.hidden,
        weights.intermediate,
    )?;
    for (index, (gate, &up)) in gate_scratch.iter_mut().zip(up_scratch.iter()).enumerate() {
        let value = silu_value(*gate) * up;
        if !value.is_finite() {
            return Err(Error::NonFinite {
                tensor: "gated scratch",
                index,
            });
        }
        *gate = value;
    }
    matmul(
        gate_scratch,
        weights.down,
        None,
        output,
        tokens,
        weights.intermediate,
        weights.hidden,
    )
}
