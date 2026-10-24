use super::{Error, finite, length, nonzero};

/// Tanh-approximate GELU on contiguous FP32 slices of equal nonzero length.
///
/// Computes `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))` in FP64,
/// rounded once to FP32. This is the tanh approximation, not erf GELU.
/// Finite FP32 inputs have finite outputs, including saturated tails.
/// Dimensions and finite inputs are checked before writes. No allocation or
/// alignment beyond `f32` is required; an empty sequence makes no calls.
pub fn gelu_tanh(input: &[f32], output: &mut [f32]) -> Result<(), Error> {
    nonzero(input.len(), "GELU width")?;
    length("GELU output", output.len(), input.len())?;
    finite("GELU input", input)?;
    let factor = (2.0 / std::f64::consts::PI).sqrt();
    for (&value, destination) in input.iter().zip(output) {
        *destination = activate(value, factor);
    }
    Ok(())
}

/// GeGLU: tanh-GELU of `gate`, rounded to FP32, then multiplied by `up`
/// in FP32. All slices have equal nonzero length; inputs remain immutable.
///
/// This preserves the separate activation/product rounding used upstream.
/// All input validation precedes writes. A product overflow can leave partial
/// output, which callers must discard on error. No numerical allocation.
pub fn gated_gelu_tanh(gate: &[f32], up: &[f32], output: &mut [f32]) -> Result<(), Error> {
    nonzero(gate.len(), "GELU width")?;
    length("GELU up", up.len(), gate.len())?;
    length("GELU output", output.len(), gate.len())?;
    finite("GELU gate", gate)?;
    finite("GELU up", up)?;
    let factor = (2.0 / std::f64::consts::PI).sqrt();
    for (index, ((&gate, &up), destination)) in gate.iter().zip(up).zip(output).enumerate() {
        let value = activate(gate, factor) * up;
        if !value.is_finite() {
            return Err(Error::NonFinite {
                tensor: "GELU output",
                index,
            });
        }
        *destination = value;
    }
    Ok(())
}

fn activate(value: f32, factor: f64) -> f32 {
    let x = f64::from(value);
    // Even the cube of a finite FP32 value fits FP64. The multiplier of x
    // lies in [0,1], so conversion back to FP32 cannot overflow.
    (0.5 * x * (1.0 + (factor * (x + 0.044715 * x * x * x)).tanh())) as f32
}

#[cfg(test)]
mod tests;
