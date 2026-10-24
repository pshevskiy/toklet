use super::{Error, finite, length, nonzero, product};

/// Contiguous `[tokens,heads,head_dim]` geometry and proportional position rules.
/// Base/factor are finite positive FP32 parameters; positions are absolute.
#[derive(Clone, Copy, Debug)]
pub struct ProportionalRotaryShape {
    pub tokens: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub rotated_dims: usize,
    pub offset: usize,
    pub base: f32,
    pub factor: f32,
}

/// Mathematical proportional rotary reference with nontraditional half pairing.
///
/// Pair `i < rotated_dims/2` joins coordinates `i` and `i + head_dim/2`.
/// The inverse frequency is `base^(-2*i/head_dim)/factor`: its denominator and
/// pair distance use the full head dimension. Inactive coordinates copy exactly.
/// Zero rotated dimensions copy all input; an empty sequence makes no calls.
///
/// Frequencies, absolute angles and trig use FP64, then sine/cosine round to
/// FP32 and rotation products/sums use FP32, like the existing scalar rotary.
/// This is not emulation of upstream backend approximations at long positions.
///
/// Head dimension is positive/even; rotated dimensions are even and no greater
/// than it. Positions through 2^53 are admitted without integer rounding.
/// All geometry, length and finite input checks precede writes. Arithmetic
/// overflow can leave partial output; discard it on error. Input stays immutable.
/// The function allocates no buffers and requires no extra alignment beyond FP32.
pub fn proportional_rotary(
    input: &[f32],
    output: &mut [f32],
    shape: &ProportionalRotaryShape,
) -> Result<(), Error> {
    let ProportionalRotaryShape {
        tokens,
        heads,
        head_dim,
        rotated_dims,
        offset,
        base,
        factor,
    } = *shape;
    nonzero(tokens, "tokens")?;
    nonzero(heads, "heads")?;
    nonzero(head_dim, "head_dim")?;
    if !head_dim.is_multiple_of(2) || !rotated_dims.is_multiple_of(2) || rotated_dims > head_dim {
        return Err(Error::InvalidParameter("proportional rotary dimensions"));
    }
    if !base.is_finite() || base <= 0.0 || !factor.is_finite() || factor <= 0.0 {
        return Err(Error::InvalidParameter("proportional rotary base/factor"));
    }
    let last = offset.checked_add(tokens - 1).ok_or(Error::SizeOverflow)?;
    if last as u128 > 1_u128 << 53 {
        return Err(Error::InvalidParameter(
            "rotary positions must be at most 2^53",
        ));
    }
    let stride = product(heads, head_dim)?;
    let count = product(tokens, stride)?;
    length("proportional rotary input", input.len(), count)?;
    length("proportional rotary output", output.len(), count)?;
    finite("proportional rotary input", input)?;
    output.copy_from_slice(input);
    for pair in 0..rotated_dims / 2 {
        // Positive finite FP32 base/factor keep this frequency and every
        // admitted angle finite in FP64, including FP32 subnormal parameters.
        let inverse =
            f64::from(base).powf(-(2.0 * pair as f64) / head_dim as f64) / f64::from(factor);
        for token in 0..tokens {
            let angle = (offset + token) as f64 * inverse;
            let (sine, cosine) = angle.sin_cos();
            let (sine, cosine) = (sine as f32, cosine as f32);
            for head in 0..heads {
                let first = token * stride + head * head_dim + pair;
                let second = first + head_dim / 2;
                let a = input[first] * cosine - input[second] * sine;
                let b = input[second] * cosine + input[first] * sine;
                if !a.is_finite() || !b.is_finite() {
                    return Err(Error::NonFinite {
                        tensor: "proportional rotary output",
                        index: if !a.is_finite() { first } else { second },
                    });
                }
                output[first] = a;
                output[second] = b;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
