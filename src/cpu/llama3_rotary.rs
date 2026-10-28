use super::{Error, finite, length, nonzero, product};

/// Validated Llama 3.1/3.2 wavelength-dependent inverse-frequency scaling.
/// This is neither ordinary RoPE nor uniform position/frequency scaling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Llama3Scaling {
    factor: f64,
    low: f64,
    high: f64,
    original: f64,
}

impl Llama3Scaling {
    pub fn new(
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original_max_positions: usize,
    ) -> Result<Self, Error> {
        if !factor.is_finite()
            || factor < 1.0
            || !low_freq_factor.is_finite()
            || low_freq_factor <= 0.0
            || !high_freq_factor.is_finite()
            || high_freq_factor <= low_freq_factor
            || original_max_positions == 0
            || original_max_positions as u128 > 1_u128 << 53
        {
            return Err(Error::InvalidParameter("Llama3 rotary scaling"));
        }
        Ok(Self {
            factor: f64::from(factor),
            low: f64::from(low_freq_factor),
            high: f64::from(high_freq_factor),
            original: original_max_positions as f64,
        })
    }

    /// Transform a positive finite ordinary inverse frequency, in FP64.
    /// Wavelengths below original/high stay unchanged; those above original/low
    /// divide by factor. The closed interval between them blends linearly in
    /// original/wavelength. In particular the exact low-frequency boundary
    /// divides by factor; it must not accidentally fall through unchanged.
    pub fn inverse_frequency(self, ordinary: f64) -> Result<f64, Error> {
        if !ordinary.is_finite() || ordinary <= 0.0 {
            return Err(Error::InvalidParameter("Llama3 inverse frequency"));
        }
        let wavelength = std::f64::consts::TAU / ordinary;
        let result = if self.factor == 1.0 || wavelength < self.original / self.high {
            ordinary
        } else if wavelength > self.original / self.low {
            ordinary / self.factor
        } else {
            let smooth = (self.original / wavelength - self.low) / (self.high - self.low);
            (1.0 - smooth) * ordinary / self.factor + smooth * ordinary
        };
        if !result.is_finite() || result <= 0.0 {
            return Err(Error::InvalidParameter("Llama3 scaled frequency underflow"));
        }
        Ok(result)
    }
}

/// Contiguous `[tokens, heads, head_dim]`, full split-half pairing, absolute
/// positions. Theta is a positive finite FP32 model parameter.
#[derive(Clone, Copy, Debug)]
pub struct Llama3RotaryShape {
    pub tokens: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub offset: usize,
    pub theta: f32,
}

/// Allocation-free mathematical Llama3 rotary reference.
///
/// Ordinary frequency is theta^(-2*j/head_dim); scaling is applied before the
/// absolute position, even for short inputs. Frequencies, angles and trig use
/// FP64, then sine/cosine round to FP32 and pair arithmetic uses FP32, matching
/// the existing scalar rotary precision contract. This does not emulate MLX
/// backend transcendental approximations or qualify a model/Metal path.
///
/// Geometry, exact lengths and finite inputs are checked before writes. Positions
/// through 2^53 are supported without rounding the integer position. Arithmetic
/// overflow can leave partial output, which must be discarded. No extra
/// alignment, aliasing or ownership guarantees beyond borrowed FP32 slices.
pub fn llama3_rotary(
    input: &[f32],
    output: &mut [f32],
    shape: &Llama3RotaryShape,
    scaling: &Llama3Scaling,
) -> Result<(), Error> {
    let Llama3RotaryShape {
        tokens,
        heads,
        head_dim,
        offset,
        theta,
    } = *shape;
    nonzero(tokens, "tokens")?;
    nonzero(heads, "heads")?;
    nonzero(head_dim, "head_dim")?;
    if !head_dim.is_multiple_of(2) {
        return Err(Error::InvalidParameter(
            "Llama3 rotary head_dim must be even",
        ));
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(Error::InvalidParameter("theta must be finite and positive"));
    }
    let last = offset.checked_add(tokens - 1).ok_or(Error::SizeOverflow)?;
    if last as u128 > 1_u128 << 53 {
        return Err(Error::InvalidParameter(
            "rotary positions must be at most 2^53",
        ));
    }
    let stride = product(heads, head_dim)?;
    let count = product(tokens, stride)?;
    length("Llama3 rotary input", input.len(), count)?;
    length("Llama3 rotary output", output.len(), count)?;
    finite("Llama3 rotary input", input)?;
    for pair in 0..head_dim / 2 {
        let ordinary = f64::from(theta).powf(-(2.0 * pair as f64) / head_dim as f64);
        let inverse = scaling.inverse_frequency(ordinary)?;
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
                        tensor: "Llama3 rotary output",
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
