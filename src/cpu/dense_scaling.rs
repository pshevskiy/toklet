//! Scalar graph boundaries. All input/parameter checks precede writes.
use super::{Error, finite, length};

fn coefficient(value: f32) -> Result<(), Error> {
    if !value.is_normal() || value <= 0. {
        return Err(Error::InvalidParameter(
            "dense scale requires positive normal FP32",
        ));
    }
    Ok(())
}
/// FP32 multiply/store; discard partial output on arithmetic overflow.
pub fn scale_in_place(values: &mut [f32], factor: f32) -> Result<(), Error> {
    coefficient(factor)?;
    finite("scaled input", values)?;
    for (index, value) in values.iter_mut().enumerate() {
        let next = *value * factor;
        if !next.is_finite() {
            return Err(Error::NonFinite {
                tensor: "scaled output",
                index,
            });
        }
        *value = next;
    }
    Ok(())
}
/// Preserve division rather than replacing it by a rounded reciprocal.
pub fn divide_in_place(values: &mut [f32], divisor: f32) -> Result<(), Error> {
    coefficient(divisor)?;
    finite("divided input", values)?;
    for (index, value) in values.iter_mut().enumerate() {
        let next = *value / divisor;
        if !next.is_finite() {
            return Err(Error::NonFinite {
                tensor: "divided output",
                index,
            });
        }
        *value = next;
    }
    Ok(())
}
/// Round the FP32 branch product before the FP32 residual addition.
pub fn residual_add_scaled(values: &mut [f32], delta: &[f32], factor: f32) -> Result<(), Error> {
    coefficient(factor)?;
    length("scaled residual delta", delta.len(), values.len())?;
    finite("scaled residual input", values)?;
    finite("scaled residual delta", delta)?;
    for (index, (value, branch)) in values.iter_mut().zip(delta).enumerate() {
        let scaled = *branch * factor;
        let next = *value + scaled;
        if !next.is_finite() {
            return Err(Error::NonFinite {
                tensor: "scaled residual output",
                index,
            });
        }
        *value = next;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scalar_boundaries_and_invalid_inputs() {
        let mut x = [0.125, -0.5];
        scale_in_place(&mut x, 12.).unwrap();
        assert_eq!(x, [1.5, -6.]);
        residual_add_scaled(&mut x, &[2., -4.], 0.25).unwrap();
        assert_eq!(x, [2., -7.]);
        divide_in_place(&mut x, 8.).unwrap();
        assert_eq!(x, [0.25, -0.875]);
        let original = x;
        for invalid in [0., -1., f32::NAN, f32::INFINITY, f32::from_bits(1)] {
            assert!(scale_in_place(&mut x, invalid).is_err());
            assert!(divide_in_place(&mut x, invalid).is_err());
            assert!(residual_add_scaled(&mut x, &[1., 1.], invalid).is_err());
            assert_eq!(x, original);
        }
        assert!(residual_add_scaled(&mut x, &[1.], 1.).is_err());
        assert!(residual_add_scaled(&mut x, &[f32::NAN, 1.], 1.).is_err());
        assert_eq!(x, original);
    }
}
