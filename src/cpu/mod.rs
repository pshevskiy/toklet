//! Readable scalar FP32 kernels, with checked operation boundaries.
//!
//! Kernels allocate no buffers. Dimension, length, token and input-finiteness
//! errors are detected before writing output. Arithmetic overflow can leave a
//! partially written output, which the caller must discard on error.

use std::fmt;

mod reductions;

pub use reductions::{MlpWeights, gated_mlp, gated_silu, rmsnorm, silu, softmax};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    ZeroDimension(&'static str),
    SizeOverflow,
    Length {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
    Token {
        position: usize,
        token: u32,
        vocab_size: usize,
    },
    NonFinite {
        tensor: &'static str,
        index: usize,
    },
    InvalidParameter(&'static str),
    FullyMasked {
        row: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension(name) => write!(formatter, "{name} must be nonzero"),
            Self::SizeOverflow => write!(formatter, "tensor size arithmetic overflow"),
            Self::Length {
                tensor,
                expected,
                actual,
            } => write!(
                formatter,
                "{tensor} length is {actual}, expected {expected}"
            ),
            Self::Token {
                position,
                token,
                vocab_size,
            } => write!(
                formatter,
                "token {token} at position {position} is outside vocabulary {vocab_size}"
            ),
            Self::NonFinite { tensor, index } => {
                write!(formatter, "non-finite value in {tensor} at index {index}")
            }
            Self::InvalidParameter(name) => write!(formatter, "invalid parameter: {name}"),
            Self::FullyMasked { row } => write!(formatter, "softmax row {row} is entirely masked"),
        }
    }
}

impl std::error::Error for Error {}

pub(crate) fn nonzero(value: usize, name: &'static str) -> Result<(), Error> {
    if value == 0 {
        Err(Error::ZeroDimension(name))
    } else {
        Ok(())
    }
}

pub(crate) fn product(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b).ok_or(Error::SizeOverflow)
}

pub(crate) fn length(tensor: &'static str, actual: usize, expected: usize) -> Result<(), Error> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::Length {
            tensor,
            expected,
            actual,
        })
    }
}

pub(crate) fn finite(tensor: &'static str, values: &[f32]) -> Result<(), Error> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        Err(Error::NonFinite { tensor, index })
    } else {
        Ok(())
    }
}

/// Compute `output = input * weights.T + bias` for contiguous row-major slices.
///
/// Input is `[tokens, in_features]`, weights are `[out_features, in_features]`,
/// and output is `[tokens, out_features]`. All dimensions must be nonzero.
/// Products and sums accumulate in FP32 in increasing feature order; optional
/// bias is applied after each dot product. No alignment beyond `f32` is required.
pub fn matmul(
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    tokens: usize,
    in_features: usize,
    out_features: usize,
) -> Result<(), Error> {
    nonzero(tokens, "tokens")?;
    nonzero(in_features, "in_features")?;
    nonzero(out_features, "out_features")?;
    length("input", input.len(), product(tokens, in_features)?)?;
    length(
        "weights",
        weights.len(),
        product(out_features, in_features)?,
    )?;
    length("output", output.len(), product(tokens, out_features)?)?;
    if let Some(bias) = bias {
        length("bias", bias.len(), out_features)?;
        finite("bias", bias)?;
    }
    finite("input", input)?;
    finite("weights", weights)?;
    for (token, (input_row, output_row)) in input
        .chunks_exact(in_features)
        .zip(output.chunks_exact_mut(out_features))
        .enumerate()
    {
        for (feature, (weight_row, destination)) in weights
            .chunks_exact(in_features)
            .zip(output_row)
            .enumerate()
        {
            let mut sum = 0.0_f32;
            for (&activation, &weight) in input_row.iter().zip(weight_row) {
                sum += activation * weight;
            }
            if let Some(bias) = bias {
                sum += bias[feature];
            }
            if !sum.is_finite() {
                return Err(Error::NonFinite {
                    tensor: "output",
                    index: token * out_features + feature,
                });
            }
            *destination = sum;
        }
    }
    Ok(())
}

/// Single-token matrix-vector projection with the same layout as [`matmul`].
pub fn matvec(
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    in_features: usize,
    out_features: usize,
) -> Result<(), Error> {
    matmul(input, weights, bias, output, 1, in_features, out_features)
}

/// Copy token rows from an embedding table `[vocab_size, hidden]` to
/// caller-owned output `[tokens.len(), hidden]`, without changing the table.
pub fn embedding(
    tokens: &[u32],
    table: &[f32],
    output: &mut [f32],
    vocab_size: usize,
    hidden: usize,
) -> Result<(), Error> {
    nonzero(tokens.len(), "tokens")?;
    nonzero(vocab_size, "vocab_size")?;
    nonzero(hidden, "hidden")?;
    length("embedding table", table.len(), product(vocab_size, hidden)?)?;
    length("output", output.len(), product(tokens.len(), hidden)?)?;
    for (position, &token) in tokens.iter().enumerate() {
        if usize::try_from(token).map_or(true, |token| token >= vocab_size) {
            return Err(Error::Token {
                position,
                token,
                vocab_size,
            });
        }
    }
    finite("embedding table", table)?;
    for (&token, row) in tokens.iter().zip(output.chunks_exact_mut(hidden)) {
        let start = token as usize * hidden;
        row.copy_from_slice(&table[start..start + hidden]);
    }
    Ok(())
}

/// Add a same-length residual to a caller-owned activation buffer in place.
pub fn residual_add(output: &mut [f32], residual: &[f32]) -> Result<(), Error> {
    nonzero(output.len(), "activation length")?;
    length("residual", residual.len(), output.len())?;
    finite("activation", output)?;
    finite("residual", residual)?;
    for (index, (destination, &value)) in output.iter_mut().zip(residual).enumerate() {
        let sum = *destination + value;
        if !sum.is_finite() {
            return Err(Error::NonFinite {
                tensor: "output",
                index,
            });
        }
        *destination = sum;
    }
    Ok(())
}
