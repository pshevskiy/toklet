//! Opt-in projection tile; scalar execution remains the default.

use super::{Error, finite, length, nonzero, product};

/// Bounded CPU projection selection at an operation boundary.
/// Both variants use FP32 storage and ordered FP64 accumulation. The tiled
/// candidate requires full-workload performance qualification before adoption.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProjectionKernel {
    #[default]
    Scalar,
    Tiled,
}

impl ProjectionKernel {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Tiled => "tiled",
        }
    }

    /// Shape is `[tokens, in_features, out_features]`. Each selected function
    /// validates borrowed slices before its statically dispatched inner loops.
    pub(crate) fn project(
        self,
        input: &[f32],
        weights: &[f32],
        bias: Option<&[f32]>,
        output: &mut [f32],
        [tokens, in_features, out_features]: [usize; 3],
    ) -> Result<(), Error> {
        match self {
            Self::Scalar => super::matmul_f64(
                input,
                weights,
                bias,
                output,
                tokens,
                in_features,
                out_features,
            ),
            Self::Tiled => matmul_f64_tiled(
                input,
                weights,
                bias,
                output,
                tokens,
                in_features,
                out_features,
            ),
        }
    }
}

/// Compute a row-major FP32 projection with ordered FP64 accumulation.
///
/// Layout, bias placement, validation, and rounding match [`super::matmul_f64`].
/// A full tile consumes four input rows and four weight rows, updating sixteen
/// independent FP64 sums in increasing feature order. It reuses each loaded
/// weight across four tokens and each activation across four outputs. Token
/// tails use a one-row tile; output tails use scalar dots. No packing, padding,
/// heap allocation, extra alignment, reassociation, or CPU feature is required.
/// The largest accumulator tile occupies 128 stack bytes.
///
/// This is an opt-in candidate kernel; models default to the scalar path. Arithmetic
/// overflow can leave partial output, which must be discarded on error.
pub fn matmul_f64_tiled(
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

    let full_tokens = tokens / 4 * 4;
    for token in (0..full_tokens).step_by(4) {
        tile::<4>(
            input,
            weights,
            bias,
            output,
            token,
            in_features,
            out_features,
        )?;
    }
    for token in full_tokens..tokens {
        tile::<1>(
            input,
            weights,
            bias,
            output,
            token,
            in_features,
            out_features,
        )?;
    }
    Ok(())
}

fn tile<const ROWS: usize>(
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    token: usize,
    in_features: usize,
    out_features: usize,
) -> Result<(), Error> {
    let inputs: [&[f32]; ROWS] = std::array::from_fn(|row| {
        let start = (token + row) * in_features;
        &input[start..start + in_features]
    });
    let full_outputs = out_features / 4 * 4;
    for feature in (0..full_outputs).step_by(4) {
        let rows: [&[f32]; 4] = std::array::from_fn(|row| {
            let start = (feature + row) * in_features;
            &weights[start..start + in_features]
        });
        let sums = dot4(inputs, rows, in_features);
        for (row, sums) in sums.iter().enumerate() {
            for (column, &sum) in sums.iter().enumerate() {
                store(
                    sum,
                    bias,
                    output,
                    (token + row) * out_features + feature + column,
                    feature + column,
                )?;
            }
        }
    }
    for feature in full_outputs..out_features {
        let start = feature * in_features;
        let weights = &weights[start..start + in_features];
        for (row, input) in inputs.iter().enumerate() {
            let mut sum = 0.0_f64;
            for (&activation, &weight) in input.iter().zip(weights) {
                sum += f64::from(activation) * f64::from(weight);
            }
            store(
                sum,
                bias,
                output,
                (token + row) * out_features + feature,
                feature,
            )?;
        }
    }
    Ok(())
}

// Only ROWS=4 and ROWS=1 are instantiated. Every cell preserves scalar feature
// order, while the compiler can schedule independent cells together.
fn dot4<const ROWS: usize>(
    inputs: [&[f32]; ROWS],
    weights: [&[f32]; 4],
    features: usize,
) -> [[f64; 4]; ROWS] {
    let mut sums = [[0.0_f64; 4]; ROWS];
    for feature in 0..features {
        let w0 = f64::from(weights[0][feature]);
        let w1 = f64::from(weights[1][feature]);
        let w2 = f64::from(weights[2][feature]);
        let w3 = f64::from(weights[3][feature]);
        for (input, sum) in inputs.iter().zip(&mut sums) {
            let activation = f64::from(input[feature]);
            sum[0] += activation * w0;
            sum[1] += activation * w1;
            sum[2] += activation * w2;
            sum[3] += activation * w3;
        }
    }
    sums
}

fn store(
    mut sum: f64,
    bias: Option<&[f32]>,
    output: &mut [f32],
    index: usize,
    feature: usize,
) -> Result<(), Error> {
    if let Some(bias) = bias {
        sum += f64::from(bias[feature]);
    }
    let sum = sum as f32;
    if !sum.is_finite() {
        return Err(Error::NonFinite {
            tensor: "output",
            index,
        });
    }
    output[index] = sum;
    Ok(())
}
