use super::{Error, finite, length, nonzero, product};

/// One raw causal depthwise convolution step, before the separate SiLU operation.
///
/// Input/output are `[channels]`, weights `[channels,kernel]` with taps ordered
/// oldest-to-current, and history/candidate `[kernel-1,channels]` oldest-to-newest.
/// Products and increasing-tap sums use FP64, rounded once to FP32 output.
/// No alignment beyond `f32`, padding, bias, masks or numerical allocation.
///
/// All dimensions, lengths and finite inputs are checked before writes. A cast
/// overflow can leave partial output; discard output/candidate on any error.
/// History stays immutable. Candidate carries raw input, not convolution output.
/// A caller may commit it only after its entire multi-layer step succeeds.
/// Kernel one has empty history; an empty sequence makes no step calls.
pub fn causal_depthwise_conv_step(
    input: &[f32],
    weight: &[f32],
    history: &[f32],
    candidate: &mut [f32],
    output: &mut [f32],
    channels: usize,
    kernel: usize,
) -> Result<(), Error> {
    nonzero(channels, "convolution channels")?;
    nonzero(kernel, "convolution kernel")?;
    let weight_elements = product(channels, kernel)?;
    let history_elements = product(kernel - 1, channels)?;
    for (name, actual, expected) in [
        ("convolution input", input.len(), channels),
        ("convolution weight", weight.len(), weight_elements),
        ("convolution history", history.len(), history_elements),
        ("convolution candidate", candidate.len(), history_elements),
        ("convolution output", output.len(), channels),
    ] {
        length(name, actual, expected)?;
    }
    for (name, values) in [
        ("convolution input", input),
        ("convolution weight", weight),
        ("convolution history", history),
    ] {
        finite(name, values)?;
    }
    for (channel, (taps, out)) in weight.chunks_exact(kernel).zip(output).enumerate() {
        let mut sum = 0.0_f64;
        for (tap, &weight) in taps[..kernel - 1].iter().enumerate() {
            sum += f64::from(history[tap * channels + channel]) * f64::from(weight);
        }
        sum += f64::from(input[channel]) * f64::from(taps[kernel - 1]);
        // FP32 products and any usize-bounded sum fit FP64. The output cast
        // alone can overflow. Previous history is never used as scratch.
        let value = sum as f32;
        if !value.is_finite() {
            return Err(Error::NonFinite {
                tensor: "convolution output",
                index: channel,
            });
        }
        *out = value;
    }
    if kernel > 1 {
        let retained = history_elements - channels;
        candidate[..retained].copy_from_slice(&history[channels..]);
        candidate[retained..].copy_from_slice(input);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
