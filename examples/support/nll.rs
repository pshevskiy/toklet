//! Stable FP64 row scoring over borrowed finite FP32 distributions.
#[derive(Default)]
pub(crate) struct Sum {
    pub(crate) value: f64,
    correction: f64,
}
impl Sum {
    pub(crate) fn add(&mut self, value: f64) {
        let adjusted = value - self.correction;
        let next = self.value + adjusted;
        self.correction = (next - self.value) - adjusted;
        self.value = next;
    }
}

/// Returns `(maximum, compensated exponential mass, negative log likelihood)`.
pub(crate) fn statistics(logits: &[f32], target: usize) -> Result<(f64, f64, f64), &'static str> {
    if logits.is_empty() || target >= logits.len() {
        return Err("loss target or vocabulary outside bounds");
    }
    let mut maximum = f64::NEG_INFINITY;
    for &value in logits {
        if !value.is_finite() {
            return Err("loss logits must be finite");
        }
        maximum = maximum.max(f64::from(value));
    }
    let mut exponential = Sum::default();
    for &value in logits {
        exponential.add((f64::from(value) - maximum).exp());
    }
    // Subtract before adding log(sum), retaining equal-logit loss even at MAX.
    let result = (maximum - f64::from(logits[target])) + exponential.value.ln();
    if !result.is_finite() || result < 0.0 {
        return Err("loss reduction produced an invalid result");
    }
    Ok((maximum, exponential.value, result))
}
