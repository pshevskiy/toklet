//! Checkpoint-defined dense graph scalars, independent of positional semantics.
use super::Error;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseScales {
    pub embedding: f32,
    /// None preserves the original division by sqrt(head_dim).
    pub attention: Option<f32>,
    pub residual: f32,
    pub logits_divisor: f32,
}
impl Default for DenseScales {
    fn default() -> Self {
        Self {
            embedding: 1.,
            attention: None,
            residual: 1.,
            logits_divisor: 1.,
        }
    }
}
impl DenseScales {
    pub(crate) fn validate(self) -> Result<(), Error> {
        if [self.embedding, self.residual, self.logits_divisor]
            .into_iter()
            .chain(self.attention)
            .any(|v| !v.is_normal() || v <= 0.)
        {
            return Err(Error::InvalidConfig(
                "dense scales require positive normal FP32 values",
            ));
        }
        Ok(())
    }
}
