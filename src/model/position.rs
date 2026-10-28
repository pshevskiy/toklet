use super::Config;
use crate::cpu::{self, Llama3RotaryShape, Llama3Scaling};

/// Immutable mathematical positional semantics, independent of cache placement
/// and backend scheduling. New codecs/positional mechanisms require real blocks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PositionPlan {
    Ordinary,
    Llama3(Llama3Scaling),
}

impl PositionPlan {
    pub(crate) fn apply(
        self,
        input: &[f32],
        output: &mut [f32],
        config: &Config,
        tokens: usize,
        heads: usize,
        offset: usize,
    ) -> Result<(), cpu::Error> {
        match self {
            Self::Ordinary => cpu::rotary(
                input,
                output,
                tokens,
                heads,
                config.head_dim,
                offset,
                config.rope_theta,
            ),
            Self::Llama3(scaling) => cpu::llama3_rotary(
                input,
                output,
                &Llama3RotaryShape {
                    tokens,
                    heads,
                    head_dim: config.head_dim,
                    offset,
                    theta: config.rope_theta,
                },
                &scaling,
            ),
        }
    }
}
