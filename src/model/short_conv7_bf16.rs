//! Concrete canonical BF16 seven-tap history. No full-model session is defined.
use super::{Error, short_conv};
use half::bf16;

pub type History = short_conv::History<7, bf16>;
pub type Prepared<'a> = short_conv::Prepared<'a, bf16>;

pub fn commit_batch(histories: &mut [History], frames: &[Prepared<'_>]) -> Result<(), Error> {
    short_conv::commit_batch_impl(histories, frames)
}
pub fn reset_batch(histories: &mut [History]) -> Result<(), Error> {
    short_conv::reset_batch_impl(histories)
}
