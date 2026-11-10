//! Concrete 3-tap transactional FP32 history; numerical policy is in [`super::short_conv`].

use super::Error;
pub use super::short_conv::Prepared;
pub type History = super::short_conv::History<3>;

pub fn commit_batch(histories: &mut [History], frames: &[Prepared<'_>]) -> Result<(), Error> {
    super::short_conv::commit_batch(histories, frames)
}

pub fn reset_batch(histories: &mut [History]) -> Result<(), Error> {
    super::short_conv::reset_batch(histories)
}
