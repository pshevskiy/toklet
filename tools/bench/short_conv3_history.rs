mod profile {
    pub use toklet::model::short_conv3::{History, Prepared, commit_batch, reset_batch};
    pub const KERNEL: usize = 3;
}
include!("short_conv_history.rs");
