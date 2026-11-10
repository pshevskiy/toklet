mod profile {
    pub use toklet::model::short_conv7::{History, Prepared, commit_batch, reset_batch};
    pub const KERNEL: usize = 7;
}
include!("short_conv_history.rs");
