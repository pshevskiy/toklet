//! Compatibility reexports for existing Gemma4 backend integrations.
#[cfg(all(
    feature = "metal",
    any(target_os = "macos", target_os = "ios"),
    target_arch = "aarch64"
))]
pub(crate) use super::super::packed_operations::sealed;
pub use super::super::packed_operations::{BackendMemory, MatrixSource, Operations, Scalar};
