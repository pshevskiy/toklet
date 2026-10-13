//! Toklet is a small Rust inference engine for Apple Silicon.
//!
//! Scalar CPU operations use borrowed contiguous slices and caller-owned output
//! buffers. Bounded sessions support generation; the CLI owns tokenization,
//! chat formatting and streamed text output.

pub mod benchmark;
pub mod cpu;
pub mod generation;
pub mod model;
pub mod q4;
pub mod sampling;
pub mod session;
pub mod weights;

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub mod metal;
