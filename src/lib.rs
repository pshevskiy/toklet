//! Toklet is a small Rust inference engine for Apple Silicon.
//!
//! Scalar CPU operations use borrowed contiguous slices and caller-owned output
//! buffers. The CLI shell currently provides help and version information.

pub mod cpu;
pub mod model;
pub mod sampling;
pub mod session;
pub mod weights;
