//! Native readers and scalar references for MLX-format model artifacts.
//!
//! This is format interoperability, not a binding to the MLX runtime. A supported
//! tensor layout does not imply model-architecture or device qualification.

pub mod affine;
pub mod compatibility;
pub mod dense;
pub(crate) mod execution;
pub mod gemma4;
mod phi3;
pub mod qwen;
mod qwen3;
pub mod reference;
