//! Native readers and scalar references for MLX-format model artifacts.
//!
//! This is format interoperability, not a binding to the MLX runtime. A supported
//! tensor layout does not imply model-architecture or device qualification.

pub mod affine;
pub mod qwen;
pub mod reference;
