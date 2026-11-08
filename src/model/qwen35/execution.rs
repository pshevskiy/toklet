//! Two concrete operation boundaries; numerical inner loops remain static.
use super::*;
use crate::mlx::affine::Layout;

pub(super) mod sealed {
    pub trait Matrix {}
    pub trait Executor {}
}

/// Immutable geometry selected during model binding, never inside a dot loop.
pub trait MatrixStorage: sealed::Matrix {
    fn layout(&self) -> Layout;
}
impl sealed::Matrix for Matrix<'_> {}
impl MatrixStorage for Matrix<'_> {
    fn layout(&self) -> Layout {
        Matrix::layout(self)
    }
}

/// Coarse projection/gather execution over already validated model bindings.
/// Bytes exclude the embedded executor struct (counted by Session metadata).
pub trait Execution<M: MatrixStorage>: sealed::Executor {
    fn owned_bytes(&self) -> Result<usize>;
    fn project(&mut self, matrix: &M, input: &[f32], output: &mut [f32]) -> Result<()>;
    fn gather(&mut self, matrix: &M, token: u32, output: &mut [f32]) -> Result<()>;
}

/// The original readable scalar implementation, with no executor allocation.
pub struct Cpu;
impl sealed::Executor for Cpu {}
impl Execution<Matrix<'_>> for Cpu {
    fn owned_bytes(&self) -> Result<usize> {
        Ok(0)
    }
    fn project(&mut self, matrix: &Matrix<'_>, input: &[f32], output: &mut [f32]) -> Result<()> {
        matrix.matvec(input, output).map_err(|e| invalid(e.0))
    }
    fn gather(&mut self, matrix: &Matrix<'_>, token: u32, output: &mut [f32]) -> Result<()> {
        matrix
            .copy_row(token as usize, output)
            .map_err(|e| invalid(e.0))
    }
}
