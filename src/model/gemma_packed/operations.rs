//! Two concrete execution paths for the same Gemma graph, selected per operation.
use super::*;

pub(crate) mod sealed {
    pub trait Matrix {}
    pub trait Operations {}
}
/// Admitted matrix geometry. Implementations are sealed to the scalar and Metal
/// Gemma paths; this is not a general tensor/backend extension interface.
pub trait MatrixSource: sealed::Matrix {
    fn layout(&self) -> crate::mlx::affine::Layout;
}
impl sealed::Matrix for Matrix<'_> {}
impl MatrixSource for Matrix<'_> {
    fn layout(&self) -> crate::mlx::affine::Layout {
        Matrix::layout(self)
    }
}
/// External scratch/descriptor ownership, excluding the backend embedded in
/// Session itself. Opaque resource policies remain separate from known bytes.
#[derive(Clone, Copy, Default, Debug)]
pub struct BackendMemory {
    pub scratch_bytes: usize,
    pub metadata_bytes: usize,
    pub metadata_allowance_bytes: usize,
}
impl BackendMemory {
    pub fn total(self) -> Result<usize, Error> {
        add_bytes(
            add_bytes(self.scratch_bytes, self.metadata_bytes)?,
            self.metadata_allowance_bytes,
        )
    }
}
/// Statically dispatched numerical substitutions. All remaining graph ordering,
/// CPU operators and KV/history publication are shared.
pub trait Operations<M: MatrixSource>: sealed::Operations {
    type Error: From<Error> + From<cpu::Error> + Into<crate::session::Error>;
    fn project(
        &mut self,
        input: &[f32],
        weight: &M,
        output: &mut [f32],
        tokens: usize,
    ) -> Result<(), Self::Error>;
    fn gather(&mut self, weight: &M, ids: &[u32], output: &mut [f32]) -> Result<(), Self::Error>;
    fn ready(&self) -> Result<(), Self::Error>;
    fn memory(&self) -> Result<BackendMemory, Error>;
}
#[derive(Default)]
pub struct Scalar;
impl sealed::Operations for Scalar {}
impl Operations<Matrix<'_>> for Scalar {
    type Error = Error;
    fn project(
        &mut self,
        input: &[f32],
        weight: &Matrix<'_>,
        output: &mut [f32],
        tokens: usize,
    ) -> Result<(), Error> {
        let l = weight.layout();
        length(
            "Gemma packed projection input",
            input.len(),
            cpu::product(tokens, l.columns())?,
        )?;
        length(
            "Gemma packed projection output",
            output.len(),
            cpu::product(tokens, l.rows())?,
        )?;
        for (row, out) in input
            .chunks_exact(l.columns())
            .zip(output.chunks_exact_mut(l.rows()))
        {
            weight
                .matvec(row, out)
                .map_err(|e| Error::InvalidConfig(e.0))?;
        }
        Ok(())
    }
    fn gather(
        &mut self,
        weight: &Matrix<'_>,
        ids: &[u32],
        output: &mut [f32],
    ) -> Result<(), Error> {
        let width = weight.layout().columns();
        length(
            "Gemma packed gather output",
            output.len(),
            cpu::product(ids.len(), width)?,
        )?;
        for (&id, row) in ids.iter().zip(output.chunks_exact_mut(width)) {
            weight
                .copy_row(id as usize, row)
                .map_err(|e| Error::InvalidConfig(e.0))?;
        }
        Ok(())
    }
    fn ready(&self) -> Result<(), Error> {
        Ok(())
    }
    fn memory(&self) -> Result<BackendMemory, Error> {
        Ok(BackendMemory::default())
    }
}
