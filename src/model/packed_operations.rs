//! Concrete packed projection/gather operations shared by Gemma graph consumers.
use super::Error;
use crate::{
    cpu::{self, length},
    mlx::affine::Matrix,
};
fn add_bytes(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(cpu::Error::SizeOverflow.into())
}

pub(crate) mod sealed {
    pub trait Matrix {}
    pub trait Operations {}
}
/// Cold shape and storage contract for the shared Gemma graph.
pub enum MatrixDescriptor {
    Affine(crate::mlx::affine::Layout),
    Bf16 {
        rows: usize,
        columns: usize,
        file_backed: bool,
    },
}

/// Implementations are sealed to the concrete scalar and Metal adapters.
pub trait MatrixSource: sealed::Matrix {
    fn descriptor(&self) -> MatrixDescriptor;
}
impl sealed::Matrix for Matrix<'_> {}
impl MatrixSource for Matrix<'_> {
    fn descriptor(&self) -> MatrixDescriptor {
        MatrixDescriptor::Affine(Matrix::layout(self))
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
    // Coarse generation lifecycle only; scalar/other backends keep no-op hooks.
    fn begin_generation_step(&mut self, _position: usize, _rows: usize, _last_row_only: bool) {}
    fn finish_generation_step(&mut self, _committed: Option<usize>) {}
    fn generation_history_replaced(&mut self) {}
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
