//! One immutable BF16 matrix with independently owned reusable session scratch.
use super::*;

pub(in crate::metal::gemma_affine) struct Matrix {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    rows: usize,
    columns: usize,
    maximum: f64,
    coefficients: Buffer<u32>,
}

impl Matrix {
    pub(in crate::metal::gemma_affine) fn payload_bytes(
        rows: usize,
        columns: usize,
    ) -> Result<usize> {
        let (sizes, _) = projection_sizes(rows, columns, 1)?;
        Ok((sizes[0] + GUARD) * 4)
    }
    pub(in crate::metal::gemma_affine) fn shape(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }
    pub(in crate::metal::gemma_affine) fn physical_bytes(&self) -> usize {
        self.coefficients.len * 4
    }
    pub(in crate::metal::gemma_affine) fn same_device(&self, context: &Bf16Context) -> bool {
        std::ptr::eq(&*self.device, &*context.0.gpu.device)
    }
}

impl Bf16Context {
    pub(in crate::metal::gemma_affine) fn resident_matrix(
        &self,
        rows: usize,
        columns: usize,
        coefficients: &[u8],
        available: usize,
    ) -> Result<Matrix> {
        let bytes = Matrix::payload_bytes(rows, columns)?;
        let required = bytes
            .checked_add(size_of::<Matrix>())
            .ok_or(Error::Options("BF16 matrix size overflow"))?;
        if required > available {
            return Err(Error::Budget {
                required,
                limit: available,
            });
        }
        if coefficients.len() != rows * columns * 2 {
            return Err(Error::Options("BF16 resident coefficient length differs"));
        }
        validate_buffer_bytes(bytes, self.0.gpu.device.maxBufferLength())?;
        let maximum = maximum_coefficient(coefficients)?;
        Ok(Matrix {
            device: self.0.gpu.device.clone(),
            rows,
            columns,
            maximum,
            coefficients: loading::upload(&self.0.gpu, coefficients)?,
        })
    }

    pub(in crate::metal::gemma_affine) fn shared_workspace(
        &self,
        rows: usize,
        columns: usize,
        capacity: usize,
        available: usize,
    ) -> Result<Workspace<'_>> {
        let required = Workspace::minimum_bytes(rows, columns, capacity)?;
        if required > available {
            return Err(Error::Budget {
                required,
                limit: available,
            });
        }
        let (sizes, _) = projection_sizes(rows, columns, capacity)?;
        for words in &sizes[1..] {
            validate_buffer_bytes((words + GUARD) * 4, self.0.gpu.device.maxBufferLength())?;
        }
        Ok(Workspace {
            context: self,
            rows,
            columns,
            capacity,
            owned: required,
            input: Buffer::new(&self.0.gpu, sizes[1] + GUARD)?,
            output: Buffer::new(&self.0.gpu, sizes[2] + GUARD)?,
            status: Buffer::new(&self.0.gpu, sizes[3] + GUARD)?,
            pending: None,
            poisoned: false,
        })
    }
}

pub(in crate::metal::gemma_affine) struct Workspace<'a> {
    context: &'a Bf16Context,
    rows: usize,
    columns: usize,
    capacity: usize,
    owned: usize,
    input: Buffer<u32>,
    output: Buffer<u32>,
    status: Buffer<u32>,
    pending: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    poisoned: bool,
}

impl Workspace<'_> {
    pub(in crate::metal::gemma_affine) fn minimum_bytes(
        rows: usize,
        columns: usize,
        capacity: usize,
    ) -> Result<usize> {
        let (sizes, _) = projection_sizes(rows, columns, capacity)?;
        sizes[1..]
            .iter()
            .try_fold(size_of::<Self>(), |sum, &words| {
                sum.checked_add((words + GUARD) * 4)
                    .ok_or(Error::Options("BF16 scratch size overflow"))
            })
    }
    pub(in crate::metal::gemma_affine) fn owned_bytes(&self) -> usize {
        self.owned
    }
    pub(in crate::metal::gemma_affine) fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    pub(in crate::metal::gemma_affine) fn apply(
        &mut self,
        matrix: &Matrix,
        input: &[f32],
        output: &mut [f32],
        tokens: usize,
    ) -> Result<()> {
        if self.poisoned || self.pending.is_some() {
            return Err(Error::Options("BF16 shared scratch poisoned; recreate"));
        }
        if matrix.shape() != (self.rows, self.columns)
            || !matrix.same_device(self.context)
            || tokens == 0
            || tokens > self.capacity
            || input.len() != tokens * self.columns
            || output.len() != tokens * self.rows
            || input.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Options("BF16 shared scratch request differs"));
        }
        admit(self.columns, matrix.maximum, input)?;
        self.poisoned = true;
        project::run(
            &self.context.0,
            project::Matrix::bf16(self.rows, self.columns, &matrix.coefficients),
            project::Buffers {
                input: &mut self.input,
                output: &mut self.output,
                status: &mut self.status,
                pending: &mut self.pending,
            },
            input,
            output,
            tokens,
        )?;
        self.poisoned = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
