//! Cumulative requested ownership, including actual outer Vec slots once.
use super::*;

pub(super) struct Plan {
    pub(super) base: usize,
    pub(super) policy_base: usize,
    pub(super) payload: usize,
    pub(super) source: usize,
    pub(super) maximum_vector_raw: usize,
    pub(super) matrix_payloads: Vec<usize>,
}
pub(super) fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or(Error::Options(OVERFLOW))
}
pub(super) fn multiply(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or(Error::Options(OVERFLOW))
}
pub(super) fn subtract(a: usize, b: usize) -> Result<usize> {
    a.checked_sub(b).ok_or(Error::Options(OVERFLOW))
}
pub(super) fn admit(required: usize, available: usize) -> Result<()> {
    if required > available {
        return Err(Error::Budget {
            required,
            limit: available,
        });
    }
    Ok(())
}
pub(super) fn vector_bytes(values: usize) -> Result<usize> {
    multiply(add(values, GUARD)?, 4)
}
impl Plan {
    pub(super) fn new(
        inputs: &source::Inputs,
        matrix_slots: usize,
        vector_slots: usize,
    ) -> Result<Self> {
        if matrix_slots < inputs.matrices.len() || vector_slots < inputs.vectors.len() {
            return Err(Error::Options("Gemma resident owner capacity differs"));
        }
        let base = add(
            add(
                size_of::<E2bWeights>(),
                multiply(matrix_slots, size_of::<Weights>())?,
            )?,
            multiply(vector_slots, size_of::<ResidentVector>())?,
        )?;
        let policy_base = add(base, policy_overhead()?)?;
        let mut payload = 0;
        let mut source = 0;
        let mut matrix_payloads = Vec::new();
        matrix_payloads
            .try_reserve_exact(inputs.matrices.len())
            .map_err(|_| Error::Native("Gemma resident plan reservation failed".into()))?;
        for matrix in &inputs.matrices {
            let plan = super::super::weights::Plan::storage(matrix.layout, TILE, usize::MAX)?;
            let bytes = subtract(plan.minimum_owned()?, size_of::<Weights>())?;
            payload = add(payload, bytes)?;
            source = source.max(plan.minimum_source()?);
            matrix_payloads.push(bytes);
        }
        let mut maximum_vector_raw = 0;
        for vector in &inputs.vectors {
            if vector.values == 0 {
                return Err(Error::Options("Gemma resident vector empty"));
            }
            payload = add(payload, vector_bytes(vector.values)?)?;
            maximum_vector_raw = maximum_vector_raw.max(multiply(vector.values, 2)?);
        }
        source = source.max(add(maximum_vector_raw, size_of::<Vec<u8>>())?);
        Ok(Self {
            base,
            policy_base,
            payload,
            source,
            maximum_vector_raw,
            matrix_payloads,
        })
    }
    pub(super) fn validate_device(&self, inputs: &source::Inputs, maximum: usize) -> Result<()> {
        for matrix in &inputs.matrices {
            super::super::weights::Plan::storage(matrix.layout, TILE, usize::MAX)?
                .validate_device(maximum)?;
        }
        for vector in &inputs.vectors {
            validate_buffer_bytes(vector_bytes(vector.values)?, maximum)?;
        }
        Ok(())
    }
    /// Include all future minimum payload. Actual retained capacities replace,
    /// rather than add to, the corresponding minimum as construction advances.
    pub(super) fn working(
        &self,
        actual_payload: usize,
        completed_minimum: usize,
        source: usize,
    ) -> Result<usize> {
        add(
            add(
                add(self.policy_base, actual_payload)?,
                subtract(self.payload, completed_minimum)?,
            )?,
            source,
        )
    }
}
