//! Cumulative pool admission. Inline Backend/Gather owners live in Session;
//! count their external payload/descriptor capacities and policies exactly once.
use super::*;

impl<'a, 'weights, 'context> Backend<'a, 'weights, 'context> {
    pub(super) fn minimum_bytes(source: &Norms<'_>, chunk_rows: usize) -> Result<usize> {
        let mut bytes = add(
            CONTEXT_ALLOWANCE_BYTES,
            multiply(source.shapes.len(), size_of::<Projection<'_>>())?,
        )?;
        for &layout in &source.shapes {
            bytes = add(
                bytes,
                subtract(
                    Projection::minimum_owned_bytes(layout, TILE, chunk_rows)?,
                    size_of::<Projection<'_>>(),
                )?,
            )?;
        }
        for &index in &source.gather_indices {
            bytes = add(
                bytes,
                subtract(
                    Gather::minimum_admitted_bytes(
                        source.matrices[index].layout().columns(),
                        chunk_rows,
                    )?,
                    size_of::<Gather<'_>>(),
                )?,
            )?;
        }
        Ok(bytes)
    }
    pub(super) fn new(
        source: &'a Norms<'weights>,
        context: &'context Context,
        chunk_rows: usize,
        budget: usize,
    ) -> HybridResult<Self> {
        if source.matrices.iter().any(|w| !w.same_device(context)) {
            return Err(
                super::super::Error::Options("Gemma hybrid Context device mismatch").into(),
            );
        }
        let minimum = Self::minimum_bytes(source, chunk_rows)?;
        admit(minimum, budget)?;
        let mut projections = owner_vec::<Projection<'context>>(source.shapes.len())?;
        let slot_extra = multiply(
            projections.capacity() - source.shapes.len(),
            size_of::<Projection<'_>>(),
        )?;
        admit(add(minimum, slot_extra)?, budget)?;
        let mut actual = add(
            CONTEXT_ALLOWANCE_BYTES,
            multiply(projections.capacity(), size_of::<Projection<'_>>())?,
        )?;
        let mut remaining = subtract(
            minimum,
            add(
                CONTEXT_ALLOWANCE_BYTES,
                multiply(source.shapes.len(), size_of::<Projection<'_>>())?,
            )?,
        )?;
        for &layout in &source.shapes {
            let expected = subtract(
                Projection::minimum_owned_bytes(layout, TILE, chunk_rows)?,
                size_of::<Projection<'_>>(),
            )?;
            remaining = subtract(remaining, expected)?;
            let local = add(
                subtract(budget, add(actual, remaining)?)?,
                size_of::<Projection<'_>>(),
            )?;
            let workspace = context.projection_session(layout, TILE, chunk_rows, local)?;
            actual = add(
                actual,
                subtract(workspace.owned_bytes(), size_of::<Projection<'_>>())?,
            )?;
            admit(add(actual, remaining)?, budget)?;
            projections.push(workspace);
        }
        let mut gather = |index: usize| -> Result<Gather<'context>> {
            let columns = source.matrices[index].layout().columns();
            let expected = subtract(
                Gather::minimum_admitted_bytes(columns, chunk_rows)?,
                size_of::<Gather<'_>>(),
            )?;
            remaining = subtract(remaining, expected)?;
            let local = add(
                subtract(budget, add(actual, remaining)?)?,
                size_of::<Gather<'_>>(),
            )?;
            let workspace = context.gather_session(columns, chunk_rows, local)?;
            actual = add(
                actual,
                subtract(workspace.admitted_bytes(), size_of::<Gather<'_>>())?,
            )?;
            admit(add(actual, remaining)?, budget)?;
            Ok(workspace)
        };
        let gathers = [
            gather(source.gather_indices[0])?,
            gather(source.gather_indices[1])?,
        ];
        if remaining != 0 {
            return Err(
                super::super::Error::Options("Gemma hybrid scratch pool incomplete").into(),
            );
        }
        let result = Self {
            source,
            projections,
            gathers,
            failed: false,
            #[cfg(test)]
            test_fault: None,
        };
        admit(result.memory_report()?.total()?, budget)?;
        Ok(result)
    }
    pub(super) fn memory_report(&self) -> Result<BackendMemory> {
        let pipeline = Gather::pipeline_allowance_bytes();
        let mut scratch = 0;
        for p in &self.projections {
            scratch = add(
                scratch,
                subtract(p.owned_bytes(), size_of::<Projection<'_>>())?,
            )?;
        }
        for g in &self.gathers {
            scratch = add(
                scratch,
                subtract(g.admitted_bytes(), add(size_of::<Gather<'_>>(), pipeline)?)?,
            )?;
        }
        Ok(BackendMemory {
            scratch_bytes: scratch,
            metadata_bytes: multiply(self.projections.capacity(), size_of::<Projection<'_>>())?,
            metadata_allowance_bytes: add(CONTEXT_ALLOWANCE_BYTES, multiply(2, pipeline)?)?,
        })
    }
}
