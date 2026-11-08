//! Cumulative pool admission. Inline Backend/Gather owners live in Session;
//! count their external payload/descriptor capacities and policies exactly once.
use super::*;

impl<'a, 'weights, 'context> Backend<'a, 'weights, 'context> {
    pub(super) fn minimum_bytes(source: &Norms<'_>, chunk_rows: usize) -> Result<usize> {
        let mut bytes = add(
            CONTEXT_BYTES,
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
        bytes = add(
            bytes,
            subtract(
                Bf16Workspace::minimum_bytes(BF16_ROWS, BF16_COLUMNS, chunk_rows)?,
                size_of::<Bf16Workspace<'_>>(),
            )?,
        )?;
        for &index in &source.gathers {
            bytes = add(
                bytes,
                subtract(
                    Gather::minimum_admitted_bytes(
                        source.affine_weight(index)?.layout().columns(),
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
        context: &'context HereticContext,
        chunk_rows: usize,
        budget: usize,
    ) -> HResult<Self> {
        if source
            .source
            .storage
            .matrices
            .iter()
            .any(|w| !w.same_device(&context.affine))
            || !source.source.bf16.same_device(&context.bf16)
        {
            return Err(Error::Options("Heretic Context device mismatch").into());
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
            CONTEXT_BYTES,
            multiply(projections.capacity(), size_of::<Projection<'_>>())?,
        )?;
        let mut remaining = subtract(
            minimum,
            add(
                CONTEXT_BYTES,
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
            let workspace = context
                .affine
                .projection_session(layout, TILE, chunk_rows, local)?;
            actual = add(
                actual,
                subtract(workspace.owned_bytes(), size_of::<Projection<'_>>())?,
            )?;
            admit(add(actual, remaining)?, budget)?;
            projections.push(workspace);
        }
        let mut gather = |index: usize| -> Result<Gather<'context>> {
            let columns = source.affine_weight(index)?.layout().columns();
            let expected = subtract(
                Gather::minimum_admitted_bytes(columns, chunk_rows)?,
                size_of::<Gather<'_>>(),
            )?;
            remaining = subtract(remaining, expected)?;
            let local = add(
                subtract(budget, add(actual, remaining)?)?,
                size_of::<Gather<'_>>(),
            )?;
            let workspace = context.affine.gather_session(columns, chunk_rows, local)?;
            actual = add(
                actual,
                subtract(workspace.admitted_bytes(), size_of::<Gather<'_>>())?,
            )?;
            admit(add(actual, remaining)?, budget)?;
            Ok(workspace)
        };
        let gathers = [gather(source.gathers[0])?, gather(source.gathers[1])?];
        let bf16_expected = subtract(
            Bf16Workspace::minimum_bytes(BF16_ROWS, BF16_COLUMNS, chunk_rows)?,
            size_of::<Bf16Workspace<'_>>(),
        )?;
        remaining = subtract(remaining, bf16_expected)?;
        let local = add(
            subtract(budget, add(actual, remaining)?)?,
            size_of::<Bf16Workspace<'_>>(),
        )?;
        let bf16 = context
            .bf16
            .shared_workspace(BF16_ROWS, BF16_COLUMNS, chunk_rows, local)?;
        actual = add(
            actual,
            subtract(bf16.owned_bytes(), size_of::<Bf16Workspace<'_>>())?,
        )?;
        admit(add(actual, remaining)?, budget)?;
        if remaining != 0 {
            return Err(Error::Options("Heretic scratch pool incomplete").into());
        }
        let result = Self {
            source,
            projections,
            gathers,
            bf16,
            failed: false,
            #[cfg(test)]
            test_fail_after: None,
        };
        admit(result.memory_report()?.total()?, budget)?;
        Ok(result)
    }
    pub(super) fn memory_report(&self) -> Result<BackendMemory> {
        let pipeline = Gather::pipeline_allowance_bytes();
        let mut scratch = subtract(self.bf16.owned_bytes(), size_of::<Bf16Workspace<'_>>())?;
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
            metadata_allowance_bytes: add(CONTEXT_BYTES, multiply(2, pipeline)?)?,
        })
    }
}
