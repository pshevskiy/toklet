//! Cold bounded source staging; numerical kernels and artifact IO stay separate.
use super::*;

const OVERFLOW: &str = "Gemma stream upload size overflow";

struct SourceScratch {
    arrays: [Vec<u8>; 3],
}
impl SourceScratch {
    fn new(lengths: [usize; 3]) -> Result<Self> {
        let reserve = |length| -> Result<Vec<u8>> {
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(length)
                .map_err(|_| Error::Native("stream source reservation failed".into()))?;
            bytes.resize(length, 0);
            Ok(bytes)
        };
        Ok(Self {
            arrays: [
                reserve(lengths[0])?,
                reserve(lengths[1])?,
                reserve(lengths[2])?,
            ],
        })
    }
    fn capacities(&self) -> [usize; 3] {
        self.arrays.each_ref().map(Vec::capacity)
    }
}
struct UploadPlan {
    resident: Plan,
    source_bytes: [usize; 3],
    minimum_working: usize,
}
impl UploadPlan {
    fn new(layout: Layout, maximum_rows: usize, capacity: usize, available: usize) -> Result<Self> {
        let resident = Plan::new(layout, capacity, maximum_rows, usize::MAX)?;
        let (_, largest) = resident.child(0)?;
        let source_bytes = [
            largest.weight_bytes(),
            largest.metadata_bytes(),
            largest.metadata_bytes(),
        ];
        let mut plan = Self {
            resident,
            source_bytes,
            minimum_working: 0,
        };
        plan.minimum_working = plan.working(
            plan.resident.chunks,
            plan.resident.full_values,
            plan.resident.chunk_values,
            source_bytes,
        )?;
        Self::admit(plan.minimum_working, available)?;
        Ok(plan)
    }
    fn working(
        &self,
        slots: usize,
        full_capacity: usize,
        chunk_capacity: usize,
        source_capacities: [usize; 3],
    ) -> Result<usize> {
        source_capacities
            .into_iter()
            .try_fold(
                self.resident.owned(slots, full_capacity, chunk_capacity)?,
                |sum, bytes| sum.checked_add(bytes),
            )
            .and_then(|n| n.checked_add(size_of::<SourceScratch>()))
            .ok_or(Error::Options(OVERFLOW))
    }
    fn admit(required: usize, limit: usize) -> Result<()> {
        if required > limit {
            return Err(Error::Budget { required, limit });
        }
        Ok(())
    }
}
impl Context {
    /// Load original packed chunks using one reusable initialized source buffer
    /// per tensor. `available_working_bytes` admits the entire resident owner
    /// plus all three source capacities/descriptors at their construction peak.
    /// Context, other models and all callback-captured/adapter state are separate.
    /// Returned `owned_bytes()` excludes the released source staging.
    ///
    /// `fill(first_row, child_layout, codes, scales, biases)` must completely
    /// overwrite all three active byte slices on every successful return. Reused
    /// initialized bytes ensure memory safety, not completeness or file identity;
    /// authentication, seek/read_exact and external effects belong to the caller.
    /// A late error or unwinding panic drops the unpublished partial owner; it
    /// does not undo callback side effects. Upload submits no GPU commands.
    pub fn partitioned_affine_from_fill<F>(
        &self,
        layout: Layout,
        maximum_rows: usize,
        capacity: usize,
        available_working_bytes: usize,
        mut fill: F,
    ) -> Result<PartitionedAffine<'_>>
    where
        F: FnMut(usize, Layout, &mut [u8], &mut [u8], &mut [u8]) -> Result<()>,
    {
        let upload = UploadPlan::new(layout, maximum_rows, capacity, available_working_bytes)?;
        let plan = &upload.resident;
        // Preflight every eventual device allocation before source reservation,
        // callback effects or partial GPU uploads.
        for index in 0..plan.chunks {
            let (_, child) = plan.child(index)?;
            for words in Sizes::new(child, capacity)?.buffers() {
                let bytes = words
                    .checked_add(GUARD)
                    .and_then(|n| n.checked_mul(4))
                    .ok_or(Error::Options(OVERFLOW))?;
                validate_buffer_bytes(bytes, self.gpu.device.maxBufferLength())?;
            }
        }
        let mut chunks = Vec::new();
        chunks
            .try_reserve_exact(plan.chunks)
            .map_err(|_| Error::Native("stream owner reservation failed".into()))?;
        let reserve = |values| -> Result<Vec<f32>> {
            let mut scratch = Vec::new();
            scratch
                .try_reserve_exact(values)
                .map_err(|_| Error::Native("stream scratch reservation failed".into()))?;
            Ok(scratch)
        };
        let mut staged = reserve(plan.full_values)?;
        let mut chunk_output = reserve(plan.chunk_values)?;
        let mut source = SourceScratch::new(upload.source_bytes)?;
        UploadPlan::admit(
            upload.working(
                chunks.capacity(),
                staged.capacity(),
                chunk_output.capacity(),
                source.capacities(),
            )?,
            available_working_bytes,
        )?;
        let owned = plan.owned(
            chunks.capacity(),
            staged.capacity(),
            chunk_output.capacity(),
        )?;
        staged.resize(plan.full_values, 0.);
        chunk_output.resize(plan.chunk_values, 0.);
        for index in 0..plan.chunks {
            let (first, child) = plan.child(index)?;
            let [codes, scales, biases] = &mut source.arrays;
            let codes = &mut codes[..child.weight_bytes()];
            let scales = &mut scales[..child.metadata_bytes()];
            let biases = &mut biases[..child.metadata_bytes()];
            fill(first, child, codes, scales, biases)?;
            // Existing affine admission checks every completed chunk's exact
            // representation/domain before synchronous shared-buffer copies.
            let child_owned = Sizes::new(child, capacity)?.owned;
            chunks.push(self.affine(child, codes, scales, biases, capacity, child_owned)?);
        }
        Ok(PartitionedAffine {
            chunks,
            staged,
            chunk_output,
            rows: layout.rows(),
            columns: layout.columns(),
            capacity,
            owned,
            poisoned: false,
        })
    }
}

#[cfg(test)]
mod tests;
