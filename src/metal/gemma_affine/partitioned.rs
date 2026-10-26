//! Bounded simultaneous affine chunks. Explicit host staging is diagnostic
//! correctness composition, not a resident whole graph or optimized GEMM.
use super::*;

mod gather;
mod upload;
pub use gather::Gather;

const MAX_ROWS: usize = 262144;
const MAX_CHUNKS: usize = 64;
const INVALID: &str = "Gemma partitioned affine fixed geometry/format outside bounds";
const OVERFLOW: &str = "Gemma partitioned affine size overflow";

struct Plan {
    layout: Layout,
    capacity: usize,
    maximum_rows: usize,
    chunks: usize,
    full_values: usize,
    chunk_values: usize,
    children_owned: usize,
    minimum_owned: usize,
}
impl Plan {
    fn new(layout: Layout, capacity: usize, maximum_rows: usize, budget: usize) -> Result<Self> {
        if layout.rows() > MAX_ROWS
            || !(1..=16384).contains(&maximum_rows)
            || !(1..=32).contains(&capacity)
        {
            return Err(Error::Options(INVALID));
        }
        let chunks = layout.rows().div_ceil(maximum_rows);
        if chunks > MAX_CHUNKS {
            return Err(Error::Options(INVALID));
        }
        let mut plan = Self {
            layout,
            capacity,
            maximum_rows,
            chunks,
            full_values: capacity
                .checked_mul(layout.rows())
                .ok_or(Error::Options(OVERFLOW))?,
            chunk_values: capacity
                .checked_mul(maximum_rows.min(layout.rows()))
                .ok_or(Error::Options(OVERFLOW))?,
            children_owned: 0,
            minimum_owned: 0,
        };
        // At most64 cold chunks, with existing format/capacity checks selected
        // before any numerical allocations or device operation.
        for index in 0..chunks {
            let (_, child) = plan.child(index)?;
            plan.children_owned = plan
                .children_owned
                .checked_add(Sizes::new(child, capacity)?.owned)
                .ok_or(Error::Options(OVERFLOW))?;
        }
        plan.minimum_owned = plan.owned(chunks, plan.full_values, plan.chunk_values)?;
        if plan.minimum_owned > budget {
            return Err(Error::Budget {
                required: plan.minimum_owned,
                limit: budget,
            });
        }
        Ok(plan)
    }
    fn child(&self, index: usize) -> Result<(usize, Layout)> {
        if index >= self.chunks {
            return Err(Error::Options(INVALID));
        }
        let first = index
            .checked_mul(self.maximum_rows)
            .ok_or(Error::Options(OVERFLOW))?;
        let count = self.maximum_rows.min(self.layout.rows() - first);
        Ok((
            first,
            Layout::new(
                count,
                self.layout.columns(),
                self.layout.bits(),
                self.layout.group_size(),
                self.layout.metadata_dtype(),
            )
            .map_err(|e| Error::Options(e.0))?,
        ))
    }
    fn owned(&self, slots: usize, full_capacity: usize, chunk_capacity: usize) -> Result<usize> {
        // children_owned already counts live Affine structs; only unused Vec
        // slots are additional. The three Vec descriptors live in Self's owner.
        let unused = slots
            .checked_sub(self.chunks)
            .and_then(|n| n.checked_mul(size_of::<Affine<'_>>()))
            .ok_or(Error::Options(OVERFLOW))?;
        let scratch = full_capacity
            .checked_add(chunk_capacity)
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .ok_or(Error::Options(OVERFLOW))?;
        self.children_owned
            .checked_add(unused)
            .and_then(|n| n.checked_add(scratch))
            .and_then(|n| n.checked_add(size_of::<PartitionedAffine<'_>>()))
            .ok_or(Error::Options(OVERFLOW))
    }
}

/// Simultaneous original packed chunks and fixed transactional host scratch.
/// Owned-byte accounting excludes Context policy, caller output, and the caller's
/// CPU source/loading overlap; admit those separately. No original CPU table or
/// expanded weights are retained by this owner after construction.
pub struct PartitionedAffine<'a> {
    chunks: Vec<Affine<'a>>,
    staged: Vec<f32>,
    chunk_output: Vec<f32>,
    rows: usize,
    columns: usize,
    capacity: usize,
    owned: usize,
    poisoned: bool,
}
impl Context {
    /// Upload bounded resident chunks after aggregate owner-only admission.
    /// The original single-owner affine geometry and numerical gates stay intact.
    pub fn partitioned_affine(
        &self,
        matrix: &Matrix<'_>,
        maximum_rows: usize,
        capacity: usize,
        available_owned_bytes: usize,
    ) -> Result<PartitionedAffine<'_>> {
        let plan = Plan::new(
            matrix.layout(),
            capacity,
            maximum_rows,
            available_owned_bytes,
        )?;
        for index in 0..plan.chunks {
            let (_, layout) = plan.child(index)?;
            for words in Sizes::new(layout, capacity)?.buffers() {
                let bytes = words
                    .checked_add(GUARD)
                    .and_then(|n| n.checked_mul(4))
                    .ok_or(Error::Options(OVERFLOW))?;
                validate_buffer_bytes(bytes, self.gpu.device.maxBufferLength())?;
            }
        }
        let (_, scales, biases) = matrix.original_tensors();
        codebook(scales, biases)?;
        let mut chunks = Vec::new();
        chunks
            .try_reserve_exact(plan.chunks)
            .map_err(|_| Error::Native("partition owner reservation failed".into()))?;
        let reserve = |values| -> Result<Vec<f32>> {
            let mut buffer = Vec::new();
            buffer
                .try_reserve_exact(values)
                .map_err(|_| Error::Native("partition scratch reservation failed".into()))?;
            Ok(buffer)
        };
        let mut staged = reserve(plan.full_values)?;
        let mut chunk_output = reserve(plan.chunk_values)?;
        let owned = plan.owned(
            chunks.capacity(),
            staged.capacity(),
            chunk_output.capacity(),
        )?;
        if owned > available_owned_bytes {
            return Err(Error::Budget {
                required: owned,
                limit: available_owned_bytes,
            });
        }
        staged.resize(plan.full_values, 0.);
        chunk_output.resize(plan.chunk_values, 0.);
        for index in 0..plan.chunks {
            let (first, layout) = plan.child(index)?;
            let view = matrix
                .row_range(first..first + layout.rows())
                .map_err(|e| Error::Options(e.0))?;
            let (codes, scales, biases) = view.original_tensors();
            let child_bytes = Sizes::new(layout, capacity)?.owned;
            chunks.push(self.affine(layout, codes, scales, biases, capacity, child_bytes)?);
        }
        Ok(PartitionedAffine {
            chunks,
            staged,
            chunk_output,
            rows: plan.layout.rows(),
            columns: plan.layout.columns(),
            capacity: plan.capacity,
            owned,
            poisoned: false,
        })
    }
}
impl PartitionedAffine<'_> {
    pub fn owned_bytes(&self) -> usize {
        self.owned
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// Caller output changes only after every chunk completes successfully.
    /// All-child numerical preflight errors preserve all buffers and reusability;
    /// any failure after staging begins permanently poisons the whole owner.
    pub fn apply(&mut self, input: &[f32], output: &mut [f32], tokens: usize) -> Result<()> {
        if self.poisoned {
            return Err(Error::Options(
                "Gemma partitioned affine owner poisoned; recreate",
            ));
        }
        if tokens == 0
            || tokens > self.capacity
            || input.len() != tokens * self.columns
            || output.len() != tokens * self.rows
            || input.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Options(
                "Gemma partitioned affine input shape/nonfinite value",
            ));
        }
        for child in &self.chunks {
            admit(self.columns, child.max_weight, input)?;
        }
        self.poisoned = true;
        let mut first = 0;
        #[cfg(test)]
        let mut completed = 0;
        for child in &mut self.chunks {
            let n = child.layout.rows();
            let active = &mut self.chunk_output[..tokens * n];
            child.apply(input, active, tokens)?;
            for (token, values) in active.chunks_exact(n).enumerate() {
                let start = token * self.rows + first;
                self.staged[start..start + n].copy_from_slice(values);
            }
            first += n;
            #[cfg(test)]
            {
                completed += 1;
                if FAILURE_AFTER.with(std::cell::Cell::get) == Some(completed) {
                    return Err(Error::Options(
                        "Gemma partitioned affine injected composition failure",
                    ));
                }
            }
        }
        debug_assert_eq!(first, self.rows);
        output.copy_from_slice(&self.staged[..tokens * self.rows]);
        self.poisoned = false;
        Ok(())
    }
}
#[cfg(test)]
thread_local! {
    static FAILURE_AFTER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(super) mod tests;
