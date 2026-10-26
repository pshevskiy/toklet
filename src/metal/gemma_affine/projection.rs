//! Fixed-geometry mutable projection scratch over independent immutable weights.
use super::*;

const INVALID: &str = "Gemma projection workspace geometry outside bounds";
const OVERFLOW: &str = "Gemma projection workspace size overflow";
const MISMATCH: &str = "Gemma projection weights device/shape/chunk mismatch";

struct Plan {
    layout: Layout,
    capacity: usize,
    maximum_rows: usize,
    input_values: usize,
    full_values: usize,
    chunk_values: usize,
    minimum_owned: usize,
}
impl Plan {
    fn new(layout: Layout, maximum_rows: usize, capacity: usize, budget: usize) -> Result<Self> {
        if layout.rows() > 262144
            || !(1..=16384).contains(&maximum_rows)
            || layout.rows().div_ceil(maximum_rows) > 64
        {
            return Err(Error::Options(INVALID));
        }
        let maximum_rows = maximum_rows.min(layout.rows());
        let child = Layout::new(
            maximum_rows,
            layout.columns(),
            layout.bits(),
            layout.group_size(),
            layout.metadata_dtype(),
        )
        .map_err(|e| Error::Options(e.0))?;
        let child_sizes = Sizes::new(child, capacity)?;
        let mut plan = Self {
            layout,
            capacity,
            maximum_rows,
            input_values: child_sizes.input,
            full_values: capacity
                .checked_mul(layout.rows())
                .ok_or(Error::Options(OVERFLOW))?,
            chunk_values: child_sizes.output,
            minimum_owned: 0,
        };
        plan.minimum_owned = plan.owned(plan.full_values, plan.chunk_values)?;
        Self::admit(plan.minimum_owned, budget)?;
        Ok(plan)
    }
    fn buffers(&self) -> [usize; 3] {
        [self.input_values, self.chunk_values, 1]
    }
    fn owned(&self, full_capacity: usize, chunk_capacity: usize) -> Result<usize> {
        if full_capacity < self.full_values || chunk_capacity < self.chunk_values {
            return Err(Error::Options(OVERFLOW));
        }
        let gpu = self.buffers().into_iter().try_fold(0_usize, |sum, words| {
            sum.checked_add(words)?.checked_add(GUARD)
        });
        gpu.and_then(|n| n.checked_add(full_capacity))
            .and_then(|n| n.checked_add(chunk_capacity))
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(size_of::<Projection<'_>>()))
            .ok_or(Error::Options(OVERFLOW))
    }
    fn admit(required: usize, limit: usize) -> Result<()> {
        if required > limit {
            return Err(Error::Budget { required, limit });
        }
        Ok(())
    }
    // Validate every child before numerical admission or any buffer mutation.
    // Fixed total geometry alone does not establish per-child scratch bounds.
    fn accepts(&self, chunks: impl Iterator<Item = Layout>) -> bool {
        let mut rows = 0_usize;
        let mut count = 0;
        for child in chunks {
            count += 1;
            if count > 64
                || child.rows() > self.maximum_rows
                || child.columns() != self.layout.columns()
                || child.bits() != self.layout.bits()
                || child.group_size() != self.layout.group_size()
                || child.metadata_dtype() != self.layout.metadata_dtype()
            {
                return false;
            }
            let Some(total) = rows.checked_add(child.rows()) else {
                return false;
            };
            if total > self.layout.rows() {
                return false;
            }
            rows = total;
        }
        count != 0 && rows == self.layout.rows()
    }
}
/// One request workspace for fixed total matrix geometry and batch capacity.
/// Owns only mutable GPU scratch and bounded transactional CPU staging; borrows
/// immutable weights per request. Accounting excludes weights, Context, caller
/// and loading/diagnostic overlap. No Rust routing allocation after admission;
/// Metal command/encoder creation may allocate.
pub struct Projection<'a> {
    context: &'a Context,
    plan: Plan,
    owned: usize,
    input: Buffer<u32>,
    output: Buffer<u32>,
    status: Buffer<u32>,
    staged: Vec<f32>,
    chunk_output: Vec<f32>,
    poisoned: bool,
    pending: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}
impl Context {
    pub fn projection_session(
        &self,
        layout: Layout,
        maximum_chunk_rows: usize,
        capacity: usize,
        available_owned_bytes: usize,
    ) -> Result<Projection<'_>> {
        let plan = Plan::new(layout, maximum_chunk_rows, capacity, available_owned_bytes)?;
        for words in plan.buffers() {
            let bytes = words
                .checked_add(GUARD)
                .and_then(|n| n.checked_mul(4))
                .ok_or(Error::Options(OVERFLOW))?;
            validate_buffer_bytes(bytes, self.gpu.device.maxBufferLength())?;
        }
        let reserve = |values| -> Result<Vec<f32>> {
            let mut out = Vec::new();
            out.try_reserve_exact(values)
                .map_err(|_| Error::Native("projection scratch reservation failed".into()))?;
            Ok(out)
        };
        let mut staged = reserve(plan.full_values)?;
        let mut chunk_output = reserve(plan.chunk_values)?;
        let owned = plan.owned(staged.capacity(), chunk_output.capacity())?;
        Plan::admit(owned, available_owned_bytes)?;
        staged.resize(plan.full_values, 0.);
        chunk_output.resize(plan.chunk_values, 0.);
        Ok(Projection {
            context: self,
            owned,
            input: Buffer::new(&self.gpu, plan.input_values + GUARD)?,
            output: Buffer::new(&self.gpu, plan.chunk_values + GUARD)?,
            status: Buffer::new(&self.gpu, 1 + GUARD)?,
            plan,
            staged,
            chunk_output,
            poisoned: false,
            pending: None,
        })
    }
}
impl Projection<'_> {
    pub(crate) fn minimum_owned_bytes(
        layout: Layout,
        maximum: usize,
        capacity: usize,
    ) -> Result<usize> {
        Ok(Plan::new(layout, maximum, capacity, usize::MAX)?.minimum_owned)
    }
    pub fn owned_bytes(&self) -> usize {
        self.owned
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// Publish the entire [tokens, rows] output only after every child completes.
    /// Preflight errors preserve all workspace/caller state. Failures after
    /// staging poison only this workspace; fresh scratch can reuse the weights.
    pub fn apply(
        &mut self,
        weights: &Weights,
        input: &[f32],
        output: &mut [f32],
        tokens: usize,
    ) -> Result<()> {
        if self.poisoned || self.pending.is_some() {
            return Err(Error::Options(
                "Gemma projection workspace poisoned; recreate",
            ));
        }
        if tokens == 0
            || tokens > self.plan.capacity
            || input.len() != tokens * self.plan.layout.columns()
            || output.len() != tokens * self.plan.layout.rows()
            || input.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Options(
                "Gemma projection input shape/nonfinite value",
            ));
        }
        if weights.layout() != self.plan.layout
            || !weights.same_device(self.context)
            || !self.plan.accepts(weights.views().map(|child| child.layout))
        {
            return Err(Error::Options(MISMATCH));
        }
        // The maximum child codebook bound controls the same monotone max/L1
        // admission for every child, before any numerical staging begins.
        admit(self.plan.layout.columns(), weights.max_abs_weight(), input)?;
        self.poisoned = true;
        let mut first = 0;
        #[cfg(test)]
        let mut completed = 0;
        for child in weights.views() {
            let rows = child.layout.rows();
            let active = &mut self.chunk_output[..tokens * rows];
            project::run(
                self.context,
                child,
                project::Buffers {
                    input: &mut self.input,
                    output: &mut self.output,
                    status: &mut self.status,
                    pending: &mut self.pending,
                },
                input,
                active,
                tokens,
            )?;
            for (token, values) in active.chunks_exact(rows).enumerate() {
                let start = token * self.plan.layout.rows() + first;
                self.staged[start..start + rows].copy_from_slice(values);
            }
            first += rows;
            #[cfg(test)]
            {
                completed += 1;
                if FAILURE_AFTER.with(std::cell::Cell::get) == Some(completed) {
                    return Err(Error::Options(
                        "Gemma projection injected composition failure",
                    ));
                }
            }
        }
        output.copy_from_slice(&self.staged[..output.len()]);
        self.poisoned = false;
        Ok(())
    }
}
#[cfg(test)]
thread_local! {static FAILURE_AFTER:std::cell::Cell<Option<usize>>=const{std::cell::Cell::new(None)};}
#[cfg(test)]
mod tests;
