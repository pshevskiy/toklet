//! Immutable original packed weights, independent of session/Context lifetimes.
use super::*;
use loading::SourceScratch;

const INVALID: &str = "Gemma immutable weights geometry outside bounds";
const OVERFLOW: &str = "Gemma immutable weights size overflow";

struct Chunk {
    layout: Layout,
    codes: Buffer<u32>,
    scales: Buffer<u32>,
    biases: Buffer<u32>,
}
impl Chunk {
    fn view(&self) -> WeightView<'_> {
        WeightView {
            layout: self.layout,
            codes: &self.codes,
            scales: &self.scales,
            biases: &self.biases,
        }
    }
}
/// Original read-only packed buffers, one retained underlying device, and cold
/// metadata. No borrowed Context, activation buffers, pipelines or session state.
/// A completed construction can outlive its Context and move with a model.
/// Requested byte accounting includes every retained handle/slot and guard;
/// Context, source-callback state and session workspaces are separate policies.
pub struct Weights {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    layout: Layout,
    chunks: Vec<Chunk>,
    owned: usize,
    max_weight: f64,
}
impl Weights {
    pub fn layout(&self) -> Layout {
        self.layout
    }
    pub fn owned_bytes(&self) -> usize {
        self.owned
    }
    /// Conservative original-codebook bound, including unused codes. Retained
    /// during loading without expanding the matrix; useful to cold admission.
    pub fn max_abs_weight(&self) -> f64 {
        self.max_weight
    }
    #[cfg(test)]
    pub(super) fn diagnostic_sizes() -> (usize, usize) {
        (size_of::<Self>(), size_of::<Chunk>())
    }
    #[cfg(test)]
    pub(super) fn diagnostic_storage(&self) -> (usize, usize, String) {
        (
            self.chunks.capacity(),
            size_of::<Chunk>(),
            format!("{:p}", &*self.device),
        )
    }
    pub(super) fn views(&self) -> impl Iterator<Item = WeightView<'_>> {
        self.chunks.iter().map(Chunk::view)
    }
    pub(super) fn same_device(&self, context: &Context) -> bool {
        std::ptr::eq(&*self.device, &*context.gpu.device)
    }
}
struct Plan {
    layout: Layout,
    maximum: usize,
    chunks: usize,
    buffer_bytes: usize,
    source_bytes: [usize; 3],
    minimum_working: usize,
}
impl Plan {
    fn new(layout: Layout, maximum: usize, available: usize) -> Result<Self> {
        if layout.rows() > 262144
            || !(1..=16384).contains(&maximum)
            || layout.rows().div_ceil(maximum) > 64
        {
            return Err(Error::Options(INVALID));
        }
        let mut plan = Self {
            layout,
            maximum,
            chunks: layout.rows().div_ceil(maximum),
            buffer_bytes: 0,
            source_bytes: [0; 3],
            minimum_working: 0,
        };
        for index in 0..plan.chunks {
            let (_, child) = plan.child(index)?;
            // Reuse the existing fixed format/domain geometry, without owning
            // its activation/status allocations or creating any device object.
            Sizes::new(child, 1)?;
            let buffers = Self::buffer_lengths(child);
            for words in buffers {
                plan.buffer_bytes = plan
                    .buffer_bytes
                    .checked_add(
                        words
                            .checked_add(GUARD)
                            .and_then(|n| n.checked_mul(4))
                            .ok_or(Error::Options(OVERFLOW))?,
                    )
                    .ok_or(Error::Options(OVERFLOW))?;
            }
        }
        let (_, largest) = plan.child(0)?;
        plan.source_bytes = [
            largest.weight_bytes(),
            largest.metadata_bytes(),
            largest.metadata_bytes(),
        ];
        plan.minimum_working = plan.working(plan.chunks, plan.source_bytes)?;
        Self::admit(plan.minimum_working, available)?;
        Ok(plan)
    }
    fn child(&self, index: usize) -> Result<(usize, Layout)> {
        if index >= self.chunks {
            return Err(Error::Options(INVALID));
        }
        let first = index * self.maximum;
        Ok((
            first,
            Layout::new(
                self.maximum.min(self.layout.rows() - first),
                self.layout.columns(),
                self.layout.bits(),
                self.layout.group_size(),
                self.layout.metadata_dtype(),
            )
            .map_err(|e| Error::Options(e.0))?,
        ))
    }
    fn buffer_lengths(layout: Layout) -> [usize; 3] {
        [
            layout.weight_bytes().div_ceil(4),
            layout.metadata_bytes().div_ceil(4),
            layout.metadata_bytes().div_ceil(4),
        ]
    }
    fn owned(&self, slots: usize) -> Result<usize> {
        if slots < self.chunks {
            return Err(Error::Options(OVERFLOW));
        }
        slots
            .checked_mul(size_of::<Chunk>())
            .and_then(|n| n.checked_add(self.buffer_bytes))
            .and_then(|n| n.checked_add(size_of::<Weights>()))
            .ok_or(Error::Options(OVERFLOW))
    }
    fn working(&self, slots: usize, source: [usize; 3]) -> Result<usize> {
        source
            .into_iter()
            .try_fold(self.owned(slots)?, |sum, n| sum.checked_add(n))
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
    /// Upload an immutable matrix through three reusable initialized source
    /// arrays. The callback must overwrite every active byte on Ok. The working
    /// budget includes all resident chunks and maximum actual source capacities;
    /// callback-captured state, Context and other owners need separate admission.
    /// Original callback errors propagate; panic/unwind drops unpublished chunks
    /// without rolling back callback effects. No GPU commands are submitted.
    pub fn weights_from_fill<F>(
        &self,
        layout: Layout,
        maximum_rows: usize,
        available_working_bytes: usize,
        mut fill: F,
    ) -> Result<Weights>
    where
        F: FnMut(usize, Layout, &mut [u8], &mut [u8], &mut [u8]) -> Result<()>,
    {
        let plan = Plan::new(layout, maximum_rows, available_working_bytes)?;
        for index in 0..plan.chunks {
            let (_, child) = plan.child(index)?;
            for words in Plan::buffer_lengths(child) {
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
            .map_err(|_| Error::Native("immutable chunk reservation failed".into()))?;
        let mut source = SourceScratch::new(plan.source_bytes)?;
        Plan::admit(
            plan.working(chunks.capacity(), source.capacities())?,
            available_working_bytes,
        )?;
        let owned = plan.owned(chunks.capacity())?;
        let mut max_weight = 0_f64;
        for index in 0..plan.chunks {
            let (first, child) = plan.child(index)?;
            let [codes, scales, biases] = &mut source.arrays;
            let codes = &mut codes[..child.weight_bytes()];
            let scales = &mut scales[..child.metadata_bytes()];
            let biases = &mut biases[..child.metadata_bytes()];
            fill(first, child, codes, scales, biases)?;
            Matrix::from_le_bytes(child, codes, scales, biases).map_err(|e| Error::Options(e.0))?;
            max_weight = max_weight.max(codebook(scales, biases)?);
            chunks.push(Chunk {
                layout: child,
                codes: loading::upload(&self.gpu, codes)?,
                scales: loading::upload(&self.gpu, scales)?,
                biases: loading::upload(&self.gpu, biases)?,
            });
        }
        Ok(Weights {
            device: self.gpu.device.clone(),
            layout,
            chunks,
            owned,
            max_weight,
        })
    }
}
#[cfg(test)]
mod tests;
