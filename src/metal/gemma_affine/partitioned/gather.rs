//! Separate lookup scratch over the already resident immutable packed rows.
use super::*;

const PIPELINE_ALLOWANCE: usize = 16 * 1024 * 1024;
const INVALID: &str = "Gemma table gather geometry outside bounds";
const OVERFLOW: &str = "Gemma table gather size overflow";

struct Sizes {
    capacity: usize,
    values: usize,
    minimum_admitted: usize,
}
impl Sizes {
    fn new(columns: usize, capacity: usize, budget: usize) -> Result<Self> {
        if !(64..=12288).contains(&columns)
            || !columns.is_multiple_of(64)
            || !(1..=32).contains(&capacity)
        {
            return Err(Error::Options(INVALID));
        }
        let mut sizes = Self {
            capacity,
            values: capacity
                .checked_mul(columns)
                .ok_or(Error::Options(OVERFLOW))?,
            minimum_admitted: 0,
        };
        sizes.minimum_admitted = sizes.admitted(sizes.values, sizes.values)?;
        if sizes.minimum_admitted > budget {
            return Err(Error::Budget {
                required: sizes.minimum_admitted,
                limit: budget,
            });
        }
        Ok(sizes)
    }
    fn buffers(&self) -> [usize; 3] {
        [self.capacity, self.values, 1]
    }
    fn admitted(&self, full_capacity: usize, chunk_capacity: usize) -> Result<usize> {
        let gpu = self
            .buffers()
            .into_iter()
            .try_fold(0_usize, |sum, words| {
                sum.checked_add(words)?.checked_add(GUARD)
            })
            .and_then(|words| words.checked_mul(4))
            .ok_or(Error::Options(OVERFLOW))?;
        full_capacity
            .checked_add(chunk_capacity)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(gpu))
            .and_then(|n| n.checked_add(size_of::<Gather<'_>>()))
            .and_then(|n| n.checked_add(PIPELINE_ALLOWANCE))
            .ok_or(Error::Options(OVERFLOW))
    }
}

/// Reused lookup workspace; borrows existing resident weights per request.
/// Its admitted bytes include buffers, CPU capacity, owner and an opaque pipeline
/// policy allowance. Caller output and the existing matrix/Context are separate.
pub struct Gather<'a> {
    context: &'a Context,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    width: usize,
    columns: usize,
    capacity: usize,
    admitted: usize,
    ids: Buffer<u32>,
    output: Buffer<u32>,
    status: Buffer<u32>,
    staged: Vec<f32>,
    chunk: Vec<f32>,
    poisoned: bool,
    pending: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}
impl Context {
    /// Cache one separate precise lookup pipeline in the new reusable workspace.
    /// Existing affine initialization and its shader remain unchanged.
    pub fn gather_session(
        &self,
        columns: usize,
        capacity: usize,
        available_bytes: usize,
    ) -> Result<Gather<'_>> {
        let sizes = Sizes::new(columns, capacity, available_bytes)?;
        for words in sizes.buffers() {
            let bytes = words
                .checked_add(GUARD)
                .and_then(|n| n.checked_mul(4))
                .ok_or(Error::Options(OVERFLOW))?;
            validate_buffer_bytes(bytes, self.gpu.device.maxBufferLength())?;
        }
        let reserve = || -> Result<Vec<f32>> {
            let mut out = Vec::new();
            out.try_reserve_exact(sizes.values)
                .map_err(|_| Error::Native("gather scratch reservation failed".into()))?;
            Ok(out)
        };
        let mut staged = reserve()?;
        let mut chunk = reserve()?;
        let admitted = sizes.admitted(staged.capacity(), chunk.capacity())?;
        if admitted > available_bytes {
            return Err(Error::Budget {
                required: admitted,
                limit: available_bytes,
            });
        }
        staged.resize(sizes.values, 0.);
        chunk.resize(sizes.values, 0.);
        autoreleasepool(|_| {
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version2_3);
            options.setMathMode(MTLMathMode::Safe);
            options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
            if options.languageVersion() != MTLLanguageVersion::Version2_3
                || options.mathMode() != MTLMathMode::Safe
                || options.mathFloatingPointFunctions() != MTLMathFloatingPointFunctions::Precise
            {
                return Err(Error::Unsupported(
                    "Gemma table gather precise math unavailable",
                ));
            }
            let library = self
                .gpu
                .device
                .newLibraryWithSource_options_error(
                    &NSString::from_str(include_str!("gather.metal")),
                    Some(&options),
                )
                .map_err(|e| Error::Native(e.to_string()))?;
            let function = library
                .newFunctionWithName(&NSString::from_str("gemma_table_gather"))
                .ok_or_else(|| Error::Native("gather kernel absent".into()))?;
            let pipeline = self
                .gpu
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| Error::Native(e.to_string()))?;
            let width = pipeline.threadExecutionWidth();
            if width == 0
                || width > pipeline.maxTotalThreadsPerThreadgroup()
                || width > u16::MAX as usize
            {
                return Err(Error::Unsupported("gather launch width unavailable"));
            }
            Ok(Gather {
                context: self,
                pipeline,
                width,
                columns,
                capacity,
                admitted,
                ids: Buffer::new(&self.gpu, capacity + GUARD)?,
                output: Buffer::new(&self.gpu, sizes.values + GUARD)?,
                status: Buffer::new(&self.gpu, 1 + GUARD)?,
                staged,
                chunk,
                poisoned: false,
                pending: None,
            })
        })
    }
}
impl Gather<'_> {
    pub(crate) fn minimum_admitted_bytes(columns: usize, capacity: usize) -> Result<usize> {
        Ok(Sizes::new(columns, capacity, usize::MAX)?.minimum_admitted)
    }
    pub(crate) fn pipeline_allowance_bytes() -> usize {
        PIPELINE_ALLOWANCE
    }
    /// Known requested storage plus opaque pipeline policy, not measured physical residency.
    pub fn admitted_bytes(&self) -> usize {
        self.admitted
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// No Rust allocation for routing/staging after admission. Metal command or
    /// encoder creation may allocate. Borrowed matrix weights are never changed.
    pub fn apply(
        &mut self,
        matrix: &PartitionedAffine<'_>,
        ids: &[usize],
        output: &mut [f32],
    ) -> Result<()> {
        self.check_request(matrix.rows, ids, output.len())?;
        if matrix.columns != self.columns
            || matrix.poisoned
            || matrix.chunks.iter().any(|child| {
                !std::ptr::eq(self.context, child.context)
                    || child.poisoned
                    || child.pending.is_some()
            })
        {
            return Err(Error::Options(
                "Gemma table gather matrix Context/shape/state mismatch",
            ));
        }
        self.apply_views(matrix.chunks.iter().map(Affine::weight_view), ids, output)
    }
    /// Gather directly from immutable original weights. The same underlying
    /// Metal device is required; the construction Context need not stay alive.
    /// Workspace failures never mutate the shared weights; a fresh workspace
    /// can reuse them after a completed-request failure without uploading again.
    pub fn apply_weights(
        &mut self,
        weights: &Weights,
        ids: &[usize],
        output: &mut [f32],
    ) -> Result<()> {
        self.check_request(weights.layout().rows(), ids, output.len())?;
        if weights.layout().columns() != self.columns || !weights.same_device(self.context) {
            return Err(Error::Options(
                "Gemma table gather weights device/shape mismatch",
            ));
        }
        self.apply_views(weights.views(), ids, output)
    }
    fn check_request(&self, rows: usize, ids: &[usize], output_len: usize) -> Result<()> {
        if self.poisoned || self.pending.is_some() {
            return Err(Error::Options(
                "Gemma table gather poisoned; recreate workspace",
            ));
        }
        if ids.is_empty()
            || ids.len() > self.capacity
            || ids.len().checked_mul(self.columns) != Some(output_len)
            || ids.iter().any(|&id| id >= rows)
        {
            return Err(Error::Options(
                "Gemma table gather request shape or indices",
            ));
        }
        Ok(())
    }
    // Two concrete iterator types select their source before encoding. Views
    // are borrowed stack descriptors, with no materialized collection or dyn
    // numerical dispatch. Both sources share every command/completion check.
    fn apply_views<'w>(
        &mut self,
        views: impl Iterator<Item = WeightView<'w>>,
        ids: &[usize],
        output: &mut [f32],
    ) -> Result<()> {
        self.poisoned = true;
        let mut first = 0;
        #[cfg(test)]
        let mut completed = 0;
        for child in views {
            let rows = child.layout.rows();
            let mut local = [0_u32; 32];
            let mut slots = [0_usize; 32];
            let mut count = 0;
            for (slot, &id) in ids.iter().enumerate() {
                if (first..first + rows).contains(&id) {
                    local[count] = (id - first) as u32;
                    slots[count] = slot;
                    count += 1;
                }
            }
            first += rows;
            if count == 0 {
                continue;
            }
            self.run(child, &local[..count])?;
            for (values, &slot) in self.chunk[..count * self.columns]
                .chunks_exact(self.columns)
                .zip(&slots[..count])
            {
                self.staged[slot * self.columns..(slot + 1) * self.columns].copy_from_slice(values);
            }
            #[cfg(test)]
            {
                completed += 1;
                if FAILURE_AFTER.with(std::cell::Cell::get) == Some(completed) {
                    return Err(Error::Options(
                        "Gemma table gather injected composition failure",
                    ));
                }
            }
        }
        output.copy_from_slice(&self.staged[..output.len()]);
        self.poisoned = false;
        Ok(())
    }
    fn run(&mut self, child: WeightView<'_>, local: &[u32]) -> Result<()> {
        autoreleasepool(|_| {
            self.ids.with_mut(|out| {
                out.fill(SENTINEL);
                out[..local.len()].copy_from_slice(local);
            });
            self.output.with_mut(|out| out.fill(SENTINEL));
            self.status.with_mut(|out| {
                out.fill(SENTINEL);
                out[0] = 0;
            });
            // Gpu::command rejects retainedReferences=false. The retained command
            // owns encoded matrix/scratch resources beyond this request borrow,
            // including an error return before completion. No CPU read precedes
            // checked completion; queue/command retention protects teardown.
            let command = self.context.gpu.command()?;
            let encoder = command
                .computeCommandEncoder()
                .ok_or_else(|| Error::Native("gather encoder absent".into()))?;
            encoder.setComputePipelineState(&self.pipeline);
            let shape = [
                local.len() as u32,
                self.columns as u32,
                child.layout.rows() as u32,
            ];
            // SAFETY: fixed ABI; validated same-Context buffers are initialized,
            // aligned, disjoint and sized with guards. Original weight buffers
            // are read-only. Uniform bytes copy immediately. Tracked resources
            // are retained by the command; host views wait for completion.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&child.codes.raw), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&child.scales.raw), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&child.biases.raw), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&self.ids.raw), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&self.output.raw), 0, 4);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&shape).cast(),
                    size_of::<[u32; 3]>(),
                    5,
                );
                encoder.setBuffer_offset_atIndex(Some(&self.status.raw), 0, 6);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (local.len() * self.columns).div_ceil(self.width),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.width,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            self.pending = Some(command);
            let command = self.pending.as_ref().unwrap();
            command.commit();
            command.waitUntilCompleted();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(Error::Completion);
            }
            let tail = |words: &[u32], active| words[active..].iter().all(|&v| v == SENTINEL);
            if !self.ids.with_slice(|v| tail(v, local.len()))
                || !self
                    .output
                    .with_slice(|v| tail(v, local.len() * self.columns))
                || !self.status.with_slice(|v| tail(v, 1))
                || !child
                    .codes
                    .with_slice(|v| tail(v, child.layout.weight_bytes().div_ceil(4)))
                || !child
                    .scales
                    .with_slice(|v| tail(v, child.layout.metadata_bytes().div_ceil(4)))
                || !child
                    .biases
                    .with_slice(|v| tail(v, child.layout.metadata_bytes().div_ceil(4)))
            {
                return Err(Error::Options(
                    "Gemma table gather guard/inactive capacity changed",
                ));
            }
            if self.status.with_slice(|v| v[0]) != 0
                || !self.output.with_slice(|v| {
                    v[..local.len() * self.columns]
                        .iter()
                        .all(|&v| f32::from_bits(v).is_finite())
                })
            {
                return Err(Error::Nonfinite);
            }
            self.output.with_slice(|words| {
                for (out, &bits) in self
                    .chunk
                    .iter_mut()
                    .zip(&words[..local.len() * self.columns])
                {
                    *out = f32::from_bits(bits);
                }
            });
            self.pending = None;
            Ok(())
        })
    }
}
#[cfg(test)]
thread_local! {static FAILURE_AFTER:std::cell::Cell<Option<usize>>=const{std::cell::Cell::new(None)};}

#[cfg(test)]
mod tests;
