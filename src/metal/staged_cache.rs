//! Bounded resident Half cache-set transaction; no graph or token ownership.
//! Model owns Operation/Plan; a session owns Set and its old/staging/status.
//! Views borrow resources only while encoding. Frames/Ready contain metadata,
//! not a Set borrow. Only Set submits commands and can mint readiness.
use super::{
    Error, Result,
    memory::{Buffer, Gpu, validate_buffer_bytes},
    profile::Command,
};
use crate::{
    capacity::{KvOwner, KvStorage, Retention},
    model::window_rows,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::*;
use std::{
    mem::size_of,
    ptr::NonNull,
    sync::atomic::{AtomicU64, Ordering},
};

const MAX: usize = 32;
const CHUNK: usize = 32;
const GUARD: usize = 7;
const PAD: u16 = 0x7e55;
const STATUS_PAD: u32 = 0x5a170031;
pub(super) const PIPELINE_ALLOWANCE: usize = 64 * 1024 * 1024;
// Caller Batch, validation/return temporaries, Ready, one ephemeral View, and
// copy descriptors are explicitly reserved in addition to Set's inline owners.
// Framework/compiler stack overhead remains in the application allowance.
const TRANSACTION_METADATA_BYTES: usize = 3 * size_of::<Batch>()
    + size_of::<Ready>()
    + size_of::<View<'static>>()
    + size_of::<CommitInfo>()
    + MAX * 2 * size_of::<window_rows::CopyRun>();
static NEXT_SET: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug)]
pub(super) struct OwnerConfig {
    pub kv_heads: usize,
    pub dim: usize,
    pub window: Option<usize>,
}
#[derive(Clone, Copy)]
struct OwnerPlan {
    config: OwnerConfig,
    retained: usize,
    old_len: usize,
    staged_len: usize,
}
const EMPTY_OWNER: OwnerPlan = OwnerPlan {
    config: OwnerConfig {
        kv_heads: 0,
        dim: 0,
        window: None,
    },
    retained: 0,
    old_len: 0,
    staged_len: 0,
};
/// Per-head active staged bytes. Fixed CHUNK head stride never becomes tokens.
fn active_range(p: OwnerPlan, head: usize, tokens: usize) -> Result<(usize, usize)> {
    if head >= p.config.kv_heads || tokens == 0 || tokens > CHUNK {
        return Err(Error::Options("cache active head/count"));
    }
    let offset = head
        .checked_mul(CHUNK)
        .and_then(|n| n.checked_mul(p.config.dim))
        .and_then(|n| n.checked_mul(2))
        .ok_or(Error::Options("cache active offset overflow"))?;
    let bytes = tokens
        .checked_mul(p.config.dim)
        .and_then(|n| n.checked_mul(2))
        .ok_or(Error::Options("cache active bytes overflow"))?;
    if offset
        .checked_add(bytes)
        .is_none_or(|end| end > (p.staged_len - GUARD) * 2)
    {
        return Err(Error::Options("cache active head extent"));
    }
    Ok((offset, bytes))
}
pub(super) struct Plan {
    capacity: usize,
    count: usize,
    owners: [OwnerPlan; MAX],
    buffer_bytes: usize,
}
impl Plan {
    pub(super) fn new(capacity: usize, configs: &[OwnerConfig]) -> Result<Self> {
        if capacity == 0 || capacity > 544 || configs.is_empty() || configs.len() > MAX {
            return Err(Error::Options("cache set shape"));
        }
        let mut plan = Self {
            capacity,
            count: configs.len(),
            owners: [EMPTY_OWNER; MAX],
            buffer_bytes: (1 + GUARD) * 4,
        };
        for (slot, &config) in plan.owners.iter_mut().zip(configs) {
            if ![(1, 32), (1, 256), (2, 64), (8, 64)].contains(&(config.kv_heads, config.dim))
                || config.window.is_some_and(|w| w == 0 || w > capacity)
            {
                return Err(Error::Options("cache owner shape"));
            }
            let layout = KvOwner {
                heads: config.kv_heads,
                head_dim: config.dim,
                storage: KvStorage::F16,
                retention: config
                    .window
                    .map_or(Retention::Full, |window| Retention::Sliding { window }),
                guard_elements: GUARD,
                copies: 1,
            }
            .at_capacity(capacity)
            .map_err(|_| Error::Options("cache owner bytes"))?;
            let staged_len = config
                .kv_heads
                .checked_mul(CHUNK)
                .and_then(|n| n.checked_mul(config.dim))
                .and_then(|n| n.checked_add(GUARD))
                .ok_or(Error::Options("cache stage overflow"))?;
            plan.buffer_bytes = plan
                .buffer_bytes
                .checked_add(layout.total_bytes)
                .and_then(|n| staged_len.checked_mul(4).and_then(|v| n.checked_add(v)))
                .ok_or(Error::Options("cache set bytes overflow"))?;
            *slot = OwnerPlan {
                config,
                retained: config.window.unwrap_or(capacity),
                old_len: layout.elements_per_array,
                staged_len,
            };
        }
        Ok(plan)
    }
    /// Session bytes only. Operation/Plan and shared compiler allowance separate.
    pub(super) fn session_bytes(&self) -> Result<usize> {
        self.buffer_bytes
            .checked_add(size_of::<Set<'_>>())
            .and_then(|n| n.checked_add(TRANSACTION_METADATA_BYTES))
            .ok_or(Error::Options("cache metadata overflow"))
    }
}
pub(super) struct Operation {
    plan: Plan,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    width: usize,
}
impl Operation {
    /// Append complete external candidate validation to the retained Phase-A
    /// command. The concrete hybrid Session stamps and validates both history
    /// banks before allowing Set::finish_math to mint Ready. No CPU value access.
    pub(super) fn encode_half_pair(
        &self,
        command: &Command<'_>,
        left: &Buffer<u16>,
        right: &Buffer<u16>,
        element_offset: usize,
        count: usize,
        status: &Buffer<u32>,
    ) -> Result<()> {
        let end = element_offset
            .checked_add(count)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or(Error::Options("cache external validator extent"))?;
        let offset = element_offset
            .checked_mul(2)
            .ok_or(Error::Options("cache external validator offset"))?;
        if count == 0
            || end > left.len
            || end > right.len
            || status.len < 1 + GUARD
            || !offset.is_multiple_of(4)
            || count
                .div_ceil(self.width)
                .checked_mul(self.width)
                .is_none_or(|n| n > u32::MAX as usize)
        {
            return Err(Error::Options("cache external validator shape"));
        }
        let buffers = [&*left.raw, &*right.raw, &*status.raw];
        for (i, b) in buffers.iter().enumerate() {
            if buffers[..i].iter().any(|other| std::ptr::eq(*b, *other)) {
                return Err(Error::Options("cache external validator alias"));
            }
        }
        let device = self.pipeline.device();
        if !std::ptr::eq(&*command.device(), &*device)
            || buffers
                .iter()
                .any(|b| !std::ptr::eq(&*b.device(), &*device))
        {
            return Err(Error::Options("cache external validator device"));
        }
        if !command.retainedReferences() || command.status() != MTLCommandBufferStatus::NotEnqueued
        {
            return Err(Error::Options("cache external validator command"));
        }
        let e = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::Native("cache external validator encoder absent".into()))?;
        e.setComputePipelineState(&self.pipeline);
        self.dispatch_pair(&e, &left.raw, &right.raw, offset, count as u32, &status.raw);
        e.endEncoding();
        Ok(())
    }
    // Caller prevalidates extents, alignment, device identity and rounded count.
    // Kept separate so finish_math retains ONE encoder across all KV heads.
    fn dispatch_pair(
        &self,
        e: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        left: &ProtocolObject<dyn MTLBuffer>,
        right: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        count: u32,
        status: &ProtocolObject<dyn MTLBuffer>,
    ) {
        // SAFETY: checked same-device initialized operands, aligned byte offsets,
        // copied scalar ABI, retained command and no CPU access until terminal.
        unsafe {
            e.setBuffer_offset_atIndex(Some(left), offset, 0);
            e.setBuffer_offset_atIndex(Some(right), offset, 1);
            e.setBytes_length_atIndex(NonNull::from(&count).cast(), size_of::<u32>(), 2);
            e.setBuffer_offset_atIndex(Some(status), 0, 3);
        }
        e.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (count as usize).div_ceil(self.width),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.width,
                height: 1,
                depth: 1,
            },
        );
    }
    pub(super) fn new(gpu: &Gpu, plan: Plan, available: usize) -> Result<Self> {
        let required = PIPELINE_ALLOWANCE + size_of::<Self>();
        if required > available {
            return Err(Error::Budget {
                required,
                limit: available,
            });
        }
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version2_3);
        let library = gpu
            .device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("staged_cache.metal")),
                Some(&options),
            )
            .map_err(|e| Error::Native(e.to_string()))?;
        let function = library
            .newFunctionWithName(&NSString::from_str("validate_half_kv"))
            .ok_or_else(|| Error::Native("Half validator absent".into()))?;
        let pipeline = gpu
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| Error::Native(e.to_string()))?;
        let width = pipeline.threadExecutionWidth();
        if width == 0
            || width > pipeline.maxTotalThreadsPerThreadgroup()
            || width > u32::MAX as usize
        {
            return Err(Error::Unsupported("Half validator launch width"));
        }
        Ok(Self {
            plan,
            pipeline,
            width,
        })
    }
    fn device(&self, gpu: &Gpu) -> Result<()> {
        if !std::ptr::eq(&*self.pipeline.device(), &*gpu.device) {
            return Err(Error::Options("cache set device"));
        }
        Ok(())
    }
}
struct Owner {
    old_k: Buffer<u16>,
    old_v: Buffer<u16>,
    new_k: Buffer<u16>,
    new_v: Buffer<u16>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Frame {
    set: u64,
    owner: usize,
    generation: u64,
    epoch: u64,
    offset: usize,
    tokens: usize,
}
const EMPTY_FRAME: Frame = Frame {
    set: 0,
    owner: 0,
    generation: 0,
    epoch: 0,
    offset: 0,
    tokens: 0,
};
pub(super) struct Batch {
    frames: [Frame; MAX],
    len: usize,
}
impl Batch {
    pub(super) fn frames(&self) -> &[Frame] {
        &self.frames[..self.len]
    }
}
pub(super) struct Ready {
    batch: Batch,
}
fn check_frames(frames: &[Frame], expected: &Batch) -> Result<()> {
    if frames.len() != expected.len {
        return Err(Error::Options("cache frame set count"));
    }
    for (actual, wanted) in frames.iter().zip(expected.frames()) {
        if actual != wanted {
            return Err(Error::Options("cache frame set order/identity"));
        }
    }
    Ok(())
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Idle,
    Encoding,
    Ready,
    Poisoned,
}
pub(super) struct CommitInfo {
    pub start: usize,
    pub end: usize,
    pub owners: usize,
    pub generation: u64,
}
/// Internal encoding-only capability. End each encoder before the next; do not
/// clone/retain raw Objective-C command or buffer handles for out-of-protocol
/// submission or later writes. Rust borrowing alone cannot prohibit those clones.
pub(super) struct View<'a> {
    pub command: Command<'a>,
    pub old_k: &'a Buffer<u16>,
    pub old_v: &'a Buffer<u16>,
    pub new_k: &'a Buffer<u16>,
    pub new_v: &'a Buffer<u16>,
    pub status: &'a Buffer<u32>,
    pub config: OwnerConfig,
    pub capacity: usize,
    pub offset: usize,
    pub tokens: usize,
}
pub(super) struct Set<'a> {
    operation: &'a Operation,
    owners: [Option<Owner>; MAX],
    status: Buffer<u32>,
    identity: u64,
    generation: u64,
    epoch: u64,
    offset: usize,
    tokens: usize,
    state: State,
    command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}
fn half_buffer(gpu: &Gpu, len: usize) -> Result<Buffer<u16>> {
    let mut b = Buffer::new(gpu, len)?;
    b.with_mut(|v| v.fill(PAD));
    Ok(b)
}
fn terminal(command: &ProtocolObject<dyn MTLCommandBuffer>) -> bool {
    matches!(
        command.status(),
        MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
    )
}
fn completed(command: &ProtocolObject<dyn MTLCommandBuffer>) -> bool {
    command.status() == MTLCommandBufferStatus::Completed && command.error().is_none()
}
impl<'a> Set<'a> {
    pub(super) fn new(gpu: &Gpu, operation: &'a Operation, available: usize) -> Result<Self> {
        operation.device(gpu)?;
        let required = operation.plan.session_bytes()?;
        if required > available {
            return Err(Error::Budget {
                required,
                limit: available,
            });
        }
        for p in &operation.plan.owners[..operation.plan.count] {
            validate_buffer_bytes(p.old_len * 2, gpu.device.maxBufferLength())?;
            validate_buffer_bytes(p.staged_len * 2, gpu.device.maxBufferLength())?;
        }
        validate_buffer_bytes((1 + GUARD) * 4, gpu.device.maxBufferLength())?;
        let identity = NEXT_SET
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| Error::Options("cache set identity exhaustion"))?;
        let mut owners: [Option<Owner>; MAX] = std::array::from_fn(|_| None);
        for (slot, p) in owners
            .iter_mut()
            .zip(&operation.plan.owners[..operation.plan.count])
        {
            *slot = Some(Owner {
                old_k: half_buffer(gpu, p.old_len)?,
                old_v: half_buffer(gpu, p.old_len)?,
                new_k: half_buffer(gpu, p.staged_len)?,
                new_v: half_buffer(gpu, p.staged_len)?,
            });
        }
        let mut status = Buffer::new(gpu, 1 + GUARD)?;
        status.with_mut(|v| {
            v.fill(STATUS_PAD);
            v[0] = 0;
        });
        Ok(Self {
            operation,
            owners,
            status,
            identity,
            generation: 0,
            epoch: 0,
            offset: 0,
            tokens: 0,
            state: State::Idle,
            command: None,
        })
    }
    pub(super) fn offset(&self) -> usize {
        self.offset
    }
    pub(super) fn is_poisoned(&self) -> bool {
        self.state == State::Poisoned
    }
    fn batch(&self) -> Batch {
        let mut frames = [EMPTY_FRAME; MAX];
        for (owner, f) in frames
            .iter_mut()
            .enumerate()
            .take(self.operation.plan.count)
        {
            *f = Frame {
                set: self.identity,
                owner,
                generation: self.generation,
                epoch: self.epoch,
                offset: self.offset,
                tokens: self.tokens,
            };
        }
        Batch {
            frames,
            len: self.operation.plan.count,
        }
    }
    fn frame(&self, f: &Frame) -> Result<()> {
        if f.set != self.identity
            || f.owner >= self.operation.plan.count
            || f.generation != self.generation
            || f.epoch != self.epoch
            || f.offset != self.offset
            || f.tokens != self.tokens
        {
            return Err(Error::Options("cache foreign/stale frame"));
        }
        Ok(())
    }
    fn frames(&self, frames: &[Frame], state: State) -> Result<()> {
        if self.state == State::Poisoned {
            return Err(Error::ReplayRequired);
        }
        if self.state != state || frames.len() != self.operation.plan.count {
            return Err(Error::Options("cache frame set state/count"));
        }
        check_frames(frames, &self.batch())?;
        self.offset
            .checked_add(self.tokens)
            .filter(|&end| end <= self.operation.plan.capacity)
            .ok_or(Error::Options("cache commit position"))?;
        Ok(())
    }
    pub(super) fn begin(&mut self, gpu: &Gpu, tokens: usize) -> Result<Batch> {
        self.operation.device(gpu)?;
        if self.state == State::Poisoned {
            return Err(Error::ReplayRequired);
        }
        if self.state != State::Idle || tokens == 0 || tokens > CHUNK {
            return Err(Error::Options("cache begin state/count"));
        }
        self.offset
            .checked_add(tokens)
            .filter(|&end| end <= self.operation.plan.capacity)
            .ok_or(Error::Options("cache begin position"))?;
        let epoch = self
            .epoch
            .checked_add(1)
            .ok_or(Error::Options("cache epoch exhaustion"))?;
        for p in &self.operation.plan.owners[..self.operation.plan.count] {
            for head in 0..p.config.kv_heads {
                active_range(*p, head, tokens)?;
            }
        }
        let raw = gpu.command()?;
        let e = raw
            .blitCommandEncoder()
            .ok_or_else(|| Error::Native("cache staging fill encoder absent".into()))?;
        for (i, p) in self.operation.plan.owners[..self.operation.plan.count]
            .iter()
            .enumerate()
        {
            let owner = self.owners[i].as_ref().unwrap();
            // Every head's tight active slice is disjoint, checked before the
            // encoder; inactive CHUNK-token gaps and guards stay untouched.
            for head in 0..p.config.kv_heads {
                let (offset, bytes) = active_range(*p, head, tokens)?;
                for b in [&owner.new_k, &owner.new_v] {
                    e.fillBuffer_range_value(
                        &b.raw,
                        NSRange {
                            location: offset,
                            length: bytes,
                        },
                        0x7e,
                    );
                }
            }
        }
        e.endEncoding();
        self.status.with_mut(|v| v[0] = 0);
        self.command = Some(raw);
        self.tokens = tokens;
        self.epoch = epoch;
        self.state = State::Encoding;
        Ok(self.batch())
    }
    pub(super) fn view(&self, frame: &Frame) -> Result<View<'_>> {
        if self.state == State::Poisoned {
            return Err(Error::ReplayRequired);
        }
        if self.state != State::Encoding {
            return Err(Error::Options("cache view not encoding"));
        }
        self.frame(frame)?;
        let raw = self.command.as_ref().unwrap();
        if raw.status() != MTLCommandBufferStatus::NotEnqueued {
            return Err(Error::Options("cache command already submitted"));
        }
        let owner = self.owners[frame.owner].as_ref().unwrap();
        let p = self.operation.plan.owners[frame.owner];
        Ok(View {
            command: Command { raw, trace: None },
            old_k: &owner.old_k,
            old_v: &owner.old_v,
            new_k: &owner.new_k,
            new_v: &owner.new_v,
            status: &self.status,
            config: p.config,
            capacity: self.operation.plan.capacity,
            offset: self.offset,
            tokens: self.tokens,
        })
    }
    pub(super) fn finish_math(&mut self, frames: &[Frame]) -> Result<Ready> {
        // Classify the actual command before even malformed frame rejection:
        // premature enqueue/submission cannot be hidden behind metadata errors.
        if self.state == State::Encoding
            && self
                .command
                .as_ref()
                .is_some_and(|c| c.status() != MTLCommandBufferStatus::NotEnqueued)
        {
            self.state = State::Poisoned;
            return Err(Error::ReplayRequired);
        }
        self.frames(frames, State::Encoding)?;
        let raw = self.command.as_ref().unwrap();
        if raw.status() != MTLCommandBufferStatus::NotEnqueued {
            self.state = State::Poisoned;
            return Err(Error::ReplayRequired);
        }
        // Bounds before encoder creation: all source owners must be validated,
        // even when their attention was never encoded.
        for p in &self.operation.plan.owners[..self.operation.plan.count] {
            for head in 0..p.config.kv_heads {
                active_range(*p, head, self.tokens)?;
            }
            let count = self.tokens * p.config.dim;
            count
                .div_ceil(self.operation.width)
                .checked_mul(self.operation.width)
                .filter(|&v| v <= u32::MAX as usize)
                .ok_or(Error::Options("cache validator rounded launch"))?;
        }
        let e = raw
            .computeCommandEncoder()
            .ok_or_else(|| Error::Native("cache validator encoder absent".into()))?;
        e.setComputePipelineState(&self.operation.pipeline);
        for (i, p) in self.operation.plan.owners[..self.operation.plan.count]
            .iter()
            .enumerate()
        {
            let owner = self.owners[i].as_ref().unwrap();
            let count = (self.tokens * p.config.dim) as u32;
            for head in 0..p.config.kv_heads {
                let (offset, _) = active_range(*p, head, self.tokens)?;
                self.operation.dispatch_pair(
                    &e,
                    &owner.new_k.raw,
                    &owner.new_v.raw,
                    offset,
                    count,
                    &self.status.raw,
                );
            }
        }
        e.endEncoding();
        raw.commit();
        raw.waitUntilCompleted();
        if !completed(raw) {
            self.state = State::Poisoned;
            return Err(Error::Completion);
        }
        if self.status.with_slice(|v| v[0]) != 0 {
            self.state = State::Idle;
            self.tokens = 0;
            return Err(Error::Nonfinite);
        }
        self.state = State::Ready;
        Ok(Ready {
            batch: self.batch(),
        })
    }
    pub(super) fn commit(&mut self, gpu: &Gpu, ready: &Ready) -> Result<CommitInfo> {
        self.copy(gpu, ready, CopyMode::All)
    }
    fn copy(&mut self, gpu: &Gpu, ready: &Ready, mode: CopyMode) -> Result<CommitInfo> {
        self.operation.device(gpu)?;
        self.frames(ready.batch.frames(), State::Ready)?;
        let next_generation = self
            .generation
            .checked_add(1)
            .ok_or(Error::Options("cache generation exhaustion"))?;
        let end = self.offset + self.tokens;
        let empty = window_rows::CopyRun {
            source_row: 0,
            slot: 0,
            rows: 0,
        };
        let mut runs = [[empty; 2]; MAX];
        for (i, p) in self.operation.plan.owners[..self.operation.plan.count]
            .iter()
            .enumerate()
        {
            runs[i] = window_rows::copy_runs(
                p.retained,
                self.offset,
                self.tokens,
                p.config.window.is_some(),
            );
            for run in runs[i] {
                let src = (run.source_row + run.rows)
                    .checked_mul(p.config.dim)
                    .and_then(|n| n.checked_mul(2))
                    .ok_or(Error::Options("cache source copy overflow"))?;
                let dst = (run.slot + run.rows)
                    .checked_mul(p.config.dim)
                    .and_then(|n| n.checked_mul(2))
                    .ok_or(Error::Options("cache destination copy overflow"))?;
                if src > self.tokens * p.config.dim * 2 || dst > p.retained * p.config.dim * 2 {
                    return Err(Error::Options("cache copy range"));
                }
                for head in 0..p.config.kv_heads {
                    let source_end = head
                        .checked_mul(CHUNK)
                        .and_then(|n| n.checked_mul(p.config.dim))
                        .and_then(|n| n.checked_mul(2))
                        .and_then(|n| n.checked_add(src))
                        .ok_or(Error::Options("cache head source overflow"))?;
                    let target_end = head
                        .checked_mul(p.retained)
                        .and_then(|n| n.checked_mul(p.config.dim))
                        .and_then(|n| n.checked_mul(2))
                        .and_then(|n| n.checked_add(dst))
                        .ok_or(Error::Options("cache head destination overflow"))?;
                    if source_end > (p.staged_len - GUARD) * 2
                        || target_end > (p.old_len - GUARD) * 2
                    {
                        return Err(Error::Options("cache head copy range"));
                    }
                }
            }
        }
        let raw = gpu.command()?;
        let e = raw
            .blitCommandEncoder()
            .ok_or_else(|| Error::Native("cache commit encoder absent".into()))?;
        let mut copied = 0;
        'owners: for (i, p) in self.operation.plan.owners[..self.operation.plan.count]
            .iter()
            .enumerate()
        {
            let owner = self.owners[i].as_ref().unwrap();
            for head in 0..p.config.kv_heads {
                for (src, dst) in [(&owner.new_k, &owner.old_k), (&owner.new_v, &owner.old_v)] {
                    for run in runs[i] {
                        if run.rows == 0 {
                            continue;
                        }
                        // SAFETY: full set and every range prevalidated before this
                        // command; tight rows have four-byte-aligned offsets/sizes.
                        // Owners are fresh non-Clone allocations and commands retain
                        // them. No CPU access until completion. Old views are sealed.
                        unsafe {
                            e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                                &src.raw,
                                (head * CHUNK + run.source_row) * p.config.dim * 2,
                                &dst.raw,
                                (head * p.retained + run.slot) * p.config.dim * 2,
                                run.rows * p.config.dim * 2,
                            );
                        }
                        copied += 1;
                        if mode.stop_after(copied) {
                            break 'owners;
                        }
                    }
                }
            }
        }
        e.endEncoding();
        self.command = Some(raw);
        self.state = State::Poisoned;
        let raw = self.command.as_ref().unwrap();
        raw.commit();
        raw.waitUntilCompleted();
        if !completed(raw) {
            return Err(Error::Completion);
        }
        if mode.injected() {
            return Err(Error::ReplayRequired);
        }
        let info = CommitInfo {
            start: self.offset,
            end,
            owners: self.operation.plan.count,
            generation: next_generation,
        };
        self.offset = end;
        self.tokens = 0;
        self.generation = next_generation;
        self.state = State::Idle;
        Ok(info)
    }
    /// Discard is valid only for a command never submitted or confirmed terminal.
    /// A poisoned copy requires zero-prefix reset/replay, never this shortcut.
    pub(super) fn discard(&mut self) -> Result<()> {
        if self.state == State::Poisoned {
            return Err(Error::ReplayRequired);
        }
        if self
            .command
            .as_ref()
            .is_some_and(|c| c.status() != MTLCommandBufferStatus::NotEnqueued && !terminal(c))
        {
            return Err(Error::ReplayRequired);
        }
        self.command = None;
        self.state = State::Idle;
        self.tokens = 0;
        Ok(())
    }
    pub(super) fn reset(&mut self) -> Result<()> {
        if self
            .command
            .as_ref()
            .is_some_and(|c| c.status() != MTLCommandBufferStatus::NotEnqueued && !terminal(c))
        {
            return Err(Error::ReplayRequired);
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(Error::Options("cache reset generation exhaustion"))?;
        let epoch = self
            .epoch
            .checked_add(1)
            .ok_or(Error::Options("cache reset epoch exhaustion"))?;
        self.command = None;
        self.state = State::Idle;
        self.offset = 0;
        self.tokens = 0;
        self.generation = generation;
        self.epoch = epoch;
        self.status.with_mut(|v| v[0] = 0);
        Ok(())
    }
    #[cfg(test)]
    pub(super) fn interrupt_copy_after_first_run(
        &mut self,
        gpu: &Gpu,
        ready: &Ready,
    ) -> Result<CommitInfo> {
        self.copy(gpu, ready, CopyMode::FirstRunThenFail)
    }
}
enum CopyMode {
    All,
    #[cfg(test)]
    FirstRunThenFail,
}
impl CopyMode {
    fn stop_after(&self, _count: usize) -> bool {
        match self {
            Self::All => false,
            #[cfg(test)]
            Self::FirstRunThenFail => _count == 1,
        }
    }
    fn injected(&self) -> bool {
        match self {
            Self::All => false,
            #[cfg(test)]
            Self::FirstRunThenFail => true,
        }
    }
}
#[cfg(test)]
mod tests;

// Test-only owned observation, never a mutable-buffer escape. Refuse mapped
// access until the Set's actual command is safe; poisoned completed copies may
// be observed for rollback evidence, but remain unavailable through the graph.
#[cfg(test)]
impl Set<'_> {
    pub(super) fn test_safe_read(&self) -> Result<()> {
        if self
            .command
            .as_ref()
            .is_some_and(|c| c.status() != MTLCommandBufferStatus::NotEnqueued && !terminal(c))
        {
            return Err(Error::ReplayRequired);
        }
        Ok(())
    }
    pub(super) fn test_completion(&self) -> Option<(bool, bool)> {
        self.command.as_ref().map(|c| (terminal(c), completed(c)))
    }
    pub(super) fn test_observe(&self) -> Result<serde_json::Value> {
        self.test_safe_read()?;
        let owners: Vec<_> = self.owners[..self.operation.plan.count]
            .iter()
            .map(|o| {
                let o = o.as_ref().unwrap();
                let buffers: Vec<_> = [&o.old_k, &o.old_v, &o.new_k, &o.new_v]
                    .iter()
                    .map(|b| {
                        serde_json::json!({
                            "address": b.raw.contents().as_ptr() as usize, "bytes": b.raw.length(),
                            "words": b.with_slice(|v| v.to_vec())
                        })
                    })
                    .collect();
                serde_json::json!(buffers)
            })
            .collect();
        Ok(
            serde_json::json!({"offset":self.offset,"generation":self.generation,"epoch":self.epoch,"identity":self.identity,"poisoned":self.is_poisoned(),
            "owners":owners,"status":self.status.with_slice(|v|v.to_vec()),
            "command":self.command.as_ref().map(|c|serde_json::json!({"status":format!("{:?}",c.status()),"error":c.error().map(|e|e.to_string()),"terminal":terminal(c),"completed":completed(c)}))}),
        )
    }
}

#[cfg(test)]
mod gqa_tests;

// Bounded real-capture observation. Existing owned tiny snapshots are unchanged.
// All callbacks finish synchronously before the next command may be submitted.
#[cfg(test)]
impl Set<'_> {
    pub(super) fn test_visit_buffers(
        &self,
        mut visit: impl FnMut(usize, usize, usize, u64, &[u16]) -> Result<()>,
    ) -> Result<()> {
        self.test_safe_read()?;
        for (owner, value) in self.owners[..self.operation.plan.count].iter().enumerate() {
            let owner_value = value.as_ref().unwrap();
            for (role, b) in [
                &owner_value.old_k,
                &owner_value.old_v,
                &owner_value.new_k,
                &owner_value.new_v,
            ]
            .into_iter()
            .enumerate()
            {
                b.with_slice(|words| {
                    visit(
                        owner,
                        role,
                        b.raw.contents().as_ptr() as usize,
                        b.raw.device().registryID(),
                        words,
                    )
                })?;
            }
        }
        Ok(())
    }
    pub(super) fn test_capture_metadata(&self) -> Result<serde_json::Value> {
        self.test_safe_read()?;
        Ok(serde_json::json!({
            "offset":self.offset,"generation":self.generation,"epoch":self.epoch,
            "identity":self.identity,"poisoned":self.is_poisoned(),
            "status":{"address":self.status.raw.contents().as_ptr() as usize,
                "bytes":self.status.raw.length(),"device":self.status.raw.device().registryID(),
                "words":self.status.with_slice(|v|v.to_vec())},
            "command":self.command.as_ref().map(|c|serde_json::json!({
                "status":format!("{:?}",c.status()),"error":c.error().map(|e|e.to_string()),
                "terminal":terminal(c),"completed":completed(c)}))
        }))
    }
}
