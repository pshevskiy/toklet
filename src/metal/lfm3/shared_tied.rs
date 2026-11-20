use super::{pair::domain_tag, *};
use crate::{
    metal::{Error, lfm3_shared_weights::BankRef},
    model::lfm3::VOCAB,
};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
enum Rejected {
    Shape,
    Cancelled,
    Status(u32),
    Native(Error),
}
impl std::fmt::Display for Rejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape => f.write_str("invalid tied-head extent or device"),
            Self::Cancelled => f.write_str("tied head cancelled before publication"),
            Self::Status(code) => write!(f, "tied-head GPU status {code}"),
            Self::Native(error) => error.fmt(f),
        }
    }
}

impl From<Error> for Rejected {
    fn from(error: Error) -> Self {
        Self::Native(error)
    }
}

struct Completed {
    // Integer census, complete ordered accounting, native hardware.
    routes: [u64; 3],
    gpu_seconds: f64,
    wait_seconds: f64,
}
fn run(
    kernels: &Kernels,
    gpu: &Gpu,
    weight: BankRef<'_>,
    input: &[bf16],
    output: &mut [bf16],
    scratch: &mut Scratch,
    cancelled: &AtomicBool,
) -> std::result::Result<(Completed, u64), Rejected> {
    let rows = output.len();
    let Some(words) = rows.checked_mul(I) else {
        return Err(Rejected::Shape);
    };
    if input.len() != I
        || rows != VOCAB
        || words > u32::MAX as usize
        || weight.words() != words
        || !weight.device_matches(&gpu.device)
    {
        return Err(Rejected::Shape);
    }
    if cancelled.load(Ordering::Acquire) {
        return Err(Rejected::Cancelled);
    }
    let Scratch {
        x,
        seen,
        routes,
        status,
        input_tag,
        weight_tags,
        submitted,
    } = scratch;
    if x.len != I
        || seen.len != rows + 1
        || routes.len != rows + 1
        || weight_tags.len != rows
        || input_tag.len != 1
        || status.len != 2
    {
        return Err(Rejected::Shape);
    }
    x.with_mut(|dst| {
        for (slot, value) in dst.iter_mut().zip(input) {
            *slot = value.to_bits();
        }
    });
    input_tag.with_mut(|dst| dst[0] = domain_tag(input.iter().map(|v| v.to_bits())));
    seen.with_mut(|d| d.fill(SENTINEL));
    routes.with_mut(|d| d.fill(ROUTE_GUARD));
    status.with_mut(|d| d.copy_from_slice(&[0, STATUS_GUARD]));
    let (dims, admission_groups) = Kernels::sealed_admission(weight, 1, I, rows);
    let command = gpu.command()?;
    if !command.retainedReferences()
        || !std::ptr::eq(&*command.device(), &*gpu.device)
        || command.status() != MTLCommandBufferStatus::NotEnqueued
    {
        return Err(Error::Completion.into());
    }
    let threads = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    for (pso, count, project) in [
        (&kernels.admission, admission_groups, false),
        (&kernels.projection, rows, true),
    ] {
        let encoder = command.computeCommandEncoder().ok_or(Error::Completion)?;
        encoder.setComputePipelineState(pso);
        // SAFETY: exact same-device extents, disjoint owned scratch, a private
        // sealed bank bound to a const device pointer in both frozen kernels,
        // native-retained references and one completion before inspection.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&x.raw), 0, 0);
            weight.bind_read_only(&encoder, 1);
            encoder.setBuffer_offset_atIndex(Some(&status.raw), 0, 2);
            encoder.setBytes_length_atIndex(NonNull::from(&dims).cast(), size_of::<[u32; 4]>(), 3);
            if project {
                encoder.setBuffer_offset_atIndex(Some(&seen.raw), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&routes.raw), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(&input_tag.raw), 0, 6);
                encoder.setBuffer_offset_atIndex(Some(&weight_tags.raw), 0, 7);
            }
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: count,
                height: 1,
                depth: 1,
            },
            threads,
        );
        encoder.endEncoding();
    }
    if cancelled.load(Ordering::Acquire) {
        return Err(Rejected::Cancelled);
    }
    let start = Instant::now();
    let next_submission = submitted.checked_add(1).ok_or(Error::Completion)?;
    command.commit();
    *submitted = next_submission;
    command.waitUntilCompleted();
    let wait_seconds = start.elapsed().as_secs_f64();
    if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
        return Err(Error::Completion.into());
    }
    let gpu_seconds = command.GPUEndTime() - command.GPUStartTime();
    if !gpu_seconds.is_finite()
        || gpu_seconds <= 0.
        || !wait_seconds.is_finite()
        || wait_seconds <= 0.
    {
        return Err(Error::Completion.into());
    }
    let code = status
        .with_slice(|d| (d[1] == STATUS_GUARD).then_some(d[0]))
        .ok_or(Error::Completion)?;
    let armed = seen.with_slice(|d| d[rows] == SENTINEL);
    let census_guard = routes.with_slice(|d| d[rows] == ROUTE_GUARD);
    if !armed || !census_guard {
        return Err(Error::Completion.into());
    }
    if code == 2
        && !(seen.with_slice(|d| d.iter().all(|&x| x == SENTINEL))
            && routes.with_slice(|d| d.iter().all(|&x| x == ROUTE_GUARD)))
    {
        return Err(Error::Completion.into());
    }
    if code != 0 {
        return Err(Rejected::Status(code));
    }
    let mut native_rows = 0_u64;
    let totals = routes
        .with_slice(|d| {
            let mut sums = [0_u64; 3];
            for &word in &d[..rows] {
                if word == 0x80000000 {
                    native_rows += 1;
                    sums[2] += (I + 32) as u64;
                    continue;
                }
                let exact = word & 4095;
                let shortcut = word >> 12;
                let hardware = (I as u32 + 32).checked_sub(exact + shortcut)?;
                for (v, x) in sums.iter_mut().zip([exact, shortcut, hardware]) {
                    *v += u64::from(x);
                }
            }
            Some(sums)
        })
        .ok_or(Error::Completion)?;
    if cancelled.load(Ordering::Acquire) {
        return Err(Rejected::Cancelled);
    }
    seen.with_slice(|d| {
        if d[..rows]
            .iter()
            .any(|&word| !bf16::from_bits(word).is_finite())
        {
            return Err(Error::Completion);
        }
        for (dst, &word) in output.iter_mut().zip(d) {
            *dst = bf16::from_bits(word);
        }
        Ok(())
    })?;
    Ok((
        Completed {
            routes: totals,
            gpu_seconds,
            wait_seconds,
        },
        native_rows,
    ))
}

struct Scratch {
    x: Buffer<u16>,
    seen: Buffer<u16>,
    routes: Buffer<u32>,
    status: Buffer<u32>,
    input_tag: Buffer<u32>,
    weight_tags: Buffer<u32>,
    submitted: u64,
}

impl Scratch {
    fn new(gpu: &Gpu, words: &[bf16], rows: usize) -> std::result::Result<Self, Error> {
        let mut weight_tags = Buffer::new(gpu, rows)?;

        weight_tags.with_mut(|tags| {
            for (tag, row) in tags.iter_mut().zip(words.chunks_exact(I)) {
                *tag = domain_tag(row.iter().map(|v| v.to_bits()));
            }
        });
        Ok(Self {
            x: Buffer::new(gpu, I)?,
            seen: Buffer::new(gpu, rows + 1)?,
            routes: Buffer::new(gpu, rows + 1)?,
            status: Buffer::new(gpu, 2)?,
            input_tag: Buffer::new(gpu, 1)?,
            weight_tags,
            submitted: 0,
        })
    }
}

pub(super) struct Workspace {
    kernels: Kernels,
    scratch: Scratch,
    identity: usize,
    completed: u64,
    clocks: [f64; 2],
}

impl Workspace {
    pub(super) fn new(
        gpu: &Gpu,
        bank: BankRef<'_>,
        words: &[bf16],
    ) -> std::result::Result<Self, crate::model::Error> {
        use crate::model::Error as ModelError;
        if words.len() != VOCAB * I || !bank.aliases(words) || !bank.device_matches(&gpu.device) {
            return Err(ModelError::InvalidConfig("LFM3 tied head bank identity"));
        }
        let kernels = Kernels::with_shader(gpu, include_str!("normal_domain.metal"))
            .ok_or(ModelError::InvalidConfig("LFM3 tied head pipelines"))?;
        let scratch = Scratch::new(gpu, words, VOCAB).map_err(|_| ModelError::Allocation)?;
        Ok(Self {
            kernels,
            scratch,
            identity: bank.identity(),
            completed: 0,
            clocks: [0.; 2],
        })
    }

    pub(super) fn project(
        &mut self,
        gpu: &Gpu,
        bank: BankRef<'_>,
        words: &[bf16],
        input: &[bf16],
        output: &mut [bf16],
        cancelled: &AtomicBool,
    ) -> std::result::Result<(), crate::model::Error> {
        use crate::model::Error as ModelError;
        if bank.identity() != self.identity || !bank.aliases(words) {
            return Err(ModelError::InvalidConfig("LFM3 tied head stale owner"));
        }
        let (done, _) = run(
            &self.kernels,
            gpu,
            bank,
            input,
            output,
            &mut self.scratch,
            cancelled,
        )
        .map_err(|error| {
            ModelError::InvalidConfig(match error {
                Rejected::Shape => "LFM3 tied head shape",
                Rejected::Cancelled => return ModelError::Cancelled,
                Rejected::Status(2) => "LFM3 tied head nonfinite input or weight",
                Rejected::Status(_) => "LFM3 tied head device status",
                Rejected::Native(_) => "LFM3 tied head completion or guard",
            })
        })?;
        let completed = self
            .completed
            .checked_add(1)
            .ok_or(crate::cpu::Error::SizeOverflow)?;
        self.completed = completed;
        self.clocks[0] += done.gpu_seconds;
        self.clocks[1] += done.wait_seconds;
        let _ = done.routes;
        Ok(())
    }

    pub(super) fn report(&self) -> serde_json::Value {
        let report = serde_json::json!({"submitted":self.scratch.submitted,"verified_completions":self.completed,
            "gpu_seconds":self.clocks[0],"commit_wait_seconds":self.clocks[1]});
        report
    }
}
