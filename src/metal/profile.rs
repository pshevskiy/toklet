//! Opt-in stage counters. Custom kernels get direct encoder timestamps; MPS
//! operations are enclosed by dependent one-lane markers because MPS owns its
//! internal encoders. Such intervals include internal scheduling, not pure ALU time.
use super::{Buffer, Error, Gpu, Result, Session};
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::*;
use std::{cell::Cell, ops::Deref, ptr::NonNull, time::Instant};

const MAX_EVENTS: usize = 1024;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct ProfileEvent {
    pub name: &'static str,
    pub host_begin_seconds: f64,
    pub host_end_seconds: f64,
    pub gpu_begin_ticks: u64,
    pub gpu_end_ticks: u64,
    /// Relative to the first sampled CPU timestamp, using Apple's paired-clock
    /// conversion. GPU clock ticks are never assumed to represent nanoseconds.
    pub gpu_begin_seconds: f64,
    pub gpu_end_seconds: f64,
}
#[derive(Clone, Copy, Default, Debug, serde::Serialize)]
pub struct ClockPair {
    pub cpu_nanoseconds: u64,
    pub gpu_ticks: u64,
}
#[derive(Debug, serde::Serialize)]
pub struct Profile {
    pub start: ClockPair,
    pub end: ClockPair,
    pub host_clock_sample_bracket_seconds: f64,
    pub events: Vec<ProfileEvent>,
}
#[derive(Clone, Copy)]
struct HostEvent {
    name: &'static str,
    begin: f64,
    end: f64,
}
struct Slot {
    pass: Retained<MTLComputePassDescriptor>,
    host: Cell<HostEvent>,
}
pub(super) struct Trace {
    buffer: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    pub sink: Buffer<u32>,
    slots: Vec<Slot>,
    cursor: Cell<usize>,
    clock: Cell<Instant>,
    start: Cell<ClockPair>,
    end: Cell<ClockPair>,
    sample_bracket: Cell<f64>,
    valid: Cell<bool>,
}
pub(super) struct Command<'a> {
    pub raw: &'a ProtocolObject<dyn MTLCommandBuffer>,
    pub trace: Option<&'a Trace>,
}
impl Deref for Command<'_> {
    type Target = ProtocolObject<dyn MTLCommandBuffer>;
    fn deref(&self) -> &Self::Target {
        self.raw
    }
}
fn sample(g: &Gpu) -> ClockPair {
    let mut cpu = 0;
    let mut gpu = 0;
    // SAFETY: live device and distinct, valid writable u64 stack pointers.
    unsafe {
        g.device
            .sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu), NonNull::from(&mut gpu));
    }
    ClockPair {
        cpu_nanoseconds: cpu,
        gpu_ticks: gpu,
    }
}
fn clock_spans(start: ClockPair, end: ClockPair) -> Result<(u64, u64)> {
    let cpu = end
        .cpu_nanoseconds
        .checked_sub(start.cpu_nanoseconds)
        .filter(|n| *n > 0)
        .ok_or_else(|| Error::Native("CPU calibration span invalid".into()))?;
    let gpu = end
        .gpu_ticks
        .checked_sub(start.gpu_ticks)
        .filter(|n| *n > 0)
        .ok_or_else(|| Error::Native("GPU calibration span invalid".into()))?;
    Ok((cpu, gpu))
}
fn convert_tick(ticks: u64, start: ClockPair, end: ClockPair) -> Result<f64> {
    let (cpu, gpu) = clock_spans(start, end)?;
    if ticks == 0
        || ticks == MTLCounterErrorValue
        || ticks < start.gpu_ticks
        || ticks > end.gpu_ticks
    {
        return Err(Error::Native("invalid timestamp sample".into()));
    }
    Ok((ticks - start.gpu_ticks) as f64 / gpu as f64 * cpu as f64 / 1e9)
}
impl Trace {
    fn new(g: &Gpu, capacity: usize) -> Result<Self> {
        if !g
            .device
            .supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary)
        {
            return Err(Error::Unsupported(
                "stage-boundary timestamp counters unavailable",
            ));
        }
        let sets = g
            .device
            .counterSets()
            .ok_or(Error::Unsupported("counter sets unavailable"))?;
        // SAFETY: public immutable timestamp counter-set name; Metal initialized.
        let name = unsafe { MTLCommonCounterSetTimestamp };
        let set = sets
            .iter()
            .find(|s| &*s.name() == name)
            .ok_or(Error::Unsupported("timestamp counter set unavailable"))?;
        let samples = capacity
            .checked_mul(2)
            .ok_or(Error::Options("counter sample overflow"))?;
        let descriptor = MTLCounterSampleBufferDescriptor::new();
        descriptor.setCounterSet(Some(&set));
        descriptor.setStorageMode(MTLStorageMode::Shared);
        if samples == 0 || samples > 2 * MAX_EVENTS {
            return Err(Error::Options("counter capacity outside bounds"));
        }
        // SAFETY: bounded nonzero sample count, pre-admitted diagnostic allowance.
        unsafe {
            descriptor.setSampleCount(samples);
        }
        descriptor.setLabel(&NSString::from_str("toklet stage timestamps"));
        let buffer = g
            .device
            .newCounterSampleBufferWithDescriptor_error(&descriptor)
            .map_err(|e| Error::Native(e.to_string()))?;
        if buffer.sampleCount() != samples {
            return Err(Error::Unsupported("counter capacity differs"));
        }
        let mut slots = Vec::with_capacity(capacity);
        for i in 0..capacity {
            let pass = MTLComputePassDescriptor::new();
            // SAFETY: Metal exposes attachment 0; indices 2*i/2*i+1 are below
            // checked sampleCount. Same-device buffer is retained by each pass.
            unsafe {
                let attachment = pass.sampleBufferAttachments().objectAtIndexedSubscript(0);
                attachment.setSampleBuffer(Some(&buffer));
                attachment.setStartOfEncoderSampleIndex(2 * i);
                attachment.setEndOfEncoderSampleIndex(2 * i + 1);
            }
            slots.push(Slot {
                pass,
                host: Cell::new(HostEvent {
                    name: "unused",
                    begin: 0.,
                    end: 0.,
                }),
            });
        }
        Ok(Self {
            buffer,
            sink: Buffer::new(g, 8)?,
            slots,
            cursor: Cell::new(0),
            clock: Cell::new(Instant::now()),
            start: Cell::new(ClockPair::default()),
            end: Cell::new(ClockPair::default()),
            sample_bracket: Cell::new(0.),
            valid: Cell::new(false),
        })
    }
    pub fn reset(&self, g: &Gpu) -> f64 {
        self.valid.set(false);
        self.cursor.set(0);
        self.clock.set(Instant::now());
        self.start.set(sample(g));
        self.sample_bracket
            .set(self.clock.get().elapsed().as_secs_f64());
        self.sample_bracket.get()
    }
    pub fn complete(&self, g: &Gpu) -> f64 {
        let clock = Instant::now();
        self.end.set(sample(g));
        self.valid.set(true);
        clock.elapsed().as_secs_f64()
    }
    pub fn encoder(
        &self,
        c: &ProtocolObject<dyn MTLCommandBuffer>,
        name: &'static str,
    ) -> Result<(
        usize,
        Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    )> {
        let index = self.cursor.get();
        let slot = self
            .slots
            .get(index)
            .ok_or(Error::Options("profile event capacity exceeded"))?;
        self.cursor.set(index + 1);
        slot.host.set(HostEvent {
            name,
            begin: self.clock.get().elapsed().as_secs_f64(),
            end: 0.,
        });
        let e = c
            .computeCommandEncoderWithDescriptor(&slot.pass)
            .ok_or_else(|| Error::Native("profile encoder absent".into()))?;
        Ok((index, e))
    }
    pub fn finish_encoder(&self, index: usize) {
        let mut h = self.slots[index].host.get();
        h.end = self.clock.get().elapsed().as_secs_f64();
        self.slots[index].host.set(h);
    }
    fn resolve(&self) -> Result<Profile> {
        if !self.valid.get() {
            return Err(Error::Options("no successful profiled chunk"));
        }
        let count = self.cursor.get();
        let start = self.start.get();
        let end = self.end.get();
        clock_spans(start, end)?;
        // SAFETY: only completed CPU/GPU work, shared buffer; bounded used range
        // 0..2*count<=sampleCount. NSData is owned until its bytes are copied.
        let data = unsafe {
            self.buffer.resolveCounterRange(NSRange {
                location: 0,
                length: 2 * count,
            })
        }
        .ok_or_else(|| Error::Native("counter resolve absent".into()))?;
        if data.length() != count * 16 {
            return Err(Error::Native("counter resolved size differs".into()));
        }
        let bytes = data.to_vec();
        let mut events = Vec::with_capacity(count);
        for (slot, bytes) in self.slots.iter().zip(bytes.chunks_exact(16)) {
            let begin = u64::from_ne_bytes(bytes[..8].try_into().unwrap());
            let finish = u64::from_ne_bytes(bytes[8..].try_into().unwrap());
            if begin == 0
                || finish == 0
                || begin == MTLCounterErrorValue
                || finish == MTLCounterErrorValue
                || finish < begin
                || begin < start.gpu_ticks
                || finish > end.gpu_ticks
            {
                return Err(Error::Native("invalid timestamp sample".into()));
            }
            let host = slot.host.get();
            events.push(ProfileEvent {
                name: host.name,
                host_begin_seconds: host.begin,
                host_end_seconds: host.end,
                gpu_begin_ticks: begin,
                gpu_end_ticks: finish,
                gpu_begin_seconds: convert_tick(begin, start, end)?,
                gpu_end_seconds: convert_tick(finish, start, end)?,
            });
        }
        Ok(Profile {
            start,
            end,
            host_clock_sample_bracket_seconds: self.sample_bracket.get(),
            events,
        })
    }
}
impl Session<'_> {
    /// Diagnostic only. Pre-admit an additional 64MiB allowance for native
    /// counter/pass objects and bounded resolve/export buffers; production has none.
    pub fn enable_profile(&mut self) -> Result<()> {
        if self.trace.is_some() || !self.history.is_empty() || self.evaluated != 0 {
            return Err(Error::Options("enable profile on a fresh session"));
        }
        // Seven custom projections use one encoder each. Original Half retains
        // bias, two dependent MPS markers, and storage conversion.
        let half_compensated = matches!(
            self.projection_mode,
            super::ProjectionMode::HalfCompensated | super::ProjectionMode::MlxAffine
        );
        let per_layer = if matches!(
            self.model.layers[0].q.storage,
            super::TensorStorage::Q4 { .. }
        ) || half_compensated
        {
            17
        } else {
            38
        };
        // Token selection replaces final validation in ordinary execution.
        // Reserve one spare slot because diagnostics may be enabled later and
        // must validate every logit row in addition to selecting the final row.
        let events = per_layer * self.model.config.num_layers
            + if half_compensated { 4 } else { 6 }
            + usize::from(self.readback_mode == super::ReadbackMode::GreedyToken);
        if events > MAX_EVENTS {
            return Err(Error::Unsupported(
                "model profiling exceeds admitted counter capacity",
            ));
        }
        let admitted = super::add(self.memory.admitted_bytes, super::ALLOWANCE)?;
        if admitted > self.options.memory_limit_bytes {
            return Err(Error::Budget {
                required: admitted,
                limit: self.options.memory_limit_bytes,
            });
        }
        let trace = autoreleasepool(|_| Trace::new(&self.model.gpu, events))?;
        self.trace = Some(trace);
        self.memory.metadata_allowance_bytes =
            super::add(self.memory.metadata_allowance_bytes, super::ALLOWANCE)?;
        self.memory.admitted_bytes = admitted;
        Ok(())
    }
    pub fn profile(&self) -> Result<Profile> {
        // Cleared before every attempted chunk, including command creation.
        // Timing and counter storage may still describe an earlier success.
        if self.replay || self.last_timing.is_none() || self.diagnostic_rows == 0 {
            return Err(Error::Options("no successful profiled chunk"));
        }
        autoreleasepool(|_| {
            self.trace
                .as_ref()
                .ok_or(Error::Options("profile disabled"))?
                .resolve()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unequal_clocks_and_large_origins_preserve_relative_nanoseconds() {
        let start = ClockPair {
            cpu_nanoseconds: u64::MAX - 1000,
            gpu_ticks: u64::MAX - 2000,
        };
        let end = ClockPair {
            cpu_nanoseconds: u64::MAX - 500,
            gpu_ticks: u64::MAX - 1900,
        };
        assert_eq!(convert_tick(start.gpu_ticks, start, end).unwrap(), 0.);
        assert!((convert_tick(start.gpu_ticks + 50, start, end).unwrap() - 250e-9).abs() < 1e-20);
        assert!((convert_tick(end.gpu_ticks, start, end).unwrap() - 500e-9).abs() < 1e-20);
    }
    #[test]
    fn invalid_clock_samples_have_no_duration_claim() {
        let start = ClockPair {
            cpu_nanoseconds: 1000,
            gpu_ticks: 100,
        };
        let end = ClockPair {
            cpu_nanoseconds: 2000,
            gpu_ticks: 200,
        };
        for tick in [0, 99, 201, u64::MAX] {
            assert!(convert_tick(tick, start, end).is_err());
        }
        assert!(convert_tick(100, start, start).is_err());
        assert!(clock_spans(end, start).is_err());
    }
}
