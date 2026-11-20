//! One synchronous ConvIn workspace for the all-layer shared-placement witness.
use super::*;

const O: usize = 3 * I;
use crate::{metal::lfm3_shared_weights::BankRef, model::Error};

const MAX: usize = 128;

pub(super) struct ConvWorkspace {
    kernels: Kernels,
    input: Buffer<u16>,
    candidate: Buffer<u16>,
    routes: Buffer<u32>,
    status: Buffer<u32>,
    submitted: u64,
    completed: u64,
    clocks: [f64; 2],
}

impl ConvWorkspace {
    pub(super) const BYTES: usize = MAX * I * 2 + (MAX * O + 1) * 6 + 8;

    pub(super) fn new(gpu: &Gpu) -> std::result::Result<Self, Error> {
        Ok(Self {
            kernels: Kernels::with_shader(gpu, include_str!("opposite_fma.metal"))
                .ok_or(Error::InvalidConfig("LFM3 shared ConvIn pipelines"))?,
            input: Buffer::new(gpu, MAX * I).map_err(|_| Error::Allocation)?,
            candidate: Buffer::new(gpu, MAX * O + 1).map_err(|_| Error::Allocation)?,
            routes: Buffer::new(gpu, MAX * O + 1).map_err(|_| Error::Allocation)?,
            status: Buffer::new(gpu, 2).map_err(|_| Error::Allocation)?,
            submitted: 0,
            completed: 0,
            clocks: [0.; 2],
        })
    }

    pub(super) fn telemetry(&self) -> (u64, u64, [f64; 2]) {
        (self.submitted, self.completed, self.clocks)
    }

    pub(super) fn validate(gpu: &Gpu, bank: BankRef<'_>) -> std::result::Result<(), Error> {
        if bank.words() != I * O || !bank.device_matches(&gpu.device) {
            return Err(Error::InvalidConfig(
                "LFM3 shared ConvIn bank device or extent",
            ));
        }
        Ok(())
    }

    pub(super) fn project(
        &mut self,
        gpu: &Gpu,
        bank: BankRef<'_>,
        input: &[bf16],
        output: &mut [bf16],
        count: usize,
        mut tile: Option<super::projection_tile::Launch<'_>>,
    ) -> std::result::Result<[u64; 3], Error> {
        Self::validate(gpu, bank)?;
        if !(1..=MAX).contains(&count) || input.len() != count * I || output.len() != count * O {
            return Err(Error::InvalidConfig("LFM3 shared ConvIn extents"));
        }
        let next = self
            .submitted
            .checked_add(1)
            .ok_or(Error::InvalidConfig("LFM3 shared ConvIn submission count"))?;
        let active = count * O;
        let (kernels, groups) = match &tile {
            Some(t) => (t.kernels, active / t.outputs_per_group),
            None => (&self.kernels, active),
        };
        self.input.with_mut(|dst| {
            if let Some(t) = tile.as_mut() {
                t.stage(input, dst, I);
            } else {
                for (slot, value) in dst.iter_mut().zip(input) {
                    *slot = value.to_bits();
                }
            }
        });
        self.candidate.with_mut(|dst| dst[..=active].fill(SENTINEL));
        self.routes.with_mut(|dst| dst[..=active].fill(ROUTE_GUARD));
        self.status
            .with_mut(|dst| dst.copy_from_slice(&[0, STATUS_GUARD]));
        let dims = [count as u32, I as u32, O as u32, 0];
        let command = gpu
            .command()
            .map_err(|_| Error::InvalidConfig("LFM3 shared ConvIn command"))?;
        for (pipeline, groups, project) in [
            (&kernels.admission, (I * O).div_ceil(32), false),
            (&kernels.projection, groups, true),
        ] {
            let encoder = command
                .computeCommandEncoder()
                .ok_or(Error::InvalidConfig("LFM3 shared ConvIn encoder"))?;
            encoder.setComputePipelineState(pipeline);
            // SAFETY: same-device immutable bank, exact const input extent and
            // private disjoint scratch. All owners survive the checked wait.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&self.input.raw), 0, 0);
                bank.bind_read_only(&encoder, 1);
                encoder.setBuffer_offset_atIndex(Some(&self.status.raw), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&dims).cast(),
                    size_of::<[u32; 4]>(),
                    3,
                );
                if project {
                    encoder.setBuffer_offset_atIndex(Some(&self.candidate.raw), 0, 4);
                    encoder.setBuffer_offset_atIndex(Some(&self.routes.raw), 0, 5);
                    if let Some(t) = &tile {
                        t.bind_tags(&encoder);
                    }
                }
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
        }
        let start = Instant::now();
        command.commit();
        self.submitted = next;
        command.waitUntilCompleted();
        let wait = start.elapsed().as_secs_f64();
        if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
            return Err(Error::InvalidConfig("LFM3 shared ConvIn completion"));
        }
        let (a, b) = (command.GPUStartTime(), command.GPUEndTime());
        if !a.is_finite() || !b.is_finite() || a <= 0. || b <= a {
            return Err(Error::InvalidConfig("LFM3 shared ConvIn completed clock"));
        }
        let (code, status_guard) = self.status.with_slice(|w| (w[0], w[1]));
        let mut census = [0; 3];
        let routes_ok = self.routes.with_slice(|w| {
            w[active] == ROUTE_GUARD
                && if code == 2 {
                    w[..active].iter().all(|&v| v == ROUTE_GUARD)
                } else if code == 0 {
                    w[..active].iter().all(|&v| {
                        super::projection_tile::record(v, 12, I, tile.is_some(), &mut census)
                    })
                } else {
                    true
                }
        });
        let output_ok = self.candidate.with_slice(|w| {
            w[active] == SENTINEL
                && if code == 2 {
                    w[..active].iter().all(|&v| v == SENTINEL)
                } else if code == 0 {
                    w[..active].iter().all(|&v| (v & 0x7fff) < 0x7f80)
                } else {
                    true
                }
        });
        if status_guard != STATUS_GUARD || !routes_ok || !output_ok {
            return Err(Error::InvalidConfig("LFM3 shared ConvIn guards"));
        }
        match code {
            0 => {}
            2 => {
                return Err(Error::InvalidConfig(
                    "LFM3 shared ConvIn nonfinite input or weight",
                ));
            }
            4 => return Err(Error::InvalidConfig("LFM3 shared ConvIn overflow")),
            _ => return Err(Error::InvalidConfig("LFM3 shared ConvIn status")),
        }
        self.candidate.with_slice(|w| {
            for (dst, &src) in output.iter_mut().zip(w) {
                *dst = bf16::from_bits(src);
            }
        });
        self.completed += 1;
        self.clocks[0] += b - a;
        self.clocks[1] += wait;
        Ok(census)
    }
}
