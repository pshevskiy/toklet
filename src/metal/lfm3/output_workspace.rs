//! One synchronous operator output workspace for the all-layer shared-placement witness.
use super::pair::domain_tag;
use super::*;
use crate::{metal::lfm3_shared_weights::BankRef, model::Error};
const O: usize = 2048;

const MAX: usize = 128;

pub(super) struct OutputWorkspace {
    kernels: Kernels,
    input: Buffer<u16>,
    candidate: Buffer<u16>,
    routes: Buffer<u32>,
    status: Buffer<u32>,
    pub(super) submitted: [u64; 2],
    completed: [u64; 2],
    native: [u64; 2],
    ordered: [u64; 2],
    clocks: [[f64; 2]; 2],
    tags: [Buffer<u32>; 24],
    tag_owners: [Option<usize>; 24],
    input_tags: Buffer<u32>,
}

impl OutputWorkspace {
    pub(super) const BUFFER_COUNT: usize = 29;
    pub(super) const BYTES: usize = MAX * I * 2 + (MAX * O + 1) * 6 + 8 + (MAX + 24 * O) * 4;

    pub(super) fn new(gpu: &Gpu) -> std::result::Result<Self, Error> {
        let mut tags = Vec::new();
        tags.try_reserve_exact(24).map_err(|_| Error::Allocation)?;
        for _ in 0..24 {
            tags.push(Buffer::new(gpu, O).map_err(|_| Error::Allocation)?);
        }
        Ok(Self {
            tags: tags
                .try_into()
                .map_err(|_| Error::InvalidConfig("LFM3 output tag count"))?,
            tag_owners: [None; 24],
            input_tags: Buffer::new(gpu, MAX).map_err(|_| Error::Allocation)?,
            kernels: Kernels::with_shader(gpu, include_str!("normal_domain.metal")).ok_or(
                Error::InvalidConfig("LFM3 shared operator output pipelines"),
            )?,
            input: Buffer::new(gpu, MAX * I).map_err(|_| Error::Allocation)?,
            candidate: Buffer::new(gpu, MAX * O + 1).map_err(|_| Error::Allocation)?,
            routes: Buffer::new(gpu, MAX * O + 1).map_err(|_| Error::Allocation)?,
            status: Buffer::new(gpu, 2).map_err(|_| Error::Allocation)?,
            submitted: [0; 2],
            completed: [0; 2],
            native: [0; 2],
            ordered: [0; 2],
            clocks: [[0.; 2]; 2],
        })
    }

    pub(super) fn seal_tags(
        &mut self,
        gpu: &Gpu,
        layer: usize,
        bank: BankRef<'_>,
        words: &[bf16],
    ) -> std::result::Result<(), Error> {
        Self::validate(gpu, bank)?;
        if words.len() != I * O {
            return Err(Error::InvalidConfig("LFM3 output tag source extent"));
        }
        let tags = self
            .tags
            .get_mut(layer)
            .ok_or(Error::InvalidConfig("LFM3 output tag layer"))?;
        if self.tag_owners[layer].is_some() {
            return Err(Error::InvalidConfig("LFM3 output tags already sealed"));
        }
        tags.with_mut(|dst| {
            for (tag, row) in dst.iter_mut().zip(words.chunks_exact(I)) {
                *tag = domain_tag(row.iter().map(|word| word.to_bits()));
            }
        });
        self.tag_owners[layer] = Some(bank.identity());
        Ok(())
    }

    fn validate(gpu: &Gpu, bank: BankRef<'_>) -> std::result::Result<(), Error> {
        if bank.words() != I * O || !bank.device_matches(&gpu.device) {
            return Err(Error::InvalidConfig(
                "LFM3 shared operator output bank device or extent",
            ));
        }
        Ok(())
    }

    pub(super) fn project(
        &mut self,
        gpu: &Gpu,
        bank: BankRef<'_>,
        (layer, role): (usize, usize),
        input: &[bf16],
        output: &mut [bf16],
        count: usize,
    ) -> std::result::Result<([f64; 2], usize), Error> {
        Self::validate(gpu, bank)?;
        if !(1..=MAX).contains(&count) || input.len() != count * I || output.len() != count * O {
            return Err(Error::InvalidConfig("LFM3 shared operator output extents"));
        }
        let next = self.submitted[role]
            .checked_add(1)
            .ok_or(Error::InvalidConfig(
                "LFM3 shared operator output submission count",
            ))?;
        let tags = self
            .tags
            .get(layer)
            .ok_or(Error::InvalidConfig("LFM3 output dispatch layer"))?;
        if self.tag_owners[layer] != Some(bank.identity()) {
            return Err(Error::InvalidConfig("LFM3 output tag owner"));
        }
        let active = count * O;
        self.input.with_mut(|dst| {
            self.input_tags.with_mut(|tags| {
                for ((buffer, x), tag) in dst[..input.len()]
                    .chunks_exact_mut(I)
                    .zip(input.chunks_exact(I))
                    .zip(tags)
                {
                    *tag = domain_tag(x.iter().zip(buffer).map(|(v, d)| {
                        *d = v.to_bits();
                        *d
                    }));
                }
            });
        });
        self.candidate.with_mut(|dst| dst[..=active].fill(SENTINEL));
        self.routes.with_mut(|dst| dst[..=active].fill(ROUTE_GUARD));
        self.status
            .with_mut(|dst| dst.copy_from_slice(&[0, STATUS_GUARD]));
        let dims = [count as u32, I as u32, O as u32, 0];
        let command = gpu
            .command()
            .map_err(|_| Error::InvalidConfig("LFM3 shared operator output command"))?;
        for (pipeline, groups, project) in [
            (&self.kernels.admission, (I * O).div_ceil(32), false),
            (&self.kernels.projection, active, true),
        ] {
            let encoder = command
                .computeCommandEncoder()
                .ok_or(Error::InvalidConfig("LFM3 shared operator output encoder"))?;
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
                    encoder.setBuffer_offset_atIndex(Some(&self.input_tags.raw), 0, 6);
                    encoder.setBuffer_offset_atIndex(Some(&tags.raw), 0, 7);
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
        self.submitted[role] = next;
        command.waitUntilCompleted();
        let wait = start.elapsed().as_secs_f64();
        if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
            return Err(Error::InvalidConfig(
                "LFM3 shared operator output completion",
            ));
        }
        let (a, b) = (command.GPUStartTime(), command.GPUEndTime());
        if !a.is_finite() || !b.is_finite() || a <= 0. || b <= a {
            return Err(Error::InvalidConfig(
                "LFM3 shared operator output completed clock",
            ));
        }
        let (code, status_guard) = self.status.with_slice(|w| (w[0], w[1]));
        let mut native = 0;
        let routes_ok = self.routes.with_slice(|w| {
            w[active] == ROUTE_GUARD
                && if code == 2 {
                    w[..active].iter().all(|&v| v == ROUTE_GUARD)
                } else if code == 0 {
                    w[..active].iter().all(|&v| {
                        if v == 0x80000000 {
                            native += 1;
                            true
                        } else {
                            (v & 4095) + (v >> 12) <= (I + 32) as u32
                        }
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
            return Err(Error::InvalidConfig("LFM3 shared operator output guards"));
        }
        match code {
            0 => {}
            2 => {
                return Err(Error::InvalidConfig(
                    "LFM3 shared operator output nonfinite input or weight",
                ));
            }
            4 => return Err(Error::InvalidConfig("LFM3 shared operator output overflow")),
            _ => return Err(Error::InvalidConfig("LFM3 shared operator output status")),
        }
        let completed = self.completed[role]
            .checked_add(1)
            .ok_or(Error::InvalidConfig("LFM3 output completion count"))?;
        let native_total = self.native[role]
            .checked_add(native as u64)
            .ok_or(Error::InvalidConfig("LFM3 output native count"))?;
        let ordered_total = self.ordered[role]
            .checked_add((active - native) as u64)
            .ok_or(Error::InvalidConfig("LFM3 output ordered count"))?;
        self.completed[role] = completed;
        self.native[role] = native_total;
        self.ordered[role] = ordered_total;
        self.clocks[role][0] += b - a;
        self.clocks[role][1] += wait;
        self.candidate.with_slice(|w| {
            for (dst, &src) in output.iter_mut().zip(w) {
                *dst = bf16::from_bits(src);
            }
        });
        Ok(([b - a, wait], native))
    }

    pub(super) fn stats(&self, published: [u64; 2]) -> serde_json::Value {
        serde_json::json!({
            "workspace_bytes": Self::BYTES, "submitted": self.submitted, "completed": self.completed,
            "published": published, "native_outputs": self.native, "ordered_outputs": self.ordered,
            "completed_gpu_seconds": [self.clocks[0][0],self.clocks[1][0]],
            "commit_wait_seconds": [self.clocks[0][1],self.clocks[1][1]],
        })
    }
}
