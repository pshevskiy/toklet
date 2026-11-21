//! Bounded one-layer gate/up BF16 pair witness using the existing ordered shader.
use super::*;

const MLP: usize = 7680;
const MAX: usize = 128;

pub(super) struct PairWorkspace {
    pub(super) kernels: Kernels,
    pub(super) gpu: Gpu,
    pub(super) staged: Buffer<u16>,
    pub(super) output: [Buffer<u16>; 2],
    pub(super) routes: [Buffer<u32>; 2],
    pub(super) status: Buffer<u32>,
    pub(super) submitted: u64,
}

impl PairWorkspace {
    pub(super) const BYTES: usize = MAX * I * 2 + (MAX * MLP + 1) * 12 + 8;

    pub(super) fn try_new() -> std::result::Result<Self, crate::model::Error> {
        assert!((2 * MAX * MLP + MAX * I) * 32 < u32::MAX as usize);
        use crate::model::Error;
        let gpu = Gpu::new().map_err(|_| Error::InvalidConfig("LFM3 pair device or queue"))?;
        let kernels = Kernels::with_shader(&gpu, include_str!("opposite_fma.metal"))
            .ok_or(Error::InvalidConfig("LFM3 pair pipelines"))?;
        let output = [
            Buffer::new(&gpu, MAX * MLP + 1),
            Buffer::new(&gpu, MAX * MLP + 1),
        ];
        let routes = [
            Buffer::new(&gpu, MAX * MLP + 1),
            Buffer::new(&gpu, MAX * MLP + 1),
        ];
        let [gate, up] = output;
        let [gate_routes, up_routes] = routes;
        Ok(Self {
            staged: Buffer::new(&gpu, MAX * I).map_err(|_| Error::Allocation)?,
            output: [
                gate.map_err(|_| Error::Allocation)?,
                up.map_err(|_| Error::Allocation)?,
            ],
            routes: [
                gate_routes.map_err(|_| Error::Allocation)?,
                up_routes.map_err(|_| Error::Allocation)?,
            ],
            status: Buffer::new(&gpu, 2).map_err(|_| Error::Allocation)?,
            kernels,
            gpu,
            submitted: 0,
        })
    }

    pub(super) fn project_selected(
        &mut self,
        input: &[bf16],
        gate: &mut [bf16],
        up: &mut [bf16],
        count: usize,
        banks: [crate::metal::lfm3_shared_weights::BankRef<'_>; 2],
        mut domain: Option<DomainLaunch<'_>>,
    ) -> std::result::Result<([f64; 2], PairRoutes), crate::model::Error> {
        use crate::model::Error;
        if !(1..=MAX).contains(&count)
            || input.len() != count * I
            || gate.len() != count * MLP
            || up.len() != count * MLP
        {
            return Err(Error::InvalidConfig("selected LFM3 MLP pair extents"));
        }
        let tile_size = domain.as_ref().map_or(1, |d| d.tile_size);
        let fma4 = count == 1 && tile_size == 4 && domain.as_ref().is_some_and(|d| d.selected);
        if tile_size != 1 && count != MAX && !fma4 {
            return Err(Error::InvalidConfig("LFM3 shared T4 prefill extent"));
        }
        let active = count * MLP;
        let projection_groups = active / tile_size;
        let staged = &mut self.staged;
        staged.with_mut(|dst| {
            if let Some(bounds) = domain.as_mut() {
                bounds.input.with_mut(|metadata| {
                    for ((row, words), slot) in dst[..input.len()]
                        .chunks_exact_mut(I)
                        .zip(input.chunks_exact(I))
                        .zip(metadata)
                    {
                        *slot = domain_tag(words.iter().zip(row).map(|(value, out)| {
                            *out = value.to_bits();
                            *out
                        }));
                    }
                });
            } else {
                for (dst, &word) in dst.iter_mut().zip(input) {
                    *dst = word.to_bits();
                }
            }
        });
        let output = &mut self.output;
        let routes = &mut self.routes;
        for buf in output.iter_mut() {
            buf.with_mut(|dst| dst[..=active].fill(SENTINEL));
        }
        for buf in routes.iter_mut() {
            buf.with_mut(|dst| dst[..=active].fill(ROUTE_GUARD));
        }
        let status = &mut self.status;
        status.with_mut(|dst| dst.copy_from_slice(&[0, STATUS_GUARD]));
        let (dims, admission_groups) = Kernels::sealed_admission(banks[0], count, I, MLP);
        let command = self
            .gpu
            .command()
            .map_err(|_| Error::InvalidConfig("LFM3 pair command allocation"))?;
        let threads = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        let projection = domain
            .as_ref()
            .filter(|d| d.selected)
            .map_or(&self.kernels.projection, |d| d.projection);
        for (pipeline, project) in [(&self.kernels.admission, false), (projection, true)] {
            for (i, bank) in banks.iter().enumerate() {
                let encoder = command
                    .computeCommandEncoder()
                    .ok_or(Error::InvalidConfig("LFM3 pair encoder allocation"))?;
                encoder.setComputePipelineState(pipeline);
                // SAFETY: the model caller checks both sealed banks and retains them until completion.
                unsafe { bank.bind_read_only(&encoder, 1) };
                // SAFETY: each buffer is same-device, initialized, tracked and
                // retained through the one wait. Both admissions precede both
                // projections; input is shared read-only, candidates disjoint.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&staged.raw), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&status.raw), 0, 2);
                    encoder.setBytes_length_atIndex(
                        NonNull::from(&dims).cast(),
                        size_of::<[u32; 4]>(),
                        3,
                    );
                    if project {
                        encoder.setBuffer_offset_atIndex(Some(&output[i].raw), 0, 4);
                        encoder.setBuffer_offset_atIndex(Some(&routes[i].raw), 0, 5);
                        if let Some(bounds) = domain.as_ref().filter(|d| d.selected) {
                            encoder.setBuffer_offset_atIndex(Some(&bounds.input.raw), 0, 6);
                            encoder.setBuffer_offset_atIndex(
                                Some(&bounds.weights.raw),
                                (2 * bounds.layer + i) * MLP * size_of::<u32>(),
                                7,
                            );
                        }
                    }
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: if project {
                            projection_groups
                        } else {
                            admission_groups
                        },
                        height: 1,
                        depth: 1,
                    },
                    threads,
                );
                encoder.endEncoding();
            }
        }
        let next = self
            .submitted
            .checked_add(1)
            .ok_or(Error::InvalidConfig("MLP pair command count"))?;
        let submitted = Instant::now();
        command.commit();
        self.submitted = next;
        command.waitUntilCompleted();
        let wait = submitted.elapsed().as_secs_f64();
        if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
            return Err(Error::InvalidConfig("ordered LFM3 MLP pair command"));
        }
        let start = command.GPUStartTime();
        let end = command.GPUEndTime();
        if !start.is_finite() || !end.is_finite() || start <= 0. || end <= start {
            return Err(Error::InvalidConfig("ordered LFM3 MLP pair clock"));
        }
        let status = status
            .with_slice(|s| (s[1] == STATUS_GUARD).then_some(s[0]))
            .ok_or(Error::InvalidConfig("ordered LFM3 MLP pair status guard"))?;
        if status != 0 {
            for (result, census) in output.iter().zip(routes.iter()) {
                let intact_output = result.with_slice(|s| {
                    s[active] == SENTINEL
                        && (status != 2 || s[..active].iter().all(|&x| x == SENTINEL))
                });
                let intact_routes = census.with_slice(|s| {
                    s[active] == ROUTE_GUARD
                        && (status != 2 || s[..active].iter().all(|&x| x == ROUTE_GUARD))
                });
                if !intact_output || !intact_routes {
                    return Err(Error::InvalidConfig("ordered LFM3 MLP pair output guard"));
                }
            }
            return Err(Error::InvalidConfig("ordered LFM3 MLP pair status"));
        }
        let mut receipt = PairRoutes {
            native: [0; 2],
            homogeneous: [0; 2],
            projection_groups,
        };
        let selected = domain.is_some_and(|d| d.selected);
        for (i, (result, census)) in output.iter().zip(routes.iter()).enumerate() {
            if !result.with_slice(|s| {
                s[count * MLP] == SENTINEL && s[..count * MLP].iter().all(|b| (b & 0x7fff) < 0x7f80)
            }) || !census.with_slice(|s| {
                s[count * MLP] == ROUTE_GUARD
                    && s[..active].iter().all(|&r| {
                        if selected && r == 1 << 31 {
                            receipt.native[i] += 1;
                            true
                        } else if selected && tile_size != 1 && r == 0xc0000000 {
                            receipt.native[i] += 1;
                            receipt.homogeneous[i] += 1;
                            true
                        } else {
                            (r & 4095) + (r >> 12) <= (I + 32) as u32
                        }
                    })
            }) {
                return Err(Error::InvalidConfig("ordered LFM3 MLP pair output guard"));
            }
        }
        if receipt
            .homogeneous
            .iter()
            .any(|&n| n % tile_size as u64 != 0)
        {
            return Err(Error::InvalidConfig("LFM3 shared pair tile markers"));
        }
        for (result, destination) in output.iter().zip([gate, up]) {
            result.with_slice(|s| {
                for (dst, &bits) in destination.iter_mut().zip(s) {
                    *dst = bf16::from_bits(bits);
                }
            });
        }
        Ok(([end - start, wait], receipt))
    }
}

pub(super) struct PairRoutes {
    pub(super) native: [u64; 2],
    pub(super) homogeneous: [u64; 2],
    pub(super) projection_groups: usize,
}

pub(super) struct DomainLaunch<'a> {
    pub(super) tile_size: usize,
    pub(super) selected: bool,
    pub(super) projection: &'a Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) input: &'a mut Buffer<u32>,
    pub(super) weights: &'a Buffer<u32>,
    pub(super) layer: usize,
}

pub(super) fn domain_tag(words: impl Iterator<Item = u16>) -> u32 {
    let mut min = 255_u32;
    let mut max = 0_u32;
    let mut flag = 0_u32;
    for bits in words {
        let raw = u32::from(bits) & 0x7fff;
        if raw == 0 {
            continue;
        }
        let e = raw >> 7;
        if e == 0 || e == 255 {
            flag |= 0x20000;
            continue;
        }
        min = min.min(e);
        max = max.max(e);
        flag |= 0x10000;
    }
    flag | (max << 8) | min
}
