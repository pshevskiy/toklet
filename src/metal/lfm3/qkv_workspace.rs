//! Shared T128 storage for the three individually completed Q/K/V projections.
use super::{pair::domain_tag, projection_tile::NormalExtent, *};
use crate::{metal::lfm3_shared_weights::BankRef, model::Error};

#[path = "qkv_t1.rs"]
mod qkv_t1;

const T: usize = 128;
const N: [usize; 3] = [2048, 512, 512];
const L: usize = 11;
const H: usize = 4;
const D: usize = 128;

struct Plane {
    output: Buffer<u16>,
    routes: Buffer<u32>,
    status: Buffer<u32>,
    weight_tags: Buffer<u32>,
}

impl Plane {
    fn new(gpu: &Gpu, n: usize) -> std::result::Result<Self, Error> {
        Ok(Self {
            output: Buffer::new(gpu, T * n + 1).map_err(|_| Error::Allocation)?,
            routes: Buffer::new(gpu, T * n + 1).map_err(|_| Error::Allocation)?,
            status: Buffer::new(gpu, 2).map_err(|_| Error::Allocation)?,
            weight_tags: Buffer::new(gpu, L * n).map_err(|_| Error::Allocation)?,
        })
    }
}

#[derive(Clone, Copy, Default)]
struct Census {
    submitted: u64,
    complete: u64,
    native: u64,
    gpu: f64,
    wait: f64,
}

// Reuse the duplicate scalar pipeline pair's inline slots. Other
// workspace fields and all GPU buffer extents retain their released shape.
const _: () = {
    assert!(size_of::<Option<Kernels>>() == size_of::<Kernels>());
    assert!(std::mem::align_of::<Option<Kernels>>() == std::mem::align_of::<Kernels>());
};

pub(super) struct Workspace {
    panel: Option<Kernels>,
    t1: Option<qkv_t1::T1>,
    input: Buffer<u16>,
    tags: Buffer<u32>,
    planes: [Plane; 3],
    owners: [[usize; 3]; L],
    census: [[Census; 3]; L],
}

const _: () = assert!(
    size_of::<Option<Workspace>>()
        == size_of::<Kernels>()
            + size_of::<Option<qkv_t1::T1>>()
            + size_of::<Buffer<u16>>()
            + size_of::<Buffer<u32>>()
            + size_of::<[Plane; 3]>()
            + size_of::<[[usize; 3]; L]>()
            + size_of::<[[Census; 3]; L]>()
);

impl Workspace {
    pub(super) const BYTE_SIZES: [usize; 14] = [
        T * I * 2,
        T * 4,
        (T * N[0] + 1) * 2,
        (T * N[0] + 1) * 4,
        8,
        L * N[0] * 4,
        (T * N[1] + 1) * 2,
        (T * N[1] + 1) * 4,
        8,
        L * N[1] * 4,
        (T * N[2] + 1) * 2,
        (T * N[2] + 1) * 4,
        8,
        L * N[2] * 4,
    ];

    pub(super) fn new(gpu: &Gpu) -> std::result::Result<Self, Error> {
        Ok(Self {
            panel: Kernels::with_shader(gpu, include_str!("panel84_attention_linear.metal")),
            t1: None,
            input: Buffer::new(gpu, T * I).map_err(|_| Error::Allocation)?,
            tags: Buffer::new(gpu, T).map_err(|_| Error::Allocation)?,
            planes: [
                Plane::new(gpu, N[0])?,
                Plane::new(gpu, N[1])?,
                Plane::new(gpu, N[2])?,
            ],
            owners: [[0; 3]; L],
            census: Default::default(),
        })
    }

    pub(super) fn panel(&self) -> Option<&Kernels> {
        self.panel.as_ref()
    }

    fn slot(layer: usize) -> std::result::Result<usize, Error> {
        if (2..=2 * L).contains(&layer) && layer.is_multiple_of(2) {
            Ok(layer / 2 - 1)
        } else {
            Err(Error::InvalidConfig("LFM3 QKV layer"))
        }
    }

    pub(super) fn seal(
        &mut self,
        gpu: &Gpu,
        layer: usize,
        banks: [(BankRef<'_>, &[bf16]); 3],
    ) -> std::result::Result<(), Error> {
        let slot = Self::slot(layer)?;
        if self.owners[slot] != [0; 3]
            || banks.iter().zip(N).any(|((bank, data), width)| {
                bank.identity() == 0
                    || !bank.device_matches(&gpu.device)
                    || bank.words() != I * width
                    || data.len() != I * width
            })
        {
            return Err(Error::InvalidConfig("LFM3 QKV seal"));
        }
        for (j, ((bank, data), plane)) in banks.into_iter().zip(&mut self.planes).enumerate() {
            let width = N[j];
            plane.weight_tags.with_mut(|tags| {
                for (tag, row) in tags[slot * width..(slot + 1) * width]
                    .iter_mut()
                    .zip(data.chunks_exact(I))
                {
                    *tag = domain_tag(row.iter().map(|w| w.to_bits()));
                }
            });
            self.owners[slot][j] = bank.identity();
        }
        Ok(())
    }

    pub(super) fn run(
        &mut self,
        gpu: &Gpu,
        scalar: &Kernels,
        layer: usize,
        banks: [BankRef<'_>; 3],
        rhs: &[bf16],
        out: [&mut [bf16]; 3],
    ) -> std::result::Result<(), Error> {
        let slot = Self::slot(layer)?;
        if rhs.len() != T * I
            || self.owners[slot] != banks.map(BankRef::identity)
            || banks.iter().zip(out.iter()).zip(N).any(|((b, y), n)| {
                b.identity() == 0
                    || !b.device_matches(&gpu.device)
                    || b.words() != I * n
                    || y.len() != T * n
            })
        {
            return Err(Error::InvalidConfig("LFM3 QKV exact owner or geometry"));
        }
        self.input.with_mut(|dst| {
            self.tags.with_mut(|tags| {
                for ((row, stored), tag) in
                    rhs.chunks_exact(I).zip(dst.chunks_exact_mut(I)).zip(tags)
                {
                    *tag = domain_tag(row.iter().zip(stored).map(|(x, w)| {
                        *w = x.to_bits();
                        *w
                    }));
                }
            })
        });
        let input_extent = self
            .panel
            .as_ref()
            .and_then(|_| self.tags.with_slice(NormalExtent::from_tags));
        for (j, (bank, plane)) in banks.into_iter().zip(&mut self.planes).enumerate() {
            let n = N[j];
            let active = T * n;
            let offset = slot * n;
            let panel = self.panel.as_ref().filter(|_| {
                input_extent.is_some_and(|x| {
                    plane
                        .weight_tags
                        .with_slice(|w| x.certifies(&w[offset..offset + n]))
                })
            });
            let kernels = panel.unwrap_or(scalar);
            let groups = if panel.is_some() { active / 32 } else { active };
            plane.output.with_mut(|w| w.fill(SENTINEL));
            plane.routes.with_mut(|w| w.fill(ROUTE_GUARD));
            plane
                .status
                .with_mut(|w| w.copy_from_slice(&[0, STATUS_GUARD]));
            let (dims, admission_groups) = Kernels::sealed_admission(bank, T, I, n);
            let command = gpu
                .command()
                .map_err(|_| Error::InvalidConfig("LFM3 QKV command"))?;
            for (pipeline, groups, project) in [
                (&kernels.admission, admission_groups, false),
                (&kernels.projection, groups, true),
            ] {
                let enc = command
                    .computeCommandEncoder()
                    .ok_or(Error::InvalidConfig("LFM3 QKV encoder"))?;
                enc.setComputePipelineState(pipeline);
                // All owners are live through completion. Bound buffers are disjoint.
                // The immutable bank and tag windows have the authenticated row size.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.input.raw), 0, 0);
                    bank.bind_read_only(&enc, 1);
                    enc.setBuffer_offset_atIndex(Some(&plane.status.raw), 0, 2);
                    enc.setBytes_length_atIndex(
                        NonNull::from(&dims).cast(),
                        size_of::<[u32; 4]>(),
                        3,
                    );
                    if project {
                        enc.setBuffer_offset_atIndex(Some(&plane.output.raw), 0, 4);
                        enc.setBuffer_offset_atIndex(Some(&plane.routes.raw), 0, 5);
                        if panel.is_none() {
                            enc.setBuffer_offset_atIndex(Some(&self.tags.raw), 0, 6);
                            enc.setBuffer_offset_atIndex(
                                Some(&plane.weight_tags.raw),
                                offset * size_of::<u32>(),
                                7,
                            );
                        }
                    }
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
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
                enc.endEncoding();
            }
            let s = &mut self.census[slot][j];
            let submitted = s
                .submitted
                .checked_add(1)
                .ok_or(Error::InvalidConfig("LFM3 QKV census overflow"))?;
            let clock = Instant::now();
            command.commit();
            s.submitted = submitted;
            command.waitUntilCompleted();
            let wait = clock.elapsed().as_secs_f64();
            let (start, end) = (command.GPUStartTime(), command.GPUEndTime());
            if command.status() != MTLCommandBufferStatus::Completed
                || command.error().is_some()
                || !start.is_finite()
                || !end.is_finite()
                || start <= 0.
                || end <= start
            {
                return Err(Error::InvalidConfig("LFM3 QKV completion"));
            }
            s.wait += wait;
            s.gpu += end - start;
            let (code, guard) = plane.status.with_slice(|w| (w[0], w[1]));
            let mut native = 0;
            let valid = plane.routes.with_slice(|w| {
                w[active] == ROUTE_GUARD
                    && (code != 2 || w[..active].iter().all(|&v| v == ROUTE_GUARD))
                    && (code != 0
                        || w[..active].iter().all(|&v| {
                            if v == 0x80000000 {
                                native += 1;
                                true
                            } else {
                                panel.is_none() && (v & 4095) + (v >> 12) <= 2080
                            }
                        }))
            });
            let words = plane.output.with_slice(|w| {
                w[active] == SENTINEL
                    && (code != 2 || w[..active].iter().all(|&v| v == SENTINEL))
                    && (code != 0 || w[..active].iter().all(|&v| v & 0x7fff < 0x7f80))
            });
            if guard != STATUS_GUARD || !valid || !words {
                return Err(Error::InvalidConfig("LFM3 QKV guards"));
            }
            if code != 0 {
                return Err(Error::InvalidConfig("LFM3 QKV projection status"));
            }
            s.native = s
                .native
                .checked_add(native)
                .ok_or(Error::InvalidConfig("LFM3 QKV native count"))?;
            s.complete += 1;
        }
        // Publish only after all three private statuses, routes and outputs
        // pass. V goes directly to the original head-major proposal, avoiding
        // a second token-major V scratch buffer.
        let [q, k, v] = out;
        for (plane, dst) in self.planes[..2].iter().zip([q, k]) {
            plane.output.with_slice(|w| {
                for (to, &bits) in dst.iter_mut().zip(w) {
                    *to = bf16::from_bits(bits);
                }
            });
        }
        self.planes[2].output.with_slice(|w| {
            for head in 0..H {
                for token in 0..T {
                    let from = token * N[2] + head * D;
                    let to = (head * T + token) * D;
                    for (dst, &bits) in v[to..to + D].iter_mut().zip(&w[from..from + D]) {
                        *dst = bf16::from_bits(bits);
                    }
                }
            }
        });
        Ok(())
    }

    pub(super) fn report(&self) -> serde_json::Value {
        let rows = self
            .census
            .iter()
            .enumerate()
            .map(|(slot, roles)| {
                let roles = roles
                    .iter()
                    .zip(N)
                    .map(|(s, n)| {
                        serde_json::json!({
                            "submitted":s.submitted,"completed":s.complete,"native":s.native,
                            "ordered":s.complete*(T*n) as u64-s.native,
                            "gpu_seconds":s.gpu,"wait_seconds":s.wait,
                        })
                    })
                    .collect::<Vec<_>>();
                serde_json::json!({"layer":2*(slot+1),"roles":roles})
            })
            .collect::<Vec<_>>();
        serde_json::json!(rows)
    }
}
