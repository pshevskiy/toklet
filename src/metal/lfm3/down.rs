//! Isolated width-7680 ordered down projection, with a separate census ABI.
use super::*;

const K: usize = 7680;
const N: usize = 2048;
const T: usize = 128;
const OPS: u32 = 7712;

type ProjectionReport = ([f64; 2], [u64; 3], [u64; 3]);

pub(super) struct DownWorkspace {
    input: Buffer<u16>,
    output: Buffer<u16>,
    routes: Buffer<u32>,
    status: Buffer<u32>,
    kernels: Kernels,
    submissions: u64,
}

fn decode(word: u32) -> Option<[u64; 3]> {
    let (exact, short) = (word & 0xffff, word >> 16);
    let hard = OPS.checked_sub(exact + short)?;
    Some([u64::from(exact), u64::from(short), u64::from(hard)])
}

impl DownWorkspace {
    pub(super) const BYTES: usize = 2 * T * K + 6 * (T * N + 1) + 8;

    pub(super) fn new(gpu: &Gpu) -> std::result::Result<Self, crate::model::Error> {
        use crate::model::Error;
        let kernels = Kernels::with_shader(gpu, include_str!("down.metal"))
            .ok_or(Error::InvalidConfig("LFM3 down pipelines"))?;
        Ok(Self {
            input: Buffer::new(gpu, T * K).map_err(|_| Error::Allocation)?,
            output: Buffer::new(gpu, T * N + 1).map_err(|_| Error::Allocation)?,
            routes: Buffer::new(gpu, T * N + 1).map_err(|_| Error::Allocation)?,
            status: Buffer::new(gpu, 2).map_err(|_| Error::Allocation)?,
            kernels,
            submissions: 0,
        })
    }

    pub(super) fn submissions(&self) -> u64 {
        self.submissions
    }

    pub(super) fn project_selected(
        &mut self,
        gpu: &Gpu,
        input: &[bf16],
        dst: &mut [bf16],
        tokens: usize,
        bind: impl Fn(&ProtocolObject<dyn MTLComputeCommandEncoder>),
        mut tile: Option<super::projection_tile::Launch<'_>>,
    ) -> std::result::Result<ProjectionReport, crate::model::Error> {
        use crate::model::Error;
        if !(1..=T).contains(&tokens) || input.len() != tokens * K || dst.len() != tokens * N {
            return Err(Error::InvalidConfig("selected LFM3 down extent"));
        }
        let active = tokens * N;
        let (kernels, groups) = match &tile {
            Some(t) => (t.kernels, active / t.outputs_per_group),
            None => (&self.kernels, active),
        };
        let next = self
            .submissions
            .checked_add(1)
            .ok_or(Error::InvalidConfig("down submission count"))?;
        self.input.with_mut(|x| {
            if let Some(t) = tile.as_mut() {
                t.stage(input, x, K);
            } else {
                for (slot, word) in x.iter_mut().zip(input) {
                    *slot = word.to_bits();
                }
            }
        });
        self.output.with_mut(|x| x[..=active].fill(SENTINEL));
        self.routes.with_mut(|x| x[..=active].fill(ROUTE_GUARD));
        self.status
            .with_mut(|x| x.copy_from_slice(&[0, STATUS_GUARD]));
        let dims = [tokens as u32, K as u32, N as u32, 0];
        let command = gpu
            .command()
            .map_err(|_| Error::InvalidConfig("LFM3 down command allocation"))?;
        let threads = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        for (ps, ordered) in [(&kernels.admission, false), (&kernels.projection, true)] {
            let encoder = command
                .computeCommandEncoder()
                .ok_or(Error::InvalidConfig("LFM3 down encoder allocation"))?;
            encoder.setComputePipelineState(ps);
            bind(&encoder);
            // SAFETY: fixed initialized tracked buffers with matching ABI,
            // private writable allocations, owned through synchronous completion.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&self.input.raw), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&self.status.raw), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&dims).cast(),
                    size_of::<[u32; 4]>(),
                    3,
                );
                if ordered {
                    encoder.setBuffer_offset_atIndex(Some(&self.output.raw), 0, 4);
                    encoder.setBuffer_offset_atIndex(Some(&self.routes.raw), 0, 5);
                    if let Some(t) = &tile {
                        t.bind_tags(&encoder);
                    }
                }
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: if ordered {
                        groups
                    } else {
                        (K * N).div_ceil(32)
                    },
                    height: 1,
                    depth: 1,
                },
                threads,
            );
            encoder.endEncoding();
        }
        let clock = Instant::now();
        command.commit();
        self.submissions = next;
        command.waitUntilCompleted();
        let wait = clock.elapsed().as_secs_f64();
        if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
            return Err(Error::InvalidConfig("ordered LFM3 down command"));
        }
        let start = command.GPUStartTime();
        let end = command.GPUEndTime();
        if !start.is_finite() || !end.is_finite() || start <= 0. || end <= start {
            return Err(Error::InvalidConfig("ordered LFM3 down timestamp"));
        }
        let flags = self
            .status
            .with_slice(|s| (s[1] == STATUS_GUARD).then_some(s[0]))
            .ok_or(Error::InvalidConfig("ordered LFM3 down status guard"))?;
        let intact_output = self.output.with_slice(|x| {
            x[active] == SENTINEL && (flags != 2 || x[..active].iter().all(|&v| v == SENTINEL))
        });
        let intact_routes = self.routes.with_slice(|x| {
            x[active] == ROUTE_GUARD
                && (flags != 2 || x[..active].iter().all(|&v| v == ROUTE_GUARD))
        });
        if !intact_output || !intact_routes {
            return Err(Error::InvalidConfig("ordered LFM3 down output guard"));
        }
        match flags {
            0 => {}
            2 => return Err(Error::InvalidConfig("ordered LFM3 down nonfinite input")),
            4 => return Err(Error::InvalidConfig("ordered LFM3 down overflow")),
            _ => return Err(Error::InvalidConfig("ordered LFM3 down status")),
        }
        let mut census = [0; 3];
        let counts = self
            .routes
            .with_slice(|words| {
                words[..active]
                    .iter()
                    .try_fold([0_u64; 3], |mut total, &word| {
                        if !super::projection_tile::record(word, 16, K, tile.is_some(), &mut census)
                        {
                            return None;
                        }
                        if matches!(word, 0xc0000000 | 0x80000000) {
                            return Some(total);
                        }
                        for (dst, part) in total.iter_mut().zip(decode(word)?) {
                            *dst = dst.checked_add(part)?;
                        }
                        Some(total)
                    })
            })
            .ok_or(Error::InvalidConfig("ordered LFM3 down census"))?;
        if counts.iter().sum::<u64>() != census[2] * u64::from(OPS)
            || !self
                .output
                .with_slice(|x| x[..active].iter().all(|&w| (w & 0x7fff) < 0x7f80))
        {
            return Err(Error::InvalidConfig("ordered LFM3 down output"));
        }
        self.output.with_slice(|x| {
            for (out, &bits) in dst.iter_mut().zip(x) {
                *out = bf16::from_bits(bits);
            }
        });
        Ok(([end - start, wait], counts, census))
    }
}
