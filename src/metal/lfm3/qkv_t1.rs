//! Joint private one-command T1 projection using the released Q/K/V planes.
use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Default)]
struct T1Census {
    submitted: u64,
    validated: u64,
    native: [u64; 3],
    ordered: [u64; 3],
    times: [f64; 2],
}

pub(super) struct T1 {
    kernels: Kernels,
    by_layer: [T1Census; L],
}

impl Workspace {
    pub(in crate::metal::lfm3) fn prepare_t1(&mut self, gpu: &Gpu) -> bool {
        if let Some(kernels) = Kernels::with_shader(gpu, include_str!("qkv_t1_fma4.metal")) {
            self.t1 = Some(T1 {
                kernels,
                by_layer: [T1Census::default(); L],
            });
            true
        } else {
            false
        }
    }

    pub(in crate::metal::lfm3) fn has_t1(&self) -> bool {
        self.t1.is_some()
    }

    pub(in crate::metal::lfm3) fn run_t1(
        &mut self,
        gpu: &Gpu,
        layer: usize,
        banks: [BankRef<'_>; 3],
        rhs: &[bf16],
        out: [&mut [bf16]; 3],
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        let slot = Self::slot(layer)?;
        if rhs.len() != I
            || banks.map(BankRef::identity) != self.owners[slot]
            || banks.iter().zip(out.iter()).zip(N).any(|((b, y), n)| {
                b.identity() == 0
                    || !b.device_matches(&gpu.device)
                    || b.words() != I * n
                    || y.len() != n
            })
        {
            return Err(Error::InvalidConfig("LFM3 T1 QKV owner or geometry"));
        }
        let t1 = self
            .t1
            .as_mut()
            .ok_or(Error::InvalidConfig("LFM3 T1 QKV pipeline absent"))?;
        if cancel.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        let tag = self.input.with_mut(|dst| {
            domain_tag(rhs.iter().zip(dst).map(|(x, y)| {
                *y = x.to_bits();
                *y
            }))
        });
        self.tags.with_mut(|tags| tags[0] = tag);
        for (plane, n) in self.planes.iter_mut().zip(N) {
            plane.output.with_mut(|x| x[..=n].fill(SENTINEL));
            plane.routes.with_mut(|x| x[..=n].fill(ROUTE_GUARD));
        }
        let status = &mut self.planes[0].status;
        status.with_mut(|x| x.copy_from_slice(&[0, STATUS_GUARD]));
        let status = &self.planes[0].status;
        let (dims, admission_groups) = Kernels::sealed_admission(banks[0], 1, I, N[0]);
        let command = gpu
            .command()
            .map_err(|_| Error::InvalidConfig("LFM3 T1 QKV command"))?;
        for (pipeline, is_projection, groups) in [
            (&t1.kernels.admission, false, admission_groups),
            (&t1.kernels.projection, true, N.iter().sum::<usize>() / 4),
        ] {
            let e = command
                .computeCommandEncoder()
                .ok_or(Error::InvalidConfig("LFM3 T1 QKV encoder"))?;
            e.setComputePipelineState(pipeline);
            // SAFETY: validated same-device sealed banks, tracked live buffers.
            // The three private output and receipt extents are disjoint.
            unsafe {
                e.setBuffer_offset_atIndex(Some(&self.input.raw), 0, 0);
                if is_projection {
                    for (role, (plane, bank)) in self.planes.iter().zip(banks).enumerate() {
                        bank.bind_read_only(&e, 1 + role);
                        e.setBuffer_offset_atIndex(Some(&plane.output.raw), 0, 5 + role);
                        e.setBuffer_offset_atIndex(Some(&plane.routes.raw), 0, 8 + role);
                        e.setBuffer_offset_atIndex(
                            Some(&plane.weight_tags.raw),
                            slot * N[role] * size_of::<u32>(),
                            12 + role,
                        );
                    }
                    e.setBuffer_offset_atIndex(Some(&status.raw), 0, 4);
                    e.setBuffer_offset_atIndex(Some(&self.tags.raw), 0, 11);
                } else {
                    banks[0].bind_read_only(&e, 1);
                    e.setBuffer_offset_atIndex(Some(&status.raw), 0, 2);
                    e.setBytes_length_atIndex(
                        NonNull::from(&dims).cast(),
                        size_of::<[u32; 4]>(),
                        3,
                    );
                }
            }
            e.dispatchThreadgroups_threadsPerThreadgroup(
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
            e.endEncoding();
        }
        let count = &mut t1.by_layer[slot];
        let submitted = count
            .submitted
            .checked_add(1)
            .ok_or(Error::InvalidConfig("LFM3 T1 QKV submitted overflow"))?;
        if cancel.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        let clock = Instant::now();
        command.commit();
        count.submitted = submitted;
        command.waitUntilCompleted();
        let wait = clock.elapsed().as_secs_f64();
        let (a, b) = (command.GPUStartTime(), command.GPUEndTime());
        if command.status() != MTLCommandBufferStatus::Completed
            || command.error().is_some()
            || !a.is_finite()
            || !b.is_finite()
            || a <= 0.
            || b <= a
        {
            return Err(Error::InvalidConfig("LFM3 T1 QKV completion"));
        }
        count.times[0] += b - a;
        count.times[1] += wait;
        let (code, guard) = status.with_slice(|x| (x[0], x[1]));
        let mut updated = *count;
        let add = |a: u64, b: u64| {
            a.checked_add(b)
                .ok_or(Error::InvalidConfig("LFM3 T1 QKV census overflow"))
        };
        for (role, (plane, n)) in self.planes.iter().zip(N).enumerate() {
            let mut native = 0;
            let routes = plane.routes.with_slice(|words| {
                words[n] == ROUTE_GUARD
                    && (code != 2 || words[..n].iter().all(|&r| r == ROUTE_GUARD))
                    && (code != 0
                        || words[..n].iter().all(|&r| {
                            if r == 0x80000000 {
                                native += 1;
                                true
                            } else {
                                (r & 4095) + (r >> 12) <= 2080
                            }
                        }))
            });
            let values = plane.output.with_slice(|words| {
                words[n] == SENTINEL
                    && (code != 2 || words[..n].iter().all(|&w| w == SENTINEL))
                    && (code != 0 || words[..n].iter().all(|&w| w & 0x7fff < 0x7f80))
            });
            if guard != STATUS_GUARD || !routes || !values {
                return Err(Error::InvalidConfig("LFM3 T1 QKV guards"));
            }
            if code == 0 {
                updated.native[role] = add(updated.native[role], native)?;
                updated.ordered[role] = add(updated.ordered[role], n as u64 - native)?;
            }
        }
        match code {
            0 => (),
            2 => return Err(Error::InvalidConfig("LFM3 T1 QKV nonfinite input")),
            4 => return Err(Error::InvalidConfig("LFM3 T1 QKV overflow")),
            _ => return Err(Error::InvalidConfig("LFM3 T1 QKV status")),
        }
        updated.validated = add(updated.validated, 1)?;
        *count = updated;
        if cancel.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        // No fallible work between the first private plane copy and the last.
        // At T1, V's token-major row already is its head-major proposal.
        for (plane, dst) in self.planes.iter().zip(out) {
            plane.output.with_slice(|words| {
                for (dst, &bits) in dst.iter_mut().zip(words) {
                    *dst = bf16::from_bits(bits);
                }
            });
        }
        Ok(())
    }

    pub(in crate::metal::lfm3) fn t1_report(&self) -> Option<serde_json::Value> {
        self.t1.as_ref().map(|owner| {
            serde_json::json!({
                "placement":"metal-t1-fma4",
                "layers":owner.by_layer.iter().enumerate().map(|(slot,c)|serde_json::json!({
                    "layer":2*(slot+1),"submitted_sets":c.submitted,"validated_sets":c.validated,
                    "native_rows_by_role":c.native,"ordered_rows_by_role":c.ordered,
                    "completed_gpu_seconds":c.times[0],"commit_wait_seconds":c.times[1]
                })).collect::<Vec<_>>()
            })
        })
    }
}
