//! Shared resident scratch for router and expert stages.
use super::{
    Error, Result,
    memory::{Buffer, Gpu, validate_buffer_bytes},
    moe_experts_half as experts, moe_router_half as router,
    profile::Command,
};
use objc2_metal::MTLDevice;
use std::mem::size_of;

#[derive(Clone, Copy)]
struct Plan {
    router: router::Plan,
    experts: experts::Plan,
}

impl Plan {
    fn new(geometry: [usize; 5], score_stride: usize, strides: [usize; 5]) -> Result<Self> {
        let [capacity, experts, selected, ..] = geometry;
        let router = router::Plan::new(capacity, experts, selected, score_stride, strides[1])?;
        let experts = experts::Plan::new(geometry, strides)?;
        if [router.output_len; 2] != experts.word[..2] {
            return Err(Error::Options("MoE shared route extent differs"));
        }
        Ok(Self { router, experts })
    }

    fn admit(self, maximum: usize, budget: usize) -> Result<usize> {
        let bytes = [
            self.router.input_len * 2,
            self.router.output_len * 4,
            self.router.output_len * 4,
        ]
        .into_iter()
        .chain(self.experts.half[3..7].iter().map(|n| n * 2));
        let mut required = size_of::<Scratch>();
        for length in bytes {
            validate_buffer_bytes(length, maximum)?;
            required = required
                .checked_add(length)
                .ok_or(Error::Options("MoE shared scratch byte overflow"))?;
        }
        if required > budget {
            return Err(Error::Budget {
                required,
                limit: budget,
            });
        }
        Ok(required)
    }
}

struct Scratch {
    plan: Plan,
    scores: Buffer<u16>,
    ids: Buffer<u32>,
    gates: Buffer<u32>,
    gate: Buffer<u16>,
    up: Buffer<u16>,
    activated: Buffer<u16>,
    values: Buffer<u16>,
}

struct LayerIo<'a> {
    input: &'a Buffer<u16>,
    fused: &'a Buffer<u16>,
    down: &'a Buffer<u16>,
    delta: &'a Buffer<u16>,
    status: &'a Buffer<u32>,
}

impl Scratch {
    fn new(gpu: &Gpu, plan: Plan, budget: usize) -> Result<Self> {
        plan.admit(gpu.device.maxBufferLength(), budget)?;
        Ok(Self {
            plan,
            scores: Buffer::new(gpu, plan.router.input_len)?,
            ids: Buffer::new(gpu, plan.router.output_len)?,
            gates: Buffer::new(gpu, plan.router.output_len)?,
            gate: Buffer::new(gpu, plan.experts.half[3])?,
            up: Buffer::new(gpu, plan.experts.half[4])?,
            activated: Buffer::new(gpu, plan.experts.half[5])?,
            values: Buffer::new(gpu, plan.experts.half[6])?,
        })
    }

    /// Scores are produced by the layer's router projection. The caller
    /// discards the command on encode failure and the candidate on GPU error.
    fn encode(
        &self,
        router: &router::Operation,
        experts: &experts::Operation,
        command: &Command<'_>,
        rows: usize,
        io: LayerIo<'_>,
    ) -> Result<()> {
        router.encode(
            command,
            self.plan.router,
            rows,
            router::Invocation {
                input: &self.scores,
                ids: &self.ids,
                gates: &self.gates,
                status: io.status,
            },
        )?;
        experts.encode(
            command,
            self.plan.experts,
            rows,
            experts::Invocation {
                half: [
                    io.input,
                    io.fused,
                    io.down,
                    &self.gate,
                    &self.up,
                    &self.activated,
                    &self.values,
                    io.delta,
                ],
                word: [&self.ids, &self.gates, io.status],
            },
        )
    }
}

#[cfg(test)]
mod tests;
