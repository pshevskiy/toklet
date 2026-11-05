//! Shared resident scratch for router and expert stages.
use super::{
    Error, Kernel, Result, Tensor, encode_half_compensated,
    memory::{Buffer, Gpu, validate_buffer_bytes},
    moe_experts_half as experts, moe_router_half as router,
    profile::Command,
};
use objc2_metal::{MTLCommandBuffer, MTLDevice, MTLResource};
use objc2_metal_performance_shaders::MPSMatrix;
use std::mem::size_of;

#[derive(Clone, Copy)]
struct Plan {
    router: router::Plan,
    experts: experts::Plan,
    geometry: [usize; 5],
    input_stride: usize,
}

impl Plan {
    fn new(geometry: [usize; 5], score_stride: usize, strides: [usize; 5]) -> Result<Self> {
        let [capacity, experts, selected, ..] = geometry;
        let router = router::Plan::new(capacity, experts, selected, score_stride, strides[1])?;
        let experts = experts::Plan::new(geometry, strides)?;
        if [router.output_len; 2] != experts.word[..2] {
            return Err(Error::Options("MoE shared route extent differs"));
        }
        Ok(Self {
            router,
            experts,
            geometry,
            input_stride: strides[0],
        })
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

struct RouterProjection<'a> {
    weights: &'a Tensor,
    zero: &'a Buffer<u16>,
    input_view: &'a MPSMatrix,
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

    fn project_scores(
        &self,
        gpu: &Gpu,
        command: &Command<'_>,
        rows: usize,
        io: &LayerIo<'_>,
        projection: RouterProjection<'_>,
    ) -> Result<()> {
        let [capacity, experts, _, hidden, _] = self.plan.geometry;
        if rows == 0
            || rows > capacity
            || projection.weights.layout.rows != experts
            || projection.weights.layout.columns != hidden
            || io.status.len < self.plan.experts.word[2]
        {
            return Err(Error::Options("MoE router projection extent"));
        }
        // SAFETY: descriptor getters inspect metadata on the retained view.
        let matches_view = unsafe {
            projection.input_view.offset() == 0
                && projection.input_view.rowBytes() == self.plan.input_stride * 2
                && std::ptr::eq(&*projection.input_view.data(), &*io.input.raw)
        };
        if !matches_view {
            return Err(Error::Options("MoE router projection input view differs"));
        }
        let buffers = [
            &*projection.weights.buffer.raw,
            &*projection.zero.raw,
            &*io.input.raw,
            &*self.scores.raw,
            &*io.status.raw,
        ];
        if !command.retainedReferences()
            || !std::ptr::eq(&*command.device(), &*gpu.device)
            || buffers
                .iter()
                .any(|buffer| !std::ptr::eq(&*buffer.device(), &*gpu.device))
        {
            return Err(Error::Options("MoE router projection resource ownership"));
        }
        for (index, buffer) in buffers.iter().enumerate() {
            if buffers[..index]
                .iter()
                .any(|other| std::ptr::eq(*buffer, *other))
            {
                return Err(Error::Options("MoE router projection operand alias"));
            }
        }
        encode_half_compensated(
            gpu,
            command,
            projection.weights,
            projection.zero,
            projection.input_view,
            &self.scores.raw,
            self.plan.router.input_stride,
            rows,
            Kernel::HalfCompensated,
            io.status,
        )
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
