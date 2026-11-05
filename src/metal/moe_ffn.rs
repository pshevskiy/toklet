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
pub(super) struct Plan {
    router: router::Plan,
    experts: experts::Plan,
    geometry: [usize; 5],
    input_stride: usize,
}

impl Plan {
    pub(super) fn expert_count(self) -> usize {
        self.geometry[1]
    }

    pub(super) fn new(
        geometry: [usize; 5],
        score_stride: usize,
        strides: [usize; 5],
    ) -> Result<Self> {
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

    pub(super) fn admit(self, maximum: usize, budget: usize) -> Result<usize> {
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

pub(super) struct Scratch {
    plan: Plan,
    scores: Buffer<u16>,
    ids: Buffer<u32>,
    gates: Buffer<u32>,
    gate: Buffer<u16>,
    up: Buffer<u16>,
    activated: Buffer<u16>,
    values: Buffer<u16>,
}

pub(super) struct LayerIo<'a> {
    pub(super) input: &'a Buffer<u16>,
    pub(super) fused: &'a Buffer<u16>,
    pub(super) down: &'a Buffer<u16>,
    pub(super) delta: &'a Buffer<u16>,
    pub(super) status: &'a Buffer<u32>,
}

pub(super) struct RouterProjection<'a> {
    pub(super) weights: &'a Tensor,
    pub(super) zero: &'a Buffer<u16>,
    pub(super) input_view: &'a MPSMatrix,
}

impl Scratch {
    pub(super) fn new(gpu: &Gpu, plan: Plan, budget: usize) -> Result<Self> {
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

    pub(super) fn project_scores(
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
    pub(super) fn encode(
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

#[cfg(all(test, target_os = "macos"))]
mod real;
#[cfg(test)]
mod tests;

use super::{ALLOWANCE, Model, add, memory::Layout, mlx_geometry};
use crate::{mlx::granite_moe as source, model, session, weights};
use std::{collections::BTreeMap, path::Path, time::Instant};

pub(super) struct LayerWeights {
    pub(super) router: Tensor,
    pub(super) fused: Buffer<u16>,
    pub(super) down: Buffer<u16>,
}

pub(super) struct Graph {
    experts_count: usize,
    selected: usize,
    pub(super) router: router::Operation,
    pub(super) experts: experts::Operation,
    receipt: source::ImportReceipt,
}

impl Graph {
    pub(super) fn pipeline_bytes() -> usize {
        router::PIPELINE_RESERVATION + experts::PIPELINE_RESERVATION + size_of::<Self>()
    }

    pub(super) fn session_plan(
        &self,
        c: &model::Config,
        options: session::Options,
        h: Layout,
    ) -> Result<Plan> {
        Plan::new(
            [
                options.prefill_chunk_size,
                self.experts_count,
                self.selected,
                c.hidden_size,
                c.intermediate_size,
            ],
            self.experts_count,
            [
                h.stride,
                self.selected,
                c.intermediate_size,
                c.hidden_size,
                h.stride,
            ],
        )
    }
}

impl Model {
    /// Load an authenticated Granite MoE artifact into resident Half buffers.
    pub fn load_granite_moe(
        directory: &Path,
        expected: source::ExpectedArtifact<'_>,
        options: weights::LoadOptions,
    ) -> Result<Self> {
        objc2::rc::autoreleasepool(|_| {
            let start = Instant::now();
            if !(1..=512).contains(&options.context_limit) {
                return Err(Error::Options("MoE context must be within 1..=512"));
            }
            let mut source = source::Source::open(directory, expected, options.memory_limit_bytes)?;
            let source_config = source.plan().config().clone();
            let mut config = source_config.dense;
            config.max_positions = config.max_positions.min(options.context_limit);
            mlx_geometry(&config, crate::mlx::dense::Architecture::Granite)?;
            let specs: Vec<_> = source
                .plan()
                .tensors()
                .iter()
                .map(|spec| {
                    let layout = match spec.shape() {
                        [n] => Some(Layout::new(1, *n, true)?),
                        [rows, cols] => Some(Layout::new(*rows, *cols, true)?),
                        [_, _, _] => None,
                        _ => return Err(Error::Options("MoE tensor rank")),
                    };
                    if layout.is_some_and(|l| l.stride != l.columns) {
                        return Err(Error::Unsupported("MoE import requires tight Half rows"));
                    }
                    let count = spec.elements();
                    let elements = match layout {
                        Some(l) => l.elements,
                        None => add(count, 7)?,
                    };
                    Ok((spec.name().to_owned(), layout, count, elements))
                })
                .collect::<Result<_>>()?;
            let weight_bytes = specs.iter().try_fold(0, |n, (_, _, _, elements)| {
                add(
                    n,
                    elements
                        .checked_mul(2)
                        .ok_or(Error::Options("MoE tensor bytes"))?,
                )
            })?;
            let storage = source.plan().storage();
            let layer_bytes = config
                .num_layers
                .checked_mul(size_of::<super::Layer>())
                .ok_or(Error::Options("MoE layer metadata overflow"))?;
            let required = [
                storage.admitted_load_bytes,
                weight_bytes
                    .checked_sub(storage.half_weights_bytes)
                    .ok_or(Error::Options("MoE weight accounting"))?,
                layer_bytes,
                Graph::pipeline_bytes(),
                ALLOWANCE,
            ]
            .into_iter()
            .try_fold(0, add)?;
            if required > options.memory_limit_bytes {
                return Err(Error::Budget {
                    required,
                    limit: options.memory_limit_bytes,
                });
            }
            let compilation = Instant::now();
            let gpu = Gpu::new()?;
            for (_, _, _, elements) in &specs {
                validate_buffer_bytes(elements * 2, gpu.device.maxBufferLength())?;
            }
            let router = router::Operation::new(
                &gpu,
                router::PIPELINE_RESERVATION + size_of::<router::Operation>(),
            )?;
            let experts = experts::Operation::new(
                &gpu,
                experts::PIPELINE_RESERVATION + size_of::<experts::Operation>(),
            )?;
            let compilation_seconds = compilation.elapsed().as_secs_f64();
            let conversion = Instant::now();
            let mut tensors = BTreeMap::new();
            let mut expert_buffers = BTreeMap::new();
            for (name, layout, count, elements) in specs {
                if let Some(layout) = layout {
                    let mut tensor = Tensor::allocate(&gpu, layout)?;
                    tensor
                        .buffer
                        .with_mut(|dst| source.import_next(&mut dst[..count]))?;
                    tensors.insert(name, tensor);
                } else {
                    let mut buffer = Buffer::new(&gpu, elements)?;
                    buffer.with_mut(|dst| source.import_next(&mut dst[..count]))?;
                    expert_buffers.insert(name, buffer);
                }
            }
            let receipt = source.finish()?;
            let conversion_seconds = conversion.elapsed().as_secs_f64();
            let mut layers = Vec::with_capacity(config.num_layers);
            for layer in 0..config.num_layers {
                let prefix = format!("model.layers.{layer}.block_sparse_moe");
                layers.push(LayerWeights {
                    router: tensors
                        .remove(&format!("{prefix}.router.layer.weight"))
                        .ok_or(Error::Options("MoE router absent"))?,
                    fused: expert_buffers
                        .remove(&format!("{prefix}.input_linear.weight"))
                        .ok_or(Error::Options("MoE gate/up absent"))?,
                    down: expert_buffers
                        .remove(&format!("{prefix}.output_linear.weight"))
                        .ok_or(Error::Options("MoE down absent"))?,
                });
            }
            if !expert_buffers.is_empty() {
                return Err(Error::Options("unused MoE expert weights"));
            }
            let mut model = Self::assemble_with_feed_forward(
                config,
                gpu,
                tensors,
                weight_bytes,
                true,
                false,
                false,
                Some(layers.into_iter()),
            )?;
            model.scales = source_config.scales;
            model.half_embedding = (model.scales.embedding != 1.)
                .then(|| half::f16::from_f32(model.scales.embedding).to_bits());
            model.half_residual = (model.scales.residual != 1.)
                .then(|| half::f16::from_f32(model.scales.residual).to_bits());
            model.moe = Some(Graph {
                experts_count: source_config.experts,
                selected: source_config.selected,
                router,
                experts,
                receipt,
            });
            model.load_timing = super::LoadTiming {
                load_seconds: start.elapsed().as_secs_f64(),
                conversion_seconds,
                compilation_seconds,
            };
            Ok(model)
        })
    }

    pub fn granite_moe_import_receipt(&self) -> Option<&source::ImportReceipt> {
        self.moe.as_ref().map(|graph| &graph.receipt)
    }
}
