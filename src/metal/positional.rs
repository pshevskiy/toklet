//! Coarse model-owned positional selection; sessions borrow one shared owner.
use super::{
    Error, Result, add, llama3_rotary, memory::Buffer, memory::Gpu, ministral3_rotary,
    profile::Command,
};
use crate::{mlx::qwen::Source, model};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputePipelineState, MTLDevice};
use std::mem::size_of;

const SETUP_LIMIT: usize = 512 * 1024 * 1024;

#[derive(Clone, Copy)]
pub(super) enum Plan {
    Llama3(llama3_rotary::Plan),
    Ministral3 { base: usize, capacity: usize },
}

impl Plan {
    pub(super) fn for_source(source: &Source) -> Result<Option<Self>> {
        if let Some(profile) = source.ministral3_profile() {
            let c = source.config();
            let selected = profile.geometry();
            if (
                c.hidden_size,
                c.intermediate_size,
                c.num_layers,
                c.query_heads,
                c.kv_heads,
                c.head_dim,
                c.vocab_size,
            ) != (
                selected.hidden_size,
                selected.intermediate_size,
                selected.layers,
                selected.query_heads,
                selected.kv_heads,
                selected.head_dim,
                selected.vocab_size,
            ) || !source.tied_output()
            {
                return Err(Error::Options("Ministral 3 selected positional geometry"));
            }
            return Ok(Some(Self::ministral3(c.max_positions)?));
        }
        match source.position_plan() {
            model::PositionPlan::Ordinary => Ok(None),
            model::PositionPlan::Llama3(scaling) => {
                Ok(Some(Self::Llama3(llama3_rotary::Plan::new(
                    source.config().head_dim,
                    source.config().rope_theta,
                    scaling,
                )?)))
            }
        }
    }

    pub(super) fn ministral3(capacity: usize) -> Result<Self> {
        Self::ministral_span(0, capacity)
    }

    fn ministral_span(base: usize, capacity: usize) -> Result<Self> {
        if !(1..=512).contains(&capacity)
            || base
                .checked_add(capacity)
                .is_none_or(|end| end > model::ministral3_rotary::SOURCE_MAX_POSITIONS)
        {
            return Err(Error::Options(
                "Ministral 3 positional context requires1..=512 rows",
            ));
        }
        let plan = Self::Ministral3 { base, capacity };
        if plan.setup_bytes()? > SETUP_LIMIT {
            return Err(Error::Budget {
                required: plan.setup_bytes()?,
                limit: SETUP_LIMIT,
            });
        }
        Ok(plan)
    }

    #[cfg(test)]
    pub(super) fn ministral3_control(base: usize, capacity: usize) -> Result<Self> {
        Self::ministral_span(base, capacity)
    }

    pub(super) fn owned_bytes(self) -> usize {
        match self {
            Self::Llama3(plan) => {
                plan.owned_bytes() - size_of::<llama3_rotary::Operation>() + size_of::<Operation>()
            }
            Self::Ministral3 { capacity, .. } => {
                ministral3_rotary::Tables::planned_bytes(capacity).expect("checked positional span")
                    - size_of::<ministral3_rotary::Tables>()
                    + size_of::<Operation>()
            }
        }
    }

    pub(super) fn pipeline_allowance(self) -> usize {
        match self {
            Self::Llama3(_) => 0,
            Self::Ministral3 { .. } => ministral3_rotary::PIPELINE_ALLOWANCE,
        }
    }

    fn temporary_bytes(self) -> Result<usize> {
        match self {
            Self::Llama3(_) => Ok(size_of::<Compiled>()),
            Self::Ministral3 { capacity, .. } => add(
                model::ministral3_rotary::Rotary::planned_bytes(capacity)
                    .map_err(|_| Error::Options("Ministral 3 CPU table admission"))?,
                size_of::<Compiled>(),
            ),
        }
    }

    pub(super) fn setup_bytes(self) -> Result<usize> {
        add(
            add(self.owned_bytes(), self.pipeline_allowance())?,
            self.temporary_bytes()?,
        )
    }

    pub(super) fn validate_buffers(self, maximum: usize) -> Result<usize> {
        let largest = match self {
            Self::Llama3(plan) => plan.buffer_bytes(),
            Self::Ministral3 { capacity, .. } => {
                let coefficients = (capacity * 128 + ministral3_rotary::GUARD) * 4;
                let gains = (capacity + ministral3_rotary::GUARD) * 2;
                super::memory::validate_buffer_bytes(gains, maximum)?;
                coefficients
            }
        };
        super::memory::validate_buffer_bytes(largest, maximum)?;
        Ok(largest)
    }

    pub(super) fn compile(self, gpu: &Gpu) -> Result<Compiled> {
        match self {
            Self::Llama3(_) => Ok(Compiled::Llama3(llama3_rotary::compile(gpu)?)),
            Self::Ministral3 { .. } => Ok(Compiled::Ministral3(ministral3_rotary::Operation::new(
                gpu,
                self.pipeline_allowance() + size_of::<ministral3_rotary::Operation>(),
            )?)),
        }
    }

    pub(super) fn upload(self, gpu: &Gpu, compiled: Compiled) -> Result<Operation> {
        self.validate_buffers(gpu.device.maxBufferLength())?;
        match (self, compiled) {
            (Self::Llama3(plan), Compiled::Llama3(pipeline)) => Ok(Operation::Llama3(
                llama3_rotary::Operation::with_pipeline(gpu, plan, plan.owned_bytes(), pipeline)?,
            )),
            (Self::Ministral3 { base, capacity }, Compiled::Ministral3(operation)) => {
                let bytes = model::ministral3_rotary::Rotary::planned_bytes(capacity)
                    .map_err(|_| Error::Options("Ministral 3 CPU table admission"))?;
                let source = model::ministral3_rotary::Rotary::new(base, capacity, bytes)
                    .map_err(|_| Error::Options("Ministral 3 CPU table construction"))?;
                let tables = ministral3_rotary::Tables::new(
                    gpu,
                    &source,
                    ministral3_rotary::Tables::planned_bytes(capacity)?,
                )?;
                // CPU coefficients/gains are temporary and leave before learned
                // tensors upload. GPU tables are shared across layers/sessions.
                drop(source);
                Ok(Operation::Ministral3 { operation, tables })
            }
            _ => Err(Error::Options("positional pipeline/plan differs")),
        }
    }
}

pub(super) enum Compiled {
    Llama3(Retained<ProtocolObject<dyn MTLComputePipelineState>>),
    Ministral3(ministral3_rotary::Operation),
}

pub(super) enum Operation {
    Llama3(llama3_rotary::Operation),
    Ministral3 {
        operation: ministral3_rotary::Operation,
        tables: ministral3_rotary::Tables,
    },
}

pub(super) struct Invocation<'a> {
    pub input: &'a Buffer<u16>,
    pub output: &'a Buffer<u16>,
    pub status: &'a Buffer<u32>,
    pub rows: usize,
    pub heads: usize,
    pub offset: usize,
    pub position_limit: usize,
}

impl Operation {
    pub(super) fn is_ministral3(&self) -> bool {
        matches!(self, Self::Ministral3 { .. })
    }

    pub(super) fn owned_bytes(&self) -> usize {
        match self {
            Self::Llama3(operation) => {
                operation.owned_bytes() - size_of::<llama3_rotary::Operation>() + size_of::<Self>()
            }
            Self::Ministral3 { tables, .. } => {
                tables.owned_bytes() - size_of::<ministral3_rotary::Tables>() + size_of::<Self>()
            }
        }
    }

    pub(super) fn pipeline_allowance(&self) -> usize {
        if self.is_ministral3() {
            ministral3_rotary::PIPELINE_ALLOWANCE
        } else {
            0
        }
    }

    pub(super) fn encode(
        &self,
        command: &Command<'_>,
        kind: ministral3_rotary::Kind,
        io: Invocation<'_>,
    ) -> Result<()> {
        match self {
            Self::Llama3(operation) => operation.encode(
                command,
                llama3_rotary::Invocation {
                    input: io.input,
                    output: io.output,
                    status: io.status,
                    rows: io.rows,
                    heads: io.heads,
                    offset: io.offset,
                    position_limit: io.position_limit,
                },
            ),
            Self::Ministral3 { operation, tables } => {
                if io
                    .offset
                    .checked_add(io.rows)
                    .is_none_or(|end| end > io.position_limit)
                {
                    return Err(Error::Options("Ministral 3 rotary exceeds model positions"));
                }
                operation.encode(
                    command,
                    ministral3_rotary::Plan::new(
                        kind,
                        io.rows,
                        io.heads,
                        io.offset,
                        tables.base(),
                        tables.capacity(),
                    )?,
                    ministral3_rotary::Invocation {
                        input: io.input,
                        output: io.output,
                        status: io.status,
                        tables,
                    },
                )
            }
        }
    }
}
