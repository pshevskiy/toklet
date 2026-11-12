//! Standalone borrowed Half rotary for the selected Ministral 3 profile.
//!
//! Immutable coefficients and Half query gains upload once from the approved
//! CPU cache. Encode performs no trig, upload, allocation, wait or
//! status reset. The caller owns completion and discards failed output. This
//! diagnostic component does not admit a source/model or extend ordinary Metal
//! context limits. Rust encode allocation qualification excludes driver work.

use super::{
    Error, Result,
    memory::{Buffer, Gpu, validate_buffer_bytes},
    profile::Command,
};
use crate::model::ministral3_rotary::{HEAD_DIM, Rotary, SOURCE_MAX_POSITIONS};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::{mem::size_of, ptr::NonNull};

pub(super) const GUARD: usize = 7;
pub(super) const SENTINEL: u32 = 0x5a17_0031;
pub(super) const HALF_SENTINEL: u16 = 0x5a17;
pub(super) const PIPELINE_ALLOWANCE: usize = 64 * 1024 * 1024;
const CAPACITY_LIMIT: usize = 512;
const HEADS_LIMIT: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Keys,
    Queries,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Plan {
    kind: Kind,
    tokens: usize,
    heads: usize,
    table_base: usize,
    table_capacity: usize,
    table_offset: usize,
    values: usize,
    operand_len: usize,
}

fn table_span(base: usize, capacity: usize) -> Result<usize> {
    if !(1..=CAPACITY_LIMIT).contains(&capacity) {
        return Err(Error::Options("Ministral 3 rotary table capacity"));
    }
    base.checked_add(capacity)
        .filter(|&end| end <= SOURCE_MAX_POSITIONS)
        .ok_or(Error::Options("Ministral 3 rotary source span"))
}

fn table_lengths(capacity: usize) -> Result<[usize; 2]> {
    table_span(0, capacity)?;
    let coefficients = capacity
        .checked_mul(HEAD_DIM)
        .and_then(|n| n.checked_add(GUARD))
        .filter(|&n| n <= u32::MAX as usize && n <= isize::MAX as usize / 4)
        .ok_or(Error::Options("Ministral 3 rotary coefficient extent"))?;
    let gains = capacity
        .checked_add(GUARD)
        .filter(|&n| n <= u32::MAX as usize && n <= isize::MAX as usize / 2)
        .ok_or(Error::Options("Ministral 3 rotary gain extent"))?;
    Ok([coefficients, gains])
}

impl Plan {
    /// Checked tight `[tokens,heads,128]` Half operands and one cached span.
    /// Diagnostic absolute positions may be high; at most512 table rows exist,
    /// and no prefix before the table base is allocated or admitted as a model.
    pub(super) fn new(
        kind: Kind,
        tokens: usize,
        heads: usize,
        absolute_offset: usize,
        table_base: usize,
        table_capacity: usize,
    ) -> Result<Self> {
        let table_end = table_span(table_base, table_capacity)?;
        if !(1..=CAPACITY_LIMIT).contains(&tokens) || !(1..=HEADS_LIMIT).contains(&heads) {
            return Err(Error::Options("Ministral 3 rotary operand geometry"));
        }
        let table_offset = absolute_offset
            .checked_sub(table_base)
            .filter(|_| {
                absolute_offset
                    .checked_add(tokens)
                    .is_some_and(|end| end <= table_end)
            })
            .ok_or(Error::Options("Ministral 3 rotary invocation span"))?;
        let values = tokens
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(HEAD_DIM))
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or(Error::Options("Ministral 3 rotary UInt32 value extent"))?;
        let operand_len = values
            .checked_add(GUARD)
            .filter(|&n| n <= u32::MAX as usize && n <= isize::MAX as usize / 2)
            .ok_or(Error::Options("Ministral 3 rotary guarded value extent"))?;
        Ok(Self {
            kind,
            tokens,
            heads,
            table_base,
            table_capacity,
            table_offset,
            values,
            operand_len,
        })
    }

    pub(super) fn input_len(self) -> usize {
        self.operand_len
    }

    pub(super) fn output_len(self) -> usize {
        self.operand_len
    }

    pub(super) fn status_len(self) -> usize {
        1 + GUARD
    }

    /// Guarded caller input/output/status allocations. Immutable tables,
    /// pipeline/context allowance, temporary CPU upload and diagnostics are
    /// counted separately; each physical owner is counted once.
    pub(super) fn operand_bytes(self) -> Result<usize> {
        self.input_len()
            .checked_add(self.output_len())
            .and_then(|n| n.checked_mul(size_of::<u16>()))
            .and_then(|bytes| {
                self.status_len()
                    .checked_mul(size_of::<u32>())
                    .and_then(|status| bytes.checked_add(status))
            })
            .ok_or(Error::Options("Ministral 3 rotary operand byte overflow"))
    }
}

/// Resident immutable FP32 `[capacity,64,{cos,sin}]` words and Half `[capacity]`
/// gains, each followed by guards. Borrowed CPU cache storage is not retained.
/// No coefficient expansion across heads or duplicate trig calculation occurs.
pub(super) struct Tables {
    coefficients: Buffer<u32>,
    gains: Buffer<u16>,
    base: usize,
    capacity: usize,
}

impl Tables {
    /// GPU coefficient/gain bytes, guards and inline descriptors/metadata.
    /// The supplied CPU cache remains a separately accounted temporary/owner.
    pub(super) fn planned_bytes(capacity: usize) -> Result<usize> {
        let [coefficients, gains] = table_lengths(capacity)?;
        coefficients
            .checked_mul(size_of::<u32>())
            .and_then(|n| {
                gains
                    .checked_mul(size_of::<u16>())
                    .and_then(|m| n.checked_add(m))
            })
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .ok_or(Error::Options("Ministral 3 rotary table byte overflow"))
    }

    pub(super) fn new(gpu: &Gpu, source: &Rotary, available: usize) -> Result<Self> {
        let (base, capacity) = (source.base(), source.capacity());
        table_span(base, capacity)?;
        let required = Self::planned_bytes(capacity)?;
        if required > available {
            return Err(Error::Budget {
                required,
                limit: available,
            });
        }
        let [coefficient_len, gain_len] = table_lengths(capacity)?;
        validate_buffer_bytes(
            coefficient_len * size_of::<u32>(),
            gpu.device.maxBufferLength(),
        )?;
        validate_buffer_bytes(gain_len * size_of::<u16>(), gpu.device.maxBufferLength())?;
        let mut coefficients = Buffer::<u32>::new(gpu, coefficient_len)?;
        let mut gains = Buffer::<u16>::new(gpu, gain_len)?;
        coefficients.with_mut(|out| -> Result<()> {
            out.fill(SENTINEL);
            for row in 0..capacity {
                let values = source
                    .coefficient_row(base + row)
                    .ok_or(Error::Options("Ministral 3 rotary CPU coefficient span"))?;
                for (pair, &[cosine, sine]) in values.iter().enumerate() {
                    let index = row * HEAD_DIM + 2 * pair;
                    out[index] = cosine.to_bits();
                    out[index + 1] = sine.to_bits();
                }
            }
            Ok(())
        })?;
        gains.with_mut(|out| -> Result<()> {
            out.fill(HALF_SENTINEL);
            for (row, value) in out[..capacity].iter_mut().enumerate() {
                *value = source
                    .query_scale(base + row)
                    .ok_or(Error::Options("Ministral 3 rotary CPU gain span"))?
                    .to_bits();
            }
            Ok(())
        })?;
        let tables = Self {
            coefficients,
            gains,
            base,
            capacity,
        };
        if tables.owned_bytes() > available {
            return Err(Error::Budget {
                required: tables.owned_bytes(),
                limit: available,
            });
        }
        Ok(tables)
    }

    pub(super) fn base(&self) -> usize {
        self.base
    }

    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(super) fn owned_bytes(&self) -> usize {
        // Constructor bounds both exact-sized allocations to512 cached rows.
        self.coefficients.raw.length() + self.gains.raw.length() + size_of::<Self>()
    }
}

pub(super) struct Operation {
    pipelines: [Retained<ProtocolObject<dyn MTLComputePipelineState>>; 2],
    widths: [usize; 2],
}

impl Operation {
    /// Pipeline/context allowance is policy, not measured physical residency.
    pub(super) fn new(gpu: &Gpu, available: usize) -> Result<Self> {
        let required = PIPELINE_ALLOWANCE + size_of::<Self>();
        if required > available {
            return Err(Error::Budget {
                required,
                limit: available,
            });
        }
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version2_3);
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        if options.languageVersion() != MTLLanguageVersion::Version2_3
            || options.mathMode() != MTLMathMode::Safe
            || options.mathFloatingPointFunctions() != MTLMathFloatingPointFunctions::Precise
        {
            return Err(Error::Unsupported(
                "Ministral 3 rotary precise math unavailable",
            ));
        }
        let library = gpu
            .device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("ministral3_rotary.metal")),
                Some(&options),
            )
            .map_err(|error| Error::Native(error.to_string()))?;
        let compile = |name: &str| -> Result<_> {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| Error::Native("Ministral 3 rotary kernel absent".into()))?;
            let pipeline = gpu
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| Error::Native(error.to_string()))?;
            let width = pipeline.threadExecutionWidth();
            if width == 0
                || width > pipeline.maxTotalThreadsPerThreadgroup()
                || width > u32::MAX as usize
            {
                return Err(Error::Unsupported(
                    "Ministral 3 rotary launch width unavailable",
                ));
            }
            Ok((pipeline, width))
        };
        let (keys, key_width) = compile("ministral3_rotary_keys")?;
        let (queries, query_width) = compile("ministral3_rotary_queries")?;
        Ok(Self {
            pipelines: [keys, queries],
            widths: [key_width, query_width],
        })
    }

    /// Encode borrowed same-device disjoint buffers into a retained, uncommitted
    /// command. All sizes, cached spans, byte extents, devices and aliases are
    /// checked before encoder creation. No input CPU reads occur here; device
    /// nonfinite/store overflow sets monotone status and permits partial writes.
    /// Caller publishes output only after checked completion and zero status,
    /// excludes CPU access/reuse while pending, and owns status initialization.
    pub(super) fn encode(
        &self,
        command: &Command<'_>,
        plan: Plan,
        io: Invocation<'_>,
    ) -> Result<()> {
        if io.tables.base() != plan.table_base || io.tables.capacity() != plan.table_capacity {
            return Err(Error::Options("Ministral 3 rotary plan/table span differs"));
        }
        let [coefficient_len, gain_len] = table_lengths(plan.table_capacity)?;
        if io.input.len < plan.input_len()
            || io.output.len < plan.output_len()
            || io.status.len < plan.status_len()
            || io.tables.coefficients.len < coefficient_len
            || io.tables.gains.len < gain_len
            || io.input.raw.length() < plan.input_len() * size_of::<u16>()
            || io.output.raw.length() < plan.output_len() * size_of::<u16>()
            || io.status.raw.length() < plan.status_len() * size_of::<u32>()
            || io.tables.coefficients.raw.length() < coefficient_len * size_of::<u32>()
            || io.tables.gains.raw.length() < gain_len * size_of::<u16>()
        {
            return Err(Error::Options("Ministral 3 rotary buffer extent"));
        }
        let buffers = [
            &*io.input.raw,
            &*io.output.raw,
            &*io.status.raw,
            &*io.tables.coefficients.raw,
            &*io.tables.gains.raw,
        ];
        for (index, buffer) in buffers.iter().enumerate() {
            if buffers[..index]
                .iter()
                .any(|other| std::ptr::eq(*buffer, *other))
            {
                return Err(Error::Options("Ministral 3 rotary buffer alias"));
            }
        }
        let (variant, label) = match plan.kind {
            Kind::Keys => (0, "ministral3_rotary_keys"),
            Kind::Queries => (1, "ministral3_rotary_queries"),
        };
        let pipeline = &self.pipelines[variant];
        let width = self.widths[variant];
        let device = pipeline.device();
        if !std::ptr::eq(&*command.device(), &*device)
            || buffers
                .iter()
                .any(|buffer| !std::ptr::eq(&*buffer.device(), &*device))
        {
            return Err(Error::Options("Ministral 3 rotary device identity"));
        }
        if !command.retainedReferences() || command.status() != MTLCommandBufferStatus::NotEnqueued
        {
            return Err(Error::Unsupported(
                "Ministral 3 rotary command retention/state",
            ));
        }
        let groups = (plan.values / 2).div_ceil(width);
        if groups
            .checked_mul(width)
            .is_none_or(|lanes| lanes > u32::MAX as usize)
        {
            return Err(Error::Options("Ministral 3 rotary rounded dispatch"));
        }
        let (index, encoder) = if let Some(trace) = command.trace {
            let (index, encoder) = trace.encoder(command.raw, label)?;
            (Some(index), encoder)
        } else {
            (
                None,
                command
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::Native("Ministral 3 rotary encoder absent".into()))?,
            )
        };
        encoder.setComputePipelineState(pipeline);
        let shape = [
            plan.tokens as u32,
            plan.heads as u32,
            plan.table_offset as u32,
        ];
        // SAFETY: closed private ABI; initialized u16/u32 storage supplies native
        // Half/FP32 alignment. Same-device resources are pairwise disjoint; every
        // UInt32 product, cached row, byte extent and rounded lane is bounded.
        // Stack shape copies now. Retained command resources live to completion;
        // caller excludes CPU access/reuse and controls status/result publication.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&io.input.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&io.tables.coefficients.raw), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&io.tables.gains.raw), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&io.output.raw), 0, 3);
            encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), size_of::<[u32; 3]>(), 4);
            encoder.setBuffer_offset_atIndex(Some(&io.status.raw), 0, 5);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        if let (Some(trace), Some(index)) = (command.trace, index) {
            trace.finish_encoder(index);
        }
        Ok(())
    }
}

pub(super) struct Invocation<'a> {
    pub input: &'a Buffer<u16>,
    pub output: &'a Buffer<u16>,
    pub status: &'a Buffer<u32>,
    pub tables: &'a Tables,
}

#[cfg(test)]
mod tests;
