//! Explicit Apple Silicon backend for the pinned Nanbeige affine Q4 artifact.
//! Immutable packed banks and pipelines are model-owned. Each session owns
//! head-major BF16 K/V, activation scratch and its published token history.

mod dense;
mod dense_basis;
mod kernel;
mod qmm;
mod session;
mod split_k;
mod steel;

use super::{ALLOWANCE, Error, Result, add, memory::Buffer};
use crate::{
    mlx::nanbeige::{
        self as plan, HIDDEN, INTERMEDIATE, KV_WIDTH, LOGIT_ROWS, MAX_CAPACITY, MAX_CHUNK,
        QUERY_WIDTH, WEIGHT_BYTES,
    },
    session::{Memory, Options},
    weights::INITIAL_MEMORY_LIMIT_BYTES,
};
use kernel::Context;
use objc2::rc::autoreleasepool;
use objc2_metal::MTLDevice;
use std::path::Path;

pub use dense::DensePrefillReceipt;
pub use qmm::QmmReceipt;
pub use session::Session;

const SOURCE_POSITION_LIMIT: usize = 262144;

struct Sizes {
    memory: Memory,
    kv_elements: usize,
    pass_elements: usize,
}

impl Sizes {
    fn new(options: Options, diagnostics: bool) -> Result<Self> {
        let Options {
            capacity,
            prefill_chunk_size: chunk,
            position_offset,
            memory_limit_bytes,
        } = options;
        if !(1..=MAX_CAPACITY).contains(&capacity)
            || !(1..=capacity.min(MAX_CHUNK)).contains(&chunk)
            || position_offset
                .checked_add(capacity)
                .is_none_or(|end| end > SOURCE_POSITION_LIMIT)
            || !(1..=INITIAL_MEMORY_LIMIT_BYTES).contains(&memory_limit_bytes)
        {
            return Err(Error::Options(
                "Nanbeige capacity, chunk, position or memory limit differs",
            ));
        }
        let kv = plan::kv_bytes(capacity).map_err(|_| Error::Options("Nanbeige KV extent"))?;
        let kv_elements = kv.elements_per_array * plan::VISITS;
        let pass_elements = chunk * HIDDEN * plan::PASSES;
        // All sizes below are bounded by the fixed geometry and checked context.
        let activation = chunk * (3 * HIDDEN + 2 * QUERY_WIDTH + 2 * KV_WIDTH + 2 * INTERMEDIATE);
        let mut memory = Memory {
            weights_bytes: WEIGHT_BYTES,
            positional_bytes: 0,
            kv_bytes: kv.total_bytes,
            scratch_bytes: 2 * activation
                + 4 * (chunk + 2)
                + if diagnostics { 4 * pass_elements } else { 0 },
            history_bytes: 4 * capacity,
            logits_bytes: 2 * LOGIT_ROWS + if diagnostics { 4 * LOGIT_ROWS } else { 0 },
            metadata_bytes: 0,
            metadata_allowance_bytes: ALLOWANCE,
            admitted_bytes: 0,
        };
        memory.admitted_bytes = [
            memory.weights_bytes,
            memory.kv_bytes,
            memory.scratch_bytes,
            memory.history_bytes,
            memory.logits_bytes,
            memory.metadata_allowance_bytes,
        ]
        .into_iter()
        .try_fold(0, add)?;
        if memory.admitted_bytes > memory_limit_bytes {
            return Err(Error::Budget {
                required: memory.admitted_bytes,
                limit: memory_limit_bytes,
            });
        }
        Ok(Self {
            memory,
            kv_elements,
            pass_elements,
        })
    }

    fn validate_device(&self, context: &Context, largest_weights: usize) -> Result<()> {
        let maximum = context.device.maxBufferLength();
        for length in [
            largest_weights,
            self.kv_elements * 2,
            self.pass_elements * 2,
        ] {
            super::memory::validate_buffer_bytes(length, maximum)?;
        }
        let recommended = context.device.recommendedMaxWorkingSetSize();
        if recommended != 0 && self.memory.admitted_bytes as u64 > recommended {
            return Err(Error::Budget {
                required: self.memory.admitted_bytes,
                limit: usize::try_from(recommended).unwrap_or(usize::MAX),
            });
        }
        Ok(())
    }
}

pub struct Model {
    context: Context,
    weights: Vec<Buffer<u32>>,
    plan: plan::Plan,
    visits: [plan::Visit; plan::VISITS],
    dense: Option<dense::DensePrefill>,
    dense_denial: Option<String>,
    qmm: Option<qmm::Qmm>,
    qmm_denial: Option<String>,
    steel: Option<steel::Steel>,
}

impl Model {
    /// Check requested session geometry and budget before reading tensor data.
    /// Stream the exact pinned artifact into its final owners and authenticate
    /// the complete byte stream before making the model available to a session.
    pub fn load(directory: &Path, options: Options, diagnostics: bool) -> Result<Self> {
        Self::load_inner(directory, options, diagnostics, false)
    }

    /// Explicit experimental selection. Failure leaves the canonical packed model.
    pub fn load_dense(directory: &Path, options: Options, diagnostics: bool) -> Result<Self> {
        Self::load_inner(directory, options, diagnostics, true)
    }

    /// Request the exact Nanbeige Mac QMM image for dense-prefill models.
    /// Optional image or device refusal retains the original fallback.
    pub fn load_dense_qmm(
        directory: &Path,
        options: Options,
        diagnostics: bool,
        metallib: &Path,
    ) -> Result<Self> {
        let mut model = Self::load_dense(directory, options, diagnostics)?;
        if let Some(dense) = &model.dense {
            let sizes = Sizes::new(options, diagnostics)?;
            match autoreleasepool(|_| {
                qmm::Qmm::prepare(
                    &model.context,
                    options,
                    &sizes,
                    dense.receipt.allocated_bytes,
                    (&model.plan, &model.weights),
                    metallib,
                )
            }) {
                Ok(qmm) => model.qmm = Some(qmm),
                Err(error) => model.qmm_denial = Some(error.to_string()),
            }
        } else {
            model.qmm_denial = Some("Nanbeige QMM requires admitted dense prefill".into());
        }
        Ok(model)
    }

    /// Prepare the experimental fixed-geometry prepacked Steel route.
    pub fn load_fixed_steel(
        directory: &Path,
        options: Options,
        qmm_metallib: &Path,
        wheel_metallib: &Path,
    ) -> Result<Self> {
        steel::Steel::validate_options(options)?;
        let mut model = Self::load_dense_qmm(directory, options, false, qmm_metallib)?;
        model.steel = Some(autoreleasepool(|_| {
            steel::Steel::prepare(&model, options, wheel_metallib)
        })?);
        Ok(model)
    }

    pub fn qmm_receipt(&self) -> Option<QmmReceipt> {
        self.qmm.as_ref().map(|qmm| qmm.receipt)
    }

    pub fn qmm_denial(&self) -> Option<&str> {
        self.qmm_denial.as_deref()
    }

    fn load_inner(
        directory: &Path,
        options: Options,
        diagnostics: bool,
        want_dense: bool,
    ) -> Result<Self> {
        let sizes = Sizes::new(options, diagnostics)?;
        let mut source = plan::Source::open(directory)?;
        autoreleasepool(|_| {
            let context = Context::new()?;
            sizes.validate_device(&context, source.largest_bank_bytes())?;
            let mut weights = Vec::new();
            while let Some(bytes) = source.next_bank_bytes() {
                let mut bank = Buffer::on_device(&context.device, bytes / 4)?;
                bank.with_bytes_mut(|destination| source.read_bank(destination))?;
                weights.push(bank);
            }
            let plan = source.finish()?;
            let (dense, dense_denial) = if want_dense {
                match dense::DensePrefill::prepare(&context, &weights, &plan, &sizes, options) {
                    Ok(pairs) => (Some(pairs), None),
                    Err(error) => (None, Some(error.to_string())),
                }
            } else {
                (None, None)
            };
            Ok(Self {
                context,
                weights,
                plan,
                visits: plan::visits(),
                dense,
                dense_denial,
                qmm: None,
                qmm_denial: None,
                steel: None,
            })
        })
    }

    pub fn dense_prefill_receipt(&self) -> Option<DensePrefillReceipt> {
        self.dense.as_ref().map(|dense| dense.receipt)
    }

    pub fn dense_prefill_denial(&self) -> Option<&str> {
        self.dense_denial.as_deref()
    }

    pub fn device_name(&self) -> String {
        self.context.device.name().to_string()
    }

    /// Active prefill row counts eligible for the prepared M32 FFN pipeline.
    pub fn ffn_m32_prefill_rows(&self) -> &'static [usize] {
        if self.context.has_ffn_m32() {
            &[64, 128]
        } else {
            &[]
        }
    }

    /// Active prefill row counts eligible for compact gate and up projections.
    pub fn gate_up_compact_prefill_rows(&self) -> &'static [usize] {
        if self.context.has_gate_up_compact() {
            &[128]
        } else {
            &[]
        }
    }

    /// Active prefill row counts eligible for the prepared K64 Down pipeline.
    pub fn down_k64_prefill_rows(&self) -> &'static [usize] {
        if self.context.has_down_k64() {
            &[128]
        } else {
            &[]
        }
    }
}
