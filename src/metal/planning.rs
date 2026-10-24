//! Explicit cold Metal resource planning. Constructors consume these same plans.
//! Numerical weight/KV/scratch buffers are not allocated by inspection; device,
//! library and pipeline initialization still allocates driver-owned resources.
use super::{
    ALLOWANCE, AttentionMode, Cache, Error, Kernel, NormalizationMode, ReadbackMode, Result, Views,
    add,
    memory::{self, Gpu, Layout},
    mlx,
};
use crate::{
    mlx::{
        dense::Architecture,
        qwen::{Source, Storage},
    },
    model, session, weights,
};
use objc2::rc::autoreleasepool;
use objc2_metal::MTLDevice;
use std::{collections::BTreeMap, path::Path};

pub(super) struct MlxWeights {
    pub tensors: BTreeMap<String, (Layout, Storage)>,
    pub weight_bytes: usize,
    pub admitted: usize,
}
impl MlxWeights {
    pub fn new(source: &Source) -> Result<Self> {
        let mut plans = BTreeMap::new();
        let mut weight_bytes = 0;
        for (name, spec) in source.tensors() {
            let (rows, columns) = match spec.shape() {
                [n] => (1, *n),
                [r, c] => (*r, *c),
                _ => return Err(Error::Options("MLX tensor rank differs")),
            };
            let layout = Layout::new(rows, columns, true)?;
            let bytes = match spec.storage() {
                Storage::Float(_) => layout.bytes(2)?,
                Storage::Affine(p) => mlx::packed_bytes(p)?,
            };
            weight_bytes = add(weight_bytes, bytes)?;
            plans.insert(name.clone(), (layout, spec.storage()));
        }
        let admitted = add(add(weight_bytes, ALLOWANCE)?, 64 * 1024)?;
        Ok(Self {
            tensors: plans,
            weight_bytes,
            admitted,
        })
    }
    pub fn admit(&self, limit: usize) -> Result<()> {
        admit(self.admitted, limit)
    }
    pub fn validate_buffers(&self, maximum: usize) -> Result<()> {
        for (layout, storage) in self.tensors.values() {
            match storage {
                Storage::Float(_) => memory::validate_buffer_bytes(layout.bytes(2)?, maximum)?,
                Storage::Affine(p) => {
                    mlx::validate_resident_layout(*p)?;
                    for bytes in [p.weight_bytes(), p.metadata_bytes().div_ceil(4) * 4] {
                        memory::validate_buffer_bytes(bytes, maximum)?;
                    }
                }
            }
        }
        Ok(())
    }
}
fn admit(required: usize, limit: usize) -> Result<()> {
    if required > limit {
        Err(Error::Budget { required, limit })
    } else {
        Ok(())
    }
}
pub(super) fn validate_attention(
    architecture: Option<Architecture>,
    mode: AttentionMode,
) -> Result<()> {
    if mode == AttentionMode::Online
        && matches!(
            architecture,
            Some(Architecture::Llama | Architecture::Mistral | Architecture::Qwen3)
        )
    {
        return Err(Error::Unsupported(
            "Llama/Mistral/Qwen3 Online attention is not qualified; explicitly select Reference attention",
        ));
    }
    Ok(())
}
pub(super) fn validate_attention_device(gpu: &Gpu, mode: AttentionMode) -> Result<()> {
    if mode == AttentionMode::Online {
        for kernel in [Kernel::AttentionDecode, Kernel::AttentionPrefill] {
            if gpu.execution_width(kernel) < 32 {
                return Err(Error::Unsupported(
                    "online attention requires SIMD width >= 32",
                ));
            }
        }
    }
    Ok(())
}
pub(super) struct SessionPlan {
    pub h: Layout,
    pub q: Layout,
    pub k: Layout,
    pub m: Layout,
    pub f: Layout,
    pub cache_count: usize,
    pub score_count: usize,
    pub zero_count: usize,
    pub memory: session::Memory,
    pub token_count: usize,
}
impl SessionPlan {
    pub fn new(
        c: &model::Config,
        weight_bytes: usize,
        options: session::Options,
        readback_mode: ReadbackMode,
        attention_mode: AttentionMode,
    ) -> Result<Self> {
        let chunk = options.prefill_chunk_size;
        if options.capacity == 0
            || options.capacity > weights::MAX_CONTEXT_LIMIT
            || chunk == 0
            || chunk > 1024
            || chunk > options.capacity
            || options
                .position_offset
                .checked_add(options.capacity)
                .is_none_or(|n| n > c.max_positions.min(weights::MAX_CONTEXT_LIMIT))
            || options.memory_limit_bytes > weights::INITIAL_MEMORY_LIMIT_BYTES
        {
            return Err(Error::Options(
                "Metal session options outside admitted model/context bounds",
            ));
        }
        let h = Layout::new(chunk, c.hidden_size, true)?;
        let query_width = c.query_heads * c.head_dim;
        let q = Layout::new(chunk, query_width, true)?;
        let k = Layout::new(chunk, c.kv_heads * c.head_dim, true)?;
        let m = Layout::new(chunk, c.intermediate_size, true)?;
        let widest = c
            .vocab_size
            .max(c.intermediate_size)
            .max(c.hidden_size)
            .max(query_width)
            .max(c.kv_heads * c.head_dim);
        let f = Layout::new(chunk, widest, false)?;
        if h.stride != c.hidden_size
            || q.stride != query_width
            || k.stride != c.kv_heads * c.head_dim
            || m.stride != c.intermediate_size
        {
            return Err(Error::Unsupported("requires contiguous Half activations"));
        }
        let cache_count = c
            .kv_heads
            .checked_mul(options.capacity)
            .and_then(|n| n.checked_mul(c.head_dim))
            .and_then(|n| n.checked_add(7))
            .ok_or(Error::Options("cache size overflow"))?;
        // Reference owns two [heads, chunk, capacity] FP32 matrices.
        // Online leaves only two eight-word guards, independent of depth.
        let score_count = if attention_mode == AttentionMode::Online {
            8
        } else {
            c.query_heads
                .checked_mul(chunk)
                .and_then(|n| n.checked_mul(options.capacity))
                .and_then(|n| n.checked_add(7))
                .ok_or(Error::Options("score size overflow"))?
        };
        let zero_count = widest
            .checked_add(7)
            .ok_or(Error::Options("zero bias overflow"))?;
        let scratch_bytes = add(
            add(
                add(
                    add(
                        h.bytes(2)?
                            .checked_mul(3)
                            .ok_or(Error::Options("scratch overflow"))?,
                        q.bytes(2)?
                            .checked_mul(3)
                            .ok_or(Error::Options("scratch overflow"))?,
                    )?,
                    k.bytes(2)?
                        .checked_mul(3)
                        .ok_or(Error::Options("scratch overflow"))?,
                )?,
                m.bytes(2)?
                    .checked_mul(4)
                    .ok_or(Error::Options("scratch overflow"))?,
            )?,
            add(
                add(
                    f.bytes(4)?,
                    score_count
                        .checked_mul(8)
                        .ok_or(Error::Options("scores overflow"))?,
                )?,
                add(
                    add(
                        add((chunk + 7) * 4, 32)?,
                        if readback_mode == ReadbackMode::GreedyToken {
                            32
                        } else {
                            0
                        },
                    )?,
                    zero_count * 2,
                )?,
            )?,
        )?;
        let kv_bytes = cache_count
            .checked_mul(4)
            .and_then(|n| n.checked_mul(c.num_layers))
            .ok_or(Error::Options("KV bytes overflow"))?;
        let history_bytes = options.capacity * 4;
        let logits_bytes = if readback_mode == ReadbackMode::GreedyToken {
            8
        } else {
            c.vocab_size * 8
        };
        let metadata_bytes =
            chunk * std::mem::size_of::<Views>() + c.num_layers * std::mem::size_of::<Cache>();
        let admitted = [
            weight_bytes,
            scratch_bytes,
            kv_bytes,
            history_bytes,
            logits_bytes,
            metadata_bytes,
            ALLOWANCE,
        ]
        .into_iter()
        .try_fold(0, add)?;
        Ok(Self {
            h,
            q,
            k,
            m,
            f,
            cache_count,
            score_count,
            zero_count,
            token_count: chunk + 7,
            memory: session::Memory {
                weights_bytes: weight_bytes,
                kv_bytes,
                scratch_bytes,
                history_bytes,
                logits_bytes,
                metadata_bytes,
                metadata_allowance_bytes: ALLOWANCE,
                admitted_bytes: admitted,
            },
        })
    }
    pub fn admit(&self, limit: usize) -> Result<()> {
        admit(self.memory.admitted_bytes, limit)
    }
    pub fn validate_buffers(&self, maximum: usize) -> Result<()> {
        for bytes in [
            self.h.bytes(2)?,
            self.q.bytes(2)?,
            self.k.bytes(2)?,
            self.m.bytes(2)?,
            self.f.bytes(4)?,
            self.cache_count
                .checked_mul(2)
                .ok_or(Error::Options("cache size overflow"))?,
            self.score_count
                .checked_mul(4)
                .ok_or(Error::Options("score size overflow"))?,
            self.zero_count
                .checked_mul(2)
                .ok_or(Error::Options("zero bias overflow"))?,
            self.token_count
                .checked_mul(4)
                .ok_or(Error::Options("token size overflow"))?,
            32,
        ] {
            memory::validate_buffer_bytes(bytes, maximum)?;
        }
        Ok(())
    }
}

/// Exact existing admission totals, not a measured process peak. Resident weights
/// are already included in both load and session totals: do not add them again.
/// Diagnostics, profiling, tokenizer/application owners and other sessions require
/// separate admission. A successful plan does not validate payload values or
/// guarantee that subsequent allocations succeed, or certify a model/device.
#[derive(Debug, serde::Serialize)]
pub struct MlxReport {
    pub schema_version: u32,
    pub device_name: String,
    pub maximum_buffer_bytes: usize,
    pub requested_load_context: usize,
    pub requested_capacity: usize,
    pub requested_chunk_size: usize,
    pub requested_position_offset: usize,
    pub requested_load_budget_bytes: usize,
    pub requested_session_budget_bytes: usize,
    pub projection: super::ProjectionMode,
    pub normalization: NormalizationMode,
    pub readback: ReadbackMode,
    pub attention: AttentionMode,
    pub source_payload_bytes: usize,
    pub source_admitted_load_bytes: usize,
    pub resident_weights_bytes: usize,
    pub resident_admitted_load_bytes: usize,
    pub session_memory: session::Memory,
    pub fits_source_load_budget: bool,
    pub fits_resident_load_budget: bool,
    pub fits_session_budget: bool,
    pub fits_requested_budgets: bool,
    pub numerical_buffers_allocated: bool,
    pub payload_validation_required: bool,
    pub qualification_evaluated: bool,
}

/// Opt-in runtime inspection for the explicit MlxAffine projection backend.
/// Reads bounded source metadata (and any redundant tied-head identity), queries
/// MPS strides, initializes Metal/MPS and compiles the existing precise pipelines.
/// Driver/library/pipeline memory is covered only by the existing 64 MiB estimate.
/// No numerical weights, KV, scratch or logits owners are created. Reopening and
/// all loading/session checks remain mandatory; this is an historical report.
/// Load and session budgets are independent, each at most the engine ceiling.
/// Reference attention and FullLogits are NOT inferred: callers select all modes.
pub fn inspect_mlx(
    directory: &Path,
    load: weights::LoadOptions,
    options: session::Options,
    normalization: NormalizationMode,
    readback: ReadbackMode,
    attention: AttentionMode,
) -> Result<MlxReport> {
    autoreleasepool(|_| {
        if load.memory_limit_bytes == 0
            || load.memory_limit_bytes > weights::INITIAL_MEMORY_LIMIT_BYTES
        {
            return Err(Error::Options("MLX loading budget must be within 6 GiB"));
        }
        // Inspect against the existing ceiling so insufficient caller budgets
        // can be reported without allocating or expanding model payloads.
        let source = Source::open_dense(
            directory,
            weights::LoadOptions {
                memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES,
                ..load
            },
        )?;
        super::mlx_geometry(source.config(), source.architecture())?;
        validate_attention(Some(source.architecture()), attention)?;
        let weights = MlxWeights::new(&source)?;
        let session = SessionPlan::new(
            source.config(),
            weights.weight_bytes,
            options,
            readback,
            attention,
        )?;
        let gpu = Gpu::new()?;
        let _pipelines = mlx::compile(&gpu)?;
        validate_attention_device(&gpu, attention)?;
        let maximum = gpu.device.maxBufferLength();
        weights.validate_buffers(maximum)?;
        session.validate_buffers(maximum)?;
        let fits_source_load_budget = source.admitted_load_bytes() <= load.memory_limit_bytes;
        let fits_resident_load_budget = weights.admitted <= load.memory_limit_bytes;
        let fits_session_budget = session.memory.admitted_bytes <= options.memory_limit_bytes;
        Ok(MlxReport {
            schema_version: 1,
            device_name: gpu.device.name().to_string(),
            maximum_buffer_bytes: maximum,
            requested_load_context: load.context_limit,
            requested_capacity: options.capacity,
            requested_chunk_size: options.prefill_chunk_size,
            requested_position_offset: options.position_offset,
            requested_load_budget_bytes: load.memory_limit_bytes,
            requested_session_budget_bytes: options.memory_limit_bytes,
            projection: super::ProjectionMode::MlxAffine,
            normalization,
            readback,
            attention,
            source_payload_bytes: source.tensor_payload_bytes(),
            source_admitted_load_bytes: source.admitted_load_bytes(),
            resident_weights_bytes: weights.weight_bytes,
            resident_admitted_load_bytes: weights.admitted,
            session_memory: session.memory,
            fits_source_load_budget,
            fits_resident_load_budget,
            fits_session_budget,
            fits_requested_budgets: fits_source_load_budget
                && fits_resident_load_budget
                && fits_session_budget,
            numerical_buffers_allocated: false,
            payload_validation_required: true,
            qualification_evaluated: false,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocation_limits_match_buffer_boundary_without_device() {
        assert!(memory::validate_buffer_bytes(1, 1).is_ok());
        assert!(memory::validate_buffer_bytes(32, 32).is_ok());
        for (bytes, maximum) in [
            (0, 1),
            (1, 0),
            (33, 32),
            (isize::MAX as usize + 1, usize::MAX),
        ] {
            assert!(matches!(
                memory::validate_buffer_bytes(bytes, maximum),
                Err(Error::Options("buffer exceeds device/index limits"))
            ));
        }
        assert!(
            matches!(admit(usize::MAX,usize::MAX-1),Err(Error::Budget{required:usize::MAX,limit}) if limit==usize::MAX-1)
        );
        assert!(admit(usize::MAX, usize::MAX).is_ok());
        assert!(matches!(
            add(usize::MAX, 1),
            Err(Error::Options("memory arithmetic overflow"))
        ));
    }
    #[test]
    fn plans_check_each_buffer_and_overflow_without_gpu() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mlx-qwen3-v1/tied");
        let source = Source::open_dense(
            &path,
            weights::LoadOptions {
                context_limit: 32,
                memory_limit_bytes: 256 * 1024 * 1024,
            },
        )
        .unwrap();
        let options = session::Options {
            capacity: 32,
            prefill_chunk_size: 3,
            position_offset: 0,
            memory_limit_bytes: 256 * 1024 * 1024,
        };
        let make = || {
            SessionPlan::new(
                source.config(),
                100,
                options,
                ReadbackMode::GreedyToken,
                AttentionMode::Reference,
            )
            .unwrap()
        };
        const LIMIT: usize = 1024 * 1024;
        assert!(make().validate_buffers(LIMIT).is_ok());
        for field in 0..10 {
            let mut plan = make();
            match field {
                0 => plan.h.elements = LIMIT / 2 + 1,
                1 => plan.q.elements = LIMIT / 2 + 1,
                2 => plan.k.elements = LIMIT / 2 + 1,
                3 => plan.m.elements = LIMIT / 2 + 1,
                4 => plan.f.elements = LIMIT / 4 + 1,
                5 => plan.cache_count = LIMIT / 2 + 1,
                6 => plan.score_count = LIMIT / 4 + 1,
                7 => plan.zero_count = LIMIT / 2 + 1,
                8 => plan.token_count = LIMIT / 4 + 1,
                9 => plan.cache_count = usize::MAX,
                _ => unreachable!(),
            }
            assert!(
                matches!(plan.validate_buffers(LIMIT), Err(Error::Options(_))),
                "field {field}"
            );
        }
        assert!(make().validate_buffers(31).is_err());
        assert!(matches!(
            SessionPlan::new(
                source.config(),
                usize::MAX,
                options,
                ReadbackMode::FullLogits,
                AttentionMode::Reference
            ),
            Err(Error::Options("memory arithmetic overflow"))
        ));
        let weights = MlxWeights::new(&source).unwrap();
        let largest = weights
            .tensors
            .values()
            .flat_map(|(layout, storage)| match storage {
                Storage::Float(_) => vec![layout.bytes(2).unwrap()],
                Storage::Affine(p) => vec![p.weight_bytes(), p.metadata_bytes().div_ceil(4) * 4],
            })
            .max()
            .unwrap();
        assert!(weights.validate_buffers(largest).is_ok());
        assert!(weights.validate_buffers(largest - 1).is_err());
    }
    #[test]
    fn unqualified_online_architectures_reject_before_device() {
        for architecture in [
            Architecture::Llama,
            Architecture::Mistral,
            Architecture::Qwen3,
        ] {
            assert!(matches!(
                validate_attention(Some(architecture), AttentionMode::Online),
                Err(Error::Unsupported(_))
            ));
            assert!(validate_attention(Some(architecture), AttentionMode::Reference).is_ok());
        }
        assert!(validate_attention(Some(Architecture::Qwen2), AttentionMode::Online).is_ok());
    }
}
