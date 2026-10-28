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
    capacity::{self, KvOwner, KvStorage, Retention},
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
    pub fn validate_buffers(&self, maximum: usize) -> Result<usize> {
        let mut largest = 0;
        for (layout, storage) in self.tensors.values() {
            match storage {
                Storage::Float(_) => {
                    let bytes = layout.bytes(2)?;
                    memory::validate_buffer_bytes(bytes, maximum)?;
                    largest = largest.max(bytes);
                }
                Storage::Affine(p) => {
                    mlx::validate_resident_layout(*p)?;
                    for bytes in [p.weight_bytes(), p.metadata_bytes().div_ceil(4) * 4] {
                        memory::validate_buffer_bytes(bytes, maximum)?;
                        largest = largest.max(bytes);
                    }
                }
            }
        }
        Ok(largest)
    }
}
fn admit(required: usize, limit: usize) -> Result<()> {
    if required > limit {
        Err(Error::Budget { required, limit })
    } else {
        Ok(())
    }
}
/// Coarse factory policy; exhaustive matching makes new families opt in explicitly.
pub(super) fn native_attention_mode(architecture: Option<Architecture>) -> AttentionMode {
    match architecture {
        None | Some(Architecture::Qwen2) => AttentionMode::Online,
        Some(Architecture::Llama | Architecture::Mistral | Architecture::Qwen3) => {
            AttentionMode::Reference
        }
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
        Self::with_context_limit(
            c,
            weight_bytes,
            options,
            readback_mode,
            attention_mode,
            weights::MAX_CONTEXT_LIMIT,
        )
    }
    pub fn new_qwen3_context_diagnostic(
        c: &model::Config,
        weight_bytes: usize,
        options: session::Options,
        readback_mode: ReadbackMode,
        attention_mode: AttentionMode,
    ) -> Result<Self> {
        // This additional validation prevents a caller from using the longer
        // planner route for another model or an unadmitted attention mode.
        crate::mlx::execution::qwen3_context_geometry(c, Architecture::Qwen3)
            .map_err(|error| Error::Options(error.message()))?;
        if attention_mode != AttentionMode::Reference {
            return Err(Error::Options(
                "Qwen3 context diagnostic requires Reference attention",
            ));
        }
        Self::with_context_limit(
            c,
            weight_bytes,
            options,
            readback_mode,
            attention_mode,
            crate::mlx::execution::QWEN3_CONTEXT_DIAGNOSTIC_LIMIT,
        )
    }
    fn with_context_limit(
        c: &model::Config,
        weight_bytes: usize,
        options: session::Options,
        readback_mode: ReadbackMode,
        attention_mode: AttentionMode,
        context_limit: usize,
    ) -> Result<Self> {
        ResolvedSessionPlan::new(
            c,
            weight_bytes,
            options,
            readback_mode,
            attention_mode,
            context_limit,
        )?
        .at_capacity(options.capacity)
    }
    pub fn admit(&self, limit: usize) -> Result<()> {
        admit(self.memory.admitted_bytes, limit)
    }
    fn buffer_bytes(&self) -> Result<[usize; 10]> {
        Ok([
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
        ])
    }
    fn largest_buffer_bytes(&self) -> Result<usize> {
        Ok(self.buffer_bytes()?.into_iter().max().unwrap_or(0))
    }
    pub fn validate_buffers(&self, maximum: usize) -> Result<()> {
        for bytes in self.buffer_bytes()? {
            memory::validate_buffer_bytes(bytes, maximum)?;
        }
        Ok(())
    }
}

fn validate_session_options(
    c: &model::Config,
    options: session::Options,
    context_limit: usize,
) -> Result<usize> {
    let chunk = options.prefill_chunk_size;
    if options.capacity == 0
        || options.capacity > context_limit
        || chunk == 0
        || chunk > 1024
        || chunk > options.capacity
        || options
            .position_offset
            .checked_add(options.capacity)
            .is_none_or(|n| n > c.max_positions.min(context_limit))
        || options.memory_limit_bytes > weights::INITIAL_MEMORY_LIMIT_BYTES
    {
        return Err(Error::Options(
            "Metal session options outside admitted model/context bounds",
        ));
    }
    Ok(chunk)
}

/// Device-specific layout resolution happens once. Candidate capacities below
/// reuse only checked integer arithmetic and the exact constructor formula.
#[derive(Clone, Copy)]
struct ResolvedSessionPlan<'a> {
    c: &'a model::Config,
    weight_bytes: usize,
    options: session::Options,
    readback_mode: ReadbackMode,
    attention_mode: AttentionMode,
    context_limit: usize,
    h: Layout,
    q: Layout,
    k: Layout,
    m: Layout,
    f: Layout,
    widest: usize,
}
impl<'a> ResolvedSessionPlan<'a> {
    fn new(
        c: &'a model::Config,
        weight_bytes: usize,
        options: session::Options,
        readback_mode: ReadbackMode,
        attention_mode: AttentionMode,
        context_limit: usize,
    ) -> Result<Self> {
        let chunk = validate_session_options(c, options, context_limit)?;
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
        Ok(Self {
            c,
            weight_bytes,
            options,
            readback_mode,
            attention_mode,
            context_limit,
            h,
            q,
            k,
            m,
            f,
            widest,
        })
    }
    fn at_capacity(&self, capacity: usize) -> Result<SessionPlan> {
        let Self {
            c,
            weight_bytes,
            mut options,
            readback_mode,
            attention_mode,
            context_limit,
            h,
            q,
            k,
            m,
            f,
            widest,
        } = *self;
        options.capacity = capacity;
        let chunk = validate_session_options(c, options, context_limit)?;
        let kv = KvOwner {
            heads: c.kv_heads,
            head_dim: c.head_dim,
            storage: KvStorage::F16,
            retention: Retention::Full,
            guard_elements: 7,
            copies: c.num_layers,
        }
        .at_capacity(options.capacity)
        .map_err(|_| Error::Options("KV bytes overflow"))?;
        let cache_count = kv.elements_per_array;
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
        // Matrix Layout already bounds queried activation strides. KV and
        // Reference score storage use separate head-major UInt32 expressions;
        // validate their actual computed guarded counts before allocation/casts.
        if cache_count > u32::MAX as usize || score_count > u32::MAX as usize {
            return Err(Error::Options("KV/score shader indexing exceeds UInt32"));
        }
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
        let kv_bytes = kv.total_bytes;
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
        Ok(SessionPlan {
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
    fn capacity_report(
        &self,
        maximum_buffer_bytes: usize,
        weight_buffer_bytes: usize,
    ) -> Result<capacity::Report> {
        capacity::search(
            self.options.prefill_chunk_size,
            self.options.capacity,
            capacity::Limits {
                budget_bytes: self.options.memory_limit_bytes,
                maximum_buffer_bytes,
            },
            |capacity| -> Result<capacity::Footprint> {
                let plan = self.at_capacity(capacity)?;
                Ok(capacity::Footprint {
                    admitted_bytes: plan.memory.admitted_bytes,
                    largest_buffer_bytes: plan.largest_buffer_bytes()?.max(weight_buffer_bytes),
                })
            },
        )
        .map_err(|error| match error {
            capacity::SearchError::InvalidBounds => {
                Error::Options("invalid capacity search bounds")
            }
            capacity::SearchError::Planner(error) => error,
        })
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
    inspect_with_policy(
        directory,
        load,
        options,
        normalization,
        readback,
        AttentionPolicy::Explicit(attention),
        false,
    )
    .map(|inspection| inspection.resources)
}

/// Same planning with the actual native model factory's attention choice.
/// All other modes remain explicit. The report contains the resolved attention;
/// new graph families must declare a factory policy before this code compiles.
pub fn inspect_mlx_with_native_attention(
    directory: &Path,
    load: weights::LoadOptions,
    options: session::Options,
    normalization: NormalizationMode,
    readback: ReadbackMode,
) -> Result<MlxReport> {
    inspect_with_policy(
        directory,
        load,
        options,
        normalization,
        readback,
        AttentionPolicy::NativeModel,
        false,
    )
    .map(|inspection| inspection.resources)
}

/// Additive read-only numerical range report, within the already admitted
/// requested upper capacity. This does not discover a model's declared maximum,
/// widen factory eligibility, reserve memory or certify a text/device profile.
/// `capacity` may fit while `resources.fits_*_load_budget` is false. Callers must
/// honor both load flags, allow for application/process owners, and re-admit at
/// construction. Existing inspect APIs/schema remain unchanged.
#[derive(Debug, serde::Serialize)]
pub struct CapacityInspection {
    pub resources: MlxReport,
    pub capacity: Option<capacity::Report>,
}
pub fn inspect_mlx_capacity(
    directory: &Path,
    load: weights::LoadOptions,
    options: session::Options,
    normalization: NormalizationMode,
    readback: ReadbackMode,
    attention: AttentionMode,
) -> Result<CapacityInspection> {
    inspect_with_policy(
        directory,
        load,
        options,
        normalization,
        readback,
        AttentionPolicy::Explicit(attention),
        true,
    )
}

enum AttentionPolicy {
    Explicit(AttentionMode),
    NativeModel,
}
fn inspect_with_policy(
    directory: &Path,
    load: weights::LoadOptions,
    options: session::Options,
    normalization: NormalizationMode,
    readback: ReadbackMode,
    policy: AttentionPolicy,
    search_capacity: bool,
) -> Result<CapacityInspection> {
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
        crate::mlx::execution::metal_source_geometry(&source)
            .map_err(|error| Error::Unsupported(error.message()))?;
        let attention = match policy {
            AttentionPolicy::Explicit(mode) => mode,
            AttentionPolicy::NativeModel => native_attention_mode(Some(source.architecture())),
        };
        validate_attention(Some(source.architecture()), attention)?;
        let weights = MlxWeights::new(&source)?;
        let resolved = ResolvedSessionPlan::new(
            source.config(),
            weights.weight_bytes,
            options,
            readback,
            attention,
            weights::MAX_CONTEXT_LIMIT,
        )?;
        let session = resolved.at_capacity(options.capacity)?;
        let gpu = Gpu::new()?;
        let _pipelines = mlx::compile(&gpu)?;
        validate_attention_device(&gpu, attention)?;
        let maximum = gpu.device.maxBufferLength();
        let weight_buffer_bytes = weights.validate_buffers(maximum)?;
        let capacity = if search_capacity {
            Some(resolved.capacity_report(maximum, weight_buffer_bytes)?)
        } else {
            session.validate_buffers(maximum)?;
            None
        };
        let fits_source_load_budget = source.admitted_load_bytes() <= load.memory_limit_bytes;
        let fits_resident_load_budget = weights.admitted <= load.memory_limit_bytes;
        let fits_session_budget = session.memory.admitted_bytes <= options.memory_limit_bytes;
        Ok(CapacityInspection {
            capacity,
            resources: MlxReport {
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
            },
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kv_and_reference_scores_reject_composite_uint32_overflow() {
        let config = |query_heads, kv_heads| model::Config {
            hidden_size: query_heads * 8,
            intermediate_size: 8,
            query_heads,
            kv_heads,
            head_dim: 8,
            num_layers: 1,
            vocab_size: 8,
            max_positions: 32768,
            rope_theta: 10000.,
            rms_norm_eps: 1e-6,
        };
        let options = session::Options {
            capacity: 32768,
            prefill_chunk_size: 1,
            position_offset: 0,
            memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES,
        };
        let safe_cache = SessionPlan::new(
            &config(16383, 16383),
            0,
            options,
            ReadbackMode::FullLogits,
            AttentionMode::Online,
        )
        .unwrap();
        assert!(safe_cache.cache_count <= u32::MAX as usize);
        assert!(matches!(
            SessionPlan::new(
                &config(16384, 16384),
                0,
                options,
                ReadbackMode::FullLogits,
                AttentionMode::Online
            ),
            Err(Error::Options("KV/score shader indexing exceeds UInt32"))
        ));
        let options = session::Options {
            prefill_chunk_size: 128,
            ..options
        };
        let safe_scores = SessionPlan::new(
            &config(1023, 1),
            0,
            options,
            ReadbackMode::FullLogits,
            AttentionMode::Reference,
        )
        .unwrap();
        assert!(safe_scores.score_count <= u32::MAX as usize);
        assert!(matches!(
            SessionPlan::new(
                &config(1024, 1),
                0,
                options,
                ReadbackMode::FullLogits,
                AttentionMode::Reference
            ),
            Err(Error::Options("KV/score shader indexing exceeds UInt32"))
        ));
        // These are pure plans with zero placeholder weights, not executable
        // models or budget-fit claims; no numerical buffers are allocated.
    }
    #[test]
    fn resolved_capacity_search_matches_constructor_totals_and_device_edge() {
        let c = model::Config {
            hidden_size: 1024,
            intermediate_size: 3072,
            query_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            num_layers: 28,
            // Keep constant vocabulary scratch below the257-row KV/score
            // buffers so this test exercises a genuine device-buffer limit.
            vocab_size: 128,
            max_positions: 512,
            rope_theta: 1e6,
            rms_norm_eps: 1e-6,
        };
        let options = session::Options {
            capacity: 512,
            prefill_chunk_size: 32,
            position_offset: 0,
            memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES,
        };
        for attention in [AttentionMode::Reference, AttentionMode::Online] {
            for readback in [ReadbackMode::FullLogits, ReadbackMode::GreedyToken] {
                // Planner-only Online shape coverage grants no Qwen3 Online eligibility.
                let mut resolved = ResolvedSessionPlan::new(
                    &c,
                    633366062,
                    options,
                    readback,
                    attention,
                    weights::MAX_CONTEXT_LIMIT,
                )
                .unwrap();
                let boundary = resolved.at_capacity(257).unwrap();
                let direct = SessionPlan::new(
                    &c,
                    633366062,
                    session::Options {
                        capacity: 257,
                        ..options
                    },
                    readback,
                    attention,
                )
                .unwrap();
                assert_eq!(boundary.memory, direct.memory);
                resolved.options.memory_limit_bytes = boundary.memory.admitted_bytes;
                let report = resolved.capacity_report(isize::MAX as usize, 1024).unwrap();
                assert_eq!(report.selected.unwrap().capacity, 257);
                assert_eq!(report.first_excluded.unwrap().capacity, 258);
                assert!(!report.first_excluded.unwrap().fits_budget);
                resolved.options.memory_limit_bytes = weights::INITIAL_MEMORY_LIMIT_BYTES;
                let buffer_edge = boundary.largest_buffer_bytes().unwrap();
                let report = resolved.capacity_report(buffer_edge, 1024).unwrap();
                let expected = (32..=512)
                    .filter(|&n| {
                        resolved
                            .at_capacity(n)
                            .unwrap()
                            .largest_buffer_bytes()
                            .unwrap()
                            <= buffer_edge
                    })
                    .next_back();
                assert_eq!(report.selected.map(|p| p.capacity), expected);
                assert_eq!(report.selected.unwrap().capacity, 257);
                assert_eq!(report.first_excluded.unwrap().capacity, 258);
                assert!(!report.first_excluded.unwrap().fits_buffer);
                assert!(resolved.at_capacity(513).is_err());
            }
        }
    }

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
    fn qwen3_context_diagnostic_resource_equations_and_boundaries_without_gpu() {
        // MPS stride metadata only; no device, numerical buffer or model load.
        let c = model::Config {
            hidden_size: 1024,
            intermediate_size: 3072,
            query_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            num_layers: 28,
            vocab_size: 151936,
            max_positions: 40960,
            rope_theta: 1e6,
            rms_norm_eps: 1e-6,
        };
        let options = session::Options {
            capacity: 40960,
            prefill_chunk_size: 32,
            position_offset: 0,
            memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES,
        };
        let make = |o, weight_bytes| {
            SessionPlan::new_qwen3_context_diagnostic(
                &c,
                weight_bytes,
                o,
                ReadbackMode::FullLogits,
                AttentionMode::Reference,
            )
        };
        const WEIGHTS: usize = 633366062;
        for capacity in [512, 1025, 4096, 8192, 16384, 32768, 40960] {
            for chunk in [32, 128] {
                let o = session::Options {
                    capacity,
                    prefill_chunk_size: chunk,
                    ..options
                };
                let plan = make(o, WEIGHTS).unwrap();
                // Independent explicit byte equations include the seven-element
                // guards, padded layout geometry, scratch owners and host readback.
                assert_eq!(
                    (
                        plan.h.stride,
                        plan.q.stride,
                        plan.k.stride,
                        plan.m.stride,
                        plan.f.stride
                    ),
                    (1024, 2048, 1024, 3072, 151936)
                );
                let kv = (8usize
                    .checked_mul(capacity)
                    .unwrap()
                    .checked_mul(128)
                    .unwrap()
                    + 7)
                    * 4
                    * 28;
                let scratch = 656900usize
                    .checked_mul(chunk)
                    .unwrap()
                    .checked_add(
                        128usize
                            .checked_mul(chunk)
                            .unwrap()
                            .checked_mul(capacity)
                            .unwrap(),
                    )
                    .unwrap()
                    .checked_add(304212)
                    .unwrap();
                let metadata =
                    chunk * std::mem::size_of::<Views>() + 28 * std::mem::size_of::<Cache>();
                let total = [
                    WEIGHTS,
                    kv,
                    scratch,
                    capacity * 4,
                    151936 * 8,
                    metadata,
                    64 * 1024 * 1024,
                ]
                .into_iter()
                .try_fold(0usize, usize::checked_add)
                .unwrap();
                assert_eq!(plan.memory.kv_bytes, kv);
                assert_eq!(plan.memory.scratch_bytes, scratch);
                assert_eq!(plan.memory.admitted_bytes, total);
                assert!(plan.admit(total).is_ok());
                assert!(
                    matches!(plan.admit(total-1),Err(Error::Budget {required,limit}) if required==total && limit==total-1)
                );
                let largest = (plan.score_count * 4)
                    .max(plan.cache_count * 2)
                    .max(plan.f.bytes(4).unwrap());
                assert!(plan.validate_buffers(largest).is_ok());
                assert!(matches!(
                    plan.validate_buffers(largest - 1),
                    Err(Error::Options("buffer exceeds device/index limits"))
                ));
                assert!(
                    plan.cache_count <= u32::MAX as usize && plan.score_count <= u32::MAX as usize
                );
            }
        }
        assert_eq!(
            make(options, WEIGHTS).unwrap().memory.admitted_bytes,
            5588577426
        );
        assert!(
            SessionPlan::new(
                &c,
                WEIGHTS,
                options,
                ReadbackMode::FullLogits,
                AttentionMode::Reference
            )
            .is_err()
        );
        for o in [
            session::Options {
                capacity: 0,
                ..options
            },
            session::Options {
                capacity: 40961,
                ..options
            },
            session::Options {
                position_offset: 1,
                ..options
            },
            session::Options {
                capacity: 1,
                prefill_chunk_size: 1,
                position_offset: usize::MAX,
                ..options
            },
            session::Options {
                prefill_chunk_size: 0,
                ..options
            },
            session::Options {
                prefill_chunk_size: 1025,
                ..options
            },
            session::Options {
                memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES + 1,
                ..options
            },
        ] {
            assert!(make(o, WEIGHTS).is_err());
        }
        assert!(
            make(
                session::Options {
                    capacity: 1,
                    prefill_chunk_size: 1,
                    position_offset: 40959,
                    ..options
                },
                WEIGHTS
            )
            .is_ok()
        );
        assert!(matches!(
            make(options, usize::MAX),
            Err(Error::Options("memory arithmetic overflow"))
        ));
        for mode in [AttentionMode::Online, AttentionMode::CompensatedReference] {
            assert!(matches!(
                SessionPlan::new_qwen3_context_diagnostic(
                    &c,
                    WEIGHTS,
                    options,
                    ReadbackMode::FullLogits,
                    mode
                ),
                Err(Error::Options(
                    "Qwen3 context diagnostic requires Reference attention"
                ))
            ));
        }
        // Existing512 route has exactly the same resource geometry/accounting.
        let c512 = model::Config {
            max_positions: 512,
            ..c
        };
        let o512 = session::Options {
            capacity: 512,
            ..options
        };
        let old = SessionPlan::new(
            &c512,
            WEIGHTS,
            o512,
            ReadbackMode::FullLogits,
            AttentionMode::Reference,
        )
        .unwrap();
        let diagnostic = SessionPlan::new_qwen3_context_diagnostic(
            &c512,
            WEIGHTS,
            o512,
            ReadbackMode::FullLogits,
            AttentionMode::Reference,
        )
        .unwrap();
        assert_eq!(old.memory, diagnostic.memory);
        assert_eq!(
            (old.cache_count, old.score_count),
            (diagnostic.cache_count, diagnostic.score_count)
        );
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
        let compensated = SessionPlan::new(
            source.config(),
            100,
            options,
            ReadbackMode::GreedyToken,
            AttentionMode::CompensatedReference,
        )
        .unwrap();
        let reference = make();
        assert_eq!(compensated.score_count, reference.score_count);
        assert_eq!(compensated.cache_count, reference.cache_count);
        assert_eq!(compensated.memory, reference.memory);
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
    fn native_attention_policy_is_exhaustive_for_current_factory_graphs() {
        for (architecture, expected) in [
            (None, AttentionMode::Online),
            (Some(Architecture::Qwen2), AttentionMode::Online),
            (Some(Architecture::Llama), AttentionMode::Reference),
            (Some(Architecture::Mistral), AttentionMode::Reference),
            (Some(Architecture::Qwen3), AttentionMode::Reference),
        ] {
            assert_eq!(native_attention_mode(architecture), expected);
        }
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
            assert!(
                validate_attention(Some(architecture), AttentionMode::CompensatedReference).is_ok()
            );
        }
        assert!(validate_attention(Some(Architecture::Qwen2), AttentionMode::Online).is_ok());
    }
}
