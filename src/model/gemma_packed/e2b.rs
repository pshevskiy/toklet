//! Cold exact E2B name/layout resolution; no tensor-name lookup during forward.
use super::*;
use crate::mlx::gemma4::{E2bPlan, payload::PayloadStore};

const PREFIX: &str = "language_model.model.";

impl Layout {
    fn e2b(plan: &E2bPlan) -> Result<Self, Error> {
        Self::e2b_layers(plan.layers())
    }
    fn e2b_layers(geometry: &[crate::mlx::gemma4::LayerPlan]) -> Result<Self, Error> {
        Self::e2b_layers_for_limit(geometry, MAX_POSITIONS)
    }
    fn e2b_layers_for_limit(
        geometry: &[crate::mlx::gemma4::LayerPlan],
        position_limit: usize,
    ) -> Result<Self, Error> {
        if !(1..=MAX_E2B_DIAGNOSTIC_POSITIONS).contains(&position_limit) {
            return Err(Error::InvalidConfig("Gemma E2B diagnostic position limit"));
        }
        if geometry.len() != 35 {
            return Err(Error::InvalidConfig("Gemma E2B geometry count"));
        }
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(geometry.len())
            .map_err(|_| Error::Allocation)?;
        for g in geometry {
            layers.push(LayerLayout {
                kind: g.kind,
                head_dim: g.head_dim,
                rotated_dims: g.rotated_dims,
                intermediate: g.intermediate,
                theta: if g.kind == KvKind::Full { 1e6 } else { 1e4 },
            });
        }
        // The cold plan is constructible publicly only through exact audited
        // config/index/header admission. Declared131072 is not execution capacity.
        Ok(Self {
            hidden: 1536,
            vocab: 262144,
            ple: 256,
            heads: 8,
            producers: 15,
            positions: position_limit,
            window: 512,
            layers,
        })
    }
}

#[cfg(all(
    feature = "metal",
    any(target_os = "macos", target_os = "ios"),
    target_arch = "aarch64"
))]
impl<'a, M: MatrixSource> Model<'a, M> {
    pub(crate) fn from_e2b_parts(
        weights: Weights<'a, M>,
        geometry: &[crate::mlx::gemma4::LayerPlan],
        backing: Backing,
        position_limit: usize,
        qat_mixed: bool,
    ) -> Result<Self, Error> {
        let profile = if qat_mixed {
            StorageProfile::MixedQat
        } else {
            StorageProfile::Affine4
        };
        Self::from_e2b_format(weights, geometry, backing, position_limit, profile)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_official_bf16_parts(
        weights: Weights<'a, M>,
        geometry: &[crate::mlx::gemma4::LayerPlan],
        backing: Backing,
        position_limit: usize,
    ) -> Result<Self, Error> {
        Self::from_e2b_format(
            weights,
            geometry,
            backing,
            position_limit,
            StorageProfile::OfficialBf16,
        )
    }

    fn from_e2b_format(
        weights: Weights<'a, M>,
        geometry: &[crate::mlx::gemma4::LayerPlan],
        backing: Backing,
        position_limit: usize,
        profile: StorageProfile,
    ) -> Result<Self, Error> {
        let result = Self::with_storage_format(
            weights,
            Layout::e2b_layers_for_limit(geometry, position_limit)?,
            Some(backing),
            profile,
        )?;
        for (index, layer) in geometry.iter().enumerate() {
            if result.plan.producer_for(index)? != layer.producer {
                return Err(Error::InvalidConfig("Gemma E2B shared producer differs"));
            }
        }
        Ok(result)
    }
}

impl<'a> Model<'a> {
    /// Borrow the exact admitted packed4/group64/BF16 E2B text payload.
    ///
    /// This scalar diagnostic graph admits only32 total positions. It supplies
    /// raw logits, not tokenizer/chat/generation, a qualified source-BF16/GPU
    /// backend or app eligibility. Keep store -> model -> session owners alive;
    /// no tensors are copied or expanded by this adapter.
    pub fn from_e2b(store: &'a PayloadStore) -> Result<Self, Error> {
        let matrix = |name: &str| {
            store
                .matrix(&format!("{PREFIX}{name}"))
                .map_err(|_| Error::InvalidConfig("Gemma E2B packed tensor"))
        };
        let vector = |name: &str| {
            store
                .vector(&format!("{PREFIX}{name}"))
                .map_err(|_| Error::InvalidConfig("Gemma E2B vector"))
        };
        let layout = Layout::e2b(store.plan())?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(layout.layers.len())
            .map_err(|_| Error::Allocation)?;
        for i in 0..layout.layers.len() {
            let base = format!("layers.{i}.");
            let m = |name: &str| matrix(&format!("{base}{name}.weight"));
            let v = |name: &str| vector(&format!("{base}{name}.weight"));
            let scalar = vector(&format!("{base}layer_scalar"))?;
            length("Gemma E2B layer scalar", scalar.len(), 1)?;
            layers.push(LayerWeights {
                input_norm: v("input_layernorm")?,
                q: m("self_attn.q_proj")?,
                q_norm: v("self_attn.q_norm")?,
                o: m("self_attn.o_proj")?,
                kv: if i < layout.producers {
                    Some(KvWeights {
                        k: m("self_attn.k_proj")?,
                        v: m("self_attn.v_proj")?,
                        k_norm: v("self_attn.k_norm")?,
                    })
                } else {
                    None
                },
                post_attention_norm: v("post_attention_layernorm")?,
                pre_ff_norm: v("pre_feedforward_layernorm")?,
                post_ff_norm: v("post_feedforward_layernorm")?,
                ff_gate: m("mlp.gate_proj")?,
                ff_up: m("mlp.up_proj")?,
                ff_down: m("mlp.down_proj")?,
                ple_gate: m("per_layer_input_gate")?,
                ple_projection: m("per_layer_projection")?,
                post_ple_norm: v("post_per_layer_input_norm")?,
                scalar: scalar[0],
            });
        }
        let weights = Weights {
            embedding: matrix("embed_tokens.weight")?,
            per_layer_embedding: matrix("embed_tokens_per_layer.weight")?,
            per_layer_projection: matrix("per_layer_model_projection.weight")?,
            per_layer_norm: vector("per_layer_projection_norm.weight")?,
            layers,
            final_norm: vector("norm.weight")?,
        };
        let mut model = Self::with_storage(weights, layout, Some(Backing::cpu(store.report())))?;
        for (i, layer) in store.plan().layers().iter().enumerate() {
            if model.plan.producer_for(i)? != layer.producer {
                return Err(Error::InvalidConfig("Gemma E2B shared producer differs"));
            }
        }
        model.content_identity = Some(*store.content_identity());
        Ok(model)
    }
}

/// Exact physical FP32 cache and proposal buffers for the pinned Mac profile.
/// Separate model/owner, scratch, codec, and application admission still applies.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn official_bf16_planned_kv(capacity: usize, chunk_rows: usize) -> Result<usize, Error> {
    use crate::metal::gemma_affine::OfficialBf16;
    if !(1..=OfficialBf16::MAX_POSITIONS).contains(&capacity)
        || !(1..=capacity.min(MAX_POSITIONS)).contains(&chunk_rows)
    {
        return Err(Error::InvalidConfig("official Gemma KV capacity or chunk"));
    }
    let layout = Layout::e2b_layers_for_limit(&crate::mlx::gemma4::layer_plans(), capacity)?;
    let cache_bytes = layout
        .shared_kv_plan()?
        .planned_narrowed_cache_bytes(capacity)?;
    let cache_metadata = add_bytes(
        std::mem::size_of::<Vec<WindowKvCache>>(),
        cpu::product(layout.producers, std::mem::size_of::<WindowKvCache>())?,
    )?;
    let cache_payload = cache_bytes
        .checked_sub(cache_metadata)
        .ok_or(cpu::Error::SizeOverflow)?;
    add_bytes(
        cache_payload,
        allocation_bytes(layout.staged_kv_values(chunk_rows)?)?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlx::gemma4::{ModalityPolicy, inspect};

    #[test]
    fn real_plan_uses_named_geometry_without_declared_context_allocations() {
        let plan = inspect(
            include_bytes!("../../../tests/fixtures/gemma4-e2b-metadata-v1/config.json"),
            include_bytes!("../../../tests/fixtures/gemma4-e2b-metadata-v1/index.json"),
            include_bytes!("../../../tests/fixtures/gemma4-e2b-metadata-v1/header.json"),
            3550670554,
            ModalityPolicy::TextOnly,
        )
        .unwrap();
        let layout = Layout::e2b(&plan).unwrap();
        assert_eq!(
            (layout.hidden, layout.vocab, layout.ple, layout.heads),
            (1536, 262144, 256, 8)
        );
        assert_eq!((layout.layers.len(), layout.producers), (35, 15));
        assert_eq!((layout.positions, layout.window), (32, 512));
        assert_eq!(plan.declared_total_positions(), 131072);
        for (i, layer) in layout.layers.iter().enumerate() {
            let full = i % 5 == 4;
            assert_eq!(
                layer.kind,
                if full { KvKind::Full } else { KvKind::Sliding }
            );
            assert_eq!(layer.head_dim, if full { 512 } else { 256 });
            assert_eq!(layer.rotated_dims, if full { 128 } else { 256 });
            assert_eq!(layer.theta, if full { 1e6 } else { 1e4 });
            assert_eq!(layer.intermediate, if i < 15 { 6144 } else { 12288 });
        }
        let counts = Scratch::counts(&layout, MAX_POSITIONS, MAX_POSITIONS).unwrap();
        assert_eq!(counts[4], 32 * 8 * 512);
        assert_eq!(counts[16], 32 * 12288);
        assert_eq!(counts[21], 32 * 262144);
        assert_eq!(counts[22], 32 * 262144);
        println!(
            "GEMMA_SESSION_CAPACITY_E2B_SCRATCH {}",
            serde_json::json!({
            "positions":32,"chunk_rows":32,"declared_positions":plan.declared_total_positions(),
            "scratch_elements":counts,"scratch_bytes":total_bytes(&counts).unwrap()})
        );
        let geometry: Vec<_> = layout
            .layers
            .iter()
            .map(|layer| LayerKvGeometry {
                kind: layer.kind,
                heads: 1,
                head_dim: layer.head_dim,
            })
            .collect();
        let original = SharedKvPlan::new(&geometry, 20, 512, 32, 512).unwrap();
        for capacity in [1, 5, 32] {
            let narrowed = original.narrowed(capacity).unwrap();
            let caches = narrowed.new_caches().unwrap();
            let payload: usize = caches
                .iter()
                .map(|cache| {
                    let (k, v) = cache.committed();
                    (k.len() + v.len()) * 4
                })
                .sum();
            assert_eq!(payload, capacity * 36864);
            assert_eq!(
                narrowed.allocated_cache_bytes(&caches).unwrap(),
                original.planned_narrowed_cache_bytes(capacity).unwrap()
            );
            println!(
                "GEMMA_SESSION_CAPACITY_E2B_ALLOCATION {}",
                serde_json::json!({
                "capacity":capacity,"cache_payload_bytes":payload,"producer_count":caches.len(),
                "whole_model_executed":false,"real_numerical_graph_qualified":false})
            );
        }
    }

    #[test]
    fn explicit_diagnostic_limit_keeps_default_and_row_scratch_bounded() {
        let plan = inspect(
            include_bytes!("../../../tests/fixtures/gemma4-e2b-metadata-v1/config.json"),
            include_bytes!("../../../tests/fixtures/gemma4-e2b-metadata-v1/index.json"),
            include_bytes!("../../../tests/fixtures/gemma4-e2b-metadata-v1/header.json"),
            3550670554,
            ModalityPolicy::TextOnly,
        )
        .unwrap();
        let ordinary = Layout::e2b(&plan).unwrap();
        assert_eq!(ordinary.positions, 32);
        for limit in [0, 1026, usize::MAX] {
            assert!(matches!(
                Layout::e2b_layers_for_limit(plan.layers(), limit),
                Err(Error::InvalidConfig("Gemma E2B diagnostic position limit"))
            ));
        }
        let mut previous = 0;
        let mut cache32 = 0;
        let mut scratch32 = 0;
        let mut cache512 = 0;
        let mut scratch512 = 0;
        for limit in [1, 32, 64, 256, 512, 513, 1024, 1025] {
            let layout = Layout::e2b_layers_for_limit(plan.layers(), limit).unwrap();
            assert_eq!(layout.positions, limit);
            let chunk = limit.min(32);
            let counts = Scratch::counts(&layout, limit, chunk).unwrap();
            assert_eq!(counts[19], 8 * chunk * limit);
            assert_eq!(counts[20], 8 * chunk * limit);
            assert_eq!(counts[21], chunk * 262144);
            assert_eq!(counts[22], chunk * 262144);
            assert!(Scratch::counts(&layout, limit + 1, chunk).is_err());
            assert!(Scratch::counts(&layout, limit, 33).is_err());
            let geometry: Vec<_> = layout
                .layers
                .iter()
                .map(|g| LayerKvGeometry {
                    kind: g.kind,
                    heads: 1,
                    head_dim: g.head_dim,
                })
                .collect();
            let cache = SharedKvPlan::new(&geometry, 20, 512, limit, limit.max(512))
                .unwrap()
                .planned_narrowed_cache_bytes(limit)
                .unwrap();
            let scratch = total_bytes(&counts).unwrap();
            let total = cache + scratch + limit * 4;
            assert!(total > previous);
            previous = total;
            if limit == 32 {
                cache32 = cache;
                scratch32 = scratch;
            }
            if limit == 512 {
                cache512 = cache;
                scratch512 = scratch;
                assert_eq!(cache - cache32, (512 - 32) * 36864);
                assert_eq!(scratch - scratch32, 2 * 8 * 32 * (512 - 32) * 4);
                assert_eq!(
                    cache - cache32 + scratch - scratch32 + (512 - 32) * 4,
                    18679680
                );
            }
            if limit > 512 {
                // Twelve local owners retain W512; only three full owners grow.
                assert_eq!(cache - cache512, (limit - 512) * 3 * 2 * 512 * 4);
                assert_eq!(scratch - scratch512, 2 * 8 * 32 * (limit - 512) * 4);
                if limit == 1025 {
                    assert_eq!(
                        cache - cache512 + scratch - scratch512 + (limit - 512) * 4,
                        7356420
                    );
                }
            }
        }
        assert!(Scratch::counts(&ordinary, 512, 32).is_err());
    }
}
