//! Device-independent geometry policy shared by MLX inspection and Metal loading.
//! No GPU, payload allocation or runtime-device qualification happens here.
use crate::model;

#[derive(Clone, Copy, Debug)]
pub(crate) enum GeometryError {
    Invalid(&'static str),
    Unsupported(&'static str),
}
impl GeometryError {
    pub(crate) fn message(self) -> &'static str {
        match self {
            Self::Invalid(message) | Self::Unsupported(message) => message,
        }
    }
}

/// Source semantics and static storage contracts precede device layout checks.
/// Compatibility remains untested until a separate artifact qualification.
pub(crate) fn metal_source_geometry(
    source: &crate::mlx::qwen::Source,
) -> Result<(), GeometryError> {
    metal_source_position_limits(source).map(|_| ())
}

/// Validated ordinary source policy, independent of runtime device/budget fit.
pub(crate) struct NativePositionLimits {
    pub graph_cap: Option<usize>,
    pub declared_window: Option<usize>,
    pub upper: usize,
}

/// Shares every source-level layout/frequency check with actual loading. Phi3's
/// declared window already bounds its full-attention prefix in the source parser;
/// it is not permission to implement a sliding cache or a larger position range.
pub(crate) fn metal_source_position_limits(
    source: &crate::mlx::qwen::Source,
) -> Result<NativePositionLimits, GeometryError> {
    source_position_limits(source, None)
}

/// Only the authenticated Llama quality loader uses this cold admission path.
/// Preserve all ordinary frequency and tensor-index checks; widen no app policy.
#[cfg(any(
    test,
    all(
        feature = "metal",
        any(target_os = "macos", target_os = "ios"),
        target_arch = "aarch64"
    )
))]
pub(crate) fn llama3_quality_geometry(
    source: &crate::mlx::qwen::Source,
) -> Result<(), GeometryError> {
    if source.architecture() != crate::mlx::dense::Architecture::Llama
        || source.config().max_positions > 1025
    {
        return Err(GeometryError::Unsupported(
            "Llama quality diagnostic requires at most1025 positions",
        ));
    }
    source_position_limits(source, Some(1025)).map(|_| ())
}

fn source_position_limits(
    source: &crate::mlx::qwen::Source,
    diagnostic_cap: Option<usize>,
) -> Result<NativePositionLimits, GeometryError> {
    use crate::mlx::{dense::Architecture, qwen::Storage};
    for factor in [
        source.dense_scales().embedding,
        source.dense_scales().residual,
    ] {
        if !half::f16::from_f32(factor).is_normal() {
            return Err(GeometryError::Unsupported(
                "dense Half coefficients must remain normal and positive",
            ));
        }
    }
    if source.position_plan() != model::PositionPlan::Ordinary
        && source.architecture() != Architecture::Llama
    {
        return Err(GeometryError::Unsupported(
            "scaled positions require the implemented Llama block composition",
        ));
    }
    let ordinary_cap = metal_geometry_context_cap(source.config(), source.architecture())?;
    let cap = diagnostic_cap.or(ordinary_cap);
    if diagnostic_cap.is_some() {
        check_context(source.config(), source.architecture(), cap)?;
    } else {
        metal_geometry(source.config(), source.architecture())?;
    }
    if matches!(
        source.architecture(),
        Architecture::Llama | Architecture::Phi3
    ) {
        let c = source.config();
        for pair in 0..c.head_dim / 2 {
            native_inverse_frequency(c.rope_theta, c.head_dim, pair, source.position_plan())?;
        }
        for logical in source.logical_tensors().values() {
            let spec = logical.tensor();
            let count = spec
                .shape()
                .iter()
                .try_fold(1usize, |n, &v| n.checked_mul(v))
                .and_then(|n| n.checked_add(7));
            if count.is_none_or(|n| n > u32::MAX as usize) {
                return Err(GeometryError::Unsupported(
                    "dense tensor extent exceeds UInt32 layout",
                ));
            }
            if let Storage::Affine(layout) = spec.storage() {
                layout.validate_gpu_indexing().map_err(|_| {
                    GeometryError::Unsupported("dense packed indexing exceeds UInt32")
                })?;
            }
        }
    }
    let declared_window = if source.architecture() == Architecture::Phi3 {
        source.declared_sliding_window()
    } else {
        None
    };
    let upper = source
        .declared_positions()
        .min(crate::weights::MAX_CONTEXT_LIMIT)
        .min(cap.unwrap_or(usize::MAX))
        .min(declared_window.unwrap_or(usize::MAX));
    if upper == 0 || source.config().max_positions > upper {
        return Err(GeometryError::Unsupported(
            "native source context exceeds declared or implemented positions",
        ));
    }
    Ok(NativePositionLimits {
        graph_cap: cap,
        declared_window,
        upper,
    })
}

/// Cold numerical representation contract shared by source inspection and the
/// immutable frequency upload. Reject FP32 zero/subnormal/overflow rather than
/// silently changing a declared mathematical transform through GPU flush-to-zero.
pub(crate) fn native_inverse_frequency(
    theta: f32,
    dimension: usize,
    pair: usize,
    plan: model::PositionPlan,
) -> Result<f32, GeometryError> {
    if !theta.is_finite()
        || theta <= 0.0
        || dimension == 0
        || !dimension.is_multiple_of(2)
        || pair >= dimension / 2
    {
        return Err(GeometryError::Invalid(
            "invalid native positional parameters",
        ));
    }
    // The initial native Llama/Phi path is bounded to positions 0..511. A base
    // >=1 keeps canonical inverse frequencies <=1 and angles finite, while
    // also excluding a subnormal theta input to the ordinary GPU power.
    if theta < 1.0 {
        return Err(GeometryError::Unsupported(
            "native dense rotary base requires theta at least 1",
        ));
    }
    let ordinary = f64::from(theta).powf(-2.0 * pair as f64 / dimension as f64);
    let inverse = match plan {
        model::PositionPlan::Ordinary => ordinary,
        model::PositionPlan::Llama3(scaling) => scaling
            .inverse_frequency(ordinary)
            .map_err(|_| GeometryError::Unsupported("invalid scaled positional frequency"))?,
    } as f32;
    if !inverse.is_normal() || inverse <= 0.0 {
        return Err(GeometryError::Unsupported(
            "positional frequency requires positive normal FP32",
        ));
    }
    Ok(inverse)
}
fn qwen_geometry(c: &model::Config) -> Result<(), GeometryError> {
    c.validate()
        .map_err(|_| GeometryError::Invalid("invalid model configuration"))?;
    let supported = match (
        c.hidden_size,
        c.intermediate_size,
        c.query_heads,
        c.kv_heads,
        c.head_dim,
    ) {
        (32, 64, 4, 2, 8) | (896, 4864, 14, 2, 64) => c.num_layers <= 24,
        (1536, 8960, 12, 2, 128) => c.num_layers == 28,
        _ => false,
    };
    if !supported || c.vocab_size > 151936 || c.rope_theta != 1e6 || c.rms_norm_eps != 1e-6 {
        return Err(GeometryError::Unsupported(
            "Metal requires supported tiny/Qwen geometry and theta/epsilon",
        ));
    }
    Ok(())
}
/// Quality experiments use a separate, explicit loader. This is a resource/
/// geometry boundary, not numerical qualification or application eligibility.
#[cfg(any(
    test,
    all(
        feature = "metal",
        any(target_os = "macos", target_os = "ios"),
        target_arch = "aarch64"
    )
))]
pub(crate) fn qwen3_quality_geometry(
    c: &model::Config,
    architecture: crate::mlx::dense::Architecture,
) -> Result<(), GeometryError> {
    c.validate()
        .map_err(|_| GeometryError::Invalid("invalid Qwen3 diagnostic configuration"))?;
    if architecture != crate::mlx::dense::Architecture::Qwen3
        || c.hidden_size != 1024
        || c.max_positions > 1025
    {
        return Err(GeometryError::Unsupported(
            "quality diagnostic requires Qwen3 0.6B and at most 1025 positions",
        ));
    }
    // Reuse every ordinary shape/rotary/epsilon restriction. Only the explicit
    // diagnostic's capacity differs; inspection and ordinary loading stay512.
    metal_geometry(
        &model::Config {
            max_positions: c.max_positions.min(512),
            ..c.clone()
        },
        architecture,
    )
}
/// Scoped to the explicit Qwen3 context diagnostic; never the shared CPU/loader
/// ceiling or an application capability. Ordinary and quality routes stay512/1025.
#[cfg(any(
    test,
    all(
        feature = "metal",
        any(target_os = "macos", target_os = "ios"),
        target_arch = "aarch64"
    )
))]
pub(crate) const QWEN3_CONTEXT_DIAGNOSTIC_LIMIT: usize = 40960;

#[cfg(any(
    test,
    all(
        feature = "metal",
        any(target_os = "macos", target_os = "ios"),
        target_arch = "aarch64"
    )
))]
pub(crate) fn qwen3_context_geometry(
    c: &model::Config,
    architecture: crate::mlx::dense::Architecture,
) -> Result<(), GeometryError> {
    if c.max_positions == 0 || c.max_positions > QWEN3_CONTEXT_DIAGNOSTIC_LIMIT {
        return Err(GeometryError::Unsupported(
            "Qwen3 context diagnostic requires1..=40960 positions",
        ));
    }
    // Reuse every original0.6B family/shape/rotary/epsilon restriction. This
    // changes only the explicit diagnostic's position admission.
    qwen3_quality_geometry(
        &model::Config {
            max_positions: c.max_positions.min(1025),
            ..c.clone()
        },
        architecture,
    )
}

/// Keep bias-free geometry admission explicit and before GPU creation. Wider dense
/// shapes/rotary variants require their own numerical and resource qualification.
pub(crate) fn metal_geometry(
    c: &model::Config,
    architecture: crate::mlx::dense::Architecture,
) -> Result<(), GeometryError> {
    let cap = metal_geometry_context_cap(c, architecture)?;
    check_context(c, architecture, cap)
}

fn check_context(
    c: &model::Config,
    architecture: crate::mlx::dense::Architecture,
    cap: Option<usize>,
) -> Result<(), GeometryError> {
    if cap.is_some_and(|limit| c.max_positions > limit) {
        use crate::mlx::dense::Architecture;
        return Err(GeometryError::Unsupported(match architecture {
            Architecture::Qwen3 => {
                "Metal Qwen3 requires the admitted tiny/0.6B geometry, context and ordinary RoPE"
            }
            Architecture::Mistral => {
                "Metal requires qualified full-attention Mistral geometry and at most 512 positions"
            }
            Architecture::Llama
            | Architecture::Phi3
            | Architecture::SmolLm3
            | Architecture::Granite => {
                "Metal dense initial implemented context is at most 512 positions"
            }
            Architecture::Qwen2 => unreachable!("Qwen2 has no extra graph cap"),
        }));
    }
    Ok(())
}

/// Select the existing ordinary graph bound once, along with its shape checks.
/// None means no additional graph cap: declarations/global parser bounds remain.
fn metal_geometry_context_cap(
    c: &model::Config,
    architecture: crate::mlx::dense::Architecture,
) -> Result<Option<usize>, GeometryError> {
    use crate::mlx::dense::Architecture;
    match architecture {
        Architecture::Qwen3 => {
            c.validate()
                .map_err(|_| GeometryError::Invalid("invalid Qwen3 configuration"))?;
            let cap = match (
                c.hidden_size,
                c.intermediate_size,
                c.query_heads,
                c.kv_heads,
                c.head_dim,
                c.num_layers,
                c.vocab_size,
            ) {
                (32, 64, 4, 2, 16, 2, 64) if c.rope_theta == 10000.0 && c.rms_norm_eps == 1e-6 => {
                    32
                }
                (1024, 3072, 16, 8, 128, 28, 151936)
                    if c.rope_theta == 1e6 && c.rms_norm_eps == 1e-6 =>
                {
                    512
                }
                _ => {
                    return Err(GeometryError::Unsupported(
                        "Metal Qwen3 requires the admitted tiny/0.6B geometry, context and ordinary RoPE",
                    ));
                }
            };
            Ok(Some(cap))
        }
        Architecture::Qwen2 => {
            qwen_geometry(c)?;
            Ok(None)
        }
        Architecture::Llama
        | Architecture::Phi3
        | Architecture::SmolLm3
        | Architecture::Granite => {
            c.validate()
                .map_err(|_| GeometryError::Invalid("invalid dense model configuration"))?;
            if !c.rms_norm_eps.is_normal() {
                return Err(GeometryError::Unsupported(
                    "native dense RMS epsilon requires positive normal FP32",
                ));
            }
            for n in [
                c.hidden_size,
                c.intermediate_size,
                c.vocab_size,
                c.query_heads.saturating_mul(c.head_dim),
                c.kv_heads.saturating_mul(c.head_dim),
            ] {
                if n.checked_add(7).is_none_or(|v| v > u32::MAX as usize) {
                    return Err(GeometryError::Unsupported(
                        "dense activation width exceeds UInt32",
                    ));
                }
            }
            Ok(Some(if architecture == Architecture::Phi3 {
                super::phi3::IMPLEMENTED_POSITIONS
            } else {
                512
            }))
        }
        Architecture::Mistral => {
            c.validate()
                .map_err(|_| GeometryError::Invalid("invalid Mistral model configuration"))?;
            let cap = match (
                c.hidden_size,
                c.intermediate_size,
                c.query_heads,
                c.kv_heads,
                c.head_dim,
                c.num_layers,
                c.vocab_size,
            ) {
                (32, 64, 4, 2, 8, 2, 64) if c.rope_theta == 10000.0 && c.rms_norm_eps == 1e-6 => 32,
                (4096, 14336, 32, 8, 128, 32, 32768)
                    if c.rope_theta == 1e6 && c.rms_norm_eps == 1e-5 =>
                {
                    512
                }
                _ => {
                    return Err(GeometryError::Unsupported(
                        "Metal requires qualified full-attention Mistral geometry and at most 512 positions",
                    ));
                }
            };
            Ok(Some(cap))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlx::dense::Architecture;
    #[test]
    fn llama_contract_accepts_implemented_blocks_without_a_checkpoint_tuple() {
        let c = model::Config {
            hidden_size: 2048,
            intermediate_size: 8192,
            query_heads: 32,
            kv_heads: 8,
            head_dim: 64,
            num_layers: 16,
            vocab_size: 128256,
            max_positions: 512,
            rope_theta: 500000.,
            rms_norm_eps: 1e-5,
        };
        assert!(metal_geometry(&c, Architecture::Llama).is_ok());
        assert!(
            metal_geometry(
                &model::Config {
                    rms_norm_eps: f32::MIN_POSITIVE,
                    ..c.clone()
                },
                Architecture::Llama
            )
            .is_ok()
        );
        assert!(matches!(
            metal_geometry(
                &model::Config {
                    rms_norm_eps: f32::from_bits(1),
                    ..c.clone()
                },
                Architecture::Llama
            ),
            Err(GeometryError::Unsupported(_))
        ));
        assert!(
            metal_geometry(
                &model::Config {
                    max_positions: 513,
                    ..c.clone()
                },
                Architecture::Llama
            )
            .is_err()
        );
        let scaled =
            model::PositionPlan::Llama3(crate::cpu::Llama3Scaling::new(32., 1., 4., 8192).unwrap());
        for plan in [model::PositionPlan::Ordinary, scaled] {
            for theta in [f32::from_bits(1), f32::MIN_POSITIVE, 0.5] {
                assert!(matches!(
                    native_inverse_frequency(theta, 128, 63, plan),
                    Err(GeometryError::Unsupported(_))
                ));
            }
            assert!(native_inverse_frequency(1., 128, 63, plan).is_ok());
        }
        for pair in 0..32 {
            assert!(
                native_inverse_frequency(c.rope_theta, 64, pair, scaled)
                    .unwrap()
                    .is_normal()
            );
        }
        assert!(
            native_inverse_frequency(
                1e38,
                128,
                63,
                model::PositionPlan::Llama3(
                    crate::cpu::Llama3Scaling::new(f32::MAX, 1., 4., 8192).unwrap()
                )
            )
            .is_err()
        );
        use crate::mlx::affine::{Layout, MetadataDtype};
        let columns = (u32::MAX as usize / 8) / 32 * 32;
        assert!(
            Layout::new(1, columns, 8, 32, MetadataDtype::F16)
                .unwrap()
                .validate_gpu_indexing()
                .is_ok()
        );
        if let Some(next) = columns.checked_add(32) {
            assert!(
                Layout::new(1, next, 8, 32, MetadataDtype::F16)
                    .and_then(|layout| layout.validate_gpu_indexing())
                    .is_err()
            );
        }
    }

    #[test]
    fn qwen3_long_context_scope_preserves_all_existing_limits() {
        let mut c = model::Config {
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
        assert_eq!(crate::weights::MAX_CONTEXT_LIMIT, 32768);
        for capacity in [
            1, 512, 513, 1025, 1026, 4096, 8192, 16384, 32768, 32769, 40959, 40960,
        ] {
            c.max_positions = capacity;
            assert!(qwen3_context_geometry(&c, Architecture::Qwen3).is_ok());
            assert_eq!(
                metal_geometry(&c, Architecture::Qwen3).is_ok(),
                capacity <= 512
            );
            assert_eq!(
                qwen3_quality_geometry(&c, Architecture::Qwen3).is_ok(),
                capacity <= 1025
            );
        }
        for capacity in [0, 40961, usize::MAX] {
            c.max_positions = capacity;
            assert!(qwen3_context_geometry(&c, Architecture::Qwen3).is_err());
        }
        c.max_positions = 40960;
        for architecture in [
            Architecture::Qwen2,
            Architecture::Llama,
            Architecture::Mistral,
        ] {
            assert!(qwen3_context_geometry(&c, architecture).is_err());
        }
        for bad in [
            model::Config {
                hidden_size: 32,
                ..c.clone()
            },
            model::Config {
                kv_heads: 4,
                ..c.clone()
            },
            model::Config {
                num_layers: 27,
                ..c.clone()
            },
            model::Config {
                rope_theta: 10000.,
                ..c.clone()
            },
            model::Config {
                rms_norm_eps: 1e-5,
                ..c.clone()
            },
        ] {
            assert!(qwen3_context_geometry(&bad, Architecture::Qwen3).is_err());
        }
    }

    #[test]
    fn qwen3_quality_context_is_explicit_and_does_not_expand_ordinary_admission() {
        let c = model::Config {
            hidden_size: 1024,
            intermediate_size: 3072,
            query_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            num_layers: 28,
            vocab_size: 151936,
            max_positions: 1025,
            rope_theta: 1e6,
            rms_norm_eps: 1e-6,
        };
        for capacity in [1, 32, 512, 513, 1024, 1025] {
            let candidate = model::Config {
                max_positions: capacity,
                ..c.clone()
            };
            assert!(qwen3_quality_geometry(&candidate, Architecture::Qwen3).is_ok());
            assert_eq!(
                metal_geometry(&candidate, Architecture::Qwen3).is_ok(),
                capacity <= 512
            );
        }
        for invalid in [0, 1026, usize::MAX] {
            assert!(
                qwen3_quality_geometry(
                    &model::Config {
                        max_positions: invalid,
                        ..c.clone()
                    },
                    Architecture::Qwen3
                )
                .is_err()
            );
        }
        for architecture in [
            Architecture::Qwen2,
            Architecture::Llama,
            Architecture::Mistral,
        ] {
            assert!(qwen3_quality_geometry(&c, architecture).is_err());
        }
        for unsupported in [
            model::Config {
                hidden_size: 32,
                ..c.clone()
            },
            model::Config {
                kv_heads: 4,
                ..c.clone()
            },
            model::Config {
                intermediate_size: 4096,
                ..c.clone()
            },
            model::Config {
                rope_theta: 10000.0,
                ..c.clone()
            },
            model::Config {
                rms_norm_eps: 1e-5,
                ..c.clone()
            },
        ] {
            assert!(qwen3_quality_geometry(&unsupported, Architecture::Qwen3).is_err());
        }
    }

    #[test]
    fn all_current_shape_policies_and_error_categories_are_device_independent() {
        for (architecture, h, ffn, q, kv, d, layers, vocabulary, positions, theta, epsilon) in [
            (Architecture::Qwen2, 32, 64, 4, 2, 8, 2, 64, 32, 1e6, 1e-6),
            (
                Architecture::Qwen2,
                896,
                4864,
                14,
                2,
                64,
                24,
                151936,
                32768,
                1e6,
                1e-6,
            ),
            (
                Architecture::Qwen2,
                1536,
                8960,
                12,
                2,
                128,
                28,
                151936,
                32768,
                1e6,
                1e-6,
            ),
            (
                Architecture::Qwen3,
                32,
                64,
                4,
                2,
                16,
                2,
                64,
                32,
                10000.0,
                1e-6,
            ),
            (
                Architecture::Qwen3,
                1024,
                3072,
                16,
                8,
                128,
                28,
                151936,
                512,
                1e6,
                1e-6,
            ),
            (
                Architecture::Llama,
                32,
                64,
                4,
                2,
                8,
                2,
                64,
                32,
                10000.0,
                1e-6,
            ),
            (
                Architecture::Llama,
                576,
                1536,
                9,
                3,
                64,
                30,
                49152,
                512,
                100000.0,
                1e-5,
            ),
            (
                Architecture::Mistral,
                32,
                64,
                4,
                2,
                8,
                2,
                64,
                32,
                10000.0,
                1e-6,
            ),
            (
                Architecture::Mistral,
                4096,
                14336,
                32,
                8,
                128,
                32,
                32768,
                512,
                1e6,
                1e-5,
            ),
        ] {
            let c = model::Config {
                hidden_size: h,
                intermediate_size: ffn,
                query_heads: q,
                kv_heads: kv,
                head_dim: d,
                num_layers: layers,
                vocab_size: vocabulary,
                max_positions: positions,
                rope_theta: theta,
                rms_norm_eps: epsilon,
            };
            assert!(metal_geometry(&c, architecture).is_ok());
            let expected_cap = match architecture {
                Architecture::Qwen2 => None,
                Architecture::Llama
                | Architecture::Phi3
                | Architecture::SmolLm3
                | Architecture::Granite => Some(512),
                Architecture::Qwen3 | Architecture::Mistral => Some(positions),
            };
            assert_eq!(
                metal_geometry_context_cap(&c, architecture).unwrap(),
                expected_cap
            );
            let invalid = model::Config {
                max_positions: 0,
                ..c.clone()
            };
            assert!(matches!(
                metal_geometry(&invalid, architecture),
                Err(GeometryError::Invalid(_))
            ));
            for unqualified in [
                model::Config {
                    rope_theta: 12345.0,
                    ..c.clone()
                },
                model::Config {
                    rms_norm_eps: epsilon * 2.0,
                    ..c.clone()
                },
            ] {
                if architecture == Architecture::Llama {
                    assert!(metal_geometry(&unqualified, architecture).is_ok());
                } else {
                    assert!(matches!(
                        metal_geometry(&unqualified, architecture),
                        Err(GeometryError::Unsupported(_))
                    ));
                }
            }
            if architecture != Architecture::Qwen2 {
                assert!(
                    metal_geometry(
                        &model::Config {
                            max_positions: if architecture == Architecture::Llama {
                                513
                            } else {
                                positions + 1
                            },
                            ..c
                        },
                        architecture
                    )
                    .is_err()
                );
            }
        }
    }
}
