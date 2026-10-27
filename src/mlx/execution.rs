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
    match architecture {
        crate::mlx::dense::Architecture::Qwen3 => {
            c.validate()
                .map_err(|_| GeometryError::Invalid("invalid Qwen3 configuration"))?;
            let supported = match (
                c.hidden_size,
                c.intermediate_size,
                c.query_heads,
                c.kv_heads,
                c.head_dim,
                c.num_layers,
                c.vocab_size,
            ) {
                (32, 64, 4, 2, 16, 2, 64) => {
                    c.max_positions <= 32 && c.rope_theta == 10000.0 && c.rms_norm_eps == 1e-6
                }
                (1024, 3072, 16, 8, 128, 28, 151936) => {
                    c.max_positions <= 512 && c.rope_theta == 1e6 && c.rms_norm_eps == 1e-6
                }
                _ => false,
            };
            if !supported {
                return Err(GeometryError::Unsupported(
                    "Metal Qwen3 requires the admitted tiny/0.6B geometry, context and ordinary RoPE",
                ));
            }
            Ok(())
        }
        crate::mlx::dense::Architecture::Qwen2 => qwen_geometry(c),
        crate::mlx::dense::Architecture::Llama => {
            c.validate()
                .map_err(|_| GeometryError::Invalid("invalid Llama model configuration"))?;
            let supported = match (
                c.hidden_size,
                c.intermediate_size,
                c.query_heads,
                c.kv_heads,
                c.head_dim,
                c.num_layers,
                c.vocab_size,
            ) {
                (32, 64, 4, 2, 8, 2, 64) => {
                    c.rope_theta == 10000.0 && c.rms_norm_eps == 1e-6 && c.max_positions <= 32
                }
                (576, 1536, 9, 3, 64, 30, 49152) => {
                    c.rope_theta == 100000.0 && c.rms_norm_eps == 1e-5 && c.max_positions <= 512
                }
                _ => false,
            };
            if !supported {
                return Err(GeometryError::Unsupported(
                    "Metal Llama requires the admitted tiny/SmolLM2 geometry, context and ordinary RoPE",
                ));
            }
            Ok(())
        }
        crate::mlx::dense::Architecture::Mistral => {
            c.validate()
                .map_err(|_| GeometryError::Invalid("invalid Mistral model configuration"))?;
            let shape = (
                c.hidden_size,
                c.intermediate_size,
                c.query_heads,
                c.kv_heads,
                c.head_dim,
                c.num_layers,
                c.vocab_size,
            );
            let supported = match shape {
                (32, 64, 4, 2, 8, 2, 64) => {
                    c.rope_theta == 10000.0 && c.rms_norm_eps == 1e-6 && c.max_positions <= 32
                }
                (4096, 14336, 32, 8, 128, 32, 32768) => {
                    c.rope_theta == 1e6 && c.rms_norm_eps == 1e-5 && c.max_positions <= 512
                }
                _ => false,
            };
            if !supported {
                return Err(GeometryError::Unsupported(
                    "Metal requires qualified full-attention Mistral geometry and at most 512 positions",
                ));
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlx::dense::Architecture;

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
                assert!(matches!(
                    metal_geometry(&unqualified, architecture),
                    Err(GeometryError::Unsupported(_))
                ));
            }
            if architecture != Architecture::Qwen2 {
                assert!(
                    metal_geometry(
                        &model::Config {
                            max_positions: positions + 1,
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
