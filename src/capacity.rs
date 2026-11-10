//! Cold checked storage and capacity arithmetic, not model/backend admission.
//!
//! Callers resolve supported blocks, physical owners, strides and device facts
//! before planning. A shared consumer is not another owner. No tensor allocation,
//! architecture-name dispatch, dtype conversion or numerical execution occurs here.
use crate::cpu::{self, nonzero, product};

/// Only storage formats implemented by the current cache owners.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvStorage {
    F16,
    Bf16,
    F32,
}
impl KvStorage {
    pub(crate) fn bytes(self) -> usize {
        match self {
            Self::F16 | Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

/// Physical retention of one existing attention-state block. Caller-owned new
/// chunk staging and scores are separate scratch, including during ring updates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Retention {
    Full,
    Sliding { window: usize },
}

/// One separate head-major K/V pair; `copies` counts physical producers only.
/// GQA uses KV heads, never expanded query heads. Guards belong to EACH array.
#[derive(Clone, Copy, Debug)]
pub struct KvOwner {
    pub heads: usize,
    pub head_dim: usize,
    pub storage: KvStorage,
    pub retention: Retention,
    pub guard_elements: usize,
    pub copies: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvBytes {
    pub elements_per_array: usize,
    pub bytes_per_array: usize,
    pub total_bytes: usize,
}
impl KvOwner {
    pub fn at_capacity(self, capacity: usize) -> Result<KvBytes, cpu::Error> {
        for (n, name) in [
            (capacity, "KV capacity"),
            (self.heads, "KV heads"),
            (self.head_dim, "KV head dimension"),
            (self.copies, "KV physical owners"),
        ] {
            nonzero(n, name)?;
        }
        let retained = match self.retention {
            Retention::Full => capacity,
            Retention::Sliding { window } => {
                nonzero(window, "KV window")?;
                capacity.min(window)
            }
        };
        let elements_per_array = product(product(self.heads, retained)?, self.head_dim)?
            .checked_add(self.guard_elements)
            .ok_or(cpu::Error::SizeOverflow)?;
        let bytes_per_array = product(elements_per_array, self.storage.bytes())?;
        if bytes_per_array > isize::MAX as usize {
            return Err(cpu::Error::SizeOverflow);
        }
        let total_bytes = product(product(bytes_per_array, 2)?, self.copies)?;
        Ok(KvBytes {
            elements_per_array,
            bytes_per_array,
            total_bytes,
        })
    }
}

/// Includes immutable weights once and ALL session owners/allowances. This is
/// an allocation estimate, not a whole-process peak or an application allowance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Footprint {
    pub admitted_bytes: usize,
    pub largest_buffer_bytes: usize,
    /// Required when a KV ceiling is supplied; omitted for legacy reports.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_bytes: Option<usize>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Limits {
    pub budget_bytes: usize,
    pub maximum_buffer_bytes: usize,
    /// None is unlimited. Some(0) is an explicit zero-byte ceiling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_budget_bytes: Option<usize>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Point {
    pub capacity: usize,
    pub footprint: Footprint,
    pub fits_budget: bool,
    pub fits_buffer: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fits_kv_budget: Option<bool>,
}
impl Point {
    fn fits(self) -> bool {
        self.fits_budget && self.fits_buffer && self.fits_kv_budget.unwrap_or(true)
    }
}
#[derive(Debug, PartialEq, Eq, serde::Serialize)]
pub struct Report {
    pub minimum: usize,
    pub maximum: usize,
    pub limits: Limits,
    pub selected: Option<Point>,
    /// selected+1, or minimum when none fits; None means the upper bound fits.
    pub first_excluded: Option<Point>,
}
#[derive(Debug, PartialEq, Eq)]
pub enum SearchError<E> {
    InvalidBounds,
    MissingKvFootprint,
    Planner(E),
}

/// Search only an already implemented/admitted interval with fixed modes,
/// chunk, offset and layouts. All constrained footprint fields MUST be nondecreasing in
/// capacity. Current full/ring caches and fixed dense scratch satisfy this;
/// a future nonmonotone allocator must supply a different search procedure.
/// Planner failures (including overflow) are errors, never interpreted as OOM.
/// The callback is statically dispatched, cold, allocation-free arithmetic.
pub fn search<E>(
    minimum: usize,
    maximum: usize,
    limits: Limits,
    mut plan: impl FnMut(usize) -> Result<Footprint, E>,
) -> Result<Report, SearchError<E>> {
    if minimum == 0 || minimum > maximum {
        return Err(SearchError::InvalidBounds);
    }
    let mut evaluate = |capacity| -> Result<Point, SearchError<E>> {
        let footprint = plan(capacity).map_err(SearchError::Planner)?;
        let fits_kv_budget = match limits.kv_budget_bytes {
            Some(limit) => {
                Some(footprint.kv_bytes.ok_or(SearchError::MissingKvFootprint)? <= limit)
            }
            None => None,
        };
        Ok(Point {
            capacity,
            footprint,
            fits_kv_budget,
            fits_budget: footprint.admitted_bytes <= limits.budget_bytes,
            fits_buffer: footprint.largest_buffer_bytes <= limits.maximum_buffer_bytes
                && footprint.largest_buffer_bytes <= isize::MAX as usize,
        })
    };
    let upper = evaluate(maximum)?;
    if upper.fits() {
        return Ok(Report {
            minimum,
            maximum,
            limits,
            selected: Some(upper),
            first_excluded: None,
        });
    }
    let lower = if minimum == maximum {
        upper
    } else {
        evaluate(minimum)?
    };
    if !lower.fits() {
        return Ok(Report {
            minimum,
            maximum,
            limits,
            selected: None,
            first_excluded: Some(lower),
        });
    }
    let (mut yes, mut no) = (lower, upper);
    while no.capacity - yes.capacity > 1 {
        let middle = yes.capacity + (no.capacity - yes.capacity) / 2;
        let point = evaluate(middle)?;
        if point.fits() {
            yes = point;
        } else {
            no = point;
        }
    }
    Ok(Report {
        minimum,
        maximum,
        limits,
        selected: Some(yes),
        first_excluded: Some(no),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concrete_cache_layouts_preserve_old_bytes() {
        let base = KvOwner {
            heads: 8,
            head_dim: 128,
            storage: KvStorage::F16,
            retention: Retention::Full,
            guard_elements: 7,
            copies: 28,
        };
        let half = base.at_capacity(512).unwrap();
        assert_eq!(half.elements_per_array, 524295);
        assert_eq!(half.bytes_per_array, 1048590);
        assert_eq!(half.total_bytes, 58721040);
        let fp32 = KvOwner {
            storage: KvStorage::F32,
            guard_elements: 0,
            ..base
        }
        .at_capacity(512)
        .unwrap();
        assert_eq!(fp32.total_bytes, 117440512);
        let local = KvOwner {
            retention: Retention::Sliding { window: 32 },
            copies: 1,
            ..base
        };
        assert_eq!(local.at_capacity(64), local.at_capacity(8192));
        assert!(
            local.at_capacity(16).unwrap().total_bytes < local.at_capacity(32).unwrap().total_bytes
        );
        assert!(
            KvOwner {
                copies: usize::MAX,
                ..base
            }
            .at_capacity(512)
            .is_err()
        );
        assert!(
            KvOwner {
                guard_elements: usize::MAX,
                ..base
            }
            .at_capacity(1)
            .is_err()
        );
        assert!(base.at_capacity(usize::MAX).is_err());
        assert!(base.at_capacity(0).is_err());
    }

    #[test]
    fn kv_ceiling_matches_independent_full_and_ring_owner_enumeration() {
        let owners = [
            KvOwner {
                heads: 2,
                head_dim: 3,
                storage: KvStorage::F16,
                retention: Retention::Full,
                guard_elements: 1,
                copies: 2,
            },
            KvOwner {
                heads: 1,
                head_dim: 2,
                storage: KvStorage::F32,
                retention: Retention::Sliding { window: 3 },
                guard_elements: 0,
                copies: 1,
            },
        ];
        for budget in [0, 80, 300, 2000] {
            for buffer in [0, 64, 128, 4096] {
                for ceiling in (0..1100).step_by(7) {
                    let report = search(
                        3,
                        16,
                        Limits {
                            budget_bytes: budget,
                            maximum_buffer_bytes: buffer,
                            kv_budget_bytes: Some(ceiling),
                        },
                        |n| {
                            let kv = owners.iter().try_fold(0usize, |sum, owner| {
                                sum.checked_add(owner.at_capacity(n)?.total_bytes)
                                    .ok_or(cpu::Error::SizeOverflow)
                            })?;
                            Ok::<_, cpu::Error>(Footprint {
                                admitted_bytes: 10 + kv + 3 * n,
                                largest_buffer_bytes: 64.max(12 * n + 2),
                                kv_bytes: Some(kv),
                            })
                        },
                    )
                    .unwrap();
                    let expected = (3..=16)
                        .filter(|&n| {
                            let kv = 48 * n + 8 + 16 * n.min(3);
                            kv <= ceiling
                                && 10 + kv + 3 * n <= budget
                                && 64.max(12 * n + 2) <= buffer
                        })
                        .next_back();
                    assert_eq!(report.selected.map(|p| p.capacity), expected);
                    assert_eq!(
                        report.first_excluded.map(|p| p.capacity),
                        match expected {
                            Some(16) => None,
                            Some(n) => Some(n + 1),
                            None => Some(3),
                        }
                    );
                }
            }
        }
    }

    #[test]
    fn kv_ceiling_exact_boundary_zero_plateau_and_missing_accounting() {
        let limits = Limits {
            budget_bytes: 1000,
            maximum_buffer_bytes: 1000,
            kv_budget_bytes: Some(24),
        };
        let plan = |n: usize| {
            Ok::<_, ()>(Footprint {
                admitted_bytes: 100 + n,
                largest_buffer_bytes: 10,
                kv_bytes: Some(8 * n.min(3)),
            })
        };
        assert_eq!(
            search(1, 32, limits, plan)
                .unwrap()
                .selected
                .unwrap()
                .capacity,
            32
        );
        let below = search(
            1,
            32,
            Limits {
                kv_budget_bytes: Some(23),
                ..limits
            },
            plan,
        )
        .unwrap();
        assert_eq!(below.selected.unwrap().capacity, 2);
        assert_eq!(below.first_excluded.unwrap().fits_kv_budget, Some(false));
        assert!(
            search(
                1,
                32,
                Limits {
                    kv_budget_bytes: Some(0),
                    ..limits
                },
                plan
            )
            .unwrap()
            .selected
            .is_none()
        );
        assert_eq!(
            search(1, 32, limits, |_| Ok::<_, ()>(Footprint {
                admitted_bytes: 1,
                largest_buffer_bytes: 1,
                kv_bytes: None
            })),
            Err(SearchError::MissingKvFootprint)
        );
        assert_eq!(
            search(1, 32, limits, |_| Err::<Footprint, _>("size overflow")),
            Err(SearchError::Planner("size overflow"))
        );
        let unlimited = search(
            1,
            32,
            Limits {
                kv_budget_bytes: None,
                ..limits
            },
            |_| {
                Ok::<_, ()>(Footprint {
                    admitted_bytes: 1,
                    largest_buffer_bytes: 1,
                    kv_bytes: None,
                })
            },
        )
        .unwrap();
        let json = serde_json::to_value(&unlimited).unwrap();
        assert!(json["limits"].get("kv_budget_bytes").is_none());
        assert!(json["selected"].get("fits_kv_budget").is_none());
        assert!(json["selected"]["footprint"].get("kv_bytes").is_none());
    }

    #[test]
    fn existing_dense_and_shared_planners_use_unique_storage() {
        use crate::model::shared_kv::{KvKind, LayerKvGeometry, SharedKvPlan};
        use crate::model::window_cache::WindowKvCache;
        use crate::model::{Config, KvCache};
        // Current Qwen2/Qwen3/Llama/Mistral geometry; no family-name sizing rule.
        for (layers, query_heads, kv_heads, dim) in [
            (24, 14, 2, 64),
            (28, 16, 8, 128),
            (30, 9, 3, 64),
            (32, 32, 8, 128),
        ] {
            let c = Config {
                vocab_size: 128,
                hidden_size: query_heads * dim,
                intermediate_size: 256,
                num_layers: layers,
                query_heads,
                kv_heads,
                head_dim: dim,
                max_positions: 512,
                rms_norm_eps: 1e-6,
                rope_theta: 1e6,
            };
            for n in [1, 32, 512] {
                assert_eq!(
                    KvCache::planned_bytes(&c, n).unwrap(),
                    layers * 2 * kv_heads * n * dim * 4
                );
            }
        }
        let local = LayerKvGeometry {
            kind: KvKind::Sliding,
            heads: 1,
            head_dim: 1,
        };
        let full = LayerKvGeometry {
            kind: KvKind::Full,
            ..local
        };
        let shared = SharedKvPlan::new(&[local, full, local, full], 2, 3, 8, 12).unwrap();
        assert_eq!(shared.producer_count(), 2);
        assert_eq!(shared.producer_for(2).unwrap(), 0);
        assert_eq!(shared.producer_for(3).unwrap(), 1);
        let metadata =
            std::mem::size_of::<Vec<WindowKvCache>>() + 2 * std::mem::size_of::<WindowKvCache>();
        assert_eq!(shared.planned_cache_bytes().unwrap(), metadata + 88);
        for n in 1..=8 {
            assert_eq!(
                shared.planned_narrowed_cache_bytes(n).unwrap(),
                metadata + 8 * (n.min(3) + n)
            );
        }
    }

    #[test]
    fn capacity_budget_buffer_and_adjacent_witness() {
        // Compare the binary result with exhaustive independent enumeration.
        for budget in 0..100 {
            for buffer in 0..40 {
                let limits = Limits {
                    budget_bytes: budget,
                    maximum_buffer_bytes: buffer,
                    kv_budget_bytes: None,
                };
                let report = search(1, 32, limits, |n| {
                    Ok::<_, ()>(Footprint {
                        admitted_bytes: 10 + 3 * n,
                        largest_buffer_bytes: 2 * n,
                        kv_bytes: None,
                    })
                })
                .unwrap();
                let expected = (1..=32)
                    .filter(|&n| 10 + 3 * n <= budget && 2 * n <= buffer)
                    .next_back();
                assert_eq!(report.selected.map(|p| p.capacity), expected);
                assert_eq!(
                    report.first_excluded.map(|p| p.capacity),
                    match expected {
                        Some(32) => None,
                        Some(n) => Some(n + 1),
                        None => Some(1),
                    }
                );
            }
        }
        assert_eq!(
            search(
                0,
                1,
                Limits {
                    budget_bytes: 1,
                    maximum_buffer_bytes: 1,
                    kv_budget_bytes: None,
                },
                |_| Ok::<_, ()>(Footprint {
                    admitted_bytes: 0,
                    largest_buffer_bytes: 0,
                    kv_bytes: None,
                })
            ),
            Err(SearchError::InvalidBounds)
        );
        assert_eq!(
            search(
                1,
                10,
                Limits {
                    budget_bytes: 1,
                    maximum_buffer_bytes: 1,
                    kv_budget_bytes: None,
                },
                |_| Err::<Footprint, _>("overflow")
            ),
            Err(SearchError::Planner("overflow"))
        );
        let report = search(
            1,
            1,
            Limits {
                budget_bytes: usize::MAX,
                maximum_buffer_bytes: usize::MAX,
                kv_budget_bytes: None,
            },
            |_| {
                Ok::<_, ()>(Footprint {
                    admitted_bytes: 1,
                    largest_buffer_bytes: isize::MAX as usize + 1,
                    kv_bytes: None,
                })
            },
        )
        .unwrap();
        assert!(report.selected.is_none());
    }
}
