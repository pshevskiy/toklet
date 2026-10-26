//! Private tiny fixture adapter. Cannot admit a public E2B graph or source.
use super::*;
use crate::mlx::affine::Matrix;

pub(crate) struct Uploaded<'a> {
    pub matrices: Vec<Weights>,
    pub description: graph::Weights<'a, usize>,
}
impl<'a> Uploaded<'a> {
    pub(crate) fn new(context: &Context, input: graph::Weights<'a>) -> Self {
        let mut matrices = Vec::new();
        let mut m = |matrix: Matrix<'_>| {
            let index = matrices.len();
            let (codes, scales, biases) = matrix.original_tensors();
            let weight = context
                .weights_from_fill(
                    matrix.layout(),
                    TILE,
                    32 * 1024 * 1024,
                    |first, _, a, b, c| {
                        assert_eq!(first, 0);
                        a.copy_from_slice(codes);
                        b.copy_from_slice(scales);
                        c.copy_from_slice(biases);
                        Ok(())
                    },
                )
                .unwrap();
            matrices.push(weight);
            index
        };
        let description = graph::Weights {
            embedding: m(input.embedding),
            per_layer_embedding: m(input.per_layer_embedding),
            per_layer_projection: m(input.per_layer_projection),
            per_layer_norm: input.per_layer_norm,
            final_norm: input.final_norm,
            layers: input
                .layers
                .into_iter()
                .map(|w| graph::LayerWeights {
                    input_norm: w.input_norm,
                    q: m(w.q),
                    q_norm: w.q_norm,
                    o: m(w.o),
                    kv: w.kv.map(|kv| graph::KvWeights {
                        k: m(kv.k),
                        v: m(kv.v),
                        k_norm: kv.k_norm,
                    }),
                    post_attention_norm: w.post_attention_norm,
                    pre_ff_norm: w.pre_ff_norm,
                    post_ff_norm: w.post_ff_norm,
                    ff_gate: m(w.ff_gate),
                    ff_up: m(w.ff_up),
                    ff_down: m(w.ff_down),
                    ple_gate: m(w.ple_gate),
                    ple_projection: m(w.ple_projection),
                    post_ple_norm: w.post_ple_norm,
                    scalar: w.scalar,
                })
                .collect(),
        };
        Self {
            matrices,
            description,
        }
    }
    pub(crate) fn norms(&self) -> Norms<'_> {
        let mut shapes = Vec::new();
        let mut slots = Vec::new();
        for (i, m) in self.matrices.iter().enumerate() {
            if i == self.description.per_layer_embedding {
                slots.push(usize::MAX);
                continue;
            }
            let layout = m.layout();
            let slot = if let Some(s) = shapes.iter().position(|v| *v == layout) {
                s
            } else {
                shapes.push(layout);
                shapes.len() - 1
            };
            slots.push(slot);
        }
        let owned = size_of::<Norms<'_>>()
            + shapes.capacity() * size_of::<crate::mlx::affine::Layout>()
            + slots.capacity() * size_of::<usize>();
        Norms {
            source: None,
            matrices: &self.matrices,
            values: Vec::new(),
            shapes,
            slots,
            gather_indices: [
                self.description.embedding,
                self.description.per_layer_embedding,
            ],
            owned,
            payload: 0,
        }
    }
    pub(crate) fn graph_weights<'n, 'weights>(
        &self,
        norms: &'n Norms<'weights>,
    ) -> graph::Weights<'n, MatrixRef<'n, 'weights>>
    where
        'a: 'n,
    {
        let m = |index| norms.matrix(index).unwrap();
        let input = &self.description;
        graph::Weights {
            embedding: m(input.embedding),
            per_layer_embedding: m(input.per_layer_embedding),
            per_layer_projection: m(input.per_layer_projection),
            per_layer_norm: input.per_layer_norm,
            final_norm: input.final_norm,
            layers: input
                .layers
                .iter()
                .map(|w| graph::LayerWeights {
                    input_norm: w.input_norm,
                    q: m(w.q),
                    q_norm: w.q_norm,
                    o: m(w.o),
                    kv: w.kv.as_ref().map(|kv| graph::KvWeights {
                        k: m(kv.k),
                        v: m(kv.v),
                        k_norm: kv.k_norm,
                    }),
                    post_attention_norm: w.post_attention_norm,
                    pre_ff_norm: w.pre_ff_norm,
                    post_ff_norm: w.post_ff_norm,
                    ff_gate: m(w.ff_gate),
                    ff_up: m(w.ff_up),
                    ff_down: m(w.ff_down),
                    ple_gate: m(w.ple_gate),
                    ple_projection: m(w.ple_projection),
                    post_ple_norm: w.post_ple_norm,
                    scalar: w.scalar,
                })
                .collect(),
        }
    }
    pub(crate) fn backing(&self, norms: &Norms<'_>) -> graph::Backing {
        let physical = self
            .matrices
            .iter()
            .flat_map(Weights::views)
            .map(|v| (v.codes.len + v.scales.len + v.biases.len) * 4)
            .sum::<usize>();
        let mut norm_values =
            self.description.per_layer_norm.len() + self.description.final_norm.len();
        for w in &self.description.layers {
            norm_values += [
                w.input_norm.len(),
                w.q_norm.len(),
                w.post_attention_norm.len(),
                w.pre_ff_norm.len(),
                w.post_ff_norm.len(),
                w.post_ple_norm.len(),
            ]
            .iter()
            .sum::<usize>();
            if let Some(kv) = &w.kv {
                norm_values += kv.k_norm.len();
            }
        }
        let owners = self.matrices.capacity() * size_of::<Weights>()
            + self
                .matrices
                .iter()
                .map(|w| w.owned_bytes() - size_of::<Weights>())
                .sum::<usize>();
        graph::Backing {
            tensor_bytes: physical + norm_values * 4,
            metadata_bytes: size_of::<Self>() + owners - physical
                + self.description.layers.capacity() * size_of::<graph::LayerWeights<'_, usize>>()
                + norms.owned,
            metadata_allowance_bytes: 0,
        }
    }
}

impl Backend<'_, '_, '_> {
    pub(crate) fn inject_failure(&mut self, gather: bool, successful_calls: usize) {
        self.test_fault = Some((gather, successful_calls));
    }
    pub(super) fn test_failure(&mut self, gather: bool) -> bool {
        if let Some((kind, remaining)) = &mut self.test_fault
            && *kind == gather
        {
            if *remaining == 0 {
                self.failed = true;
                self.test_fault = None;
                return true;
            }
            *remaining -= 1;
        }
        false
    }
}
impl Uploaded<'_> {
    pub(crate) fn snapshot(&self) -> serde_json::Value {
        use sha2::{Digest, Sha256};
        let buffer = |b: &Buffer<u32>| {
            let digest = b.with_slice(|words| {
                let mut h = Sha256::new();
                for word in words {
                    h.update(word.to_le_bytes());
                }
                format!("{:x}", h.finalize())
            });
            serde_json::json!({"identity":format!("{:p}",&*b.raw),"words":b.len,"sha256":digest})
        };
        serde_json::json!(self.matrices.iter().map(|w|serde_json::json!({"rows":w.layout().rows(),"columns":w.layout().columns(),
            "chunks":w.views().map(|v|serde_json::json!({"rows":v.layout.rows(),"codes":buffer(v.codes),"scales":buffer(v.scales),"biases":buffer(v.biases)})).collect::<Vec<_>>()})).collect::<Vec<_>>())
    }
}
