//! Stable operation identities and logical order for the selected BF16/G2 graph.
//! This inspects authenticated metadata only, not tensor payloads or a device.
use super::{PREFIX, TextPlan, invalid};
use crate::weights::Error;
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    TokenEmbedding,
    PerLayerEmbedding,
    PerLayerInputProjection,
    Query,
    Key,
    Value,
    AttentionOutput,
    FeedForwardGate,
    FeedForwardUp,
    FeedForwardDown,
    PerLayerGate,
    PerLayerProjection,
    OutputHead,
}

/// Version1 identities are semantic roles, never reconstructed from event order.
/// The tied head and token gather deliberately name the same physical source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct OperationId {
    pub role: Role,
    pub layer: Option<u16>,
}

impl OperationId {
    pub fn tensor_name(self) -> Result<String, Error> {
        let suffix = match (self.role, self.layer) {
            (Role::TokenEmbedding | Role::OutputHead, None) => "embed_tokens.weight",
            (Role::PerLayerEmbedding, None) => "embed_tokens_per_layer.weight",
            (Role::PerLayerInputProjection, None) => "per_layer_model_projection.weight",
            (role, Some(layer)) if usize::from(layer) < super::super::LAYERS => {
                let suffix = match role {
                    Role::Query => "self_attn.q_proj.weight",
                    Role::Key => "self_attn.k_proj.weight",
                    Role::Value => "self_attn.v_proj.weight",
                    Role::AttentionOutput => "self_attn.o_proj.weight",
                    Role::FeedForwardGate => "mlp.gate_proj.weight",
                    Role::FeedForwardUp => "mlp.up_proj.weight",
                    Role::FeedForwardDown => "mlp.down_proj.weight",
                    Role::PerLayerGate => "per_layer_input_gate.weight",
                    Role::PerLayerProjection => "per_layer_projection.weight",
                    _ => return Err(invalid("Gemma BF16 operation role/layer differs")),
                };
                return Ok(format!("{PREFIX}layers.{layer}.{suffix}"));
            }
            _ => return Err(invalid("Gemma BF16 operation role/layer differs")),
        };
        Ok(format!("{PREFIX}{suffix}"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Execution {
    ResidentBf16CpuGather,
    AuthenticatedFileBf16CpuGather,
    TiledBf16MetalProjection,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Operation {
    pub id: OperationId,
    /// Rank2 enumeration in TextPlan's ordered source map, matching delivery.
    pub source_matrix_slot: usize,
    pub source_tensor: String,
    pub source_rows: usize,
    pub source_columns: usize,
    pub source_bytes: usize,
    pub prefill_rows: usize,
    pub decode_rows: usize,
    pub execution: Execution,
}

#[derive(Debug, Serialize)]
pub struct Plan {
    pub schema_version: u32,
    pub policy: &'static str,
    pub chunk_rows: usize,
    pub layer_count: usize,
    pub producer_count: usize,
    pub operations: Vec<Operation>,
    pub cpu_recipe: [&'static str; 5],
    pub payload_authenticated: bool,
    pub numerical_buffers_allocated: bool,
}

pub(crate) fn operation_ids(
    layers: &[crate::mlx::gemma4::LayerPlan],
) -> Result<Vec<OperationId>, Error> {
    if layers.len() != super::super::LAYERS {
        return Err(invalid("Gemma BF16 execution layer count"));
    }
    let mut ids = Vec::new();
    ids.try_reserve_exact(3 + layers.len() * 9 + 1)
        .map_err(|_| Error::Allocation)?;
    let mut add = |role, layer| ids.push(OperationId { role, layer });
    for role in [
        Role::TokenEmbedding,
        Role::PerLayerEmbedding,
        Role::PerLayerInputProjection,
    ] {
        add(role, None);
    }
    for (index, g) in layers.iter().enumerate() {
        let layer =
            Some(u16::try_from(index).map_err(|_| invalid("Gemma BF16 layer ID overflow"))?);
        add(Role::Query, layer);
        if g.producer == index {
            add(Role::Key, layer);
            add(Role::Value, layer);
        }
        for role in [
            Role::AttentionOutput,
            Role::FeedForwardGate,
            Role::FeedForwardUp,
            Role::FeedForwardDown,
            Role::PerLayerGate,
            Role::PerLayerProjection,
        ] {
            add(role, layer);
        }
    }
    add(Role::OutputHead, None);
    Ok(ids)
}

pub(crate) fn prefill_rows(
    id: OperationId,
    layers: &[crate::mlx::gemma4::LayerPlan],
    rows: usize,
) -> Result<usize, Error> {
    if !(1..=32).contains(&rows) {
        return Err(invalid("Gemma BF16 execution chunk outside1..32"));
    }
    if id.role == Role::OutputHead {
        return Ok(1);
    }
    if let Some(layer) = id.layer {
        let index = usize::from(layer);
        let g = layers
            .get(index)
            .ok_or_else(|| invalid("Gemma BF16 operation layer absent"))?;
        return Ok(if g.producer == index { rows } else { 1 });
    }
    Ok(rows)
}

/// Logical gather/projection order for one newly evaluated chunk. CPU operators
/// between these calls are described separately; this is not a GPU trace or a
/// prediction of an entire request's chunk count/cache action.
pub fn inspect(plan: &TextPlan, chunk_rows: usize) -> Result<Plan, Error> {
    if !(1..=32).contains(&chunk_rows) {
        return Err(invalid("Gemma BF16 execution chunk outside1..32"));
    }
    let mut operations = Vec::new();
    operations
        .try_reserve_exact(plan.text_tensors().len())
        .map_err(|_| Error::Allocation)?;
    for id in operation_ids(plan.layers())? {
        let rows = prefill_rows(id, plan.layers(), chunk_rows)?;
        let execution = match id.role {
            Role::TokenEmbedding => Execution::ResidentBf16CpuGather,
            Role::PerLayerEmbedding => Execution::AuthenticatedFileBf16CpuGather,
            _ => Execution::TiledBf16MetalProjection,
        };
        let name = id.tensor_name()?;
        let (slot, (_, tensor)) = plan
            .text_tensors()
            .iter()
            .filter(|(_, t)| t.shape.len() == 2)
            .enumerate()
            .find(|(_, (key, _))| **key == name)
            .ok_or_else(|| invalid("Gemma BF16 operation source absent"))?;
        operations.push(Operation {
            id,
            source_matrix_slot: slot,
            source_tensor: name,
            source_rows: tensor.shape[0],
            source_columns: tensor.shape[1],
            source_bytes: tensor.range.len(),
            prefill_rows: rows,
            decode_rows: 1,
            execution,
        });
    }
    Ok(Plan {
        schema_version: 1,
        policy: "official BF16 native factory G2 last-row prefill; FP32 graph/KV; full CPU logits",
        chunk_rows,
        layer_count: plan.layers().len(),
        producer_count: plan
            .layers()
            .iter()
            .enumerate()
            .filter(|(i, g)| g.producer == *i)
            .count(),
        operations,
        cpu_recipe: [
            "embedding gather -> scaled input -> PLE gather/projection -> per-layer input normalization/combination",
            "each producer: input RMSNorm -> Q projection/norm/rotary -> K projection/norm/rotary -> V projection/unit normalization -> private KV proposal",
            "each layer: causal shared-KV CPU attention -> output projection/post norm/residual -> pre-FF norm -> gate/up projections/GELU -> down projection/post norm/residual -> PLE gate/GELU/projection/post norm/residual/scalar",
            "G2 keeps all producer rows, then only final hidden/PLE row through shared consumers and final norm/head/softcap",
            "KV/history publish only after every layer/head succeeds; CPU phases are not individually timed by this metadata report",
        ],
        payload_authenticated: false,
        numerical_buffers_allocated: false,
    })
}

#[cfg(test)]
mod tests;
