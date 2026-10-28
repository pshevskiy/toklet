//! Cold source naming and original row mapping for the Phi3 dense composition.
use super::{logical::RowPartition, *};

/// Convert the common separate-projection inventory into the physical export.
/// Logical partitions still feed the ordinary decoder's separate projections.
pub(super) fn inventory(
    config: &Config,
    expected: &mut BTreeMap<String, Vec<usize>>,
) -> Result<Vec<RowPartition>, Error> {
    let query = config
        .query_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| invalid("Phi3 query row overflow"))?;
    let kv = config
        .kv_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| invalid("Phi3 KV row overflow"))?;
    let key_end = add(query, kv)?;
    let qkv_end = add(key_end, kv)?;
    let mlp_end = add(config.intermediate_size, config.intermediate_size)?;
    let count = config
        .num_layers
        .checked_mul(2)
        .ok_or_else(|| invalid("Phi3 partition count overflow"))?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| Error::Allocation)?;
    for layer in 0..config.num_layers {
        let prefix = format!("model.layers.{layer}");
        for (parent, end, parts) in [
            (
                format!("{prefix}.self_attn.qkv_proj.weight"),
                qkv_end,
                vec![
                    (format!("{prefix}.self_attn.q_proj.weight"), 0..query),
                    (format!("{prefix}.self_attn.k_proj.weight"), query..key_end),
                    (
                        format!("{prefix}.self_attn.v_proj.weight"),
                        key_end..qkv_end,
                    ),
                ],
            ),
            (
                format!("{prefix}.mlp.gate_up_proj.weight"),
                mlp_end,
                vec![
                    (
                        format!("{prefix}.mlp.gate_proj.weight"),
                        0..config.intermediate_size,
                    ),
                    (
                        format!("{prefix}.mlp.up_proj.weight"),
                        config.intermediate_size..mlp_end,
                    ),
                ],
            ),
        ] {
            for (name, rows) in &parts {
                let shape = expected
                    .remove(name)
                    .ok_or_else(|| invalid("Phi3 logical projection missing from inventory"))?;
                if shape != [rows.end - rows.start, config.hidden_size] {
                    return Err(invalid("Phi3 logical projection shape differs"));
                }
            }
            if expected
                .insert(parent.clone(), vec![end, config.hidden_size])
                .is_some()
            {
                return Err(invalid("Phi3 physical projection repeated"));
            }
            result.push(RowPartition {
                physical_name: parent,
                parts,
            });
        }
    }
    Ok(result)
}
