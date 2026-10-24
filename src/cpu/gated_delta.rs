use super::{Error, finite, length, nonzero, product};

/// One unmasked, scalar-gate recurrent step. Dimensions are checked at entry.
#[derive(Debug, Clone, Copy)]
pub struct GatedDeltaShape {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

/// Update a separate candidate state and read it through normalized Q.
///
/// Q/K are `[key_heads,key_dim]`, V/output `[value_heads,value_dim]`, gates
/// `[value_heads]`, and both states `[value_heads,value_dim,key_dim]` with
/// contiguous key dimension. Value head `h` reads key head `h / (Hv / Hk)`.
/// Q/K normalization (including the Q readout scale) and gate construction
/// belong to separate operations. Decay/beta must be scalar values in `[0,1]`.
/// All products/reductions use FP32 in increasing key-coordinate order.
///
/// No allocations, copies of the previous state, masks or session mutations.
/// Length/shape/input errors precede all writes. Arithmetic failure can leave
/// partial candidate/output buffers; discard both. The input state stays intact
/// on every result. A session may commit the candidate only after its entire
/// multi-layer step succeeds, which this standalone operator does not manage.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay: &[f32],
    beta: &[f32],
    state: &[f32],
    candidate: &mut [f32],
    output: &mut [f32],
    shape: &GatedDeltaShape,
) -> Result<(), Error> {
    let GatedDeltaShape {
        key_heads,
        value_heads,
        key_dim,
        value_dim,
    } = *shape;
    for (size, name) in [
        (key_heads, "key_heads"),
        (value_heads, "value_heads"),
        (key_dim, "key_dim"),
        (value_dim, "value_dim"),
    ] {
        nonzero(size, name)?;
    }
    if !value_heads.is_multiple_of(key_heads) {
        return Err(Error::InvalidParameter(
            "value_heads must be divisible by key_heads",
        ));
    }
    let key_elements = product(key_heads, key_dim)?;
    let value_elements = product(value_heads, value_dim)?;
    let state_elements = product(value_elements, key_dim)?;
    for (name, actual, expected) in [
        ("delta Q", q.len(), key_elements),
        ("delta K", k.len(), key_elements),
        ("delta V", v.len(), value_elements),
        ("delta decay", decay.len(), value_heads),
        ("delta beta", beta.len(), value_heads),
        ("delta state", state.len(), state_elements),
        ("delta candidate", candidate.len(), state_elements),
        ("delta output", output.len(), value_elements),
    ] {
        length(name, actual, expected)?;
    }
    for (name, values) in [
        ("delta Q", q),
        ("delta K", k),
        ("delta V", v),
        ("delta decay", decay),
        ("delta beta", beta),
        ("delta state", state),
    ] {
        finite(name, values)?;
    }
    if decay.iter().chain(beta).any(|x| !(0.0..=1.0).contains(x)) {
        return Err(Error::InvalidParameter("delta scalar gate outside [0,1]"));
    }
    let heads_per_key = value_heads / key_heads;
    for head in 0..value_heads {
        let key_base = (head / heads_per_key) * key_dim;
        for value in 0..value_dim {
            let index = head * value_dim + value;
            let base = index * key_dim;
            let mut prediction = 0.0_f32;
            for dim in 0..key_dim {
                let decayed = state[base + dim] * decay[head];
                candidate[base + dim] = decayed;
                prediction += decayed * k[key_base + dim];
            }
            let correction = (v[index] - prediction) * beta[head];
            if !prediction.is_finite() || !correction.is_finite() {
                return Err(Error::NonFinite {
                    tensor: "delta correction",
                    index,
                });
            }
            let mut readout = 0.0_f32;
            for dim in 0..key_dim {
                let next = candidate[base + dim] + k[key_base + dim] * correction;
                if !next.is_finite() {
                    return Err(Error::NonFinite {
                        tensor: "delta candidate",
                        index: base + dim,
                    });
                }
                candidate[base + dim] = next;
                readout += next * q[key_base + dim];
            }
            if !readout.is_finite() {
                return Err(Error::NonFinite {
                    tensor: "delta output",
                    index,
                });
            }
            output[index] = readout;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
