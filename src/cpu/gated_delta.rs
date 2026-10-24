use super::{Error, finite, length, nonzero, product};

/// One unmasked, scalar-gate recurrent step. Dimensions are checked at entry.
#[derive(Debug, Clone, Copy)]
pub struct GatedDeltaShape {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

/// L2-normalize raw Q/K `[heads,key_dim]`, folding one readout scale into Q.
///
/// Epsilon is added to sum(x²), not its mean. Squares, increasing-coordinate
/// sums and normalization use FP64; outputs are rounded once to FP32. Q also
/// receives `key_dim**-0.5`. There are no weights or heap allocations. All
/// dimensions, lengths, finite inputs and epsilon are checked before writes.
/// This is distinct from the transformer's per-head RMSNorm operation.
pub fn gated_delta_normalize(
    q: &[f32],
    k: &[f32],
    q_output: &mut [f32],
    k_output: &mut [f32],
    heads: usize,
    key_dim: usize,
    epsilon: f64,
) -> Result<(), Error> {
    nonzero(heads, "heads")?;
    nonzero(key_dim, "key_dim")?;
    let elements = product(heads, key_dim)?;
    for (name, actual) in [
        ("delta raw Q", q.len()),
        ("delta raw K", k.len()),
        ("delta normalized Q", q_output.len()),
        ("delta normalized K", k_output.len()),
    ] {
        length(name, actual, elements)?;
    }
    finite("delta raw Q", q)?;
    finite("delta raw K", k)?;
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Error::InvalidParameter("delta L2 epsilon"));
    }
    let readout_scale = (key_dim as f64).sqrt().recip();
    for head in 0..heads {
        let base = head * key_dim;
        let mut q_squared = 0.0_f64;
        let mut k_squared = 0.0_f64;
        for dim in 0..key_dim {
            let q = f64::from(q[base + dim]);
            let k = f64::from(k[base + dim]);
            q_squared += q * q;
            k_squared += k * k;
        }
        // Finite FP32 squares/sums fit FP64 even at the maximum usize count.
        // Positive finite epsilon leaves nonzero finite denominators. Each
        // normalized coordinate is bounded by one, so output casts stay finite.
        let q_norm = (q_squared + epsilon).sqrt();
        let k_norm = (k_squared + epsilon).sqrt();
        for dim in 0..key_dim {
            q_output[base + dim] = ((f64::from(q[base + dim]) / q_norm) * readout_scale) as f32;
            k_output[base + dim] = (f64::from(k[base + dim]) / k_norm) as f32;
        }
    }
    Ok(())
}

/// Construct scalar decay/update gates from FP32 raw vectors `[value_heads]`.
///
/// Decay is `exp(-exp(a_log)*softplus(a+dt_bias))`; beta is sigmoid(b).
/// Stable softplus/sigmoid and other intermediate arithmetic use FP64, then
/// outputs round once to FP32. Gates remain in `[0,1]`. An overflowing FP64
/// rate/exponent rejects rather than silently replacing its intermediate.
/// All lengths and finite inputs are checked before writes. Arithmetic failure
/// may leave partial output vectors; discard both. No state or storage is owned.
pub fn gated_delta_gates(
    a: &[f32],
    b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    decay: &mut [f32],
    beta: &mut [f32],
) -> Result<(), Error> {
    let heads = a.len();
    nonzero(heads, "value_heads")?;
    for (name, actual) in [
        ("delta raw beta", b.len()),
        ("delta log rate", a_log.len()),
        ("delta dt bias", dt_bias.len()),
        ("delta decay", decay.len()),
        ("delta beta", beta.len()),
    ] {
        length(name, actual, heads)?;
    }
    for (name, values) in [
        ("delta raw decay", a),
        ("delta raw beta", b),
        ("delta log rate", a_log),
        ("delta dt bias", dt_bias),
    ] {
        finite(name, values)?;
    }
    for head in 0..heads {
        let x = f64::from(a[head]) + f64::from(dt_bias[head]);
        let softplus = x.max(0.0) + (-x.abs()).exp().ln_1p();
        let alpha = f64::from(a_log[head]).exp();
        let rate = alpha * softplus;
        if !alpha.is_finite() || !rate.is_finite() {
            return Err(Error::NonFinite {
                tensor: "delta decay exponent",
                index: head,
            });
        }
        let b = f64::from(b[head]);
        let exp_negative_abs = (-b.abs()).exp();
        let update = if b >= 0.0 {
            1.0 / (1.0 + exp_negative_abs)
        } else {
            exp_negative_abs / (1.0 + exp_negative_abs)
        };
        decay[head] = (-rate).exp() as f32;
        beta[head] = update as f32;
    }
    Ok(())
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
