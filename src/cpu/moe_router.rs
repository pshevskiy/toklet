//! Deterministic routing for the first bounded sparse-expert consumer.
use super::{Error, finite, length, nonzero, product};

#[derive(Clone, Copy, Debug)]
pub struct RouterShape {
    pub tokens: usize,
    pub experts: usize,
    pub selected: usize,
}

/// Scores are row-major `[tokens, experts]`; IDs/gates are `[tokens, selected]`.
/// Rank by descending finite score, breaking ties with ascending expert ID.
/// Normalize only the selected scores, retaining that ranked order. Subtract,
/// exponential, ordered sum and division use FP32 boundaries. A negative
/// subtraction overflow becomes -infinity and contributes zero probability.
/// All shape/length/finiteness checks precede writes; no allocation occurs.
/// This bounded scalar reference supports at most 32 experts, not an entire MoE.
pub fn route_topk(
    scores: &[f32],
    ids: &mut [u32],
    gates: &mut [f32],
    shape: RouterShape,
) -> Result<(), Error> {
    let RouterShape {
        tokens,
        experts,
        selected,
    } = shape;
    nonzero(tokens, "router tokens")?;
    nonzero(experts, "router experts")?;
    nonzero(selected, "router selected")?;
    if experts > 32 || selected > experts {
        return Err(Error::InvalidParameter(
            "router requires selected <= experts <= 32",
        ));
    }
    length("router scores", scores.len(), product(tokens, experts)?)?;
    let output = product(tokens, selected)?;
    length("router IDs", ids.len(), output)?;
    length("router gates", gates.len(), output)?;
    finite("router scores", scores)?;
    for token in 0..tokens {
        let row = &scores[token * experts..(token + 1) * experts];
        let ids = &mut ids[token * selected..(token + 1) * selected];
        let gates = &mut gates[token * selected..(token + 1) * selected];
        // Deliberately readable reference: bounded repeated scans, not a heap.
        for slot in 0..selected {
            let mut best = experts;
            for expert in 0..experts {
                if !ids[..slot].contains(&(expert as u32))
                    && (best == experts || row[expert] > row[best])
                {
                    best = expert;
                }
            }
            // selected <= experts ensures a distinct candidate at every slot.
            ids[slot] = best as u32;
        }
        let maximum = row[ids[0] as usize];
        let mut total = 0_f32;
        for (gate, &id) in gates.iter_mut().zip(ids.iter()) {
            *gate = (row[id as usize] - maximum).exp();
            total += *gate;
        }
        // Highest score contributes exp(0)=1; all <=32 terms lie in [0,1].
        for gate in gates {
            *gate /= total;
        }
    }
    Ok(())
}
