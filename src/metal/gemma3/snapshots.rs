//! Coarse ordered GPU snapshots. Numeric stage slots are resolved once at
//! construction; no string map, dynamic callbacks or CPU operators in forward.
use super::*;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLResource};

const LAYER_STAGES: [(&str, usize); 22] = [
    ("input_norm", 0),
    ("q_projection", 1),
    ("k_projection", 2),
    ("v_projection", 2),
    ("q_norm", 1),
    ("k_norm", 2),
    ("q_rotary", 1),
    ("k_rotary", 2),
    ("attention_scores", 4),
    ("attention_probabilities", 4),
    ("attention_context", 1),
    ("attention_projection", 0),
    ("post_attention_norm", 0),
    ("attention_residual", 0),
    ("pre_ff_norm", 0),
    ("gate_projection", 3),
    ("up_projection", 3),
    ("gate_gelu", 3),
    ("gated", 3),
    ("down_projection", 0),
    ("post_ff_norm", 0),
    ("output", 0),
];
pub(super) struct Snapshot {
    layer: Option<usize>,
    stage: &'static str,
    width: usize,
    half: bool,
    scores: bool,
    qk: bool,
    elements: usize,
    buffer: Option<Buffer<u32>>,
}
/// Shape contains one unused trailing 1 for rank2 arrays. Scores/probabilities
/// use [4, tokens, keys] with absolute keys [key_start,key_start+keys).
#[derive(Clone, Copy, Debug)]
pub struct SnapshotInfo {
    pub layer: Option<usize>,
    pub stage: &'static str,
    pub shape: [usize; 3],
    pub rank: usize,
    pub absolute_query_start: usize,
    pub absolute_key_start: usize,
}
impl SnapshotInfo {
    pub fn values(&self) -> usize {
        self.shape.into_iter().product()
    }
}
pub(super) fn plan(c: &Config, enabled: bool) -> Result<(Vec<Snapshot>, usize, usize, usize)> {
    let mut result = Vec::new();
    if !enabled {
        return Ok((result, 0, 0, 0));
    }
    result
        .try_reserve_exact(4 + c.num_layers * 22)
        .map_err(|_| Error::Options("Gemma3 snapshot metadata allocation"))?;
    let mut gpu_bytes = 0;
    let mut readback_bytes = 0;
    let mut push = |layer, stage, width, half, scores, qk| -> Result<()> {
        let active = if scores {
            mul(4 * B, c.max_positions)?
        } else {
            mul(B, width)?
        };
        let bytes = mul(add(active, GUARD)?, if half { 2 } else { 4 })?.div_ceil(4) * 4;
        gpu_bytes = add(gpu_bytes, bytes)?;
        readback_bytes = add(readback_bytes, mul(active, 4)?)?;
        result.push(Snapshot {
            layer,
            stage,
            width,
            half,
            scores,
            qk,
            elements: bytes / 4,
            buffer: None,
        });
        Ok(())
    };
    push(None, "embedding_lookup", c.hidden_size, true, false, false)?;
    push(None, "embedding", c.hidden_size, true, false, false)?;
    for layer in 0..c.num_layers {
        for (stage, kind) in LAYER_STAGES {
            let width = match kind {
                0 => c.hidden_size,
                1 => 4 * c.head_dim,
                2 => c.head_dim,
                3 => c.intermediate_size,
                _ => c.max_positions,
            };
            let qk = matches!(
                stage,
                "q_projection"
                    | "k_projection"
                    | "v_projection"
                    | "q_norm"
                    | "k_norm"
                    | "q_rotary"
                    | "k_rotary"
            );
            push(Some(layer), stage, width, kind != 4, kind == 4, qk)?;
        }
    }
    push(None, "final_norm", c.hidden_size, true, false, false)?;
    push(None, "logits", c.vocab_size, false, false, false)?;
    let metadata = mul(result.capacity(), size_of::<Snapshot>())?;
    Ok((result, gpu_bytes, readback_bytes, metadata))
}
pub(super) fn validate(plans: &[Snapshot], maximum: usize) -> Result<()> {
    for p in plans {
        validate_buffer_bytes(mul(p.elements, 4)?, maximum)?;
    }
    Ok(())
}
pub(super) fn allocate(gpu: &Gpu, mut plans: Vec<Snapshot>) -> Result<Vec<Snapshot>> {
    for p in &mut plans {
        p.buffer = Some(Buffer::new(gpu, p.elements)?);
    }
    Ok(plans)
}
impl Session<'_> {
    pub(super) fn capture(
        &self,
        command: &Command<'_>,
        slot: usize,
        source: &ProtocolObject<dyn MTLBuffer>,
    ) -> Result<()> {
        if self.snapshots.is_empty() {
            return Ok(());
        }
        let s = self
            .snapshots
            .get(slot)
            .ok_or(Error::Options("Gemma3 snapshot slot"))?;
        let out = s.buffer.as_ref().unwrap();
        // Copy the fixed capacity before scratch reuse. Readback selects active
        // rows/columns; inactive rows, score padding and guards are not evidence.
        let active = if s.scores {
            4 * B * self.model.config.max_positions
        } else {
            B * s.width
        };
        let bytes = active * if s.half { 2 } else { 4 };
        if !bytes.is_multiple_of(4)
            || bytes > source.length()
            || bytes > out.raw.length()
            || !std::ptr::eq(&*source.device(), &*out.raw.device())
        {
            return Err(Error::Options("Gemma3 snapshot buffer/alignment"));
        }
        let e = command
            .blitCommandEncoder()
            .ok_or_else(|| Error::Native("Gemma3 snapshot encoder absent".into()))?;
        // SAFETY: fresh disjoint same-device owners, checked aligned byte ranges,
        // retained resources, tracked dependencies, no host access before finish.
        unsafe {
            e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                source, 0, &out.raw, 0, bytes,
            );
        }
        e.endEncoding();
        Ok(())
    }
    pub fn diagnostic_count(&self) -> usize {
        self.snapshots.len()
    }
    pub fn diagnostic_info(&self, index: usize) -> Result<SnapshotInfo> {
        if self.is_poisoned() {
            return Err(Error::ReplayRequired);
        }
        if self.diagnostic_rows == 0 {
            return Err(Error::Options("Gemma3 no completed diagnostic chunk"));
        }
        let s = self
            .snapshots
            .get(index)
            .ok_or(Error::Options("Gemma3 diagnostic index"))?;
        let n = self.diagnostic_rows;
        let (shape, rank, key_start) = if s.scores {
            let layer = &self.model.layers[s.layer.unwrap()];
            let window = window(layer.spec, self.model.local_window);
            let start = crate::model::window_rows::view_start(self.diagnostic_start, window);
            ([4, n, self.diagnostic_start + n - start], 3, start)
        } else if s.qk {
            (
                [
                    n,
                    s.width / self.model.config.head_dim,
                    self.model.config.head_dim,
                ],
                3,
                0,
            )
        } else {
            ([n, s.width, 1], 2, 0)
        };
        Ok(SnapshotInfo {
            layer: s.layer,
            stage: s.stage,
            shape,
            rank,
            absolute_query_start: self.diagnostic_start,
            absolute_key_start: key_start,
        })
    }
    /// Caller-owned output avoids hidden host allocations. The ledger reserves
    /// all maximal FP32 readback arrays; callers retaining more copies account
    /// those separately as external resident state.
    pub fn copy_diagnostic(&self, index: usize, output: &mut [f32]) -> Result<SnapshotInfo> {
        let info = self.diagnostic_info(index)?;
        if output.len() != info.values() {
            return Err(Error::Options("Gemma3 diagnostic output length"));
        }
        let s = &self.snapshots[index];
        s.buffer.as_ref().unwrap().with_slice(|bits| {
            if s.scores {
                let keys = info.shape[2];
                for head in 0..4 {
                    for row in 0..self.diagnostic_rows {
                        for key in 0..keys {
                            output[(head * self.diagnostic_rows + row) * keys + key] =
                                f32::from_bits(
                                    bits[(head * B + row) * self.model.config.max_positions + key],
                                );
                        }
                    }
                }
            } else {
                for (i, out) in output.iter_mut().enumerate() {
                    *out = if s.half {
                        let word = bits[i / 2];
                        let half = if i % 2 == 0 {
                            word as u16
                        } else {
                            (word >> 16) as u16
                        };
                        half::f16::from_bits(half).to_f32()
                    } else {
                        f32::from_bits(bits[i])
                    };
                }
            }
        });
        Ok(info)
    }
}

#[cfg(test)]
impl Session<'_> {
    pub(super) fn test_snapshot_words(&self) -> Result<Vec<Vec<u32>>> {
        self.cache.test_safe_read()?;
        Ok(self
            .snapshots
            .iter()
            .map(|s| s.buffer.as_ref().unwrap().with_slice(|v| v.to_vec()))
            .collect())
    }
}

#[cfg(test)]
impl Session<'_> {
    // Cold/quiescent test initialization only; callers do not reseed after work.
    pub(super) fn test_seed_snapshot_guards(&mut self, half: u16, word: u32) -> Result<()> {
        self.cache.test_safe_read()?;
        for s in &mut self.snapshots {
            let (count, value) = if s.half {
                (4, u32::from(half) | (u32::from(half) << 16))
            } else {
                (7, word)
            };
            s.buffer.as_mut().unwrap().with_mut(|v| {
                let start = v.len() - count;
                v[start..].fill(value);
            });
        }
        Ok(())
    }
}
