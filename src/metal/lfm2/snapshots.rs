//! Fixed57 stage slots, resolved cold. State observations belong to real chunk
//! boundaries, not fabricated within-chunk committed prefixes.
use super::*;
const CONV: [(&str, usize); 17] = [
    ("operator_norm", H),
    ("bcx_projection", 3 * H),
    ("b", H),
    ("c", H),
    ("x", H),
    ("u", H),
    ("convolution", H),
    ("gated_conv", H),
    ("operator_projection", H),
    ("operator_residual", H),
    ("ffn_norm", H),
    ("gate_projection", I),
    ("up_projection", I),
    ("gate_silu", I),
    ("ffn_gated", I),
    ("down_projection", H),
    ("output", H),
];
const ATTN: [(&str, usize); 20] = [
    ("operator_norm", H),
    ("q_projection", Q),
    ("k_projection", K),
    ("v_projection", K),
    ("q_norm", Q),
    ("k_norm", K),
    ("q_rotary", Q),
    ("k_rotary", K),
    ("attention_scores", CAP),
    ("attention_probabilities", CAP),
    ("attention_context", Q),
    ("operator_projection", H),
    ("operator_residual", H),
    ("ffn_norm", H),
    ("gate_projection", I),
    ("up_projection", I),
    ("gate_silu", I),
    ("ffn_gated", I),
    ("down_projection", H),
    ("output", H),
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
pub(super) fn plan(enabled: bool) -> Result<(Vec<Snapshot>, usize, usize, usize)> {
    let mut out = Vec::new();
    if !enabled {
        return Ok((out, 0, 0, 0));
    }
    out.try_reserve_exact(57)
        .map_err(|_| Error::Options("LFM2 diagnostic metadata allocation"))?;
    let (mut gpu, mut host) = (0, 0);
    let mut push = |layer, stage, width, half, scores, qk| -> Result<()> {
        let active = if scores { 4 * B * CAP } else { B * width };
        // u32-backed Half snapshots include the extra alignment Half word after
        // seven guards. Ordinary Half scratch has exactly seven guard words.
        let bytes = mul(add(active, GUARD)?, if half { 2 } else { 4 })?.div_ceil(4) * 4;
        gpu = add(gpu, bytes)?;
        host = add(host, mul(active, 4)?)?;
        out.push(Snapshot {
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
    push(None, "embedding_lookup", H, true, false, false)?;
    for layer in 0..3 {
        let table: &[(&str, usize)] = if layer == 1 { &ATTN } else { &CONV };
        for &(stage, width) in table {
            let scores = matches!(stage, "attention_scores" | "attention_probabilities");
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
            push(Some(layer), stage, width, !scores, scores, qk)?;
        }
    }
    push(None, "final_norm", H, true, false, false)?;
    push(None, "logits", V, false, false, false)?;
    let metadata = mul(out.capacity(), size_of::<Snapshot>())?;
    Ok((out, gpu, host, metadata))
}
pub(super) fn validate(plans: &[Snapshot], maximum: usize) -> Result<()> {
    for p in plans {
        validate_buffer_bytes(mul(p.elements, 4)?, maximum)?;
    }
    Ok(())
}
pub(super) fn allocate(gpu: &Gpu, mut plans: Vec<Snapshot>) -> Result<Vec<Snapshot>> {
    for p in &mut plans {
        let mut buffer = Buffer::new(gpu, p.elements)?;
        buffer.with_mut(|v| {
            let guards = if p.half { 4 } else { 7 };
            let start = v.len() - guards;
            v[start..].fill(if p.half {
                u32::from(HALF_PAD) * 0x10001
            } else {
                WORD_PAD
            });
        });
        p.buffer = Some(buffer);
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
            .ok_or(Error::Options("LFM2 snapshot slot"))?;
        self.capture_view(
            command,
            slot,
            source,
            0,
            if s.scores { 4 * CAP } else { s.width },
        )
    }
    pub(super) fn capture_view(
        &self,
        command: &Command<'_>,
        slot: usize,
        source: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        stride: usize,
    ) -> Result<()> {
        if self.snapshots.is_empty() {
            return Ok(());
        }
        let s = self
            .snapshots
            .get(slot)
            .ok_or(Error::Options("LFM2 snapshot slot"))?;
        let out = s.buffer.as_ref().unwrap();
        let scalar = if s.half { 2 } else { 4 };
        let width = if s.scores { 4 * CAP } else { s.width };
        // Score physical layout is copied whole; its host reader applies head,
        // query and key strides. BCx B/C/x views copy one tight row at a time.
        let end = add(add(offset, mul(B - 1, stride)?)?, width)?;
        if stride < width
            || !mul(offset, scalar)?.is_multiple_of(4)
            || !mul(stride, scalar)?.is_multiple_of(4)
            || !mul(width, scalar)?.is_multiple_of(4)
            || mul(end, scalar)? > source.length()
            || mul(B * width, scalar)? > out.raw.length()
            || !std::ptr::eq(&*source.device(), &*out.raw.device())
        {
            return Err(Error::Options("LFM2 snapshot extent/alignment/device"));
        }
        let e = command
            .blitCommandEncoder()
            .ok_or_else(|| Error::Native("LFM2 snapshot encoder absent".into()))?;
        // SAFETY: checked aligned source/destination extents, same-device fresh
        // disjoint snapshot, retained command; mapped host access only terminal.
        unsafe {
            if stride == width {
                e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    source,
                    offset * scalar,
                    &out.raw,
                    0,
                    B * width * scalar,
                );
            } else {
                for row in 0..B {
                    e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        source,
                        (offset + row * stride) * scalar,
                        &out.raw,
                        row * width * scalar,
                        width * scalar,
                    );
                }
            }
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
            return Err(Error::Options("LFM2 no completed diagnostic chunk"));
        }
        let s = self
            .snapshots
            .get(index)
            .ok_or(Error::Options("LFM2 diagnostic index"))?;
        let n = self.diagnostic_rows;
        let (shape, rank) = if s.scores {
            ([4, n, self.diagnostic_start + n], 3)
        } else if s.qk {
            ([n, s.width / D, D], 3)
        } else {
            ([n, s.width, 1], 2)
        };
        Ok(SnapshotInfo {
            layer: s.layer,
            stage: s.stage,
            shape,
            rank,
            absolute_query_start: self.diagnostic_start,
            absolute_key_start: 0,
        })
    }
    pub fn copy_diagnostic(&self, index: usize, out: &mut [f32]) -> Result<SnapshotInfo> {
        let info = self.diagnostic_info(index)?;
        if out.len() != info.values() {
            return Err(Error::Options("LFM2 diagnostic output length"));
        }
        let s = &self.snapshots[index];
        s.buffer.as_ref().unwrap().with_slice(|v| {
            if s.scores {
                let keys = info.shape[2];
                for head in 0..4 {
                    for row in 0..self.diagnostic_rows {
                        for key in 0..keys {
                            out[(head * self.diagnostic_rows + row) * keys + key] =
                                f32::from_bits(v[(head * B + row) * CAP + key]);
                        }
                    }
                }
            } else {
                for (j, dst) in out.iter_mut().enumerate() {
                    *dst = if s.half {
                        half::f16::from_bits((v[j / 2] >> (16 * (j % 2))) as u16).to_f32()
                    } else {
                        f32::from_bits(v[j])
                    };
                }
            }
        });
        Ok(info)
    }
}
#[cfg(test)]
impl Session<'_> {
    pub(super) fn snapshot_buffers(&self) -> Result<Vec<(usize, bool, Vec<u32>)>> {
        self.cache.test_safe_read()?;
        Ok(self
            .snapshots
            .iter()
            .map(|s| {
                let b = s.buffer.as_ref().unwrap();
                (
                    b.raw.contents().as_ptr() as usize,
                    s.half,
                    b.with_slice(|v| v.to_vec()),
                )
            })
            .collect())
    }
    pub(super) fn snapshot_guards(&self) {
        for (_, half, v) in self.snapshot_buffers().unwrap() {
            let count = if half { 4 } else { 7 };
            let expected = if half {
                u32::from(HALF_PAD) * 0x10001
            } else {
                WORD_PAD
            };
            assert!(v[v.len() - count..].iter().all(|&x| x == expected));
        }
    }
}
