//! Fixed57 tiny or293 selected-real stage slots, resolved cold. State observations belong to real chunk
//! boundaries, not fabricated within-chunk committed prefixes.
use super::*;
fn conv(g: Geometry) -> [(&'static str, usize); 17] {
    [
        ("operator_norm", g.h),
        ("bcx_projection", 3 * g.h),
        ("b", g.h),
        ("c", g.h),
        ("x", g.h),
        ("u", g.h),
        ("convolution", g.h),
        ("gated_conv", g.h),
        ("operator_projection", g.h),
        ("operator_residual", g.h),
        ("ffn_norm", g.h),
        ("gate_projection", g.i),
        ("up_projection", g.i),
        ("gate_silu", g.i),
        ("ffn_gated", g.i),
        ("down_projection", g.h),
        ("output", g.h),
    ]
}
fn attn(g: Geometry) -> [(&'static str, usize); 20] {
    [
        ("operator_norm", g.h),
        ("q_projection", g.q),
        ("k_projection", g.k),
        ("v_projection", g.k),
        ("q_norm", g.q),
        ("k_norm", g.k),
        ("q_rotary", g.q),
        ("k_rotary", g.k),
        ("attention_scores", g.capacity),
        ("attention_probabilities", g.capacity),
        ("attention_context", g.q),
        ("operator_projection", g.h),
        ("operator_residual", g.h),
        ("ffn_norm", g.h),
        ("gate_projection", g.i),
        ("up_projection", g.i),
        ("gate_silu", g.i),
        ("ffn_gated", g.i),
        ("down_projection", g.h),
        ("output", g.h),
    ]
}
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
pub(super) fn plan(enabled: bool, g: Geometry) -> Result<(Vec<Snapshot>, usize, usize, usize)> {
    let mut out = Vec::new();
    if !enabled {
        return Ok((out, 0, 0, 0));
    }
    out.try_reserve_exact(g.stages())
        .map_err(|_| Error::Options("LFM2 diagnostic metadata allocation"))?;
    let (mut gpu, mut host) = (0, 0);
    let mut push = |layer, stage, width, half, scores, qk| -> Result<()> {
        let active = if scores {
            g.query_heads * B * g.capacity
        } else {
            B * width
        };
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
    push(None, "embedding_lookup", g.h, true, false, false)?;
    let conv = conv(g);
    let attn = attn(g);
    for layer in 0..g.layers {
        let table: &[(&str, usize)] = match g.kind(layer) {
            Kind::Attention(_) => &attn,
            Kind::Convolution(_) => &conv,
        };
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
    push(None, "final_norm", g.h, true, false, false)?;
    push(None, "logits", g.v, false, false, false)?;
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
            if s.scores {
                self.model.geometry.query_heads * self.model.geometry.capacity
            } else {
                s.width
            },
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
        let width = if s.scores {
            self.model.geometry.query_heads * self.model.geometry.capacity
        } else {
            s.width
        };
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
        let g = self.model.geometry;
        let n = self.diagnostic_rows;
        let (shape, rank) = if s.scores {
            ([g.query_heads, n, self.diagnostic_start + n], 3)
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
        let g = self.model.geometry;
        let s = &self.snapshots[index];
        s.buffer.as_ref().unwrap().with_slice(|v| {
            if s.scores {
                let keys = info.shape[2];
                for head in 0..g.query_heads {
                    for row in 0..self.diagnostic_rows {
                        for key in 0..keys {
                            out[(head * self.diagnostic_rows + row) * keys + key] =
                                f32::from_bits(v[(head * B + row) * g.capacity + key]);
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

#[cfg(test)]
mod cold_tests {
    use super::*;
    #[test]
    fn selected_real_snapshot_slots_shapes_and_aligned_owner_bytes() {
        let g = geometry(&super::super::tests::real_config(512)).unwrap();
        let (plans, gpu, host, metadata) = plan(true, g).unwrap();
        assert_eq!((plans.len(), gpu, host), (293, 54530796, 88080384));
        assert_eq!(metadata, plans.capacity() * size_of::<Snapshot>());
        assert_eq!(plans[0].stage, "embedding_lookup");
        assert_eq!(plans[291].stage, "final_norm");
        assert_eq!(plans[292].stage, "logits");
        let (mut slot, mut scores, mut qk) = (1, 0, 0);
        let mut names = std::collections::BTreeSet::new();
        for layer in 0..16 {
            let length = if matches!(g.kind(layer), Kind::Attention(_)) {
                20
            } else {
                17
            };
            for p in &plans[slot..slot + length] {
                assert_eq!(p.layer, Some(layer));
                assert!(names.insert((p.layer, p.stage)));
                assert!(p.buffer.is_none());
                if p.scores {
                    scores += 1;
                    assert_eq!(
                        (p.width, p.elements, p.half),
                        (512, 16 * 32 * 512 + 7, false)
                    );
                } else {
                    assert_eq!(p.elements * 4, (32 * p.width + 8) * 2);
                }
                if p.qk {
                    qk += 1;
                    assert!(p.width == 1024 || p.width == 512);
                }
            }
            slot += length;
        }
        assert_eq!((slot, scores, qk), (291, 12, 42));
        assert_eq!(plans[292].elements, 32 * 65536 + 7);
        assert!(!plans[292].half);
        validate(&plans, 8_388_636).unwrap();
        assert!(validate(&plans, 8_388_635).is_err());
        let (_, small_gpu, small_host, _) = plan(
            true,
            geometry(&super::super::tests::real_config(1)).unwrap(),
        )
        .unwrap();
        assert_eq!(gpu - small_gpu, 12 * 16 * 32 * 511 * 4);
        assert_eq!(host - small_host, 12 * 16 * 32 * 511 * 4);
    }
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(super) struct CaptureSlot {
    pub slot: usize,
    pub layer: Option<usize>,
    pub stage: &'static str,
    pub width: usize,
    pub half: bool,
    pub scores: bool,
    pub address: usize,
    pub device: u64,
}
#[cfg(test)]
impl Session<'_> {
    // Raw terminal scratch is observable even when publication failed. The
    // caller labels that attempt; this never changes diagnostic validity.
    pub(super) fn visit_capture_slots(
        &self,
        mut visit: impl FnMut(CaptureSlot, &[u32]) -> Result<()>,
    ) -> Result<()> {
        self.cache.test_safe_read()?;
        for (slot, s) in self.snapshots.iter().enumerate() {
            let b = s.buffer.as_ref().unwrap();
            let info = CaptureSlot {
                slot,
                layer: s.layer,
                stage: s.stage,
                width: s.width,
                half: s.half,
                scores: s.scores,
                address: b.raw.contents().as_ptr() as usize,
                device: b.raw.device().registryID(),
            };
            b.with_slice(|words| visit(info, words))?;
        }
        Ok(())
    }
}
