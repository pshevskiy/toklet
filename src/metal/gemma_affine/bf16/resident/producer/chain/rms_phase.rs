//! Session-owned persistent receipts for the explicit paired RMS graph family.
//! One bank,22 disjoint operations, no CPU overwrites between A/B dispatches.
use super::*;
use rms::precision::two_phase as paired;
const OPERATIONS: usize = 22;
const ORDER: [usize; 8] = [0, 1, 2, 7, 3, 4, 5, 6];

pub(super) enum Family {
    Row(rms::precision::Pipelines),
    Paired(paired::Pipelines),
}
impl Family {
    pub(super) fn paired(&self) -> Option<&paired::Pipelines> {
        match self {
            Self::Row(_) => None,
            Self::Paired(pipelines) => Some(pipelines),
        }
    }
}
#[derive(Clone, Copy)]
struct Operation {
    node: usize,
    norm: usize,
    status: usize,
    width: usize,
    factor: usize,
    context: Region,
    header: Region,
    tag: u32,
}
#[derive(Clone, Copy)]
pub(in crate::metal::gemma_affine::bf16::resident::producer) struct Plan {
    operations: [Operation; OPERATIONS],
    pub words: usize,
}
impl Plan {
    pub(super) fn new(rows: usize) -> Result<Self> {
        if !(1..=32).contains(&rows) {
            return Err(Error::Options("paired RMS chunk admission differs"));
        }
        let mut cursor = GUARD;
        let mut index = 0;
        let empty = Operation {
            node: 0,
            norm: 0,
            status: 0,
            width: 0,
            factor: 0,
            context: Region { start: 0, len: 0 },
            header: Region { start: 0, len: 0 },
            tag: 0,
        };
        let mut operations = [empty; OPERATIONS];
        for (node, d) in [256, 512, 256].into_iter().enumerate() {
            for norm in ORDER {
                if node == 2 && (norm == 2 || norm == 7) {
                    continue;
                }
                let factor = if norm == 1 { 8 } else { 1 };
                let context = Region {
                    start: cursor,
                    len: rows * factor * 8,
                };
                cursor = add(context.end(), GUARD)?;
                let header = Region {
                    start: cursor,
                    len: 4,
                };
                cursor = add(header.end(), GUARD)?;
                operations[index] = Operation {
                    node,
                    norm,
                    status: rms::precision::STATUS_SLOTS[norm],
                    width: if [1, 2, 7].contains(&norm) { d } else { H },
                    factor,
                    context,
                    header,
                    tag: index as u32 + 1,
                };
                index += 1;
            }
        }
        if index != OPERATIONS || cursor != 344 * rows + 403 {
            return Err(Error::Options("paired RMS cold plan differs"));
        }
        Ok(Self {
            operations,
            words: cursor,
        })
    }
    pub(super) fn initialize_statuses(&self, storage: &mut pool::Storage) -> usize {
        for op in &self.operations {
            storage
                .status_mut(op.node)
                .with_mut(|s| s[op.status] = paired::PENDING);
        }
        OPERATIONS * 4
    }
    pub(super) fn payload_bytes(&self) -> Result<usize> {
        self.words
            .checked_mul(8)
            .ok_or(Error::Options("paired RMS snapshot byte overflow"))
    }
    pub(in crate::metal::gemma_affine::bf16::resident::producer) fn slots<'a>(
        &self,
        scratch: &'a Scratch,
        active_rows: usize,
        node: usize,
    ) -> Result<[Option<paired::Slot<'a>>; 8]> {
        let mut out = [None; 8];
        for op in self.operations.iter().filter(|op| op.node == node) {
            let rows = active_rows * op.factor;
            out[op.norm] = Some(paired::Slot {
                context: op.context.range(&scratch.buffer, rows * 8)?,
                header: op.header.range(&scratch.buffer, 4)?,
                op_tag: op.tag,
            });
        }
        Ok(out)
    }
}
#[derive(Clone, Copy)]
struct ReceiptShape {
    rows: usize,
    width: usize,
    epoch: u32,
    tag: u32,
    groups: u32,
}
fn row_receipt(d: &[u32], shape: ReceiptShape, row: usize, final_reason: u32) -> Result<u32> {
    if d.len() != 8 {
        return Err(Error::Options("paired RMS row word count differs"));
    }
    let mut required = 0;
    let mode = d[4] & 3;
    let scale = d[2] as i32;
    let check = d[..6]
        .iter()
        .fold(row as u32 ^ shape.tag ^ 0x9e37_79b9, |v, &x| v ^ x);
    if d[4] >> 2 != shape.epoch
        || mode == 3
        || d[5] != paired::VERSION | shape.rows as u32 | ((shape.width as u32) << 9)
        || d[6] != check
        || d[7] != shape.groups
    {
        return Err(Error::Options("paired RMS row receipt differs"));
    }
    if final_reason & !rms::precision::KNOWN_REASON_BITS != 0 || d[3] & !final_reason != 0 {
        return Err(Error::Options("paired RMS row reasons differ"));
    }
    match mode {
        0 => {
            let hm = d[0] & 0x7fff_ffff;
            let lm = d[1] & 0x7fff_ffff;
            if !(-10..=40).contains(&scale)
                || d[3] & !0x3f0 != 0
                || d[3] & 512 == 0
                    && (d[0] & 0x8000_0000 != 0
                        || !(0x0080_0000..0x7f80_0000).contains(&hm)
                        || lm != 0 && !(0x0080_0000..0x7f80_0000).contains(&lm))
            {
                return Err(Error::Options(
                    "paired RMS normal descriptor domain differs",
                ));
            }
            required |= u32::from(d[3] != 0) * 2;
        }
        1..=2 => {
            let fatal = mode == 2;
            let allowed = if fatal { 0x8000_000f } else { 15 };
            let max_scale = if fatal { 128 } else { 127 };
            if d[0] != 0
                || d[1] != 0
                || d[3] == 0
                || d[3] & !allowed != 0
                || fatal != (d[3] & 0x8000_0000 != 0)
                || !(-10..=max_scale).contains(&scale)
            {
                return Err(Error::Options("paired RMS early descriptor domain differs"));
            }
            required |= if fatal { 1 } else { 2 };
        }
        _ => unreachable!(),
    }
    Ok(required)
}

pub(in crate::metal::gemma_affine::bf16::resident::producer) struct Scratch {
    pub buffer: Buffer<u32>,
    snapshot: Vec<u32>,
    snapshotted: bool,
}
impl Scratch {
    pub(super) fn new(gpu: &Gpu, plan: &Plan) -> Result<Self> {
        let mut buffer = Buffer::new(gpu, plan.words)?;
        buffer.with_mut(|s| s.fill(SENTINEL));
        let mut snapshot = Vec::new();
        snapshot
            .try_reserve_exact(plan.words)
            .map_err(|_| Error::Native("paired RMS snapshot admission failed".into()))?;
        snapshot.resize(plan.words, SENTINEL);
        Ok(Self {
            buffer,
            snapshot,
            snapshotted: true,
        })
    }
    pub(super) fn bytes(&self) -> usize {
        (self.buffer.len + self.snapshot.capacity()) * 4
    }
    pub(super) fn validate(&self, gpu: &Gpu, plan: &Plan) -> Result<()> {
        validate_storage(gpu, &self.buffer, plan.words)?;
        if self.snapshot.len() != plan.words {
            return Err(Error::Options("paired RMS snapshot extent differs"));
        }
        Ok(())
    }
    pub(super) fn unchanged(&self) -> Result<usize> {
        if !self.snapshotted || !self.buffer.with_slice(|s| s == self.snapshot) {
            return Err(Error::Options(
                "paired RMS evidence changed after validation",
            ));
        }
        Ok(self.buffer.len * 8)
    }
    pub(super) fn reset(
        &mut self,
        plan: &Plan,
        frame: &GraphFrame,
        pipelines: &paired::Pipelines,
    ) -> Result<usize> {
        let epoch = paired::epoch(frame.epoch)?;
        // Everything fallible is checked before any CPU overwrite.
        let mut totals = [0u32; OPERATIONS];
        let mut active = 0;
        for (i, op) in plan.operations.iter().enumerate() {
            let rows = frame.nodes[op.node].rows * op.factor;
            if rows == 0 || rows * 8 > op.context.len {
                return Err(Error::Options("paired RMS active rows exceed admission"));
            }
            totals[i] = u32::try_from(rows * pipelines.groups(op.width, op.norm != 7)?)
                .map_err(|_| Error::Options("paired RMS group count overflow"))?;
            active += rows;
        }
        self.buffer.with_mut(|s| {
            s.fill(SENTINEL);
            for (i, op) in plan.operations.iter().enumerate() {
                let rows = frame.nodes[op.node].rows * op.factor;
                for row in 0..rows {
                    s[op.context.start + row * 8 + 4] = 0;
                }
                s[op.header.start..op.header.end()].copy_from_slice(&[epoch, op.tag, totals[i], 0]);
            }
        });
        self.snapshotted = false;
        Ok(self.buffer.len * 4 + active * 4 + OPERATIONS * 16)
    }
    fn guards(&self, plan: &Plan, frame: &GraphFrame) -> Result<()> {
        let valid = self.buffer.with_slice(|s| {
            let mut good = s[..GUARD].iter().all(|&x| x == SENTINEL);
            for op in &plan.operations {
                let n = frame.nodes[op.node].rows * op.factor * 8;
                good &= n <= op.context.len
                    && s[op.context.start + n..op.context.end() + GUARD]
                        .iter()
                        .all(|&x| x == SENTINEL);
                good &= s[op.header.end()..op.header.end() + GUARD]
                    .iter()
                    .all(|&x| x == SENTINEL);
            }
            good
        });
        if !valid {
            return Err(Error::Options("paired RMS guard/inactive capacity changed"));
        }
        Ok(())
    }
    pub(super) fn snapshot_completed(
        &mut self,
        plan: &Plan,
        frame: &GraphFrame,
        pipelines: &paired::Pipelines,
        storage: &pool::Storage,
        layouts: [LayerLayout; 3],
    ) -> Result<usize> {
        self.guards(plan, frame)?;
        let epoch = paired::epoch(frame.epoch)?;
        let mut row_bytes = 0;
        for op in &plan.operations {
            let rows = frame.nodes[op.node].rows * op.factor;
            let groups = pipelines.groups(op.width, op.norm != 7)? as u32;
            let status = storage.status(op.node).with_slice(|s| s[op.status]);
            if status & !3 != 0 {
                return Err(Error::Options("paired RMS pending/structural status"));
            }
            let reason = layouts[op.node].rms_plan().0[op.norm]
                .ok_or(Error::Options("paired RMS reason role absent"))?;
            let result = self.buffer.with_slice(|s| {
                storage.reasons(op.node).with_slice(|reasons| {
                    let header = &s[op.header.start..op.header.end()];
                    if header != [epoch, op.tag, rows as u32 * groups, rows as u32 * groups] {
                        return Err(Error::Options(
                            "paired RMS operation completion/header differs",
                        ));
                    }
                    let mut numeric = 0;
                    for row in 0..rows {
                        let d = &s[op.context.start + row * 8..op.context.start + row * 8 + 8];
                        numeric |= row_receipt(
                            d,
                            ReceiptShape {
                                rows,
                                width: op.width,
                                epoch,
                                tag: op.tag,
                                groups,
                            },
                            row,
                            reasons[reason.start + row],
                        )?;
                    }
                    // Extra original numeric1/2 flags remain subject to graph-order replay.
                    // Missing mandatory A reasons is a contract failure; do not
                    // promote later ordinary numeric failures into structural ones.
                    if numeric & !status != 0 {
                        return Err(Error::Options("paired RMS mandatory status missing"));
                    }
                    Ok(())
                })
            });
            result?;
            row_bytes += rows * 4;
        }
        self.copy_snapshot();
        Ok(self.buffer.len * 8 + row_bytes + OPERATIONS * 4)
    }
    fn copy_snapshot(&mut self) {
        self.buffer.with_slice(|s| self.snapshot.copy_from_slice(s));
        self.snapshotted = true;
    }
    /// Existing lifecycle allows a drained transaction abandoned before numeric
    /// encoding to reuse the owner. PENDING here is host initialization only.
    pub(super) fn snapshot_unencoded(
        &mut self,
        plan: &Plan,
        frame: &GraphFrame,
        pipelines: &paired::Pipelines,
    ) -> Result<()> {
        self.guards(plan, frame)?;
        let epoch = paired::epoch(frame.epoch)?;
        let valid = self.buffer.with_slice(|s| {
            plan.operations.iter().all(|op| {
                let rows = frame.nodes[op.node].rows * op.factor;
                let groups = pipelines.groups(op.width, op.norm != 7).unwrap();
                s[op.header.start..op.header.end()] == [epoch, op.tag, (rows * groups) as u32, 0]
                    && (0..rows).all(|row| {
                        (0..8).all(|word| {
                            s[op.context.start + row * 8 + word]
                                == if word == 4 { 0 } else { SENTINEL }
                        })
                    })
            })
        });
        if !valid {
            return Err(Error::Options(
                "paired RMS abandoned initialization changed",
            ));
        }
        self.copy_snapshot();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row(shape: ReceiptShape, mode: u32, reason: u32, scale: i32) -> [u32; 8] {
        let mut d = [
            if mode == 0 { 1f32.to_bits() } else { 0 },
            0,
            scale as u32,
            reason,
            shape.epoch << 2 | mode,
            paired::VERSION | shape.rows as u32 | ((shape.width as u32) << 9),
            0,
            shape.groups,
        ];
        d[6] = d[..6].iter().fold(shape.tag ^ 0x9e37_79b9, |v, &x| v ^ x);
        d
    }
    fn reseal(d: &mut [u32; 8], shape: ReceiptShape) {
        d[6] = d[..6].iter().fold(shape.tag ^ 0x9e37_79b9, |v, &x| v ^ x);
    }
    #[test]
    fn paired_plan_bounds_every_actual_chunk_and_preserves_status_roles() {
        for rows in 1..=32 {
            let p = Plan::new(rows).unwrap();
            assert_eq!(p.words, 344 * rows + 403);
            assert_eq!(p.payload_bytes().unwrap(), 2 * (1376 * rows + 1612));
            assert_eq!(
                p.operations
                    .iter()
                    .map(|o| o.context.len / 8)
                    .sum::<usize>(),
                43 * rows
            );
            let mut cursor = GUARD;
            for (i, o) in p.operations.iter().enumerate() {
                assert_eq!(o.context.start, cursor);
                assert!(o.context.len <= 2048);
                cursor = o.context.end() + GUARD;
                assert_eq!(o.header.start, cursor);
                cursor = o.header.end() + GUARD;
                assert_eq!(o.tag, i as u32 + 1);
                assert_eq!(o.status, rms::precision::STATUS_SLOTS[o.norm]);
                assert!(o.node != 2 || o.norm != 2 && o.norm != 7);
            }
            assert_eq!(cursor, p.words);
        }
        assert_eq!(Plan::new(32).unwrap().payload_bytes().unwrap(), 91288);
        assert!(Plan::new(0).is_err());
        assert!(Plan::new(33).is_err());
        assert!(Plan::new(usize::MAX).is_err());
        assert_eq!((paired::PENDING, paired::STRUCTURAL), (4, 8));
    }
    #[test]
    fn paired_receipts_reject_coherent_corruption_and_keep_numeric_priority() {
        for epoch in [1, paired::MAX_EPOCH] {
            let shape = ReceiptShape {
                rows: 256,
                width: 512,
                epoch,
                tag: 22,
                groups: 16,
            };
            for (mode, reason, scale, needed) in [
                (0, 0, -10, 0),
                (0, 16, 40, 2),
                (1, 8, 127, 2),
                (2, 0x8000_0004, 128, 1),
            ] {
                let d = row(shape, mode, reason, scale);
                assert_eq!(row_receipt(&d, shape, 0, reason).unwrap(), needed);
                for (word, value) in [
                    (4, ((epoch - 1) << 2) | mode),
                    (5, 0),
                    (7, 0),
                    (4, epoch << 2 | 3),
                ] {
                    let mut bad = d;
                    bad[word] = value;
                    reseal(&mut bad, shape);
                    assert!(row_receipt(&bad, shape, 0, reason).is_err());
                }
                let mut bad = d;
                bad[6] ^= 1;
                assert!(row_receipt(&bad, shape, 0, reason).is_err());
                assert!(row_receipt(&d, shape, 1, reason).is_err());
            }
            let mut normal = row(shape, 0, 0, 0);
            normal[2] = (-11i32) as u32;
            reseal(&mut normal, shape);
            assert!(row_receipt(&normal, shape, 0, 0).is_err());
            normal = row(shape, 0, 0, 0);
            normal[0] = f32::INFINITY.to_bits();
            reseal(&mut normal, shape);
            assert!(row_receipt(&normal, shape, 0, 0).is_err());
            let normal = row(shape, 0, 0, 0);
            // An extra ordinary fatal1/known reason remains numeric: graph
            // ordering may discard it after an earlier whole-chain replay.
            assert_eq!(row_receipt(&normal, shape, 0, 0x8000_0000).unwrap(), 0);
            let uncertain = row(shape, 0, 16, 0);
            assert!(row_receipt(&uncertain, shape, 0, 0).is_err());
        }
    }
}
