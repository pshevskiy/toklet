//! Test-only append-only raw evidence. No numerical transformation or GPU work.
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{self, Write},
    path::Path,
};

pub(super) const RAW_CAP: usize = 256 * 1024 * 1024;
pub(super) const JOURNAL_CAP: usize = 16 * 1024 * 1024;
const BLOCK: usize = 64 * 1024;
const MAX_ARRAYS: usize = 15_000;
const MAX_BOUNDARIES: usize = 160;
const MAX_OTHER: usize = 2_000;

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[derive(Clone, Copy, Serialize)]
pub(super) struct Tag {
    // [session, attempt, record-kind, stage/owner]. The header fixes each enum.
    pub id: [u32; 4],
    pub shape: [usize; 3],
    pub rank: usize,
    pub query: [usize; 2],
    pub keys: [usize; 2],
}
#[derive(Clone, Serialize)]
pub(super) struct ArrayRef {
    pub seq: usize,
    pub offset: usize,
    pub bytes: usize,
    pub sha256: String,
}
#[derive(Serialize)]
struct ArrayRecord {
    kind: &'static str,
    #[serde(flatten)]
    range: ArrayRef,
    dtype: &'static str,
    #[serde(flatten)]
    tag: Tag,
}
#[derive(Serialize)]
struct Event<'a, T> {
    seq: usize,
    kind: &'static str,
    body: &'a T,
}
#[derive(Clone, Copy)]
pub(super) enum Class {
    Header,
    Boundary,
    Other,
}
#[derive(Serialize)]
pub(super) struct Summary {
    pub raw_bytes: usize,
    pub raw_sha256: String,
    pub journal_bytes: usize,
    pub journal_sha256: String,
    pub records: usize,
    pub arrays: usize,
    pub boundaries: usize,
    pub other: usize,
}
struct Capped {
    bytes: Vec<u8>,
    cap: usize,
}
impl Write for Capped {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self
            .bytes
            .len()
            .checked_add(bytes.len())
            .is_none_or(|n| n > self.cap)
        {
            return Err(invalid("capture JSON record exceeds cap"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(super) struct Writer {
    raw: File,
    journal: File,
    raw_bytes: usize,
    journal_bytes: usize,
    seq: usize,
    arrays: usize,
    boundaries: usize,
    other: usize,
    raw_hash: Sha256,
    journal_hash: Sha256,
    failed: bool,
}
impl Writer {
    pub fn create(directory: &Path) -> io::Result<Self> {
        // Parent owner creates and authenticates the exclusive directory. Neither
        // file can replace an existing output or follow an existing final symlink.
        let raw = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("capture.bin"))?;
        let journal = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("journal.jsonl"))?;
        Ok(Self {
            raw,
            journal,
            raw_bytes: 0,
            journal_bytes: 0,
            seq: 0,
            arrays: 0,
            boundaries: 0,
            other: 0,
            raw_hash: Sha256::new(),
            journal_hash: Sha256::new(),
            failed: false,
        })
    }
    fn writable(&self) -> io::Result<()> {
        if self.failed {
            Err(invalid("capture writer already failed"))
        } else {
            Ok(())
        }
    }
    fn json<T: Serialize>(&mut self, value: &T, cap: usize) -> io::Result<()> {
        self.writable()?;
        let result = (|| {
            let mut encoded = Capped {
                bytes: Vec::with_capacity(cap),
                cap,
            };
            serde_json::to_writer(&mut encoded, value).map_err(io::Error::other)?;
            encoded.write_all(b"\n")?;
            let end = self
                .journal_bytes
                .checked_add(encoded.bytes.len())
                .ok_or_else(|| invalid("capture journal overflow"))?;
            if end > JOURNAL_CAP {
                return Err(invalid("capture journal cap"));
            }
            self.journal.write_all(&encoded.bytes)?;
            self.journal.flush()?;
            self.journal_hash.update(&encoded.bytes);
            self.journal_bytes = end;
            self.seq = self
                .seq
                .checked_add(1)
                .ok_or_else(|| invalid("capture sequence overflow"))?;
            Ok(())
        })();
        self.failed |= result.is_err();
        result
    }
    pub fn event<T: Serialize>(
        &mut self,
        kind: &'static str,
        body: &T,
        class: Class,
    ) -> io::Result<()> {
        self.writable()?;
        if self.seq == 0 && !matches!(class, Class::Header) {
            self.failed = true;
            return Err(invalid("capture header required first"));
        }
        let cap = match class {
            Class::Header if self.seq == 0 => 262_144,
            Class::Header => {
                self.failed = true;
                return Err(invalid("capture header is not first"));
            }
            Class::Boundary if self.boundaries < MAX_BOUNDARIES => 40_000,
            Class::Other if self.other < MAX_OTHER => 512,
            _ => {
                self.failed = true;
                return Err(invalid("capture record inventory cap"));
            }
        };
        self.json(
            &Event {
                seq: self.seq,
                kind,
                body,
            },
            cap,
        )?;
        match class {
            Class::Header => {}
            Class::Boundary => self.boundaries += 1,
            Class::Other => self.other += 1,
        }
        Ok(())
    }
    fn array<const N: usize>(
        &mut self,
        tag: Tag,
        count: usize,
        dtype: &'static str,
        words: impl IntoIterator<Item = [u8; N]>,
    ) -> io::Result<ArrayRef> {
        self.writable()?;
        let result = (|| {
            if !matches!(N, 1 | 2 | 4) {
                return Err(invalid("capture scalar width"));
            }
            // BLOCK is divisible by each admitted concrete scalar width.
            if self.seq == 0
                || self.arrays >= MAX_ARRAYS
                || !(1..=3).contains(&tag.rank)
                || tag.shape[..tag.rank].contains(&0)
                || tag.shape[tag.rank..].iter().any(|&x| x != 1)
                || tag.shape[..tag.rank]
                    .iter()
                    .try_fold(1usize, |a, &b| a.checked_mul(b))
                    != Some(count)
            {
                return Err(invalid("capture array shape/order/inventory"));
            }
            let bytes = count
                .checked_mul(N)
                .ok_or_else(|| invalid("capture raw count overflow"))?;
            let end = self
                .raw_bytes
                .checked_add(bytes)
                .ok_or_else(|| invalid("capture raw offset overflow"))?;
            if end > RAW_CAP {
                return Err(invalid("capture raw cap"));
            }
            let offset = self.raw_bytes;
            let mut hash = Sha256::new();
            let mut block = [0u8; BLOCK];
            let (mut used, mut seen) = (0, 0);
            for word in words {
                if seen == count {
                    return Err(invalid("capture iterator longer than shape"));
                }
                block[used..used + N].copy_from_slice(&word);
                used += N;
                seen += 1;
                if used == BLOCK {
                    self.raw.write_all(&block)?;
                    hash.update(block);
                    self.raw_hash.update(block);
                    used = 0;
                }
            }
            if used != 0 {
                self.raw.write_all(&block[..used])?;
                hash.update(&block[..used]);
                self.raw_hash.update(&block[..used]);
            }
            // Even on a short iterator the partial raw prefix above is retained.
            if seen != count {
                return Err(invalid("capture iterator shorter than shape"));
            }
            self.raw.flush()?;
            self.raw_bytes = end;
            let range = ArrayRef {
                seq: self.seq,
                offset,
                bytes,
                sha256: format!("{:x}", hash.finalize()),
            };
            self.json(
                &ArrayRecord {
                    kind: "array",
                    range: range.clone(),
                    dtype,
                    tag,
                },
                384,
            )?;
            self.arrays += 1;
            Ok(range)
        })();
        self.failed |= result.is_err();
        result
    }
    pub fn bytes(&mut self, tag: Tag, bytes: &[u8]) -> io::Result<ArrayRef> {
        self.array(tag, bytes.len(), "u8", bytes.iter().map(|&b| [b]))
    }
    pub fn half(
        &mut self,
        tag: Tag,
        count: usize,
        words: impl IntoIterator<Item = u16>,
    ) -> io::Result<ArrayRef> {
        self.array(tag, count, "f16le", words.into_iter().map(u16::to_le_bytes))
    }
    pub fn float(
        &mut self,
        tag: Tag,
        count: usize,
        words: impl IntoIterator<Item = u32>,
    ) -> io::Result<ArrayRef> {
        self.array(tag, count, "f32le", words.into_iter().map(u32::to_le_bytes))
    }
    /// Hashes of the exact file prefixes before an optional end event. The end
    /// event may carry this summary. finish() separately hashes the COMPLETE
    /// journal including that event; its summary goes to stdout/owner receipt.
    /// Never put a whole-journal self-hash inside the journal itself.
    pub fn prefix(&self) -> io::Result<Summary> {
        self.writable()?;
        Ok(Summary {
            raw_bytes: self.raw_bytes,
            raw_sha256: format!("{:x}", self.raw_hash.clone().finalize()),
            journal_bytes: self.journal_bytes,
            journal_sha256: format!("{:x}", self.journal_hash.clone().finalize()),
            records: self.seq,
            arrays: self.arrays,
            boundaries: self.boundaries,
            other: self.other,
        })
    }
    pub fn finish(mut self) -> io::Result<Summary> {
        self.writable()?;
        self.raw.flush()?;
        self.journal.flush()?;
        self.raw.sync_all()?;
        self.journal.sync_all()?;
        Ok(Summary {
            raw_bytes: self.raw_bytes,
            raw_sha256: format!("{:x}", self.raw_hash.finalize()),
            journal_bytes: self.journal_bytes,
            journal_sha256: format!("{:x}", self.journal_hash.finalize()),
            records: self.seq,
            arrays: self.arrays,
            boundaries: self.boundaries,
            other: self.other,
        })
    }
}

pub(super) fn hash16(words: &[u16]) -> String {
    let mut hash = Sha256::new();
    // A fixed block avoids a model-sized Vec and an update call per scalar.
    let mut bytes = [0u8; BLOCK];
    for chunk in words.chunks(BLOCK / 2) {
        for (dst, word) in bytes.chunks_exact_mut(2).zip(chunk) {
            dst.copy_from_slice(&word.to_le_bytes());
        }
        hash.update(&bytes[..chunk.len() * 2]);
    }
    format!("{:x}", hash.finalize())
}
pub(super) fn hash32(words: &[u32]) -> String {
    let mut hash = Sha256::new();
    let mut bytes = [0u8; BLOCK];
    for chunk in words.chunks(BLOCK / 4) {
        for (dst, word) in bytes.chunks_exact_mut(4).zip(chunk) {
            dst.copy_from_slice(&word.to_le_bytes());
        }
        hash.update(&bytes[..chunk.len() * 4]);
    }
    format!("{:x}", hash.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    struct Directory(std::path::PathBuf);
    impl Directory {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!(
                "toklet-capture-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
    fn tag(n: usize) -> Tag {
        Tag {
            id: [1, 2, 3, 4],
            shape: [n, 1, 1],
            rank: 1,
            query: [7, 1],
            keys: [0, 0],
        }
    }
    fn ready(path: &Path) -> Writer {
        let mut w = Writer::create(path).unwrap();
        w.event(
            "start",
            &serde_json::json!({"format":"synthetic"}),
            Class::Header,
        )
        .unwrap();
        w
    }
    #[test]
    fn capture_words_ranges_hashes_and_footer_scopes() {
        let dir = Directory::new();
        let mut w = ready(&dir.0);
        let a = w.half(tag(5), 5, [0, 0x8000, 1, 0x7e55, 0x7bff]).unwrap();
        let b = w
            .float(tag(4), 4, [0, 0x80000000, 0x7fc00001, 0x3f800000])
            .unwrap();
        let c = w.bytes(tag(3), &[0, 127, 255]).unwrap();
        assert_eq!(
            (a.offset, a.bytes, b.offset, b.bytes, c.offset, c.bytes),
            (0, 10, 10, 16, 26, 3)
        );
        w.event(
            "boundary",
            &serde_json::json!({"refs":[a.seq,b.seq,c.seq]}),
            Class::Boundary,
        )
        .unwrap();
        let prefix = w.prefix().unwrap();
        w.event("end", &prefix, Class::Other).unwrap();
        let summary = w.finish().unwrap();
        let raw = std::fs::read(dir.0.join("capture.bin")).unwrap();
        let journal = std::fs::read(dir.0.join("journal.jsonl")).unwrap();
        assert_eq!(raw.len(), 29);
        assert_eq!(&raw[..4], &[0, 0, 0, 128]);
        assert_eq!(summary.raw_sha256, format!("{:x}", Sha256::digest(&raw)));
        assert_eq!(
            summary.journal_sha256,
            format!("{:x}", Sha256::digest(&journal))
        );
        assert_eq!(
            prefix.journal_sha256,
            format!("{:x}", Sha256::digest(&journal[..prefix.journal_bytes]))
        );
        assert_ne!(prefix.journal_sha256, summary.journal_sha256);
        let records: Vec<serde_json::Value> = journal
            .split(|&b| b == b'\n')
            .filter(|r| !r.is_empty())
            .map(|r| serde_json::from_slice(r).unwrap())
            .collect();
        for (i, r) in records.iter().enumerate() {
            assert_eq!(r["seq"], i);
        }
        assert_eq!(
            (
                summary.records,
                summary.arrays,
                summary.boundaries,
                summary.other
            ),
            (6, 3, 1, 1)
        );
        assert_eq!(hash16(&[0, 0x8000, 1, 0x7e55, 0x7bff]), a.sha256);
        assert_eq!(hash32(&[0, 0x80000000, 0x7fc00001, 0x3f800000]), b.sha256);
    }
    #[test]
    fn capture_short_long_iterator_poison_and_partial_retention() {
        let dir = Directory::new();
        let mut w = ready(&dir.0);
        assert!(w.half(tag(2), 2, [0x3555]).is_err());
        assert_eq!(
            std::fs::read(dir.0.join("capture.bin")).unwrap(),
            [0x55, 0x35]
        );
        assert!(w.prefix().is_err());
        assert!(w.event("false_success", &true, Class::Other).is_err());
        assert!(w.finish().is_err());
        let dir = Directory::new();
        let mut w = ready(&dir.0);
        assert!(
            w.half(
                tag(BLOCK / 2),
                BLOCK / 2,
                std::iter::repeat_n(7, BLOCK / 2 + 1)
            )
            .is_err()
        );
        assert_eq!(
            std::fs::metadata(dir.0.join("capture.bin")).unwrap().len(),
            BLOCK as u64
        );
        assert!(w.finish().is_err());
    }
    #[test]
    fn capture_caps_checked_before_output_and_failure_is_sticky() {
        let dir = Directory::new();
        let mut w = ready(&dir.0);
        w.raw_bytes = RAW_CAP - 1;
        assert!(w.half(tag(1), 1, [1]).is_err());
        assert_eq!(
            std::fs::metadata(dir.0.join("capture.bin")).unwrap().len(),
            0
        );
        assert!(w.finish().is_err());
        let dir = Directory::new();
        let mut w = ready(&dir.0);
        let before = w.journal_bytes;
        assert!(
            w.event("oversized", &"x".repeat(512), Class::Other)
                .is_err()
        );
        assert_eq!(
            std::fs::metadata(dir.0.join("journal.jsonl"))
                .unwrap()
                .len(),
            before as u64
        );
        assert!(w.finish().is_err());
        let dir = Directory::new();
        let mut w = ready(&dir.0);
        w.journal_bytes = JOURNAL_CAP;
        assert!(w.event("cap", &true, Class::Other).is_err());
        assert!(w.finish().is_err());
        let dir = Directory::new();
        let mut w = ready(&dir.0);
        w.arrays = MAX_ARRAYS;
        assert!(w.bytes(tag(1), &[1]).is_err());
        assert!(w.finish().is_err());
        let dir = Directory::new();
        let mut w = ready(&dir.0);
        w.boundaries = MAX_BOUNDARIES;
        assert!(w.event("cap", &true, Class::Boundary).is_err());
        assert!(w.finish().is_err());
    }
    #[test]
    fn capture_create_new_shape_scalar_and_checked_arithmetic() {
        let dir = Directory::new();
        let w = ready(&dir.0);
        assert!(Writer::create(&dir.0).is_err());
        drop(w);
        for (shape, count) in [
            ([0, 1, 1], 0),
            ([2, 1, 1], 1),
            ([1, 2, 1], 1),
            ([usize::MAX, 1, 1], usize::MAX),
        ] {
            let dir = Directory::new();
            let mut w = ready(&dir.0);
            let mut t = tag(count);
            t.shape = shape;
            assert!(w.half(t, count, []).is_err());
            assert!(w.finish().is_err());
        }
        let dir = Directory::new();
        let mut w = ready(&dir.0);
        assert!(w.array::<3>(tag(1), 1, "unsupported", [[0; 3]]).is_err());
        assert!(w.finish().is_err());
        let dir = Directory::new();
        let mut w = Writer::create(&dir.0).unwrap();
        assert!(w.half(tag(1), 1, [1]).is_err());
        assert!(w.finish().is_err());
    }
}

#[test]
#[ignore = "explicit Apple GPU tiny diagnostic visitor qualification"]
fn tiny_terminal_borrowed_visitors_match_owned_observations() {
    use super::*;
    use serde_json::json;
    use std::collections::BTreeSet;
    let (config, weights) = super::tests::inputs();
    let model = Model::from_raw_weights(
        config,
        weights,
        2 * 1024 * 1024,
        crate::weights::INITIAL_MEMORY_LIMIT_BYTES,
    )
    .unwrap();
    let mut session = model
        .session(Options {
            memory_limit_bytes: crate::weights::INITIAL_MEMORY_LIMIT_BYTES,
            external_resident_bytes: 32 * 1024 * 1024,
            all_logits: true,
            diagnostics: true,
        })
        .unwrap();
    let result = session.forward(&[1, 7, 3]);
    println!(
        "LFM2_CAPTURE_IO {}",
        json!({"kind":"forward","result":format!("{result:?}"),"phase_a":session.test_phase_a})
    );
    std::io::stdout().flush().unwrap();
    // Existing terminal check is authoritative before either new or old read.
    session.cache.test_safe_read().unwrap();
    let old = session.cache.test_observe().unwrap();
    let metadata = session.cache.test_capture_metadata().unwrap();
    let mut cache_ok = true;
    let mut cache_count = 0;
    let mut addresses = BTreeSet::new();
    session.cache.test_visit_buffers(|owner,role,address,device,words| {
        let reference=&old["owners"][owner][role];
        let same=reference["address"].as_u64()==Some(address as u64)
            &&reference["bytes"].as_u64()==Some((words.len()*2)as u64)
            &&reference["words"].as_array().is_some_and(|v|v.len()==words.len()
                &&v.iter().zip(words).all(|(a,&b)|a.as_u64()==Some(u64::from(b))));
        let unique=addresses.insert(address);cache_ok&=same&&unique;cache_count+=1;
        println!("LFM2_CAPTURE_IO {}",json!({"kind":"cache","owner":owner,"role":role,"address":address,
            "bytes":words.len()*2,"device":device,"sha256":hash16(words),"equal":same,"unique":unique}));
        Ok(())
    }).unwrap();
    let status_address = metadata["status"]["address"].as_u64().unwrap() as usize;
    let status_unique = addresses.insert(status_address);
    let status_ok = metadata["status"]["words"] == old["status"]
        && metadata["status"]["bytes"] == 32
        && status_unique;
    println!(
        "LFM2_CAPTURE_IO {}",
        json!({"kind":"status","actual":metadata,"equal":status_ok,"unique":status_unique})
    );
    let snapshots = session.snapshot_buffers().unwrap();
    let mut snapshot_count = 0;
    let mut snapshots_ok = true;
    session.visit_capture_slots(|slot,words| {
        let (address,half,old_words)=&snapshots[slot.slot];
        let same=*address==slot.address&&*half==slot.half&&old_words==words;
        let unique=addresses.insert(slot.address);snapshots_ok&=same&&unique;snapshot_count+=1;
        println!("LFM2_CAPTURE_IO {}",json!({"kind":"snapshot","slot":slot.slot,"layer":slot.layer,
            "stage":slot.stage,"width":slot.width,"half":slot.half,"scores":slot.scores,
            "address":slot.address,"device":slot.device,"bytes":words.len()*4,"sha256":hash32(words),"equal":same,"unique":unique}));
        Ok(())
    }).unwrap();
    let mut tensors = vec![&model.embedding];
    for l in &model.layers {
        tensors.extend([&l.gate, &l.up, &l.down]);
    }
    for a in &model.attention {
        tensors.extend([&a.q, &a.k, &a.v, &a.o]);
    }
    for c in &model.convolution {
        tensors.extend([&c.bcx, &c.output]);
    }
    let mut metadata_count = 0;
    let mut packed_ok = true;
    for (matrix, tensor) in tensors.iter().enumerate() {
        let TensorStorage::Mlx(packed) = &tensor.storage else {
            panic!("tiny original packed matrix required");
        };
        let owned = packed.test_metadata();
        packed.test_visit_metadata(|role,address,device,words| {
            let same=owned[role].0==address&&owned[role].1==words;
            let unique=addresses.insert(address);packed_ok&=same&&unique;metadata_count+=1;
            println!("LFM2_CAPTURE_IO {}",json!({"kind":"packed","matrix":matrix,"role":role,"address":address,
                "bytes":words.len()*4,"device":device,"sha256":hash32(words),"equal":same,"unique":unique}));
            Ok(())
        }).unwrap();
    }
    // Callback failures propagate rather than being swallowed/continuing.
    let mut visits = 0;
    let stopped = session.cache.test_visit_buffers(|_, _, _, _, _| {
        visits += 1;
        Err(Error::Options("synthetic capture callback failure"))
    });
    let mut snapshot_visits = 0;
    let snapshot_stopped = session.visit_capture_slots(|_, _| {
        snapshot_visits += 1;
        Err(Error::Options("synthetic capture callback failure"))
    });
    println!(
        "LFM2_CAPTURE_IO {}",
        json!({"kind":"callback_errors","cache":format!("{stopped:?}"),
        "cache_calls":visits,"snapshot":format!("{snapshot_stopped:?}"),"snapshot_calls":snapshot_visits})
    );
    // Deliberately enqueue the existing cache-stamp command without committing:
    // neither visitor may map memory or call the observation closure yet.
    let batch = session.cache.begin(&model.gpu, 1).unwrap();
    let view = session.cache.view(&batch.frames()[0]).unwrap();
    view.command.enqueue();
    let mut pending_cache_calls = 0;
    let pending_cache = session.cache.test_visit_buffers(|_, _, _, _, _| {
        pending_cache_calls += 1;
        Ok(())
    });
    let mut pending_snapshot_calls = 0;
    let pending_snapshot = session.visit_capture_slots(|_, _| {
        pending_snapshot_calls += 1;
        Ok(())
    });
    let pending_status = session.cache.test_capture_metadata();
    println!(
        "LFM2_CAPTURE_IO {}",
        json!({"kind":"pending_refusal","cache":format!("{pending_cache:?}"),
        "snapshot":format!("{pending_snapshot:?}"),"status":format!("{pending_status:?}"),
        "cache_calls":pending_cache_calls,"snapshot_calls":pending_snapshot_calls})
    );
    std::io::stdout().flush().unwrap();
    view.command.commit();
    view.command.waitUntilCompleted();
    let terminal = view.command.status() == MTLCommandBufferStatus::Completed
        && view.command.error().is_none();
    let after = session.cache.test_capture_metadata().unwrap();
    println!(
        "LFM2_CAPTURE_IO {}",
        json!({"kind":"completed_stamp","terminal_success":terminal,"actual":after})
    );
    session.cache.discard().unwrap();
    println!(
        "LFM2_CAPTURE_IO {}",
        json!({"kind":"summary","forwards":1,"tokens":3,"extra_stamp_commands":1,
        "cache_buffers":cache_count,"snapshots":snapshot_count,"packed_metadata":metadata_count,
        "cache_equal":cache_ok,"snapshots_equal":snapshots_ok,"metadata_equal":packed_ok,
        "status_equal":status_ok,"real_artifact_executed":false})
    );
    std::io::stdout().flush().unwrap();
    assert!(result.is_ok());
    assert!(cache_ok && status_ok && snapshots_ok && packed_ok);
    assert_eq!((cache_count, snapshot_count, metadata_count), (4, 57, 36));
    assert!(stopped.is_err() && snapshot_stopped.is_err());
    assert_eq!((visits, snapshot_visits), (1, 1));
    assert!(pending_cache.is_err() && pending_snapshot.is_err() && pending_status.is_err());
    assert_eq!((pending_cache_calls, pending_snapshot_calls), (0, 0));
    assert!(terminal);
}
