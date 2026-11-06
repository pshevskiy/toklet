//! Authenticated official BF16 per-layer embedding rows over a held descriptor.
use super::{CONTAINER_BYTES, HEADER_BYTES, ModalityPolicy, TextPlan, add_bytes, inspect, invalid};
use crate::weights::Error;
use sha2::{Digest, Sha256};
use std::{
    fs::{File, Metadata},
    io,
    ops::Range,
    os::unix::fs::{FileExt, MetadataExt},
};

const CONTAINER_SHA256: &str = "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550";
const SCAN_BYTES: usize = 64 * 1024;
const METADATA_ALLOWANCE_BYTES: usize = 8 * 1024 * 1024;
const MAX_BATCH: usize = 32;
const ROW_DIGEST_BYTES: usize = size_of::<[u8; 32]>();

#[derive(Clone, Copy)]
struct Table {
    start: u64,
    end: u64,
    rows: usize,
    columns: usize,
    row_bytes: usize,
}

impl Table {
    fn new(start: u64, rows: usize, columns: usize, file_bytes: u64) -> Result<Self, Error> {
        let row_bytes = columns
            .checked_mul(2)
            .filter(|n| (1..=SCAN_BYTES).contains(n))
            .ok_or_else(|| invalid("Gemma4 BF16 PLE row width"))?;
        let end = row_bytes
            .checked_mul(rows)
            .and_then(|n| start.checked_add(n as u64))
            .filter(|&n| rows != 0 && n <= file_bytes)
            .ok_or_else(|| invalid("Gemma4 BF16 PLE row range"))?;
        Ok(Self {
            start,
            end,
            rows,
            columns,
            row_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Memory {
    pub retained_bytes: usize,
    pub scan_bytes: usize,
    pub metadata_allowance_bytes: usize,
    pub load_bytes: usize,
}

/// Owns the exact file descriptor used for whole-file authentication and
/// the original SHA256 of each PLE row. Sessions own their gather scratch.
pub struct Source {
    file: File,
    table: Table,
    row_digests: Vec<[u8; 32]>,
    memory: Memory,
}

impl Source {
    pub fn open_text_only(
        file: File,
        config: &[u8],
        available_bytes: usize,
    ) -> Result<Self, Error> {
        let (before, _, table) = read_plan(&file, config, available_bytes)?;
        authenticate(file, before, table, CONTAINER_SHA256, available_bytes)
    }

    pub fn memory(&self) -> Memory {
        self.memory
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.table.rows, self.table.columns)
    }

    pub fn workspace_bytes(&self, max_rows: usize) -> Result<usize, Error> {
        Ok(Workspace::layout(self.table.row_bytes, max_rows)?.1)
    }

    pub fn workspace(&self, max_rows: usize, available_bytes: usize) -> Result<Workspace, Error> {
        Workspace::new(self.table.row_bytes, max_rows, available_bytes)
    }

    /// Authenticate every requested row in private scratch before changing any
    /// caller output. The stable descriptor and per-row hashes bind consumed
    /// bytes even if the original path is replaced or its contents change.
    pub fn gather(
        &self,
        workspace: &mut Workspace,
        ids: &[u32],
        output: &mut [f32],
    ) -> Result<(), Error> {
        let table = self.table;
        if ids.is_empty()
            || ids.len() > workspace.max_rows
            || workspace.row_bytes != table.row_bytes
            || ids.len().checked_mul(table.columns) != Some(output.len())
            || ids.iter().any(|&id| id as usize >= table.rows)
        {
            return Err(invalid("Gemma4 BF16 PLE gather request differs"));
        }
        let raw = &mut workspace.raw[..ids.len() * table.row_bytes];
        for (&id, row) in ids.iter().zip(raw.chunks_exact_mut(table.row_bytes)) {
            let offset = table.start + u64::from(id) * table.row_bytes as u64;
            read(&self.file, row, offset)?;
            let digest: [u8; 32] = Sha256::digest(&*row).into();
            if digest != self.row_digests[id as usize] {
                return Err(invalid("Gemma4 BF16 PLE row authentication differs"));
            }
        }
        // Authentication admitted only finite BF16. Widening is exact and
        // infallible, so publication starts only after every possible error.
        for (value, raw) in output.iter_mut().zip(raw.chunks_exact(2)) {
            let word = u16::from_le_bytes([raw[0], raw[1]]);
            *value = f32::from_bits(u32::from(word) << 16);
        }
        Ok(())
    }
}

pub struct Workspace {
    row_bytes: usize,
    max_rows: usize,
    raw: Vec<u8>,
}

impl Workspace {
    fn new(row_bytes: usize, max_rows: usize, available_bytes: usize) -> Result<Self, Error> {
        let (len, minimum) = Self::layout(row_bytes, max_rows)?;
        admit(minimum, available_bytes)?;
        let mut raw = reserved(len)?;
        admit(
            add_bytes(size_of::<Self>(), raw.capacity())?,
            available_bytes,
        )?;
        raw.resize(len, 0);
        Ok(Self {
            row_bytes,
            max_rows,
            raw,
        })
    }

    fn layout(row_bytes: usize, max_rows: usize) -> Result<(usize, usize), Error> {
        if !(1..=MAX_BATCH).contains(&max_rows) {
            return Err(invalid("Gemma4 BF16 PLE workspace rows outside bounds"));
        }
        let len = row_bytes
            .checked_mul(max_rows)
            .ok_or_else(|| invalid("Gemma4 BF16 PLE workspace size overflow"))?;
        Ok((len, add_bytes(size_of::<Self>(), len)?))
    }

    pub fn owned_bytes(&self) -> usize {
        size_of::<Self>() + self.raw.capacity()
    }
}

fn read_plan(
    file: &File,
    config: &[u8],
    available_bytes: usize,
) -> Result<([u64; 7], TextPlan, Table), Error> {
    let before = file_identity(file)?;
    if before[2] != CONTAINER_BYTES {
        return Err(invalid("Gemma4 BF16 source length differs"));
    }
    admit(
        memory(super::super::VOCAB, SCAN_BYTES)?.load_bytes,
        available_bytes,
    )?;
    let mut len = [0u8; 8];
    read(file, &mut len, 0)?;
    if u64::from_le_bytes(len) != HEADER_BYTES as u64 {
        return Err(invalid("Gemma4 BF16 source header length differs"));
    }
    let mut header = reserved(HEADER_BYTES)?;
    header.resize(HEADER_BYTES, 0);
    read(file, &mut header, 8)?;
    let plan = inspect(config, &header, before[2], ModalityPolicy::TextOnly)?;
    let ple = plan.per_layer_embedding();
    let table = Table::new(
        add_bytes(plan.payload_offset(), ple.range.start)? as u64,
        ple.shape[0],
        ple.shape[1],
        before[2],
    )?;
    if table.end != add_bytes(plan.payload_offset(), ple.range.end)? as u64 {
        return Err(invalid("Gemma4 BF16 PLE source range differs"));
    }
    Ok((before, plan, table))
}

fn memory(digest_capacity: usize, scan_bytes: usize) -> Result<Memory, Error> {
    let digests = digest_capacity
        .checked_mul(ROW_DIGEST_BYTES)
        .ok_or_else(|| invalid("Gemma4 BF16 PLE digest capacity overflow"))?;
    let retained = add_bytes(size_of::<Source>(), digests)?;
    Ok(Memory {
        retained_bytes: retained,
        scan_bytes,
        metadata_allowance_bytes: METADATA_ALLOWANCE_BYTES,
        load_bytes: add_bytes(add_bytes(retained, scan_bytes)?, METADATA_ALLOWANCE_BYTES)?,
    })
}

fn authenticate(
    file: File,
    before: [u64; 7],
    table: Table,
    expected: &str,
    available_bytes: usize,
) -> Result<Source, Error> {
    let mut scan = Scan::new(table.rows, available_bytes)?;
    hash_span(&file, &mut scan.scratch, 0..table.start, &mut scan.digest)?;
    scan.ple(&file, table)?;
    hash_span(
        &file,
        &mut scan.scratch,
        table.end..before[2],
        &mut scan.digest,
    )?;
    scan.finish(file, before, table, expected)
}

struct Scan {
    scratch: Vec<u8>,
    row_digests: Vec<[u8; 32]>,
    digest: Sha256,
    memory: Memory,
}

impl Scan {
    fn new(rows: usize, available_bytes: usize) -> Result<Self, Error> {
        admit(memory(rows, SCAN_BYTES)?.load_bytes, available_bytes)?;
        let row_digests = reserved(rows)?;
        let mut scratch = reserved(SCAN_BYTES)?;
        let memory = memory(row_digests.capacity(), scratch.capacity())?;
        admit(memory.load_bytes, available_bytes)?;
        scratch.resize(SCAN_BYTES, 0);
        Ok(Self {
            scratch,
            row_digests,
            digest: Sha256::new(),
            memory,
        })
    }

    fn ple(&mut self, file: &File, table: Table) -> Result<(), Error> {
        let batch_bytes = SCAN_BYTES / table.row_bytes * table.row_bytes;
        let mut offset = table.start;
        while offset < table.end {
            let bytes = (table.end - offset).min(batch_bytes as u64) as usize;
            let block = &mut self.scratch[..bytes];
            read(file, block, offset)?;
            self.digest.update(&*block);
            for row in block.chunks_exact(table.row_bytes) {
                if row
                    .chunks_exact(2)
                    .any(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7f80 == 0x7f80)
                {
                    return Err(invalid("nonfinite Gemma4 BF16 PLE row"));
                }
                self.row_digests.push(Sha256::digest(row).into());
            }
            offset += bytes as u64;
        }
        Ok(())
    }

    fn finish(
        self,
        file: File,
        before: [u64; 7],
        table: Table,
        expected: &str,
    ) -> Result<Source, Error> {
        if self.row_digests.len() != table.rows
            || format!("{:x}", self.digest.finalize()) != expected
            || file_identity(&file)? != before
        {
            return Err(invalid("Gemma4 BF16 source authentication differs"));
        }
        Ok(Source {
            file,
            table,
            row_digests: self.row_digests,
            memory: self.memory,
        })
    }
}

fn hash_span(
    file: &File,
    scratch: &mut [u8],
    span: Range<u64>,
    digest: &mut Sha256,
) -> Result<(), Error> {
    let mut offset = span.start;
    while offset < span.end {
        let bytes = (span.end - offset).min(scratch.len() as u64) as usize;
        let block = &mut scratch[..bytes];
        read(file, block, offset)?;
        digest.update(&*block);
        offset += bytes as u64;
    }
    Ok(())
}

fn file_identity(file: &File) -> Result<[u64; 7], Error> {
    let m: Metadata = io_result(file.metadata())?;
    if !m.is_file() {
        return Err(invalid("Gemma4 BF16 source must be a regular file"));
    }
    Ok([
        m.dev(),
        m.ino(),
        m.len(),
        m.mtime() as u64,
        m.mtime_nsec() as u64,
        m.ctime() as u64,
        m.ctime_nsec() as u64,
    ])
}

fn read(file: &File, bytes: &mut [u8], offset: u64) -> Result<(), Error> {
    io_result(file.read_exact_at(bytes, offset))
}

fn io_result<T>(result: io::Result<T>) -> Result<T, Error> {
    result.map_err(|source| Error::Io {
        path: "Gemma4 BF16 source descriptor".into(),
        source,
    })
}

fn reserved<T>(count: usize) -> Result<Vec<T>, Error> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Allocation)?;
    Ok(values)
}

fn admit(required: usize, limit: usize) -> Result<(), Error> {
    if required > limit {
        return Err(Error::Budget { required, limit });
    }
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub(crate) mod delivery;

#[cfg(test)]
mod tests;
