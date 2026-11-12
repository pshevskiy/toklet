//! Lossless diagnostic export and sparse repeat differences, outside kernels.
use half::f16;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::FileExt,
    path::Path,
};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy)]
pub enum Dtype {
    Half,
    Float,
}
impl Dtype {
    fn width(self) -> usize {
        match self {
            Self::Half => 2,
            Self::Float => 4,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Half => "<f2",
            Self::Float => "<f4",
        }
    }
}

pub fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut block = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut block)?;
        if n == 0 {
            break;
        }
        hash.update(&block[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[derive(Clone, Serialize)]
pub struct Patch {
    pub offset: usize,
    pub bytes: usize,
    pub file: String,
    pub sha256: String,
}
#[derive(Clone, Serialize)]
pub struct Array {
    pub shape: [usize; 2],
    pub dtype: &'static str,
    pub bytes: usize,
    pub sha256: String,
    pub baseline_file: String,
    pub mismatching_bytes: usize,
    pub patches: Vec<Patch>,
}
pub struct Stage {
    name: String,
    rows: usize,
    columns: usize,
    dtype: Dtype,
    written: usize,
    hash: Sha256,
    mismatches: usize,
    patches: Vec<Patch>,
    baseline_hash: Option<String>,
}
impl Stage {
    /// Conservative helper admission uses 512 bytes for every retained patch,
    /// including Vec spare capacity and both bounded owned strings.
    pub fn patch_count(&self) -> usize {
        self.patches.len()
    }

    pub fn new(name: &str, rows: usize, columns: usize, dtype: Dtype) -> Result<Self> {
        if name.is_empty()
            || name.len() > 100
            || name.starts_with('.')
            || name.contains("..")
            || !name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'.')
            || !(1..=512).contains(&rows)
            || !(1..=100352).contains(&columns)
        {
            return Err("bounded named diagnostic array required".into());
        }
        Ok(Self {
            name: name.into(),
            rows,
            columns,
            dtype,
            written: 0,
            hash: Sha256::new(),
            mismatches: 0,
            patches: Vec::new(),
            baseline_hash: None,
        })
    }

    /// Baseline output is written once. Repeats retain changed byte blocks;
    /// unchanged bytes remain reconstructible from the authenticated baseline.
    pub fn push(
        &mut self,
        values: &[f32],
        rows: usize,
        folder: &Path,
        baseline: Option<&Path>,
    ) -> Result<()> {
        let count = rows
            .checked_mul(self.columns)
            .ok_or("diagnostic shape overflow")?;
        let width = self.dtype.width();
        let size = count.checked_mul(width).ok_or("diagnostic byte overflow")?;
        let expected = self.rows * self.columns * width;
        if rows == 0
            || values.len() != count
            || self.written.checked_add(size).is_none_or(|n| n > expected)
        {
            return Err("complete bounded diagnostic rows required".into());
        }
        for &value in values {
            if !value.is_finite()
                || matches!(self.dtype, Dtype::Half)
                    && f16::from_f32(value).to_f32().to_bits() != value.to_bits()
            {
                return Err("diagnostic cannot be encoded losslessly".into());
            }
        }
        let leaf = format!("{}.bin", self.name);
        let mut original = baseline.map(|p| File::open(p.join(&leaf))).transpose()?;
        if original
            .as_ref()
            .is_some_and(|f| f.metadata().map_or(true, |m| m.len() != expected as u64))
        {
            return Err("complete baseline array required".into());
        }
        if let Some(path) = baseline.filter(|_| self.written == 0) {
            self.baseline_hash = Some(hash_file(&path.join(&leaf))?);
        }
        let mut destination = if baseline.is_none() {
            ensure_disk_space(folder, size as u64)?;
            let path = folder.join(&leaf);
            let file = if self.written == 0 {
                OpenOptions::new().write(true).create_new(true).open(path)?
            } else {
                OpenOptions::new().append(true).open(path)?
            };
            if file.metadata()?.len() != self.written as u64 {
                return Err("diagnostic append extent differs".into());
            }
            Some(file)
        } else {
            None
        };
        let mut encoded = vec![0; 64 * 1024];
        let mut prior = vec![0; 64 * 1024];
        for block in values.chunks(encoded.len() / width) {
            let bytes = block.len() * width;
            match self.dtype {
                Dtype::Half => {
                    for (value, out) in block.iter().zip(encoded[..bytes].chunks_exact_mut(2)) {
                        out.copy_from_slice(&f16::from_f32(*value).to_le_bytes());
                    }
                }
                Dtype::Float => {
                    for (value, out) in block.iter().zip(encoded[..bytes].chunks_exact_mut(4)) {
                        out.copy_from_slice(&value.to_le_bytes());
                    }
                }
            }
            self.hash.update(&encoded[..bytes]);
            if let Some(file) = &mut destination {
                file.write_all(&encoded[..bytes])?;
            } else if let Some(file) = &mut original {
                let mut read = 0;
                while read < bytes {
                    let n = file.read_at(&mut prior[read..bytes], (self.written + read) as u64)?;
                    if n == 0 {
                        return Err("baseline array ended early".into());
                    }
                    read += n;
                }
                let bad = encoded[..bytes]
                    .iter()
                    .zip(&prior[..bytes])
                    .filter(|(a, b)| a != b)
                    .count();
                if bad > 0 {
                    ensure_disk_space(folder, bytes as u64)?;
                    // Adjacent changed blocks share one payload file. This also
                    // bounds metadata for a completely different repeat.
                    if let Some(last) = self
                        .patches
                        .last_mut()
                        .filter(|p| p.offset + p.bytes == self.written)
                    {
                        OpenOptions::new()
                            .append(true)
                            .open(folder.join(&last.file))?
                            .write_all(&encoded[..bytes])?;
                        last.bytes += bytes;
                        // Authenticate the completed extent once in finish;
                        // repeatedly hashing an expanding patch is quadratic.
                        last.sha256.clear();
                    } else {
                        let leaf = format!("{}.at{:016x}.patch", self.name, self.written);
                        OpenOptions::new()
                            .write(true)
                            .create_new(true)
                            .open(folder.join(&leaf))?
                            .write_all(&encoded[..bytes])?;
                        self.patches.push(Patch {
                            offset: self.written,
                            bytes,
                            file: leaf,
                            sha256: String::new(),
                        });
                    }
                    self.mismatches += bad;
                }
            }
            self.written += bytes;
        }
        Ok(())
    }

    pub fn finish(mut self, folder: &Path, baseline: Option<&Path>) -> Result<Array> {
        if self.written != self.rows * self.columns * self.dtype.width() {
            return Err("incomplete diagnostic array".into());
        }
        let hash = format!("{:x}", self.hash.finalize());
        let leaf = format!("{}.bin", self.name);
        if baseline.is_none() && hash_file(&folder.join(&leaf))? != hash {
            return Err("saved complete baseline differs".into());
        }
        if let Some(baseline) = baseline {
            if Some(hash_file(&baseline.join(&leaf))?) != self.baseline_hash {
                return Err("baseline changed during repeat capture".into());
            }
            for patch in &mut self.patches {
                let path = folder.join(&patch.file);
                if path.metadata()?.len() != patch.bytes as u64 {
                    return Err("saved patch extent differs".into());
                }
                patch.sha256 = hash_file(&path)?;
            }
            if reconstructed_hash(&baseline.join(&leaf), folder, &self.patches, self.written)?
                != hash
            {
                return Err("saved repeat reconstruction differs".into());
            }
        }
        Ok(Array {
            shape: [self.rows, self.columns],
            dtype: self.dtype.label(),
            bytes: self.written,
            sha256: hash,
            baseline_file: leaf,
            mismatching_bytes: self.mismatches,
            patches: self.patches,
        })
    }
}

/// Re-read every saved reconstruction byte, independently of the incoming
/// diagnostic hash. No complete array is retained in memory.
fn reconstructed_hash(
    baseline: &Path,
    folder: &Path,
    patches: &[Patch],
    bytes: usize,
) -> Result<String> {
    let original = File::open(baseline)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 64 * 1024];
    let mut offset = 0;
    let mut consume = |file: &File, mut at: usize, end: usize| -> Result<()> {
        while at < end {
            let n = buffer.len().min(end - at);
            file.read_exact_at(&mut buffer[..n], at as u64)?;
            hash.update(&buffer[..n]);
            at += n;
        }
        Ok(())
    };
    for patch in patches {
        if patch.offset < offset
            || patch.bytes == 0
            || patch
                .offset
                .checked_add(patch.bytes)
                .is_none_or(|end| end > bytes)
        {
            return Err("invalid repeat patch range".into());
        }
        consume(&original, offset, patch.offset)?;
        let file = File::open(folder.join(&patch.file))?;
        consume(&file, 0, patch.bytes)?;
        offset = patch.offset + patch.bytes;
    }
    consume(&original, offset, bytes)?;
    Ok(format!("{:x}", hash.finalize()))
}

pub fn ensure_disk_space(folder: &Path, additional: u64) -> Result<()> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(folder.as_os_str().as_bytes())?;
    let mut status = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: NUL-terminated live path and valid writable statvfs storage;
    // assume_init is reached only after the system call initializes it.
    if unsafe { libc::statvfs(path.as_ptr(), status.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let status = unsafe { status.assume_init() };
    let available = u128::from(status.f_bavail) * u128::from(status.f_frsize);
    if available < u128::from(additional) + 2 * 1024 * 1024 * 1024 {
        return Err("diagnostic output would cross the 2 GiB free-space floor".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!(
                "toklet-granite-array-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn half_rows_preserve_original_words_and_complete_extent() {
        let t = Temp::new();
        let mut s = Stage::new("embedding", 2, 2, Dtype::Half).unwrap();
        s.push(&[1., -0.], 1, &t.0, None).unwrap();
        s.push(&[2., 3.], 1, &t.0, None).unwrap();
        let a = s.finish(&t.0, None).unwrap();
        assert_eq!(a.bytes, 8);
        assert_eq!(
            std::fs::read(t.0.join("embedding.bin")).unwrap(),
            [0, 0x3c, 0, 0x80, 0, 0x40, 0, 0x42]
        );
        // Exhaust every finite original Half word, including signed zero,
        // subnormals and both signs, across the encoder's 64 KiB block tail.
        let words: Vec<_> = (0..=u16::MAX)
            .filter(|word| f16::from_bits(*word).is_finite())
            .collect();
        let values: Vec<_> = words
            .iter()
            .map(|word| f16::from_bits(*word).to_f32())
            .collect();
        let mut all = Stage::new("all_half_words", 1, words.len(), Dtype::Half).unwrap();
        all.push(&values, 1, &t.0, None).unwrap();
        let array = all.finish(&t.0, None).unwrap();
        let expected: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        assert_eq!(
            std::fs::read(t.0.join("all_half_words.bin")).unwrap(),
            expected
        );
        assert_eq!(array.bytes, 2 * words.len());
    }
    #[test]
    fn changed_repeat_blocks_reconstruct_the_entire_actual_array() {
        let base = Temp::new();
        let delta = Temp::new();
        let mut first = Stage::new("logits", 2, 2, Dtype::Float).unwrap();
        first.push(&[1., 2., 3., 4.], 2, &base.0, None).unwrap();
        first.finish(&base.0, None).unwrap();
        let mut repeat = Stage::new("logits", 2, 2, Dtype::Float).unwrap();
        repeat.push(&[1., 2.], 1, &delta.0, Some(&base.0)).unwrap();
        repeat.push(&[3., 5.], 1, &delta.0, Some(&base.0)).unwrap();
        let a = repeat.finish(&delta.0, Some(&base.0)).unwrap();
        assert_eq!(a.patches.len(), 1);
        assert_eq!(a.patches[0].offset, 8);
        let mut raw = std::fs::read(base.0.join("logits.bin")).unwrap();
        for p in &a.patches {
            raw[p.offset..p.offset + p.bytes]
                .copy_from_slice(&std::fs::read(delta.0.join(&p.file)).unwrap());
        }
        assert_eq!(format!("{:x}", Sha256::digest(raw)), a.sha256);
        // Adjacent changed pushes merge without losing offset/hash evidence.
        let merged = Temp::new();
        let mut repeat = Stage::new("logits", 2, 2, Dtype::Float).unwrap();
        repeat.push(&[5., 6.], 1, &merged.0, Some(&base.0)).unwrap();
        repeat.push(&[7., 8.], 1, &merged.0, Some(&base.0)).unwrap();
        assert_eq!(repeat.patch_count(), 1);
        let array = repeat.finish(&merged.0, Some(&base.0)).unwrap();
        assert_eq!(array.patches.len(), 1);
        assert_eq!((array.patches[0].offset, array.patches[0].bytes), (0, 16));
        assert_eq!(
            hash_file(&merged.0.join(&array.patches[0].file)).unwrap(),
            array.sha256
        );
    }
    #[test]
    fn unchanged_repeat_keeps_no_duplicate_payload() {
        let base = Temp::new();
        let delta = Temp::new();
        let values = [1., 2.];
        let mut first = Stage::new("output", 1, 2, Dtype::Half).unwrap();
        first.push(&values, 1, &base.0, None).unwrap();
        let before = first.finish(&base.0, None).unwrap();
        let mut repeat = Stage::new("output", 1, 2, Dtype::Half).unwrap();
        repeat.push(&values, 1, &delta.0, Some(&base.0)).unwrap();
        let after = repeat.finish(&delta.0, Some(&base.0)).unwrap();
        assert_eq!(before.sha256, after.sha256);
        assert!(after.patches.is_empty());
        assert_eq!(std::fs::read_dir(&delta.0).unwrap().count(), 0);
    }
    #[test]
    fn nonrepresentable_nonfinite_and_incomplete_arrays_reject() {
        let t = Temp::new();
        for value in [0.1, f32::INFINITY, f32::NAN] {
            let mut s = Stage::new("bad", 1, 1, Dtype::Half).unwrap();
            assert!(s.push(&[value], 1, &t.0, None).is_err());
        }
        assert!(
            Stage::new("short", 2, 1, Dtype::Float)
                .unwrap()
                .finish(&t.0, None)
                .is_err()
        );
        assert!(Stage::new("../escape", 1, 1, Dtype::Float).is_err());
        let mut full = Stage::new("full", 1, 1, Dtype::Float).unwrap();
        assert!(full.push(&[], 0, &t.0, None).is_err());
        assert!(full.push(&[1., 2.], 1, &t.0, None).is_err());
        full.push(&[1.], 1, &t.0, None).unwrap();
        assert!(full.push(&[2.], 1, &t.0, None).is_err());
        full.finish(&t.0, None).unwrap();
        let delta = Temp::new();
        let mut repeat = Stage::new("full", 1, 1, Dtype::Float).unwrap();
        repeat.push(&[1.], 1, &delta.0, Some(&t.0)).unwrap();
        std::fs::write(t.0.join("full.bin"), 2_f32.to_le_bytes()).unwrap();
        assert!(repeat.finish(&delta.0, Some(&t.0)).is_err());
        let mut patched = Stage::new("full", 1, 1, Dtype::Float).unwrap();
        patched.push(&[3.], 1, &delta.0, Some(&t.0)).unwrap();
        let patch = delta.0.join("full.at0000000000000000.patch");
        std::fs::write(patch, 4_f32.to_le_bytes()).unwrap();
        assert!(patched.finish(&delta.0, Some(&t.0)).is_err());
    }
}
