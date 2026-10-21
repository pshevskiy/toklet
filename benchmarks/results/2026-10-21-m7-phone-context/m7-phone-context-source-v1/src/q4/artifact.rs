//! Source-bound packed model inventory. No GPU or model-sized CPU allocation.
use super::{Format, HEADER_BYTES, Matrix};
use crate::weights::{Error, INITIAL_MEMORY_LIMIT_BYTES};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::{Component, Path, PathBuf},
};

const METADATA_LIMIT: usize = 1024 * 1024;
const ALLOWANCE: usize = 64 * 1024 * 1024;
const CONVERSION: usize = 64 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    file: String,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    name: String,
    file: String,
    rows: usize,
    columns: usize,
    packed_bytes: usize,
    packed_sha256: String,
    source_f32_sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format: String,
    quantizer: String,
    config_sha256: String,
    sources: Vec<Source>,
    matrices: Vec<Entry>,
    packed_matrix_bytes: usize,
    exclusions: Vec<String>,
    excluded_dtype: String,
}
fn invalid(reason: &str) -> Error {
    Error::Invalid(reason.into())
}
fn io<T>(path: &Path, result: std::io::Result<T>) -> Result<T, Error> {
    result.map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })
}
fn plain(name: &str) -> Result<(), Error> {
    if name.is_empty()
        || name.contains(['/', '\\'])
        || !matches!(
            Path::new(name).components().collect::<Vec<_>>().as_slice(),
            [Component::Normal(_)]
        )
    {
        return Err(invalid(
            "Q4 artifact filenames must be plain local filenames",
        ));
    }
    Ok(())
}
fn sha(value: &str) -> Result<(), Error> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v))
    {
        return Err(invalid("invalid Q4 SHA256 identity"));
    }
    Ok(())
}
fn digest(path: &Path, expected_bytes: Option<usize>) -> Result<String, Error> {
    let mut file = io(path, File::open(path))?;
    let size = io(path, file.metadata())?.len();
    if expected_bytes.is_some_and(|n| n as u64 != size) {
        return Err(invalid("Q4 artifact length mismatch"));
    }
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; CONVERSION];
    let mut count = 0_u64;
    loop {
        let length = io(path, file.read(&mut buffer))?;
        if length == 0 {
            break;
        }
        count = count
            .checked_add(length as u64)
            .ok_or_else(|| invalid("Q4 digest length overflow"))?;
        if count > size {
            return Err(invalid("Q4 source grew during hash"));
        }
        hash.update(&buffer[..length]);
    }
    if count != size {
        return Err(invalid("Q4 source shortened during hash"));
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// A validated model inventory. FP16 exclusions remain in the original source.
/// Callers add their actual excluded weight/session buffers to this admission.
pub struct Artifact {
    directory: PathBuf,
    source: PathBuf,
    manifest: Manifest,
    format: Format,
    pub packed_matrix_bytes: usize,
    pub largest_matrix_bytes: usize,
    pub admitted_packed_load_bytes: usize,
}
impl Artifact {
    pub fn format(&self) -> Format {
        self.format
    }
    pub fn open(
        directory: &Path,
        source: &Path,
        expected: &BTreeMap<String, Vec<usize>>,
        source_files: &[String],
        limit: usize,
    ) -> Result<Self, Error> {
        if limit == 0 || limit > INITIAL_MEMORY_LIMIT_BYTES {
            return Err(invalid("Q4 loading budget must fit 6 GiB"));
        }
        if expected.is_empty()
            || expected.len() > 7 * 1024
            || expected.values().any(|s| s.len() != 2)
            || source_files.is_empty()
            || source_files.len() > 1025
        {
            return Err(invalid("Q4 expected inventory exceeds bounded format"));
        }
        let path = directory.join("manifest.json");
        let file = io(&path, File::open(&path))?;
        let size = io(&path, file.metadata())?.len();
        if size > METADATA_LIMIT as u64 {
            return Err(invalid("Q4 manifest exceeds 1 MiB"));
        }
        let mut data = Vec::new();
        io(
            &path,
            file.take((METADATA_LIMIT + 1) as u64)
                .read_to_end(&mut data),
        )?;
        if data.len() as u64 != size {
            return Err(invalid("Q4 manifest size changed"));
        }
        let manifest: Manifest = serde_json::from_slice(&data)?;
        let format = match manifest.format.as_str() {
            "toklet-q4-model-v1" if manifest.quantizer == Format::V1.quantizer() => Format::V1,
            "toklet-q4-model-v2" if manifest.quantizer == Format::V2.quantizer() => Format::V2,
            _ => return Err(invalid("unsupported Q4 model format or quantizer")),
        };
        if manifest.excluded_dtype != "F16"
            || manifest.exclusions != ["norms", "qkv_biases", "tied_embedding_output"]
        {
            return Err(invalid("unsupported Q4 model format or exclusions"));
        }
        sha(&manifest.config_sha256)?;
        let required_files: BTreeSet<_> = source_files.iter().map(String::as_str).collect();
        if required_files.len() != source_files.len() {
            return Err(invalid("duplicate expected source file"));
        }
        let mut names = BTreeSet::new();
        for entry in &manifest.sources {
            plain(&entry.file)?;
            sha(&entry.sha256)?;
            if !names.insert(entry.file.as_str()) {
                return Err(invalid("duplicate Q4 source identity"));
            }
        }
        if names != required_files {
            return Err(invalid("Q4 source inventory mismatch"));
        }
        let mut shapes = expected.clone();
        let mut bytes = 0_usize;
        let mut largest = 0;
        for entry in &manifest.matrices {
            plain(&entry.file)?;
            sha(&entry.packed_sha256)?;
            sha(&entry.source_f32_sha256)?;
            if entry.file != format!("{}.q4", entry.name)
                || entry.rows == 0
                || entry.columns == 0
                || shapes.remove(&entry.name).as_deref() != Some(&[entry.rows, entry.columns])
            {
                return Err(invalid("Q4 matrix inventory or shape mismatch"));
            }
            let stride = entry
                .columns
                .div_ceil(format.group())
                .checked_mul(format.record_bytes())
                .and_then(|v| v.checked_add(63))
                .map(|v| v / 64 * 64)
                .ok_or_else(|| invalid("Q4 size overflow"))?;
            let length = entry
                .rows
                .checked_mul(stride)
                .and_then(|n| n.checked_add(HEADER_BYTES))
                .ok_or_else(|| invalid("Q4 size overflow"))?;
            if length != entry.packed_bytes || length > isize::MAX as usize {
                return Err(invalid("Q4 matrix byte count mismatch"));
            }
            bytes = bytes
                .checked_add(length)
                .ok_or_else(|| invalid("Q4 total overflow"))?;
            largest = largest.max(length);
        }
        if !shapes.is_empty() || bytes != manifest.packed_matrix_bytes {
            return Err(invalid("Q4 packed inventory or total mismatch"));
        }
        let admitted = bytes
            .checked_add(largest)
            .and_then(|v| v.checked_add(ALLOWANCE + CONVERSION))
            .ok_or_else(|| invalid("Q4 admission overflow"))?;
        if admitted > limit {
            return Err(Error::Budget {
                required: admitted,
                limit,
            });
        }
        let artifact = Self {
            directory: directory.to_owned(),
            source: source.to_owned(),
            manifest,
            format,
            packed_matrix_bytes: bytes,
            largest_matrix_bytes: largest,
            admitted_packed_load_bytes: admitted,
        };
        artifact.verify_sources()?;
        for entry in &artifact.manifest.matrices {
            if digest(&directory.join(&entry.file), Some(entry.packed_bytes))?
                != entry.packed_sha256
            {
                return Err(invalid("Q4 packed artifact SHA256 mismatch"));
            }
        }
        Ok(artifact)
    }
    pub fn verify_sources(&self) -> Result<(), Error> {
        if digest(&self.source.join("config.json"), None)? != self.manifest.config_sha256 {
            return Err(invalid("Q4 source config SHA256 mismatch"));
        }
        for entry in &self.manifest.sources {
            if digest(&self.source.join(&entry.file), None)? != entry.sha256 {
                return Err(invalid("Q4 source SHA256 mismatch"));
            }
        }
        Ok(())
    }
    /// Bounded one-matrix staging allocation; rehash the exact bytes used.
    pub fn read_matrix(&self, name: &str) -> Result<Vec<u8>, Error> {
        let entry = self
            .manifest
            .matrices
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| invalid("Q4 matrix absent"))?;
        let path = self.directory.join(&entry.file);
        let file = io(&path, File::open(&path))?;
        if io(&path, file.metadata())?.len() != entry.packed_bytes as u64 {
            return Err(invalid("Q4 matrix length changed"));
        }
        let mut data = Vec::new();
        data.try_reserve_exact(entry.packed_bytes)
            .map_err(|_| Error::Allocation)?;
        data.resize(entry.packed_bytes, 0);
        let mut file = file;
        io(&path, file.read_exact(&mut data))?;
        let mut trailing = [0];
        if io(&path, file.read(&mut trailing))? != 0 {
            return Err(invalid("Q4 matrix grew"));
        }
        if format!("{:x}", Sha256::digest(&data)) != entry.packed_sha256 {
            return Err(invalid("Q4 matrix changed after admission"));
        }
        let matrix = Matrix::parse(&data).map_err(|e| invalid(e.0))?;
        if matrix.rows() != entry.rows
            || matrix.columns() != entry.columns
            || matrix.format() != self.format
        {
            return Err(invalid("Q4 header shape differs from manifest"));
        }
        Ok(data)
    }
}
