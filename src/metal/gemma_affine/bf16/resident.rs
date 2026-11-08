//! Private one-pass delivery of original BF16 projection storage.
use super::{super::resident::bindings::Bindings, *};
use crate::mlx::gemma4::bf16::{
    TextPlan,
    rows::{Source, delivery::Prepared},
};
use std::{fs::File, io::Read};

pub(super) mod graph;
pub(super) mod planning;
pub(super) mod profile;

const TILE_ROWS: usize = 8192;
const MATRIX_SLOTS: usize = 278;
const VECTOR_SLOTS: usize = 262;
const VECTOR_STAGE_BYTES: usize = 1536 * 2;

struct Matrix {
    rows: usize,
    columns: usize,
    tiles: Vec<Buffer<u32>>,
    maximum: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Bf16LoadReport {
    pub retained_bytes: usize,
    pub admitted_load_bytes: usize,
    pub source_retained_bytes: usize,
    pub maximum_abs_coefficient: f64,
}

/// Immutable resident projection and tied-embedding storage. The PLE table
/// stays on the authenticated source descriptor. Graph access is separate.
pub struct Bf16Weights {
    matrices: Vec<Option<Matrix>>,
    vectors: Vec<Vec<f32>>,
    source: Source,
    bindings: Bindings,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    report: Bf16LoadReport,
}

impl Bf16Weights {
    pub fn load_text_only(
        context: &Bf16Context,
        file: File,
        config: &[u8],
        available_bytes: usize,
    ) -> Result<Self> {
        let prepared = Prepared::new(file, config, available_bytes)?;
        Self::load_prepared(context, prepared, available_bytes)
    }

    #[cfg(test)]
    pub(crate) fn load_modified_parent_for_quality(
        context: &Bf16Context,
        file: File,
        config: &[u8],
        available_bytes: usize,
    ) -> Result<Self> {
        Self::load_prepared(
            context,
            Prepared::modified_parent(file, config, available_bytes)?,
            available_bytes,
        )
    }

    fn load_prepared(
        context: &Bf16Context,
        prepared: Prepared,
        available_bytes: usize,
    ) -> Result<Self> {
        let source_minimum = prepared.minimum_source_memory()?;
        let plan = prepared.plan();
        let gpu = &context.0.gpu;
        let storage_minimum = plan_storage(plan, gpu.device.maxBufferLength())?;
        let policies = add(CONTEXT_ALLOWANCE_BYTES, VECTOR_STAGE_BYTES)?;
        let fixed = add(policies, source_minimum.load_bytes)?;
        check_budget(add(storage_minimum, fixed)?, available_bytes)?;

        let bindings = Bindings::bf16(plan)?;
        let mut matrices = reserve::<Option<Matrix>>(MATRIX_SLOTS)?;
        let mut vectors = reserve::<Vec<f32>>(VECTOR_SLOTS)?;
        let mut storage_bytes = add(
            storage_minimum,
            add(
                multiply(
                    matrices.capacity() - MATRIX_SLOTS,
                    size_of::<Option<Matrix>>(),
                )?,
                multiply(vectors.capacity() - VECTOR_SLOTS, size_of::<Vec<f32>>())?,
            )?,
        )?;
        check_budget(add(storage_bytes, fixed)?, available_bytes)?;
        matrices.resize_with(MATRIX_SLOTS, || None);
        vectors.resize_with(VECTOR_SLOTS, Vec::new);

        let source_limit = available_bytes - add(policies, storage_bytes)?;
        let source = prepared.deliver::<Error>(
            source_limit,
            |slot, _name, tensor, source_memory, reader| {
                let storage_limit = available_bytes - add(policies, source_memory.load_bytes)?;
                match *tensor.shape.as_slice() {
                    [rows, columns] => {
                        let target = matrices
                            .get_mut(slot)
                            .filter(|item| item.is_none() && slot != bindings.ple_embedding_index())
                            .ok_or(Error::Options("Gemma BF16 matrix delivery slot differs"))?;
                        *target = Some(upload_matrix(
                            gpu,
                            reader,
                            rows,
                            columns,
                            &mut storage_bytes,
                            storage_limit,
                        )?);
                    }
                    [values] => {
                        let target = vectors
                            .get_mut(slot)
                            .filter(|v| v.is_empty())
                            .ok_or(Error::Options("Gemma BF16 vector delivery slot differs"))?;
                        *target = load_vector(reader, values, &mut storage_bytes, storage_limit)?;
                    }
                    _ => return Err(Error::Options("Gemma BF16 delivery geometry differs")),
                }
                Ok(())
            },
        )?;
        if matrices
            .iter()
            .enumerate()
            .any(|(index, matrix)| matrix.is_none() != (index == bindings.ple_embedding_index()))
            || vectors.iter().any(Vec::is_empty)
        {
            return Err(Error::Options("Gemma BF16 delivery inventory incomplete"));
        }

        let source_memory = source.memory();
        let retained_bytes = add(storage_bytes, source_memory.retained_bytes)?;
        let admitted_load_bytes = add(add(storage_bytes, source_memory.load_bytes)?, policies)?;
        check_budget(admitted_load_bytes, available_bytes)?;
        let report = Bf16LoadReport {
            retained_bytes,
            admitted_load_bytes,
            source_retained_bytes: source_memory.retained_bytes,
            maximum_abs_coefficient: matrices
                .iter()
                .flatten()
                .fold(0.0_f64, |v, m| v.max(m.maximum)),
        };
        Ok(Self {
            matrices,
            vectors,
            source,
            bindings,
            device: gpu.device.clone(),
            report,
        })
    }

    pub fn report(&self) -> &Bf16LoadReport {
        &self.report
    }

    pub fn matrix_count(&self) -> usize {
        self.matrices.iter().flatten().count()
    }

    pub fn vector_count(&self) -> usize {
        self.vectors.len()
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub fn per_layer_embeddings(&self) -> &Source {
        &self.source
    }

    /// Copy BF16 rows from the tied resident table. All request validation
    /// precedes publication. The table is immutable and originally finite.
    pub fn gather_embeddings(&self, ids: &[u32], output: &mut [f32]) -> Result<()> {
        let weights = self.matrices[self.bindings.embedding_index()]
            .as_ref()
            .ok_or(Error::Options("Gemma BF16 resident embedding absent"))?;
        if !(1..=32).contains(&ids.len())
            || ids.len().checked_mul(weights.columns) != Some(output.len())
            || ids.iter().any(|&id| id as usize >= weights.rows)
        {
            return Err(Error::Options("Gemma BF16 resident gather request differs"));
        }
        let row_words = weights.columns / 2;
        for (&id, out) in ids.iter().zip(output.chunks_exact_mut(weights.columns)) {
            let id = id as usize;
            let start = (id % TILE_ROWS) * row_words;
            weights.tiles[id / TILE_ROWS].with_slice(|words| {
                for (pair, &word) in out
                    .chunks_exact_mut(2)
                    .zip(&words[start..start + row_words])
                {
                    pair[0] = f32::from_bits((word & 0xffff) << 16);
                    pair[1] = f32::from_bits(word & 0xffff_0000);
                }
            });
        }
        Ok(())
    }
}

fn plan_storage(plan: &TextPlan, device_max: usize) -> Result<usize> {
    // Source owns and charges its inline representation and row digests.
    let mut bytes = add(
        size_of::<Bf16Weights>() - size_of::<Source>(),
        add(
            multiply(MATRIX_SLOTS, size_of::<Option<Matrix>>())?,
            multiply(VECTOR_SLOTS, size_of::<Vec<f32>>())?,
        )?,
    )?;
    let ple = plan.per_layer_embedding();
    for tensor in plan.text_tensors().values() {
        if std::ptr::eq(tensor, ple) {
            continue;
        }
        let owned = match *tensor.shape.as_slice() {
            [rows, columns] => {
                projection_sizes(rows.min(TILE_ROWS), columns, 1)?;
                let count = rows.div_ceil(TILE_ROWS);
                let maximum = tile_bytes(rows.min(TILE_ROWS), columns)?;
                validate_buffer_bytes(maximum, device_max)?;
                add(
                    tensor.range.len(),
                    multiply(count, add(size_of::<Buffer<u32>>(), GUARD * 4)?)?,
                )?
            }
            [values] if values * 2 <= VECTOR_STAGE_BYTES => multiply(values, 4)?,
            _ => return Err(Error::Options("Gemma BF16 resident plan geometry differs")),
        };
        bytes = add(bytes, owned)?;
    }
    Ok(bytes)
}

fn upload_matrix(
    gpu: &Gpu,
    reader: &mut impl Read,
    rows: usize,
    columns: usize,
    storage: &mut usize,
    limit: usize,
) -> Result<Matrix> {
    let count = rows.div_ceil(TILE_ROWS);
    let mut tiles = reserve::<Buffer<u32>>(count)?;
    charge_capacity(
        storage,
        limit,
        tiles.capacity(),
        count,
        size_of::<Buffer<u32>>(),
    )?;
    let mut maximum = 0.0_f64;
    for first in (0..rows).step_by(TILE_ROWS) {
        let bytes = tile_bytes((rows - first).min(TILE_ROWS), columns)?;
        let mut tile = Buffer::<u32>::new(gpu, bytes / 4)?;
        let tile_maximum = tile.with_bytes_mut(|storage| {
            let (coefficients, guard) = storage.split_at_mut(bytes - GUARD * 4);
            for word in guard.chunks_exact_mut(4) {
                word.copy_from_slice(&SENTINEL.to_le_bytes());
            }
            read(reader, coefficients)?;
            maximum_coefficient(coefficients)
        })?;
        maximum = maximum.max(tile_maximum);
        tiles.push(tile);
    }
    Ok(Matrix {
        rows,
        columns,
        tiles,
        maximum,
    })
}

fn load_vector(
    reader: &mut impl Read,
    values: usize,
    storage: &mut usize,
    limit: usize,
) -> Result<Vec<f32>> {
    let mut vector = reserve::<f32>(values)?;
    charge_capacity(storage, limit, vector.capacity(), values, size_of::<f32>())?;
    let mut raw = [0u8; VECTOR_STAGE_BYTES];
    let raw = &mut raw[..values * 2];
    read(reader, raw)?;
    for word in raw.chunks_exact(2) {
        let value = f32::from_bits(u32::from(u16::from_le_bytes([word[0], word[1]])) << 16);
        if !value.is_finite() {
            return Err(Error::Nonfinite);
        }
        vector.push(value);
    }
    Ok(vector)
}

fn tile_bytes(rows: usize, columns: usize) -> Result<usize> {
    add(multiply(multiply(rows, columns)?, 2)?, GUARD * 4)
}

fn charge_capacity(
    storage: &mut usize,
    limit: usize,
    actual: usize,
    minimum: usize,
    item: usize,
) -> Result<()> {
    *storage = add(*storage, multiply(actual - minimum, item)?)?;
    check_budget(*storage, limit)
}

fn reserve<T>(count: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Native("Gemma BF16 storage reservation failed".into()))?;
    Ok(values)
}

fn read(reader: &mut impl Read, bytes: &mut [u8]) -> Result<()> {
    reader.read_exact(bytes).map_err(|source| {
        crate::weights::Error::Io {
            path: "Gemma BF16 verified tensor permit".into(),
            source,
        }
        .into()
    })
}

pub(super) fn check_budget(required: usize, limit: usize) -> Result<()> {
    if required > limit {
        return Err(Error::Budget { required, limit });
    }
    Ok(())
}

pub(super) fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or(Error::Options("Gemma BF16 storage byte overflow"))
}

fn multiply(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or(Error::Options("Gemma BF16 storage byte overflow"))
}

#[cfg(test)]
impl Bf16Weights {
    pub(crate) fn test_retained_digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut sha = Sha256::new();
        let mut raw = [0; 4096];
        for matrix in self.matrices.iter().flatten() {
            for tile in &matrix.tiles {
                tile.with_slice(|words| {
                    for block in words.chunks(raw.len() / 4) {
                        for (word, bytes) in block.iter().zip(raw.chunks_exact_mut(4)) {
                            bytes.copy_from_slice(&word.to_le_bytes());
                        }
                        sha.update(&raw[..block.len() * 4]);
                    }
                });
            }
        }
        for vector in &self.vectors {
            for value in vector {
                sha.update(value.to_bits().to_le_bytes());
            }
        }
        sha.finalize().into()
    }
}

#[cfg(test)]
mod tests;
