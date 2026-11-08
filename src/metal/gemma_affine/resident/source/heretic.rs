//! The pinned mixed source uses the same two-pass consumption proofs as E2B.
//! A separate captured BF16 component is not misrepresented as an affine matrix.
use super::*;
use crate::mlx::gemma4::uncensored::{self, MatrixStorage};

pub(in crate::metal::gemma_affine::resident) struct Prepared {
    pub(in crate::metal::gemma_affine::resident) cold: uncensored::Plan,
    pub(in crate::metal::gemma_affine::resident) inputs: Inputs,
    bf16_region: usize,
}

impl Prepared {
    pub(in crate::metal::gemma_affine::resident) fn read(
        config: &[u8],
        reader: &mut (impl Read + Seek),
        options: &LoadOptions,
    ) -> Result<Self> {
        if config.len() > MAX_HEADER {
            return Err(Error::Options("Heretic source config bound"));
        }
        seek(reader, 0)?;
        let mut prefix = [0; 8];
        exact(reader, &mut prefix)?;
        let bytes = usize::try_from(u64::from_le_bytes(prefix))
            .map_err(|_| Error::Options("Heretic header overflow"))?;
        if !(1..=MAX_HEADER).contains(&bytes) {
            return Err(Error::Options("Heretic source header bound"));
        }
        let mut header = reserve(bytes)?;
        if header.capacity() > MAX_HEADER {
            return Err(Error::Options("Heretic header capacity bound"));
        }
        header.resize(bytes, 0);
        exact(reader, &mut header)?;
        let cold = uncensored::inspect(config, &header, options.container_bytes, options.modality)?;
        let mut physical = reserve(2649)?;
        physical.extend(cold.text_tensors().iter().map(|(n, t)| (n, t, true)));
        physical.extend(
            cold.unused_shared_kv_tensors()
                .iter()
                .map(|(n, t)| (n, t, false)),
        );
        physical.extend(cold.media_tensors().iter().map(|(n, t)| (n, t, false)));
        physical.sort_unstable_by_key(|(_, t, _)| t.range.start);
        let mut regions = reserve(physical.len())?;
        let mut names = BTreeMap::new();
        let mut cursor = 0;
        let start = 8_u64 + header.len() as u64;
        for (name, tensor, text) in physical {
            if tensor.range.start != cursor {
                return Err(Error::Options("Heretic source span differs"));
            }
            names.insert(name.as_str(), regions.len());
            regions.push(Region {
                offset: start
                    .checked_add(tensor.range.start as u64)
                    .ok_or(Error::Options(OVERFLOW))?,
                bytes: tensor.range.len(),
                text,
                digest: [0; 32],
                consumed: false,
            });
            cursor = tensor.range.end;
        }
        if start.checked_add(cursor as u64) != Some(options.container_bytes) {
            return Err(Error::Options("Heretic source coverage differs"));
        }
        let lookup = |name: &str| {
            names
                .get(name)
                .copied()
                .ok_or(Error::Options("Heretic source component missing"))
        };
        let mut matrices = reserve(277)?;
        let mut bf16_region = None;
        for matrix in cold.matrices() {
            match matrix.storage {
                MatrixStorage::Affine(layout) => {
                    let base = matrix
                        .name
                        .strip_suffix(".weight")
                        .ok_or(Error::Options("Heretic matrix stem"))?;
                    matrices.push(MatrixInput {
                        layout,
                        regions: [
                            lookup(&matrix.name)?,
                            lookup(&format!("{base}.scales"))?,
                            lookup(&format!("{base}.biases"))?,
                        ],
                    });
                }
                MatrixStorage::Bf16 { rows, columns } => {
                    if bf16_region.is_some() || (rows, columns) != (8960, 1536) {
                        return Err(Error::Options("Heretic BF16 role differs"));
                    }
                    bf16_region = Some(lookup(&matrix.name)?);
                }
            }
        }
        let mut vectors = reserve(262)?;
        for (name, tensor) in cold
            .text_tensors()
            .iter()
            .filter(|(_, t)| t.shape.len() == 1)
        {
            vectors.push(VectorInput {
                values: tensor.shape[0],
                region: lookup(name)?,
            });
        }
        if matrices.len() != 277 || vectors.len() != 262 {
            return Err(Error::Options("Heretic source logical count differs"));
        }
        let bf16_region = bf16_region.ok_or(Error::Options("Heretic BF16 role missing"))?;
        drop(names);
        let inputs = Inputs {
            prefix,
            header,
            regions,
            matrices,
            vectors,
            config_sha256: Sha256::digest(config).into(),
            // No index is consumed: this pinned recipe names one fixed file.
            index_sha256: Sha256::digest([]).into(),
            authenticated: false,
        };
        inputs.check_metadata_capacities()?;
        Ok(Self {
            cold,
            inputs,
            bf16_region,
        })
    }
    pub(in crate::metal::gemma_affine::resident) fn read_bf16(
        &mut self,
        reader: &mut (impl Read + Seek),
        raw: &mut [u8],
    ) -> Result<()> {
        self.inputs.component_read(reader, self.bf16_region, raw)
    }
}

#[cfg(test)]
mod tests;
