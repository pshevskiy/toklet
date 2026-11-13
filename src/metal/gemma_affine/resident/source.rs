//! Cold two-pass byte authentication. No model or identity escapes this module.
use super::*;
use crate::mlx::gemma4::{E2bPlan, Storage, inspect, inspect_qat};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
#[cfg(target_os = "macos")]
pub(super) mod heretic;
#[cfg(test)]
pub(super) mod test_support;
#[cfg(test)]
mod tests;

const MAX_HEADER: usize = 1024 * 1024;
const STREAM: usize = 64 * 1024;

pub(super) struct Region {
    pub(super) offset: u64,
    pub(super) bytes: usize,
    text: bool,
    digest: [u8; 32],
    consumed: bool,
}
pub(super) struct MatrixInput {
    pub(super) layout: Layout,
    pub(super) regions: [usize; 3],
}
pub(super) struct VectorInput {
    pub(super) values: usize,
    pub(super) region: usize,
}
pub(super) struct Inputs {
    prefix: [u8; 8],
    header: Vec<u8>,
    pub(super) regions: Vec<Region>,
    pub(super) matrices: Vec<MatrixInput>,
    pub(super) vectors: Vec<VectorInput>,
    config_sha256: [u8; 32],
    index_sha256: [u8; 32],
    authenticated: bool,
}
pub(super) struct Prepared {
    pub(super) cold: E2bPlan,
    pub(super) inputs: Inputs,
}
fn input_io<T>(result: std::io::Result<T>) -> Result<T> {
    result.map_err(|source| {
        crate::weights::Error::Io {
            path: "Gemma source stream".into(),
            source,
        }
        .into()
    })
}
fn seek(reader: &mut impl Seek, offset: u64) -> Result<()> {
    if input_io(reader.seek(SeekFrom::Start(offset)))? != offset {
        return Err(Error::Options("Gemma source seek differs"));
    }
    Ok(())
}
fn exact(reader: &mut impl Read, destination: &mut [u8]) -> Result<()> {
    input_io(reader.read_exact(destination))
}
fn reserve<T>(count: usize) -> Result<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| Error::Native("Gemma source metadata reservation failed".into()))?;
    Ok(result)
}
impl Prepared {
    pub(super) fn read(
        config: &[u8],
        index: &[u8],
        reader: &mut (impl Read + Seek),
        options: &LoadOptions,
        kind: SourceKind,
    ) -> Result<Self> {
        if config.len() > MAX_HEADER || index.len() > MAX_HEADER {
            return Err(Error::Options("Gemma source metadata bound"));
        }
        seek(reader, 0)?;
        let mut prefix = [0; 8];
        exact(reader, &mut prefix)?;
        let count = usize::try_from(u64::from_le_bytes(prefix))
            .map_err(|_| Error::Options("Gemma source header overflow"))?;
        if !(1..=MAX_HEADER).contains(&count) {
            return Err(Error::Options("Gemma source header bound"));
        }
        let mut header = reserve(count)?;
        if header.capacity() > MAX_HEADER {
            return Err(Error::Options("Gemma source header capacity bound"));
        }
        header.resize(count, 0);
        exact(reader, &mut header)?;
        let cold = match kind {
            SourceKind::Affine4 => inspect(
                config,
                index,
                &header,
                options.container_bytes,
                options.modality,
            )?,
            SourceKind::QatMixed => inspect_qat(config, index, &header, options.modality)?,
        };
        let mut physical = Vec::new();
        physical
            .try_reserve_exact(cold.text_tensors().len() + cold.excluded_tensors().len())
            .map_err(|_| Error::Native("Gemma source inventory reservation failed".into()))?;
        physical.extend(cold.text_tensors().iter().map(|(n, t)| (n, t, true)));
        physical.extend(cold.excluded_tensors().iter().map(|(n, t)| (n, t, false)));
        physical.sort_unstable_by_key(|(_, tensor, _)| tensor.range.start);
        let mut regions = reserve(physical.len())?;
        let mut names = BTreeMap::new();
        let mut cursor = 0;
        let start = 8_u64 + header.len() as u64;
        for (name, tensor, text) in physical {
            if name.len() > 512 || tensor.range.start != cursor {
                return Err(Error::Options("Gemma source name/coverage bound"));
            }
            names.insert(name.as_str(), regions.len());
            regions.push(Region {
                offset: start
                    .checked_add(tensor.range.start as u64)
                    .ok_or(Error::Options(OVERFLOW))?,
                bytes: tensor.range.end - tensor.range.start,
                text,
                digest: [0; 32],
                consumed: false,
            });
            cursor = tensor.range.end;
        }
        if start.checked_add(cursor as u64) != Some(options.container_bytes) {
            return Err(Error::Options("Gemma source complete coverage differs"));
        }
        let lookup = |name: &str| {
            names
                .get(name)
                .copied()
                .ok_or(Error::Options("Gemma source component missing"))
        };
        let mut matrices = reserve(278)?;
        let mut vectors = reserve(262)?;
        for tensor in cold.logical_tensors() {
            match tensor.storage {
                Storage::F32Vector => {
                    return Err(Error::Options("Gemma BF16-only vector recipe required"));
                }
                Storage::Affine(layout) => {
                    let base = tensor
                        .name
                        .strip_suffix(".weight")
                        .ok_or(Error::Options("Gemma source matrix stem"))?;
                    matrices.push(MatrixInput {
                        layout,
                        regions: [
                            lookup(&tensor.name)?,
                            lookup(&format!("{base}.scales"))?,
                            lookup(&format!("{base}.biases"))?,
                        ],
                    });
                }
                Storage::Bf16Vector => {
                    let crate::mlx::gemma4::Shape::Vector(values) = tensor.shape else {
                        return Err(Error::Options("Gemma source vector shape"));
                    };
                    vectors.push(VectorInput {
                        values,
                        region: lookup(&tensor.name)?,
                    });
                }
            }
        }
        if matrices.len() != 278 || vectors.len() != 262 {
            return Err(Error::Options("Gemma source logical inventory differs"));
        }
        drop(names);
        let inputs = Inputs {
            prefix,
            header,
            regions,
            matrices,
            vectors,
            config_sha256: Sha256::digest(config).into(),
            index_sha256: Sha256::digest(index).into(),
            authenticated: false,
        };
        Ok(Self { cold, inputs })
    }
}
impl Inputs {
    /// Starts at the position immediately after the SAME captured header. No
    /// seek/reopen can substitute other bytes for the parsed prefix in this hash.
    pub(super) fn authenticate(
        &mut self,
        reader: &mut impl Read,
        options: &LoadOptions,
    ) -> Result<()> {
        if self.authenticated || self.regions.iter().any(|r| r.consumed) {
            return Err(Error::Options("Gemma source authentication repeated"));
        }
        let mut whole = Sha256::new();
        whole.update(self.prefix);
        whole.update(&self.header);
        let mut stream = [0; STREAM];
        let mut count = 8_u64 + self.header.len() as u64;
        for region in &mut self.regions {
            if count != region.offset {
                return Err(Error::Options("Gemma source stream offset differs"));
            }
            let mut digest = Sha256::new();
            let mut remaining = region.bytes;
            while remaining != 0 {
                let active = remaining.min(stream.len());
                exact(reader, &mut stream[..active])?;
                whole.update(&stream[..active]);
                digest.update(&stream[..active]);
                remaining -= active;
                count = count
                    .checked_add(active as u64)
                    .ok_or(Error::Options(OVERFLOW))?;
            }
            if region.text {
                region.digest = digest.finalize().into();
            }
        }
        let eof = loop {
            match reader.read(&mut [0_u8; 1]) {
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                result => break input_io(result)?,
            }
        };
        let digest: [u8; 32] = whole.finalize().into();
        if count != options.container_bytes || eof != 0 || digest != options.container_sha256 {
            return Err(Error::Options("Gemma source container identity differs"));
        }
        self.authenticated = true;
        Ok(())
    }
    pub(super) fn check_metadata_capacities(&self) -> Result<()> {
        let bytes = add(
            add(
                self.header.capacity(),
                multiply(self.regions.capacity(), size_of::<Region>())?,
            )?,
            add(
                multiply(self.matrices.capacity(), size_of::<MatrixInput>())?,
                multiply(self.vectors.capacity(), size_of::<VectorInput>())?,
            )?,
        )?;
        // Keep explicit numerical descriptors inside the larger cold policy;
        // the remainder covers the bounded parsed plan/maps and proposal frames.
        if bytes > COLD_POLICY / 4 {
            return Err(Error::Options("Gemma source descriptor capacity bound"));
        }
        Ok(())
    }
    pub(super) fn matrix_read(&self, index: usize) -> Result<MatrixRead> {
        if !self.authenticated {
            return Err(Error::Options("Gemma source is not authenticated"));
        }
        let input = &self.matrices[index];
        let mut expected = [(0_u64, 0_usize, [0_u8; 32]); 3];
        for (entry, &index) in expected.iter_mut().zip(&input.regions) {
            let region = &self.regions[index];
            if !region.text || region.consumed {
                return Err(Error::Options("Gemma source matrix coverage repeated"));
            }
            *entry = (region.offset, region.bytes, region.digest);
        }
        Ok(MatrixRead {
            layout: input.layout,
            regions: input.regions,
            expected,
            covered: [0; 3],
            digests: std::array::from_fn(|_| Sha256::new()),
        })
    }
    pub(super) fn vector_read(
        &mut self,
        reader: &mut (impl Read + Seek),
        index: usize,
        raw: &mut [u8],
    ) -> Result<()> {
        self.component_read(reader, self.vectors[index].region, raw)
    }
    // Shared raw BF16/vector component proof; interpretation follows only after
    // the same selected bytes match their first-pass authenticated digest.
    fn component_read(
        &mut self,
        reader: &mut (impl Read + Seek),
        index: usize,
        raw: &mut [u8],
    ) -> Result<()> {
        let region = &mut self.regions[index];
        if !self.authenticated || !region.text || region.consumed || raw.len() != region.bytes {
            return Err(Error::Options("Gemma source vector coverage differs"));
        }
        seek(reader, region.offset)?;
        exact(reader, raw)?;
        let digest: [u8; 32] = Sha256::digest(&*raw).into();
        if digest != region.digest {
            return Err(Error::Options("Gemma source vector changed between passes"));
        }
        region.consumed = true;
        Ok(())
    }
    pub(super) fn finish(&self, options: &LoadOptions) -> Result<Identity> {
        if !self.authenticated || self.regions.iter().any(|r| r.text && !r.consumed) {
            return Err(Error::Options("Gemma source consumption incomplete"));
        }
        Ok(Identity {
            config_sha256: self.config_sha256,
            index_sha256: self.index_sha256,
            header_sha256: Sha256::digest(&self.header).into(),
            container_sha256: options.container_sha256,
            container_bytes: options.container_bytes,
            text_components: self.regions.iter().filter(|r| r.text).count(),
        })
    }
}
pub(super) struct MatrixRead {
    layout: Layout,
    regions: [usize; 3],
    expected: [(u64, usize, [u8; 32]); 3],
    covered: [usize; 3],
    digests: [Sha256; 3],
}
impl MatrixRead {
    pub(super) fn fill(
        &mut self,
        reader: &mut (impl Read + Seek),
        first: usize,
        layout: Layout,
        buffers: [&mut [u8]; 3],
    ) -> Result<()> {
        if layout.columns() != self.layout.columns()
            || layout.bits() != self.layout.bits()
            || layout.group_size() != self.layout.group_size()
            || layout.metadata_dtype() != self.layout.metadata_dtype()
            || first
                .checked_add(layout.rows())
                .is_none_or(|end| end > self.layout.rows())
        {
            return Err(Error::Options("Gemma source chunk layout differs"));
        }
        let strides = [
            self.layout.weight_bytes() / self.layout.rows(),
            self.layout.metadata_bytes() / self.layout.rows(),
            self.layout.metadata_bytes() / self.layout.rows(),
        ];
        for (role, raw) in buffers.into_iter().enumerate() {
            let expected_first = first
                .checked_mul(strides[role])
                .ok_or(Error::Options(OVERFLOW))?;
            let expected_bytes = layout
                .rows()
                .checked_mul(strides[role])
                .ok_or(Error::Options(OVERFLOW))?;
            let end = self.covered[role]
                .checked_add(raw.len())
                .ok_or(Error::Options(OVERFLOW))?;
            if expected_first != self.covered[role]
                || raw.len() != expected_bytes
                || end > self.expected[role].1
            {
                return Err(Error::Options("Gemma source chunk coverage differs"));
            }
            let offset = self.expected[role]
                .0
                .checked_add(expected_first as u64)
                .ok_or(Error::Options(OVERFLOW))?;
            seek(reader, offset)?;
            exact(reader, raw)?;
            self.digests[role].update(&*raw);
            self.covered[role] = end;
        }
        Ok(())
    }
    pub(super) fn finish(self, inputs: &mut Inputs) -> Result<()> {
        for (role, digest) in self.digests.into_iter().enumerate() {
            let actual: [u8; 32] = digest.finalize().into();
            if self.covered[role] != self.expected[role].1
                || actual != self.expected[role].2
                || inputs.regions[self.regions[role]].consumed
            {
                return Err(Error::Options(
                    "Gemma source matrix changed/incomplete between passes",
                ));
            }
        }
        for index in self.regions {
            inputs.regions[index].consumed = true;
        }
        Ok(())
    }
}
