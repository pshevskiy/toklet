//! One checked physical traversal supplies unpublished resident tensors.
use super::*;
use crate::mlx::gemma4::PhysicalTensor;
use std::io::Read;

pub(crate) struct Prepared {
    file: File,
    before: [u64; 7],
    plan: TextPlan,
    table: Table,
    expected_sha256: &'static str,
}

impl Prepared {
    pub(crate) fn new(file: File, config: &[u8], available_bytes: usize) -> Result<Self, Error> {
        let (before, plan, table) = read_plan(&file, config, available_bytes)?;
        Ok(Self {
            file,
            before,
            plan,
            table,
            expected_sha256: CONTAINER_SHA256,
        })
    }

    #[cfg(test)]
    pub(crate) fn modified_parent(
        file: File,
        config: &[u8],
        available: usize,
    ) -> Result<Self, Error> {
        let (before, plan, table) = read_plan_with(
            &file,
            config,
            available,
            super::super::modified_parent::inspect,
        )?;
        Ok(Self {
            file,
            before,
            plan,
            table,
            expected_sha256: super::super::modified_parent::CONTAINER_SHA256,
        })
    }

    pub(crate) fn plan(&self) -> &TextPlan {
        &self.plan
    }

    pub(crate) fn minimum_source_memory(&self) -> Result<Memory, Error> {
        memory(self.table.rows, SCAN_BYTES)
    }
    pub(crate) fn workspace_bytes(&self, rows: usize) -> Result<usize, Error> {
        Ok(Workspace::layout(self.table.row_bytes, rows)?.1)
    }

    /// The receiver must keep its buffers private until this returns the
    /// authenticated source. Every selected non-PLE tensor must be consumed
    /// exactly. The receiver cannot seek or read excluded bytes.
    pub(crate) fn deliver<E>(
        self,
        source_bytes: usize,
        receive: impl FnMut(
            usize,
            &str,
            &PhysicalTensor,
            Memory,
            &mut TensorReader<'_>,
        ) -> Result<(), E>,
    ) -> Result<Source, E>
    where
        E: From<Error>,
    {
        let expected = self.expected_sha256;
        self.deliver_pinned(source_bytes, expected, receive)
    }

    fn deliver_pinned<E>(
        self,
        source_bytes: usize,
        expected: &str,
        mut receive: impl FnMut(
            usize,
            &str,
            &PhysicalTensor,
            Memory,
            &mut TensorReader<'_>,
        ) -> Result<(), E>,
    ) -> Result<Source, E>
    where
        E: From<Error>,
    {
        let mut scan = Scan::new(self.table.rows, source_bytes)?;
        let mut spans = reserved(self.plan.text_tensors().len())?;
        let mut matrix_slot = 0;
        let mut vector_slot = 0;
        for (name, tensor) in self.plan.text_tensors() {
            let slot = match tensor.shape.len() {
                1 => &mut vector_slot,
                2 => &mut matrix_slot,
                _ => return Err(invalid("Gemma4 BF16 delivery rank differs").into()),
            };
            spans.push((*slot, name, tensor));
            *slot += 1;
        }
        spans.sort_unstable_by_key(|(_, _, t)| t.range.start);
        let mut cursor = 0;
        for (slot, name, tensor) in spans {
            let start = add_bytes(self.plan.payload_offset(), tensor.range.start)? as u64;
            let end = add_bytes(self.plan.payload_offset(), tensor.range.end)? as u64;
            if cursor > start || end > self.before[2] {
                return Err(invalid("Gemma4 BF16 delivery span differs").into());
            }
            hash_span(
                &self.file,
                &mut scan.scratch,
                cursor..start,
                &mut scan.digest,
            )?;
            if name == super::super::PLE_TABLE {
                if (start, end) != (self.table.start, self.table.end) {
                    return Err(invalid("Gemma4 BF16 PLE delivery span differs").into());
                }
                scan.ple(&self.file, self.table)?;
            } else {
                let mut tensor_read = TensorReader {
                    file: &self.file,
                    cursor: start,
                    end,
                    digest: &mut scan.digest,
                };
                receive(slot, name, tensor, scan.memory, &mut tensor_read)?;
                if tensor_read.cursor != end {
                    return Err(invalid("Gemma4 BF16 tensor was not consumed exactly").into());
                }
            }
            cursor = end;
        }
        hash_span(
            &self.file,
            &mut scan.scratch,
            cursor..self.before[2],
            &mut scan.digest,
        )?;
        Ok(scan.finish(self.file, self.before, self.table, expected)?)
    }
}

pub(crate) struct TensorReader<'a> {
    file: &'a File,
    digest: &'a mut Sha256,
    cursor: u64,
    end: u64,
}

impl Read for TensorReader<'_> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let len = (self.end - self.cursor).min(output.len() as u64) as usize;
        if len == 0 {
            return Ok(0);
        }
        let count = self.file.read_at(&mut output[..len], self.cursor)?;
        self.digest.update(&output[..count]);
        self.cursor += count as u64;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::{super::tests::TestFile, *};
    use crate::mlx::gemma4::{Dtype, layer_plans};

    #[test]
    fn modified_parent_preparation_keeps_a_distinct_payload_pin() {
        use super::super::super::modified_parent as parent;
        let location = TestFile::new();
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(location.path())
            .unwrap();
        file.set_len(super::super::super::CONTAINER_BYTES).unwrap();
        let header =
            include_bytes!("../../../../../tests/fixtures/gemma-unquantized-header-v1/header.json");
        file.write_all_at(&(header.len() as u64).to_le_bytes(), 0)
            .unwrap();
        file.write_all_at(header, 8).unwrap();
        let prepared =
            Prepared::modified_parent(file.try_clone().unwrap(), parent::CONFIG, usize::MAX)
                .unwrap();
        assert_eq!(prepared.expected_sha256, parent::CONTAINER_SHA256);
        assert_ne!(prepared.expected_sha256, CONTAINER_SHA256);
        assert!(Prepared::new(file.try_clone().unwrap(), parent::CONFIG, usize::MAX).is_err());
        let original =
            include_bytes!("../../../../../tests/fixtures/gemma-unquantized-header-v1/config.json");
        assert!(
            Prepared::modified_parent(file.try_clone().unwrap(), original, usize::MAX).is_err()
        );
        assert!(Prepared::modified_parent(file, parent::CONFIG, 0).is_err());
    }

    fn fixture() -> (TestFile, Prepared, Vec<u8>) {
        let location = TestFile::new();
        let base = super::super::super::HEADER_BYTES + 8;
        let mut raw = vec![0x2a; base + 32];
        raw[base + 32 - 1] = 0x3f;
        let ple = Table::new((base + 16) as u64, 2, 2, raw.len() as u64).unwrap();
        std::fs::write(location.path(), &raw).unwrap();
        let file = File::open(location.path()).unwrap();
        let before = file_identity(&file).unwrap();
        let plan = TextPlan {
            layers: layer_plans(),
            text: [
                (
                    "tensor".into(),
                    PhysicalTensor {
                        dtype: Dtype::Bf16,
                        shape: vec![2],
                        range: 4..8,
                    },
                ),
                (
                    super::super::super::PLE_TABLE.into(),
                    PhysicalTensor {
                        dtype: Dtype::Bf16,
                        shape: vec![2, 2],
                        range: 16..24,
                    },
                ),
            ]
            .into_iter()
            .collect(),
            media_bytes: 0,
            unused_shared_kv_bytes: 0,
            resident_tensor_bytes: 4,
        };
        (
            location,
            Prepared {
                file,
                before,
                plan,
                table: ple,
                expected_sha256: CONTAINER_SHA256,
            },
            raw,
        )
    }

    #[test]
    fn tensor_permit_is_bounded_and_requires_exact_consumption() {
        let (_location, prepared, raw) = fixture();
        let expected = format!("{:x}", Sha256::digest(&raw));
        let source = prepared
            .deliver_pinned::<Error>(usize::MAX, &expected, |slot, name, _, _, reader| {
                assert_eq!(slot, 0);
                assert_eq!(name, "tensor");
                let mut row = [0u8; 16];
                assert_eq!(reader.read(&mut row).unwrap(), 4);
                assert_eq!(&row[..4], &[0x2a; 4]);
                assert_eq!(reader.read(&mut row).unwrap(), 0);
                Ok(())
            })
            .unwrap();
        assert_eq!(source.row_digests.len(), 2);

        let (_location, prepared, _) = fixture();
        let short =
            prepared.deliver_pinned::<Error>(usize::MAX, &expected, |_, _, _, _, reader| {
                reader.read_exact(&mut [0u8; 3]).unwrap();
                Ok(())
            });
        assert!(
            matches!(short, Err(Error::Invalid(reason)) if reason.contains("consumed exactly"))
        );
    }

    #[test]
    fn final_excluded_mutation_cannot_authenticate_already_delivered_bytes() {
        let (location, prepared, raw) = fixture();
        let expected = format!("{:x}", Sha256::digest(&raw));
        let writer = std::fs::OpenOptions::new()
            .write(true)
            .open(location.path())
            .unwrap();
        let mut delivered = false;
        let rejected =
            prepared.deliver_pinned::<Error>(usize::MAX, &expected, |_, _, _, _, reader| {
                reader.read_exact(&mut [0u8; 4]).unwrap();
                delivered = true;
                writer.write_all_at(&[0x7a], raw.len() as u64 - 1).unwrap();
                Ok(())
            });
        assert!(delivered);
        assert!(
            matches!(rejected, Err(Error::Invalid(reason)) if reason.contains("authentication"))
        );
    }
}
