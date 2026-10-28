//! Cold logical tensor selection. No descriptors survive into numerical kernels.
use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Selection {
    Whole,
    Rows(Range<usize>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LogicalTensorSpec {
    physical_name: String,
    selection: Selection,
    tensor: TensorSpec,
}
impl LogicalTensorSpec {
    pub(crate) fn tensor(&self) -> &TensorSpec {
        &self.tensor
    }
}

/// Ordered, complete partition of one physical rank-two tensor. Names become
/// ordinary graph input names after loading; rows always address original data.
pub(super) struct RowPartition {
    pub physical_name: String,
    pub parts: Vec<(String, Range<usize>)>,
}

fn payload(spec: &TensorSpec) -> Result<usize, Error> {
    if !matches!(spec.shape.as_slice(), [_] | [_, _]) || spec.shape.contains(&0) {
        return Err(invalid(
            "logical source requires nonempty rank-one/two tensor",
        ));
    }
    let count = spec
        .shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| invalid("logical tensor extent overflow"))?;
    let expected = match spec.storage {
        Storage::Float(dtype) => count
            .checked_mul(dtype.bytes())
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| invalid("logical tensor byte overflow"))?,
        Storage::Affine(layout) => {
            if spec.shape != [layout.rows(), layout.columns()] {
                return Err(invalid("logical affine shape differs from layout"));
            }
            layout.total_bytes()
        }
    };
    if expected != spec.payload_bytes {
        return Err(invalid("logical source payload differs from layout"));
    }
    Ok(expected)
}

/// Retains each Whole owner or replaces it by disjoint original row intervals.
/// No bytes are copied or expanded. Every interval and byte sum is checked once
/// before publishing the plan. Physical source names/overrides remain unchanged.
pub(super) fn plan(
    physical: &BTreeMap<String, TensorSpec>,
    partitions: &[RowPartition],
) -> Result<BTreeMap<String, LogicalTensorSpec>, Error> {
    let mut selected = BTreeMap::new();
    for partition in partitions {
        if !physical.contains_key(&partition.physical_name)
            || selected
                .insert(partition.physical_name.as_str(), partition)
                .is_some()
        {
            return Err(invalid("missing or repeated partition parent"));
        }
    }
    let mut count = physical.len();
    for partition in partitions {
        if partition.parts.is_empty() {
            return Err(invalid("empty logical partition"));
        }
        count = count
            .checked_add(partition.parts.len() - 1)
            .ok_or_else(|| invalid("logical tensor count overflow"))?;
    }
    if count > 8192 {
        return Err(invalid("logical tensor count exceeds source bound"));
    }
    let mut result = BTreeMap::new();
    let mut physical_bytes = 0;
    let mut logical_bytes = 0;
    let mut insert = |name: &str, physical_name: &str, selection, tensor: TensorSpec| {
        if name.is_empty() || result.contains_key(name) {
            return Err(invalid("empty or repeated logical tensor name"));
        }
        logical_bytes = add(logical_bytes, payload(&tensor)?)?;
        result.insert(
            name.to_owned(),
            LogicalTensorSpec {
                physical_name: physical_name.to_owned(),
                selection,
                tensor,
            },
        );
        Ok(())
    };
    for (name, spec) in physical {
        physical_bytes = add(physical_bytes, payload(spec)?)?;
        if let Some(partition) = selected.get(name.as_str()) {
            let [rows, columns] = spec.shape.as_slice() else {
                return Err(invalid("only rank-two tensors can be row-partitioned"));
            };
            let mut cursor = 0;
            for (logical_name, range) in &partition.parts {
                if range.start != cursor || range.start >= range.end || range.end > *rows {
                    return Err(invalid(
                        "logical rows must partition the complete parent in order",
                    ));
                }
                let row_count = range.end - range.start;
                let (storage, payload_bytes) = match spec.storage {
                    Storage::Affine(layout) => {
                        let selected = layout
                            .row_span(range.clone())
                            .map_err(|e| invalid(e.to_string()))?
                            .layout();
                        (Storage::Affine(selected), selected.total_bytes())
                    }
                    Storage::Float(dtype) => {
                        let bytes = row_count
                            .checked_mul(*columns)
                            .and_then(|n| n.checked_mul(dtype.bytes()))
                            .ok_or_else(|| invalid("logical floating row byte overflow"))?;
                        (Storage::Float(dtype), bytes)
                    }
                };
                insert(
                    logical_name,
                    name,
                    Selection::Rows(range.clone()),
                    TensorSpec {
                        shape: vec![row_count, *columns],
                        storage,
                        payload_bytes,
                    },
                )?;
                cursor = range.end;
            }
            if cursor != *rows {
                return Err(invalid("logical rows do not cover the complete parent"));
            }
        } else {
            insert(name, name, Selection::Whole, spec.clone())?;
        }
    }
    if logical_bytes != physical_bytes || result.len() != count {
        return Err(invalid(
            "logical partition does not preserve unique source payload",
        ));
    }
    Ok(result)
}

impl Source {
    pub(crate) fn logical_tensors(&self) -> &BTreeMap<String, LogicalTensorSpec> {
        &self.logical
    }
    pub(crate) fn visit_logical_float(
        &mut self,
        name: &str,
        consume: impl FnMut(f32) -> Result<(), Error>,
    ) -> Result<(), Error> {
        // Clone only cold name/range metadata to release the source-map borrow.
        // Whole still enters the original audited read, even for rank-two data.
        let result = (|| {
            let logical = self
                .logical
                .get(name)
                .ok_or_else(|| invalid("missing logical floating tensor"))?;
            if !matches!(logical.tensor.storage, Storage::Float(_)) {
                return Err(invalid("logical tensor is not floating"));
            }
            let physical_name = logical.physical_name.clone();
            match logical.selection.clone() {
                Selection::Whole => self.visit_float(&physical_name, consume),
                Selection::Rows(rows) => self.visit_float_rows(&physical_name, rows, consume),
            }
        })();
        if result.is_err() {
            self.source.poison_audit();
        }
        result
    }
    pub(crate) fn read_logical_affine(
        &mut self,
        name: &str,
        weights: &mut [u8],
        scales: &mut [u8],
        biases: &mut [u8],
    ) -> Result<Layout, Error> {
        let result = (|| {
            let logical = self
                .logical
                .get(name)
                .ok_or_else(|| invalid("missing logical affine tensor"))?;
            if !matches!(logical.tensor.storage, Storage::Affine(_)) {
                return Err(invalid("logical tensor is not affine"));
            }
            let physical_name = logical.physical_name.clone();
            match logical.selection.clone() {
                Selection::Whole => self.read_affine(&physical_name, weights, scales, biases),
                Selection::Rows(rows) => {
                    self.read_affine_rows(&physical_name, rows, weights, scales, biases)
                }
            }
        })();
        if result.is_err() {
            self.source.poison_audit();
        }
        result
    }
}
