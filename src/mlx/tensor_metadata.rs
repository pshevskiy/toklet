//! Concrete cold tensor metadata shared by the pinned model inspectors.
//! No tensor payload, numerical dispatch or execution eligibility is introduced.
use super::affine::Layout;
use crate::weights::Error;
use serde_json::Value;
use std::{collections::BTreeMap, ops::Range};
fn invalid(reason: impl Into<String>) -> Error {
    Error::Invalid(reason.into())
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Vector(usize),
    Matrix { rows: usize, columns: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    Bf16Vector,
    /// Original FP32 small parameters for an explicitly selected source only.
    F32Vector,
    Affine(Layout),
}

#[derive(Debug)]
pub struct TensorPlan {
    pub name: String,
    pub shape: Shape,
    pub storage: Storage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    U32,
    Bf16,
    F32,
}
impl Dtype {
    fn bytes(self) -> usize {
        match self {
            Self::U32 | Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
}

#[derive(Debug)]
pub struct PhysicalTensor {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Relative to the single shard's payload, after its length/header bytes.
    pub range: Range<usize>,
}

pub(super) fn same_semantics(actual: &Value, expected: &Value) -> bool {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) => {
            if let Some(integer) = e.as_u64() {
                a.as_u64() == Some(integer)
            } else {
                // `arbitrary_precision` retains exponent spelling in Number.
                // Float-valued graph options compare their exact finite FP64
                // value, while dimensions/counts above remain JSON integers.
                match (a.as_f64(), e.as_f64()) {
                    (Some(a), Some(e)) => a.is_finite() && e.is_finite() && a == e,
                    _ => false,
                }
            }
        }
        (Value::Object(a), Value::Object(e)) => {
            a.len() == e.len()
                && e.iter()
                    .all(|(key, value)| a.get(key).is_some_and(|v| same_semantics(v, value)))
        }
        (Value::Array(a), Value::Array(e)) => {
            a.len() == e.len() && a.iter().zip(e).all(|(a, e)| same_semantics(a, e))
        }
        _ => actual == expected,
    }
}

// These family factories construct only validated Q4/group64/BF16 plans.
// This mapper is deliberately crate-private and is not a general affine schema.
pub(super) fn expected_physical(
    logical: &[TensorPlan],
) -> BTreeMap<String, (Dtype, Vec<usize>, usize)> {
    expected_physical_impl(logical, false)
}

/// Only the distinct2B factory admits original FP32 small vectors. Existing
/// source factories retain their U32/BF16 mapper and descriptor whitelist.
pub(super) fn expected_physical_with_f32(
    logical: &[TensorPlan],
) -> BTreeMap<String, (Dtype, Vec<usize>, usize)> {
    expected_physical_impl(logical, true)
}

fn expected_physical_impl(
    logical: &[TensorPlan],
    f32_vectors: bool,
) -> BTreeMap<String, (Dtype, Vec<usize>, usize)> {
    let mut result = BTreeMap::new();
    for tensor in logical {
        match (tensor.shape, tensor.storage) {
            (Shape::Vector(size), Storage::Bf16Vector) => {
                result.insert(tensor.name.clone(), (Dtype::Bf16, vec![size], size * 2));
            }
            (Shape::Vector(size), Storage::F32Vector) if f32_vectors => {
                result.insert(tensor.name.clone(), (Dtype::F32, vec![size], size * 4));
            }
            (Shape::Matrix { rows, columns }, Storage::Affine(layout)) => {
                result.insert(
                    tensor.name.clone(),
                    (
                        Dtype::U32,
                        vec![rows, layout.row_bytes() / size_of::<u32>()],
                        layout.weight_bytes(),
                    ),
                );
                let base = tensor.name.strip_suffix(".weight").unwrap();
                for name in ["scales", "biases"] {
                    result.insert(
                        format!("{base}.{name}"),
                        (
                            Dtype::Bf16,
                            vec![rows, columns / layout.group_size()],
                            layout.metadata_bytes(),
                        ),
                    );
                }
            }
            _ => unreachable!("constructed logical tensor storage"),
        }
    }
    result
}

fn integer(value: &Value) -> Result<usize, Error> {
    value
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| invalid("bounded unsigned tensor metadata integer required"))
}
pub(super) fn descriptor(value: &Value, payload: usize) -> Result<PhysicalTensor, Error> {
    descriptor_impl(value, payload, false)
}

/// Explicit small-FP32 opt-in, used only after the selected2B recipe fixes the
/// exact names/shapes. This does not widen any existing source descriptor.
pub(super) fn descriptor_with_f32(value: &Value, payload: usize) -> Result<PhysicalTensor, Error> {
    descriptor_impl(value, payload, true)
}

fn descriptor_impl(
    value: &Value,
    payload: usize,
    f32_vectors: bool,
) -> Result<PhysicalTensor, Error> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("tensor descriptor object required"))?;
    if object.len() != 3
        || !["dtype", "shape", "data_offsets"]
            .iter()
            .all(|key| object.contains_key(*key))
    {
        return Err(invalid("exact tensor descriptor fields required"));
    }
    let dtype = match value["dtype"].as_str() {
        Some("U32") => Dtype::U32,
        Some("BF16") => Dtype::Bf16,
        Some("F32") if f32_vectors => Dtype::F32,
        _ => return Err(invalid("unsupported tensor metadata dtype")),
    };
    let shape = value["shape"]
        .as_array()
        .ok_or_else(|| invalid("tensor shape array required"))?;
    if shape.len() > 8 {
        return Err(invalid("tensor rank exceeds bounded header"));
    }
    let shape: Vec<usize> = shape.iter().map(integer).collect::<Result<_, _>>()?;
    if dtype == Dtype::F32 && shape.len() != 1 {
        return Err(invalid("only explicit FP32 vectors are supported"));
    }
    if shape.iter().any(|&n| n >= 1 << 31) {
        return Err(invalid("tensor dimension exceeds bounded header"));
    }
    let elements = shape.iter().try_fold(1_usize, |n, &d| {
        n.checked_mul(d)
            .ok_or_else(|| invalid("tensor element count overflow"))
    })?;
    let bytes = elements
        .checked_mul(dtype.bytes())
        .ok_or_else(|| invalid("tensor byte count overflow"))?;
    let offsets = value["data_offsets"]
        .as_array()
        .filter(|v| v.len() == 2)
        .ok_or_else(|| invalid("tensor offset pair required"))?;
    let start = integer(&offsets[0])?;
    let end = integer(&offsets[1])?;
    if end > payload || start > end || end - start != bytes {
        return Err(invalid("tensor payload span/shape differs"));
    }
    Ok(PhysicalTensor {
        dtype,
        shape,
        range: start..end,
    })
}
