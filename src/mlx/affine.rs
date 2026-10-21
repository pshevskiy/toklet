//! Borrowed rank-two MLX affine matrices: separate U32 codes/scales/biases.
//! See `docs/mlx-affine.md` for storage and reconstruction precision contracts.

use half::{bf16, f16};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error(pub &'static str);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for Error {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataDtype {
    F16,
    BF16,
    F32,
}
impl MetadataDtype {
    pub fn bytes(self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::F32 => 4,
        }
    }
}

/// Validated, contiguous row-major tensor layout. MLX has no padded row tails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    rows: usize,
    columns: usize,
    bits: u8,
    group_size: usize,
    dtype: MetadataDtype,
    row_bytes: usize,
    weight_bytes: usize,
    metadata_bytes: usize,
    total_bytes: usize,
}
impl Layout {
    pub fn new(
        rows: usize,
        columns: usize,
        bits: u8,
        group_size: usize,
        dtype: MetadataDtype,
    ) -> Result<Self, Error> {
        if rows == 0 || columns == 0 {
            return Err(Error("MLX affine dimensions must be positive"));
        }
        if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) {
            return Err(Error("unsupported MLX affine bit width"));
        }
        if !matches!(group_size, 32 | 64 | 128) || !columns.is_multiple_of(group_size) {
            return Err(Error(
                "invalid MLX affine group size or column divisibility",
            ));
        }
        let overflow = Error("MLX affine size overflow");
        let elements = rows.checked_mul(columns).ok_or(overflow)?;
        let row_bytes = columns.checked_mul(usize::from(bits)).ok_or(overflow)? / 8;
        let weight_bytes = rows.checked_mul(row_bytes).ok_or(overflow)?;
        let metadata_bytes = (elements / group_size)
            .checked_mul(dtype.bytes())
            .ok_or(overflow)?;
        let total_bytes = metadata_bytes
            .checked_mul(2)
            .and_then(|v| v.checked_add(weight_bytes))
            .ok_or(overflow)?;
        if elements > isize::MAX as usize || total_bytes > isize::MAX as usize {
            return Err(overflow);
        }
        Ok(Self {
            rows,
            columns,
            bits,
            group_size,
            dtype,
            row_bytes,
            weight_bytes,
            metadata_bytes,
            total_bytes,
        })
    }
    pub fn rows(self) -> usize {
        self.rows
    }
    pub fn columns(self) -> usize {
        self.columns
    }
    pub fn bits(self) -> u8 {
        self.bits
    }
    pub fn group_size(self) -> usize {
        self.group_size
    }
    pub fn metadata_dtype(self) -> MetadataDtype {
        self.dtype
    }
    pub fn row_bytes(self) -> usize {
        self.row_bytes
    }
    pub fn weight_bytes(self) -> usize {
        self.weight_bytes
    }
    pub fn metadata_bytes(self) -> usize {
        self.metadata_bytes
    }
    /// Shared borrowed storage counted once; no expanded model allocation.
    pub fn total_bytes(self) -> usize {
        self.total_bytes
    }
}

// The metadata choice is made at an operation boundary, never per element.
trait Metadata {
    const BYTES: usize;
    fn read(bytes: &[u8]) -> f32;
}
struct Half;
impl Metadata for Half {
    const BYTES: usize = 2;
    fn read(bytes: &[u8]) -> f32 {
        f16::from_le_bytes(bytes[..2].try_into().unwrap()).to_f32()
    }
}
struct BFloat;
impl Metadata for BFloat {
    const BYTES: usize = 2;
    fn read(bytes: &[u8]) -> f32 {
        bf16::from_le_bytes(bytes[..2].try_into().unwrap()).to_f32()
    }
}
struct Float;
impl Metadata for Float {
    const BYTES: usize = 4;
    fn read(bytes: &[u8]) -> f32 {
        f32::from_le_bytes(bytes[..4].try_into().unwrap())
    }
}
fn finite<M: Metadata>(bytes: &[u8]) -> bool {
    bytes.chunks_exact(M::BYTES).all(|v| M::read(v).is_finite())
}

/// Immutable byte-backed tensors need no native-pointer alignment guarantee.
/// Scales may be negative or zero. Group-wise `biases` are not a linear `bias`.
pub struct Matrix<'a> {
    layout: Layout,
    weights: &'a [u8],
    scales: &'a [u8],
    biases: &'a [u8],
}
impl<'a> Matrix<'a> {
    pub fn from_le_bytes(
        layout: Layout,
        weights: &'a [u8],
        scales: &'a [u8],
        biases: &'a [u8],
    ) -> Result<Self, Error> {
        if weights.len() != layout.weight_bytes
            || scales.len() != layout.metadata_bytes
            || biases.len() != layout.metadata_bytes
        {
            return Err(Error("MLX affine tensor byte lengths do not match layout"));
        }
        let valid = match layout.dtype {
            MetadataDtype::F16 => finite::<Half>(scales) && finite::<Half>(biases),
            MetadataDtype::BF16 => finite::<BFloat>(scales) && finite::<BFloat>(biases),
            MetadataDtype::F32 => finite::<Float>(scales) && finite::<Float>(biases),
        };
        if !valid {
            return Err(Error("MLX affine metadata must be finite"));
        }
        Ok(Self {
            layout,
            weights,
            scales,
            biases,
        })
    }
    pub fn layout(&self) -> Layout {
        self.layout
    }

    fn code<const B: u8>(&self, row: usize, column: usize) -> u8 {
        let bit = column * usize::from(B);
        let index = row * self.layout.row_bytes + bit / 8;
        let shift = bit % 8;
        let mut word = u16::from(self.weights[index]);
        if shift + usize::from(B) > 8 {
            word |= u16::from(self.weights[index + 1]) << 8;
        }
        ((word >> shift) & ((1u16 << B) - 1)) as u8
    }
    fn at<M: Metadata, const B: u8>(&self, row: usize, column: usize) -> f32 {
        let group = (row * self.layout.columns + column) / self.layout.group_size;
        let offset = group * M::BYTES;
        reconstruct(
            M::read(&self.scales[offset..]),
            M::read(&self.biases[offset..]),
            self.code::<B>(row, column),
        )
    }
    /// Scalar reference: metadata -> FP32, FP64 affine multiply/add, FP32 result.
    /// This does not emulate MLX reduced-precision or fused-matmul rounding.
    pub fn value(&self, row: usize, column: usize) -> Result<f32, Error> {
        if row >= self.layout.rows || column >= self.layout.columns {
            return Err(Error("MLX affine index out of bounds"));
        }
        let result = match self.layout.dtype {
            MetadataDtype::F16 => self.value_bits::<Half>(row, column),
            MetadataDtype::BF16 => self.value_bits::<BFloat>(row, column),
            MetadataDtype::F32 => self.value_bits::<Float>(row, column),
        };
        if !result.is_finite() {
            return Err(Error("MLX affine reconstruction overflow"));
        }
        Ok(result)
    }
    fn value_bits<M: Metadata>(&self, row: usize, column: usize) -> f32 {
        match self.layout.bits {
            2 => self.at::<M, 2>(row, column),
            3 => self.at::<M, 3>(row, column),
            4 => self.at::<M, 4>(row, column),
            5 => self.at::<M, 5>(row, column),
            6 => self.at::<M, 6>(row, column),
            8 => self.at::<M, 8>(row, column),
            _ => unreachable!("validated MLX bit width"),
        }
    }
    /// Reference only: FP64 products/feature-order accumulation, FP32 output.
    /// Caller output is untouched on shape, nonfinite-input or overflow failure.
    pub fn matvec(&self, input: &[f32], output: &mut [f32]) -> Result<(), Error> {
        if input.len() != self.layout.columns || output.len() != self.layout.rows {
            return Err(Error("MLX affine matvec shape mismatch"));
        }
        if input.iter().any(|x| !x.is_finite()) {
            return Err(Error("MLX affine input must be finite"));
        }
        match self.layout.dtype {
            MetadataDtype::F16 => self.matvec_bits::<Half>(input, output),
            MetadataDtype::BF16 => self.matvec_bits::<BFloat>(input, output),
            MetadataDtype::F32 => self.matvec_bits::<Float>(input, output),
        }
    }
    fn matvec_bits<M: Metadata>(&self, input: &[f32], output: &mut [f32]) -> Result<(), Error> {
        match self.layout.bits {
            2 => self.matvec_typed::<M, 2>(input, output),
            3 => self.matvec_typed::<M, 3>(input, output),
            4 => self.matvec_typed::<M, 4>(input, output),
            5 => self.matvec_typed::<M, 5>(input, output),
            6 => self.matvec_typed::<M, 6>(input, output),
            8 => self.matvec_typed::<M, 8>(input, output),
            _ => unreachable!("validated MLX bit width"),
        }
    }
    fn matvec_typed<M: Metadata, const B: u8>(
        &self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), Error> {
        for row in 0..self.layout.rows {
            if !(self.dot::<M, B>(row, input) as f32).is_finite() {
                return Err(Error("MLX affine matvec overflow"));
            }
        }
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.dot::<M, B>(row, input) as f32;
        }
        Ok(())
    }
    fn dot<M: Metadata, const B: u8>(&self, row: usize, input: &[f32]) -> f64 {
        let mut sum = 0.0;
        for (group, values) in input.chunks_exact(self.layout.group_size).enumerate() {
            let offset = (row * (self.layout.columns / self.layout.group_size) + group) * M::BYTES;
            let scale = M::read(&self.scales[offset..]);
            let bias = M::read(&self.biases[offset..]);
            for (lane, &x) in values.iter().enumerate() {
                let column = group * self.layout.group_size + lane;
                let weight = reconstruct(scale, bias, self.code::<B>(row, column));
                sum += f64::from(weight) * f64::from(x);
            }
        }
        sum
    }
}
fn reconstruct(scale: f32, bias: f32, code: u8) -> f32 {
    (f64::from(scale) * f64::from(code) + f64::from(bias)) as f32
}
