//! Borrowed, validated Toklet Q4 v1 matrix and scalar reconstruction oracle.
//! See `docs/q4-v1.md`. Parsing allocates nothing; reconstruction is FP32.

use half::f16;
use std::fmt;

mod artifact;
pub use artifact::Artifact;
pub const QUANTIZER: &str = "toklet-q4-affine-f32-source-half-metadata-rne-v1";

pub const GROUP: usize = 64;
pub const RECORD_BYTES: usize = 36;
pub const HEADER_BYTES: usize = 64;
pub const MAGIC: &[u8; 8] = b"TOKQ4V1\0";

/// The seven eligible row-major matrices per layer, resolved once at loading.
pub fn matrix_shapes(
    config: &crate::model::Config,
) -> Result<std::collections::BTreeMap<String, Vec<usize>>, crate::model::Error> {
    config.validate()?;
    if config.num_layers > 1024 {
        return Err(crate::model::Error::InvalidConfig(
            "Q4 layer count exceeds bounded artifact inventory",
        ));
    }
    let h = config.hidden_size;
    let m = config.intermediate_size;
    let k = config.kv_heads * config.head_dim;
    let mut shapes = std::collections::BTreeMap::new();
    for layer in 0..config.num_layers {
        for (name, rows, columns) in [
            ("self_attn.q_proj.weight", h, h),
            ("self_attn.k_proj.weight", k, h),
            ("self_attn.v_proj.weight", k, h),
            ("self_attn.o_proj.weight", h, h),
            ("mlp.gate_proj.weight", m, h),
            ("mlp.up_proj.weight", m, h),
            ("mlp.down_proj.weight", h, m),
        ] {
            shapes.insert(format!("model.layers.{layer}.{name}"), vec![rows, columns]);
        }
    }
    Ok(shapes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error(pub &'static str);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for Error {}

/// Byte-backed storage imposes no alignment requirement on the Rust slice.
/// Artifact row starts are aligned to 64 bytes relative to the file origin.
pub struct Matrix<'a> {
    bytes: &'a [u8],
    rows: usize,
    columns: usize,
    stride: usize,
}
fn integer(bytes: &[u8], offset: usize) -> Result<usize, Error> {
    usize::try_from(u64::from_le_bytes(
        bytes[offset..offset + 8].try_into().unwrap(),
    ))
    .map_err(|_| Error("Q4 dimension exceeds address space"))
}
fn metadata(bytes: &[u8]) -> (f32, f32) {
    (
        f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32(),
        f16::from_bits(u16::from_le_bytes([bytes[2], bytes[3]])).to_f32(),
    )
}
impl<'a> Matrix<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        if bytes.len() < HEADER_BYTES
            || &bytes[..8] != MAGIC
            || bytes[8..12] != 1_u32.to_le_bytes()
            || bytes[12..16] != 64_u32.to_le_bytes()
            || bytes[48..64].iter().any(|&b| b != 0)
        {
            return Err(Error("invalid Q4 v1 header"));
        }
        let rows = integer(bytes, 16)?;
        let columns = integer(bytes, 24)?;
        if rows == 0 || columns == 0 {
            return Err(Error("Q4 matrix dimensions must be positive"));
        }
        let groups = columns.div_ceil(GROUP);
        let logical = groups
            .checked_mul(RECORD_BYTES)
            .ok_or(Error("Q4 size overflow"))?;
        let stride = logical.checked_add(63).ok_or(Error("Q4 size overflow"))? / 64 * 64;
        let payload = rows.checked_mul(stride).ok_or(Error("Q4 size overflow"))?;
        let total = payload
            .checked_add(HEADER_BYTES)
            .ok_or(Error("Q4 size overflow"))?;
        if total > isize::MAX as usize
            || integer(bytes, 32)? != stride
            || integer(bytes, 40)? != payload
            || bytes.len() != total
        {
            return Err(Error("Q4 payload length or row stride mismatch"));
        }
        for row in bytes[HEADER_BYTES..].chunks_exact(stride) {
            if row[logical..].iter().any(|&b| b != 0) {
                return Err(Error("Q4 row padding must be zero"));
            }
            for (group, record) in row[..logical].chunks_exact(RECORD_BYTES).enumerate() {
                let (scale, bias) = metadata(record);
                if !scale.is_finite() || scale.is_sign_negative() || !bias.is_finite() {
                    return Err(Error(
                        "Q4 metadata must contain finite bias and nonnegative scale",
                    ));
                }
                let valid = (columns - group * GROUP).min(GROUP);
                for lane in 0..GROUP {
                    let code = (record[4 + lane / 2] >> (4 * (lane % 2))) & 15;
                    if (lane >= valid || scale == 0.0) && code != 0 {
                        return Err(Error("Q4 constant or tail codes must be zero"));
                    }
                }
            }
        }
        Ok(Self {
            bytes,
            rows,
            columns,
            stride,
        })
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn row_stride(&self) -> usize {
        self.stride
    }
    pub fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }
    pub fn value(&self, row: usize, column: usize) -> Result<f32, Error> {
        if row >= self.rows || column >= self.columns {
            return Err(Error("Q4 index outside matrix"));
        }
        Ok(self.at(row, column))
    }
    fn at(&self, row: usize, column: usize) -> f32 {
        let start = HEADER_BYTES + row * self.stride + column / GROUP * RECORD_BYTES;
        let record = &self.bytes[start..start + RECORD_BYTES];
        let (scale, bias) = metadata(record);
        let lane = column % GROUP;
        let code = (record[4 + lane / 2] >> (4 * (lane % 2))) & 15;
        // The Half scale times a four-bit code is exact in FP32. The addition
        // rounds once to FP32; the oracle does not cast reconstruction to Half.
        scale * f32::from(code) + bias
    }
    /// Independent oracle: FP64 products/accumulation, FP32 output, no expansion.
    pub fn matvec(&self, input: &[f32], output: &mut [f32]) -> Result<(), Error> {
        if input.len() != self.columns || output.len() != self.rows {
            return Err(Error("Q4 matvec shape mismatch"));
        }
        if input.iter().any(|v| !v.is_finite()) {
            return Err(Error("Q4 input must be finite"));
        }
        // Validate the whole result before publishing any caller output.
        for row in 0..self.rows {
            if !(self.dot(row, input) as f32).is_finite() {
                return Err(Error("Q4 output overflow"));
            }
        }
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.dot(row, input) as f32;
        }
        Ok(())
    }
    fn dot(&self, row: usize, input: &[f32]) -> f64 {
        input
            .iter()
            .enumerate()
            .map(|(column, &x)| f64::from(self.at(row, column)) * f64::from(x))
            .sum()
    }
}
