//! Precise projection from original row-major BF16 Metal coefficients.
use super::*;

/// Dedicated BF16 projection pipeline and device/queue owner.
pub struct Bf16Context(Context);

impl Bf16Context {
    pub fn new(available_bytes: usize) -> Result<Self> {
        Context::with_projection_pipeline(available_bytes, 16).map(Self)
    }

    pub fn device_name(&self) -> String {
        self.0.device_name()
    }

    /// Upload the exact little-endian BF16 words once. The separate context
    /// allowance and caller-owned slices are outside the projection budget.
    pub fn projection(
        &self,
        rows: usize,
        columns: usize,
        coefficients: &[u8],
        capacity: usize,
        available_bytes: usize,
    ) -> Result<Bf16Projection<'_>> {
        let (sizes, owned) = projection_sizes(rows, columns, capacity)?;
        if owned > available_bytes {
            return Err(Error::Budget {
                required: owned,
                limit: available_bytes,
            });
        }
        if coefficients.len() != sizes[0] * 4 {
            return Err(Error::Options("Gemma BF16 coefficient length differs"));
        }
        let maximum = maximum_coefficient(coefficients)?;
        let gpu = &self.0.gpu;
        for words in sizes {
            validate_buffer_bytes((words + GUARD) * 4, gpu.device.maxBufferLength())?;
        }
        Ok(Bf16Projection {
            context: &self.0,
            rows,
            columns,
            capacity,
            owned,
            maximum,
            coefficients: loading::upload(gpu, coefficients)?,
            input: Buffer::new(gpu, sizes[1] + GUARD)?,
            output: Buffer::new(gpu, sizes[2] + GUARD)?,
            status: Buffer::new(gpu, sizes[3] + GUARD)?,
            pending: None,
            poisoned: false,
        })
    }
}

fn projection_sizes(rows: usize, columns: usize, capacity: usize) -> Result<([usize; 4], usize)> {
    if !(1..=16384).contains(&rows)
        || !(64..=12288).contains(&columns)
        || !columns.is_multiple_of(64)
        || !(1..=32).contains(&capacity)
    {
        return Err(Error::Options(
            "Gemma BF16 projection geometry outside bounds",
        ));
    }
    let sizes = [rows * columns / 2, capacity * columns, capacity * rows, 1];
    let owned = sizes
        .into_iter()
        .try_fold(0usize, |sum, words| {
            sum.checked_add(words)?.checked_add(GUARD)
        })
        .and_then(|words| words.checked_mul(4))
        .and_then(|bytes| bytes.checked_add(size_of::<Bf16Projection<'_>>()))
        .ok_or(Error::Options("Gemma BF16 projection size overflow"))?;
    Ok((sizes, owned))
}

fn maximum_coefficient(raw: &[u8]) -> Result<f64> {
    let mut maximum = 0;
    for word in raw.chunks_exact(2) {
        let bits = u16::from_le_bytes([word[0], word[1]]) & 0x7fff;
        if bits >= 0x7f80 || (1..0x80).contains(&bits) {
            return Err(Error::Unsupported("Gemma BF16 coefficient domain"));
        }
        maximum = maximum.max(bits);
    }
    let value = f32::from_bits(u32::from(maximum) << 16);
    if value > MAX_WEIGHT {
        return Err(Error::Unsupported("Gemma BF16 coefficient magnitude"));
    }
    Ok(f64::from(value))
}

/// Bounded standalone projection. Coefficients and FP32 scratch stay
/// resident. Output is published only after the shared completion checks.
pub struct Bf16Projection<'a> {
    context: &'a Context,
    rows: usize,
    columns: usize,
    capacity: usize,
    owned: usize,
    maximum: f64,
    coefficients: Buffer<u32>,
    input: Buffer<u32>,
    output: Buffer<u32>,
    status: Buffer<u32>,
    pending: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    poisoned: bool,
}

impl Bf16Projection<'_> {
    pub fn minimum_owned_bytes(rows: usize, columns: usize, capacity: usize) -> Result<usize> {
        Ok(projection_sizes(rows, columns, capacity)?.1)
    }

    pub fn owned_bytes(&self) -> usize {
        self.owned
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub fn apply(&mut self, input: &[f32], output: &mut [f32], tokens: usize) -> Result<()> {
        if self.poisoned || self.pending.is_some() {
            return Err(Error::Options("Gemma BF16 projection poisoned; recreate"));
        }
        if tokens == 0
            || tokens > self.capacity
            || input.len() != tokens * self.columns
            || output.len() != tokens * self.rows
            || input.iter().any(|value| !value.is_finite())
        {
            return Err(Error::Options("Gemma BF16 projection request differs"));
        }
        admit(self.columns, self.maximum, input)?;
        self.poisoned = true;
        project::run(
            self.context,
            project::Matrix::bf16(self.rows, self.columns, &self.coefficients),
            project::Buffers {
                input: &mut self.input,
                output: &mut self.output,
                status: &mut self.status,
                pending: &mut self.pending,
            },
            input,
            output,
            tokens,
        )?;
        self.poisoned = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
