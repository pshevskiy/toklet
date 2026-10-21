//! Opt-in snapshots, encoded as ordered GPU blits. Ordinary sessions allocate
//! none. Host conversion happens only after the completed chunk boundary.
use super::{Buffer, Error, Result, Session, add};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder};

pub(super) struct Snapshot {
    pub layer: Option<usize>,
    pub stage: &'static str,
    pub columns: usize,
    pub stride: usize,
    pub half: bool,
    pub buffer: Buffer<u32>,
}
#[derive(Debug)]
pub struct Diagnostic {
    pub layer: Option<usize>,
    pub stage: &'static str,
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<f32>,
}
impl Session<'_> {
    /// Admit diagnostics before execution; selected layers use row-major stage
    /// snapshots. No additional CPU/GPU wait is introduced. Re-enabling is rejected.
    pub fn enable_diagnostics(&mut self, layers: &[usize]) -> Result<()> {
        if !self.diagnostics.is_empty() || self.evaluated != 0 || self.pending_len() != 0 {
            return Err(Error::Options("enable diagnostics on a fresh session"));
        }
        if layers
            .iter()
            .enumerate()
            .any(|(i, l)| *l >= self.model.config.num_layers || layers[..i].contains(l))
        {
            return Err(Error::Options("invalid diagnostic layers"));
        }
        let c = &self.model.config;
        let mut plans = vec![
            (None, "embedding", c.hidden_size, self.h.stride, true),
            (None, "final_norm", c.hidden_size, self.h.stride, true),
            (None, "logits", c.vocab_size, self.f.stride, false),
        ];
        for &layer in layers {
            for (stage, columns, stride) in [
                ("input_norm", c.hidden_size, self.h.stride),
                ("q", c.hidden_size, self.h.stride),
                ("k", c.kv_heads * c.head_dim, self.k.stride),
                ("v", c.kv_heads * c.head_dim, self.k.stride),
                ("q_rotary", c.hidden_size, self.h.stride),
                ("k_rotary", c.kv_heads * c.head_dim, self.k.stride),
                ("attention_context", c.hidden_size, self.h.stride),
                ("attention_projection", c.hidden_size, self.h.stride),
                ("attention_residual", c.hidden_size, self.h.stride),
                ("post_attention_norm", c.hidden_size, self.h.stride),
                ("gate", c.intermediate_size, self.m.stride),
                ("silu", c.intermediate_size, self.m.stride),
                ("up", c.intermediate_size, self.m.stride),
                ("gated", c.intermediate_size, self.m.stride),
                ("down", c.hidden_size, self.h.stride),
                ("output", c.hidden_size, self.h.stride),
            ] {
                plans.push((Some(layer), stage, columns, stride, true));
            }
        }
        let chunk = self.options.prefill_chunk_size;
        let bytes = plans.iter().try_fold(0, |sum, (_, _, _, stride, half)| {
            add(sum, (chunk * stride + 7) * (if *half { 2 } else { 4 }))
        })?;
        // u32 backing rounds each byte count up; include snapshot metadata.
        let readback_bytes = plans.iter().try_fold(0, |sum, (_, _, columns, _, _)| {
            add(sum, chunk * columns * 4)
        })?;
        let bytes = add(
            add(bytes, readback_bytes)?,
            plans.len() * (2 + std::mem::size_of::<Snapshot>()),
        )?;
        let admitted = add(self.memory.admitted_bytes, bytes)?;
        if admitted > self.options.memory_limit_bytes {
            return Err(Error::Budget {
                required: admitted,
                limit: self.options.memory_limit_bytes,
            });
        }
        let mut snapshots = Vec::with_capacity(plans.len());
        for (layer, stage, columns, stride, half) in plans {
            let bytes = (chunk * stride + 7) * (if half { 2 } else { 4 });
            snapshots.push(Snapshot {
                layer,
                stage,
                columns,
                stride,
                half,
                buffer: Buffer::new(&self.model.gpu, bytes.div_ceil(4))?,
            });
        }
        self.diagnostics = snapshots;
        self.memory.scratch_bytes = add(self.memory.scratch_bytes, bytes)?;
        self.memory.admitted_bytes = admitted;
        Ok(())
    }
    /// Snapshots belong to the last successful chunk, including all its rows.
    pub fn diagnostics(&self) -> Result<Vec<Diagnostic>> {
        if self.last_timing.is_none() || self.replay || self.diagnostic_rows == 0 {
            return Err(Error::Options("no completed diagnostic chunk"));
        }
        Ok(self
            .diagnostics
            .iter()
            .map(|s| {
                let values = s.buffer.with_slice(|words| {
                    let mut values = Vec::with_capacity(self.diagnostic_rows * s.columns);
                    for row in 0..self.diagnostic_rows {
                        for col in 0..s.columns {
                            let i = row * s.stride + col;
                            let value = if s.half {
                                let word = words[i / 2];
                                let bits = if i % 2 == 0 {
                                    word as u16
                                } else {
                                    (word >> 16) as u16
                                };
                                half::f16::from_bits(bits).to_f32()
                            } else {
                                f32::from_bits(words[i])
                            };
                            values.push(value);
                        }
                    }
                    values
                });
                Diagnostic {
                    layer: s.layer,
                    stage: s.stage,
                    rows: self.diagnostic_rows,
                    columns: s.columns,
                    values,
                }
            })
            .collect())
    }
    pub(super) fn snapshot(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        layer: Option<usize>,
        stage: &'static str,
        source: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
    ) -> Result<()> {
        if let Some(s) = self
            .diagnostics
            .iter()
            .find(|s| s.layer == layer && s.stage == stage)
        {
            let bytes = self.options.prefill_chunk_size * s.stride * if s.half { 2 } else { 4 };
            if !bytes.is_multiple_of(4) || bytes > source.length() || bytes > s.buffer.raw.length()
            {
                return Err(Error::Options("diagnostic blit bounds/alignment differ"));
            }
            let e = command
                .blitCommandEncoder()
                .ok_or_else(|| Error::Native("diagnostic encoder absent".into()))?;
            // SAFETY: same-device initialized disjoint retained buffers. Copy only
            // aligned logical payload bytes (excluding guards); snapshot holds them.
            unsafe {
                e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    source,
                    0,
                    &s.buffer.raw,
                    0,
                    bytes,
                );
            }
            e.endEncoding();
        }
        Ok(())
    }
}
