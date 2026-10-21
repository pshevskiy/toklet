use super::{Config, Error, product, total_bytes, zeros};

pub(crate) struct LayerKv {
    k: Vec<f32>,
    v: Vec<f32>,
}

/// Only session.evaluated_len defines validity. Writes beyond it may survive a
/// failed chunk but are overwritten on replay. K/V are separate head-major arrays.
pub(crate) struct KvCache {
    pub(super) layers: Vec<LayerKv>,
    pub(super) capacity: usize,
    pub(crate) allocated_bytes: usize,
    pub(crate) metadata_bytes: usize,
}

impl KvCache {
    pub(crate) fn planned_metadata_bytes(c: &Config) -> Result<usize, Error> {
        product(c.num_layers, std::mem::size_of::<LayerKv>()).map_err(Into::into)
    }
    pub(crate) fn planned_bytes(c: &Config, capacity: usize) -> Result<usize, Error> {
        let count = product(product(c.kv_heads, capacity)?, c.head_dim)?;
        product(total_bytes(&[count, count])?, c.num_layers).map_err(Into::into)
    }

    pub(crate) fn new(c: &Config, capacity: usize) -> Result<Self, Error> {
        Self::planned_bytes(c, capacity)?;
        Self::planned_metadata_bytes(c)?;
        let count = product(product(c.kv_heads, capacity)?, c.head_dim)?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(c.num_layers)
            .map_err(|_| Error::Allocation)?;
        let mut allocated_bytes = 0_usize;
        for _ in 0..c.num_layers {
            let layer = LayerKv {
                k: zeros(count)?,
                v: zeros(count)?,
            };
            allocated_bytes = allocated_bytes
                .checked_add(total_bytes(&[layer.k.capacity(), layer.v.capacity()])?)
                .ok_or(crate::cpu::Error::SizeOverflow)?;
            layers.push(layer);
        }
        let metadata_bytes = product(layers.capacity(), std::mem::size_of::<LayerKv>())?;
        Ok(Self {
            layers,
            capacity,
            allocated_bytes,
            metadata_bytes,
        })
    }
}

impl LayerKv {
    /// Convert only the new token-major chunk. No earlier token is moved.
    pub(super) fn append(
        &mut self,
        k: &[f32],
        v: &[f32],
        c: &Config,
        capacity: usize,
        offset: usize,
    ) {
        let chunk_stride = c.kv_heads * c.head_dim;
        for head in 0..c.kv_heads {
            for token in 0..k.len() / chunk_stride {
                let source = token * chunk_stride + head * c.head_dim;
                let target = (head * capacity + offset + token) * c.head_dim;
                self.k[target..target + c.head_dim]
                    .copy_from_slice(&k[source..source + c.head_dim]);
                self.v[target..target + c.head_dim]
                    .copy_from_slice(&v[source..source + c.head_dim]);
            }
        }
    }

    pub(super) fn attend(
        &self,
        q: &[f32],
        context: &mut [f32],
        scores: &mut [f32],
        probabilities: &mut [f32],
        shape: &crate::cpu::CachedAttentionShape,
    ) -> Result<(), Error> {
        crate::cpu::attention_cached(q, &self.k, &self.v, context, scores, probabilities, shape)?;
        Ok(())
    }
}
