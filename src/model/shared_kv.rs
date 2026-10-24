//! Concrete local/global KV producer routing, outside numerical hot loops.
//!
//! A plan owns immutable routes. Callers own cache arrays and borrowed proposals;
//! the first `layers - shared_layers` entries are the only cache owners. Shared
//! layers borrow their last preceding same-kind producer's current-step view.
//! Model identity and session/token-history provenance remain caller-owned.

use super::Error;
use super::window_cache::{KvView, PreparedKv, WindowKvCache, commit_batch};
use crate::cpu::{self, length, nonzero, product};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvKind {
    Sliding,
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerKvGeometry {
    pub kind: KvKind,
    pub heads: usize,
    pub head_dim: usize,
}

struct Route {
    geometry: LayerKvGeometry,
    producer: usize,
}

pub struct SharedKvPlan {
    routes: Vec<Route>,
    producers: usize,
    window: usize,
    global_capacity: usize,
    max_positions: usize,
}

impl SharedKvPlan {
    pub fn planned_metadata_bytes(layers: usize) -> Result<usize, Error> {
        nonzero(layers, "KV routing layers")?;
        let bytes = product(layers, std::mem::size_of::<Route>())?;
        if bytes > isize::MAX as usize {
            return Err(cpu::Error::SizeOverflow.into());
        }
        bytes
            .checked_add(std::mem::size_of::<Self>())
            .ok_or(cpu::Error::SizeOverflow.into())
    }

    pub fn new(
        layers: &[LayerKvGeometry],
        shared_layers: usize,
        window: usize,
        global_capacity: usize,
        max_positions: usize,
    ) -> Result<Self, Error> {
        Self::planned_metadata_bytes(layers.len())?;
        if shared_layers >= layers.len() {
            return Err(Error::InvalidConfig("KV routing needs a producer"));
        }
        nonzero(window, "KV routing window")?;
        nonzero(global_capacity, "KV routing global capacity")?;
        if window > max_positions || global_capacity > max_positions {
            return Err(Error::InvalidConfig("KV routing position/capacity limit"));
        }
        let producers = layers.len() - shared_layers;
        // Validate all geometry and producer availability before allocating routes.
        for (layer, &geometry) in layers.iter().enumerate() {
            let (capacity, mode) = match geometry.kind {
                KvKind::Sliding => (window, Some(window)),
                KvKind::Full => (global_capacity, None),
            };
            WindowKvCache::planned_bytes(
                geometry.heads,
                geometry.head_dim,
                capacity,
                max_positions,
                mode,
            )?;
            if layer >= producers {
                let source = layers[..producers]
                    .iter()
                    .rfind(|source| source.kind == geometry.kind)
                    .ok_or(Error::InvalidConfig("missing shared KV producer kind"))?;
                if *source != geometry {
                    return Err(Error::InvalidConfig("shared KV producer geometry differs"));
                }
            }
        }
        let mut routes: Vec<Route> = Vec::new();
        routes
            .try_reserve_exact(layers.len())
            .map_err(|_| Error::Allocation)?;
        let bytes = product(routes.capacity(), std::mem::size_of::<Route>())?;
        bytes
            .checked_add(std::mem::size_of::<Self>())
            .ok_or(cpu::Error::SizeOverflow)?;
        for (layer, &geometry) in layers.iter().enumerate() {
            let producer = if layer < producers {
                layer
            } else {
                // All requested kinds were validated above; resolve once at load.
                layers[..producers]
                    .iter()
                    .rposition(|source| source.kind == geometry.kind)
                    .unwrap()
            };
            routes.push(Route { geometry, producer });
        }
        let result = Self {
            routes,
            producers,
            window,
            global_capacity,
            max_positions,
        };
        result.planned_cache_bytes()?;
        Ok(result)
    }

    pub fn layer_count(&self) -> usize {
        self.routes.len()
    }

    pub fn producer_count(&self) -> usize {
        self.producers
    }

    pub fn producer_for(&self, layer: usize) -> Result<usize, Error> {
        self.routes
            .get(layer)
            .map(|route| route.producer)
            .ok_or(Error::InvalidConfig("KV routing layer"))
    }

    pub fn allocated_metadata_bytes(&self) -> usize {
        self.routes.capacity() * std::mem::size_of::<Route>() + std::mem::size_of::<Self>()
    }

    fn storage(&self, producer: usize) -> (usize, Option<usize>) {
        match self.routes[producer].geometry.kind {
            KvKind::Sliding => (self.window, Some(self.window)),
            KvKind::Full => (self.global_capacity, None),
        }
    }

    /// Cache Vec header, occupied owner objects and their two FP32 buffers.
    /// Route metadata, new-token staging and numerical scratch are separate.
    pub fn planned_cache_bytes(&self) -> Result<usize, Error> {
        let mut total = std::mem::size_of::<Vec<WindowKvCache>>();
        for producer in 0..self.producers {
            let geometry = self.routes[producer].geometry;
            let (capacity, window) = self.storage(producer);
            let bytes = WindowKvCache::planned_bytes(
                geometry.heads,
                geometry.head_dim,
                capacity,
                self.max_positions,
                window,
            )?;
            total = total.checked_add(bytes).ok_or(cpu::Error::SizeOverflow)?;
        }
        Ok(total)
    }

    /// Allocate only producer caches once, outside decode. The caller must admit
    /// these and all other model/session owners against its actual device budget.
    /// This checked size estimate is not a process-memory admission decision.
    pub fn new_caches(&self) -> Result<Vec<WindowKvCache>, Error> {
        self.planned_cache_bytes()?;
        let mut caches = Vec::new();
        caches
            .try_reserve_exact(self.producers)
            .map_err(|_| Error::Allocation)?;
        for producer in 0..self.producers {
            let geometry = self.routes[producer].geometry;
            let (capacity, window) = self.storage(producer);
            caches.push(WindowKvCache::new(
                geometry.heads,
                geometry.head_dim,
                capacity,
                self.max_positions,
                window,
            )?);
        }
        self.allocated_cache_bytes(&caches)?;
        Ok(caches)
    }

    /// Includes Vec excess capacity and header without double-counting occupied
    /// owner objects, already included in each cache's allocated_bytes result.
    pub fn allocated_cache_bytes(&self, caches: &Vec<WindowKvCache>) -> Result<usize, Error> {
        self.check_caches(caches)?;
        let unused = caches.capacity() - caches.len();
        let mut total = product(unused, std::mem::size_of::<WindowKvCache>())?
            .checked_add(std::mem::size_of::<Vec<WindowKvCache>>())
            .ok_or(cpu::Error::SizeOverflow)?;
        for cache in caches {
            total = total
                .checked_add(cache.allocated_bytes())
                .ok_or(cpu::Error::SizeOverflow)?;
        }
        Ok(total)
    }

    fn check_caches(&self, caches: &[WindowKvCache]) -> Result<(), Error> {
        length("KV producer owners", caches.len(), self.producers)?;
        for (producer, cache) in caches.iter().enumerate() {
            let geometry = self.routes[producer].geometry;
            let (capacity, window) = self.storage(producer);
            if !cache.matches_geometry(
                geometry.heads,
                geometry.head_dim,
                capacity,
                self.max_positions,
                window,
            ) {
                return Err(Error::InvalidConfig(
                    "KV owner differs from routing geometry",
                ));
            }
        }
        Ok(())
    }

    pub fn prepare_producer<'input>(
        &self,
        producer: usize,
        caches: &[WindowKvCache],
        k: &'input [f32],
        v: &'input [f32],
        tokens: usize,
    ) -> Result<PreparedKv<'input>, Error> {
        if producer >= self.producers {
            return Err(Error::InvalidConfig("shared layer has no private KV cache"));
        }
        self.check_caches(caches)?;
        caches[producer].prepare(k, v, tokens)
    }

    fn validate_prefix(
        &self,
        caches: &[WindowKvCache],
        frames: &[PreparedKv<'_>],
    ) -> Result<(), Error> {
        self.check_caches(caches)?;
        if frames.len() > self.producers {
            return Err(Error::InvalidConfig(
                "KV publication exceeds producer count",
            ));
        }
        let offset = caches[0].offset();
        let generation = caches[0].generation();
        // Check even unpublished owners' committed metadata before any view.
        for cache in caches {
            if cache.offset() != offset || cache.generation() != generation {
                return Err(Error::InvalidConfig(
                    "KV committed owners do not describe one prefix",
                ));
            }
        }
        let mut end = None;
        for (cache, frame) in caches.iter().zip(frames) {
            let view = cache.view(frame)?;
            if end.is_some_and(|position| view.end() != position) {
                return Err(Error::InvalidConfig(
                    "KV published producers do not describe one step",
                ));
            }
            end = Some(view.end());
        }
        Ok(())
    }

    fn validate_batch(
        &self,
        caches: &[WindowKvCache],
        frames: &[PreparedKv<'_>],
    ) -> Result<(), Error> {
        length("KV producer proposals", frames.len(), self.producers)?;
        self.validate_prefix(caches, frames)
    }

    /// Sequential execution publishes an ordered prefix of producer proposals.
    /// Validate every committed owner and every published frame before borrowing
    /// a selected producer. Future token counts are checked when published.
    /// Staging and frame-container capacity are preallocated/accounted by caller.
    pub fn published_view<'cache, 'input>(
        &self,
        layer: usize,
        caches: &'cache [WindowKvCache],
        published: &[PreparedKv<'input>],
    ) -> Result<KvView<'cache, 'input>, Error> {
        let producer = self.producer_for(layer)?;
        self.validate_prefix(caches, published)?;
        if producer >= published.len() {
            return Err(Error::InvalidConfig("KV producer not published"));
        }
        caches[producer].view(&published[producer])
    }

    /// Validate every producer, including ones the requested layer does not use.
    /// A view borrows the selected owner's actual old/new spans; no append/copy.
    pub fn view<'cache, 'input>(
        &self,
        layer: usize,
        caches: &'cache [WindowKvCache],
        frames: &[PreparedKv<'input>],
    ) -> Result<KvView<'cache, 'input>, Error> {
        let producer = self.producer_for(layer)?;
        self.validate_batch(caches, frames)?;
        caches[producer].view(&frames[producer])
    }

    pub fn commit(
        &self,
        caches: &mut [WindowKvCache],
        frames: &[PreparedKv<'_>],
    ) -> Result<(), Error> {
        self.validate_batch(caches, frames)?;
        commit_batch(caches, frames)
    }
}

#[cfg(test)]
mod tests;
