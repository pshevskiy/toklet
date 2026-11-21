//! Cold certificates and existing-route census for the two T128 projections.
use super::{pair::domain_tag, *};
use crate::{metal::lfm3_shared_weights::BankRef, model::Error};

const T: usize = 128;
const LAYERS: usize = 24;
const CONVS: usize = 13;
const K: [usize; 2] = [2048, 7680];
const N: [usize; 2] = [6144, 2048];

#[derive(Clone, Copy, Default)]
struct Census {
    calls: u64,
    tiled: u64,
    rows: [u64; 3],
}

pub(super) struct Tiles {
    kernels: [Kernels; 2],
    input: [Buffer<u32>; 2],
    weights: [Buffer<u32>; 2],
    // Identity and compact tag-buffer row offset. Zero identity means absent.
    banks: [[(usize, usize); LAYERS]; 2],
    sealed: [usize; 2],
    counts: [[[Census; 3]; LAYERS]; 2],
}

impl Tiles {
    pub(super) const BYTES: usize = (2 * T + CONVS * N[0] + LAYERS * N[1]) * size_of::<u32>();

    pub(super) fn new(gpu: &Gpu) -> std::result::Result<Self, Error> {
        let buffer = |n| Buffer::new(gpu, n).map_err(|_| Error::Allocation);
        let kernel = |text| {
            Kernels::with_shader(gpu, text)
                .ok_or(Error::InvalidConfig("LFM3 projection tile pipelines"))
        };
        Ok(Self {
            kernels: [
                kernel(include_str!("panel84_convin.metal"))?,
                kernel(include_str!("panel84_down.metal"))?,
            ],
            input: [buffer(T)?, buffer(T)?],
            weights: [buffer(CONVS * N[0])?, buffer(LAYERS * N[1])?],
            banks: [[(0, 0); LAYERS]; 2],
            sealed: [0; 2],
            counts: Default::default(),
        })
    }

    pub(super) fn seal(
        &mut self,
        role: usize,
        layer: usize,
        bank: BankRef<'_>,
        data: &[bf16],
    ) -> std::result::Result<(), Error> {
        let (k, n, limit) = (K[role], N[role], [CONVS, LAYERS][role]);
        if self.banks[role][layer].0 != 0
            || bank.identity() == 0
            || data.len() != k * n
            || self.sealed[role] == limit
        {
            return Err(Error::InvalidConfig("LFM3 projection tile seal"));
        }
        let offset = self.sealed[role] * n;
        self.weights[role].with_mut(|dst| {
            for (tag, row) in dst[offset..offset + n].iter_mut().zip(data.chunks_exact(k)) {
                *tag = domain_tag(row.iter().map(|w| w.to_bits()));
            }
        });
        self.banks[role][layer] = (bank.identity(), offset);
        self.sealed[role] += 1;
        Ok(())
    }

    pub(super) fn launch<'p>(
        &'p mut self,
        role: usize,
        layer: usize,
        bank: BankRef<'_>,
        tokens: usize,
        decode_projection: Option<&'p Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    ) -> std::result::Result<Option<Launch<'p>>, Error> {
        let (identity, offset) = self.banks[role][layer];
        if self.sealed != [CONVS, LAYERS] || identity == 0 || identity != bank.identity() {
            return Err(Error::InvalidConfig("LFM3 projection tile bank identity"));
        }
        let fma4 = decode_projection.is_some() && tokens == 1 && role == 1;
        Ok((tokens == T || fma4).then(|| Launch {
            kernels: &self.kernels[role],
            decode_projection: if fma4 { decode_projection } else { None },
            outputs_per_group: if fma4 { 4 } else { 32 },
            input: &mut self.input[role],
            weights: &self.weights[role],
            offset_bytes: offset * size_of::<u32>(),
        }))
    }

    pub(super) fn completed(
        &mut self,
        role: usize,
        layer: usize,
        tokens: usize,
        tiled: bool,
        rows: [u64; 3],
    ) {
        let s = &mut self.counts[role][layer][match tokens {
            T => 0,
            1 => 1,
            _ => 2,
        }];
        // These counts can only advance after a checked workspace submission.
        s.calls += 1;
        s.tiled += u64::from(tiled);
        for (total, delta) in s.rows.iter_mut().zip(rows) {
            *total += delta;
        }
    }

    pub(super) fn report(&self) -> serde_json::Value {
        let mut result = serde_json::Map::new();
        for (role, name) in ["convin", "down"].into_iter().enumerate() {
            let rows = self.counts[role]
                .iter()
                .enumerate()
                .filter(|(layer, _)| self.banks[role][*layer].0 != 0)
                .map(|(layer, shapes)| {
                    let shape = |s: &Census, decode_down| {
                        let mut out = serde_json::json!({
                            "completed": s.calls, "tiled": if decode_down { 0 } else { s.tiled },
                            "homogeneous_rows": if decode_down { 0 } else { s.rows[0] },
                            "mixed_native_rows": s.rows[1], "ordered_rows": s.rows[2],
                        });
                        if decode_down {
                            let quartets = s.rows[0] / 4;
                            out["fma4_calls"] = serde_json::json!(s.tiled);
                            out["fma4_quartets"] = serde_json::json!(quartets);
                            out["fallback_quartets"] = serde_json::json!(s.tiled * (N[1] / 4) as u64 - quartets);
                        }
                        out
                    };
                    serde_json::json!({"layer":layer,"t128":shape(&shapes[0],false),
                                       "t1":shape(&shapes[1],role == 1),"tail":shape(&shapes[2],false)})
                })
                .collect::<Vec<_>>();
            result.insert(name.to_owned(), serde_json::json!(rows));
        }
        result.insert("mode".to_owned(), serde_json::json!("panel84"));
        result.into()
    }
}

pub(super) struct Launch<'a> {
    pub(super) kernels: &'a Kernels,
    pub(super) decode_projection: Option<&'a Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(super) outputs_per_group: usize,
    input: &'a mut Buffer<u32>,
    weights: &'a Buffer<u32>,
    offset_bytes: usize,
}

impl Launch<'_> {
    pub(super) fn stage(&mut self, input: &[bf16], dst: &mut [u16], k: usize) {
        self.input.with_mut(|tags| {
            for ((source, target), tag) in
                input.chunks_exact(k).zip(dst.chunks_exact_mut(k)).zip(tags)
            {
                *tag = domain_tag(source.iter().zip(target).map(|(x, y)| {
                    *y = x.to_bits();
                    *y
                }));
            }
        });
    }

    pub(super) fn bind_tags(&self, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        // SAFETY: distinct same-device allocations remain live through the
        // original synchronous wait. Sealing bounds the selected complete row span.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.input.raw), 0, 6);
        }
        self.bind_weight_tags(encoder);
    }

    pub(super) fn bind_weight_tags(&self, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        // SAFETY: The complete row span was checked when this launch was sealed.
        unsafe { encoder.setBuffer_offset_atIndex(Some(&self.weights.raw), self.offset_bytes, 7) };
    }
}

pub(super) fn record(word: u32, shift: u32, k: usize, tiled: bool, rows: &mut [u64; 3]) -> bool {
    let index = match word {
        0xc0000000 if tiled => 0,
        0x80000000 if tiled => 1,
        n if (n & ((1 << shift) - 1)) + (n >> shift) <= (k + 32) as u32 => 2,
        _ => return false,
    };
    rows[index] += 1;
    true
}

// The extrema of the Cartesian product are sums of the independent
// input and weight extrema. Every individual pair then meets normal_pair.
#[derive(Clone, Copy)]
pub(super) struct NormalExtent {
    min: i32,
    max: i32,
}

impl NormalExtent {
    pub(super) fn from_tags(tags: &[u32]) -> Option<Self> {
        let mut min = 255;
        let mut max = 0;
        for &tag in tags {
            if tag & 0x30000 != 0x10000 {
                return None;
            }
            min = min.min((tag & 255) as i32);
            max = max.max(((tag >> 8) & 255) as i32);
        }
        (min <= max).then_some(Self { min, max })
    }

    pub(super) fn certifies(self, weights: &[u32]) -> bool {
        Self::from_tags(weights)
            .is_some_and(|w| self.min + w.min - 254 >= -112 && self.max + w.max - 254 <= 109)
    }
}
