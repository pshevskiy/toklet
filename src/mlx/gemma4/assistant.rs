//! Exact official E2B assistant metadata and ordered-head vocabulary routing.
//! Numerical draft execution and target verification are separate.
use super::{Dtype, PhysicalTensor, descriptor, invalid};
use crate::weights::{
    Error,
    audit::open_regular,
    content::{ExpectedFile, FileAudit, Manifest},
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Read, mem::size_of, ops::Range, path::Path};

pub const REPOSITORY: &str = "google/gemma-4-E2B-it-assistant";
pub const REVISION: &str = "2d874ef7d29f9a30599a1e4b3c1cbc9595f005df";
pub const CONTAINER_BYTES: usize = 157_565_344;
pub const VOCAB: usize = 262_144;
pub const CENTROIDS: usize = 2048;
pub const CANDIDATES: usize = 32 * (VOCAB / CENTROIDS);
const CONFIG_BYTES: usize = 2_315;
const HEADER_BYTES: usize = 5_520;
const HEADER_SHA256: &str = "6179f5c61f819bf278f6199e454d2fe5b5846dda74a6cd7adb93672c492960a3";
const ORDERING: &str = "masked_embedding.token_ordering";
const TOP_CENTROIDS: usize = 32;
const IDS_PER_CENTROID: usize = VOCAB / CENTROIDS;
const ORDERING_BYTES: usize = VOCAB * size_of::<i64>();

pub struct Plan {
    tensors: BTreeMap<String, PhysicalTensor>,
}

impl Plan {
    pub fn tensors(&self) -> &BTreeMap<String, PhysicalTensor> {
        &self.tensors
    }

    /// Payload-relative byte range containing little-endian signed token IDs.
    pub fn ordering_range(&self) -> Range<usize> {
        0..ORDERING_BYTES
    }

    pub const fn payload_offset(&self) -> usize {
        HEADER_BYTES + 8
    }
}

fn pinned(bytes: &[u8], length: usize, digest: &str) -> bool {
    bytes.len() == length && format!("{:x}", Sha256::digest(bytes)) == digest
}

/// Check the exact selected assistant config and header. No tensor payload is
/// read or authenticated by this metadata-only entry point.
pub fn inspect(config: &[u8], header: &[u8], container_bytes: usize) -> Result<Plan, Error> {
    if container_bytes != CONTAINER_BYTES
        || (config.len() != CONFIG_BYTES
            || <[u8; 32]>::from(Sha256::digest(config)) != FILES[0].sha256)
        || !pinned(header, HEADER_BYTES, HEADER_SHA256)
    {
        return Err(invalid("assistant pinned source metadata differs"));
    }
    let config = crate::mlx::qwen::parse_json(config)?;
    if config["model_type"] != "gemma4_assistant"
        || config["architectures"] != json!(["Gemma4AssistantForCausalLM"])
    {
        return Err(invalid("assistant architecture differs"));
    }
    let document = crate::mlx::qwen::parse_json(header)?;
    let header = document
        .as_object()
        .filter(|h| h.len() == 51)
        .ok_or_else(|| invalid("assistant inventory differs"))?;
    if header.get("__metadata__") != Some(&json!({"format":"pt"})) {
        return Err(invalid("assistant container format differs"));
    }
    let payload = container_bytes - HEADER_BYTES - 8;
    if header.get(ORDERING)
        != Some(&json!({
            "dtype": "I64", "shape": [VOCAB], "data_offsets": [0, ORDERING_BYTES]
        }))
    {
        return Err(invalid("assistant token ordering descriptor differs"));
    }
    let mut spans = Vec::with_capacity(header.len() - 1);
    spans.push(0..ORDERING_BYTES);
    let mut tensors = BTreeMap::new();
    for (name, value) in header {
        if name == "__metadata__" || name == ORDERING {
            continue;
        }
        let tensor = descriptor(value, payload)?;
        if tensor.dtype != Dtype::Bf16 {
            return Err(invalid("assistant matrix representation differs"));
        }
        spans.push(tensor.range.clone());
        tensors.insert(name.clone(), tensor);
    }
    spans.sort_unstable_by_key(|span| span.start);
    let mut cursor = 0;
    for span in spans {
        if span.start != cursor || span.is_empty() {
            return Err(invalid(
                "assistant tensor spans do not partition the payload",
            ));
        }
        cursor = span.end;
    }
    if cursor != payload {
        return Err(invalid("assistant tensor coverage differs"));
    }
    Ok(Plan { tensors })
}

/// Bounded metadata, audit and reader storage during assistant source setup.
pub const SOURCE_ALLOWANCE_BYTES: usize = 512 * 1024;
/// Cold ordering input plus normalized output and permutation validation.
pub const ORDERING_ALLOWANCE_BYTES: usize =
    ORDERING_BYTES + VOCAB * (size_of::<u32>() + size_of::<bool>());

const FILES: [ExpectedFile<'static>; 2] = [
    ExpectedFile {
        name: "config.json",
        bytes: CONFIG_BYTES as u64,
        sha256: [
            0x7f, 0x42, 0xf5, 0x59, 0xa6, 0xa6, 0x9f, 0xfa, 0xea, 0xf6, 0xb6, 0x1a, 0x1e, 0xce,
            0x3a, 0x56, 0x2a, 0x2e, 0xd5, 0xad, 0x00, 0xb8, 0xd3, 0x0f, 0x16, 0x91, 0x7b, 0xa5,
            0xab, 0x1b, 0xcb, 0xe9,
        ],
    },
    ExpectedFile {
        name: "model.safetensors",
        bytes: CONTAINER_BYTES as u64,
        sha256: [
            0x93, 0x68, 0x2e, 0xb1, 0xc9, 0x76, 0x39, 0xd1, 0x8f, 0x00, 0x77, 0x04, 0xdc, 0x88,
            0x0b, 0xd7, 0x4c, 0xbe, 0x53, 0x0a, 0xda, 0xf7, 0xb1, 0xbb, 0x56, 0x12, 0x13, 0x86,
            0x3f, 0xda, 0xd2, 0xa6,
        ],
    },
];

/// Retained exact regular-file reader. The full pinned shard is authenticated
/// before any tensor is delivered. Every later complete range is reverified.
pub struct Source {
    plan: Plan,
    audit: FileAudit,
    ends: Vec<u64>,
}

impl Source {
    pub fn open(directory: &Path, available_bytes: usize) -> Result<Self, Error> {
        if available_bytes < SOURCE_ALLOWANCE_BYTES {
            return Err(Error::Budget {
                required: SOURCE_ALLOWANCE_BYTES,
                limit: available_bytes,
            });
        }
        let mut config = [0; CONFIG_BYTES];
        let mut config_file = open_regular(&directory.join(FILES[0].name))?;
        config_file
            .read_exact(&mut config)
            .map_err(|e| invalid(e.to_string()))?;
        if config_file
            .read(&mut [0])
            .map_err(|e| invalid(e.to_string()))?
            != 0
        {
            return Err(invalid("assistant config exceeds pinned length"));
        }
        let mut shard = open_regular(&directory.join(FILES[1].name))?;
        let bytes = usize::try_from(shard.metadata().map_err(|e| invalid(e.to_string()))?.len())
            .map_err(|_| invalid("assistant container exceeds address width"))?;
        let mut length = [0; 8];
        shard
            .read_exact(&mut length)
            .map_err(|e| invalid(e.to_string()))?;
        if u64::from_le_bytes(length) != HEADER_BYTES as u64 {
            return Err(invalid("assistant header length differs"));
        }
        let mut header = [0; HEADER_BYTES];
        shard
            .read_exact(&mut header)
            .map_err(|e| invalid(e.to_string()))?;
        let plan = inspect(&config, &header, bytes)?;
        let offset = plan.payload_offset();
        let mut ends = vec![offset as u64, (offset + plan.ordering_range().end) as u64];
        ends.extend(
            plan.tensors
                .values()
                .map(|tensor| (offset + tensor.range.end) as u64),
        );
        ends.sort_unstable();
        let manifest = Manifest::new(REPOSITORY, REVISION, &FILES)?;
        let audit = FileAudit::open(directory, &manifest, 1, &ends, SOURCE_ALLOWANCE_BYTES)?;
        Ok(Self { plan, audit, ends })
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// A destination must discard bytes if authenticated delivery fails.
    pub fn visit_bf16(
        &mut self,
        name: &str,
        consume: impl FnMut(&[u8]) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let tensor = self
            .plan
            .tensors
            .get(name)
            .ok_or_else(|| invalid("unknown assistant tensor"))?;
        let index = self.range_index(tensor.range.end)?;
        self.audit.visit(index, consume)
    }

    fn range_index(&self, end: usize) -> Result<usize, Error> {
        self.ends
            .binary_search(&((self.plan.payload_offset() + end) as u64))
            .map_err(|_| invalid("assistant authenticated range differs"))
    }

    /// Admit ordering storage in addition to the retained source reader.
    pub fn ordered_head(&mut self, available_bytes: usize) -> Result<OrderedHead, Error> {
        if available_bytes < ORDERING_ALLOWANCE_BYTES {
            return Err(Error::Budget {
                required: ORDERING_ALLOWANCE_BYTES,
                limit: available_bytes,
            });
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(ORDERING_BYTES)
            .map_err(|_| Error::Allocation)?;
        let index = self.range_index(self.plan.ordering_range().end)?;
        self.audit.visit(index, |chunk| {
            bytes.extend_from_slice(chunk);
            Ok(())
        })?;
        OrderedHead::from_ordering_bytes(&bytes)
    }
}

/// Maps each centroid bucket entry to its canonical vocabulary token ID.
pub struct OrderedHead {
    ordering: Vec<u32>,
}

impl OrderedHead {
    fn from_ordering_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() != ORDERING_BYTES {
            return Err(invalid("assistant token ordering length differs"));
        }
        let mut ordering = Vec::new();
        ordering
            .try_reserve_exact(VOCAB)
            .map_err(|_| Error::Allocation)?;
        let mut seen = Vec::new();
        seen.try_reserve_exact(VOCAB)
            .map_err(|_| Error::Allocation)?;
        seen.resize(VOCAB, false);
        for word in bytes.chunks_exact(size_of::<i64>()) {
            let id = i64::from_le_bytes(word.try_into().expect("complete word"));
            let index = usize::try_from(id).map_err(|_| invalid("negative assistant token ID"))?;
            let marker = seen
                .get_mut(index)
                .ok_or_else(|| invalid("assistant token ID outside vocabulary"))?;
            if std::mem::replace(marker, true) {
                return Err(invalid("duplicate assistant token ID"));
            }
            ordering.push(index as u32);
        }
        Ok(Self { ordering })
    }

    /// Fill canonical IDs for the 32 selected centroids, 128 IDs per centroid.
    /// Equal finite scores prefer the lower centroid index. Failed validation
    /// leaves the caller's output untouched.
    pub fn candidate_ids(&self, scores: &[f32], output: &mut [u32]) -> Result<(), Error> {
        if scores.len() != CENTROIDS
            || output.len() != CANDIDATES
            || scores.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "assistant centroid score or candidate shape differs",
            ));
        }
        let mut indices: [usize; CENTROIDS] = std::array::from_fn(|index| index);
        let compare = |&a: &usize, &b: &usize| {
            scores[b]
                .partial_cmp(&scores[a])
                .expect("finite centroid scores")
                .then(a.cmp(&b))
        };
        let (chosen, _, _) = indices.select_nth_unstable_by(TOP_CENTROIDS, compare);
        chosen.sort_unstable_by(compare);
        for (&index, output) in chosen.iter().zip(output.chunks_exact_mut(IDS_PER_CENTROID)) {
            let start = index * IDS_PER_CENTROID;
            output.copy_from_slice(&self.ordering[start..start + IDS_PER_CENTROID]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_permutation_and_deterministic_centroid_candidates() {
        let mut bytes: Vec<u8> = (0..VOCAB as i64).rev().flat_map(i64::to_le_bytes).collect();
        let head = OrderedHead::from_ordering_bytes(&bytes).unwrap();
        let mut scores = [0_f32; CENTROIDS];
        scores[5] = 1.;
        scores[100] = 1.;
        scores[CENTROIDS - 1] = 2.;
        let mut output = [u32::MAX; CANDIDATES];
        head.candidate_ids(&scores, &mut output).unwrap();
        let chosen = [CENTROIDS - 1, 5, 100]
            .into_iter()
            .chain((0..CENTROIDS).filter(|&i| i != 5 && i != 100 && i != CENTROIDS - 1))
            .take(TOP_CENTROIDS);
        let expected: Vec<u32> = chosen
            .flat_map(|i| {
                (i * IDS_PER_CENTROID..(i + 1) * IDS_PER_CENTROID)
                    .map(|row| (VOCAB - 1 - row) as u32)
            })
            .collect();
        assert_eq!(output.as_slice(), expected.as_slice());
        scores[19] = f32::NAN;
        assert!(head.candidate_ids(&scores, &mut output).is_err());
        assert_eq!(output.as_slice(), expected.as_slice());
        assert!(OrderedHead::from_ordering_bytes(&bytes[..bytes.len() - 1]).is_err());
        for invalid in [VOCAB as i64, -1, 0] {
            bytes[..8].copy_from_slice(&invalid.to_le_bytes());
            assert!(OrderedHead::from_ordering_bytes(&bytes).is_err());
        }
    }

    #[test]
    #[ignore = "explicit pinned official config and safetensors header"]
    fn official_assistant_cold_metadata() {
        let path = std::env::var_os("TOKLET_GEMMA_ASSISTANT_METADATA").expect("pinned metadata");
        let directory = Path::new(&path);
        let mut config = std::fs::read(directory.join("config.json")).unwrap();
        let mut header = std::fs::read(directory.join("header.json")).unwrap();
        let plan = inspect(&config, &header, CONTAINER_BYTES).unwrap();
        assert_eq!(plan.tensors().len(), 49);
        assert_eq!(
            plan.tensors()["model.embed_tokens.weight"].shape,
            [VOCAB, 256]
        );
        assert_eq!(plan.tensors()["pre_projection.weight"].shape, [256, 3072]);
        assert_eq!(plan.tensors()["post_projection.weight"].shape, [1536, 256]);
        assert_eq!(plan.ordering_range().len(), ORDERING_BYTES);
        assert!(inspect(&config, &header, CONTAINER_BYTES - 1).is_err());
        header[0] ^= 1;
        assert!(inspect(&config, &header, CONTAINER_BYTES).is_err());
        header[0] ^= 1;
        config[0] ^= 1;
        assert!(inspect(&config, &header, CONTAINER_BYTES).is_err());
    }

    #[test]
    #[ignore = "explicit complete official assistant artifact and authenticated range delivery"]
    fn official_assistant_metadata_and_ordering() {
        let path = std::env::var_os("TOKLET_GEMMA_ASSISTANT_DIRECTORY")
            .expect("verified assistant source");
        let directory = Path::new(&path);
        assert!(matches!(
            Source::open(directory, SOURCE_ALLOWANCE_BYTES - 1),
            Err(Error::Budget { .. })
        ));
        let mut source = Source::open(directory, SOURCE_ALLOWANCE_BYTES).unwrap();
        let plan = source.plan();
        assert_eq!(plan.tensors().len(), 49);
        assert_eq!(
            plan.tensors()["model.embed_tokens.weight"].shape,
            [VOCAB, 256]
        );
        assert_eq!(plan.tensors()["pre_projection.weight"].shape, [256, 3072]);
        assert_eq!(plan.tensors()["post_projection.weight"].shape, [1536, 256]);
        assert_eq!(plan.ordering_range().len(), ORDERING_BYTES);
        assert!(matches!(
            source.ordered_head(ORDERING_ALLOWANCE_BYTES - 1),
            Err(Error::Budget { .. })
        ));
        let head = source.ordered_head(ORDERING_ALLOWANCE_BYTES).unwrap();
        let mut ids = [0; CANDIDATES];
        let scores = std::array::from_fn::<_, CENTROIDS, _>(|i| ((i * 73) % 31) as f32 - 15.);
        head.candidate_ids(&scores, &mut ids).unwrap();
        let mut ordering_sha = Sha256::new();
        for id in ids {
            ordering_sha.update(id.to_le_bytes());
        }
        assert_eq!(
            format!("{:x}", ordering_sha.finalize()),
            "6d0f784e3eaee5dada1224fc63b5b2312ad8e3beb1f7adb6cbbc35dc61ad8df7"
        );
        assert!(ids.iter().all(|&id| id < VOCAB as u32));
        assert_eq!(
            ids.iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            CANDIDATES
        );
        let mut delivered = 0;
        let mut centroid_sha = Sha256::new();
        source
            .visit_bf16("masked_embedding.centroids.weight", |bytes| {
                delivered += bytes.len();
                centroid_sha.update(bytes);
                Ok(())
            })
            .unwrap();
        assert_eq!(delivered, 2 * CENTROIDS * 256);
        assert_eq!(
            format!("{:x}", centroid_sha.finalize()),
            "481fc5c561b2ac12f0f971d66a631d1ab260c62b9785a638e091a73679786054"
        );
        println!(
            "OFFICIAL_GEMMA_ASSISTANT_META {}",
            serde_json::json!({
                "container_bytes":CONTAINER_BYTES,"tensors":50,"ordering":VOCAB,"centroids":CENTROIDS,
                "candidates":CANDIDATES,"centroid_bytes":delivered
            })
        );
    }
}
