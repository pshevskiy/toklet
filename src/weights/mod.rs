//! Local Qwen2 safetensors import, with bounded streaming dtype conversion.
//!
//! Parses and validates every tensor before numerical loading. FP32, FP16 and
//! BF16 little-endian row-major data become one immutable FP32 model; no full
//! source checkpoint or per-tensor conversion copy is retained.

use crate::{
    cpu,
    model::{self, Config, LayerWeights, Model, Projection, Weights},
};
use half::{bf16, f16};
use safetensors::{Dtype, tensor::Metadata};
use serde::{
    Deserialize,
    de::{self, MapAccess, Visitor},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Component, Path, PathBuf},
    time::Instant,
};

const CONVERSION_BYTES: usize = 64 * 1024;
const MAX_JSON_BYTES: usize = 1024 * 1024;
const MAX_HEADER_BYTES: usize = 16 * 1024 * 1024;
const METADATA_ALLOWANCE_BYTES: usize = 64 * 1024 * 1024;
pub const INITIAL_MEMORY_LIMIT_BYTES: usize = 6 * 1024 * 1024 * 1024;
pub const INITIAL_CONTEXT_LIMIT: usize = 2048;

#[derive(Clone, Copy)]
pub struct LoadOptions {
    pub context_limit: usize,
    /// Admission estimate for weights, conversion buffer and metadata only.
    /// Sessions/application buffers must be admitted separately within 6 GiB.
    pub memory_limit_bytes: usize,
}
impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            context_limit: INITIAL_CONTEXT_LIMIT,
            memory_limit_bytes: INITIAL_MEMORY_LIMIT_BYTES,
        }
    }
}

pub struct LoadReport {
    pub tensor_count: usize,
    pub source_tensor_bytes: usize,
    pub source_dtypes: Vec<String>,
    pub fp32_weights_bytes: usize,
    pub conversion_buffer_bytes: usize,
    pub metadata_allowance_bytes: usize,
    pub admitted_load_bytes: usize,
    pub load_seconds: f64,
    /// Nested within load time: payload decoding, finite checks and writes.
    /// Excludes file reads, seeks, allocation and final model validation.
    pub conversion_seconds: f64,
}
pub struct LoadedModel {
    pub model: Model,
    pub report: LoadReport,
}

#[derive(Debug)]
pub enum Error {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json(serde_json::Error),
    Safetensors(safetensors::SafeTensorError),
    Model(model::Error),
    Invalid(String),
    Budget {
        required: usize,
        limit: usize,
    },
    Allocation,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::Json(error) => write!(f, "invalid artifact JSON: {error}"),
            Self::Safetensors(error) => write!(f, "invalid safetensors metadata: {error}"),
            Self::Model(error) => error.fmt(f),
            Self::Invalid(reason) => f.write_str(reason),
            Self::Budget { required, limit } => write!(
                f,
                "loading requires an estimated {required} bytes, exceeding {limit}"
            ),
            Self::Allocation => write!(f, "unable to allocate model loading buffers"),
        }
    }
}
impl std::error::Error for Error {}
impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}
impl From<safetensors::SafeTensorError> for Error {
    fn from(e: safetensors::SafeTensorError) -> Self {
        Self::Safetensors(e)
    }
}
impl From<model::Error> for Error {
    fn from(e: model::Error) -> Self {
        Self::Model(e)
    }
}
impl From<cpu::Error> for Error {
    fn from(e: cpu::Error) -> Self {
        Self::Model(e.into())
    }
}

fn invalid(reason: impl Into<String>) -> Error {
    Error::Invalid(reason.into())
}
fn io<T>(path: &Path, result: std::io::Result<T>) -> Result<T, Error> {
    result.map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })
}
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(cpu::Error::SizeOverflow.into())
}
fn bytes(count: usize) -> Result<usize, Error> {
    Ok(model::allocation_bytes(count)?)
}

#[derive(Deserialize)]
struct QwenConfig {
    model_type: String,
    architectures: Vec<String>,
    hidden_act: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    tie_word_embeddings: bool,
    #[serde(default)]
    use_sliding_window: bool,
    #[serde(default)]
    attention_dropout: f64,
    #[serde(default)]
    attention_bias: Option<bool>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    quantization_config: Option<serde_json::Value>,
    #[serde(default, alias = "dtype")]
    torch_dtype: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}
impl QwenConfig {
    fn validate(self, options: LoadOptions) -> Result<Config, Error> {
        if self.model_type != "qwen2"
            || self.architectures != ["Qwen2ForCausalLM"]
            || self.hidden_act != "silu"
            || !self.tie_word_embeddings
            || self.use_sliding_window
            || self.attention_dropout != 0.0
            || self.attention_bias == Some(false)
            || self.rope_scaling.is_some()
            || self.quantization_config.is_some()
        {
            return Err(invalid(
                "only tied, unquantized Qwen2 with SiLU, QKV bias, ordinary rotary positions and full causal attention is supported",
            ));
        }
        if self
            .torch_dtype
            .as_deref()
            .is_some_and(|dtype| !["float32", "float16", "bfloat16"].contains(&dtype))
        {
            return Err(invalid("unsupported declared torch dtype"));
        }
        for name in self.extra.keys() {
            if ![
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
                "initializer_range",
                "transformers_version",
                "use_cache",
                "sliding_window",
                "max_window_layers",
                "_name_or_path",
            ]
            .contains(&name.as_str())
            {
                return Err(invalid(format!("unsupported config field: {name}")));
            }
        }
        cpu::nonzero(self.num_attention_heads, "num_attention_heads")?;
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(invalid(
                "hidden_size must be divisible by num_attention_heads",
            ));
        }
        let head_dim = self.hidden_size / self.num_attention_heads;
        if self.head_dim.is_some_and(|dimension| dimension != head_dim) {
            return Err(invalid(
                "declared head_dim disagrees with hidden_size/num_attention_heads",
            ));
        }
        if options.context_limit == 0
            || options.context_limit > INITIAL_CONTEXT_LIMIT
            || options.context_limit > self.max_position_embeddings
        {
            return Err(invalid(
                "context limit must fit model positions and initial 2048-token limit",
            ));
        }
        let config = Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_layers: self.num_hidden_layers,
            query_heads: self.num_attention_heads,
            kv_heads: self.num_key_value_heads,
            head_dim,
            max_positions: options.context_limit,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
        };
        config.validate()?;
        Ok(config)
    }
}

/// Reject duplicate object keys before passing the header to the reused parser.
struct UniqueKeys;
impl<'de> Deserialize<'de> for UniqueKeys {
    fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Keys;
        impl<'de> Visitor<'de> for Keys {
            type Value = UniqueKeys;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an object with unique keys")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                let mut names = BTreeSet::new();
                while let Some(name) = map.next_key::<String>()? {
                    if !names.insert(name) {
                        return Err(de::Error::custom("duplicate tensor name"));
                    }
                    map.next_value::<de::IgnoredAny>()?;
                }
                Ok(UniqueKeys)
            }
        }
        deserializer.deserialize_map(Keys)
    }
}

fn unique_weight_map<'de, D: de::Deserializer<'de>>(
    deserializer: D,
) -> Result<BTreeMap<String, String>, D::Error> {
    struct Entries;
    impl<'de> Visitor<'de> for Entries {
        type Value = BTreeMap<String, String>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("unique tensor-to-shard entries")
        }
        fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
            let mut entries = BTreeMap::new();
            while let Some((name, file)) = map.next_entry::<String, String>()? {
                if entries.insert(name, file).is_some() {
                    return Err(de::Error::custom("duplicate index tensor"));
                }
            }
            Ok(entries)
        }
    }
    deserializer.deserialize_map(Entries)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Index {
    #[serde(deserialize_with = "unique_weight_map")]
    weight_map: BTreeMap<String, String>,
    #[serde(default)]
    metadata: BTreeMap<String, serde_json::Value>,
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, Error> {
    let mut file = io(path, File::open(path))?;
    let count =
        usize::try_from(io(path, file.metadata())?.len()).map_err(|_| cpu::Error::SizeOverflow)?;
    if count > MAX_JSON_BYTES {
        return Err(invalid(format!(
            "{} exceeds 1 MiB JSON limit",
            path.display()
        )));
    }
    let mut data = Vec::new();
    data.try_reserve_exact(count)
        .map_err(|_| Error::Allocation)?;
    data.resize(count, 0);
    io(path, file.read_exact(&mut data))?;
    Ok(serde_json::from_slice(&data)?)
}

struct Shard {
    path: PathBuf,
    file: File,
    metadata: Metadata,
    data_start: u64,
    header_bytes: usize,
}
impl Shard {
    fn open(path: PathBuf) -> Result<Self, Error> {
        let mut file = io(&path, File::open(&path))?;
        let file_bytes = io(&path, file.metadata())?.len();
        let mut length = [0; 8];
        io(&path, file.read_exact(&mut length))?;
        let count =
            usize::try_from(u64::from_le_bytes(length)).map_err(|_| cpu::Error::SizeOverflow)?;
        if count > MAX_HEADER_BYTES {
            return Err(invalid("safetensors header exceeds 16 MiB limit"));
        }
        let data_start = add(count, 8)? as u64;
        if data_start > file_bytes {
            return Err(invalid("truncated safetensors header"));
        }
        let mut header = Vec::new();
        header
            .try_reserve_exact(count)
            .map_err(|_| Error::Allocation)?;
        header.resize(count, 0);
        io(&path, file.read_exact(&mut header))?;
        if header.first() != Some(&b'{') {
            return Err(invalid("safetensors header must start with an object"));
        }
        serde_json::from_slice::<UniqueKeys>(&header)?;
        let metadata: Metadata = serde_json::from_slice(&header)?;
        if data_start.checked_add(metadata.data_len() as u64) != Some(file_bytes) {
            return Err(invalid(
                "safetensors offsets do not cover the complete file",
            ));
        }
        Ok(Self {
            path,
            file,
            metadata,
            data_start,
            header_bytes: count,
        })
    }
}

struct Source {
    shards: Vec<Shard>,
    locations: BTreeMap<String, usize>,
    buffer: Vec<u8>,
    conversion_seconds: f64,
}
impl Source {
    fn open(directory: &Path) -> Result<Self, Error> {
        let index_path = directory.join("model.safetensors.index.json");
        let index = if index_path.try_exists().map_err(|source| Error::Io {
            path: index_path.clone(),
            source,
        })? {
            Some(read_json::<Index>(&index_path)?)
        } else {
            None
        };
        let filenames: BTreeSet<String> = index
            .as_ref()
            .map(|index| index.weight_map.values().cloned().collect())
            .unwrap_or_else(|| BTreeSet::from(["model.safetensors".into()]));
        if filenames.is_empty() {
            return Err(invalid("empty safetensors index"));
        }
        let mut shards = Vec::new();
        let mut locations = BTreeMap::new();
        let mut header_bytes = 0;
        for filename in filenames {
            let mut components = Path::new(&filename).components();
            if !matches!(components.next(), Some(Component::Normal(_)))
                || components.next().is_some()
                || !filename.ends_with(".safetensors")
            {
                return Err(invalid(
                    "shard filenames must be plain local .safetensors filenames",
                ));
            }
            let shard = Shard::open(directory.join(&filename))?;
            header_bytes = add(header_bytes, shard.header_bytes)?;
            if header_bytes > MAX_HEADER_BYTES {
                return Err(invalid("aggregate safetensors headers exceed 16 MiB limit"));
            }
            for name in shard.metadata.offset_keys() {
                if let Some(index) = &index
                    && index.weight_map.get(&name) != Some(&filename)
                {
                    return Err(invalid(format!(
                        "index disagrees with shard for tensor {name}"
                    )));
                }
                if locations.insert(name, shards.len()).is_some() {
                    return Err(invalid("duplicate tensor across shards"));
                }
            }
            shards.push(shard);
        }
        if let Some(index) = index {
            if locations.len() != index.weight_map.len() {
                return Err(invalid("index references absent tensors"));
            }
            if let Some(total) = index.metadata.get("total_size") {
                let actual = shards
                    .iter()
                    .try_fold(0_usize, |sum, shard| add(sum, shard.metadata.data_len()))?;
                if total.as_u64() != Some(actual as u64) {
                    return Err(invalid("index total_size disagrees with tensor bytes"));
                }
            }
        }
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(CONVERSION_BYTES)
            .map_err(|_| Error::Allocation)?;
        buffer.resize(CONVERSION_BYTES, 0);
        Ok(Self {
            shards,
            locations,
            buffer,
            conversion_seconds: 0.0,
        })
    }

    fn validate(&self, expected: &BTreeMap<String, Vec<usize>>) -> Result<(), Error> {
        for (name, shape) in expected {
            let index = self
                .locations
                .get(name)
                .ok_or_else(|| invalid(format!("missing tensor: {name}")))?;
            let info = self.shards[*index].metadata.info(name).unwrap();
            if &info.shape != shape {
                return Err(invalid(format!(
                    "tensor {name} shape {:?}, expected {shape:?}",
                    info.shape
                )));
            }
            if !matches!(info.dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) {
                return Err(invalid(format!(
                    "tensor {name} has unsupported dtype {:?}",
                    info.dtype
                )));
            }
        }
        for name in self.locations.keys() {
            if !expected.contains_key(name) {
                return Err(invalid(format!("unexpected tensor: {name}")));
            }
        }
        Ok(())
    }

    fn visit(
        &mut self,
        name: &str,
        mut consume: impl FnMut(f32) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let index = *self
            .locations
            .get(name)
            .ok_or_else(|| invalid(format!("missing tensor: {name}")))?;
        let shard = &mut self.shards[index];
        let info = shard.metadata.info(name).unwrap();
        let (begin, end) = info.data_offsets;
        io(
            &shard.path,
            shard
                .file
                .seek(SeekFrom::Start(shard.data_start + begin as u64)),
        )?;
        // Select the decoding format once, outside all per-value loops.
        fn stream<const WIDTH: usize>(
            shard: &mut Shard,
            buffer: &mut [u8],
            mut remaining: usize,
            decode: impl Fn([u8; WIDTH]) -> f32,
            consume: &mut impl FnMut(f32) -> Result<(), Error>,
            conversion_seconds: &mut f64,
        ) -> Result<(), Error> {
            while remaining > 0 {
                let count = remaining.min(buffer.len());
                io(&shard.path, shard.file.read_exact(&mut buffer[..count]))?;
                let conversion_start = Instant::now();
                for raw in buffer[..count].chunks_exact(WIDTH) {
                    let value = decode(raw.try_into().unwrap());
                    if !value.is_finite() {
                        return Err(invalid("nonfinite weight value in tensor payload"));
                    }
                    consume(value)?;
                }
                *conversion_seconds += conversion_start.elapsed().as_secs_f64();
                remaining -= count;
            }
            Ok(())
        }
        match info.dtype {
            Dtype::F32 => stream(
                shard,
                &mut self.buffer,
                end - begin,
                f32::from_le_bytes,
                &mut consume,
                &mut self.conversion_seconds,
            ),
            Dtype::F16 => stream(
                shard,
                &mut self.buffer,
                end - begin,
                |raw| f16::from_le_bytes(raw).to_f32(),
                &mut consume,
                &mut self.conversion_seconds,
            ),
            Dtype::BF16 => stream(
                shard,
                &mut self.buffer,
                end - begin,
                |raw| bf16::from_le_bytes(raw).to_f32(),
                &mut consume,
                &mut self.conversion_seconds,
            ),
            _ => Err(invalid("unsupported weight dtype")),
        }
    }

    fn read(&mut self, name: &str, count: usize) -> Result<Vec<f32>, Error> {
        bytes(count)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| Error::Allocation)?;
        self.visit(name, |value| {
            values.push(value);
            Ok(())
        })?;
        if values.len() != count {
            return Err(invalid(format!("tensor {name} element count mismatch")));
        }
        Ok(values)
    }
}

fn expected(c: &Config) -> Result<BTreeMap<String, Vec<usize>>, Error> {
    let h = c.hidden_size;
    let kv = cpu::product(c.kv_heads, c.head_dim)?;
    let mut tensors = BTreeMap::from([
        ("model.embed_tokens.weight".into(), vec![c.vocab_size, h]),
        ("model.norm.weight".into(), vec![h]),
    ]);
    for layer in 0..c.num_layers {
        let prefix = format!("model.layers.{layer}");
        for name in ["input_layernorm", "post_attention_layernorm"] {
            tensors.insert(format!("{prefix}.{name}.weight"), vec![h]);
        }
        for (name, rows) in [("q", h), ("k", kv), ("v", kv)] {
            tensors.insert(
                format!("{prefix}.self_attn.{name}_proj.weight"),
                vec![rows, h],
            );
            tensors.insert(format!("{prefix}.self_attn.{name}_proj.bias"), vec![rows]);
        }
        tensors.insert(format!("{prefix}.self_attn.o_proj.weight"), vec![h, h]);
        for name in ["gate", "up"] {
            tensors.insert(
                format!("{prefix}.mlp.{name}_proj.weight"),
                vec![c.intermediate_size, h],
            );
        }
        tensors.insert(
            format!("{prefix}.mlp.down_proj.weight"),
            vec![h, c.intermediate_size],
        );
    }
    Ok(tensors)
}

/// Load only local artifacts. No download, tokenizer, GPU or generation occurs.
///
/// Configuration and all tensor names/shapes/dtypes/index mappings are validated
/// before allocating FP32 weight storage. Metadata input is bounded (1 MiB JSON,
/// 16 MiB aggregate headers); a 64 MiB metadata allowance is an admission estimate,
/// not an instrumented physical-peak measurement. Conversion uses one 64 KiB
/// buffer for the entire load. An optional tied `lm_head.weight` is checked
/// elementwise against embeddings and discarded without allocating a second table.
pub fn load_qwen(directory: &Path, options: LoadOptions) -> Result<LoadedModel, Error> {
    let start = Instant::now();
    if options.memory_limit_bytes == 0 || options.memory_limit_bytes > INITIAL_MEMORY_LIMIT_BYTES {
        return Err(invalid(
            "loading memory limit must be within the initial 6 GiB budget",
        ));
    }
    let c = read_json::<QwenConfig>(&directory.join("config.json"))?.validate(options)?;
    // Check the complete model-size arithmetic before constructing name maps.
    let h = c.hidden_size;
    let kv = cpu::product(c.kv_heads, c.head_dim)?;
    let hm = cpu::product(h, c.intermediate_size)?;
    let per_layer = [
        h,
        h,
        cpu::product(h, h)?,
        h,
        cpu::product(kv, h)?,
        kv,
        cpu::product(kv, h)?,
        kv,
        cpu::product(h, h)?,
        hm,
        hm,
        hm,
    ]
    .into_iter()
    .try_fold(0, add)?;
    let elements = add(
        add(cpu::product(c.vocab_size, h)?, h)?,
        cpu::product(per_layer, c.num_layers)?,
    )?;
    let fp32_weights_bytes = bytes(elements)?;
    let admitted_load_bytes = add(
        add(fp32_weights_bytes, METADATA_ALLOWANCE_BYTES)?,
        CONVERSION_BYTES,
    )?;
    if admitted_load_bytes > options.memory_limit_bytes {
        return Err(Error::Budget {
            required: admitted_load_bytes,
            limit: options.memory_limit_bytes,
        });
    }
    if add(cpu::product(c.num_layers, 12)?, 2)? > 8192 {
        return Err(invalid("initial loader supports at most 8192 tensors"));
    }
    let mut source = Source::open(directory)?;
    let mut expected = expected(&c)?;
    let has_output = source.locations.contains_key("lm_head.weight");
    if has_output {
        expected.insert("lm_head.weight".into(), vec![c.vocab_size, h]);
    }
    source.validate(&expected)?;
    let source_tensor_bytes = source
        .shards
        .iter()
        .try_fold(0, |sum, shard| add(sum, shard.metadata.data_len()))?;
    let source_dtypes = source
        .shards
        .iter()
        .flat_map(|shard| {
            shard
                .metadata
                .tensors()
                .into_values()
                .map(|info| format!("{:?}", info.dtype))
                .collect::<Vec<_>>()
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let embedding = source.read("model.embed_tokens.weight", cpu::product(c.vocab_size, h)?)?;
    if has_output {
        let mut index = 0;
        source.visit("lm_head.weight", |value| {
            if embedding.get(index) != Some(&value) {
                return Err(invalid("lm_head.weight does not equal tied embeddings"));
            }
            index += 1;
            Ok(())
        })?;
        if index != embedding.len() {
            return Err(invalid("tied output element count mismatch"));
        }
    }
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(c.num_layers)
        .map_err(|_| Error::Allocation)?;
    for layer in 0..c.num_layers {
        let prefix = format!("model.layers.{layer}");
        let mut projection = |name, rows| -> Result<Projection, Error> {
            Ok(Projection {
                weight: source.read(
                    &format!("{prefix}.self_attn.{name}_proj.weight"),
                    cpu::product(rows, h)?,
                )?,
                bias: source.read(&format!("{prefix}.self_attn.{name}_proj.bias"), rows)?,
            })
        };
        let q = projection("q", h)?;
        let k = projection("k", kv)?;
        let v = projection("v", kv)?;
        layers.push(LayerWeights {
            input_norm: source.read(&format!("{prefix}.input_layernorm.weight"), h)?,
            post_attention_norm: source
                .read(&format!("{prefix}.post_attention_layernorm.weight"), h)?,
            q,
            k,
            v,
            o: source.read(
                &format!("{prefix}.self_attn.o_proj.weight"),
                cpu::product(h, h)?,
            )?,
            gate: source.read(&format!("{prefix}.mlp.gate_proj.weight"), hm)?,
            up: source.read(&format!("{prefix}.mlp.up_proj.weight"), hm)?,
            down: source.read(&format!("{prefix}.mlp.down_proj.weight"), hm)?,
        });
    }
    let final_norm = source.read("model.norm.weight", h)?;
    let model = Model::new(
        c,
        Weights {
            embedding,
            layers,
            final_norm,
        },
    )?;
    let report = LoadReport {
        tensor_count: expected.len(),
        source_tensor_bytes,
        source_dtypes,
        fp32_weights_bytes,
        conversion_buffer_bytes: source.buffer.capacity(),
        metadata_allowance_bytes: METADATA_ALLOWANCE_BYTES,
        admitted_load_bytes,
        load_seconds: start.elapsed().as_secs_f64(),
        conversion_seconds: source.conversion_seconds,
    };
    Ok(LoadedModel { model, report })
}

/// Validated streaming source for the Metal Half loader. No full FP32 model is
/// constructed. Tensor names/shapes are resolved before any GPU allocation.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub(crate) struct HalfSource {
    pub config: Config,
    pub shapes: BTreeMap<String, Vec<usize>>,
    source: Source,
    embedding_digest: Option<[u8; 32]>,
}
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
impl HalfSource {
    pub fn open(directory: &Path, options: LoadOptions) -> Result<Self, Error> {
        if options.memory_limit_bytes == 0
            || options.memory_limit_bytes > INITIAL_MEMORY_LIMIT_BYTES
        {
            return Err(invalid("Half loading budget must be within 6 GiB"));
        }
        let config = read_json::<QwenConfig>(&directory.join("config.json"))?.validate(options)?;
        let source = Source::open(directory)?;
        let shapes = expected(&config)?;
        let mut validation = shapes.clone();
        if source.locations.contains_key("lm_head.weight") {
            validation.insert(
                "lm_head.weight".into(),
                vec![config.vocab_size, config.hidden_size],
            );
        }
        source.validate(&validation)?;
        Ok(Self {
            config,
            shapes,
            source,
            embedding_digest: None,
        })
    }
    pub fn conversion_seconds(&self) -> f64 {
        self.source.conversion_seconds
    }
    /// Stream a tensor directly to its admitted Half allocation. Finite source
    /// values are checked by Source::visit; Half overflow is rejected here.
    pub fn visit_half(
        &mut self,
        name: &str,
        mut consume: impl FnMut(u16) -> Result<(), Error>,
    ) -> Result<(), Error> {
        use sha2::{Digest, Sha256};
        let mut digest = Sha256::new();
        let embedding = name == "model.embed_tokens.weight";
        self.source.visit(name, |value| {
            if embedding {
                digest.update(if value == 0.0 { 0.0f32 } else { value }.to_le_bytes());
            }
            let value = f16::from_f32(value);
            if !value.is_finite() {
                return Err(invalid("weight exceeds finite Half range"));
            }
            consume(value.to_bits())
        })?;
        if embedding {
            self.embedding_digest = Some(digest.finalize().into());
        }
        Ok(())
    }
    /// A redundant lm_head must equal the original source embedding values,
    /// not merely their rounded Half representation. No second table is held.
    pub fn check_tied_output(&mut self) -> Result<(), Error> {
        use sha2::{Digest, Sha256};
        if self.source.locations.contains_key("lm_head.weight") {
            let expected = self
                .embedding_digest
                .ok_or_else(|| invalid("embedding must load before tied output check"))?;
            let mut digest = Sha256::new();
            self.source.visit("lm_head.weight", |value| {
                digest.update(if value == 0.0 { 0.0f32 } else { value }.to_le_bytes());
                Ok(())
            })?;
            let actual: [u8; 32] = digest.finalize().into();
            if actual != expected {
                return Err(invalid("lm_head differs from tied source embeddings"));
            }
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub(crate) mod half_tests;
