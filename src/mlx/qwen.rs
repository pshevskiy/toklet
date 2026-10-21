//! Validated local MLX Qwen2 artifact source, without full-model expansion.
//!
//! Opening validates config, index and tensor metadata. Payload reads check
//! floating values/affine metadata; no GPU allocation or model execution occurs.

use super::affine::{Layout, Matrix, MetadataDtype};
use crate::{
    model::Config,
    weights::{self, Error, LoadOptions},
};
use safetensors::Dtype;
use serde::{
    Deserialize,
    de::{self, MapAccess, SeqAccess, Visitor},
};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::File,
    io::Read,
    path::Path,
};

const MAX_CONFIG_BYTES: usize = 1024 * 1024;
const METADATA_ALLOWANCE: usize = 64 * 1024 * 1024;
const STREAM_BYTES: usize = 64 * 1024;

fn invalid(reason: impl Into<String>) -> Error {
    Error::Invalid(reason.into())
}
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b)
        .ok_or_else(|| invalid("MLX artifact size overflow"))
}

/// A floating tensor retains its source precision; affine matrices retain all
/// three original packed buffers. This is an artifact plan, not a GPU layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Storage {
    Float(MetadataDtype),
    Affine(Layout),
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    shape: Vec<usize>,
    storage: Storage,
    payload_bytes: usize,
}
impl TensorSpec {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn storage(&self) -> Storage {
        self.storage
    }
    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Quantization {
    bits: u8,
    group_size: usize,
}
impl Quantization {
    fn parse(value: &Map<String, Value>, defaults: bool) -> Result<Self, Error> {
        for name in value.keys() {
            if !["bits", "group_size", "mode"].contains(&name.as_str()) && !defaults {
                return Err(invalid(format!(
                    "unsupported MLX quantization option: {name}"
                )));
            }
        }
        if value
            .get("mode")
            .is_some_and(|v| v.as_str() != Some("affine"))
        {
            return Err(invalid("only MLX affine quantization is supported"));
        }
        let bits = value
            .get("bits")
            .and_then(Value::as_u64)
            .and_then(|v| u8::try_from(v).ok())
            .ok_or_else(|| invalid("MLX bits must be an integer"))?;
        let group_size = value
            .get("group_size")
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| invalid("MLX group_size must be an integer"))?;
        Layout::new(1, 128, bits, group_size, MetadataDtype::F16)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(Self { bits, group_size })
    }
}
struct QuantizationPlan {
    default: Quantization,
    overrides: BTreeMap<String, Option<Quantization>>,
}
impl QuantizationPlan {
    fn parse(value: Value) -> Result<Self, Error> {
        let object = value
            .as_object()
            .ok_or_else(|| invalid("MLX quantization must be an object"))?;
        let default = Quantization::parse(object, true)?;
        let mut overrides = BTreeMap::new();
        for (name, value) in object {
            if ["bits", "group_size", "mode"].contains(&name.as_str()) {
                continue;
            }
            let quantization = if value == &Value::Bool(false) {
                None
            } else {
                Some(Quantization::parse(
                    value
                        .as_object()
                        .ok_or_else(|| invalid(format!("invalid MLX layer instruction: {name}")))?,
                    false,
                )?)
            };
            overrides.insert(name.clone(), quantization);
        }
        Ok(Self { default, overrides })
    }
    fn resolve(&self, module: &str) -> Option<Quantization> {
        self.overrides
            .get(module)
            .copied()
            .unwrap_or(Some(self.default))
    }
}

/// Immutable metadata and open shard handles. Tensor-name lookups belong to
/// loading only; execution should retain resolved concrete buffers/operations.
pub struct Source {
    config: Config,
    tensors: BTreeMap<String, TensorSpec>,
    source: weights::Source,
    tensor_payload_bytes: usize,
    admitted_load_bytes: usize,
}
impl Source {
    /// No weights are expanded/allocated. Admit source-precision tensor payloads
    /// plus a 64 MiB metadata estimate and 64 KiB shared stream buffer. This does
    /// not admit a session, GPU padding/conversion, tokenizer or application.
    /// A redundant tied head must be representation- and byte-identical; it is
    /// streamed for verification but omitted from owned tensor payload totals.
    pub fn open(directory: &Path, options: LoadOptions) -> Result<Self, Error> {
        if options.memory_limit_bytes == 0
            || options.memory_limit_bytes > weights::INITIAL_MEMORY_LIMIT_BYTES
        {
            return Err(invalid("MLX loading budget must be within 6 GiB"));
        }
        let mut document = read_config(&directory.join("config.json"))?;
        let object = document
            .as_object_mut()
            .ok_or_else(|| invalid("MLX config must be an object"))?;
        let quantization = object.remove("quantization");
        let alternate = object.remove("quantization_config");
        let quantization = match (quantization, alternate) {
            (Some(a), Some(b)) if a != b => {
                return Err(invalid("conflicting MLX quantization declarations"));
            }
            (Some(a), _) | (_, Some(a)) => a,
            _ => {
                return Err(invalid(
                    "MLX affine artifact requires quantization settings",
                ));
            }
        };
        let plan = QuantizationPlan::parse(quantization)?;
        let config = weights::qwen_config(document, options)?;
        // Bound name-map construction before looping over declared layers.
        let count = config
            .num_layers
            .checked_mul(12)
            .and_then(|v| v.checked_add(2))
            .ok_or_else(|| invalid("MLX tensor count overflow"))?;
        if count > 8192 {
            return Err(invalid(
                "MLX Qwen loader supports at most 8192 logical tensors",
            ));
        }
        let mut expected = weights::expected(&config)?;
        let source = weights::Source::open(directory)?;
        let has_head = source.info("lm_head.weight").is_some();
        if has_head {
            expected.insert(
                "lm_head.weight".into(),
                vec![config.vocab_size, config.hidden_size],
            );
        }
        let modules: BTreeSet<_> = expected
            .iter()
            .filter(|(_, shape)| shape.len() == 2)
            .map(|(name, _)| name.strip_suffix(".weight").unwrap())
            .collect();
        for name in plan.overrides.keys() {
            if !modules.contains(name.as_str()) {
                return Err(invalid(format!("unknown MLX quantized module: {name}")));
            }
        }
        let mut physical = BTreeSet::new();
        let mut tensors = BTreeMap::new();
        let mut tensor_payload_bytes = 0;
        for (name, shape) in &expected {
            let info = source
                .info(name)
                .ok_or_else(|| invalid(format!("missing tensor: {name}")))?;
            physical.insert(name.clone());
            let module = name.strip_suffix(".weight");
            let scales = module.map(|p| format!("{p}.scales"));
            let packed =
                shape.len() == 2 && scales.as_ref().is_some_and(|p| source.info(p).is_some());
            let storage = if packed {
                let module = module.unwrap();
                let quantization = plan.resolve(module).ok_or_else(|| {
                    invalid(format!("excluded MLX module has packed tensors: {module}"))
                })?;
                let scales = scales.as_ref().unwrap();
                let biases = format!("{module}.biases");
                let scale_info = source.info(scales).unwrap();
                let bias_info = source
                    .info(&biases)
                    .ok_or_else(|| invalid(format!("missing tensor: {biases}")))?;
                let dtype = metadata_dtype(scale_info.dtype)?;
                let layout = Layout::new(
                    shape[0],
                    shape[1],
                    quantization.bits,
                    quantization.group_size,
                    dtype,
                )
                .map_err(|e| invalid(format!("{name}: {e}")))?;
                let metadata_shape = [shape[0], shape[1] / layout.group_size()];
                if info.dtype != Dtype::U32
                    || info.shape != [shape[0], layout.row_bytes() / 4]
                    || scale_info.shape != metadata_shape
                    || bias_info.shape != metadata_shape
                    || bias_info.dtype != scale_info.dtype
                {
                    return Err(invalid(format!(
                        "MLX packed tensor shape/dtype mismatch: {name}"
                    )));
                }
                physical.insert(scales.clone());
                physical.insert(biases);
                Storage::Affine(layout)
            } else {
                if &info.shape != shape {
                    return Err(invalid(format!("tensor shape mismatch: {name}")));
                }
                Storage::Float(metadata_dtype(info.dtype)?)
            };
            let payload_bytes = match storage {
                Storage::Affine(layout) => layout.total_bytes(),
                Storage::Float(_) => info.data_offsets.1 - info.data_offsets.0,
            };
            if name != "lm_head.weight" {
                tensor_payload_bytes = add(tensor_payload_bytes, payload_bytes)?;
            }
            tensors.insert(
                name.clone(),
                TensorSpec {
                    shape: shape.clone(),
                    storage,
                    payload_bytes,
                },
            );
        }
        if source.names().any(|name| !physical.contains(name)) {
            return Err(invalid("unexpected tensor in MLX Qwen artifact"));
        }
        let admitted_load_bytes =
            add(add(tensor_payload_bytes, METADATA_ALLOWANCE)?, STREAM_BYTES)?;
        if admitted_load_bytes > options.memory_limit_bytes {
            return Err(Error::Budget {
                required: admitted_load_bytes,
                limit: options.memory_limit_bytes,
            });
        }
        let mut result = Self {
            config,
            tensors,
            source,
            tensor_payload_bytes,
            admitted_load_bytes,
        };
        if has_head {
            result.check_tied_output()?;
            result.tensors.remove("lm_head.weight");
        }
        Ok(result)
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn tensors(&self) -> &BTreeMap<String, TensorSpec> {
        &self.tensors
    }
    /// Original payloads counted once, without a second tied output table.
    pub fn tensor_payload_bytes(&self) -> usize {
        self.tensor_payload_bytes
    }
    pub fn admitted_load_bytes(&self) -> usize {
        self.admitted_load_bytes
    }

    /// Stream decoded floating values. A failing read or consumer may already
    /// have written a partial destination; discard it before publishing a model.
    pub fn visit_float(
        &mut self,
        name: &str,
        consume: impl FnMut(f32) -> Result<(), Error>,
    ) -> Result<(), Error> {
        if !matches!(
            self.tensors.get(name).map(|s| s.storage),
            Some(Storage::Float(_))
        ) {
            return Err(invalid("MLX tensor is not a planned floating tensor"));
        }
        self.source.visit(name, consume)
    }
    /// Read directly into three caller-admitted packed allocations, then check
    /// finite affine metadata. Buffers are partial/unpublished on any error.
    pub fn read_affine(
        &mut self,
        name: &str,
        weights: &mut [u8],
        scales: &mut [u8],
        biases: &mut [u8],
    ) -> Result<Layout, Error> {
        let Some(Storage::Affine(layout)) = self.tensors.get(name).map(|s| s.storage) else {
            return Err(invalid("MLX tensor is not a planned affine tensor"));
        };
        if weights.len() != layout.weight_bytes()
            || scales.len() != layout.metadata_bytes()
            || biases.len() != layout.metadata_bytes()
        {
            return Err(invalid("MLX destination sizes do not match planned tensor"));
        }
        let module = name.strip_suffix(".weight").unwrap();
        self.read_raw(name, weights)?;
        self.read_raw(&format!("{module}.scales"), scales)?;
        self.read_raw(&format!("{module}.biases"), biases)?;
        Matrix::from_le_bytes(layout, weights, scales, biases)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(layout)
    }
    fn read_raw(&mut self, name: &str, destination: &mut [u8]) -> Result<(), Error> {
        let mut offset = 0;
        self.source.visit_raw(name, |bytes| {
            let end = add(offset, bytes.len())?;
            let target = destination
                .get_mut(offset..end)
                .ok_or_else(|| invalid("MLX raw read size mismatch"))?;
            target.copy_from_slice(bytes);
            offset = end;
            Ok(())
        })?;
        if offset != destination.len() {
            return Err(invalid("MLX raw read size mismatch"));
        }
        Ok(())
    }
    fn check_tied_output(&mut self) -> Result<(), Error> {
        if self.tensors["lm_head.weight"] != self.tensors["model.embed_tokens.weight"] {
            return Err(invalid(
                "redundant tied MLX head has a different representation",
            ));
        }
        let suffixes: &[&str] = match self.tensors["lm_head.weight"].storage {
            Storage::Float(_) => &["weight"],
            Storage::Affine(_) => &["weight", "scales", "biases"],
        };
        for suffix in suffixes {
            let mut digest = |name: &str| -> Result<[u8; 32], Error> {
                let mut hash = Sha256::new();
                self.source.visit_raw(name, |bytes| {
                    hash.update(bytes);
                    Ok(())
                })?;
                Ok(hash.finalize().into())
            };
            if digest(&format!("model.embed_tokens.{suffix}"))?
                != digest(&format!("lm_head.{suffix}"))?
            {
                return Err(invalid(
                    "redundant tied MLX head differs from embedding bytes",
                ));
            }
        }
        Ok(())
    }
}

fn metadata_dtype(dtype: Dtype) -> Result<MetadataDtype, Error> {
    match dtype {
        Dtype::F16 => Ok(MetadataDtype::F16),
        Dtype::BF16 => Ok(MetadataDtype::BF16),
        Dtype::F32 => Ok(MetadataDtype::F32),
        _ => Err(invalid(format!(
            "unsupported MLX floating dtype: {dtype:?}"
        ))),
    }
}

// Validate nested uniqueness before serde_json::Value can erase duplicate
// quantization instructions. The ordinary parser retains exact integer syntax.
struct Unique;
impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Check;
        impl<'de> Visitor<'de> for Check {
            type Value = Unique;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("JSON with unique object keys")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Unique, M::Error> {
                let mut keys = BTreeSet::new();
                while let Some(key) = map.next_key::<String>()? {
                    if !keys.insert(key) {
                        return Err(de::Error::custom("duplicate MLX config key"));
                    }
                    map.next_value::<Unique>()?;
                }
                Ok(Unique)
            }
            fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> Result<Unique, S::Error> {
                while seq.next_element::<Unique>()?.is_some() {}
                Ok(Unique)
            }
            fn visit_bool<E: de::Error>(self, _: bool) -> Result<Unique, E> {
                Ok(Unique)
            }
            fn visit_i64<E: de::Error>(self, _: i64) -> Result<Unique, E> {
                Ok(Unique)
            }
            fn visit_u64<E: de::Error>(self, _: u64) -> Result<Unique, E> {
                Ok(Unique)
            }
            fn visit_f64<E: de::Error>(self, _: f64) -> Result<Unique, E> {
                Ok(Unique)
            }
            fn visit_str<E: de::Error>(self, _: &str) -> Result<Unique, E> {
                Ok(Unique)
            }
            fn visit_unit<E: de::Error>(self) -> Result<Unique, E> {
                Ok(Unique)
            }
        }
        d.deserialize_any(Check)
    }
}
fn read_config(path: &Path) -> Result<Value, Error> {
    let io = |source| Error::Io {
        path: path.to_owned(),
        source,
    };
    let file = File::open(path).map_err(io)?;
    if file.metadata().map_err(io)?.len() > MAX_CONFIG_BYTES as u64 {
        return Err(invalid("MLX config exceeds 1 MiB"));
    }
    let mut data = Vec::new();
    // Limit again while reading in case a local file grows after stat.
    file.take((MAX_CONFIG_BYTES + 1) as u64)
        .read_to_end(&mut data)
        .map_err(io)?;
    if data.len() > MAX_CONFIG_BYTES {
        return Err(invalid("MLX config exceeds 1 MiB"));
    }
    serde_json::from_slice::<Unique>(&data)?;
    Ok(serde_json::from_slice(&data)?)
}
