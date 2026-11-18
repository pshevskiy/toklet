//! Validated text geometry and visit schedule for the fixed SenseNova Loop-9B text profiles.
//!
//! Physical layers own weights. A visit owns the distinct state used by that
//! execution of the layer, including both convolution and recurrent state for
//! linear attention and both K and V for full attention.

use super::Error;
use serde::Deserialize;

pub mod mhc;
pub mod rotary;
pub mod state;
pub mod text;

pub const OFFICIAL_AWQ_REVISION: &str = "9250902822d85911d23781f0ddad9d7601f5033a";
pub const MLX_REVISION: &str = "b9a36a9750f28b7a0916dd403a87d1f79a02e1cc";
pub const PHYSICAL_LAYERS: usize = 32;
pub const EXECUTION_VISITS: usize = 56;
const LOOP_START: usize = 4;
const LOOP_END: usize = 28;
const LOOP_PASSES: usize = 2;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Attention {
    LinearAttention,
    FullAttention,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualStreams {
    Plain,
    Expand,
    Carry,
    Reduce,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Visit {
    pub physical_layer: usize,
    pub state_slot: usize,
    pub attention: Attention,
    pub residual_streams: ResidualStreams,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct TextShape {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
}

const TEXT: TextShape = TextShape {
    hidden_size: 4096,
    intermediate_size: 12288,
    vocab_size: 248320,
    head_dim: 256,
    num_attention_heads: 16,
    num_key_value_heads: 4,
    linear_conv_kernel_dim: 4,
    linear_key_head_dim: 128,
    linear_num_key_heads: 16,
    linear_value_head_dim: 128,
    linear_num_value_heads: 32,
    max_position_embeddings: 262144,
    rms_norm_eps: 1e-6,
};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct Yarn {
    pub rope_theta: f64,
    pub factor: f64,
    pub original_max_position_embeddings: usize,
    pub partial_rotary_factor: f64,
    pub mrope_interleaved: bool,
    pub mrope_section: [usize; 3],
}

const YARN: Yarn = Yarn {
    rope_theta: 10_000_000.0,
    factor: 2.0,
    original_max_position_embeddings: 262144,
    partial_rotary_factor: 0.25,
    mrope_interleaved: true,
    mrope_section: [11, 11, 10],
};

#[derive(Deserialize)]
struct PublishedConfig {
    model_type: String,
    architectures: [String; 1],
    hidden_size: usize,
    tie_word_embeddings: bool,
    use_cache: bool,
    loop_settings: LoopSettings,
    mhc: Mhc,
    text_config: TextConfig,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LoopSettings {
    start: usize,
    end: usize,
    counts: [usize; 1],
    eval_count: usize,
    use_gate: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Mhc {
    start: usize,
    end: usize,
    streams: usize,
    iterations: usize,
    fixed_identity: bool,
    pre_scale: f32,
}

#[derive(Deserialize)]
// Direct numeric fields keep serde_json's arbitrary-precision number representation
// out of serde's flatten buffer.
struct TextConfig {
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    head_dim: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_value_head_dim: usize,
    linear_num_value_heads: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    model_type: String,
    hidden_act: String,
    num_hidden_layers: usize,
    full_attention_interval: usize,
    layer_types: [Attention; PHYSICAL_LAYERS],
    attention_bias: bool,
    attention_dropout: f32,
    attn_output_gate: bool,
    mamba_ssm_dtype: String,
    partial_rotary_factor: f64,
    tie_word_embeddings: bool,
    #[serde(rename = "mlp_only_layers")]
    _mlp_only_layers: [usize; 0],
    mtp_num_hidden_layers: usize,
    mtp_use_dedicated_embeddings: bool,
    rope_parameters: RotaryConfig,
    use_cache: bool,
}

#[derive(Deserialize)]
struct RotaryConfig {
    rope_theta: f64,
    factor: f64,
    original_max_position_embeddings: usize,
    partial_rotary_factor: f64,
    mrope_interleaved: bool,
    mrope_section: [usize; 3],
    rope_type: String,
}

impl TextConfig {
    fn shape(&self) -> TextShape {
        TextShape {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            head_dim: self.head_dim,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            linear_conv_kernel_dim: self.linear_conv_kernel_dim,
            linear_key_head_dim: self.linear_key_head_dim,
            linear_num_key_heads: self.linear_num_key_heads,
            linear_value_head_dim: self.linear_value_head_dim,
            linear_num_value_heads: self.linear_num_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
        }
    }
}

impl RotaryConfig {
    fn parameters(&self) -> Yarn {
        Yarn {
            rope_theta: self.rope_theta,
            factor: self.factor,
            original_max_position_embeddings: self.original_max_position_embeddings,
            partial_rotary_factor: self.partial_rotary_factor,
            mrope_interleaved: self.mrope_interleaved,
            mrope_section: self.mrope_section,
        }
    }
}

#[derive(Clone, Copy)]
enum Profile {
    OfficialAwq,
    Mlx,
}

pub struct Plan {
    visits: [Visit; EXECUTION_VISITS],
    text: TextShape,
    use_cache: bool,
}

impl Plan {
    /// Admit the published official AWQ text geometry from bounded JSON bytes.
    /// Weight format admission and checkpoint identity are separate concerns.
    pub fn from_official_awq_config(bytes: &[u8]) -> Result<Self, Error> {
        Self::parse(bytes, Profile::OfficialAwq)
    }

    /// Admit text geometry from the published MLX conversion profile.
    /// The MLX artifact inspector authenticates the model and weight bytes.
    pub fn from_mlx_config(bytes: &[u8]) -> Result<Self, Error> {
        Self::parse(bytes, Profile::Mlx)
    }

    fn parse(bytes: &[u8], profile: Profile) -> Result<Self, Error> {
        let (
            architecture,
            max_positions,
            default_cache,
            training_gate,
            architecture_error,
            json_error,
        ) = match profile {
            Profile::OfficialAwq => (
                "Qwen3_5LoopForConditionalGeneration",
                262144,
                false,
                Some(false),
                "SenseNova official AWQ architecture",
                "SenseNova official AWQ configuration JSON",
            ),
            Profile::Mlx => (
                "SenseNovaLoopForConditionalGeneration",
                524288,
                true,
                None,
                "SenseNova MLX architecture",
                "SenseNova MLX configuration JSON",
            ),
        };
        if bytes.len() > 64 * 1024 {
            return Err(Error::InvalidConfig("SenseNova configuration byte bound"));
        }
        let config: PublishedConfig =
            serde_json::from_slice(bytes).map_err(|_| Error::InvalidConfig(json_error))?;
        if config.model_type != "sensenova_loop"
            || config.architectures != [architecture]
            || config.hidden_size != TEXT.hidden_size
            || config.tie_word_embeddings
        {
            return Err(Error::InvalidConfig(architecture_error));
        }
        let loop_settings = config.loop_settings;
        if (loop_settings.start, loop_settings.end) != (LOOP_START, LOOP_END)
            || loop_settings.counts != [LOOP_PASSES]
            || loop_settings.eval_count != LOOP_PASSES
            || loop_settings.use_gate != training_gate
        {
            return Err(Error::InvalidConfig("SenseNova fixed decoder loop"));
        }
        let mhc = config.mhc;
        if (mhc.start, mhc.end) != (LOOP_START, LOOP_END)
            || mhc.streams != 2
            || mhc.iterations != 20
            || mhc.fixed_identity
            || mhc.pre_scale != 1.0
        {
            return Err(Error::InvalidConfig("SenseNova two-stream mHC routing"));
        }
        let text = config.text_config;
        let shape = text.shape();
        if config.use_cache != default_cache || text.use_cache != default_cache {
            return Err(Error::InvalidConfig("SenseNova text cache default"));
        }
        if shape
            != (TextShape {
                max_position_embeddings: max_positions,
                ..TEXT
            })
            || text.model_type != "qwen3_5_text"
            || text.hidden_act != "silu"
            || text.num_hidden_layers != PHYSICAL_LAYERS
            || text.full_attention_interval != 4
            || text.attention_bias
            || text.attention_dropout != 0.0
            || !text.attn_output_gate
            || text.mamba_ssm_dtype != "float32"
            || text.partial_rotary_factor != YARN.partial_rotary_factor
            || text.tie_word_embeddings
            || text.mtp_num_hidden_layers != 0
            || text.mtp_use_dedicated_embeddings
        {
            return Err(Error::InvalidConfig("SenseNova text backbone geometry"));
        }
        if text.rope_parameters.rope_type != "yarn" || text.rope_parameters.parameters() != YARN {
            return Err(Error::InvalidConfig("SenseNova text YaRN positions"));
        }
        for (layer, &attention) in text.layer_types.iter().enumerate() {
            let expected = if layer % 4 == 3 {
                Attention::FullAttention
            } else {
                Attention::LinearAttention
            };
            if attention != expected {
                return Err(Error::InvalidConfig("SenseNova physical attention types"));
            }
        }
        let loop_width = LOOP_END - LOOP_START;
        let loop_visits_end = LOOP_START + LOOP_PASSES * loop_width;
        let visits = std::array::from_fn(|state_slot| {
            let physical_layer = match state_slot {
                0..LOOP_START => state_slot,
                slot if slot < loop_visits_end => LOOP_START + (slot - LOOP_START) % loop_width,
                slot => LOOP_END + slot - loop_visits_end,
            };
            let residual_streams = match physical_layer {
                LOOP_START => ResidualStreams::Expand,
                layer if layer == LOOP_END - 1 => ResidualStreams::Reduce,
                layer if (LOOP_START..LOOP_END).contains(&layer) => ResidualStreams::Carry,
                _ => ResidualStreams::Plain,
            };
            Visit {
                physical_layer,
                state_slot,
                attention: text.layer_types[physical_layer],
                residual_streams,
            }
        });
        Ok(Self {
            visits,
            text: shape,
            use_cache: default_cache,
        })
    }

    pub fn visits(&self) -> &[Visit; EXECUTION_VISITS] {
        &self.visits
    }

    pub fn text_shape(&self) -> TextShape {
        self.text
    }

    pub fn default_use_cache(&self) -> bool {
        self.use_cache
    }

    pub fn yarn(&self) -> Yarn {
        YARN
    }
}
