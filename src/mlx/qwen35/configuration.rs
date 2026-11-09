//! Exact source graph options; declared context and precision are not execution.
use super::*;

pub(super) fn validate(bytes: &[u8]) -> Result<(), Error> {
    let text = text(HIDDEN, INTERMEDIATE, true);
    let expected = json!({
        "architectures":["Qwen3_5ForConditionalGeneration"],"dtype":"bfloat16","eos_token_id":248044,
        "image_token_id":248056,"model_type":"qwen3_5",
        "quantization":{"group_size":64,"bits":4,"mode":"affine"},
        "quantization_config":{"group_size":64,"bits":4,"mode":"affine"},
        "text_config":text,"tie_word_embeddings":true,"transformers_version":"5.2.0","video_token_id":248057,
        "vision_end_token_id":248054,"vision_start_token_id":248053
    });
    if !same_semantics(&super::super::qwen::parse_json(bytes)?, &expected) {
        return Err(invalid("Qwen3.5-4B source graph options differ"));
    }
    Ok(())
}

// The two concrete32-layer sources share graph options, apart from their
// hidden/intermediate widths and tied versus independent output storage.
pub(super) fn text(
    hidden: usize,
    intermediate: usize,
    tied_output: bool,
) -> serde_json::Map<String, Value> {
    let layers: Vec<_> = (0..LAYERS)
        .map(|i| {
            if i % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    let mut text = serde_json::Map::new();
    for part in [
        json!({"attention_bias":false,"attention_dropout":0.0,"attn_output_gate":true,"bos_token_id":null,
            "dtype":"bfloat16","eos_token_id":248044,"full_attention_interval":4,"head_dim":256,
            "hidden_act":"silu","hidden_size":hidden,"initializer_range":0.02,"intermediate_size":intermediate,
            "layer_types":layers}),
        json!({"linear_conv_kernel_dim":4,"linear_key_head_dim":128,"linear_num_key_heads":16,
            "linear_num_value_heads":32,"linear_value_head_dim":128,"mamba_ssm_dtype":"float32",
            "max_position_embeddings":262144,"mlp_only_layers":[],"model_type":"qwen3_5_text",
            "mtp_num_hidden_layers":1,"mtp_use_dedicated_embeddings":false,"num_attention_heads":16,
            "num_hidden_layers":32,"num_key_value_heads":4}),
        json!({"pad_token_id":null,"partial_rotary_factor":0.25,"rms_norm_eps":1e-6,
            "rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],
                "partial_rotary_factor":0.25,"rope_theta":10000000,"type":"default"},
            "tie_word_embeddings":tied_output,"use_cache":true,"vocab_size":248320}),
    ] {
        let Value::Object(part) = part else {
            unreachable!("literal config object")
        };
        text.extend(part);
    }
    text
}
