//! Streaming conversion tests exercise no GPU APIs.
use super::*;
use safetensors::{Dtype, tensor::TensorView};
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT: AtomicUsize = AtomicUsize::new(0);
pub(crate) struct Artifact(pub PathBuf);
impl Drop for Artifact {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
pub(crate) fn artifact(dtype: Dtype, tied: Option<f32>, overflow_token: bool) -> Artifact {
    let path = std::env::temp_dir().join(format!(
        "toklet-half-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&path).unwrap();
    let config = serde_json::json!({"model_type":"qwen2", "architectures":["Qwen2ForCausalLM"],"hidden_act":"silu","hidden_size":32,"intermediate_size":64,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"vocab_size":64,"max_position_embeddings":2048,"rms_norm_eps":1e-6,"rope_theta":1e6,"tie_word_embeddings":true});
    fs::write(
        path.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let c: QwenConfig = serde_json::from_value(config).unwrap();
    let c = c.validate(LoadOptions::default()).unwrap();
    let mut tensors = BTreeMap::new();
    for (name, shape) in expected(&c).unwrap() {
        let count: usize = shape.iter().product();
        let bytes: Vec<u8> = (0..count)
            .flat_map(|i| {
                let value: f32 = if overflow_token && name.ends_with("layernorm.weight") {
                    65504.0
                } else if overflow_token && name == "model.embed_tokens.weight" {
                    if i / 32 == 1 { 1.0 } else { 0.0 }
                } else if overflow_token && name.ends_with(".bias") {
                    0.0
                } else {
                    1.0
                };
                match dtype {
                    Dtype::F32 => value.to_le_bytes().to_vec(),
                    Dtype::F16 => half::f16::from_f32(value).to_bits().to_le_bytes().to_vec(),
                    Dtype::BF16 => half::bf16::from_f32(value).to_bits().to_le_bytes().to_vec(),
                    _ => unreachable!(),
                }
            })
            .collect();
        tensors.insert(name, (shape, bytes));
    }
    if let Some(value) = tied {
        tensors.insert(
            "lm_head.weight".into(),
            (vec![64, 32], value.to_le_bytes().repeat(64 * 32)),
        );
    }
    let views: Vec<_> = tensors
        .iter()
        .map(|(name, (shape, bytes))| {
            (
                name.as_str(),
                TensorView::new(dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, None, &path.join("model.safetensors")).unwrap();
    Artifact(path)
}
#[test]
fn half_source_converts_f32_f16_bf16_without_gpu() {
    for dtype in [Dtype::F32, Dtype::F16, Dtype::BF16] {
        let a = artifact(dtype, None, false);
        let mut source = HalfSource::open(&a.0, LoadOptions::default()).unwrap();
        let mut n = 0;
        source
            .visit_half("model.embed_tokens.weight", |bits| {
                assert_eq!(bits, half::f16::ONE.to_bits());
                n += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(n, 64 * 32);
        source.check_tied_output().unwrap();
    }
}
#[test]
fn tied_output_checks_original_values_before_half_rounding() {
    let a = artifact(Dtype::F32, Some(1.0001), false);
    let mut source = HalfSource::open(&a.0, LoadOptions::default()).unwrap();
    source
        .visit_half("model.embed_tokens.weight", |_| Ok(()))
        .unwrap();
    assert_eq!(half::f16::from_f32(1.0001), half::f16::ONE);
    assert!(source.check_tied_output().is_err());
}
