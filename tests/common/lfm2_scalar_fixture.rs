use super::engine;
use engine::mlx::affine::{Layout, Matrix, MetadataDtype};
use engine::model::{
    Config,
    lfm2::{AttentionWeights, ConvWeights, Kind, LayerWeights, Weights},
};
use half::bf16;
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use sha2::{Digest, Sha256};
const WEIGHTS: &[u8] = include_bytes!("../fixtures/lfm2-scalar-v1/model.safetensors");
const ARRAYS: &[u8] = include_bytes!("../fixtures/lfm2-scalar-v1/arrays.bin");
pub fn metadata() -> Value {
    let raw = include_bytes!("../fixtures/lfm2-scalar-v1/manifest.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "876b55cd158ccc98b71b63fcd3b4c03d06e979fdada58dbca64921e2fe8d736b"
    );
    let m: Value = serde_json::from_slice(raw).unwrap();
    for (name, bytes) in [("model.safetensors", WEIGHTS), ("arrays.bin", ARRAYS)] {
        assert_eq!(m["files"][name]["bytes"], bytes.len());
        assert_eq!(
            m["files"][name]["sha256"],
            format!("{:x}", Sha256::digest(bytes))
        );
    }
    assert_eq!(m["stage_groups"], 252);
    assert_eq!(m["gate"]["atol"].as_f64(), Some(1e-5));
    assert_eq!(m["gate"]["rtol"].as_f64(), Some(1e-4));
    m
}
pub fn values(item: &Value) -> Vec<f32> {
    assert_eq!(item["dtype"], "f32le");
    let start = item["offset"].as_u64().unwrap() as usize;
    let n = item["bytes"].as_u64().unwrap() as usize;
    let raw = &ARRAYS[start..start + n];
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        item["sha256"].as_str().unwrap()
    );
    let elements: usize = item["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .product();
    assert_eq!(n, elements * 4);
    raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn packed(t: &SafeTensors<'static>, name: &str, r: usize, c: usize) -> Matrix<'static> {
    let w = t.tensor(&format!("{name}.weight")).unwrap();
    let s = t.tensor(&format!("{name}.scales")).unwrap();
    let b = t.tensor(&format!("{name}.biases")).unwrap();
    assert_eq!(w.dtype(), Dtype::U32);
    assert_eq!(s.dtype(), Dtype::BF16);
    assert_eq!(b.dtype(), Dtype::BF16);
    assert_eq!(w.shape(), [r, c / 8]);
    assert_eq!(s.shape(), [r, c / 64]);
    assert_eq!(s.shape(), b.shape());
    Matrix::from_le_bytes(
        Layout::new(r, c, 4, 64, MetadataDtype::BF16).unwrap(),
        w.data(),
        s.data(),
        b.data(),
    )
    .unwrap()
}
fn gamma(t: &SafeTensors<'_>, name: &str) -> Vec<bf16> {
    let a = t.tensor(name).unwrap();
    assert_eq!(a.dtype(), Dtype::BF16);
    a.data()
        .chunks_exact(2)
        .map(|v| bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())))
        .collect()
}
pub fn inputs() -> (Config, Weights<'static>) {
    let t = SafeTensors::deserialize(WEIGHTS).unwrap();
    let layers = (0..3)
        .map(|i| {
            let p = format!("model.layers.{i}");
            LayerWeights {
                kind: if i == 1 {
                    Kind::Attention(0)
                } else {
                    Kind::Convolution(i / 2)
                },
                gate: packed(&t, &format!("{p}.feed_forward.w1"), 256, 256),
                up: packed(&t, &format!("{p}.feed_forward.w3"), 256, 256),
                down: packed(&t, &format!("{p}.feed_forward.w2"), 256, 256),
                operator_gamma: gamma(&t, &format!("{p}.operator_norm.weight")),
                ffn_gamma: gamma(&t, &format!("{p}.ffn_norm.weight")),
            }
        })
        .collect();
    let p = "model.layers.1.self_attn";
    let attention = vec![AttentionWeights {
        q: packed(&t, &format!("{p}.q_proj"), 256, 256),
        k: packed(&t, &format!("{p}.k_proj"), 128, 256),
        v: packed(&t, &format!("{p}.v_proj"), 128, 256),
        o: packed(&t, &format!("{p}.out_proj"), 256, 256),
        q_gamma: gamma(&t, &format!("{p}.q_layernorm.weight")),
        k_gamma: gamma(&t, &format!("{p}.k_layernorm.weight")),
    }];
    let convolution = [0, 2]
        .into_iter()
        .map(|i| {
            let p = format!("model.layers.{i}.conv");
            ConvWeights {
                bcx: packed(&t, &format!("{p}.in_proj"), 768, 256),
                output: packed(&t, &format!("{p}.out_proj"), 256, 256),
                coefficients: gamma(&t, &format!("{p}.conv.weight")),
            }
        })
        .collect();
    (
        Config {
            vocab_size: 64,
            hidden_size: 256,
            intermediate_size: 256,
            num_layers: 3,
            query_heads: 4,
            kv_heads: 2,
            head_dim: 64,
            max_positions: 12,
            rms_norm_eps: 1e-5,
            rope_theta: 1e6,
        },
        Weights {
            embedding: packed(&t, "model.embed_tokens", 64, 256),
            layers,
            attention,
            convolution,
            final_gamma: gamma(&t, "model.embedding_norm.weight"),
        },
    )
}
