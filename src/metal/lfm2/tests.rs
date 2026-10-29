use super::*;
use crate::model::lfm2::{AttentionWeights, ConvWeights, LayerWeights};
use safetensors::{Dtype, SafeTensors};
const WEIGHTS: &[u8] = include_bytes!("../../../tests/fixtures/lfm2-scalar-v1/model.safetensors");
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
        crate::mlx::affine::Layout::new(r, c, 4, 64, MetadataDtype::BF16).unwrap(),
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
pub(super) fn inputs() -> (Config, Weights<'static>) {
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

#[test]
fn tiny_geometry_routes_and_original_raw_gains() {
    let (c, w) = inputs();
    geometry(&c).unwrap();
    assert_eq!(
        w.layers.iter().map(|l| l.kind).collect::<Vec<_>>(),
        [
            Kind::Convolution(0),
            Kind::Attention(0),
            Kind::Convolution(1)
        ]
    );
    assert_eq!(w.convolution[0].bcx.layout().rows(), 3 * H);
    assert_ne!(w.convolution[0].coefficients, w.convolution[1].coefficients);
    for index in 0..5 {
        let mut bad = c.clone();
        match index {
            0 => bad.kv_heads = 1,
            1 => bad.max_positions = 13,
            2 => bad.hidden_size = 1024,
            3 => bad.rms_norm_eps = 1e-6,
            _ => bad.rope_theta = 10000.,
        };
        assert!(geometry(&bad).is_err());
    }
    let direct = [
        bf16::from_bits(0),
        bf16::from_bits(0x3a00),
        bf16::from_bits(0xbf80),
    ];
    raw(&direct, 3).unwrap();
    assert_eq!(direct.map(|v| v.to_f32()), [0., 2f32.powi(-11), -1.]);
    assert!(raw(&[bf16::from_bits(0x7fc0)], 1).is_err());
    assert!(raw(&direct, 2).is_err());
}
#[test]
fn numerical_ledger_includes_fused_parent_and_all_bank_owners() {
    let (_, w) = inputs();
    let mut packed = w.embedding.layout().total_bytes();
    let mut count = 1;
    for l in &w.layers {
        for a in [&l.gate, &l.up, &l.down] {
            packed += a.layout().total_bytes();
            count += 1;
        }
    }
    for a in &w.attention {
        for a in [&a.q, &a.k, &a.v, &a.o] {
            packed += a.layout().total_bytes();
            count += 1;
        }
    }
    for a in &w.convolution {
        for a in [&a.bcx, &a.output] {
            packed += a.layout().total_bytes();
            count += 1;
        }
    }
    assert_eq!((count, packed), (18, 746496));
    assert_eq!((7 * (H + GUARD) + 2 * (D + GUARD)) * 4, 7932);
    assert_eq!(2 * (3 * H + GUARD) * 4, 6200);
    assert_eq!((3 * H + GUARD) * 2, 1550);
    assert_eq!(
        (B * (8 * H + 2 * Q + 3 * K + 3 * I) + 14 * GUARD) * 2,
        237764
    );
    assert_eq!(extent(4 * B, CAP).unwrap() * 8, 12344);
    assert_eq!(4 * (2 * H + GUARD) * 2, 4152);
    let (p, gpu, host, _) = snapshots::plan(true, geometry(&inputs().0).unwrap()).unwrap();
    assert_eq!((p.len(), gpu, host), (57, 938932, 1855488));
    assert!(extent(usize::MAX, 2).is_err());
    assert!(mul(usize::MAX, 2).is_err());
    assert!(admit(10, 9).is_err());
    admit(10, 10).unwrap();
}
#[test]
fn invalid_inputs_are_rejected_before_gpu_construction() {
    let (c, mut w) = inputs();
    w.layers[2].kind = Kind::Convolution(0);
    assert!(matches!(
        Model::from_raw_weights(
            c,
            w,
            2 * 1024 * 1024,
            crate::weights::INITIAL_MEMORY_LIMIT_BYTES
        ),
        Err(Error::Options("LFM2 fixed owner routes"))
    ));
    let (c, w) = inputs();
    assert!(matches!(
        Model::from_raw_weights(c, w, 0, crate::weights::INITIAL_MEMORY_LIMIT_BYTES),
        Err(Error::Options("LFM2 cold source overlap understated"))
    ));
    let (c, w) = inputs();
    assert!(matches!(
        Model::from_raw_weights(c, w, 2 * 1024 * 1024, 1),
        Err(Error::Budget { .. })
    ));
}

pub(super) fn real_config(capacity: usize) -> Config {
    Config {
        vocab_size: 65536,
        hidden_size: 1024,
        intermediate_size: 4608,
        num_layers: 16,
        query_heads: 16,
        kv_heads: 8,
        head_dim: 64,
        max_positions: capacity,
        rms_norm_eps: 1e-5,
        rope_theta: 1e6,
    }
}
#[test]
fn selected_real_routes_and_bounded_views_are_resolved_cold() {
    for capacity in [1, 31, 32, 64, 511, 512] {
        let g = geometry(&real_config(capacity)).unwrap();
        assert_eq!((g.views(), g.stages()), (capacity.min(32), 293));
        let (mut kv, mut history, mut slots) = (0, 0, 1);
        for layer in 0..16 {
            if [2, 5, 8, 10, 12, 14].contains(&layer) {
                assert_eq!(g.kind(layer), Kind::Attention(kv));
                kv += 1;
                slots += 20;
            } else {
                assert_eq!(g.kind(layer), Kind::Convolution(history));
                history += 1;
                slots += 17;
            }
        }
        assert_eq!((kv, history, slots + 2), (6, 10, 293));
        assert!(g.cache_plan().unwrap().session_bytes().unwrap() > 0);
    }
    for capacity in [0, 513, 128000, usize::MAX] {
        assert!(geometry(&real_config(capacity)).is_err());
    }
    for field in 0..9 {
        let mut c = real_config(512);
        match field {
            0 => c.vocab_size -= 1,
            1 => c.hidden_size = 256,
            2 => c.intermediate_size = 4096,
            3 => c.num_layers = 15,
            4 => c.query_heads = 8,
            5 => c.kv_heads = 4,
            6 => c.head_dim = 128,
            7 => c.rms_norm_eps = 1e-6,
            _ => c.rope_theta = 10000.,
        }
        assert!(geometry(&c).is_err());
    }
}
#[test]
fn selected_real_step_bounds_do_not_admit_capacity_sized_mps_views() {
    let g = geometry(&real_config(512)).unwrap();
    g.admit_step(480, &[65535; 32]).unwrap();
    g.admit_step(511, &[1]).unwrap();
    for (offset, tokens) in [
        (0, vec![]),
        (0, vec![1; 33]),
        (481, vec![1; 32]),
        (512, vec![1]),
        (usize::MAX, vec![1]),
        (0, vec![65536]),
    ] {
        assert!(g.admit_step(offset, &tokens).is_err());
    }
    let single = geometry(&real_config(1)).unwrap();
    single.admit_step(0, &[1]).unwrap();
    assert!(single.admit_step(0, &[1, 2]).is_err());
    // Pairs validate every candidate, not one history per attention owner.
    let owners: Vec<_> = (0..g.convolution).collect();
    assert_eq!(owners.chunks_exact(2).count(), 5);
    assert_eq!(
        owners
            .chunks_exact(2)
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
        owners
    );
    assert_eq!(g.convolution * 2 * g.h, 20480);
}
#[test]
fn selected_real_original_layout_and_numerical_owner_ledger() {
    let g = geometry(&real_config(512)).unwrap();
    let bytes = |r, c| {
        let layout = crate::mlx::affine::Layout::new(r, c, 4, 64, MetadataDtype::BF16).unwrap();
        mlx::validate_resident_layout(layout).unwrap();
        let resident = mlx::packed_bytes(layout).unwrap();
        assert_eq!(resident, layout.total_bytes());
        resident
    };
    let packed = bytes(g.v, g.h)
        + g.layers * (2 * bytes(g.i, g.h) + bytes(g.h, g.i))
        + g.attention * (2 * bytes(g.q, g.h) + 2 * bytes(g.k, g.h))
        + g.convolution * (bytes(3 * g.h, g.h) + bytes(g.h, g.h));
    assert_eq!(
        (
            1 + 3 * g.layers + 4 * g.attention + 2 * g.convolution,
            packed
        ),
        (93, 199360512)
    );
    assert_eq!(g.half_width(), 4608);
    assert_eq!(geometry(&inputs().0).unwrap().half_width(), 768);
    assert_eq!(g.norm_bytes().unwrap(), 139500);
    assert_eq!(g.coefficient_bytes().unwrap(), 123160);
    assert_eq!(g.zero_bytes().unwrap(), 131086);
    assert_eq!(g.history_bytes().unwrap(), 82200);
    assert_eq!(g.scratch_bytes().unwrap(), 3735804);
    assert!(g.cache_plan().unwrap().session_bytes().unwrap() >= 6685040);
    assert_eq!(extent(B, g.v).unwrap() * 4 * 2, 16777272);
    assert_eq!(extent(1, g.v).unwrap() * 4 * 2, 524344);
    let (_, gpu, host, _) = snapshots::plan(true, g).unwrap();
    assert_eq!((gpu, host), (54530796, 88080384));
    let (none, gpu, host, metadata) = snapshots::plan(false, g).unwrap();
    assert!(none.is_empty());
    assert_eq!((gpu, host, metadata), (0, 0, 0));
    let sum = packed
        + g.norm_bytes().unwrap()
        + g.coefficient_bytes().unwrap()
        + g.zero_bytes().unwrap()
        + g.history_bytes().unwrap()
        + g.scratch_bytes().unwrap();
    admit(sum, sum).unwrap();
    assert!(
        matches!(admit(sum, sum - 1), Err(Error::Budget { required, limit }) if required == sum && limit == sum - 1)
    );
}
