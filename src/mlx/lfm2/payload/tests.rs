use super::*;
use crate::mlx::{
    affine::{Layout, MetadataDtype},
    lfm2::{Dtype, PhysicalTensor},
};
use std::{
    collections::BTreeMap,
    io::{self, Cursor},
};
const CONFIG: &[u8] =
    include_bytes!("../../../../tests/fixtures/lfm2-350m-metadata-v1/config.json");
const INDEX: &[u8] = include_bytes!("../../../../tests/fixtures/lfm2-350m-metadata-v1/index.json");
const HEADER: &[u8] =
    include_bytes!("../../../../tests/fixtures/lfm2-350m-metadata-v1/header.json");
const PREFIX: &[u8] = b"private synthetic header, not original350M data";
fn hash(x: &[u8]) -> [u8; 32] {
    Sha256::digest(x).into()
}
fn start() -> Sha256 {
    let mut h = Sha256::new();
    h.update(PREFIX);
    h
}
fn options(raw: &[u8]) -> LoadOptions {
    let mut sha = start();
    sha.update(raw);
    LoadOptions {
        identity: ContentIdentity {
            config_sha256: hash(b"config"),
            index_sha256: hash(b"index"),
            header_sha256: hash(PREFIX),
            container_sha256: sha.finalize().into(),
            container_bytes: (PREFIX.len() + raw.len()) as u64,
        },
        memory_limit_bytes: usize::MAX,
        external_resident_bytes: 0,
    }
}
fn admitted(
    mut reader: impl Read,
    plan: Plan,
    options: LoadOptions,
) -> Result<PayloadStore, Error> {
    load_admitted(
        &mut reader,
        plan,
        start(),
        PREFIX.len() as u64,
        PREFIX.len() - 8,
        options,
    )
}
fn insert(
    physical: &mut BTreeMap<String, PhysicalTensor>,
    raw: &mut Vec<u8>,
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: &[u8],
) {
    let start = raw.len();
    raw.extend(bytes);
    assert!(
        physical
            .insert(
                name,
                PhysicalTensor {
                    dtype,
                    shape,
                    range: start..raw.len()
                }
            )
            .is_none()
    );
}
fn words(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
        .collect()
}
// All original names and routes, but deliberately tiny private storage. These
// forged dimensions cannot enter public inspect/load and never execute a model.
fn tiny_names() -> (Plan, Vec<u8>) {
    let mut common = super::super::common_tensors().unwrap();
    let mut convolutions = super::super::convolutions(&common).unwrap();
    let mut raw = vec![0x6d]; // odd metadata alignment intentionally exercises byte decoding
    let mut physical = BTreeMap::new();
    let layout = Layout::new(2, 64, 4, 64, MetadataDtype::BF16).unwrap();
    for (ordinal, t) in common.iter_mut().enumerate() {
        let marker = ordinal as f32;
        match t.storage {
            Storage::Affine(_) => {
                t.shape = Shape::Matrix {
                    rows: 2,
                    columns: 64,
                };
                t.storage = Storage::Affine(layout);
                let base = t.name.strip_suffix(".weight").unwrap();
                insert(
                    &mut physical,
                    &mut raw,
                    t.name.clone(),
                    Dtype::U32,
                    vec![2, 8],
                    &[0; 64],
                );
                insert(
                    &mut physical,
                    &mut raw,
                    format!("{base}.scales"),
                    Dtype::Bf16,
                    vec![2, 1],
                    &words(&[1., 1.]),
                );
                insert(
                    &mut physical,
                    &mut raw,
                    format!("{base}.biases"),
                    Dtype::Bf16,
                    vec![2, 1],
                    &words(&[marker, marker]),
                );
            }
            Storage::Bf16Vector => {
                t.shape = Shape::Vector(2);
                insert(
                    &mut physical,
                    &mut raw,
                    t.name.clone(),
                    Dtype::Bf16,
                    vec![2],
                    &words(&[marker, -0.]),
                );
            }
        }
    }
    for t in &mut convolutions {
        t.channels = 2;
        let x = 200. + t.history_owner as f32;
        insert(
            &mut physical,
            &mut raw,
            t.coefficient_name.clone(),
            Dtype::Bf16,
            vec![2, 3, 1],
            &words(&[x, 0., -0., 1., -1., 0.5]),
        );
    }
    assert!(raw.len() < 9 * 1024);
    (
        Plan {
            layers: super::super::layers(),
            common,
            convolutions,
            physical,
            payload_bytes: raw.len(),
        },
        raw,
    )
}
struct Fragmented {
    bytes: Cursor<Vec<u8>>,
    width: usize,
    fail: Option<u64>,
}
impl Read for Fragmented {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.fail.is_some_and(|n| self.bytes.position() >= n) {
            return Err(io::Error::other("injected read failure"));
        }
        let n = out.len().min(self.width);
        self.bytes.read(&mut out[..n])
    }
}
#[test]
fn fragmented_original_bytes_and_unaligned_direct_words_are_preserved() {
    for width in [1, 3, 65536] {
        let (plan, raw) = tiny_names();
        let expected = options(&raw);
        let store = admitted(
            Fragmented {
                bytes: Cursor::new(raw.clone()),
                width,
                fail: None,
            },
            plan,
            expected,
        )
        .unwrap();
        assert_eq!(store.payload, raw);
        assert_eq!(*store.content_identity(), expected.identity);
        let r = store.report();
        assert_eq!(
            (
                r.packed_matrices,
                r.direct_bf16_norms,
                r.rank3_bf16_convolutions
            ),
            (93, 45, 10)
        );
        assert_eq!(r.container_bytes, (PREFIX.len() + raw.len()) as u64);
        assert_eq!(r.header_bytes, PREFIX.len() - 8);
        assert_eq!(
            r.admitted_load_bytes,
            store.payload.capacity()
                + METADATA_RESERVATION
                + STREAM_BYTES
                + r.header_capacity_bytes
        );
        assert_eq!(
            store.bytes("model.embed_tokens.weight").unwrap().as_ptr(),
            store.payload[1..].as_ptr()
        );
        assert_eq!(
            store
                .direct_gain("model.embedding_norm.weight")
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            [0x3f80, 0x8000]
        );
        assert_eq!(
            store
                .convolution_coefficients("model.layers.0.conv.conv.weight")
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            [0x4348, 0, 0x8000, 0x3f80, 0xbf80, 0x3f00]
        );
        assert!(store.matrix("model.embedding_norm.weight").is_err());
        assert!(store.direct_gain("model.embed_tokens.weight").is_err());
        assert!(
            store
                .direct_gain("model.layers.0.conv.conv.weight")
                .is_err()
        );
        assert!(
            store
                .convolution_coefficients("model.embedding_norm.weight")
                .is_err()
        );
    }
}
#[test]
fn failed_truncated_trailing_and_changed_streams_publish_no_store() {
    for mode in 0..6 {
        let (plan, mut raw) = tiny_names();
        let mut opt = options(&raw);
        let fail = if mode == 0 { Some(3) } else { None };
        match mode {
            1 => {
                raw.pop();
            }
            2 => raw.push(0),
            3 => opt.identity.container_sha256[0] ^= 1,
            4 => opt.identity.container_bytes += 1,
            5 => raw[1] ^= 1,
            _ => {}
        }
        assert!(
            admitted(
                Fragmented {
                    bytes: Cursor::new(raw),
                    width: 1,
                    fail
                },
                plan,
                opt
            )
            .is_err()
        );
    }
}
#[test]
fn reauthenticated_bad_values_and_finite_fma_rescue_are_rejected() {
    for (name, bits) in [
        ("model.embed_tokens.scales", 0x7fc0u16),
        ("model.embed_tokens.biases", 0xff80),
        ("model.embedding_norm.weight", 0x7f80),
        ("model.layers.0.conv.conv.weight", 0x7fc1),
    ] {
        let (plan, mut raw) = tiny_names();
        let r = plan.physical[name].range.clone();
        raw[r.start..r.start + 2].copy_from_slice(&bits.to_le_bytes());
        let opt = options(&raw);
        assert!(admitted(Cursor::new(raw), plan, opt).is_err());
    }
    // q=15 product overflows F32, but adding the negative bias in F64 then
    // rounding is finite. The selected separate-product source domain rejects.
    let (plan, mut raw) = tiny_names();
    let s = plan.physical["model.embed_tokens.scales"].range.start;
    let b = plan.physical["model.embed_tokens.biases"].range.start;
    let scale = bf16::from_bits(0x7e00);
    let bias = bf16::from_bits(0xff7f);
    assert!(!(scale.to_f32() * 15.).is_finite());
    assert!(((f64::from(scale.to_f32()) * 15. + f64::from(bias.to_f32())) as f32).is_finite());
    raw[s..s + 2].copy_from_slice(&scale.to_le_bytes());
    raw[b..b + 2].copy_from_slice(&bias.to_le_bytes());
    let opt = options(&raw);
    assert!(admitted(Cursor::new(raw), plan, opt).is_err());
    assert!(decode_bf16(&[0, 0, 0], 2).is_err());
    assert!(add(usize::MAX, 1).is_err());
    assert!(multiply(usize::MAX, 2).is_err());
}
#[test]
fn load_budget_counts_header_and_external_owner_before_payload_read() {
    struct Never;
    impl Read for Never {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            panic!("must refuse before payload read")
        }
    }
    let (plan, raw) = tiny_names();
    let mut opt = options(&raw);
    opt.external_resident_bytes = 177;
    let required = raw.len() + METADATA_RESERVATION + STREAM_BYTES + PREFIX.len() - 8 + 177;
    opt.memory_limit_bytes = required - 1;
    assert!(matches!(admitted(Never,plan,opt),Err(Error::Budget{required:r,..})if r==required));
    let (plan, raw) = tiny_names();
    opt.memory_limit_bytes = required;
    let store = admitted(Cursor::new(raw), plan, opt).unwrap();
    assert_eq!(store.report.admitted_load_bytes, required);
    let (plan, raw) = tiny_names();
    let mut opt = options(&raw);
    opt.external_resident_bytes = usize::MAX;
    assert!(admitted(Never, plan, opt).is_err());
}
#[test]
fn all_cold_routes_names_tied_parent_and_temporary_capacities_are_resolved() {
    let (plan, raw) = tiny_names();
    let opt = options(&raw);
    let store = admitted(Cursor::new(raw), plan, opt).unwrap();
    let marker = |name: &str| {
        store
            .plan
            .logical_tensors()
            .iter()
            .position(|t| t.name == name)
            .unwrap() as f32
    };
    for capacity in [1, 64, 512] {
        let input = store
            .prepare_direct_weights(capacity, 777, usize::MAX)
            .unwrap();
        assert_eq!(
            (
                input.weights.layers.len(),
                input.weights.attention.len(),
                input.weights.convolution.len()
            ),
            (16, 6, 10)
        );
        assert_eq!(input.config.max_positions, capacity);
        assert_eq!(input.weights.embedding.value(0, 0).unwrap(), 0.);
        assert_eq!(input.weights.final_gamma[0].to_f32(), 1.);
        let mut matrices = 1;
        for (i, l) in input.weights.layers.iter().enumerate() {
            let b = format!("model.layers.{i}.");
            for (m, s) in [
                (&l.gate, "feed_forward.w1"),
                (&l.up, "feed_forward.w3"),
                (&l.down, "feed_forward.w2"),
            ] {
                assert_eq!(m.value(0, 0).unwrap(), marker(&format!("{b}{s}.weight")));
                matrices += 1;
            }
            for (g, s) in [
                (&l.operator_gamma, "operator_norm"),
                (&l.ffn_gamma, "ffn_norm"),
            ] {
                assert_eq!(g[0].to_f32(), marker(&format!("{b}{s}.weight")));
                assert_eq!(g[1].to_bits(), 0x8000);
            }
            match l.kind {
                Kind::Attention(owner) => {
                    assert!(
                        matches!(store.plan.layers()[i],LayerRecipe::FullAttention(p)if p.kv_owner==owner)
                    );
                    let a = &input.weights.attention[owner];
                    for (m, s) in [
                        (&a.q, "q_proj"),
                        (&a.k, "k_proj"),
                        (&a.v, "v_proj"),
                        (&a.o, "out_proj"),
                    ] {
                        assert_eq!(
                            m.value(0, 0).unwrap(),
                            marker(&format!("{b}self_attn.{s}.weight"))
                        );
                        matrices += 1;
                    }
                    assert_eq!(
                        a.q_gamma[0].to_f32(),
                        marker(&format!("{b}self_attn.q_layernorm.weight"))
                    );
                    assert_eq!(
                        a.k_gamma[0].to_f32(),
                        marker(&format!("{b}self_attn.k_layernorm.weight"))
                    );
                }
                Kind::Convolution(owner) => {
                    assert_eq!(
                        store.plan.layers()[i],
                        LayerRecipe::ShortConv3 {
                            history_owner: owner
                        }
                    );
                    let c = &input.weights.convolution[owner];
                    assert_eq!(
                        c.bcx.value(0, 0).unwrap(),
                        marker(&format!("{b}conv.in_proj.weight"))
                    );
                    assert_eq!(
                        c.output.value(0, 0).unwrap(),
                        marker(&format!("{b}conv.out_proj.weight"))
                    );
                    matrices += 2;
                    assert_eq!(c.coefficients[0].to_f32(), 200. + owner as f32);
                }
            }
        }
        assert_eq!(matrices, 93);
        let needed = store.payload.capacity()
            + METADATA_RESERVATION
            + 777
            + input_capacity(&input.weights).unwrap();
        assert_eq!(input.source_resident_bytes, needed);
        assert_eq!(
            input_capacity(&input.weights).unwrap(),
            temporary_bytes(
                &store.plan,
                input.weights.layers.capacity(),
                input.weights.attention.capacity(),
                input.weights.convolution.capacity()
            )
            .unwrap()
        );
        drop(input);
        assert!(store.prepare_direct_weights(capacity, 777, needed).is_ok());
        assert!(matches!(
            store.prepare_direct_weights(capacity, 777, needed - 1),
            Err(Error::Budget { .. })
        ));
    }
    assert!(
        store
            .prepare_direct_weights(64, usize::MAX, usize::MAX)
            .is_err()
    );
}
#[test]
fn strict_metadata_identity_and_header_fail_before_payload() {
    let identity = ContentIdentity {
        config_sha256: hash(CONFIG),
        index_sha256: hash(INDEX),
        header_sha256: hash(HEADER),
        container_sha256: [0; 32],
        container_bytes: 199527820,
    };
    let mut data = (HEADER.len() as u64).to_le_bytes().to_vec();
    data.extend(HEADER);
    for mode in 0..4 {
        let mut id = identity;
        let mut reader = Cursor::new(data.clone());
        match mode {
            0 => id.config_sha256[0] ^= 1,
            1 => id.index_sha256[0] ^= 1,
            2 => id.header_sha256[0] ^= 1,
            _ => {}
        }
        let limit = if mode == 3 {
            METADATA_RESERVATION + STREAM_BYTES + HEADER.len() + 128
        } else {
            usize::MAX
        };
        assert!(
            load(
                CONFIG,
                INDEX,
                &mut reader,
                LoadOptions {
                    identity: id,
                    memory_limit_bytes: limit,
                    external_resident_bytes: 0
                }
            )
            .is_err()
        );
        assert_eq!(
            reader.position(),
            if mode < 2 { 0 } else { data.len() as u64 }
        );
    }
    for len in [0, MAX_HEADER_BYTES as u64 + 1, u64::MAX] {
        assert!(
            load(
                CONFIG,
                INDEX,
                Cursor::new(len.to_le_bytes()),
                LoadOptions {
                    identity,
                    memory_limit_bytes: usize::MAX,
                    external_resident_bytes: 0
                }
            )
            .is_err()
        );
    }
    let mut unsupported: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    unsupported["conv_bias"] = true.into();
    let config = serde_json::to_vec(&unsupported).unwrap();
    let mut id = identity;
    id.config_sha256 = hash(&config);
    assert!(
        load(
            &config,
            INDEX,
            Cursor::new(data),
            LoadOptions {
                identity: id,
                memory_limit_bytes: usize::MAX,
                external_resident_bytes: 0
            }
        )
        .is_err()
    );
}
#[test]
fn selected_geometry_is_cold_data_not_backend_or_source_window_admission() {
    for capacity in [1, 12, 64, 511, 512] {
        let c = execution_config(capacity).unwrap();
        c.validate().unwrap();
        assert_eq!(
            (
                c.hidden_size,
                c.intermediate_size,
                c.vocab_size,
                c.num_layers,
                c.query_heads,
                c.kv_heads,
                c.head_dim
            ),
            (1024, 4608, 65536, 16, 16, 8, 64)
        );
        assert_eq!(c.max_positions, capacity);
        assert_eq!(c.rms_norm_eps.to_bits(), 1e-5_f32.to_bits());
    }
    for capacity in [0, 513, usize::MAX] {
        assert!(execution_config(capacity).is_err());
    }
    let plan = inspect(CONFIG, INDEX, HEADER, 199527820).unwrap();
    assert_eq!(plan.payload_bytes(), 199491072);
    assert_eq!(plan.declared_total_positions(), 128000);
    assert_eq!(plan.implemented_total_positions(), None);
    assert_eq!(plan.qualified_total_positions(), None);
    assert_eq!(
        temporary_bytes(&plan, 0, 0, 0).unwrap(),
        (33 * 1024 + 12 * 64 + 10 * 3 * 1024) * 2
    );
}
