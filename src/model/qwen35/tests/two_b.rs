//! Independent tiny ratio-one graph; no downloaded2B source or route claim.
use super::*;

const MANIFEST: &str =
    include_str!("../../../../tests/fixtures/qwen35-2b-cpu-graph-v1/manifest.json");
const WEIGHTS: &[u8] =
    include_bytes!("../../../../tests/fixtures/qwen35-2b-cpu-graph-v1/weights.bin");
const REFERENCE: &[u8] =
    include_bytes!("../../../../tests/fixtures/qwen35-2b-cpu-graph-v1/reference.bin");

fn data() -> &'static Value {
    static DATA: OnceLock<Value> = OnceLock::new();
    DATA.get_or_init(|| {
        let m: Value = serde_json::from_str(MANIFEST).unwrap();
        assert_eq!(m["format"], "toklet-qwen35-cpu-graph-v1");
        assert_eq!(m["fixture_only"], true);
        assert_eq!(m["gate"]["atol"].as_f64(), Some(1e-5));
        assert_eq!(m["gate"]["rtol"].as_f64(), Some(1e-4));
        assert_eq!(
            format!("{:x}", Sha256::digest(WEIGHTS)),
            m["weights_sha256"]
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(REFERENCE)),
            m["reference_sha256"]
        );
        assert_eq!(m["weights_bytes"].as_u64().unwrap() as usize, WEIGHTS.len());
        assert_eq!(
            m["reference_bytes"].as_u64().unwrap() as usize,
            REFERENCE.len()
        );
        let c: Config = serde_json::from_value(m["geometry"].clone()).unwrap();
        assert_eq!(
            (c.hidden, c.intermediate, c.vocab, c.layers),
            (64, 128, 32, 4)
        );
        assert_eq!(
            (c.key_heads, c.value_heads, c.key_dim, c.value_dim),
            (2, 2, 32, 32)
        );
        assert_eq!(
            (c.query_heads, c.kv_heads, c.head_dim, c.rotated_dims),
            (4, 2, 16, 8)
        );
        m
    })
}
fn raw(component: &Value) -> &'static [u8] {
    let start = component["offset"].as_u64().unwrap() as usize;
    let n = component["bytes"].as_u64().unwrap() as usize;
    let bytes = &WEIGHTS[start..start + n];
    assert_eq!(format!("{:x}", Sha256::digest(bytes)), component["sha256"]);
    bytes
}
struct TwoBFixture {
    vectors: BTreeMap<String, Vec<f32>>,
}
impl TwoBFixture {
    fn new() -> Self {
        let vectors = data()["weights"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(_, v)| v["kind"] != "matrix")
            .map(|(name, v)| {
                let values = match v["kind"].as_str().unwrap() {
                    "bf16" => raw(&v["data"])
                        .chunks_exact(2)
                        .map(|v| {
                            f32::from_bits(
                                u32::from(u16::from_le_bytes(v.try_into().unwrap())) << 16,
                            )
                        })
                        .collect(),
                    "f32" => raw(&v["data"])
                        .chunks_exact(4)
                        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                        .collect(),
                    _ => panic!("unexpected tiny vector storage"),
                };
                (name.clone(), values)
            })
            .collect();
        Self { vectors }
    }
    fn matrix(&self, name: &str) -> Matrix<'_> {
        let row = &data()["weights"][name];
        let shape = row["shape"].as_array().unwrap();
        Matrix::from_le_bytes(
            Layout::new(
                shape[0].as_u64().unwrap() as usize,
                shape[1].as_u64().unwrap() as usize,
                4,
                64,
                MetadataDtype::BF16,
            )
            .unwrap(),
            raw(&row["codes"]),
            raw(&row["scales"]),
            raw(&row["biases"]),
        )
        .unwrap()
    }
    fn model(&self) -> Model<'_> {
        let c: Config = serde_json::from_value(data()["geometry"].clone()).unwrap();
        let mut layers = Vec::new();
        let p = "language_model.model.";
        for i in 0..c.layers {
            let base = format!("{p}layers.{i}.");
            let mat = |role: &str| self.matrix(&format!("{base}{role}.weight"));
            let vector = |role: &str| self.vectors[&format!("{base}{role}")].as_slice();
            let mixing = if i % 4 == 3 {
                MixingWeights::Attention(AttentionWeights {
                    q_gate: mat("self_attn.q_proj"),
                    k: mat("self_attn.k_proj"),
                    v: mat("self_attn.v_proj"),
                    out: mat("self_attn.o_proj"),
                    q_norm: vector("self_attn.q_norm.weight"),
                    k_norm: vector("self_attn.k_norm.weight"),
                })
            } else {
                MixingWeights::Delta(DeltaWeights {
                    qkv: mat("linear_attn.in_proj_qkv"),
                    z: mat("linear_attn.in_proj_z"),
                    a: mat("linear_attn.in_proj_a"),
                    b: mat("linear_attn.in_proj_b"),
                    out: mat("linear_attn.out_proj"),
                    convolution: vector("linear_attn.conv1d.weight"),
                    a_log: vector("linear_attn.A_log"),
                    dt_bias: vector("linear_attn.dt_bias"),
                    norm: vector("linear_attn.norm.weight"),
                })
            };
            layers.push(LayerWeights {
                input_norm: vector("input_layernorm.weight"),
                post_norm: vector("post_attention_layernorm.weight"),
                gate: mat("mlp.gate_proj"),
                up: mat("mlp.up_proj"),
                down: mat("mlp.down_proj"),
                mixing,
            });
        }
        let model = Model::new(
            c,
            Weights {
                embedding: self.matrix(&format!("{p}embed_tokens.weight")),
                final_norm: &self.vectors[&format!("{p}norm.weight")],
                layers,
            },
        )
        .unwrap();
        assert_eq!(
            model.learned_bytes,
            data()["backing_bytes"].as_u64().unwrap() as usize
        );
        assert!(model.output_head.is_none());
        model
    }
}
fn reference(row: &Value, profile: &str) -> Vec<f64> {
    let start = row["offset"].as_u64().unwrap() as usize;
    let count = row["count"].as_u64().unwrap() as usize;
    let width = if profile == "ideal_fp64" { 8 } else { 4 };
    let bytes = &REFERENCE[start..start + count * width];
    assert_eq!(bytes.len(), row["bytes"].as_u64().unwrap() as usize);
    assert_eq!(format!("{:x}", Sha256::digest(bytes)), row["sha256"]);
    if width == 8 {
        bytes
            .chunks_exact(8)
            .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
            .collect()
    } else {
        bytes
            .chunks_exact(4)
            .map(|v| f64::from(f32::from_le_bytes(v.try_into().unwrap())))
            .collect()
    }
}
fn ids(case: &Value) -> Vec<u32> {
    case["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

#[test]
fn independent_ratio_one_profiles_cover_all_stages_states_and_tied_logits() {
    let fixture = TwoBFixture::new();
    let model = fixture.model();
    let mut compared = 0;
    let mut maximum = 0.0f64;
    for case in data()["cases"].as_array().unwrap() {
        let tokens = ids(case);
        let offset = case["position_offset"].as_u64().unwrap() as usize;
        let mut expected: BTreeMap<_, Vec<_>> = BTreeMap::new();
        for profile in ["ideal_fp64", "staged_fp32"] {
            for row in case["profiles"][profile].as_array().unwrap() {
                expected
                    .entry(key(
                        row["token"].as_u64().unwrap() as usize,
                        row["layer"].as_u64().map(|v| v as usize),
                        row["stage"].as_str().unwrap(),
                    ))
                    .or_default()
                    .push(reference(row, profile));
            }
        }
        for chunk in [1, 3, 17] {
            let mut session = model
                .session(Options {
                    position_offset: offset,
                    ..options()
                })
                .unwrap();
            let mut seen = BTreeMap::new();
            for part in tokens.chunks(chunk) {
                session
                    .forward_observed(part, &AtomicBool::new(false), &mut |o| {
                        if o.stage == "before_commit" {
                            return Ok(());
                        }
                        assert_eq!(o.position, offset + o.token);
                        let key = observed_key(&o);
                        let profiles = &expected[&key];
                        assert_eq!(profiles.len(), 2);
                        for values in profiles {
                            maximum = maximum.max(close(o.values, values));
                            compared += o.values.len();
                        }
                        assert!(seen.insert(key, ()).is_none());
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(session.work().tokens_committed, part.len());
                assert_eq!(session.work().recurrence_steps_completed, 3 * part.len());
                assert_eq!(session.work().attention_heads_completed, 4 * part.len());
                assert_eq!(session.work().projections_completed, 32 * part.len());
            }
            assert_eq!(seen.len(), expected.len());
            assert_eq!(session.history(), tokens);
        }
    }
    println!("QWEN35_TWO_B_TINY compared={compared},max_ratio={maximum}; fixture only");
}

#[test]
fn ratio_one_f32_decay_owners_keep_original_non_bf16_words() {
    let fixture = TwoBFixture::new();
    let model = fixture.model();
    for (layer, owner) in model.layers.iter().enumerate() {
        let MixingWeights::Delta(delta) = &owner.weights.mixing else {
            continue;
        };
        let name = format!("language_model.model.layers.{layer}.linear_attn.A_log");
        assert_eq!(data()["weights"][&name]["kind"], "f32");
        assert_eq!(delta.a_log.len(), 2);
        assert_eq!(delta.a_log.as_ptr(), fixture.vectors[&name].as_ptr());
        for (actual, word) in delta
            .a_log
            .iter()
            .zip(raw(&data()["weights"][&name]["data"]).chunks_exact(4))
        {
            assert_eq!(
                actual.to_bits(),
                u32::from_le_bytes(word.try_into().unwrap())
            );
            assert_ne!(actual.to_bits() & 0xffff, 0);
        }
    }
}

#[test]
fn ratio_one_transactions_retry_partitions_reset_and_memory_admission() {
    let fixture = TwoBFixture::new();
    let model = fixture.model();
    let tokens = ids(&data()["cases"][1]);
    let mut clean = model.session(options()).unwrap();
    let output = clean.forward(&tokens).unwrap().to_vec();
    let target = snapshot(&clean);
    let admitted = clean.memory();
    assert!(matches!(
        model.session(Options {
            memory_limit_bytes: admitted.admitted_bytes - 1,
            ..options()
        }),
        Err(Error::MemoryBudget { .. })
    ));
    for chunk in [1, 3, 8] {
        let mut session = model.session(options()).unwrap();
        let mut partitioned = Vec::new();
        for part in tokens.chunks(chunk) {
            partitioned.extend_from_slice(session.forward(part).unwrap());
        }
        assert_float_bits(&partitioned, &output);
        assert_eq!(snapshot(&session), target);
        session.reset();
        assert!(session.history().is_empty() && session.logits().is_none());
        assert_eq!(session.memory(), admitted);
        assert_float_bits(session.forward(&tokens).unwrap(), &output);
        assert_eq!(snapshot(&session), target);
    }
    let mut baseline = model.session(options()).unwrap();
    baseline.forward(&tokens[..1]).unwrap();
    let prefix = snapshot(&baseline);
    baseline.forward(&tokens[1..3]).unwrap();
    let suffix = snapshot(&baseline);
    for stage in [
        "delta_decay",
        "delta_state",
        "attention_scores",
        "before_commit",
    ] {
        for cancel_request in [false, true] {
            let mut session = model.session(options()).unwrap();
            session.forward(&tokens[..1]).unwrap();
            let cancel = AtomicBool::new(false);
            let mut injected = false;
            let result = session.forward_observed(&tokens[1..3], &cancel, &mut |o| {
                if !injected && o.stage == stage {
                    injected = true;
                    if cancel_request {
                        cancel.store(true, Ordering::Relaxed);
                    } else {
                        return Err(Error::Observation("ratio-one control"));
                    }
                }
                Ok(())
            });
            assert!(injected && result.is_err());
            assert_eq!(snapshot(&session), prefix);
            assert_eq!(session.work().tokens_committed, 0);
            assert_eq!(session.memory(), admitted);
            session.forward(&tokens[1..3]).unwrap();
            assert_eq!(snapshot(&session), suffix);
        }
    }
}
