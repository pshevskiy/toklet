use super::*;
use crate::weights::audit::tests::Fixture;
fn wrapper(source: weights::Source, storage: Storage) -> Source {
    Source {
        declared_positions: 8,
        config: Config {
            vocab_size: 1,
            hidden_size: 32,
            intermediate_size: 32,
            num_layers: 1,
            query_heads: 1,
            kv_heads: 1,
            head_dim: 32,
            max_positions: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.,
        },
        architecture: Architecture::Llama,
        tied_output: true,
        tensors: BTreeMap::from([(
            "x.weight".into(),
            TensorSpec {
                shape: vec![1, 32],
                storage,
                payload_bytes: 36,
            },
        )]),
        source,
        tensor_payload_bytes: 36,
        admitted_load_bytes: METADATA_ALLOWANCE + STREAM_BYTES + 36,
    }
}
#[test]
fn public_affine_post_read_validation_poison_is_sticky() {
    let mut f = Fixture::new(vec![
        ("x.weight", Dtype::U32, vec![1, 8], vec![0; 32]),
        (
            "x.scales",
            Dtype::F16,
            vec![1, 1],
            0x7e00_u16.to_le_bytes().to_vec(),
        ),
        ("x.biases", Dtype::F16, vec![1, 1], vec![0; 2]),
    ]);
    let placeholder = weights::Source::open(&f.path).unwrap();
    let raw = std::mem::replace(&mut f.source, placeholder);
    let layout = Layout::new(1, 32, 8, 32, MetadataDtype::F16).unwrap();
    let mut source = wrapper(raw, Storage::Affine(layout));
    assert!(
        source
            .read_affine("x.weight", &mut [0; 32], &mut [0; 2], &mut [0; 2])
            .is_err()
    );
    assert!(source.content_identity().is_none());
    assert!(
        source
            .read_affine("x.weight", &mut [0; 32], &mut [0; 2], &mut [0; 2])
            .is_err()
    );
    assert!(source.content_identity().is_none());
}
#[test]
fn public_float_consumer_failure_is_sticky() {
    let mut f = Fixture::new(vec![("x.weight", Dtype::F32, vec![1, 32], vec![0; 128])]);
    let raw = std::mem::replace(&mut f.source, weights::Source::open(&f.path).unwrap());
    let mut source = wrapper(raw, Storage::Float(MetadataDtype::F32));
    source.visit_float("x.weight", |_| Ok(())).unwrap();
    assert!(source.content_identity().is_some());
    assert!(
        source
            .visit_float("x.weight", |_| Err(invalid("consumer")))
            .is_err()
    );
    assert!(source.content_identity().is_none());
    assert!(source.visit_float("x.weight", |_| Ok(())).is_err());
}
#[test]
fn verified_metadata_budget_rejects_before_opening_files() {
    let options = LoadOptions {
        context_limit: 8,
        memory_limit_bytes: 1,
    };
    assert!(matches!(
        Source::open_smol_audited(Path::new("/nonexistent/toklet"), options),
        Err(Error::Budget { .. })
    ));
}

#[test]
fn public_wrong_name_and_destination_poison_prior_evidence() {
    let mut f = Fixture::new(vec![("x.weight", Dtype::F32, vec![1, 32], vec![0; 128])]);
    let raw = std::mem::replace(&mut f.source, weights::Source::open(&f.path).unwrap());
    let mut source = wrapper(raw, Storage::Float(MetadataDtype::F32));
    source.visit_float("x.weight", |_| Ok(())).unwrap();
    let copied = source.content_identity().unwrap();
    assert!(source.visit_float("unknown", |_| Ok(())).is_err());
    assert!(source.content_identity().is_none());
    assert!(source.visit_float("x.weight", |_| Ok(())).is_err());
    assert_eq!(copied.verified_reads(), 1);
    let mut f = Fixture::new(vec![
        ("x.weight", Dtype::U32, vec![1, 8], vec![0; 32]),
        ("x.scales", Dtype::F16, vec![1, 1], vec![0; 2]),
        ("x.biases", Dtype::F16, vec![1, 1], vec![0; 2]),
    ]);
    let raw = std::mem::replace(&mut f.source, weights::Source::open(&f.path).unwrap());
    let layout = Layout::new(1, 32, 8, 32, MetadataDtype::F16).unwrap();
    let mut source = wrapper(raw, Storage::Affine(layout));
    source
        .read_affine("x.weight", &mut [0; 32], &mut [0; 2], &mut [0; 2])
        .unwrap();
    let copied = source.content_identity().unwrap();
    assert!(
        source
            .read_affine("x.weight", &mut [0; 31], &mut [0; 2], &mut [0; 2])
            .is_err()
    );
    assert!(source.content_identity().is_none());
    assert!(
        source
            .read_affine("x.weight", &mut [0; 32], &mut [0; 2], &mut [0; 2])
            .is_err()
    );
    assert_eq!(copied.verified_reads(), 3);
}
