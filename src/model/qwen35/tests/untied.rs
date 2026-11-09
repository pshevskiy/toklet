//! Explicit two-table graph controls; original tied fixtures remain immutable.
use super::*;
const MANIFEST: &str =
    include_str!("../../../../tests/fixtures/qwen35-untied-graph-v1/manifest.json");
const WEIGHTS: &[u8] =
    include_bytes!("../../../../tests/fixtures/qwen35-untied-graph-v1/weights.bin");
const REFERENCE: &[u8] =
    include_bytes!("../../../../tests/fixtures/qwen35-untied-graph-v1/reference.bin");
const HEAD: &str = "language_model.lm_head.weight";

pub(super) fn data() -> &'static Value {
    static VALUE: OnceLock<Value> = OnceLock::new();
    VALUE.get_or_init(|| {
        let value: Value = serde_json::from_str(MANIFEST).unwrap();
        assert_eq!(value["format"], "toklet-qwen35-untied-graph-v1");
        assert_eq!(value["gate"]["atol"].as_f64(), Some(1e-5));
        assert_eq!(value["gate"]["rtol"].as_f64(), Some(1e-4));
        assert_eq!(
            format!("{:x}", Sha256::digest(WEIGHTS)),
            value["weights_sha256"]
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(REFERENCE)),
            value["reference_sha256"]
        );
        assert_eq!(&WEIGHTS[..super::WEIGHTS.len()], super::WEIGHTS);
        value
    })
}
fn raw(component: &Value) -> &'static [u8] {
    let offset = component["offset"].as_u64().unwrap() as usize;
    let size = component["bytes"].as_u64().unwrap() as usize;
    let result = &WEIGHTS[offset..offset + size];
    assert_eq!(format!("{:x}", Sha256::digest(result)), component["sha256"]);
    result
}
pub(super) fn head() -> Matrix<'static> {
    let row = &data()["weights"][HEAD];
    Matrix::from_le_bytes(
        Layout::new(32, 64, 4, 64, MetadataDtype::BF16).unwrap(),
        raw(&row["codes"]),
        raw(&row["scales"]),
        raw(&row["biases"]),
    )
    .unwrap()
}
fn model(fixture: &Fixture) -> Model<'_> {
    let (c, w) = fixture.inputs();
    let model = Model::new_untied(c, w, head()).unwrap();
    assert_eq!(
        model.learned_bytes,
        data()["backing_bytes"].as_u64().unwrap() as usize
    );
    model
}
fn values(row: &Value, profile: &str) -> Vec<f64> {
    let offset = row["offset"].as_u64().unwrap() as usize;
    let count = row["count"].as_u64().unwrap() as usize;
    let width = if profile == "ideal_fp64" { 8 } else { 4 };
    let bytes = &REFERENCE[offset..offset + count * width];
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
pub(super) fn expected(case: &Value) -> BTreeMap<(usize, Option<usize>, String), Vec<Vec<f64>>> {
    let mut result: BTreeMap<_, Vec<_>> = BTreeMap::new();
    for profile in ["ideal_fp64", "staged_fp32"] {
        for row in case["profiles"][profile].as_array().unwrap() {
            result
                .entry(key(
                    row["token"].as_u64().unwrap() as usize,
                    row["layer"].as_u64().map(|v| v as usize),
                    row["stage"].as_str().unwrap(),
                ))
                .or_default()
                .push(values(row, profile));
        }
    }
    result
}

#[test]
fn all_independent_untied_stages_states_logits_and_partitions() {
    let f = Fixture::new();
    let m = model(&f);
    let mut compared = 0;
    let mut maximum = 0.0f64;
    assert!(m.output_head.is_some());
    for case in data()["cases"].as_array().unwrap() {
        let ids: Vec<u32> = case["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let offset = case["position_offset"].as_u64().unwrap() as usize;
        let references = expected(case);
        for chunk in [1, 3, 17] {
            let mut session = m
                .session(Options {
                    position_offset: offset,
                    ..options()
                })
                .unwrap();
            let mut seen = BTreeMap::new();
            for input in ids.chunks(chunk) {
                session
                    .forward_observed(input, &AtomicBool::new(false), &mut |o| {
                        if o.stage == "before_commit" {
                            return Ok(());
                        }
                        assert_eq!(o.position, offset + o.token);
                        let key = observed_key(&o);
                        let reference = &references[&key];
                        assert_eq!(reference.len(), 2);
                        for row in reference {
                            maximum = maximum.max(close(o.values, row));
                            compared += o.values.len();
                        }
                        assert!(seen.insert(key, ()).is_none());
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(session.work().projections_completed, input.len() * 32);
                assert_eq!(session.work().tokens_committed, input.len());
            }
            assert_eq!(seen.len(), references.len());
            assert_eq!(session.history(), ids);
        }
    }
    println!("UNTIED_GRAPH compared={compared},max_ratio={maximum}; tiny fixtures only");
}

#[test]
fn input_lookup_stays_original_and_head_substitution_changes_only_logits() {
    let f = Fixture::new();
    let tied = f.model();
    let untied = model(&f);
    assert!(tied.output_head.is_none());
    let mut a = tied.session(options()).unwrap();
    let mut b = untied.session(options()).unwrap();
    let ids = tokens();
    let mut stages = BTreeMap::new();
    a.forward_observed(&ids, &AtomicBool::new(false), &mut |o| {
        if o.stage != "before_commit" {
            stages.insert(observed_key(&o), o.values.to_vec());
        }
        Ok(())
    })
    .unwrap();
    let mut different = 0;
    b.forward_observed(&ids, &AtomicBool::new(false), &mut |o| {
        if o.stage == "before_commit" {
            return Ok(());
        }
        let old = &stages[&observed_key(&o)];
        if o.stage == "logits" {
            different += usize::from(
                old.iter()
                    .zip(o.values)
                    .any(|(a, b)| a.to_bits() != b.to_bits()),
            );
        } else {
            assert_float_bits(o.values, old);
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(different, ids.len());
    let wrong = &data()["cases"][1];
    let expected = expected(wrong);
    let reference = &expected[&key(ids.len() - 1, None, "logits")][1];
    assert!(
        a.logits()
            .unwrap()
            .iter()
            .zip(reference)
            .any(|(&a, &e)| (f64::from(a) - e).abs() > 1e-5 + 1e-4 * e.abs())
    );
}

#[test]
fn separate_head_geometry_packing_and_memory_are_explicit() {
    let f = Fixture::new();
    let tied = f.model();
    let untied = model(&f);
    assert_eq!(
        untied.learned_bytes - tied.learned_bytes,
        head().layout().total_bytes()
    );
    assert_eq!(
        untied.model_bytes().unwrap() - tied.model_bytes().unwrap(),
        head().layout().total_bytes()
    );
    assert_eq!(untied.owned_bytes, tied.owned_bytes);
    let wanted = untied.session(options()).unwrap().memory();
    assert!(matches!(
        untied.session(Options {
            memory_limit_bytes: wanted.admitted_bytes - 1,
            ..options()
        }),
        Err(Error::MemoryBudget { .. })
    ));
    let (c, w) = f.inputs();
    let wrong = f.matrix("language_model.model.layers.0.linear_attn.in_proj_a.weight");
    assert!(Model::new_untied(c, w, wrong).is_err());
    let codes = vec![0; 32 * 64];
    let metadata = vec![0; 64];
    let wrong = Matrix::from_le_bytes(
        Layout::new(32, 64, 8, 64, MetadataDtype::BF16).unwrap(),
        &codes,
        &metadata,
        &metadata,
    )
    .unwrap();
    let (c, w) = f.inputs();
    let measure = Measurement::start();
    let result = Model::new_untied(c, w, wrong);
    let allocations = measure.finish();
    assert!(result.is_err());
    assert_eq!(allocations, 0);
}

#[test]
fn late_head_observer_failure_cancel_and_retry_preserve_committed_prefix() {
    let f = Fixture::new();
    let m = model(&f);
    let mut clean = m.session(options()).unwrap();
    clean.forward(&[1]).unwrap();
    let prefix = snapshot(&clean);
    clean.forward(&[7]).unwrap();
    let target = snapshot(&clean);
    for stage in ["final_norm", "logits", "before_commit"] {
        let mut s = m.session(options()).unwrap();
        s.forward(&[1]).unwrap();
        let memory = s.memory();
        assert!(
            s.forward_observed(
                &[7],
                &AtomicBool::new(false),
                &mut |o| if o.stage == stage {
                    Err(Error::Observation("late head"))
                } else {
                    Ok(())
                }
            )
            .is_err()
        );
        assert_eq!(snapshot(&s), prefix);
        assert_eq!(s.work().tokens_committed, 0);
        assert_eq!(s.work().outcome, Outcome::Error);
        assert_eq!(s.memory(), memory);
        s.forward(&[7]).unwrap();
        assert_eq!(snapshot(&s), target);
    }
    let mut s = m.session(options()).unwrap();
    s.forward(&[1]).unwrap();
    let cancel = AtomicBool::new(false);
    assert!(matches!(
        s.forward_observed(&[7], &cancel, &mut |o| {
            if o.stage == "final_norm" {
                cancel.store(true, Ordering::Relaxed);
            }
            Ok(())
        }),
        Err(Error::Cancelled)
    ));
    assert_eq!(snapshot(&s), prefix);
    cancel.store(false, Ordering::Relaxed);
    s.forward_with_cancel(&[7], &cancel).unwrap();
    assert_eq!(snapshot(&s), target);
}

#[test]
fn warmed_untied_reset_replay_and_cancellation_allocate_no_rust_heap() {
    let f = Fixture::new();
    let m = model(&f);
    let mut s = m.session(options()).unwrap();
    s.forward(&[1]).unwrap();
    s.reset();
    let memory = s.memory();
    for _ in 0..3 {
        for ids in [&[1, 7, 13][..], &[3][..], &[9, 5][..]] {
            let measure = Measurement::start();
            let result = s.forward(ids);
            let allocations = measure.finish();
            result.unwrap();
            assert_eq!(allocations, 0);
            assert_eq!(s.memory(), memory);
        }
        let measure = Measurement::start();
        s.reset();
        assert_eq!(measure.finish(), 0);
    }
    let cancel = AtomicBool::new(true);
    let measure = Measurement::start();
    let result = s.forward_with_cancel(&[1], &cancel);
    let allocations = measure.finish();
    assert!(matches!(result, Err(Error::Cancelled)));
    assert_eq!(allocations, 0);
}

struct FailHead {
    output_rows: usize,
    fail: bool,
}
impl execution::sealed::Executor for FailHead {}
impl Execution<Matrix<'_>> for FailHead {
    fn owned_bytes(&self) -> Result<usize> {
        Ok(0)
    }
    fn project(&mut self, m: &Matrix<'_>, input: &[f32], output: &mut [f32]) -> Result<()> {
        if m.layout().rows() == self.output_rows && self.fail {
            self.fail = false;
            return Err(Error::Observation("output execution failed"));
        }
        Cpu.project(m, input, output)
    }
    fn gather(&mut self, m: &Matrix<'_>, token: u32, output: &mut [f32]) -> Result<()> {
        Cpu.gather(m, token, output)
    }
}
#[test]
fn failed_separate_head_is_not_completed_or_committed() {
    let f = Fixture::new();
    let m = model(&f);
    let output_rows = head().layout().rows();
    let mut s = Session::with_execution(
        &m,
        options(),
        FailHead {
            output_rows,
            fail: false,
        },
    )
    .unwrap();
    s.forward(&[1]).unwrap();
    let prefix = snapshot(&s);
    s.execution.fail = true;
    let measure = Measurement::start();
    let result = s.forward(&[7]);
    let allocations = measure.finish();
    assert!(result.is_err());
    assert_eq!(allocations, 0);
    assert_eq!(snapshot(&s), prefix);
    assert_eq!(s.work().projections_completed, 31);
    assert_eq!(s.work().tokens_completed, 0);
    assert_eq!(s.work().tokens_committed, 0);
    s.forward(&[7]).unwrap();
    let mut clean = m.session(options()).unwrap();
    clean.forward(&[1, 7]).unwrap();
    assert_eq!(snapshot(&s), snapshot(&clean));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;

#[test]
#[ignore = "explicit pinned9B CPU binding/one-step head-routing diagnostic; no independent9B graph or quality qualification"]
fn actual_nine_b_binding_one_step_and_memory() {
    use crate::mlx::qwen35::{ModalityPolicy, nine_b::payload};
    use std::{fs::File, io::Read, path::PathBuf, time::Instant};
    println!();
    let folder = PathBuf::from(std::env::var_os("TOKLET_QWEN35_9B_MODEL_DIRECTORY").unwrap());
    let file = File::open(folder.join("config.json")).unwrap();
    assert!(file.metadata().unwrap().is_file());
    assert_eq!(file.metadata().unwrap().len(), 2431);
    let mut config = Vec::with_capacity(2432);
    std::io::Read::take(file, 2432)
        .read_to_end(&mut config)
        .unwrap();
    assert_eq!(config.len(), 2431);
    let start = Instant::now();
    let store = payload::load(
        &config,
        File::open(folder.join("model.safetensors")).unwrap(),
        payload::LoadOptions {
            memory_limit_bytes: MEMORY_LIMIT,
            modality: ModalityPolicy::TextOnly,
        },
    )
    .unwrap();
    let load_ns = start.elapsed().as_nanos();
    let start = Instant::now();
    let model = Model::from_nine_b_cpu_store(&store).unwrap();
    let bind_ns = start.elapsed().as_nanos();
    assert!(model.output_head.is_some());
    assert_eq!(
        (
            model.config.hidden,
            model.config.intermediate,
            model.config.vocab,
            model.config.layers
        ),
        (4096, 12288, 248320, 32)
    );
    assert_eq!(
        model.learned_bytes,
        store.report().retained_tensor_capacity_bytes
    );
    assert_eq!(model.learned_bytes, 5_040_162_816);
    let mut session = model
        .session(Options {
            capacity: 512,
            chunk: 32,
            position_offset: 0,
            memory_limit_bytes: MEMORY_LIMIT,
        })
        .unwrap();
    let memory = session.memory();
    let mut final_norm = vec![0.; 4096];
    let start = Instant::now();
    session
        .forward_observed(&[1], &AtomicBool::new(false), &mut |o| {
            if o.stage == "final_norm" {
                final_norm.copy_from_slice(o.values);
            }
            Ok(())
        })
        .unwrap();
    let forward_ns = start.elapsed().as_nanos();
    let head = store.matrix(&store.plan().output_head().name).unwrap();
    let mut direct = vec![0.; 248320];
    head.matvec(&final_norm, &mut direct).unwrap();
    assert_float_bits(session.logits().unwrap(), &direct);
    assert_eq!(session.work().tokens_committed, 1);
    assert_eq!(session.work().projections_completed, 249);
    assert_eq!(session.history(), [1]);
    let source_ids = store.identity();
    assert_eq!(source_ids.container_bytes(), 5_038_160_953);
    let mut hash = Sha256::new();
    for &v in session.logits().unwrap() {
        hash.update(v.to_le_bytes());
    }
    println!(
        "QWEN35_9B_HEAD_BINDING {}",
        serde_json::json!({"load_ns":load_ns,"bind_ns":bind_ns,"forward_observed_ns":forward_ns,"memory":memory,"work":session.work(),"source_bytes":source_ids.container_bytes(),"retained_capacity_bytes":store.report().retained_tensor_capacity_bytes,"distinct_head":true,"source_head_dispatch_bits_equal":true,"logits_sha256":format!("{:x}",hash.finalize()),"independent_full_graph_qualified":false,"quality_qualified":false,"managed_adoption_qualified":false})
    );
}
