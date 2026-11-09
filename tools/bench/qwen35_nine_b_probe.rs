//! Same-source9B scalar/hybrid one-input comparison, selected at compile time.
use std::{
    fs::File,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use toklet::model::qwen35::{Execution, MatrixStorage, Memory, Options, Outcome, Session, Work};
const LIMIT: usize = 6 * 1024 * 1024 * 1024;
const DIAGNOSTIC: usize = 64 * 1024 * 1024;

fn config(folder: &Path) -> Vec<u8> {
    let file = File::open(folder.join("config.json")).unwrap();
    assert!(file.metadata().unwrap().is_file());
    assert!(file.metadata().unwrap().len() <= 65536);
    let mut bytes = Vec::new();
    file.take(65537).read_to_end(&mut bytes).unwrap();
    assert!(bytes.len() <= 65536);
    bytes
}
fn save(path: &Path, values: &[f32]) {
    let mut out = BufWriter::with_capacity(65536, File::create(path).unwrap());
    for v in values {
        out.write_all(&v.to_le_bytes()).unwrap();
    }
    out.flush().unwrap();
    out.get_ref().sync_all().unwrap();
}
fn compare(expected: &mut Option<Vec<f32>>, actual: &[f32]) {
    if let Some(old) = expected {
        assert_eq!(old.len(), actual.len());
        assert!(
            old.iter()
                .zip(actual)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    } else {
        *expected = Some(actual.to_vec());
    }
}
fn emit(ordinal: usize, ns: u128, output_values: usize, m: Memory, w: Work) {
    assert!(m.admitted_bytes + DIAGNOSTIC <= LIMIT);
    println!(
        "NINE_B_SAMPLE {{\"ordinal\":{ordinal},\"warmup\":{},\"forward_ns\":{ns},\"output_values\":{output_values},\"weights_model_bytes\":{},\"recurrent_bytes\":{},\"convolution_bytes\":{},\"kv_bytes\":{},\"scratch_bytes\":{},\"logits_bytes\":{},\"history_bytes\":{},\"metadata_bytes\":{},\"metadata_allowance_bytes\":{},\"admitted_bytes\":{},\"diagnostic_allowance_bytes\":{DIAGNOSTIC},\"tokens_completed\":{},\"tokens_committed\":{},\"projections_completed\":{},\"gathers_completed\":{},\"convolution_steps_completed\":{},\"recurrence_steps_completed\":{},\"attention_heads_completed\":{},\"repeated_logits_bits_equal\":true,\"numerical_adoption_qualified\":false,\"managed_adoption_qualified\":false}}",
        ordinal < 3,
        m.weights_and_model_bytes,
        m.recurrent_bank_bytes,
        m.convolution_bank_bytes,
        m.kv_bytes,
        m.scratch_bytes,
        m.logits_bytes,
        m.history_bytes,
        m.metadata_bytes,
        m.metadata_allowance_bytes,
        m.admitted_bytes,
        w.tokens_completed,
        w.tokens_committed,
        w.projections_completed,
        w.gathers_completed,
        w.convolution_steps_completed,
        w.recurrence_steps_completed,
        w.attention_heads_completed
    );
}

fn measure<M: MatrixStorage, E: Execution<M>>(session: &mut Session<'_, '_, M, E>, output: &Path) {
    let mut expected = None;
    for ordinal in 0..13 {
        session.reset();
        let start = Instant::now();
        let ns;
        let n;
        {
            let values = session.forward(&[1]).unwrap();
            ns = start.elapsed().as_nanos();
            n = values.len();
            compare(&mut expected, values);
            if ordinal == 12 {
                save(output, values);
            }
        }
        let work = session.work();
        assert!(matches!(work.outcome, Outcome::Success));
        assert_eq!(work.tokens_completed, 1);
        assert_eq!(work.tokens_committed, 1);
        assert_eq!(work.projections_completed, 249);
        assert_eq!(work.gathers_completed, 1);
        assert_eq!(work.convolution_steps_completed, 24);
        assert_eq!(work.recurrence_steps_completed, 24);
        assert_eq!(work.attention_heads_completed, 128);
        emit(ordinal, ns, n, session.memory(), work);
    }
}
#[cfg(not(toklet_metal9b))]
fn run(folder: &Path, output: &Path) {
    use toklet::{
        mlx::qwen35::{ModalityPolicy, nine_b::payload},
        model::qwen35::Model,
    };
    let config = config(folder);
    let start = Instant::now();
    let store = payload::load(
        &config,
        File::open(folder.join("model.safetensors")).unwrap(),
        payload::LoadOptions {
            memory_limit_bytes: LIMIT - DIAGNOSTIC,
            modality: ModalityPolicy::TextOnly,
        },
    )
    .unwrap();
    let load_ns = start.elapsed().as_nanos();
    assert_eq!(store.report().retained_tensor_capacity_bytes, 5_040_162_816);
    let start = Instant::now();
    let model = Model::from_nine_b_cpu_store(&store).unwrap();
    let bind_ns = start.elapsed().as_nanos();
    let mut session = model
        .session(Options {
            capacity: 512,
            chunk: 32,
            position_offset: 0,
            memory_limit_bytes: LIMIT - DIAGNOSTIC,
        })
        .unwrap();
    println!(
        "NINE_B_INITIAL {{\"backend\":\"cpu\",\"load_ns\":{load_ns},\"context_ns\":0,\"upload_ns\":0,\"bind_ns\":{bind_ns},\"source_bytes\":{},\"gpu_matrices\":0,\"gpu_weight_bytes\":0,\"retained_cpu_matrix_bytes\":5035917312}}",
        store.identity().container_bytes()
    );
    measure(&mut session, output);
}
#[cfg(toklet_metal9b)]
fn run(folder: &Path, output: &Path) {
    use toklet::{
        metal::gemma_affine::{CONTEXT_ALLOWANCE_BYTES, Context},
        mlx::qwen35::{ModalityPolicy, nine_b::payload},
        model::qwen35::MetalModel,
    };
    let config = config(folder);
    let start = Instant::now();
    let store = payload::load(
        &config,
        File::open(folder.join("model.safetensors")).unwrap(),
        payload::LoadOptions {
            memory_limit_bytes: LIMIT - DIAGNOSTIC - CONTEXT_ALLOWANCE_BYTES,
            modality: ModalityPolicy::TextOnly,
        },
    )
    .unwrap();
    let load_ns = start.elapsed().as_nanos();
    let start = Instant::now();
    let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let context_ns = start.elapsed().as_nanos();
    let start = Instant::now();
    let store = store.into_metal(&context, LIMIT - DIAGNOSTIC).unwrap();
    let upload_ns = start.elapsed().as_nanos();
    let report = store.report();
    assert_eq!(report.matrices, 250);
    assert_eq!(report.retained_cpu_matrix_bytes, 0);
    let start = Instant::now();
    let model = MetalModel::from_nine_b_metal_store(&store).unwrap();
    let bind_ns = start.elapsed().as_nanos();
    let mut session = model
        .session(
            &context,
            Options {
                capacity: 512,
                chunk: 32,
                position_offset: 0,
                memory_limit_bytes: LIMIT - DIAGNOSTIC,
            },
        )
        .unwrap();
    println!(
        "NINE_B_INITIAL {{\"backend\":\"metal_hybrid\",\"load_ns\":{load_ns},\"context_ns\":{context_ns},\"upload_ns\":{upload_ns},\"bind_ns\":{bind_ns},\"source_bytes\":{},\"gpu_matrices\":{},\"gpu_weight_bytes\":{},\"retained_cpu_matrix_bytes\":{},\"admitted_load_policy_bytes\":{}}}",
        store.identity().container_bytes(),
        report.matrices,
        report.packed_gpu_bytes,
        report.retained_cpu_matrix_bytes,
        report.admitted_load_policy_bytes
    );
    measure(&mut session, output);
}
fn main() {
    let folder = PathBuf::from(std::env::args_os().nth(1).expect("model directory"));
    let output = PathBuf::from(std::env::args_os().nth(2).expect("fresh output"));
    assert!(!output.exists());
    run(&folder, &output);
}
