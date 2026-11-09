//! Matched4B Metal regression or9B CPU untied diagnostic, selected at compile time.
use std::{
    fs::File,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use toklet::model::qwen35::{Memory, Options, Work};
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
        "HEAD_SAMPLE {{\"ordinal\":{ordinal},\"warmup\":{},\"forward_ns\":{ns},\"output_values\":{output_values},\"weights_model_bytes\":{},\"recurrent_bytes\":{},\"convolution_bytes\":{},\"kv_bytes\":{},\"scratch_bytes\":{},\"logits_bytes\":{},\"history_bytes\":{},\"metadata_bytes\":{},\"metadata_allowance_bytes\":{},\"admitted_bytes\":{},\"diagnostic_allowance_bytes\":{DIAGNOSTIC},\"tokens_completed\":{},\"tokens_committed\":{},\"projections_completed\":{},\"gathers_completed\":{},\"convolution_steps_completed\":{},\"recurrence_steps_completed\":{},\"attention_heads_completed\":{},\"repeated_logits_bits_equal\":true,\"numerical_adoption_qualified\":false,\"managed_adoption_qualified\":false}}",
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
#[cfg(toklet_9b)]
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
    let mut expected = None;
    println!(
        "HEAD_INITIAL {{\"mode\":\"9b_cpu_untied_one\",\"load_ns\":{load_ns},\"bind_ns\":{bind_ns},\"source_bytes\":{}}}",
        store.identity().container_bytes()
    );
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
        assert_eq!(session.work().tokens_committed, 1);
        assert_eq!(session.work().projections_completed, 249);
        emit(ordinal, ns, n, session.memory(), session.work());
    }
}
#[cfg(not(toklet_9b))]
fn run(folder: &Path, output: &Path) {
    use toklet::{
        metal::gemma_affine::{CONTEXT_ALLOWANCE_BYTES, Context},
        mlx::qwen35::{ModalityPolicy, payload},
        model::qwen35::MetalModel,
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
    let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let store = store.into_metal(&context, LIMIT - DIAGNOSTIC).unwrap();
    let load_ns = start.elapsed().as_nanos();
    let start = Instant::now();
    let model = MetalModel::from_metal_store(&store).unwrap();
    let bind_ns = start.elapsed().as_nanos();
    let mut session = model
        .session(
            &context,
            Options {
                capacity: 512,
                chunk: 32,
                position_offset: 37,
                memory_limit_bytes: LIMIT - DIAGNOSTIC,
            },
        )
        .unwrap();
    let mut expected = None;
    let ids = [1, 7, 13, 3, 9, 5, 11, 2, 17, 4, 6, 10, 19, 8, 21, 23, 29];
    println!(
        "HEAD_INITIAL {{\"mode\":\"4b_metal_tied_context17_offset37\",\"load_ns\":{load_ns},\"bind_ns\":{bind_ns},\"source_bytes\":{}}}",
        store.identity().container_bytes()
    );
    for ordinal in 0..13 {
        session.reset();
        let start = Instant::now();
        let ns;
        let n;
        {
            let values = session.forward(&ids).unwrap();
            ns = start.elapsed().as_nanos();
            n = values.len();
            compare(&mut expected, values);
            if ordinal == 12 {
                save(output, values);
            }
        }
        assert_eq!(session.work().tokens_committed, 17);
        assert_eq!(session.work().projections_completed, 17 * 249);
        emit(ordinal, ns, n, session.memory(), session.work());
    }
}
fn main() {
    let folder = PathBuf::from(std::env::args_os().nth(1).expect("model directory"));
    let output = PathBuf::from(std::env::args_os().nth(2).expect("fresh logit output"));
    assert!(!output.exists());
    run(&folder, &output);
}
