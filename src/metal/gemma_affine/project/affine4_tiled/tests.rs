use super::*;
use crate::metal::gemma_affine::tests::{fixture, l1_fixture};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    io::{BufWriter, Read, Write},
    path::PathBuf,
};
const LIMIT: usize = 512 * 1024 * 1024;
const ORACLE: usize = 128 * 1024 * 1024;
const HELPER: usize = 32 * 1024 * 1024;

#[test]
fn q4_strided_geometry_and_general_fp32_codebook_proof() {
    for rows in [0, 262145, usize::MAX] {
        assert!(Geometry::new(rows, 64, 1).is_err());
    }
    for columns in [0, 63, 65, 12352, usize::MAX] {
        assert!(Geometry::new(1, columns, 1).is_err());
    }
    for tokens in [0, 33, usize::MAX] {
        assert!(Geometry::new(1, 64, tokens).is_err());
    }
    let g = Geometry::new(262144, 12288, 32).unwrap();
    let last = Layout::new(16384, 12288, 4, 64, MetadataDtype::BF16).unwrap();
    let c = g.chunk(last, 245760).unwrap();
    assert_eq!(c.words, [32, 12288, 16384, 262144, 245760]);
    assert_eq!((c.codes, c.metadata, c.groups), (25165824, 1572864, 524288));
    assert!(g.chunk(last, 245761).is_err());
    let maximum = f32::from_bits(0x3f95_7000); // all16-code maximum, not BF16
    let proof = InputProof::new(1536, f64::from(maximum)).unwrap();
    assert_eq!(proof.maximum_bits, maximum.to_bits());
    assert_ne!(proof.maximum_bits & 65535, 0);
    for value in [
        f64::NAN,
        f64::INFINITY,
        -1.,
        f64::from(f32::from_bits(1)),
        f64::from(MAX_WEIGHT) * 2.,
        f64::from(maximum) + f64::EPSILON,
    ] {
        assert!(InputProof::new(1536, value).is_err());
    }
    let zero = InputProof::new(64, 0.).unwrap();
    assert_eq!(zero.max_bits, MAX_FINITE_BITS);
    assert_eq!(zero.l1_bits, MAX_FINITE_BITS);
    for bits in [proof.max_bits, proof.max_bits + 1] {
        let (s, u) = bounds(1536, f64::from(maximum), f32::from_bits(bits));
        assert_eq!(
            s <= f64::from(f32::MAX) / 4. && u <= ORDERED_ERROR_LIMIT,
            bits == proof.max_bits
        );
    }
}
#[test]
fn q4_status_contracts_precede_ordered_reference_and_speculative_product_fatal() {
    assert_eq!(decision(0, 0).unwrap(), Decision::Ready);
    assert_eq!(decision(2, 0).unwrap(), Decision::ReferenceReplay);
    assert_eq!(decision(2, 1).unwrap(), Decision::ReferenceReplay);
    for (a, p) in [(1, 0), (3, 1), (0, 1), (8, 0), (2, 2), (0, 2)] {
        assert!(decision(a, p).is_err());
    }
}

fn checked(command: &ProtocolObject<dyn MTLCommandBuffer>) {
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert!(command.error().is_none());
}
fn complete(command: &ProtocolObject<dyn MTLCommandBuffer>) {
    command.commit();
    command.waitUntilCompleted();
    checked(command);
}
fn hash_words(hash: &mut Sha256, words: &[u32]) {
    let mut bytes = [0; 4096];
    for chunk in words.chunks(1024) {
        for (dst, word) in bytes.chunks_exact_mut(4).zip(chunk) {
            dst.copy_from_slice(&word.to_le_bytes());
        }
        hash.update(&bytes[..chunk.len() * 4]);
    }
}
fn owner_hash(owner: &Weights) -> String {
    let mut hash = Sha256::new();
    for view in owner.views() {
        for buffer in [view.codes, view.scales, view.biases] {
            buffer.with_slice(|s| hash_words(&mut hash, s));
        }
    }
    format!("{:x}", hash.finalize())
}
fn guards(owner: &Weights) {
    for view in owner.views() {
        let g = Geometry::new(owner.layout().rows(), owner.layout().columns(), 1)
            .unwrap()
            .chunk(view.layout, 0)
            .unwrap();
        for (buffer, n) in [view.codes, view.scales, view.biases]
            .into_iter()
            .zip([g.codes, g.metadata, g.metadata])
        {
            buffer.with_slice(|s| assert!(s[n..].iter().all(|&w| w == SENTINEL)));
        }
    }
}
fn weights(
    context: &Context,
    layout: Layout,
    chunk_rows: usize,
    codes: &[u8],
    scales: &[u8],
    biases: &[u8],
) -> Result<Weights> {
    let per_codes = layout.weight_bytes() / layout.rows();
    let per_metadata = layout.metadata_bytes() / layout.rows();
    context.weights_from_fill(
        layout,
        chunk_rows,
        64 * 1024 * 1024,
        |first, child, c, s, b| {
            let rows = child.rows();
            c.copy_from_slice(&codes[first * per_codes..(first + rows) * per_codes]);
            s.copy_from_slice(&scales[first * per_metadata..(first + rows) * per_metadata]);
            b.copy_from_slice(&biases[first * per_metadata..(first + rows) * per_metadata]);
            Ok(())
        },
    )
}
struct Scratch {
    input: Buffer<u32>,
    output: Buffer<u32>,
    status: Buffer<u32>,
    tokens: usize,
    rows: usize,
    columns: usize,
}
impl Scratch {
    fn new(context: &Context, rows: usize, columns: usize, tokens: usize) -> Self {
        let mut s = Self {
            input: Buffer::new(&context.gpu, tokens * columns + 2 * GUARD).unwrap(),
            output: Buffer::new(&context.gpu, tokens * rows + 2 * GUARD).unwrap(),
            status: Buffer::new(&context.gpu, 2 + 2 * GUARD).unwrap(),
            tokens,
            rows,
            columns,
        };
        s.reset(&vec![0.; tokens * columns]);
        s
    }
    fn reset(&mut self, input: &[f32]) {
        assert_eq!(input.len(), self.tokens * self.columns);
        self.input.with_mut(|s| {
            s.fill(SENTINEL);
            for (dst, x) in s[GUARD..GUARD + input.len()].iter_mut().zip(input) {
                *dst = x.to_bits();
            }
        });
        self.output.with_mut(|s| s.fill(SENTINEL));
        self.status.with_mut(|s| {
            s.fill(SENTINEL);
            s[GUARD..GUARD + 2].fill(0);
        });
    }
    fn buffers(&self) -> DeviceBuffers<'_> {
        DeviceBuffers {
            input: BufferRange::new(&self.input, GUARD, self.tokens * self.columns).unwrap(),
            output: BufferRange::new(&self.output, GUARD, self.tokens * self.rows).unwrap(),
            admission_status: BufferRange::new(&self.status, GUARD, 1).unwrap(),
            product_status: BufferRange::new(&self.status, GUARD + 1, 1).unwrap(),
        }
    }
    fn statuses(&self) -> (u32, u32) {
        self.status.with_slice(|s| (s[GUARD], s[GUARD + 1]))
    }
    fn guards(&self) {
        for (buffer, n) in [
            (&self.input, self.tokens * self.columns),
            (&self.output, self.tokens * self.rows),
            (&self.status, 2),
        ] {
            buffer.with_slice(|s| {
                assert!(
                    s[..GUARD]
                        .iter()
                        .chain(&s[GUARD + n..])
                        .all(|&w| w == SENTINEL)
                );
            });
        }
    }
    fn bytes(&self) -> usize {
        (self.input.len + self.output.len + self.status.len) * 4 + size_of::<Self>()
    }
}
struct Raw {
    root: PathBuf,
    file: BufWriter<std::fs::File>,
    bytes: usize,
    cases: Vec<Value>,
}
impl Raw {
    fn new() -> Self {
        let root = PathBuf::from(std::env::var_os("TOKLET_AFFINE4_BORROWED_RAW_DIR").unwrap());
        std::fs::create_dir(&root).unwrap();
        let file = BufWriter::new(std::fs::File::create_new(root.join("arrays.bin")).unwrap());
        Self {
            root,
            file,
            bytes: 0,
            cases: Vec::new(),
        }
    }
    fn words(&mut self, words: &[u32]) -> Value {
        assert!(self.bytes + words.len() * 4 <= 256 * 1024 * 1024);
        let offset = self.bytes;
        let mut hash = Sha256::new();
        let mut block = [0; 4096];
        for chunk in words.chunks(1024) {
            for (dst, w) in block.chunks_exact_mut(4).zip(chunk) {
                dst.copy_from_slice(&w.to_le_bytes());
            }
            let bytes = &block[..chunk.len() * 4];
            self.file.write_all(bytes).unwrap();
            hash.update(bytes);
        }
        self.bytes += words.len() * 4;
        json!({"offset":offset,"bytes":words.len()*4,"words":words.len(),"sha256":format!("{:x}",hash.finalize())})
    }
    fn generated_words(&mut self, count: usize, mut word: impl FnMut(usize) -> u32) -> Value {
        assert!(self.bytes + count * 4 <= 256 * 1024 * 1024);
        let offset = self.bytes;
        let mut hash = Sha256::new();
        let mut block = [0; 4096];
        for first in (0..count).step_by(1024) {
            let n = (count - first).min(1024);
            for (index, dst) in block[..n * 4].chunks_exact_mut(4).enumerate() {
                dst.copy_from_slice(&word(first + index).to_le_bytes());
            }
            self.file.write_all(&block[..n * 4]).unwrap();
            hash.update(&block[..n * 4]);
        }
        self.bytes += count * 4;
        json!({"offset":offset,"bytes":count*4,"words":count,"sha256":format!("{:x}",hash.finalize())})
    }
    fn floats(&mut self, values: &[f32]) -> Value {
        let offset = self.bytes;
        let mut hash = Sha256::new();
        let mut block = [0; 4096];
        assert!(self.bytes + values.len() * 4 <= 256 * 1024 * 1024);
        for chunk in values.chunks(1024) {
            for (dst, x) in block.chunks_exact_mut(4).zip(chunk) {
                dst.copy_from_slice(&x.to_le_bytes());
            }
            let bytes = &block[..chunk.len() * 4];
            self.file.write_all(bytes).unwrap();
            hash.update(bytes);
        }
        self.bytes += values.len() * 4;
        json!({"offset":offset,"bytes":values.len()*4,"words":values.len(),"sha256":format!("{:x}",hash.finalize())})
    }
    fn finish(mut self) {
        self.file.flush().unwrap();
        drop(self.file);
        let mut hash = Sha256::new();
        let mut file = std::fs::File::open(self.root.join("arrays.bin")).unwrap();
        let mut block = [0; 65536];
        loop {
            let n = file.read(&mut block).unwrap();
            if n == 0 {
                break;
            }
            hash.update(&block[..n]);
        }
        let result = json!({"format":"toklet-affine4-borrowed-v1","passed":true,"cases":self.cases,"raw":{"path":self.root.join("arrays.bin"),"bytes":self.bytes,"sha256":format!("{:x}",hash.finalize())},"no_default_route":true,"no_graph_quality_or_performance_claim":true,"numeric_gate":{"absolute":1e-5,"relative":1e-4,"relaxed":false}});
        let mut file = std::fs::File::create_new(self.root.join("result.json")).unwrap();
        serde_json::to_writer_pretty(&mut file, &result).unwrap();
        file.write_all(b"\n").unwrap();
    }
}
// Minimal test transaction boundary, separate from the encoder. The real
// graph must retain its own pending/poison/cancel/publication owner.
fn publish_trial(
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    encoded: Result<usize>,
    scratch: &Scratch,
    caller: &mut [f32],
    cancel: bool,
) -> Result<Decision> {
    if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
        return Err(Error::Completion);
    }
    for (buffer, n) in [
        (&scratch.input, scratch.tokens * scratch.columns),
        (&scratch.output, scratch.tokens * scratch.rows),
        (&scratch.status, 2),
    ] {
        if !buffer.with_slice(|s| {
            s[..GUARD]
                .iter()
                .chain(&s[GUARD + n..])
                .all(|&w| w == SENTINEL)
        }) {
            return Err(Error::Options("Q4 trial guard/inactive integrity differs"));
        }
    }
    encoded?;
    let (admission, product) = scratch.statuses();
    let decision = decision(admission, product)?;
    if cancel || caller.len() != scratch.tokens * scratch.rows {
        return Err(Error::Options(
            "Q4 trial cancellation/output extent differs",
        ));
    }
    if decision == Decision::Ready {
        let finite = scratch.output.with_slice(|s| {
            s[GUARD..GUARD + caller.len()]
                .iter()
                .all(|&w| f32::from_bits(w).is_finite())
        });
        if !finite {
            return Err(Error::Nonfinite);
        }
        let count = caller.len();
        scratch.output.with_slice(|s| {
            for (dst, &w) in caller.iter_mut().zip(&s[GUARD..GUARD + count]) {
                *dst = f32::from_bits(w);
            }
        });
    }
    Ok(decision)
}

fn gate(actual: f32, reference: f64) {
    assert!(actual.is_finite() && reference.is_finite());
    assert!((f64::from(actual) - reference).abs() <= 1e-5 + 1e-4 * reference.abs());
}

// Every target uses the same loaded fixture owner as the unchanged original
// synchronous oracle; direct live CommonQ4 graph integration is a later task.

fn exercise(
    context: &Context,
    pipeline: &Affine4Pipelines,
    owner: &Weights,
    input: &[f32],
    tokens: usize,
    name: &str,
    raw: &mut Raw,
) -> (Vec<f32>, bool, Value) {
    let layout = owner.layout();
    let matrix = Affine4MatrixView::new(context, owner).unwrap();
    let hash_before = owner_hash(owner);
    let mut scratch = Scratch::new(context, layout.rows(), layout.columns(), tokens);
    scratch.reset(input);
    let mut oracle = context
        .projection_session(layout, matrix.maximum_chunk_rows(), tokens, ORACLE)
        .unwrap();
    let mut reference = vec![99.; tokens * layout.rows()];
    let mut published = vec![99.; tokens * layout.rows()];
    let ledger = CONTEXT_ALLOWANCE_BYTES
        + Affine4Pipelines::policy_bytes()
        + HELPER
        + owner.owned_bytes()
        + scratch.bytes()
        + oracle.owned_bytes()
        + reference.capacity() * 4
        + published.capacity() * 4
        + input.len() * 4;
    assert!(
        ledger <= LIMIT,
        "admitted qualification owner/policy sum{ledger}"
    );
    let command = context.gpu.command().unwrap();
    let dispatches = pipeline
        .encode(&command, matrix, scratch.buffers(), tokens)
        .unwrap();
    assert_eq!(command.status(), MTLCommandBufferStatus::NotEnqueued);
    complete(&command);
    scratch.guards();
    guards(owner);
    let statuses = scratch.statuses();
    let selected = decision(statuses.0, statuses.1);
    let host_admitted = input.iter().all(|x| x.is_finite())
        && admit(layout.columns(), owner.max_abs_weight(), input).is_ok();
    let reference_start = std::time::Instant::now();
    let reference_result = oracle.apply(owner, input, &mut reference, tokens);
    let reference_seconds = reference_start.elapsed().as_secs_f64();
    let reference_ok = reference_result.is_ok();
    let reference_error = reference_result.as_ref().err().map(ToString::to_string);
    // Capture both candidate and same-owner original results before any gate;
    // even a rejected input retains its unchanged99 caller-reference output.
    let input_raw = scratch
        .input
        .with_slice(|s| raw.words(&s[GUARD..GUARD + input.len()]));
    let candidate_raw = scratch
        .output
        .with_slice(|s| raw.words(&s[GUARD..GUARD + published.len()]));
    let reference_raw = raw.floats(&reference);
    let status_raw = scratch.status.with_slice(|s| raw.words(s));
    println!(
        "AFFINE4_BORROWED_CANDIDATE {}",
        json!({"name":name,"rows":layout.rows(),"columns":layout.columns(),"tokens":tokens,"input":input_raw,"candidate":candidate_raw,"original_reference":reference_raw,"guarded_status":status_raw,"admission_status":statuses.0,"product_status":statuses.1,"reference_ok":reference_ok,"reference_error":reference_error})
    );
    if reference_ok {
        match selected.as_ref().unwrap() {
            Decision::Ready => scratch.output.with_slice(|s| {
                for (index, (&word, &expected)) in s[GUARD..GUARD + reference.len()]
                    .iter()
                    .zip(&reference)
                    .enumerate()
                {
                    assert_eq!(
                        word,
                        expected.to_bits(),
                        "same-owner original bits differ at{index}"
                    );
                }
                for (dst, &word) in published.iter_mut().zip(&s[GUARD..GUARD + reference.len()]) {
                    *dst = f32::from_bits(word);
                }
            }),
            Decision::ReferenceReplay => published.copy_from_slice(&reference),
        }
    } else {
        assert!(published.iter().all(|&x| x == 99.));
        // A candidate admission miss may return the exact unchanged original
        // host rejection. It does not exempt any rejected numerical family.
        assert!(
            selected.is_err() || matches!(selected, Ok(Decision::ReferenceReplay)),
            "GPU admitted outside original host proof domain"
        );
    }
    scratch.guards();
    guards(owner);
    assert_eq!(owner_hash(owner), hash_before);
    let published_raw = raw.floats(&published);
    let mut record = json!({"name":name,"rows":layout.rows(),"columns":layout.columns(),"tokens":tokens,"chunk_rows":owner.views().map(|v|v.layout.rows()).collect::<Vec<_>>(),"chunks":matrix.chunk_count(),"maximum_bits":matrix.maximum_bits(),"max_input_bits":matrix.proof.max_bits,"L1_input_bits":matrix.proof.l1_bits,"input":input_raw,"candidate":candidate_raw,"original_reference":reference_raw,"published":published_raw,"guarded_status":status_raw,"admission_status":statuses.0});
    for extra in [
        json!({"product_status":statuses.1,"candidate_commands":1,"candidate_waits":1,"candidate_dispatches":dispatches,"reference_calls_for_validation_or_replay":1,"reference_commands":if reference_ok{Some(matrix.chunk_count())}else if !host_admitted{Some(0)}else{None},"reference_waits":if reference_ok{Some(matrix.chunk_count())}else if !host_admitted{Some(0)}else{None},"reference_input_copy_bytes":if reference_ok{Some(matrix.chunk_count()*input.len()*4)}else if !host_admitted{Some(0)}else{None},"reference_readback_bytes":if reference_ok{Some(tokens*layout.rows()*4)}else if !host_admitted{Some(0)}else{None},"reference_seconds":reference_seconds,"candidate_coefficient_copy_bytes":0,"caller_active_input_stage_bytes":input.len()*4,"copy_scope":"logical active copies excluding cold initialization, not physical bus traffic","reference_ok":reference_ok,"reference_error":reference_error}),
        json!({"whole_case_replay":matches!(selected,Ok(Decision::ReferenceReplay)) && reference_ok,"caller_output_unchanged_on_failure":!reference_ok,"source_hash_before":hash_before,"source_hash_after":owner_hash(owner),"owner_bytes":owner.owned_bytes(),"scratch_bytes":scratch.bytes(),"reference_owned_bytes":oracle.owned_bytes()+reference.capacity()*4,"publication_owned_bytes":published.capacity()*4,"logical_owner_policy_admission_bytes":ledger,"context_policy_bytes":CONTEXT_ALLOWANCE_BYTES,"pipeline_policy_bytes":Affine4Pipelines::policy_bytes(),"native_PSO_bytes_known":false}),
    ] {
        let Value::Object(fields) = extra else {
            unreachable!("capture literal must be an object");
        };
        record.as_object_mut().unwrap().extend(fields);
    }
    println!("AFFINE4_BORROWED_CASE {record}");
    raw.cases.push(record.clone());
    (published, reference_ok, record)
}

#[test]
#[ignore = "explicit original31 affine families+19L1 families+6native arithmetic cases, same-owner borrowed encoding"]
fn original_frozen_and_native_families_preserve_reference_and_rejection() {
    autoreleasepool(|_| {
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let pipeline = Affine4Pipelines::new(&context, Affine4Pipelines::policy_bytes()).unwrap();
        let mut raw = Raw::new();
        for (family, fixture) in fixture::fixtures().into_iter().enumerate() {
            for case in &fixture.cases {
                autoreleasepool(|_| {
                    let name = format!("frozen{family}/{}", case["name"].as_str().unwrap());
                    let layout = fixture.layout(case);
                    let result = weights(
                        &context,
                        layout,
                        layout.rows().min(MAX_CHUNK_ROWS),
                        fixture.bytes(case, "weight"),
                        fixture.bytes(case, "scales"),
                        fixture.bytes(case, "biases"),
                    );
                    if case["constructor"]["admitted"] != true {
                        assert!(result.is_err());
                        raw.cases.push(json!({"name":name,"constructor_rejected":true,"GPU_commands":0,"error":result.err().unwrap().to_string()}));
                        return;
                    }
                    let owner = result.unwrap();
                    let input = fixture.input(case);
                    let tokens = case["batch"].as_u64().unwrap() as usize;
                    let (published, okay, _) =
                        exercise(&context, &pipeline, &owner, &input, tokens, &name, &mut raw);
                    assert_eq!(
                        okay,
                        case["input_admission"]["admitted"] == true,
                        "original source acceptance{name}"
                    );
                    if okay {
                        for key in ["math_fp64", "ordered_fp64"] {
                            let expected = case[key].as_array().unwrap();
                            assert_eq!(published.len(), expected.len());
                            for (&a, e) in published.iter().zip(expected) {
                                gate(a, e.as_f64().unwrap());
                            }
                        }
                    }
                });
            }
        }
        for case in l1_fixture::cases() {
            autoreleasepool(|_| {
                let owner = weights(
                    &context,
                    case.layout,
                    case.layout.rows(),
                    &case.codes,
                    &case.scales,
                    &case.biases,
                )
                .unwrap();
                let tokens = case.reference["batch"].as_u64().unwrap() as usize;
                let name = format!("L1/{}", case.reference["name"].as_str().unwrap());
                let (published, okay, _) = exercise(
                    &context,
                    &pipeline,
                    &owner,
                    &case.input,
                    tokens,
                    &name,
                    &mut raw,
                );
                assert_eq!(okay, case.reference["combined_admitted"] == true);
                if okay {
                    for key in ["ordered_fp64", "mathematical_fp64"] {
                        let expected = case.reference[key].as_array().unwrap();
                        assert_eq!(published.len(), expected.len());
                        for (&a, e) in published.iter().zip(expected) {
                            gate(a, e.as_f64().unwrap());
                        }
                    }
                }
            });
        }
        for (name, weight, first, second) in [
            ("subnormal-positive", 16777216., f32::from_bits(1), 0.),
            ("subnormal-negative", 16777216., -f32::from_bits(1), 0.),
            (
                "two-subnormal-products",
                0.5,
                f32::MIN_POSITIVE,
                f32::MIN_POSITIVE,
            ),
            ("single-subnormal-product", 0.5, f32::MIN_POSITIVE, 0.),
            (
                "normal-residual",
                1.0078125,
                f32::from_bits(0x3f80_0001),
                -1.,
            ),
            ("positive-zero", 1., 0., 0.),
        ] {
            autoreleasepool(|_| {
                let weight: f32 = weight;
                let layout = Layout::new(1, 64, 4, 64, MetadataDtype::BF16).unwrap();
                let bias = ((weight.to_bits() >> 16) as u16).to_le_bytes();
                let owner = weights(&context, layout, 1, &[0; 32], &[0; 2], &bias).unwrap();
                let mut input = [0.; 64];
                input[0] = first;
                input[1] = second;
                let (_, okay, _) = exercise(
                    &context,
                    &pipeline,
                    &owner,
                    &input,
                    1,
                    &format!("native/{name}"),
                    &mut raw,
                );
                assert!(okay);
            });
        }
        assert_eq!(raw.cases.len(), 56);
        raw.finish();
    });
}

fn dyadic_owner(context: &Context, rows: usize, columns: usize, chunk: usize) -> Weights {
    let layout = Layout::new(rows, columns, 4, 64, MetadataDtype::BF16).unwrap();
    context
        .weights_from_fill(
            layout,
            chunk,
            64 * 1024 * 1024,
            |first, child, codes, scales, biases| {
                codes.fill(0);
                for local in 0..child.rows() {
                    let row = first + local;
                    codes[local * columns / 2] = (row % 16) as u8;
                    let scale = if row.is_multiple_of(2) {
                        1_f32 / 64.
                    } else {
                        -1_f32 / 64.
                    };
                    let bias = (row / chunk + 1) as f32 / 64.;
                    for group in 0..columns / 64 {
                        let index = (local * (columns / 64) + group) * 2;
                        scales[index..index + 2]
                            .copy_from_slice(&((scale.to_bits() >> 16) as u16).to_le_bytes());
                        biases[index..index + 2]
                            .copy_from_slice(&((bias.to_bits() >> 16) as u16).to_le_bytes());
                    }
                }
                Ok(())
            },
        )
        .unwrap()
}
fn sparse_input(tokens: usize, columns: usize) -> Vec<f32> {
    let mut input = vec![0.; tokens * columns];
    for (token, row) in input.chunks_exact_mut(columns).enumerate() {
        row[0] = (token as f32 - 16.) / 1048576.;
    }
    input
}
fn dyadic_output(row: usize, token: usize, chunk: usize) -> f32 {
    let code = (row % 16) as i32;
    let signed = if row.is_multiple_of(2) { code } else { -code };
    let coefficient = (row / chunk + 1) as i32 + signed;
    let input = token as i32 - 16;
    if coefficient == 0 || input == 0 {
        0.
    } else {
        (coefficient * input) as f32 / 67108864.
    }
}
#[test]
#[ignore = "explicit complete nonuniform Q4 tiling including16384boundary/full262144x64 and usefulFF/PLE dimensions"]
fn complete_strided_synthetic_shapes_cover_every_output() {
    autoreleasepool(|_| {
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let pipeline = Affine4Pipelines::new(&context, Affine4Pipelines::policy_bytes()).unwrap();
        let mut raw = Raw::new();
        for (rows, columns, chunk) in [
            (16383, 64, 16384),
            (16384, 64, 16384),
            (16385, 64, 16384),
            (17, 1536, 7),
            (262144, 64, 16384),
            (8960, 1536, 16384),
            (12288, 1536, 16384),
            (1536, 12288, 16384),
        ] {
            for tokens in [1, 32] {
                autoreleasepool(|_| {
                    let owner = dyadic_owner(&context, rows, columns, chunk);
                    let input = sparse_input(tokens, columns);
                    let name = format!("dyadic-{rows}x{columns}-T{tokens}-chunk{chunk}");
                    // Materialize independent one-term dyadic expectations
                    // before the candidate has any command or dispatch.
                    let expected_raw = raw.generated_words(rows * tokens, |index| {
                        dyadic_output(index % rows, index / rows, chunk).to_bits()
                    });
                    let (published, okay, _) =
                        exercise(&context, &pipeline, &owner, &input, tokens, &name, &mut raw);
                    assert!(okay);
                    for (index, &actual) in published.iter().enumerate() {
                        let expected = dyadic_output(index % rows, index / rows, chunk);
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "independent one-nonzero dyadic product at{index}"
                        );
                    }
                    raw.cases.last_mut().unwrap()["expected_before_candidate"] = expected_raw;
                    raw.cases.last_mut().unwrap()["synthetic_dyadic"] = json!({"chunk_rows":chunk,"first_input_integer":"token-16","input_denominator":1048576,"code":"global_row%16 only at column0; other codes0","scale":"(-1)^global_row/64","bias":"(global_row/chunk_rows+1)/64","all_output_bits_checked":true,"selected_real_weights":false});
                });
            }
        }
        assert_eq!(raw.cases.len(), 16);
        raw.finish();
    });
}

#[test]
#[ignore = "explicit non-BF16 reconstructed and unused-code maximum/disjoint status/sticky/latefailure controls"]
fn checked_preflight_statuses_and_partial_failures_preserve_caller_output() {
    autoreleasepool(|_| {
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let pipeline = Affine4Pipelines::new(&context, Affine4Pipelines::policy_bytes()).unwrap();
        let mut raw = Raw::new();
        // q7 is used everywhere, while the authenticated maximum is at unused
        // q15. Its exact FP32 low16 are nonzero and must not be BF16-rounded.
        let layout = Layout::new(3, 64, 4, 64, MetadataDtype::BF16).unwrap();
        let codes = vec![0x77; layout.weight_bytes()];
        let scales = [0xcd, 0x3d].repeat(3);
        let biases = [0xab, 0xbe].repeat(3);
        let owner = weights(&context, layout, 2, &codes, &scales, &biases).unwrap();
        let view = Affine4MatrixView::new(&context, &owner).unwrap();
        assert_eq!(view.maximum_bits(), 0x3f95_7000);
        assert_ne!(view.maximum_bits() & 65535, 0);
        let input = vec![1_f32 / 1048576.; 64];
        let (published, okay, _) = exercise(
            &context,
            &pipeline,
            &owner,
            &input,
            1,
            "non-BF16-unused-code-max",
            &mut raw,
        );
        assert!(okay);
        for value in published {
            gate(value, 64. * 0.36669921875 / 1048576.);
        }
        let mut scratch = Scratch::new(&context, 3, 64, 1);
        scratch.reset(&input);
        for kind in 0..8 {
            let command = context.gpu.command().unwrap();
            let mut buffers = scratch.buffers();
            match kind {
                0 => buffers.output = BufferRange::new(&scratch.input, GUARD, 3).unwrap(),
                1 => buffers.product_status = buffers.admission_status,
                2 => buffers.admission_status = BufferRange::new(&scratch.input, GUARD, 1).unwrap(),
                3 => buffers.output = BufferRange::new(&scratch.output, GUARD, 2).unwrap(),
                4 => buffers.input = BufferRange::new(&scratch.input, GUARD, 63).unwrap(),
                5 => {
                    let codes = owner.views().next().unwrap().codes;
                    buffers.product_status = BufferRange::new(codes, codes.len - 1, 1).unwrap();
                }
                6 => {
                    buffers.output = BufferRange {
                        raw: buffers.output.raw,
                        offset: usize::MAX,
                        len: 3,
                    }
                }
                7 => buffers.output = BufferRange::new(&scratch.status, GUARD, 3).unwrap(),
                _ => unreachable!(),
            }
            let error = pipeline.encode(&command, view, buffers, 1).err().unwrap();
            complete(&command);
            scratch.guards();
            assert_eq!(scratch.statuses(), (0, 0));
            scratch
                .output
                .with_slice(|s| assert!(s.iter().all(|&w| w == SENTINEL)));
            raw.cases.push(json!({"name":format!("preflight-{kind}"),"error":error.to_string(),"commands_drained":1,"candidate_dispatches":0,"output_untouched":true,"status_untouched":true}));
        }
        for queued in [false, true] {
            let command = if queued {
                let c = context.gpu.command().unwrap();
                c.enqueue();
                c
            } else {
                context
                    .gpu
                    .queue
                    .commandBufferWithUnretainedReferences()
                    .unwrap()
            };
            assert!(
                pipeline
                    .encode(&command, view, scratch.buffers(), 1)
                    .is_err()
            );
            if queued {
                complete(&command);
            }
            assert_eq!(scratch.statuses(), (0, 0));
            scratch
                .output
                .with_slice(|s| assert!(s.iter().all(|&w| w == SENTINEL)));
            raw.cases.push(json!({"name":if queued{"queued-command-rejected-before-encoding"}else{"unretained-command-rejected-before-encoding"},"candidate_dispatches":0,"empty_command_drained":queued,"status_untouched":true,"output_untouched":true}));
        }
        // An initialized status is caller state. Encoding ORs it, never resets.
        scratch.status.with_mut(|s| s[GUARD] = 2);
        let command = context.gpu.command().unwrap();
        let count = pipeline
            .encode(&command, view, scratch.buffers(), 1)
            .unwrap();
        complete(&command);
        scratch.guards();
        assert_eq!(scratch.statuses().0, 2);
        assert_eq!(
            decision(scratch.statuses().0, scratch.statuses().1).unwrap(),
            Decision::ReferenceReplay
        );
        raw.cases.push(json!({"name":"sticky-admission-reference","dispatches":count,"statuses":scratch.statuses(),"caller_replay_required":true}));
        scratch.reset(&input);
        scratch.status.with_mut(|s| s[GUARD + 1] = 1);
        let command = context.gpu.command().unwrap();
        pipeline
            .encode(&command, view, scratch.buffers(), 1)
            .unwrap();
        complete(&command);
        scratch.guards();
        assert_eq!(scratch.statuses().1, 1);
        assert!(decision(scratch.statuses().0, scratch.statuses().1).is_err());
        raw.cases.push(json!({"name":"sticky-product-fatal","statuses":scratch.statuses(),"publishable":false}));
        // Deliberate malformed last-child metadata exercises the full first
        // pass. The ordinary owner API exposes no such mutable buffer access.
        let mut malformed = dyadic_owner(&context, 17, 64, 7);
        let valid = Affine4MatrixView::new(&context, &malformed).unwrap();
        let saved = (
            valid.layout,
            valid.chunks,
            valid.maximum_chunk_rows,
            valid.proof,
        );
        malformed.diagnostic_shorten_chunk_codes(2).unwrap();
        assert!(Affine4MatrixView::new(&context, &malformed).is_err());
        let fake = Affine4MatrixView {
            weights: &malformed,
            layout: saved.0,
            chunks: saved.1,
            maximum_chunk_rows: saved.2,
            proof: saved.3,
        };
        let mut small = Scratch::new(&context, 17, 64, 1);
        let input = sparse_input(1, 64);
        small.reset(&input);
        let command = context.gpu.command().unwrap();
        let error = pipeline
            .encode(&command, fake, small.buffers(), 1)
            .err()
            .unwrap();
        complete(&command);
        assert_eq!(small.statuses(), (0, 0));
        small
            .output
            .with_slice(|s| assert!(s.iter().all(|&w| w == SENTINEL)));
        small.guards();
        raw.cases.push(json!({"name":"late-bad-chunk-precedes-all-encoders","error":error.to_string(),"candidate_dispatches":0,"status_untouched":true,"output_untouched":true}));
        let valid_owner = dyadic_owner(&context, 17, 64, 7);
        let valid = Affine4MatrixView::new(&context, &valid_owner).unwrap();
        for index in [0, 1, 2] {
            autoreleasepool(|_| {
                small.reset(&input);
                pipeline.fail_before_chunk.set(Some(index));
                let command = context.gpu.command().unwrap();
                let encoded = pipeline.encode(&command, valid, small.buffers(), 1);
                pipeline.fail_before_chunk.set(None);
                assert!(encoded.is_err());
                complete(&command);
                let mut caller = vec![99.; 17];
                let error = publish_trial(&command, encoded, &small, &mut caller, false)
                    .err()
                    .unwrap();
                assert!(caller.iter().all(|&x| x == 99.));
                assert!(
                    matches!(&error,Error::Native(message) if message=="injected Q4 late encoder absence")
                );
                // The caller drains retained resources, discards all private
                // products and exposes no falsely exact aggregate count.
                small.guards();
                guards(&valid_owner);
                let caller_raw = raw.floats(&caller);
                raw.cases.push(json!({"name":format!("late-encoder-{index}"),"error":error.to_string(),"candidate_dispatches":Value::Null,"drained_commands":1,"drained_waits":1,"caller_output":caller_raw,"caller_output_untouched":true,"private_output_discarded":true,"requires_poison_or_new_transaction":true}));
            });
        }
        raw.finish();
    });
}

#[test]
#[ignore = "explicit32 warm same-owner borrowed Q4 encodes with zero Rust allocations"]
fn warmed_borrowed_q4_encoding_allocates_no_rust_memory() {
    autoreleasepool(|_| {
        use crate::model::gemma_reference::tests::Measurement;
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let pipeline = Affine4Pipelines::new(&context, Affine4Pipelines::policy_bytes()).unwrap();
        let owner = dyadic_owner(&context, 16385, 64, 16384);
        let matrix = Affine4MatrixView::new(&context, &owner).unwrap();
        let input = sparse_input(1, 64);
        let mut scratch = Scratch::new(&context, 16385, 64, 1);
        scratch.reset(&input);
        for _ in 0..2 {
            autoreleasepool(|_| {
                let command = context.gpu.command().unwrap();
                pipeline
                    .encode(&command, matrix, scratch.buffers(), 1)
                    .unwrap();
                complete(&command);
                scratch.guards();
            });
        }
        let measurement = Measurement::start();
        for _ in 0..32 {
            autoreleasepool(|_| {
                scratch.reset(&input);
                let command = context.gpu.command().unwrap();
                let count = pipeline
                    .encode(&command, matrix, scratch.buffers(), 1)
                    .unwrap();
                assert_eq!(count, 3);
                complete(&command);
                scratch.guards();
                assert_eq!(scratch.statuses(), (0, 0));
            });
        }
        let allocations = measurement.finish();
        assert_eq!(allocations, 0);
        let mut raw = Raw::new();
        let output = scratch
            .output
            .with_slice(|s| raw.words(&s[GUARD..GUARD + 16385]));
        raw.cases.push(json!({"name":"warm32-borrowed-Q4","iterations":32,"Rust_allocations":allocations,"candidate_commands":32,"candidate_waits":32,"dispatches_per_evaluation":3,"output":output,"native_allocations_measured":false}));
        raw.finish();
    });
}

fn bounded_fixture_file(path: &std::path::Path, bytes: usize, expected: &str) -> Vec<u8> {
    assert_eq!(path.metadata().unwrap().len(), bytes as u64);
    let mut value = Vec::with_capacity(bytes);
    std::fs::File::open(path)
        .unwrap()
        .take(bytes as u64 + 1)
        .read_to_end(&mut value)
        .unwrap();
    assert_eq!(value.len(), bytes);
    assert_eq!(format!("{:x}", Sha256::digest(&value)), expected);
    value
}
fn fixture_part<'a>(data: &'a [u8], desc: &Value, bytes: usize) -> &'a [u8] {
    let offset = usize::try_from(desc["offset"].as_u64().unwrap()).unwrap();
    assert_eq!(desc["bytes"].as_u64().unwrap(), bytes as u64);
    let value = data
        .get(offset..offset.checked_add(bytes).unwrap())
        .unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(value)),
        desc["sha256"].as_str().unwrap()
    );
    value
}
#[test]
#[ignore = "explicit bounded authenticated common-Q4 PLE matrix and six head coefficient row fixtures; published APIs only"]
fn authenticated_real_coefficient_fixtures_use_one_published_weights_owner() {
    autoreleasepool(|_| {
        let root = PathBuf::from(std::env::var_os("TOKLET_AFFINE4_REAL_FIXTURE").unwrap());
        let protocol: Value = serde_json::from_slice(&bounded_fixture_file(
            &root.join("protocol.json"),
            10931,
            "742a5e911af0692a72610f52ccb828bdb2b436a0b2d95fc78a49f3b59755554e",
        ))
        .unwrap();
        let data = bounded_fixture_file(
            &root.join("arrays.bin"),
            261184,
            "51f5a0d945dff7c3621020e1bd1db832e4d39426952941e1e9f74563326db75c",
        );
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let pipeline = Affine4Pipelines::new(&context, Affine4Pipelines::policy_bytes()).unwrap();
        let mut raw = Raw::new();
        for (key, rows, tokens, chunk) in [("PLE", 256, 4, 128), ("head_rows", 6, 1, 3)] {
            autoreleasepool(|_| {
                let arrays = &protocol[key]["arrays"];
                let layout = Layout::new(rows, 1536, 4, 64, MetadataDtype::BF16).unwrap();
                let codes = fixture_part(&data, &arrays["codes_u32"], layout.weight_bytes());
                let scales = fixture_part(&data, &arrays["scales_u32"], layout.metadata_bytes());
                let biases = fixture_part(&data, &arrays["biases_u32"], layout.metadata_bytes());
                let input_bytes = fixture_part(&data, &arrays["input"], tokens * 1536 * 4);
                let input = input_bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>();
                let independent = Matrix::from_le_bytes(layout, codes, scales, biases).unwrap();
                let mut expected = vec![0.; tokens * rows];
                for (x, y) in input
                    .chunks_exact(1536)
                    .zip(expected.chunks_exact_mut(rows))
                {
                    independent.matvec(x, y).unwrap();
                }
                // Independent original CPU result and saved PLE GPU identity
                // exist before any candidate dispatch on this fixture owner.
                let expected_raw = raw.floats(&expected);
                let owner = weights(&context, layout, chunk, codes, scales, biases).unwrap();
                let name = format!("real-fixture/{key}");
                let (published, okay, _) =
                    exercise(&context, &pipeline, &owner, &input, tokens, &name, &mut raw);
                assert!(okay);
                for (&a, &e) in published.iter().zip(&expected) {
                    gate(a, f64::from(e));
                }
                if key == "PLE" {
                    let saved =
                        fixture_part(&data, &arrays["expected_original_gpu"], tokens * rows * 4);
                    for (index, (&actual, bytes)) in
                        published.iter().zip(saved.chunks_exact(4)).enumerate()
                    {
                        assert_eq!(
                            actual.to_bits(),
                            u32::from_le_bytes(bytes.try_into().unwrap()),
                            "untouched saved original PLE bits at{index}"
                        );
                    }
                }
                let row = raw.cases.last_mut().unwrap();
                row["independent_CPU_before_candidate"] = expected_raw;
                row["source_fixture"] = json!({"protocol_sha256":"742a5e911af0692a72610f52ccb828bdb2b436a0b2d95fc78a49f3b59755554e","case":key,"canonical_identity_sha256":protocol["canonical_identity_sha256"],"selected_global_head_rows":if key=="head_rows"{protocol[key]["selected_global_rows"].clone()}else{Value::Null},"live_CommonQ4_owner_integration":false,"full_real_head_independent_coverage":false});
            });
        }
        assert_eq!(raw.cases.len(), 2);
        raw.finish();
    });
}

#[test]
#[ignore = "explicit readonly input sharing and real test-boundary cancellation/guard/status publication controls"]
fn readonly_sharing_and_checked_trial_publication() {
    autoreleasepool(|_| {
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let pipeline = Affine4Pipelines::new(&context, Affine4Pipelines::policy_bytes()).unwrap();
        let layout = Layout::new(64, 64, 4, 64, MetadataDtype::BF16).unwrap();
        let owner = weights(
            &context,
            layout,
            64,
            &vec![0; layout.weight_bytes()],
            &vec![0; layout.metadata_bytes()],
            &vec![0; layout.metadata_bytes()],
        )
        .unwrap();
        let matrix = Affine4MatrixView::new(&context, &owner).unwrap();
        let source = owner_hash(&owner);
        let input = vec![0.; 64];
        let mut scratch = Scratch::new(&context, 64, 64, 1);
        let mut raw = Raw::new();
        for (kind, name) in [
            "readonly-code-input-sharing",
            "canceled-before-publication",
            "guard-defeats-reference-request",
            "unknown-admission-status",
            "unknown-product-status-defeats-reference",
        ]
        .into_iter()
        .enumerate()
        {
            autoreleasepool(|_| {
                scratch.reset(&input);
                let command = context.gpu.command().unwrap();
                let mut buffers = scratch.buffers();
                if kind == 0 {
                    buffers.input =
                        BufferRange::new(owner.views().next().unwrap().codes, 0, 64).unwrap();
                }
                let encoded = pipeline.encode(&command, matrix, buffers, 1);
                complete(&command);
                if kind == 2 {
                    scratch.output.with_mut(|s| s[GUARD + 64] = 0);
                    scratch.status.with_mut(|s| s[GUARD] = 2);
                }
                if kind == 3 {
                    scratch.status.with_mut(|s| s[GUARD] = 8);
                }
                if kind == 4 {
                    scratch.status.with_mut(|s| s[GUARD + 1] = 2);
                    scratch.status.with_mut(|s| s[GUARD] = 2);
                }
                let mut caller = vec![99.; 64];
                let result = publish_trial(&command, encoded, &scratch, &mut caller, kind == 1);
                if kind == 0 {
                    assert_eq!(result.unwrap(), Decision::Ready);
                    assert!(caller.iter().all(|&x| x.to_bits() == 0));
                } else {
                    assert!(result.is_err());
                    assert!(caller.iter().all(|&x| x == 99.));
                }
                assert_eq!(owner_hash(&owner), source);
                guards(&owner);
                let caller_raw = raw.floats(&caller);
                let status_raw = scratch.status.with_slice(|s| raw.words(s));
                raw.cases.push(json!({"name":name,"caller_output":caller_raw,"guarded_status":status_raw,"caller_published":kind==0,"candidate_commands":1,"candidate_waits":1,"source_immutable":true,"trial_only_no_model_API":true}));
            });
        }
        raw.finish();
    });
}
