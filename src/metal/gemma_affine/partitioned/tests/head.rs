//! Ignored complete real tied-head diagnostic. No ordinary test opens a model.
use super::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

const N: usize = 262144;
const K: usize = 1536;
const B: usize = 4;
const TILE: usize = 16384;
const MODEL_BYTES: u64 = 3550670554;
const RAW_CAP: usize = 64 * 1024 * 1024;
const PROCESS_CAP: usize = 1024 * 1024 * 1024;
fn hash(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}
fn hash_words(words: impl Iterator<Item = u32>) -> String {
    // Bounded stack conversion, never a clone of an entire resident allocation.
    let mut digest = Sha256::new();
    let mut scratch = [0_u8; 4096];
    let mut used = 0;
    for word in words {
        scratch[used..used + 4].copy_from_slice(&word.to_le_bytes());
        used += 4;
        if used == scratch.len() {
            digest.update(scratch);
            used = 0;
        }
    }
    digest.update(&scratch[..used]);
    format!("{:x}", digest.finalize())
}
fn floats_hash(values: &[f32]) -> String {
    hash_words(values.iter().map(|v| v.to_bits()))
}
fn buffer_identity(buffer: &Buffer<u32>) -> Value {
    buffer.with_slice(|words| {
        json!({"allocation":format!("{:p}",&*buffer.raw),"words":words.len(),"bytes":words.len()*4,
            "sha256":hash_words(words.iter().copied()),"guards":words[words.len()-GUARD..].to_vec()})
    })
}
fn observation(owner: &PartitionedAffine<'_>) -> Value {
    let children: Vec<_> = owner.chunks.iter().map(|child| {
        let safe = child.pending.as_ref().is_none_or(|command| command.status() == MTLCommandBufferStatus::Completed);
        let device = safe.then(|| json!({"codes":buffer_identity(&child.codes),"scales":buffer_identity(&child.scales),"biases":buffer_identity(&child.biases),
            "input":buffer_identity(&child.input),"output":buffer_identity(&child.output),"status":buffer_identity(&child.status)}));
        json!({"rows":child.layout.rows(),"owned_bytes":child.owned_bytes(),"poisoned":child.is_poisoned(),"pending":child.pending.is_some(),"shared_read_safe":safe,"device":device})
    }).collect();
    json!({"rows":owner.rows,"columns":owner.columns,"capacity":owner.capacity,"owned_bytes":owner.owned_bytes(),"poisoned":owner.is_poisoned(),
        "owner_slots":owner.chunks.capacity(),"full_scratch_capacity":owner.staged.capacity(),"chunk_scratch_capacity":owner.chunk_output.capacity(),
        "staged_sha256":floats_hash(&owner.staged),"chunk_scratch_sha256":floats_hash(&owner.chunk_output),"children":children})
}
fn bounded(path: &Path, cap: usize, wanted: &str) -> Vec<u8> {
    let mut raw = Vec::new();
    File::open(path)
        .unwrap()
        .take(cap as u64 + 1)
        .read_to_end(&mut raw)
        .unwrap();
    assert!(raw.len() <= cap);
    assert_eq!(hash(&raw), wanted);
    raw
}
fn source(
    file: &mut File,
    header: &Value,
    name: &str,
    dtype: &str,
    shape: Value,
    bytes: usize,
) -> (Vec<u8>, Value) {
    let item = &header[name];
    assert_eq!(item["dtype"], dtype);
    assert_eq!(item["shape"], shape);
    let first = item["data_offsets"][0].as_u64().unwrap();
    let end = item["data_offsets"][1].as_u64().unwrap();
    assert_eq!(end.checked_sub(first), Some(bytes as u64));
    let absolute = first.checked_add(318228).unwrap();
    assert!(end.checked_add(318228).unwrap() <= MODEL_BYTES);
    let mut raw = vec![0; bytes];
    file.seek(SeekFrom::Start(absolute)).unwrap();
    file.read_exact(&mut raw).unwrap();
    let record = json!({"name":name,"dtype":dtype,"shape":shape,"offset":absolute,"bytes":bytes,"sha256":hash(&raw),"capacity":raw.capacity()});
    (raw, record)
}
// Named, conservative workload policy. Parsed metadata/serialization are policy
// allowances, not allocator residency measurements. Actual process peaks gate
// independently in the owner. All known overlap is counted simultaneously.
fn workload(owner: usize, source: usize, reference: usize, arrays: [usize; 5]) -> Value {
    let [input, expected, scalar, caller, baseline] = arrays;
    let categories = json!({"resident_owner":owner,"context_policy":CONTEXT_ALLOWANCE_BYTES,"source_cpu_loading_overlap":source,
        "header_bytes":318228,"frozen_binary":reference,"frozen_json_and_manifest_caps":131072,"parsed_header_metadata_policy":4194304,
        "parsed_reference_metadata_policy":1048576,"input_f32":input*4,"expected_f32":expected*4,"scalar_f32":scalar*4,"caller_f32":caller*4,"baseline_f32":baseline*4,
        "caller_raw_conversion":4194304,"child_raw_conversion":262144,"hash_read_workspace":65536,"record_serialization_policy":4194304,"record_metadata_policy":4194304});
    let total = categories
        .as_object()
        .unwrap()
        .values()
        .try_fold(0_u64, |sum, value| sum.checked_add(value.as_u64().unwrap()))
        .unwrap();
    assert!(total <= PROCESS_CAP as u64);
    json!({"categories":categories,"total_policy_bytes":total,"native_process_allowance_bytes":PROCESS_CAP,"remainder_is_free_device_memory":false})
}
struct Capture {
    file: File,
    bytes: usize,
}
impl Capture {
    fn append(
        &mut self,
        key: &str,
        role: &str,
        words: impl Iterator<Item = u32> + Clone,
        tokens: usize,
        rows: usize,
    ) -> Value {
        let count = tokens.checked_mul(rows).unwrap();
        let bytes = count.checked_mul(4).unwrap();
        assert!(bytes <= B * N * 4 && words.clone().count() == count);
        let next = self.bytes.checked_add(bytes).unwrap();
        assert!(next <= RAW_CAP);
        let item = json!({"key":key,"role":role,"dtype":"F32","shape":[tokens,rows],"offset":self.bytes,"bytes":bytes,"sha256":hash_words(words.clone())});
        report("GEMMA_HEAD_ARRAY_ATTEMPT", item.clone());
        let mut scratch = [0_u8; 4096];
        let mut used = 0;
        for word in words {
            scratch[used..used + 4].copy_from_slice(&word.to_le_bytes());
            used += 4;
            if used == scratch.len() {
                self.file.write_all(&scratch).unwrap();
                used = 0;
            }
        }
        self.file.write_all(&scratch[..used]).unwrap();
        self.file.flush().unwrap();
        self.bytes = next;
        report("GEMMA_HEAD_ARRAY", item.clone());
        item
    }
}
fn publish_outcome<W: Write>(writer: &mut W, outcome: &Value) {
    writeln!(writer, "GEMMA_HEAD_STEP_OUTCOME {outcome}").unwrap();
    writer.flush().unwrap();
}
fn step(
    owner: &mut PartitionedAffine<'_>,
    capture: &mut Capture,
    phase: &str,
    index: usize,
    input: &[f32],
    tokens: usize,
    caller: &mut [f32],
) {
    let key = format!("{phase}/{index}");
    let caller = &mut caller[..tokens * N];
    caller.fill(42.);
    report(
        "GEMMA_HEAD_STEP_ATTEMPT",
        json!({"key":key,"phase":phase,"index":index,"tokens":tokens,"input_sha256":floats_hash(input)}),
    );
    let result = owner.apply(input, caller, tokens);
    let state = observation(owner);
    let outcome = json!({"key":key,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"caller_sha256":floats_hash(caller),"observation":state});
    // Preserve the original cause and safe complete-buffer observations before
    // any fallible raw publication. Keep caller and device results separate.
    publish_outcome(&mut std::io::stdout(), &outcome);
    let mut arrays = Vec::new();
    for (child_index, child) in owner.chunks.iter().enumerate() {
        if child
            .pending
            .as_ref()
            .is_none_or(|command| command.status() == MTLCommandBufferStatus::Completed)
        {
            let item = child.output.with_slice(|words| {
                capture.append(
                    &key,
                    &format!("child-{child_index}-observed-output"),
                    words[..tokens * child.layout.rows()].iter().copied(),
                    tokens,
                    child.layout.rows(),
                )
            });
            arrays.push(item);
        }
    }
    let caller_array = capture.append(
        &key,
        "transactional-caller",
        caller.iter().map(|v| v.to_bits()),
        tokens,
        N,
    );
    report(
        "GEMMA_HEAD_STEP",
        json!({"key":key,"phase":phase,"index":index,"tokens":tokens,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"caller":caller_array,"observed_child_outputs":arrays,"observed_child_outputs_are_failed_request_results":false}),
    );
    result.unwrap();
    assert!(!owner.is_poisoned());
}
#[test]
fn outcome_precedes_a_failed_full_head_raw_writer_without_gpu() {
    let mut capture = Capture {
        file: File::open(file!()).unwrap(),
        bytes: 0,
    };
    let original = json!({"completed":false,"error":"original completed head apply error","caller_sha256":"sentinel caller","safe_child_observations":["completed output retained separately"]});
    let mut log = Vec::new();
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        publish_outcome(&mut log, &original);
        capture.append(
            "failure",
            "child-observed-output",
            [0_u32].into_iter(),
            1,
            1,
        );
    }));
    assert!(failed.is_err());
    assert_eq!(capture.bytes, 0);
    let line = String::from_utf8(log).unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(line.strip_prefix("GEMMA_HEAD_STEP_OUTCOME ").unwrap())
            .unwrap(),
        original
    );
}
#[test]
#[ignore = "explicit resident original real Gemma full tied-head qualification"]
fn real_tied_head_resident_reuse() {
    autoreleasepool(|_| {
        let model_path = std::env::var_os("TOKLET_GEMMA_HEAD_MODEL").expect("explicit model");
        let reference_path =
            std::env::var_os("TOKLET_GEMMA_HEAD_REFERENCE").expect("explicit frozen reference");
        let output_path =
            std::env::var_os("TOKLET_GEMMA_HEAD_OUTPUT").expect("exclusive raw output");
        let layout = Layout::new(N, K, 4, 64, MetadataDtype::BF16).unwrap();
        let plan = Plan::new(layout, B, TILE, usize::MAX).unwrap();
        let initial_policy = workload(
            plan.minimum_owned,
            226492416,
            4218880,
            [B * K, B * N, B * N, B * N, B * N],
        );
        report("GEMMA_HEAD_PLAN", initial_policy);
        let frozen = Path::new(&reference_path);
        bounded(
            &frozen.join("observation.json"),
            65536,
            "1c03155565a4a0ef5d0b71ce6c5d0600b3061f85c6c8275561530c9897f08579",
        );
        bounded(
            &frozen.join("manifest.json"),
            65536,
            "40a56af7e34f975142cdb0fd8a89e8683846ebd9f29189a4fa4117f34062aeed",
        );
        let reference = bounded(
            &frozen.join("arrays.bin"),
            8 * 1024 * 1024,
            "ba0e853cbf15459e3774afdac233237c68f36560395eb85c2ccb549409a7742c",
        );
        let decode = |raw: &[u8]| -> Vec<f32> {
            raw.chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect()
        };
        let input = decode(&reference[..24576]);
        let expected = decode(&reference[24576..]);
        let mut ordered = vec![0.; B * N];
        let mut caller = vec![42.; B * N];
        let mut baseline = vec![0.; B * N];
        let mut model = File::open(model_path).unwrap();
        assert_eq!(model.metadata().unwrap().len(), MODEL_BYTES);
        let mut prefix = [0; 8];
        model.read_exact(&mut prefix).unwrap();
        assert_eq!(u64::from_le_bytes(prefix), 318220);
        let mut header = vec![0; 318220];
        model.read_exact(&mut header).unwrap();
        assert_eq!(
            hash(&header),
            "4d1819ededded19812059691c0f77cfec0fe10c9c89c720950f919bfc26544a7"
        );
        let header: Value = serde_json::from_slice(&header).unwrap();
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let rejected = context.affine(layout, &[], &[], &[], B, PROCESS_CAP);
        report(
            "GEMMA_HEAD_OLD_OWNER",
            json!({"rejected":rejected.is_err(),"error":rejected.as_ref().err().map(ToString::to_string),"large_spans_read":false}),
        );
        assert!(matches!(
            rejected,
            Err(Error::Options(
                "Gemma affine fixed geometry/format outside bounds"
            ))
        ));
        let (codes, code_record) = source(
            &mut model,
            &header,
            "language_model.model.embed_tokens.weight",
            "U32",
            json!([N, K / 8]),
            layout.weight_bytes(),
        );
        let (scales, scale_record) = source(
            &mut model,
            &header,
            "language_model.model.embed_tokens.scales",
            "BF16",
            json!([N, K / 64]),
            layout.metadata_bytes(),
        );
        let (biases, bias_record) = source(
            &mut model,
            &header,
            "language_model.model.embed_tokens.biases",
            "BF16",
            json!([N, K / 64]),
            layout.metadata_bytes(),
        );
        let policy = workload(
            plan.minimum_owned,
            codes.capacity() + scales.capacity() + biases.capacity(),
            reference.capacity(),
            [
                input.capacity(),
                expected.capacity(),
                ordered.capacity(),
                caller.capacity(),
                baseline.capacity(),
            ],
        );
        let mut capture = Capture {
            file: File::options()
                .write(true)
                .create_new(true)
                .open(output_path)
                .unwrap(),
            bytes: 0,
        };
        report(
            "GEMMA_HEAD_META",
            json!({"device":context.device_name(),"simd_width":context.width,"owner_struct_bytes":size_of::<PartitionedAffine<'_>>(),"child_struct_bytes":size_of::<Affine<'_>>(),"policy":policy,"source_spans":[code_record,scale_record,bias_record],"binary_cap_bytes":RAW_CAP,"whole_graph_qualified":false,"native_activation_parity":false}),
        );
        let mut owner = {
            let matrix = Matrix::from_le_bytes(layout, &codes, &scales, &biases).unwrap();
            for token in 0..B {
                report(
                    "GEMMA_HEAD_ORDERED_ATTEMPT",
                    json!({"token":token,"contract":"native ordered FP64 dot rounded FP32"}),
                );
                matrix
                    .matvec(
                        &input[token * K..(token + 1) * K],
                        &mut ordered[token * N..(token + 1) * N],
                    )
                    .unwrap();
                let array = capture.append(
                    &format!("ordered/{token}"),
                    "native-ordered-fp32",
                    ordered[token * N..(token + 1) * N]
                        .iter()
                        .map(|v| v.to_bits()),
                    1,
                    N,
                );
                report(
                    "GEMMA_HEAD_ORDERED",
                    json!({"token":token,"array":array,"contract":"native ordered FP64 dot rounded FP32"}),
                );
            }
            let rejected = context.partitioned_affine(&matrix, TILE, B, plan.minimum_owned - 1);
            report(
                "GEMMA_HEAD_BUDGET",
                json!({"required":plan.minimum_owned,"limit":plan.minimum_owned-1,"rejected":rejected.is_err(),"error":rejected.as_ref().err().map(ToString::to_string)}),
            );
            assert!(
                matches!(rejected,Err(Error::Budget{required,limit}) if required==plan.minimum_owned && limit==plan.minimum_owned-1)
            );
            report(
                "GEMMA_HEAD_CONSTRUCTOR_ATTEMPT",
                json!({"rows":N,"columns":K,"maximum_rows":TILE,"capacity":B,"required":plan.minimum_owned}),
            );
            let result = context.partitioned_affine(&matrix, TILE, B, plan.minimum_owned);
            report(
                "GEMMA_HEAD_CONSTRUCTOR_OUTCOME",
                json!({"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string)}),
            );
            result.unwrap()
        };
        drop((codes, scales, biases));
        report(
            "GEMMA_HEAD_CONSTRUCTOR",
            json!({"source_cpu_bytes_retained":0,"observation":observation(&owner)}),
        );
        for repeat in 0..2 {
            step(
                &mut owner,
                &mut capture,
                "repeat",
                repeat,
                &input,
                B,
                &mut caller,
            );
            for (actual, math) in caller.iter().zip(&expected) {
                assert!(
                    actual.is_finite()
                        && (f64::from(*actual) - f64::from(*math)).abs()
                            <= 1e-5 + 1e-4 * f64::from(*math).abs()
                );
            }
            for (actual, scalar) in caller.iter().zip(&ordered) {
                assert!(
                    actual.is_finite()
                        && (f64::from(*actual) - f64::from(*scalar)).abs()
                            <= 1e-5 + 1e-4 * f64::from(*scalar).abs()
                );
            }
            if repeat == 0 {
                baseline.copy_from_slice(&caller);
            } else {
                assert!(
                    caller
                        .iter()
                        .zip(&baseline)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
        }
        for token in 0..B {
            step(
                &mut owner,
                &mut capture,
                "part",
                token,
                &input[token * K..(token + 1) * K],
                1,
                &mut caller,
            );
            assert!(
                caller[..N]
                    .iter()
                    .zip(&baseline[token * N..(token + 1) * N])
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
        step(
            &mut owner,
            &mut capture,
            "resume",
            0,
            &input,
            B,
            &mut caller,
        );
        assert!(
            caller
                .iter()
                .zip(&baseline)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert_eq!(capture.bytes, 37748736);
        report(
            "GEMMA_HEAD_FINAL",
            json!({"steps":7,"chunks":16,"scalar_rows":4,"raw_bytes":capture.bytes,"original_cpu_packed_table_retained":false,"all_chunks_resident":true,"chunks_execute_sequentially":true,"whole_graph_qualified":false,"native_activation_parity":false,"performance_qualified":false}),
        );
    });
}
