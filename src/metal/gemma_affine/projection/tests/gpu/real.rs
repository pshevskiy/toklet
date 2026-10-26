//! Complete real tied head with immutable weights and reused projection scratch.
use super::*;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};
const N: usize = 262144;
const K: usize = 1536;
const B: usize = 4;
const TILE: usize = 16384;
const RAW_CAP: usize = 64 * 1024 * 1024;
const WEIGHT_BYTES: usize = 226495664;
const WORKSPACE_BYTES: usize = 4743496;
const LOOKUP_BYTES: usize = 16851208;
fn report(label: &str, value: Value) {
    record(&mut std::io::stdout(), label, &value);
}
fn record(writer: &mut impl Write, label: &str, value: &Value) {
    writeln!(writer, "GEMMA_PROJECTION_HEAD_{label} {value}").unwrap();
    writer.flush().unwrap();
}
fn outcome_then<W: Write, T>(
    writer: &mut W,
    outcome: &Value,
    append: impl FnOnce(&mut W) -> T,
) -> T {
    record(writer, "STEP_OUTCOME", outcome);
    append(writer)
}
fn hash(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}
fn word_hash(words: impl Iterator<Item = u32>) -> String {
    let mut h = Sha256::new();
    let mut stack = [0_u8; 4096];
    let mut used = 0;
    for word in words {
        stack[used..used + 4].copy_from_slice(&word.to_le_bytes());
        used += 4;
        if used == stack.len() {
            h.update(stack);
            used = 0;
        }
    }
    h.update(&stack[..used]);
    format!("{:x}", h.finalize())
}
fn frozen(path: &Path, bytes: usize, wanted: &str) -> Vec<u8> {
    let mut file = File::open(path).unwrap();
    let mut raw = vec![0; bytes];
    file.read_exact(&mut raw).unwrap();
    assert_eq!(file.read(&mut [0_u8; 1]).unwrap(), 0);
    assert_eq!(raw.capacity(), bytes);
    assert_eq!(hash(&raw), wanted);
    raw
}
fn policy() -> Value {
    let categories = json!({"resident_weights":WEIGHT_BYTES,"projection_workspace":WORKSPACE_BYTES,"lookup_policy":LOOKUP_BYTES,
        "source_staging":14155848,"two_contexts":2*CONTEXT_ALLOWANCE_BYTES,"head_reference":4218880,"lookup_reference":167936,
        "frozen_metadata":131072,"header_bytes":318228,"metadata_policy":8388608,"caller":4194304,"input":24576,"lookup_caller":24576,
        "serialization":8388608,"diagnostic_state":16777216,"hash_io_stack":131072});
    let total: u64 = categories
        .as_object()
        .unwrap()
        .values()
        .map(|v| v.as_u64().unwrap())
        .sum();
    assert_eq!(total, 439229020);
    json!({"categories":categories,"total_policy_bytes":total,"native_process_allowance_bytes":1024*1024*1024,"exact_physical_residency":false})
}
fn buffer_state(value: &Buffer<u32>, read: bool) -> Value {
    json!({"allocation":format!("{:p}",&*value.raw),"words":value.len,"bytes":value.len*4,
        "contents":read.then(||value.with_slice(|words|json!({"sha256":word_hash(words.iter().copied()),"guards":words[words.len()-GUARD..].to_vec()})))})
}
fn weight_state(weights: &Weights, read: bool) -> Value {
    let (slots, chunk_bytes, device) = weights.diagnostic_storage();
    let chunks:Vec<_>=weights.views().map(|v|json!({"rows":v.layout.rows(),"codes":buffer_state(v.codes,read),"scales":buffer_state(v.scales,read),"biases":buffer_state(v.biases,read)})).collect();
    json!({"rows":N,"columns":K,"owned_bytes":weights.owned_bytes(),"owner_slots":slots,"chunk_struct_bytes":chunk_bytes,"device":device,"chunks":chunks})
}
fn state(session: &Projection<'_>) -> Value {
    let read = safe(session);
    json!({"owned_bytes":session.owned_bytes(),"staged_capacity":session.staged.capacity(),"chunk_capacity":session.chunk_output.capacity(),
        "poisoned":session.is_poisoned(),"pending":session.pending.is_some(),"shared_read_safe":read,
        "staged_sha256":word_hash(session.staged.iter().map(|v|v.to_bits())),"chunk_sha256":word_hash(session.chunk_output.iter().map(|v|v.to_bits())),
        "input":buffer_state(&session.input,read),"output":buffer_state(&session.output,read),"status":buffer_state(&session.status,read)})
}
struct Capture {
    file: File,
    bytes: usize,
    arrays: usize,
}
impl Capture {
    fn append(
        &mut self,
        key: &str,
        role: &str,
        dtype: &str,
        words: impl Iterator<Item = u32> + Clone,
    ) -> Value {
        self.append_to(&mut std::io::stdout(), key, role, dtype, words)
    }
    fn append_to(
        &mut self,
        records: &mut impl Write,
        key: &str,
        role: &str,
        dtype: &str,
        words: impl Iterator<Item = u32> + Clone,
    ) -> Value {
        let count = words.clone().count();
        let bytes = count.checked_mul(4).unwrap();
        let next = self.bytes.checked_add(bytes).unwrap();
        assert!(bytes <= B * N * 4 && next <= RAW_CAP);
        let item = json!({"key":key,"role":role,"dtype":dtype,"words":count,"bytes":bytes,"offset":self.bytes,"sha256":word_hash(words.clone())});
        record(records, "ARRAY_ATTEMPT", &item);
        let mut stack = [0_u8; 4096];
        let mut used = 0;
        for word in words {
            stack[used..used + 4].copy_from_slice(&word.to_le_bytes());
            used += 4;
            if used == stack.len() {
                self.file.write_all(&stack).unwrap();
                used = 0;
            }
        }
        self.file.write_all(&stack[..used]).unwrap();
        self.file.flush().unwrap();
        self.bytes = next;
        self.arrays += 1;
        record(records, "ARRAY", &item);
        item
    }
    fn workspace(&mut self, key: &str, session: &Projection<'_>) -> Vec<Value> {
        let mut arrays = Vec::new();
        if safe(session) {
            for (role, buffer) in [
                ("input", &session.input),
                ("output", &session.output),
                ("status", &session.status),
            ] {
                arrays.push(
                    buffer.with_slice(|words| self.append(key, role, "U32", words.iter().copied())),
                );
            }
        }
        arrays.push(self.append(
            key,
            "staged",
            "F32",
            session.staged.iter().map(|v| v.to_bits()),
        ));
        arrays.push(self.append(
            key,
            "chunk",
            "F32",
            session.chunk_output.iter().map(|v| v.to_bits()),
        ));
        arrays
    }
}
fn spans(header: &Value) -> Value {
    let mut spans = Vec::new();
    for (role, suffix, dtype, shape, stride) in [
        ("codes", "weight", "U32", json!([N, K / 8]), K / 2),
        ("scales", "scales", "BF16", json!([N, K / 64]), K / 32),
        ("biases", "biases", "BF16", json!([N, K / 64]), K / 32),
    ] {
        let name = format!("language_model.model.embed_tokens.{suffix}");
        let entry = &header[&name];
        assert_eq!(entry["dtype"], dtype);
        assert_eq!(entry["shape"], shape);
        let start = entry["data_offsets"][0].as_u64().unwrap();
        let end = entry["data_offsets"][1].as_u64().unwrap();
        assert_eq!(end.checked_sub(start), Some((N * stride) as u64));
        assert!(end <= 3550670554 - 318228);
        spans.push(json!({"role":role,"name":name,"dtype":dtype,"shape":shape,"offset":start+318228,"bytes":N*stride,"stride":stride}));
    }
    json!(spans)
}
fn lookup(
    capture: &mut Capture,
    session: &mut Gather<'_>,
    weights: &Weights,
    index: usize,
    caller: &mut [f32],
    expected: &[u8],
) {
    caller.fill(42.);
    report(
        "LOOKUP_ATTEMPT",
        json!({"index":index,"ids":[2,100,257,100]}),
    );
    let result = session.apply_weights(weights, &[2, 100, 257, 100], caller);
    report(
        "LOOKUP_OUTCOME",
        json!({"index":index,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"caller_sha256":word_hash(caller.iter().map(|v|v.to_bits()))}),
    );
    let array = capture.append(
        &format!("lookup/{index}"),
        "caller",
        "F32",
        caller.iter().map(|v| v.to_bits()),
    );
    report(
        "LOOKUP",
        json!({"index":index,"array":array,"private_lookup_scratch_captured":false}),
    );
    result.unwrap();
    assert!(
        caller
            .iter()
            .zip(expected.chunks_exact(4))
            .all(|(v, raw)| v.to_le_bytes() == raw)
    );
}
#[test]
fn real_head_workspace_and_shared_table_policy_need_no_model_or_gpu() {
    assert_eq!(
        Plan::new(layout(N, K), TILE, B, usize::MAX)
            .unwrap()
            .minimum_owned,
        WORKSPACE_BYTES
    );
    assert_eq!(policy()["total_policy_bytes"], 439229020);
    assert_eq!(
        4743256 + 3 * (4743256 + 4194304) + 4 * (4743256 + 1048576) + 2 * 24576,
        54772416
    );
}
#[test]
fn original_outcome_and_array_attempt_precede_read_only_writer_failure_without_gpu() {
    let mut capture = Capture {
        file: File::open(file!()).unwrap(),
        bytes: 0,
        arrays: 0,
    };
    let original = json!({"completed":false,"error":"original completed projection error","safe_final_scratch":"retained as state, not failed results"});
    let mut records = Vec::new();
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        outcome_then(&mut records, &original, |writer| {
            capture.append_to(writer, "failure", "caller", "F32", [0_u32].into_iter())
        });
    }));
    assert!(failure.is_err());
    assert_eq!((capture.bytes, capture.arrays), (0, 0));
    let text = String::from_utf8(records).unwrap();
    let lines: Vec<_> = text.lines().collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(
        serde_json::from_str::<Value>(
            lines[0]
                .strip_prefix("GEMMA_PROJECTION_HEAD_STEP_OUTCOME ")
                .unwrap()
        )
        .unwrap(),
        original
    );
    let attempt: Value = serde_json::from_str(
        lines[1]
            .strip_prefix("GEMMA_PROJECTION_HEAD_ARRAY_ATTEMPT ")
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        attempt,
        json!({"key":"failure","role":"caller","dtype":"F32","words":1,"bytes":4,"offset":0,"sha256":hash(&[0;4])})
    );
}
#[test]
#[ignore = "explicit full real immutable tied-head projection and shared embedding lookup"]
fn real_tied_head_reuses_immutable_weights_and_fresh_projection_scratch() {
    autoreleasepool(|_| {
        report("PLAN", policy());
        let model_path =
            std::env::var_os("TOKLET_GEMMA_PROJECTION_MODEL").expect("explicit original model");
        let head_path = std::env::var_os("TOKLET_GEMMA_PROJECTION_HEAD_REFERENCE")
            .expect("explicit head reference");
        let lookup_path = std::env::var_os("TOKLET_GEMMA_PROJECTION_LOOKUP_REFERENCE")
            .expect("explicit lookup reference");
        let output_path =
            std::env::var_os("TOKLET_GEMMA_PROJECTION_OUTPUT").expect("exclusive output path");
        let folder = Path::new(&head_path);
        frozen(
            &folder.join("observation.json"),
            2145,
            "1c03155565a4a0ef5d0b71ce6c5d0600b3061f85c6c8275561530c9897f08579",
        );
        frozen(
            &folder.join("manifest.json"),
            385,
            "40a56af7e34f975142cdb0fd8a89e8683846ebd9f29189a4fa4117f34062aeed",
        );
        let reference = frozen(
            &folder.join("arrays.bin"),
            4218880,
            "ba0e853cbf15459e3774afdac233237c68f36560395eb85c2ccb549409a7742c",
        );
        let lookup_reference = frozen(
            &Path::new(&lookup_path).join("arrays.bin"),
            167936,
            "aea89e9e83d266ae5e648e1c4bebb01509370ef29ef777dea7018b5822cbf469",
        );
        let input: Vec<_> = reference[..24576]
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        assert_eq!(input.capacity(), B * K);
        let mut caller = vec![42.; B * N];
        assert_eq!(caller.capacity(), B * N);
        let mut lookup_caller = vec![42.; 4 * K];
        assert_eq!(lookup_caller.capacity(), 4 * K);
        let weights = autoreleasepool(|_| {
            let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
            report(
                "CONSTRUCTION_CONTEXT",
                json!({"device":context.device_name(),"identity":format!("{:p}",&*context.gpu.device)}),
            );
            let mut model = File::open(&model_path).unwrap();
            assert_eq!(model.metadata().unwrap().len(), 3550670554);
            let mut prefix = [0; 8];
            model.read_exact(&mut prefix).unwrap();
            assert_eq!(u64::from_le_bytes(prefix), 318220);
            let mut raw = vec![0; 318220];
            model.read_exact(&mut raw).unwrap();
            assert_eq!(
                hash(&raw),
                "4d1819ededded19812059691c0f77cfec0fe10c9c89c720950f919bfc26544a7"
            );
            let header: Value = serde_json::from_slice(&raw).unwrap();
            let spans = spans(&header);
            report("SPANS", json!({"entries":spans.clone()}));
            report(
                "UPLOAD_ATTEMPT",
                json!({"working_budget":240651512,"maximum_rows":TILE}),
            );
            let result=context.weights_from_fill(layout(N,K),TILE,240651512,|first,child,codes,scales,biases| {
                let arrays=[codes,scales,biases];report("FILL_ATTEMPT",json!({"first_row":first,"rows":child.rows(),"addresses":arrays.each_ref().map(|v|format!("{:p}",v.as_ptr())),"lengths":arrays.each_ref().map(|v|v.len())}));
                let mut consumed=Vec::new();
                for (array,span) in arrays.into_iter().zip(spans.as_array().unwrap()) {
                    let offset=span["offset"].as_u64().unwrap()+first as u64*span["stride"].as_u64().unwrap();
                    model.seek(SeekFrom::Start(offset)).map_err(|e|Error::Native(e.to_string()))?;model.read_exact(array).map_err(|e|Error::Native(e.to_string()))?;
                    consumed.push(json!({"offset":offset,"bytes":array.len(),"sha256":hash(array)}));
                }
                report("FILL",json!({"first_row":first,"consumed":consumed}));Ok(())
            });
            report(
                "UPLOAD_OUTCOME",
                json!({"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string)}),
            );
            let weights = result.unwrap();
            report("WEIGHTS", weight_state(&weights, true));
            assert_eq!(weights.owned_bytes(), WEIGHT_BYTES);
            weights
        });
        report(
            "CONSTRUCTION_CONTEXT_DROPPED",
            json!({"owners":1,"device":weights.diagnostic_storage().2}),
        );
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        report(
            "EXECUTION_CONTEXT",
            json!({"device":context.device_name(),"identity":format!("{:p}",&*context.gpu.device),"same_device":weights.same_device(&context)}),
        );
        report(
            "WORKSPACE_ATTEMPT",
            json!({"projection_budget":WORKSPACE_BYTES,"lookup_budget":LOOKUP_BYTES}),
        );
        let projected = context.projection_session(layout(N, K), TILE, B, WORKSPACE_BYTES);
        report(
            "PROJECTION_OUTCOME",
            json!({"completed":projected.is_ok(),"error":projected.as_ref().err().map(ToString::to_string)}),
        );
        let mut session = projected.unwrap();
        let gathered = context.gather_session(K, 4, LOOKUP_BYTES);
        report(
            "GATHER_OUTCOME",
            json!({"completed":gathered.is_ok(),"error":gathered.as_ref().err().map(ToString::to_string)}),
        );
        let mut gather = gathered.unwrap();
        assert_eq!(gather.admitted_bytes(), LOOKUP_BYTES);
        let mut capture = Capture {
            file: File::options()
                .write(true)
                .create_new(true)
                .open(output_path)
                .unwrap(),
            bytes: 0,
            arrays: 0,
        };
        report("COLD_OBSERVATION", state(&session));
        let arrays = capture.workspace("cold", &session);
        report("COLD", json!({"arrays":arrays}));
        lookup(
            &mut capture,
            &mut gather,
            &weights,
            0,
            &mut lookup_caller,
            &lookup_reference[..24576],
        );
        for (phase, index, tokens) in [
            ("repeat", 0, 4),
            ("repeat", 1, 4),
            ("part", 0, 1),
            ("part", 1, 1),
            ("part", 2, 1),
            ("part", 3, 1),
            ("resume", 0, 4),
        ] {
            let key = format!("{phase}/{index}");
            let input = if tokens == 4 {
                &input[..]
            } else {
                &input[index * K..(index + 1) * K]
            };
            let caller = &mut caller[..tokens * N];
            caller.fill(42.);
            report(
                "STEP_ATTEMPT",
                json!({"key":key,"tokens":tokens,"input_sha256":word_hash(input.iter().map(|v|v.to_bits()))}),
            );
            let result = session.apply(&weights, input, caller, tokens);
            let outcome = json!({"key":key,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"caller_sha256":word_hash(caller.iter().map(|v|v.to_bits())),"workspace":state(&session),"weights":weight_state(&weights,safe(&session))});
            let (arrays, returned) = outcome_then(&mut std::io::stdout(), &outcome, |_| {
                let arrays = capture.workspace(&key, &session);
                let returned =
                    capture.append(&key, "caller", "F32", caller.iter().map(|v| v.to_bits()));
                (arrays, returned)
            });
            report(
                "STEP",
                json!({"key":key,"arrays":arrays,"caller":returned,"overwritten_child_device_outputs_separately_captured":false,"captured_scratch_is_failed_request_result":false}),
            );
            result.unwrap();
            let expected = if tokens == 4 {
                &reference[24576..]
            } else {
                &reference[24576 + index * N * 4..24576 + (index + 1) * N * 4]
            };
            assert!(
                caller
                    .iter()
                    .zip(expected.chunks_exact(4))
                    .all(|(v, raw)| v.to_le_bytes() == raw)
            );
        }
        lookup(
            &mut capture,
            &mut gather,
            &weights,
            1,
            &mut lookup_caller,
            &lookup_reference[..24576],
        );
        report("FINAL_WEIGHTS", weight_state(&weights, true));
        report(
            "RESULT",
            json!({"steps":7,"lookups":2,"arrays":capture.arrays,"binary_bytes":capture.bytes,"chunks":16,"construction_context_dropped":true,"weights_shared_with_lookup":true,
            "overwritten_child_device_outputs_separately_captured":false,"private_lookup_scratch_captured":false,"source_cpu_table_retained":false,"foreign_device_qualified":false,
            "whole_graph_qualified":false,"original_bf16_source_qualified":false,"performance_qualified":false}),
        );
        assert_eq!(capture.arrays, 49);
        assert_eq!(capture.bytes, 54772416);
    });
}
