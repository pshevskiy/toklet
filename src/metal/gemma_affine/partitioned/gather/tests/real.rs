//! Explicit real checkpoint boundary. Ordinary tests never open model files.
use super::super::super::Plan;
use super::*;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

const N: usize = 262144;
const TILE: usize = 16384;
const CAP: usize = 32;
const MODEL_BYTES: u64 = 3550670554;
const RAW_CAP: usize = 64 * 1024 * 1024;
const PROCESS_CAP: usize = 3 * 1024 * 1024 * 1024;
const ORIGINAL: [usize; 4] = [2, 100, 257, 100];

fn record_then<W: Write, T>(
    writer: &mut W,
    label: &str,
    value: &Value,
    append: impl FnOnce() -> std::io::Result<T>,
) -> std::io::Result<T> {
    writeln!(writer, "GEMMA_TABLES_{label} {value}")?;
    writer.flush()?;
    append()
}
fn emit(label: &str, value: Value) {
    record_then(&mut std::io::stdout().lock(), label, &value, || Ok(())).unwrap();
}
fn hash(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}
fn word_hash(words: impl Iterator<Item = u32>) -> String {
    let mut h = Sha256::new();
    let mut scratch = [0_u8; 4096];
    let mut used = 0;
    for word in words {
        scratch[used..used + 4].copy_from_slice(&word.to_le_bytes());
        used += 4;
        if used == scratch.len() {
            h.update(scratch);
            used = 0;
        }
    }
    h.update(&scratch[..used]);
    format!("{:x}", h.finalize())
}
fn bounded(path: &Path, cap: usize, wanted: &str) -> Vec<u8> {
    let mut bytes = vec![0; cap];
    let mut file = File::open(path).unwrap();
    file.read_exact(&mut bytes).unwrap();
    assert_eq!(file.read(&mut [0_u8; 1]).unwrap(), 0);
    assert_eq!(bytes.capacity(), cap);
    assert_eq!(hash(&bytes), wanted);
    bytes
}
fn buffer_state(buffer: &Buffer<u32>) -> Value {
    buffer.with_slice(|v| {
        json!({"allocation":format!("{:p}",&*buffer.raw),"words":v.len(),"bytes":v.len()*4,
        "sha256":word_hash(v.iter().copied()),"guards":v[v.len()-GUARD..].to_vec()})
    })
}
fn matrix_state(matrix: &PartitionedAffine<'_>) -> Value {
    let children: Vec<_> = matrix.chunks.iter().map(|c| {
        let safe = c.pending.as_ref().is_none_or(|p|p.status()==MTLCommandBufferStatus::Completed);
        json!({"rows":c.layout.rows(),"owned_bytes":c.owned_bytes(),"poisoned":c.poisoned,"pending":c.pending.is_some(),"shared_read_safe":safe,
            "device":safe.then(||json!({"codes":buffer_state(&c.codes),"scales":buffer_state(&c.scales),"biases":buffer_state(&c.biases),
                "input":buffer_state(&c.input),"output":buffer_state(&c.output),"status":buffer_state(&c.status)}))})
    }).collect();
    json!({"rows":matrix.rows,"columns":matrix.columns,"capacity":matrix.capacity,"owned_bytes":matrix.owned_bytes(),"poisoned":matrix.poisoned,
        "owner_slots":matrix.chunks.capacity(),"full_scratch_capacity":matrix.staged.capacity(),"chunk_scratch_capacity":matrix.chunk_output.capacity(),
        "staged_sha256":word_hash(matrix.staged.iter().map(|v|v.to_bits())),"chunk_scratch_sha256":word_hash(matrix.chunk_output.iter().map(|v|v.to_bits())),"children":children})
}
fn layout(k: usize) -> Layout {
    Layout::new(N, k, 4, 64, MetadataDtype::BF16).unwrap()
}
fn sizes(k: usize) -> (usize, usize, usize) {
    let plan = Plan::new(layout(k), 1, TILE, usize::MAX).unwrap();
    let (_, child) = plan.child(0).unwrap();
    let source = child.weight_bytes() + child.metadata_bytes() * 2 + 3 * size_of::<Vec<u8>>();
    (
        plan.minimum_owned,
        Sizes::new(k, CAP, usize::MAX).unwrap().minimum_admitted,
        source,
    )
}
fn policy() -> Value {
    let (e, eg, es) = sizes(1536);
    let (p, pg, ps) = sizes(8960);
    let categories = json!({"resident_owners":e+p,"gather_workspaces":eg+pg,"source_staging":es.max(ps),"context_policy":CONTEXT_ALLOWANCE_BYTES,
        "header_bytes":318228,"header_and_reference_parser_policy":8388608,"frozen_lookup":167936,"frozen_metadata_policy":131072,
        "caller_f32":32*8960*4,"serialization_policy":8388608,"hash_and_io_stack":131072,"diagnostic_metadata_policy":8388608});
    let total = categories
        .as_object()
        .unwrap()
        .values()
        .try_fold(0_u64, |sum, v| sum.checked_add(v.as_u64().unwrap()))
        .unwrap();
    assert!(total <= PROCESS_CAP as u64);
    json!({"categories":categories,"total_policy_bytes":total,"native_process_allowance_bytes":PROCESS_CAP,"mac_working_set_ceiling_bytes":6_u64*1024*1024*1024,
        "exact_physical_residency":false,"phone_qualified":false})
}
fn source_spans(header: &Value, stem: &str, k: usize) -> Value {
    let mut entries = Vec::new();
    for (role, suffix, dtype, shape, stride) in [
        ("codes", "weight", "U32", json!([N, k / 8]), k / 2),
        ("scales", "scales", "BF16", json!([N, k / 64]), k / 32),
        ("biases", "biases", "BF16", json!([N, k / 64]), k / 32),
    ] {
        let name = format!("language_model.model.{stem}.{suffix}");
        let entry = &header[&name];
        assert_eq!(entry["dtype"], dtype);
        assert_eq!(entry["shape"], shape);
        let a = entry["data_offsets"][0].as_u64().unwrap();
        let b = entry["data_offsets"][1].as_u64().unwrap();
        assert_eq!(b.checked_sub(a), Some((N * stride) as u64));
        assert!(b.checked_add(318228).unwrap() <= MODEL_BYTES);
        entries.push(json!({"role":role,"name":name,"dtype":dtype,"shape":shape,"offset":a+318228,"bytes":b-a,"stride":stride}));
    }
    json!(entries)
}
fn upload<'a>(
    context: &'a Context,
    file: &mut File,
    header: &Value,
    role: &str,
    stem: &str,
    k: usize,
) -> PartitionedAffine<'a> {
    let (resident, _, source) = sizes(k);
    let spans = source_spans(header, stem, k);
    emit(
        "UPLOAD_ATTEMPT",
        json!({"role":role,"columns":k,"maximum_rows":TILE,"capacity":1,"working_budget":resident+source,"spans":spans}),
    );
    let result = context.partitioned_affine_from_fill(layout(k),TILE,1,resident+source,|first,child,codes,scales,biases|{
        let arrays = [codes,scales,biases];
        let addresses: Vec<_> = arrays.iter().map(|v|format!("{:p}",v.as_ptr())).collect();
        emit("FILL_ATTEMPT",json!({"role":role,"first_row":first,"rows":child.rows(),"addresses":addresses,"lengths":arrays.each_ref().map(|v|v.len())}));
        let mut consumed = Vec::new();
        for (array,span) in arrays.into_iter().zip(spans.as_array().unwrap()) {
            let offset = span["offset"].as_u64().unwrap()+first as u64*span["stride"].as_u64().unwrap();
            file.seek(SeekFrom::Start(offset)).map_err(|e|Error::Native(e.to_string()))?;
            file.read_exact(array).map_err(|e|Error::Native(e.to_string()))?;
            consumed.push(json!({"offset":offset,"bytes":array.len(),"sha256":hash(array)}));
        }
        emit("FILL",json!({"role":role,"first_row":first,"consumed":consumed}));
        Ok(())
    });
    emit(
        "UPLOAD_OUTCOME",
        json!({"role":role,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string)}),
    );
    let matrix = result.unwrap();
    // Actual capacities are admitted by the constructor; aggregate policy
    // reserves the complete resident plus its peak source, independently.
    emit(
        "MATRIX",
        json!({"role":role,"source_cpu_bytes_retained":0,"observation":matrix_state(&matrix)}),
    );
    assert_eq!(matrix.owned_bytes(), resident);
    matrix
}
struct Capture {
    file: File,
    bytes: usize,
}
impl Capture {
    fn array(
        &mut self,
        key: &str,
        role: &str,
        words: impl Iterator<Item = u32> + Clone,
        count: usize,
    ) -> Value {
        assert_eq!(words.clone().count(), count);
        let bytes = count.checked_mul(4).unwrap();
        let end = self.bytes.checked_add(bytes).unwrap();
        assert!(end <= RAW_CAP);
        let entry = json!({"key":key,"role":role,"offset":self.bytes,"bytes":bytes,"words":count,"sha256":word_hash(words.clone())});
        emit("ARRAY_ATTEMPT", entry.clone());
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
        self.bytes = end;
        emit("ARRAY", entry.clone());
        entry
    }
    fn session(&mut self, key: &str, session: &Gather<'_>, caller: &[f32]) -> Value {
        let safe = session
            .pending
            .as_ref()
            .is_none_or(|c| c.status() == MTLCommandBufferStatus::Completed);
        let host:Vec<_>=[("caller",caller),("staged",&session.staged),("chunk",&session.chunk)].into_iter().map(|(role,values)|json!({"role":role,"words":values.len(),"sha256":word_hash(values.iter().map(|v|v.to_bits()))})).collect();
        let devices: Vec<_> = if safe {
            [
                ("ids", &session.ids),
                ("output", &session.output),
                ("status", &session.status),
            ]
            .into_iter()
            .map(|(role, buffer)| json!({"role":role,"identity":buffer_state(buffer)}))
            .collect()
        } else {
            Vec::new()
        };
        let descriptor = json!({"key":key,"columns":session.columns,"capacity":session.capacity,"admitted_bytes":session.admitted_bytes(),"poisoned":session.poisoned,"pending":session.pending.is_some(),"shared_read_safe":safe,"full_scratch_capacity":session.staged.capacity(),"chunk_scratch_capacity":session.chunk.capacity(),"host":host,"device":devices});
        // All completion-safe allocation identities and complete-buffer hashes
        // reach stdout before the first fallible raw capture write.
        record_then(
            &mut std::io::stdout().lock(),
            "SNAPSHOT",
            &descriptor,
            || Ok(self.session_arrays(key, session, caller, safe)),
        )
        .unwrap()
    }
    fn session_arrays(
        &mut self,
        key: &str,
        session: &Gather<'_>,
        caller: &[f32],
        safe: bool,
    ) -> Value {
        let mut arrays = Vec::new();
        for (role, values) in [
            ("caller", caller),
            ("staged", &session.staged),
            ("chunk", &session.chunk),
        ] {
            arrays.push(self.array(key, role, values.iter().map(|v| v.to_bits()), values.len()));
        }
        let mut devices = Vec::new();
        if safe {
            for (role, buffer) in [
                ("ids", &session.ids),
                ("output", &session.output),
                ("status", &session.status),
            ] {
                let entry =
                    buffer.with_slice(|v| self.array(key, role, v.iter().copied(), v.len()));
                devices.push(
                    json!({"role":role,"allocation":format!("{:p}",&*buffer.raw),"array":entry}),
                );
            }
        }
        json!({"columns":session.columns,"capacity":session.capacity,"admitted_bytes":session.admitted_bytes(),"poisoned":session.poisoned,"pending":session.pending.is_some(),"shared_read_safe":safe,
            "full_scratch_capacity":session.staged.capacity(),"chunk_scratch_capacity":session.chunk.capacity(),"arrays":arrays,"device":devices})
    }
}
fn step(
    capture: &mut Capture,
    session: &mut Gather<'_>,
    matrix: &PartitionedAffine<'_>,
    role: &str,
    index: usize,
    ids: &[usize],
    reference: &[u8],
) {
    let key = format!("{role}/{index}");
    let mut caller = vec![42.; ids.len() * matrix.columns];
    assert_eq!(caller.capacity(), caller.len());
    emit(
        "STEP_ATTEMPT",
        json!({"key":key,"role":role,"index":index,"ids":ids,"caller_capacity":caller.capacity()}),
    );
    let result = session.apply(matrix, ids, &mut caller);
    // Original cause first; safe actual bytes are preserved before assertions.
    emit(
        "STEP_OUTCOME",
        json!({"key":key,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"poisoned":session.poisoned}),
    );
    let state = capture.session(&key, session, &caller);
    emit(
        "STEP",
        json!({"key":key,"observation":state,"workspace_output_contract":"last executed child; inactive tail sentinel; CPU chunk tail may be stale"}),
    );
    result.unwrap();
    if ids == ORIGINAL {
        let actual: Vec<_> = caller.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(actual, reference);
    }
}
#[test]
fn original_outcome_and_safe_device_descriptor_survive_first_raw_write_failure_without_gpu() {
    let mut log = Vec::new();
    let outcome = json!({"completed":false,"error":"original failure"});
    let descriptor = json!({"device":[{"allocation":"0x1234","sha256":"complete safe buffer digest"}],"shared_read_safe":true});
    let result = record_then(&mut log, "STEP_OUTCOME", &outcome, || Ok(()));
    result.unwrap();
    let result: std::io::Result<()> = record_then(&mut log, "SNAPSHOT", &descriptor, || {
        Err(std::io::Error::new(
            std::io::ErrorKind::WriteZero,
            "first raw write",
        ))
    });
    assert_eq!(result.unwrap_err().to_string(), "first raw write");
    let expected =
        format!("GEMMA_TABLES_STEP_OUTCOME {outcome}\nGEMMA_TABLES_SNAPSHOT {descriptor}\n");
    assert_eq!(String::from_utf8(log).unwrap(), expected);
}
#[test]
fn aggregate_plan_covers_two_real_tables_loading_overlap_and_both_lookup_workspaces_without_gpu() {
    assert_eq!(sizes(1536), (228760112, 17367416, 14155848));
    assert_eq!(sizes(8960), (1323948592, 20218232, 82575432));
    assert!(policy()["total_policy_bytes"].as_u64().unwrap() < PROCESS_CAP as u64);
}
#[test]
#[ignore = "explicit two full real resident tables and independent complete lookup qualification"]
fn real_streamed_tables_remain_resident_across_lookups() {
    autoreleasepool(|_| {
        let ledger = policy();
        emit("PLAN", ledger);
        let model = std::env::var_os("TOKLET_GEMMA_TABLES_MODEL").unwrap();
        let mut source = File::open(model).unwrap();
        assert_eq!(source.metadata().unwrap().len(), MODEL_BYTES);
        let mut prefix = [0; 8];
        source.read_exact(&mut prefix).unwrap();
        assert_eq!(u64::from_le_bytes(prefix), 318220);
        let mut headerraw = vec![0; 318220];
        source.read_exact(&mut headerraw).unwrap();
        assert_eq!(headerraw.capacity(), 318220);
        assert_eq!(
            hash(&headerraw),
            "4d1819ededded19812059691c0f77cfec0fe10c9c89c720950f919bfc26544a7"
        );
        let header: Value = serde_json::from_slice(&headerraw).unwrap();
        let frozen = std::env::var_os("TOKLET_GEMMA_TABLES_REFERENCE").unwrap();
        let reference = bounded(
            &Path::new(&frozen).join("arrays.bin"),
            167936,
            "aea89e9e83d266ae5e648e1c4bebb01509370ef29ef777dea7018b5822cbf469",
        );
        assert_eq!(reference.capacity(), 167936);
        let output = std::env::var_os("TOKLET_GEMMA_TABLES_OUTPUT").unwrap();
        let mut capture = Capture {
            file: File::options()
                .write(true)
                .create_new(true)
                .open(output)
                .unwrap(),
            bytes: 0,
        };
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        emit(
            "META",
            json!({"device":context.device_name(),"owner_struct_bytes":size_of::<PartitionedAffine<'_>>(),"child_struct_bytes":size_of::<Affine<'_>>(),"gather_struct_bytes":size_of::<Gather<'_>>(),"header_capacity":headerraw.capacity(),"reference_capacity":reference.capacity(),"binary_cap_bytes":RAW_CAP,"whole_graph_qualified":false,"performance_qualified":false}),
        );
        let embedding = upload(
            &context,
            &mut source,
            &header,
            "embedding",
            "embed_tokens",
            1536,
        );
        let ple = upload(
            &context,
            &mut source,
            &header,
            "ple_lookup",
            "embed_tokens_per_layer",
            8960,
        );
        let make = |role: &str, k| {
            emit(
                "GATHER_ATTEMPT",
                json!({"role":role,"columns":k,"capacity":CAP,"budget":sizes(k).1}),
            );
            let result = context.gather_session(k, CAP, sizes(k).1);
            emit(
                "GATHER_OUTCOME",
                json!({"role":role,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string)}),
            );
            result.unwrap()
        };
        let mut e = make("embedding", 1536);
        let mut p = make("ple_lookup", 8960);
        assert_eq!(e.admitted_bytes(), sizes(1536).1);
        assert_eq!(p.admitted_bytes(), sizes(8960).1);
        emit(
            "WORKSPACES",
            json!({"embedding":capture.session("embedding/cold",&e,&[]),"ple_lookup":capture.session("ple_lookup/cold",&p,&[])}),
        );
        let boundary: Vec<_> = (0..16)
            .flat_map(|i| [i * TILE, (i + 1) * TILE - 1])
            .collect();
        let reverse: Vec<_> = boundary.iter().rev().copied().collect();
        let requests: [&[usize]; 6] = [
            &ORIGINAL,
            &boundary,
            &reverse,
            &[N - 1],
            &[TILE; 32],
            &ORIGINAL,
        ];
        for (index, ids) in requests.into_iter().enumerate() {
            step(
                &mut capture,
                &mut e,
                &embedding,
                "embedding",
                index,
                ids,
                &reference[..24576],
            );
            step(
                &mut capture,
                &mut p,
                &ple,
                "ple_lookup",
                index,
                ids,
                &reference[24576..],
            );
        }
        emit(
            "MATRIX_FINAL",
            json!({"role":"embedding","observation":matrix_state(&embedding)}),
        );
        emit(
            "MATRIX_FINAL",
            json!({"role":"ple_lookup","observation":matrix_state(&ple)}),
        );
        emit(
            "RESULT",
            json!({"tables":2,"chunks":32,"callbacks":32,"steps":12,"binary_bytes":capture.bytes,"actual_table_lookup_qualified":true,"whole_graph_qualified":false,"source_or_precision_failures_waived":false,"phone_qualified":false,"performance_qualified":false}),
        );
    });
}

mod immutable;
