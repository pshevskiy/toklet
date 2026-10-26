//! Same frozen original tables and capture protocol, with weight-only owners.
use super::*;

fn sizes(k: usize) -> (usize, usize, usize) {
    let (owner, chunk) = Weights::diagnostic_sizes();
    let layout = layout(k);
    let resident = layout.total_bytes() + 16 * 3 * GUARD * 4 + 16 * chunk + owner;
    let source = TILE * k / 2 + 2 * TILE * k / 32 + 3 * size_of::<Vec<u8>>();
    (
        resident,
        Sizes::new(k, CAP, usize::MAX).unwrap().minimum_admitted,
        source,
    )
}
fn policy() -> Value {
    let (e, eg, es) = sizes(1536);
    let (p, pg, ps) = sizes(8960);
    let categories = json!({"resident_owners":e+p,"gather_workspaces":eg+pg,"source_staging":es.max(ps),"two_context_policies":2*CONTEXT_ALLOWANCE_BYTES,
        "header_bytes":318228,"header_and_reference_parser_policy":8388608,"frozen_lookup":167936,"frozen_metadata_policy":131072,
        "caller_f32":32*8960*4,"serialization_policy":8388608,"hash_and_io_stack":131072,"diagnostic_metadata_policy":8388608});
    let total = categories
        .as_object()
        .unwrap()
        .values()
        .try_fold(0_u64, |sum, v| sum.checked_add(v.as_u64().unwrap()))
        .unwrap();
    assert!(total <= PROCESS_CAP as u64);
    json!({"categories":categories,"total_policy_bytes":total,"native_process_allowance_bytes":PROCESS_CAP,"mac_working_set_ceiling_bytes":6_u64*1024*1024*1024,"exact_physical_residency":false,"phone_qualified":false})
}
fn observation(weights: &Weights) -> Value {
    let (slots, chunk_struct_bytes, device) = weights.diagnostic_storage();
    let children:Vec<_>=weights.views().map(|v|json!({"rows":v.layout.rows(),"codes":buffer_state(v.codes),"scales":buffer_state(v.scales),"biases":buffer_state(v.biases)})).collect();
    json!({"rows":weights.layout().rows(),"columns":weights.layout().columns(),"owned_bytes":weights.owned_bytes(),"owner_slots":slots,"chunk_struct_bytes":chunk_struct_bytes,"device":device,"children":children})
}
fn upload(
    context: &Context,
    file: &mut File,
    header: &Value,
    role: &str,
    stem: &str,
    k: usize,
) -> Weights {
    let (resident, _, source) = sizes(k);
    let spans = source_spans(header, stem, k);
    emit(
        "UPLOAD_ATTEMPT",
        json!({"role":role,"columns":k,"maximum_rows":TILE,"working_budget":resident+source,"spans":spans}),
    );
    let result=context.weights_from_fill(layout(k),TILE,resident+source,|first,child,codes,scales,biases|{
        let arrays=[codes,scales,biases];
        emit("FILL_ATTEMPT",json!({"role":role,"first_row":first,"rows":child.rows(),"addresses":arrays.each_ref().map(|v|format!("{:p}",v.as_ptr())),"lengths":arrays.each_ref().map(|v|v.len())}));
        let mut consumed=Vec::new();
        for(array,span)in arrays.into_iter().zip(spans.as_array().unwrap()){
            let offset=span["offset"].as_u64().unwrap()+first as u64*span["stride"].as_u64().unwrap();
            file.seek(SeekFrom::Start(offset)).map_err(|e|Error::Native(e.to_string()))?;
            file.read_exact(array).map_err(|e|Error::Native(e.to_string()))?;
            consumed.push(json!({"offset":offset,"bytes":array.len(),"sha256":hash(array)}));
        }
        emit("FILL",json!({"role":role,"first_row":first,"consumed":consumed}));Ok(())
    });
    emit(
        "UPLOAD_OUTCOME",
        json!({"role":role,"completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string)}),
    );
    let weights = result.unwrap();
    emit(
        "IMMUTABLE",
        json!({"role":role,"source_cpu_bytes_retained":0,"observation":observation(&weights)}),
    );
    assert_eq!(weights.owned_bytes(), resident);
    weights
}
fn step(
    capture: &mut Capture,
    session: &mut Gather<'_>,
    weights: &Weights,
    role: &str,
    index: usize,
    ids: &[usize],
    reference: &[u8],
) {
    let key = format!("{role}/{index}");
    let mut caller = vec![42.; ids.len() * weights.layout().columns()];
    assert_eq!(caller.capacity(), caller.len());
    emit(
        "STEP_ATTEMPT",
        json!({"key":key,"role":role,"index":index,"ids":ids,"caller_capacity":caller.capacity()}),
    );
    let result = session.apply_weights(weights, ids, &mut caller);
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
        let raw: Vec<_> = caller.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(raw, reference);
    }
}
#[test]
fn immutable_real_table_plan_separates_weights_from_both_session_workspaces_without_gpu() {
    assert_eq!(sizes(1536), (226495664, 17367416, 14155848));
    assert_eq!(sizes(8960), (1321209008, 20218232, 82575432));
    assert!(policy()["total_policy_bytes"].as_u64().unwrap() < PROCESS_CAP as u64);
}
#[test]
#[ignore = "explicit complete immutable real tables outliving construction Context with resident lookups"]
fn immutable_real_tables_outlive_context_and_share_lookup_workspaces() {
    autoreleasepool(|_| {
        emit("PLAN", policy());
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
        let (embedding, ple) = autoreleasepool(|_| {
            let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
            let (owner, chunk) = Weights::diagnostic_sizes();
            emit(
                "META",
                json!({"device":context.device_name(),"device_identity":format!("{:p}",&*context.gpu.device),"owner_struct_bytes":owner,"chunk_struct_bytes":chunk,"gather_struct_bytes":size_of::<Gather<'_>>(),"header_capacity":headerraw.capacity(),"reference_capacity":reference.capacity(),"binary_cap_bytes":RAW_CAP,"whole_graph_qualified":false,"performance_qualified":false}),
            );
            (
                upload(
                    &context,
                    &mut source,
                    &header,
                    "embedding",
                    "embed_tokens",
                    1536,
                ),
                upload(
                    &context,
                    &mut source,
                    &header,
                    "ple_lookup",
                    "embed_tokens_per_layer",
                    8960,
                ),
            )
        });
        // Both returned owners have no Context borrow and own retained devices.
        // Only identity metadata is inspected here; full bytes are captured
        // cold above and after successful GPU consumption below.
        emit(
            "CONSTRUCTION_CONTEXT_DROPPED",
            json!({"weights":2,"devices":[embedding.diagnostic_storage().2,ple.diagnostic_storage().2]}),
        );
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        emit(
            "LOOKUP_CONTEXT",
            json!({"device":context.device_name(),"device_identity":format!("{:p}",&*context.gpu.device),"same_devices":[embedding.same_device(&context),ple.same_device(&context)]}),
        );
        assert!(embedding.same_device(&context) && ple.same_device(&context));
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
            "IMMUTABLE_FINAL",
            json!({"role":"embedding","observation":observation(&embedding)}),
        );
        emit(
            "IMMUTABLE_FINAL",
            json!({"role":"ple_lookup","observation":observation(&ple)}),
        );
        emit(
            "RESULT",
            json!({"tables":2,"chunks":32,"callbacks":32,"steps":12,"binary_bytes":capture.bytes,"construction_context_dropped":true,"actual_table_lookup_qualified":true,"whole_graph_qualified":false,"source_or_precision_failures_waived":false,"foreign_device_qualified":false,"phone_qualified":false,"performance_qualified":false}),
        );
    });
}
