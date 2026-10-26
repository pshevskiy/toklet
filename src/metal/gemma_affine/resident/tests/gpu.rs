//! Explicit small storage transaction witness; ordinary tests never create Metal.
use super::super::super::partitioned::tests::fixture;
use super::*;
use serde_json::{Value, json};
use std::io::{Read, Seek, SeekFrom, Write};

fn emit(label: &str, value: Value) {
    let mut output = std::io::stdout().lock();
    writeln!(output, "GEMMA_RESIDENT_TINY_{label} {value}").unwrap();
    output.flush().unwrap();
}
fn storage_state(storage: &Storage) -> Value {
    let buffer = |value: &Buffer<u32>| {
        json!({"allocation":format!("{:p}",&*value.raw),"words":value.len,
               "data":value.with_slice(<[u32]>::to_vec)})
    };
    let matrices:Vec<_>=storage.matrices.iter().map(|weights|json!({
        "rows":weights.layout().rows(),"columns":weights.layout().columns(),
        "owned_bytes":weights.owned_bytes(),"maximum_weight":weights.max_abs_weight(),
        "chunks":weights.views().map(|view|json!({"rows":view.layout.rows(),
            "codes":buffer(view.codes),"scales":buffer(view.scales),"biases":buffer(view.biases)})).collect::<Vec<_>>()
    })).collect();
    json!({"resident_bytes":storage.resident_bytes().unwrap(),"matrix_slots":storage.matrices.capacity(),
        "vector_slots":storage.vectors.capacity(),"matrices":matrices,
        "vectors":storage.vectors.iter().map(|v|json!({"values":v.values,"buffer":buffer(&v.buffer)})).collect::<Vec<_>>()})
}
fn outcome(label: &str, result: &Result<(Storage, LoadReport)>, old: &Storage) {
    let value = match result {
        Ok((storage, report)) => {
            json!({"key":label,"ok":true,"resident_bytes":report.resident_requested_bytes,
            "policy_bytes":report.admitted_load_policy_bytes,"source_bytes":report.maximum_source_capacity_bytes,
            "matrix_slots":report.matrix_owner_slots,"vector_slots":report.vector_owner_slots,
            "candidate":storage_state(storage)})
        }
        Err(error) => json!({"key":label,"ok":false,"error":error.to_string(),
                          "failed_candidate_buffers_captured":false}),
    };
    emit(
        "OUTCOME",
        json!({"outcome":value,"previous":storage_state(old)}),
    );
}
enum Change {
    None,
    Code(usize),
    Read(usize),
}
#[derive(Debug)]
struct OriginalReadFailure;
impl std::fmt::Display for OriginalReadFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("resident original late read failure")
    }
}
impl std::error::Error for OriginalReadFailure {}
struct Controlled {
    inner: Cursor<Vec<u8>>,
    change: Change,
    reads: usize,
    seeks: usize,
    authenticated_eof: bool,
}
impl Controlled {
    fn new(inner: Cursor<Vec<u8>>, change: Change) -> Self {
        Self {
            inner,
            change,
            reads: 0,
            seeks: 0,
            authenticated_eof: false,
        }
    }
    fn state(&self) -> Value {
        json!({"position":self.inner.position(),"bytes_read":self.reads,"seeks":self.seeks,"historical_eof_observed":self.authenticated_eof})
    }
}
impl Read for Controlled {
    fn read(&mut self, raw: &mut [u8]) -> std::io::Result<usize> {
        if self.authenticated_eof
            && let Change::Read(offset) = self.change
            && self.inner.position() as usize >= offset
        {
            return Err(std::io::Error::other(OriginalReadFailure));
        }
        let count = self.inner.read(raw)?;
        self.reads += count;
        if count == 0 && !raw.is_empty() && !self.authenticated_eof {
            self.authenticated_eof = true;
            if let Change::Code(offset) = self.change {
                self.inner.get_mut()[offset] ^= 1;
            }
        }
        Ok(count)
    }
}
impl Seek for Controlled {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.seeks += 1;
        self.inner.seek(position)
    }
}
fn inputs(
    data: &Value,
    bad_vector: bool,
    bad_metadata: bool,
) -> (source::Inputs, Cursor<Vec<u8>>, LoadOptions) {
    let matrices: Vec<_> = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(fixture::matrix)
        .collect();
    let mut owned: Vec<_> = matrices
        .iter()
        .map(|m| {
            let (a, b, c) = m.original_tensors();
            (m.layout(), [a.to_vec(), b.to_vec(), c.to_vec()])
        })
        .collect();
    if bad_metadata {
        owned[1].1[2][..2].copy_from_slice(&1_u16.to_le_bytes());
    }
    let arrays: Vec<_> = owned
        .iter()
        .map(|(layout, arrays)| (*layout, arrays.each_ref().map(Vec::as_slice)))
        .collect();
    let norm = [0, 128, 0, 0, 128, 63, 0, 191];
    let scalar = if bad_vector { [128, 127] } else { [128, 63] };
    source::test_support::stream(&arrays, &[&norm, &scalar])
}
#[test]
#[ignore = "explicit Metal source/ownership transaction qualification"]
fn immutable_bundle_loading_preserves_sources_failures_and_context_independence() {
    let data = fixture::fixture();
    let (mut source, mut reader, mut options) = inputs(&data, false, false);
    let process_allowance = options.available_bytes;
    let diagnostic_policy = 32 * 1024 * 1024;
    options.available_bytes -= diagnostic_policy + reader.get_ref().capacity();
    let plan = Plan::new(&source, 2, 2).unwrap();
    let minimum = plan.working(0, 0, plan.source).unwrap();
    emit(
        "PLAN",
        json!({"bundle_struct_bytes":size_of::<E2bWeights>(),"vector_struct_bytes":size_of::<ResidentVector>(),
        "weights_struct_bytes":size_of::<Weights>(),"minimum_policy_bytes":minimum,"base":plan.base,
        "payload":plan.payload,"source":plan.source,"cold_policy":COLD_POLICY,"context_policy":CONTEXT_ALLOWANCE_BYTES,
        "hash_stack_policy":HASH_STACK_POLICY,"real_e2b_model":false,"whole_graph_qualified":false,
        "process_allowance_bytes":process_allowance,"diagnostic_policy_bytes":diagnostic_policy,
        "reader_backing_bytes":reader.get_ref().capacity(),"initial_available_bytes":options.available_bytes}),
    );
    let (old, old_report) = autoreleasepool(|_| {
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        load_storage(&context, &mut source, &mut reader, &options).unwrap()
    });
    let identity = source.finish(&options).unwrap();
    drop(source);
    drop(reader);
    let original = storage_state(&old);
    emit(
        "INITIAL",
        json!({"source_context_dropped":true,"identity_components":identity.text_components(),
        "resident_bytes":old_report.resident_requested_bytes,"policy_bytes":old_report.admitted_load_policy_bytes,
        "source_bytes":old_report.maximum_source_capacity_bytes,"storage":original}),
    );
    let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    for index in 0..old.matrices.len() {
        let case = &data["cases"][index];
        let input = fixture::floats(&case["arrays"]["input"]);
        let expected = fixture::floats(&case["arrays"]["output_f32"]);
        let weights = &old.matrices[index];
        let mut workspace = context
            .projection_session(weights.layout(), TILE, 3, 1024 * 1024)
            .unwrap();
        let mut output = vec![f32::from_bits(SENTINEL); expected.len()];
        let result = workspace.apply(weights, &input, &mut output, 3);
        emit(
            "PROJECTION",
            json!({"matrix":index,"ok":result.is_ok(),"bits":output.iter().map(|v|v.to_bits()).collect::<Vec<_>>()}),
        );
        result.unwrap();
        assert_eq!(
            output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    for mode in [
        "budget",
        "changed",
        "late-read",
        "late-vector",
        "late-codebook",
        "valid-replacement",
    ] {
        let (mut candidate, reader, mut next) =
            inputs(&data, mode == "late-vector", mode == "late-codebook");
        let next_plan = Plan::new(&candidate, 2, 2).unwrap();
        let exact = next_plan.working(0, 0, next_plan.source).unwrap();
        next.available_bytes = if mode == "budget" {
            exact - 1
        } else {
            process_allowance
                - diagnostic_policy
                - old_report.resident_requested_bytes
                - reader.get_ref().capacity()
        };
        let change = match mode {
            "changed" => {
                Change::Code(candidate.regions[candidate.matrices[0].regions[0]].offset as usize)
            }
            "late-read" => {
                Change::Read(candidate.regions[candidate.vectors[1].region].offset as usize)
            }
            _ => Change::None,
        };
        let mut reader = Controlled::new(reader, change);
        let result = load_storage(&context, &mut candidate, &mut reader, &next);
        outcome(mode, &result, &old);
        emit("READER", json!({"key":mode,"state":reader.state()}));
        assert_eq!(storage_state(&old), original);
        match mode {
            "budget" => {
                assert!(
                    matches!(result,Err(Error::Budget{required,limit})if required==exact&&limit==exact-1)
                );
                assert_eq!(reader.inner.position(), 10);
                assert_eq!(reader.reads, 0);
                assert_eq!(reader.seeks, 0);
            }
            "late-read" => assert!(
                matches!(result,Err(Error::Weights(crate::weights::Error::Io{source,..}))if source.get_ref().unwrap().is::<OriginalReadFailure>())
            ),
            "late-codebook" => assert!(matches!(
                result,
                Err(Error::Unsupported("Gemma affine metadata domain"))
            )),
            "valid-replacement" => {
                let (new, report) = result.unwrap();
                let new_identity = candidate.finish(&next).unwrap();
                assert_eq!(new_identity, identity);
                assert!(report.admitted_load_policy_bytes <= next.available_bytes);
                assert_eq!(new.matrices.len(), 2);
                assert_eq!(new.vectors.len(), 2);
            }
            _ => assert!(result.is_err()),
        }
    }
    emit(
        "COMPLETE",
        json!({"passed":true,"steps":6,"context_dropped":true,"projection_values":90,
        "failed_candidate_buffers_captured":false,"real_e2b_model":false,"whole_graph_qualified":false,
        "source_or_precision_failures_waived":false,"performance_qualified":false}),
    );
}
