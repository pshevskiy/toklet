//! Whole-checkpoint storage evidence; never expands a packed matrix on the CPU.
use super::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

fn emit(label: &str, value: Value) {
    let mut output = std::io::stdout().lock();
    writeln!(output, "GEMMA_RESIDENT_REAL_{label} {value}").unwrap();
    output.flush().unwrap();
}
fn identity(metadata: &std::fs::Metadata) -> [u64; 7] {
    [
        metadata.dev(),
        metadata.ino(),
        metadata.len(),
        metadata.mtime() as u64,
        metadata.mtime_nsec() as u64,
        metadata.ctime() as u64,
        metadata.ctime_nsec() as u64,
    ]
}
struct Counted<'a> {
    file: &'a mut File,
    bytes: &'a Cell<usize>,
}
impl Read for Counted<'_> {
    fn read(&mut self, destination: &mut [u8]) -> std::io::Result<usize> {
        let count = self.file.read(destination)?;
        self.bytes.set(self.bytes.get() + count);
        Ok(count)
    }
}
impl Seek for Counted<'_> {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.file.seek(position)
    }
}
fn buffer(value: &Buffer<u32>) -> Value {
    let digest = value.with_slice(|words| {
        let mut digest = Sha256::new();
        let mut bytes = [0_u8; 4096];
        for part in words.chunks(1024) {
            for (word, target) in part.iter().zip(bytes.chunks_exact_mut(4)) {
                target.copy_from_slice(&word.to_le_bytes());
            }
            digest.update(&bytes[..part.len() * 4]);
        }
        format!("{:x}", digest.finalize())
    });
    json!({"allocation":format!("{:p}",&*value.raw),"bytes":value.len*4,"sha256":digest})
}
#[test]
#[ignore = "explicit complete original Gemma E2B immutable GPU storage qualification"]
fn complete_original_e2b_storage_matches_independent_inventory_after_context_teardown() {
    let path = std::env::var_os("TOKLET_GEMMA_RESIDENT_MODEL").expect("explicit model path");
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&path)
        .unwrap();
    let metadata = file.metadata().unwrap();
    assert!(metadata.is_file());
    let before = identity(&metadata);
    assert_eq!(before, identity(&std::fs::symlink_metadata(&path).unwrap()));
    let config = include_bytes!("../../../../../tests/fixtures/gemma4-e2b-metadata-v1/config.json");
    let index = include_bytes!("../../../../../tests/fixtures/gemma4-e2b-metadata-v1/index.json");
    let header = include_bytes!("../../../../../tests/fixtures/gemma4-e2b-metadata-v1/header.json");
    let process_policy = 6 * 1024 * 1024 * 1024;
    let diagnostic_policy = 32 * 1024 * 1024;
    let mut options = LoadOptions {
        container_bytes: 3550670554,
        container_sha256: [
            3, 142, 57, 163, 122, 118, 103, 55, 61, 44, 57, 145, 55, 84, 70, 177, 12, 150, 174, 29,
            113, 122, 104, 103, 72, 112, 52, 61, 179, 118, 183, 110,
        ],
        available_bytes: process_policy - diagnostic_policy,
        modality: ModalityPolicy::TextOnly,
    };
    let (prepared, _, _) = real_metadata();
    let minimum = Plan::new(&prepared.inputs, 278, 262)
        .unwrap()
        .working(0, 0, 82575432)
        .unwrap();
    drop(prepared);
    let weights = autoreleasepool(|_| {
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let bytes = Cell::new(0);
        options.available_bytes = minimum - 1;
        let rejected = E2bWeights::load(
            &context,
            config,
            index,
            Counted {
                file: &mut file,
                bytes: &bytes,
            },
            options,
        );
        emit(
            "BUDGET",
            json!({"rejected":rejected.is_err(),"required":minimum,
            "limit":options.available_bytes,"source_bytes_read":bytes.get(),
            "error":rejected.as_ref().err().map(ToString::to_string)}),
        );
        assert!(
            matches!(rejected,Err(Error::Budget{required,limit})if required==minimum&&limit==minimum-1)
        );
        assert_eq!(bytes.get(), 8 + header.len());
        options.available_bytes = process_policy - diagnostic_policy;
        let result = E2bWeights::load(&context, config, index, &mut file, options);
        emit(
            "LOAD",
            json!({"ok":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string)}),
        );
        result.unwrap()
    });
    let report = weights.report();
    emit(
        "REPORT",
        json!({"context_dropped":true,"matrices":weights.matrix_count(),
        "vectors":weights.vector_count(),"identity_components":weights.identity().text_components(),
        "resident_bytes":report.resident_requested_bytes,"policy_bytes":report.admitted_load_policy_bytes,
        "source_bytes":report.maximum_source_capacity_bytes,"matrix_slots":report.matrix_owner_slots,
        "vector_slots":report.vector_owner_slots,"bundle_abi":size_of::<E2bWeights>(),
        "vector_abi":size_of::<ResidentVector>(),"weights_abi":size_of::<Weights>(),
        "process_policy_bytes":process_policy,"diagnostic_policy_bytes":diagnostic_policy,
        "available_bytes":options.available_bytes,
        "config_digest":weights.identity().config_sha256(),"index_digest":weights.identity().index_sha256(),
        "header_digest":weights.identity().header_sha256(),"container_bytes":weights.identity().container_bytes(),
        "container_identity":weights.identity().container_sha256().iter().map(|b|format!("{b:02x}")).collect::<String>()}),
    );
    for (number, matrix) in weights.storage.matrices.iter().enumerate() {
        let chunks: Vec<_> = matrix
            .views()
            .enumerate()
            .map(|(chunk, v)| {
                json!({"first":chunk*TILE,"rows":v.layout.rows(),
            "codes":buffer(v.codes),"scales":buffer(v.scales),"biases":buffer(v.biases)})
            })
            .collect();
        emit(
            "MATRIX",
            json!({"index":number,"rows":matrix.layout().rows(),
            "columns":matrix.layout().columns(),"owned_bytes":matrix.owned_bytes(),
            "chunk_slots":matrix.diagnostic_storage().0,"chunk_abi":matrix.diagnostic_storage().1,
            "device":matrix.diagnostic_storage().2,"chunks":chunks}),
        );
    }
    for (number, vector) in weights.storage.vectors.iter().enumerate() {
        emit(
            "VECTOR",
            json!({"index":number,"values":vector.values,"buffer":buffer(&vector.buffer)}),
        );
    }
    assert_eq!(before, identity(&file.metadata().unwrap()));
    assert_eq!(before, identity(&std::fs::symlink_metadata(&path).unwrap()));
    assert_eq!(
        weights.storage.resident_bytes().unwrap(),
        report.resident_requested_bytes
    );
    assert!(report.admitted_load_policy_bytes <= options.available_bytes);
    emit(
        "COMPLETE",
        json!({"passed":true,"context_dropped":true,"model_matrices":278,
        "model_vectors":262,"source_path_and_descriptor_unchanged":true,
        "whole_graph_qualified":false,"performance_qualified":false,"phone_qualified":false,
        "source_or_precision_failures_waived":false}),
    );
}
