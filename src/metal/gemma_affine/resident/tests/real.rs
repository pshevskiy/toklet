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
#[ignore = "explicit pinned QAT immutable mixed storage, no graph"]
fn complete_qat_storage_authenticates_and_owns_mixed_payload() {
    let root = std::path::PathBuf::from(std::env::var_os("TOKLET_GEMMA_QAT_METADATA").unwrap());
    let path = root.join("model/model.safetensors");
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&path)
        .unwrap();
    let before = identity(&file.metadata().unwrap());
    assert_eq!(before, identity(&std::fs::symlink_metadata(&path).unwrap()));
    let config = std::fs::read(root.join("config.json")).unwrap();
    let index = std::fs::read(root.join("model.safetensors.index.json")).unwrap();
    let options = LoadOptions {
        container_bytes: QAT_CONTAINER_BYTES,
        container_sha256: QAT_CONTAINER_SHA256,
        available_bytes: 6 * 1024 * 1024 * 1024,
        modality: ModalityPolicy::TextOnly,
    };
    let prepared =
        source::Prepared::read(&config, &index, &mut file, &options, SourceKind::QatMixed).unwrap();
    let plan = Plan::new(&prepared.inputs, 278, 262).unwrap();
    let required = plan.working(0, 0, plan.source).unwrap();
    let (original, _, _) = real_metadata();
    let affine4 = Plan::new(&original.inputs, 278, 262).unwrap();
    assert_eq!(plan.base, affine4.base);
    assert_eq!(plan.source, affine4.source);
    assert_eq!(plan.payload - affine4.payload, 778_567_680);
    assert_eq!(size_of::<QatWeights>(), size_of::<E2bWeights>());
    let resident = autoreleasepool(|_| {
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let count = Cell::new(0);
        let rejected = QatWeights::load_text_only(
            &context,
            &config,
            &index,
            Counted {
                file: &mut file,
                bytes: &count,
            },
            required - 1,
        );
        assert!(
            matches!(rejected,Err(Error::Budget { required: bytes, limit })
            if bytes==required && limit==required-1)
        );
        assert_eq!(count.get(), 320_319);
        QatWeights::load_text_only(&context, &config, &index, &mut file, required).unwrap()
    });
    let report = resident.report();
    assert_eq!(report.resident_requested_bytes, plan.base + plan.payload);
    assert_eq!(report.admitted_load_policy_bytes, required);
    assert_eq!(report.maximum_source_capacity_bytes, plan.source);
    assert_eq!((report.matrices, report.vectors), (278, 262));
    assert_eq!(
        resident.identity().container_sha256(),
        &QAT_CONTAINER_SHA256
    );
    assert_eq!(resident.identity().text_components(), 1096);
    let affine8: Vec<_> = resident
        .0
        .storage
        .matrices
        .iter()
        .enumerate()
        .filter(|(_, matrix)| matrix.layout().bits() == 8)
        .collect();
    assert_eq!(affine8.len(), 105);
    for &(matrix_index, matrix) in [affine8.first().unwrap(), affine8.last().unwrap()] {
        let mut views = matrix.views();
        let view = views.next().unwrap();
        assert!(views.next().is_none());
        let regions = prepared.inputs.matrices[matrix_index].regions;
        for (region, buffer) in regions
            .into_iter()
            .zip([view.codes, view.scales, view.biases])
        {
            let source = &prepared.inputs.regions[region];
            file.seek(SeekFrom::Start(source.offset)).unwrap();
            let mut original = Sha256::new();
            let mut temporary = [0_u8; 4096];
            let mut remaining = source.bytes;
            while remaining > 0 {
                let count = remaining.min(temporary.len());
                file.read_exact(&mut temporary[..count]).unwrap();
                original.update(&temporary[..count]);
                remaining -= count;
            }
            let retained = buffer.with_slice(|words| {
                let mut digest = Sha256::new();
                let mut remaining = source.bytes;
                for word in words {
                    let count = remaining.min(4);
                    if count == 0 {
                        assert_eq!(*word, SENTINEL);
                    } else {
                        digest.update(&word.to_le_bytes()[..count]);
                        remaining -= count;
                    }
                }
                assert_eq!(remaining, 0);
                digest.finalize()
            });
            assert_eq!(original.finalize(), retained);
        }
    }
    assert_eq!(before, identity(&file.metadata().unwrap()));
    assert_eq!(before, identity(&std::fs::symlink_metadata(&path).unwrap()));
    emit(
        "QAT_STORAGE",
        json!({"resident_bytes":report.resident_requested_bytes,
        "policy_bytes":report.admitted_load_policy_bytes,"source_bytes":report.maximum_source_capacity_bytes,
        "matrices":report.matrices,"affine8":affine8.len(),"vectors":report.vectors,
        "budget_minus_one_read_bytes":320319,"checked_original_components":6}),
    );
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
