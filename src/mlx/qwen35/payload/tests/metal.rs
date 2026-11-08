//! Explicit source-backed FP32 projection diagnostic; not a Qwen GPU graph.
use super::*;
use crate::{
    metal::gemma_affine::{CONTEXT_ALLOWANCE_BYTES, Context, Projection},
    model::gemma_reference::tests::Measurement,
};
use std::{fs::File, io::Write, path::PathBuf, time::Instant};

const LIMIT: usize = 6 * 1024 * 1024 * 1024;
const DIAGNOSTIC: usize = 256 * 1024 * 1024;
const CHUNK: usize = 16384;
const BATCH: usize = 17;

#[test]
#[ignore = "explicit authenticated Qwen4B all249 FP32 Metal projections, not a GPU graph"]
fn actual_fp32_projections() {
    let directory = PathBuf::from(std::env::var_os("TOKLET_QWEN35_MODEL_DIRECTORY").unwrap());
    let operands = PathBuf::from(std::env::var_os("TOKLET_QWEN35_PROJECTION_INPUT").unwrap());
    let output = PathBuf::from(std::env::var_os("TOKLET_QWEN35_PROJECTION_CAPTURE").unwrap());
    std::fs::create_dir(&output).unwrap();
    let file = File::open(&operands).unwrap();
    assert!(
        file.metadata().unwrap().is_file() && file.metadata().unwrap().len() <= 8 * 1024 * 1024
    );
    let mut raw = Vec::new();
    file.take(8 * 1024 * 1024 + 1)
        .read_to_end(&mut raw)
        .unwrap();
    assert!(raw.len() <= 8 * 1024 * 1024);
    assert_eq!(
        format!("{:x}", Sha256::digest(&raw)),
        std::env::var("TOKLET_QWEN35_PROJECTION_INPUT_SHA256").unwrap()
    );
    let start = Instant::now();
    let store = load(
        CONFIG,
        File::open(directory.join("model.safetensors")).unwrap(),
        LoadOptions {
            memory_limit_bytes: LIMIT - DIAGNOSTIC,
            modality: ModalityPolicy::TextOnly,
        },
    )
    .unwrap();
    let load_ns = start.elapsed().as_nanos();
    let columns: usize = store.matrices.values().map(|p| p.layout.columns()).sum();
    let rows: usize = store.matrices.values().map(|p| p.layout.rows()).sum();
    assert_eq!(store.matrices.len(), 249);
    assert_eq!(rows, 1_380_352);
    assert_eq!(raw.len(), columns.checked_mul(4).unwrap());
    // CPU packed owner, context, caller inputs/outputs and projection scratch
    // coexist. Reserve all of them and upload overlap before any GPU owner.
    // DIAGNOSTIC covers cold maps/Vec owners/reporting/framework uncertainty.
    let caller_bytes = columns
        .checked_add(rows)
        .unwrap()
        .checked_mul(BATCH * 4)
        .unwrap();
    let scratch = store
        .matrices
        .values()
        .map(|p| Projection::minimum_owned_bytes(p.layout, CHUNK, BATCH).unwrap())
        .try_fold(0usize, usize::checked_add)
        .unwrap();
    let baseline = store
        .report
        .admitted_load_bytes
        .checked_add(CONTEXT_ALLOWANCE_BYTES)
        .unwrap()
        .checked_add(DIAGNOSTIC)
        .unwrap()
        .checked_add(raw.capacity())
        .unwrap()
        .checked_add(caller_bytes)
        .unwrap()
        .checked_add(scratch)
        .unwrap();
    assert!(baseline < LIMIT);
    let start = Instant::now();
    let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let mut owners = Vec::with_capacity(249);
    let mut workspaces = Vec::with_capacity(249);
    let mut inputs = Vec::with_capacity(249);
    let mut outputs = Vec::with_capacity(249);
    let mut cursor = 0;
    let mut resident_bytes = 0usize;
    let mut actual_scratch = 0usize;
    println!();
    for (name, p) in &store.matrices {
        let layout = p.layout;
        let available = LIMIT
            .checked_sub(baseline)
            .unwrap()
            .checked_sub(resident_bytes)
            .unwrap();
        let weights = context
            .weights_from_fill(
                layout,
                CHUNK,
                available,
                |first, child, codes, scales, biases| {
                    let span = layout.row_span(first..first + child.rows()).unwrap();
                    assert_eq!(span.layout(), child);
                    codes.copy_from_slice(&p.codes[span.weight_range()]);
                    scales.copy_from_slice(&p.scales[span.metadata_range()]);
                    biases.copy_from_slice(&p.biases[span.metadata_range()]);
                    Ok(())
                },
            )
            .unwrap();
        resident_bytes = resident_bytes.checked_add(weights.owned_bytes()).unwrap();
        assert!(baseline.checked_add(resident_bytes).unwrap() <= LIMIT);
        let session = context
            .projection_session(
                layout,
                CHUNK,
                BATCH,
                Projection::minimum_owned_bytes(layout, CHUNK, BATCH).unwrap(),
            )
            .unwrap();
        actual_scratch = actual_scratch.checked_add(session.owned_bytes()).unwrap();
        let mut input = vec![0.; BATCH * layout.columns()];
        for (token, row) in input.chunks_exact_mut(layout.columns()).enumerate() {
            let scalar = [1., -1., 0.5][token % 3];
            for (j, value) in row.iter_mut().enumerate() {
                let i = cursor + 4 * j;
                *value = f32::from_le_bytes(raw[i..i + 4].try_into().unwrap()) * scalar;
                assert!(value.is_finite());
            }
        }
        cursor += layout.columns() * 4;
        println!(
            "QWEN35_PROJECTION_OWNER {}",
            json!({"name":name,"rows":layout.rows(),"columns":layout.columns(),"gpu_weight_bytes":weights.owned_bytes(),"scratch_bytes":session.owned_bytes()})
        );
        owners.push(weights);
        workspaces.push(session);
        inputs.push(input);
        outputs.push(vec![f32::from_bits(0x5a17_0031); BATCH * layout.rows()]);
    }
    assert_eq!(cursor, raw.len());
    assert_eq!(actual_scratch, scratch);
    let cpu_views: Vec<_> = store
        .matrices
        .keys()
        .map(|name| store.matrix(name).unwrap())
        .collect();
    let setup_ns = start.elapsed().as_nanos();
    let mut capture = std::io::BufWriter::with_capacity(
        64 * 1024,
        File::create(output.join("rows.f32le")).unwrap(),
    );
    let mut offset = 0usize;
    for batch in [1, BATCH] {
        for (i, (name, p)) in store.matrices.iter().enumerate() {
            let input = &inputs[i][..batch * p.layout.columns()];
            let out = &mut outputs[i][..batch * p.layout.rows()];
            // Numerical preflight rejection must preserve caller outputs and workspace.
            let sentinel: Vec<_> = out.iter().map(|v| v.to_bits()).collect();
            let mut invalid = input.to_vec();
            invalid[0] = f32::NAN;
            assert!(
                workspaces[i]
                    .apply(&owners[i], &invalid, out, batch)
                    .is_err()
            );
            assert!(out.iter().map(|v| v.to_bits()).eq(sentinel));
            assert!(!workspaces[i].is_poisoned());
            workspaces[i].apply(&owners[i], input, out, batch).unwrap();
            let measure = Measurement::start();
            let result = workspaces[i].apply(&owners[i], input, out, batch);
            let allocations = measure.finish();
            result.unwrap();
            assert_eq!(allocations, 0);
            let mut digest = Sha256::new();
            for value in out.iter() {
                let bytes = value.to_le_bytes();
                capture.write_all(&bytes).unwrap();
                digest.update(bytes);
            }
            println!(
                "QWEN35_PROJECTION_ARRAY {}",
                json!({"name":name,"batch":batch,"rows":p.layout.rows(),"columns":p.layout.columns(),"offset":offset,"bytes":out.len()*4,"sha256":format!("{:x}",digest.finalize()),"warmed_rust_allocations":allocations,"preflight_transactional":true})
            );
            offset += out.len() * 4;
        }
    }
    capture.flush().unwrap();
    capture.get_ref().sync_all().unwrap();
    assert_eq!(offset, 24_846_336 * 4);
    // Alternating CPU/GPU first order by iteration/cohort. Diagnostic I/O and
    // allocation observation stay outside completed sequence timing.
    for cohort in 0..2 {
        for iteration in 0..13 {
            for slot in 0..2 {
                let gpu = (slot + iteration + cohort) % 2 == 1;
                let start = Instant::now();
                for (i, p) in store.matrices.values().enumerate() {
                    let input = &inputs[i][..p.layout.columns()];
                    let out = &mut outputs[i][..p.layout.rows()];
                    if gpu {
                        workspaces[i].apply(&owners[i], input, out, 1).unwrap();
                    } else {
                        cpu_views[i].matvec(input, out).unwrap();
                    }
                    std::hint::black_box(&*out);
                }
                let elapsed_ns = start.elapsed().as_nanos();
                println!(
                    "QWEN35_PROJECTION_SAMPLE {}",
                    json!({"cohort":cohort,"iteration":iteration,"warmup":iteration<3,"engine":if gpu {"metal_fp32"} else {"scalar_cpu"},"elapsed_ns":elapsed_ns,"projections":249})
                );
            }
        }
    }
    println!(
        "QWEN35_PROJECTION_REPORT {}",
        json!({"source_sha256":CONTAINER_SHA256,"input_sha256":format!("{:x}",Sha256::digest(&raw)),"matrices":249,"batches":[1,17],"capture_bytes":offset,"load_ns":load_ns,"setup_ns":setup_ns,"admitted_bytes":baseline+resident_bytes,"packed_gpu_bytes":resident_bytes,"scratch_bytes":scratch,"caller_bytes":caller_bytes,"warmed_rust_allocations":0,"gpu_graph_qualified":false,"quality_qualified":false,"managed_adoption_qualified":false})
    );
}
