//! Prepared CPU BF16 projection baselines; no model graph or throughput claim.
use half::bf16;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{hint::black_box, time::Instant};
use toklet::cpu::lfm3_linear_bf16;

const META: &[u8] = include_bytes!("../../tests/fixtures/lfm3-projection-bf16-v1/fixture.json");
const SOURCE: &[u8] = include_bytes!("../../tests/fixtures/lfm3-projection-bf16-v1/source.bf16le");

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn input(ordinal: usize, tokens: usize, columns: usize) -> Vec<bf16> {
    let pool: Vec<u16> = [0, 0x8000]
        .into_iter()
        .flat_map(|sign| (0..=0x4180).map(move |v| v | sign))
        .collect();
    (0..tokens * columns)
        .map(|index| {
            let selector = index * 37 + index / columns * 11 + ordinal * 3;
            bf16::from_bits(if index % 2 == 1 {
                ((0x3d80 + selector % 1025) | ((selector / 1025 % 2) << 15)) as u16
            } else {
                pool[selector % pool.len()]
            })
        })
        .collect()
}
fn word_sha(values: &[bf16]) -> String {
    let mut h = Sha256::new();
    for value in values {
        h.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn verify_source(output: &[bf16], tokens: usize, rows: usize, ordinal: usize) -> usize {
    assert_eq!(
        sha(META),
        "0f5f8a970d116b32073aa6090bd64b91825f39db2adfc9311eb5995b6367cd74"
    );
    let meta: Value = serde_json::from_slice(META).unwrap();
    let matrix = &meta["matrices"][ordinal];
    let case = meta["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| {
            c["ordinal"].as_u64() == Some(ordinal as u64) && c["tokens"].as_u64() == Some(128)
        })
        .unwrap();
    let start = case["output"]["byte_offset"].as_u64().unwrap() as usize;
    let bytes = case["output"]["byte_count"].as_u64().unwrap() as usize;
    assert_eq!(
        sha(&SOURCE[start..start + bytes]),
        case["output"]["sha256"].as_str().unwrap()
    );
    let mut checked = 0;
    for (selected, row) in matrix["selected_rows"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let row = row.as_u64().unwrap() as usize;
        if row >= rows {
            continue;
        }
        for token in 0..tokens {
            let offset = start + (token * 4 + selected) * 2;
            let expected = u16::from_le_bytes(SOURCE[offset..offset + 2].try_into().unwrap());
            assert_eq!(output[token * rows + row].to_bits(), expected);
            checked += 1;
        }
    }
    assert!(checked > 0);
    checked
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err("release build required".into());
    }
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 6 || args[0] != "--profile" || args[2] != "--tokens" || args[4] != "--weights"
    {
        return Err("usage: --profile PROFILE --tokens 1|32|128 --weights BANK".into());
    }
    let (rows, columns, ordinal, expected) = match args[1].as_str() {
        "conv-in" => (
            6144,
            2048,
            1,
            "6f0734e0164573ba1b9a3ae4a6cda2e95271eb9b9069b14a722d282bd445bafb",
        ),
        "conv-out" => (
            2048,
            2048,
            2,
            "dcc9ea78253ce5851ac8d11d4b8c27fae238234cfcb456974af0d9b947bcb625",
        ),
        "mlp-up" => (
            7680,
            2048,
            3,
            "2e17e3b36eb42bf8b1bb4bef63bf0a43405b7299125f39783af202dfecb97534",
        ),
        "mlp-down" => (
            2048,
            7680,
            4,
            "a04e4df91c6c0fcb37ea2fc5c9db171b6540a7903c039413efbaaa4361780659",
        ),
        "attention-gate" => (
            16,
            2048,
            79,
            "310f3e94d844ad80577f543d6f8ff86f139114061a9987d1b0542316f4ea9ab4",
        ),
        "attention-kv" => (
            512,
            2048,
            80,
            "b0454e5463eedb6be30f9737475620e0731a826cad0fea630f95fda43d64c10e",
        ),
        "tied-head-tile" => (
            1024,
            2048,
            0,
            "fd5bc035cf51258776e90a63433d8bb980f218676a0934fde3f1aab029e4efe3",
        ),
        _ => return Err("unknown projection profile".into()),
    };
    let tokens: usize = args[3].parse()?;
    if !matches!(tokens, 1 | 32 | 128) {
        return Err("unsupported token count".into());
    }
    let begin = Instant::now();
    let metadata = std::fs::metadata(&args[5])?;
    if !metadata.is_file() || metadata.len() != (rows * columns * 2) as u64 {
        return Err("source coefficient extent differs".into());
    }
    let raw = std::fs::read(&args[5])?;
    if sha(&raw) != expected {
        return Err("authenticated source bank differs".into());
    }
    let raw_bytes = raw.capacity();
    let weights: Vec<_> = raw
        .chunks_exact(2)
        .map(|b| bf16::from_le_bytes(b.try_into().unwrap()))
        .collect();
    drop(raw);
    let bank_read_hash_decode_ms = begin.elapsed().as_secs_f64() * 1000.;
    let begin = Instant::now();
    let x = input(ordinal, tokens, columns);
    let mut output = vec![bf16::ZERO; tokens * rows];
    let caller_preparation_ms = begin.elapsed().as_secs_f64() * 1000.;
    let begin = Instant::now();
    lfm3_linear_bf16(black_box(&x), &weights, &mut output, tokens, columns, rows)?;
    let first_ns = begin.elapsed().as_nanos();
    let verified = verify_source(&output, tokens, rows, ordinal);
    let mut warm_ns = Vec::with_capacity(10);
    for trial in 0..13 {
        let begin = Instant::now();
        lfm3_linear_bf16(black_box(&x), &weights, &mut output, tokens, columns, rows)?;
        let ns = begin.elapsed().as_nanos();
        black_box(&output);
        if trial >= 3 {
            warm_ns.push(ns);
        }
    }
    assert_eq!(verify_source(&output, tokens, rows, ordinal), verified);
    println!(
        "{}",
        serde_json::json!({
            "format":"toklet-lfm3-projection-bf16-micro-v1","profile":args[1],"storage":"BF16",
            "rows":rows,"columns":columns,"tokens":tokens,"source_matrix_ordinal":ordinal,
            "weight_sha256":expected,"input_sha256":word_sha(&x),"warmups":3,"measured_trials":10,
            "bank_read_hash_decode_ms":bank_read_hash_decode_ms,"caller_preparation_ms":caller_preparation_ms,
            "first_projection_ns":first_ns,"warm_projection_ns":warm_ns,
            "warm_projection_ms":warm_ns.iter().map(|v|*v as f64/1e6).collect::<Vec<_>>(),
            "source_coordinates_checked_first_and_last":verified,"full_matrix":args[1]!="tied-head-tile",
            "owned_weight_bytes":weights.capacity()*2,"caller_input_output_bytes":(x.capacity()+output.capacity())*2,
        "load_transient_raw_capacity_bytes":raw_bytes,"numerical_buffer_header_bytes":3*std::mem::size_of::<Vec<bf16>>(),
            "embedded_fixture_bytes":META.len()+SOURCE.len(),"logical_source_partial_float_count":36,
            "logical_main_multiply_adds":tokens*rows*columns,
            "scope":"One resident source matrix call including public guards/stores, six full representative matrices or explicitly partial1024row tied-head tile. First/warm projections exclude separately named bank read/hash/decode and caller preparation. Source authentication may warm storage. No graph, full128000row head, request/TTFT/throughput/dtype speedup or quality/adoption qualification.",
            "model_execution_qualified":false,"quality_adoption_qualified":false
        })
    );
    Ok(())
}
