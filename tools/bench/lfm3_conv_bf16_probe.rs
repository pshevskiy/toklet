//! Prepared canonical BF16 convolution/binary operator baseline; no model graph.
use half::bf16;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{hint::black_box, io, time::Instant};
use toklet::cpu::{self, lfm3_add_bf16, lfm3_conv7_bf16, lfm3_head_gate_bf16, lfm3_mul_bf16};

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "usage: lfm3_conv_bf16_probe --tokens 1|32|128",
    )
}

fn options() -> Result<usize, io::Error> {
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() != Some("--tokens") {
        return Err(invalid());
    }
    let tokens = args
        .next()
        .ok_or_else(invalid)?
        .parse()
        .map_err(|_| invalid())?;
    if args.next().is_some() || !matches!(tokens, 1 | 32 | 128) {
        return Err(invalid());
    }
    Ok(tokens)
}

fn phase<F: FnMut() -> Result<(), cpu::Error>>(
    repeats: usize,
    mut work: F,
) -> Result<f64, cpu::Error> {
    let begin = Instant::now();
    for _ in 0..repeats {
        work()?;
    }
    Ok(begin.elapsed().as_secs_f64() * 1000. / repeats as f64)
}

fn input_hash(values: &[bf16]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tokens = options()?;
    let manifest = include_bytes!("../../tests/fixtures/lfm3-conv-bf16-v1/parent.json");
    let raw = include_bytes!("../../tests/fixtures/lfm3-conv-bf16-v1/weights.bf16le");
    if format!("{:x}", Sha256::digest(manifest))
        != "c90abf0c6fed114ba8f3e1459d3a5e82a4fd94c0347249bde41cac0c1794493a"
        || format!("{:x}", Sha256::digest(raw))
            != "f305ef6e0054d9128004f7ac9c4eccb7dd5bf0e0f1bc3143ef39de3355ed9e62"
        || raw.len() != 13 * 2048 * 7 * 2
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Pinned source coefficient banks differ",
        )
        .into());
    }
    // Frozen banks are already in fixture/name order. Decode once, with no
    // tensor-name lookup, descriptor allocation or conversion in timed work.
    let weights: Vec<_> = raw
        .chunks_exact(2)
        .map(|v| bf16::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let input: Vec<_> = (0..tokens * 7680)
        .map(|i| bf16::from_f32(((i * 13 % 257) as f32 - 128.) / 32.))
        .collect();
    let right: Vec<_> = (0..input.len())
        .map(|i| bf16::from_f32(((i * 17 % 257) as f32 - 128.) / 32.))
        .collect();
    let history: Vec<_> = (0..6 * 2048)
        .map(|i| bf16::from_f32(((i * 19 % 257) as f32 - 128.) / 32.))
        .collect();
    let gates: Vec<_> = (0..tokens * 16)
        .map(|i| bf16::from_f32((i * 23 % 129) as f32 / 128.))
        .collect();
    let mut output = vec![bf16::ZERO; input.len()];
    let mut conv_output = vec![bf16::ZERO; tokens * 2048];
    let repetitions = (32 / tokens).max(1);
    let mut samples: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::with_capacity(10));
    let mut first = [0.; 6];
    for trial in 0..14 {
        let repeats = if trial == 0 { 1 } else { repetitions };
        let begin = Instant::now();
        let conv = phase(repeats, || {
            for bank in weights.chunks_exact(2048 * 7) {
                lfm3_conv7_bf16(
                    black_box(&input[..tokens * 2048]),
                    bank,
                    &history,
                    &mut conv_output,
                    tokens,
                    2048,
                )?;
            }
            Ok(())
        })?;
        black_box(&conv_output);
        let conv_product = phase(repeats, || {
            for _ in 0..26 {
                lfm3_mul_bf16(
                    black_box(&input[..tokens * 2048]),
                    &right[..tokens * 2048],
                    &mut output[..tokens * 2048],
                )?;
            }
            Ok(())
        })?;
        black_box(&output);
        let mlp_product = phase(repeats, || {
            for _ in 0..24 {
                lfm3_mul_bf16(black_box(&input), &right, &mut output)?;
            }
            Ok(())
        })?;
        black_box(&output);
        let residual = phase(repeats, || {
            for _ in 0..48 {
                lfm3_add_bf16(
                    black_box(&input[..tokens * 2048]),
                    &right[..tokens * 2048],
                    &mut output[..tokens * 2048],
                )?;
            }
            Ok(())
        })?;
        black_box(&output);
        let head_gate = phase(repeats, || {
            for _ in 0..11 {
                lfm3_head_gate_bf16(
                    black_box(&input[..tokens * 2048]),
                    &gates,
                    &mut conv_output,
                    tokens,
                )?;
            }
            Ok(())
        })?;
        black_box(&conv_output);
        let total = begin.elapsed().as_secs_f64() * 1000. / repeats as f64;
        let observed = [conv, conv_product, mlp_product, residual, head_gate, total];
        if trial == 0 {
            first = observed;
        } else if trial >= 4 {
            for (samples, value) in samples.iter_mut().zip(observed) {
                samples.push(value);
            }
        }
    }
    println!(
        "{}",
        json!({
            "format":"toklet-lfm3-bf16-convolution-binary-micro-v1","profile":"lfm3-parent-bf16-cpu-v1",
            "tokens":tokens,"channels":2048,"kernel":7,"coefficient_banks":13,
            "convolution_calls":13,"conv_product_calls":26,"conv_product_values":26*tokens*2048,
            "mlp_product_calls":24,"mlp_product_values":24*tokens*7680,
            "residual_calls":48,"residual_values":48*tokens*2048,"head_gate_calls":11,
            "head_gate_values":11*tokens*16*128,"warmups":3,"measured_trials":10,
            "repetitions_per_trial":repetitions,"first_conv_conv_product_mlp_product_residual_head_gate_total_ms":first,
            "warm_conv_ms":samples[0],"warm_conv_product_ms":samples[1],"warm_mlp_product_ms":samples[2],
            "warm_residual_ms":samples[3],"warm_head_gate_ms":samples[4],"warm_component_total_ms":samples[5],
            "owned_coefficient_bytes":weights.capacity()*2,"owned_descriptor_bytes":0,
            "caller_input_output_history_gate_bytes":(input.capacity()+right.capacity()+history.capacity()+gates.capacity()+output.capacity()+conv_output.capacity())*2,
            "embedded_fixture_bytes":manifest.len()+raw.len(),"logical_conv_accumulator_bytes":16,
            "input_sha256":input_hash(&input),"right_sha256":input_hash(&right),
            "history_sha256":input_hash(&history),"gates_sha256":input_hash(&gates),
            "scope":"First/warm prepared CPU operators, validation/stores included. Thirteen authenticated coefficient banks; synthetic inputs and immutable history. Grouped phases are not graph order. No projections, norms, activations, attention, KV, owner commits, codec or complete-model throughput.",
            "model_execution_qualified":false
        })
    );
    Ok(())
}
