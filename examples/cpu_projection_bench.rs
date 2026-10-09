//! Projection microbenchmark only; no model, KV, tokenization, or sampling.
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    hint::black_box,
    io::{self, BufRead, Write},
    time::Instant,
};
use toklet::cpu;

fn data(count: usize, salt: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (((i % 1021) * 37 + salt) % 1021) as f32 / 512.0 - 1.0)
        .collect()
}

fn hash(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args_os().len() != 1 {
        return Err("cpu_projection_bench accepts no arguments".into());
    }
    let mut stdout = io::stdout().lock();
    let mut stdin = io::stdin().lock();
    for (tokens, inputs, outputs) in [
        (1, 896, 896),
        (128, 896, 896),
        (1, 896, 128),
        (128, 896, 128),
        (1, 896, 4864),
        (128, 896, 4864),
        (1, 4864, 896),
        (128, 4864, 896),
        (1, 896, 151936),
    ] {
        let input = data(tokens * inputs, 17);
        let weights = data(inputs * outputs, 91);
        let mut scalar = vec![0.0; tokens * outputs];
        let mut tiled = scalar.clone();
        let input_sha = hash(&input);
        let weights_sha = hash(&weights);
        let mut samples = Vec::with_capacity(10);
        for iteration in 0..13 {
            let candidate_first = iteration % 2 == 1;
            let mut elapsed = [0.0; 2];
            // Selection happens once per complete operation, never per cell.
            for candidate in if candidate_first {
                [true, false]
            } else {
                [false, true]
            } {
                let output = if candidate { &mut tiled } else { &mut scalar };
                let start = Instant::now();
                if candidate {
                    cpu::matmul_f64_tiled(
                        black_box(&input),
                        black_box(&weights),
                        None,
                        black_box(output),
                        tokens,
                        inputs,
                        outputs,
                    )?;
                } else {
                    cpu::matmul_f64(
                        black_box(&input),
                        black_box(&weights),
                        None,
                        black_box(output),
                        tokens,
                        inputs,
                        outputs,
                    )?;
                }
                elapsed[usize::from(candidate)] = start.elapsed().as_secs_f64();
            }
            for (index, (scalar, tiled)) in scalar.iter().zip(&tiled).enumerate() {
                if scalar.to_bits() != tiled.to_bits() {
                    return Err(format!("scalar/candidate differ at index {index}").into());
                }
            }
            if iteration >= 3 {
                samples.push(json!({"candidate_first":candidate_first,
                    "scalar_seconds":elapsed[0], "tiled_seconds":elapsed[1]}));
            }
        }
        writeln!(
            stdout,
            "{}",
            json!({
                "tokens":tokens, "in_features":inputs, "out_features":outputs,
                "warmup_pairs":3, "measured_pairs":10, "samples":samples,
                "input_sha256":input_sha, "weights_sha256":weights_sha,
                "output_sha256":hash(&tiled), "bitwise_equal_every_pair":true,
                "input_bytes":input.len()*4, "weights_bytes":weights.len()*4,
                "output_buffers_bytes":(scalar.len()+tiled.len())*4,
                "candidate_accumulator_stack_bytes":128,
            })
        )?;
        stdout.flush()?;
        // Permit the external driver to inspect this process after timed work.
        let mut acknowledgement = String::new();
        if stdin.read_line(&mut acknowledgement)? == 0 || acknowledgement != "next\n" {
            return Err("expected next acknowledgement".into());
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
