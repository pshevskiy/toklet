//! Fixed-shape canonical LFM3 CPU component baseline; no model/codec execution.
use half::bf16;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{hint::black_box, io, time::Instant};
use toklet::cpu::{lfm3_rmsnorm_bf16, lfm3_sigmoid_bf16, lfm3_silu_bf16};

#[derive(Deserialize)]
struct Fixture {
    norms: Norms,
}
#[derive(Deserialize)]
struct Norms {
    vectors: Vec<Norm>,
}
#[derive(Deserialize)]
struct Norm {
    width: usize,
    heads: usize,
    byte_offset: usize,
    byte_count: usize,
}

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "usage: lfm3_bf16_probe --tokens 1|32|128",
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tokens = options()?;
    let manifest = include_bytes!("../../tests/fixtures/lfm3-bf16-v1/parent.json");
    let weights = include_bytes!("../../tests/fixtures/lfm3-bf16-v1/norms.bf16le");
    if format!("{:x}", Sha256::digest(manifest))
        != "9f1494cd22e636e4e1787bcd73d7e1d11a5d89be72336792e8331357eb2fca91"
        || format!("{:x}", Sha256::digest(weights))
            != "cba5b1876070346343f92af911f3b905fde757515bf4532d9673f07206d7c134"
    {
        return Err(
            io::Error::new(io::ErrorKind::InvalidData, "Pinned norm fixture differs").into(),
        );
    }
    // Deserialize only the cold vector descriptors; large input/expected arrays
    // are skipped. Resolve and validate every gain range before timing.
    let Fixture { norms } = serde_json::from_slice(manifest)?;
    if norms.vectors.len() != 119 {
        return Err(invalid().into());
    }
    for norm in &norms.vectors {
        if !matches!((norm.width, norm.heads), (2048, 1) | (128, 4 | 16))
            || norm.byte_count != 2 * norm.width
            || !norm.byte_offset.is_multiple_of(2)
            || norm
                .byte_offset
                .checked_add(norm.byte_count)
                .is_none_or(|end| end > weights.len())
        {
            return Err(invalid().into());
        }
    }
    let gains: Vec<_> = weights
        .chunks_exact(2)
        .map(|bytes| bf16::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let norm_input: Vec<_> = (0..tokens * 2048)
        .map(|i| bf16::from_f32(((i * 13 % 257) as f32 - 128.) / 32.))
        .collect();
    let silu_input: Vec<_> = (0..tokens * 7680)
        .map(|i| bf16::from_f32(((i * 17 % 257) as f32 - 128.) / 32.))
        .collect();
    let sigmoid_input: Vec<_> = (0..tokens * 16)
        .map(|i| bf16::from_f32(((i * 19 % 257) as f32 - 128.) / 32.))
        .collect();
    let mut norm_output = vec![bf16::ZERO; norm_input.len()];
    let mut silu_output = vec![bf16::ZERO; silu_input.len()];
    let mut sigmoid_output = vec![bf16::ZERO; sigmoid_input.len()];
    let repetitions = (32 / tokens).max(1);
    let mut norm_samples = Vec::with_capacity(10);
    let mut silu_samples = Vec::with_capacity(10);
    let mut sigmoid_samples = Vec::with_capacity(10);
    let mut total_samples = Vec::with_capacity(10);
    let mut first = [0.; 4];
    for trial in 0..14 {
        let repeats = if trial == 0 { 1 } else { repetitions };
        let begin = Instant::now();
        for _ in 0..repeats {
            for norm in &norms.vectors {
                let count = tokens * norm.heads * norm.width;
                let start = norm.byte_offset / 2;
                lfm3_rmsnorm_bf16(
                    black_box(&norm_input[..count]),
                    &gains[start..start + norm.width],
                    &mut norm_output[..count],
                    tokens * norm.heads,
                    norm.width,
                )?;
            }
        }
        let norm_ms = begin.elapsed().as_secs_f64() * 1000. / repeats as f64;
        black_box(&norm_output);
        let phase = Instant::now();
        for _ in 0..repeats {
            for _ in 0..24 {
                lfm3_silu_bf16(black_box(&silu_input), &mut silu_output)?;
            }
        }
        let silu_ms = phase.elapsed().as_secs_f64() * 1000. / repeats as f64;
        black_box(&silu_output);
        let phase = Instant::now();
        for _ in 0..repeats {
            for _ in 0..11 {
                lfm3_sigmoid_bf16(black_box(&sigmoid_input), &mut sigmoid_output)?;
            }
        }
        let sigmoid_ms = phase.elapsed().as_secs_f64() * 1000. / repeats as f64;
        black_box(&sigmoid_output);
        let total_ms = begin.elapsed().as_secs_f64() * 1000. / repeats as f64;
        if trial == 0 {
            first = [norm_ms, silu_ms, sigmoid_ms, total_ms];
        } else if trial >= 4 {
            norm_samples.push(norm_ms);
            silu_samples.push(silu_ms);
            sigmoid_samples.push(sigmoid_ms);
            total_samples.push(total_ms);
        }
    }
    let input_hash = |values: &[bf16]| {
        let mut hash = Sha256::new();
        for v in values {
            hash.update(v.to_bits().to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    };
    println!(
        "{}",
        json!({
            "format":"toklet-lfm3-bf16-component-micro-v1",
            "profile":"lfm3-parent-bf16-cpu-v1","tokens":tokens,
            "norm_calls":119,"norm_values":norms.vectors.iter().map(|v|tokens*v.heads*v.width).sum::<usize>(),
            "silu_calls":24,"silu_values":tokens*7680*24,"sigmoid_calls":11,"sigmoid_values":tokens*16*11,
            "warmups":3,"measured_trials":10,"repetitions_per_trial":repetitions,
            "first_norm_silu_sigmoid_total_ms":first,
            "warm_norm_ms":norm_samples,"warm_silu_ms":silu_samples,
            "warm_sigmoid_ms":sigmoid_samples,"warm_component_total_ms":total_samples,
            "owned_gain_bytes":gains.capacity()*2,
            "owned_descriptor_bytes":norms.vectors.capacity()*std::mem::size_of::<Norm>(),
            "caller_input_output_bytes":(norm_input.capacity()+norm_output.capacity()+silu_input.capacity()+
                silu_output.capacity()+sigmoid_input.capacity()+sigmoid_output.capacity())*2,
            "embedded_fixture_bytes":manifest.len()+weights.len(),
            "logical_norm_accumulator_bytes":128,
            "norm_input_sha256":input_hash(&norm_input),"silu_input_sha256":input_hash(&silu_input),
            "sigmoid_input_sha256":input_hash(&sigmoid_input),
            "scope":"First/warm CPU operator calls over fixed buffers, validation/stores included. Source norm gains only; no projections, products, residuals, convolution, attention, KV, codec or complete-model throughput.",
            "model_execution_qualified":false
        })
    );
    Ok(())
}
