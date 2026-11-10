//! Fixed source-profile microbenchmark, independent of model downloads/GPU/UI.
use half::bf16;
use serde_json::json;
use std::{hint::black_box, io, time::Instant};
use toklet::model::lfm3_rotary::{HEAD_DIM, Profile, Rotary};

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "usage: lfm3_rotary_probe --tokens 1|32|128 --offset 0|511|1023",
    )
}

fn options() -> Result<(usize, usize), io::Error> {
    let mut args = std::env::args().skip(1);
    let (mut tokens, mut offset) = (None, None);
    while let Some(arg) = args.next() {
        let slot = match arg.as_str() {
            "--tokens" => &mut tokens,
            "--offset" => &mut offset,
            _ => return Err(invalid()),
        };
        if slot.is_some() {
            return Err(invalid());
        }
        *slot = Some(
            args.next()
                .ok_or_else(invalid)?
                .parse()
                .map_err(|_| invalid())?,
        );
    }
    match (tokens, offset) {
        (Some(t @ (1 | 32 | 128)), Some(p @ (0 | 511 | 1023))) => Ok((t, p)),
        _ => Err(invalid()),
    }
}

fn tables() -> Result<(Rotary, Rotary), toklet::model::Error> {
    let bytes = Rotary::planned_bytes(2048)?;
    Ok((
        Rotary::new(Profile::FullAttention, 2048, bytes)?,
        Rotary::new(Profile::SlidingAttention, 2048, bytes)?,
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (tokens, offset) = options()?;
    let started = Instant::now();
    let (full, sliding) = tables()?;
    let initial_table_construction_ms = started.elapsed().as_secs_f64() * 1000.;
    let mut construction = Vec::with_capacity(10);
    for trial in 0..13 {
        let started = Instant::now();
        let pair = tables()?;
        let elapsed = started.elapsed().as_secs_f64() * 1000.;
        black_box(&pair);
        if trial >= 3 {
            construction.push(elapsed);
        }
    }
    let layers = [
        &full, &sliding, &full, &sliding, &full, &sliding, &full, &sliding, &full, &sliding, &full,
    ];
    let q = vec![bf16::from_f32(0.25); tokens * 16 * HEAD_DIM];
    let k = vec![bf16::from_f32(-0.375); tokens * 4 * HEAD_DIM];
    let mut q_out = vec![bf16::ZERO; q.len()];
    let mut k_out = vec![bf16::ZERO; k.len()];
    let repetitions = (128 / tokens).max(1);
    let mut warm = Vec::with_capacity(10);
    for trial in 0..13 {
        let started = Instant::now();
        for _ in 0..repetitions {
            for table in layers {
                table.apply(black_box(&q), &mut q_out, tokens, 16, offset)?;
                table.apply(black_box(&k), &mut k_out, tokens, 4, offset)?;
            }
        }
        let elapsed = started.elapsed().as_secs_f64() * 1000. / repetitions as f64;
        black_box((&q_out, &k_out));
        if trial >= 3 {
            warm.push(elapsed);
        }
    }
    println!(
        "{}",
        json!({
            "format":"toklet-lfm3-rotary-source-micro-v1",
            "tokens":tokens,"offset":offset,"head_dim":HEAD_DIM,
            "query_heads":16,"kv_heads":4,"full_layers":6,"sliding_layers":5,
            "table_capacity":2048,"warmups":3,"measured_trials":10,
            "repetitions_per_trial":repetitions,
            "initial_table_construction_ms":initial_table_construction_ms,
            "repeated_two_table_construction_ms":construction,
            "warm_11_layer_qk_pass_ms":warm,
            "coefficient_owner_bytes":full.allocated_bytes()?+sliding.allocated_bytes()?,
            "caller_input_output_bytes":(q.capacity()+k.capacity()+q_out.capacity()+k_out.capacity())*2,
            "last_q_bf16":q_out.last().unwrap().to_bits(),
            "last_k_bf16":k_out.last().unwrap().to_bits(),
            "scope":"CPU source-profile operator only; construction excludes process startup and deallocation; warm time includes finite validation and stores. No weights, KV, codec, sampling or model throughput.",
            "full_model_execution_qualified":false
        })
    );
    Ok(())
}
