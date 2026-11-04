//! Shared bounded qualification worker for two executed dense model geometries.
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path};
use toklet::{metal, sampling, session, weights};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Two concrete consumers of the same session/decision probe, not an engine API.
pub struct Spec {
    geometry: (usize, usize, usize, usize, usize, usize, usize),
    architecture: toklet::mlx::dense::Architecture,
    stops: [u32; 2],
}
const QWEN15: Spec = Spec {
    geometry: (1536, 8960, 28, 12, 2, 128, 151936),
    architecture: toklet::mlx::dense::Architecture::Qwen2,
    stops: [151645, 151643],
};
const SMOLLM3: Spec = Spec {
    geometry: (2048, 11008, 36, 16, 4, 128, 128256),
    architecture: toklet::mlx::dense::Architecture::SmolLm3,
    stops: [128012, 128001],
};

const BUDGET: usize = 4 * 1024 * 1024 * 1024;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Inputs {
    prompt: Vec<u32>,
    expected: Vec<u32>,
    stops: Vec<u32>,
    boundary: Vec<u32>,
}
fn make(model: &metal::Model, chunk: usize) -> Result<metal::Session<'_>> {
    Ok(metal::Session::with_attention_mode(
        model,
        session::Options {
            capacity: 512,
            prefill_chunk_size: chunk,
            position_offset: 0,
            memory_limit_bytes: BUDGET,
        },
        metal::ProjectionMode::MlxAffine,
        metal::NormalizationMode::Compensated,
        metal::ReadbackMode::FullLogits,
        metal::AttentionMode::Reference,
    )?)
}
fn row(s: &metal::Session<'_>) -> Result<Vec<f32>> {
    let values = s.logits().ok_or("missing distribution")?;
    if values.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite distribution".into());
    }
    Ok(values.to_vec())
}
fn same(a: &[f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() || a.iter().zip(b).any(|(x, y)| x.to_bits() != y.to_bits()) {
        return Err("chunk/reset distribution differs".into());
    }
    Ok(())
}
pub fn run(smollm3: bool) -> Result<()> {
    let spec = if smollm3 { SMOLLM3 } else { QWEN15 };
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: dense probe MODEL INPUT_JSON FRESH_OUTPUT".into());
    }
    let input_path = Path::new(&args[2]);
    if fs::metadata(input_path)?.len() > 1024 * 1024 {
        return Err("input exceeds bound".into());
    }
    let input: Inputs = serde_json::from_slice(&fs::read(input_path)?)?;
    if input.prompt.is_empty()
        || input.expected.is_empty()
        || input.expected.len() > 8
        || input.prompt.len() + input.expected.len() > 512
        || input.boundary.len() != 512
        || input.stops != spec.stops
        || input.expected[..input.expected.len() - 1]
            .iter()
            .any(|t| input.stops.contains(t))
    {
        return Err("frozen input/stop bounds differ".into());
    }
    let directory = Path::new(&args[1]);
    let output = Path::new(&args[3]);
    fs::create_dir(output)?;
    let load = weights::LoadOptions {
        context_limit: 512,
        memory_limit_bytes: BUDGET,
    };
    let options = session::Options {
        capacity: 512,
        prefill_chunk_size: 32,
        position_offset: 0,
        memory_limit_bytes: BUDGET,
    };
    let plan = metal::planning::inspect_mlx(
        directory,
        load,
        options,
        metal::NormalizationMode::Compensated,
        metal::ReadbackMode::FullLogits,
        metal::AttentionMode::Reference,
    )?;
    println!("{}", json!({"event":"plan","resources":plan}));
    if !plan.fits_requested_budgets {
        return Err("native admission rejected".into());
    }
    let model = metal::Model::load_mlx(directory, load)?;
    let c = model.config();
    if (
        c.hidden_size,
        c.intermediate_size,
        c.num_layers,
        c.query_heads,
        c.kv_heads,
        c.head_dim,
        c.vocab_size,
    ) != spec.geometry
        || model.mlx_architecture() != Some(spec.architecture)
    {
        return Err("unexpected probe geometry or architecture".into());
    }
    let mut s = make(&model, 32)?;
    println!(
        "{}",
        json!({"event":"loaded","device":model.device_name(),"memory":s.memory(),"weights":model.model_resident_bytes()?})
    );
    s.append(&input.prompt)?;
    s.evaluate()?;
    let initial = row(&s)?;
    let mut ids = Vec::new();
    for (index, expected) in input.expected.iter().enumerate() {
        let values = row(&s)?;
        let id = sampling::greedy(&values)?;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.join(format!("row-{index}.f32le")))?;
        let mut hash = Sha256::new();
        for value in &values {
            let bytes = value.to_le_bytes();
            file.write_all(&bytes)?;
            hash.update(bytes);
        }
        println!(
            "{}",
            json!({"event":"decision","index":index,"id":id,"expected":expected,"row_sha256":format!("{:x}",hash.finalize()),"committed":s.evaluated_len()})
        );
        if id != *expected {
            return Err("stock/native greedy disagreement; first row retained".into());
        }
        ids.push(id);
        if input.stops.contains(&id) {
            break;
        }
        s.append(&[id])?;
        if index + 1 < input.expected.len() {
            s.evaluate()?;
        }
    }
    let generation_history = s.history().to_vec();
    let generation_evaluated = s.evaluated_len();
    let generation_row = row(&s)?;
    // Drop each session before admitting its replacement. No duplicate weights.
    drop(s);
    let mut s = make(&model, 1)?;
    s.append(&input.prompt)?;
    s.evaluate()?;
    same(&initial, &row(&s)?)?;
    for (index, id) in ids.iter().enumerate() {
        if input.stops.contains(id) {
            break;
        }
        s.append(&[*id])?;
        if index + 1 < ids.len() {
            s.evaluate()?;
        }
    }
    if s.history() != generation_history || s.evaluated_len() != generation_evaluated {
        return Err("generation schedule state differs".into());
    }
    same(&generation_row, &row(&s)?)?;
    s.reset();
    s.append(&input.boundary)?;
    s.evaluate()?;
    let boundary = row(&s)?;
    let history = s.history().to_vec();
    let committed = s.evaluated_len();
    if !matches!(
        s.append(&[2]),
        Err(metal::Error::Options(
            "token ID/count outside session capacity"
        ))
    ) || s.history() != history
        || s.evaluated_len() != committed
    {
        return Err("overflow changed state".into());
    }
    same(&boundary, &row(&s)?)?;
    s.reset();
    s.append(&input.prompt)?;
    s.evaluate()?;
    same(&initial, &row(&s)?)?;
    drop(s);
    let mut s = make(&model, 32)?;
    s.append(&input.boundary)?;
    s.evaluate()?;
    same(&boundary, &row(&s)?)?;
    if !matches!(
        s.append(&[2]),
        Err(metal::Error::Options(
            "token ID/count outside session capacity"
        ))
    ) || s.history() != input.boundary
        || s.evaluated_len() != 512
    {
        return Err("chunk32 overflow changed state".into());
    }
    same(&boundary, &row(&s)?)?;
    s.reset();
    s.append(&input.prompt)?;
    s.evaluate()?;
    same(&initial, &row(&s)?)?;
    println!(
        "{}",
        json!({"event":"done","passed":true,"ids":ids,"committed_generation":generation_evaluated,
        "pending_generation":generation_history.len()-generation_evaluated,"capacity":512,"chunks":[1,32],
        "overflow_preserved":true,"reset_reproduced":true,"memory":s.memory(),"app_adopted":false,"full_fidelity_qualified":false})
    );
    Ok(())
}
