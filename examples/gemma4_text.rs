//! Explicit bounded scalar E2B text probe; no tokenizer/chat or app capability.
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::{File, OpenOptions},
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
};
use toklet::{
    mlx::gemma4::{ModalityPolicy, payload},
    model::gemma_packed::Model,
};

fn metadata(path: PathBuf, expected: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 1024 * 1024 || format!("{:x}", Sha256::digest(&bytes)) != expected {
        return Err("pinned Gemma4 metadata identity differs".into());
    }
    Ok(bytes)
}
fn capture(root: &Path, name: &str, values: &[f32]) -> Result<serde_json::Value, Box<dyn Error>> {
    let mut output = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join(name))?,
    );
    let mut sha = Sha256::new();
    for value in values {
        let bytes = value.to_le_bytes();
        output.write_all(&bytes)?;
        sha.update(bytes);
    }
    output.flush()?;
    Ok(
        serde_json::json!({"file":name,"dtype":"<f4","shape":[4,262144],
        "bytes":values.len()*4,"sha256":format!("{:x}",sha.finalize())}),
    )
}
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let root = PathBuf::from(
        args.next()
            .ok_or("usage: gemma4_text CHECKPOINT_DIRECTORY NEW_OUTPUT_DIRECTORY")?,
    );
    let output = PathBuf::from(
        args.next()
            .ok_or("usage: gemma4_text CHECKPOINT_DIRECTORY NEW_OUTPUT_DIRECTORY")?,
    );
    if args.next().is_some() {
        return Err("usage: gemma4_text CHECKPOINT_DIRECTORY NEW_OUTPUT_DIRECTORY".into());
    }
    std::fs::create_dir(&output)?;
    let config = metadata(
        root.join("config.json"),
        "6397cb6eca41b911d1dcab74e17941351057bd759284052a2331918ff6f9246c",
    )?;
    let index = metadata(
        root.join("model.safetensors.index.json"),
        "edb157dbf495e23f37377af4a628a9ad13c4ee7937f93ccb36ec9e9a19940f16",
    )?;
    // Independently frozen LFS digest, never supplied by the consumed container.
    let digest = [
        0x03, 0x8e, 0x39, 0xa3, 0x7a, 0x76, 0x67, 0x37, 0x3d, 0x2c, 0x39, 0x91, 0x37, 0x54, 0x46,
        0xb1, 0x0c, 0x96, 0xae, 0x1d, 0x71, 0x7a, 0x68, 0x67, 0x48, 0x70, 0x34, 0x3d, 0xb3, 0x76,
        0xb7, 0x6e,
    ];
    let store = payload::load(
        &config,
        &index,
        File::open(root.join("model.safetensors"))?,
        payload::LoadOptions {
            container_bytes: 3550670554,
            container_sha256: digest,
            memory_limit_bytes: toklet::weights::INITIAL_MEMORY_LIMIT_BYTES,
            modality: ModalityPolicy::TextOnly,
        },
    )?;
    let model = Model::from_e2b(&store)?;
    let budget = toklet::weights::INITIAL_MEMORY_LIMIT_BYTES;
    let mut session = model.new_session(budget)?;
    let session_bytes = session.allocated_bytes()?;
    let ids = [2, 100, 257, 100];
    let mut arrays = Vec::new();
    for (name, schedule) in [
        ("batched.logits.bin", &[4][..]),
        ("decode.logits.bin", &[1, 1, 1, 1][..]),
        ("chunked.logits.bin", &[1, 3][..]),
    ] {
        session.reset()?;
        let mut values = Vec::new();
        values.try_reserve_exact(4 * 262144)?;
        let mut offset = 0;
        for &count in schedule {
            values.extend_from_slice(session.forward(&ids[offset..offset + count])?);
            offset += count;
        }
        if session.tokens() != ids
            || values.len() != 4 * 262144
            || session.allocated_bytes()? != session_bytes
        {
            return Err("E2B committed prefix/storage differs".into());
        }
        arrays.push(capture(&output, name, &values)?);
        let before = session.logits().to_vec();
        if session.forward(&[1; 29]).is_ok()
            || session.tokens() != ids
            || session.logits() != before
        {
            return Err("E2B token33 rejection changed committed state".into());
        }
    }
    let report = serde_json::json!({
        "format":"toklet-gemma4-short-scalar-probe-v1","tokens":ids,
        "repository":"mlx-community/gemma-4-e2b-it-4bit",
        "revision":"238767527555cb75a05732a84dff5d6ba0dd6809",
        "model_bytes_including_store":model.model_bytes(),"session_bytes":session_bytes,
        "planned_session_bytes":model.planned_session_bytes()?,"memory_limit_bytes":budget,
        "store_tensor_capacity_bytes":store.report().tensor_capacity_bytes,
        "store_metadata_reservation_bytes":store.report().metadata_reservation_bytes,
        "arrays":arrays,"native_scalar_executed":true,"max_executable_positions":32,
        "local_cache_slots":512,"global_cache_slots":32,"token33_rejection_preserves_public_prefix":true,
        "numerical_comparison_complete":false,"original_bf16_qualified":false,
        "tokenizer_chat_generation_qualified":false,"metal_qualified":false,
        "performance_qualified":false,"quality_qualified":false,"device_fit_qualified":false,
        "app_eligibility_changed":false
    });
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join("observation.json"))?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    println!(
        "{}",
        serde_json::json!({"native_scalar_executed":true,"model_bytes":model.model_bytes(),"session_bytes":session_bytes})
    );
    Ok(())
}
