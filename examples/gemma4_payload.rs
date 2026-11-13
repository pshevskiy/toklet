//! Explicit data-only cold-load probe for the pinned Gemma4 E2B candidate.
//! Does not execute the graph, tokenize, grant app support or assert device fit.
use sha2::{Digest, Sha256};
use std::{error::Error, fs::File, io::Read, path::PathBuf};
use toklet::mlx::gemma4::{ModalityPolicy, payload};

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
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let root = PathBuf::from(
        args.next()
            .ok_or("usage: gemma4_payload CHECKPOINT_DIRECTORY")?,
    );
    if args.next().is_some() {
        return Err("usage: gemma4_payload CHECKPOINT_DIRECTORY".into());
    }
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
    // Resolve every cold public view, exercising final name/layout ownership
    // before constructing the named adapter (without executing its graph).
    for tensor in store.plan().logical_tensors() {
        match tensor.storage {
            toklet::mlx::gemma4::Storage::F32Vector => {
                return Err("Gemma BF16-only vector recipe required".into());
            }
            toklet::mlx::gemma4::Storage::Affine(layout) => {
                if store.matrix(&tensor.name)?.layout() != layout {
                    return Err("packed view differs".into());
                }
            }
            toklet::mlx::gemma4::Storage::Bf16Vector => {
                let toklet::mlx::gemma4::Shape::Vector(count) = tensor.shape else {
                    return Err("vector shape differs".into());
                };
                if store.vector(&tensor.name)?.len() != count {
                    return Err("vector view differs".into());
                }
            }
        }
    }
    let identity = store.content_identity();
    let model = toklet::model::gemma_packed::Model::from_e2b(&store)?;
    if model.content_identity() != Some(identity)
        || identity.config_sha256() != &<[u8; 32]>::from(Sha256::digest(&config))
        || identity.index_sha256() != &<[u8; 32]>::from(Sha256::digest(&index))
        || identity.container_sha256() != &digest
        || identity.container_bytes() != 3550670554
        || model.model_bytes() > toklet::weights::INITIAL_MEMORY_LIMIT_BYTES
    {
        return Err("loaded model content identity or budget differs".into());
    }
    let hex = |bytes: &[u8; 32]| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let r = store.report();
    println!(
        "{}",
        serde_json::json!({
            "format":"toklet-gemma4-cold-load-probe-v1",
            "content_identity":{"config_sha256":hex(identity.config_sha256()),
                "index_sha256":hex(identity.index_sha256()),
                "container_sha256":hex(identity.container_sha256()),
                "container_bytes":identity.container_bytes()},
            "model_content_identity_matches_store":true,
            "model_bytes_including_store":model.model_bytes(),
            "repository":"mlx-community/gemma-4-e2b-it-4bit",
            "revision":"238767527555cb75a05732a84dff5d6ba0dd6809",
            "container_bytes":r.container_bytes,
            "container_sha256":r.container_sha256.iter().map(|b|format!("{b:02x}")).collect::<String>(),
            "text_source_bytes":r.text_source_bytes,"excluded_source_bytes":r.excluded_source_bytes,
            "packed_matrices":r.packed_matrices,"fp32_vectors":r.fp32_vectors,
            "tensor_capacity_bytes":r.tensor_capacity_bytes,"metadata_reservation_bytes":r.metadata_reservation_bytes,
            "admitted_load_bytes":r.admitted_load_bytes,"public_cold_views_checked":store.plan().logical_tensors().len(),
            "real_payload_integrity_and_values_admitted":true,"inference_executed":false,
            "device_fit_qualified":false,"performance_qualified":false,"app_eligibility_changed":false
        })
    );
    Ok(())
}
