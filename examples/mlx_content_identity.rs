//! Explicit source-only Smol consumption diagnostic; no model/codec/GPU execution.
use std::{error::Error, path::Path};
use toklet::{
    mlx::qwen::{Source, Storage},
    weights::LoadOptions,
};
const LIMIT: usize = 512 * 1024 * 1024;
const DIAGNOSTIC_RESERVE: usize = 64 * 1024 * 1024;
fn zeroed(count: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut values = Vec::new();
    values.try_reserve_exact(count)?;
    values.resize(count, 0);
    Ok(values)
}
fn read(source: &mut Source, name: &str, storage: Storage) -> Result<(), Box<dyn Error>> {
    match storage {
        Storage::Affine(layout) => {
            if layout.total_bytes() > DIAGNOSTIC_RESERVE {
                return Err("tensor exceeds diagnostic reserve".into());
            }
            let mut weights = zeroed(layout.weight_bytes())?;
            let mut scales = zeroed(layout.metadata_bytes())?;
            let mut biases = zeroed(layout.metadata_bytes())?;
            source.read_affine(name, &mut weights, &mut scales, &mut biases)?;
        }
        Storage::Float(_) => source.visit_float(name, |_| Ok(()))?,
    }
    Ok(())
}
fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|v| format!("{v:02x}")).collect()
}
fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = std::env::args_os().collect();
    if arguments.len() != 2 {
        return Err("usage: mlx_content_identity LOCAL_SMOL_DIRECTORY".into());
    }
    let mut source = Source::open_smol_audited(
        Path::new(&arguments[1]),
        LoadOptions {
            context_limit: 8,
            memory_limit_bytes: LIMIT,
        },
    )?;
    let admitted = source.admitted_load_bytes();
    if admitted
        .checked_add(DIAGNOSTIC_RESERVE)
        .is_none_or(|n| n > LIMIT)
    {
        return Err("source and diagnostic exceed budget".into());
    }
    let owned = source.content_audit_owned_bytes().ok_or("missing audit")?;
    let entries: Vec<_> = source
        .tensors()
        .iter()
        .map(|(name, spec)| (name.clone(), spec.storage()))
        .collect();
    let embedding = source
        .tensors()
        .get("model.embed_tokens.weight")
        .ok_or("missing tied embedding")?
        .storage();
    assert!(source.content_identity().is_none());
    read(&mut source, "model.embed_tokens.weight", embedding)?;
    assert!(source.content_identity().is_none());
    let mut packed = 0;
    let mut floating = 0;
    for (name, storage) in &entries {
        read(&mut source, name, *storage)?;
        match storage {
            Storage::Affine(_) => packed += 1,
            Storage::Float(_) => floating += 1,
        };
        assert_eq!(source.content_audit_owned_bytes(), Some(owned));
    }
    // Two additional embedding reads represent input/tied-output roles. Every
    // raw weight/scale/bias visit must reverify, even after all ranges were seen.
    read(&mut source, "model.embed_tokens.weight", embedding)?;
    let identity = source.content_identity().ok_or("incomplete consumption")?;
    assert_eq!(identity.physical_ranges(), 694);
    assert_eq!(identity.verified_reads(), 700);
    assert_eq!(packed, 211);
    assert_eq!(floating, 61);
    drop(source); // The copied historical evidence owns no file/path reference.
    println!(
        "{}",
        serde_json::json!({"format":"toklet-smol-native-source-content-v1","repository":identity.repository(),"revision":identity.revision(),"config_sha256":hex(identity.config_sha256()),"index_sha256":hex(identity.index_sha256()),"header_sha256":hex(identity.header_sha256()),"authenticated_reference_container_sha256":hex(identity.reference_container_sha256()),"container_bytes":identity.container_bytes(),"payload_bytes":identity.payload_bytes(),"physical_ranges":identity.physical_ranges(),"verified_reads":identity.verified_reads(),"logical_tensors":entries.len(),"packed_matrices":packed,"floating_vectors":floating,"additional_embedding_reads":2,"source_admitted_bytes":admitted,"audit_owned_bytes":owned,"diagnostic_reserve_bytes":DIAGNOSTIC_RESERVE,"process_policy_bytes":LIMIT,"identity_survives_source_drop":true,"new_sequential_container_sha_computed":false,"model_assembled":false,"inference_executed":false,"codec_pairing_qualified":false,"app_eligibility_changed":false,"phone_fit_qualified":false,"performance_qualified":false})
    );
    Ok(())
}
