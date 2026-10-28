//! Fixed cold identity checks shared by the explicit Phi3 diagnostics.
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const REPOSITORY: &str = "mlx-community/Phi-3-mini-4k-instruct-4bit";
pub const REVISION: &str = "5b3819ed6317784fb20eddeae9bed984f778d0d0";
pub const VOCAB: usize = 32064;
pub const FILES: &[(&str, u64, &str)] = &[
    (
        "config.json",
        1030,
        "0e2e43bc4358b4cabbcc33c496f34e170fdfe04612a47428f1691d1e9ec5a568",
    ),
    (
        "model.safetensors.index.json",
        32695,
        "4ef199a5c6c29fa54d84d3710bfad4b5424b75bbd68872f9e1e12234bd3fd9c3",
    ),
    (
        "tokenizer.json",
        1844436,
        "d0f067e1e15cd0a36ebef3668024882cb67a80b86fb4b7b4b128481f0d474db7",
    ),
    (
        "tokenizer_config.json",
        3333,
        "d6e13c85fbde9cf71f663da027cf558ab2bb9df80bd60c718be10dbba8d2a2be",
    ),
    (
        "special_tokens_map.json",
        569,
        "474d699677e264edd3851559548cbbb61801ad8da6b249c81455007602313c34",
    ),
    (
        "model.safetensors",
        2149696167,
        "8d75680621a09474f6601e9176f2f61f92a5e4c079d68d583901f51699fda50a",
    ),
];
pub fn hash_file(path: &Path, size: u64, expected: &str) -> Result<()> {
    let metadata = path.symlink_metadata()?;
    if !metadata.is_file() || metadata.len() != size {
        return Err("pinned regular file size differs".into());
    }
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut bytes = [0; 64 * 1024];
    let mut count = 0_u64;
    loop {
        let n = f.read(&mut bytes)?;
        if n == 0 {
            break;
        }
        count += n as u64;
        if count > size {
            return Err("pinned file grew".into());
        }
        hasher.update(&bytes[..n]);
    }
    if count != size || format!("{:x}", hasher.finalize()) != expected {
        return Err("pinned file digest differs".into());
    }
    Ok(())
}
