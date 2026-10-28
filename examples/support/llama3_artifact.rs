//! Fixed cold identity checks shared by the explicit Llama3 diagnostics.
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const REPOSITORY: &str = "mlx-community/Llama-3.2-1B-Instruct-8bit";
pub const REVISION: &str = "d48cdf0a4ea22d893b7c63a99d6a693e24822795";
pub const VOCAB: usize = 128256;
pub const FILES: &[(&str, u64, &str)] = &[
    (
        "config.json",
        1041,
        "847794a29e25927b4b98f35f5e66829e1b8bfd78f7743a9582780b4fd4a8ef9d",
    ),
    (
        "model.safetensors.index.json",
        26160,
        "c444df144fe9d86b42f8b1d797432c1f8379225316e41b1b9551e90d63136155",
    ),
    (
        "tokenizer.json",
        9085657,
        "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
    ),
    (
        "tokenizer_config.json",
        54528,
        "9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e",
    ),
    (
        "special_tokens_map.json",
        296,
        "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec",
    ),
    (
        "model.safetensors",
        1313157436,
        "cd3582a78766654e3e2107e3b4d51d9cb18e6754b659bfb44a34328ef9d7fa52",
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
