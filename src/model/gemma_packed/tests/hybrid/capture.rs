//! Bounded raw observations are emitted before numerical assertions.
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::PathBuf,
};
pub(super) struct Capture {
    file: File,
    bytes: usize,
    records: usize,
    digest: Sha256,
    limit: usize,
}
impl Capture {
    pub(super) fn new() -> Self {
        Self::bounded(64 * 1024 * 1024)
    }
    pub(super) fn bounded(limit: usize) -> Self {
        assert!(limit > 0 && limit <= 128 * 1024 * 1024);
        let path = PathBuf::from(
            std::env::var_os("TOKLET_GEMMA_HYBRID_OUTPUT")
                .expect("explicit fresh diagnostic output path"),
        );
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        Self {
            file,
            bytes: 0,
            records: 0,
            digest: Sha256::new(),
            limit,
        }
    }
    pub(super) fn record(&mut self, name: &str, values: &[f32], before: usize, count: usize) {
        let bytes = values.len().checked_mul(4).unwrap();
        assert!(self.bytes.checked_add(bytes).unwrap() <= self.limit && self.records < 16384);
        println!(
            "GEMMA_HYBRID_ARRAY_ATTEMPT {}",
            json!({"record":self.records,"name":name,"offset":self.bytes,"bytes":bytes,"before":before,"count":count,"values":values.len()})
        );
        std::io::stdout().flush().unwrap();
        let mut digest = Sha256::new();
        let mut raw = [0u8; 4096];
        for part in values.chunks(1024) {
            for (&value, to) in part.iter().zip(raw.chunks_exact_mut(4)) {
                to.copy_from_slice(&value.to_le_bytes());
            }
            let bytes = &raw[..part.len() * 4];
            self.file.write_all(bytes).unwrap();
            digest.update(bytes);
            self.digest.update(bytes);
        }
        println!(
            "GEMMA_HYBRID_ARRAY {}",
            json!({"record":self.records,"name":name,"offset":self.bytes,"bytes":bytes,"before":before,"count":count,"values":values.len(),"sha256":format!("{:x}",digest.finalize())})
        );
        std::io::stdout().flush().unwrap();
        self.bytes += bytes;
        self.records += 1;
    }
    pub(super) fn finish(self) {
        self.file.sync_all().unwrap();
        assert_eq!(self.file.metadata().unwrap().len(), self.bytes as u64);
        println!(
            "GEMMA_HYBRID_CAPTURE {}",
            json!({"records":self.records,"bytes":self.bytes,"sha256":format!("{:x}",self.digest.finalize()),"whole_e2b_qualified":false})
        );
        std::io::stdout().flush().unwrap();
    }
}
