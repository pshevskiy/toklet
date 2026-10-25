//! Bounded raw E2B generation diagnostic; no tokenizer/chat or app capability.
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    error::Error,
    fs::{File, OpenOptions},
    io::{self, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};
use toklet::{
    generation::{self, Input, Request, StopReason},
    mlx::gemma4::{ModalityPolicy, payload},
    model::gemma_packed::{GenerationSession, Model},
    session::{self, InferenceSession, Memory, Options},
};

const PROMPT: [u32; 4] = [2, 100, 257, 100];
const VOCAB: usize = 262144;
const ROWS: usize = 16;

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

// This example alone streams diagnostics at the coarse sampling boundary.
// Capture precedes append, while the distribution is still valid. No numerical
// session API or kernel is modified, and no full vocabulary copy is retained.
struct Capture {
    output: BufWriter<File>,
    arrays: Vec<serde_json::Value>,
    bytes: usize,
    scenario: usize,
    scenario_rows: usize,
    error: Option<String>,
}
impl Capture {
    fn new(root: &Path) -> io::Result<Self> {
        let mut arrays = Vec::new();
        arrays.try_reserve_exact(ROWS).map_err(io::Error::other)?;
        Ok(Self {
            output: BufWriter::with_capacity(
                64 * 1024,
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(root.join("arrays.bin"))?,
            ),
            arrays,
            bytes: 0,
            scenario: 0,
            scenario_rows: 0,
            error: None,
        })
    }
    fn row(&mut self, prefix: &[u32], values: &[f32]) -> io::Result<()> {
        if self.arrays.len() >= ROWS
            || self.scenario_rows >= 4
            || values.len() != VOCAB
            || !(4..=7).contains(&prefix.len())
        {
            return Err(io::Error::other("fixed diagnostic capture bound differs"));
        }
        self.arrays.push(serde_json::json!({
            "scenario":self.scenario,"step":self.scenario_rows,"prefix":prefix,
            "offset":self.bytes,"dtype":"<f4","shape":[VOCAB],
            "bytes":VOCAB*4,"completed":false
        }));
        let mut sha = Sha256::new();
        let mut finite = true;
        for value in values {
            let bytes = value.to_le_bytes();
            self.output.write_all(&bytes)?;
            sha.update(bytes);
            finite &= value.is_finite();
        }
        self.output.flush()?;
        self.bytes += VOCAB * 4;
        let item = self
            .arrays
            .last_mut()
            .expect("descriptor was just appended");
        item["sha256"] = serde_json::json!(format!("{:x}", sha.finalize()));
        item["completed"] = true.into();
        self.scenario_rows += 1;
        if !finite {
            return Err(io::Error::other("nonfinite captured distribution"));
        }
        Ok(())
    }
}

struct Observed<'a, 'model, 'weights> {
    inner: &'a mut GenerationSession<'model, 'weights>,
    capture: &'a RefCell<Capture>,
}
impl InferenceSession for Observed<'_, '_, '_> {
    fn history(&self) -> &[u32] {
        self.inner.history()
    }
    fn evaluated_len(&self) -> usize {
        self.inner.evaluated_len()
    }
    fn pending_len(&self) -> usize {
        self.inner.pending_len()
    }
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn memory(&self) -> Memory {
        self.inner.memory()
    }
    fn logits(&self) -> Option<&[f32]> {
        self.inner.logits()
    }
    fn greedy_token(&self) -> Option<Result<u32, toklet::cpu::Error>> {
        let values = self.inner.logits()?;
        let mut capture = self.capture.borrow_mut();
        if capture.error.is_some() {
            return None;
        }
        if let Err(error) = capture.row(self.inner.history(), values) {
            capture.error = Some(error.to_string());
            return None;
        }
        self.inner.greedy_token()
    }
    fn append(&mut self, tokens: &[u32]) -> Result<(), session::Error> {
        self.inner.append(tokens)
    }
    fn set_tokens(&mut self, tokens: &[u32]) -> Result<(), session::Error> {
        self.inner.set_tokens(tokens)
    }
    fn evaluate_next_chunk(&mut self) -> Result<usize, session::Error> {
        self.inner.evaluate_next_chunk()
    }
    fn discard_pending(&mut self) {
        self.inner.discard_pending();
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
}

fn scenarios(
    model: &Model<'_>,
    capture: &RefCell<Capture>,
    record: &mut serde_json::Value,
) -> Result<(), Box<dyn Error>> {
    let mut session = model.new_generation_session(Options {
        capacity: 32,
        prefill_chunk_size: 4,
        position_offset: 0,
        memory_limit_bytes: toklet::weights::INITIAL_MEMORY_LIMIT_BYTES,
    })?;
    let memory = session.memory();
    record["memory"] = serde_json::to_value(memory)?;
    for (index, name) in ["uninterrupted", "split", "cancel_resume", "caller_stop"]
        .iter()
        .enumerate()
    {
        {
            let mut saved = capture.borrow_mut();
            saved.scenario = index;
            saved.scenario_rows = 0;
        }
        let mut observed = Observed {
            inner: &mut session,
            capture,
        };
        let cancelled = AtomicBool::new(false);
        let mut emitted = Vec::with_capacity(4);
        let mut reports = Vec::with_capacity(2);
        let result = (|| -> Result<(), Box<dyn Error>> {
            let first = generation::generate(
                &mut observed,
                Request {
                    input: Input::Reset(&PROMPT),
                    max_new_tokens: if index == 1 { 2 } else { 4 },
                    // Explicit caller stop, not a qualified model EOS policy.
                    stop_tokens: if index == 3 { &[107] } else { &[] },
                },
                &cancelled,
                |event| {
                    emitted.push(event.token_id);
                    if index == 2 && emitted.len() == 2 {
                        cancelled.store(true, Ordering::Relaxed);
                    }
                    Ok(())
                },
            )?;
            let reason = first.reason;
            reports.push(first);
            if reason
                != match index {
                    2 => StopReason::Cancelled,
                    3 => StopReason::StopToken { token_id: 107 },
                    _ => StopReason::OutputLimit,
                }
            {
                return Err("first generation stop reason differs".into());
            }
            if index != 0 {
                cancelled.store(false, Ordering::Relaxed);
                reports.push(generation::generate(
                    &mut observed,
                    Request {
                        input: Input::Continue,
                        max_new_tokens: if index == 3 { 3 } else { 2 },
                        stop_tokens: &[],
                    },
                    &cancelled,
                    |event| {
                        emitted.push(event.token_id);
                        Ok(())
                    },
                )?);
            }
            if capture.borrow().scenario_rows != 4
                || observed.history().len() != 8
                || observed.evaluated_len() != 7
                || observed.pending_len() != 1
                || observed.logits().is_some()
                || observed.memory() != memory
            {
                return Err("final generation state or storage differs".into());
            }
            Ok(())
        })();
        record["scenarios"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "name":name,"reports":reports,"emitted_ids":emitted,
                "history":observed.history(),"evaluated":observed.evaluated_len(),
                "pending":observed.pending_len(),"logits_available":observed.logits().is_some(),
                "completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string)
            }));
        result?;
    }
    Ok(())
}

fn execute(
    root: &Path,
    capture: &RefCell<Capture>,
    record: &mut serde_json::Value,
) -> Result<(), Box<dyn Error>> {
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
    record["model_bytes_including_store"] = model.model_bytes().into();
    scenarios(&model, capture, record)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let usage = "usage: gemma4_generate CHECKPOINT_DIRECTORY NEW_OUTPUT_DIRECTORY";
    let root = PathBuf::from(args.next().ok_or(usage)?);
    let output = PathBuf::from(args.next().ok_or(usage)?);
    if args.next().is_some() {
        return Err(usage.into());
    }
    std::fs::create_dir(&output)?;
    let capture = RefCell::new(Capture::new(&output)?);
    let mut record = serde_json::json!({
        "format":"toklet-gemma4-raw-generation-probe-v1","passed":false,
        "repository":"mlx-community/gemma-4-e2b-it-4bit",
        "revision":"238767527555cb75a05732a84dff5d6ba0dd6809",
        "prompt":PROMPT,"scenarios":[],"primary_error":null,"endpoint_errors":[],
        "max_executable_positions":32,"local_cache_slots":512,"global_cache_slots":32,
        "memory_limit_bytes":toklet::weights::INITIAL_MEMORY_LIMIT_BYTES,
        "diagnostic_byte_limit":ROWS*VOCAB*4,"numerical_comparison_complete":false,
        "original_bf16_qualified":false,"tokenizer_chat_or_model_stop_policy_qualified":false,
        "metal_qualified":false,"performance_qualified":false,"quality_qualified":false,
        "device_fit_qualified":false,"app_eligibility_changed":false
    });
    if let Err(error) = execute(&root, &capture, &mut record) {
        record["primary_error"] = error.to_string().into();
    }
    let mut capture = capture.into_inner();
    record["capture_error"] = serde_json::to_value(&capture.error)?;
    if let Err(error) = capture.output.flush() {
        record["endpoint_errors"]
            .as_array_mut()
            .unwrap()
            .push(error.to_string().into());
    }
    record["arrays"] = serde_json::to_value(&capture.arrays)?;
    record["completed_array_bytes"] = capture.bytes.into();
    record["passed"] = (record["primary_error"].is_null()
        && capture.error.is_none()
        && record["endpoint_errors"].as_array().unwrap().is_empty()
        && capture.arrays.len() == ROWS)
        .into();
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join("observation.json"))?;
    serde_json::to_writer_pretty(&mut file, &record)?;
    file.write_all(b"\n")?;
    file.flush()?;
    println!(
        "{}",
        serde_json::json!({"passed":record["passed"],"arrays":capture.arrays.len(),"primary_error":record["primary_error"],"capture_error":record["capture_error"]})
    );
    if record["passed"] != true {
        return Err("raw E2B generation probe failed; observations retained".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct Temporary(PathBuf);
    impl Temporary {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "toklet-gemma-generate-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temporary {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn finite_rows_preserve_exact_bytes_offsets_and_prewrite_bounds() {
        let temp = Temporary::new();
        let mut capture = Capture::new(&temp.0).unwrap();
        let mut values = vec![0.0; VOCAB];
        values[..4].copy_from_slice(&[-0.0, 1.25, f32::MIN_POSITIVE, f32::from_bits(1)]);
        capture.row(&PROMPT, &values).unwrap();
        capture.row(&[2, 100, 257, 100, 107], &values).unwrap();
        let raw = std::fs::read(temp.0.join("arrays.bin")).unwrap();
        assert_eq!(raw.len(), 2 * VOCAB * 4);
        for (index, item) in capture.arrays.iter().enumerate() {
            let row = &raw[index * VOCAB * 4..(index + 1) * VOCAB * 4];
            assert_eq!(item["offset"], index * VOCAB * 4);
            assert_eq!(item["completed"], true);
            assert_eq!(item["sha256"], format!("{:x}", Sha256::digest(row)));
            for (bytes, value) in row.chunks_exact(4).zip(&values) {
                assert_eq!(bytes, value.to_le_bytes());
            }
        }
        assert!(capture.row(&[2; 3], &values).is_err());
        assert!(capture.row(&[2; 8], &values).is_err());
        capture.scenario_rows = 4;
        assert!(capture.row(&PROMPT, &values).is_err());
        assert_eq!(capture.arrays.len(), 2);
        capture.scenario_rows = 0;
        capture.arrays.resize(ROWS, serde_json::Value::Null);
        assert!(capture.row(&PROMPT, &values).is_err());
        assert_eq!(capture.bytes, raw.len());
        assert_eq!(std::fs::read(temp.0.join("arrays.bin")).unwrap(), raw);
    }

    #[test]
    fn invalid_shape_rejects_before_capture_nonfinite_retains_complete_raw() {
        let temp = Temporary::new();
        let mut capture = Capture::new(&temp.0).unwrap();
        assert!(capture.row(&PROMPT, &[0.0]).is_err());
        assert!(capture.arrays.is_empty());
        let mut values = vec![0.0; VOCAB];
        values[42] = f32::from_bits(0x7fc00042);
        assert!(capture.row(&PROMPT, &values).is_err());
        assert_eq!(capture.arrays.len(), 1);
        assert_eq!(capture.arrays[0]["completed"], true);
        let raw = std::fs::read(temp.0.join("arrays.bin")).unwrap();
        assert_eq!(raw.len(), VOCAB * 4);
        assert_eq!(&raw[42 * 4..43 * 4], &values[42].to_le_bytes());
        assert_eq!(
            capture.arrays[0]["sha256"],
            format!("{:x}", Sha256::digest(&raw))
        );
    }

    #[test]
    fn actual_write_failure_retains_incomplete_descriptor() {
        let temp = Temporary::new();
        let path = temp.0.join("readonly.bin");
        File::create_new(&path).unwrap();
        let mut capture = Capture {
            output: BufWriter::with_capacity(8, File::open(&path).unwrap()),
            arrays: Vec::new(),
            bytes: 0,
            scenario: 0,
            scenario_rows: 0,
            error: None,
        };
        let error = capture.row(&PROMPT, &vec![0.0; VOCAB]).unwrap_err();
        assert!(error.raw_os_error().is_some());
        assert_eq!(capture.arrays.len(), 1);
        assert_eq!(capture.arrays[0]["completed"], false);
        assert!(capture.arrays[0].get("sha256").is_none());
        assert_eq!(capture.bytes, 0);
        assert_eq!(capture.scenario_rows, 0);
        assert_eq!(std::fs::metadata(path).unwrap().len(), 0);
    }
}
