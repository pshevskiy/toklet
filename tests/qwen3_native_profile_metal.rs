//! Explicit native-profile state validation; no CPU trajectory tolerance.
#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
};
use toklet::{
    metal::{self, AttentionMode, NormalizationMode, ProjectionMode, ReadbackMode},
    session,
    weights::LoadOptions,
};

const CONTRACT: &str = "05c6046c10945b637735c9a86de82d17f42372dd6e0fec1124b22c4ff47afb24";
const VOCAB: usize = 151936;
const BUDGET: usize = 1536 * 1024 * 1024;
type Identity = (u64, u64, u64, i64, i64, i64, i64);

fn identity(file: &File) -> Identity {
    let m = file.metadata().unwrap();
    (
        m.dev(),
        m.ino(),
        m.len(),
        m.mtime(),
        m.mtime_nsec(),
        m.ctime(),
        m.ctime_nsec(),
    )
}
fn open(path: &Path) -> File {
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .unwrap();
    assert!(f.metadata().unwrap().is_file());
    f
}
fn hash(file: &mut File) -> String {
    file.seek(SeekFrom::Start(0)).unwrap();
    let mut remaining = file.metadata().unwrap().len();
    assert!(remaining <= 1024 * 1024 * 1024);
    let mut buffer = [0; 65536];
    let mut h = Sha256::new();
    while remaining > 0 {
        let n = usize::try_from(remaining.min(buffer.len() as u64)).unwrap();
        file.read_exact(&mut buffer[..n]).unwrap();
        h.update(&buffer[..n]);
        remaining -= n as u64;
    }
    assert_eq!(file.read(&mut buffer[..1]).unwrap(), 0);
    format!("{:x}", h.finalize())
}
struct Owner {
    path: PathBuf,
    file: File,
    before: Identity,
    digest: String,
}
impl Owner {
    fn new(path: PathBuf, expected: &str) -> Self {
        let mut file = open(&path);
        let before = identity(&file);
        let digest = hash(&mut file);
        assert_eq!(digest, expected, "{}", path.display());
        assert_eq!(identity(&file), before);
        Self {
            path,
            file,
            before,
            digest,
        }
    }
    fn bytes(&mut self, limit: usize) -> Vec<u8> {
        assert!(self.before.2 <= limit as u64);
        self.file.seek(SeekFrom::Start(0)).unwrap();
        let mut bytes = vec![0; self.before.2 as usize];
        self.file.read_exact(&mut bytes).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), self.digest);
        bytes
    }
    fn close_check(&mut self) {
        assert_eq!(identity(&self.file), self.before);
        assert_eq!(hash(&mut self.file), self.digest);
        assert_eq!(identity(&self.file), self.before);
        assert_eq!(identity(&open(&self.path)), self.before);
    }
}
fn write(file: &mut File, record: &Value) {
    serde_json::to_writer(&mut *file, record).unwrap();
    file.write_all(b"\n").unwrap();
    file.flush().unwrap();
}
fn create(path: &Path) -> File {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap()
}
fn row_hash(row: &[f32]) -> String {
    assert_eq!(row.len(), VOCAB);
    let mut h = Sha256::new();
    for x in row {
        h.update(x.to_bits().to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn new_session(model: &metal::Model, chunk: usize) -> metal::Session<'_> {
    metal::Session::with_attention_mode(
        model,
        session::Options {
            capacity: 512,
            prefill_chunk_size: chunk,
            position_offset: 0,
            memory_limit_bytes: BUDGET,
        },
        ProjectionMode::MlxAffine,
        NormalizationMode::Compensated,
        ReadbackMode::FullLogits,
        AttentionMode::Reference,
    )
    .unwrap()
}
fn state(s: &metal::Session<'_>) -> Value {
    json!({"history":s.history(), "evaluated":s.evaluated_len(), "pending":s.pending_len(),
        "logits_sha256":s.logits().map(row_hash)})
}

fn trial(
    model: &metal::Model,
    id: &str,
    tokens: &[u32],
    chunk: usize,
    folder: &Path,
    expected: Option<&[String]>,
) -> Vec<String> {
    let label = format!("{id}-chunk{chunk}");
    let mut journal = create(&folder.join(format!("{label}.jsonl")));
    let mut s = new_session(model, chunk);
    s.enable_diagnostics(&[]).unwrap();
    s.append(tokens).unwrap();
    write(
        &mut journal,
        &json!({"kind":"start", "case":id,"chunk":chunk,"tokens":tokens.len(),
        "profile":"MlxAffine/Compensated/Reference/FP16/FullLogits", "memory":s.memory()}),
    );
    assert_eq!(
        (s.history(), s.evaluated_len(), s.pending_len()),
        (tokens, 0, tokens.len())
    );
    let mut rows = Vec::with_capacity(tokens.len());
    while s.evaluated_len() < tokens.len() {
        let start = s.evaluated_len();
        let n = s.evaluate_next_chunk().unwrap();
        assert_eq!(n, chunk.min(tokens.len() - start));
        let snapshots = s.diagnostics().unwrap();
        assert_eq!(
            snapshots.iter().map(|d| d.stage).collect::<Vec<_>>(),
            ["embedding", "final_norm", "logits"]
        );
        let logits = snapshots.into_iter().find(|d| d.stage == "logits").unwrap();
        assert_eq!(
            (logits.layer, logits.rows, logits.columns),
            (None, n, VOCAB)
        );
        for (r, row) in logits.values.chunks_exact(VOCAB).enumerate() {
            let position = start + r;
            let digest = row_hash(row);
            let invalid = row.iter().position(|x| !x.is_finite());
            let mismatch = expected.is_some_and(|x| x[position] != digest);
            write(
                &mut journal,
                &json!({"kind":"row", "position":position,"sha256":digest,
                "first_nonfinite":invalid.map(|i| json!({"vocabulary_id":i,"bits":row[i].to_bits()})),
                "expected_sha256":expected.map(|x| &x[position]), "matches":!mismatch}),
            );
            if invalid.is_some() || mismatch {
                // At most one complete failing row, retained before assertion;
                // successful rows retain only authenticated hashes.
                let mut witness = create(&folder.join(format!("{label}-failure.json")));
                write(
                    &mut witness,
                    &json!({"position":position,"actual_fp32_bits":row.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),
                    "expected_row_sha256":expected.map(|x| &x[position])}),
                );
            }
            assert!(invalid.is_none(), "{label} row{position} nonfinite");
            assert!(!mismatch, "{label} row{position} schedule mismatch");
            rows.push(digest);
        }
    }
    assert_eq!(rows.len(), tokens.len());
    assert_eq!(
        (s.history(), s.evaluated_len(), s.pending_len()),
        (tokens, tokens.len(), 0)
    );
    assert_eq!(row_hash(s.logits().unwrap()), *rows.last().unwrap());
    let complete = state(&s);
    s.set_tokens(tokens).unwrap();
    assert_eq!(s.evaluate_next_chunk().unwrap(), 0);
    let cached = state(&s);
    write(
        &mut journal,
        &json!({"kind":"cached", "before":complete,"after":cached}),
    );
    assert_eq!(cached, complete);
    let too_many = vec![tokens[0]; 513 - tokens.len()];
    let append_result = s.append(&too_many);
    let after_append = state(&s);
    write(
        &mut journal,
        &json!({"kind":"capacity_rejection", "operation":"append", "requested_history":513,
            "result":format!("{append_result:?}"), "before":complete,"after":after_append}),
    );
    assert!(matches!(
        append_result,
        Err(metal::Error::Options(
            "token ID/count outside session capacity"
        ))
    ));
    assert_eq!(after_append, complete);
    let set_result = s.set_tokens(&vec![tokens[0]; 513]);
    let after_set = state(&s);
    write(
        &mut journal,
        &json!({"kind":"capacity_rejection", "operation":"set_tokens", "requested_history":513,
            "result":format!("{set_result:?}"), "before":complete,"after":after_set}),
    );
    assert!(matches!(
        set_result,
        Err(metal::Error::Options(
            "token ID/count outside session capacity"
        ))
    ));
    assert_eq!(after_set, complete);
    // A reset replay with the last prescribed token held pending tests exact
    // prefix reuse, discard/reappend and continuation without invented tokens.
    s.reset();
    assert_eq!(
        (s.history().len(), s.evaluated_len(), s.pending_len()),
        (0, 0, 0)
    );
    assert!(s.logits().is_none());
    s.append(&tokens[..tokens.len() - 1]).unwrap();
    s.evaluate().unwrap();
    assert_eq!(row_hash(s.logits().unwrap()), rows[tokens.len() - 2]);
    let prefix = state(&s);
    s.append(&tokens[tokens.len() - 1..]).unwrap();
    assert_eq!((s.evaluated_len(), s.pending_len()), (tokens.len() - 1, 1));
    s.discard_pending();
    assert_eq!(state(&s), prefix);
    s.append(&tokens[tokens.len() - 1..]).unwrap();
    assert_eq!(s.evaluate_next_chunk().unwrap(), 1);
    let replay = state(&s);
    write(
        &mut journal,
        &json!({"kind":"replay", "prefix":prefix,"complete":complete,"replayed":replay,
        "rejected_513_append_and_set_atomic":true}),
    );
    assert_eq!(replay, complete);
    write(
        &mut journal,
        &json!({"kind":"complete", "rows":rows.len(),"passed":true}),
    );
    rows
}

#[test]
#[ignore = "Explicit frozen native-profile validation; opens eight held-out arrays and real Metal model"]
fn native_profile_validation_rows_and_state() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut owners = vec![Owner::new(
        root.join("benchmarks/qwen3-native-profile-v1/contract.json"),
        CONTRACT,
    )];
    let contract: Value = serde_json::from_slice(&owners[0].bytes(65536)).unwrap();
    assert_eq!(contract["profile_id"], "qwen3-0.6b-mlx8-fp16-reference-v1");
    let validation = &contract["conformance"];
    let manifest = &validation["validation_manifest"];
    owners.push(Owner::new(
        root.join(manifest["path"].as_str().unwrap()),
        manifest["identity"]["sha256"].as_str().unwrap(),
    ));
    assert_eq!(
        owners.last().unwrap().before.2,
        manifest["identity"]["bytes"].as_u64().unwrap()
    );
    let cases = validation["validation_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 8);
    let mut inputs = Vec::new();
    for case in cases {
        let filename = case["file"].as_str().unwrap();
        assert_eq!(Path::new(filename).file_name().unwrap(), filename);
        let mut owner = Owner::new(
            root.join("benchmarks/qwen3-context-candidate-v2")
                .join(filename),
            case["token_array"]["sha256"].as_str().unwrap(),
        );
        let bytes = owner.bytes(2048);
        assert_eq!(
            bytes.len() as u64,
            case["token_array"]["bytes"].as_u64().unwrap()
        );
        let tokens: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(bytes.len(), tokens.len() * 4);
        assert_eq!(tokens.len() as u64, case["tokens"].as_u64().unwrap());
        assert!(tokens.iter().all(|t| (*t as usize) < VOCAB));
        inputs.push(tokens);
        owners.push(owner);
    }
    let source = root.join("models/MLX-Qwen3-0.6B-8bit-11de968");
    for (name, digest) in contract["model"]["files"].as_object().unwrap() {
        owners.push(Owner::new(source.join(name), digest.as_str().unwrap()));
    }
    let folder = PathBuf::from(
        std::env::var_os("TOKLET_NATIVE_PROFILE_OUTPUT").expect("explicit fresh output directory"),
    );
    assert!(folder.is_absolute());
    fs::create_dir(&folder).unwrap();
    let mut summary = create(&folder.join("validation.jsonl"));
    let model = metal::Model::load_mlx(
        &source,
        LoadOptions {
            context_limit: 512,
            memory_limit_bytes: BUDGET,
        },
    )
    .unwrap();
    let mut prefixes: BTreeMap<String, (Vec<u32>, Vec<String>)> = BTreeMap::new();
    let mut total = 0;
    for (case, tokens) in cases.iter().zip(&inputs) {
        let id = case["id"].as_str().unwrap();
        let rows = trial(&model, id, tokens, 128, &folder, None);
        let other = trial(&model, id, tokens, 32, &folder, Some(&rows));
        assert_eq!(rows, other);
        let key = case["source"].as_str().unwrap().to_owned();
        if let Some((old_tokens, old_rows)) = prefixes.get(&key) {
            assert!(old_tokens.len() < tokens.len());
            assert_eq!(old_tokens, &tokens[..old_tokens.len()]);
            let mismatch = old_rows.iter().zip(&rows).position(|(a, b)| a != b);
            write(
                &mut summary,
                &json!({"kind":"prefix", "case":id,"prefix_rows":old_rows.len(),
                "first_mismatch":mismatch.map(|i|json!({"position":i,"previous":old_rows[i],"current":rows[i]}))}),
            );
            assert!(mismatch.is_none(), "{id} nested prefix mismatch");
        }
        total += rows.len();
        prefixes.insert(key, (tokens.clone(), rows));
        write(
            &mut summary,
            &json!({"kind":"case", "id":id,"rows":tokens.len(),"passed":true}),
        );
    }
    assert_eq!((total, prefixes.len()), (2175, 3));
    drop(model);
    for owner in &mut owners {
        owner.close_check();
    }
    write(
        &mut summary,
        &json!({"kind":"complete", "contract_sha256":CONTRACT,"cases":8,
        "rows_per_schedule":total,"chunks":[128,32],"source_input_endpoints_checked":true,
        "conformance_passed":true,"quality_evaluated":false,"profile_qualified":false,
        "historical_fidelity_rejections_unchanged":true}),
    );
}
