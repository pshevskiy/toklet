//! Frozen C512 canonical-FP32 qualification. FP64 remains a separate diagnostic.
use serde_json::Value;
use sha2::{Digest, Sha256};

const MANIFEST_SHA: &str = "21d54e567025f70fdda01d7f1bede1a71f730c6b7c10aec62f5aaffec52a7f08";
const COMPARISON_SHA: &str = "1bbd129b5bf1811e6cdc898a433983c95d0547c9aedc05a03ad9c361564ce801";

fn frozen() -> (Value, Value) {
    let manifest = include_bytes!("fixtures/qwen3-context512-v1/manifest.json");
    let comparison = include_bytes!("fixtures/qwen3-context512-v1/comparison.json");
    assert_eq!(format!("{:x}", Sha256::digest(manifest)), MANIFEST_SHA);
    assert_eq!(format!("{:x}", Sha256::digest(comparison)), COMPARISON_SHA);
    (
        serde_json::from_slice(manifest).unwrap(),
        serde_json::from_slice(comparison).unwrap(),
    )
}

#[test]
fn qwen3_context512_schema_preserves_canonical_gates_and_known_precision_failures() {
    let (m, comparison) = frozen();
    let old: Value =
        serde_json::from_slice(include_bytes!("fixtures/mlx-qwen3-real-v1/manifest.json")).unwrap();
    assert_eq!(m["plan"]["context"], 512);
    assert_eq!(m["plan"]["layers"], serde_json::json!([0, 1, 27]));
    assert_eq!(m["records"].as_object().unwrap().len(), 4);
    for record in m["records"].as_object().unwrap().values() {
        assert_eq!(record["arrays"].as_object().unwrap().len(), 63);
        assert_eq!(record["bytes"], 508_297_216);
    }
    for (name, gate) in m["plan"]["gates"].as_object().unwrap() {
        assert_eq!(gate, &old["arrays"][format!("cases.tile17.{name}")]["gate"]);
    }
    assert_eq!(comparison["numerical_gates_passed"], false);
    assert_eq!(comparison["common_greedy_oracle_ready"], true);
    assert_eq!(comparison["causal_passed"], true);
    assert_eq!(comparison["mask_passed"], true);
    let failures: u64 = comparison["comparisons"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| v["failed_values"].as_u64().unwrap())
        .sum();
    assert_eq!(failures, 7282);
    let original = m["plan"]["cases"]["original"].as_array().unwrap();
    let future = m["plan"]["cases"]["future_tail"].as_array().unwrap();
    assert_eq!(original.len(), 512);
    assert_eq!(&original[..511], &future[..511]);
    assert_eq!(
        (original[511].as_u64(), future[511].as_u64()),
        (Some(0), Some(17))
    );
    for contract in ["fp32", "fp64"] {
        let ids = &comparison["selections"][format!("original.{contract}")]["ids"];
        assert_eq!(ids.as_array().unwrap().len(), 512);
        assert_eq!(ids[509], 1879);
        assert_eq!(ids[510], 0);
    }
}

#[cfg(all(feature = "metal", target_arch = "aarch64", target_os = "macos"))]
mod gpu {
    use super::*;
    use half::f16;
    use serde::Serialize;
    use serde_json::json;
    use std::{
        collections::BTreeMap,
        fs::{self, File, OpenOptions},
        io::{Read, Seek, SeekFrom, Write},
        os::unix::fs::{MetadataExt, OpenOptionsExt},
        path::{Path, PathBuf},
        sync::atomic::{AtomicBool, Ordering},
    };
    use toklet::{
        generation::{self, Input, Request, StopReason},
        metal::{self, AttentionMode, NormalizationMode, ProjectionMode, ReadbackMode},
        session,
        weights::LoadOptions,
    };
    const BUDGET: usize = 1536 * 1024 * 1024;
    const VOCABULARY: usize = 151936;
    const RAW_BYTES: usize = 504_102_912;
    const SOURCE: &str = "models/MLX-Qwen3-0.6B-8bit-11de968";
    const CAPTURE: &str = "artifacts/m8-qwen3-long-reference-checks-v1/capture-a";
    struct Recipe {
        name: &'static str,
        chunk: usize,
        segments: &'static [usize],
    }
    const RECIPES: [Recipe; 4] = [
        Recipe {
            name: "chunk32",
            chunk: 32,
            segments: &[512],
        },
        Recipe {
            name: "chunk128",
            chunk: 128,
            segments: &[512],
        },
        Recipe {
            name: "uneven",
            chunk: 128,
            segments: &[17, 33, 79, 1, 127, 128, 127],
        },
        Recipe {
            name: "late_single",
            chunk: 32,
            segments: &[511, 1],
        },
    ];

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }
    fn create(path: &Path) -> File {
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap()
    }
    fn json_file(path: &Path, value: &Value) {
        let mut file = create(path);
        serde_json::to_writer_pretty(&mut file, value).unwrap();
        file.write_all(b"\n").unwrap();
        file.flush().unwrap();
    }
    fn hash(file: &mut File) -> String {
        file.seek(SeekFrom::Start(0)).unwrap();
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 65536];
        let mut remaining = usize::try_from(file.metadata().unwrap().len()).unwrap();
        assert!(remaining <= 1024 * 1024 * 1024);
        while remaining > 0 {
            let count = remaining.min(buffer.len());
            file.read_exact(&mut buffer[..count]).unwrap();
            hasher.update(&buffer[..count]);
            remaining -= count;
        }
        assert_eq!(
            file.read(&mut buffer[..1]).unwrap(),
            0,
            "captured file grew"
        );
        format!("{:x}", hasher.finalize())
    }
    fn captured_file(path: &Path) -> File {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path)
            .unwrap();
        assert!(file.metadata().unwrap().is_file());
        file
    }
    fn identity(file: &File) -> (u64, u64, u64, i64, i64, i64, i64) {
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
    struct Oracle {
        file: File,
        path: PathBuf,
        record: Value,
        identity: (u64, u64, u64, i64, i64, i64, i64),
        row: Vec<u8>,
    }
    impl Oracle {
        fn open(record: &Value) -> Self {
            let filename = record["file"].as_str().unwrap();
            assert_eq!(Path::new(filename).file_name().unwrap(), filename);
            let path = root().join(CAPTURE).join(filename);
            assert!(!path.is_symlink());
            let mut file = captured_file(&path);
            assert_eq!(file.metadata().unwrap().len(), 508_297_216);
            assert_eq!(hash(&mut file), record["sha256"].as_str().unwrap());
            let identity = identity(&file);
            Self {
                file,
                path,
                record: record.clone(),
                identity,
                row: Vec::new(),
            }
        }
        fn verify(&mut self) {
            assert_eq!(identity(&self.file), self.identity);
            assert_eq!(
                hash(&mut self.file),
                self.record["sha256"].as_str().unwrap()
            );
            assert_eq!(identity(&captured_file(&self.path)), self.identity);
            assert!(!self.path.is_symlink());
        }
        fn compare(
            &mut self,
            name: &str,
            start: usize,
            columns: usize,
            actual: &[f32],
            gate: &Value,
            stats: &mut Stats,
        ) {
            let item = &self.record["arrays"][name];
            let shape = item["shape"].as_array().unwrap();
            assert_eq!(shape[0], 512);
            let expected_columns: usize = shape[1..]
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .product();
            assert_eq!(columns, expected_columns);
            assert_eq!(actual.len() % columns, 0);
            assert!(start + actual.len() / columns <= 512);
            let half = item["dtype"] == "<f2";
            assert!(half || item["dtype"] == "<f4");
            let width = if half { 2 } else { 4 };
            self.row.resize(columns * width, 0);
            let offset = item["offset"].as_u64().unwrap() + (start * columns * width) as u64;
            self.file.seek(SeekFrom::Start(offset)).unwrap();
            let atol = gate["atol"].as_f64().unwrap();
            let rtol = gate["rtol"].as_f64().unwrap();
            for (r, row) in actual.chunks_exact(columns).enumerate() {
                self.file.read_exact(&mut self.row).unwrap();
                if half {
                    for (column, (&a, b)) in row.iter().zip(self.row.chunks_exact(2)).enumerate() {
                        let expected = f16::from_le_bytes([b[0], b[1]]).to_f32();
                        stats.observe(start + r, column, a, expected, atol, rtol);
                    }
                } else {
                    for (column, (&a, b)) in row.iter().zip(self.row.chunks_exact(4)).enumerate() {
                        let expected = f32::from_le_bytes(b.try_into().unwrap());
                        stats.observe(start + r, column, a, expected, atol, rtol);
                    }
                }
            }
        }
    }
    #[derive(Default, Serialize)]
    struct Stats {
        values: u64,
        failed_values: u64,
        nonfinite_values: u64,
        maximum_absolute: f64,
        maximum_gate_fraction: f64,
        first_failures: Vec<Value>,
    }
    impl Stats {
        fn observe(
            &mut self,
            position: usize,
            feature: usize,
            actual: f32,
            expected: f32,
            atol: f64,
            rtol: f64,
        ) {
            self.values += 1;
            if !actual.is_finite() || !expected.is_finite() {
                self.nonfinite_values += 1;
                return;
            }
            let delta = (f64::from(actual) - f64::from(expected)).abs();
            let bound = atol + rtol * f64::from(expected).abs();
            self.maximum_absolute = self.maximum_absolute.max(delta);
            self.maximum_gate_fraction = self.maximum_gate_fraction.max(delta / bound);
            if delta > bound {
                self.failed_values += 1;
                if self.first_failures.len() < 8 {
                    self.first_failures.push(json!({"position":position,"feature":feature,"actual":actual,"expected":expected,"delta":delta,"bound":bound}));
                }
            }
        }
        fn passed(&self) -> bool {
            self.failed_values == 0 && self.nonfinite_values == 0
        }
    }
    struct Raw {
        file: File,
        journal: File,
        bytes: usize,
        digest: Sha256,
        scratch: Vec<u8>,
    }
    impl Raw {
        fn new(folder: &Path, label: &str) -> Self {
            Self {
                file: create(&folder.join(format!("{label}.bin"))),
                journal: create(&folder.join(format!("{label}.jsonl"))),
                bytes: 0,
                digest: Sha256::new(),
                scratch: Vec::with_capacity(65536),
            }
        }
        fn array(&mut self, name: &str, start: usize, rows: usize, columns: usize, values: &[f32]) {
            let offset = self.bytes;
            assert_eq!(values.len(), rows * columns);
            assert!(
                self.bytes
                    .checked_add(values.len().checked_mul(4).unwrap())
                    .unwrap()
                    <= RAW_BYTES
            );
            let mut digest = Sha256::new();
            for stripe in values.chunks(16384) {
                self.scratch.clear();
                for value in stripe {
                    self.scratch.extend_from_slice(&value.to_le_bytes());
                }
                self.file.write_all(&self.scratch).unwrap();
                digest.update(&self.scratch);
                self.digest.update(&self.scratch);
                self.bytes += self.scratch.len();
            }
            let record = json!({"name":name,"start":start,"rows":rows,"columns":columns,"dtype":"<f4","offset":offset,"bytes":values.len()*4,"sha256":format!("{:x}",digest.finalize())});
            serde_json::to_writer(&mut self.journal, &record).unwrap();
            self.journal.write_all(b"\n").unwrap();
        }
        fn flush(&mut self) {
            self.file.flush().unwrap();
            self.journal.flush().unwrap();
        }
    }
    fn name(d: &metal::Diagnostic) -> String {
        let stage = match d.stage {
            "q" => "q_projection",
            "k" => "k_projection",
            "v" => "v_projection",
            "gate" => "gate_projection",
            "silu" => "gate_silu",
            "up" => "up_projection",
            "down" => "down_projection",
            other => other,
        };
        d.layer.map_or_else(
            || stage.to_owned(),
            |layer| format!("layers.{layer}.{stage}"),
        )
    }
    fn session(model: &metal::Model, chunk: usize) -> metal::Session<'_> {
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
    fn tokens(m: &Value, case: &str) -> Vec<u32> {
        m["plan"]["cases"][case]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect()
    }
    fn saved(s: &metal::Session<'_>) -> (Vec<u32>, usize, usize, Option<Vec<f32>>) {
        (
            s.history().to_vec(),
            s.evaluated_len(),
            s.pending_len(),
            s.logits().map(<[f32]>::to_vec),
        )
    }

    fn trial(
        model: &metal::Model,
        m: &Value,
        expected: &Value,
        case: &str,
        recipe: &Recipe,
        folder: &Path,
    ) -> Value {
        let (recipe, chunk, segments) = (recipe.name, recipe.chunk, recipe.segments);
        let label = format!("{recipe}-{case}");
        let input = tokens(m, case);
        assert_eq!(segments.iter().sum::<usize>(), 512);
        let mut oracle = Oracle::open(&m["records"][format!("{case}.fp32")]);
        let mut wide = Oracle::open(&m["records"][format!("{case}.fp64")]);
        let names: Vec<_> = m["plan"]["gates"]
            .as_object()
            .unwrap()
            .keys()
            .filter(|name| {
                !name.ends_with("attention_scores") && !name.ends_with("attention_probabilities")
            })
            .cloned()
            .collect();
        assert_eq!(names.len(), 57);
        let mut canonical: BTreeMap<String, Stats> = names
            .iter()
            .map(|n| (n.clone(), Stats::default()))
            .collect();
        let mut fp64: BTreeMap<String, Stats> = names
            .iter()
            .map(|n| (n.clone(), Stats::default()))
            .collect();
        let mut writer = Raw::new(folder, &label);
        let mut s = session(model, chunk);
        s.enable_diagnostics(&[0, 1, 27]).unwrap();
        let mut appended = 0;
        let mut snapshots = 0;
        let mut ids = Vec::new();
        for &segment in segments {
            s.append(&input[appended..appended + segment]).unwrap();
            appended += segment;
            loop {
                let start = s.evaluated_len();
                let rows = s.evaluate_next_chunk().unwrap();
                if rows == 0 {
                    break;
                }
                let snapshots_now = s.diagnostics().unwrap();
                assert_eq!(snapshots_now.len(), 57);
                let mut diagnostics: BTreeMap<_, _> =
                    snapshots_now.into_iter().map(|d| (name(&d), d)).collect();
                assert_eq!(diagnostics.len(), 57);
                for name in &names {
                    let d = diagnostics.remove(name).unwrap();
                    assert_eq!(d.rows, rows);
                    let shape = m["records"][format!("{case}.fp32")]["arrays"][name]["shape"]
                        .as_array()
                        .unwrap();
                    let columns: usize = shape[1..]
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .product();
                    assert_eq!(d.columns, columns);
                    writer.array(name, start, rows, d.columns, &d.values);
                    let gate = &m["plan"]["gates"][name];
                    oracle.compare(
                        name,
                        start,
                        d.columns,
                        &d.values,
                        gate,
                        canonical.get_mut(name).unwrap(),
                    );
                    wide.compare(
                        name,
                        start,
                        d.columns,
                        &d.values,
                        gate,
                        fp64.get_mut(name).unwrap(),
                    );
                    if name == "logits" {
                        assert_eq!(d.columns, VOCABULARY);
                        for row in d.values.chunks_exact(VOCABULARY) {
                            ids.push(toklet::sampling::greedy(row).ok());
                        }
                    }
                    snapshots += 1;
                }
                assert!(diagnostics.is_empty());
                writer.flush();
            }
        }
        assert_eq!(
            (s.history(), s.evaluated_len(), s.pending_len()),
            (input.as_slice(), 512, 0)
        );
        let before = saved(&s);
        let error = s.append(&[11]).unwrap_err();
        assert!(
            matches!(
                error,
                metal::Error::Options("token ID/count outside session capacity")
            ),
            "{error:?}"
        );
        assert_eq!(saved(&s), before);
        let error = s.set_tokens(&[9707; 513]).unwrap_err();
        assert!(
            matches!(
                error,
                metal::Error::Options("token ID/count outside session capacity")
            ),
            "{error:?}"
        );
        assert_eq!(saved(&s), before);
        s.set_tokens(&input).unwrap();
        assert_eq!(s.evaluate_next_chunk().unwrap(), 0);
        assert_eq!(saved(&s), before);
        assert_eq!(ids.len(), 512);
        let canonical_ids = expected["selections"][format!("{case}.fp32")]["ids"]
            .as_array()
            .unwrap();
        let mismatches: Vec<_> = ids
            .iter()
            .zip(canonical_ids)
            .enumerate()
            .filter_map(|(i, (a, b))| (*a != b.as_u64().map(|v| v as u32)).then_some(i))
            .collect();
        assert_eq!(writer.bytes, RAW_BYTES);
        for name in &names {
            let shape = m["records"][format!("{case}.fp32")]["arrays"][name]["shape"]
                .as_array()
                .unwrap();
            let count: u64 = shape.iter().map(|v| v.as_u64().unwrap()).product();
            assert_eq!(canonical[name].values, count);
            assert_eq!(fp64[name].values, count);
        }
        oracle.verify();
        wide.verify();
        let result = json!({"case":case,"recipe":recipe,"chunk":chunk,"segments":segments,"snapshots":snapshots,"rows_per_state":512,"native_bytes":writer.bytes,"native_sha256":format!("{:x}",writer.digest.finalize()),"canonical":canonical,"fp64_diagnostic":fp64,"canonical_argmax_mismatches":mismatches,"argmax_ids":ids,"admitted_bytes":s.memory().admitted_bytes,"state_checks_passed":true,"canonical_passed":canonical.values().all(Stats::passed),"fp64_diagnostic_passed":fp64.values().all(Stats::passed)});
        json_file(&folder.join(format!("{label}.json")), &result);
        result
    }

    fn request(input: Input<'_>, maximum: usize) -> Request<'_> {
        Request {
            input,
            max_new_tokens: maximum,
            stop_tokens: &[],
        }
    }
    fn contracts(model: &metal::Model, m: &Value, folder: &Path) -> Value {
        let prompt = tokens(m, "original");
        let mut s = session(model, 32);
        let flag = AtomicBool::new(false);
        let mut emitted = Vec::new();
        let cancelled = generation::generate(
            &mut s,
            request(Input::Prompt(&prompt[..510]), 2),
            &flag,
            |event| {
                emitted.push(event.token_id);
                flag.store(true, Ordering::Relaxed);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(cancelled.reason, StopReason::Cancelled);
        assert_eq!(emitted, [1879]);
        assert_eq!(
            (s.evaluated_len(), s.pending_len(), s.history().len()),
            (510, 1, 511)
        );
        assert_eq!(s.history(), &prompt[..511]);
        let before = saved(&s);
        let untouched = generation::generate(
            &mut s,
            request(Input::Reset(&prompt[..4]), 1),
            &flag,
            |_| panic!("cancelled reset emitted"),
        )
        .unwrap();
        assert_eq!(untouched.reason, StopReason::Cancelled);
        assert_eq!(saved(&s), before);
        flag.store(false, Ordering::Relaxed);
        generation::generate(&mut s, request(Input::Continue, 1), &flag, |event| {
            emitted.push(event.token_id);
            Ok(())
        })
        .unwrap();
        assert_eq!(emitted, [1879, 0]);
        assert_eq!((s.evaluated_len(), s.pending_len()), (511, 1));
        assert_eq!(s.history(), prompt);
        let before = saved(&s);
        let error = generation::generate(&mut s, request(Input::Continue, 1), &flag, |_| {
            panic!("overflow emitted")
        })
        .unwrap_err();
        assert!(
            matches!(
                error,
                generation::Error::InvalidRequest(
                    "prompt plus requested output exceeds session capacity"
                )
            ),
            "{error:?}"
        );
        assert_eq!(saved(&s), before);
        let error = generation::generate(&mut s, request(Input::Reset(&prompt), 1), &flag, |_| {
            panic!("reset overflow emitted")
        })
        .unwrap_err();
        assert!(
            matches!(
                error,
                generation::Error::InvalidRequest(
                    "prompt plus requested output exceeds session capacity"
                )
            ),
            "{error:?}"
        );
        assert_eq!(saved(&s), before);
        generation::generate(&mut s, request(Input::Continue, 0), &flag, |_| {
            panic!("prefill-only emitted")
        })
        .unwrap();
        assert_eq!((s.evaluated_len(), s.pending_len()), (512, 0));
        let complete = s.logits().unwrap().to_vec();
        let mut raw = Raw::new(folder, "contracts-final-logits");
        raw.array("logits", 511, 1, VOCABULARY, &complete);
        raw.flush();
        let mut canonical = Stats::default();
        let mut fp64 = Stats::default();
        let mut oracle = Oracle::open(&m["records"]["original.fp32"]);
        let mut wide = Oracle::open(&m["records"]["original.fp64"]);
        oracle.compare(
            "logits",
            511,
            VOCABULARY,
            &complete,
            &m["plan"]["gates"]["logits"],
            &mut canonical,
        );
        wide.compare(
            "logits",
            511,
            VOCABULARY,
            &complete,
            &m["plan"]["gates"]["logits"],
            &mut fp64,
        );
        oracle.verify();
        wide.verify();
        drop((oracle, wide));
        drop(s);
        let mut fresh = session(model, 32);
        fresh.set_tokens(&prompt).unwrap();
        fresh.evaluate().unwrap();
        assert_eq!(fresh.logits().unwrap(), complete);
        drop(fresh);
        let mut s = session(model, 32);
        let failed = generation::generate(
            &mut s,
            request(Input::Prompt(&prompt[..510]), 2),
            &flag,
            |_| Err(std::io::Error::other("injected output failure")),
        );
        match failed {
            Err(generation::Error::Output(error)) => {
                assert_eq!(error.kind(), std::io::ErrorKind::Other);
                assert_eq!(error.to_string(), "injected output failure");
            }
            other => panic!("unexpected output failure result: {other:?}"),
        }
        assert_eq!(
            (s.evaluated_len(), s.pending_len(), s.history().len()),
            (510, 1, 511)
        );
        assert_eq!(s.history(), &prompt[..511]);
        generation::generate(&mut s, request(Input::Continue, 1), &flag, |event| {
            assert_eq!(event.token_id, 0);
            Ok(())
        })
        .unwrap();
        generation::generate(&mut s, request(Input::Continue, 0), &flag, |_| {
            panic!("flush emitted")
        })
        .unwrap();
        assert_eq!(s.logits().unwrap(), complete);
        drop(s);
        let mut partial = session(model, 128);
        partial.set_tokens(&prompt).unwrap();
        assert_eq!(partial.evaluate_next_chunk().unwrap(), 128);
        partial.discard_pending();
        assert_eq!(partial.history(), &prompt[..128]);
        partial.append(&prompt[128..]).unwrap();
        partial.evaluate().unwrap();
        let partial_logits = partial.logits().unwrap().to_vec();
        drop(partial);
        let mut same = session(model, 128);
        same.set_tokens(&prompt).unwrap();
        same.evaluate().unwrap();
        assert_eq!(same.logits().unwrap(), partial_logits);
        let result = json!({"passed":true,"sequence":"original","capacity":512,"callback_cancel_prompt":510,"requested_output":2,"emitted_before_cancel":1,"resumed_ids":emitted,"pending_then_flushed":true,"rejected_513_state_unchanged":true,"output_failure_retry":true,"sampled_history_fresh_replay":true,"partial_discard_reappend":true,"mid_gpu_cancellation_qualified":false,"canonical_fp32_passed":canonical.passed(),"canonical":canonical,"fp64_diagnostic":fp64});
        json_file(&folder.join("contracts.json"), &result);
        result
    }

    fn future_prefix(folder: &Path, recipe: &str) {
        let left = folder.join(format!("{recipe}-original"));
        let right = folder.join(format!("{recipe}-future_tail"));
        let journal = |path: PathBuf| {
            let mut text = String::new();
            captured_file(&path)
                .take(4 * 1024 * 1024 + 1)
                .read_to_string(&mut text)
                .unwrap();
            assert!(text.len() <= 4 * 1024 * 1024);
            text
        };
        let a = journal(left.with_extension("jsonl"));
        let b = journal(right.with_extension("jsonl"));
        let mut fa = File::open(left.with_extension("bin")).unwrap();
        let mut fb = File::open(right.with_extension("bin")).unwrap();
        let mut aa = vec![0; 65536];
        let mut bb = vec![0; 65536];
        assert_eq!(a.lines().count(), b.lines().count());
        for (a, b) in a.lines().zip(b.lines()) {
            let a: Value = serde_json::from_str(a).unwrap();
            let b: Value = serde_json::from_str(b).unwrap();
            for key in ["name", "start", "rows", "columns", "offset", "bytes"] {
                assert_eq!(a[key], b[key]);
            }
            let start = a["start"].as_u64().unwrap() as usize;
            let rows = a["rows"].as_u64().unwrap() as usize;
            let columns = a["columns"].as_u64().unwrap() as usize;
            let rows = rows.min(511usize.saturating_sub(start));
            let mut remaining = rows * columns * 4;
            fa.seek(SeekFrom::Start(a["offset"].as_u64().unwrap()))
                .unwrap();
            fb.seek(SeekFrom::Start(b["offset"].as_u64().unwrap()))
                .unwrap();
            while remaining > 0 {
                let count = remaining.min(aa.len());
                fa.read_exact(&mut aa[..count]).unwrap();
                fb.read_exact(&mut bb[..count]).unwrap();
                assert_eq!(aa[..count], bb[..count]);
                remaining -= count;
            }
        }
    }

    #[test]
    #[ignore = "Explicit Mac GPU experimental CompensatedReference capture; saved-logit screening is separate"]
    fn native_qwen3_compensated_context512_logits_only() {
        const LOGIT_BYTES: usize = 512 * VOCABULARY * 4;
        let (m, _) = frozen();
        let folder = PathBuf::from(
            std::env::var_os("TOKLET_COMPENSATED_CONTEXT512_OUTPUT")
                .expect("owned fresh compensated capture output required"),
        );
        assert!(
            folder.is_absolute()
                && folder.components().all(|part| !matches!(
                    part,
                    std::path::Component::ParentDir | std::path::Component::CurDir
                ))
        );
        assert!(
            folder
                .parent()
                .unwrap()
                .canonicalize()
                .unwrap()
                .starts_with(root().join("artifacts").canonicalize().unwrap())
        );
        fs::create_dir(&folder).unwrap();
        // Retain the authenticated descriptors through execution, then verify
        // their bytes and path identities again before declaring completion.
        let mut sources = Vec::new();
        for (name, pin) in m["model_pins"].as_object().unwrap() {
            let path = root().join(SOURCE).join(name);
            let mut file = captured_file(&path);
            let before = identity(&file);
            assert_eq!(hash(&mut file), pin.as_str().unwrap());
            assert_eq!(identity(&file), before);
            sources.push((path, pin.as_str().unwrap().to_owned(), file, before));
        }
        let model = metal::Model::load_mlx(
            &root().join(SOURCE),
            LoadOptions {
                context_limit: 512,
                memory_limit_bytes: BUDGET,
            },
        )
        .unwrap();
        let device = model.device_name();
        let mut results = Vec::new();
        for case in ["original", "future_tail"] {
            let input = tokens(&m, case);
            assert_eq!(input.len(), 512);
            let label = format!("compensated-{case}");
            let mut writer = Raw::new(&folder, &label);
            let mut s = metal::Session::with_attention_mode(
                &model,
                session::Options {
                    capacity: 512,
                    prefill_chunk_size: 128,
                    position_offset: 0,
                    memory_limit_bytes: BUDGET,
                },
                ProjectionMode::MlxAffine,
                NormalizationMode::Compensated,
                ReadbackMode::FullLogits,
                AttentionMode::CompensatedReference,
            )
            .unwrap();
            // Existing empty-layer diagnostics admit all-row logits plus two
            // small embedding/final-norm snapshots. Persist only logits.
            s.enable_diagnostics(&[]).unwrap();
            s.append(&input).unwrap();
            let mut frames = 0;
            for start in (0..512).step_by(128) {
                assert_eq!(s.evaluated_len(), start);
                assert_eq!(s.evaluate_next_chunk().unwrap(), 128);
                let diagnostics = s.diagnostics().unwrap();
                assert_eq!(
                    diagnostics.iter().map(|d| d.stage).collect::<Vec<_>>(),
                    ["embedding", "final_norm", "logits"]
                );
                let logits = diagnostics
                    .into_iter()
                    .find(|d| d.stage == "logits")
                    .unwrap();
                assert_eq!(
                    (logits.layer, logits.rows, logits.columns),
                    (None, 128, VOCABULARY)
                );
                writer.array("logits", start, 128, VOCABULARY, &logits.values);
                writer.flush();
                assert_eq!(writer.bytes, (start + 128) * VOCABULARY * 4);
                frames += 1;
            }
            assert_eq!(s.evaluate_next_chunk().unwrap(), 0);
            assert_eq!(
                (s.history(), s.evaluated_len(), s.pending_len()),
                (input.as_slice(), 512, 0)
            );
            assert_eq!((frames, writer.bytes), (4, LOGIT_BYTES));
            let native_sha = format!("{:x}", writer.digest.clone().finalize());
            let mut raw = captured_file(&folder.join(format!("{label}.bin")));
            assert_eq!(identity(&raw), identity(&writer.file));
            assert_eq!(raw.metadata().unwrap().len(), LOGIT_BYTES as u64);
            assert_eq!(hash(&mut raw), native_sha);
            let mut journal = captured_file(&folder.join(format!("{label}.jsonl")));
            assert_eq!(identity(&journal), identity(&writer.journal));
            let journal_sha = hash(&mut journal);
            let result = json!({"case":case,"attention_mode":"CompensatedReference", "chunk":128,
                "frames":frames,"rows":512,"vocabulary":VOCABULARY,"dtype":"<f4",
                "native_bytes":writer.bytes,"native_sha256":native_sha,
                "journal_bytes":journal.metadata().unwrap().len(),"journal_sha256":journal_sha,
                "history":s.history(),"evaluated_len":s.evaluated_len(),"pending_len":s.pending_len(),
                "history_matches_frozen_input":true,"memory":s.memory(),"capture_complete":true,
                "numerical_screen_evaluated":false,"numerically_qualified":false});
            json_file(&folder.join(format!("{label}.json")), &result);
            results.push(result);
            println!(
                "TOKLET_QWEN3_COMPENSATED_CAPTURE {}",
                json!({"case":case,"frames":frames,"native_bytes":writer.bytes,"native_sha256":native_sha})
            );
        }
        future_prefix(&folder, "compensated");
        drop(model);
        for (path, expected, file, before) in &mut sources {
            assert_eq!(identity(file), *before);
            assert_eq!(hash(file), *expected);
            assert_eq!(identity(file), *before);
            assert_eq!(identity(&captured_file(path)), *before);
        }
        let summary = json!({"device":device,"manifest_sha256":MANIFEST_SHA,
            "comparison_sha256":COMPARISON_SHA,"attention_mode":"CompensatedReference",
            "projection_mode":"MlxAffine","normalization_mode":"Compensated",
            "capacity":512,"chunk":128,"position_offset":0,"activation_dtype":"FP16","kv_dtype":"FP16",
            "persisted_stages":["logits"],"allocated_diagnostic_stages":["embedding","final_norm","logits"],
            "cases":results,"future_prefix_exact":true,"source_endpoints_checked":true,
            "capture_complete":true,"numerical_screen_evaluated":false,"numerically_qualified":false,
            "historical_qualification_changed":false,"quality_evaluated":false,"validation_opened":false});
        json_file(&folder.join("compensated-summary.json"), &summary);
        // Completion means the authenticated capture and state checks passed.
        // The unchanged output screen classifies these saved logits separately.
    }

    #[test]
    #[ignore = "Explicit Mac GPU, frozen real checkpoint and complete C512 CPU captures"]
    fn native_qwen3_context512_all_rows_chunks_and_pending_contracts() {
        let (m, expected) = frozen();
        let folder = PathBuf::from(
            std::env::var_os("TOKLET_CONTEXT512_OUTPUT")
                .expect("owned fresh artifact output required"),
        );
        assert!(
            folder.is_absolute()
                && folder.components().all(|part| !matches!(
                    part,
                    std::path::Component::ParentDir | std::path::Component::CurDir
                ))
        );
        assert!(
            folder
                .parent()
                .unwrap()
                .canonicalize()
                .unwrap()
                .starts_with(root().join("artifacts").canonicalize().unwrap())
        );
        fs::create_dir(&folder).unwrap();
        for (name, pin) in m["model_pins"].as_object().unwrap() {
            assert_eq!(
                hash(&mut captured_file(&root().join(SOURCE).join(name))),
                pin.as_str().unwrap()
            );
        }
        let model = metal::Model::load_mlx(
            &root().join(SOURCE),
            LoadOptions {
                context_limit: 512,
                memory_limit_bytes: BUDGET,
            },
        )
        .unwrap();
        let mut results = Vec::new();
        for recipe in &RECIPES {
            for case in ["original", "future_tail"] {
                results.push(trial(&model, &m, &expected, case, recipe, &folder));
            }
            future_prefix(&folder, recipe.name);
        }
        let contracts = contracts(&model, &m, &folder);
        for (name, pin) in m["model_pins"].as_object().unwrap() {
            assert_eq!(
                hash(&mut captured_file(&root().join(SOURCE).join(name))),
                pin.as_str().unwrap()
            );
        }
        let canonical_passed = results.iter().all(|r| {
            r["canonical_passed"] == true
                && r["canonical_argmax_mismatches"]
                    .as_array()
                    .unwrap()
                    .is_empty()
        }) && contracts["canonical_fp32_passed"] == true;
        let snapshots: u64 = results
            .iter()
            .map(|r| r["snapshots"].as_u64().unwrap())
            .sum();
        assert_eq!(snapshots, 5016);
        let summary = json!({"device":model.device_name(),"manifest_sha256":MANIFEST_SHA,"comparison_sha256":COMPARISON_SHA,"capacity":512,"recipes":results,"stage_snapshots":snapshots,"contracts":contracts,"canonical_fp32_passed":canonical_passed,"fp64_is_diagnostic_only":true,"future_prefix_exact":true,"source_quality_qualified":false,"held_out_quality_qualified":false,"full_model_window_qualified":false,"phone_qualified":false,"precision_or_capacity_changed":false});
        json_file(&folder.join("summary.json"), &summary);
        println!(
            "TOKLET_QWEN3_CONTEXT512 {}",
            json!({"canonical_fp32_passed":canonical_passed,"stage_snapshots":snapshots,"source_quality_qualified":false,"phone_qualified":false})
        );
        assert!(
            canonical_passed,
            "canonical numerical failures retained in raw output/summary"
        );
    }
}
