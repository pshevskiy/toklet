//! Explicit full-Granite diagnostic capture. This does not qualify quality,
//! performance, a managed artifact, or an application route.
#[cfg(any(test, all(target_os = "macos", target_arch = "aarch64")))]
#[path = "support/granite_graph_io.rs"]
mod graph_io;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod native {
    use super::graph_io::{self as io, Array, Dtype, Result, Stage};
    use serde::{Deserialize, Serialize};
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::{
        collections::{BTreeMap, BTreeSet},
        fs::{self, File, Metadata, OpenOptions},
        io::{BufWriter, Read, Write},
        os::unix::fs::{MetadataExt, OpenOptionsExt},
        path::{Path, PathBuf},
    };
    use toklet::{metal, sampling, session, weights};

    const PROTOCOL_SHA: &str = "e4dec7ec0ca7e08e87bfbf538a71fca703d5553484e5a0e2eddc250418572c1d";
    const DATA_SHA: &str = "02f938b9f2a82d61c8609e19d2d65fcd7b32e52cc5b1cc7834b9843b9c4eec1e";
    const CONFIG_SHA: &str = "889ff79da95a5526811498cd47cee14c92a9d494fbd5dca70a3beb1c71b3df9b";
    const BUDGET: usize = 6 * 1024 * 1024 * 1024;
    const HELPER: usize = 32 * 1024 * 1024;
    // Protocol, source/manifest state, row vectors, baseline hash tables, writer
    // buffers and temporary exporter metadata fit in this separate allowance.
    // Retained arrays/patch metadata receive the remaining helper reservation.
    const FIXED_HELPER: usize = 8 * 1024 * 1024;
    const RAW_BYTES: u64 = 5_279_121_408;
    const LAYERS: usize = 40;
    const VOCAB: usize = 100_352;
    const STAGES: [&str; 16] = [
        "input_norm",
        "q",
        "k",
        "v",
        "q_rotary",
        "k_rotary",
        "attention_context",
        "attention_projection",
        "attention_residual",
        "post_attention_norm",
        "gate",
        "silu",
        "up",
        "gated",
        "down",
        "output",
    ];

    // Unknown top-level protocol fields stay authenticated by its external
    // whole-file pin. Numerical inputs have explicit integer/scalar types.
    #[derive(Deserialize)]
    struct Protocol {
        format: String,
        model: ModelPins,
        geometry: Geometry,
        cases: Vec<Case>,
        native: NativeOptions,
        memory_and_disk: Limits,
    }
    #[derive(Deserialize)]
    struct ModelPins {
        data_sha256: String,
        config_sha256: String,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Geometry {
        hidden: usize,
        intermediate: usize,
        layers: usize,
        vocabulary: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        theta: f64,
        rms_eps: f64,
        attention_scale: f64,
        embedding_scale: f64,
        residual_scale: f64,
        logits_divisor: f64,
        tied_head: bool,
    }
    #[derive(Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct Case {
        name: String,
        tokens: Vec<u32>,
        offset: usize,
    }
    #[derive(Deserialize)]
    struct NativeOptions {
        capacity: usize,
        chunks: Vec<usize>,
        projection: String,
        normalization: String,
        attention: String,
        readback: String,
    }
    #[derive(Deserialize)]
    struct Limits {
        engine_limit_bytes: usize,
        helper_reserve_bytes: usize,
        disk_floor_bytes: u64,
        raw_bytes_per_reference_profile_or_lossless_native_capture: u64,
    }
    impl Protocol {
        fn validate(&self) -> Result<()> {
            let g = &self.geometry;
            if self.format != "toklet-granite-full-graph-prospective-v1"
                || self.model.data_sha256 != DATA_SHA
                || self.model.config_sha256 != CONFIG_SHA
                || (
                    g.hidden,
                    g.intermediate,
                    g.layers,
                    g.vocabulary,
                    g.query_heads,
                    g.kv_heads,
                    g.head_dim,
                ) != (2560, 8192, LAYERS, VOCAB, 40, 8, 64)
                || (
                    g.theta,
                    g.rms_eps,
                    g.attention_scale,
                    g.embedding_scale,
                    g.residual_scale,
                    g.logits_divisor,
                    g.tied_head,
                ) != (10_000_000., 1e-5, 0.015625, 1., 1., 1., false)
                || self.native.capacity != 512
                || self.native.chunks != [128, 32, 1]
                || self.native.projection != "MlxAffine"
                || self.native.normalization != "Compensated"
                || self.native.attention != "Reference"
                || self.native.readback != "FullLogits plus separate GreedyToken state checks"
                || self.memory_and_disk.engine_limit_bytes != BUDGET
                || self.memory_and_disk.helper_reserve_bytes != HELPER
                || self.memory_and_disk.disk_floor_bytes != 2 * 1024 * 1024 * 1024
                || self
                    .memory_and_disk
                    .raw_bytes_per_reference_profile_or_lossless_native_capture
                    != RAW_BYTES
                || self.cases.len() != 10
            {
                return Err("frozen Granite protocol contract differs".into());
            }
            let mut names = BTreeSet::new();
            let mut raw = 0_u64;
            for case in &self.cases {
                if case.name.is_empty()
                    || case.name.len() > 40
                    || !case
                        .name
                        .bytes()
                        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
                    || !names.insert(&case.name)
                    || case.tokens.is_empty()
                    || case.tokens.len() > 512
                    || case
                        .offset
                        .checked_add(case.tokens.len())
                        .is_none_or(|n| n > 512)
                    || case.tokens.iter().any(|t| *t as usize >= VOCAB)
                {
                    return Err("ten unique bounded frozen token cases required".into());
                }
                raw = raw
                    .checked_add(case.tokens.len() as u64 * bytes_per_token())
                    .ok_or("raw reservation overflow")?;
            }
            if raw != RAW_BYTES {
                return Err("full raw stage reservation differs".into());
            }
            Ok(())
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
    struct Stamp {
        device: u64,
        inode: u64,
        len: u64,
        mtime: i64,
        mtime_nsec: i64,
        ctime: i64,
        ctime_nsec: i64,
        mode: u32,
        uid: u32,
        gid: u32,
        nlink: u64,
    }
    impl From<Metadata> for Stamp {
        fn from(m: Metadata) -> Self {
            Self {
                device: m.dev(),
                inode: m.ino(),
                len: m.len(),
                mtime: m.mtime(),
                mtime_nsec: m.mtime_nsec(),
                ctime: m.ctime(),
                ctime_nsec: m.ctime_nsec(),
                mode: m.mode(),
                uid: m.uid(),
                gid: m.gid(),
                nlink: m.nlink(),
            }
        }
    }
    struct StableFile {
        path: PathBuf,
        file: File,
        stamp: Stamp,
    }
    impl StableFile {
        fn open(path: &Path, maximum: u64, directory: bool) -> Result<Self> {
            let named = fs::symlink_metadata(path)?;
            if (directory && !named.is_dir())
                || (!directory && (!named.is_file() || named.len() == 0 || named.len() > maximum))
            {
                return Err("bounded regular file or real source directory required".into());
            }
            let file = OpenOptions::new()
                .read(true)
                .custom_flags(
                    libc::O_NOFOLLOW
                        | libc::O_CLOEXEC
                        | if directory { libc::O_DIRECTORY } else { 0 },
                )
                .open(path)?;
            let stamp = Stamp::from(named);
            if Stamp::from(file.metadata()?) != stamp {
                return Err("named/descriptor source identity differs".into());
            }
            Ok(Self {
                path: path.into(),
                file,
                stamp,
            })
        }
        fn verify(&self) -> Result<()> {
            if Stamp::from(fs::symlink_metadata(&self.path)?) != self.stamp
                || Stamp::from(self.file.metadata()?) != self.stamp
            {
                return Err("source identity/metadata changed".into());
            }
            Ok(())
        }
        fn hash(&self) -> Result<String> {
            use std::os::unix::fs::FileExt;
            self.verify()?;
            let mut hash = Sha256::new();
            let mut block = [0; 64 * 1024];
            let mut at = 0;
            while at < self.stamp.len {
                let n = block.len().min((self.stamp.len - at) as usize);
                self.file.read_exact_at(&mut block[..n], at)?;
                hash.update(&block[..n]);
                at += n as u64;
            }
            if self.file.read_at(&mut block[..1], at)? != 0 {
                return Err("source grew beyond authenticated extent".into());
            }
            self.verify()?;
            Ok(format!("{:x}", hash.finalize()))
        }
        fn bytes(&self, sha: &str) -> Result<Vec<u8>> {
            self.verify()?;
            let mut bytes = Vec::with_capacity(self.stamp.len as usize);
            (&self.file)
                .take(self.stamp.len + 1)
                .read_to_end(&mut bytes)?;
            self.verify()?;
            if bytes.len() as u64 != self.stamp.len
                || format!("{:x}", Sha256::digest(&bytes)) != sha
            {
                return Err("externally pinned bounded input differs".into());
            }
            Ok(bytes)
        }
    }
    struct Source {
        directory: StableFile,
        data: StableFile,
        config: StableFile,
    }
    impl Source {
        fn open(path: &Path) -> Result<Self> {
            let source = Self {
                directory: StableFile::open(path, 0, true)?,
                data: StableFile::open(
                    &path.join("model.safetensors"),
                    4 * 1024 * 1024 * 1024,
                    false,
                )?,
                config: StableFile::open(&path.join("config.json"), 1024 * 1024, false)?,
            };
            source.verify()?;
            Ok(source)
        }
        fn verify(&self) -> Result<()> {
            self.directory.verify()?;
            self.data.verify()?;
            self.config.verify()?;
            match fs::symlink_metadata(self.directory.path.join("model.safetensors.index.json")) {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                _ => Err(
                    "exact converted single-file source must have no index; no repair is permitted"
                        .into(),
                ),
            }
        }
        fn evidence(&self) -> Result<Value> {
            self.verify()?;
            let data = self.data.hash()?;
            let config = self.config.hash()?;
            self.verify()?;
            if data != DATA_SHA || config != CONFIG_SHA {
                return Err("whole converted source SHA256 differs".into());
            }
            Ok(
                json!({"directory":self.directory.path,"directory_metadata":self.directory.stamp,
                "data":{"name":"model.safetensors","sha256":data,"metadata":self.data.stamp},
                "config":{"name":"config.json","sha256":config,"metadata":self.config.stamp},
                "index_absent":true,"named_and_retained_descriptor_metadata_equal":true,
                "scope":"Historical whole-source and stable named/descriptor observations; no managed grant"}),
            )
        }
    }

    fn specs() -> BTreeMap<String, (usize, Dtype)> {
        let mut all = BTreeMap::from([
            ("embedding".into(), (2560, Dtype::Half)),
            ("final_norm".into(), (2560, Dtype::Half)),
            ("logits".into(), (VOCAB, Dtype::Float)),
        ]);
        for layer in 0..LAYERS {
            for stage in STAGES {
                let columns = match stage {
                    "k" | "v" | "k_rotary" => 512,
                    "gate" | "silu" | "up" | "gated" => 8192,
                    _ => 2560,
                };
                all.insert(format!("layers.{layer}.{stage}"), (columns, Dtype::Half));
            }
        }
        all
    }
    fn bytes_per_token() -> u64 {
        specs()
            .values()
            .map(|(width, dtype)| *width as u64 * if matches!(dtype, Dtype::Float) { 4 } else { 2 })
            .sum()
    }
    fn make<'m>(
        model: &'m metal::Model,
        case: &Case,
        requested: usize,
        readback: metal::ReadbackMode,
    ) -> Result<metal::Session<'m>> {
        Ok(metal::Session::with_attention_mode(
            model,
            session::Options {
                capacity: 512 - case.offset,
                position_offset: case.offset,
                prefill_chunk_size: requested.min(512 - case.offset),
                memory_limit_bytes: BUDGET,
            },
            metal::ProjectionMode::MlxAffine,
            metal::NormalizationMode::Compensated,
            readback,
            metal::AttentionMode::Reference,
        )?)
    }
    fn row(s: &metal::Session<'_>) -> Result<Vec<f32>> {
        let values = s
            .logits()
            .ok_or("completed full-logit distribution absent")?;
        if values.len() != VOCAB || values.iter().any(|x| !x.is_finite()) {
            return Err("complete finite vocabulary row required".into());
        }
        Ok(values.to_vec())
    }
    fn same(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
    }
    fn row_hash(values: &[f32]) -> String {
        let mut hash = Sha256::new();
        for value in values {
            hash.update(value.to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }
    fn expected_chunks(length: usize, chunk: usize) -> Vec<usize> {
        (0..length)
            .step_by(chunk)
            .map(|at| chunk.min(length - at))
            .collect()
    }
    #[derive(Serialize)]
    struct Trace {
        folder: String,
        baseline_folder: String,
        effective_chunk: usize,
        effective_chunks: Vec<usize>,
        arrays: BTreeMap<String, Array>,
        final_logits_sha256: String,
        final_greedy_token: u32,
        diagnostic_last_row_matches_logits: bool,
    }
    impl Trace {
        fn helper_charge(&self) -> usize {
            self.arrays
                .values()
                .map(|array| 512 + 512 * array.patches.len())
                .sum()
        }
    }
    struct CaptureOutput<'a> {
        root: &'a Path,
        folder: &'a str,
        baseline_folder: &'a str,
        repeat: bool,
        helper_available: usize,
    }
    fn capture(
        s: &mut metal::Session<'_>,
        case: &Case,
        chunk: usize,
        output: CaptureOutput<'_>,
    ) -> Result<(Trace, Vec<f32>)> {
        let CaptureOutput {
            root,
            folder,
            baseline_folder,
            repeat,
            helper_available,
        } = output;
        let path = root.join(folder);
        fs::create_dir_all(path.parent().ok_or("output parent absent")?)?;
        fs::create_dir(&path)?;
        let baseline_path = root.join(baseline_folder);
        let baseline = repeat.then_some(baseline_path.as_path());
        let expected = specs();
        let mut arrays: BTreeMap<_, _> = expected
            .iter()
            .map(|(name, (columns, dtype))| {
                Ok((
                    name.clone(),
                    Stage::new(name, case.tokens.len(), *columns, *dtype)?,
                ))
            })
            .collect::<Result<_>>()?;
        let mut chunks = Vec::new();
        let mut committed = 0;
        let mut last = Vec::new();
        let mut helper_charge = 512 * arrays.len();
        if helper_charge > helper_available {
            return Err("diagnostic metadata exceeds helper reservation".into());
        }
        loop {
            let rows = s.evaluate_next_chunk()?;
            if rows == 0 {
                break;
            }
            if committed + rows > case.tokens.len() {
                return Err("runtime committed rows beyond the case".into());
            }
            let diagnostics = s.diagnostics()?;
            if diagnostics.len() != 643 {
                return Err("all 643 native-visible diagnostic stages required".into());
            }
            let mut names = BTreeSet::new();
            for diagnostic in diagnostics {
                let name = match diagnostic.layer {
                    Some(layer) => format!("layers.{layer}.{}", diagnostic.stage),
                    None => diagnostic.stage.into(),
                };
                let (columns, _) = expected.get(&name).ok_or("unexpected diagnostic stage")?;
                if !names.insert(name.clone())
                    || diagnostic.rows != rows
                    || diagnostic.columns != *columns
                {
                    return Err("duplicate or incorrectly shaped diagnostic stage".into());
                }
                if name == "logits" {
                    last.clear();
                    last.extend_from_slice(&diagnostic.values[(rows - 1) * VOCAB..]);
                }
                let stage = arrays.get_mut(&name).ok_or("diagnostic exporter absent")?;
                let prior_patches = stage.patch_count();
                stage.push(&diagnostic.values, rows, &path, baseline)?;
                helper_charge += 512 * (stage.patch_count() - prior_patches);
                if helper_charge > helper_available {
                    return Err("fragmented mismatch metadata exceeds the admitted 32 MiB helper reservation; partial payloads retained".into());
                }
            }
            committed += rows;
            chunks.push(rows);
            println!(
                "{}",
                json!({"event":"native_chunk","case":case.name,"folder":folder,
                "rows":rows,"committed":committed,"effective_chunk":chunk,"memory":s.memory()})
            );
        }
        if committed != case.tokens.len() || s.evaluated_len() != committed || s.pending_len() != 0
        {
            return Err("incomplete committed diagnostic case".into());
        }
        let final_logits = row(s)?;
        let last_matches = same(&last, &final_logits);
        let arrays = arrays
            .into_iter()
            .map(|(name, stage)| Ok((name, stage.finish(&path, baseline)?)))
            .collect::<Result<_>>()?;
        Ok((
            Trace {
                folder: folder.into(),
                baseline_folder: baseline_folder.into(),
                effective_chunk: chunk,
                effective_chunks: chunks,
                arrays,
                final_logits_sha256: row_hash(&final_logits),
                final_greedy_token: sampling::greedy(&final_logits)?,
                diagnostic_last_row_matches_logits: last_matches,
            },
            final_logits,
        ))
    }

    #[derive(Serialize)]
    struct NativeCase {
        name: String,
        requested_chunk: usize,
        position_offset: usize,
        tokens: Vec<u32>,
        #[serde(flatten)]
        trace: Trace,
        state_checks: BTreeMap<String, bool>,
        state_evidence: Value,
        memory: session::Memory,
        #[serde(skip_serializing_if = "Option::is_none")]
        reset_replay: Option<Trace>,
    }
    impl NativeCase {
        fn helper_charge(&self) -> usize {
            8192 + self.trace.helper_charge()
                + self.reset_replay.as_ref().map_or(0, Trace::helper_charge)
        }
    }
    fn simple_evaluate(s: &mut metal::Session<'_>, length: usize) -> Result<Vec<usize>> {
        let mut chunks = Vec::new();
        let mut total = 0;
        loop {
            let rows = s.evaluate_next_chunk()?;
            if rows == 0 {
                break;
            }
            total += rows;
            if total > length {
                return Err("state evaluation exceeded bounded history".into());
            }
            chunks.push(rows);
        }
        if total != length {
            return Err("state evaluation ended before the full history".into());
        }
        // The caller retains an exact schedule comparison as a state result.
        Ok(chunks)
    }
    fn run_case(
        model: &metal::Model,
        case: &Case,
        requested: usize,
        root: &Path,
        baseline_hashes: Option<&BTreeMap<String, String>>,
        helper_available: usize,
    ) -> Result<NativeCase> {
        let chunk = requested.min(512 - case.offset);
        let folder = format!("{}/chunk{requested}", case.name);
        let baseline_folder = format!("{}/chunk128", case.name);
        let mut s = make(model, case, requested, metal::ReadbackMode::FullLogits)?;
        s.enable_diagnostics(&(0..LAYERS).collect::<Vec<_>>())?;
        let memory = s.memory();
        if memory.admitted_bytes > BUDGET {
            return Err("diagnostic admission exceeded 6 GiB".into());
        }
        s.set_tokens(&case.tokens)?;
        let (trace, before) = capture(
            &mut s,
            case,
            chunk,
            CaptureOutput {
                root,
                folder: &folder,
                baseline_folder: &baseline_folder,
                repeat: requested != 128,
                helper_available,
            },
        )?;
        let mut checks = BTreeMap::from([
            (
                "exact_chunk_schedule".into(),
                trace.effective_chunks == expected_chunks(case.tokens.len(), chunk),
            ),
            (
                "diagnostic_last_row_matches_logits".into(),
                trace.diagnostic_last_row_matches_logits,
            ),
            (
                "full_logit_greedy_selection".into(),
                s.greedy_token().transpose()? == Some(trace.final_greedy_token),
            ),
            (
                "schedule_all_stage_bytes_equal".into(),
                trace.arrays.iter().all(|(name, a)| {
                    a.mismatching_bytes == 0
                        && baseline_hashes.is_none_or(|h| h.get(name) == Some(&a.sha256))
                }),
            ),
        ]);
        s.set_tokens(&case.tokens)?;
        let no_work = s.evaluate_next_chunk()?;
        checks.insert(
            "fully_cached_prompt_zero_work_unchanged_logits".into(),
            no_work == 0
                && s.history() == case.tokens
                && s.evaluated_len() == case.tokens.len()
                && s.logits().is_some_and(|values| same(values, &before)),
        );
        let overflow_append = s
            .append(&vec![1; s.capacity() - case.tokens.len() + 1])
            .is_err();
        let overflow_set = s.set_tokens(&vec![1; s.capacity() + 1]).is_err();
        let invalid_id = s.append(&[u32::MAX]).is_err();
        checks.insert(
            "capacity_rejection_preserves_prefix".into(),
            overflow_append
                && overflow_set
                && invalid_id
                && s.history() == case.tokens
                && s.evaluated_len() == case.tokens.len()
                && s.logits().is_some_and(|v| same(v, &before)),
        );
        // Restore the original requested history if a state observation failed;
        // diagnostics from the primary numerical capture are already retained.
        s.set_tokens(&case.tokens)?;
        let pending_applicable = case.tokens.len() < s.capacity();
        if pending_applicable {
            s.append(&[trace.final_greedy_token])?;
            let pending = s.pending_len() == 1
                && s.evaluated_len() == case.tokens.len()
                && s.history()[case.tokens.len()] == trace.final_greedy_token
                && s.logits().is_none()
                && s.greedy_token().is_none();
            s.discard_pending();
            checks.insert(
                "pending_sampled_token_is_uncommitted_and_discardable".into(),
                pending
                    && s.history() == case.tokens
                    && s.pending_len() == 0
                    && s.logits().is_some_and(|values| same(values, &before)),
            );
        }
        s.reset();
        checks.insert(
            "reset_clears_history_KV_prefix_and_distribution".into(),
            s.history().is_empty()
                && s.evaluated_len() == 0
                && s.pending_len() == 0
                && s.logits().is_none()
                && s.greedy_token().is_none(),
        );
        s.set_tokens(&case.tokens)?;
        let (reset_replay, replay_chunks, replay_logits) = if requested == 128 {
            let replay_folder = format!("{folder}/reset_replay");
            let (replay, values) = capture(
                &mut s,
                case,
                chunk,
                CaptureOutput {
                    root,
                    folder: &replay_folder,
                    baseline_folder: &baseline_folder,
                    repeat: true,
                    helper_available: helper_available.saturating_sub(trace.helper_charge()),
                },
            )?;
            checks.insert(
                "reset_replay_all_stage_bytes_equal".into(),
                replay.arrays.iter().all(|(name, a)| {
                    a.mismatching_bytes == 0
                        && trace.arrays.get(name).is_some_and(|b| a.sha256 == b.sha256)
                }),
            );
            let chunks = replay.effective_chunks.clone();
            (Some(replay), chunks, values)
        } else {
            let chunks = simple_evaluate(&mut s, case.tokens.len())?;
            (None, chunks, row(&s)?)
        };
        checks.insert(
            "reset_replay_exact_schedule_and_final_logits".into(),
            replay_chunks == expected_chunks(case.tokens.len(), chunk)
                && same(&before, &replay_logits),
        );
        // Concrete longest-prefix edit on the frozen short/future-changed pair.
        // Compare reused and freshly built KV, holding one session at a time.
        let changed_prefix = if case.name == "short" {
            let mut changed = case.tokens.clone();
            *changed.last_mut().ok_or("empty short case")? = 17;
            s.set_tokens(&changed)?;
            let retained = s.evaluated_len();
            let changed_chunks = simple_evaluate(&mut s, 1)?;
            let reused = row(&s)?;
            drop(s);
            let mut fresh = make(model, case, requested, metal::ReadbackMode::FullLogits)?;
            fresh.set_tokens(&changed)?;
            let fresh_chunks = simple_evaluate(&mut fresh, changed.len())?;
            let fresh_values = row(&fresh)?;
            checks.insert(
                "changed_prefix_replay_equals_fresh_session".into(),
                retained == 3
                    && changed_chunks == [1]
                    && fresh_chunks == expected_chunks(changed.len(), chunk)
                    && same(&reused, &fresh_values),
            );
            Some(
                json!({"tokens":changed,"retained_committed_prefix":retained,
                "reused_logits_sha256":row_hash(&reused),"fresh_logits_sha256":row_hash(&fresh_values)}),
            )
        } else {
            drop(s);
            None
        };
        let mut greedy = make(model, case, requested, metal::ReadbackMode::GreedyToken)?;
        greedy.set_tokens(&case.tokens)?;
        let greedy_chunks = simple_evaluate(&mut greedy, case.tokens.len())?;
        let selected = greedy.greedy_token().transpose()?;
        checks.insert(
            "greedy_token_readback_equals_full_logit_selection".into(),
            selected == Some(trace.final_greedy_token)
                && greedy.logits().is_none()
                && greedy_chunks == expected_chunks(case.tokens.len(), chunk),
        );
        let greedy_memory = greedy.memory();
        drop(greedy);
        Ok(NativeCase {
            name: case.name.clone(),
            requested_chunk: requested,
            position_offset: case.offset,
            tokens: case.tokens.clone(),
            trace,
            state_checks: checks,
            state_evidence: json!({"pending_sampled_token_applicable":pending_applicable,
                "pending_inapplicable_reason":if pending_applicable { Value::Null } else { json!("full admitted capacity") },
                "reset_replay_chunks":replay_chunks,"reset_replay_final_logits_sha256":row_hash(&replay_logits),
                "changed_prefix":changed_prefix,"greedy_token":selected,"greedy_memory":greedy_memory,
                "session_ownership":"Serial borrowed sessions; one immutable model counted once",
                "cancellation_or_device_failure_injection":"Unavailable public API; not claimed"}),
            memory,
            reset_replay,
        })
    }

    #[derive(Serialize)]
    struct Manifest {
        format: &'static str,
        protocol_sha256: &'static str,
        model_data_sha256: &'static str,
        model_config_sha256: &'static str,
        complete: bool,
        state_checks_passed: bool,
        graph_qualified: bool,
        quality_qualified: bool,
        performance_qualified: bool,
        managed_adoption_qualified: bool,
        cases: Vec<NativeCase>,
        source_before: Option<Value>,
        source_after: Option<Value>,
        runtime: Value,
        error: Option<String>,
    }
    impl Manifest {
        fn new() -> Self {
            Self {
                format: "toklet-granite-native-graph-v1",
                protocol_sha256: PROTOCOL_SHA,
                model_data_sha256: DATA_SHA,
                model_config_sha256: CONFIG_SHA,
                complete: false,
                state_checks_passed: false,
                graph_qualified: false,
                quality_qualified: false,
                performance_qualified: false,
                managed_adoption_qualified: false,
                cases: Vec::new(),
                source_before: None,
                source_after: None,
                runtime: json!({}),
                error: None,
            }
        }
        fn save(&self, output: &Path) -> Result<()> {
            io::ensure_disk_space(output, HELPER as u64)?;
            let temporary = output.join("manifest.partial.json");
            let mut file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&temporary)?;
            let mut writer = BufWriter::with_capacity(64 * 1024, &mut file);
            serde_json::to_writer_pretty(&mut writer, self)?;
            writer.write_all(b"\n")?;
            writer.flush()?;
            drop(writer);
            file.sync_all()?;
            fs::rename(temporary, output.join("manifest.json"))?;
            Ok(())
        }
    }
    fn collect(
        model_path: &Path,
        protocol_path: &Path,
        output: &Path,
        report: &mut Manifest,
    ) -> Result<()> {
        let input = StableFile::open(protocol_path, 2 * 1024 * 1024, false)?;
        let protocol: Protocol = serde_json::from_slice(&input.bytes(PROTOCOL_SHA)?)?;
        protocol.validate()?;
        let source = Source::open(model_path)?;
        report.source_before = Some(source.evidence()?);
        io::ensure_disk_space(output, RAW_BYTES + HELPER as u64)?;
        let binary = std::env::current_exe()?;
        report.runtime = json!({"binary":binary,"binary_sha256":io::hash_file(&binary)?,
            "worker_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("granite_graph_capture.rs"))),
            "io_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("support/granite_graph_io.rs"))),
            "engine_revision":option_env!("TOKLET_ENGINE_REVISION").unwrap_or("unrecorded"),
            "engine_limit_bytes":BUDGET,"helper_reserve_bytes":HELPER,"disk_floor_bytes":2_u64*1024*1024*1024,
            "ordinary_context_limit":512,"immutable_model_loads":1,"simultaneous_sessions":1,
            "protocol_metadata":input.stamp,"target":"Apple Silicon macOS Metal",
            "projection":"MlxAffine","normalization":"Compensated","attention":"Reference"});
        report.save(output)?;
        let model = metal::Model::load_mlx(
            model_path,
            weights::LoadOptions {
                context_limit: 512,
                memory_limit_bytes: BUDGET,
            },
        )?;
        source.verify()?;
        let c = model.config();
        if model.mlx_architecture() != Some(toklet::mlx::dense::Architecture::Granite)
            || (
                c.hidden_size,
                c.intermediate_size,
                c.num_layers,
                c.vocab_size,
                c.query_heads,
                c.kv_heads,
                c.head_dim,
                c.max_positions,
            ) != (2560, 8192, LAYERS, VOCAB, 40, 8, 64, 512)
            || c.rope_theta.to_bits() != 10_000_000_f32.to_bits()
            || c.rms_norm_eps.to_bits() != 1e-5_f32.to_bits()
        {
            return Err("loaded Granite model geometry differs".into());
        }
        report.runtime["device"] = json!(model.device_name());
        report.runtime["model_resident_bytes"] = json!(model.model_resident_bytes()?);
        for case in &protocol.cases {
            let mut baseline_hashes = None;
            for &requested in &protocol.native.chunks {
                source.verify()?;
                input.verify()?;
                println!(
                    "{}",
                    json!({"event":"native_case_start","case":case.name,"requested_chunk":requested})
                );
                let retained: usize = report.cases.iter().map(NativeCase::helper_charge).sum();
                let available = HELPER
                    .checked_sub(FIXED_HELPER + retained + 8192)
                    .ok_or("retained metadata exceeds the 32 MiB helper reservation")?;
                let result = run_case(
                    &model,
                    case,
                    requested,
                    output,
                    baseline_hashes.as_ref(),
                    available,
                )?;
                if requested == 128 {
                    baseline_hashes = Some(
                        result
                            .trace
                            .arrays
                            .iter()
                            .map(|(n, a)| (n.clone(), a.sha256.clone()))
                            .collect(),
                    );
                }
                report.cases.push(result);
                report.save(output)?;
            }
        }
        report.source_after = Some(source.evidence()?);
        input.verify()?;
        report.complete = report.cases.len() == 30;
        report.state_checks_passed = report.complete
            && report
                .cases
                .iter()
                .all(|c| c.state_checks.values().all(|passed| *passed));
        Ok(())
    }
    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args_os().collect();
        if args.len() != 5 || args[3].to_str() != Some(PROTOCOL_SHA) {
            return Err("usage: granite_graph_capture MODEL_DIR FROZEN_PROTOCOL_JSON EXACT_PROTOCOL_SHA256 NEW_OUTPUT_DIR".into());
        }
        let model_path = Path::new(&args[1]);
        if !fs::symlink_metadata(model_path)?.is_dir() {
            return Err("real source directory required".into());
        }
        let model_path = model_path.canonicalize()?;
        let protocol_path = Path::new(&args[2]).to_path_buf();
        let output = Path::new(&args[4]);
        let parent = output
            .parent()
            .ok_or("output parent required")?
            .canonicalize()?;
        if parent.starts_with(&model_path) {
            return Err("output must be outside the immutable source directory".into());
        }
        fs::create_dir(output)?;
        let output = output.canonicalize()?;
        let mut report = Manifest::new();
        let result = collect(&model_path, &protocol_path, &output, &mut report);
        if let Err(error) = &result {
            report.error = Some(error.to_string());
        }
        report.save(&output)?;
        result?;
        if !report.state_checks_passed {
            return Err("all captures retained; exact state/schedule observations differ and remain unqualified".into());
        }
        println!(
            "{}",
            json!({"event":"native_capture_complete","cases":report.cases.len(),
            "state_checks_passed":report.state_checks_passed,"graph_qualified":false,"manifest":output.join("manifest.json")})
        );
        Ok(())
    }
}

fn main() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Err(error) = native::run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        eprintln!("Granite graph capture requires Apple Silicon macOS Metal");
        std::process::exit(1);
    }
}
