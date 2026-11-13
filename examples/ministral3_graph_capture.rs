//! Bounded, explicit full-model diagnostics. Raw captures require independent
//! comparison; this executable does not qualify a model or application route.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod native {
    use half::f16;
    use serde::{Deserialize, Serialize};
    use sha2::{Digest, Sha256};
    use std::{
        collections::BTreeMap,
        fs::{self, File, Metadata, OpenOptions},
        io::Write,
        mem::size_of,
        os::unix::{
            ffi::OsStrExt,
            fs::{FileExt, MetadataExt, OpenOptionsExt},
        },
        path::{Path, PathBuf},
        time::Instant,
    };
    use toklet::{metal, session, weights};

    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    const PROTOCOL_SHA: &str = "1e38c54ca03d6eddb73578cd2e12a7ea14bc9f6d875f397645558be750429932";
    const DATA_SHA: &str = "fe0acffd3bcac67f1c4bcf4e396f310dba94687e24c2a6a798e288ef48f4f7be";
    const CONFIG_SHA: &str = "0adbb3ce9e2c6c3fd050e2aa795ca9e92e839e939defb9721965e6bb20522f2d";
    const DATA_BYTES: u64 = 2_662_082_312;
    const BUDGET: usize = 6 * 1024 * 1024 * 1024;
    const HELPER: usize = 32 * 1024 * 1024;
    // Cold input parsing, hash/write buffers, path/receipt state and allocator
    // slack. Diagnostic Vec capacities and retained array records are additional.
    const FIXED_HELPER: usize = 4 * 1024 * 1024;
    const LAYERS: usize = 26;
    const VOCAB: usize = 131_072;
    const CAPACITY: usize = 512;
    const STAGES: [(&str, usize); 16] = [
        ("input_norm", 3072),
        ("q", 4096),
        ("k", 1024),
        ("v", 1024),
        ("q_rotary", 4096),
        ("k_rotary", 1024),
        ("attention_context", 4096),
        ("attention_projection", 3072),
        ("attention_residual", 3072),
        ("post_attention_norm", 3072),
        ("gate", 9216),
        ("silu", 9216),
        ("up", 9216),
        ("gated", 9216),
        ("down", 3072),
        ("output", 3072),
    ];

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Protocol {
        format: String,
        model: ModelPins,
        cases: Vec<Case>,
        chunks: Vec<usize>,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ModelPins {
        data_sha256: String,
        config_sha256: String,
    }
    #[derive(Clone, Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct Case {
        name: String,
        tokens: Vec<u32>,
        offset: usize,
    }
    #[derive(Deserialize)]
    struct Build {
        binary: BinaryPin,
        capture_source: SourcePin,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct BinaryPin {
        path: PathBuf,
        bytes: u64,
        sha256: String,
    }
    #[derive(Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct SourcePin {
        bytes: usize,
        sha256: String,
    }
    struct Args {
        model: PathBuf,
        protocol: PathBuf,
        protocol_sha: String,
        build: PathBuf,
        build_sha: String,
        output: PathBuf,
    }
    fn args() -> Result<Args> {
        let mut values = BTreeMap::new();
        let mut args = std::env::args_os().skip(1);
        while let Some(key) = args.next() {
            let key = key.into_string().map_err(|_| "non-UTF8 flag")?;
            if ![
                "--model",
                "--protocol",
                "--protocol-sha256",
                "--build-manifest",
                "--build-manifest-sha256",
                "--fresh-output",
            ]
            .contains(&key.as_str())
            {
                return Err(format!("unknown flag: {key}").into());
            }
            if values
                .insert(key, args.next().ok_or("missing flag value")?)
                .is_some()
            {
                return Err("duplicate flag".into());
            }
        }
        let mut take = |key: &str| values.remove(key).ok_or_else(|| format!("missing {key}"));
        Ok(Args {
            model: take("--model")?.into(),
            protocol: take("--protocol")?.into(),
            protocol_sha: take("--protocol-sha256")?
                .into_string()
                .map_err(|_| "non-UTF8 hash")?,
            build: take("--build-manifest")?.into(),
            build_sha: take("--build-manifest-sha256")?
                .into_string()
                .map_err(|_| "non-UTF8 hash")?,
            output: take("--fresh-output")?.into(),
        })
    }
    fn need(ok: bool, message: &str) -> Result<()> {
        if ok { Ok(()) } else { Err(message.into()) }
    }
    fn hash(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
    struct Identity {
        dev: u64,
        ino: u64,
        bytes: u64,
        mtime: i64,
        mtime_nsec: i64,
        ctime: i64,
        ctime_nsec: i64,
        mode: u32,
        nlink: u64,
    }
    impl Identity {
        fn of(m: &Metadata) -> Self {
            Self {
                dev: m.dev(),
                ino: m.ino(),
                bytes: m.len(),
                mtime: m.mtime(),
                mtime_nsec: m.mtime_nsec(),
                ctime: m.ctime(),
                ctime_nsec: m.ctime_nsec(),
                mode: m.mode(),
                nlink: m.nlink(),
            }
        }
        fn same_object(&self, other: &Self) -> bool {
            self.dev == other.dev && self.ino == other.ino && self.mode == other.mode
        }
    }
    #[derive(Serialize)]
    struct Receipt {
        path: PathBuf,
        bytes: u64,
        sha256: String,
        before_identity: Identity,
        after_identity: Identity,
        after_sha256: Option<String>,
    }
    struct HeldFile {
        path: PathBuf,
        file: File,
        before: Identity,
    }
    impl HeldFile {
        fn open(path: &Path, limit: u64) -> Result<Self> {
            let before = fs::symlink_metadata(path)?;
            need(
                before.is_file() && before.len() > 0 && before.len() <= limit,
                "input must be a bounded, nonempty regular file",
            )?;
            let file = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
                .open(path)?;
            let held = Self {
                path: path.to_path_buf(),
                file,
                before: Identity::of(&before),
            };
            held.verify()?;
            Ok(held)
        }
        fn verify(&self) -> Result<Identity> {
            let path = fs::symlink_metadata(&self.path)?;
            let descriptor = self.file.metadata()?;
            need(
                path.is_file()
                    && descriptor.is_file()
                    && Identity::of(&path) == self.before
                    && Identity::of(&descriptor) == self.before,
                "input descriptor/path identity changed",
            )?;
            Ok(Identity::of(&descriptor))
        }
        fn sha(&self) -> Result<String> {
            self.verify()?;
            let result = hash_range(&self.file, 0, self.before.bytes)?;
            self.verify()?;
            Ok(result)
        }
        fn read(&self) -> Result<Vec<u8>> {
            self.verify()?;
            let mut bytes = vec![0; usize::try_from(self.before.bytes)?];
            self.file.read_exact_at(&mut bytes, 0)?;
            self.verify()?;
            Ok(bytes)
        }
        fn receipt(&self, sha256: String) -> Result<Receipt> {
            Ok(Receipt {
                path: self.path.clone(),
                bytes: self.before.bytes,
                sha256,
                before_identity: self.before.clone(),
                after_identity: self.verify()?,
                after_sha256: None,
            })
        }
        fn finish(&self, receipt: &mut Receipt) -> Result<()> {
            let after = self.sha()?;
            need(after == receipt.sha256, "input bytes changed at endpoint")?;
            receipt.after_identity = self.verify()?;
            receipt.after_sha256 = Some(after);
            Ok(())
        }
    }
    fn hash_range(file: &File, mut offset: u64, mut bytes: u64) -> Result<String> {
        offset.checked_add(bytes).ok_or("file extent overflow")?;
        let mut hash = Sha256::new();
        let mut block = [0_u8; 64 * 1024];
        while bytes > 0 {
            let n = bytes.min(block.len() as u64) as usize;
            file.read_exact_at(&mut block[..n], offset)?;
            hash.update(&block[..n]);
            offset += n as u64;
            bytes -= n as u64;
        }
        Ok(format!("{:x}", hash.finalize()))
    }
    fn directory(path: &Path) -> Result<(File, Identity)> {
        need(
            fs::symlink_metadata(path)?.is_dir(),
            "expected real directory",
        )?;
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)?;
        let identity = Identity::of(&file.metadata()?);
        need(
            identity == Identity::of(&fs::symlink_metadata(path)?),
            "directory identity mismatch",
        )?;
        Ok((file, identity))
    }
    fn verify_directory(path: &Path, file: &File, before: &Identity) -> Result<()> {
        let path_meta = fs::symlink_metadata(path)?;
        need(
            path_meta.is_dir()
                && before.same_object(&Identity::of(&path_meta))
                && before.same_object(&Identity::of(&file.metadata()?)),
            "directory replaced",
        )
    }
    fn disk(folder: &Path, additional: u64) -> Result<()> {
        let path = std::ffi::CString::new(folder.as_os_str().as_bytes())?;
        let mut status = std::mem::MaybeUninit::<libc::statvfs>::uninit();
        // SAFETY: live NUL-terminated path and writable statvfs storage. The
        // initialized value is read only after the successful system call.
        if unsafe { libc::statvfs(path.as_ptr(), status.as_mut_ptr()) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let status = unsafe { status.assume_init() };
        need(
            u128::from(status.f_bavail) * u128::from(status.f_frsize)
                >= u128::from(additional) + 2 * 1024 * 1024 * 1024,
            "capture would cross the 2 GiB disk floor",
        )
    }
    fn write_new(root: &Path, path: &Path, bytes: &[u8]) -> Result<()> {
        disk(root, bytes.len() as u64)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(())
    }

    struct Spec {
        layer: Option<usize>,
        stage: &'static str,
        columns: usize,
    }
    impl Spec {
        fn name(&self) -> String {
            self.layer.map_or_else(
                || self.stage.to_owned(),
                |l| format!("layer{l:02}.{}", self.stage),
            )
        }
        fn float(&self) -> bool {
            self.layer.is_none() && self.stage == "logits"
        }
    }
    fn specs() -> Vec<Spec> {
        let mut out = Vec::with_capacity(419);
        for (stage, columns) in [("embedding", 3072), ("final_norm", 3072), ("logits", VOCAB)] {
            out.push(Spec {
                layer: None,
                stage,
                columns,
            });
        }
        for layer in 0..LAYERS {
            for (stage, columns) in STAGES {
                out.push(Spec {
                    layer: Some(layer),
                    stage,
                    columns,
                });
            }
        }
        out
    }
    #[derive(Serialize)]
    struct Array {
        name: String,
        layer: Option<usize>,
        stage: &'static str,
        path: String,
        dtype: &'static str,
        shape: [usize; 2],
        bytes: u64,
        sha256: String,
    }
    impl Array {
        fn heap(&self) -> usize {
            self.name.capacity() + self.path.capacity() + self.sha256.capacity()
        }
    }
    struct Writer {
        array: Array,
        identity: Option<Identity>,
        hash: Sha256,
        written: u64,
    }
    impl Writer {
        fn append(&mut self, root: &Path, values: &[f32]) -> Result<()> {
            let float = self.array.dtype == "<f4";
            need(
                values.iter().all(|v| {
                    v.is_finite() && (float || f16::from_f32(*v).to_f32().to_bits() == v.to_bits())
                }),
                "snapshot is nonfinite or not losslessly representable in its storage dtype",
            )?;
            let bytes = values
                .len()
                .checked_mul(if float { 4 } else { 2 })
                .ok_or("array size overflow")?;
            need(
                self.written
                    .checked_add(bytes as u64)
                    .is_some_and(|n| n <= self.array.bytes),
                "array row count overflow",
            )?;
            disk(root, bytes as u64)?;
            let path = root.join(&self.array.path);
            let mut options = OpenOptions::new();
            options
                .write(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
            if self.identity.is_none() {
                options.create_new(true);
            } else {
                options.append(true);
            }
            let mut file = options.open(&path)?;
            let identity = Identity::of(&file.metadata()?);
            need(
                file.metadata()?.is_file()
                    && identity.bytes == self.written
                    && self
                        .identity
                        .as_ref()
                        .is_none_or(|old| old.same_object(&identity)),
                "output descriptor replaced or resized",
            )?;
            self.identity = Some(identity);
            let mut block = [0_u8; 64 * 1024];
            let width = if float { 4 } else { 2 };
            for values in values.chunks(block.len() / width) {
                for (value, slot) in values.iter().zip(block.chunks_exact_mut(width)) {
                    if float {
                        slot.copy_from_slice(&value.to_le_bytes());
                    } else {
                        slot.copy_from_slice(&f16::from_f32(*value).to_bits().to_le_bytes());
                    }
                }
                let block = &block[..values.len() * width];
                disk(root, block.len() as u64)?;
                file.write_all(block)?;
                self.hash.update(block);
            }
            file.sync_all()?;
            self.written += bytes as u64;
            need(
                Identity::of(&fs::symlink_metadata(path)?)
                    .same_object(self.identity.as_ref().ok_or("missing output identity")?),
                "output path replaced",
            )?;
            Ok(())
        }
        fn finish(mut self, root: &Path) -> Result<Array> {
            need(
                self.written == self.array.bytes,
                "incomplete diagnostic array",
            )?;
            self.array.sha256 = format!("{:x}", self.hash.finalize());
            let held = HeldFile::open(&root.join(&self.array.path), self.array.bytes)?;
            need(
                held.before.bytes == self.array.bytes
                    && self
                        .identity
                        .as_ref()
                        .is_some_and(|old| old.same_object(&held.before))
                    && held.sha()? == self.array.sha256,
                "array readback hash mismatch",
            )?;
            Ok(self.array)
        }
    }
    #[derive(Serialize)]
    struct Chunk {
        start: usize,
        absolute_start: usize,
        rows: usize,
        wall_seconds: f64,
        diagnostics_seconds: f64,
        timing: metal::Timing,
        memory: session::Memory,
    }
    #[derive(Serialize)]
    struct Run {
        case: String,
        tokens: Vec<u32>,
        offset: usize,
        chunk: usize,
        repeat: &'static str,
        row_offset: usize,
        arrays: Vec<Array>,
        chunks: Vec<Chunk>,
        final_logits_sha256: String,
        final_greedy_token: u32,
        diagnostic_final_row_equals_logits: bool,
        checks: BTreeMap<&'static str, bool>,
        pending_token_applicable: bool,
    }
    impl Run {
        fn heap(&self) -> usize {
            self.case.capacity()
                + self.tokens.capacity() * 4
                + self.arrays.capacity() * size_of::<Array>()
                + self.arrays.iter().map(Array::heap).sum::<usize>()
                + self.chunks.capacity() * size_of::<Chunk>()
                + self.final_logits_sha256.capacity()
                + self.checks.len() * 128
        }
    }
    #[derive(Serialize)]
    struct Manifest {
        format: &'static str,
        completed: bool,
        graph_qualified: bool,
        error: Option<String>,
        active_run: Option<String>,
        protocol: Option<Receipt>,
        build_manifest: Option<Receipt>,
        binary: Option<Receipt>,
        capture_source: SourcePin,
        model_data: Option<Receipt>,
        model_config: Option<Receipt>,
        model_directory: Option<Identity>,
        device: Option<String>,
        load_seconds: Option<f64>,
        engine_budget_bytes: usize,
        helper_limit_bytes: usize,
        helper_high_water_bytes: usize,
        runs: Vec<Run>,
        checks: BTreeMap<&'static str, bool>,
    }
    impl Manifest {
        fn new() -> Self {
            let source = include_bytes!("ministral3_graph_capture.rs");
            Self {
                format: "toklet-ministral3-native-graph-capture-v1",
                completed: false,
                graph_qualified: false,
                error: None,
                active_run: None,
                protocol: None,
                build_manifest: None,
                binary: None,
                capture_source: SourcePin {
                    bytes: source.len(),
                    sha256: hash(source),
                },
                model_data: None,
                model_config: None,
                model_directory: None,
                device: None,
                load_seconds: None,
                engine_budget_bytes: BUDGET - HELPER,
                helper_limit_bytes: HELPER,
                helper_high_water_bytes: FIXED_HELPER,
                runs: Vec::with_capacity(22),
                checks: BTreeMap::new(),
            }
        }
        fn heap(&self) -> usize {
            self.runs.capacity() * size_of::<Run>() + self.runs.iter().map(Run::heap).sum::<usize>()
        }
        fn helper(&mut self, additional: usize) -> Result<()> {
            let bytes = FIXED_HELPER
                .checked_add(self.heap())
                .and_then(|n| n.checked_add(additional))
                .ok_or("helper size overflow")?;
            self.helper_high_water_bytes = self.helper_high_water_bytes.max(bytes);
            need(bytes <= HELPER, "diagnostic helper exceeds 32 MiB")
        }
        fn save(&mut self, root: &Path, initial: bool) -> Result<()> {
            // No readback arrays are live during JSON serialization. The fixed
            // reservation and live typed metadata are charged separately.
            let mut bytes = serde_json::to_vec_pretty(self)?;
            self.helper(bytes.capacity())?;
            // Include the serialization reservation in the saved high-water
            // bound too. Reuse the same Vec; no second JSON buffer is live.
            bytes.clear();
            serde_json::to_writer_pretty(&mut bytes, self)?;
            self.helper(bytes.capacity())?;
            if initial {
                write_new(root, &root.join("manifest.json"), &bytes)
            } else {
                let partial = root.join("manifest.next.json");
                write_new(root, &partial, &bytes)?;
                fs::rename(partial, root.join("manifest.json"))?;
                File::open(root)?.sync_all()?;
                Ok(())
            }
        }
    }
    fn row_hash(values: &[f32]) -> String {
        let mut sha = Sha256::new();
        for value in values {
            sha.update(value.to_le_bytes());
        }
        format!("{:x}", sha.finalize())
    }
    fn geometry(model: &metal::Model) -> Result<()> {
        let c = model.config();
        need(
            c.vocab_size == VOCAB
                && c.hidden_size == 3072
                && c.intermediate_size == 9216
                && c.num_layers == LAYERS
                && c.query_heads == 32
                && c.kv_heads == 8
                && c.head_dim == 128
                && c.max_positions == CAPACITY
                && c.rms_norm_eps.to_bits() == 1e-5_f32.to_bits()
                && c.rope_theta.to_bits() == 1e6_f32.to_bits(),
            "loaded geometry differs from frozen profile",
        )
    }
    fn make<'m>(model: &'m metal::Model, case: &Case, chunk: usize) -> Result<metal::Session<'m>> {
        let mut s = metal::Session::with_attention_mode(
            model,
            session::Options {
                capacity: CAPACITY - case.offset,
                prefill_chunk_size: chunk.min(CAPACITY - case.offset),
                position_offset: case.offset,
                memory_limit_bytes: BUDGET - HELPER,
            },
            metal::ProjectionMode::MlxAffine,
            metal::NormalizationMode::Compensated,
            metal::ReadbackMode::FullLogits,
            metal::AttentionMode::Reference,
        )?;
        s.enable_diagnostics(&(0..LAYERS).collect::<Vec<_>>())?;
        need(
            s.memory().admitted_bytes <= BUDGET - HELPER,
            "session exceeds engine budget",
        )?;
        Ok(s)
    }
    fn capture(
        s: &mut metal::Session<'_>,
        case: &Case,
        chunk: usize,
        repeat: &'static str,
        row_offset: usize,
        root: &Path,
        m: &mut Manifest,
    ) -> Result<Run> {
        let folder = format!("chunk-{chunk}/{}-{repeat}", case.name);
        m.active_run = Some(folder.clone());
        let path = root.join(&folder);
        fs::create_dir(&path)?;
        let (dir, dir_identity) = directory(&path)?;
        let specs = specs();
        let rows = case
            .tokens
            .len()
            .checked_sub(row_offset)
            .ok_or("row offset outside case")?;
        need(
            rows > 0 && s.history() == case.tokens && s.evaluated_len() == row_offset,
            "capture must start at its declared uncommitted row",
        )?;
        let mut writers = specs
            .iter()
            .map(|spec| {
                let name = spec.name();
                let bytes = rows * spec.columns * if spec.float() { 4 } else { 2 };
                Writer {
                    array: Array {
                        path: format!("{folder}/{name}.bin"),
                        name,
                        layer: spec.layer,
                        stage: spec.stage,
                        dtype: if spec.float() { "<f4" } else { "<f2" },
                        shape: [rows, spec.columns],
                        bytes: bytes as u64,
                        sha256: String::new(),
                    },
                    identity: None,
                    hash: Sha256::new(),
                    written: 0,
                }
            })
            .collect::<Vec<_>>();
        disk(root, writers.iter().map(|w| w.array.bytes).sum())?;
        let writer_heap = writers.capacity() * size_of::<Writer>()
            + writers.iter().map(|w| w.array.heap()).sum::<usize>();
        let mut chunks = Vec::with_capacity(rows);
        let mut final_row_equal = false;
        while s.pending_len() > 0 {
            let start = s.evaluated_len();
            let expected = s.pending_len().min(chunk.min(s.capacity()));
            let tick = Instant::now();
            let n = s.evaluate_next_chunk()?;
            let wall_seconds = tick.elapsed().as_secs_f64();
            need(
                n == expected && n > 0 && s.evaluated_len() == start + n,
                "chunk did not commit its exact scheduled prefix",
            )?;
            let readback = specs
                .iter()
                .map(|v| n * v.columns * size_of::<f32>())
                .sum::<usize>()
                + specs.len() * size_of::<metal::Diagnostic>();
            m.helper(writer_heap + chunks.capacity() * size_of::<Chunk>() + readback)?;
            let tick = Instant::now();
            let diagnostics = s.diagnostics()?;
            let diagnostics_seconds = tick.elapsed().as_secs_f64();
            need(
                diagnostics.len() == 419,
                "expected all 419 diagnostic stages",
            )?;
            let actual_readback = diagnostics.capacity() * size_of::<metal::Diagnostic>()
                + diagnostics
                    .iter()
                    .map(|d| d.values.capacity() * size_of::<f32>())
                    .sum::<usize>();
            m.helper(writer_heap + chunks.capacity() * size_of::<Chunk>() + actual_readback)?;
            for ((diagnostic, spec), writer) in diagnostics.iter().zip(&specs).zip(&mut writers) {
                need(
                    diagnostic.layer == spec.layer
                        && diagnostic.stage == spec.stage
                        && diagnostic.columns == spec.columns
                        && diagnostic.rows == n
                        && diagnostic.values.len() == n * spec.columns,
                    "diagnostic stage inventory, shape or order differs",
                )?;
                if spec.float() && s.pending_len() == 0 {
                    let row = &diagnostic.values[(n - 1) * VOCAB..];
                    final_row_equal = s.logits().is_some_and(|v| {
                        v.len() == row.len()
                            && v.iter().zip(row).all(|(a, b)| a.to_bits() == b.to_bits())
                    });
                }
                writer.append(root, &diagnostic.values)?;
            }
            chunks.push(Chunk {
                start,
                absolute_start: case.offset + start,
                rows: n,
                wall_seconds,
                diagnostics_seconds,
                timing: s.timing().ok_or("missing completed GPU timing")?,
                memory: s.memory(),
            });
        }
        need(
            final_row_equal
                && s.history() == case.tokens
                && s.evaluated_len() == case.tokens.len()
                && s.evaluate_next_chunk()? == 0,
            "completed distribution/state mismatch",
        )?;
        let logits = s.logits().ok_or("missing completed logits")?;
        let final_logits_sha256 = row_hash(logits);
        let final_greedy_token = s.greedy_token().ok_or("missing completed token")??;
        let arrays = writers
            .into_iter()
            .map(|w| w.finish(root))
            .collect::<Result<Vec<_>>>()?;
        verify_directory(&path, &dir, &dir_identity)?;
        let run = Run {
            case: case.name.clone(),
            tokens: case.tokens.clone(),
            offset: case.offset,
            chunk,
            repeat,
            row_offset,
            arrays,
            chunks,
            final_logits_sha256,
            final_greedy_token,
            diagnostic_final_row_equals_logits: final_row_equal,
            checks: BTreeMap::new(),
            pending_token_applicable: case.tokens.len() < s.capacity(),
        };
        m.helper(run.heap())?;
        m.active_run = None;
        Ok(run)
    }
    fn state_controls(s: &mut metal::Session<'_>, case: &Case, run: &mut Run) -> Result<()> {
        s.set_tokens(&case.tokens)?;
        run.checks.insert(
            "fully_cached_prompt_zero_work",
            s.evaluate_next_chunk()? == 0
                && s.evaluated_len() == case.tokens.len()
                && s.pending_len() == 0
                && s.logits()
                    .is_some_and(|v| row_hash(v) == run.final_logits_sha256),
        );
        let invalid = s.append(&[u32::MAX]).is_err();
        let overflow = s.set_tokens(&vec![1; s.capacity() + 1]).is_err();
        run.checks.insert(
            "invalid_inputs_preserve_committed_prefix",
            invalid
                && overflow
                && s.history() == case.tokens
                && s.evaluated_len() == case.tokens.len()
                && s.logits()
                    .is_some_and(|v| row_hash(v) == run.final_logits_sha256),
        );
        if run.pending_token_applicable {
            s.append(&[run.final_greedy_token])?;
            let pending = s.evaluated_len() == case.tokens.len()
                && s.pending_len() == 1
                && s.logits().is_none()
                && s.greedy_token().is_none();
            s.discard_pending();
            run.checks.insert(
                "pending_token_uncommitted_discard_restores_distribution",
                pending
                    && s.history() == case.tokens
                    && s.pending_len() == 0
                    && s.logits()
                        .is_some_and(|v| row_hash(v) == run.final_logits_sha256),
            );
        }
        need(
            run.checks.values().all(|v| *v),
            "cached/pending/validation state control failed",
        )
    }
    fn all_arrays_equal(a: &Run, b: &Run) -> bool {
        a.arrays.len() == 419
            && b.arrays.len() == 419
            && a.arrays.iter().zip(&b.arrays).all(|(a, b)| {
                a.name == b.name
                    && a.dtype == b.dtype
                    && a.shape == b.shape
                    && a.bytes == b.bytes
                    && a.sha256 == b.sha256
            })
    }
    fn changed_equal(root: &Path, changed: &Run, fresh: &Run) -> Result<bool> {
        need(
            changed.row_offset == 3
                && fresh.row_offset == 0
                && changed.arrays.len() == 419
                && fresh.arrays.len() == 419,
            "changed-prefix comparison inventory",
        )?;
        for (a, b) in changed.arrays.iter().zip(&fresh.arrays) {
            if a.name != b.name || a.dtype != b.dtype || a.shape != [1, b.shape[1]] {
                return Ok(false);
            }
            let width = if b.dtype == "<f4" { 4 } else { 2 };
            let file = HeldFile::open(&root.join(&b.path), b.bytes)?;
            let equal =
                hash_range(&file.file, (3 * b.shape[1] * width) as u64, a.bytes)? == a.sha256;
            file.verify()?;
            if !equal {
                return Ok(false);
            }
        }
        Ok(changed.final_logits_sha256 == fresh.final_logits_sha256)
    }

    fn run(args: &Args, root: &Path, m: &mut Manifest) -> Result<()> {
        need(
            args.protocol_sha == PROTOCOL_SHA,
            "protocol argument is not the frozen protocol SHA",
        )?;
        let protocol = HeldFile::open(&args.protocol, 64 * 1024)?;
        let protocol_hash = protocol.sha()?;
        need(
            protocol_hash == PROTOCOL_SHA,
            "frozen protocol hash mismatch",
        )?;
        let p: Protocol = serde_json::from_slice(&protocol.read()?)?;
        need(
            p.format == "toklet-ministral3-native-graph-protocol-v1"
                && p.model.data_sha256 == DATA_SHA
                && p.model.config_sha256 == CONFIG_SHA
                && p.chunks == [1, 3]
                && p.cases.len() == 5
                && p.cases.iter().all(|c| {
                    !c.tokens.is_empty()
                        && c.tokens.len() <= 8
                        && c.offset
                            .checked_add(c.tokens.len())
                            .is_some_and(|n| n <= CAPACITY)
                        && c.tokens.iter().all(|v| (*v as usize) < VOCAB)
                }),
            "frozen protocol fields invalid",
        )?;
        m.protocol = Some(protocol.receipt(protocol_hash)?);
        let build = HeldFile::open(&args.build, 1024 * 1024)?;
        let build_hash = build.sha()?;
        need(build_hash == args.build_sha, "build manifest hash mismatch")?;
        let b: Build = serde_json::from_slice(&build.read()?)?;
        need(
            b.capture_source.bytes == m.capture_source.bytes
                && b.capture_source.sha256 == m.capture_source.sha256,
            "build capture source pin mismatch",
        )?;
        m.build_manifest = Some(build.receipt(build_hash)?);
        let binary_path = fs::canonicalize(std::env::current_exe()?)?;
        need(
            fs::canonicalize(&b.binary.path)? == binary_path,
            "build binary path mismatch",
        )?;
        let binary = HeldFile::open(&binary_path, 512 * 1024 * 1024)?;
        let binary_hash = binary.sha()?;
        need(
            binary.before.bytes == b.binary.bytes && binary_hash == b.binary.sha256,
            "actual binary differs from build manifest",
        )?;
        m.binary = Some(binary.receipt(binary_hash)?);
        let model_path = fs::canonicalize(&args.model)?;
        let (model_dir, dir_identity) = directory(&model_path)?;
        m.model_directory = Some(dir_identity.clone());
        let data = HeldFile::open(&model_path.join("model.safetensors"), DATA_BYTES)?;
        let config = HeldFile::open(&model_path.join("config.json"), 256 * 1024)?;
        let data_hash = data.sha()?;
        let config_hash = config.sha()?;
        need(
            data.before.bytes == DATA_BYTES && data_hash == DATA_SHA && config_hash == CONFIG_SHA,
            "model data/config does not match frozen child",
        )?;
        m.model_data = Some(data.receipt(data_hash)?);
        m.model_config = Some(config.receipt(config_hash)?);
        verify_directory(&model_path, &model_dir, &dir_identity)?;
        let tick = Instant::now();
        let model = metal::Model::load_ministral3_diagnostic(
            &model_path,
            weights::LoadOptions {
                context_limit: CAPACITY,
                memory_limit_bytes: BUDGET - HELPER,
            },
        )?;
        m.load_seconds = Some(tick.elapsed().as_secs_f64());
        m.device = Some(model.device_name());
        geometry(&model)?;
        data.verify()?;
        config.verify()?;
        verify_directory(&model_path, &model_dir, &dir_identity)?;
        need(
            model.model_resident_bytes()? < BUDGET - HELPER,
            "resident model exceeds engine budget",
        )?;
        for &chunk in &p.chunks {
            fs::create_dir(root.join(format!("chunk-{chunk}")))?;
            for case in &p.cases {
                let mut s = make(&model, case, chunk)?;
                s.append(&case.tokens)?;
                need(
                    s.evaluated_len() == 0
                        && s.pending_len() == case.tokens.len()
                        && s.logits().is_none(),
                    "fresh input unexpectedly committed",
                )?;
                let mut initial = capture(&mut s, case, chunk, "initial", 0, root, m)?;
                state_controls(&mut s, case, &mut initial)?;
                m.runs.push(initial);
                s.reset();
                need(
                    s.history().is_empty()
                        && s.evaluated_len() == 0
                        && s.pending_len() == 0
                        && s.logits().is_none()
                        && s.greedy_token().is_none(),
                    "reset did not clear state",
                )?;
                s.set_tokens(&case.tokens)?;
                let mut replay = capture(&mut s, case, chunk, "reset_replay", 0, root, m)?;
                let initial = m.runs.last().ok_or("missing initial capture")?;
                let equal = all_arrays_equal(initial, &replay)
                    && replay.final_logits_sha256 == initial.final_logits_sha256;
                replay
                    .checks
                    .insert("reset_replay_all_stage_bytes_equal", equal);
                m.runs.push(replay);
                need(equal, "reset replay differs from initial raw stage bytes")?;
                if case.name == "prompt" {
                    let future = p
                        .cases
                        .iter()
                        .find(|c| c.name == "future_changed")
                        .ok_or("missing changed case")?;
                    s.set_tokens(&future.tokens)?;
                    need(
                        s.evaluated_len() == 3 && s.pending_len() == 1 && s.logits().is_none(),
                        "changed prompt did not retain exactly three committed tokens",
                    )?;
                    let changed = capture(&mut s, future, chunk, "changed_prefix", 3, root, m)?;
                    m.runs.push(changed);
                }
            }
            let changed = m
                .runs
                .iter()
                .find(|r| r.chunk == chunk && r.repeat == "changed_prefix")
                .ok_or("missing changed-prefix capture")?;
            let fresh = m
                .runs
                .iter()
                .find(|r| r.chunk == chunk && r.case == "future_changed" && r.repeat == "initial")
                .ok_or("missing fresh changed-case capture")?;
            let equal = changed_equal(root, changed, fresh)?;
            m.runs
                .iter_mut()
                .find(|r| r.chunk == chunk && r.repeat == "changed_prefix")
                .ok_or("missing changed-prefix capture")?
                .checks
                .insert("changed_prefix_all_suffix_bytes_equal_fresh", equal);
            need(equal, "changed-prefix reuse differs from fresh raw suffix")?;
        }
        need(
            m.runs.len() == 22
                && m.runs.iter().all(|r| {
                    r.arrays.len() == 419
                        && r.diagnostic_final_row_equals_logits
                        && r.checks.values().all(|v| *v)
                }),
            "incomplete run inventory",
        )?;
        drop(model);
        data.finish(m.model_data.as_mut().ok_or("missing model receipt")?)?;
        config.finish(m.model_config.as_mut().ok_or("missing config receipt")?)?;
        protocol.finish(m.protocol.as_mut().ok_or("missing protocol receipt")?)?;
        build.finish(m.build_manifest.as_mut().ok_or("missing build receipt")?)?;
        binary.finish(m.binary.as_mut().ok_or("missing binary receipt")?)?;
        verify_directory(&model_path, &model_dir, &dir_identity)?;
        m.checks.insert(
            "all_input_descriptor_path_and_byte_endpoints_unchanged",
            true,
        );
        m.checks
            .insert("complete_frozen_case_chunk_stage_inventory", true);
        m.completed = true;
        Ok(())
    }
    pub fn main() -> Result<()> {
        let args = args()?;
        need(!args.output.exists(), "fresh output must not exist")?;
        fs::create_dir(&args.output)?;
        let root = fs::canonicalize(&args.output)?;
        let (output_dir, output_identity) = directory(&root)?;
        let mut m = Manifest::new();
        m.save(&root, true)?;
        let result = run(&args, &root, &mut m)
            .and_then(|()| verify_directory(&root, &output_dir, &output_identity));
        if let Err(error) = &result {
            m.completed = false;
            m.error = Some(error.to_string());
        }
        m.save(&root, false)?;
        result
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        struct Temp(PathBuf);
        impl Temp {
            fn new() -> Self {
                let path = std::env::temp_dir().join(format!(
                    "toklet-ministral3-capture-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                fs::create_dir(&path).unwrap();
                Self(path)
            }
        }
        impl Drop for Temp {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        #[test]
        fn full_stage_geometry_and_storage_inventory() {
            let s = specs();
            assert_eq!(s.len(), 419);
            assert_eq!(s.iter().filter(|s| s.float()).count(), 1);
            assert_eq!(s.iter().map(|s| s.columns).sum::<usize>(), 1_974_272);
            assert_eq!(s[2].columns, 131_072);
            assert_eq!(s[4].name(), "layer00.q");
            assert_eq!(s[418].name(), "layer25.output");
        }
        #[test]
        fn held_input_detects_replacement_and_short_extent() {
            let temp = Temp::new();
            let path = temp.0.join("input");
            fs::write(&path, b"abcd").unwrap();
            let held = HeldFile::open(&path, 4).unwrap();
            assert!(hash_range(&held.file, 0, 5).is_err());
            fs::rename(&path, temp.0.join("old")).unwrap();
            fs::write(&path, b"abcd").unwrap();
            assert!(held.verify().is_err());
        }
        fn writer(dtype: &'static str, bytes: u64) -> Writer {
            Writer {
                array: Array {
                    name: "stage".into(),
                    layer: None,
                    stage: "embedding",
                    path: "stage.bin".into(),
                    dtype,
                    shape: [2, 2],
                    bytes,
                    sha256: String::new(),
                },
                identity: None,
                hash: Sha256::new(),
                written: 0,
            }
        }
        #[test]
        fn half_writer_preserves_bytes_checks_readback_and_rejects_invalid() {
            let temp = Temp::new();
            let mut w = writer("<f2", 8);
            assert!(w.append(&temp.0, &[f32::NAN]).is_err());
            assert!(w.append(&temp.0, &[1.0001]).is_err());
            assert!(!temp.0.join("stage.bin").exists());
            w.append(&temp.0, &[0.0, -0.0]).unwrap();
            w.append(&temp.0, &[1.0, -2.0]).unwrap();
            let a = w.finish(&temp.0).unwrap();
            let bytes = [0, 0, 0, 128, 0, 60, 0, 192];
            assert_eq!(fs::read(temp.0.join(a.path)).unwrap(), bytes);
            assert_eq!(a.sha256, hash(&bytes));
        }
        #[test]
        fn float_writer_admits_full_vocab_and_refuses_overwrite_or_truncation() {
            let temp = Temp::new();
            let mut w = writer("<f4", 8);
            w.append(&temp.0, &[1.25]).unwrap();
            assert!(writer("<f4", 8).append(&temp.0, &[2.0]).is_err());
            fs::write(temp.0.join("stage.bin"), []).unwrap();
            assert!(w.append(&temp.0, &[2.0]).is_err());
            assert!(w.finish(&temp.0).is_err());
            let mut w = writer("<f4", (VOCAB * 4) as u64);
            w.array.path = "logits.bin".into();
            w.array.shape = [1, VOCAB];
            w.append(&temp.0, &vec![1.25; VOCAB]).unwrap();
            assert_eq!(w.finish(&temp.0).unwrap().bytes, 524_288);
        }
        #[test]
        fn failure_manifest_is_replaced_atomically_without_removing_raw_partial() {
            let temp = Temp::new();
            let mut m = Manifest::new();
            m.save(&temp.0, true).unwrap();
            fs::write(temp.0.join("retained.bin"), [1, 2]).unwrap();
            m.error = Some("controlled failure".into());
            m.active_run = Some("chunk-1/prompt-initial".into());
            m.save(&temp.0, false).unwrap();
            let value: serde_json::Value =
                serde_json::from_slice(&fs::read(temp.0.join("manifest.json")).unwrap()).unwrap();
            assert_eq!(value["completed"], false);
            assert_eq!(value["graph_qualified"], false);
            assert_eq!(value["error"], "controlled failure");
            assert_eq!(fs::read(temp.0.join("retained.bin")).unwrap(), [1, 2]);
            assert!(!temp.0.join("manifest.next.json").exists());
            assert!(m.save(&temp.0, true).is_err());
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    native::main()
}
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("Ministral 3 native graph capture requires Apple Silicon and Metal");
    std::process::exit(2);
}
