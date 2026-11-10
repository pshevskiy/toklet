//! Explicit deeper numerical capture. Ordinary model/session policy is unchanged.
use super::artifact::{FILES, REPOSITORY, REVISION, Result, VOCAB, hash_file};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs::File, io::Read, path::Path};

const CONTRACT_SHA: &str = "b5464b7d67cee7fa8d0df28afc5fb0e68f0c5d550232e572f93858dc4f61bad3";
const QUALITY_SHA: &str = "fe954ea5afa82f3fc33981c139727bef4fb3cd10f1b6485b7f0001f3b43875a7";
const TOKENS_SHA: &str = "297451dd509fd620d91a997dac58ad40f514f4fa2f63efcc92253a314c7476cc";
const COUNTS: [usize; 3] = [128, 512, 1025];
const CAPACITY: usize = 1025;
const CAPTURE_LIMIT: usize = 4 * 1024 * 1024 * 1024;

fn digest(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut block = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut block)?;
        if n == 0 {
            break;
        }
        hash.update(&block[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn document(path: &Path, expected: &str) -> Result<Value> {
    if expected.len() != 64
        || !expected
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("explicit lowercase SHA256 required".into());
    }
    let info = path.symlink_metadata()?;
    if !info.is_file() || info.len() > 2 * 1024 * 1024 {
        return Err("bounded regular JSON required".into());
    }
    hash_file(path, info.len(), expected)?;
    let mut raw = Vec::new();
    File::open(path)?
        .take(2 * 1024 * 1024 + 1)
        .read_to_end(&mut raw)?;
    if raw.len() > 2 * 1024 * 1024 {
        return Err("JSON grew beyond bound".into());
    }
    Ok(serde_json::from_slice(&raw)?)
}

fn validate(reference: &Value, contract: &Value) -> Result<BTreeMap<String, usize>> {
    if reference["format"] != "toklet-minicpm5-context1025-reference-v1"
        || reference["completed"].as_bool() != Some(true)
        || reference["native_outcomes_seen"].as_bool() != Some(false)
        || reference["quality_scored"].as_bool() != Some(false)
        || reference["managed_adoption_qualified"].as_bool() != Some(false)
        || reference["gates_recomputed_or_relaxed"].as_bool() != Some(false)
        || reference["contract_sha256"] != CONTRACT_SHA
        || reference["quality_manifest_sha256"] != QUALITY_SHA
        || reference["model"]["repository"] != REPOSITORY
        || reference["model"]["revision"] != REVISION
        || reference["model"]["files"]["model.safetensors"] != FILES[2].2
        || reference["gates"] != contract["gates"]
        || reference["columns"] != contract["columns"]
        || reference["cases"] != contract["cases"]
        || contract["native"]["capacity"].as_u64() != Some(CAPACITY as u64)
        || contract["native"]["chunks"] != json!([1, 32])
    {
        return Err("fixed independent reference/contract differs".into());
    }
    let columns: BTreeMap<String, usize> = serde_json::from_value(contract["columns"].clone())?;
    let arrays = reference["arrays"]
        .as_object()
        .ok_or("CPU arrays required")?;
    if columns.len() != 51 || arrays.len() != 306 {
        return Err("complete CPU stage/profile coverage required".into());
    }
    for profile in ["fp32", "fp64"] {
        let mut offset = 0usize;
        let mut spans = Vec::with_capacity(153);
        for n in COUNTS {
            for (stage, &cols) in &columns {
                let name = format!("{profile}.prefix{n}.{stage}");
                let a = arrays.get(&name).ok_or("missing CPU stage")?;
                let shape = a["shape"].as_array().ok_or("CPU shape required")?;
                let dtype = if stage == "logits" { "<f4" } else { "<f2" };
                let bytes = n
                    .checked_mul(cols)
                    .and_then(|x| x.checked_mul(if stage == "logits" { 4 } else { 2 }))
                    .ok_or("CPU size overflow")?;
                if shape.len() != 2
                    || shape[0].as_u64() != Some(n as u64)
                    || shape[1].as_u64() != Some(cols as u64)
                    || a["dtype"] != dtype
                    || a["gate"] != *stage
                    || a["file"] != format!("{profile}.bin")
                    || a["bytes"].as_u64() != Some(bytes as u64)
                {
                    return Err("CPU array storage/geometry differs".into());
                }
                let start = usize::try_from(a["offset"].as_u64().ok_or("CPU offset required")?)?;
                spans.push((start, start.checked_add(bytes).ok_or("CPU span overflow")?));
                offset = offset.checked_add(bytes).ok_or("CPU extent overflow")?;
            }
        }
        if reference["files"][format!("{profile}.bin")]["bytes"].as_u64() != Some(offset as u64) {
            return Err("CPU file coverage differs".into());
        }
        spans.sort_unstable();
        let mut end = 0;
        for (first, next) in spans {
            if first != end {
                return Err("CPU spans have holes or overlaps".into());
            }
            end = next;
        }
        if end != offset {
            return Err("CPU span coverage differs".into());
        }
    }
    Ok(columns)
}

fn tokens(raw: &[u8]) -> Result<Vec<u32>> {
    if raw.len() != 40004 {
        return Err("complete10001-ID quality payload required".into());
    }
    let ids: Vec<_> = raw
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    if ids.iter().any(|&id| id as usize >= VOCAB) {
        return Err("quality ID outside vocabulary".into());
    }
    Ok(ids)
}

fn stage(layer: Option<usize>, value: &str) -> String {
    let value = match value {
        "q" => "q_projection",
        "k" => "k_projection",
        "v" => "v_projection",
        "gate" => "gate_projection",
        "up" => "up_projection",
        "silu" => "gate_silu",
        "down" => "down_projection",
        other => other,
    };
    layer.map_or_else(|| value.to_owned(), |i| format!("layers.{i}.{value}"))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn run() -> Result<()> {
    use std::{
        collections::BTreeSet,
        fs::{self, OpenOptions},
        io::{BufWriter, Write},
        path::PathBuf,
        time::Instant,
    };
    use toklet::{
        metal::{self, AttentionMode, NormalizationMode, ProjectionMode, ReadbackMode},
        session,
        weights::LoadOptions,
    };
    const TOTAL: usize = 6 * 1024 * 1024 * 1024;
    const APP: usize = 32 * 1024 * 1024;
    let args: Vec<_> = std::env::args_os().skip(2).collect();
    if args.len() != 7 {
        return Err("usage: minicpm5_probe context MODEL REFERENCE REF-SHA CONTRACT CONTRACT-SHA QUALITY-INPUTS NEW-OUTPUT".into());
    }
    let model_path = PathBuf::from(&args[0]);
    let reference_path = PathBuf::from(&args[1]);
    let reference_sha = args[2].to_str().ok_or("reference SHA must be UTF8")?;
    let contract_path = PathBuf::from(&args[3]);
    if args[4].to_str() != Some(CONTRACT_SHA) {
        return Err("exact prospective context contract required".into());
    }
    let quality = PathBuf::from(&args[5]);
    let out = PathBuf::from(&args[6]);
    if out.symlink_metadata().is_ok() {
        return Err("fresh output required".into());
    }
    let contract = document(&contract_path, CONTRACT_SHA)?;
    let reference = document(&reference_path.join("manifest.json"), reference_sha)?;
    let columns = validate(&reference, &contract)?;
    let q = document(&quality.join("manifest.json"), QUALITY_SHA)?;
    if q["loss_targets"].as_u64() != Some(10000) || q["task_cases"].as_u64() != Some(100) {
        return Err("full quality inputs required".into());
    }
    hash_file(&quality.join("held_out_loss.u32le"), 40004, TOKENS_SHA)?;
    let ids = tokens(&fs::read(quality.join("held_out_loss.u32le"))?)?;
    for name in ["fp32.bin", "fp64.bin"] {
        hash_file(
            &reference_path.join(name),
            reference["files"][name]["bytes"]
                .as_u64()
                .ok_or("reference file bytes required")?,
            reference["files"][name]["sha256"]
                .as_str()
                .ok_or("reference file hash required")?,
        )?;
    }
    for &(name, size, pin) in FILES {
        hash_file(&model_path.join(name), size, pin)?;
    }
    fs::create_dir(&out)?;
    let origin = Instant::now();
    let load = Instant::now();
    let model = metal::Model::load_mlx_minicpm5_context_diagnostic(
        &model_path,
        LoadOptions {
            context_limit: CAPACITY,
            memory_limit_bytes: TOTAL - APP,
        },
    )?;
    let load_wall_ns = u64::try_from(load.elapsed().as_nanos())?;
    if model.weights_bytes() != 2674218150
        || model.positional_resident_bytes() != 0
        || model.mlx_declared_positions() != Some(131072)
    {
        return Err("resident model identity differs".into());
    }
    struct Capture {
        binary: BufWriter<File>,
        journal: BufWriter<File>,
        bytes: usize,
        arrays: usize,
    }
    struct Frame<'a> {
        kind: &'a str,
        count: usize,
        chunk: usize,
        name: &'a str,
        start: usize,
        rows: usize,
        cols: usize,
    }
    impl Capture {
        fn save(&mut self, frame: Frame<'_>, values: &[f32]) -> Result<()> {
            let Frame {
                kind,
                count,
                chunk,
                name,
                start,
                rows,
                cols,
            } = frame;
            let bytes = rows
                .checked_mul(cols)
                .and_then(|v| v.checked_mul(4))
                .ok_or("capture extent overflow")?;
            if rows == 0
                || values.len() != rows * cols
                || self
                    .bytes
                    .checked_add(bytes)
                    .is_none_or(|v| v > CAPTURE_LIMIT)
                || self.arrays >= 90000
            {
                return Err("bounded complete native frame required".into());
            }
            let offset = self.bytes;
            let mut hash = Sha256::new();
            for &value in values {
                if !value.is_finite() {
                    return Err("nonfinite native frame".into());
                }
                let raw = value.to_le_bytes();
                hash.update(raw);
                self.binary.write_all(&raw)?;
            }
            self.binary.flush()?;
            let row = json!({"event":"completed","kind":kind,"case":format!("prefix{count}"),"chunk":chunk,"stage":name,
                "row_start":start,"shape":[rows,cols],"dtype":"<f4","offset":offset,"bytes":bytes,"sha256":format!("{:x}",hash.finalize())});
            serde_json::to_writer(&mut self.journal, &row)?;
            self.journal.write_all(b"\n")?;
            self.journal.flush()?;
            self.bytes += bytes;
            self.arrays += 1;
            Ok(())
        }
    }
    let fresh = |name: &str| {
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join(name))
    };
    let mut capture = Capture {
        binary: BufWriter::new(fresh("arrays.bin")?),
        journal: BufWriter::new(fresh("arrays.jsonl")?),
        bytes: 0,
        arrays: 0,
    };
    let mut metrics = BufWriter::new(fresh("chunks.jsonl")?);
    let mut controls = Vec::new();
    let mut max_admitted = 0usize;
    let mut chunks = 0usize;
    for count in COUNTS {
        for chunk in [1, 32] {
            let input = &ids[..count];
            let mut session = metal::Session::with_attention_mode(
                &model,
                session::Options {
                    capacity: CAPACITY,
                    prefill_chunk_size: chunk,
                    position_offset: 0,
                    memory_limit_bytes: TOTAL - APP,
                },
                ProjectionMode::MlxAffine,
                NormalizationMode::Compensated,
                ReadbackMode::FullLogits,
                AttentionMode::Reference,
            )?;
            session.enable_diagnostics(&[0, 1, 41])?;
            max_admitted = max_admitted.max(
                session
                    .memory()
                    .admitted_bytes
                    .checked_add(APP)
                    .ok_or("admission overflow")?,
            );
            if max_admitted > TOTAL {
                return Err("combined diagnostic budget exceeds6GiB".into());
            }
            session.set_tokens(input)?;
            let mut start = 0usize;
            loop {
                let begun = u64::try_from(origin.elapsed().as_nanos())?;
                let wall = Instant::now();
                let rows = session.evaluate_next_chunk()?;
                let wall_ns = u64::try_from(wall.elapsed().as_nanos())?;
                let ended = u64::try_from(origin.elapsed().as_nanos())?;
                if rows == 0 {
                    break;
                }
                if rows != chunk.min(count - start)
                    || session.evaluated_len() != start + rows
                    || session.pending_len() != count - start - rows
                {
                    return Err("completed chunk history differs".into());
                }
                let diagnostics = session.diagnostics()?;
                let mut seen = BTreeSet::new();
                for d in diagnostics {
                    let name = stage(d.layer, d.stage);
                    if d.rows != rows
                        || columns.get(&name) != Some(&d.columns)
                        || !seen.insert(name.clone())
                    {
                        return Err("diagnostic stage/row inventory differs".into());
                    }
                    capture.save(
                        Frame {
                            kind: "prefill",
                            count,
                            chunk,
                            name: &name,
                            start,
                            rows,
                            cols: d.columns,
                        },
                        &d.values,
                    )?;
                }
                if seen.len() != 51 {
                    return Err("complete51stage capture required".into());
                }
                let row = json!({"case":format!("prefix{count}"),"chunk":chunk,"row_start":start,"rows":rows,
                    "committed_after":session.evaluated_len(),"pending_after":session.pending_len(),
                    "started_since_run_ns":begun,"completed_since_run_ns":ended,"wall_ns":wall_ns,
                    "completed_timing":session.timing().ok_or("completed timing absent")?,"session_memory":session.memory()});
                serde_json::to_writer(&mut metrics, &row)?;
                metrics.write_all(b"\n")?;
                metrics.flush()?;
                chunks += 1;
                if start == 0 {
                    session.discard_pending();
                    if session.history() != &input[..rows]
                        || session.evaluated_len() != rows
                        || session.pending_len() != 0
                    {
                        return Err("discard lost committed prefix".into());
                    }
                    session.set_tokens(input)?;
                    if session.evaluated_len() != rows || session.pending_len() != count - rows {
                        return Err("exact prefix not reused".into());
                    }
                }
                start += rows;
            }
            if start != count {
                return Err("incomplete context".into());
            }
            let before: Vec<_> = session.logits().ok_or("full distribution absent")?.to_vec();
            session.set_tokens(input)?;
            if session.evaluate_next_chunk()? != 0 || session.logits() != Some(before.as_slice()) {
                return Err("cached full prompt changed distribution".into());
            }
            if count < CAPACITY {
                session.append(&[1])?;
                if session.pending_len() != 1 || session.greedy_token().is_some() {
                    return Err("pending sampled token exposed stale logits".into());
                }
                session.discard_pending();
            } else if session.append(&[1]).is_ok() {
                return Err("one-past append accepted".into());
            }
            if session.set_tokens(&vec![1; CAPACITY + 1]).is_ok()
                || session.set_tokens(&[VOCAB as u32]).is_ok()
                || session.history() != input
                || session.evaluated_len() != count
                || session.logits() != Some(before.as_slice())
            {
                return Err("rejection changed valid prefix".into());
            }
            let mut changed = input.to_vec();
            changed[count - 1] ^= 1;
            session.set_tokens(&changed)?;
            if session.evaluated_len() != count - 1 || session.pending_len() != 1 {
                return Err("changed-prefix invalidation differs".into());
            }
            session.discard_pending();
            session.append(&input[count - 1..])?;
            session.evaluate()?;
            capture.save(
                Frame {
                    kind: "corrected_last",
                    count,
                    chunk,
                    name: "logits",
                    start: count - 1,
                    rows: 1,
                    cols: VOCAB,
                },
                session.logits().ok_or("corrected logits absent")?,
            )?;
            session.reset();
            if !session.history().is_empty()
                || session.evaluated_len() != 0
                || session.pending_len() != 0
            {
                return Err("reset did not clear logical state".into());
            }
            session.set_tokens(input)?;
            session.evaluate()?;
            if !session
                .logits()
                .ok_or("replayed logits absent")?
                .iter()
                .zip(&before)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                return Err("same-schedule replay is not bitwise equal".into());
            }
            controls.push(json!({"case":format!("prefix{count}"),"chunk":chunk,"partial_discard_and_exact_prefix":true,
                "fully_cached":true,"pending_or_boundary_append":true,"rejections_preserve_state":true,
                "changed_prefix_discard_and_restore":true,"corrected_logits_captured_for_original_gate":true,"reset_replay_bitwise":true}));
        }
    }
    capture.binary.flush()?;
    capture.journal.flush()?;
    metrics.flush()?;
    if chunks != 1718 || capture.arrays != 87624 || capture.bytes != 3546040320 {
        return Err("whole capture coverage differs".into());
    }
    for &(name, size, pin) in FILES {
        hash_file(&model_path.join(name), size, pin)?;
    }
    let files: BTreeMap<_, _> = ["arrays.bin", "arrays.jsonl", "chunks.jsonl"]
        .iter()
        .map(|&name| -> Result<_> {
            Ok((
                name,
                json!({"bytes":out.join(name).metadata()?.len(),"sha256":digest(&out.join(name))?}),
            ))
        })
        .collect::<Result<_>>()?;
    let report = json!({"format":"toklet-minicpm5-context1025-native-v1","completed":true,"repository":REPOSITORY,"revision":REVISION,
        "reference_sha256":reference_sha,"contract_sha256":CONTRACT_SHA,"quality_manifest_sha256":QUALITY_SHA,
        "counts":COUNTS,"chunks":[1,32],"capacity":CAPACITY,"ordinary_capacity_unchanged":512,
        "rms_store":"serial baseline","arrays":capture.arrays,"array_bytes":capture.bytes,"completed_chunks":chunks,
        "maximum_combined_admitted_bytes":max_admitted,"memory_limit_bytes":TOTAL,"application_allowance_bytes":APP,
        "load_wall_ns":load_wall_ns,"elapsed_ns":u64::try_from(origin.elapsed().as_nanos())?,"functional":controls,"files":files,
        "numerical_comparison_pending":true,"quality_scored":false,"managed_adoption_qualified":false,
        "performance_or_overhead_qualified":false,"physical_iphone_qualified":false});
    serde_json::to_writer_pretty(fresh("capture.json")?, &report)?;
    println!(
        "{}",
        json!({"completed":true,"arrays":capture.arrays,"bytes":capture.bytes,"comparison_pending":true})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_quality_ids_reject_shortened_or_out_of_vocabulary() {
        assert_eq!(tokens(&vec![0; 40004]).unwrap().len(), 10001);
        assert!(tokens(&vec![0; 40000]).is_err());
        let mut raw = vec![0; 40004];
        raw[..4].copy_from_slice(&(VOCAB as u32).to_le_bytes());
        assert!(tokens(&raw).is_err());
    }
    #[test]
    fn stage_names_preserve_original_gate_mapping() {
        assert_eq!(stage(Some(41), "q"), "layers.41.q_projection");
        assert_eq!(stage(Some(0), "silu"), "layers.0.gate_silu");
        assert_eq!(stage(None, "logits"), "logits");
        assert_eq!(stage(Some(1), "output"), "layers.1.output");
    }
    #[test]
    fn incomplete_or_promoted_source_reference_rejects_without_gpu() {
        for value in [
            json!({}),
            json!({"completed":true}),
            json!({"format":"toklet-minicpm5-context1025-reference-v1","completed":1}),
        ] {
            assert!(validate(&value, &json!({})).is_err());
        }
    }
}
