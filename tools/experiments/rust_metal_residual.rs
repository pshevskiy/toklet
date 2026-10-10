use super::*;
use std::{collections::HashSet, io::Read, path::Path};

const MAX_INPUT: u64 = 1024 * 1024;
const STATE_GUARD: u16 = 0x51aa;
const DELTA_GUARD: u16 = 0x529b;
const _: () = assert!(size_of::<u32>() == 4 && align_of::<u32>() == 4);

struct Case {
    id: String,
    rows: usize,
    hidden: usize,
    state: Vec<u16>,
    delta: Vec<u16>,
}
fn scalar(value: &Value, name: &str) -> Result<usize, Error> {
    value[name]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| error("fixture", format!("invalid unsigned {name}")))
}
fn bits(value: &Value, name: &str, count: usize, guard: u16) -> Result<Vec<u16>, Error> {
    let source = value[name]
        .as_array()
        .ok_or_else(|| error("fixture", "missing residual Half bits"))?;
    if source.len() != count {
        return Err(error("fixture", "residual Half count differs"));
    }
    let mut values = Vec::with_capacity(count + GUARDS);
    for item in source {
        let number = item
            .as_u64()
            .and_then(|n| u16::try_from(n).ok())
            .ok_or_else(|| error("fixture", "residual bits outside UInt16"))?;
        if number & 0x7c00 == 0x7c00 {
            return Err(error("fixture", "nonfinite residual input Half"));
        }
        values.push(number);
    }
    values.resize(count + GUARDS, guard);
    Ok(values)
}
fn load(path: &Path) -> Result<Vec<Case>, Error> {
    let file = std::fs::File::open(path).map_err(|e| error("fixture", e.to_string()))?;
    let size = file
        .metadata()
        .map_err(|e| error("fixture", e.to_string()))?
        .len();
    if size > MAX_INPUT {
        return Err(error("fixture", "residual input exceeds one MiB"));
    }
    let mut data = Vec::new();
    file.take(MAX_INPUT + 1)
        .read_to_end(&mut data)
        .map_err(|e| error("fixture", e.to_string()))?;
    if data.len() as u64 != size || data.len() as u64 > MAX_INPUT {
        return Err(error(
            "fixture",
            "residual input changed size or exceeds bounds",
        ));
    }
    let value: Value =
        serde_json::from_slice(&data).map_err(|e| error("fixture", e.to_string()))?;
    if value["format"] != "toklet-metal-residual-input-v1" {
        return Err(error("fixture", "residual format differs"));
    }
    let sources = value["cases"]
        .as_array()
        .ok_or_else(|| error("fixture", "missing residual cases"))?;
    if sources.is_empty() || sources.len() > 32 {
        return Err(error("fixture", "residual cases outside 1..32"));
    }
    let mut names = HashSet::new();
    let mut elements = 0usize;
    let mut cases = Vec::with_capacity(sources.len());
    for source in sources {
        let id = source["id"]
            .as_str()
            .ok_or_else(|| error("fixture", "missing residual identity"))?;
        if id.is_empty() || id.len() > 128 || !id.is_ascii() || !names.insert(id) {
            return Err(error("fixture", "invalid or duplicate residual identity"));
        }
        let rows = scalar(source, "rows")?;
        let hidden = scalar(source, "hidden")?;
        if !(1..=32).contains(&rows) || ![32, 896].contains(&hidden) {
            return Err(error("fixture", "residual prototype shape outside bounds"));
        }
        let count = rows
            .checked_mul(hidden)
            .ok_or_else(|| error("admission", "residual shape overflow"))?;
        elements = elements
            .checked_add(count)
            .ok_or_else(|| error("admission", "residual inventory overflow"))?;
        if elements > 8192 {
            return Err(error("admission", "residual inventory exceeds 8192 values"));
        }
        cases.push(Case {
            id: id.to_owned(),
            rows,
            hidden,
            state: bits(source, "state_bits", count, STATE_GUARD)?,
            delta: bits(source, "delta_bits", count, DELTA_GUARD)?,
        });
    }
    // JSON/data temporaries drop before any native allocation.
    Ok(cases)
}
fn storage(count: usize) -> Result<usize, Error> {
    let bytes = count
        .checked_add(GUARDS)
        .and_then(|n| n.checked_mul(size_of::<u16>()))
        .ok_or_else(|| error("admission", "residual storage overflow"))?;
    if count == 0 || bytes > isize::MAX as usize {
        return Err(error(
            "admission",
            "residual storage exceeds pointer bounds",
        ));
    }
    Ok(bytes)
}
fn admitted(case: &Case, limit: usize) -> Result<usize, Error> {
    let count = case
        .rows
        .checked_mul(case.hidden)
        .ok_or_else(|| error("admission", "residual shape overflow"))?;
    // For each buffer: shared + host baseline + readback. Retained diagnostics
    // and bounded fixture temporaries use the fixed runtime allowance.
    let bytes = storage(count)?
        .checked_mul(6)
        .and_then(|n| n.checked_add(ALLOWANCE))
        .ok_or_else(|| error("admission", "residual working-set overflow"))?;
    if limit > LIMIT || bytes > limit {
        return Err(error("admission", "residual exceeds supplied budget"));
    }
    Ok(bytes)
}
fn execute(probe: &mut Probe, case: &Case, limit: usize) -> Result<Value, Error> {
    let budget = admitted(case, limit)?;
    let count = case.rows * case.hidden;
    let bytes = storage(count)?;
    let launch = groups(count, probe.width)?;
    let mut state = Shared::<u16>::new(&probe.device, bytes)?;
    let mut delta = Shared::<u16>::new(&probe.device, bytes)?;
    delta.write(&case.delta)?;
    let native_count = count as u32;
    let mut repetitions = Vec::with_capacity(13);
    for repetition in 0..13 {
        // Restore the complete baseline outside GPU access. A rejected result
        // is invalid, not rolled back; this standalone call returns immediately.
        state.write(&case.state)?;
        repetitions.push(autoreleasepool(|_| {
            let started = Instant::now();
            let command = probe.queue.commandBuffer().ok_or_else(|| error("command", "command unavailable"))?;
            if !command.retainedReferences() { return Err(error("command", "unretained resources unsupported")); }
            let encoder = command.computeCommandEncoder().ok_or_else(|| error("command", "encoder unavailable"))?;
            encoder.setComputePipelineState(&probe.pipeline);
            // SAFETY: two distinct owned Half allocations at slots0/1, offsets0,
            // checked lengths/alignment. Slot2 copies a UInt32 count (4/4).
            // Each lane only reads/writes its own state element; delta immutable.
            // Rounded grid/count fit UInt32 and shader checks count before reads.
            // Retained resources/owners stay alive through wait; no host access
            // from commit until successful completion/status/error checks.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&state.raw), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&delta.raw), 0, 1);
                encoder.setBytes_length_atIndex(NonNull::from(&native_count).cast(), size_of::<u32>(), 2);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: launch, height: 1, depth: 1 }, MTLSize { width: probe.width, height: 1, depth: 1 });
            encoder.endEncoding(); command.commit(); command.waitUntilCompleted();
            let seconds = started.elapsed().as_secs_f64();
            if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
                return Err(error("completion", format!("residual GPU failure: {:?}", command.error())));
            }
            let actual = state.read();
            if delta.read() != case.delta || actual[count..].iter().any(|&bits| bits != STATE_GUARD) {
                return Err(error("validation", "residual delta/state guards changed"));
            }
            if actual[..count].iter().any(|&bits| bits & 0x7c00 == 0x7c00) {
                return Err(error("validation", "completed residual contains nonfinite Half; in-place result invalid and discarded"));
            }
            Ok(json!({"phase":if repetition<3 {"warmup"} else {"sample"},"repetition":if repetition<3 {repetition+1} else {repetition-2},
                "command_completed":true,"immutable_delta_and_state_guards":true,"encode_commit_wait_seconds":seconds,"output_bits":actual}))
        })?);
    }
    Ok(
        json!({"id":case.id,"rows_count":case.rows,"hidden":case.hidden,"guards":GUARDS,"output_guard_bits":STATE_GUARD,
        "threadgroup_width":probe.width,"threadgroups":launch,"shared_buffer_bytes":bytes*2,"host_baseline_and_readback_bytes":bytes*4,
        "admitted_bytes":budget,"in_place_state":true,"baseline_restored_before_each_repetition":true,"allocations_reused_across_repetitions":true,"rows":repetitions}),
    )
}
pub(super) fn run(path: &Path, limit: usize) -> Result<Value, Error> {
    let cases = load(path)?;
    let maximum = cases
        .iter()
        .map(|case| admitted(case, limit))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or_else(|| error("admission", "empty residual cases"))?;
    autoreleasepool(|_| {
        let (mut probe, metadata) = prepare(
            include_str!("rust_metal_residual.metal"),
            "toklet_residual",
            maximum,
        )?;
        let mut results = Vec::with_capacity(cases.len());
        for case in &cases {
            results.push(autoreleasepool(|_| execute(&mut probe, case, limit))?);
        }
        Ok(
            json!({"operation":"residual_in_place_fp16_storage_fp32_add","metadata":metadata,"cases":results}),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    struct Temporary(PathBuf);
    impl Drop for Temporary {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    fn fixture(value: &Value) -> Temporary {
        let path = std::env::temp_dir().join(format!(
            "toklet-residual-{}-{}.json",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        writeln!(file, "{value}").unwrap();
        Temporary(path)
    }
    fn sample() -> Value {
        json!({"format":"toklet-metal-residual-input-v1","cases":[{"id":"manual","rows":1,"hidden":32,"state_bits":vec![0x3c00u16;32],"delta_bits":vec![0x0001u16;32]}]})
    }
    #[test]
    fn all_inputs_are_finite_unsigned_half_before_native_work() {
        for (key, bit) in [
            ("state_bits", json!(0x7c00)),
            ("delta_bits", json!(0xfc01)),
            ("delta_bits", json!(-1)),
            ("state_bits", json!(65536)),
        ] {
            let mut value = sample();
            value["cases"][0][key][31] = bit;
            let file = fixture(&value);
            assert_eq!(load(&file.0).err().unwrap().stage, "fixture");
        }
        // Finite inputs that sum outside Half range are intentionally admitted;
        // native completion must reject their invalid result, without rollback.
        let mut value = sample();
        value["cases"][0]["state_bits"] = json!(vec![0x7bffu16; 32]);
        value["cases"][0]["delta_bits"] = json!(vec![0x7bffu16; 32]);
        assert!(load(&fixture(&value).0).is_ok());
    }
    #[test]
    fn shape_names_and_cumulative_inventory_are_bounded() {
        let mut value = sample();
        let duplicate = value["cases"][0].clone();
        value["cases"].as_array_mut().unwrap().push(duplicate);
        assert!(load(&fixture(&value).0).is_err());
        let mut value = sample();
        value["cases"][0]["rows"] = json!(10);
        value["cases"][0]["hidden"] = json!(896);
        assert_eq!(load(&fixture(&value).0).err().unwrap().stage, "admission");
        let mut value = sample();
        value["cases"][0]["hidden"] = json!(33);
        assert_eq!(load(&fixture(&value).0).err().unwrap().stage, "fixture");
    }
    #[test]
    fn admission_counts_in_place_storage_without_a_third_buffer() {
        let file = fixture(&sample());
        let cases = load(&file.0).unwrap();
        assert_eq!(cases[0].state[32..], [STATE_GUARD; 7]);
        assert_eq!(cases[0].delta[32..], [DELTA_GUARD; 7]);
        let total = ALLOWANCE + 78 * 6;
        assert_eq!(admitted(&cases[0], total).unwrap(), total);
        assert!(admitted(&cases[0], total - 1).is_err());
        assert!(storage(usize::MAX).is_err());
        assert!(storage(0).is_err());
    }
}
