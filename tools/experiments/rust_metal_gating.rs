use super::*;
use std::{collections::HashSet, io::Read, path::Path};

const MAX_INPUT: u64 = 1024 * 1024;
const GATE_GUARD: u16 = 0x51aa;
const UP_GUARD: u16 = 0x529b;
const OUTPUT_GUARD: u16 = 0x55a1;
const _: () = assert!(size_of::<u32>() == 4 && align_of::<u32>() == 4);

struct Case {
    id: String,
    rows: usize,
    width: usize,
    gate: Vec<u16>,
    up: Vec<u16>,
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
        .ok_or_else(|| error("fixture", "missing gating Half bits"))?;
    if source.len() != count {
        return Err(error("fixture", "gating Half count differs"));
    }
    let mut values = Vec::with_capacity(count + GUARDS);
    for item in source {
        let number = item
            .as_u64()
            .and_then(|n| u16::try_from(n).ok())
            .ok_or_else(|| error("fixture", "gating bits outside UInt16"))?;
        if number & 0x7c00 == 0x7c00 {
            return Err(error("fixture", "nonfinite gating input Half"));
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
        return Err(error("fixture", "gating input exceeds one MiB"));
    }
    let mut data = Vec::new();
    file.take(MAX_INPUT + 1)
        .read_to_end(&mut data)
        .map_err(|e| error("fixture", e.to_string()))?;
    if data.len() as u64 != size || data.len() as u64 > MAX_INPUT {
        return Err(error(
            "fixture",
            "gating input changed size or exceeds bounds",
        ));
    }
    let value: Value =
        serde_json::from_slice(&data).map_err(|e| error("fixture", e.to_string()))?;
    if value["format"] != "toklet-metal-gating-input-v1" {
        return Err(error("fixture", "gating format differs"));
    }
    let sources = value["cases"]
        .as_array()
        .ok_or_else(|| error("fixture", "missing gating cases"))?;
    if sources.is_empty() || sources.len() > 32 {
        return Err(error("fixture", "gating cases outside 1..32"));
    }
    let mut names = HashSet::new();
    let mut elements = 0usize;
    let mut cases = Vec::with_capacity(sources.len());
    for source in sources {
        let id = source["id"]
            .as_str()
            .ok_or_else(|| error("fixture", "missing gating identity"))?;
        if id.is_empty() || id.len() > 128 || !id.is_ascii() || !names.insert(id) {
            return Err(error("fixture", "invalid or duplicate gating identity"));
        }
        let rows = scalar(source, "rows")?;
        let width = scalar(source, "width")?;
        if !(1..=32).contains(&rows) || !matches!(width, 64 | 4864) {
            return Err(error("fixture", "gating prototype shape outside bounds"));
        }
        let count = rows
            .checked_mul(width)
            .ok_or_else(|| error("admission", "gating shape overflow"))?;
        elements = elements
            .checked_add(count)
            .ok_or_else(|| error("admission", "gating inventory overflow"))?;
        if elements > 8192 {
            return Err(error("admission", "gating inventory exceeds 8192 values"));
        }
        cases.push(Case {
            id: id.to_owned(),
            rows,
            width,
            gate: bits(source, "gate_bits", count, GATE_GUARD)?,
            up: bits(source, "up_bits", count, UP_GUARD)?,
        });
    }
    // Parsing and byte temporaries drop before any native allocation.
    Ok(cases)
}
fn storage(count: usize) -> Result<usize, Error> {
    let bytes = count
        .checked_add(GUARDS)
        .and_then(|n| n.checked_mul(size_of::<u16>()))
        .ok_or_else(|| error("admission", "gating storage overflow"))?;
    if count == 0 || u32::try_from(count).is_err() || bytes > isize::MAX as usize {
        return Err(error("admission", "gating storage exceeds indexing bounds"));
    }
    Ok(bytes)
}
fn admitted(case: &Case, limit: usize) -> Result<usize, Error> {
    let count = case
        .rows
        .checked_mul(case.width)
        .ok_or_else(|| error("admission", "gating shape overflow"))?;
    // Three shared allocations, two host baselines and three readbacks. Bounded
    // retained fixture/JSON/repetition records use the fixed runtime allowance.
    let bytes = storage(count)?
        .checked_mul(8)
        .and_then(|n| n.checked_add(ALLOWANCE))
        .ok_or_else(|| error("admission", "gating working-set overflow"))?;
    if limit > LIMIT || bytes > limit {
        return Err(error("admission", "gating exceeds supplied budget"));
    }
    Ok(bytes)
}
fn execute(probe: &mut Probe, case: &Case, limit: usize) -> Result<Value, Error> {
    let budget = admitted(case, limit)?;
    let count = case.rows * case.width;
    let bytes = storage(count)?;
    let launch = groups(count, probe.width)?;
    let mut gate = Shared::<u16>::new(&probe.device, bytes)?;
    let mut up = Shared::<u16>::new(&probe.device, bytes)?;
    let mut output = Shared::<u16>::new(&probe.device, bytes)?;
    gate.write(&case.gate)?;
    up.write(&case.up)?;
    let native_count = count as u32;
    let mut repetitions = Vec::with_capacity(13);
    for repetition in 0..13 {
        output.fill(OUTPUT_GUARD);
        repetitions.push(autoreleasepool(|_| {
            let started = Instant::now();
            let command = probe.queue.commandBuffer().ok_or_else(|| error("command", "command unavailable"))?;
            if !command.retainedReferences() { return Err(error("command", "unretained resources unsupported")); }
            let encoder = command.computeCommandEncoder().ok_or_else(|| error("command", "encoder unavailable"))?;
            encoder.setComputePipelineState(&probe.pipeline);
            // SAFETY: three distinct owned Half allocations at slots0/1/2,
            // offset0, checked lengths/alignment; slot3 copies UInt32 count4/4.
            // Each lane reads two immutable operands and writes its own output;
            // rounded grid fits UInt32 and shader checks count before access.
            // Retained references and owners live through wait; no host access
            // from commit until successful completion/status/error checks.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&gate.raw), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&up.raw), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&output.raw), 0, 2);
                encoder.setBytes_length_atIndex(NonNull::from(&native_count).cast(), size_of::<u32>(), 3);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: launch, height: 1, depth: 1 }, MTLSize { width: probe.width, height: 1, depth: 1 });
            encoder.endEncoding(); command.commit(); command.waitUntilCompleted();
            let seconds = started.elapsed().as_secs_f64();
            if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
                return Err(error("completion", format!("gating GPU failure: {:?}", command.error())));
            }
            let actual = output.read();
            if gate.read() != case.gate || up.read() != case.up || actual[count..].iter().any(|&bits| bits != OUTPUT_GUARD) {
                return Err(error("validation", "gating inputs/output guards changed"));
            }
            if actual[..count].iter().any(|&bits| bits & 0x7c00 == 0x7c00) {
                return Err(error("validation", "completed gating contains nonfinite Half; result discarded"));
            }
            Ok(json!({"phase":if repetition<3 {"warmup"} else {"sample"},"repetition":if repetition<3 {repetition+1} else {repetition-2},
                "command_completed":true,"immutable_inputs_and_guards":true,"encode_commit_wait_seconds":seconds,"output_bits":actual}))
        })?);
    }
    Ok(
        json!({"id":case.id,"rows_count":case.rows,"width":case.width,"guards":GUARDS,"output_guard_bits":OUTPUT_GUARD,
        "threadgroup_width":probe.width,"threadgroups":launch,"shared_buffer_bytes":bytes*3,"host_input_and_readback_bytes":bytes*5,
        "admitted_bytes":budget,"distinct_owned_output":true,"allocations_reused_across_repetitions":true,"rows":repetitions}),
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
        .ok_or_else(|| error("admission", "empty gating cases"))?;
    autoreleasepool(|_| {
        let (mut probe, metadata) = prepare(
            include_str!("rust_metal_gating.metal"),
            "toklet_gating",
            maximum,
        )?;
        let mut results = Vec::with_capacity(cases.len());
        for case in &cases {
            results.push(autoreleasepool(|_| execute(&mut probe, case, limit))?);
        }
        Ok(
            json!({"operation":"gating_fp16_storage_fp32_product","metadata":metadata,"cases":results}),
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
            "toklet-gating-{}-{}.json",
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
        json!({"format":"toklet-metal-gating-input-v1","cases":[{"id":"manual","rows":1,"width":64,"gate_bits":vec![0x3c00u16;64],"up_bits":vec![0x0001u16;64]}]})
    }
    #[test]
    fn both_operands_are_finite_unsigned_half_before_device_work() {
        for key in ["gate_bits", "up_bits"] {
            for bit in [
                json!(0x7c00),
                json!(0xfc00),
                json!(0x7e00),
                json!(-1),
                json!(65536),
                json!(1.0),
            ] {
                let mut value = sample();
                value["cases"][0][key][63] = bit;
                assert!(load(&fixture(&value).0).is_err());
            }
        }
        let mut value = sample();
        value["cases"][0]["gate_bits"] = json!(vec![0x7bffu16; 64]);
        value["cases"][0]["up_bits"] = json!(vec![0x4000u16; 64]);
        assert!(load(&fixture(&value).0).is_ok());
        // Finite operands may overflow Half; completed output must be rejected.
    }
    #[test]
    fn shape_identity_and_total_inventory_are_bounded() {
        for (key, number) in [("rows", 0), ("rows", 33), ("width", 32)] {
            let mut value = sample();
            value["cases"][0][key] = json!(number);
            assert!(load(&fixture(&value).0).is_err());
        }
        for key in ["gate_bits", "up_bits"] {
            let mut value = sample();
            value["cases"][0][key] = json!(vec![0u16; 63]);
            assert!(load(&fixture(&value).0).is_err());
        }
        for name in ["", "non-ascii-λ"] {
            let mut value = sample();
            value["cases"][0]["id"] = json!(name);
            assert!(load(&fixture(&value).0).is_err());
        }
        let mut value = sample();
        let duplicate = value["cases"][0].clone();
        value["cases"].as_array_mut().unwrap().push(duplicate);
        assert!(load(&fixture(&value).0).is_err());
        let value = json!({"format":"toklet-metal-gating-input-v1","cases":(0..5).map(|i|json!({"id":format!("{i}"),"rows":32,"width":64,"gate_bits":vec![0u16;2048],"up_bits":vec![0u16;2048]})).collect::<Vec<_>>()});
        assert_eq!(load(&fixture(&value).0).err().unwrap().stage, "admission");
        let cases = load(&fixture(&sample()).0).unwrap();
        assert_eq!(cases[0].gate[64..], [GATE_GUARD; 7]);
        assert_eq!(cases[0].up[64..], [UP_GUARD; 7]);
    }
    #[test]
    fn budget_counts_three_shared_buffers_once_and_bounds_storage() {
        assert!(storage(0).is_err());
        assert!(storage(usize::MAX).is_err());
        let case = Case {
            id: String::new(),
            rows: 32,
            width: 64,
            gate: vec![0; 2055],
            up: vec![0; 2055],
        };
        let total = ALLOWANCE + 2055 * 2 * 8;
        assert_eq!(total, 67141744);
        assert_eq!(admitted(&case, total).unwrap(), total);
        assert!(admitted(&case, total - 1).is_err());
        assert!(admitted(&case, LIMIT + 1).is_err());
    }
    #[test]
    fn wide_rows_preserve_the_original_aggregate_inventory_limit() {
        let mut value = sample();
        value["cases"][0]["width"] = json!(4864);
        value["cases"][0]["gate_bits"] = json!(vec![0x3c00u16; 4864]);
        value["cases"][0]["up_bits"] = json!(vec![0x0001u16; 4864]);
        let cases = load(&fixture(&value).0).unwrap();
        assert_eq!(cases[0].gate.len(), 4871);
        assert_eq!(cases[0].up.len(), 4871);
        let budget = 67186800;
        assert_eq!(admitted(&cases[0], budget).unwrap(), budget);
        assert!(admitted(&cases[0], budget - 1).is_err());
        let duplicate = value["cases"][0].clone();
        value["cases"].as_array_mut().unwrap().push(duplicate);
        value["cases"][1]["id"] = json!("second");
        assert_eq!(load(&fixture(&value).0).err().unwrap().stage, "admission");
        value["cases"].as_array_mut().unwrap().pop();
        value["cases"][0]["rows"] = json!(2);
        value["cases"][0]["gate_bits"] = json!(vec![0u16; 9728]);
        value["cases"][0]["up_bits"] = json!(vec![0u16; 9728]);
        assert_eq!(load(&fixture(&value).0).err().unwrap().stage, "admission");
    }
    #[test]
    fn oversized_bytes_and_case_counts_fail_before_native_allocation() {
        let file = fixture(&sample());
        std::fs::write(&file.0, vec![b' '; (MAX_INPUT + 1) as usize]).unwrap();
        assert_eq!(load(&file.0).err().unwrap().stage, "fixture");
        for count in [0, 33] {
            let value = json!({"format":"toklet-metal-gating-input-v1","cases":(0..count).map(|i|json!({"id":format!("{i}"),"rows":1,"width":64,"gate_bits":vec![0u16;64],"up_bits":vec![0u16;64]})).collect::<Vec<_>>()});
            assert_eq!(load(&fixture(&value).0).err().unwrap().stage, "fixture");
        }
    }
}
