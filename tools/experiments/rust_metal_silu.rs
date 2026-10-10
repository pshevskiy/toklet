use super::*;
use std::{collections::HashSet, io::Read, path::Path};

const MAX_INPUT: u64 = 1024 * 1024;
const INPUT_GUARD: u16 = 0x51aa;
const OUTPUT_GUARD: u16 = 0x55a1;
const _: () = assert!(size_of::<u32>() == 4 && align_of::<u32>() == 4);

struct Case {
    id: String,
    rows: usize,
    width: usize,
    input: Vec<u16>,
}
fn scalar(value: &Value, name: &str) -> Result<usize, Error> {
    value[name]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| error("fixture", format!("invalid unsigned {name}")))
}
fn load(path: &Path) -> Result<Vec<Case>, Error> {
    let file = std::fs::File::open(path).map_err(|e| error("fixture", e.to_string()))?;
    let size = file
        .metadata()
        .map_err(|e| error("fixture", e.to_string()))?
        .len();
    if size > MAX_INPUT {
        return Err(error("fixture", "SiLU input exceeds one MiB"));
    }
    let mut data = Vec::new();
    file.take(MAX_INPUT + 1)
        .read_to_end(&mut data)
        .map_err(|e| error("fixture", e.to_string()))?;
    if data.len() as u64 != size || data.len() as u64 > MAX_INPUT {
        return Err(error(
            "fixture",
            "SiLU input changed size or exceeds bounds",
        ));
    }
    let value: Value =
        serde_json::from_slice(&data).map_err(|e| error("fixture", e.to_string()))?;
    if value["format"] != "toklet-metal-silu-input-v1" {
        return Err(error("fixture", "SiLU format differs"));
    }
    let sources = value["cases"]
        .as_array()
        .ok_or_else(|| error("fixture", "missing SiLU cases"))?;
    if sources.is_empty() || sources.len() > 32 {
        return Err(error("fixture", "SiLU cases outside 1..32"));
    }
    let mut names = HashSet::new();
    let mut elements = 0usize;
    let mut cases = Vec::with_capacity(sources.len());
    for source in sources {
        let id = source["id"]
            .as_str()
            .ok_or_else(|| error("fixture", "missing SiLU identity"))?;
        if id.is_empty() || id.len() > 128 || !id.is_ascii() || !names.insert(id) {
            return Err(error("fixture", "invalid or duplicate SiLU identity"));
        }
        let rows = scalar(source, "rows")?;
        let width = scalar(source, "width")?;
        if !(1..=32).contains(&rows) || width != 64 {
            return Err(error("fixture", "SiLU prototype shape outside bounds"));
        }
        let count = rows
            .checked_mul(width)
            .ok_or_else(|| error("admission", "SiLU shape overflow"))?;
        elements = elements
            .checked_add(count)
            .ok_or_else(|| error("admission", "SiLU inventory overflow"))?;
        if elements > 8192 {
            return Err(error("admission", "SiLU inventory exceeds 8192 values"));
        }
        let source = source["input_bits"]
            .as_array()
            .ok_or_else(|| error("fixture", "missing SiLU Half bits"))?;
        if source.len() != count {
            return Err(error("fixture", "SiLU Half count differs"));
        }
        let mut input = Vec::with_capacity(count + GUARDS);
        for item in source {
            let bits = item
                .as_u64()
                .and_then(|n| u16::try_from(n).ok())
                .ok_or_else(|| error("fixture", "SiLU bits outside UInt16"))?;
            if bits & 0x7c00 == 0x7c00 {
                return Err(error("fixture", "nonfinite SiLU input Half"));
            }
            input.push(bits);
        }
        input.resize(count + GUARDS, INPUT_GUARD);
        cases.push(Case {
            id: id.to_owned(),
            rows,
            width,
            input,
        });
    }
    // Parsing and byte temporaries drop before any native allocation.
    Ok(cases)
}
fn storage(count: usize) -> Result<usize, Error> {
    let bytes = count
        .checked_add(GUARDS)
        .and_then(|n| n.checked_mul(size_of::<u16>()))
        .ok_or_else(|| error("admission", "SiLU storage overflow"))?;
    if count == 0 || u32::try_from(count).is_err() || bytes > isize::MAX as usize {
        return Err(error("admission", "SiLU storage exceeds indexing bounds"));
    }
    Ok(bytes)
}
fn admitted(case: &Case, limit: usize) -> Result<usize, Error> {
    let count = case
        .rows
        .checked_mul(case.width)
        .ok_or_else(|| error("admission", "SiLU shape overflow"))?;
    // Shared input/output, one host baseline, two diagnostic readbacks. Bounded
    // retained fixture/JSON/repetition records use the fixed runtime allowance.
    let bytes = storage(count)?
        .checked_mul(5)
        .and_then(|n| n.checked_add(ALLOWANCE))
        .ok_or_else(|| error("admission", "SiLU working-set overflow"))?;
    if limit > LIMIT || bytes > limit {
        return Err(error("admission", "SiLU exceeds supplied budget"));
    }
    Ok(bytes)
}
fn execute(probe: &mut Probe, case: &Case, limit: usize) -> Result<Value, Error> {
    let budget = admitted(case, limit)?;
    let count = case.rows * case.width;
    let bytes = storage(count)?;
    let launch = groups(count, probe.width)?;
    let mut input = Shared::<u16>::new(&probe.device, bytes)?;
    let mut output = Shared::<u16>::new(&probe.device, bytes)?;
    input.write(&case.input)?;
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
            // SAFETY: distinct owned Half input/output at slots0/1, offset0,
            // checked lengths/alignment; slot2 copies a UInt32 count (4/4).
            // Each lane writes one output; rounded grid fits UInt32 and shader
            // checks count before access. Retained resources and owners live
            // through wait; host access starts only after completion checks.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&input.raw), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&output.raw), 0, 1);
                encoder.setBytes_length_atIndex(NonNull::from(&native_count).cast(), size_of::<u32>(), 2);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: launch, height: 1, depth: 1 }, MTLSize { width: probe.width, height: 1, depth: 1 });
            encoder.endEncoding(); command.commit(); command.waitUntilCompleted();
            let seconds = started.elapsed().as_secs_f64();
            if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
                return Err(error("completion", format!("SiLU GPU failure: {:?}", command.error())));
            }
            let actual = output.read();
            if input.read() != case.input || actual[count..].iter().any(|&bits| bits != OUTPUT_GUARD) {
                return Err(error("validation", "SiLU input/output guards changed"));
            }
            if actual[..count].iter().any(|&bits| bits & 0x7c00 == 0x7c00) {
                return Err(error("validation", "completed SiLU contains nonfinite Half; result discarded"));
            }
            Ok(json!({"phase":if repetition<3 {"warmup"} else {"sample"},"repetition":if repetition<3 {repetition+1} else {repetition-2},
                "command_completed":true,"immutable_input_and_guards":true,"encode_commit_wait_seconds":seconds,"output_bits":actual}))
        })?);
    }
    Ok(
        json!({"id":case.id,"rows_count":case.rows,"width":case.width,"guards":GUARDS,"output_guard_bits":OUTPUT_GUARD,
        "threadgroup_width":probe.width,"threadgroups":launch,"shared_buffer_bytes":bytes*2,"host_input_and_readback_bytes":bytes*3,
        "admitted_bytes":budget,"allocations_reused_across_repetitions":true,"rows":repetitions}),
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
        .ok_or_else(|| error("admission", "empty SiLU cases"))?;
    autoreleasepool(|_| {
        let (mut probe, metadata) = prepare(
            include_str!("rust_metal_silu.metal"),
            "toklet_silu",
            maximum,
        )?;
        let mut results = Vec::with_capacity(cases.len());
        for case in &cases {
            results.push(autoreleasepool(|_| execute(&mut probe, case, limit))?);
        }
        Ok(json!({"operation":"silu_fp16_storage_stable_fp32","metadata":metadata,"cases":results}))
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
            "toklet-silu-{}-{}.json",
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
        json!({"format":"toklet-metal-silu-input-v1","cases":[{"id":"manual","rows":1,"width":64,"input_bits":vec![0x3c00u16;64]}]})
    }
    #[test]
    fn rejects_nonfinite_unsigned_shape_and_inventory_before_device_work() {
        for bit in [
            json!(0x7c00),
            json!(0xfc00),
            json!(0x7e00),
            json!(-1),
            json!(65536),
            json!(1.0),
        ] {
            let mut value = sample();
            value["cases"][0]["input_bits"][0] = bit;
            assert!(load(&fixture(&value).0).is_err());
        }
        for (key, number) in [("rows", 0), ("rows", 33), ("width", 32)] {
            let mut value = sample();
            value["cases"][0][key] = json!(number);
            assert!(load(&fixture(&value).0).is_err());
        }
        let mut value = sample();
        value["cases"][0]["input_bits"] = json!(vec![0u16; 63]);
        assert!(load(&fixture(&value).0).is_err());
        let mut value = sample();
        let item = value["cases"][0].clone();
        value["cases"].as_array_mut().unwrap().push(item);
        assert!(load(&fixture(&value).0).is_err());
        let value = json!({"format":"toklet-metal-silu-input-v1","cases":(0..5).map(|i|json!({"id":format!("{i}"),"rows":32,"width":64,"input_bits":vec![0u16;2048]})).collect::<Vec<_>>()});
        assert!(load(&fixture(&value).0).is_err());
        assert_eq!(load(&fixture(&sample()).0).unwrap()[0].input.len(), 71);
    }
    #[test]
    fn checked_storage_and_budget_count_shared_memory_once() {
        assert!(storage(0).is_err());
        assert!(storage(usize::MAX).is_err());
        let case = Case {
            id: String::new(),
            rows: 1,
            width: 64,
            input: vec![0; 71],
        };
        let total = ALLOWANCE + 71 * 2 * 5;
        assert_eq!(admitted(&case, total).unwrap(), total);
        assert!(admitted(&case, total - 1).is_err());
        assert!(admitted(&case, LIMIT + 1).is_err());
    }
}
