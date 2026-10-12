use super::*;
use std::{collections::HashSet, io::Read, path::Path};

const MAX_INPUT: u64 = 1024 * 1024;
const INPUT_GUARD: u16 = 0x51aa;
const OUTPUT_GUARD: u16 = 0x55a1;
const _: () = assert!(size_of::<u32>() == 4 && align_of::<u32>() == 4);

struct Case {
    id: String,
    rows: usize,
    heads: usize,
    offset: usize,
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
        return Err(error("fixture", "rotary input exceeds one MiB"));
    }
    let mut data = Vec::new();
    file.take(MAX_INPUT + 1)
        .read_to_end(&mut data)
        .map_err(|e| error("fixture", e.to_string()))?;
    if data.len() as u64 != size || data.len() as u64 > MAX_INPUT {
        return Err(error(
            "fixture",
            "rotary input changed size or exceeds bounds",
        ));
    }
    let value: Value =
        serde_json::from_slice(&data).map_err(|e| error("fixture", e.to_string()))?;
    if value["format"] != "toklet-metal-rotary-input-v1" {
        return Err(error("fixture", "rotary format differs"));
    }
    let sources = value["cases"]
        .as_array()
        .ok_or_else(|| error("fixture", "missing rotary cases"))?;
    if sources.is_empty() || sources.len() > 32 {
        return Err(error("fixture", "rotary cases outside 1..32"));
    }
    let mut names = HashSet::new();
    let mut elements = 0usize;
    let mut cases = Vec::with_capacity(sources.len());
    for source in sources {
        let id = source["id"]
            .as_str()
            .ok_or_else(|| error("fixture", "missing rotary identity"))?;
        if id.is_empty() || id.len() > 128 || !id.is_ascii() || !names.insert(id) {
            return Err(error("fixture", "invalid or duplicate rotary identity"));
        }
        let rows = scalar(source, "rows")?;
        let heads = scalar(source, "heads")?;
        let offset = scalar(source, "offset")?;
        if !(1..=32).contains(&rows)
            || !matches!(heads, 2 | 4)
            || offset.checked_add(rows).is_none_or(|end| end > 2048)
            || source["dimension"].as_u64() != Some(8)
            || source["theta"].as_f64() != Some(1_000_000.0)
        {
            return Err(error("fixture", "rotary prototype shape outside bounds"));
        }
        let count = rows
            .checked_mul(heads * 8)
            .ok_or_else(|| error("admission", "rotary shape overflow"))?;
        elements = elements
            .checked_add(count)
            .ok_or_else(|| error("admission", "rotary inventory overflow"))?;
        if elements > 8192 {
            return Err(error("admission", "rotary inventory exceeds 8192 values"));
        }
        let source = source["input_bits"]
            .as_array()
            .ok_or_else(|| error("fixture", "missing rotary Half bits"))?;
        if source.len() != count {
            return Err(error("fixture", "rotary Half count differs"));
        }
        let mut input = Vec::with_capacity(count + GUARDS);
        for item in source {
            let bits = item
                .as_u64()
                .and_then(|n| u16::try_from(n).ok())
                .ok_or_else(|| error("fixture", "rotary bits outside UInt16"))?;
            if bits & 0x7c00 == 0x7c00 {
                return Err(error("fixture", "nonfinite rotary input Half"));
            }
            input.push(bits);
        }
        input.resize(count + GUARDS, INPUT_GUARD);
        cases.push(Case {
            id: id.to_owned(),
            rows,
            heads,
            offset,
            input,
        });
    }
    // Parsing and byte temporaries drop before any native allocation.
    Ok(cases)
}
fn storage(count: usize, width: usize) -> Result<usize, Error> {
    let bytes = groups(count / 2, width)?
        .checked_mul(width)
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(GUARDS))
        .and_then(|n| n.checked_mul(size_of::<u16>()))
        .ok_or_else(|| error("admission", "rotary storage overflow"))?;
    if count == 0
        || !count.is_multiple_of(2)
        || u32::try_from(count).is_err()
        || bytes > isize::MAX as usize
    {
        return Err(error("admission", "rotary storage exceeds indexing bounds"));
    }
    Ok(bytes)
}
fn admitted(case: &Case, limit: usize, width: usize) -> Result<usize, Error> {
    let count = case
        .rows
        .checked_mul(case.heads * 8)
        .ok_or_else(|| error("admission", "rotary shape overflow"))?;
    // Shared input/output, one host baseline, two diagnostic readbacks. Bounded
    // retained fixture/JSON/repetition records use the fixed runtime allowance.
    let bytes = storage(count, width)?
        .checked_mul(5)
        .and_then(|n| n.checked_add(ALLOWANCE))
        .ok_or_else(|| error("admission", "rotary working-set overflow"))?;
    if limit > LIMIT || bytes > limit {
        return Err(error("admission", "rotary exceeds supplied budget"));
    }
    Ok(bytes)
}
fn execute(probe: &mut Probe, case: &Case, limit: usize) -> Result<Value, Error> {
    let budget = admitted(case, limit, probe.width)?;
    let count = case.rows * case.heads * 8;
    let bytes = storage(count, probe.width)?;
    let launch = groups(count / 2, probe.width)?;
    let mut input = Shared::<u16>::new(&probe.device, bytes)?;
    let mut output = Shared::<u16>::new(&probe.device, bytes)?;
    let mut baseline = vec![INPUT_GUARD; bytes / 2];
    baseline[..count].copy_from_slice(&case.input[..count]);
    input.write(&baseline)?;
    let shape = [
        case.rows as u32,
        case.heads as u32,
        8u32,
        case.offset as u32,
    ];
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
            // checked lengths/alignment; slot2 copies four UInt32 values (16/4).
            // Each lane writes its two unique split-half components. Checked mapped
            // storage covers rounded pair grid; shader checks pair count first. Retained resources and owners live
            // through wait; host access starts only after completion checks.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&input.raw), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&output.raw), 0, 1);
                encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), size_of::<[u32; 4]>(), 2);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: launch, height: 1, depth: 1 }, MTLSize { width: probe.width, height: 1, depth: 1 });
            encoder.endEncoding(); command.commit(); command.waitUntilCompleted();
            let seconds = started.elapsed().as_secs_f64();
            if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
                return Err(error("completion", format!("rotary GPU failure: {:?}", command.error())));
            }
            let actual = output.read();
            if input.read() != baseline || actual[count..].iter().any(|&bits| bits != OUTPUT_GUARD) {
                return Err(error("validation", "rotary input/output guards changed"));
            }
            if actual[..count].iter().any(|&bits| bits & 0x7c00 == 0x7c00) {
                return Err(error("validation", "completed rotary contains nonfinite Half; result discarded"));
            }
            Ok(json!({"phase":if repetition<3 {"warmup"} else {"sample"},"repetition":if repetition<3 {repetition+1} else {repetition-2},
                "command_completed":true,"immutable_input_and_guards":true,"encode_commit_wait_seconds":seconds,"output_bits":actual}))
        })?);
    }
    Ok(
        json!({"id":case.id,"rows_count":case.rows,"heads":case.heads,"dimension":8,"theta":1_000_000.0,"offset":case.offset,"pairing":"split_half","guards":bytes / 2 - count,"output_guard_bits":OUTPUT_GUARD,
        "threadgroup_width":probe.width,"threadgroups":launch,"shared_buffer_bytes":bytes*2,"host_input_and_readback_bytes":bytes*3,
        "admitted_bytes":budget,"allocations_reused_across_repetitions":true,"rows":repetitions}),
    )
}
pub(super) fn run(path: &Path, limit: usize) -> Result<Value, Error> {
    let cases = load(path)?;
    let maximum = cases
        .iter()
        .map(|case| admitted(case, limit, 1))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or_else(|| error("admission", "empty rotary cases"))?;
    autoreleasepool(|_| {
        let (mut probe, mut metadata) = prepare(
            include_str!("rust_metal_rotary.metal"),
            "toklet_rotary",
            maximum,
        )?;
        let maximum = cases
            .iter()
            .map(|case| admitted(case, limit, probe.width))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .ok_or_else(|| error("admission", "empty rotary cases"))?;
        metadata["maximum_admitted_bytes"] = json!(maximum);
        let mut results = Vec::with_capacity(cases.len());
        for case in &cases {
            results.push(autoreleasepool(|_| execute(&mut probe, case, limit))?);
        }
        Ok(
            json!({"operation":"rotary_split_half_fp32_half_store","metadata":metadata,"cases":results}),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    fn load_value(value: &Value) -> Result<Vec<Case>, Error> {
        let path = std::env::temp_dir().join(format!(
            "toklet-rotary-{}-{}.json",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        writeln!(file, "{value}").unwrap();
        let result = load(&path);
        std::fs::remove_file(path).unwrap();
        result
    }
    fn sample() -> Value {
        json!({"format":"toklet-metal-rotary-input-v1","cases":[{"id":"manual","rows":1,"heads":2,"dimension":8,"theta":1_000_000.0,"offset":2047,"input_bits":vec![0x3c00u16;16]}]})
    }
    #[test]
    fn parser_bounds_positions_pairing_topology_and_finite_bits() {
        assert_eq!(load_value(&sample()).unwrap()[0].offset, 2047);
        for (key, value) in [
            ("rows", json!(0)),
            ("rows", json!(33)),
            ("heads", json!(3)),
            ("dimension", json!(7)),
            ("offset", json!(2048)),
            ("offset", json!(u64::MAX)),
            ("offset", json!(-1)),
            ("theta", json!(10000)),
            ("theta", json!("1000000")),
        ] {
            let mut source = sample();
            source["cases"][0][key] = value;
            assert!(load_value(&source).is_err());
        }
        for bit in [
            json!(0x7c00),
            json!(0x7e00),
            json!(0xfc00),
            json!(-1),
            json!(65536),
            json!(1.0),
        ] {
            let mut source = sample();
            source["cases"][0]["input_bits"][0] = bit;
            assert!(load_value(&source).is_err());
        }
        let mut source = sample();
        source["cases"][0]["input_bits"] = json!([0]);
        assert!(load_value(&source).is_err());
        let item = sample()["cases"][0].clone();
        source = sample();
        source["cases"].as_array_mut().unwrap().push(item);
        assert!(load_value(&source).is_err());
    }
    #[test]
    fn mapped_pair_grid_and_memory_arithmetic_are_checked() {
        assert_eq!(storage(16, 32).unwrap(), (64 + 7) * 2);
        assert_eq!(storage(64, 32).unwrap(), (64 + 7) * 2);
        for (count, width) in [(0, 32), (15, 32), (16, 0), (usize::MAX, 32)] {
            assert!(storage(count, width).is_err());
        }
        let case = Case {
            id: String::new(),
            rows: 1,
            heads: 2,
            offset: 0,
            input: vec![0; 23],
        };
        let total = ALLOWANCE + (64 + 7) * 2 * 5;
        assert_eq!(admitted(&case, total, 32).unwrap(), total);
        assert!(admitted(&case, total - 1, 32).is_err());
        assert!(admitted(&case, LIMIT + 1, 32).is_err());
    }
}
