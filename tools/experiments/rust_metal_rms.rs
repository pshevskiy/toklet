use super::*;
use std::{collections::HashSet, io::Read, path::Path};

const MAX_INPUT_BYTES: u64 = 1024 * 1024;
const MAX_ELEMENTS: usize = 8192;
const INPUT_GUARD: u16 = 0x51aa;
const WEIGHT_GUARD: u16 = 0x529b;
const OUTPUT_GUARD: u16 = 0x55a1;

#[repr(C)]
struct Params {
    rows: u32,
    hidden: u32,
    epsilon_bits: u32,
}
const _: () = assert!(size_of::<Params>() == 12 && align_of::<Params>() == 4);

struct Case {
    id: String,
    rows: usize,
    hidden: usize,
    epsilon_bits: u32,
    input: Vec<u16>,
    weight: Vec<u16>,
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
        .ok_or_else(|| error("fixture", "missing Half bits"))?;
    if source.len() != count {
        return Err(error("fixture", "Half count differs"));
    }
    let mut values = Vec::with_capacity(count + GUARDS);
    for item in source {
        let number = item
            .as_u64()
            .and_then(|n| u16::try_from(n).ok())
            .ok_or_else(|| error("fixture", "Half bits outside UInt16"))?;
        if number & 0x7c00 == 0x7c00 {
            return Err(error("fixture", "nonfinite stored Half"));
        }
        values.push(number);
    }
    values.resize(count + GUARDS, guard);
    Ok(values)
}
fn load(path: &Path) -> Result<Vec<Case>, Error> {
    let file = std::fs::File::open(path).map_err(|e| error("fixture", e.to_string()))?;
    let length = file
        .metadata()
        .map_err(|e| error("fixture", e.to_string()))?
        .len();
    if length > MAX_INPUT_BYTES {
        return Err(error("fixture", "RMS input exceeds one MiB"));
    }
    let mut payload = Vec::new();
    file.take(MAX_INPUT_BYTES + 1)
        .read_to_end(&mut payload)
        .map_err(|e| error("fixture", e.to_string()))?;
    if payload.len() as u64 != length || payload.len() as u64 > MAX_INPUT_BYTES {
        return Err(error("fixture", "RMS input changed size or exceeds limit"));
    }
    let value: Value =
        serde_json::from_slice(&payload).map_err(|e| error("fixture", e.to_string()))?;
    if value["format"] != "toklet-metal-rms-input-v1" {
        return Err(error("fixture", "RMS format differs"));
    }
    let sources = value["cases"]
        .as_array()
        .ok_or_else(|| error("fixture", "missing RMS cases"))?;
    if sources.is_empty() || sources.len() > 32 {
        return Err(error("fixture", "RMS case count outside 1..32"));
    }
    let mut names = HashSet::new();
    let mut elements = 0usize;
    let mut cases = Vec::with_capacity(sources.len());
    for source in sources {
        let id = source["id"]
            .as_str()
            .ok_or_else(|| error("fixture", "missing RMS identity"))?;
        if id.is_empty() || id.len() > 128 || !id.is_ascii() || !names.insert(id) {
            return Err(error("fixture", "invalid or duplicate RMS identity"));
        }
        let rows = scalar(source, "rows")?;
        let hidden = scalar(source, "hidden")?;
        if !(1..=32).contains(&rows) || ![32, 896].contains(&hidden) {
            return Err(error(
                "fixture",
                "RMS prototype shape outside validation bounds",
            ));
        }
        let count = rows
            .checked_mul(hidden)
            .ok_or_else(|| error("admission", "RMS shape overflow"))?;
        elements = elements
            .checked_add(count)
            .ok_or_else(|| error("admission", "RMS inventory overflow"))?;
        if elements > MAX_ELEMENTS {
            return Err(error(
                "admission",
                "RMS diagnostic inventory exceeds 8192 values",
            ));
        }
        let epsilon_bits = u32::try_from(scalar(source, "epsilon_bits")?)
            .map_err(|_| error("fixture", "epsilon bits overflow"))?;
        let epsilon = f32::from_bits(epsilon_bits);
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(error("fixture", "epsilon must be finite and positive"));
        }
        cases.push(Case {
            id: id.to_owned(),
            rows,
            hidden,
            epsilon_bits,
            input: bits(source, "input_bits", count, INPUT_GUARD)?,
            weight: bits(source, "weight_bits", hidden, WEIGHT_GUARD)?,
        });
    }
    // Payload/JSON temporaries are released before any native allocation.
    Ok(cases)
}
fn half_bytes(count: usize) -> Result<usize, Error> {
    let bytes = count
        .checked_add(GUARDS)
        .and_then(|n| n.checked_mul(size_of::<u16>()))
        .ok_or_else(|| error("admission", "Half byte count overflow"))?;
    if count == 0 || bytes > isize::MAX as usize {
        return Err(error("admission", "Half count exceeds pointer bounds"));
    }
    Ok(bytes)
}
fn admitted(case: &Case, limit: usize) -> Result<usize, Error> {
    // Input: shared + host + readback; output: shared + readback; weights:
    // shared + host + readback. Fixed bounded fixture/diagnostics use allowance.
    let bytes = half_bytes(case.rows * case.hidden)?
        .checked_mul(5)
        .and_then(|n| half_bytes(case.hidden).ok()?.checked_mul(3)?.checked_add(n))
        .and_then(|n| n.checked_add(ALLOWANCE))
        .ok_or_else(|| error("admission", "RMS working-set overflow"))?;
    if limit > LIMIT || bytes > limit {
        return Err(error("admission", "RMS exceeds supplied budget"));
    }
    Ok(bytes)
}
fn execute(probe: &mut Probe, case: &Case, limit: usize) -> Result<Value, Error> {
    let admitted = admitted(case, limit)?;
    let count = case.rows * case.hidden;
    let input_bytes = half_bytes(count)?;
    let weight_bytes = half_bytes(case.hidden)?;
    let launch = groups(case.rows, probe.width)?;
    let mut input = Shared::<u16>::new(&probe.device, input_bytes)?;
    let mut weight = Shared::<u16>::new(&probe.device, weight_bytes)?;
    let mut output = Shared::<u16>::new(&probe.device, input_bytes)?;
    input.write(&case.input)?;
    weight.write(&case.weight)?;
    let params = Params {
        rows: case.rows as u32,
        hidden: case.hidden as u32,
        epsilon_bits: case.epsilon_bits,
    };
    let mut rows = Vec::with_capacity(13);
    for repetition in 0..13 {
        output.fill(OUTPUT_GUARD);
        rows.push(autoreleasepool(|_| {
            let started = Instant::now();
            let command = probe
                .queue
                .commandBuffer()
                .ok_or_else(|| error("command", "command buffer unavailable"))?;
            if !command.retainedReferences() {
                return Err(error("command", "unretained resources unsupported"));
            }
            let encoder = command
                .computeCommandEncoder()
                .ok_or_else(|| error("command", "compute encoder unavailable"))?;
            encoder.setComputePipelineState(&probe.pipeline);
            // SAFETY: shader uses readonly Half input[rows,hidden] at0 and
            // weight[hidden] at1, writable Half output[rows,hidden] at2 and
            // the repr(C) 12-byte/4-aligned UInt32 params copied at3. Exact
            // lengths/alignment checked by Shared<u16>; bit payloads finite.
            // All shader products/rounded row indices fit UInt32 under load's
            // bounded inventory. Tail rows return before access. Distinct
            // private owners and retained command references live through the
            // wait. Host access resumes only after successful completion.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&input.raw), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&weight.raw), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&output.raw), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&params).cast(),
                    size_of::<Params>(),
                    3,
                );
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: launch,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: probe.width,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            command.commit();
            command.waitUntilCompleted();
            let seconds = started.elapsed().as_secs_f64();
            if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
                return Err(error(
                    "completion",
                    format!("RMS GPU failure: {:?}", command.error()),
                ));
            }
            let actual = output.read();
            if input.read() != case.input || weight.read() != case.weight {
                return Err(error("validation", "RMS input/weight or guards changed"));
            }
            if actual[count..].iter().any(|&bits| bits != OUTPUT_GUARD) {
                return Err(error("validation", "RMS output guard changed"));
            }
            if actual[..count].iter().any(|&bits| bits & 0x7c00 == 0x7c00) {
                return Err(error("arithmetic", "RMS output is not finite Half"));
            }
            Ok(json!({"phase":if repetition<3 {"warmup"}else{"sample"},
                "repetition":if repetition<3 {repetition+1}else{repetition-2},
                "command_completed":true,"immutable_input_weight_and_guards":true,
                "encode_commit_wait_seconds":seconds,"output_bits":actual}))
        })?);
    }
    Ok(
        json!({"id":case.id,"rows_count":case.rows,"hidden":case.hidden,
        "epsilon_bits":case.epsilon_bits,"guards":GUARDS,"output_guard_bits":OUTPUT_GUARD,
        "shared_buffer_bytes":input_bytes*2+weight_bytes,
        "host_upload_and_readback_bytes":input_bytes*3+weight_bytes*2,
        "admitted_bytes":admitted,"threadgroup_width":probe.width,"threadgroups":launch,
        "allocations_reused_across_repetitions":true,"rows":rows}),
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
        .ok_or_else(|| error("admission", "empty cases"))?;
    autoreleasepool(|_| {
        let (mut probe, metadata) = prepare(
            include_str!("rust_metal_rms.metal"),
            "toklet_rms_norm",
            maximum,
        )?;
        let mut results = Vec::with_capacity(cases.len());
        for case in &cases {
            results.push(autoreleasepool(|_| execute(&mut probe, case, limit))?);
        }
        Ok(
            json!({"metadata":metadata,"operation":"rms_norm_fp16_storage_fp32_serial", "cases":results}),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    struct Temporary(PathBuf);
    use std::path::PathBuf;
    impl Drop for Temporary {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    fn fixture(value: &Value) -> Temporary {
        let path = std::env::temp_dir().join(format!(
            "toklet-rms-{}-{}.json",
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
        json!({"format":"toklet-metal-rms-input-v1","cases":[{"id":"manual","rows":1,"hidden":32,
            "epsilon_bits":1.0e-6f32.to_bits(),"input_bits":vec![0x3c00u16;32],"weight_bits":vec![0x3c00u16;32]}]})
    }
    #[test]
    fn artifact_rejects_nonfinite_bits_and_invalid_epsilon_before_native_work() {
        for variant in ["input", "weight", "epsilon"] {
            let mut value = sample();
            match variant {
                "input" => value["cases"][0]["input_bits"][0] = json!(0x7c00),
                "weight" => value["cases"][0]["weight_bits"][0] = json!(0x7e00),
                _ => value["cases"][0]["epsilon_bits"] = json!(0),
            }
            let file = fixture(&value);
            assert_eq!(load(&file.0).err().unwrap().stage, "fixture");
        }
    }
    #[test]
    fn inventory_rejects_duplicate_names_and_oversized_shapes() {
        let mut value = sample();
        let duplicate = value["cases"][0].clone();
        value["cases"].as_array_mut().unwrap().push(duplicate);
        let file = fixture(&value);
        assert_eq!(load(&file.0).err().unwrap().stage, "fixture");
        let mut value = sample();
        value["cases"][0]["rows"] = json!(33);
        let file = fixture(&value);
        assert_eq!(load(&file.0).err().unwrap().stage, "fixture");
    }
    #[test]
    fn half_layout_and_budget_admission_include_all_buffers() {
        let file = fixture(&sample());
        let cases = load(&file.0).unwrap();
        assert_eq!(cases[0].input.len(), 39);
        assert_eq!(cases[0].input[32..], [INPUT_GUARD; 7]);
        assert_eq!(cases[0].weight[32..], [WEIGHT_GUARD; 7]);
        let total = ALLOWANCE + 78 * 8;
        assert_eq!(admitted(&cases[0], total).unwrap(), total);
        assert!(admitted(&cases[0], total - 1).is_err());
        assert!(half_bytes(usize::MAX).is_err());
    }
}
