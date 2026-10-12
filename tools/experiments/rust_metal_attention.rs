use super::*;
use std::{io::Read, path::Path};
const HALF_GUARD: u16 = 0x55a1;
const FLOAT_GUARD: u32 = 0x4642fedc;
struct Case {
    id: String,
    rows: usize,
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
}
fn load(path: &Path) -> Result<Case, Error> {
    let file = std::fs::File::open(path).map_err(|e| error("fixture", e.to_string()))?;
    let size = file
        .metadata()
        .map_err(|e| error("fixture", e.to_string()))?
        .len();
    if size > 1024 * 1024 {
        return Err(error("fixture", "attention input exceeds one MiB"));
    }
    let mut data = Vec::new();
    file.take(1024 * 1024 + 1)
        .read_to_end(&mut data)
        .map_err(|e| error("fixture", e.to_string()))?;
    if data.len() as u64 != size || data.len() > 1024 * 1024 {
        return Err(error(
            "fixture",
            "attention input changed size or exceeds bounds",
        ));
    }
    let value: Value =
        serde_json::from_slice(&data).map_err(|e| error("fixture", e.to_string()))?;
    if value["format"] != "toklet-metal-attention-input-v1" {
        return Err(error("fixture", "attention format differs"));
    }
    let id = value["id"]
        .as_str()
        .ok_or_else(|| error("fixture", "missing attention identity"))?;
    if id.is_empty() || id.len() > 128 || !id.is_ascii() {
        return Err(error("fixture", "invalid attention identity"));
    }
    let rows = projection::number(&value, "rows")?;
    if !(1..=32).contains(&rows)
        || value["query_heads"].as_u64() != Some(4)
        || value["kv_heads"].as_u64() != Some(2)
        || value["dimension"].as_u64() != Some(8)
        || value["capacity"].as_u64() != Some(32)
    {
        return Err(error("fixture", "attention prototype shape outside bounds"));
    }
    Ok(Case {
        id: id.to_owned(),
        rows,
        q: projection::half(&value, "q_bits", rows * 32)?,
        k: projection::half(&value, "k_bits", 512)?,
        v: projection::half(&value, "v_bits", 512)?,
    })
}
fn mapped(rows: usize, width: usize) -> Result<(usize, usize), Error> {
    if !(1..=32).contains(&rows) {
        return Err(error("admission", "attention rows outside bounds"));
    }
    let units = groups(rows * 4, width)?
        .checked_mul(width)
        .ok_or_else(|| error("admission", "attention grid overflow"))?;
    let half = units
        .checked_mul(8)
        .and_then(|n| n.checked_add(GUARDS))
        .ok_or_else(|| error("admission", "attention Half storage overflow"))?;
    let single = units
        .checked_mul(rows)
        .and_then(|n| n.checked_add(GUARDS))
        .ok_or_else(|| error("admission", "attention FP32 storage overflow"))?;
    Ok((half, single))
}
fn admitted(rows: usize, width: usize, limit: usize) -> Result<usize, Error> {
    let (h, f) = mapped(rows, width)?;
    // Three immutable inputs and three outputs; shared once, full baselines
    // and input readbacks, plus retained first/current output frames. Bounded
    // diagnostic JSON/fixture records live under the fixed runtime allowance.
    let shared = h
        .checked_mul(4)
        .and_then(|n| n.checked_add(519 * 4))
        .and_then(|n| n.checked_add(f.checked_mul(8)?))
        .ok_or_else(|| error("admission", "attention bytes overflow"))?;
    let total = shared
        .checked_mul(3)
        .and_then(|n| n.checked_add(ALLOWANCE))
        .ok_or_else(|| error("admission", "attention admission overflow"))?;
    if limit > LIMIT || total > limit {
        return Err(error("admission", "attention exceeds supplied budget"));
    }
    Ok(total)
}
fn baseline(values: &[u16], size: usize) -> Vec<u16> {
    let mut output = vec![HALF_GUARD; size];
    output[..values.len()].copy_from_slice(values);
    output
}
#[derive(PartialEq)]
struct Frame {
    scores: Vec<u32>,
    probabilities: Vec<u32>,
    context: Vec<u16>,
}
fn frame(
    scores: &Shared<u32>,
    probabilities: &Shared<u32>,
    context: &Shared<u16>,
    rows: usize,
) -> Result<Frame, Error> {
    let result = Frame {
        scores: scores.read(),
        probabilities: probabilities.read(),
        context: context.read(),
    };
    let n = rows * rows * 4;
    let h = rows * 32;
    if result.scores[n..]
        .iter()
        .chain(&result.probabilities[n..])
        .any(|&v| v != FLOAT_GUARD)
        || result.context[h..].iter().any(|&v| v != HALF_GUARD)
    {
        return Err(error(
            "validation",
            "attention inactive or tail guards changed",
        ));
    }
    if result.scores[..n]
        .iter()
        .chain(&result.probabilities[..n])
        .any(|&v| !f32::from_bits(v).is_finite())
        || result.context[..h].iter().any(|&v| v & 0x7c00 == 0x7c00)
    {
        return Err(error(
            "validation",
            "completed attention nonfinite; result discarded",
        ));
    }
    Ok(result)
}
pub(super) fn run(path: &Path, limit: usize) -> Result<Value, Error> {
    let case = load(path)?;
    let initial = admitted(case.rows, 1, limit)?;
    autoreleasepool(|_| {
        let (probe, mut metadata) = prepare(
            include_str!("rust_metal_attention.metal"),
            "toklet_attention",
            initial,
        )?;
        let total = admitted(case.rows, probe.width, limit)?;
        metadata["maximum_admitted_bytes"] = json!(total);
        let (h, f) = mapped(case.rows, probe.width)?;
        let mut q = Shared::<u16>::new(&probe.device, h * 2)?;
        let mut k = Shared::<u16>::new(&probe.device, 519 * 2)?;
        let mut v = Shared::<u16>::new(&probe.device, 519 * 2)?;
        let mut scores = Shared::<u32>::new(&probe.device, f * 4)?;
        let mut probabilities = Shared::<u32>::new(&probe.device, f * 4)?;
        let mut context = Shared::<u16>::new(&probe.device, h * 2)?;
        let qb = baseline(&case.q, h);
        let kb = baseline(&case.k, 519);
        let vb = baseline(&case.v, 519);
        q.write(&qb)?;
        k.write(&kb)?;
        v.write(&vb)?;
        let native_rows = case.rows as u32;
        let mut first: Option<Frame> = None;
        let mut repetitions = Vec::with_capacity(13);
        for repetition in 0..13 {
            scores.fill(FLOAT_GUARD);
            probabilities.fill(FLOAT_GUARD);
            context.fill(HALF_GUARD);
            repetitions.push(autoreleasepool(|_|{
                let started=Instant::now();let command=probe.queue.commandBuffer().ok_or_else(||error("command","command unavailable"))?;
                if !command.retainedReferences(){return Err(error("command","unretained resources unsupported"));}
                let encoder=command.computeCommandEncoder().ok_or_else(||error("command","encoder unavailable"))?;encoder.setComputePipelineState(&probe.pipeline);
                // SAFETY: six distinct owned/tracked buffers at offset0. Half
                // inputs/output are2-byte aligned, raw FP32 output storage4-byte
                // aligned. Mapped sizes cover rounded query grid plus7 guards.
                // K/V cover2*32*8+7; bounded topology gives in-range head/position
                // addresses. Each lane owns a disjoint score/probability row and
                // context8-tuple; its ordered passes need no inter-lane barrier.
                // Slot6 copies UInt32 rows (4/4). Owners live through completion;
                // all CPU buffer access occurs before submit or after checked wait.
                unsafe{
                    encoder.setBuffer_offset_atIndex(Some(&q.raw),0,0);encoder.setBuffer_offset_atIndex(Some(&k.raw),0,1);encoder.setBuffer_offset_atIndex(Some(&v.raw),0,2);
                    encoder.setBuffer_offset_atIndex(Some(&scores.raw),0,3);encoder.setBuffer_offset_atIndex(Some(&probabilities.raw),0,4);encoder.setBuffer_offset_atIndex(Some(&context.raw),0,5);
                    encoder.setBytes_length_atIndex(NonNull::from(&native_rows).cast(),size_of::<u32>(),6);
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(MTLSize{width:groups(case.rows*4,probe.width)?,height:1,depth:1},MTLSize{width:probe.width,height:1,depth:1});
                encoder.endEncoding();command.commit();command.waitUntilCompleted();let seconds=started.elapsed().as_secs_f64();
                if command.status()!=MTLCommandBufferStatus::Completed || command.error().is_some(){return Err(error("completion",format!("attention GPU failure: {:?}",command.error())));}
                let current=frame(&scores,&probabilities,&context,case.rows)?;
                if q.read()!=qb || k.read()!=kb || v.read()!=vb{return Err(error("validation","attention immutable inputs or guards changed"));}
                if first.as_ref().is_some_and(|expected|expected!=&current){return Err(error("validation","attention repeated intermediate frame differs"));}
                let output=current.context[..case.rows*32].to_vec();if first.is_none(){first=Some(current);}
                Ok(json!({"phase":if repetition<3{"warmup"}else{"sample"},"repetition":if repetition<3{repetition+1}else{repetition-2},"command_completed":true,"inputs_guards_intermediates_valid":true,"commit_wait_pairs":1,"encode_commit_wait_seconds":seconds,"context_bits":output}))
            })?);
        }
        let first = first.ok_or_else(|| error("validation", "missing attention first frame"))?;
        let shared = h * 4 + 519 * 4 + f * 8;
        Ok(
            json!({"operation":"tiny_causal_gqa_fp32_half_context","id":case.id,"rows_count":case.rows,"query_heads":4,"kv_heads":2,"dimension":8,"capacity":32,"metadata":metadata,
            "threadgroup_width":probe.width,"threadgroups":groups(case.rows*4,probe.width)?,"half_storage_elements":h,"fp32_storage_elements":f,"half_guard_bits":HALF_GUARD,"fp32_guard_bits":FLOAT_GUARD,
            "shared_buffer_bytes":shared,"host_baseline_frames_readback_bytes":2*shared,"admitted_bytes":total,"allocations_reused":true,
            "first_score_bits":first.scores,"first_probability_bits":first.probabilities,"first_context_bits":first.context,"rows":repetitions}),
        )
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mapped_query_grid_and_admission_are_bounded() {
        assert_eq!(mapped(3, 32).unwrap(), (263, 103));
        assert_eq!(mapped(32, 32).unwrap(), (1031, 4103));
        for (rows, width) in [(0, 32), (33, 32), (1, 0), (32, usize::MAX)] {
            assert!(mapped(rows, width).is_err());
        }
        let total = ALLOWANCE + 3 * (263 * 4 + 519 * 4 + 103 * 8);
        assert_eq!(admitted(3, 32, total).unwrap(), total);
        assert!(admitted(3, 32, total - 1).is_err());
    }
    #[test]
    fn malformed_topology_and_nonfinite_storage_fail_before_device_work() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let sample = json!({"format":"toklet-metal-attention-input-v1","id":"manual","rows":1,"query_heads":4,"kv_heads":2,"dimension":8,"capacity":32,
            "q_bits":vec![0u16;32],"k_bits":vec![0u16;512],"v_bits":vec![0u16;512]});
        let check = |value: &Value| {
            let path = std::env::temp_dir().join(format!(
                "toklet-attention-{}-{}.json",
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
        };
        assert_eq!(check(&sample).unwrap().q.len(), 32);
        for (key, n) in [
            ("rows", 0),
            ("rows", 33),
            ("query_heads", 3),
            ("kv_heads", 4),
            ("dimension", 7),
            ("capacity", 33),
        ] {
            let mut value = sample.clone();
            value[key] = json!(n);
            assert!(check(&value).is_err());
        }
        for key in ["q_bits", "k_bits", "v_bits"] {
            let mut value = sample.clone();
            value[key][0] = json!(0x7c00);
            assert!(check(&value).is_err());
            value[key] = json!([0]);
            assert!(check(&value).is_err());
        }
    }
}
