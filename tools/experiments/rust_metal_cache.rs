use super::*;
use std::{collections::HashSet, io::Read, path::Path};
const MAX_INPUT: u64 = 1024 * 1024;
const INPUT_GUARD: u16 = 0x51aa;
const K_GUARD: u16 = 0x55a1;
const V_GUARD: u16 = 0x529b;
const CAPACITY: usize = 32;
const WIDTH: usize = 16;
const CACHE_COUNT: usize = CAPACITY * WIDTH;
struct Case {
    id: String,
    rows: usize,
    offset: usize,
    k: Vec<u16>,
    v: Vec<u16>,
    initial_k: Vec<u16>,
    initial_v: Vec<u16>,
}
fn load(path: &Path) -> Result<Vec<Case>, Error> {
    let file = std::fs::File::open(path).map_err(|e| error("fixture", e.to_string()))?;
    let size = file
        .metadata()
        .map_err(|e| error("fixture", e.to_string()))?
        .len();
    if size > MAX_INPUT {
        return Err(error("fixture", "cache input exceeds one MiB"));
    }
    let mut data = Vec::new();
    file.take(MAX_INPUT + 1)
        .read_to_end(&mut data)
        .map_err(|e| error("fixture", e.to_string()))?;
    if data.len() as u64 != size || data.len() as u64 > MAX_INPUT {
        return Err(error(
            "fixture",
            "cache input changed size or exceeds bounds",
        ));
    }
    let value: Value =
        serde_json::from_slice(&data).map_err(|e| error("fixture", e.to_string()))?;
    if value["format"] != "toklet-metal-cache-input-v1" {
        return Err(error("fixture", "cache format differs"));
    }
    let sources = value["cases"]
        .as_array()
        .ok_or_else(|| error("fixture", "missing cache cases"))?;
    if sources.is_empty() || sources.len() > 32 {
        return Err(error("fixture", "cache cases outside1..32"));
    }
    let mut cases = Vec::with_capacity(sources.len());
    let mut names = HashSet::new();
    let mut inventory = 0usize;
    for source in sources {
        let id = source["id"]
            .as_str()
            .ok_or_else(|| error("fixture", "missing cache identity"))?;
        if id.is_empty() || id.len() > 128 || !id.is_ascii() || !names.insert(id) {
            return Err(error("fixture", "invalid or duplicate cache identity"));
        }
        let rows = projection::number(source, "rows")?;
        let offset = projection::number(source, "offset")?;
        if !(1..=32).contains(&rows)
            || offset.checked_add(rows).is_none_or(|end| end > CAPACITY)
            || source["heads"].as_u64() != Some(2)
            || source["dimension"].as_u64() != Some(8)
            || source["capacity"].as_u64() != Some(CAPACITY as u64)
        {
            return Err(error(
                "fixture",
                "cache prototype shape/position outside bounds",
            ));
        }
        let count = rows
            .checked_mul(WIDTH)
            .ok_or_else(|| error("admission", "cache input size overflow"))?;
        inventory = inventory
            .checked_add(2 * count + 2 * CACHE_COUNT)
            .ok_or_else(|| error("admission", "cache inventory overflow"))?;
        if inventory > 32768 {
            return Err(error("admission", "cache inventory exceeds32768 values"));
        }
        cases.push(Case {
            id: id.to_owned(),
            rows,
            offset,
            k: projection::half(source, "k_bits", count)?,
            v: projection::half(source, "v_bits", count)?,
            initial_k: projection::half(source, "initial_k_bits", CACHE_COUNT)?,
            initial_v: projection::half(source, "initial_v_bits", CACHE_COUNT)?,
        });
    }
    Ok(cases)
}
fn storage(count: usize, width: usize) -> Result<usize, Error> {
    groups(count, width)?
        .checked_mul(width)
        .and_then(|n| n.checked_add(GUARDS))
        .and_then(|n| n.checked_mul(2))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| error("admission", "cache mapped storage overflow"))
}
fn admitted(case: &Case, limit: usize, width: usize) -> Result<usize, Error> {
    let count = case
        .rows
        .checked_mul(WIDTH)
        .ok_or_else(|| error("admission", "cache count overflow"))?;
    let shared = storage(count, width)?
        .checked_mul(2)
        .and_then(|n| n.checked_add((CACHE_COUNT + GUARDS) * 2 * 2))
        .ok_or_else(|| error("admission", "cache shared overflow"))?;
    let total = shared
        .checked_mul(3)
        .and_then(|n| n.checked_add(ALLOWANCE))
        .ok_or_else(|| error("admission", "cache admission overflow"))?;
    if limit > LIMIT || total > limit {
        return Err(error("admission", "cache exceeds supplied budget"));
    }
    Ok(total)
}
fn baseline(values: &[u16], count: usize, guard: u16) -> Vec<u16> {
    let mut output = vec![guard; count];
    output[..values.len()].copy_from_slice(values);
    output
}
fn valid_cache(actual: &[u16], initial: &[u16], chunk: &[u16], case: &Case, guard: u16) -> bool {
    if actual.len() != CACHE_COUNT + GUARDS || actual[CACHE_COUNT..].iter().any(|&n| n != guard) {
        return false;
    }
    for (index, &value) in actual[..CACHE_COUNT].iter().enumerate() {
        let head = index / (CAPACITY * 8);
        let position = (index / 8) % CAPACITY;
        let feature = index % 8;
        let expected = if (case.offset..case.offset + case.rows).contains(&position) {
            chunk[((position - case.offset) * 2 + head) * 8 + feature]
        } else {
            initial[index]
        };
        if value != expected {
            return false;
        }
    }
    true
}
fn execute(probe: &mut Probe, case: &Case, limit: usize) -> Result<Value, Error> {
    let count = case.rows * WIDTH;
    let input_bytes = storage(count, probe.width)?;
    let cache_bytes = (CACHE_COUNT + GUARDS) * 2;
    let total = admitted(case, limit, probe.width)?;
    let launch = groups(count, probe.width)?;
    let mut k = Shared::<u16>::new(&probe.device, input_bytes)?;
    let mut v = Shared::<u16>::new(&probe.device, input_bytes)?;
    let mut cache_k = Shared::<u16>::new(&probe.device, cache_bytes)?;
    let mut cache_v = Shared::<u16>::new(&probe.device, cache_bytes)?;
    let kb = baseline(&case.k, input_bytes / 2, INPUT_GUARD);
    let vb = baseline(&case.v, input_bytes / 2, INPUT_GUARD);
    let ckb = baseline(&case.initial_k, CACHE_COUNT + GUARDS, K_GUARD);
    let cvb = baseline(&case.initial_v, CACHE_COUNT + GUARDS, V_GUARD);
    k.write(&kb)?;
    v.write(&vb)?;
    let shape = [case.rows as u32, case.offset as u32, CAPACITY as u32];
    let mut rows = Vec::with_capacity(13);
    for repetition in 0..13 {
        // Diagnostic cases restore saved state outside timing. Persistent allocations
        // are reused; this is not a session or steady-state inference claim.
        cache_k.write(&ckb)?;
        cache_v.write(&cvb)?;
        rows.push(autoreleasepool(|_|{
            let started=Instant::now();let command=probe.queue.commandBuffer().ok_or_else(||error("command","command unavailable"))?;
            if !command.retainedReferences(){return Err(error("command","unretained resources unsupported"));}
            let encoder=command.computeCommandEncoder().ok_or_else(||error("command","encoder unavailable"))?;
            encoder.setComputePipelineState(&probe.pipeline);
            // SAFETY: four distinct owned initialized Half buffers at offset0.
            // Read-only token-major K/V cover rounded grid+7; writeable separate
            // head-major caches cover2*capacity*8+7. Copied shape is three UInt32
            // values (12/4). Validated rows/offset imply unique in-bounds targets;
            // shader masks inactive lanes before access. Tracked owners live
            // through checked completion; host accesses occur before/after GPU.
            unsafe{
                encoder.setBuffer_offset_atIndex(Some(&k.raw),0,0);encoder.setBuffer_offset_atIndex(Some(&v.raw),0,1);
                encoder.setBuffer_offset_atIndex(Some(&cache_k.raw),0,2);encoder.setBuffer_offset_atIndex(Some(&cache_v.raw),0,3);
                encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(),size_of::<[u32;3]>(),4);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(MTLSize{width:launch,height:1,depth:1},MTLSize{width:probe.width,height:1,depth:1});
            encoder.endEncoding();command.commit();command.waitUntilCompleted();let seconds=started.elapsed().as_secs_f64();
            if command.status()!=MTLCommandBufferStatus::Completed || command.error().is_some(){return Err(error("completion",format!("cache GPU failure: {:?}",command.error())));}
            let ak=cache_k.read();let av=cache_v.read();
            if k.read()!=kb || v.read()!=vb || !valid_cache(&ak,&case.initial_k,&case.k,case,K_GUARD) || !valid_cache(&av,&case.initial_v,&case.v,case,V_GUARD){return Err(error("validation","cache inputs or target/untouched/guard bytes differ"));}
            Ok(json!({"phase":if repetition<3{"warmup"}else{"sample"},"repetition":if repetition<3{repetition+1}else{repetition-2},
                "command_completed":true,"inputs_targets_untouched_and_guards_valid":true,"encode_commit_wait_seconds":seconds,"cache_k_bits":ak,"cache_v_bits":av}))
        })?);
    }
    Ok(
        json!({"id":case.id,"rows_count":case.rows,"offset":case.offset,"heads":2,"dimension":8,"capacity":CAPACITY,"layout":"heads_capacity_dimension",
        "input_guard_values":input_bytes/2-count,"cache_guard_values":GUARDS,"k_guard_bits":K_GUARD,"v_guard_bits":V_GUARD,
        "threadgroup_width":probe.width,"threadgroups":launch,"shared_buffer_bytes":2*(input_bytes+cache_bytes),"host_baseline_readback_bytes":4*(input_bytes+cache_bytes),
        "admitted_bytes":total,"allocations_reused_across_repetitions":true,"committed_position_advanced":false,"rows":rows}),
    )
}
pub(super) fn run(path: &Path, limit: usize) -> Result<Value, Error> {
    let cases = load(path)?;
    let initial = cases
        .iter()
        .map(|c| admitted(c, limit, 1))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or_else(|| error("admission", "empty cache cases"))?;
    autoreleasepool(|_| {
        let (mut probe, mut metadata) = prepare(
            include_str!("rust_metal_cache.metal"),
            "toklet_cache_write",
            initial,
        )?;
        let maximum = cases
            .iter()
            .map(|c| admitted(c, limit, probe.width))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .ok_or_else(|| error("admission", "empty cache cases"))?;
        metadata["maximum_admitted_bytes"] = json!(maximum);
        let mut results = Vec::with_capacity(cases.len());
        for case in &cases {
            results.push(autoreleasepool(|_| execute(&mut probe, case, limit))?);
        }
        Ok(
            json!({"operation":"kv_half_token_to_head_major_write","metadata":metadata,"cases":results}),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    fn value() -> Value {
        json!({"format":"toklet-metal-cache-input-v1","cases":[{"id":"manual","rows":1,"offset":31,"heads":2,"dimension":8,"capacity":32,
        "k_bits":vec![0u16;16],"v_bits":vec![0u16;16],"initial_k_bits":vec![0u16;512],"initial_v_bits":vec![0u16;512]}]})
    }
    fn parse(value: &Value) -> Result<Vec<Case>, Error> {
        let path = std::env::temp_dir().join(format!(
            "toklet-cache-{}-{}.json",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        writeln!(f, "{value}").unwrap();
        let result = load(&path);
        std::fs::remove_file(path).unwrap();
        result
    }
    #[test]
    fn parser_checks_shapes_capacity_unsigned_and_finite_input_before_native_work() {
        assert_eq!(parse(&value()).unwrap()[0].offset, 31);
        for (key, n) in [
            ("rows", json!(0)),
            ("rows", json!(33)),
            ("offset", json!(32)),
            ("offset", json!(u64::MAX)),
            ("offset", json!(-1)),
            ("heads", json!(4)),
            ("dimension", json!(7)),
            ("capacity", json!(33)),
        ] {
            let mut v = value();
            v["cases"][0][key] = n;
            assert!(parse(&v).is_err());
        }
        for key in ["k_bits", "v_bits", "initial_k_bits", "initial_v_bits"] {
            let mut v = value();
            v["cases"][0][key][0] = json!(0x7c00);
            assert!(parse(&v).is_err());
            v["cases"][0][key] = json!([0]);
            assert!(parse(&v).is_err());
        }
        let mut v = value();
        let c = v["cases"][0].clone();
        v["cases"].as_array_mut().unwrap().push(c);
        assert!(parse(&v).is_err());
    }
    #[test]
    fn unique_head_major_targets_preserve_prefix_tail_and_guard_bytes() {
        let mut case = parse(&value()).unwrap().remove(0);
        case.k = (0..16).map(|n| n as u16).collect();
        let mut actual = vec![0; CACHE_COUNT + GUARDS];
        actual[CACHE_COUNT..].fill(K_GUARD);
        actual[31 * 8..32 * 8].copy_from_slice(&case.k[..8]);
        actual[63 * 8..64 * 8].copy_from_slice(&case.k[8..]);
        assert!(valid_cache(
            &actual,
            &case.initial_k,
            &case.k,
            &case,
            K_GUARD
        ));
        for index in [0, 31 * 8, 63 * 8, CACHE_COUNT] {
            let mut bad = actual.clone();
            bad[index] ^= 1;
            assert!(!valid_cache(&bad, &case.initial_k, &case.k, &case, K_GUARD));
        }
        let total = ALLOWANCE + 3 * (2 * (32 + 7) * 2 + 2 * (512 + 7) * 2);
        assert_eq!(admitted(&case, total, 32).unwrap(), total);
        assert!(admitted(&case, total - 1, 32).is_err());
        assert!(storage(usize::MAX, 32).is_err());
        assert!(storage(16, 0).is_err());
    }
}
