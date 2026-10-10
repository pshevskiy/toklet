use super::*;
use std::{collections::HashSet, io::Read, path::Path};

const TABLE_GUARD: u16 = 0x51aa;
const TOKEN_GUARD: u32 = 0xdead_beef;
const OUTPUT_GUARD: u16 = 0x55a1;
const MAX_INPUT: u64 = 1024 * 1024;

#[repr(C)]
struct Params {
    count: u32,
    hidden: u32,
    vocab: u32,
}
const _: () = assert!(size_of::<Params>() == 12 && align_of::<Params>() == 4);
struct Case {
    id: String,
    tokens: Vec<u32>,
}
struct Fixture {
    hidden: usize,
    vocab: usize,
    table: Vec<u16>,
    cases: Vec<Case>,
}

fn scalar(value: &Value, name: &str) -> Result<usize, Error> {
    value[name]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| error("fixture", format!("invalid unsigned {name}")))
}
fn load(path: &Path) -> Result<Fixture, Error> {
    let file = std::fs::File::open(path).map_err(|e| error("fixture", e.to_string()))?;
    let size = file
        .metadata()
        .map_err(|e| error("fixture", e.to_string()))?
        .len();
    if size > MAX_INPUT {
        return Err(error("fixture", "embedding input exceeds one MiB"));
    }
    let mut data = Vec::new();
    file.take(MAX_INPUT + 1)
        .read_to_end(&mut data)
        .map_err(|e| error("fixture", e.to_string()))?;
    if data.len() as u64 != size || data.len() as u64 > MAX_INPUT {
        return Err(error(
            "fixture",
            "embedding input changed size or exceeds bounds",
        ));
    }
    let value: Value =
        serde_json::from_slice(&data).map_err(|e| error("fixture", e.to_string()))?;
    if value["format"] != "toklet-metal-embedding-input-v1" {
        return Err(error("fixture", "embedding format differs"));
    }
    let hidden = scalar(&value, "hidden")?;
    let vocab = scalar(&value, "vocab")?;
    if ![32, 896].contains(&hidden) || !(1..=64).contains(&vocab) {
        return Err(error("fixture", "embedding prototype shape outside bounds"));
    }
    let count = hidden
        .checked_mul(vocab)
        .ok_or_else(|| error("admission", "table shape overflow"))?;
    if count > 65536 {
        return Err(error("admission", "embedding table exceeds 65536 values"));
    }
    let source = value["table_bits"]
        .as_array()
        .ok_or_else(|| error("fixture", "missing Half table"))?;
    if source.len() != count {
        return Err(error("fixture", "Half table count differs"));
    }
    let mut table = Vec::with_capacity(count + GUARDS);
    for item in source {
        let bits = item
            .as_u64()
            .and_then(|n| u16::try_from(n).ok())
            .ok_or_else(|| error("fixture", "table bits outside UInt16"))?;
        if bits & 0x7c00 == 0x7c00 {
            return Err(error("fixture", "nonfinite Half table"));
        }
        table.push(bits);
    }
    table.resize(count + GUARDS, TABLE_GUARD);
    let sources = value["cases"]
        .as_array()
        .ok_or_else(|| error("fixture", "missing embedding cases"))?;
    if sources.is_empty() || sources.len() > 32 {
        return Err(error("fixture", "embedding cases outside 1..32"));
    }
    let mut names = HashSet::new();
    let mut elements = 0usize;
    let mut cases = Vec::with_capacity(sources.len());
    for source in sources {
        let id = source["id"]
            .as_str()
            .ok_or_else(|| error("fixture", "missing embedding identity"))?;
        if id.is_empty() || id.len() > 128 || !id.is_ascii() || !names.insert(id) {
            return Err(error("fixture", "invalid or duplicate embedding identity"));
        }
        let ids = source["tokens"]
            .as_array()
            .ok_or_else(|| error("fixture", "missing token IDs"))?;
        if !(1..=32).contains(&ids.len()) {
            return Err(error("fixture", "embedding rows outside 1..32"));
        }
        elements = elements
            .checked_add(
                ids.len()
                    .checked_mul(hidden)
                    .ok_or_else(|| error("admission", "output shape overflow"))?,
            )
            .ok_or_else(|| error("admission", "output inventory overflow"))?;
        if elements > 8192 {
            return Err(error(
                "admission",
                "embedding output inventory exceeds 8192 values",
            ));
        }
        let mut tokens = Vec::with_capacity(ids.len() + GUARDS);
        for item in ids {
            let token = item
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| error("fixture", "token outside UInt32"))?;
            if token as usize >= vocab {
                return Err(error("fixture", "token outside vocabulary"));
            }
            tokens.push(token);
        }
        tokens.resize(ids.len() + GUARDS, TOKEN_GUARD);
        cases.push(Case {
            id: id.to_owned(),
            tokens,
        });
    }
    // JSON/data temporaries drop before native allocations. All IDs validated.
    Ok(Fixture {
        hidden,
        vocab,
        table,
        cases,
    })
}
fn storage(count: usize, size: usize) -> Result<usize, Error> {
    let bytes = count
        .checked_add(GUARDS)
        .and_then(|n| n.checked_mul(size))
        .ok_or_else(|| error("admission", "embedding storage overflow"))?;
    if count == 0 || bytes > isize::MAX as usize {
        return Err(error(
            "admission",
            "embedding storage exceeds pointer bounds",
        ));
    }
    Ok(bytes)
}
fn admitted(fixture: &Fixture, rows: usize, limit: usize) -> Result<usize, Error> {
    let bytes = storage(fixture.hidden * fixture.vocab, 2)?
        .checked_mul(3)
        .and_then(|n| storage(rows, 4).ok()?.checked_mul(3)?.checked_add(n))
        .and_then(|n| {
            storage(rows * fixture.hidden, 2)
                .ok()?
                .checked_mul(2)?
                .checked_add(n)
        })
        .and_then(|n| n.checked_add(ALLOWANCE))
        .ok_or_else(|| error("admission", "embedding working-set overflow"))?;
    if limit > LIMIT || bytes > limit {
        return Err(error("admission", "embedding exceeds supplied budget"));
    }
    Ok(bytes)
}
fn execute(
    probe: &mut Probe,
    fixture: &Fixture,
    table: &Shared<u16>,
    case: &Case,
    limit: usize,
) -> Result<Value, Error> {
    let rows = case.tokens.len() - GUARDS;
    let count = rows * fixture.hidden;
    let table_bytes = storage(fixture.hidden * fixture.vocab, 2)?;
    let id_bytes = storage(rows, 4)?;
    let output_bytes = storage(count, 2)?;
    let budget = admitted(fixture, rows, limit)?;
    let mut tokens = Shared::<u32>::new(&probe.device, id_bytes)?;
    let mut output = Shared::<u16>::new(&probe.device, output_bytes)?;
    tokens.write(&case.tokens)?;
    let params = Params {
        count: count as u32,
        hidden: fixture.hidden as u32,
        vocab: fixture.vocab as u32,
    };
    let launch = groups(count, probe.width)?;
    let mut repetitions = Vec::with_capacity(13);
    for repetition in 0..13 {
        output.fill(OUTPUT_GUARD);
        repetitions.push(autoreleasepool(|_| {
            let started=Instant::now();
            let command=probe.queue.commandBuffer().ok_or_else(|| error("command","command unavailable"))?;
            if !command.retainedReferences() {return Err(error("command","unretained resources unsupported"));}
            let encoder=command.computeCommandEncoder().ok_or_else(|| error("command","encoder unavailable"))?;
            encoder.setComputePipelineState(&probe.pipeline);
            // SAFETY: ushort table[V,H] at0, uint IDs[rows] at1, ushort output
            // [rows,H] at2; offsets0. Scalar alignment/lengths validated by
            // private Shared owners. Params at3 is copied repr(C)12bytes/align4.
            // IDs validated before device discovery. Rounded grid, count and
            // table indices fit UInt32 under loader bounds. Shader checks count
            // and vocab before reads. Distinct retained buffers remain alive
            // through wait; no host access occurs between commit and completion.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&table.raw),0,0);
                encoder.setBuffer_offset_atIndex(Some(&tokens.raw),0,1);
                encoder.setBuffer_offset_atIndex(Some(&output.raw),0,2);
                encoder.setBytes_length_atIndex(NonNull::from(&params).cast(),size_of::<Params>(),3);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(MTLSize{width:launch,height:1,depth:1},MTLSize{width:probe.width,height:1,depth:1});
            encoder.endEncoding();command.commit();command.waitUntilCompleted();
            let seconds=started.elapsed().as_secs_f64();
            if command.status()!=MTLCommandBufferStatus::Completed || command.error().is_some() {
                return Err(error("completion",format!("embedding GPU failure: {:?}",command.error())));
            }
            let actual=output.read();
            if table.read()!=fixture.table || tokens.read()!=case.tokens {
                return Err(error("validation","embedding table/IDs or guards changed"));
            }
            if actual[count..].iter().any(|&bits| bits!=OUTPUT_GUARD) {
                return Err(error("validation","embedding output guard changed"));
            }
            Ok(json!({"phase":if repetition<3 {"warmup"}else{"sample"},"repetition":if repetition<3 {repetition+1}else{repetition-2},
                "command_completed":true,"immutable_table_tokens_and_guards":true,"encode_commit_wait_seconds":seconds,"output_bits":actual}))
        })?);
    }
    Ok(
        json!({"id":case.id,"rows_count":rows,"hidden":fixture.hidden,"vocab":fixture.vocab,
        "guards":GUARDS,"output_guard_bits":OUTPUT_GUARD,"threadgroup_width":probe.width,"threadgroups":launch,
        "shared_buffer_bytes":table_bytes+id_bytes+output_bytes,"host_upload_and_readback_bytes":table_bytes*2+id_bytes*2+output_bytes,
        "admitted_bytes":budget,"allocations_reused_across_repetitions":true,"rows":repetitions}),
    )
}
pub(super) fn run(path: &Path, limit: usize) -> Result<Value, Error> {
    let fixture = load(path)?;
    let maximum = fixture
        .cases
        .iter()
        .map(|case| admitted(&fixture, case.tokens.len() - GUARDS, limit))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or_else(|| error("admission", "empty embedding cases"))?;
    autoreleasepool(|_| {
        let (mut probe, metadata) = prepare(
            include_str!("rust_metal_embedding.metal"),
            "toklet_embedding",
            maximum,
        )?;
        // Exactly one table allocation/upload persists through all cases.
        let mut table =
            Shared::<u16>::new(&probe.device, storage(fixture.hidden * fixture.vocab, 2)?)?;
        table.write(&fixture.table)?;
        let mut results = Vec::with_capacity(fixture.cases.len());
        for case in &fixture.cases {
            results.push(autoreleasepool(|_| {
                execute(&mut probe, &fixture, &table, case, limit)
            })?);
        }
        Ok(
            json!({"operation":"embedding_fp16_bit_copy","metadata":metadata,"table_allocation_reused_across_cases":true,"cases":results}),
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
            "toklet-embedding-{}-{}.json",
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
        json!({"format":"toklet-metal-embedding-input-v1","hidden":32,"vocab":2,
            "table_bits":vec![0x3c00u16;64],"cases":[{"id":"manual","tokens":[1,0,1]}]})
    }
    #[test]
    fn rejects_all_invalid_ids_and_nonfinite_table_before_native_work() {
        for variant in ["last_id", "half_inf", "unsigned"] {
            let mut value = sample();
            match variant {
                "last_id" => value["cases"][0]["tokens"][2] = json!(2),
                "half_inf" => value["table_bits"][63] = json!(0x7c00),
                _ => value["cases"][0]["tokens"][0] = json!(-1),
            }
            let file = fixture(&value);
            assert_eq!(load(&file.0).err().unwrap().stage, "fixture");
        }
    }
    #[test]
    fn duplicate_cases_and_inventory_limits_fail_before_device_work() {
        let mut value = sample();
        let duplicate = value["cases"][0].clone();
        value["cases"].as_array_mut().unwrap().push(duplicate);
        let file = fixture(&value);
        assert!(load(&file.0).is_err());
        let mut value = sample();
        value["hidden"] = json!(896);
        value["vocab"] = json!(1);
        value["table_bits"] = json!(vec![0x3c00u16; 896]);
        value["cases"][0]["tokens"] = json!(vec![0; 10]);
        let file = fixture(&value);
        assert_eq!(load(&file.0).err().unwrap().stage, "admission");
    }
    #[test]
    fn guarded_storage_and_admission_count_one_shared_table() {
        let file = fixture(&sample());
        let value = load(&file.0).unwrap();
        assert_eq!(value.table[64..], [TABLE_GUARD; 7]);
        assert_eq!(value.cases[0].tokens[3..], [TOKEN_GUARD; 7]);
        let total = ALLOWANCE + 142 * 3 + 40 * 3 + 206 * 2;
        assert_eq!(admitted(&value, 3, total).unwrap(), total);
        assert!(admitted(&value, 3, total - 1).is_err());
        assert!(storage(usize::MAX, 2).is_err());
        assert!(storage(0, 4).is_err());
    }
}
