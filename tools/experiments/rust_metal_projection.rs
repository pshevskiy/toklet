use super::*;
use objc2::AnyThread;
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication, MPSSupportsMTLDevice,
};
use std::{collections::HashSet, io::Read, path::Path};

const MAX_INPUT: u64 = 1024 * 1024;
const HALF_GUARD: u16 = 0x55a1;
const FLOAT_GUARD: u32 = 0x7fc0_5678;
#[repr(C)]
struct Params {
    rows: u32,
    outputs: u32,
    float_stride: u32,
    half_stride: u32,
}
const _: () = assert!(size_of::<Params>() == 16 && align_of::<Params>() == 4);
struct Case {
    id: String,
    rows: usize,
    inputs: usize,
    outputs: usize,
    input: Vec<u16>,
    weight: Vec<u16>,
    bias: Vec<u16>,
}
fn number(value: &Value, name: &str) -> Result<usize, Error> {
    value[name]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| error("fixture", "projection dimension differs"))
}
fn half(value: &Value, name: &str, count: usize) -> Result<Vec<u16>, Error> {
    let source = value[name]
        .as_array()
        .ok_or_else(|| error("fixture", "missing projection Half bits"))?;
    if source.len() != count {
        return Err(error("fixture", "projection Half count differs"));
    }
    source
        .iter()
        .map(|v| {
            let bits = v
                .as_u64()
                .and_then(|n| u16::try_from(n).ok())
                .ok_or_else(|| error("fixture", "projection Half bits outside UInt16"))?;
            if bits & 0x7c00 == 0x7c00 {
                return Err(error("fixture", "nonfinite projection input"));
            }
            Ok(bits)
        })
        .collect()
}
fn load(path: &Path) -> Result<Vec<Case>, Error> {
    let file = std::fs::File::open(path).map_err(|e| error("fixture", e.to_string()))?;
    let size = file
        .metadata()
        .map_err(|e| error("fixture", e.to_string()))?
        .len();
    if size > MAX_INPUT {
        return Err(error("fixture", "projection input exceeds one MiB"));
    }
    let mut bytes = Vec::new();
    file.take(MAX_INPUT + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| error("fixture", e.to_string()))?;
    if bytes.len() as u64 != size || bytes.len() as u64 > MAX_INPUT {
        return Err(error(
            "fixture",
            "projection input changed or exceeds bounds",
        ));
    }
    let data: Value =
        serde_json::from_slice(&bytes).map_err(|e| error("fixture", e.to_string()))?;
    if data["format"] != "toklet-metal-projection-input-v1" {
        return Err(error("fixture", "projection format differs"));
    }
    let values = data["cases"]
        .as_array()
        .ok_or_else(|| error("fixture", "missing projection cases"))?;
    if values.is_empty() || values.len() > 16 {
        return Err(error("fixture", "projection case count outside 1..16"));
    }
    let mut names = HashSet::new();
    let mut inventory = 0usize;
    let mut cases = Vec::new();
    for v in values {
        let id = v["id"]
            .as_str()
            .ok_or_else(|| error("fixture", "projection identity differs"))?;
        if id.is_empty() || id.len() > 128 || !id.is_ascii() || !names.insert(id) {
            return Err(error("fixture", "invalid or duplicate projection identity"));
        }
        let rows = number(v, "rows")?;
        let inputs = number(v, "inputs")?;
        let outputs = number(v, "outputs")?;
        if !(1..=32).contains(&rows) || !(1..=64).contains(&inputs) || !(1..=64).contains(&outputs)
        {
            return Err(error("fixture", "projection shape outside bounds"));
        }
        let a = rows
            .checked_mul(inputs)
            .ok_or_else(|| error("admission", "projection shape overflow"))?;
        let b = outputs
            .checked_mul(inputs)
            .ok_or_else(|| error("admission", "projection shape overflow"))?;
        inventory = inventory
            .checked_add(a + b + outputs + rows * outputs)
            .ok_or_else(|| error("admission", "projection inventory overflow"))?;
        if inventory > 32768 {
            return Err(error("admission", "projection inventory exceeds32768"));
        }
        cases.push(Case {
            id: id.to_owned(),
            rows,
            inputs,
            outputs,
            input: half(v, "input_bits", a)?,
            weight: half(v, "weight_bits", b)?,
            bias: half(v, "bias_bits", outputs)?,
        });
    }
    Ok(cases)
}
#[derive(Clone, Copy)]
struct Layout {
    rows: usize,
    columns: usize,
    stride: usize,
    elements: usize,
    bytes: usize,
}
fn layout(
    rows: usize,
    columns: usize,
    scalar_bytes: usize,
    row_bytes: usize,
    storage_rows: usize,
) -> Result<Layout, Error> {
    if rows == 0
        || columns == 0
        || storage_rows < rows
        || ![2, 4].contains(&scalar_bytes)
        || row_bytes
            < columns
                .checked_mul(scalar_bytes)
                .ok_or_else(|| error("admission", "projection row overflow"))?
        || !row_bytes.is_multiple_of(scalar_bytes)
    {
        return Err(error("admission", "invalid projection layout"));
    }
    let stride = row_bytes / scalar_bytes;
    let elements = storage_rows
        .checked_mul(stride)
        .and_then(|n| n.checked_add(GUARDS))
        .ok_or_else(|| error("admission", "projection layout overflow"))?;
    let bytes = elements
        .checked_mul(scalar_bytes)
        .ok_or_else(|| error("admission", "projection byte overflow"))?;
    if u32::try_from(elements).is_err() || bytes > isize::MAX as usize {
        return Err(error("admission", "projection indexing overflow"));
    }
    Ok(Layout {
        rows,
        columns,
        stride,
        elements,
        bytes,
    })
}
fn queried(
    rows: usize,
    columns: usize,
    scalar_bytes: usize,
    storage_rows: usize,
) -> Result<Layout, Error> {
    // SAFETY: small nonzero columns and qualified Float16/Float32 types. This
    // class query allocates no tensor; returned alignment/size is checked.
    let row_bytes = unsafe {
        MPSMatrixDescriptor::rowBytesForColumns_dataType(
            columns,
            if scalar_bytes == 2 {
                MPSDataType::Float16
            } else {
                MPSDataType::Float32
            },
        )
    };
    layout(rows, columns, scalar_bytes, row_bytes, storage_rows)
}
fn matrix<T: Scalar>(buffer: &Shared<T>, l: Layout, dtype: MPSDataType) -> Retained<MPSMatrix> {
    // SAFETY: descriptor reflects checked physical stride and owned buffer size;
    // offset0, positive dimensions and type match T. Both objects stay owned
    // through command completion and host access. No temporary MPS allocation.
    unsafe {
        let d = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            l.rows,
            l.columns,
            l.stride * size_of::<T>(),
            dtype,
        );
        MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &buffer.raw, &d)
    }
}
fn baseline(values: &[u16], l: Layout) -> Vec<u16> {
    let mut out = vec![HALF_GUARD; l.elements];
    for (row, values) in values.chunks_exact(l.columns).enumerate() {
        out[row * l.stride..row * l.stride + l.columns].copy_from_slice(values);
    }
    out
}
fn budget(
    a: Layout,
    b: Layout,
    c: Layout,
    out: Layout,
    bias_bytes: usize,
    limit: usize,
) -> Result<(usize, usize, usize), Error> {
    let inputs = a
        .bytes
        .checked_add(b.bytes)
        .and_then(|n| n.checked_add(bias_bytes))
        .ok_or_else(|| error("admission", "projection input sum overflow"))?;
    let outputs = c
        .bytes
        .checked_add(out.bytes)
        .ok_or_else(|| error("admission", "projection output sum overflow"))?;
    let shared = inputs
        .checked_add(outputs)
        .ok_or_else(|| error("admission", "projection shared sum overflow"))?;
    // Two immutable host baselines/readbacks per input; one readback per output.
    let host = inputs
        .checked_mul(2)
        .and_then(|n| n.checked_add(outputs))
        .ok_or_else(|| error("admission", "projection host sum overflow"))?;
    let total = shared
        .checked_add(host)
        .and_then(|n| n.checked_add(ALLOWANCE))
        .ok_or_else(|| error("admission", "projection budget overflow"))?;
    if limit > LIMIT || total > limit {
        return Err(error("admission", "projection exceeds supplied budget"));
    }
    Ok((shared, host, total))
}
fn execute(
    probe: &Probe,
    bias_pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    case: &Case,
    limit: usize,
) -> Result<Value, Error> {
    let count = case.rows * case.outputs;
    let launch = groups(count, probe.width)?;
    let rounded = launch * probe.width;
    let storage_rows = rounded.div_ceil(case.outputs);
    let a = queried(case.rows, case.inputs, 2, case.rows)?;
    let b = queried(case.outputs, case.inputs, 2, case.outputs)?;
    // Full mapped inactive dispatch region + sentinels, not just seven values.
    let c = queried(case.rows, case.outputs, 4, storage_rows)?;
    let out = queried(case.rows, case.outputs, 2, storage_rows)?;
    let bias_bytes = (case.outputs + GUARDS) * 2;
    let (shared, host, admitted) = budget(a, b, c, out, bias_bytes, limit)?;
    let av = baseline(&case.input, a);
    let bv = baseline(&case.weight, b);
    let mut biasv = vec![HALF_GUARD; case.outputs + GUARDS];
    biasv[..case.outputs].copy_from_slice(&case.bias);
    let mut ab = Shared::<u16>::new(&probe.device, a.bytes)?;
    ab.write(&av)?;
    let mut bb = Shared::<u16>::new(&probe.device, b.bytes)?;
    bb.write(&bv)?;
    let mut bias = Shared::<u16>::new(&probe.device, bias_bytes)?;
    bias.write(&biasv)?;
    let mut cb = Shared::<f32>::new(&probe.device, c.bytes)?;
    let mut ob = Shared::<u16>::new(&probe.device, out.bytes)?;
    let am = matrix(&ab, a, MPSDataType::Float16);
    let bm = matrix(&bb, b, MPSDataType::Float16);
    let cm = matrix(&cb, c, MPSDataType::Float32);
    // SAFETY: same device, nontransposed activations [rows,in], transposed
    // row-major weights [out,in], F32 result [rows,out], alpha1/beta1 after
    // GPU bias initialization. Dimensions/strides/storage checked before this.
    let kernel = unsafe {
        MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
        MPSMatrixMultiplication::alloc(),&probe.device,false,true,case.rows,case.outputs,case.inputs,1.0,1.0)
    };
    let params = Params {
        rows: case.rows as u32,
        outputs: case.outputs as u32,
        float_stride: c.stride as u32,
        half_stride: out.stride as u32,
    };
    let mut repetitions = Vec::with_capacity(13);
    for repetition in 0..13 {
        cb.fill(f32::from_bits(FLOAT_GUARD));
        ob.fill(HALF_GUARD);
        repetitions.push(autoreleasepool(|_| {
            let start=Instant::now();
            let command=probe.queue.commandBuffer().ok_or_else(||error("command","projection command unavailable"))?;
            if !command.retainedReferences(){return Err(error("command","unretained projection resources unsupported"));}
            for phase in 0..2 {
                let encoder=command.computeCommandEncoder().ok_or_else(||error("command","projection encoder unavailable"))?;
                encoder.setComputePipelineState(if phase==0{bias_pipeline}else{&probe.pipeline});
                // SAFETY: distinct sized aligned buffers; UInt4 params16/4
                // copied at slot2. Shader maps checked strides and bounds count.
                // Owners/retained references survive commit/wait. CPU access is
                // excluded during GPU work; tracked hazards order all encoders.
                unsafe{
                    encoder.setBuffer_offset_atIndex(Some(if phase==0{&bias.raw}else{&cb.raw}),0,0);
                    encoder.setBuffer_offset_atIndex(Some(if phase==0{&cb.raw}else{&ob.raw}),0,1);
                    encoder.setBytes_length_atIndex(NonNull::from(&params).cast(),size_of::<Params>(),2);
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(MTLSize{width:launch,height:1,depth:1},MTLSize{width:probe.width,height:1,depth:1});encoder.endEncoding();
                if phase==0 {
                    // SAFETY: owned matrices/kernel from same device, no alias,
                    // exactly admitted dimensions; tracked bias->MPS->store
                    // dependencies in one command. Resources live through wait.
                    unsafe{kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(&command,&am,&bm,&cm);}
                }
            }
            command.commit();command.waitUntilCompleted();
            let seconds=start.elapsed().as_secs_f64();
            if command.status()!=MTLCommandBufferStatus::Completed || command.error().is_some(){return Err(error("completion","projection GPU failure"));}
            let actual=ob.read();let floats=cb.read();
            if ab.read()!=av || bb.read()!=bv || bias.read()!=biasv {return Err(error("validation","projection immutable inputs changed"));}
            let mut bits=Vec::with_capacity(count);let mut fp32=Vec::with_capacity(count);
            for row in 0..case.rows {
                for col in 0..case.outputs {
                    let value=actual[row*out.stride+col];let single=floats[row*c.stride+col];
                    if value&0x7c00==0x7c00 || !single.is_finite(){return Err(error("validation","completed projection nonfinite; result discarded"));}
                    bits.push(value);fp32.push(single.to_bits());
                }
            }
            for (index,&value) in actual.iter().enumerate(){
                if (index/out.stride>=case.rows || index%out.stride>=case.outputs) && value!=HALF_GUARD {return Err(error("validation","projection Half padding/guards changed"));}
            }
            // MPS may use FP32 row padding; only storage past logical matrix
            // payload is required unchanged. No arithmetic consumes padding.
            if floats[case.rows*c.stride..].iter().any(|v|v.to_bits()!=FLOAT_GUARD){return Err(error("validation","projection FP32 tail guards changed"));}
            Ok(json!({"phase":if repetition<3{"warmup"}else{"sample"},"repetition":if repetition<3{repetition+1}else{repetition-2},
              "command_completed":true,"inputs_and_guards_unchanged":true,"encode_commit_wait_seconds":seconds,"output_bits":bits,"fp32_bits":fp32}))
        })?);
    }
    Ok(
        json!({"id":case.id,"rows_count":case.rows,"inputs":case.inputs,"outputs":case.outputs,"threadgroup_width":probe.width,"threadgroups":launch,
      "input_stride":a.stride,"weight_stride":b.stride,"fp32_stride":c.stride,"output_stride":out.stride,"storage_rows":storage_rows,
      "shared_buffer_bytes":shared,"host_baseline_readback_bytes":host,"admitted_bytes":admitted,"rows":repetitions}),
    )
}
pub(super) fn run(path: &Path, limit: usize) -> Result<Value, Error> {
    let cases = load(path)?;
    if !(ALLOWANCE..=LIMIT).contains(&limit) {
        return Err(error(
            "admission",
            "projection budget outside runtime allowance",
        ));
    }
    autoreleasepool(|_| {
        let (probe, mut metadata) = prepare(
            include_str!("rust_metal_projection.metal"),
            "toklet_projection_store",
            ALLOWANCE,
        )?;
        // SAFETY: queried device is a live retained Metal device; this is a
        // capability query only, before matrix allocations or command encoding.
        if !unsafe { MPSSupportsMTLDevice(Some(&probe.device)) } {
            return Err(error("capability", "MPS does not support this device"));
        }
        metadata["mps_device_supported"] = json!(true);
        let function = probe
            .library
            .newFunctionWithName(&NSString::from_str("toklet_projection_bias"))
            .ok_or_else(|| error("pipeline", "projection bias unavailable"))?;
        let bias = probe
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| error("pipeline", e.to_string()))?;
        // One common width respects both pipelines and the queried device limit.
        let mut probe = probe;
        probe.width = probe
            .width
            .min(bias.threadExecutionWidth())
            .min(bias.maxTotalThreadsPerThreadgroup());
        if probe.width == 0 {
            return Err(error("capability", "invalid bias launch width"));
        }
        let mut results = Vec::new();
        for case in &cases {
            results.push(autoreleasepool(|_| execute(&probe, &bias, case, limit))?);
        }
        metadata["maximum_admitted_bytes"] = json!(
            results
                .iter()
                .map(|v| v["admitted_bytes"].as_u64().unwrap())
                .max()
                .unwrap()
        );
        metadata["bias_execution_width"] = json!(bias.threadExecutionWidth());
        metadata["bias_max_threads"] = json!(bias.maxTotalThreadsPerThreadgroup());
        Ok(
            json!({"operation":"mps_fp16_inputs_fp32_bias_and_result_fp16_store","metadata":metadata,"cases":results}),
        )
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(value: &Value) -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "toklet-projection-{}-{}.json",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        writeln!(file, "{value}").unwrap();
        path
    }
    fn loaded(value: &Value) -> Result<Vec<Case>, Error> {
        let path = fixture(value);
        let result = load(&path);
        std::fs::remove_file(path).unwrap();
        result
    }
    fn sample() -> Value {
        json!({"format":"toklet-metal-projection-input-v1","cases":[{"id":"manual","rows":1,"inputs":2,"outputs":1,"input_bits":[0x3c00,0x3c00],"weight_bits":[0x3c00,0x1000],"bias_bits":[0x1000]}]})
    }
    #[test]
    fn malformed_shapes_arrays_and_finite_half_values_fail_before_device_work() {
        assert!(loaded(&sample()).is_ok());
        for (key, value) in [
            ("rows", 0),
            ("rows", 33),
            ("inputs", 0),
            ("inputs", 65),
            ("outputs", 0),
            ("outputs", 65),
        ] {
            let mut source = sample();
            source["cases"][0][key] = json!(value);
            assert!(loaded(&source).is_err());
        }
        for key in ["input_bits", "weight_bits", "bias_bits"] {
            for value in [
                json!(0x7c00),
                json!(0xfc00),
                json!(0x7e00),
                json!(-1),
                json!(65536),
                json!(1.0),
            ] {
                let mut source = sample();
                source["cases"][0][key][0] = value;
                assert!(loaded(&source).is_err());
            }
            let mut source = sample();
            source["cases"][0][key] = json!([]);
            assert!(loaded(&source).is_err());
        }
        let mut source = sample();
        source["cases"][0]["id"] = json!("λ");
        assert!(loaded(&source).is_err());
        let mut source = sample();
        let duplicate = source["cases"][0].clone();
        source["cases"].as_array_mut().unwrap().push(duplicate);
        assert!(loaded(&source).is_err());
        let source = json!({"format":"toklet-metal-projection-input-v1","cases":(0..4).map(|i|json!({"id":format!("{i}"),"rows":32,"inputs":64,"outputs":64,"input_bits":vec![0u16;2048],"weight_bits":vec![0u16;4096],"bias_bits":vec![0u16;64]})).collect::<Vec<_>>()});
        assert_eq!(loaded(&source).err().unwrap().stage, "admission");
        for count in [0, 17] {
            let source = json!({"format":"toklet-metal-projection-input-v1","cases":vec![sample()["cases"][0].clone();count]});
            assert!(loaded(&source).is_err());
        }
        let path = fixture(&sample());
        std::fs::write(&path, vec![b' '; (MAX_INPUT + 1) as usize]).unwrap();
        assert!(load(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn checked_strides_sizes_and_budget_before_native_buffers() {
        let a = layout(2, 3, 2, 16, 2).unwrap();
        let c = layout(2, 2, 4, 16, 16).unwrap();
        let out = layout(2, 2, 2, 16, 16).unwrap();
        assert_eq!(a.elements, 23);
        assert_eq!(c.elements, 71);
        assert_eq!(out.elements, 135);
        for (rows, cols, size, stride, padded) in [
            (0, 3, 2, 16, 2),
            (2, 3, 2, 4, 2),
            (2, 3, 2, 7, 2),
            (2, 3, 3, 16, 2),
            (2, 3, 2, 16, 1),
            (usize::MAX, 3, 2, 16, usize::MAX),
        ] {
            assert!(layout(rows, cols, size, stride, padded).is_err());
        }
        let (_, _, total) = budget(a, a, c, out, 18, LIMIT).unwrap();
        assert!(budget(a, a, c, out, 18, total - 1).is_err());
        assert!(budget(a, a, c, out, 18, LIMIT + 1).is_err());
        let v = baseline(&[1, 2, 3, 4, 5, 6], a);
        assert_eq!(&v[..3], &[1, 2, 3]);
        assert_eq!(&v[8..11], &[4, 5, 6]);
        assert!(v[11..].iter().all(|&n| n == HALF_GUARD));
    }
}
