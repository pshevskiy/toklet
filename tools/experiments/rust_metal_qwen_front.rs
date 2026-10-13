// Resident layer0 input RMSNorm -> Q/K/V. Numerical qualification only.
use super::*;
use mlp::{ProjectionStep, active_float, active_half, encode_projection, input, kernel, totals};
use objc2_metal_performance_shaders::MPSDataType;
use projection::{half, matrix, number, queried, weight_file};
use std::{io::Read, path::Path};
struct Case {
    id: String,
    rows: usize,
    input: Vec<u16>,
    norm: Vec<u16>,
    weights: [Vec<u16>; 3],
    biases: [Vec<u16>; 3],
}
fn parse(v: &Value, parent: &Path) -> Result<Case, Error> {
    if v["format"] != "toklet-metal-qwen-front-input-v1" {
        return Err(error("fixture", "Qwen front format differs"));
    }
    let id = v["id"]
        .as_str()
        .filter(|id| !id.is_empty() && id.len() <= 128 && id.is_ascii())
        .ok_or_else(|| error("fixture", "Qwen front identity differs"))?;
    let rows = number(v, "rows")?;
    if !(1..=4).contains(&rows) || v["epsilon_bits"].as_u64() != Some(1.0e-6f32.to_bits() as u64) {
        return Err(error("fixture", "Qwen front rows/epsilon differ"));
    }
    let input = half(v, "input_bits", rows * 896)?;
    let norm = half(v, "norm_weight_bits", 896)?;
    let biases = [
        half(v, "q_bias_bits", 896)?,
        half(v, "k_bias_bits", 128)?,
        half(v, "v_bias_bits", 128)?,
    ];
    let weight = |op: &str, count: usize| {
        weight_file(
            parent,
            &json!({"weight_file":v[format!("{op}_weight_file")]}),
            count,
        )
    };
    Ok(Case {
        id: id.to_owned(),
        rows,
        input,
        norm,
        weights: [
            weight("q", 896 * 896)?,
            weight("k", 128 * 896)?,
            weight("v", 128 * 896)?,
        ],
        biases,
    })
}
fn load(path: &Path) -> Result<Case, Error> {
    let file = std::fs::File::open(path).map_err(|e| error("fixture", e.to_string()))?;
    let size = file
        .metadata()
        .map_err(|e| error("fixture", e.to_string()))?
        .len();
    if size > 1024 * 1024 {
        return Err(error("fixture", "Qwen front input exceeds one MiB"));
    }
    let mut data = Vec::new();
    file.take(1024 * 1024 + 1)
        .read_to_end(&mut data)
        .map_err(|e| error("fixture", e.to_string()))?;
    if data.len() as u64 != size || data.len() > 1024 * 1024 {
        return Err(error("fixture", "Qwen front input changed length"));
    }
    let v = serde_json::from_slice(&data).map_err(|e| error("fixture", e.to_string()))?;
    parse(&v, path.parent().unwrap_or_else(|| Path::new(".")))
}
#[derive(PartialEq)]
struct Frame {
    half: Vec<Vec<u16>>,
    single: Vec<Vec<u32>>,
}
pub(super) fn run(path: &Path, limit: usize) -> Result<Value, Error> {
    let c = load(path)?;
    autoreleasepool(|_| {
        let (probe, mut metadata, pipelines, caps) = layer::setup()?;
        let n = c.rows;
        let il = queried(n, 896, 2, n)?;
        let nl = queried(1, 896, 2, 1)?;
        let qw = queried(896, 896, 2, 896)?;
        let kw = queried(128, 896, 2, 128)?;
        let qb = queried(1, 896, 2, 1)?;
        let kb = queried(1, 128, 2, 1)?;
        let normalized = queried(n, 896, 2, groups(n, probe.width)? * probe.width)?;
        let mut half_layouts = vec![normalized];
        let mut float_layouts = Vec::new();
        for width in [896, 128, 128] {
            let capacity = (groups(n * width, probe.width)? * probe.width).div_ceil(width);
            half_layouts.push(queried(n, width, 2, capacity)?);
            float_layouts.push(queried(n, width, 4, capacity)?);
        }
        if il.stride != 896 || normalized.stride != 896 {
            return Err(error(
                "capability",
                "Qwen RMS requires contiguous Half rows",
            ));
        }
        let input_layouts = [il, nl, qw, kw, kw, qb, kb, kb];
        let sum = |layouts: &[projection::Layout]| {
            layouts.iter().try_fold(0usize, |n, l| {
                n.checked_add(l.bytes)
                    .ok_or_else(|| error("admission", "Qwen front bytes overflow"))
            })
        };
        let (shared, total) = totals(
            sum(&input_layouts)?,
            sum(&half_layouts)?
                .checked_add(sum(&float_layouts)?)
                .ok_or_else(|| error("admission", "Qwen front output bytes overflow"))?,
            limit,
        )?;
        let inputs = [
            input(&probe.device, il, &c.input)?,
            input(&probe.device, nl, &c.norm)?,
            input(&probe.device, qw, &c.weights[0])?,
            input(&probe.device, kw, &c.weights[1])?,
            input(&probe.device, kw, &c.weights[2])?,
            input(&probe.device, qb, &c.biases[0])?,
            input(&probe.device, kb, &c.biases[1])?,
            input(&probe.device, kb, &c.biases[2])?,
        ];
        let mut halves = half_layouts
            .iter()
            .map(|l| Shared::<u16>::new(&probe.device, l.bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let mut singles = float_layouts
            .iter()
            .map(|l| Shared::<u32>::new(&probe.device, l.bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let left = matrix(&halves[0], normalized, MPSDataType::Float16)?;
        let weights = [
            matrix(&inputs[2].buffer, qw, MPSDataType::Float16)?,
            matrix(&inputs[3].buffer, kw, MPSDataType::Float16)?,
            matrix(&inputs[4].buffer, kw, MPSDataType::Float16)?,
        ];
        let results = (0..3)
            .map(|i| matrix(&singles[i], float_layouts[i], MPSDataType::Float32))
            .collect::<Result<Vec<_>, _>>()?;
        let kernels = [
            kernel(&probe.device, n, 896, 896),
            kernel(&probe.device, n, 896, 128),
            kernel(&probe.device, n, 896, 128),
        ];
        let mut first: Option<Frame> = None;
        let mut repetitions = Vec::with_capacity(13);
        for repetition in 0..13 {
            for b in &mut halves {
                b.fill(0x55a1);
            }
            for b in &mut singles {
                b.fill(0x7fc05678);
            }
            let (frame, seconds) = autoreleasepool(|_| {
                let start = Instant::now();
                let command = probe
                    .queue
                    .commandBuffer()
                    .ok_or_else(|| error("command", "Qwen front command unavailable"))?;
                if !command.retainedReferences() {
                    return Err(error(
                        "command",
                        "unretained Qwen front resources unsupported",
                    ));
                }
                // Qualified RMS ABI: Half input/norm/output at0/1/2, UInt32 params12/4
                // at3. Contiguous width896, rounded row storage, no alias. All initialized
                // owners survive checked completion. No host access between encodings.
                layer::encode(
                    &command,
                    &probe,
                    &pipelines[4],
                    &[&inputs[0].buffer.raw, &inputs[1].buffer.raw, &halves[0].raw],
                    &[n as u32, 896, 1.0e-6f32.to_bits()],
                    n,
                )?;
                for i in 0..3 {
                    encode_projection(
                        &command,
                        &probe,
                        &pipelines[0],
                        ProjectionStep {
                            bias: &inputs[5 + i].buffer,
                            floats: &singles[i],
                            output: &halves[1 + i],
                            fl: float_layouts[i],
                            hl: half_layouts[1 + i],
                            kernel: &kernels[i],
                            left: &left,
                            right: &weights[i],
                            result: &results[i],
                        },
                    )?;
                }
                command.commit();
                command.waitUntilCompleted();
                let seconds = start.elapsed().as_secs_f64();
                if command.status() != MTLCommandBufferStatus::Completed
                    || command.error().is_some()
                {
                    return Err(error("completion", "Qwen front GPU command failed"));
                }
                if inputs.iter().any(|i| i.buffer.read() != i.baseline) {
                    return Err(error("validation", "Qwen front immutable inputs changed"));
                }
                let half = halves
                    .iter()
                    .zip(&half_layouts)
                    .map(|(b, l)| active_half(b, *l))
                    .collect::<Result<Vec<_>, _>>()?;
                let single = singles
                    .iter()
                    .zip(&float_layouts)
                    .map(|(b, l)| active_float(b, *l))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((Frame { half, single }, seconds))
            })
            .map_err(|e: Error| {
                if e.stage == "validation"
                    && e.message == "completed MLP nonfinite; result discarded"
                {
                    error(
                        "validation",
                        "completed Qwen front nonfinite; result discarded",
                    )
                } else {
                    e
                }
            })?;
            if first.as_ref().is_some_and(|f| f != &frame) {
                return Err(error("validation", "Qwen front repeated frames differ"));
            }
            let output = frame.half[1..].to_vec();
            if first.is_none() {
                first = Some(frame);
            }
            repetitions.push(json!({"phase":if repetition<3{"warmup"}else{"sample"},"repetition":if repetition<3{repetition+1}else{repetition-2},"command_completed":true,"commit_wait_pairs":1,"inputs_guards_and_intermediates_unchanged":true,"encode_commit_wait_seconds":seconds,"output_bits":output}));
        }
        let first = first.unwrap();
        metadata["maximum_admitted_bytes"] = json!(total);
        metadata["mps_device_supported"] = json!(true);
        metadata["graph_pipeline_caps"] = json!(caps);
        Ok(
            json!({"operation":"resident_qwen_layer0_input_norm_qkv","id":c.id,"rows_count":n,"metadata":metadata,"threadgroup_width":probe.width,"input_layouts":input_layouts.map(layer::describe),"half_layouts":half_layouts.into_iter().map(layer::describe).collect::<Vec<_>>(),"float_layouts":float_layouts.into_iter().map(layer::describe).collect::<Vec<_>>(),"shared_buffer_bytes":shared,"host_snapshot_readback_bytes":shared*2,"admitted_bytes":total,"allocations_reused":true,"first_half_intermediates":first.half,"first_fp32_intermediates":first.single,"rows":repetitions}),
        )
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parser_rejects_shape_epsilon_and_nonfinite_before_weight_reads() {
        let v = json!({"format":"toklet-metal-qwen-front-input-v1","id":"manual","rows":1,"epsilon_bits":1.0e-6f32.to_bits(),"input_bits":vec![0u16;896],"norm_weight_bits":vec![0u16;896],"q_bias_bits":vec![0u16;896],"k_bias_bits":vec![0u16;128],"v_bias_bits":vec![0u16;128]});
        for (key, val) in [
            ("rows", json!(0)),
            ("rows", json!(5)),
            ("epsilon_bits", json!(1)),
            ("input_bits", json!([])),
            ("norm_weight_bits", json!(vec![0x7c00u16; 896])),
            ("id", json!("")),
        ] {
            let mut v = v.clone();
            v[key] = val;
            assert!(parse(&v, Path::new("/nonexistent")).is_err());
        }
    }
}
