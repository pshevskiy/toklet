// Tiny full-prefix numerical decoder qualification; no application/session API.
use super::*;
use mlp::{active_float, active_half, input, kernel, totals};
use objc2_metal_performance_shaders::MPSDataType;
use projection::{half, matrix, number, queried};
use std::{io::Read, path::Path};
struct Case {
    id: String,
    rows: usize,
    position: usize,
    tokens: Vec<u32>,
    table: Vec<u16>,
    norm: Vec<u16>,
    layers: [layer::Case; 2],
}
fn parse(v: &Value) -> Result<Case, Error> {
    if v["format"] != "toklet-metal-decoder-input-v1" {
        return Err(error("fixture", "decoder format differs"));
    }
    let id = v["id"]
        .as_str()
        .filter(|id| !id.is_empty() && id.len() <= 128 && id.is_ascii())
        .ok_or_else(|| error("fixture", "decoder identity differs"))?;
    let rows = number(v, "rows")?;
    let position = number(v, "position")?;
    if !(1..=32).contains(&rows) || position.checked_add(rows).is_none_or(|n| n > 2048) {
        return Err(error("fixture", "decoder shape/position outside bounds"));
    }
    let tokens = v["tokens"]
        .as_array()
        .filter(|a| a.len() == rows)
        .ok_or_else(|| error("fixture", "decoder token shape differs"))?
        .iter()
        .map(|n| {
            n.as_u64()
                .filter(|n| *n < 64)
                .map(|n| n as u32)
                .ok_or_else(|| error("fixture", "decoder token outside vocabulary"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let layers = v["layers"]
        .as_array()
        .filter(|a| a.len() == 2)
        .ok_or_else(|| error("fixture", "decoder requires two layers"))?;
    let parse_layer = |index: usize| {
        if !layers[index].is_object() {
            return Err(error("fixture", "decoder layer must be an object"));
        }
        let mut value = layers[index].clone();
        // Layer fixture parser is reused for weights/shape validation. Incoming values
        // are supplied exclusively by GPU buffers in encode, never by this placeholder.
        if value.get("input_bits").is_some() {
            return Err(error("fixture", "decoder forbids saved layer inputs"));
        }
        value["format"] = json!("toklet-metal-layer-input-v1");
        value["rows"] = json!(rows);
        value["position"] = json!(position);
        value["id"] = json!(format!("{id}.layers.{index}"));
        value["input_bits"] = json!(vec![0u16; rows * 32]);
        layer::parse(&value)
    };
    Ok(Case {
        id: id.to_owned(),
        rows,
        position,
        tokens,
        table: half(v, "embedding_weight_bits", 2048)?,
        norm: half(v, "final_norm_weight_bits", 32)?,
        layers: [parse_layer(0)?, parse_layer(1)?],
    })
}
fn load(path: &Path) -> Result<Case, Error> {
    let file = std::fs::File::open(path).map_err(|e| error("fixture", e.to_string()))?;
    let size = file
        .metadata()
        .map_err(|e| error("fixture", e.to_string()))?
        .len();
    if size > 1024 * 1024 {
        return Err(error("fixture", "decoder input exceeds one MiB"));
    }
    let mut data = Vec::new();
    file.take(1024 * 1024 + 1)
        .read_to_end(&mut data)
        .map_err(|e| error("fixture", e.to_string()))?;
    if data.len() as u64 != size || data.len() > 1024 * 1024 {
        return Err(error("fixture", "decoder changed size or exceeds bounds"));
    }
    let v = serde_json::from_slice(&data).map_err(|e| error("fixture", e.to_string()))?;
    parse(&v)
}
fn greedy(bits: &[u32]) -> Result<usize, Error> {
    if bits.len() != 64 {
        return Err(error("validation", "decoder logits vocabulary differs"));
    }
    let mut best = 0;
    let mut maximum = f32::NEG_INFINITY;
    for (index, &bits) in bits.iter().enumerate() {
        let value = f32::from_bits(bits);
        if !value.is_finite() {
            return Err(error(
                "validation",
                "decoder nonfinite logits; result discarded",
            ));
        }
        if value > maximum {
            best = index;
            maximum = value;
        }
    }
    Ok(best)
}
pub(super) fn run(path: &Path, limit: usize) -> Result<Value, Error> {
    let c = load(path)?;
    autoreleasepool(|_| {
        let (mut probe, mut metadata, pipelines, caps) = layer::setup()?;
        let function = probe
            .library
            .newFunctionWithName(&NSString::from_str("toklet_embedding"))
            .ok_or_else(|| error("pipeline", "embedding unavailable"))?;
        let embedding = probe
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| error("pipeline", e.to_string()))?;
        let width = embedding
            .threadExecutionWidth()
            .min(embedding.maxTotalThreadsPerThreadgroup());
        if width == 0 {
            return Err(error("capability", "embedding width zero"));
        }
        probe.width = probe.width.min(width);
        let plan = layer::Plan::new(c.rows, &probe)?;
        let tl = queried(64, 32, 2, 64)?;
        let nl = queried(1, 32, 2, 1)?;
        let zl = queried(1, 64, 2, 1)?;
        let hl = plan.half_layouts[0];
        let fl = queried(c.rows, 64, 4, c.rows)?;
        let token_bytes = (c.rows + GUARDS)
            .checked_mul(4)
            .ok_or_else(|| error("admission", "decoder token bytes overflow"))?;
        let outer = tl
            .bytes
            .checked_add(nl.bytes)
            .and_then(|n| n.checked_add(zl.bytes))
            .and_then(|n| n.checked_add(token_bytes))
            .and_then(|n| n.checked_add(hl.bytes.checked_mul(2)?))
            .and_then(|n| n.checked_add(fl.bytes))
            .ok_or_else(|| error("admission", "decoder bytes overflow"))?;
        let layers = plan
            .bytes()?
            .checked_mul(2)
            .ok_or_else(|| error("admission", "decoder layer bytes overflow"))?;
        // Combined admission before any numerical buffer; tied table is counted once.
        let (shared, total) = totals(outer, layers, limit)?;
        if tl.stride != 32 || hl.stride != 32 {
            return Err(error(
                "capability",
                "decoder contiguous Half layout differs",
            ));
        }
        let table = input(&probe.device, tl, &c.table)?;
        let norm = input(&probe.device, nl, &c.norm)?;
        let zero = input(&probe.device, zl, &[0u16; 64])?;
        let mut tokens = Shared::<u32>::new(&probe.device, token_bytes)?;
        let mut token_snapshot = vec![0x7fc05678; c.rows + GUARDS];
        token_snapshot[..c.rows].copy_from_slice(&c.tokens);
        tokens.write(&token_snapshot)?;
        let mut embedded = Shared::<u16>::new(&probe.device, hl.bytes)?;
        let mut normalized = Shared::<u16>::new(&probe.device, hl.bytes)?;
        let mut logits = Shared::<u32>::new(&probe.device, fl.bytes)?;
        let mut graphs = [
            layer::Prepared::new(&c.layers[0], plan, &probe)?,
            layer::Prepared::new(&c.layers[1], plan, &probe)?,
        ];
        let left = matrix(&normalized, hl, MPSDataType::Float16)?;
        let right = matrix(&table.buffer, tl, MPSDataType::Float16)?;
        let result = matrix(&logits, fl, MPSDataType::Float32)?;
        let output = kernel(&probe.device, c.rows, 32, 64);
        let mut first = None;
        let mut repetitions = Vec::with_capacity(13);
        for repetition in 0..13 {
            for g in &mut graphs {
                g.reset()?;
            }
            embedded.fill(0x55a1);
            normalized.fill(0x55a1);
            logits.fill(0x7fc05678);
            let (frame, seconds) = autoreleasepool(|_| {
                let start = Instant::now();
                let command = probe
                    .queue
                    .commandBuffer()
                    .ok_or_else(|| error("command", "decoder command unavailable"))?;
                if !command.retainedReferences() {
                    return Err(error("command", "unretained decoder resources unsupported"));
                }
                layer::encode(
                    &command,
                    &probe,
                    &embedding,
                    &[&table.buffer.raw, &tokens.raw, &embedded.raw],
                    &[(c.rows * 32) as u32, 32, 64],
                    c.rows * 32,
                )?;
                graphs[0].encode(&command, &probe, &pipelines, &embedded, c.rows, c.position)?;
                graphs[1].encode(
                    &command,
                    &probe,
                    &pipelines,
                    &graphs[0].halves[17],
                    c.rows,
                    c.position,
                )?;
                layer::encode(
                    &command,
                    &probe,
                    &pipelines[4],
                    &[&graphs[1].halves[17].raw, &norm.buffer.raw, &normalized.raw],
                    &[c.rows as u32, 32, 1.0e-6f32.to_bits()],
                    c.rows,
                )?;
                layer::encode(
                    &command,
                    &probe,
                    &pipelines[0],
                    &[&zero.buffer.raw, &logits.raw],
                    &[c.rows as u32, 64, fl.stride as u32, hl.stride as u32],
                    c.rows * 64,
                )?;
                // SAFETY: admitted nonalias same-device Half [rows,32]/[64,32] and
                // FP32 [rows,64] matrices. The identical resident embedding buffer is the
                // transposed right matrix; alpha=beta=1 with zero initialized FP32 result.
                // Tracked encoder dependencies and retained owners survive the checked wait.
                unsafe {
                    output.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                        &command, &left, &right, &result,
                    );
                }
                command.commit();
                command.waitUntilCompleted();
                let seconds = start.elapsed().as_secs_f64();
                if command.status() != MTLCommandBufferStatus::Completed
                    || command.error().is_some()
                {
                    return Err(error("completion", "decoder GPU command failed"));
                }
                if tokens.read() != token_snapshot
                    || [&table, &norm, &zero]
                        .iter()
                        .any(|i| i.buffer.read() != i.baseline)
                {
                    return Err(error("validation", "decoder immutable inputs changed"));
                }
                let layer_frames = [graphs[0].frame()?, graphs[1].frame()?];
                Ok((
                    (
                        layer_frames,
                        active_half(&embedded, hl)?,
                        active_half(&normalized, hl)?,
                        active_float(&logits, fl)?,
                    ),
                    seconds,
                ))
            })?;
            if first.as_ref().is_some_and(|f| f != &frame) {
                return Err(error("validation", "decoder repeated intermediates differ"));
            }
            let last = frame.3[(c.rows - 1) * 64..].to_vec();
            let token = greedy(&last)?;
            if first.is_none() {
                first = Some(frame);
            }
            repetitions.push(json!({"phase":if repetition<3{"warmup"}else{"sample"},"repetition":if repetition<3{repetition+1}else{repetition-2},"command_completed":true,"commit_wait_pairs":1,"inputs_guards_and_intermediates_unchanged":true,"encode_commit_wait_seconds":seconds,"last_logits_bits":last,"token_id":token}));
        }
        let (first_layers, first_embedding, first_normalized, first_logits) = first.unwrap();
        metadata["maximum_admitted_bytes"] = json!(total);
        metadata["mps_device_supported"] = json!(true);
        metadata["graph_pipeline_caps"] = json!(caps);
        metadata["embedding_pipeline_caps"] = json!({"execution_width":embedding.threadExecutionWidth(),"max_threads":embedding.maxTotalThreadsPerThreadgroup()});
        Ok(
            json!({"operation":"resident_tiny_decoder_half_boundaries","id":c.id,"rows_count":c.rows,"position":c.position,"metadata":metadata,"threadgroup_width":probe.width,"layer_weight_layouts":plan.input_layouts[1..].iter().copied().map(layer::describe).collect::<Vec<_>>(),"layer_half_layouts":plan.half_layouts.map(layer::describe),"layer_float_layouts":plan.float_layouts.map(layer::describe),"outer_layouts":([tl,nl,zl,hl,hl,fl].map(layer::describe)),"token_buffer_bytes":token_bytes,"shared_buffer_bytes":shared,"host_snapshot_readback_bytes":shared*2,"admitted_bytes":total,"allocations_reused":true,"tied_embedding_allocation":true,"first_layers":first_layers.map(|f|json!({"half":f.half,"single":f.single})),"first_embedding_bits":first_embedding,"first_normalized_bits":first_normalized,"first_logits_bits":first_logits,"rows":repetitions}),
        )
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn greedy_finite_and_lowest_id_ties() {
        let mut bits = vec![0f32.to_bits(); 64];
        bits[8] = 1f32.to_bits();
        bits[3] = 1f32.to_bits();
        assert_eq!(greedy(&bits).unwrap(), 3);
        bits[1] = f32::NAN.to_bits();
        assert!(greedy(&bits).is_err());
        assert!(greedy(&[]).is_err());
    }
    #[test]
    fn decoder_layer_entries_must_be_objects() {
        for entry in [json!(true), json!(9), json!([]), json!(null)] {
            let v = json!({"format":"toklet-metal-decoder-input-v1","id":"manual","rows":1,"position":0,"tokens":[1],"embedding_weight_bits":vec![0u16;2048],"final_norm_weight_bits":vec![0u16;32],"layers":[entry,{}]});
            assert_eq!(
                parse(&v).err().unwrap().message,
                "decoder layer must be an object"
            );
        }
    }
    #[test]
    fn decoder_parser_rejects_token_shapes_and_vocabulary_before_weights() {
        let v = json!({"format":"toklet-metal-decoder-input-v1","id":"manual","rows":1,"position":0,"tokens":[64]});
        assert!(parse(&v).err().unwrap().message.contains("vocabulary"));
        for (key, val) in [
            ("rows", json!(33)),
            ("position", json!(2048)),
            ("tokens", json!([])),
            ("tokens", json!([1.0])),
        ] {
            let mut v = v.clone();
            v[key] = val;
            assert!(parse(&v).is_err());
        }
    }
}
