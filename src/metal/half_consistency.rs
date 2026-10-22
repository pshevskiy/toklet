//! Explicit numerical investigation; no production arithmetic or gate changes.
use super::*;
use half::f16;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufRead, BufWriter, Read, Write},
    path::{Path, PathBuf},
};
// Reuse the frozen scoring math; its standalone transport helpers are unused here.
#[allow(dead_code)]
#[path = "../../examples/support/loss.rs"]
mod loss;
const TOKENS: [u32; 4] = [9707, 11, 358, 498];
const EXTRA: usize = 256 * 1024 * 1024;
const CAPTURE_LIMIT: usize = 128 * 1024 * 1024;
const STAGES: [&str; 16] = [
    "input_norm",
    "q",
    "k",
    "v",
    "q_rotary",
    "k_rotary",
    "attention_context",
    "attention_projection",
    "attention_residual",
    "post_attention_norm",
    "gate",
    "silu",
    "up",
    "gated",
    "down",
    "output",
];
struct Data {
    layer: Option<usize>,
    stage: &'static str,
    columns: usize,
    words: Vec<u32>,
}
struct Captured {
    stages: Vec<Data>,
    cache: Vec<Value>,
    admitted: usize,
}
fn digest(path: &Path) -> String {
    let mut file = fs::File::open(path).unwrap();
    let mut h = Sha256::new();
    let mut b = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut b).unwrap();
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    format!("{:x}", h.finalize())
}
fn specs(c: &model::Config, all: bool) -> Vec<Data> {
    let mut result = vec![Data {
        layer: None,
        stage: "embedding",
        columns: c.hidden_size,
        words: vec![],
    }];
    if all {
        for layer in 0..c.num_layers {
            for stage in STAGES {
                let columns = match stage {
                    "k" | "v" | "k_rotary" => c.kv_heads * c.head_dim,
                    "gate" | "silu" | "up" | "gated" => c.intermediate_size,
                    _ => c.hidden_size,
                };
                result.push(Data {
                    layer: Some(layer),
                    stage,
                    columns,
                    words: vec![],
                });
            }
        }
    }
    result.push(Data {
        layer: None,
        stage: "final_norm",
        columns: c.hidden_size,
        words: vec![],
    });
    result.push(Data {
        layer: None,
        stage: "logits",
        columns: c.vocab_size,
        words: vec![],
    });
    let bytes = result
        .iter()
        .try_fold(0usize, |sum, s| {
            sum.checked_add(s.columns.checked_mul(4 * 4).unwrap())
        })
        .unwrap();
    assert!(bytes < CAPTURE_LIMIT / 2);
    for d in &mut result {
        d.words = vec![0; 4 * d.columns];
    }
    result
}
fn write_words(dir: &Path, name: &str, shape: &[usize], words: &[u32]) -> Value {
    assert_eq!(shape.iter().product::<usize>(), words.len());
    let bytes = words.len().checked_mul(4).unwrap();
    assert!(bytes <= 10 * 1024 * 1024);
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(dir.join(name))
        .unwrap();
    let mut writer = BufWriter::new(file);
    let mut h = Sha256::new();
    for chunk in words.chunks(1024) {
        let mut b = [0u8; 4096];
        for (out, &word) in b.chunks_exact_mut(4).zip(chunk) {
            out.copy_from_slice(&word.to_le_bytes());
        }
        let b = &b[..chunk.len() * 4];
        writer.write_all(b).unwrap();
        h.update(b);
    }
    writer.flush().unwrap();
    json!({"file":name,"shape":shape,"bytes":bytes,"sha256":format!("{:x}", h.finalize()),"dtype":"f32le"})
}
fn write_halves(dir: &Path, name: &str, shape: &[usize], values: &[u16]) -> Value {
    assert_eq!(shape.iter().product::<usize>(), values.len());
    assert!(values.len() * 2 <= 10 * 1024 * 1024);
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(dir.join(name))
        .unwrap();
    let mut out = BufWriter::new(file);
    let mut h = Sha256::new();
    for chunk in values.chunks(1024) {
        let mut bytes = [0u8; 2048];
        for (b, &v) in bytes.chunks_exact_mut(2).zip(chunk) {
            b.copy_from_slice(&v.to_le_bytes());
        }
        let b = &bytes[..chunk.len() * 2];
        out.write_all(b).unwrap();
        h.update(b);
    }
    out.flush().unwrap();
    json!({"file":name,"shape":shape,"bytes":values.len()*2,"sha256":format!("{:x}", h.finalize()),"dtype":"f16le"})
}
fn capture(model: &Model, chunk: usize, all: bool, dir: &Path, tag: &str) -> Captured {
    autoreleasepool(|_| {
        let mut stages = specs(&model.config, all);
        let mut s = Session::new(
            model,
            session::Options {
                capacity: 1025,
                prefill_chunk_size: chunk,
                memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES - EXTRA,
                ..Default::default()
            },
        )
        .unwrap();
        let layers: Vec<_> = if all { (0..28).collect() } else { vec![] };
        s.enable_diagnostics(&layers).unwrap();
        let admitted = s.memory().admitted_bytes.checked_add(EXTRA).unwrap();
        assert!(admitted <= weights::INITIAL_MEMORY_LIMIT_BYTES);
        s.set_tokens(&TOKENS).unwrap();
        let mut start = 0;
        while s.pending_len() > 0 {
            let n = s.evaluate_next_chunk().unwrap();
            assert_eq!(n, chunk.min(4 - start));
            let values = s.diagnostics().unwrap();
            assert_eq!(values.len(), stages.len());
            let mut seen = std::collections::BTreeSet::new();
            for d in values {
                assert!(seen.insert((d.layer, d.stage)));
                let dst = stages
                    .iter_mut()
                    .find(|v| v.layer == d.layer && v.stage == d.stage)
                    .unwrap();
                assert_eq!(
                    (d.rows, d.columns, d.values.len()),
                    (n, dst.columns, n * dst.columns)
                );
                for (out, v) in dst.words[start * dst.columns..(start + n) * dst.columns]
                    .iter_mut()
                    .zip(d.values)
                {
                    assert!(v.is_finite());
                    *out = v.to_bits();
                }
            }
            assert_eq!(seen.len(), stages.len());
            start += n;
            assert_eq!((s.evaluated_len(), s.pending_len()), (start, 4 - start));
        }
        assert_eq!(start, 4);
        let mut cache = Vec::new();
        if all {
            for (layer, c) in s.cache.iter().enumerate() {
                for (kind, b) in [("k", &c.k), ("v", &c.v)] {
                    let mut values = Vec::with_capacity(2 * 4 * 128);
                    b.with_slice(|words| {
                        for head in 0..2 {
                            for p in 0..4 {
                                let base = (head * 1025 + p) * 128;
                                values.extend_from_slice(&words[base..base + 128]);
                            }
                        }
                    });
                    let mut item = write_halves(
                        dir,
                        &format!("{tag}.cache.{layer}.{kind}.f16le"),
                        &[2, 4, 128],
                        &values,
                    );
                    item["layer"] = json!(layer);
                    item["kind"] = json!(kind);
                    cache.push(item);
                }
            }
        }
        println!("{tag}:{}stages,admitted{admitted}", stages.len());
        Captured {
            stages,
            cache,
            admitted,
        }
    })
}
fn export(dir: &Path, tag: &str, c: &Captured) -> Value {
    let arrays: Vec<_> = c
        .stages
        .iter()
        .enumerate()
        .map(|(order, d)| {
            let mut item = write_words(
                dir,
                &format!("{tag}.{order:03}.f32le"),
                &[4, d.columns],
                &d.words,
            );
            item["layer"] = json!(d.layer);
            item["stage"] = json!(d.stage);
            item["half_storage"] = json!(d.stage != "logits");
            item
        })
        .collect();
    json!({"stages":arrays,"cache":c.cache,"admitted_bytes_including_extra":c.admitted})
}
fn half_key(bits: u16) -> u32 {
    let magnitude = u32::from(bits & 0x7fff);
    if bits & 0x8000 != 0 {
        0x8000 - magnitude
    } else {
        0x8000 + magnitude
    }
}
fn differences(a: &Captured, b: &Captured) -> Vec<Value> {
    assert_eq!(a.stages.len(), 451);
    assert_eq!(b.stages.len(), 451);
    a.stages.iter().zip(&b.stages).enumerate().map(|(order,(a,b))|{assert_eq!((a.layer,a.stage,a.columns),(b.layer,b.stage,b.columns));assert_eq!(a.words.len(),4*a.columns);assert_eq!(a.words.len(),b.words.len());let mut count=0;let mut first=None;let mut maximum=0f64;let mut ulp=0;
        for (i,(&x,&y)) in a.words.iter().zip(&b.words).enumerate(){let xval=f32::from_bits(x);let yval=f32::from_bits(y);if x!=y{count+=1;if first.is_none(){first=Some([i/a.columns,i%a.columns]);}maximum=maximum.max((f64::from(xval)-f64::from(yval)).abs());if a.stage!="logits" {ulp=ulp.max(half_key(f16::from_f32(xval).to_bits()).abs_diff(half_key(f16::from_f32(yval).to_bits())));}}}
        json!({"order":order,"layer":a.layer,"stage":a.stage,"unequal_values":count,"first_coordinate":first,"maximum_absolute_difference":maximum,"maximum_half_ulp_distance":if a.stage=="logits"{None}else{Some(ulp)}})
    }).collect()
}
fn logical_tensor_sha(t: &Tensor) -> String {
    let mut h = Sha256::new();
    t.buffer.with_slice(|words| {
        for row in 0..t.layout.rows {
            for chunk in
                words[row * t.layout.stride..row * t.layout.stride + t.layout.columns].chunks(1024)
            {
                let mut b = [0u8; 2048];
                for (v, &word) in b.chunks_exact_mut(2).zip(chunk) {
                    v.copy_from_slice(&word.to_le_bytes());
                }
                h.update(&b[..chunk.len() * 2]);
            }
        }
    });
    format!("{:x}", h.finalize())
}
#[allow(clippy::too_many_arguments)]
fn replay(
    model: &Model,
    weight: &Tensor,
    bias_values: &[u16],
    input: &Data,
    output: &Data,
    chunk: usize,
    dir: &Path,
    tag: &str,
    float_output: bool,
) -> Value {
    autoreleasepool(|_| {
        let g = &model.gpu;
        let il = Layout::new(chunk, weight.layout.columns, true).unwrap();
        let fl = Layout::new(
            chunk,
            model.config.vocab_size.max(model.config.intermediate_size),
            false,
        )
        .unwrap();
        let hl = Layout::new(chunk, weight.layout.rows, true).unwrap();
        let temporary = il
            .bytes(2)
            .unwrap()
            .checked_add(fl.bytes(4).unwrap())
            .unwrap()
            .checked_add(if float_output {
                0
            } else {
                hl.bytes(2).unwrap()
            })
            .unwrap()
            .checked_add((bias_values.len() + 7) * 2 + 32)
            .unwrap();
        assert!(temporary < EXTRA / 2);
        let mut x = Buffer::<u16>::new(g, il.elements).unwrap();
        let mut bias = Buffer::<u16>::new(g, bias_values.len() + 7).unwrap();
        bias.with_mut(|v| v[..bias_values.len()].copy_from_slice(bias_values));
        let floats = Buffer::<u32>::new(g, fl.elements).unwrap();
        let out = if float_output {
            None
        } else {
            Some(Buffer::<u16>::new(g, hl.elements).unwrap())
        };
        let mut status = Buffer::<u32>::new(g, 8).unwrap();
        let mut raw_values = Vec::with_capacity(output.words.len());
        let mut stored = Vec::with_capacity(output.words.len());
        let rows_per_call = chunk.min(4);
        for start in (0..4).step_by(rows_per_call) {
            let n = rows_per_call;
            status.with_mut(|a| a.fill(0));
            x.with_mut(|a| {
                for row in 0..n {
                    for col in 0..il.columns {
                        a[row * il.stride + col] = f16::from_f32(f32::from_bits(
                            input.words[(start + row) * il.columns + col],
                        ))
                        .to_bits();
                    }
                }
            });
            let left = matrix(&x, il.view(n, il.columns).unwrap(), true).unwrap();
            let result = matrix(&floats, fl.view(n, weight.layout.rows).unwrap(), false).unwrap();
            let kernel = multiply(g, n, weight.layout.columns, weight.layout.rows);
            let raw = g.command().unwrap();
            let command = Command {
                raw: &raw,
                trace: None,
            };
            encode_multiply(
                g,
                &command,
                weight,
                &bias,
                &kernel,
                &left,
                &result,
                &floats,
                fl,
                n,
                weight.layout.rows,
            )
            .unwrap();
            if let Some(out) = &out {
                g.encode(
                    &command,
                    Kernel::Store,
                    &[&floats.raw, &out.raw],
                    &[
                        n as u32,
                        weight.layout.rows as u32,
                        fl.stride as u32,
                        hl.stride as u32,
                    ],
                    n * weight.layout.rows,
                    Some(&status),
                )
                .unwrap();
            }
            raw.commit();
            raw.waitUntilCompleted();
            assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
            assert!(raw.error().is_none());
            assert!(status.with_slice(|a| a.iter().all(|&v| v == 0)));
            floats.with_slice(|a| {
                for row in 0..n {
                    for &bits in &a[row * fl.stride..row * fl.stride + weight.layout.rows] {
                        assert!(f32::from_bits(bits).is_finite());
                        raw_values.push(bits);
                    }
                }
            });
            if let Some(out) = &out {
                out.with_slice(|a| {
                    for row in 0..n {
                        stored.extend(
                            a[row * hl.stride..row * hl.stride + weight.layout.rows]
                                .iter()
                                .map(|&v| f16::from_bits(v).to_f32().to_bits()),
                        );
                    }
                });
            }
        }
        let equal = if float_output {
            raw_values == output.words
        } else {
            stored == output.words
        };
        let raw = write_words(
            dir,
            &format!("{tag}.raw.f32le"),
            &[4, weight.layout.rows],
            &raw_values,
        );
        let half = if float_output {
            Value::Null
        } else {
            write_words(
                dir,
                &format!("{tag}.stored.f32le"),
                &[4, weight.layout.rows],
                &stored,
            )
        };
        json!({"chunk_capacity":chunk,"active_rows_per_call":rows_per_call,"input_stride_half_elements":il.stride,"result_stride_f32_elements":fl.stride,"stored_stride_half_elements":if float_output{None}else{Some(hl.stride)},"weight_stride_half_elements":weight.layout.stride,"transpose_left":false,"transpose_right":true,"alpha":1,"beta":1,"temporary_bytes":temporary,"matches_graph_output_bits":equal,"raw":raw,"stored":half})
    })
}
fn projection_replay(
    model: &Model,
    a: &Captured,
    b: &Captured,
    stats: &[Value],
    dir: &Path,
) -> Value {
    let Some(first) = stats
        .iter()
        .find(|v| v["unequal_values"].as_u64().unwrap() > 0)
    else {
        return json!({"skipped":"No divergence"});
    };
    let order = first["order"].as_u64().unwrap() as usize;
    let d = &a.stages[order];
    let (weight, bias, predecessor, float_output) = if let Some(layer) = d.layer {
        let l = &model.layers[layer];
        let selected = match d.stage {
            "q" => (&l.q, l.q_bias.as_ref(), "input_norm"),
            "k" => (&l.k, l.k_bias.as_ref(), "input_norm"),
            "v" => (&l.v, l.v_bias.as_ref(), "input_norm"),
            "attention_projection" => (&l.o, None, "attention_context"),
            "gate" => (&l.gate, None, "post_attention_norm"),
            "up" => (&l.up, None, "post_attention_norm"),
            "down" => (&l.down, None, "gated"),
            _ => {
                return json!({"skipped":"First differing operation is not a projection","first":first});
            }
        };
        (selected.0, selected.1, selected.2, false)
    } else if d.stage == "logits" {
        (model.output_weights(), None, "final_norm", true)
    } else {
        return json!({"skipped":"First differing operation is not a projection","first":first});
    };
    assert!(matches!(&weight.storage, TensorStorage::Half(_)));
    let input = a
        .stages
        .iter()
        .find(|v| v.layer == d.layer && v.stage == predecessor)
        .unwrap();
    let other = b
        .stages
        .iter()
        .find(|v| v.layer == d.layer && v.stage == predecessor)
        .unwrap();
    let agree = input.words == other.words;
    if !agree {
        return json!({"skipped":"Relevant complete projection inputs differ","first":first});
    }
    let mut indices: Vec<_> = (0..weight.layout.rows)
        .filter(|&col| {
            (0..4).any(|row| {
                d.words[row * d.columns + col] != b.stages[order].words[row * d.columns + col]
            })
        })
        .take(64)
        .collect();
    assert!(!indices.is_empty());
    indices.sort_unstable();
    let bias_values: Vec<u16> = if let Some(bias) = bias {
        bias.buffer.with_slice(|v| v[..weight.layout.rows].to_vec())
    } else {
        vec![0; weight.layout.rows]
    };
    let subset: Vec<u16> = weight.buffer.with_slice(|v| {
        indices
            .iter()
            .flat_map(|&row| {
                v[row * weight.layout.stride..row * weight.layout.stride + weight.layout.columns]
                    .iter()
                    .copied()
            })
            .collect()
    });
    let bias_subset: Vec<_> = indices.iter().map(|&i| bias_values[i]).collect();
    let input_array = write_words(dir, "replay.input.f32le", &[4, input.columns], &input.words);
    let weights = write_halves(
        dir,
        "replay.weights.f16le",
        &[indices.len(), weight.layout.columns],
        &subset,
    );
    let biases = write_halves(dir, "replay.bias.f16le", &[indices.len()], &bias_subset);
    let weight_sha = logical_tensor_sha(weight);
    let bias_sha = bias.map(logical_tensor_sha);
    let single = replay(
        model,
        weight,
        &bias_values,
        input,
        d,
        1,
        dir,
        "replay.single",
        float_output,
    );
    let batch = replay(
        model,
        weight,
        &bias_values,
        input,
        &b.stages[order],
        128,
        dir,
        "replay.batch",
        float_output,
    );
    json!({"first":first,"complete_inputs_equal":true,"float_output":float_output,"input":input_array,"full_weight_shape":[weight.layout.rows,weight.layout.columns],"resident_logical_half_weight_sha256":weight_sha,"resident_logical_half_bias_sha256":bias_sha,"bias_is_zero":bias.is_none(),"weight_output_indices":indices,"weights":weights,"bias":biases,"single":single,"batch":batch})
}
#[test]
#[ignore = "Explicit bounded all28-layer Half consistency localization, no arithmetic/quality acceptance"]
fn larger_half_probe_capture() {
    autoreleasepool(|_| {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let dir = std::env::var_os("TOKLET_HALF_CAPTURE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                root.join(format!(
                    "artifacts/m5-half-consistency-{}",
                    std::process::id()
                ))
            });
        fs::create_dir(&dir).unwrap();
        let baseline_path =
            root.join("benchmarks/results/2026-10-14-larger-half-held-out-loss-v1.json");
        let mut baseline_bytes = Vec::new();
        fs::File::open(&baseline_path)
            .unwrap()
            .take(2 * 1024 * 1024 + 1)
            .read_to_end(&mut baseline_bytes)
            .unwrap();
        assert!(baseline_bytes.len() <= 2 * 1024 * 1024);
        let baseline_sha = format!("{:x}", Sha256::digest(&baseline_bytes));
        assert_eq!(
            baseline_sha,
            "301c6a9a245153d93e401dae93208facd263e68e4e1a446beacb11c93c6d6767"
        );
        let baseline: Value = serde_json::from_slice(&baseline_bytes).unwrap();
        assert_eq!(
            baseline["error"],
            "ValueError: Unchanged short-probe NLL gate failed"
        );
        assert_eq!(baseline["probes"][0]["tokens"], json!(TOKENS));
        let source = root.join("models/Qwen2.5-1.5B-Instruct");
        assert_eq!(
            digest(&source.join("model.safetensors")),
            "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
        );
        assert_eq!(
            digest(&source.join("config.json")),
            "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670"
        );
        let model = Model::load(
            &source,
            weights::LoadOptions {
                memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES - EXTRA,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            (
                model.config.hidden_size,
                model.config.intermediate_size,
                model.config.kv_heads,
                model.config.head_dim,
                model.config.num_layers,
                model.config.vocab_size
            ),
            (1536, 8960, 2, 128, 28, 151936)
        );
        let single_base = capture(&model, 1, false, &dir, "single_base");
        let batch_base = capture(&model, 128, false, &dir, "batch_base");
        for (index, c) in [&single_base, &batch_base].iter().enumerate() {
            let logits = &c.stages[2];
            for row in 0..3 {
                let values: Vec<_> = logits.words[row * 151936..(row + 1) * 151936]
                    .iter()
                    .map(|&v| f32::from_bits(v))
                    .collect();
                let mut h = Sha256::new();
                for chunk in logits.words[row * 151936..(row + 1) * 151936].chunks(1024) {
                    let mut bytes = [0u8; 4096];
                    for (b, &v) in bytes.chunks_exact_mut(4).zip(chunk) {
                        b.copy_from_slice(&v.to_le_bytes());
                    }
                    h.update(&bytes[..chunk.len() * 4]);
                }
                assert_eq!(
                    format!("{:x}", h.finalize()),
                    baseline["runs"][index]["losses"][row]["logits_sha256"]
                        .as_str()
                        .unwrap()
                );
                assert!(
                    (loss::loss(&values, TOKENS[row + 1] as usize).unwrap()
                        - baseline["runs"][index]["losses"][row]["negative_log_likelihood"]
                            .as_f64()
                            .unwrap())
                    .abs()
                        < 1e-12
                );
            }
        }
        let single = capture(&model, 1, true, &dir, "single");
        let batch = capture(&model, 128, true, &dir, "batch");
        let diagnostics_equal =
            [(&single_base, &single), (&batch_base, &batch)].map(|(base, all)| {
                base.stages.iter().all(|d| {
                    d.words
                        == all
                            .stages
                            .iter()
                            .find(|v| v.layer == d.layer && v.stage == d.stage)
                            .unwrap()
                            .words
                })
            });
        let stats = differences(&single, &batch);
        let replay = projection_replay(&model, &single, &batch, &stats, &dir);
        let retained_bytes = [&single_base, &batch_base, &single, &batch]
            .iter()
            .map(|c| c.stages.iter().map(|d| d.words.len() * 4).sum::<usize>())
            .sum::<usize>();
        assert!(retained_bytes < CAPTURE_LIMIT);
        let record = json!({"format":"toklet-half-consistency-capture-v1","scope":"All-layer localization and first-operation replay only; no arithmetic or full quality acceptance","tokens":TOKENS,"capacity":1025,"chunks":[1,128],"projection_mode":"Original","normalization_mode":"Original","device":model.device_name(),"baseline_sha256":baseline_sha,"extra_admitted_bytes":EXTRA,"retained_host_words_bytes":retained_bytes,"diagnostics_preserve_scorer_stage_bits":diagnostics_equal,"single_base":export(&dir,"single_base",&single_base),"batch_base":export(&dir,"batch_base",&batch_base),"single":export(&dir,"single",&single),"batch":export(&dir,"batch",&batch),"stage_differences":stats,"projection_replay":replay});
        let bytes = serde_json::to_vec_pretty(&record).unwrap();
        assert!(bytes.len() < 2 * 1024 * 1024);
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.join("manifest.json"))
            .unwrap()
            .write_all(&bytes)
            .unwrap();
        println!("\nTOKLET_HALF_CAPTURE_DONE");
        std::io::stdout().flush().unwrap();
        let mut line = String::new();
        assert_eq!(
            std::io::stdin()
                .lock()
                .take(16)
                .read_line(&mut line)
                .unwrap(),
            9
        );
        assert_eq!(line, "continue\n");
    });
}
