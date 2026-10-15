//! Isolated M6 component experiment, never part of the production source tree.
//! See benchmarks/m6-kv-component-v1/manifest.json in the parent repository.
use super::*;
use half::f16;
use objc2_foundation::NSString;
use objc2_metal::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    io::{self, Read, Write},
    ptr::NonNull,
};

const GROUP: usize = 32;
const GUARD: usize = 7;
const DEPTH: usize = 32768;
const FRAMEWORK: usize = 512 * 1024 * 1024;
const DRIVER: usize = 64 * 1024 * 1024;
const SCRATCH: usize = 1024 * 1024;
const LIMIT: usize = 6 * 1024 * 1024 * 1024;
#[path = "kv_probe_witness.rs"]
mod witness;

struct PackedGroup {
    words: [u16; GROUP / 2],
    scale: u16,
    clamped: usize,
    max_error: f64,
    square_error: f64,
}

fn reconstructed(word: u16, odd: bool, scale: u16) -> Result<f32> {
    let byte = (word >> if odd { 8 } else { 0 }) as u8;
    let code = i8::from_le_bytes([byte]);
    if scale & 0x8000 != 0
        || scale & 0x7c00 == 0x7c00
        || code == i8::MIN
        || (scale == 0 && code != 0)
    {
        return Err(Error::Options("invalid component I8 code/scale"));
    }
    Ok(f32::from(code) * f16::from_bits(scale).to_f32())
}

fn pack_group(input: &[u16]) -> Result<PackedGroup> {
    if input.len() != GROUP {
        return Err(Error::Options("component requires complete32-value groups"));
    }
    let mut maximum = 0.0f32;
    for &bits in input {
        let value = f16::from_bits(bits).to_f32();
        if !value.is_finite() {
            return Err(Error::Nonfinite);
        }
        maximum = maximum.max(value.abs());
    }
    let scale = if maximum == 0.0 {
        0
    } else {
        f16::from_f32(maximum / 127.0).to_bits().max(1)
    };
    let step = f16::from_bits(scale).to_f32();
    let mut result = PackedGroup {
        words: [0; GROUP / 2],
        scale,
        clamped: 0,
        max_error: 0.0,
        square_error: 0.0,
    };
    for (i, &bits) in input.iter().enumerate() {
        let original = f16::from_bits(bits).to_f32();
        let rounded = if scale == 0 {
            0.0
        } else {
            (original / step).round_ties_even()
        };
        let bounded = rounded.clamp(-127.0, 127.0);
        result.clamped += usize::from(rounded != bounded);
        let byte = (bounded as i8).to_le_bytes()[0];
        result.words[i / 2] |= u16::from(byte) << ((i % 2) * 8);
        let value = reconstructed(result.words[i / 2], i % 2 != 0, scale)?;
        let error = f64::from(value) - f64::from(original);
        result.max_error = result.max_error.max(error.abs());
        result.square_error += error * error;
    }
    Ok(result)
}

#[derive(Clone, Copy)]
struct Geometry {
    layers: usize,
    heads: usize,
    d: usize,
}
impl Geometry {
    fn named(name: &str) -> Result<Self> {
        match name {
            "small" => Ok(Self {
                layers: 24,
                heads: 14,
                d: 64,
            }),
            "larger" => Ok(Self {
                layers: 28,
                heads: 12,
                d: 128,
            }),
            _ => Err(Error::Options("component requires small/larger geometry")),
        }
    }
    fn query(self) -> usize {
        self.heads * self.d
    }
    fn cache(self) -> usize {
        2 * DEPTH * self.d
    }
    fn shape(self) -> [u32; 8] {
        [
            1,
            self.heads as u32,
            2,
            self.d as u32,
            DEPTH as u32,
            1,
            DEPTH as u32 - 1,
            DEPTH as u32,
        ]
    }
    fn admitted(self) -> Result<usize> {
        let layer_words = add(self.query(), GUARD)?
            .checked_add(
                add(self.cache(), GUARD)?
                    .checked_mul(2)
                    .ok_or(Error::Options("KV bytes overflow"))?,
            )
            .and_then(|n| n.checked_add((self.cache() / 2 + GUARD) * 2))
            .and_then(|n| n.checked_add((self.cache() / GROUP + GUARD) * 2))
            .ok_or(Error::Options("component layer bytes overflow"))?;
        let buffers = layer_words
            .checked_mul(2)
            .and_then(|n| n.checked_mul(self.layers))
            .and_then(|n| n.checked_add((self.query() + GUARD) * 2 + 8 * 4))
            .ok_or(Error::Options("component allocation bytes overflow"))?;
        // One32768-entry FP64 score row plus small output/query/packing storage.
        add(
            add(buffers, FRAMEWORK)?,
            DRIVER + DEPTH * 8 + self.query() * 16 + SCRATCH,
        )
    }
}

fn generated(index: usize, seed: u64) -> u16 {
    let mut value = (index as u64).wrapping_add(seed.wrapping_mul(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^= value >> 31;
    f16::from_f32(((value >> 53) as f32 / 512.0) - 2.0).to_bits()
}

struct Layer {
    q: Buffer<u16>,
    k: Buffer<u16>,
    v: Buffer<u16>,
    kc: Buffer<u16>,
    vc: Buffer<u16>,
    ks: Buffer<u16>,
    vs: Buffer<u16>,
}

fn digest(buffer: &Buffer<u16>) -> String {
    buffer.with_slice(|words| {
        let mut hash = Sha256::new();
        let mut bytes = [0u8; 4096];
        for chunk in words.chunks(bytes.len() / 2) {
            for (out, value) in bytes.chunks_exact_mut(2).zip(chunk) {
                out.copy_from_slice(&value.to_le_bytes());
            }
            hash.update(&bytes[..chunk.len() * 2]);
        }
        format!("{:x}", hash.finalize())
    })
}

fn fill(buffer: &mut Buffer<u16>, payload: usize, seed: u64) {
    buffer.with_mut(|words| {
        words.fill(0x7e00);
        for (i, value) in words[..payload].iter_mut().enumerate() {
            *value = generated(i, seed);
        }
    });
}

fn quantize(
    source: &Buffer<u16>,
    codes: &mut Buffer<u16>,
    scales: &mut Buffer<u16>,
    payload: usize,
) -> Result<Value> {
    let mut maximum = 0.0f64;
    let mut square = 0.0f64;
    let mut clamped = 0usize;
    source.with_slice(|original| {
        codes.with_mut(|packed| {
            scales.with_mut(|metadata| {
                packed.fill(0x8080);
                metadata.fill(0x7e00);
                for (i, group) in original[..payload].chunks_exact(GROUP).enumerate() {
                    let value = pack_group(group)?;
                    packed[i * GROUP / 2..(i + 1) * GROUP / 2].copy_from_slice(&value.words);
                    metadata[i] = value.scale;
                    maximum = maximum.max(value.max_error);
                    square += value.square_error;
                    clamped += value.clamped;
                }
                Ok::<(), Error>(())
            })
        })
    })?;
    Ok(json!({"values":payload,"max_reconstruction_error":maximum,
              "rms_reconstruction_error":(square/payload as f64).sqrt(),"clamped_values":clamped}))
}

impl Layer {
    fn new(gpu: &Gpu, g: Geometry, index: usize) -> Result<(Self, Value)> {
        let mut layer = Self {
            q: Buffer::new(gpu, g.query() + GUARD)?,
            k: Buffer::new(gpu, g.cache() + GUARD)?,
            v: Buffer::new(gpu, g.cache() + GUARD)?,
            kc: Buffer::new(gpu, g.cache() / 2 + GUARD)?,
            vc: Buffer::new(gpu, g.cache() / 2 + GUARD)?,
            ks: Buffer::new(gpu, g.cache() / GROUP + GUARD)?,
            vs: Buffer::new(gpu, g.cache() / GROUP + GUARD)?,
        };
        let seed = index as u64 * 3 + 1;
        fill(&mut layer.q, g.query(), seed);
        fill(&mut layer.k, g.cache(), seed + 1);
        fill(&mut layer.v, g.cache(), seed + 2);
        let k = quantize(&layer.k, &mut layer.kc, &mut layer.ks, g.cache())?;
        let v = quantize(&layer.v, &mut layer.vc, &mut layer.vs, g.cache())?;
        Ok((layer, json!({"layer":index,"k":k,"v":v})))
    }
    fn hashes(&self) -> Vec<String> {
        [
            &self.q, &self.k, &self.v, &self.kc, &self.vc, &self.ks, &self.vs,
        ]
        .map(digest)
        .to_vec()
    }
}

type Pipeline = Retained<objc2::runtime::ProtocolObject<dyn MTLComputePipelineState>>;
struct Candidate {
    pipeline: Pipeline,
    checked: Pipeline,
    words: Pipeline,
    checked_words: Pipeline,
    width: usize,
}
impl Candidate {
    fn new(gpu: &Gpu) -> Result<Self> {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version2_3);
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        if options.languageVersion() != MTLLanguageVersion::Version2_3
            || options.mathMode() != MTLMathMode::Safe
            || options.mathFloatingPointFunctions() != MTLMathFloatingPointFunctions::Precise
        {
            return Err(Error::Unsupported("component precise policy unavailable"));
        }
        let source = format!(
            "{}\n{}",
            include_str!("kernels.metal"),
            include_str!("kv_probe.metal")
        );
        let library = gpu
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options))
            .map_err(|e| Error::Native(e.to_string()))?;
        let make = |name: &str| -> Result<Pipeline> {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or(Error::Options("component kernel missing"))?;
            gpu.device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| Error::Native(e.to_string()))
        };
        let pipeline = make("attention_decode_i8_probe")?;
        let checked = make("attention_decode_i8_probe_checked")?;
        let words = make("i8_words_probe")?;
        let checked_words = make("i8_words_probe_checked")?;
        let width = pipeline.threadExecutionWidth();
        if width < 32
            || width != gpu.execution_width(Kernel::AttentionDecode)
            || width > pipeline.maxTotalThreadsPerThreadgroup()
            || width != checked.threadExecutionWidth()
            || width > checked.maxTotalThreadsPerThreadgroup()
        {
            return Err(Error::Unsupported("component queried width differs"));
        }
        for pipeline in [&words, &checked_words] {
            if pipeline.threadExecutionWidth() == 0
                || pipeline.threadExecutionWidth() > pipeline.maxTotalThreadsPerThreadgroup()
            {
                return Err(Error::Unsupported(
                    "component word witness width unavailable",
                ));
            }
        }
        Ok(Self {
            pipeline,
            checked,
            words,
            checked_words,
            width,
        })
    }
    fn encode(
        &self,
        raw: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>,
        layer: &Layer,
        output: &Buffer<u16>,
        status: &Buffer<u32>,
        g: Geometry,
        checked: bool,
    ) -> Result<()> {
        if layer.q.len != g.query() + GUARD
            || output.len != g.query() + GUARD
            || layer.kc.len != g.cache() / 2 + GUARD
            || layer.vc.len != g.cache() / 2 + GUARD
            || layer.ks.len != g.cache() / GROUP + GUARD
            || layer.vs.len != g.cache() / GROUP + GUARD
            || status.len != 8
        {
            return Err(Error::Options("component encoder extents differ"));
        }
        let encoder = raw
            .computeCommandEncoder()
            .ok_or(Error::Options("component encoder absent"))?;
        encoder.setComputePipelineState(if checked {
            &self.checked
        } else {
            &self.pipeline
        });
        let shape = g.shape();
        // SAFETY: private ABI slots0..5 are exact same-device live typed buffers,
        // slot6 copies eight u32 shape words; slot7 is aligned initialized atomic
        // status. D64/128 and queried width>=32 bound four feature slots. Each
        // layer uses a serial encoder overwriting the SAME tracked output owner
        // (WAW dependency). Owners and retained commands outlive completed wait;
        // mapped CPU access is excluded until that wait succeeds.
        unsafe {
            for (i, buffer) in [
                &layer.q.raw,
                &layer.kc.raw,
                &layer.vc.raw,
                &layer.ks.raw,
                &layer.vs.raw,
                &output.raw,
            ]
            .into_iter()
            .enumerate()
            {
                encoder.setBuffer_offset_atIndex(Some(buffer), 0, i);
            }
            encoder.setBytes_length_atIndex(
                NonNull::new(shape.as_ptr().cast_mut()).unwrap().cast(),
                std::mem::size_of_val(&shape),
                6,
            );
            encoder.setBuffer_offset_atIndex(Some(&status.raw), 0, 7);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: g.heads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.width,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        Ok(())
    }
}

// Independent FP64 softmax over exact stored/reconstructed inputs. No production
// attention, quantizer, or cache-read helper participates in this oracle.
fn oracle(
    g: Geometry,
    q: &[u16],
    key: impl Fn(usize) -> f64,
    value: impl Fn(usize) -> f64,
) -> Vec<f64> {
    let mut output = vec![0.0; g.query()];
    let mut scores = vec![0.0f64; DEPTH];
    for head in 0..g.heads {
        let kv = head / (g.heads / 2);
        for (p, score) in scores.iter_mut().enumerate() {
            let mut dot = 0.0;
            for f in 0..g.d {
                dot += f64::from(f16::from_bits(q[head * g.d + f]).to_f32())
                    * key((kv * DEPTH + p) * g.d + f);
            }
            *score = dot / (g.d as f64).sqrt();
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for weight in &mut scores {
            *weight = (*weight - maximum).exp();
        }
        let mass: f64 = scores.iter().sum();
        for f in 0..g.d {
            let mut numerator = 0.0;
            for (p, weight) in scores.iter().enumerate() {
                numerator += weight * value((kv * DEPTH + p) * g.d + f);
            }
            output[head * g.d + f] = numerator / mass;
        }
    }
    output
}

#[derive(Clone, Copy)]
enum Format {
    Fp16,
    I8,
    I8Checked,
}
impl Format {
    fn name(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::I8 => "i8",
            Self::I8Checked => "i8_checked",
        }
    }
}

fn expected(layer: &Layer, g: Geometry, format: Format) -> Vec<f64> {
    layer.q.with_slice(|q| match format {
        Format::Fp16 => layer.k.with_slice(|k| {
            layer.v.with_slice(|v| {
                oracle(
                    g,
                    q,
                    |i| f64::from(f16::from_bits(k[i]).to_f32()),
                    |i| f64::from(f16::from_bits(v[i]).to_f32()),
                )
            })
        }),
        Format::I8 | Format::I8Checked => layer.kc.with_slice(|k| {
            layer.vc.with_slice(|v| {
                layer.ks.with_slice(|ks| {
                    layer.vs.with_slice(|vs| {
                        oracle(
                            g,
                            q,
                            |i| {
                                let bytes = k[i / 2].to_le_bytes();
                                let code = i8::from_le_bytes([bytes[i % 2]]);
                                f64::from(f32::from(code) * f16::from_bits(ks[i / GROUP]).to_f32())
                            },
                            |i| {
                                let bytes = v[i / 2].to_le_bytes();
                                let code = i8::from_le_bytes([bytes[i % 2]]);
                                f64::from(f32::from(code) * f16::from_bits(vs[i / GROUP]).to_f32())
                            },
                        )
                    })
                })
            })
        }),
    })
}

fn command(
    gpu: &Gpu,
    candidate: &Candidate,
    layers: &[Layer],
    g: Geometry,
    format: Format,
    output: &mut Buffer<u16>,
    status: &mut Buffer<u32>,
) -> Result<Value> {
    autoreleasepool(|_| {
        output.with_mut(|words| words.fill(0x7bff));
        status.with_mut(|words| words.fill(0));
        let clock = Instant::now();
        let raw = gpu.command()?;
        let encoding = Instant::now();
        // Format selection is outside the layer loop. Every serial encoder
        // writes the identical tracked output allocation; status is not a fence.
        match format {
            Format::Fp16 => {
                for layer in layers {
                    gpu.encode(
                        &Command {
                            raw: &raw,
                            trace: None,
                        },
                        Kernel::AttentionDecode,
                        &[&layer.q.raw, &layer.k.raw, &layer.v.raw, &output.raw],
                        &g.shape(),
                        g.heads,
                        Some(status),
                    )?;
                }
            }
            Format::I8 | Format::I8Checked => {
                let checked = matches!(format, Format::I8Checked);
                for layer in layers {
                    candidate.encode(&raw, layer, output, status, g, checked)?;
                }
            }
        }
        let host_encode_seconds = encoding.elapsed().as_secs_f64();
        raw.commit();
        raw.waitUntilCompleted();
        let completed_host_seconds = clock.elapsed().as_secs_f64();
        if raw.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Completion);
        }
        let gpu_start = raw.GPUStartTime();
        let gpu_end = raw.GPUEndTime();
        if !gpu_start.is_finite() || !gpu_end.is_finite() || gpu_end <= gpu_start {
            return Err(Error::Options("component completed GPU timing unavailable"));
        }
        if !status.with_slice(|words| words.iter().all(|&word| word == 0)) {
            return Err(Error::Nonfinite);
        }
        if !output.with_slice(|words| words[g.query()..].iter().all(|&word| word == 0x7bff)) {
            return Err(Error::Options("component output guard changed"));
        }
        Ok(json!({"format":format.name(),"operator_count":layers.len(),
                  "completed_gpu_seconds":gpu_end-gpu_start,"completed_host_seconds":completed_host_seconds,
                  "host_encode_seconds":host_encode_seconds,"command_count":1,"status_zero":true,"output_guard_unchanged":true}))
    })
}

fn invalid_cases(
    gpu: &Gpu,
    candidate: &Candidate,
    layer: &mut Layer,
    g: Geometry,
    output: &mut Buffer<u16>,
    status: &mut Buffer<u32>,
) -> Result<usize> {
    let saved = [
        layer.q.with_slice(|b| b[0]),
        layer.kc.with_slice(|b| b[0]),
        layer.vc.with_slice(|b| b[0]),
        layer.ks.with_slice(|b| b[0]),
        layer.vs.with_slice(|b| b[0]),
    ];
    for case in 0..7 {
        match case {
            0 => layer.kc.with_mut(|b| b[0] = (b[0] & 0xff00) | 0x80),
            1 => layer.vc.with_mut(|b| b[0] = (b[0] & 0xff00) | 0x80),
            2 => layer.ks.with_mut(|b| b[0] = 0x8000),
            3 => layer.vs.with_mut(|b| b[0] = 0x7e00),
            4 => layer.ks.with_mut(|b| b[0] = 0x7c00),
            5 => {
                layer.ks.with_mut(|b| b[0] = 0);
                layer.kc.with_mut(|b| b[0] = (b[0] & 0xff00) | 1);
            }
            _ => layer.q.with_mut(|b| b[0] = 0x7e00),
        }
        let prepared = layer.hashes();
        let result = command(
            gpu,
            candidate,
            std::slice::from_ref(layer),
            g,
            Format::I8Checked,
            output,
            status,
        );
        let unchanged = layer.hashes() == prepared;
        // command has completed or was never submitted before any error return.
        // Restore only intentionally changed words; no full buffer copy exists.
        layer.q.with_mut(|b| b[0] = saved[0]);
        layer.kc.with_mut(|b| b[0] = saved[1]);
        layer.vc.with_mut(|b| b[0] = saved[2]);
        layer.ks.with_mut(|b| b[0] = saved[3]);
        layer.vs.with_mut(|b| b[0] = saved[4]);
        let expected_status = if case == 6 { 3 } else { 2 };
        if !unchanged
            || !matches!(result, Err(Error::Nonfinite))
            || !status.with_slice(|b| b[0] == expected_status && b[1..].iter().all(|&v| v == 0))
            || !output.with_slice(|b| b[g.query()..].iter().all(|&v| v == 0x7bff))
        {
            return Err(Error::Options(
                "component invalid input/status/guard witness failed",
            ));
        }
    }
    Ok(7)
}

fn correctness(
    gpu: &Gpu,
    candidate: &Candidate,
    layers: &mut [Layer],
    g: Geometry,
    output: &mut Buffer<u16>,
    status: &mut Buffer<u32>,
) -> Result<Value> {
    let valid_words = witness::words(gpu, candidate, status)?;
    let valid_attention = witness::attention(gpu, candidate, &mut layers[0], g, output, status)?;
    let rejected = invalid_cases(gpu, candidate, &mut layers[0], g, output, status)?;
    let mut maximum = 0.0f64;
    let mut compared = 0usize;
    let mut approximation_maximum = 0.0f64;
    let mut approximation_square = 0.0f64;
    for layer in layers {
        let reference = expected(layer, g, Format::Fp16);
        let quantized = expected(layer, g, Format::I8);
        for (&a, &b) in reference.iter().zip(&quantized) {
            approximation_maximum = approximation_maximum.max((a - b).abs());
            approximation_square += (a - b) * (a - b);
        }
        for (format, values) in [
            (Format::Fp16, &reference),
            (Format::I8, &quantized),
            (Format::I8Checked, &quantized),
        ] {
            command(
                gpu,
                candidate,
                std::slice::from_ref(layer),
                g,
                format,
                output,
                status,
            )?;
            output.with_slice(|words| {
                for (&word, &expected) in words.iter().zip(values) {
                    let actual = f64::from(f16::from_bits(word).to_f32());
                    let delta = (actual - expected).abs();
                    if !actual.is_finite()
                        || !expected.is_finite()
                        || delta > 1e-4 + 1e-3 * expected.abs()
                    {
                        return Err(Error::Options(
                            "component frozen FP64 arithmetic gate failed",
                        ));
                    }
                    maximum = maximum.max(delta);
                    compared += 1;
                }
                Ok(())
            })?;
        }
    }
    Ok(
        json!({"compared_outputs":compared,"expected_outputs":3*g.layers*g.query(),
        "every_layer_checked":true,"rejected_invalid_cases":rejected,"max_arithmetic_error":maximum,
        "exact_word_values_checked":valid_words,"valid_attention_outputs_checked":valid_attention,
        "quantization_attention_max_error":approximation_maximum,
        "quantization_attention_rms_error":(approximation_square/(g.layers*g.query()) as f64).sqrt(),
        "quantization_error_is_not_arithmetic_error":true}),
    )
}

fn io_error(error: impl std::fmt::Display) -> Error {
    Error::Native(error.to_string())
}

fn emit_and_wait(value: Value) -> Result<()> {
    let mut bytes = serde_json::to_vec(&value).map_err(io_error)?;
    bytes.push(b'\n');
    if bytes.len() > 65536 {
        return Err(Error::Options("component event exceeds bound"));
    }
    io::stdout().lock().write_all(&bytes).map_err(io_error)?;
    io::stdout().lock().flush().map_err(io_error)?;
    let mut ack = [0u8; 9];
    io::stdin().lock().read_exact(&mut ack).map_err(io_error)?;
    if &ack != b"continue\n" {
        return Err(Error::Options("component acknowledgement differs"));
    }
    Ok(())
}

/// Entry point exists only in this isolated experimental source snapshot.
pub fn run(model: &str, collection: &str) -> Result<()> {
    if !matches!(collection, "primary" | "confirmation") {
        return Err(Error::Options("component requires named collection"));
    }
    let g = Geometry::named(model)?;
    let admitted = g.admitted()?;
    if admitted > LIMIT {
        return Err(Error::Budget {
            required: admitted,
            limit: LIMIT,
        });
    }
    autoreleasepool(|_| {
        let gpu = Gpu::new()?;
        let candidate = Candidate::new(&gpu)?;
        let mut layers = Vec::with_capacity(g.layers);
        let mut quantization = Vec::with_capacity(g.layers);
        for i in 0..g.layers {
            let (layer, errors) = Layer::new(&gpu, g, i)?;
            layers.push(layer);
            quantization.push(errors);
        }
        let mut output = Buffer::<u16>::new(&gpu, g.query() + GUARD)?;
        let mut status = Buffer::<u32>::new(&gpu, 8)?;
        let hashes: Vec<_> = layers.iter().map(Layer::hashes).collect();
        emit_and_wait(json!({"event":"ready","format":"toklet-m6-kv-component-v1",
            "model_shape":model,"collection":collection,"layers":g.layers,"query_heads":g.heads,
            "kv_heads":2,"head_dim":g.d,"capacity":DEPTH,"execution_width":candidate.width,
            "device":gpu.device.name().to_string(),"debug_assertions":cfg!(debug_assertions),
            "admitted_bytes":admitted,"framework_allowance_bytes":FRAMEWORK,"driver_allowance_bytes":DRIVER,
            "scratch_allowance_bytes":SCRATCH,"oracle_score_bytes":DEPTH*8,"oracle_output_bytes":g.query()*16,
            "input_hashes":hashes,"quantization":quantization,"m6_complete":false}))?;
        let check = correctness(&gpu, &candidate, &mut layers, g, &mut output, &mut status)?;
        emit_and_wait(json!({"event":"correctness","result":check}))?;
        for pair in 0..13 {
            let order = if pair % 2 == 0 {
                [Format::Fp16, Format::I8]
            } else {
                [Format::I8, Format::Fp16]
            };
            for format in order {
                let sample = command(
                    &gpu,
                    &candidate,
                    &layers,
                    g,
                    format,
                    &mut output,
                    &mut status,
                )?;
                emit_and_wait(
                    json!({"event":"sample","pair":pair,"phase":if pair<3 {"warmup"}else{"sample"},"sample":sample}),
                )?;
            }
        }
        if layers.iter().map(Layer::hashes).collect::<Vec<_>>() != hashes {
            return Err(Error::Options(
                "component inputs/scales/codes/guards changed",
            ));
        }
        emit_and_wait(json!({"event":"done","samples":26,"inputs_unchanged":true,
            "every_layer_checked":true,"cache_write_cost_included":false,"m6_complete":false}))?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_has_explicit_signed_byte_order_and_nearest_even_ties() {
        let mut group = [0u16; 32];
        for (out, value) in group
            .iter_mut()
            .zip([-127.0, 127.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5])
        {
            *out = f16::from_f32(value).to_bits();
        }
        let packed = pack_group(&group).unwrap();
        assert_eq!(packed.scale, 0x3c00); // Exact1.0 Half scale.
        assert_eq!(&packed.words[..4], &[0x7f81, 0, 0xfe02, 0xfe02]);
        assert_eq!(packed.clamped, 0);
        assert_eq!(reconstructed(0x7f81, false, 0x3c00).unwrap(), -127.0);
        assert_eq!(reconstructed(0x7f81, true, 0x3c00).unwrap(), 127.0);
    }

    #[test]
    fn zero_subnormal_scale_fallback_and_clamp_are_distinct() {
        let zeros = pack_group(&[0; 32]).unwrap();
        assert_eq!(zeros.scale, 0);
        assert_eq!(zeros.words, [0; 16]);
        assert_eq!(zeros.max_error, 0.0);
        let mut tiny = [0; 32];
        tiny[0] = 1;
        tiny[1] = 0x8001;
        let packed = pack_group(&tiny).unwrap();
        assert_eq!(packed.scale, 1);
        assert_eq!(packed.words[0], 0xff01);
        assert_eq!(packed.max_error, 0.0);
        let mut clipped = [0; 32];
        clipped[0] = 128; //128 minimum Half subnormals.
        let packed = pack_group(&clipped).unwrap();
        assert_eq!(packed.scale, 1);
        assert_eq!(packed.words[0], 127);
        assert_eq!(packed.clamped, 1);
        assert_eq!(packed.max_error, 2f64.powi(-24));
    }

    #[test]
    fn format_rejects_partial_nonfinite_reserved_and_noncanonical_groups() {
        assert!(pack_group(&[0; 31]).is_err());
        for value in [0x7e00, 0x7c00, 0xfc00] {
            let mut group = [0; 32];
            group[0] = value;
            assert!(pack_group(&group).is_err());
        }
        for (word, scale) in [
            (0x0080, 0x3c00),
            (1, 0),
            (0, 0x8000),
            (0, 0x7c00),
            (0, 0x7e00),
        ] {
            assert!(reconstructed(word, false, scale).is_err());
        }
        // Full-range finite Half values may reconstruct above65504 in FP32;
        // no hidden Half cast or promise of finite Half attention output.
        let mut extreme = [0; 32];
        extreme[0] = 0x7bff;
        let packed = pack_group(&extreme).unwrap();
        let reconstructed = reconstructed(packed.words[0], false, packed.scale).unwrap();
        assert!(reconstructed.is_finite());
        assert_eq!(packed.max_error, (f64::from(reconstructed) - 65504.0).abs());
    }

    #[test]
    fn admitted_shapes_are_bounded_and_layer_inputs_are_distinct() {
        assert!(Geometry::named("other").is_err());
        for name in ["small", "larger"] {
            let g = Geometry::named(name).unwrap();
            assert!(g.admitted().unwrap() < LIMIT);
            assert!(g.d.is_multiple_of(GROUP));
            let signatures: std::collections::BTreeSet<_> = (0..g.layers)
                .map(|layer| {
                    (0..64)
                        .map(|i| generated(i, layer as u64 * 3 + 2))
                        .collect::<Vec<_>>()
                })
                .collect();
            assert_eq!(signatures.len(), g.layers);
        }
    }
}
