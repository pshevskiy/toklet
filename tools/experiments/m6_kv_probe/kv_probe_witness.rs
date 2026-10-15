//! Untimed format witnesses. Exact FP32 bits make tiny scales observable without
//! changing the attention arithmetic tolerance or introducing a Half cast.
use super::*;

pub(super) fn words(gpu: &Gpu, candidate: &Candidate, status: &mut Buffer<u32>) -> Result<usize> {
    const N: usize = 128;
    let mut codes = Buffer::<u16>::new(gpu, N / 2 + GUARD)?;
    let mut scales = Buffer::<u16>::new(gpu, N / GROUP + GUARD)?;
    let mut output = Buffer::<u32>::new(gpu, N + GUARD)?;
    codes.with_mut(|b| {
        b.fill(0x8080);
        b[..16].fill(0); // Canonical zero-scale group.
        for (i, word) in b[16..N / 2].iter_mut().enumerate() {
            // Codes[0,1,-1,2,-2,127,-127,0], independently hand-packed.
            *word = [0x0100, 0x02ff, 0x7ffe, 0x0081][i % 4];
        }
    });
    scales.with_mut(|b| {
        b.fill(0x7e00);
        b[..4].copy_from_slice(&[0, 1, 0x3c00, 0x7bff]);
    });
    let golden = [
        [0u32; 8],
        [
            0, 0x33800000, 0xb3800000, 0x34000000, 0xb4000000, 0x36fe0000, 0xb6fe0000, 0,
        ],
        [0.0f32, 1.0, -1.0, 2.0, -2.0, 127.0, -127.0, 0.0].map(f32::to_bits),
        [
            0.0f32, 65504.0, -65504.0, 131008.0, -131008.0, 8319008.0, -8319008.0, 0.0,
        ]
        .map(f32::to_bits),
    ];
    let hashes = [digest(&codes), digest(&scales)];
    for pipeline in [&candidate.words, &candidate.checked_words] {
        output.with_mut(|b| b.fill(0xdeadbeef));
        status.with_mut(|b| b.fill(0));
        let raw = gpu.command()?;
        let encoder = raw
            .computeCommandEncoder()
            .ok_or(Error::Options("word witness encoder absent"))?;
        encoder.setComputePipelineState(pipeline);
        let count = [N as u32];
        // SAFETY: fixed128-value payload,64u16 code words and4u16 scale words
        // plus seven guards; output is128u32 plus seven guards. Uniform slot3
        // copies one u32, status slot4 is eight initialized aligned u32 words.
        // Width is queried/checked. All resources are same-device, owned and
        // retained through completed wait; no mapped CPU access is in flight.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&codes.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&scales.raw), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&output.raw), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new(count.as_ptr().cast_mut()).unwrap().cast(),
                4,
                3,
            );
            encoder.setBuffer_offset_atIndex(Some(&status.raw), 0, 4);
        }
        let width = pipeline.threadExecutionWidth();
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: N.div_ceil(width),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        raw.commit();
        raw.waitUntilCompleted();
        if raw.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Completion);
        }
        if !status.with_slice(|b| b.iter().all(|&v| v == 0)) {
            return Err(Error::Nonfinite);
        }
        output.with_slice(|b| {
            for (i, &bits) in b[..N].iter().enumerate() {
                if bits != golden[i / GROUP][i % 8] {
                    return Err(Error::Options("exact GPU I8 reconstruction bits differ"));
                }
            }
            if !b[N..].iter().all(|&v| v == 0xdeadbeef) {
                return Err(Error::Options("word witness output guard differs"));
            }
            Ok(())
        })?;
    }
    if [digest(&codes), digest(&scales)] != hashes {
        return Err(Error::Options("word witness input/scale/guards changed"));
    }
    Ok(2 * N)
}

pub(super) fn attention(
    gpu: &Gpu,
    candidate: &Candidate,
    layer: &mut Layer,
    g: Geometry,
    output: &mut Buffer<u16>,
    status: &mut Buffer<u32>,
) -> Result<usize> {
    let hashes = layer.hashes();
    let mut compared = 0usize;
    for pattern in 0..2 {
        layer.q.with_mut(|b| b[..g.query()].fill(0));
        layer.k.with_mut(|b| b[..g.cache()].fill(0));
        layer.v.with_mut(|b| {
            for (i, word) in b[..g.cache()].iter_mut().enumerate() {
                *word = f16::from_f32(if pattern == 0 {
                    0.0
                } else if i % 2 == 0 {
                    127.0
                } else {
                    -127.0
                })
                .to_bits();
            }
        });
        quantize(&layer.k, &mut layer.kc, &mut layer.ks, g.cache())?;
        quantize(&layer.v, &mut layer.vc, &mut layer.vs, g.cache())?;
        let prepared = layer.hashes();
        for format in [Format::Fp16, Format::I8, Format::I8Checked] {
            command(
                gpu,
                candidate,
                std::slice::from_ref(layer),
                g,
                format,
                output,
                status,
            )?;
            if layer.hashes() != prepared {
                return Err(Error::Options(
                    "prepared attention witness inputs/guards changed",
                ));
            }
            output.with_slice(|b| {
                for (i, &bits) in b[..g.query()].iter().enumerate() {
                    // Q/K zero give uniform softmax; each V feature is constant
                    // over positions. Its exact mean is0 or signed127. Both
                    // integer numerator and mass fit exactly in FP32 at32K.
                    let expected = if pattern == 0 {
                        0.0
                    } else if i % 2 == 0 {
                        127.0
                    } else {
                        -127.0
                    };
                    if bits != f16::from_f32(expected).to_bits() {
                        return Err(Error::Options(
                            "uniform zero/extreme attention witness differs",
                        ));
                    }
                    compared += 1;
                }
                Ok(())
            })?;
        }
    }
    // Restore layer0's exact deterministic original values before any timing.
    // Refill in place; no original/reconstructed full CPU cache is retained.
    fill(&mut layer.q, g.query(), 1);
    fill(&mut layer.k, g.cache(), 2);
    fill(&mut layer.v, g.cache(), 3);
    quantize(&layer.k, &mut layer.kc, &mut layer.ks, g.cache())?;
    quantize(&layer.v, &mut layer.vc, &mut layer.vs, g.cache())?;
    if layer.hashes() != hashes {
        return Err(Error::Options("attention witness restore differs"));
    }
    Ok(compared)
}
