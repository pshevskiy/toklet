//! Bounded startup validation of the actual dense preparation and consume PSOs.
use super::{
    Buffer, Context, Error, Result,
    kernel::{Op, encode_pipeline, k64_input_aligned},
};
use half::bf16;
use objc2::sel;
use objc2::{
    rc::Retained,
    runtime::{NSObjectProtocol, ProtocolObject},
};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLComputePipelineState,
};

const M: usize = 32;
const K: usize = 128;
const N: usize = 64;
const PAD: usize = 16;
const SHAPE: [u32; 4] = [M as u32, K as u32, N as u32, 0];
const GROUPS: [usize; 2] = [N / 32, 1];
const WEIGHTS: usize = N * K;

struct SignedBank {
    words: Buffer<u32>,
    scales: Buffer<u16>,
    biases: Buffer<u16>,
}

fn signed(role: usize, row: usize, col: usize) -> (u32, u16, u16) {
    let seed = role + 3 * row + 7 * (col / 64);
    let sign = if seed.is_multiple_of(2) { 0 } else { 0x8000 };
    let offset = col % 64;
    let q = ((seed + offset + offset / 16) % 16) as u32;
    (
        q,
        sign | (0x3b00 + (seed % 32) as u16),
        (sign ^ 0x8000) | (0x3a80 + (seed % 64) as u16),
    )
}

fn bank(context: &Context, role: usize) -> Result<SignedBank> {
    let device = &context.device;
    let mut words = Buffer::<u32>::on_device(device, WEIGHTS / 8)?;
    let mut scales = Buffer::<u16>::on_device(device, WEIGHTS / 64)?;
    let mut biases = Buffer::<u16>::on_device(device, WEIGHTS / 64)?;
    words.with_mut(|dst| {
        for row in 0..N {
            for col in 0..K {
                let (nibble, _, _) = signed(role, row, col);
                dst[(row * K + col) / 8] |= nibble << (4 * (col % 8));
            }
        }
    });
    scales.with_mut(|dst| {
        for (idx, scale) in dst.iter_mut().enumerate() {
            *scale = signed(role, idx / 2, (idx % 2) * 64).1;
        }
    });
    biases.with_mut(|dst| {
        for (idx, bias) in dst.iter_mut().enumerate() {
            *bias = signed(role, idx / 2, (idx % 2) * 64).2;
        }
    });
    Ok(SignedBank {
        words,
        scales,
        biases,
    })
}

fn prepared_exact(pairs: &[u32]) -> bool {
    let mut words = pairs[..WEIGHTS]
        .iter()
        .flat_map(|&x| [x as u16, (x >> 16) as u16]);
    for tile in 0..N / 32 {
        for k8 in 0..K / 8 {
            for half in 0..2 {
                for lane in 0..32 {
                    let quad = lane / 4;
                    let m = (quad & 4) + lane / 2 % 4;
                    let n = (quad & 2) * 2 + lane % 2 * 2;
                    for role in [11, 42] {
                        for delta in [0, 1, 8, 9] {
                            let (q, scale, bias) =
                                signed(role, tile * 32 + half * 16 + n + delta, k8 * 8 + m);
                            let value = bf16::from_bits(scale).to_f32() * q as f32
                                + bf16::from_bits(bias).to_f32();
                            if words.next() != Some(bf16::from_f32(value).to_bits()) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
    }
    words.next().is_none() && pairs[WEIGHTS..].iter().all(|&x| x == u32::MAX)
}

pub(super) fn check(
    context: &Context,
    prepare: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    consume: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
) -> Result<()> {
    if !context.has_gate_up_compact() {
        return Err(Error::Unsupported("Nanbeige packed startup comparator"));
    }
    let device = &context.device;
    let gate = bank(context, 11)?;
    let up = bank(context, 42)?;
    let mut input = Buffer::<u16>::on_device(device, M * K)?;
    input.with_mut(|dst| {
        for (index, value) in dst.iter_mut().enumerate() {
            let sign = if (index / 3).is_multiple_of(2) {
                0
            } else {
                0x8000
            };
            *value = sign | (0x3c00 + (index % 512) as u16);
        }
    });
    let mut pair = Buffer::<u32>::on_device(device, WEIGHTS + PAD)?;
    pair.with_mut(|x| x.fill(u32::MAX));
    let [mut packed_gate, mut packed_up, mut dense] = [
        Buffer::<u16>::on_device(device, M * N + PAD)?,
        Buffer::<u16>::on_device(device, M * N + PAD)?,
        Buffer::<u16>::on_device(device, M * N + PAD)?,
    ];
    for out in [&mut packed_gate, &mut packed_up, &mut dense] {
        out.with_mut(|x| x.fill(u16::MAX));
    }
    let mut status = Buffer::<u32>::on_device(device, 2)?;
    status.with_mut(|x| x[1] = u32::MAX);
    if !pair.raw.respondsToSelector(sel!(gpuAddress))
        || pair.raw.gpuAddress() == 0
        || !pair.raw.gpuAddress().is_multiple_of(16)
        || !k64_input_aligned(&input.raw)
    {
        return Err(Error::Unsupported(
            "Nanbeige dense signed startup alignment",
        ));
    }
    let command = context.command()?;
    let encoder = command
        .computeCommandEncoder()
        .ok_or_else(|| Error::Native("Nanbeige dense signed startup encoder".into()))?;
    encode_pipeline(
        &encoder,
        prepare,
        128,
        &[
            &gate.words.raw,
            &gate.scales.raw,
            &gate.biases.raw,
            &up.words.raw,
            &up.scales.raw,
            &up.biases.raw,
            &pair.raw,
        ],
        &SHAPE,
        [(WEIGHTS / 4).div_ceil(128), 1],
    );
    encode_pipeline(
        &encoder,
        consume,
        128,
        &[&pair.raw, &input.raw, &dense.raw, &status.raw],
        &SHAPE,
        GROUPS,
    );
    for (bank, output) in [(&gate, &packed_gate), (&up, &packed_up)] {
        context.encode(
            &encoder,
            Op::GateUpCompact,
            &[
                &bank.words.raw,
                &bank.scales.raw,
                &bank.biases.raw,
                &input.raw,
                &output.raw,
                &status.raw,
            ],
            &SHAPE,
            GROUPS,
        );
    }
    context.encode(
        &encoder,
        Op::Swiglu,
        &[&packed_gate.raw, &packed_up.raw, &status.raw],
        &[M as u32, N as u32],
        [(M * N).div_ceil(32), 1],
    );
    encoder.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    #[cfg(test)]
    if command.status() == MTLCommandBufferStatus::Completed && STARTUP_POISON.replace(false) {
        pair.with_mut(|words| words[0] ^= 1);
    }
    if command.status() != MTLCommandBufferStatus::Completed
        || !status.with_slice(|x| x == [0, u32::MAX])
        || !pair.with_slice(prepared_exact)
        || !packed_gate.with_slice(|a| dense.with_slice(|b| a == b))
        || ![&packed_gate, &packed_up, &dense].iter().all(|out| {
            out.with_slice(|words| {
                words[..M * N].iter().all(|&x| x & 0x7f80 != 0x7f80)
                    && words[M * N..].iter().all(|&x| x == u16::MAX)
            })
        })
    {
        return Err(Error::Unsupported("Nanbeige dense signed startup parity"));
    }
    Ok(())
}

#[cfg(test)]
std::thread_local! {
    static STARTUP_POISON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(super) fn poison_next_startup() {
    assert!(!STARTUP_POISON.replace(true));
}

#[cfg(test)]
pub(super) fn startup_poison_pending() -> bool {
    STARTUP_POISON.get()
}
