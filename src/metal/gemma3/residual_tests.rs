//! Explicit clipped-Half primitive checks; no model inference or speed claim.
use super::*;
use crate::model::gemma_reference::tests::Measurement;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
};
use serde_json::json;

const H: u16 = 0x7e55;
const S: u32 = 0x5a170031;
// Independent exact Half-word expectations: saturation, ties-to-even, signed
// zero, minimum subnormals and ordinary finite sums. Entry1 is the saved real
// BOS witness 64864 +14520; no model/golden output is used to compute these.
const CASES: [(u16, u16, u16); 16] = [
    (0x7bff, 0x7bff, 0x7bff),
    (0x7beb, 0x7317, 0x7bff),
    (0xfbeb, 0xf317, 0xfbff),
    (0x7bff, 0xfbff, 0),
    (0, 0x8000, 0),
    (0x8000, 0x8000, 0x8000),
    (1, 1, 2),
    (1, 0x8001, 0),
    (0x7bfe, 0x4c00, 0x7bfe),
    (0x7bff, 0x4c00, 0x7bff),
    (0x7bff, 0xcc00, 0x7bfe),
    (0xfbff, 0x4c00, 0xfbfe),
    (0x3c00, 0x1000, 0x3c00),
    (0x3c01, 0x1000, 0x3c02),
    (0x3c00, 0xbc00, 0),
    (0x8001, 0x8001, 0x8002),
];
fn data(gpu: &Gpu, active: &[u16]) -> Buffer<u16> {
    let mut b = Buffer::new(gpu, active.len() + 7).unwrap();
    b.with_mut(|out| {
        out.fill(H);
        out[..active.len()].copy_from_slice(active);
    });
    b
}
fn status(gpu: &Gpu, value: u32) -> Buffer<u32> {
    let mut b = Buffer::new(gpu, 8).unwrap();
    b.with_mut(|out| {
        out.fill(S);
        out[0] = value;
    });
    b
}
fn encode(
    gpu: &Gpu,
    command: &Command<'_>,
    kernel: Kernel,
    x: &Buffer<u16>,
    delta: &Buffer<u16>,
    status: &Buffer<u32>,
    n: usize,
) -> Result<()> {
    gpu.encode(
        command,
        kernel,
        &[&x.raw, &delta.raw],
        &[n as u32],
        n,
        Some(status),
    )
}
fn observe(
    command: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>,
    x: &Buffer<u16>,
    delta: &Buffer<u16>,
    status: &Buffer<u32>,
    label: &str,
    result: &Result<()>,
) -> (Vec<u16>, Vec<u16>, Vec<u32>) {
    println!(
        "HALF_RESIDUAL_RAW {}",
        json!({"kind":"completion","label":label,"encode":format!("{result:?}"),"status":format!("{:?}",command.status()),"error":command.error().map(|e|e.to_string())})
    );
    // Never map operands while native completion is uncertain.
    assert!(matches!(
        command.status(),
        MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
    ));
    let a = x.with_slice(|v| v.to_vec());
    let b = delta.with_slice(|v| v.to_vec());
    let state = status.with_slice(|v| v.to_vec());
    println!(
        "HALF_RESIDUAL_RAW {}",
        json!({"kind":"operands","label":label,"x":a,"delta":b,"status":state})
    );
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert!(command.error().is_none());
    (a, b, state)
}
fn execute(
    gpu: &Gpu,
    kernel: Kernel,
    x: &Buffer<u16>,
    delta: &Buffer<u16>,
    status: &Buffer<u32>,
    n: usize,
    label: &str,
) -> (Vec<u16>, Vec<u16>, Vec<u32>) {
    let raw = gpu.command().unwrap();
    let c = Command {
        raw: &raw,
        trace: None,
    };
    let result = encode(gpu, &c, kernel, x, delta, status, n);
    raw.commit();
    raw.waitUntilCompleted();
    let captured = observe(&raw, x, delta, status, label, &result);
    result.unwrap();
    captured
}
fn guards(a: &[u16], b: &[u16], state: &[u32], n: usize) {
    assert_eq!(&a[n..], &[H; 7]);
    assert_eq!(&b[n..], &[H; 7]);
    assert_eq!(&state[1..], &[S; 7]);
}

#[test]
#[ignore = "Explicit Apple GPU finite clamped-Half residual primitive and legacy rejection"]
fn clipped_half_add_preserves_stores_inputs_status_and_allocations() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        for n in [1, 33, 7 * 64, 32 * 640] {
            let left: Vec<_> = (0..n).map(|i| CASES[i % CASES.len()].0).collect();
            let right: Vec<_> = (0..n).map(|i| CASES[i % CASES.len()].1).collect();
            let x = data(&gpu, &left);
            let delta = data(&gpu, &right);
            let state = status(&gpu, 0);
            println!(
                "HALF_RESIDUAL_RAW {}",
                json!({"kind":"input","label":format!("bulk{n}"),"x":left,"delta":right})
            );
            let (a, b, flags) = execute(
                &gpu,
                Kernel::ResidualClamp,
                &x,
                &delta,
                &state,
                n,
                &format!("bulk{n}"),
            );
            guards(&a, &b, &flags, n);
            assert_eq!(flags[0], 0);
            assert_eq!(&b[..n], right);
            for (i, &actual) in a[..n].iter().enumerate() {
                assert_eq!(actual, CASES[i % CASES.len()].2, "scalar{i}/n{n}");
            }
        }
        // A true same-command store boundary: saturated max +(-max) becomes0.
        // Delaying saturation until after both additions would produce max.
        let n = 64;
        let x = data(&gpu, &[0x7bff; 64]);
        let delta = data(&gpu, &[0x7bff; 64]);
        let negative = data(&gpu, &[0xfbff; 64]);
        let saved = data(&gpu, &[0; 64]);
        let state = status(&gpu, 0);
        let raw = gpu.command().unwrap();
        let c = Command {
            raw: &raw,
            trace: None,
        };
        let first = encode(&gpu, &c, Kernel::ResidualClamp, &x, &delta, &state, n);
        let blit = raw.blitCommandEncoder().unwrap();
        // SAFETY: disjoint retained same-device initialized buffers, aligned
        // checked64-Half range, tracked dependencies, no host access in flight.
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &x.raw,
                0,
                &saved.raw,
                0,
                n * 2,
            );
        }
        blit.endEncoding();
        let second = encode(&gpu, &c, Kernel::ResidualClamp, &x, &negative, &state, n);
        raw.commit();
        raw.waitUntilCompleted();
        let (a, b, flags) = observe(&raw, &x, &delta, &state, "chain", &second);
        let midpoint = saved.with_slice(|v| v.to_vec());
        let neg = negative.with_slice(|v| v.to_vec());
        println!(
            "HALF_RESIDUAL_RAW {}",
            json!({"kind":"chain","first_encode":format!("{first:?}"),"midpoint":midpoint,"negative":neg})
        );
        first.unwrap();
        second.unwrap();
        guards(&a, &b, &flags, n);
        assert_eq!(&a[..n], &[0; 64]);
        assert_eq!(&b[..n], &[0x7bff; 64]);
        assert_eq!(&midpoint[..n], &[0x7bff; 64]);
        assert_eq!(&midpoint[n..], &[H; 7]);
        assert_eq!(&neg[..n], &[0xfbff; 64]);
        assert_eq!(&neg[n..], &[H; 7]);
        assert_eq!(flags[0], 0);

        let n = 33;
        let mut left = vec![0; n];
        let mut right = vec![0; n];
        for (i, word) in [(0, 0x7e11), (7, 0xfe22), (11, 0x7d01)] {
            left[i] = word;
        }
        right[3] = 0x7c00;
        right[9] = 0xfc00;
        let mut x = data(&gpu, &left);
        let mut delta = data(&gpu, &right);
        let mut state = status(&gpu, 4);
        println!(
            "HALF_RESIDUAL_RAW {}",
            json!({"kind":"input","label":"invalid","x":left,"delta":right})
        );
        let (a, b, flags) = execute(
            &gpu,
            Kernel::ResidualClamp,
            &x,
            &delta,
            &state,
            n,
            "invalid",
        );
        guards(&a, &b, &flags, n);
        assert_eq!(&a[..n], left);
        assert_eq!(&b[..n], right);
        assert_eq!(flags[0], 5);
        x.with_mut(|v| v[..n].fill(0));
        delta.with_mut(|v| v[..n].fill(0));
        state.with_mut(|v| v[0] = 0);
        let (a, b, flags) = execute(
            &gpu,
            Kernel::ResidualClamp,
            &x,
            &delta,
            &state,
            n,
            "recovery",
        );
        guards(&a, &b, &flags, n);
        assert!(a[..n].iter().all(|&v| v == 0));
        assert_eq!(flags[0], 0);

        let legacy = data(&gpu, &[0x7bff]);
        let other = data(&gpu, &[0x7bff]);
        let old_status = status(&gpu, 0);
        let (a, b, flags) = execute(
            &gpu,
            Kernel::Residual,
            &legacy,
            &other,
            &old_status,
            1,
            "legacy",
        );
        guards(&a, &b, &flags, 1);
        assert_eq!(a[0], 0x7c00);
        assert_ne!(flags[0] & 1, 0);

        let raw = gpu.command().unwrap();
        let c = Command {
            raw: &raw,
            trace: None,
        };
        let mut counts = Vec::with_capacity(8);
        for i in 0..8 {
            let measurement = Measurement::start();
            let result = encode(&gpu, &c, Kernel::ResidualClamp, &x, &delta, &state, n);
            let count = measurement.finish();
            println!(
                "HALF_RESIDUAL_RAW {}",
                json!({"kind":"allocation_call","index":i,"count":count,"encode":format!("{result:?}")})
            );
            result.unwrap();
            counts.push(count);
        }
        raw.commit();
        raw.waitUntilCompleted();
        let (a, b, flags) = observe(&raw, &x, &delta, &state, "measured", &Ok(()));
        guards(&a, &b, &flags, n);
        assert!(a[..n].iter().all(|&v| v == 0));
        assert_eq!(flags[0], 0);
        println!(
            "HALF_RESIDUAL_RAW {}",
            json!({"kind":"summary","active_bulk_values":20962,"invalid_sites":5,"same_command_boundary_values":64,"allocation_counts":counts,"release_zero_required":!cfg!(debug_assertions)})
        );
        if !cfg!(debug_assertions) {
            assert!(counts.iter().all(|&v| v == 0));
        }
    });
}
