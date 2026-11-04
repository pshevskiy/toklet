//! Exact authored witnesses distinguish Half coefficient and product boundaries.
use super::*;
use half::f16;

#[test]
#[ignore = "Explicit Apple GPU scalar boundaries and padded strides"]
fn dense_scalars_preserve_half_boundaries_and_padding() {
    objc2::rc::autoreleasepool(|_| {
        let g = Gpu::new().unwrap();
        let sentinel = 0x3555;
        let mut x = Buffer::<u16>::new(&g, 23).unwrap();
        let mut delta = Buffer::<u16>::new(&g, 23).unwrap();
        x.with_mut(|b| {
            b.fill(sentinel);
            b[0] = f16::from_f64(0.125).to_bits();
            b[1] = 0;
            b[2] = f16::from_f64(-0.5).to_bits();
            b[8..11].fill(0);
        });
        delta.with_mut(|b| {
            b.fill(sentinel);
            b[0] = f16::from_f64(0.0013875961303710938).to_bits();
            b[1] = f16::from_f64(0.0009794235229492188).to_bits();
            b[2] = f16::from_f64(1.).to_bits();
            b[8..11].fill(f16::from_f64(1.).to_bits());
        });
        let before_delta = delta.with_slice(|b| b.to_vec());
        let status = Buffer::<u32>::new(&g, 8).unwrap();
        let raw = g.command().unwrap();
        g.encode(
            &Command {
                raw: &raw,
                trace: None,
            },
            Kernel::ResidualScaled,
            &[&x.raw, &delta.raw],
            &[2, 3, 8, u32::from(f16::from_f64(0.22).to_bits())],
            6,
            Some(&status),
        )
        .unwrap();
        raw.commit();
        raw.waitUntilCompleted();
        x.with_slice(|b| {
            assert_eq!(b[0], f16::from_f64(0.125244140625).to_bits());
            assert_eq!(b[1], f16::from_f64(0.00021541118621826172).to_bits());
            for (i, &word) in b.iter().enumerate() {
                if !(0..3).contains(&i) && !(8..11).contains(&i) {
                    assert_eq!(word, sentinel);
                }
            }
        });
        delta.with_slice(|b| assert_eq!(b, before_delta));
        let raw = g.command().unwrap();
        g.encode(
            &Command {
                raw: &raw,
                trace: None,
            },
            Kernel::ScaleHalf,
            &[&x.raw],
            &[2, 3, 8, u32::from(f16::from_f64(12.).to_bits())],
            6,
            Some(&status),
        )
        .unwrap();
        let mut logits = Buffer::<u32>::new(&g, 23).unwrap();
        logits.with_mut(|b| {
            b.fill(0x7fa12345);
            for i in [0, 1, 2, 8, 9, 10] {
                b[i] = 2_f32.to_bits();
            }
        });
        g.encode(
            &Command {
                raw: &raw,
                trace: None,
            },
            Kernel::DivideFloat,
            &[&logits.raw],
            &[2, 3, 8, 8_f32.to_bits()],
            6,
            Some(&status),
        )
        .unwrap();
        raw.commit();
        raw.waitUntilCompleted();
        x.with_slice(|b| {
            assert_eq!(b[0], f16::from_f64(1.5029296875).to_bits());
            for (i, &word) in b.iter().enumerate() {
                if !(0..3).contains(&i) && !(8..11).contains(&i) {
                    assert_eq!(word, sentinel);
                }
            }
        });
        logits.with_slice(|b| {
            for (i, &word) in b.iter().enumerate() {
                assert_eq!(
                    word,
                    if (0..3).contains(&i) || (8..11).contains(&i) {
                        0.25_f32.to_bits()
                    } else {
                        0x7fa12345
                    }
                );
            }
        });
        status.with_slice(|b| assert!(b.iter().all(|&x| x == 0)));
    });
}
