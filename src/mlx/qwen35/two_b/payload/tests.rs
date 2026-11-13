//! Tiny literal byte stream exercises the common cold mechanism, including
//! original FP32 values that BF16 cannot preserve. No real model is allocated.
use super::*;
use crate::mlx::{
    affine::{Layout, MetadataDtype},
    qwen35::{
        ConvolutionPlan, Dtype, PhysicalTensor, Shape, Storage, TensorPlan,
        two_b::{PayloadBytes, recipe},
    },
};
use std::{collections::BTreeMap, io::Cursor};
const MATRIX: &str = "language_model.model.embed_tokens.weight";
const NORM: &str = "language_model.model.norm.weight";
const ALOG: &str = "language_model.model.layers.0.linear_attn.A_log";
const CONV: &str = "language_model.model.layers.0.linear_attn.conv1d.weight";
const PREFIX: &[u8] = b"\x01\0\0\0\0\0\0\0x";
const F32_BITS: [u32; 3] = [0x8000_0000, 0x3f80_0055, 0xbf12_3456];
fn stream() -> Vec<u8> {
    let mut bytes = PREFIX.to_vec();
    bytes.extend([0x77; 32]);
    bytes.extend(0x3f80u16.to_le_bytes()); // scale1
    bytes.extend(0xc000u16.to_le_bytes()); // bias-2
    for n in [0x3f80u16, 0x8000] {
        bytes.extend(n.to_le_bytes());
    }
    for n in F32_BITS {
        bytes.extend(n.to_le_bytes());
    }
    for n in [0x3f00u16, 0xbf80, 0x4000, 0x0000] {
        bytes.extend(n.to_le_bytes());
    }
    assert_eq!(bytes.len(), 69);
    bytes
}
fn fixture() -> Plan {
    let bytes = stream();
    let mut physical = BTreeMap::new();
    for (name, dtype, shape, start, end) in [
        (MATRIX, Dtype::U32, vec![1, 8], 0, 32),
        (
            "language_model.model.embed_tokens.scales",
            Dtype::Bf16,
            vec![1, 1],
            32,
            34,
        ),
        (
            "language_model.model.embed_tokens.biases",
            Dtype::Bf16,
            vec![1, 1],
            34,
            36,
        ),
        (NORM, Dtype::Bf16, vec![2], 36, 40),
        (ALOG, Dtype::F32, vec![3], 40, 52),
        (CONV, Dtype::Bf16, vec![1, 4, 1], 52, 60),
    ] {
        physical.insert(
            name.into(),
            PhysicalTensor {
                dtype,
                shape,
                range: start..end,
            },
        );
    }
    Plan {
        layers: recipe::build().unwrap().0,
        common: vec![
            TensorPlan {
                name: MATRIX.into(),
                shape: Shape::Matrix {
                    rows: 1,
                    columns: 64,
                },
                storage: Storage::Affine(Layout::new(1, 64, 4, 64, MetadataDtype::BF16).unwrap()),
            },
            TensorPlan {
                name: NORM.into(),
                shape: Shape::Vector(2),
                storage: Storage::Bf16Vector,
            },
            TensorPlan {
                name: ALOG.into(),
                shape: Shape::Vector(3),
                storage: Storage::F32Vector,
            },
        ],
        convolutions: vec![ConvolutionPlan {
            layer: 0,
            history_owner: 0,
            name: CONV.into(),
            channels: 1,
            taps: 4,
        }],
        physical,
        bytes: PayloadBytes {
            affine_codes: 32,
            affine_metadata: 4,
            bf16_vectors: 4,
            f32_vectors: 12,
            bf16_convolutions: 8,
            total: 60,
        },
        pins: ArtifactPins {
            config_bytes: 11,
            config_sha256: Sha256::digest(b"tiny-config").into(),
            header_bytes: 1,
            header_sha256: Sha256::digest(b"x").into(),
            container_bytes: 69,
            container_sha256: Sha256::digest(&bytes).into(),
        },
    }
}
fn private_load<R: Read>(
    plan: Plan,
    reader: &mut R,
    expected: &[u8; 32],
    limit: usize,
) -> Result<PayloadStore, Error> {
    let mut sha = Sha256::new();
    sha.update(PREFIX);
    let identity = ContentIdentity::pending(b"tiny-config", b"x", 69);
    let components = cold::read_components(
        reader,
        source_recipe(&plan),
        sha,
        identity,
        9,
        &digest_text(expected),
        limit,
    )?;
    Ok(PayloadStore { plan, components })
}
fn tiny() -> PayloadStore {
    let bytes = stream();
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    private_load(
        fixture(),
        &mut Cursor::new(&bytes[9..]),
        &digest,
        usize::MAX,
    )
    .unwrap()
}

#[test]
fn original_fp32_bits_bf16_values_and_tied_matrix_are_preserved() {
    let s = tiny();
    assert_eq!(
        s.vector(ALOG)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        F32_BITS
    );
    assert_ne!(
        half::bf16::from_f32(s.vector(ALOG).unwrap()[1])
            .to_f32()
            .to_bits(),
        F32_BITS[1]
    );
    assert_eq!(
        s.vector(NORM)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        [1.0f32.to_bits(), 0x8000_0000]
    );
    assert_eq!(s.convolution(CONV).unwrap(), [0.5, -1.0, 2.0, 0.0]);
    assert_eq!(s.matrix(MATRIX).unwrap().value(0, 0).unwrap(), 5.0);
    assert_eq!(
        (
            s.report().packed_matrices,
            s.report().fp32_vectors,
            s.report().fp32_convolutions
        ),
        (1, 2, 1)
    );
    assert_eq!(s.report().source_payload_bytes, 60);
    assert_eq!(s.report().retained_tensor_capacity_bytes, 72);
    assert_eq!(
        s.identity().container_sha256(),
        &<[u8; 32]>::from(Sha256::digest(stream()))
    );
    assert!(
        s.matrix("missing").is_err()
            && s.vector("missing").is_err()
            && s.convolution("missing").is_err()
    );
}

struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
        panic!("budget/type admission must precede payload read")
    }
}
#[test]
fn exact_budget_minus_one_and_missing_f32_optin_reject_before_read() {
    let p = fixture();
    let required = cold::reservation_for(&source_recipe(&p)).unwrap().total;
    assert!(matches!(private_load(p,&mut NoRead,&[1;32],required-1),
        Err(Error::Budget { required:r,limit }) if r==required && limit==required-1));
    let bytes = stream();
    let digest = Sha256::digest(&bytes).into();
    assert!(private_load(fixture(), &mut Cursor::new(&bytes[9..]), &digest, required).is_ok());
    let mut p = fixture();
    p.bytes.f32_vectors = 0;
    assert!(private_load(p, &mut NoRead, &[1; 32], usize::MAX).is_err());
    let mut p = fixture();
    p.physical.get_mut(ALOG).unwrap().dtype = Dtype::Bf16;
    assert!(private_load(p, &mut NoRead, &[1; 32], usize::MAX).is_err());
    let mut p = fixture();
    p.common[2].shape = Shape::Vector(usize::MAX);
    assert!(private_load(p, &mut NoRead, &[1; 32], usize::MAX).is_err());
}

#[test]
fn authenticated_nonfinite_original_f32_and_malformed_lengths_never_publish() {
    for bits in [0x7f80_0000u32, 0x7fc0_0001, 0xff80_0000] {
        let mut bytes = stream();
        bytes[49..53].copy_from_slice(&bits.to_le_bytes());
        let digest = Sha256::digest(&bytes).into();
        assert!(
            private_load(
                fixture(),
                &mut Cursor::new(&bytes[9..]),
                &digest,
                usize::MAX
            )
            .is_err()
        );
    }
    let mut p = fixture();
    p.physical.get_mut(ALOG).unwrap().range.end -= 1;
    assert!(private_load(p, &mut NoRead, &[1; 32], usize::MAX).is_err());
}

#[test]
fn corrupt_short_trailing_and_failed_io_streams_never_publish() {
    let bytes = stream();
    let digest = Sha256::digest(&bytes).into();
    for end in [0, 1, 31, 32, 35, 39, 43, 51, 59] {
        assert!(
            private_load(
                fixture(),
                &mut Cursor::new(&bytes[9..9 + end]),
                &digest,
                usize::MAX
            )
            .is_err()
        );
    }
    let mut corrupt = bytes[9..].to_vec();
    corrupt[0] ^= 1;
    assert!(private_load(fixture(), &mut Cursor::new(corrupt), &digest, usize::MAX).is_err());
    let mut tail = bytes[9..].to_vec();
    tail.push(0);
    assert!(private_load(fixture(), &mut Cursor::new(tail), &digest, usize::MAX).is_err());
    struct Failed;
    impl Read for Failed {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("control"))
        }
    }
    assert!(private_load(fixture(), &mut Failed, &digest, usize::MAX).is_err());
}

#[test]
fn interrupted_short_reads_keep_original_unaligned_fp32_values() {
    struct Short<'a> {
        bytes: &'a [u8],
        interrupted: bool,
    }
    impl Read for Short<'_> {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            if !self.interrupted {
                self.interrupted = true;
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            let n = out.len().min(3).min(self.bytes.len());
            out[..n].copy_from_slice(&self.bytes[..n]);
            self.bytes = &self.bytes[n..];
            Ok(n)
        }
    }
    let bytes = stream();
    let digest = Sha256::digest(&bytes).into();
    let s = private_load(
        fixture(),
        &mut Short {
            bytes: &bytes[9..],
            interrupted: false,
        },
        &digest,
        usize::MAX,
    )
    .unwrap();
    assert_eq!(
        s.vector(ALOG)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        F32_BITS
    );
}

#[test]
fn public_load_requires_fixed_real_recipe_metadata_not_tiny_private_controls() {
    let p = fixture();
    let options = LoadOptions {
        memory_limit_bytes: usize::MAX,
        modality: ModalityPolicy::TextOnly,
    };
    assert!(load(b"tiny-config", &mut NoRead, options, &p.pins).is_err());
    let (config, header, pins) = super::super::tests::documents();
    let mut header_stream = (header.len() as u64).to_le_bytes().to_vec();
    header_stream.extend(&header);
    let required = cold::reservation_for(&source_recipe(
        &inspect(
            &config,
            &header,
            pins.container_bytes,
            ModalityPolicy::TextOnly,
            &pins,
        )
        .unwrap(),
    ))
    .unwrap()
    .total;
    let mut reader = Cursor::new(header_stream);
    assert!(
        matches!(load(&config,&mut reader,LoadOptions { memory_limit_bytes:required-1,
        modality:ModalityPolicy::TextOnly },&pins),Err(Error::Budget { required:r,limit }) if r==required &&limit==required-1)
    );
    assert_eq!(reader.position(), (8 + header.len()) as u64);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;
