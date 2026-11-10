use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn hash(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}

fn digest(values: &[bf16]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn decode(raw: &[u8]) -> Vec<bf16> {
    assert!(raw.len().is_multiple_of(2));
    raw.chunks_exact(2)
        .map(|word| bf16::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn fixture() -> Value {
    let bytes = include_bytes!("../../../tests/fixtures/lfm3-bf16-v1/parent.json");
    assert_eq!(
        hash(bytes),
        "9f1494cd22e636e4e1787bcd73d7e1d11a5d89be72336792e8331357eb2fca91"
    );
    serde_json::from_slice(bytes).unwrap()
}

fn inputs(pool: &[bf16], ordinal: usize, rows: usize, width: usize) -> Vec<bf16> {
    (0..rows * width)
        .map(|index| {
            let selector = index * 37 + index / width * 11 + ordinal * 3;
            if index.is_multiple_of(2) {
                pool[selector % pool.len()]
            } else {
                bf16::from_bits(((0x3d80 + selector % 1025) | ((selector / 1025 % 2) << 15)) as u16)
            }
        })
        .collect()
}

#[test]
fn all_authenticated_gains_and_canonical_norm_outputs_match_bitwise() {
    let fixture = fixture();
    let norms = &fixture["norms"];
    let gains = include_bytes!("../../../tests/fixtures/lfm3-bf16-v1/norms.bf16le");
    assert_eq!(hash(gains), norms["weights_sha256"].as_str().unwrap());
    let pool: Vec<_> = norms["input_pool"]
        .as_array()
        .unwrap()
        .iter()
        .map(|word| bf16::from_bits(word.as_u64().unwrap() as u16))
        .collect();
    let mut output_words = 0;
    let mut statistics = 0;
    for case in norms["cases"].as_array().unwrap() {
        let n = |key: &str| case[key].as_u64().unwrap() as usize;
        let entry = &norms["vectors"][n("ordinal")];
        assert_eq!(entry["name"], case["gain"]);
        let start = entry["byte_offset"].as_u64().unwrap() as usize;
        let count = entry["byte_count"].as_u64().unwrap() as usize;
        let raw = &gains[start..start + count];
        assert_eq!(hash(raw), entry["sha256"].as_str().unwrap());
        let gain = decode(raw);
        let input = inputs(&pool, n("ordinal"), n("rows"), n("width"));
        assert_eq!(digest(&input), case["input_sha256"].as_str().unwrap());
        let mut output = vec![bf16::NAN; input.len()];
        lfm3_rmsnorm_bf16(&input, &gain, &mut output, n("rows"), n("width")).unwrap();
        assert_eq!(
            digest(&output),
            case["output_sha256"].as_str().unwrap(),
            "{}",
            case["name"]
        );
        for (index, row) in input.chunks_exact(n("width")).enumerate() {
            let variance = squared_sum(row) / n("width") as f32;
            let inverse = 1. / (variance + EPSILON).sqrt();
            assert_eq!(
                variance.to_bits(),
                case["variance_fp32_bits"][index].as_u64().unwrap() as u32,
                "variance {} row{index}",
                case["name"]
            );
            assert_eq!(
                inverse.to_bits(),
                case["inverse_fp32_bits"][index].as_u64().unwrap() as u32,
                "inverse {} row{index}",
                case["name"]
            );
            statistics += 1;
        }
        // Split row batches preserve their exact independent normalization.
        let mut split = vec![bf16::NAN; input.len()];
        for (source, target) in input
            .chunks_exact(n("width"))
            .zip(split.chunks_exact_mut(n("width")))
        {
            lfm3_rmsnorm_bf16(source, &gain, target, 1, n("width")).unwrap();
        }
        assert!(
            split
                .iter()
                .zip(output)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        output_words += input.len();
    }
    assert_eq!(output_words, 4_082_688);
    assert_eq!(statistics, 5706);
}

#[test]
fn full_finite_bf16_sigmoid_and_silu_match_canonical_bits() {
    let input: Vec<_> = (0..=u16::MAX)
        .map(bf16::from_bits)
        .filter(|word| word.is_finite())
        .collect();
    assert_eq!(input.len(), 65_280);
    let fixture = fixture();
    let mut output = vec![bf16::NAN; input.len()];
    let expected = include_bytes!("../../../tests/fixtures/lfm3-bf16-v1/sigmoid.bf16le");
    assert_eq!(
        hash(expected),
        fixture["pointwise"]["functions"]["sigmoid"]["output_sha256"]
            .as_str()
            .unwrap()
    );
    lfm3_sigmoid_bf16(&input, &mut output).unwrap();
    assert_eq!(digest(&output), hash(expected));
    let expected = include_bytes!("../../../tests/fixtures/lfm3-bf16-v1/silu.bf16le");
    assert_eq!(
        hash(expected),
        fixture["pointwise"]["functions"]["silu"]["output_sha256"]
            .as_str()
            .unwrap()
    );
    lfm3_silu_bf16(&input, &mut output).unwrap();
    assert_eq!(digest(&output), hash(expected));
}

#[test]
fn epsilon_dominated_and_signed_zero_norm_corners_match() {
    let raw = include_bytes!("../../../tests/fixtures/lfm3-bf16-v1/corners.json");
    assert_eq!(
        hash(raw),
        "5435676ae03cdb32bc3e6a76bed6f525149818adfb6fc7a442f30e30abc2cdc7"
    );
    let data: Value = serde_json::from_slice(raw).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let width = case["width"].as_u64().unwrap() as usize;
        let input: Vec<_> = (0..width)
            .map(|i| {
                bf16::from_bits(case["input_word"].as_u64().unwrap_or_else(|| {
                    case[if i == 0 {
                        "impulse_word"
                    } else {
                        "background_word"
                    }]
                    .as_u64()
                    .unwrap()
                }) as u16)
            })
            .collect();
        let gain: Vec<_> = (0..width)
            .map(|i| {
                bf16::from_bits(
                    case[if i.is_multiple_of(2) {
                        "even_gain_word"
                    } else {
                        "odd_gain_word"
                    }]
                    .as_u64()
                    .unwrap() as u16,
                )
            })
            .collect();
        let mut output = vec![bf16::NAN; width];
        lfm3_rmsnorm_bf16(&input, &gain, &mut output, 1, width).unwrap();
        assert_eq!(digest(&output), case["output_sha256"].as_str().unwrap());
        let expected: Vec<_> = case["first_output_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u16)
            .collect();
        assert_eq!(
            output[..8].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn geometry_nonfinite_and_every_nonfinite_unary_word_reject_before_writes() {
    let sentinel = bf16::from_bits(0x7fc1);
    let input = vec![bf16::ONE; 256];
    let gain = vec![bf16::ONE; 128];
    for (rows, width, input_len, gain_len, output_len) in [
        (0, 128, 0, 128, 0),
        (1, 0, 0, 0, 0),
        (1, 64, 64, 64, 64),
        (usize::MAX, 128, 128, 128, 128),
        (1, 128, 127, 128, 128),
        (1, 128, 128, 127, 128),
        (1, 128, 128, 128, 127),
    ] {
        let mut output = vec![sentinel; output_len];
        assert!(
            lfm3_rmsnorm_bf16(
                &input[..input_len],
                &gain[..gain_len],
                &mut output,
                rows,
                width
            )
            .is_err()
        );
        assert!(output.iter().all(|v| v.to_bits() == sentinel.to_bits()));
    }
    for index in [0, 255] {
        let mut bad = input.clone();
        bad[index] = bf16::INFINITY;
        let mut output = vec![sentinel; 256];
        assert!(lfm3_rmsnorm_bf16(&bad, &gain, &mut output, 2, 128).is_err());
        assert!(output.iter().all(|v| v.to_bits() == sentinel.to_bits()));
    }
    let mut bad_gain = gain.clone();
    bad_gain[127] = bf16::NAN;
    let mut output = vec![sentinel; 256];
    assert!(lfm3_rmsnorm_bf16(&input, &bad_gain, &mut output, 2, 128).is_err());
    assert!(output.iter().all(|v| v.to_bits() == sentinel.to_bits()));
    for word in (0..=u16::MAX)
        .map(bf16::from_bits)
        .filter(|v| !v.is_finite())
    {
        let source = [bf16::ONE, word, bf16::ONE];
        let mut output = [sentinel; 3];
        assert!(lfm3_sigmoid_bf16(&source, &mut output).is_err());
        assert!(output.iter().all(|v| v.to_bits() == sentinel.to_bits()));
        assert!(lfm3_silu_bf16(&source, &mut output).is_err());
        assert!(output.iter().all(|v| v.to_bits() == sentinel.to_bits()));
    }
    assert!(lfm3_sigmoid_bf16(&[], &mut []).is_err());
    assert!(lfm3_silu_bf16(&[], &mut []).is_err());
    assert!(lfm3_sigmoid_bf16(&input, &mut output[..1]).is_err());
    assert!(lfm3_silu_bf16(&input, &mut output[..1]).is_err());
}

#[test]
fn numerical_overflow_is_discardable_and_retry_is_valid() {
    let gain = vec![bf16::ONE; 128];
    let mut input = vec![bf16::ONE; 256];
    input[128..].fill(bf16::MAX);
    let mut output = vec![bf16::NAN; 256];
    assert!(lfm3_rmsnorm_bf16(&input, &gain, &mut output, 2, 128).is_err());
    assert!(
        output[..128]
            .iter()
            .all(|v| v.to_bits() == bf16::ONE.to_bits())
    );
    assert!(output[128..].iter().all(|v| v.is_nan()));
    input.fill(bf16::ONE);
    lfm3_rmsnorm_bf16(&input, &gain, &mut output, 2, 128).unwrap();
    assert!(output.iter().all(|v| v.to_bits() == bf16::ONE.to_bits()));
    let mut input = vec![bf16::ZERO; 128];
    input[0] = bf16::from_f32(16.);
    let mut gain = gain;
    gain[0] = bf16::MAX;
    output.fill(bf16::NAN);
    assert!(lfm3_rmsnorm_bf16(&input, &gain, &mut output[..128], 1, 128).is_err());
    assert!(output[0].is_nan());
}
