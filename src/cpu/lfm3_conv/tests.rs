use super::*;
use crate::cpu::{lfm3_add_bf16, lfm3_head_gate_bf16, lfm3_mul_bf16};
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

fn fixture() -> Value {
    let raw = include_bytes!("../../../tests/fixtures/lfm3-conv-bf16-v1/parent.json");
    assert_eq!(
        hash(raw),
        "c90abf0c6fed114ba8f3e1459d3a5e82a4fd94c0347249bde41cac0c1794493a"
    );
    serde_json::from_slice(raw).unwrap()
}

fn inputs(ordinal: usize, rows: usize, width: usize) -> Vec<bf16> {
    let pool: Vec<_> = [0, 0x8000]
        .into_iter()
        .flat_map(|sign| (0..=0x4180).map(move |word| bf16::from_bits(word | sign)))
        .collect();
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
fn every_source_bank_and_chunk_schedule_matches_canonical_bf16() {
    let fixture = fixture();
    let conv = &fixture["convolution"];
    let raw = include_bytes!("../../../tests/fixtures/lfm3-conv-bf16-v1/weights.bf16le");
    assert_eq!(hash(raw), conv["weights_sha256"].as_str().unwrap());
    assert_eq!(conv["banks"].as_array().unwrap().len(), 13);
    assert_eq!(conv["cases"].as_array().unwrap().len(), 104);
    assert_eq!(conv["candidate_word_differences"], 0);
    assert_eq!(conv["fp32_mathematical_diagnostic"]["failures"], 0);
    let mut total = 0;
    for case in conv["cases"].as_array().unwrap() {
        let ordinal = case["ordinal"].as_u64().unwrap() as usize;
        let tokens = case["tokens"].as_u64().unwrap() as usize;
        let c = case["channels"].as_u64().unwrap() as usize;
        let bank = &conv["banks"][ordinal];
        assert_eq!(bank["name"], case["coefficient"]);
        let start = bank["byte_offset"].as_u64().unwrap() as usize;
        let count = bank["byte_count"].as_u64().unwrap() as usize;
        let raw = &raw[start..start + count];
        assert_eq!(hash(raw), bank["sha256"].as_str().unwrap());
        let weights: Vec<_> = raw
            .chunks_exact(2)
            .map(|w| bf16::from_le_bytes(w.try_into().unwrap()))
            .collect();
        let input = inputs(ordinal, tokens, c);
        let history = if case["history_profile"] == "zero-history" {
            vec![bf16::ZERO; 6 * c]
        } else {
            inputs(ordinal + 31, 6, c)
        };
        assert_eq!(digest(&input), case["input_sha256"].as_str().unwrap());
        assert_eq!(digest(&history), case["history_sha256"].as_str().unwrap());
        let mut output = vec![bf16::NAN; input.len()];
        lfm3_conv7_bf16(&input, &weights, &history, &mut output, tokens, c).unwrap();
        assert_eq!(
            digest(&output),
            case["output_sha256"].as_str().unwrap(),
            "{}",
            case["name"]
        );
        // Each subcall borrows the exact six preceding raw BF16 rows. This
        // tests the leaf's chunk equivalence, without claiming an owner commit.
        let joined: Vec<_> = history.iter().chain(&input).copied().collect();
        let mut split = vec![bf16::NAN; input.len()];
        let mut after = 0;
        for count in [1, 6, 7, 17, 32].into_iter().cycle() {
            if after == tokens {
                break;
            }
            let count = count.min(tokens - after);
            lfm3_conv7_bf16(
                &input[after * c..(after + count) * c],
                &weights,
                &joined[after * c..(after + 6) * c],
                &mut split[after * c..(after + count) * c],
                count,
                c,
            )
            .unwrap();
            after += count;
        }
        assert_eq!(digest(&split), case["output_sha256"].as_str().unwrap());
        total += input.len();
    }
    assert_eq!(total, 8_146_944);
}

#[test]
fn binary_grids_match_source_including_overflow_rejection() {
    let fixture = fixture();
    let left: Vec<_> = (0..=u16::MAX)
        .filter(|word| word & 0x7f80 != 0x7f80)
        .map(bf16::from_bits)
        .collect();
    assert_eq!(left.len(), 65_280);
    assert_eq!(
        digest(&left),
        fixture["binary"]["left_sha256"].as_str().unwrap()
    );
    assert_eq!(fixture["binary"]["cases"].as_array().unwrap().len(), 34);
    for case in fixture["binary"]["cases"].as_array().unwrap() {
        let anchor = bf16::from_bits(case["anchor"].as_u64().unwrap() as u16);
        let multiply = case["operation"] == "mul";
        let mut observed = Vec::with_capacity(left.len());
        let mut rejected = 0;
        for &value in &left {
            let mut output = [bf16::NAN];
            let result = if multiply {
                lfm3_mul_bf16(&[value], &[anchor], &mut output)
            } else {
                lfm3_add_bf16(&[value], &[anchor], &mut output)
            };
            if result.is_ok() {
                observed.push(output[0]);
            } else {
                assert!(matches!(result, Err(Error::NonFinite { index: 0, .. })));
                assert!(output[0].is_nan());
                // Finite source operands can overflow to signed infinity,
                // never NaN. Insert that rejection marker into the source grid.
                let sign = if multiply {
                    value.to_bits() ^ anchor.to_bits()
                } else {
                    value.to_bits()
                } & 0x8000;
                observed.push(bf16::from_bits(sign | 0x7f80));
                rejected += 1;
            }
        }
        assert_eq!(rejected, case["nonfinite_output_words"].as_u64().unwrap());
        assert_eq!(
            digest(&observed),
            case["output_sha256"].as_str().unwrap(),
            "{}",
            case["name"]
        );
    }
}

#[test]
fn attention_gates_broadcast_each_head_without_expansion() {
    let fixture = fixture();
    for case in fixture["head_gate_cases"].as_array().unwrap() {
        let tokens = case["tokens"].as_u64().unwrap() as usize;
        let input = inputs(73, tokens * 16, 128);
        let pool = [0, 1, 128, 0x3c00, 0x3f00, 0x3f7f, 0x3f80];
        let gates: Vec<_> = (0..tokens * 16)
            .map(|index| bf16::from_bits(pool[(index * 5 + index / 16 * 3) % 7]))
            .collect();
        assert_eq!(digest(&input), case["input_sha256"].as_str().unwrap());
        assert_eq!(digest(&gates), case["gate_sha256"].as_str().unwrap());
        let mut output = vec![bf16::NAN; input.len()];
        lfm3_head_gate_bf16(&input, &gates, &mut output, tokens).unwrap();
        assert_eq!(digest(&output), case["output_sha256"].as_str().unwrap());
    }
}

#[test]
fn admission_errors_preserve_output_and_history() {
    let input = vec![bf16::ONE; 4];
    let weight = vec![bf16::ONE; 14];
    let history = vec![bf16::ONE; 12];
    let sentinel = bf16::from_f32(-7.);
    let mut output = vec![sentinel; 4];
    for (tokens, channels) in [(0, 2), (2, 0), (usize::MAX, 2), (2, 3)] {
        assert!(lfm3_conv7_bf16(&input, &weight, &history, &mut output, tokens, channels).is_err());
        assert_eq!(output, vec![sentinel; 4]);
    }
    for bank in 0..3 {
        let mut operands = [input.clone(), weight.clone(), history.clone()];
        *operands[bank].last_mut().unwrap() = bf16::NAN;
        assert!(
            lfm3_conv7_bf16(&operands[0], &operands[1], &operands[2], &mut output, 2, 2).is_err()
        );
        assert_eq!(output, vec![sentinel; 4]);
    }
    for (x, w, h, out) in [
        (3, 14, 12, 4),
        (4, 13, 12, 4),
        (4, 14, 11, 4),
        (4, 14, 12, 3),
    ] {
        assert!(
            lfm3_conv7_bf16(
                &input[..x],
                &weight[..w],
                &history[..h],
                &mut output[..out],
                2,
                2
            )
            .is_err()
        );
        assert_eq!(output, vec![sentinel; 4]);
    }
    for operation in [lfm3_add_bf16, lfm3_mul_bf16] {
        assert!(operation(&[], &[], &mut []).is_err());
        assert!(operation(&input, &input[..3], &mut output).is_err());
        assert!(operation(&input, &input, &mut output[..3]).is_err());
        assert_eq!(output, vec![sentinel; 4]);
    }
    for word in (0..=u16::MAX).filter(|word| word & 0x7f80 == 0x7f80) {
        let bad = [bf16::ONE, bf16::from_bits(word)];
        for operation in [lfm3_add_bf16, lfm3_mul_bf16] {
            let mut out = [sentinel; 2];
            assert!(operation(&bad, &[bf16::ONE; 2], &mut out).is_err());
            assert_eq!(out, [sentinel; 2]);
            assert!(operation(&[bf16::ONE; 2], &bad, &mut out).is_err());
            assert_eq!(out, [sentinel; 2]);
        }
    }
    let mut gate_output = vec![sentinel; 2048];
    assert!(
        lfm3_head_gate_bf16(
            &vec![bf16::ONE; 2048],
            &[bf16::ONE; 15],
            &mut gate_output,
            1
        )
        .is_err()
    );
    let mut gates = [bf16::ONE; 16];
    gates[15] = bf16::NAN;
    assert!(lfm3_head_gate_bf16(&vec![bf16::ONE; 2048], &gates, &mut gate_output, 1).is_err());
    assert_eq!(gate_output, vec![sentinel; 2048]);
    let mut gate_input = vec![bf16::ONE; 2048];
    gate_input[2047] = bf16::NAN;
    assert!(lfm3_head_gate_bf16(&gate_input, &[bf16::ONE; 16], &mut gate_output, 1).is_err());
    assert!(lfm3_head_gate_bf16(&[], &[], &mut [], 0).is_err());
    assert!(
        lfm3_head_gate_bf16(&gate_input, &[bf16::ONE; 16], &mut gate_output, usize::MAX).is_err()
    );
    assert_eq!(gate_output, vec![sentinel; 2048]);
    assert_eq!(history, vec![bf16::ONE; 12]);
}

#[test]
fn late_numerical_errors_leave_discardable_output_and_valid_retry() {
    let sentinel = bf16::from_f32(-7.);
    let input = [bf16::ONE, bf16::from_f32(2.)];
    let mut weight = [bf16::ZERO; 14];
    weight[6] = bf16::ONE;
    weight[13] = bf16::MAX;
    let history = [bf16::ZERO; 12];
    let mut output = [sentinel; 2];
    assert!(matches!(
        lfm3_conv7_bf16(&input, &weight, &history, &mut output, 1, 2),
        Err(Error::NonFinite { index: 1, .. })
    ));
    assert_eq!(output, [bf16::ONE, sentinel]);
    weight[13] = bf16::ONE;
    lfm3_conv7_bf16(&input, &weight, &history, &mut output, 1, 2).unwrap();
    assert_eq!(output, input);
    for operation in [lfm3_add_bf16, lfm3_mul_bf16] {
        output.fill(sentinel);
        assert!(matches!(
            operation(
                &[bf16::ONE, bf16::MAX],
                &[bf16::ONE, bf16::MAX],
                &mut output
            ),
            Err(Error::NonFinite { index: 1, .. })
        ));
        assert_eq!(output[1], sentinel);
        operation(&input, &[bf16::ONE; 2], &mut output).unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }
    assert_eq!(history, [bf16::ZERO; 12]);
    let mut input = vec![bf16::ONE; 2048];
    input[2047] = bf16::MAX;
    let mut gates = [bf16::ONE; 16];
    gates[15] = bf16::MAX;
    let mut output = vec![sentinel; 2048];
    assert!(matches!(
        lfm3_head_gate_bf16(&input, &gates, &mut output, 1),
        Err(Error::NonFinite { index: 2047, .. })
    ));
    assert_eq!(output[0], bf16::ONE);
    assert_eq!(output[2047], sentinel);
    gates[15] = bf16::ONE;
    lfm3_head_gate_bf16(&input, &gates, &mut output, 1).unwrap();
    assert_eq!(output, input);
}
