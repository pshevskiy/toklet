use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

#[test]
fn finite_only_widening_matches_independent_half_conversion_for_every_finite_word() {
    let mut count = 0;
    for bits in 0..=u16::MAX {
        let value = bf16::from_bits(bits);
        if value.is_finite() {
            assert_eq!(
                widen_finite(value).to_bits(),
                value.to_f32().to_bits(),
                "finite BF16 word {bits:04x}"
            );
            count += 1;
        }
    }
    assert_eq!(count, 65_280);
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn data(name: &str) -> &'static [u8] {
    match name {
        "weights.bf16le" => {
            include_bytes!("../../../tests/fixtures/lfm3-projection-bf16-v1/weights.bf16le")
        }
        "source.bf16le" => {
            include_bytes!("../../../tests/fixtures/lfm3-projection-bf16-v1/source.bf16le")
        }
        "reference.f64le" => {
            include_bytes!("../../../tests/fixtures/lfm3-projection-bf16-v1/reference.f64le")
        }
        "corner-input.bf16le" => {
            include_bytes!("../../../tests/fixtures/lfm3-projection-bf16-v1/corner-input.bf16le")
        }
        "corner-weight.bf16le" => {
            include_bytes!("../../../tests/fixtures/lfm3-projection-bf16-v1/corner-weight.bf16le")
        }
        "corner-source.bf16le" => {
            include_bytes!("../../../tests/fixtures/lfm3-projection-bf16-v1/corner-source.bf16le")
        }
        "finite-grid.bf16le" => {
            include_bytes!("../../../tests/fixtures/lfm3-projection-bf16-v1/finite-grid.bf16le")
        }
        _ => panic!("Unknown frozen payload"),
    }
}
fn n(v: &Value, key: &str) -> usize {
    v[key].as_u64().unwrap().try_into().unwrap()
}
fn text<'a>(v: &'a Value, key: &str) -> &'a str {
    v[key].as_str().unwrap()
}
fn fixture() -> &'static Value {
    static FIXTURE: OnceLock<Value> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let raw = include_bytes!("../../../tests/fixtures/lfm3-projection-bf16-v1/fixture.json");
        assert_eq!(
            sha(raw),
            "0f5f8a970d116b32073aa6090bd64b91825f39db2adfc9311eb5995b6367cd74"
        );
        let v: Value = serde_json::from_slice(raw).unwrap();
        for (name, meta) in v["files"].as_object().unwrap() {
            assert_eq!(data(name).len(), n(meta, "bytes"));
            assert_eq!(sha(data(name)), text(meta, "sha256"));
        }
        v
    })
}
fn raw_span(v: &Value) -> &'static [u8] {
    let begin = n(v, "byte_offset");
    let raw = &data(text(v, "file"))[begin..begin + n(v, "byte_count")];
    assert_eq!(sha(raw), text(v, "sha256"));
    raw
}
fn words(v: &Value) -> Vec<bf16> {
    raw_span(v)
        .chunks_exact(2)
        .map(|b| bf16::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn word_sha(values: &[bf16]) -> String {
    sha(&values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect::<Vec<_>>())
}
fn input(ordinal: usize, rows: usize, width: usize) -> Vec<bf16> {
    static POOL: OnceLock<Vec<u16>> = OnceLock::new();
    let pool = POOL.get_or_init(|| {
        [0, 0x8000]
            .into_iter()
            .flat_map(|sign| (0..=0x4180).map(move |v| v | sign))
            .collect()
    });
    (0..rows * width)
        .map(|index| {
            let selector = index * 37 + index / width * 11 + ordinal * 3;
            let word = if index % 2 == 1 {
                ((0x3d80 + selector % 1025) | ((selector / 1025 % 2) << 15)) as u16
            } else {
                pool[selector % pool.len()]
            };
            bf16::from_bits(word)
        })
        .collect()
}

#[test]
fn frozen_complete_rank_two_inventory_and_payloads() {
    let f = fixture();
    let contract = include_bytes!("../../../tests/fixtures/lfm3-projection-bf16-v1/contract.json");
    assert_eq!(
        sha(contract),
        "ad76bdd309690fb7c7063930f714d26f09a49618565cce4a3aed52e305eccc6c"
    );
    assert_eq!(text(f, "contract_sha256"), sha(contract));
    assert_eq!(
        text(f, "producer_sha256"),
        sha(include_bytes!(
            "../../../tools/reference/lfm3_projection_bf16_fixture.py"
        ))
    );
    assert_eq!(f["matrices"].as_array().unwrap().len(), 154);
    assert_eq!(f["cases"].as_array().unwrap().len(), 616);
    assert_eq!(n(f, "learned_output_words"), 94248);
    assert_eq!(n(f, "corner_output_words"), 742);
    assert_eq!(n(f, "corner_rejection_markers"), 3);
    let c: Value = serde_json::from_slice(contract).unwrap();
    for (entry, original) in f["matrices"]
        .as_array()
        .unwrap()
        .iter()
        .zip(c["matrices"].as_array().unwrap())
    {
        assert_eq!(entry["name"], original["name"]);
        assert_eq!(entry["source_shape"], original["shape"]);
        assert_eq!(entry["selected_rows"], original["selected_rows"]);
        assert_eq!(
            n(&entry["weights"], "byte_count"),
            original["shape"][1].as_u64().unwrap() as usize * 4 * 2
        );
    }
}

#[test]
fn learned_rows_exact_source_and_original_fp32_diagnostic() {
    let f = fixture();
    let mut checked = 0;
    let mut failures = Vec::new();
    let mut maximum = 0_f64;
    for matrix in f["matrices"].as_array().unwrap() {
        let ordinal = n(matrix, "ordinal");
        let weights = words(&matrix["weights"]);
        for case in f["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| n(c, "ordinal") == ordinal)
        {
            let (tokens, columns, rows) = (
                n(case, "tokens"),
                n(case, "in_features"),
                n(case, "out_features"),
            );
            let x = input(ordinal, tokens, columns);
            assert_eq!(word_sha(&x), text(case, "input_sha256"));
            let mut output = vec![bf16::ZERO; tokens * rows];
            lfm3_linear_bf16(&x, &weights, &mut output, tokens, columns, rows).unwrap();
            assert_eq!(
                word_sha(&output),
                text(&case["output"], "sha256"),
                "{}",
                text(case, "name")
            );
            let reference: Vec<_> = raw_span(&case["mathematical_reference"])
                .chunks_exact(8)
                .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
                .collect();
            for (token, values) in x.chunks_exact(columns).enumerate() {
                for (row, coefficients) in weights.chunks_exact(columns).enumerate() {
                    let actual = f64::from(dot(values, coefficients));
                    let expected = reference[token * rows + row];
                    let error = (actual - expected).abs();
                    maximum = maximum.max(error);
                    if !actual.is_finite() || error > 1e-5 + 1e-4 * expected.abs() {
                        failures.push((text(case, "name"), token, row, actual, expected));
                    }
                    checked += 1;
                }
            }
        }
    }
    println!(
        "LFM3_PROJECTION source_words={checked} fp32_diagnostic_failures={} maximum_absolute_error={maximum:.12e}",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "First failures: {:?}",
        &failures[..failures.len().min(16)]
    );
    assert_eq!(checked, 94248);
}

#[test]
fn learned_token_partitions_preserve_source_words() {
    let f = fixture();
    for matrix in f["matrices"].as_array().unwrap() {
        let ordinal = n(matrix, "ordinal");
        let case = f["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| n(c, "ordinal") == ordinal && n(c, "tokens") == 128)
            .unwrap();
        let columns = n(case, "in_features");
        let x = input(ordinal, 128, columns);
        let weights = words(&matrix["weights"]);
        let mut output = vec![bf16::ZERO; 128 * 4];
        let mut offset = 0;
        for tokens in [1, 7, 17, 103] {
            lfm3_linear_bf16(
                &x[offset * columns..(offset + tokens) * columns],
                &weights,
                &mut output[offset * 4..(offset + tokens) * 4],
                tokens,
                columns,
                4,
            )
            .unwrap();
            offset += tokens;
        }
        assert_eq!(offset, 128);
        assert_eq!(word_sha(&output), text(&case["output"], "sha256"));
    }
}

#[test]
fn tails_midpoints_cancellation_and_source_rejection_markers() {
    let mut checked = 0;
    let mut rejections = 0;
    for c in fixture()["corners"].as_array().unwrap() {
        let x = words(&c["input"]);
        let weights = words(&c["weight"]);
        let (tokens, columns, rows) = (n(c, "tokens"), n(c, "in_features"), n(c, "out_features"));
        let mut output = vec![bf16::ZERO; tokens * rows];
        let result = lfm3_linear_bf16(&x, &weights, &mut output, tokens, columns, rows);
        if n(c, "rejection_markers") == 0 {
            result.unwrap();
            assert_eq!(
                word_sha(&output),
                text(&c["output"], "sha256"),
                "{}",
                text(c, "name")
            );
        } else {
            assert!(
                matches!(result, Err(Error::NonFinite { .. })),
                "{}",
                text(c, "name")
            );
            rejections += n(c, "rejection_markers");
        }
        checked += tokens * rows;
    }
    assert_eq!(checked, 742);
    assert_eq!(rejections, 3);
}

#[test]
fn every_finite_bf16_input_matches_source_identity_grid() {
    let finite: Vec<_> = (0..=u16::MAX)
        .filter(|word| word & 0x7f80 != 0x7f80)
        .collect();
    assert_eq!(finite.len(), 65280);
    let mut output = Vec::new();
    for (batch, chunk) in finite.chunks(257).enumerate() {
        let mut x = vec![bf16::ZERO; chunk.len() * 32];
        for (row, &word) in chunk.iter().enumerate() {
            x[row * 32 + (batch * 257 + row) % 32] = bf16::from_bits(word);
        }
        let mut values = vec![bf16::ZERO; chunk.len()];
        lfm3_linear_bf16(&x, &[bf16::ONE; 32], &mut values, chunk.len(), 32, 1).unwrap();
        output.extend(values);
    }
    assert_eq!(
        word_sha(&output),
        text(&fixture()["files"]["finite-grid.bf16le"], "sha256")
    );
}

#[test]
fn admission_and_all_nonfinite_patterns_precede_output_writes() {
    let sentinel = bf16::from_f32(17.);
    let mut x = [bf16::ONE; 33];
    let mut weights = [bf16::ONE; 99];
    let mut output = [sentinel; 3];
    let mut patterns = 0;
    for word in 0..=u16::MAX {
        if word & 0x7f80 != 0x7f80 {
            continue;
        }
        x[32] = bf16::from_bits(word);
        assert!(lfm3_linear_bf16(&x, &weights, &mut output, 1, 33, 3).is_err());
        assert_eq!(output, [sentinel; 3]);
        x[32] = bf16::ONE;
        weights[98] = bf16::from_bits(word);
        assert!(lfm3_linear_bf16(&x, &weights, &mut output, 1, 33, 3).is_err());
        assert_eq!(output, [sentinel; 3]);
        weights[98] = bf16::ONE;
        patterns += 1;
    }
    assert_eq!(patterns, 256);
    for (tokens, columns, rows) in [
        (0, 33, 3),
        (1, 0, 3),
        (1, 33, 0),
        (usize::MAX, 2, 3),
        (1, 2, usize::MAX),
    ] {
        assert!(lfm3_linear_bf16(&x, &weights, &mut output, tokens, columns, rows).is_err());
        assert_eq!(output, [sentinel; 3]);
    }
    assert!(lfm3_linear_bf16(&x[..32], &weights, &mut output, 1, 33, 3).is_err());
    assert!(lfm3_linear_bf16(&x, &weights[..98], &mut output, 1, 33, 3).is_err());
    assert!(lfm3_linear_bf16(&x, &weights, &mut output[..2], 1, 33, 3).is_err());
    assert_eq!(output, [sentinel; 3]);
}

#[test]
fn numerical_overflow_is_discardable_and_valid_retry_succeeds() {
    let max = bf16::from_bits(0x7f7f);
    let mut x = vec![bf16::ZERO; 66];
    x[33..].fill(max);
    let weight = [max; 33];
    let snapshot = x.clone();
    let sentinel = bf16::from_f32(17.);
    let mut output = [sentinel; 2];
    assert!(matches!(
        lfm3_linear_bf16(&x, &weight, &mut output, 2, 33, 1),
        Err(Error::NonFinite { index: 1, .. })
    ));
    assert_eq!(output, [bf16::ZERO, sentinel]);
    assert_eq!(x, snapshot);
    x[33..].fill(bf16::ZERO);
    lfm3_linear_bf16(&x, &weight, &mut output, 2, 33, 1).unwrap();
    assert_eq!(output, [bf16::ZERO; 2]);
    let x = [max, bf16::from_bits(0x7b00)];
    let sum = dot(&x, &[bf16::ONE; 2]);
    assert!(sum.is_finite());
    assert!(!bf16::from_f32(sum).is_finite());
    assert!(lfm3_linear_bf16(&x, &[bf16::ONE; 2], &mut output[..1], 1, 2, 1).is_err());
}

#[test]
fn row_major_subslices_require_only_bf16_alignment() {
    let guard = bf16::from_f32(17.);
    let input: Vec<_> = [17., 1., 2., 3., 4., 5., 6., 17.]
        .map(bf16::from_f32)
        .to_vec();
    let weights: Vec<_> = [17., 1., 1., 1., -1., -1., -1., 17.]
        .map(bf16::from_f32)
        .to_vec();
    let mut output = [guard; 6];
    lfm3_linear_bf16(&input[1..7], &weights[1..7], &mut output[1..5], 2, 3, 2).unwrap();
    assert_eq!(output, [17., 6., -6., 15., -15., 17.].map(bf16::from_f32));
}
