use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn fixture() -> Value {
    let raw = include_bytes!("../../../tests/fixtures/lfm3-rotary-bf16-v1.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "3588b6de41c00c4536135bc687c27c06bed22a48eed909f6c6901c7e0b2faba4"
    );
    serde_json::from_slice(raw).unwrap()
}

fn profile(name: &str) -> Profile {
    match name {
        "full_attention" => Profile::FullAttention,
        "sliding_attention" => Profile::SlidingAttention,
        _ => panic!("unknown fixture profile"),
    }
}

fn digest(values: &[bf16]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn table_hash(table: &Rotary) -> String {
    let mut hash = Sha256::new();
    for row in &table.coefficients {
        for value in row {
            hash.update(value.to_bits().to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

#[test]
fn frozen_source_tables_and_all_forty_rotation_cases_match_bitwise() {
    let fixture = fixture();
    let pool: Vec<_> = fixture["input_pool"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| bf16::from_bits(v.as_u64().unwrap() as u16))
        .collect();
    let mut count = 0;
    for row in fixture["profiles"].as_array().unwrap() {
        let name = row["name"].as_str().unwrap();
        let kind = profile(name);
        let table = Rotary::new(kind, 2048, Rotary::planned_bytes(2048).unwrap()).unwrap();
        assert_eq!(table.profile(), kind);
        assert_eq!(table.capacity(), 2048);
        assert_eq!(
            table.allocated_bytes().unwrap(),
            Rotary::planned_bytes(2048).unwrap()
        );
        for (got, want) in table
            .frequencies()
            .iter()
            .zip(row["frequency_bits"].as_array().unwrap())
        {
            assert_eq!(got.to_bits(), want.as_u64().unwrap() as u32);
        }
        assert_eq!(
            table_hash(&table),
            row["prefix_coefficient_sha256"].as_str().unwrap()
        );
        for case in fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|case| case["profile"] == name)
        {
            let n = |key: &str| case[key].as_u64().unwrap() as usize;
            let (tokens, heads, offset) = (n("tokens"), n("heads"), n("offset"));
            let stride = heads * HEAD_DIM;
            let input: Vec<_> = (0..tokens * stride)
                .map(|index| {
                    pool[(index * 73
                        + index / stride * 11
                        + (index / HEAD_DIM % heads) * 7
                        + offset * 3)
                        % pool.len()]
                })
                .collect();
            assert_eq!(digest(&input), case["input_sha256"].as_str().unwrap());
            let mut output = vec![bf16::NAN; input.len()];
            table
                .apply(&input, &mut output, tokens, heads, offset)
                .unwrap();
            assert_eq!(
                digest(&output),
                case["output_sha256"].as_str().unwrap(),
                "{name}/{offset}/{tokens}/{heads}"
            );
            for chunks in [
                vec![tokens],
                vec![1; tokens],
                if tokens == 17 {
                    vec![1, 3, 5, 8]
                } else {
                    vec![1]
                },
            ] {
                let mut joined = vec![bf16::NAN; input.len()];
                let mut start = 0;
                for length in chunks {
                    let range = start * stride..(start + length) * stride;
                    table
                        .apply(
                            &input[range.clone()],
                            &mut joined[range],
                            length,
                            heads,
                            offset + start,
                        )
                        .unwrap();
                    start += length;
                }
                assert_eq!(start, tokens);
                assert!(
                    joined
                        .iter()
                        .zip(&output)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
            count += input.len();
        }
        assert_eq!(
            table_hash(&table),
            row["prefix_coefficient_sha256"].as_str().unwrap()
        );
    }
    assert_eq!(count, 460_800);
}

#[test]
#[ignore = "Explicit source-limit table construction and sparse-position qualification"]
fn frozen_sparse_positions_through_source_limit_match_bitwise() {
    for row in fixture()["profiles"].as_array().unwrap() {
        let table = Rotary::new(
            profile(row["name"].as_str().unwrap()),
            SOURCE_MAX_POSITIONS,
            Rotary::planned_bytes(SOURCE_MAX_POSITIONS).unwrap(),
        )
        .unwrap();
        for sparse in row["sparse"].as_array().unwrap() {
            let position = sparse["position"].as_u64().unwrap() as usize;
            let got: Vec<_> = table
                .coefficient_row(position)
                .unwrap()
                .iter()
                .flat_map(|pair| pair.iter().map(|x| x.to_bits()))
                .collect();
            let want: Vec<_> = sparse["coefficient_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u16)
                .collect();
            assert_eq!(got, want, "{position}");
        }
        assert!(table.coefficient_row(SOURCE_MAX_POSITIONS).is_none());
        assert!(table.coefficient_row(usize::MAX).is_none());
    }
}

#[test]
fn signed_zero_pairs_execute_source_operations_instead_of_copying() {
    let table = Rotary::new(Profile::FullAttention, 2, Rotary::planned_bytes(2).unwrap()).unwrap();
    let mut input = vec![bf16::from_bits(0x8000); HEAD_DIM];
    input[33] = bf16::ZERO;
    let mut output = vec![bf16::NAN; HEAD_DIM];
    table.apply(&input, &mut output, 1, 1, 1).unwrap();
    assert_eq!(output[32].to_bits(), 0);
    assert_eq!(output[96].to_bits(), 0x8000);
    assert_eq!(output[33].to_bits(), 0);
    assert_eq!(output[97].to_bits(), 0);
}

#[test]
fn admission_and_entire_input_preflight_before_output_writes() {
    for capacity in [0, SOURCE_MAX_POSITIONS + 1, usize::MAX] {
        assert!(Rotary::planned_bytes(capacity).is_err());
        assert!(Rotary::new(Profile::FullAttention, capacity, usize::MAX).is_err());
    }
    let bytes = Rotary::planned_bytes(2).unwrap();
    assert!(Rotary::new(Profile::FullAttention, 2, bytes - 1).is_err());
    let table = Rotary::new(Profile::FullAttention, 2, bytes).unwrap();
    let good = vec![bf16::ONE; 2 * HEAD_DIM];
    let sentinel = bf16::from_bits(0x7fc1);
    for (tokens, heads, offset, input_len, output_len) in [
        (0, 1, 0, 0, 0),
        (1, 0, 0, 0, 0),
        (1, 1, 2, 128, 128),
        (2, 1, usize::MAX, 256, 256),
        (1, usize::MAX, 0, 128, 128),
        (1, 1, 0, 127, 128),
        (1, 1, 0, 128, 127),
        (1, 1, 0, 128, 129),
    ] {
        let mut output = vec![sentinel; output_len];
        assert!(
            table
                .apply(&good[..input_len], &mut output, tokens, heads, offset)
                .is_err()
        );
        assert!(output.iter().all(|v| v.to_bits() == sentinel.to_bits()));
    }
    for bits in [0x7f80, 0xff80, 0x7fc1, 0xffc1] {
        for index in [0, good.len() - 1] {
            let mut input = good.clone();
            input[index] = bf16::from_bits(bits);
            let mut output = vec![sentinel; input.len()];
            assert!(table.apply(&input, &mut output, 2, 1, 0).is_err());
            assert!(output.iter().all(|v| v.to_bits() == sentinel.to_bits()));
        }
    }
}

#[test]
fn late_numerical_overflow_discards_output_and_owner_remains_reusable() {
    let table = Rotary::new(
        Profile::SlidingAttention,
        2,
        Rotary::planned_bytes(2).unwrap(),
    )
    .unwrap();
    let before = table_hash(&table);
    let mut input = vec![bf16::ZERO; 2 * HEAD_DIM];
    input[HEAD_DIM] = bf16::MAX;
    input[HEAD_DIM + 64] = bf16::MAX;
    let mut output = vec![bf16::NAN; input.len()];
    assert!(table.apply(&input, &mut output, 2, 1, 0).is_err());
    assert!(output[..HEAD_DIM].iter().all(|v| v.to_bits() == 0));
    assert!(output[HEAD_DIM].is_nan());
    input.fill(bf16::ONE);
    table.apply(&input, &mut output, 2, 1, 0).unwrap();
    assert!(output.iter().all(|v| v.is_finite()));
    assert_eq!(table_hash(&table), before);
}
