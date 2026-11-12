//! Immutable cached Half YaRN rotation for the selected Ministral 3 profile.
//!
//! This primitive neither admits a source configuration nor constructs or
//! qualifies a model. It leaves the independent scalar rotary references and
//! ordinary Metal position limit unchanged.

use super::Error;
use crate::cpu::{self, length, nonzero, product};
use half::f16;

pub const HEAD_DIM: usize = 128;
pub const SOURCE_MAX_POSITIONS: usize = 262_144;
pub const ORIGINAL_MAX_POSITIONS: usize = 16_384;
const PAIRS: usize = HEAD_DIM / 2;
const THETA: f64 = 1_000_000.;
const FACTOR: f32 = 16.;
const QUERY_BETA: f32 = 0.1;

/// Model-owned source periods, FP32 `[capacity,64,{cos,sin}]` coefficients,
/// and Half `[capacity]` query gains. Row zero belongs to `base`, not position
/// zero. Pair `i` joins dimensions `i` and `i+64`. Heads/layers share this
/// immutable owner; all activation, scratch and KV storage belongs to callers.
///
/// The selected source mscale/all-dimension ratio is one. Keys only rotate;
/// queries rotate to Half, then multiply by the cached Half gain and store Half
/// again. Periods are denominators: angles divide the absolute position by the
/// stored period, without replacing this source arithmetic with reciprocals.
pub struct Rotary {
    base: usize,
    periods: [f32; PAIRS],
    coefficients: Vec<[f32; 2]>,
    query_scales: Vec<f16>,
}

fn storage_bytes(count: usize, element_bytes: usize) -> Result<usize, Error> {
    let bytes = product(count, element_bytes)?;
    if bytes > isize::MAX as usize {
        return Err(cpu::Error::SizeOverflow.into());
    }
    Ok(bytes)
}

fn owner_bytes(coefficient_capacity: usize, gain_capacity: usize) -> Result<usize, Error> {
    let coefficients = storage_bytes(coefficient_capacity, std::mem::size_of::<[f32; 2]>())?;
    let gains = storage_bytes(gain_capacity, std::mem::size_of::<f16>())?;
    std::mem::size_of::<Rotary>()
        .checked_add(coefficients)
        .and_then(|bytes| bytes.checked_add(gains))
        .ok_or(cpu::Error::SizeOverflow.into())
}

impl Rotary {
    /// Inline object (including periods/Vec descriptors) plus requested table
    /// storage. Allocator bookkeeping and caller input/output storage are
    /// separate. No prefix before `base` is allocated.
    pub fn planned_bytes(capacity: usize) -> Result<usize, Error> {
        nonzero(capacity, "Ministral 3 rotary capacity")?;
        if capacity > SOURCE_MAX_POSITIONS {
            return Err(Error::InvalidConfig(
                "Ministral 3 rotary source position limit",
            ));
        }
        owner_bytes(product(capacity, PAIRS)?, capacity)
    }

    /// Admit an absolute-position span against the remaining model budget,
    /// before reserving. Actual Vec capacities are checked again after reserve.
    /// This does not assert that any complete model fits that budget.
    pub fn new(base: usize, capacity: usize, available_bytes: usize) -> Result<Self, Error> {
        let planned = Self::planned_bytes(capacity)?;
        let after = base.checked_add(capacity).ok_or(cpu::Error::SizeOverflow)?;
        if after > SOURCE_MAX_POSITIONS {
            return Err(Error::Capacity {
                offset: base,
                tokens: capacity,
                limit: SOURCE_MAX_POSITIONS,
            });
        }
        if planned > available_bytes {
            return Err(Error::InvalidConfig("Ministral 3 rotary admitted memory"));
        }
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(product(capacity, PAIRS)?)
            .map_err(|_| Error::Allocation)?;
        let mut query_scales = Vec::new();
        query_scales
            .try_reserve_exact(capacity)
            .map_err(|_| Error::Allocation)?;
        let mut table = Self {
            base,
            periods: [0.; PAIRS],
            coefficients,
            query_scales,
        };
        if table.allocated_bytes()? > available_bytes {
            return Err(Error::InvalidConfig("Ministral 3 rotary allocated memory"));
        }

        // Correction dimensions use FP64 logarithms and truncation, exactly as
        // the selected source. Its upper clamp is dims-1, not pair-count-1.
        let correction = |rotations: f64| {
            HEAD_DIM as f64
                * (ORIGINAL_MAX_POSITIONS as f64 / (rotations * std::f64::consts::TAU)).ln()
                / (2. * THETA.ln())
        };
        let low = correction(32.).floor().max(0.) as f32;
        let high = correction(1.).ceil().min((HEAD_DIM - 1) as f64) as f32;
        for (pair, period) in table.periods.iter_mut().enumerate() {
            let exponent = (2 * pair) as f32 / HEAD_DIM as f32;
            let extra = THETA.powf(f64::from(exponent)) as f32;
            let inter = FACTOR * extra;
            let ramp = ((pair as f32 - low) / (high - low)).clamp(0., 1.);
            let mask = 1. - ramp;
            // Preserve the source numerator/denominator and separate FP32
            // operations. Algebraic simplification changes stored period words.
            let numerator = inter * extra;
            let extra_weight = 1. - mask;
            let weighted_inter = inter * mask;
            let weighted_extra = extra * extra_weight;
            let denominator = weighted_inter + weighted_extra;
            *period = numerator / denominator;
        }
        for position in base..after {
            for &period in &table.periods {
                let angle = position as f32 / period;
                let (sine, cosine) = f64::from(angle).sin_cos();
                table.coefficients.push([cosine as f32, sine as f32]);
            }
            let epoch = position / ORIGINAL_MAX_POSITIONS;
            let logarithm = ((1 + epoch) as f64).ln() as f32;
            let scaled = QUERY_BETA * logarithm;
            table.query_scales.push(f16::from_f32(1. + scaled));
        }
        Ok(table)
    }

    /// Inline object plus actual reserved capacities, counted once.
    pub fn allocated_bytes(&self) -> Result<usize, Error> {
        owner_bytes(self.coefficients.capacity(), self.query_scales.capacity())
    }

    pub fn base(&self) -> usize {
        self.base
    }

    pub fn capacity(&self) -> usize {
        self.query_scales.len()
    }

    pub fn periods(&self) -> &[f32; PAIRS] {
        &self.periods
    }

    /// Borrow one absolute-position row; no copy or per-head expansion.
    pub fn coefficient_row(&self, absolute_position: usize) -> Option<&[[f32; 2]]> {
        let row = absolute_position.checked_sub(self.base)?;
        if row >= self.capacity() {
            return None;
        }
        Some(&self.coefficients[row * PAIRS..(row + 1) * PAIRS])
    }

    pub fn query_scale(&self, absolute_position: usize) -> Option<f16> {
        self.query_scales
            .get(absolute_position.checked_sub(self.base)?)
            .copied()
    }

    /// Rotate contiguous caller-owned `[tokens,heads,128]` Half keys.
    ///
    /// Absolute span, nonzero sizes, exact lengths and all finite inputs validate
    /// before writes. Half inputs widen exactly; separate FP32 products and
    /// add/subtract precede a nearest-even Half store. Numerical/store overflow
    /// may leave partial output, which the caller must discard. This immutable
    /// owner remains reusable and publishes no session state on either outcome.
    /// No allocation, trig, dynamic dispatch or profile selection occurs here.
    pub fn apply_keys(
        &self,
        input: &[f16],
        output: &mut [f16],
        tokens: usize,
        heads: usize,
        offset: usize,
    ) -> Result<(), Error> {
        self.apply::<false>(input, output, tokens, heads, offset)
    }

    /// Rotate Half queries, store Half, multiply by the already-Half cached gain
    /// and store Half a second time. Validation, error and ownership guarantees
    /// are the same as [`Self::apply_keys`].
    pub fn apply_queries(
        &self,
        input: &[f16],
        output: &mut [f16],
        tokens: usize,
        heads: usize,
        offset: usize,
    ) -> Result<(), Error> {
        self.apply::<true>(input, output, tokens, heads, offset)
    }

    fn apply<const QUERY: bool>(
        &self,
        input: &[f16],
        output: &mut [f16],
        tokens: usize,
        heads: usize,
        offset: usize,
    ) -> Result<(), Error> {
        nonzero(tokens, "Ministral 3 rotary tokens")?;
        nonzero(heads, "Ministral 3 rotary heads")?;
        let after = offset.checked_add(tokens).ok_or(cpu::Error::SizeOverflow)?;
        let limit = self.base + self.capacity(); // Constructor checked this sum.
        if offset < self.base || after > limit {
            return Err(Error::Capacity {
                offset,
                tokens,
                limit,
            });
        }
        let stride = product(heads, HEAD_DIM)?;
        let count = product(tokens, stride)?;
        length("Ministral 3 rotary input", input.len(), count)?;
        length("Ministral 3 rotary output", output.len(), count)?;
        if let Some(index) = input.iter().position(|value| !value.is_finite()) {
            return Err(cpu::Error::NonFinite {
                tensor: "Ministral 3 rotary input",
                index,
            }
            .into());
        }
        let start = offset - self.base;
        for (token, (source, target)) in input
            .chunks_exact(stride)
            .zip(output.chunks_exact_mut(stride))
            .enumerate()
        {
            let row = start + token;
            let first = row * PAIRS;
            let coefficients = &self.coefficients[first..first + PAIRS];
            let gain = self.query_scales[row].to_f32();
            for (head, (source, target)) in source
                .chunks_exact(HEAD_DIM)
                .zip(target.chunks_exact_mut(HEAD_DIM))
                .enumerate()
            {
                for (pair, &[cosine, sine]) in coefficients.iter().enumerate() {
                    let (a, b) = (source[pair].to_f32(), source[pair + PAIRS].to_f32());
                    let ac = a * cosine;
                    let bs = b * sine;
                    let bc = b * cosine;
                    let asine = a * sine;
                    let mut first = f16::from_f32(ac - bs);
                    let mut second = f16::from_f32(bc + asine);
                    if !first.is_finite() || !second.is_finite() {
                        return Err(cpu::Error::NonFinite {
                            tensor: "Ministral 3 rotary output",
                            index: token * stride
                                + head * HEAD_DIM
                                + pair
                                + if first.is_finite() { PAIRS } else { 0 },
                        }
                        .into());
                    }
                    // Const specialization selects keys/queries outside the hot
                    // loop. The first Half store is an explicit dtype boundary.
                    if QUERY {
                        first = f16::from_f32(first.to_f32() * gain);
                        second = f16::from_f32(second.to_f32() * gain);
                        if !first.is_finite() || !second.is_finite() {
                            return Err(cpu::Error::NonFinite {
                                tensor: "Ministral 3 scaled query output",
                                index: token * stride
                                    + head * HEAD_DIM
                                    + pair
                                    + if first.is_finite() { PAIRS } else { 0 },
                            }
                            .into());
                        }
                    }
                    target[pair] = first;
                    target[pair + PAIRS] = second;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use sha2::{Digest, Sha256};

    const PROFILE_SHA: &str = "0f8d25bec77f2f6901e86830b7da22b75808fccb7800ce939f17f386206f4c9f";

    fn check_binding(binding: &Value, raw: &[u8]) {
        assert_eq!(binding["bytes"].as_u64().unwrap(), raw.len() as u64);
        assert_eq!(
            binding["sha256"].as_str().unwrap(),
            format!("{:x}", Sha256::digest(raw))
        );
    }

    fn fixture() -> (Value, Value) {
        let profile_raw = include_bytes!("../../tests/fixtures/ministral3-rotary-v1/profile.json");
        assert_eq!(format!("{:x}", Sha256::digest(profile_raw)), PROFILE_SHA);
        let profile: Value = serde_json::from_slice(profile_raw).unwrap();
        let raw = include_bytes!("../../tests/fixtures/ministral3-rotary-v1/fixture.json");
        assert!(raw.len() <= 2_097_152);
        let fixture: Value = serde_json::from_slice(raw).unwrap();
        assert_eq!(fixture["format"], "toklet-ministral3-yarn-fixture-v1");
        assert_eq!(
            fixture["profile"]["file"],
            "tests/fixtures/ministral3-rotary-v1/profile.json"
        );
        check_binding(&fixture["profile"], profile_raw);
        let producers = fixture["producers"].as_object().unwrap();
        assert_eq!(producers.len(), 2);
        for (path, bytes) in [
            (
                "tools/reference/ministral3_rotary_fixture.py",
                include_bytes!("../../tools/reference/ministral3_rotary_fixture.py").as_slice(),
            ),
            (
                "tools/reference/test_ministral3_rotary_fixture.py",
                include_bytes!("../../tools/reference/test_ministral3_rotary_fixture.py")
                    .as_slice(),
            ),
        ] {
            check_binding(&producers[path], bytes);
        }
        assert_eq!(fixture["gates"], profile["gates"]);
        assert_eq!(
            fixture["external_source_pins"],
            profile["external_source_pins"]
        );
        assert_eq!(fixture["cases"].as_array().unwrap().len(), 11);
        (fixture, profile)
    }

    fn word32(value: &Value) -> u32 {
        u32::try_from(value.as_u64().unwrap()).unwrap()
    }

    fn half_words(value: &Value) -> Vec<f16> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|word| f16::from_bits(u16::try_from(word.as_u64().unwrap()).unwrap()))
            .collect()
    }

    fn floats(value: &Value) -> Vec<f64> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|word| f64::from(f32::from_bits(word32(word))))
            .collect()
    }

    fn halves(value: &Value) -> Vec<f64> {
        half_words(value)
            .iter()
            .map(|value| f64::from(value.to_f32()))
            .collect()
    }

    // Each exact case/array has its own independently derived absolute gate.
    // Native values never contribute to the spread, and arrays are never pooled.
    fn check_array(name: &str, native: &[f64], main: &[f64], wide: &[f64]) {
        assert!(!main.is_empty(), "{name}");
        assert_eq!(native.len(), main.len(), "{name}");
        assert_eq!(main.len(), wide.len(), "{name}");
        assert!(main.iter().chain(wide).all(|value| value.is_finite()));
        let spread = main
            .iter()
            .zip(wide)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f64, f64::max);
        let atol = 1e-4_f64.max(2. * spread);
        for (profile, reference) in [("main", main), ("wide", wide)] {
            for (index, (&got, &want)) in native.iter().zip(reference).enumerate() {
                let error = (got - want).abs();
                let gate = atol + 1e-3 * want.abs();
                assert!(
                    got.is_finite() && error <= gate,
                    "{name}/{profile}[{index}] got={got} want={want} error={error} gate={gate}"
                );
            }
        }
    }

    fn check_inputs(input: &[f16], pattern: &str, seed: usize, profile: &Value) {
        let edges = half_words(&profile["input_recipe"]["edge_words"]);
        for (index, &value) in input.iter().enumerate() {
            let expected = match pattern {
                "normal" => f16::from_f32((((index * 37 + seed) % 257) as i32 - 128) as f32 / 128.),
                "edges" => edges[(index + seed + 7 * (index / 64)) % 16],
                _ => panic!("unknown frozen input pattern"),
            };
            assert!(value.is_finite());
            assert_eq!(value.to_bits(), expected.to_bits(), "input[{index}]");
        }
    }

    #[test]
    fn frozen_eleven_cases_pass_each_independent_array_gate_and_exact_period_gain_words() {
        let (fixture, profile) = fixture();
        for (case, selected) in fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .zip(profile["cases"].as_array().unwrap())
        {
            for (key, value) in selected.as_object().unwrap() {
                assert_eq!(&case[key], value, "case metadata {key}");
            }
            let n = |key: &str| case[key].as_u64().unwrap() as usize;
            let (offset, tokens) = (n("offset"), n("tokens"));
            let table =
                Rotary::new(offset, tokens, Rotary::planned_bytes(tokens).unwrap()).unwrap();
            assert_eq!(table.base(), offset);
            assert_eq!(table.capacity(), tokens);
            let main_periods = &fixture["periods"]["main"];
            assert_eq!(main_periods.as_array().unwrap().len(), PAIRS);
            for (got, want) in table.periods().iter().zip(main_periods.as_array().unwrap()) {
                assert_eq!(got.to_bits(), word32(want));
            }
            let native_periods: Vec<_> = table.periods().iter().map(|&v| f64::from(v)).collect();
            check_array(
                "periods",
                &native_periods,
                &floats(main_periods),
                &floats(&fixture["periods"]["wide"]),
            );
            let main = &case["references"]["main"];
            let wide = &case["references"]["wide"];
            let coefficients: Vec<_> = (offset..offset + tokens)
                .flat_map(|position| table.coefficient_row(position).unwrap())
                .flat_map(|pair| pair.iter().map(|&value| f64::from(value)))
                .collect();
            assert_eq!(coefficients.len(), tokens * PAIRS * 2);
            check_array(
                "coefficients",
                &coefficients,
                &floats(&main["coefficients"]),
                &floats(&wide["coefficients"]),
            );
            let scales: Vec<_> = (offset..offset + tokens)
                .map(|position| table.query_scale(position).unwrap())
                .collect();
            let main_scales = half_words(&main["query_scale"]);
            assert_eq!(scales.len(), main_scales.len());
            for (got, want) in scales.iter().zip(&main_scales) {
                assert_eq!(got.to_bits(), want.to_bits());
            }
            check_array(
                "query_scale",
                &scales
                    .iter()
                    .map(|value| f64::from(value.to_f32()))
                    .collect::<Vec<_>>(),
                &halves(&main["query_scale"]),
                &halves(&wide["query_scale"]),
            );
            for (input_name, output_name, heads, query, seed) in [
                ("key_input", "keys", n("kv_heads"), false, n("k_seed")),
                (
                    "query_input",
                    "queries",
                    n("query_heads"),
                    true,
                    n("q_seed"),
                ),
            ] {
                let input = half_words(&case[input_name]);
                assert_eq!(input.len(), tokens * heads * HEAD_DIM);
                check_inputs(&input, case["pattern"].as_str().unwrap(), seed, &profile);
                let mut output = vec![f16::NAN; input.len()];
                if query {
                    table
                        .apply_queries(&input, &mut output, tokens, heads, offset)
                        .unwrap();
                } else {
                    table
                        .apply_keys(&input, &mut output, tokens, heads, offset)
                        .unwrap();
                }
                let native: Vec<_> = output
                    .iter()
                    .map(|value| f64::from(value.to_f32()))
                    .collect();
                check_array(
                    output_name,
                    &native,
                    &halves(&main[output_name]),
                    &halves(&wide[output_name]),
                );
            }
        }
    }

    fn table_hash(table: &Rotary) -> String {
        let mut hash = Sha256::new();
        hash.update(table.base().to_le_bytes());
        for period in table.periods() {
            hash.update(period.to_bits().to_le_bytes());
        }
        for pair in &table.coefficients {
            for value in pair {
                hash.update(value.to_bits().to_le_bytes());
            }
        }
        for scale in &table.query_scales {
            hash.update(scale.to_bits().to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }

    fn assert_same_half(a: &[f16], b: &[f16]) {
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn reset_replay_chunks_and_separately_based_tables_are_byte_exact() {
        let (base, tokens, heads) = (ORIGINAL_MAX_POSITIONS - 4, 17, 3);
        let table = Rotary::new(base, tokens, Rotary::planned_bytes(tokens).unwrap()).unwrap();
        let before = table_hash(&table);
        let stride = heads * HEAD_DIM;
        let input: Vec<_> = (0..tokens * stride)
            .map(|index| f16::from_f32(((index * 37 % 257) as i32 - 128) as f32 / 128.))
            .collect();
        for query in [false, true] {
            let apply = |table: &Rotary, input: &[f16], output: &mut [f16], tokens, offset| {
                if query {
                    table.apply_queries(input, output, tokens, heads, offset)
                } else {
                    table.apply_keys(input, output, tokens, heads, offset)
                }
            };
            let mut expected = vec![f16::NAN; input.len()];
            apply(&table, &input, &mut expected, tokens, base).unwrap();
            let mut output = vec![f16::NAN; input.len()];
            for _ in 0..3 {
                output.fill(f16::NAN); // Caller reset; the owner has no mutable session state.
                apply(&table, &input, &mut output, tokens, base).unwrap();
                assert_same_half(&output, &expected);
            }
            for lengths in [vec![1; tokens], vec![1, 3, 5, 8]] {
                output.fill(f16::NAN);
                let mut start = 0;
                for length in lengths {
                    let span = start * stride..(start + length) * stride;
                    apply(
                        &table,
                        &input[span.clone()],
                        &mut output[span],
                        length,
                        base + start,
                    )
                    .unwrap();
                    start += length;
                }
                assert_eq!(start, tokens);
                assert_same_half(&output, &expected);
            }
            for token in 0..tokens {
                let separate =
                    Rotary::new(base + token, 1, Rotary::planned_bytes(1).unwrap()).unwrap();
                assert_eq!(
                    separate.coefficient_row(base + token),
                    table.coefficient_row(base + token)
                );
                assert_eq!(
                    separate.query_scale(base + token),
                    table.query_scale(base + token)
                );
                let span = token * stride..(token + 1) * stride;
                apply(
                    &separate,
                    &input[span.clone()],
                    &mut output[span],
                    1,
                    base + token,
                )
                .unwrap();
            }
            assert_same_half(&output, &expected);
        }
        assert_eq!(table_hash(&table), before);
    }

    #[test]
    fn span_overflow_budget_and_entire_input_validation_precede_writes() {
        for capacity in [0, SOURCE_MAX_POSITIONS + 1, usize::MAX] {
            assert!(Rotary::planned_bytes(capacity).is_err());
            assert!(Rotary::new(0, capacity, usize::MAX).is_err());
        }
        for (base, capacity) in [
            (SOURCE_MAX_POSITIONS, 1),
            (SOURCE_MAX_POSITIONS - 1, 2),
            (usize::MAX, 1),
        ] {
            assert!(Rotary::new(base, capacity, usize::MAX).is_err());
        }
        assert!(storage_bytes(usize::MAX, 2).is_err());
        assert!(storage_bytes(isize::MAX as usize, 2).is_err());
        assert!(owner_bytes(usize::MAX, 1).is_err());
        let bytes = Rotary::planned_bytes(3).unwrap();
        assert!(Rotary::new(ORIGINAL_MAX_POSITIONS - 1, 3, bytes - 1).is_err());
        let base = ORIGINAL_MAX_POSITIONS - 1;
        let table = Rotary::new(base, 3, bytes).unwrap();
        let before = table_hash(&table);
        let good = vec![f16::ONE; 3 * HEAD_DIM + 1];
        let sentinel = f16::from_bits(0x7e01);
        for query in [false, true] {
            let apply = |input: &[f16], output: &mut [f16], tokens, heads, offset| {
                if query {
                    table.apply_queries(input, output, tokens, heads, offset)
                } else {
                    table.apply_keys(input, output, tokens, heads, offset)
                }
            };
            for (tokens, heads, offset, input_len, output_len) in [
                (0, 1, base, 0, 0),
                (1, 0, base, 0, 0),
                (1, 1, base - 1, 128, 128),
                (1, 1, base + 3, 128, 128),
                (3, 1, base + 1, 384, 384),
                (2, 1, usize::MAX, 256, 256),
                (1, usize::MAX, base, 128, 128),
                (3, usize::MAX / HEAD_DIM, base, 384, 384),
                (1, 1, base, 127, 128),
                (1, 1, base, 129, 128),
                (1, 1, base, 128, 127),
                (1, 1, base, 128, 129),
            ] {
                let mut output = vec![sentinel; output_len];
                assert!(apply(&good[..input_len], &mut output, tokens, heads, offset).is_err());
                assert!(
                    output
                        .iter()
                        .all(|value| value.to_bits() == sentinel.to_bits())
                );
            }
            for word in [0x7c00, 0xfc00, 0x7e01, 0xfe01] {
                for index in [0, 3 * HEAD_DIM - 1] {
                    let mut input = good[..3 * HEAD_DIM].to_vec();
                    input[index] = f16::from_bits(word);
                    let mut output = vec![sentinel; input.len()];
                    assert!(apply(&input, &mut output, 3, 1, base).is_err());
                    assert!(
                        output
                            .iter()
                            .all(|value| value.to_bits() == sentinel.to_bits())
                    );
                }
            }
        }
        assert_eq!(table_hash(&table), before);
    }

    #[test]
    fn high_absolute_span_counts_actual_storage_without_allocating_a_prefix() {
        let capacity = 8;
        let bytes = Rotary::planned_bytes(capacity).unwrap();
        assert_eq!(
            bytes,
            std::mem::size_of::<Rotary>() + capacity * PAIRS * 8 + capacity * 2
        );
        for base in [0, SOURCE_MAX_POSITIONS - capacity] {
            let table = Rotary::new(base, capacity, bytes).unwrap();
            assert_eq!(table.base(), base);
            assert_eq!(table.capacity(), capacity);
            assert_eq!(table.coefficients.len(), capacity * PAIRS);
            assert_eq!(table.query_scales.len(), capacity);
            assert_eq!(
                table.allocated_bytes().unwrap(),
                owner_bytes(table.coefficients.capacity(), table.query_scales.capacity()).unwrap()
            );
            assert!(table.allocated_bytes().unwrap() <= bytes);
            assert!(table.coefficient_row(base).is_some());
            assert!(table.coefficient_row(base + capacity - 1).is_some());
            for outside in [base + capacity, usize::MAX] {
                assert!(table.coefficient_row(outside).is_none());
                assert!(table.query_scale(outside).is_none());
            }
            if base > 0 {
                assert!(table.coefficient_row(base - 1).is_none());
                assert!(table.query_scale(base - 1).is_none());
            }
        }
    }

    #[test]
    fn source_zero_arithmetic_preserves_the_declared_half_store_signs() {
        let table = Rotary::new(0, 1, Rotary::planned_bytes(1).unwrap()).unwrap();
        let input = vec![f16::from_bits(0x8000); HEAD_DIM];
        for query in [false, true] {
            let mut output = vec![f16::NAN; HEAD_DIM];
            if query {
                table.apply_queries(&input, &mut output, 1, 1, 0).unwrap();
            } else {
                table.apply_keys(&input, &mut output, 1, 1, 0).unwrap();
            }
            assert!(output[..PAIRS].iter().all(|value| value.to_bits() == 0));
            assert!(
                output[PAIRS..]
                    .iter()
                    .all(|value| value.to_bits() == 0x8000)
            );
        }
    }

    #[test]
    fn rotation_and_second_half_store_overflow_discard_output_and_leave_owner_reusable() {
        let table = Rotary::new(0, 2, Rotary::planned_bytes(2).unwrap()).unwrap();
        let before = table_hash(&table);
        let mut input = vec![f16::ZERO; 2 * HEAD_DIM];
        input[HEAD_DIM] = f16::MAX;
        input[HEAD_DIM + PAIRS] = f16::MAX;
        for query in [false, true] {
            let mut output = vec![f16::NAN; input.len()];
            let result = if query {
                table.apply_queries(&input, &mut output, 2, 1, 0)
            } else {
                table.apply_keys(&input, &mut output, 2, 1, 0)
            };
            assert!(matches!(
                result,
                Err(Error::Kernel(cpu::Error::NonFinite {
                    tensor: "Ministral 3 rotary output",
                    ..
                }))
            ));
            assert!(output[..HEAD_DIM].iter().all(|value| value.to_bits() == 0));
            assert!(output[HEAD_DIM].is_nan());
        }
        input.fill(f16::ONE);
        let mut output = vec![f16::NAN; input.len()];
        table.apply_queries(&input, &mut output, 2, 1, 0).unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
        assert_eq!(table_hash(&table), before);

        let position = ORIGINAL_MAX_POSITIONS;
        let table = Rotary::new(position, 1, Rotary::planned_bytes(1).unwrap()).unwrap();
        let before = table_hash(&table);
        let mut input = vec![f16::ZERO; HEAD_DIM];
        input[PAIRS - 1] = f16::MAX;
        let mut keys = vec![f16::NAN; HEAD_DIM];
        table.apply_keys(&input, &mut keys, 1, 1, position).unwrap();
        let mut queries = vec![f16::NAN; HEAD_DIM];
        assert!(matches!(
            table.apply_queries(&input, &mut queries, 1, 1, position),
            Err(Error::Kernel(cpu::Error::NonFinite {
                tensor: "Ministral 3 scaled query output",
                ..
            }))
        ));
        input.fill(f16::ONE);
        table
            .apply_queries(&input, &mut queries, 1, 1, position)
            .unwrap();
        assert!(queries.iter().all(|value| value.is_finite()));
        assert_eq!(table_hash(&table), before);
    }
}
