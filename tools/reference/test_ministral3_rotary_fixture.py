"""Portable controls; no model download, external numerical runtime or full fixture."""

import copy
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ministral3_rotary_fixture as oracle


ROOT = Path(__file__).resolve().parents[2]


def profile():
    return oracle.load_profile(ROOT / oracle.PROFILE_FILE)[0]


def tiny_profile():
    result = profile()
    result["cases"] = [dict(result["cases"][0], query_heads=1)]
    return result


def mock_reference(case, geometry, period_words, queries, keys, wide=False):
    del geometry, period_words, wide
    return dict(coefficients=[oracle.f32_word(1.0), 0] * (case["tokens"] * 64),
                query_scale=[0x3C00] * case["tokens"],
                keys=[0] * len(keys), queries=[0] * len(queries))


def mock_fixture(selected):
    binding = dict(file=oracle.PROFILE_FILE, bytes=oracle.PROFILE_BYTES, sha256=oracle.PROFILE_SHA256)
    with patch.object(oracle, "periods_main", return_value=[oracle.f32_word(1.0)] * 64), \
            patch.object(oracle, "periods_wide", return_value=[oracle.f32_word(1.0)] * 64), \
            patch.object(oracle, "reference", side_effect=mock_reference):
        return oracle.build_fixture(selected, binding, oracle.producer_records(ROOT))


class BinaryStorage(unittest.TestCase):
    def test_fp32_ties_signed_zero_and_extreme_finite(self):
        self.assertEqual(oracle.f32_word(1 + 2 ** -24), 0x3F800000)
        self.assertEqual(oracle.f32_word(1 + 3 * 2 ** -24), 0x3F800002)
        self.assertEqual(oracle.f32_word(-0.0), 0x80000000)
        self.assertEqual(oracle.f32_word(2 ** -149), 1)
        self.assertEqual(oracle.f32_word(2 ** -150), 0)
        self.assertEqual(oracle.f32_word(oracle.from_f32(0x7F7FFFFF)), 0x7F7FFFFF)

    def test_half_ties_subnormals_and_signed_zero(self):
        self.assertEqual(oracle.half_word(1 + 2 ** -11), 0x3C00)
        self.assertEqual(oracle.half_word(1 + 3 * 2 ** -11), 0x3C02)
        self.assertEqual(oracle.half_word(-0.0), 0x8000)
        self.assertEqual(oracle.half_word(2 ** -24), 1)
        self.assertEqual(oracle.half_word(-2 ** -25), 0x8000)
        self.assertEqual(oracle.half_word(65504.0), 0x7BFF)

    def test_decimal_storage_has_no_fp64_midpoint_substitution(self):
        with localcontext() as context:
            context.prec = 80
            half_tie = Decimal(1) + Decimal(2) ** -11
            f32_tie = Decimal(1) + Decimal(2) ** -24
            tail = Decimal("1e-70")
            self.assertEqual(oracle.dhalf(half_tie), 0x3C00)
            self.assertEqual(oracle.dhalf(half_tie + tail), 0x3C01)
            self.assertEqual(oracle.dhalf(half_tie - tail), 0x3C00)
            self.assertEqual(oracle.d32(f32_tie), 0x3F800000)
            self.assertEqual(oracle.d32(f32_tie + tail), 0x3F800001)
            self.assertEqual(oracle.d32(f32_tie - tail), 0x3F800000)
            self.assertEqual(oracle.dhalf(-(Decimal(2) ** -25)), 0x8000)
            self.assertEqual(oracle.dhalf(Decimal(2) ** -24), 1)
            self.assertEqual(oracle.d32(Decimal(2) ** -149), 1)
        self.assertEqual(oracle.d32(Decimal("-0")), 0x80000000)
        self.assertEqual(oracle.dhalf(Decimal("-0")), 0x8000)

    def test_exact_widen_storage_roundtrip(self):
        for word in (0, 0x8000, 1, 0x8001, 0x3FF, 0x400, 0x3555, 0xB555, 0x7BFF):
            self.assertEqual(oracle.dhalf(oracle.widen(oracle.from_half(word))), word)
        for word in (0, 0x80000000, 1, 0x80000001, 0x7FFFFF, 0x800000, 0x3F800000, 0x7F7FFFFF):
            self.assertEqual(oracle.d32(oracle.widen(oracle.from_f32(word))), word)

    def test_nonfinite_bad_type_and_overflow_reject(self):
        for value in (True, "1", float("nan"), float("inf"), -float("inf"), 10 ** 1000):
            for function in (oracle.f32, oracle.half_word):
                with self.subTest(value=str(value)[:30], function=function.__name__), self.assertRaises(ValueError):
                    function(value)
        for word in (True, -1, 0x10000, 0x7C00, 0xFC00, 0x7E00):
            with self.subTest(word=word), self.assertRaises(ValueError):
                oracle.from_half(word)
        for word in (True, -1, 0x100000000, 0x7F800000, 0x7FC00000):
            with self.subTest(word=word), self.assertRaises(ValueError):
                oracle.from_f32(word)
        for value in (Decimal("NaN"), Decimal("Infinity"), Decimal("1e1000")):
            for function in (oracle.d32, oracle.dhalf):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    function(value)
        with self.assertRaises(ValueError):
            oracle.f32(1e300)
        with self.assertRaises(ValueError):
            oracle.half_word(65520.0)


class IndependentMath(unittest.TestCase):
    def test_decimal_trig_axes_and_period_reduction(self):
        self.assertEqual(oracle.decimal_sincos(Decimal(0)), (Decimal(1), Decimal(0)))
        with localcontext() as context:
            context.prec = 80
            for angle, expected in ((oracle.PI / 2, (0, 1)),
                                    (oracle.PI, (-1, 0)),
                                    (-oracle.PI / 2, (0, -1)),
                                    (oracle.PI * 20000, (1, 0))):
                cosine, sine = oracle.decimal_sincos(angle)
                self.assertLess(abs(cosine - expected[0]), Decimal("1e-70"))
                self.assertLess(abs(sine - expected[1]), Decimal("1e-70"))

    def test_decimal_trig_unit_circle_and_std_math_sanity(self):
        with localcontext() as context:
            context.prec = 80
            for value in ("0.125", "-2.75", "262143"):
                cosine, sine = oracle.decimal_sincos(Decimal(value))
                self.assertLess(abs(cosine * cosine + sine * sine - 1), Decimal("1e-77"))
                self.assertAlmostEqual(float(cosine), math.cos(float(value)), places=14)
                self.assertAlmostEqual(float(sine), math.sin(float(value)), places=14)
        with self.assertRaises(ValueError):
            oracle.decimal_sincos(Decimal("NaN"))

    def test_correction_range_and_source_clamps(self):
        geometry = dict(oracle.GEOMETRY)
        self.assertEqual(oracle.correction_range(geometry), (20, 37))
        self.assertEqual(oracle.correction_range(geometry, wide=True), (20, 37))
        geometry.update(beta_fast=1e30, beta_slow=1e-30)
        self.assertEqual(oracle.correction_range(geometry), (0, 127))
        self.assertEqual(oracle.correction_range(geometry, wide=True), (0, 127))

    def test_gain_is_half_and_changes_at_integer_epoch_boundaries(self):
        geometry = oracle.GEOMETRY
        expected = ((0, 0x3C00), (16383, 0x3C00), (16384, 0x3C47),
                    (32767, 0x3C47), (32768, 0x3C70), (262143, 0x3D1C))
        for position, word in expected:
            self.assertEqual(oracle.gain_main(position, geometry), word)
            self.assertEqual(oracle.gain_wide(position, geometry), word)
        for position in (-1, True, 262144):
            with self.assertRaises(ValueError):
                oracle.gain_main(position, geometry)

    def test_tiny_zero_position_rotation_and_query_second_store(self):
        case = tiny_profile()["cases"][0]
        values = [oracle.half_word(0.25)] * 128
        periods = [oracle.f32_word(1.0)] * 64
        for wide in (False, True):
            result = oracle.reference(case, oracle.GEOMETRY, periods, values, values, wide=wide)
            self.assertEqual(result["coefficients"], [oracle.f32_word(1.0), 0] * 64)
            self.assertEqual(result["query_scale"], [0x3C00])
            self.assertEqual(result["queries"], values)
            self.assertEqual(result["keys"], values)
        # Patch only coefficients via a synthetic very large period.  At an
        # epoch boundary rotation stores Half first, then query gain stores Half.
        case = dict(case, offset=16384)
        periods = [oracle.f32_word(2 ** 100)] * 64
        result = oracle.reference(case, oracle.GEOMETRY, periods, values, values)
        expected = oracle.half_word(oracle.f32(0.25 * oracle.from_half(result["query_scale"][0])))
        self.assertEqual(result["keys"], values)
        self.assertEqual(result["queries"], [expected] * 128)

    def test_input_recipe_is_deterministic_including_signed_subnormals(self):
        selected = profile()
        case = selected["cases"][0]
        words = oracle.make_input(case, 3, 17, selected["input_recipe"])
        self.assertEqual(len(words), 384)
        self.assertEqual(words, oracle.make_input(case, 3, 17, selected["input_recipe"]))
        self.assertEqual(words[0], oracle.half_word((17 - 128) / 128))
        edges = oracle.make_input(dict(case, pattern="edges"), 1, 0, selected["input_recipe"])
        self.assertEqual(edges[:16], selected["input_recipe"]["edge_words"])
        self.assertEqual(set(edges), set(selected["input_recipe"]["edge_words"]))


class ProfileAndProvenance(unittest.TestCase):
    def test_frozen_profile_and_tracked_producer_bindings(self):
        selected, record = oracle.load_profile(ROOT / oracle.PROFILE_FILE)
        self.assertEqual(record, dict(file=oracle.PROFILE_FILE, bytes=8522, sha256=oracle.PROFILE_SHA256))
        self.assertEqual(len(selected["cases"]), 11)
        records = oracle.producer_records(ROOT)
        self.assertEqual(set(records), set(oracle.PRODUCER_FILES))
        for name, record in records.items():
            raw = (ROOT / name).read_bytes()
            self.assertEqual(record, dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))

    def test_changed_truncated_or_oversized_profile_rejects(self):
        raw = (ROOT / oracle.PROFILE_FILE).read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            for changed in (raw[:-1], raw + b" ", raw.replace(b"1000000.0", b"1000001.0")):
                path.write_bytes(changed)
                with self.assertRaises(ValueError):
                    oracle.load_profile(path)

    def test_profile_type_extent_recipe_and_gate_controls(self):
        mutations = (
            lambda p: p["geometry"].update(original_max_positions=16384.0),
            lambda p: p["geometry"].update(theta=float("nan")),
            lambda p: p["cases"][0].update(tokens=True),
            lambda p: p["cases"][0].update(offset=262143, tokens=2),
            lambda p: p["cases"][0].update(query_heads=3, kv_heads=2),
            lambda p: p["cases"][0].update(pattern="native"),
            lambda p: p["input_recipe"].update(edge_words=[0x7C00] * 16),
            lambda p: p["gates"].update(rtol=0.01),
            lambda p: p["bounds"].update(fixture_bytes=True),
            lambda p: p["source_config"]["rope_parameters"].update(type="default"),
        )
        for mutate in mutations:
            selected = profile()
            mutate(selected)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                oracle.validate_profile(selected)

    def test_integer_size_overflow_and_bool_controls(self):
        self.assertEqual(oracle.product(8, 32, 128), 32768)
        for sizes in ((1 << 63, 3), (True, 1), (-1, 1), (1 << 64, 0)):
            with self.assertRaises(ValueError):
                oracle.product(*sizes)

    def test_json_duplicate_and_nonfinite_controls(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'\xff', b'{'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                oracle.parse_json(raw)

    def test_source_pins_and_modified_config_validate_without_external_runtime(self):
        selected = profile()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = dict(models=dict(ministral_modified=dict(
                selected["model_sources"]["modified"], text_config=selected["source_config"])))
            raw = json.dumps(metadata).encode()
            (root / "preflight-summary.json").write_bytes(raw)
            (root / "formula.py").write_bytes(b"source bytes; never imported\n")
            selected["external_source_pins"] = {
                "preflight-summary.json": hashlib.sha256(raw).hexdigest(),
                "formula.py": hashlib.sha256((root / "formula.py").read_bytes()).hexdigest(),
            }
            oracle.verify_sources(root, selected)
            (root / "formula.py").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                oracle.verify_sources(root, selected)
            selected["external_source_pins"].pop("formula.py")
            selected["source_config"] = dict(selected["source_config"], dtype="float16")
            with self.assertRaises(ValueError):
                oracle.verify_sources(root, selected)
            selected["external_source_pins"] = {"../escape": "0" * 64}
            with self.assertRaises(ValueError):
                oracle.verify_sources(root, selected)
        with self.assertRaises(ValueError):
            oracle.verify_sources(Path("relative"), selected)


class FixtureAndOutput(unittest.TestCase):
    def test_mock_schema_is_deterministic_and_explicitly_unqualified(self):
        selected = tiny_profile()
        first = mock_fixture(selected)
        raw = oracle.serialize_fixture(first, selected)
        self.assertEqual(raw, oracle.serialize_fixture(mock_fixture(selected), selected))
        self.assertLess(len(raw), 16000)
        self.assertEqual(first["reference_profiles"], dict(
            main="main_simulated_FP32_with_std_math", wide="wide_Decimal80"))
        self.assertEqual(first["primitive_dtype"], "Half")
        self.assertEqual(first["source_dtype"], "bfloat16")
        for name in ("source_admission", "model_execution", "metal", "quality", "performance", "adoption"):
            self.assertIs(first[name + "_qualified"], False)

    def test_mock_schema_rejects_nonfinite_words_and_wrong_sizes(self):
        selected = tiny_profile()
        original = mock_fixture(selected)
        mutations = (
            lambda f: f["periods"]["main"].__setitem__(0, 0x7FC00000),
            lambda f: f["periods"]["wide"].pop(),
            lambda f: f["cases"][0]["query_input"].__setitem__(0, True),
            lambda f: f["cases"][0]["references"]["wide"]["keys"].__setitem__(0, 0x7C00),
            lambda f: f["cases"][0]["references"]["main"]["queries"].pop(),
            lambda f: f["cases"][0].update(offset=1),
            lambda f: f["cases"][0]["references"].pop("wide"),
        )
        for mutate in mutations:
            fixture = copy.deepcopy(original)
            mutate(fixture)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                oracle.serialize_fixture(fixture, selected)

    def test_mock_fixture_bound_cannot_be_enlarged(self):
        selected = tiny_profile()
        fixture = mock_fixture(selected)
        with patch.object(oracle, "MAX_FIXTURE_BYTES", 32), self.assertRaises(ValueError):
            oracle.serialize_fixture(fixture, selected)

    def test_build_requires_exact_profile_and_two_source_bindings(self):
        selected = tiny_profile()
        records = oracle.producer_records(ROOT)
        binding = dict(file=oracle.PROFILE_FILE, bytes=8522, sha256=oracle.PROFILE_SHA256)
        with self.assertRaises(ValueError):
            oracle.build_fixture(selected, dict(binding, sha256="0" * 64), records)
        with self.assertRaises(ValueError):
            oracle.build_fixture(selected, binding, {})
        changed = copy.deepcopy(records)
        changed[oracle.PRODUCER_FILES[0]]["bytes"] = True
        with self.assertRaises(ValueError):
            oracle.build_fixture(selected, binding, changed)

    def test_fresh_output_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "fixture.json"
            oracle.publish(b"{}\n", path)
            self.assertEqual(path.read_bytes(), b"{}\n")
            with self.assertRaises(ValueError):
                oracle.publish(b"changed", path)
            self.assertEqual(path.read_bytes(), b"{}\n")
            for invalid in (Path("fixture.json"), path.parent / "other.json",
                            path.parent / "missing" / "fixture.json",
                            path.parent / ".." / "fixture.json"):
                with self.subTest(path=invalid), self.assertRaises(ValueError):
                    oracle.check_output(invalid)

    def test_symlink_input_and_output_ancestors_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            outside = root / "outside"
            outside.mkdir()
            alias = root / "alias"
            alias.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                oracle.publish(b"{}\n", alias / "fixture.json")
            original = root / "source"
            original.write_bytes(b"x")
            link = root / "link"
            link.symlink_to(original)
            with self.assertRaises(ValueError):
                oracle.read_regular(link, 5)
            dangling = root / "fixture.json"
            dangling.symlink_to(root / "nonexistent")
            with self.assertRaises(ValueError):
                oracle.check_output(dangling)
            self.assertEqual(list(outside.iterdir()), [])

    def test_bounded_reads_and_output_bytes_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"
            source.write_bytes(b"1234")
            self.assertEqual(oracle.read_regular(source, 4), b"1234")
            with self.assertRaises(ValueError):
                oracle.read_regular(source, 3)
            with self.assertRaises(ValueError):
                oracle.read_regular(root, 3)
            for raw in (b"", "{}", b"x" * (oracle.MAX_FIXTURE_BYTES + 1)):
                with self.assertRaises(ValueError):
                    oracle.publish(raw, root / "fixture.json")
            self.assertFalse((root / "fixture.json").exists())


if __name__ == "__main__":
    unittest.main()
