"""Independent, bounded standard-library witnesses for one frozen Half YaRN profile.

This is prospective CPU primitive arithmetic, not BF16 model emulation or a
capture of MLX execution.  No source package is imported.  The main witness
rounds every FP32 operation; the Decimal80 witness uses its own trigonometric
series and rounds only at the storage boundaries declared in profile.json.
"""

import argparse
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct


PROFILE_FILE = "tests/fixtures/ministral3-rotary-v1/profile.json"
PROFILE_BYTES = 8522
PROFILE_SHA256 = "0f8d25bec77f2f6901e86830b7da22b75808fccb7800ce939f17f386206f4c9f"
PRODUCER_FILES = (
    "tools/reference/ministral3_rotary_fixture.py",
    "tools/reference/test_ministral3_rotary_fixture.py",
)
FORMAT = "toklet-ministral3-yarn-fixture-v1"
MAX_FIXTURE_BYTES = 2 * 1024 * 1024
MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_INTEGER = (1 << 64) - 1
PI = Decimal(
    "3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148"
)
GEOMETRY = dict(
    head_dim=128, theta=1000000.0, factor=16.0, beta_fast=32.0,
    beta_slow=1.0, mscale=1.0, mscale_all_dim=1.0,
    original_max_positions=16384, source_max_positions=262144,
    query_scaling_beta=0.1, traditional=False, truncate=True,
)
GATES = dict(
    atol="max(1e-4,2*max_abs(main-wide)) per exact case/array", rtol=0.001,
    against_both_profiles=True, pooling_forbidden=True,
    native_derived_enlargement_forbidden=True, chunk_and_replay_Half_bytes_exact=True,
    period_words_exact_main=True, query_gain_words_exact_main=True,
)


def integer(value, name, minimum=0, maximum=MAX_INTEGER):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"invalid {name}")
    return value


def product(*sizes):
    result = 1
    for size in sizes:
        integer(size, "size")
        if size and result > MAX_INTEGER // size:
            raise ValueError("size overflow")
        result *= size
    return result


def finite(value):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("expected a finite real number")
    return value


def f32(value):
    finite(value)
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("FP32 overflow") from error
    if not math.isfinite(result):
        raise ValueError("nonfinite FP32")
    return result


def f32_word(value):
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


def from_f32(word):
    integer(word, "FP32 word", maximum=0xFFFFFFFF)
    value = struct.unpack("<f", struct.pack("<I", word))[0]
    finite(value)
    return value


def half_word(value):
    finite(value)
    try:
        word = struct.unpack("<H", struct.pack("<e", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("Half overflow") from error
    from_half(word)
    return word


def from_half(word):
    integer(word, "Half word", maximum=0xFFFF)
    value = struct.unpack("<e", struct.pack("<H", word))[0]
    finite(value)
    return value


def _round_ratio(numerator, denominator):
    quotient, remainder = divmod(numerator, denominator)
    return quotient + (2 * remainder > denominator or
                       (2 * remainder == denominator and quotient % 2 == 1))


def decimal_word(value, exponent_bits, fraction_bits):
    """Exact rational RNE conversion: no Decimal -> FP64 -> target double round."""
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError("expected finite Decimal")
    sign = int(value.is_signed()) << (exponent_bits + fraction_bits)
    if value.is_zero():
        return sign
    numerator, denominator = value.copy_abs().as_integer_ratio()
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        exponent -= numerator < denominator << exponent
    else:
        exponent -= numerator << -exponent < denominator
    bias = (1 << (exponent_bits - 1)) - 1
    minimum, maximum = 1 - bias, bias
    exponent = max(exponent, minimum)
    shift = fraction_bits - exponent
    if shift >= 0:
        significand = _round_ratio(numerator << shift, denominator)
    else:
        significand = _round_ratio(numerator, denominator << -shift)
    if significand == 1 << (fraction_bits + 1):
        significand >>= 1
        exponent += 1
    if exponent > maximum:
        raise ValueError("Decimal storage overflow")
    if significand < 1 << fraction_bits:
        return sign | significand
    return sign | ((exponent + bias) << fraction_bits) | (significand - (1 << fraction_bits))


def d32(value):
    return decimal_word(value, 8, 23)


def dhalf(value):
    return decimal_word(value, 5, 10)


def widen(value):
    return Decimal.from_float(value)


def decimal_sincos(angle):
    """Independent Decimal80 range reduction and sine/cosine Taylor series."""
    if not isinstance(angle, Decimal) or not angle.is_finite():
        raise ValueError("invalid angle")
    with localcontext() as context:
        context.prec = 80
        tau = 2 * PI
        x = angle % tau
        if x > PI:
            x -= tau
        elif x < -PI:
            x += tau
        square = x * x
        sine, cosine = x, Decimal(1)
        sin_term, cos_term = x, Decimal(1)
        # |x| <= pi; 100 terms comfortably exceed 80 decimal digits.
        for index in range(1, 101):
            sin_term = -sin_term * square / ((2 * index) * (2 * index + 1))
            cos_term = -cos_term * square / ((2 * index - 1) * (2 * index))
            new_sine, new_cosine = sine + sin_term, cosine + cos_term
            if new_sine == sine and new_cosine == cosine:
                return +cosine, +sine
            sine, cosine = new_sine, new_cosine
    raise ValueError("trigonometric series did not converge")


def correction_range(geometry, wide=False):
    dim, original, theta = (geometry[name] for name in
                            ("head_dim", "original_max_positions", "theta"))
    if wide:
        with localcontext() as context:
            context.prec = 80
            def correction(beta):
                return (Decimal(dim) * (Decimal(original) /
                        (Decimal(str(beta)) * 2 * PI)).ln() /
                        (2 * Decimal(str(theta)).ln()))
            low = int(correction(geometry["beta_fast"]).to_integral_value(rounding=ROUND_FLOOR))
            high = int(correction(geometry["beta_slow"]).to_integral_value(rounding=ROUND_CEILING))
    else:
        def correction(beta):
            return dim * math.log(original / (beta * 2 * math.pi)) / (2 * math.log(theta))
        low, high = math.floor(correction(geometry["beta_fast"])), math.ceil(correction(geometry["beta_slow"]))
    return max(low, 0), min(high, dim - 1)


def periods_main(geometry):
    low, high = correction_range(geometry)
    upper = f32(high if high != low else high + 0.001)
    denominator = f32(upper - f32(low))
    result = []
    for pair in range(geometry["head_dim"] // 2):
        exponent = f32(f32(2 * pair) / f32(geometry["head_dim"]))
        extra = f32(math.pow(geometry["theta"], exponent))
        inter = f32(f32(geometry["factor"]) * extra)
        ramp = min(1.0, max(0.0, f32(f32(f32(pair) - f32(low)) / denominator)))
        mask = f32(1.0 - ramp)
        numerator = f32(inter * extra)
        divisor = f32(f32(inter * mask) + f32(extra * f32(1.0 - mask)))
        result.append(f32_word(f32(numerator / divisor)))
    return result


def periods_wide(geometry):
    with localcontext() as context:
        context.prec = 80
        low, high = correction_range(geometry, wide=True)
        upper = Decimal(high) if high != low else Decimal(high) + Decimal("0.001")
        result = []
        for pair in range(geometry["head_dim"] // 2):
            exponent = Decimal(2 * pair) / geometry["head_dim"]
            extra = (Decimal(str(geometry["theta"])).ln() * exponent).exp()
            inter = Decimal(str(geometry["factor"])) * extra
            ramp = min(Decimal(1), max(Decimal(0), (Decimal(pair) - low) / (upper - low)))
            mask = 1 - ramp
            result.append(d32((inter * extra) / (inter * mask + extra * (1 - mask))))
        return result


def gain_main(position, geometry):
    integer(position, "position", maximum=geometry["source_max_positions"] - 1)
    epoch = position // geometry["original_max_positions"]
    logarithm = f32(math.log(f32(1 + epoch)))
    return half_word(f32(1.0 + f32(f32(geometry["query_scaling_beta"]) * logarithm)))


def gain_wide(position, geometry):
    integer(position, "position", maximum=geometry["source_max_positions"] - 1)
    with localcontext() as context:
        context.prec = 80
        epoch = position // geometry["original_max_positions"]
        return dhalf(1 + Decimal(str(geometry["query_scaling_beta"])) * Decimal(1 + epoch).ln())


def make_input(case, heads, seed, recipe):
    count = product(case["tokens"], heads, 128)
    if case["pattern"] == "normal":
        return [half_word((((index * 37 + seed) % 257) - 128) / 128)
                for index in range(count)]
    edges = recipe["edge_words"]
    return [edges[(index + seed + 7 * (index // 64)) % 16] for index in range(count)]


def reference(case, geometry, period_words, queries, keys, wide=False):
    dim, pairs = geometry["head_dim"], geometry["head_dim"] // 2
    coefficients, scales = [], []
    with localcontext() as context:
        context.prec = 80
        for token in range(case["tokens"]):
            position = case["offset"] + token
            scales.append(gain_wide(position, geometry) if wide else gain_main(position, geometry))
            for period_word in period_words:
                period = from_f32(period_word)
                if wide:
                    cosine, sine = decimal_sincos(Decimal(position) / widen(period))
                    coefficients.extend((d32(cosine), d32(sine)))
                else:
                    angle = f32(f32(position) / period)
                    coefficients.extend((f32_word(math.cos(angle)), f32_word(math.sin(angle))))

        def rotate(values, heads, apply_gain):
            if len(values) != product(case["tokens"], heads, dim):
                raise ValueError("rotation input size mismatch")
            output = [0] * len(values)
            for token in range(case["tokens"]):
                gain = from_half(scales[token])
                for head in range(heads):
                    start = (token * heads + head) * dim
                    for pair in range(pairs):
                        a, b = (from_half(values[start + pair]),
                                from_half(values[start + pairs + pair]))
                        coefficient = (token * pairs + pair) * 2
                        cosine, sine = map(from_f32, coefficients[coefficient:coefficient + 2])
                        if wide:
                            a, b, cosine, sine = map(widen, (a, b, cosine, sine))
                            first, second = dhalf(a * cosine - b * sine), dhalf(b * cosine + a * sine)
                            if apply_gain:
                                first = dhalf(widen(from_half(first)) * widen(gain))
                                second = dhalf(widen(from_half(second)) * widen(gain))
                        else:
                            first = half_word(f32(f32(a * cosine) - f32(b * sine)))
                            second = half_word(f32(f32(b * cosine) + f32(a * sine)))
                            if apply_gain:
                                first = half_word(f32(from_half(first) * gain))
                                second = half_word(f32(from_half(second) * gain))
                        output[start + pair], output[start + pairs + pair] = first, second
            return output

        return dict(coefficients=coefficients, query_scale=scales,
                    keys=rotate(keys, case["kv_heads"], False),
                    queries=rotate(queries, case["query_heads"], True))


def _object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON key")
        result[name] = value
    return result


def parse_json(raw):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    try:
        return json.loads(raw, object_pairs_hook=_object, parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid JSON") from error


def read_regular(path, limit):
    path = Path(path)
    try:
        flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
        with os.fdopen(os.open(path, flags), "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
                raise ValueError("invalid input file or input size")
            raw = stream.read(limit + 1)
    except OSError as error:
        raise ValueError("cannot read input") from error
    if len(raw) > limit:
        raise ValueError("input grew beyond size bound")
    return raw


def digest(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def validate_profile(profile):
    if type(profile) is not dict or profile.get("geometry") != GEOMETRY:
        raise ValueError("unsupported geometry")
    # Equality alone accepts bool-as-int; validate the numeric types too.
    for name, value in profile["geometry"].items():
        if type(value) is not type(GEOMETRY[name]):
            raise ValueError("invalid geometry type")
        if name in ("traditional", "truncate"):
            if type(value) is not bool:
                raise ValueError("invalid geometry flag")
        else:
            finite(value)
    if profile.get("gates") != GATES:
        raise ValueError("changed gates")
    cases = profile.get("cases")
    if type(cases) is not list or not cases or len(cases) > 11:
        raise ValueError("invalid cases")
    names = set()
    for case in cases:
        if type(case) is not dict or set(case) != {
                "name", "offset", "tokens", "query_heads", "kv_heads", "pattern", "q_seed", "k_seed"}:
            raise ValueError("invalid case fields")
        if type(case["name"]) is not str or not case["name"] or case["name"] in names:
            raise ValueError("invalid case name")
        names.add(case["name"])
        integer(case["offset"], "offset", maximum=GEOMETRY["source_max_positions"] - 1)
        integer(case["tokens"], "tokens", 1, 8)
        integer(case["query_heads"], "query heads", 1, 32)
        integer(case["kv_heads"], "KV heads", 1, 8)
        integer(case["q_seed"], "query seed", maximum=255)
        integer(case["k_seed"], "key seed", maximum=255)
        if (case["offset"] + case["tokens"] > GEOMETRY["source_max_positions"] or
                case["query_heads"] % case["kv_heads"] or case["pattern"] not in ("normal", "edges")):
            raise ValueError("invalid case extent or pattern")
        product(case["tokens"], case["query_heads"], GEOMETRY["head_dim"])
    recipe = profile.get("input_recipe", {})
    if (recipe.get("normal") != "(((index*37+seed)%257)-128)/128, then Half RNE" or
            recipe.get("edges") != "edge_words[(index+seed+7*(index//64))%16]" or
            type(recipe.get("edge_words")) is not list or len(recipe["edge_words"]) != 16):
        raise ValueError("invalid input recipe")
    for word in recipe["edge_words"]:
        from_half(word)
    bounds = profile.get("bounds", {})
    if type(bounds.get("fixture_bytes")) is not int or bounds["fixture_bytes"] != MAX_FIXTURE_BYTES:
        raise ValueError("invalid fixture bound")
    config = profile.get("source_config", {})
    rope = config.get("rope_parameters", {})
    if (config.get("dtype") != "bfloat16" or config.get("model_type") != "ministral3" or
            rope.get("rope_type") != "yarn" or rope.get("type") != "yarn"):
        raise ValueError("unsupported source profile")


def load_profile(path):
    raw = read_regular(path, PROFILE_BYTES)
    if digest(raw) != dict(bytes=PROFILE_BYTES, sha256=PROFILE_SHA256):
        raise ValueError("frozen profile identity mismatch")
    profile = parse_json(raw)
    validate_profile(profile)
    return profile, dict(file=PROFILE_FILE, **digest(raw))


def producer_records(repository_root):
    return {name: digest(read_regular(Path(repository_root) / name, 128 * 1024))
            for name in PRODUCER_FILES}


def verify_sources(source_root, profile):
    source_root = Path(source_root)
    if not source_root.is_absolute() or not source_root.is_dir():
        raise ValueError("source root must be an absolute directory")
    captured = None
    for name, expected_hash in profile["external_source_pins"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe source path")
        raw = read_regular(source_root / path, MAX_INPUT_BYTES)
        if digest(raw)["sha256"] != expected_hash:
            raise ValueError(f"external source identity mismatch: {name}")
        if name.endswith("preflight-summary.json"):
            captured = parse_json(raw)
    try:
        selected = captured["models"]["ministral_modified"]
        if selected["text_config"] != profile["source_config"]:
            raise ValueError("selected modified text config differs")
        for key in ("repository", "revision", "config_sha256"):
            if selected[key] != profile["model_sources"]["modified"][key]:
                raise ValueError("selected modified model pin differs")
    except (KeyError, TypeError) as error:
        raise ValueError("missing selected source metadata") from error


def build_fixture(profile, profile_record, producers):
    validate_profile(profile)
    if profile_record != dict(file=PROFILE_FILE, bytes=PROFILE_BYTES, sha256=PROFILE_SHA256):
        raise ValueError("invalid profile binding")
    if set(producers) != set(PRODUCER_FILES):
        raise ValueError("invalid producer bindings")
    for record in producers.values():
        if (type(record) is not dict or set(record) != {"bytes", "sha256"} or
                type(record["sha256"]) is not str or len(record["sha256"]) != 64 or
                any(c not in "0123456789abcdef" for c in record["sha256"])):
            raise ValueError("invalid producer record")
        integer(record["bytes"], "producer bytes", 1, 128 * 1024)
    geometry = profile["geometry"]
    periods = dict(main=periods_main(geometry), wide=periods_wide(geometry))
    cases = []
    for case in profile["cases"]:
        queries = make_input(case, case["query_heads"], case["q_seed"], profile["input_recipe"])
        keys = make_input(case, case["kv_heads"], case["k_seed"], profile["input_recipe"])
        references = {name: reference(case, geometry, words, queries, keys, wide=name == "wide")
                      for name, words in periods.items()}
        cases.append(dict(case, query_input=queries, key_input=keys, references=references))
    return dict(
        format=FORMAT, profile=profile_record, producers=producers,
        external_source_pins=profile["external_source_pins"],
        reference_profiles=dict(main="main_simulated_FP32_with_std_math", wide="wide_Decimal80"),
        periods=periods, cases=cases, gates=profile["gates"],
        source_dtype="bfloat16", primitive_dtype="Half",
        scope=profile["scope"], source_admission_qualified=False,
        model_execution_qualified=False, metal_qualified=False,
        quality_qualified=False, performance_qualified=False, adoption_qualified=False,
    )


def validate_fixture(fixture, profile):
    if fixture.get("format") != FORMAT or fixture.get("gates") != GATES:
        raise ValueError("invalid fixture format or gates")
    periods = fixture.get("periods", {})
    if set(periods) != {"main", "wide"}:
        raise ValueError("invalid reference profiles")
    for words in periods.values():
        if type(words) is not list or len(words) != 64 or any(from_f32(word) <= 0 for word in words):
            raise ValueError("invalid periods")
    if len(fixture.get("cases", [])) != len(profile["cases"]):
        raise ValueError("case count mismatch")
    for case, frozen in zip(fixture["cases"], profile["cases"]):
        if {name: case.get(name) for name in frozen} != frozen:
            raise ValueError("case metadata mismatch")
        sizes = dict(query_input=product(frozen["tokens"], frozen["query_heads"], 128),
                     key_input=product(frozen["tokens"], frozen["kv_heads"], 128))
        for name, length in sizes.items():
            words = case.get(name)
            if type(words) is not list or len(words) != length:
                raise ValueError("input word count mismatch")
            for word in words:
                from_half(word)
        references = case.get("references", {})
        if set(references) != {"main", "wide"}:
            raise ValueError("missing reference")
        expected = dict(coefficients=(product(frozen["tokens"], 64, 2), from_f32),
                        query_scale=(frozen["tokens"], from_half),
                        keys=(sizes["key_input"], from_half), queries=(sizes["query_input"], from_half))
        for arrays in references.values():
            if set(arrays) != set(expected):
                raise ValueError("unexpected reference arrays")
            for name, (length, decoder) in expected.items():
                words = arrays[name]
                if type(words) is not list or len(words) != length:
                    raise ValueError("reference word count mismatch")
                for word in words:
                    decoder(word)


def serialize_fixture(fixture, profile):
    validate_fixture(fixture, profile)
    raw = (json.dumps(fixture, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(raw) > MAX_FIXTURE_BYTES:
        raise ValueError("fixture exceeds prospective byte bound")
    return raw


def check_output(path):
    path = Path(path)
    if not path.is_absolute() or path.name != "fixture.json":
        raise ValueError("fresh output must be an absolute fixture.json path")
    if any(part == ".." for part in path.parts):
        raise ValueError("output traversal")
    if path.exists() or path.is_symlink():
        raise ValueError("output already exists")
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError("symlink output ancestor")
    if not path.parent.is_dir():
        raise ValueError("output parent must exist")
    return path


def publish(raw, path):
    path = check_output(path)
    if type(raw) is not bytes or not raw or len(raw) > MAX_FIXTURE_BYTES:
        raise ValueError("invalid output bytes")
    try:
        with path.open("xb") as stream:
            stream.write(raw)
    except OSError as error:
        raise ValueError("fresh output publication failed") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fresh-output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = check_output(args.fresh_output)
    repository_root = Path(__file__).resolve().parents[2]
    profile, record = load_profile(repository_root / PROFILE_FILE)
    verify_sources(args.source_root, profile)
    fixture = build_fixture(profile, record, producer_records(repository_root))
    raw = serialize_fixture(fixture, profile)
    publish(raw, output)
    print(json.dumps(dict(output=str(output), **digest(raw), cases=len(fixture["cases"]),
                          profiles=list(fixture["reference_profiles"])), sort_keys=True))


if __name__ == "__main__":
    main()
