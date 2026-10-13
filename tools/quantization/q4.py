"""Toklet Q4 v1 packer, standard-library only; see docs/q4-v1.md."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

GROUP = 64
RECORD = 36
HEADER = 64
MAGIC = b"TOKQ4V1\0"
LIMIT = 6 * 1024**3
POLICY = "toklet-q4-affine-f32-source-half-metadata-rne-v1"


def layout(rows, columns, limit=LIMIT):
    if type(rows) is not int or type(columns) is not int or rows <= 0 or columns <= 0:
        raise ValueError("positive integer dimensions required")
    stride = (((columns + 63) // 64 * RECORD + 63) // 64) * 64
    payload = rows * stride
    if payload + HEADER > limit or payload > 2**63 - 1:
        raise ValueError("Q4 artifact exceeds size limit")
    return stride, payload


def half(value):
    try:
        result = struct.unpack("<e", struct.pack("<e", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("Q4 metadata exceeds finite Half range") from error
    if not math.isfinite(result):
        raise ValueError("Q4 metadata must be finite")
    return 0.0 if result == 0.0 else result


def header(rows, columns, limit=LIMIT):
    stride, payload = layout(rows, columns, limit)
    return struct.pack("<8sIIQQQQ16x", MAGIC, 1, GROUP, rows, columns, stride, payload)


def pack_row(values):
    # Even library callers use the same decoded FP32 source policy as the CLI.
    try:
        values = [struct.unpack("<f", struct.pack("<f", v))[0] for v in values]
    except (OverflowError, struct.error) as error:
        raise ValueError("Q4 source exceeds FP32 range") from error
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("Q4 row must contain finite values")
    result = bytearray()
    for start in range(0, len(values), GROUP):
        group = values[start:start + GROUP]
        minimum, maximum = min(group), max(group)
        bias = half(minimum)
        scale = half((maximum - minimum) / 15.0)
        # A nonconstant group must retain a positive scale, including subnormals.
        if maximum != minimum and scale == 0.0:
            scale = 2**-24
        codes = [0] * GROUP
        if scale:
            codes[:len(group)] = [min(15, max(0, round((v - bias) / scale))) for v in group]
        result.extend(struct.pack("<ee", scale, bias))
        result.extend(codes[i] | codes[i + 1] << 4 for i in range(0, GROUP, 2))
    result.extend(bytes((-len(result)) % 64))
    return bytes(result)


def pack_matrix(rows, columns, limit=LIMIT):
    result = bytearray(header(len(rows), columns, limit))
    for row in rows:
        if len(row) != columns:
            raise ValueError("Q4 row shape mismatch")
        result.extend(pack_row(row))
    return bytes(result)


def pack_file(source, destination, rows, columns, limit=LIMIT):
    """Stream one FP32LE row at a time; never overwrite an existing artifact."""
    metadata = header(rows, columns, limit)
    if source.stat().st_size != rows * columns * 4:
        raise ValueError("FP32 input length mismatch")
    # Bound conversion memory independently of compressed output admission.
    if columns * 4 > 16 * 1024**2:
        raise ValueError("FP32 row exceeds conversion limit")
    with source.open("rb") as src:
        owned = destination.open("xb")  # failed exclusive open never owns an existing file
        try:
            with owned as dst:
                source_hash = hashlib.sha256()
                payload_hash = hashlib.sha256(metadata)
                dst.write(metadata)
                for _ in range(rows):
                    data = src.read(columns * 4)
                    if len(data) != columns * 4:
                        raise ValueError("FP32 source shortened during packing")
                    source_hash.update(data)
                    packed = pack_row(struct.unpack(f"<{columns}f", data))
                    payload_hash.update(packed)
                    dst.write(packed)
                if src.read(1):
                    raise ValueError("FP32 source grew during packing")
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    return {"format": "toklet-q4-matrix-v1", "quantizer": POLICY,
            "source_dtype": "f32le", "source_sha256": source_hash.hexdigest(),
            "packed_sha256": payload_hash.hexdigest(), "rows": rows,
            "columns": columns, "packed_bytes": HEADER + layout(rows, columns, limit)[1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--columns", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(pack_file(args.source, args.destination, args.rows, args.columns), sort_keys=True))


if __name__ == "__main__":
    main()
