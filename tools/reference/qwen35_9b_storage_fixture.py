"""Independent packed byte/row fixture; no Toklet, model, NumPy or GPU import."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

EMBED = 'language_model.model.embed_tokens'
HEAD = 'language_model.lm_head'
NORM = 'language_model.model.norm.weight'
CONV = 'language_model.model.layers.0.linear_attn.conv1d.weight'


def fixture(destination):
    destination.mkdir()
    body = bytearray(); physical = {}; expected = {}

    def append(name, dtype, shape, raw):
        start = len(body); body.extend(raw)
        physical[name] = dict(dtype=dtype, shape=shape, data_offsets=[start, len(body)],
                              sha256=hashlib.sha256(raw).hexdigest())

    def words(values):
        return struct.pack('<'+'H'*len(values), *values)

    def codes(rows, reverse=False):
        # Original integer cells determine both source packing and expected rows.
        cells = [[15-column % 16 if reverse else column % 16 for column in range(64)]
                 for _ in range(rows)]
        packed = bytearray()
        for row in cells:
            for first in range(0, 64, 8):
                word = sum(row[first+lane] << (4*lane) for lane in range(8))
                packed.extend(struct.pack('<I', word))
        return bytes(packed), cells

    embedding, e = codes(2); head, h = codes(3, True)
    append(EMBED+'.weight', 'U32', [2, 8], embedding)
    append(CONV, 'BF16', [2, 4, 1], words([0x3f80,0xc000,0x3f00,0x8000,0,0xbf80,0x4000,0x3e80]))
    append(HEAD+'.weight', 'U32', [3, 8], head)
    append(EMBED+'.scales', 'BF16', [2, 1], words([0x3f80,0xbf00]))
    append(NORM, 'BF16', [3], words([0x3f80,0xbf80,0x8000]))
    append(HEAD+'.biases', 'BF16', [3, 1], words([0x3f80,0,0xc040]))
    append(EMBED+'.biases', 'BF16', [2, 1], words([0xc000,0x4040]))
    append(HEAD+'.scales', 'BF16', [3, 1], words([0x3e80,0xbf80,0x4000]))
    expected[EMBED+'.weight'] = [[cell-2.0 for cell in e[0]], [3.0-.5*cell for cell in e[1]]]
    expected[HEAD+'.weight'] = [[1.0+.25*cell for cell in h[0]], [-1.0*cell+0.0 for cell in h[1]],
                                [2.0*cell-3.0 for cell in h[2]]]
    prefix = bytes([0xab])*9; stream = prefix+body
    vector_bits = {NORM:[0x3f800000,0xbf800000,0x80000000],
                   CONV:[0x3f800000,0xc0000000,0x3f000000,0x80000000,0,0xbf800000,0x40000000,0x3e800000]}
    record = dict(format='toklet-qwen35-9b-storage-fixture-v1',prefix_bytes=9,
                  source_bytes=len(stream),source_sha256=hashlib.sha256(stream).hexdigest(),
                  payload_bytes=len(body), physical=physical, expected_rows=expected,
                  expected_small_fp32_bits=vector_bits, retained_tensor_bytes=224,
                  requested_load_bytes=64*1024**2+65536+16+224,
                  fixture_only=True, public_source_eligible=False, engine_executed=False)
    (destination/'stream.bin').write_bytes(stream)
    (destination/'inventory.json').write_text(json.dumps(record,indent=2)+'\n')
    (destination/'README.md').write_text('''# Independent 9B storage fixture

This 211-byte artificial stream has a nine-byte unaligned prefix, two distinct
original affine4/BF16 matrices (embedding2x64, head3x64), a three-value vector
and a two-channel/four-tap convolution. Physical order differs from logical
recipe order. The Python generator packs original integer cells and records
independent expected rows and exact FP32 small-value bits, including signed zero.

Only private test constructors can supply its deliberately tiny cold recipe.
It is rejected by every pinned public loader and grants no real-model execution,
numerical, memory-fit, context, quality or managed-route qualification.

[Generator](../../../tools/reference/qwen35_9b_storage_fixture.py).
''')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    result=fixture(args.destination)
    print(json.dumps({k:result[k] for k in ['source_bytes','payload_bytes','source_sha256']}))
