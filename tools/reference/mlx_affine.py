"""Explicit, pinned MLX affine storage oracle; --check uses only the stdlib.

The code patterns below are literal low-bit-first encodings of all unsigned
codes in order. No production Toklet parser/packer is imported. Export runs
MLX CPU and GPU dequantization with metadata promoted to FP32, separately saves
native-dtype CPU/GPU outputs, and freezes an ordered FP64 matvec reference.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import struct

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'tests/fixtures/mlx-affine-v1'
PATTERNS = {
    2: bytes.fromhex('e4'),
    3: bytes.fromhex('88c6fa'),
    4: bytes.fromhex('1032547698badcfe'),
    5: bytes.fromhex('2088418a3928a9c59a7b30ca49abbd38ebcdbbff'),
    6: bytes.fromhex('40200c44611c48a22c4ce33c50244d54655d58a66d5ce77d60288e64699e68aaae6cebbe702ccf746ddf78aeef7cefff'),
    8: bytes(range(256)),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(folder=FIXTURE):
    manifest = json.loads((folder / 'manifest.json').read_text())
    raw = (folder / 'arrays.bin').read_bytes()
    if manifest['format'] != 'toklet-mlx-affine-reference-v1' or len(raw) != manifest['bytes'] or digest(raw) != manifest['sha256']:
        raise ValueError('Affine fixture identity differs')
    if len(manifest['cases']) != 54:
        raise ValueError('Complete affine parameter inventory required')
    for case in manifest['cases']:
        for entry in case['arrays'].values():
            start, size = entry['offset'], entry['bytes']
            data = raw[start:start + size]
            if len(data) != size or digest(data) != entry['sha256']:
                raise ValueError('Affine array identity differs')
    return manifest


def export(folder):
    if folder.exists():
        raise FileExistsError('Never replace an existing affine fixture')
    if version('mlx') != '0.32.3':
        raise ValueError('Requires pinned MLX 0.32.3')
    import mlx.core as mx

    folder.mkdir(parents=True)
    raw = bytearray()
    cases = []

    def put(data):
        entry = dict(offset=len(raw), bytes=len(data), sha256=digest(data))
        raw.extend(data)
        return entry

    def array_bytes(array):
        mx.eval(array)
        return bytes(array.view(mx.uint8).reshape(-1).tolist())

    dtypes = {'F16': mx.float16, 'BF16': mx.bfloat16, 'F32': mx.float32}
    for bits, pattern in PATTERNS.items():
        row = pattern * (256 * bits // 8 // len(pattern))
        packed = row + bytes(v ^ 255 for v in row)
        words = struct.unpack('<' + 'I' * (len(packed) // 4), packed)
        for group in (32, 64, 128):
            count = 2 * 256 // group
            for dtype_name, dtype in dtypes.items():
                # Half-exact scales make FP32 CPU/GPU affine arithmetic exact
                # before the final sum rounding, including the 8-bit codes.
                scale_source = [-.25, .125, 0.0, .333251953125, 2**-24, -2**-10]
                bias_source = [1., -2., .0001, -.5, 2., .75, -1.]
                with mx.stream(mx.cpu):
                    scales = mx.array([scale_source[i % len(scale_source)] for i in range(count)], dtype=dtype).reshape(2, -1)
                    biases = mx.array([bias_source[i % len(bias_source)] for i in range(count)], dtype=dtype).reshape(2, -1)
                    weight = mx.array(words, dtype=mx.uint32).reshape(2, -1)
                    cpu = mx.dequantize(weight, scales.astype(mx.float32), biases.astype(mx.float32), group_size=group, bits=bits)
                    native_cpu = mx.dequantize(weight, scales, biases, group_size=group, bits=bits)
                    cpu_bytes = array_bytes(cpu)
                    native_cpu_bytes = array_bytes(native_cpu)
                    scale_bytes, bias_bytes = array_bytes(scales), array_bytes(biases)
                with mx.stream(mx.gpu):
                    gpu = mx.dequantize(weight, scales.astype(mx.float32), biases.astype(mx.float32), group_size=group, bits=bits)
                    native_gpu = mx.dequantize(weight, scales, biases, group_size=group, bits=bits)
                    gpu_bytes = array_bytes(gpu)
                    native_gpu_bytes = array_bytes(native_gpu)
                if cpu_bytes != gpu_bytes:
                    raise ValueError('FP32 CPU/GPU reconstruction differs for fixed exact-product inputs')
                values = struct.unpack('<512f', cpu_bytes)
                scale32 = scales.astype(mx.float32).reshape(-1).tolist()
                bias32 = biases.astype(mx.float32).reshape(-1).tolist()
                # Independently verify each literal packed-code sequence against
                # its semantic code values, not an alternate bit-unpacking loop.
                for r in range(2):
                    for c in range(256):
                        code = c % (1 << bits)
                        if r: code = ((1 << bits) - 1) - code
                        index = (r * 256 + c) // group
                        expected = struct.pack('<f', float(scale32[index]) * code + float(bias32[index]))
                        if expected != cpu_bytes[(r * 256 + c) * 4:(r * 256 + c + 1) * 4]:
                            raise ValueError('Literal code/metadata oracle differs')
                inputs = [(i % 7 - 3) / 8 for i in range(256)]
                result = []
                for r in range(2):
                    total = 0.0
                    for c in range(256):
                        total += float(values[r * 256 + c]) * inputs[c]
                    result.append(total)
                arrays = dict(weight=put(packed), scales=put(scale_bytes), biases=put(bias_bytes),
                              reconstructed_f32=put(cpu_bytes), native_cpu=put(native_cpu_bytes), native_gpu=put(native_gpu_bytes),
                              input_f32=put(struct.pack('<256f', *inputs)), matvec_f32=put(struct.pack('<2f', *result)))
                cases.append(dict(bits=bits, group_size=group, metadata_dtype=dtype_name, rows=2, columns=256, arrays=arrays,
                                  native_dtype_cpu_gpu_equal=native_cpu_bytes == native_gpu_bytes))
    manifest = dict(format='toklet-mlx-affine-reference-v1', mlx_version=version('mlx'),
                    python_version=platform.python_version(), machine=platform.machine(), os=platform.mac_ver()[0],
                    exporter_sha256=digest(Path(__file__).read_bytes()),
                    oracle_binary=dict(name=Path(mx.__file__).name, bytes=Path(mx.__file__).stat().st_size, sha256=digest(Path(mx.__file__).read_bytes())),
                    backends=['CPU', 'Metal GPU'], policy='Promote original metadata to FP32; compare CPU/GPU dequantization bits for exact-product inputs. Native-dtype CPU/GPU outputs are separate rounding witnesses, not the Rust FP32 reconstruction contract. Matvec: reconstructed FP32 weights, ordered FP64 products/accumulation, final FP32.',
                    cases=cases, bytes=len(raw), sha256=digest(raw))
    (folder / 'arrays.bin').write_bytes(raw)
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify(folder)
    print(json.dumps(dict(cases=len(cases), bytes=len(raw), sha256=digest(raw))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--export', type=Path)
    mode.add_argument('--check', type=Path)
    args = parser.parse_args()
    if args.export:
        export(args.export)
    else:
        result = verify(args.check)
        print(json.dumps(dict(verified=True, cases=len(result['cases']), bytes=result['bytes'])))
