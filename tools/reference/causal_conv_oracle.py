"""Independent tiny unmasked causal convolution and post-convolution SiLU math.

No Toklet or MLX arithmetic is imported. A separate explicit reference driver
must establish the pinned upstream operator/layout correspondence before target
implementation. Empty input and kernel-one history are mathematical policies.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import causal_conv_fixture as fixture
import reference_runtime as runtime


def run(inputs, weights, initial):
    import numpy as np
    inputs, weights, initial = [np.asarray(x) for x in (inputs, weights, initial)]
    if any(x.dtype not in (np.dtype('float32'), np.dtype('float64')) or not np.isfinite(x).all() for x in (inputs, weights, initial)):
        raise ValueError('Finite FP32/FP64 convolution inputs required')
    if inputs.ndim != 2 or weights.ndim != 2 or initial.ndim != 2:
        raise ValueError('Expected input[T,C], weight[C,K], history[K-1,C]')
    tokens, channels = inputs.shape
    other_channels, kernel = weights.shape
    if (not 0 <= tokens <= 8 or not 1 <= channels <= 128 or not 1 <= kernel <= 4
            or other_channels != channels or initial.shape != (kernel - 1, channels)):
        raise ValueError('Convolution geometry differs')
    inputs, weights, initial = [x.astype(np.float64) for x in (inputs, weights, initial)]
    joined = np.concatenate([initial, inputs])
    windows = np.stack([joined[t:t + kernel] for t in range(tokens)]) if tokens else np.empty((0, kernel, channels))
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        output = np.einsum('tkc,ck->tc', windows, weights)
        e = np.exp(-np.abs(output))
        sigmoid = np.where(output >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
        activated = output * sigmoid
    history = np.stack([joined[t + 1:t + kernel] for t in range(tokens)]) if tokens else np.empty((0, kernel - 1, channels))
    final = joined[-(kernel - 1):].copy() if kernel > 1 else np.empty((0, channels))
    if not np.isfinite(output).all() or not np.isfinite(activated).all():
        raise ValueError('Nonfinite convolution math result')
    return output, activated, history, final


def make_case(name, tokens, channels, kernel):
    import numpy as np
    def values(shape, multiplier, offset, scale):
        grid = (np.arange(math.prod(shape), dtype=np.int64) * multiplier + offset) % 31 - 15
        return (grid.astype(np.float64) * scale).astype(np.float32).reshape(shape)
    inputs = values((tokens, channels), 7, 4, 0.125)
    weights = values((channels, kernel), 11, 3, 0.0625)
    initial = values((kernel - 1, channels), 13, 8, 0.25)
    output, activated, history, final = run(inputs, weights, initial)
    return dict(name=name, tokens=tokens, channels=channels, kernel=kernel,
        arrays=dict(input=inputs, weight=weights, initial_history=initial, output=output,
            activated=activated, history=history, final_history=final))


def export():
    import numpy as np
    blob = bytearray()
    rows = []
    for spec in fixture.SPECS:
        row = make_case(*spec)
        descriptors = {}
        for name, value in row['arrays'].items():
            dtype = '<f4' if value.dtype == np.dtype('float32') else '<f8'
            raw = np.ascontiguousarray(value, dtype=dtype).tobytes()
            blob.extend(b'\0' * ((-len(blob)) % 8))
            if len(blob) + len(raw) > fixture.MAX_BYTES:
                raise ValueError('Convolution reference exceeds binary bound')
            descriptors[name] = dict(dtype=dtype, shape=list(value.shape), offset=len(blob), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            blob.extend(raw)
        rows.append(dict(row, arrays=descriptors))
    manifest = dict(format=fixture.FORMAT, producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        upstream_model_revision=fixture.MODEL_REVISION, upstream_model_source_sha256=fixture.MODEL_SOURCE_SHA,
        precision='FP32 inputs/weights/history; FP64 convolution/SiLU/reference history',
        weight_layout='[channels,kernel] oldest-to-current tap; upstream [channels,kernel,1]',
        history_layout='[kernel-1,channels] oldest-to-newest', runtime=dict(numpy=np.__version__),
        native_implemented=False, upstream_parity_checked=False, masked_supported=False, device_qualified=False,
        binary=dict(path='arrays.bin', bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(), alignment=8), cases=rows)
    return {'manifest.json': (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode(), 'arrays.bin': bytes(blob)}


def publish(outputs, destination):
    if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):
        raise ValueError('Empty existing regular convolution output required')
    if set(outputs) != {'manifest.json', 'arrays.bin'}:
        raise ValueError('Exact convolution output inventory required')
    for name, raw in outputs.items():
        with (destination / name).open('xb') as stream:
            stream.write(raw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--runtime-manifest', required=True, type=Path)
    parser.add_argument('--expected-runtime-sha256', required=True)
    parser.add_argument('--model-source', required=True, type=Path)
    options = parser.parse_args()
    runtime.captured(options.model_source, 128 * 1024, fixture.MODEL_SOURCE_SHA)
    admitted = runtime.admit_runtime(options.runtime_manifest, options.expected_runtime_sha256,
        [Path(__file__), Path(fixture.__file__)])
    import numpy as np
    if np.__file__ != admitted['imports']['numpy']:
        raise ValueError('Imported NumPy differs from admitted convolution producer')
    publish(export(), options.output)
