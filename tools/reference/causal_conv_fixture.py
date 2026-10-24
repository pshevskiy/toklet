"""Bounded independent depthwise causal-convolution fixture format."""
import hashlib
import json
import math
from pathlib import Path
import struct

import reference_runtime as runtime

FORMAT = 'toklet-causal-depthwise-math-reference-v1'
MAX_BYTES = 4 * 1024 * 1024
MODEL_REVISION = '0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d'
MODEL_SOURCE_SHA = '851449fd9989d6bf4b3c4fed92ddc3be5f1e781eb2246f89c54d7eff2518d9c6'
SPECS = (('empty', 0, 3, 4), ('scalar', 1, 1, 1),
         ('asymmetric', 7, 3, 4), ('channel-tail', 6, 5, 3), ('wide', 3, 128, 4))


def layout(tokens, channels, kernel):
    return dict(input=('<f4', [tokens, channels]), weight=('<f4', [channels, kernel]),
        initial_history=('<f4', [kernel - 1, channels]), output=('<f8', [tokens, channels]),
        activated=('<f8', [tokens, channels]), history=('<f8', [tokens, kernel - 1, channels]),
        final_history=('<f8', [kernel - 1, channels]))


def verify(folder, manifest_sha, binary_sha):
    folder = Path(folder)
    if folder.is_symlink() or not folder.is_dir() or {p.name for p in folder.iterdir()} != {'manifest.json', 'arrays.bin'}:
        raise ValueError('Exact regular convolution fixture directory required')
    raw = runtime.captured(folder / 'manifest.json', 128 * 1024, runtime.sha(manifest_sha))
    binary = runtime.captured(folder / 'arrays.bin', MAX_BYTES, runtime.sha(binary_sha))
    manifest = json.loads(raw, object_pairs_hook=runtime.json_unique,
        parse_constant=runtime.bad_json_number, parse_float=runtime.finite_json_number)
    if (not isinstance(manifest, dict) or set(manifest) != {
                'format', 'producer_sha256', 'precision', 'weight_layout', 'history_layout',
                'runtime', 'native_implemented', 'upstream_parity_checked', 'masked_supported',
                'device_qualified', 'binary', 'cases', 'upstream_model_revision',
                'upstream_model_source_sha256'} or manifest.get('format') != FORMAT
            or manifest.get('precision') != 'FP32 inputs/weights/history; FP64 convolution/SiLU/reference history'
            or manifest.get('weight_layout') != '[channels,kernel] oldest-to-current tap; upstream [channels,kernel,1]'
            or manifest.get('history_layout') != '[kernel-1,channels] oldest-to-newest'
            or manifest.get('runtime') != dict(numpy=runtime.VERSIONS['numpy'])
            or manifest.get('upstream_model_revision') != MODEL_REVISION
            or manifest.get('upstream_model_source_sha256') != MODEL_SOURCE_SHA
            or any(manifest.get(key) is not False for key in ('native_implemented', 'upstream_parity_checked', 'masked_supported', 'device_qualified'))):
        raise ValueError('Convolution reference format/numerical scope differs')
    runtime.sha(manifest.get('producer_sha256'))
    descriptor = manifest.get('binary')
    if descriptor != dict(path='arrays.bin', bytes=len(binary), sha256=binary_sha, alignment=8):
        raise ValueError('Convolution binary descriptor differs')
    if type(descriptor['bytes']) is not int or type(descriptor['alignment']) is not int:
        raise ValueError('Convolution binary size/alignment must be integers')
    cases = manifest.get('cases')
    if not isinstance(cases, list) or len(cases) != len(SPECS):
        raise ValueError('Convolution case inventory differs')
    cursor = 0
    for row, (name, tokens, channels, kernel) in zip(cases, SPECS):
        if not isinstance(row, dict) or set(row) != {'name', 'tokens', 'channels', 'kernel', 'arrays'}:
            raise ValueError('Convolution case descriptor differs')
        if (row['name'] != name or row['tokens'] != tokens or row['channels'] != channels or row['kernel'] != kernel
                or any(type(row[key]) is not int for key in ('tokens', 'channels', 'kernel'))):
            raise ValueError('Convolution case geometry differs')
        contract = layout(tokens, channels, kernel)
        if not isinstance(row['arrays'], dict) or set(row['arrays']) != set(contract):
            raise ValueError('Convolution array inventory differs')
        for key, (dtype, shape) in contract.items():
            item = row['arrays'][key]
            if (not isinstance(item, dict) or set(item) != {'dtype', 'shape', 'offset', 'bytes', 'sha256'}
                    or item['dtype'] != dtype or item['shape'] != shape or not isinstance(item['shape'], list)
                    or any(type(n) is not int for n in item['shape'])):
                raise ValueError('Convolution array type/shape differs')
            size = math.prod(shape) * (4 if dtype == '<f4' else 8)
            offset = cursor + (-cursor) % 8
            if (type(item['offset']) is not int or type(item['bytes']) is not int
                    or item['offset'] != offset or item['bytes'] != size
                    or offset + size > len(binary) or any(binary[cursor:offset])):
                raise ValueError('Convolution array span/padding differs')
            payload = binary[offset:offset + size]
            if hashlib.sha256(payload).hexdigest() != runtime.sha(item['sha256']):
                raise ValueError('Convolution array identity differs')
            if any(not math.isfinite(x[0]) for x in struct.iter_unpack('<f' if dtype == '<f4' else '<d', payload)):
                raise ValueError('Nonfinite convolution reference')
            cursor = offset + size
    if cursor != len(binary):
        raise ValueError('Unaccounted convolution bytes')
    return manifest
