"""Independent FP64 causal/sliding GQA reference before native implementation.

Absolute query/key origins are translated together; causal and window predicates
use logical positions. Cache storage/commit/shared ownership are separate tasks.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import signal
import struct
import sys

import reference_runtime as runtime

FORMAT = 'toklet-gemma-local-attention-math-v1'
MAX_JSON, MAX_RAW = 2*1024*1024, 1024*1024
REVISION = '0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d'
SOURCES = {'base.py': '63d8e58a4b6f8c0902719499c693647fbf486bbd5d22b09868a07d37e08c83d6'}
KEYS = ('queries', 'keys', 'q_heads', 'kv_heads', 'dims', 'window', 'origin', 'scale')
SPECS = (
    ('empty', (0,0,2,1,4,3,0,1.)),
    ('single', (1,1,2,1,4,3,0,1.)),
    ('prefill', (5,5,4,2,3,3,0,1.)),
    ('cached', (3,8,4,2,4,3,0,1.)),
    ('trimmed_origin', (3,5,2,1,3,3,17,1.)),
    ('window_one', (4,6,2,1,4,1,9,1.)),
    ('full', (3,8,4,2,3,None,0,1.)),
    ('wide_window', (3,7,6,2,5,16,4,.5)),
    ('actual_global', (2,4,8,1,512,None,0,1.)),
    ('actual_local', (3,5,8,1,256,3,9,1.)),
    ('actual_window_boundary', (3,515,2,1,2,512,0,1.)),
)
ATOL, RTOL = 1e-5, 1e-4


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()


def geometry(queries, keys, q_heads, kv_heads, dims, window, origin, scale):
    if (any(type(x) is not int for x in (queries, keys, q_heads, kv_heads, dims, origin))
            or not 0 <= queries <= 8 or not queries <= keys <= 515 or (queries == 0) != (keys == 0)
            or not 1 <= kv_heads <= q_heads <= 8 or q_heads % kv_heads
            or not 1 <= dims <= 512 or not 0 <= origin <= 32
            or q_heads*queries*dims > 16384 or kv_heads*keys*dims > 16384 or q_heads*queries*keys > 32768
            or (window is not None and (type(window) is not int or not 1 <= window <= 512))
            or type(scale) not in (int,float) or not 0 < scale <= 8 or not math.isfinite(scale)):
        raise ValueError('Bounded causal/window GQA geometry required')
    if struct.unpack('<f',struct.pack('<f',scale))[0] != scale:
        raise ValueError('Exactly representable FP32 attention scale required')


def math_values(q, k, v, queries, keys, q_heads, kv_heads, dims, window, origin, scale):
    geometry(queries, keys, q_heads, kv_heads, dims, window, origin, scale)
    import numpy as np
    q,k,v = [np.asarray(x) for x in (q,k,v)]
    if ([x.shape for x in (q,k,v)] != [(q_heads,queries,dims),(kv_heads,keys,dims),(kv_heads,keys,dims)]
            or any(x.dtype != np.dtype('float32') or not np.isfinite(x).all() for x in (q,k,v))):
        raise ValueError('Finite FP32 attention arrays with exact shapes required')
    mask = np.array([[origin+j <= origin+keys-queries+i and
        (window is None or origin+keys-queries+i-(origin+j) < window)
        for j in range(keys)] for i in range(queries)],dtype=np.bool_).reshape(queries,keys)
    scores = np.zeros((q_heads,queries,keys),np.float64)
    probabilities = np.zeros_like(scores)
    output = np.zeros((q_heads,queries,dims),np.float64)
    for head in range(q_heads):
        kv_head = head // (q_heads//kv_heads)
        scores[head] = (q[head].astype(np.float64) @ k[kv_head].astype(np.float64).T) * scale
        for row in range(queries):
            active = mask[row]
            included = scores[head,row,active]
            weights = np.exp(included-included.max())
            probabilities[head,row,active] = weights/weights.sum()
        output[head] = probabilities[head] @ v[kv_head].astype(np.float64)
    return dict(mask=mask,scores=scores,probabilities=probabilities,output=output)


def export():
    import numpy as np
    cases=[]
    for name,spec in SPECS:
        params=dict(zip(KEYS,spec));n,k,h,hk,d,*_=spec
        def values(shape,multiplier,shift):
            return (((np.arange(math.prod(shape))*multiplier+shift)%31)-15).astype(np.float32).reshape(shape)*np.float32(.125)
        q,k,v=values((h,n,d),7,4),values((hk,k,d),11,3),values((hk,k,d),13,7)
        arrays=dict(q=q,k=k,v=v,**math_values(q,k,v,**params))
        cases.append(dict(name=name,**params,arrays={key:dict(dtype='|b1' if value.dtype==np.dtype('bool') else '<f4' if value.dtype==np.dtype('float32') else '<f8',shape=list(value.shape),values=value.reshape(-1).tolist()) for key,value in arrays.items()}))
    return encoded(dict(format=FORMAT,source_revision=REVISION,source_sha256=SOURCES,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        precision='FP32 inputs/scale; independent FP64 score, stable softmax and context',
        runtime=dict(numpy=np.__version__),native_graph_implemented=False,device_qualified=False,cases=cases))


def verify(path,expected_sha):
    raw=runtime.captured(path,MAX_JSON,runtime.sha(expected_sha))
    data=json.loads(raw,object_pairs_hook=runtime.json_unique,parse_constant=runtime.bad_json_number,parse_float=runtime.finite_json_number)
    if (not isinstance(data,dict) or set(data)!={'format','source_revision','source_sha256','producer_sha256','precision','runtime','native_graph_implemented','device_qualified','cases'}
            or data['format']!=FORMAT or data['source_revision']!=REVISION or data['source_sha256']!=SOURCES
            or data['precision']!='FP32 inputs/scale; independent FP64 score, stable softmax and context'
            or data['runtime']!=dict(numpy=runtime.VERSIONS['numpy']) or data['native_graph_implemented'] is not False or data['device_qualified'] is not False
            or not isinstance(data['cases'],list) or len(data['cases'])!=len(SPECS)):
        raise ValueError('Local attention identity/scope differs')
    runtime.sha(data['producer_sha256'])
    for row,(name,spec) in zip(data['cases'],SPECS):
        if not isinstance(row,dict) or set(row)!={'name','arrays',*KEYS} or row['name']!=name or tuple(row[key] for key in KEYS)!=spec:
            raise ValueError('Local attention case inventory differs')
        geometry(*(row[key] for key in KEYS))
        n,k,h,hk,d,*_=spec
        shapes=dict(q=[h,n,d],k=[hk,k,d],v=[hk,k,d],mask=[n,k],scores=[h,n,k],probabilities=[h,n,k],output=[h,n,d])
        if not isinstance(row['arrays'],dict) or set(row['arrays'])!=set(shapes):raise ValueError('Attention array inventory differs')
        for key,shape in shapes.items():
            item=row['arrays'][key];dtype='|b1' if key=='mask' else '<f4' if key in ('q','k','v') else '<f8'
            if (not isinstance(item,dict) or set(item)!={'dtype','shape','values'} or item['dtype']!=dtype or item['shape']!=shape
                    or not isinstance(item['shape'],list) or any(type(x) is not int for x in item['shape'])
                    or not isinstance(item['values'],list) or len(item['values'])!=math.prod(shape)):
                raise ValueError('Attention array descriptor differs')
            for value in item['values']:
                if key=='mask':
                    if type(value) is not bool:raise ValueError('Boolean attention mask required')
                else:
                    if type(value) not in (float,int) or not -sys.float_info.max<=value<=sys.float_info.max or not math.isfinite(value):raise ValueError('Finite attention value required')
                    if dtype=='<f4':
                        try:
                            if struct.unpack('<f',struct.pack('<f',value))[0]!=value:raise ValueError('Exactly representable FP32 input required')
                        except OverflowError as error:raise ValueError('FP32 input overflow') from error
    return data


def source_nodes(sources):
    names={'create_causal_mask','scaled_dot_product_attention'}
    nodes=[node for node in ast.parse(sources['base.py']).body if isinstance(node,ast.FunctionDef) and node.name in names]
    if len(nodes)!=2 or {node.name for node in nodes}!=names:raise ValueError('Unique pinned mask/SDPA definitions required')
    return {node.name:node for node in nodes}


def compare(reference, reference_sha, sources, device, destination, runtime_manifest, runtime_sha):
    if device not in ('cpu', 'gpu'):
        raise ValueError('Explicit local attention reference device required')
    data = verify(reference, reference_sha)
    if (set(sources) != set(SOURCES) or any(not isinstance(raw, bytes) or len(raw) > 128 * 1024
            or hashlib.sha256(raw).hexdigest() != SOURCES[name] for name, raw in sources.items())):
        raise ValueError('Authenticated local attention source buffers differ')
    nodes = source_nodes(sources)
    admitted = runtime.admit_runtime(runtime_manifest, runtime_sha, [Path(__file__)])
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    from typing import Optional
    if (np.__file__ != admitted['imports']['numpy'] or mx.__file__ != admitted['imports']['mlx.core']
            or nn.__file__ != admitted['imports']['mlx.nn']):
        raise ValueError('Actual local attention imports differ from admitted files')
    if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):
        raise ValueError('Empty regular local attention output required')
    previous_device = mx.default_device()
    previous_signals = {}
    blob = bytearray()
    rows, comparisons, cleanup = [], [], []
    primary = None

    def interrupted(number, _frame):
        for value in (signal.SIGTERM, signal.SIGINT):
            signal.signal(value, signal.SIG_IGN)
        raise InterruptedError('Received signal ' + str(number))

    def array(case, name):
        item = case['arrays'][name]
        return np.array(item['values'], dtype=item['dtype']).reshape(item['shape'])

    def observe(row, name, value, expected):
        mx.eval(value)
        actual = np.asarray(value)
        if actual.dtype.hasobject or actual.dtype.kind not in 'biufc' or actual.nbytes > MAX_RAW:
            row['arrays'][name] = dict(raw_unavailable='Unsupported numeric observation', dtype=str(actual.dtype), shape=list(actual.shape))
            raise ValueError('Bounded numeric observation required')
        dtype = actual.dtype.newbyteorder('<')
        raw = np.ascontiguousarray(actual, dtype=dtype).tobytes()
        padding = (-len(blob)) % 8
        if len(blob) + padding + len(raw) > MAX_RAW:
            raise ValueError('local attention observation buffer exceeds bound')
        blob.extend(b'\0' * padding)
        row['arrays'][name] = dict(dtype=dtype.str, shape=list(actual.shape), offset=len(blob), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        blob.extend(raw)
        required_dtype = np.dtype('bool') if expected.dtype == np.dtype('bool') else np.dtype('float32')
        if actual.dtype != required_dtype or actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError('Observed local attention dtype/shape/finite values differ')
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        ratio = difference if required_dtype == np.dtype('bool') else difference / (ATOL + RTOL * np.abs(expected))
        if required_dtype == np.dtype('bool') and np.any(difference):
            raise ValueError('Causal window mask must match exactly')
        comparisons.append(dict(case=row['name'], name=name, passed=bool(np.all(ratio <= 1)),
            maximum_absolute_error=float(difference.max()) if difference.size else 0,
            maximum_error_to_policy_ratio=float(ratio.max()) if ratio.size else 0))

    try:
        for value in (signal.SIGTERM, signal.SIGINT):
            previous_signals[value] = signal.signal(value, interrupted)
        mx.set_default_device(mx.cpu if device == 'cpu' else mx.gpu)
        environment = dict(mx=mx, nn=nn, Optional=Optional)
        for filename, node in nodes.items():
            exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), environment)
        for case in data['cases']:
            row = dict(name=case['name'], completed=False, upstream_evaluated=False, arrays={})
            rows.append(row)
            if case['queries'] == 0:
                row.update(completed=True, reason='Empty sequence is mathematical no-call policy')
                continue
            q, k, v = [mx.array(array(case, name))[None] for name in ('q', 'k', 'v')]
            mask = environment['create_causal_mask'](case['queries'], case['keys']-case['queries'], window_size=case['window'])
            observe(row, 'mask', mask, array(case, 'mask'))
            value = environment['scaled_dot_product_attention'](q, k, v, cache=None, scale=case['scale'], mask=mask)
            observe(row, 'output', value[0], array(case, 'output'))
            for actual, name in zip((q, k, v), ('q', 'k', 'v')):
                if not np.array_equal(np.asarray(actual)[0], array(case, name)):
                    raise ValueError('Attention mutated immutable inputs')
            row.update(completed=True, upstream_evaluated=True)
    except (Exception, KeyboardInterrupt) as error:
        primary = type(error).__name__ + ': ' + str(error)
    finally:
        for value in previous_signals:
            signal.signal(value, signal.SIG_IGN)
        try:
            mx.set_default_device(previous_device)
        except Exception as error:
            cleanup.append('Restore device: ' + type(error).__name__ + ': ' + str(error))
    passed = primary is None and not cleanup and len(rows) == len(SPECS) and all(row['completed'] for row in rows) and len(comparisons) == 20 and all(row['passed'] for row in comparisons)
    report = dict(format='toklet-gemma-local-attention-observation-v1', passed=passed,
        mathematical_reference_sha256=reference_sha, source_revision=REVISION, source_sha256=SOURCES,
        admitted_runtime=admitted, requested_device=device, source_policy=dict(atol=ATOL, rtol=RTOL),
        selected_source_definitions={name: dict(line=node.lineno, source=ast.unparse(node)) for name, node in nodes.items()},
        cases=rows, comparisons=comparisons, binary=dict(path='observed.bin', bytes=len(blob), alignment=8, sha256=hashlib.sha256(blob).hexdigest()),
        primary_error=primary, cleanup_errors=cleanup, native_graph_implemented=False, device_qualified=False,
        scope='Tiny causal/sliding GQA source/operator comparison; no native graph or performance qualification')
    try:
        for name, raw in [('observed.bin', bytes(blob)), ('observation.json', encoded(report))]:
            with (destination / name).open('xb') as stream:
                stream.write(raw)
    finally:
        for value, handler in previous_signals.items():
            signal.signal(value, handler)
    return passed



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['export', 'compare'])
    parser.add_argument('--sources', required=True, type=Path)
    parser.add_argument('--runtime-manifest', required=True, type=Path)
    parser.add_argument('--expected-runtime-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--expected-reference-sha256')
    parser.add_argument('--device', choices=['cpu', 'gpu'])
    options = parser.parse_args()
    sources = {name: runtime.captured(options.sources / name, 128 * 1024, expected) for name, expected in SOURCES.items()}
    source_nodes(sources)
    if options.mode == 'compare':
        if options.reference is None or options.expected_reference_sha256 is None or options.device is None:
            parser.error('Comparison requires explicit reference identity and device')
        verify(options.reference, options.expected_reference_sha256)
    if options.mode == 'export':
        admitted = runtime.admit_runtime(options.runtime_manifest, options.expected_runtime_sha256, [Path(__file__)])
        import numpy as np
        if np.__file__ != admitted['imports']['numpy']:
            raise ValueError('Imported NumPy differs from admitted producer')
        with options.output.open('xb') as stream:
            stream.write(export())
    else:
        passed = compare(options.reference, options.expected_reference_sha256, sources, options.device, options.output, options.runtime_manifest, options.expected_runtime_sha256)
        print(json.dumps(dict(passed=passed, mathematical_reference_only=True)))
        raise SystemExit(0 if passed else 1)
