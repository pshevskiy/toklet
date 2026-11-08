"""Independent source-row FP64 projection oracle over frozen real FP32 operands.

The operands come from native context diagnostics. Native projection outputs are
never reference targets. This qualifies isolated operations, not a coupled graph.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import time
import affine4_tiles as tiles
import bf16_source
import check_qwen35_payload as pin
import gemma4_affine_reference as runtime_owner
import qwen35_cpu_graph as graph
import qwen35_real_weights as weights

need = weights.need
INPUT_SHA = '4a55cd6be84a387571b04fbb4490be0f3014fca36c1b9a55b2fd331551fb9112'
CONTEXT_LOG_SHA = 'eac873f075360ea8cd10521a81ef573331d4896009c78f96fc5a157057be4d1c'
VALUES = 24846336
MAX_CAPTURE = 256*1024**2
ROOT = Path(__file__).resolve().parents[2]


def operand(name):
    if name == 'language_model.model.embed_tokens.weight': return None, 'final_norm'
    match = re.fullmatch(r'language_model\.model\.layers\.(\d+)\.(.+)\.weight', name)
    need(match is not None and 0 <= int(match[1]) < 32, 'Unsupported source matrix name')
    stage = {
        'linear_attn.in_proj_qkv': 'input_norm', 'linear_attn.in_proj_z': 'input_norm',
        'linear_attn.in_proj_a': 'input_norm', 'linear_attn.in_proj_b': 'input_norm',
        'linear_attn.out_proj': 'delta_gated',
        'self_attn.q_proj': 'input_norm', 'self_attn.k_proj': 'input_norm',
        'self_attn.v_proj': 'input_norm', 'self_attn.o_proj': 'attention_gated',
        'mlp.gate_proj': 'post_norm', 'mlp.up_proj': 'post_norm', 'mlp.down_proj': 'mlp_activated',
    }.get(match[2])
    need(stage is not None, 'Unsupported projection operand role')
    return int(match[1]), stage


def document(path):
    need(not path.is_symlink() and path.is_file() and path.stat().st_size <= 2*1024**2, 'Bounded evidence document required')
    return json.loads(path.read_bytes(), object_pairs_hook=weights.unique)


def operands(folder):
    """Validate the sealed context capture, then extract only declared inputs."""
    receipt = document(folder/'checks.json')
    need(receipt['passed'] is True and receipt['exit_code'] == 0 and type(receipt['exit_code']) is int
         and receipt['group_absent'] is True and receipt['error'] is None
         and receipt['changed_inputs'] == [], 'Closed successful native operand source required')
    inventory = {'console.log', 'time-and-errors.log', 'inputs.json', 'capture/rows.f32le'}
    need(set(receipt['files_sha256']) == inventory
         and receipt['files_sha256']['console.log'] == CONTEXT_LOG_SHA, 'Exact operand source seals required')
    for name, digest in receipt['files_sha256'].items():
        p = folder/name
        bound = 128*1024**2 if name.endswith('f32le') else 2*1024**2
        need(not p.is_symlink() and p.is_file() and p.stat().st_size <= bound, 'Bounded operand evidence required')
        need(hashlib.sha256(p.read_bytes()).hexdigest() == digest, 'Operand source seal differs')
    body = (folder/'capture/rows.f32le').read_bytes()
    need(hashlib.sha256(body).hexdigest() == INPUT_SHA and len(body) == 65334272, 'Frozen native operand body differs')
    rows = [json.loads(line.split(' ', 1)[1], object_pairs_hook=weights.unique)
            for line in (folder/'console.log').read_text().splitlines()
            if line.startswith('QWEN35_CONTEXT_ARRAY ')]
    need(len(rows) == 1051, 'Complete native source inventory required')
    arrays = {}; offset = 0
    for row in rows:
        need(row['token'] == 16 and type(row['token']) is int and row['position'] == 53
             and type(row['offset']) is int and row['offset'] == offset
             and type(row['bytes']) is int and row['bytes'] == row['count']*4,
             'Native source coordinate/coverage differs')
        raw = body[offset:offset+row['bytes']]; offset += len(raw)
        need(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'Native array seal differs')
        key = (row['layer'], row['stage'], row['head'])
        need(key not in arrays, 'Duplicate native array')
        arrays[key] = raw
    need(offset == len(body), 'Complete native body coverage required')
    result = bytearray(); descriptors = []
    for name, (kind, shape) in sorted(graph.specs(weights.GEOMETRY).items()):
        if kind != 'matrix': continue
        layer, stage = operand(name)
        raw = arrays[(layer, stage, None)]
        need(len(raw) == shape[1]*4 and all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',raw)), 'Exact finite operand shape required')
        descriptors.append(dict(name=name, rows=shape[0], columns=shape[1], offset=len(result), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), layer=layer, stage=stage))
        result.extend(raw)
    need(len(descriptors) == 249 and sum(v['rows'] for v in descriptors) == 1380352
         and len(result) <= 8*1024**2, 'Frozen projection inventory differs')
    return bytes(result), descriptors


def extras():
    return [Path(module.__file__).resolve() for module in [weights, graph, bf16_source, pin]] + [Path(__file__).resolve()]


def collect(output, directory, native, admission, endpoint):
    import numpy as np
    need(np.__version__ == '2.3.2', 'Pinned NumPy2.3.2 required')
    need(output.resolve().is_relative_to(ROOT/'artifacts'), 'Local fresh artifact output required')
    output.mkdir()
    record = dict(format='toklet-qwen35-projection-reference-v1', completed=False,
        source_sha256=pin.FILE_SHA, input_capture_sha256=INPUT_SHA, gate=graph.GATE,
        batches=[1,17], scalars=[1,-1,0.5], values=VALUES, arrays=[],
        ledger=dict(python_state_and_temporaries=2*1024**3,interpreter_and_runtime=512*1024**2,
            source_and_tile_buffers=128*1024**2,small_vectors=64*1024**2,capture_limit=MAX_CAPTURE),
        runtime_admission=admission, primary_error=None, endpoint_errors=[], source=None,
        gpu_graph_qualified=False, quality_qualified=False, managed_adoption_qualified=False)
    owner = None
    try:
        raw, descriptors = operands(native)
        (output/'operands.f32le').write_bytes(raw)
        record['operands_sha256'] = hashlib.sha256(raw).hexdigest()
        record['operands'] = descriptors
        deadline = time.monotonic()+1800
        with bf16_source.Source(directory/'model.safetensors', pin.FILE_BYTES, pin.FILE_SHA, deadline=deadline) as owner:
            matrices, _ = weights.weights(owner, (directory/'config.json').read_bytes(), deadline)
            need(sorted(matrices) == [v['name'] for v in descriptors], 'Source matrix inventory differs')
            with (output/'rows.f64le').open('xb') as stream:
                for batch in [1,17]:
                    for item in descriptors:
                        x = np.frombuffer(raw[item['offset']:item['offset']+item['bytes']], dtype='<f4')
                        inputs = np.asarray([x * np.float32([1,-1,0.5][j%3]) for j in range(batch)], dtype='<f4')
                        # Independent BLAS FP64 dot against exact reconstructed FP32 weights.
                        actual = tiles.project_rows(matrices[item['name']].rows, inputs.astype(np.float64), np.float64)
                        need(actual.shape == (batch,item['rows']) and np.isfinite(actual).all(), 'Finite full reference output required')
                        payload = np.asarray(actual,dtype='<f8').tobytes()
                        need(stream.tell()+len(payload) <= VALUES*8 <= MAX_CAPTURE, 'Reference capture bound exceeded')
                        record['arrays'].append(dict(name=item['name'],batch=batch,rows=item['rows'],columns=item['columns'],offset=stream.tell(),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
                        need(stream.write(payload) == len(payload), 'Short reference write')
                    print(json.dumps(dict(event='batch-complete',batch=batch)),flush=True)
                need(stream.tell() == VALUES*8, 'Complete reference capture required')
            digest = hashlib.sha256()
            for offset in range(0,owner.size,bf16_source.HASH_BLOCK):
                digest.update(owner.read(offset,min(bf16_source.HASH_BLOCK,owner.size-offset),deadline=deadline))
            need(digest.hexdigest() == pin.FILE_SHA, 'Source endpoint differs')
            record['source_endpoint_sha256'] = digest.hexdigest()
        record['rows_sha256'] = runtime_owner.sha(output/'rows.f64le')
        record['completed'] = True
    except BaseException as error:
        record['primary_error'] = dict(type=type(error).__name__,message=str(error)[:4096])
    finally:
        if owner is not None:
            if not owner.closed:
                try: owner.close()
                except BaseException as error: record['endpoint_errors'].append(str(error)[:4096])
            record['source'] = owner.audit()
        try: record['runtime_endpoint'] = endpoint()
        except BaseException as error: record['endpoint_errors'].append(str(error)[:4096])
        if record['endpoint_errors']: record['completed'] = False
        (output/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model-directory','native-context','output','runtime-manifest','native-manifest']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-runtime-sha256',required=True)
    parser.add_argument('--expected-native-sha256',required=True)
    args = parser.parse_args()
    admission, endpoint = runtime_owner.runtime_context(args.runtime_manifest,args.expected_runtime_sha256,
        args.native_manifest,args.expected_native_sha256,extra_sources=extras())
    result = collect(args.output,args.model_directory,args.native_context,admission,endpoint)
    print(json.dumps({k:result[k] for k in ['completed','primary_error','endpoint_errors']}))
    raise SystemExit(0 if result['completed'] else 1)
