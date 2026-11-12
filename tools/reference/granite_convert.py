"""Stream the frozen Granite BF16 parent into an explicit mixed affine4/8 child.

No model is constructed. A candidate is published only after complete parent
authentication, every component readback, independent FP32 reconstruction,
unchanged runtime/source metadata and a complete child hash close successfully.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import time

import granite_convert_layout as layout
import granite_quantize as quant
import granite_quant_runtime as runtime
from granite_quant_probe import upstream_f32

need = quant.need
BLOCK = 4 * 1024**2


def sources():
    folder = Path(__file__).resolve().parent
    return [folder / name for name in (
        'granite_convert.py', 'test_granite_convert.py',
        'granite_convert_layout.py', 'test_granite_convert_layout.py',
        'granite_quantize.py', 'test_granite_quantize.py',
        'granite_quant_runtime.py', 'test_granite_quant_runtime.py',
        'granite_quant_probe.py', 'minicpm5_quantize.py',
        'minicpm5_quant_runtime.py', 'minicpm5_parent.py')]


def read_exact(fd, count, offset):
    need(type(count) is int and 0 <= count <= BLOCK
         and type(offset) is int and offset >= 0, 'Bounded exact source read required')
    result = bytearray()
    while len(result) < count:
        raw = os.pread(fd, count - len(result), offset + len(result))
        need(bool(raw), 'Source descriptor ended before its declared extent')
        result.extend(raw)
    return bytes(result)


def hash_range(fd, count, offset=0):
    need(type(count) is int and count >= 0, 'Nonnegative hash extent required')
    digest = hashlib.sha256()
    for first in range(0, count, 1024**2):
        digest.update(read_exact(fd, min(1024**2, count - first), offset + first))
    return digest.hexdigest()


def verify_named(fd, path, expected=None):
    owned, named = os.fstat(fd), path.lstat()
    need(stat.S_ISREG(owned.st_mode) and stat.S_ISREG(named.st_mode)
         and runtime.identity(owned) == runtime.identity(named),
         'Consumed descriptor and named regular source differ')
    if expected is not None:
        need(runtime.identity(owned) == expected, 'Source identity changed during conversion')
    return runtime.identity(owned)


def converted_config(config, matrices):
    """Add only the explicit storage recipe and exact legacy default-RoPE alias."""
    need(type(config) is dict and config.get('model_type') == 'granite'
         and config.get('architectures') == ['GraniteForCausalLM']
         and 'quantization' not in config and 'quantization_config' not in config,
         'Unquantized Granite configuration required')
    parameters = config.get('rope_parameters')
    need(type(parameters) is dict and set(parameters) == {'rope_type', 'rope_theta'}
         and parameters['rope_type'] == 'default'
         and type(parameters['rope_theta']) is int and parameters['rope_theta'] > 0
         and config.get('rope_scaling') is None, 'Explicit integer default Granite RoPE required')
    if 'rope_theta' in config:
        need(type(config['rope_theta']) is int
             and config['rope_theta'] == parameters['rope_theta'], 'Conflicting RoPE theta aliases')
    need(config.get('dtype') == 'bfloat16'
         and config.get('torch_dtype', 'bfloat16') == 'bfloat16', 'BF16 source declaration required')
    result = copy.deepcopy(config)
    result['rope_theta'] = parameters['rope_theta']
    recipe = dict(group_size=64, bits=4, mode='affine')
    for matrix in matrices:
        need(matrix['bits'] in (4, 8) and type(matrix['bits']) is int
             and matrix['group_size'] == 64 and matrix['mode'] == 'affine',
             'Explicit affine4/8 group-64 matrix recipe required')
        if matrix['bits'] == 8:
            need(matrix['module'] not in recipe, 'Duplicate matrix override')
            recipe[matrix['module']] = dict(group_size=64, bits=8, mode='affine')
    result['quantization'] = recipe
    return result


def reserve_output(output, evidence):
    need(output.resolve() != evidence.resolve(), 'Distinct candidate/evidence directories required')
    for path in (output, evidence):
        need(not path.exists() and not path.is_symlink(), 'Never replace an existing output or evidence path')
    output.mkdir()
    evidence.mkdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('environment-root', 'runtime-manifest', 'runtime-sha256',
                 'contract', 'contract-sha256', 'source-authentication',
                 'fresh-output', 'fresh-evidence'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    output, evidence = Path(args.fresh_output), Path(args.fresh_evidence)
    reserve_output(output, evidence)
    contract_raw = runtime.common.captured(Path(args.contract), 2 * 1024**2,
                                           runtime.common.sha(args.contract_sha256))
    contract = json.loads(contract_raw, object_pairs_hook=runtime.common.json_unique,
                          parse_constant=runtime.common.bad_json_number)
    need(contract['format'] == 'toklet-granite-heretic-mixed-affine-v1', 'Frozen Granite contract required')
    recipe, source = contract['recipe'], contract['source']
    need(recipe['device'] == 'gpu' and recipe['input_dtype'] == recipe['metadata_dtype'] == 'BF16'
         and recipe['maximum_rows_per_tile'] == quant.ROWS
         and recipe['maximum_columns'] == quant.MAX_COLUMNS
         and recipe['quantization_runtime'] == runtime.VERSIONS, 'Frozen conversion mechanism differs')
    auth_raw = runtime.common.captured(Path(args.source_authentication), 2 * 1024**2,
                                       contract['source_authentication_receipt_sha256'])
    auth = json.loads(auth_raw, object_pairs_hook=runtime.common.json_unique)
    need(auth['passed'] is True, 'Complete prior source authentication required')
    matching = [row for row in auth['sources'] if row['repository'] == source['repository']
                and row['revision'] == source['revision'] and row['directory'] == source['directory']]
    need(len(matching) == 1, 'Exact authenticated source package required')
    metadata = {item['file']: item for item in matching[0]['metadata']}
    source_dir = Path(source['directory'])
    config_path, index_path = source_dir / 'config.json', source_dir / 'model.safetensors.index.json'
    config_raw = runtime.common.captured(config_path, 1024**2, source['config_sha256'])
    index_raw = runtime.common.captured(index_path, 1024**2, source['index_sha256'])
    config = json.loads(config_raw, object_pairs_hook=runtime.common.json_unique,
                        parse_constant=runtime.common.bad_json_number)
    child_config = converted_config(config, recipe['matrices'])
    planned = layout.output_layout(recipe['matrices'], recipe['norms'])
    need(planned['body_bytes'] == contract['expected_output']['body_bytes']
         and planned['file_bytes'] == contract['expected_output']['file_bytes']
         and hashlib.sha256(planned['prefix']).hexdigest() == contract['expected_output']['header_sha256']
         and len(planned['matrices']) == 282 and len(planned['logical']) == 363,
         'Complete frozen output layout differs')
    space = os.statvfs(output)
    need(space.f_bavail * space.f_frsize >= planned['file_bytes'] + 2 * 1024**3,
         'Output reservation must preserve the disk floor')
    opened, destination = [], None
    try:
        owners = {}
        for container in source['containers']:
            name = container['container']
            need(Path(name).name == name and name not in ('.', '..'), 'Safe source shard leaf required')
            path = source_dir / name
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
            opened.append(dict(fd=fd, path=path, pin=container))
            obj = opened[-1]
            obj['identity'] = verify_named(fd, path)
            need(obj['identity'][2] == container['expected_bytes'], 'Source shard extent differs')
            prefix = read_exact(fd, 8, 0)
            count = struct.unpack('<Q', prefix)[0]
            need(0 < count <= 2 * 1024**2, 'Bounded source header required')
            raw = read_exact(fd, count, 8)
            obj['prefix'] = prefix + raw
            obj['origin'] = count + 8
            obj['spans'] = layout.source_spans(prefix, raw, container['expected_bytes'], planned['logical'])
            for span in obj['spans']:
                need(span['name'] not in owners, 'Source tensor has multiple shard owners')
                owners[span['name']] = name
        need(set(owners) == set(planned['logical']), 'Source logical coverage differs')
        total = sum(s['end'] - s['start'] for obj in opened for s in obj['spans'])
        layout.validate_index(index_raw, [(obj['path'].name, obj['spans']) for obj in opened], total)
        runtime.bind_environment(args.environment_root)
        admitted, endpoint = runtime.admit(Path(args.runtime_manifest), args.runtime_sha256, sources())
        import numpy as np
        import mlx.core as mx
        mx.set_cache_limit(0)
        partial = output / 'model.safetensors.partial'
        destination = os.open(partial, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        os.ftruncate(destination, planned['file_bytes'])
        layout.write_all(destination, planned['prefix'], 0)
        matrix_reports, norm_reports, object_reports = [], [], []
        tile_count = reconstructed_values = 0
        with (evidence / 'records.jsonl').open('x') as journal:
            def event(value):
                journal.write(json.dumps(value, separators=(',', ':')) + '\n')
                journal.flush()

            for obj in opened:
                fd, pin = obj['fd'], obj['pin']
                whole = hashlib.sha256(obj['prefix'])
                for span in obj['spans']:
                    name, shape = span['name'], span['shape']
                    tensor = hashlib.sha256()
                    if len(shape) == 1:
                        raw = read_exact(fd, span['end'] - span['start'], obj['origin'] + span['start'])
                        words = np.frombuffer(raw, dtype='<u2')
                        need(not np.any((words & 0x7f80) == 0x7f80), 'Nonfinite source norm')
                        whole.update(raw)
                        layout.write_norm(destination, planned, name, raw)
                        report = dict(name=name, bytes=len(raw), copied_BF16_sha256=hashlib.sha256(raw).hexdigest())
                        norm_reports.append(report)
                        event(dict(event='norm_copied', parent_object_authenticated=False, **report))
                        del raw, words
                        continue
                    plan = planned['matrices'][name]
                    field_hashes = {key: hashlib.sha256() for key in ('weight', 'scales', 'biases')}
                    oracle_hash = hashlib.sha256()
                    for first in range(0, shape[0], quant.ROWS):
                        rows = min(quant.ROWS, shape[0] - first)
                        before = time.monotonic_ns()
                        raw = read_exact(fd, rows * shape[1] * 2,
                                         obj['origin'] + span['start'] + first * shape[1] * 2)
                        read_ns = time.monotonic_ns() - before
                        whole.update(raw)
                        tensor.update(raw)
                        before = time.monotonic_ns()
                        buffers = quant.quantize(raw, rows, shape[1], plan['bits'], 'gpu', np, mx)
                        quantize_ns = time.monotonic_ns() - before
                        before = time.monotonic_ns()
                        independent = quant.reconstruct_f32(buffers, rows, shape[1], plan['bits'], np)
                        independent_ns = time.monotonic_ns() - before
                        before = time.monotonic_ns()
                        upstream = upstream_f32(buffers, rows, shape[1], plan['bits'], 'gpu', np, mx)
                        upstream_ns = time.monotonic_ns() - before
                        equal = independent == upstream
                        event(dict(event='matrix_tile', parent_object_authenticated=False, name=name,
                                   first_row=first, rows=rows, columns=shape[1], bits=plan['bits'],
                                   source_offset=obj['origin'] + span['start'] + first * shape[1] * 2,
                                   source_bytes=len(raw), source_sha256=hashlib.sha256(raw).hexdigest(),
                                   fields={key: dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
                                           for key, value in buffers.items()},
                                   independent_F32_sha256=hashlib.sha256(independent).hexdigest(),
                                   upstream_F32_sha256=hashlib.sha256(upstream).hexdigest(),
                                   reconstructed_values=rows * shape[1], reconstruction_equal=equal,
                                   source_read_ns=read_ns, completed_quantize_readback_ns=quantize_ns,
                                   independent_decode_ns=independent_ns, completed_upstream_decode_ns=upstream_ns))
                        need(equal, 'Independent FP32 reconstruction differs')
                        layout.write_matrix_tile(destination, planned, name, first, rows, buffers)
                        for key, value in buffers.items():
                            field_hashes[key].update(value)
                        oracle_hash.update(independent)
                        tile_count += 1
                        reconstructed_values += rows * shape[1]
                        del raw, buffers, independent, upstream
                    fields = {}
                    for key, item in plan['fields'].items():
                        declared = planned['header'][item['name']]
                        start, end = declared['data_offsets']
                        stored = hash_range(destination, end - start, planned['origin'] + start)
                        need(stored == field_hashes[key].hexdigest(), 'Complete stored matrix component hash differs')
                        fields[key] = dict(bytes=end - start, sha256=stored)
                    report = dict(name=name, shape=shape, bits=plan['bits'],
                                  source_tensor_sha256=tensor.hexdigest(), fields=fields,
                                  independently_reconstructed_F32_sha256=oracle_hash.hexdigest())
                    matrix_reports.append(report)
                    event(dict(event='matrix_completed', parent_object_authenticated=False, **report))
                need(read_exact(fd, 0, pin['expected_bytes']) == b''
                     and os.pread(fd, 1, pin['expected_bytes']) == b'', 'Trailing source bytes')
                need(whole.hexdigest() == pin['expected_sha256'], 'Complete consumed parent SHA256 differs')
                verify_named(fd, obj['path'], obj['identity'])
                report = dict(file=obj['path'].name, bytes=pin['expected_bytes'], sha256=whole.hexdigest(),
                              tensors=len(obj['spans']), authenticated=True)
                object_reports.append(report)
                event(dict(event='parent_authenticated', **report))
        need(len(matrix_reports) == 282 and len(norm_reports) == 81 and len(object_reports) == 2,
             'Complete matrix/norm/object coverage required')
        need(runtime.common.captured(config_path, 1024**2, source['config_sha256']) == config_raw
             and runtime.common.captured(index_path, 1024**2, source['index_sha256']) == index_raw,
             'Source configuration/index changed')
        for obj in opened:
            verify_named(obj['fd'], obj['path'], obj['identity'])
        final_runtime = endpoint()
        os.fsync(destination)
        need(os.fstat(destination).st_size == planned['file_bytes'], 'Child extent differs')
        need(read_exact(destination, planned['origin'], 0) == planned['prefix'], 'Child header differs')
        before_hash = verify_named(destination, partial)
        child_sha = hash_range(destination, planned['file_bytes'])
        verify_named(destination, partial, before_hash)
        final = output / 'model.safetensors'
        os.link(partial, final, follow_symlinks=False)
        need(os.fstat(destination).st_ino == final.lstat().st_ino, 'Published child descriptor differs')
        partial.unlink()
        os.fchmod(destination, 0o444)
        copied = []
        for name in ('generation_config.json', 'tokenizer_config.json', 'chat_template.jinja'):
            if name not in metadata:
                continue
            pin = metadata[name]
            raw = runtime.common.captured(source_dir / name, 2 * 1024**2, pin['sha256'])
            need(len(raw) == pin['bytes'], 'Ancillary metadata extent differs')
            with (output / name).open('xb') as file:
                file.write(raw)
            copied.append(pin)
        config_bytes = (json.dumps(child_config, indent=2) + '\n').encode()
        with (output / 'config.json').open('xb') as file:
            file.write(config_bytes)
        result = dict(format='toklet-granite-mixed-affine-conversion-v1', passed=True,
                      contract_sha256=args.contract_sha256, source_repository=source['repository'],
                      source_revision=source['revision'], source_objects=object_reports,
                      admitted_runtime=admitted, endpoint_origins=final_runtime,
                      child=dict(path=str(final), bytes=planned['file_bytes'], sha256=child_sha,
                                 header_sha256=hashlib.sha256(planned['prefix']).hexdigest(),
                                 config_sha256=hashlib.sha256(config_bytes).hexdigest()),
                      matrices=matrix_reports, exact_norms=norm_reports, ancillary_metadata=copied,
                      matrix_count=282, norm_count=81, tiles=tile_count,
                      independently_reconstructed_values=reconstructed_values,
                      source_tensor_payloads_consumed=True, model_constructed=False,
                      codec_payloads_qualified=False, model_numerics_quality_fit_performance_adoption_qualified=False)
        raw = (json.dumps(result, indent=2) + '\n').encode()
        with (evidence / 'manifest.json').open('xb') as file:
            file.write(raw)
        with (output / 'toklet-conversion.json').open('xb') as file:
            file.write(raw)
        print(json.dumps({key: result[key] for key in (
            'passed', 'matrix_count', 'norm_count', 'tiles', 'independently_reconstructed_values',
            'model_constructed', 'codec_payloads_qualified', 'model_numerics_quality_fit_performance_adoption_qualified')}))
        return 0
    finally:
        if destination is not None:
            os.close(destination)
        for obj in opened:
            os.close(obj['fd'])


if __name__ == '__main__':
    raise SystemExit(main())
