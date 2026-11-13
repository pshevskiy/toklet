"""Bounded selected Ministral BF16-to-mixed-affine text conversion.

The complete two-object parent, including all excluded vision/projector bytes,
is authenticated. Every matrix tile has an independent NumPy FP32 reconstruction
and completed upstream FP32 witness. No model is constructed or loaded.
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

import granite_convert as generic
import granite_quant_runtime as runtime
import ministral3_convert_layout as layout
import ministral3_quantize as quant
import ministral3_quant_probe as probe

need = quant.need
BLOCK = generic.BLOCK
read_exact = generic.read_exact
hash_range = generic.hash_range
verify_named = generic.verify_named


def sources():
    """Pin the producer and complete transitive generic/new helper/test closure."""
    folder = Path(__file__).resolve().parent
    return sorted({*generic.sources(), *probe.sources(), *(folder / name for name in (
        'ministral3_convert.py', 'test_ministral3_convert.py',
        'ministral3_convert_layout.py', 'test_ministral3_convert_layout.py'))})


def strict_equal(a, b):
    return json.dumps(a, sort_keys=True, separators=(',', ':'), allow_nan=False) == \
        json.dumps(b, sort_keys=True, separators=(',', ':'), allow_nan=False)


def converted_config(config, matrices):
    """Preserve the complete source wrapper; add only the explicit storage recipe."""
    need(type(config) is dict and config.get('model_type') == 'mistral3'
         and config.get('architectures') == ['Mistral3ForConditionalGeneration']
         and 'quantization' not in config and 'quantization_config' not in config,
         'Unquantized selected Ministral wrapper required')
    text = config.get('text_config')
    need(type(text) is dict and text.get('model_type') == 'ministral3'
         and 'quantization' not in text and 'quantization_config' not in text,
         'Selected unquantized Ministral text configuration required')
    need(config.get('dtype') == 'bfloat16' and config.get('torch_dtype', 'bfloat16') == 'bfloat16'
         and text.get('dtype', config['dtype']) == 'bfloat16'
         and text.get('torch_dtype', 'bfloat16') == 'bfloat16'
         and text.get('tie_word_embeddings') is True
         and config.get('tie_word_embeddings', text['tie_word_embeddings']) is True,
         'BF16 and tied source declarations required')
    for name, value in dict(hidden_size=3072, intermediate_size=9216, num_hidden_layers=26,
                            num_attention_heads=32, num_key_value_heads=8, head_dim=128,
                            vocab_size=131072, max_position_embeddings=262144).items():
        need(type(text.get(name)) is int and text[name] == value, 'Selected text geometry differs')
    need(text.get('hidden_act') == 'silu' and text.get('sliding_window', False) is None
         and text.get('rms_norm_eps') == 1e-5 and type(text.get('rms_norm_eps')) in (int, float)
         and text.get('attention_dropout', 0) == 0
         and text.get('attention_bias', False) is False and text.get('mlp_bias', False) is False
         and text.get('rope_scaling') is None and text.get('rope_traditional', False) is False
         and text.get('rope_interleaved', False) is False,
         'Selected normalization/attention/rotary semantics differ')
    parameters = text.get('rope_parameters')
    selected = dict(beta_fast=32.0, beta_slow=1.0, factor=16.0, llama_4_scaling_beta=0.1,
                    mscale=1.0, mscale_all_dim=1.0, original_max_position_embeddings=16384,
                    rope_theta=1000000.0, rope_type='yarn', type='yarn')
    need(type(parameters) is dict and set(parameters) == set(selected), 'Explicit complete YaRN aliases required')
    for name, value in selected.items():
        got = parameters[name]
        need(type(got) is type(value) and got == value, 'Selected YaRN parameter or alias differs')
    need('rope_theta' not in text or (type(text['rope_theta']) in (int, float)
                                     and text['rope_theta'] == 1000000), 'Conflicting flat theta alias')
    expected, _ = layout.selected_recipe()
    need(strict_equal(matrices, expected), 'Exact selected matrix storage recipe required')
    recipe = dict(group_size=64, bits=4, mode='affine')
    for matrix in matrices:
        if matrix['bits'] == 8:
            recipe[matrix['module']] = dict(group_size=64, bits=8, mode='affine')
    result = copy.deepcopy(config)
    result['quantization'] = recipe
    return result


def reserve_output(output, evidence):
    for path in (output, evidence):
        need(not path.exists() and not path.is_symlink(), 'Never replace an existing output or evidence path')
    a, b = output.resolve(), evidence.resolve()
    need(a != b and not a.is_relative_to(b) and not b.is_relative_to(a),
         'Distinct nonnested candidate/evidence directories required')
    output.mkdir(mode=0o700)
    evidence.mkdir(mode=0o700)


def directory_identity(path):
    value = path.lstat()
    need(stat.S_ISDIR(value.st_mode), 'Owned output directory replaced')
    return value.st_dev, value.st_ino


def capture_named(path, maximum, expected):
    """Small metadata capture with descriptor/path/extent checks at both ends."""
    runtime.common.sha(expected)
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        identity = verify_named(fd, path)
        need(0 <= identity[2] <= maximum, 'Bounded metadata extent required')
        raw = read_exact(fd, identity[2], 0)
        need(os.pread(fd, 1, identity[2]) == b'' and hashlib.sha256(raw).hexdigest() == expected,
             'Captured metadata identity differs')
        verify_named(fd, path, identity)
        return raw, identity
    finally:
        os.close(fd)


def write_exclusive(path, raw):
    need(type(raw) is bytes and len(raw) <= 2 * 1024**2, 'Bounded output metadata required')
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        layout.write_all(fd, raw, 0)
        os.fsync(fd)
        identity = verify_named(fd, path)
        need(identity[2] == len(raw) and read_exact(fd, len(raw), 0) == raw,
             'Output metadata readback differs')
        verify_named(fd, path, identity)
        return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                    before_identity=list(identity), after_identity=list(identity))
    finally:
        os.close(fd)


def promote_partial(fd, partial, final, identity):
    """Exclusive same-inode publication; never replace another destination."""
    verify_named(fd, partial, identity)
    os.link(partial, final, follow_symlinks=False)
    owned, named = os.fstat(fd), final.lstat()
    need(stat.S_ISREG(named.st_mode) and (owned.st_dev, owned.st_ino) == (named.st_dev, named.st_ino),
         'Published child descriptor differs')
    partial.unlink()
    return verify_named(fd, final)


def preserve_failed_partial(fd, partial, final):
    """Retain a published owned inode after later manifest finalization fails."""
    if partial.exists() or partial.is_symlink() or not final.exists():
        return False
    owned, named = os.fstat(fd), final.lstat()
    if not stat.S_ISREG(named.st_mode) or (owned.st_dev, owned.st_ino) != (named.st_dev, named.st_ino):
        return False
    os.link(final, partial, follow_symlinks=False)
    return True


def validate_contract(contract):
    need(type(contract) is dict and contract.get('format') == 'toklet-ministral3-heretic-mixed-affine-v1',
         'Frozen selected Ministral contract required')
    recipe = contract['recipe']
    need(recipe['device'] == 'gpu' and recipe['input_dtype'] == recipe['metadata_dtype'] == 'BF16'
         and type(recipe['maximum_rows_per_tile']) is int and recipe['maximum_rows_per_tile'] == quant.ROWS
         and type(recipe['maximum_columns']) is int and recipe['maximum_columns'] == quant.MAX_COLUMNS
         and type(recipe['group_size']) is int and recipe['group_size'] == 64
         and strict_equal(recipe['quantization_runtime'], runtime.VERSIONS),
         'Frozen conversion mechanism differs')
    matrices, norms = layout.selected_recipe()
    inventory = layout.selected_inventory()
    exclusions = [dict(name=name, **item) for name, item in inventory.items()
                  if not name.startswith('language_model.model.')]
    need(strict_equal(recipe['matrices'], matrices) and strict_equal(recipe['norms'], norms)
         and strict_equal(contract['expected_source_inventory'], inventory)
         and strict_equal(contract['source_exclusions'], exclusions)
         and strict_equal(contract['expected_output'], layout.EXPECTED_OUTPUT),
         'Frozen complete source/child inventory differs')
    need(contract['engine_limit_bytes'] == 6 * 1024**3
         and contract['audit_limit_bytes'] == 512 * 1024**2
         and contract['helper_reserve_bytes'] == 32 * 1024**2
         and contract['ordinary_Metal_context_cap'] == 512,
         'Original bounds must remain unchanged')
    runtime.common.sha(contract['source_authentication_receipt_sha256'])
    return recipe, inventory


def validate_authentication(auth, contract):
    need(type(auth) is dict and auth.get('passed') is True
         and type(auth.get('remaining_expected_bytes')) is int and auth['remaining_expected_bytes'] == 0
         and type(auth.get('sources')) is list and len(auth['sources']) == 2,
         'Complete prior source authentication required')
    selected = None
    for source in (contract['original_source'], contract['source']):
        matching = [row for row in auth['sources'] if type(row) is dict
                    and all(row.get(k) == source[k] for k in ('label', 'repository', 'revision', 'directory'))]
        need(len(matching) == 1, 'Exact authenticated source package required')
        row = matching[0]
        need(row.get('tensor_payloads_authenticated') is True and type(row.get('containers')) is list
             and len(row['containers']) == 2 and type(row.get('metadata')) is list,
             'Complete authenticated parent objects required')
        expected = {item['container']: item for item in source['containers']}
        need(len(expected) == 2 and {item['file'] for item in row['containers']} == set(expected),
             'Authenticated parent object coverage differs')
        for item in row['containers']:
            pin = expected[item['file']]
            need(type(item['bytes']) is int and item['bytes'] == pin['expected_bytes']
                 and item['sha256'] == pin['expected_sha256']
                 and item.get('payload_authenticated') is True
                 and item.get('pinned_header_ranges_match') is True,
                 'Authenticated parent object pin differs')
        metadata = {}
        for item in row['metadata']:
            name = layout.leaf(item['file'])
            need(name not in metadata and type(item['bytes']) is int and 0 <= item['bytes'] <= 64 * 1024**2,
                 'Unique bounded authenticated metadata required')
            runtime.common.sha(item['sha256'])
            metadata[name] = item
        for name, pin in [('config.json', source['config_sha256']),
                          ('model.safetensors.index.json', source['index_sha256'])]:
            need(name in metadata and metadata[name]['sha256'] == pin,
                 'Authenticated source metadata pin differs')
        if source is contract['source']:
            selected = metadata
    return selected


def verify_saved_fields(fd, matrices, norms):
    """Close every saved tensor hash before the whole-child hash/promotion."""
    fields = [field for matrix in matrices for field in matrix['fields'].values()]
    fields.extend(norm['child'] for norm in norms)
    for field in fields:
        need(hash_range(fd, field['bytes'], field['offset']) == field['sha256'],
             'Final saved tensor hash differs')


def verify_metadata_records(records):
    for record in records:
        raw, identity = capture_named(Path(record['path']), 2 * 1024**2, record['sha256'])
        need(len(raw) == record['bytes'] and list(identity) == record['after_identity'],
             'Saved configuration/ancillary metadata endpoint differs')


def component_record(planned, name, first, rows, raw):
    item = planned['header'][name]
    width = item['shape'][1]
    stride = (item['data_offsets'][1] - item['data_offsets'][0]) // item['shape'][0]
    return dict(name=name, dtype=item['dtype'], shape=[rows, width],
                offset=planned['origin'] + item['data_offsets'][0] + first * stride,
                bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def source_descriptor(obj, span):
    return dict(source_file=obj['path'].name, source_offset=obj['origin'] + span['start'],
                source_bytes=span['end'] - span['start'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('environment-root', 'runtime-manifest', 'runtime-sha256', 'contract',
                 'contract-sha256', 'source-authentication', 'fresh-output', 'fresh-evidence'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    output, evidence = Path(args.fresh_output), Path(args.fresh_evidence)
    reserve_output(output, evidence)
    directories = [(path, directory_identity(path)) for path in (output, evidence)]
    metadata_inputs = []

    def capture(path, maximum, pin):
        path = Path(path)
        raw, identity = capture_named(path, maximum, pin)
        metadata_inputs.append((path, maximum, pin, raw, identity))
        return raw

    def endpoint_inputs():
        for path, maximum, pin, raw, identity in metadata_inputs:
            actual, after = capture_named(path, maximum, pin)
            need(actual == raw and after == identity, 'Input metadata descriptor/path endpoint differs')
        need(all(directory_identity(path) == identity for path, identity in directories),
             'Output/evidence directory identity changed')

    contract_raw = capture(args.contract, 2 * 1024**2, args.contract_sha256)
    contract = json.loads(contract_raw, object_pairs_hook=runtime.common.json_unique,
                          parse_constant=runtime.common.bad_json_number)
    recipe, inventory = validate_contract(contract)
    source = contract['source']
    auth_raw = capture(args.source_authentication, 2 * 1024**2,
                       contract['source_authentication_receipt_sha256'])
    auth = json.loads(auth_raw, object_pairs_hook=runtime.common.json_unique,
                      parse_constant=runtime.common.bad_json_number)
    metadata = validate_authentication(auth, contract)
    lineage = {}
    for name in ('raw_receipt', 'download_guard', 'acquisition_wrapper'):
        pin = auth.get(name)
        need(type(pin) is dict and type(pin.get('bytes')) is int and pin['bytes'] >= 0,
             'Bound source authentication lineage required')
        path = Path(pin['path'])
        if not path.is_absolute():
            path = Path(args.environment_root) / path
        raw = capture(path, 2 * 1024**2, pin['sha256'])
        need(len(raw) == pin['bytes'], 'Source authentication lineage extent differs')
        lineage[name] = dict(path=str(path), bytes=len(raw), sha256=pin['sha256'])
        if name == 'download_guard':
            need(pin.get('passed') is True, 'Positive download guard required')
    source_dir = Path(source['directory'])
    need(source_dir.is_dir() and not source_dir.is_symlink(), 'Regular selected source directory required')
    source_directory_identity = directory_identity(source_dir)
    config_raw = capture(source_dir / 'config.json', 1024**2, source['config_sha256'])
    index_raw = capture(source_dir / 'model.safetensors.index.json', 1024**2, source['index_sha256'])
    need(len(config_raw) == metadata['config.json']['bytes']
         and len(index_raw) == metadata['model.safetensors.index.json']['bytes'],
         'Authenticated source metadata extent differs')
    config = json.loads(config_raw, object_pairs_hook=runtime.common.json_unique,
                        parse_constant=runtime.common.bad_json_number)
    child_config = converted_config(config, recipe['matrices'])
    planned = layout.output_layout(recipe['matrices'], recipe['norms'])
    need(planned['body_bytes'] == layout.EXPECTED_OUTPUT['body_bytes']
         and planned['file_bytes'] == layout.EXPECTED_OUTPUT['file_bytes']
         and hashlib.sha256(planned['prefix']).hexdigest() == layout.EXPECTED_OUTPUT['header_sha256']
         and len(planned['matrices']) == 183 and len(planned['logical']) == 236
         and len(planned['header']) - 1 == 602, 'Complete frozen output layout differs')
    space = os.statvfs(output)
    need(space.f_bavail * space.f_frsize >= planned['file_bytes'] + 2 * 1024**3 + 256 * 1024**2,
         'Output reservation must preserve disk floor and work reserve')
    runtime_raw = capture(args.runtime_manifest, 2 * 1024**2, args.runtime_sha256)
    opened, destination, completed = [], None, False
    partial, final = output / 'model.safetensors.partial', output / 'model.safetensors'
    try:
        owners = {}
        for pin in source['containers']:
            path = source_dir / layout.leaf(pin['container'])
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
            obj = dict(fd=fd, path=path, pin=pin)
            opened.append(obj)
            obj['identity'] = verify_named(fd, path)
            need(obj['identity'][2] == pin['expected_bytes'], 'Source shard extent differs')
            prefix = read_exact(fd, 8, 0)
            count = struct.unpack('<Q', prefix)[0]
            need(0 < count <= layout.MAX_HEADER and count + 8 == pin['payload_origin'],
                 'Bounded pinned source header extent required')
            raw = read_exact(fd, count, 8)
            ranges = pin['files']
            need(len(ranges) == 2 and ranges[0]['range'] == [0, 7]
                 and ranges[1]['range'] == [8, count + 7]
                 and ranges[0]['bytes'] == len(prefix) and ranges[1]['bytes'] == len(raw)
                 and hashlib.sha256(prefix).hexdigest() == ranges[0]['sha256']
                 and hashlib.sha256(raw).hexdigest() == ranges[1]['sha256'], 'Pinned parent header bytes differ')
            obj.update(prefix=prefix + raw, origin=count + 8, header_sha256=hashlib.sha256(raw).hexdigest())
            obj['spans'] = layout.source_spans(prefix, raw, pin['expected_bytes'], inventory)
            for span in obj['spans']:
                need(span['name'] not in owners, 'Source tensor has multiple shard owners')
                owners[span['name']] = path.name
        total = sum(span['end'] - span['start'] for obj in opened for span in obj['spans'])
        layout.validate_index(index_raw, [(obj['path'].name, obj['spans']) for obj in opened], total, inventory)
        endpoint_inputs()
        runtime.bind_environment(args.environment_root)
        admitted, endpoint = runtime.admit(Path(args.runtime_manifest), args.runtime_sha256, sources())
        import numpy as np
        import mlx.core as mx
        mx.set_cache_limit(0)
        destination = os.open(partial, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        os.ftruncate(destination, planned['file_bytes'])
        layout.write_all(destination, planned['prefix'], 0)
        matrix_reports, norm_reports, exclusions, object_reports = [], [], [], []
        tile_count = reconstructed_values = journal_records = 0
        journal_path = evidence / 'records.jsonl'
        with journal_path.open('x') as journal:
            def event(value):
                nonlocal journal_records
                journal.write(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')
                journal.flush()
                journal_records += 1

            for obj in opened:
                fd, pin = obj['fd'], obj['pin']
                whole = hashlib.sha256(obj['prefix'])
                for span in obj['spans']:
                    name, shape = span['name'], span['shape']
                    descriptor = source_descriptor(obj, span)
                    tensor = hashlib.sha256()
                    if name not in planned['logical']:
                        started = time.monotonic_ns()
                        for first in range(0, descriptor['source_bytes'], BLOCK):
                            raw = read_exact(fd, min(BLOCK, descriptor['source_bytes'] - first),
                                             descriptor['source_offset'] + first)
                            whole.update(raw)
                            tensor.update(raw)
                        report = dict(name=name, dtype='BF16', shape=shape, **descriptor,
                                      source_sha256=tensor.hexdigest(), excluded=True,
                                      reason='vision_or_projector_outside_text_child',
                                      source_read_ns=time.monotonic_ns() - started)
                        exclusions.append(report)
                        event(dict(event='excluded_tensor', parent_object_authenticated=False, **report))
                        del raw
                        continue
                    if len(shape) == 1:
                        started = time.monotonic_ns()
                        raw = read_exact(fd, descriptor['source_bytes'], descriptor['source_offset'])
                        read_ns = time.monotonic_ns() - started
                        words = np.frombuffer(raw, dtype='<u2')
                        need(not np.any((words & 0x7f80) == 0x7f80), 'Nonfinite source norm')
                        whole.update(raw)
                        started = time.monotonic_ns()
                        layout.write_norm(destination, planned, name, raw)
                        write_ns = time.monotonic_ns() - started
                        item = planned['header'][name]
                        digest = hashlib.sha256(raw).hexdigest()
                        child = dict(name=name, dtype='BF16', shape=shape,
                                     offset=planned['origin'] + item['data_offsets'][0],
                                     bytes=len(raw), sha256=digest)
                        report = dict(name=name, dtype='BF16', shape=shape, **descriptor,
                                      source_sha256=digest, bytes=len(raw), copied_BF16_sha256=digest,
                                      child=child, norm_bytes_equal=True, finite_text_input=True,
                                      source_read_ns=read_ns, component_write_readback_ns=write_ns)
                        norm_reports.append(report)
                        event(dict(event='norm_copied', parent_object_authenticated=False, **report))
                        del raw, words
                        continue
                    plan = planned['matrices'][name]
                    field_hashes = {key: hashlib.sha256() for key in ('weight', 'scales', 'biases')}
                    oracle_hash, upstream_hash = hashlib.sha256(), hashlib.sha256()
                    matrix_tiles = 0
                    for first in range(0, shape[0], quant.ROWS):
                        rows = min(quant.ROWS, shape[0] - first)
                        offset = descriptor['source_offset'] + first * shape[1] * 2
                        started = time.monotonic_ns()
                        raw = read_exact(fd, rows * shape[1] * 2, offset)
                        read_ns = time.monotonic_ns() - started
                        whole.update(raw)
                        tensor.update(raw)
                        started = time.monotonic_ns()
                        buffers = quant.quantize(raw, rows, shape[1], plan['bits'], 'gpu', np, mx)
                        quantize_ns = time.monotonic_ns() - started
                        started = time.monotonic_ns()
                        independent = quant.reconstruct_f32(buffers, rows, shape[1], plan['bits'], np)
                        independent_ns = time.monotonic_ns() - started
                        started = time.monotonic_ns()
                        upstream = probe.upstream_f32(buffers, rows, shape[1], plan['bits'], 'gpu', np, mx)
                        upstream_ns = time.monotonic_ns() - started
                        equal = independent == upstream
                        fields = {key: component_record(planned, item['name'], first, rows, buffers[key])
                                  for key, item in plan['fields'].items()}
                        tile_record = dict(event='matrix_tile', parent_object_authenticated=False, name=name,
                                   first_row=first, rows=rows, columns=shape[1], bits=plan['bits'],
                                   source_file=obj['path'].name, source_offset=offset,
                                   source_bytes=len(raw), source_sha256=hashlib.sha256(raw).hexdigest(),
                                   fields=fields, independent_F32_sha256=hashlib.sha256(independent).hexdigest(),
                                   upstream_F32_sha256=hashlib.sha256(upstream).hexdigest(),
                                   reconstructed_values=rows * shape[1], reconstruction_equal=equal,
                                   source_read_ns=read_ns, completed_quantize_readback_ns=quantize_ns,
                                   independent_decode_ns=independent_ns, completed_upstream_decode_ns=upstream_ns,
                                   component_write_readback_ns=0, component_readback=False)
                        if not equal:
                            event(tile_record)
                            need(False, 'Independent FP32 reconstruction differs')
                        started = time.monotonic_ns()
                        try:
                            layout.write_matrix_tile(destination, planned, name, first, rows, buffers)
                        except BaseException:
                            tile_record['component_write_readback_ns'] = time.monotonic_ns() - started
                            event(tile_record)
                            raise
                        tile_record['component_write_readback_ns'] = time.monotonic_ns() - started
                        tile_record['component_readback'] = True
                        event(tile_record)
                        for key, value in buffers.items():
                            field_hashes[key].update(value)
                        oracle_hash.update(independent)
                        upstream_hash.update(upstream)
                        matrix_tiles += 1
                        tile_count += 1
                        reconstructed_values += rows * shape[1]
                        del raw, buffers, independent, upstream
                    fields = {}
                    for key, item in plan['fields'].items():
                        declared = planned['header'][item['name']]
                        start, end = declared['data_offsets']
                        stored = hash_range(destination, end - start, planned['origin'] + start)
                        need(stored == field_hashes[key].hexdigest(), 'Complete stored matrix component hash differs')
                        fields[key] = dict(name=item['name'], dtype=declared['dtype'], shape=declared['shape'],
                                           offset=planned['origin'] + start, bytes=end - start, sha256=stored)
                    report = dict(name=name, shape=shape, bits=plan['bits'], **descriptor,
                                  source_sha256=tensor.hexdigest(), source_tensor_sha256=tensor.hexdigest(),
                                  fields=fields, independently_reconstructed_F32_sha256=oracle_hash.hexdigest(),
                                  upstream_F32_sha256=upstream_hash.hexdigest(),
                                  reconstructed_values=shape[0] * shape[1], tiles=matrix_tiles,
                                  reconstruction_equal=True)
                    matrix_reports.append(report)
                    event(dict(event='matrix_completed', parent_object_authenticated=False, **report))
                need(os.pread(fd, 1, pin['expected_bytes']) == b'', 'Trailing source bytes')
                need(whole.hexdigest() == pin['expected_sha256'], 'Complete consumed parent SHA256 differs')
                after = verify_named(fd, obj['path'], obj['identity'])
                report = dict(file=obj['path'].name, path=str(obj['path']), bytes=pin['expected_bytes'],
                              sha256=whole.hexdigest(), tensors=len(obj['spans']), authenticated=True,
                              prefix_bytes=obj['origin'], prefix_sha256=hashlib.sha256(obj['prefix']).hexdigest(),
                              header_bytes=obj['origin'] - 8, header_sha256=obj['header_sha256'],
                              before_identity=list(obj['identity']), after_identity=list(after))
                object_reports.append(report)
                event(dict(event='parent_authenticated', **report))
            os.fsync(journal.fileno())
            journal_identity_after_write = verify_named(journal.fileno(), journal_path)
        need(len(matrix_reports) == 183 and len(norm_reports) == 53 and len(exclusions) == 222
             and len(object_reports) == 2 and tile_count == 4252 and reconstructed_values == 3428843520,
             'Complete matrix/norm/exclusion/object/tile/value coverage required')
        config_bytes = (json.dumps(child_config, indent=2, allow_nan=False) + '\n').encode()
        config_record = write_exclusive(output / 'config.json', config_bytes)
        copied = []
        for name in ('generation_config.json', 'tokenizer_config.json', 'chat_template.jinja'):
            if name in metadata:
                pin = metadata[name]
                raw = capture(source_dir / name, 2 * 1024**2, pin['sha256'])
                need(len(raw) == pin['bytes'], 'Ancillary metadata extent differs')
                copied.append(dict(file=name, source_path=str(source_dir / name),
                                   **write_exclusive(output / name, raw)))
                del raw
        endpoint_inputs()
        verify_metadata_records([config_record, *copied])
        need(directory_identity(source_dir) == source_directory_identity, 'Source directory replaced')
        for obj in opened:
            verify_named(obj['fd'], obj['path'], obj['identity'])
        verify_saved_fields(destination, matrix_reports, norm_reports)
        os.fchmod(destination, 0o444)
        os.fsync(destination)
        need(os.fstat(destination).st_size == planned['file_bytes'], 'Child extent differs')
        need(read_exact(destination, planned['origin'], 0) == planned['prefix'], 'Child header differs')
        before_hash = verify_named(destination, partial)
        child_sha = hash_range(destination, planned['file_bytes'])
        verify_named(destination, partial, before_hash)
        endpoint_inputs()
        verify_metadata_records([config_record, *copied])
        for obj in opened:
            verify_named(obj['fd'], obj['path'], obj['identity'])
        final_runtime = endpoint()
        need(all(directory_identity(path) == identity for path, identity in directories),
             'Output/evidence directory replaced before publication')
        child_identity = promote_partial(destination, partial, final, before_hash)
        journal_fd = os.open(journal_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            journal_identity = verify_named(journal_fd, journal_path, journal_identity_after_write)
            journal_pin = dict(path=str(journal_path), bytes=journal_identity[2],
                               sha256=hash_range(journal_fd, journal_identity[2]), records=journal_records)
            verify_named(journal_fd, journal_path, journal_identity)
        finally:
            os.close(journal_fd)
        need(verify_named(destination, final) == child_identity, 'Published child endpoint differs')
        result = dict(format='toklet-ministral3-mixed-affine-conversion-v1', passed=True,
                      contract_sha256=args.contract_sha256,
                      source_authentication=dict(path=str(Path(args.source_authentication)), bytes=len(auth_raw),
                                                 sha256=contract['source_authentication_receipt_sha256']),
                      source_authentication_lineage=lineage,
                      runtime=dict(path=str(Path(args.runtime_manifest)), bytes=len(runtime_raw), sha256=args.runtime_sha256),
                      source_repository=source['repository'], source_revision=source['revision'],
                      source_objects=object_reports, admitted_runtime=admitted, endpoint_origins=final_runtime,
                      journal=journal_pin,
                      child=dict(path=str(final), bytes=planned['file_bytes'], sha256=child_sha,
                                 header_sha256=hashlib.sha256(planned['prefix']).hexdigest(),
                                 config_sha256=config_record['sha256'], config=config_record,
                                 before_identity=list(child_identity), after_identity=list(child_identity)),
                      matrices=matrix_reports, exact_norms=norm_reports, exclusions=exclusions,
                      ancillary_metadata=copied, matrix_count=183, norm_count=53, excluded_count=222,
                      tiles=tile_count, independently_reconstructed_values=reconstructed_values,
                      complete_positive_predicates=dict(source_authentication=True, component_readback=True,
                          independent_upstream_f32_equal=True, finite_text_input=True, finite_metadata=True,
                          finite_reconstruction=True, norm_bytes_equal=True, endpoint_unchanged=True),
                      source_tensor_payloads_consumed=True, model_constructed=False,
                      codec_payloads_qualified=False, model_numerics_quality_fit_performance_adoption_qualified=False)
        manifest = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
        write_exclusive(evidence / 'manifest.json', manifest)
        write_exclusive(output / 'toklet-conversion.json', manifest)
        completed = True
        print(json.dumps({key: result[key] for key in ('passed', 'matrix_count', 'norm_count', 'excluded_count',
                         'tiles', 'independently_reconstructed_values', 'model_constructed',
                         'codec_payloads_qualified', 'model_numerics_quality_fit_performance_adoption_qualified')}))
        return 0
    finally:
        try:
            if destination is not None and not completed:
                preserve_failed_partial(destination, partial, final)
        finally:
            if destination is not None:
                os.close(destination)
            for obj in opened:
                os.close(obj['fd'])


if __name__ == '__main__':
    raise SystemExit(main())
