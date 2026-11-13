"""Portable saved-byte controls: stdlib, tracked metadata and tiny files only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import ministral3_convert_audit as audit

FIXTURES = Path(__file__).resolve().parents[2] / 'tests/fixtures/ministral3-conversion-v1'
SHA = 'a' * 64


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture_recipe():
    inventory = {}
    for number in (1, 2):
        stem = f'ministral_modified-model-0000{number}-of-00002.safetensors'
        header = audit.loads((FIXTURES / (stem + '-header.bin')).read_bytes())
        inventory.update({name: dict(dtype=row['dtype'], shape=row['shape'])
                          for name, row in header.items() if name != '__metadata__'})
    matrices, norms, exclusions = [], [], []
    for name, row in sorted(inventory.items()):
        if not name.startswith('language_model.model.'):
            exclusions.append(dict(name=name, **row))
        elif len(row['shape']) == 2:
            module = name.removesuffix('.weight')
            matrices.append(dict(module=module, shape=row['shape'],
                                 bits=8 if module.endswith(('.embed_tokens', '.mlp.down_proj', '.self_attn.o_proj')) else 4,
                                 group_size=64, mode='affine'))
        else:
            norms.append(dict(name=name, **row, bytes=row['shape'][0] * 2))
    return inventory, matrices, norms, exclusions


def contract():
    inventory, matrices, norms, exclusions = fixture_recipe()
    return dict(format='toklet-ministral3-heretic-mixed-affine-v1',
                source_authentication_receipt_sha256=SHA, expected_output=dict(audit.EXPECTED),
                recipe=dict(device='gpu', input_dtype='BF16', metadata_dtype='BF16',
                            maximum_rows_per_tile=224, maximum_columns=9216, group_size=64,
                            quantization_runtime=dict(audit.VERSIONS), matrices=matrices, norms=norms),
                expected_source_inventory=inventory, source_exclusions=exclusions,
                audit_limit_bytes=512 * 1024 ** 2, helper_reserve_bytes=32 * 1024 ** 2,
                engine_limit_bytes=6 * 1024 ** 3, ordinary_Metal_context_cap=512,
                model_full_graph_quality_codec_fit_request_performance_adoption_qualified=False,
                whole_goal_complete=False, numerical_outputs_exist=False)


class AuditControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.journals = 0

    def file(self, name='bytes', raw=b'0123456789'):
        path = self.root / name
        path.write_bytes(raw)
        return path

    def test_import_is_stdlib_and_numerical_cold(self):
        script = ("import sys; from importlib.abc import MetaPathFinder; "
                  "exec('class Deny(MetaPathFinder):\\n def find_spec(self,name,*args):\\n  "
                  "assert name.split(\".\")[0] not in (\"numpy\",\"mlx\",\"torch\"),name'); "
                  "sys.meta_path.insert(0,Deny()); import ministral3_convert_audit; "
                  "assert not any(n.split('.')[0] in ('numpy','mlx','torch') for n in sys.modules)")
        result = subprocess.run([sys.executable, '-B', '-s', '-c', script],
                                cwd=Path(audit.__file__).parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_tracked_metadata_produces_frozen_child_prefix(self):
        planned, inventory = audit.contract_layout(contract())
        self.assertEqual(len(inventory), 458)
        self.assertEqual(len(planned['matrices']), 183)
        self.assertEqual(len(planned['header']) - 1, 602)
        self.assertEqual(digest(planned['prefix']), audit.EXPECTED['header_sha256'])
        self.assertEqual(planned['file_bytes'], 2662082312)
        self.assertEqual(planned['body_bytes'], 2662004736)

    def test_production_contract_cannot_be_scaled_down(self):
        for alter in (lambda c: c['recipe']['matrices'].pop(),
                      lambda c: c['recipe']['norms'].pop(),
                      lambda c: c['source_exclusions'].pop(),
                      lambda c: c['expected_output'].update(matrix_tiles=1),
                      lambda c: c.update(source_authentication_receipt_sha256=None),
                      lambda c: c['recipe'].update(maximum_rows_per_tile=256),
                      lambda c: c.update(audit_limit_bytes=6 * 1024 ** 3)):
            value = contract()
            alter(value)
            with self.assertRaises(ValueError):
                audit.contract_layout(value)

    def test_bool_float_integer_confusion_rejects(self):
        for invalid in (True, False, 1.0, -1, '1', None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                audit.uint(invalid)
        for actual, expected in ((True, 1), (1.0, 1), ([True], [1]), ({'x': 0}, {'x': False})):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                audit.same(actual, expected, 'strict')
        value = contract()
        value['recipe']['group_size'] = 64.0
        with self.assertRaises(ValueError):
            audit.contract_layout(value)

    def test_json_duplicates_and_nonfinite_reject(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                audit.loads(raw)

    def test_actual_bounded_capture_hash(self):
        path = self.file()
        self.assertEqual(audit.capture(path, 10, digest(b'0123456789')), b'0123456789')
        for maximum, expected in ((9, None), (10, SHA)):
            with self.assertRaises(ValueError):
                audit.capture(path, maximum, expected)

    def test_hash_reads_no_more_than_one_mib(self):
        raw = b'x' * (audit.HASH_BLOCK * 2 + 37)
        path = self.file(raw=raw)
        calls = []
        original = os.pread
        def read(fd, count, offset):
            calls.append(count)
            return original(fd, count, offset)
        with audit.SavedFile(path) as saved, patch.object(os, 'pread', read):
            self.assertEqual(saved.digest(), digest(raw))
        self.assertLessEqual(max(calls), audit.HASH_BLOCK)
        self.assertEqual(sum(calls), len(raw))

    def test_tile_bound_rejects_before_read(self):
        with audit.SavedFile(self.file()) as saved:
            for count, offset in ((audit.TILE_BLOCK + 1, 0), (1, -1), (True, 0), (20, 0)):
                with self.subTest(count=count), self.assertRaises(ValueError):
                    saved.read(count, offset)

    def test_short_reads_complete_but_empty_read_rejects(self):
        path = self.file()
        original = os.pread
        with audit.SavedFile(path) as saved:
            with patch.object(os, 'pread', side_effect=lambda fd, count, offset: original(fd, min(count, 2), offset)):
                self.assertEqual(saved.read(10), b'0123456789')
            with patch.object(os, 'pread', return_value=b''):
                with self.assertRaises(ValueError):
                    saved.read(1)

    def test_symlink_file_and_nonregular_file_reject(self):
        path = self.file()
        link = self.root / 'symlink'
        link.symlink_to(path)
        with self.assertRaises((OSError, ValueError)):
            audit.SavedFile(link)
        with self.assertRaises(ValueError):
            audit.SavedFile(self.root)
        fifo = self.root / 'fifo'
        os.mkfifo(fifo)
        with self.assertRaises(ValueError):
            audit.SavedFile(fifo)

    def test_path_replacement_and_file_mutation_reject(self):
        for action in ('replace', 'mutate', 'grow', 'truncate'):
            path = self.file(action)
            saved = audit.SavedFile(path)
            if action == 'replace':
                replacement = self.file('replacement', b'0123456789')
                replacement.replace(path)
            elif action == 'mutate':
                with path.open('r+b') as stream:
                    stream.write(b'X')
            elif action == 'grow':
                with path.open('ab') as stream:
                    stream.write(b'X')
            else:
                path.write_bytes(b'X')
            with self.assertRaises(ValueError):
                saved.close()

    def test_declared_file_size_rejects_boolean_and_mismatch(self):
        path = self.file()
        for size in (True, 9, 11):
            with self.assertRaises(ValueError):
                audit.SavedFile(path, size)

    def test_endpoint_receipts_require_exact_actual_identity(self):
        with audit.SavedFile(self.file()) as saved:
            receipt = dict(before_identity=saved.before, after_identity=saved.before)
            audit.endpoints(receipt, saved)
            bad = copy.deepcopy(receipt)
            bad['before_identity'][2] = True
            with self.assertRaises(ValueError):
                audit.endpoints(bad, saved)
            bad = copy.deepcopy(receipt)
            bad['after_identity'][1] += 1
            with self.assertRaises(ValueError):
                audit.endpoints(bad, saved)

    def test_tracked_headers_index_and_rank4_exclusion(self):
        inventory, _, _, _ = fixture_recipe()
        objects = []
        for number in (1, 2):
            name = f'model-0000{number}-of-00002.safetensors'
            stem = 'ministral_modified-' + name
            prefix = (FIXTURES / (stem + '-prefix.bin')).read_bytes()
            raw = (FIXTURES / (stem + '-header.bin')).read_bytes()
            header = audit.loads(raw)
            size = len(raw) + 8 + max(row['data_offsets'][1] for key, row in header.items() if key != '__metadata__')
            spans = audit.source_spans(prefix, raw, size, inventory)
            objects.append((name, spans))
        self.assertTrue(any(len(row['shape']) == 4 for _, spans in objects for row in spans))
        index = (FIXTURES / 'ministral_modified-model.safetensors.index.json').read_bytes()
        total = sum(row['end'] - row['start'] for _, spans in objects for row in spans)
        audit.validate_index(index, objects, total)
        parsed = audit.loads(index)
        parsed['metadata']['total_size'] -= 2
        with self.assertRaises(ValueError):
            audit.validate_index(json.dumps(parsed), objects, total)
        with self.assertRaises(ValueError):
            audit.validate_index(index, [objects[0], objects[0]], total)

    def test_stock_index_rejects_inconsistent_parameter_count_without_repair(self):
        inventory, _, _, _ = fixture_recipe()
        objects = []
        for number in (1, 2):
            name = f'model-0000{number}-of-00002.safetensors'
            stem = 'ministral_original-' + name
            prefix = (FIXTURES / (stem + '-prefix.bin')).read_bytes()
            raw = (FIXTURES / (stem + '-header.bin')).read_bytes()
            header = audit.loads(raw)
            size = len(raw) + 8 + max(row['data_offsets'][1] for key, row in header.items() if key != '__metadata__')
            objects.append((name, audit.source_spans(prefix, raw, size, inventory)))
        raw = (FIXTURES / 'ministral_original-model.safetensors.index.json').read_bytes()
        total = sum(row['end'] - row['start'] for _, spans in objects for row in spans)
        self.assertEqual(total, 7698180096)
        self.assertNotEqual(audit.loads(raw)['metadata']['total_parameters'] * 2, total)
        with self.assertRaisesRegex(ValueError, 'parameter count'):
            audit.validate_index(raw, objects, total)

    def test_source_span_holes_overlap_wrong_shape_dtype_unknown(self):
        inventory = {'a': dict(dtype='BF16', shape=[2]), 'b': dict(dtype='BF16', shape=[1, 1, 1, 2])}
        header = {'a': dict(dtype='BF16', shape=[2], data_offsets=[0, 4]),
                  'b': dict(dtype='BF16', shape=[1, 1, 1, 2], data_offsets=[4, 8])}
        def parse(value):
            raw = json.dumps(value).encode()
            return audit.source_spans(struct.pack('<Q', len(raw)), raw, len(raw) + 16, inventory)
        self.assertEqual(len(parse(header)), 2)
        for change in (dict(data_offsets=[5, 9]), dict(data_offsets=[3, 7]),
                       dict(shape=[4]), dict(dtype='F32'), dict(shape=[True])):
            changed = copy.deepcopy(header)
            changed['b'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse(changed)
        changed = dict(header, unknown=header['b'])
        with self.assertRaises(ValueError):
            parse(changed)

    def test_component_full_and_tile_actual_byte_hashes(self):
        raw = bytes(range(64))
        declaration = dict(dtype='U32', shape=[4, 2], data_offsets=[16, 48])
        with audit.SavedFile(self.file(raw=raw)) as saved:
            full = audit.component(saved, declaration)
            tile = audit.component(saved, declaration, 1, 2)
        self.assertEqual(full, dict(dtype='U32', shape=[4, 2], offset=16, bytes=32, sha256=digest(raw[16:48])))
        self.assertEqual(tile, dict(dtype='U32', shape=[2, 2], offset=24, bytes=16, sha256=digest(raw[24:40])))

    def journal(self, events, count=None, suffix=b''):
        raw = b''.join(json.dumps(event).encode() + b'\n' for event in events) + suffix
        self.journals += 1
        path = self.file('journal' + str(self.journals), raw)
        saved = audit.SavedFile(path)
        self.addCleanup(saved.close)
        return audit.Journal(saved, dict(records=len(events) if count is None else count, sha256=digest(raw)))

    def test_journal_helper_positive_exact_order_and_completion(self):
        rows = [dict(event='norm_copied', parent_object_authenticated=False, name='n'),
                dict(event='parent_authenticated', authenticated=True)]
        journal = self.journal(rows)
        journal.matches('norm_copied', dict(name='n'))
        journal.matches('parent_authenticated', dict(authenticated=True))
        journal.finish()

    def test_journal_missing_reordered_duplicate_extra_reject(self):
        row = dict(event='norm_copied', parent_object_authenticated=False, name='n')
        journal = self.journal([row, row])
        journal.matches('norm_copied', dict(name='n'))
        with self.assertRaises(ValueError):
            journal.finish()
        journal.matches('norm_copied', dict(name='n'))
        journal.finish()
        journal = self.journal([row])
        with self.assertRaises(ValueError):
            journal.take('matrix_tile')
        journal = self.journal([row])
        journal.take('norm_copied')
        with self.assertRaises(ValueError):
            journal.take('parent_authenticated')
        journal = self.journal([row], count=2)
        journal.take('norm_copied')
        with self.assertRaises(ValueError):
            journal.finish()

    def test_journal_premature_parent_and_invalid_termination_reject(self):
        for value in (True, 0, None):
            journal = self.journal([dict(event='matrix_tile', parent_object_authenticated=value)])
            with self.assertRaises(ValueError):
                journal.take('matrix_tile')
        journal = self.journal([], suffix=b'{"event":"matrix_tile"}')
        with self.assertRaises(ValueError):
            journal.take('matrix_tile')
        journal = self.journal([], suffix=b'x' * 65536 + b'\n')
        with self.assertRaises(ValueError):
            journal.take('matrix_tile')

    def test_journal_manifest_mismatch_rejects(self):
        journal = self.journal([dict(event='norm_copied', parent_object_authenticated=False, name='wrong')])
        with self.assertRaises(ValueError):
            journal.matches('norm_copied', dict(name='correct'))

    def test_witness_equal_hashes_and_exact_count_clocks(self):
        row = dict(reconstruction_equal=True, reconstructed_values=128,
                   independent_F32_sha256=SHA, upstream_F32_sha256=SHA,
                   source_read_ns=1, completed_quantize_readback_ns=2,
                   independent_decode_ns=3, completed_upstream_decode_ns=4)
        audit.witness(row, 2, 64)
        for changed in (dict(reconstruction_equal=1), dict(upstream_F32_sha256='b' * 64),
                        dict(reconstructed_values=127), dict(reconstructed_values=128.0),
                        dict(source_read_ns=True), dict(independent_decode_ns=-1)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                audit.witness(dict(row, **changed), 2, 64)

    def test_duplicate_missing_unknown_manifest_reports_reject(self):
        good = [dict(name='a'), dict(name='b')]
        self.assertEqual(set(audit.reports(good, {'a', 'b'})), {'a', 'b'})
        for value in (good[:1], [good[0], good[0]], good + [dict(name='c')], [good[0], dict(name='c')]):
            with self.assertRaises(ValueError):
                audit.reports(value, {'a', 'b'})

    def test_config_preserves_all_source_semantics_and_yarn_aliases(self):
        raw = (FIXTURES / 'ministral_modified-config.json').read_bytes()
        config = audit.loads(raw)
        _, matrices, _, _ = fixture_recipe()
        converted = audit.child_config(config, matrices)
        self.assertEqual({k: v for k, v in converted.items() if k != 'quantization'}, config)
        self.assertEqual(len(converted['quantization']) - 3, 53)
        self.assertEqual(converted['quantization']['bits'], 4)
        with self.assertRaises(ValueError):
            audit.child_config(converted, matrices)

    def tile_fixture(self):
        matrices = [dict(module='language_model.model.x', shape=[3, 64], bits=4, group_size=64, mode='affine')]
        planned = audit.output_layout(matrices, [])
        source_raw = bytes(range(128)) * 3
        child_raw = planned['prefix'] + bytes(range(planned['body_bytes']))
        source_path = self.file('tile-source', source_raw)
        child_path = self.file('tile-child', child_raw)
        span = dict(name='language_model.model.x.weight', shape=[3, 64], start=0, end=384)
        plan = planned['matrices'][span['name']]
        fields = {}
        with audit.SavedFile(child_path) as child:
            for key, field in plan['fields'].items():
                declaration = dict(planned['header'][field['name']])
                declaration['data_offsets'] = [value + planned['origin'] for value in declaration['data_offsets']]
                fields[key] = dict(name=field['name'], **audit.component(child, declaration, 1, 2))
        tile = dict(event='matrix_tile', parent_object_authenticated=False,
                    name=span['name'], first_row=1, rows=2, columns=64, bits=4,
                    source_file='part', source_offset=128, source_bytes=256,
                    source_sha256=digest(source_raw[128:]), fields=fields,
                    independent_F32_sha256=SHA, upstream_F32_sha256=SHA,
                    reconstructed_values=128, reconstruction_equal=True,
                    source_read_ns=1, completed_quantize_readback_ns=1,
                    independent_decode_ns=1, completed_upstream_decode_ns=1,
                    component_write_readback_ns=1, component_readback=True)
        return source_path, child_path, span, plan, planned, tile

    def test_saved_tile_positive_and_adversarial_controls(self):
        source_path, child_path, span, plan, planned, tile = self.tile_fixture()
        with audit.SavedFile(source_path) as source, audit.SavedFile(child_path) as child:
            self.assertEqual(audit.check_tile(tile, source, child, span, plan, planned, 'part', 0, 1, 2), 128)
            for changed in (dict(first_row=0), dict(rows=True), dict(source_file='wrong'),
                            dict(source_offset=0), dict(source_sha256='b' * 64),
                            dict(upstream_F32_sha256='b' * 64), dict(component_readback=False),
                            dict(parent_object_authenticated=True), dict(extra='unexpected')):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    audit.check_tile(dict(tile, **changed), source, child, span, plan, planned, 'part', 0, 1, 2)
            bad = copy.deepcopy(tile)
            bad['fields']['weight']['sha256'] = 'b' * 64
            with self.assertRaises(ValueError):
                audit.check_tile(bad, source, child, span, plan, planned, 'part', 0, 1, 2)
            bad = copy.deepcopy(tile)
            bad['fields']['weight']['offset'] += 4
            with self.assertRaises(ValueError):
                audit.check_tile(bad, source, child, span, plan, planned, 'part', 0, 1, 2)

    def test_actual_saved_tile_corruption_rejects(self):
        for which in ('source', 'child'):
            source_path, child_path, span, plan, planned, tile = self.tile_fixture()
            target = source_path if which == 'source' else child_path
            raw = bytearray(target.read_bytes())
            raw[-1] ^= 1
            target.write_bytes(raw)
            with audit.SavedFile(source_path) as source, audit.SavedFile(child_path) as child:
                with self.assertRaises(ValueError):
                    audit.check_tile(tile, source, child, span, plan, planned, 'part', 0, 1, 2)

    def authentication_fixture(self):
        sources, receipts = [], []
        for number in (1, 2):
            source = dict(repository='repo' + str(number), revision='rev', directory='/source' + str(number),
                          config_sha256=SHA, index_sha256='b' * 64,
                          containers=[dict(container='part1', expected_bytes=10, expected_sha256=SHA),
                                      dict(container='part2', expected_bytes=20, expected_sha256='c' * 64)])
            sources.append(source)
            receipts.append(dict(repository=source['repository'], revision='rev', directory=source['directory'],
                tensor_payloads_authenticated=True,
                metadata=[dict(file='config.json', bytes=2, sha256=SHA),
                          dict(file='model.safetensors.index.json', bytes=2, sha256='b' * 64)],
                containers=[dict(file=row['container'], bytes=row['expected_bytes'], sha256=row['expected_sha256'],
                                 pinned_header_ranges_match=True, payload_authenticated=True)
                            for row in source['containers']]))
        return dict(passed=True, remaining_expected_bytes=0, sources=receipts), dict(original_source=sources[0], source=sources[1])

    def test_prior_authentication_requires_both_complete_sources(self):
        authentication, frozen = self.authentication_fixture()
        self.assertEqual(audit.source_authentication(authentication, frozen), authentication['sources'][1])
        for mutate in (lambda a: a.update(passed=1), lambda a: a.update(remaining_expected_bytes=False),
                       lambda a: a['sources'].pop(), lambda a: a['sources'][1]['containers'].pop(),
                       lambda a: a['sources'][0].update(tensor_payloads_authenticated=False),
                       lambda a: a['sources'][0]['containers'][0].update(payload_authenticated=False),
                       lambda a: a['sources'][1]['containers'][1].update(bytes=True),
                       lambda a: a['sources'][1]['metadata'][0].update(sha256='d' * 64)):
            bad = copy.deepcopy(authentication)
            mutate(bad)
            with self.assertRaises((ValueError, KeyError)):
                audit.source_authentication(bad, frozen)

    def source_guard_fixture(self):
        reservation = audit.EXPECTED['file_bytes'] + 2 * 1024 ** 3 + 256 * audit.HASH_BLOCK
        return dict(passed=True, failure=None, exit_code=0,
                    maximum_observed_charge_with_reserve=100 * audit.HASH_BLOCK,
                    terminal_direct_pair_rss_charge_with_reserve=80 * audit.HASH_BLOCK,
                    minimum_free_bytes=4 * 1024 ** 3, sample_count=10,
                    owned_live_after_cleanup=[], cleanup_errors=[],
                    last_runtime_gate=dict(passed=True, remaining_download_bytes=0,
                        child_bytes_reserved=audit.EXPECTED['file_bytes'], disk_floor_bytes=2 * 1024 ** 3,
                        work_reserve_bytes=256 * audit.HASH_BLOCK,
                        required_remaining_bytes=reservation, free_bytes=reservation + audit.HASH_BLOCK))

    def test_source_guard_positive_and_adversarial_controls(self):
        guard = self.source_guard_fixture()
        audit.source_guard(guard)
        for changed in (dict(passed=1), dict(exit_code=True), dict(sample_count=0),
                        dict(maximum_observed_charge_with_reserve=512 * audit.HASH_BLOCK + 1),
                        dict(minimum_free_bytes=2 * 1024 ** 3 - 1), dict(owned_live_after_cleanup=[1]),
                        dict(cleanup_errors=['failed'])):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                audit.source_guard(dict(guard, **changed))
        for changed in (dict(remaining_download_bytes=True), dict(work_reserve_bytes=0),
                        dict(child_bytes_reserved=1), dict(free_bytes=0), dict(required_remaining_bytes=1)):
            bad = copy.deepcopy(guard)
            bad['last_runtime_gate'].update(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                audit.source_guard(bad)

    def closure_fixture(self):
        authentication, frozen = self.authentication_fixture()
        authentication['format'] = 'toklet-ministral3-complete-source-authentication-v1'
        for index, (source, receipt) in enumerate(zip((frozen['original_source'], frozen['source']), authentication['sources'])):
            source['directory'] = str(self.root / 'models' / ('source' + str(index)))
            receipt['directory'] = source['directory']
            for number, container in enumerate(receipt['containers']):
                relative = f'artifacts/source{index}-part{number}/authenticated.json'
                container['download_result'] = relative
                container['final_identity'] = [1, 2 + index * 2 + number, container['bytes'], 4, 5]
                path = self.root / relative
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(container))
        raw = json.dumps(authentication).encode()
        raw_path = self.file('raw-source.json', raw)
        guard = self.source_guard_fixture()
        guard_raw = json.dumps(guard).encode()
        guard_path = self.file('download-guard.json', guard_raw)
        code = self.file('acquisition-code.py', b'# bounded source acquisition\n')
        code_lease = {str(code): digest(code.read_bytes())}
        wrapper_raw = json.dumps(dict(passed=True, guard=guard,
            sources_before=code_lease, sources_after=code_lease,
            numerical_runtime_or_model_requested=False)).encode()
        wrapper_path = self.file('acquisition-wrapper.json', wrapper_raw)
        closure = dict(authentication, repository_root=str(self.root),
                       acquisition_wrapper=dict(path=str(wrapper_path), bytes=len(wrapper_raw), sha256=digest(wrapper_raw)),
                       raw_receipt=dict(path=str(raw_path), bytes=len(raw), sha256=digest(raw)),
                       download_guard=dict(path=str(guard_path), bytes=len(guard_raw), sha256=digest(guard_raw), passed=True))
        return closure, frozen

    def test_closed_source_receipt_positive_and_adversarial_controls(self):
        closure, frozen = self.closure_fixture()
        self.assertEqual(audit.source_closure(closure, frozen), closure['sources'][1])
        for mutate in (lambda a: a['raw_receipt'].update(bytes=True),
                       lambda a: a['download_guard'].update(passed=1),
                       lambda a: a['download_guard'].update(sha256='b' * 64),
                       lambda a: a.update(repository_root='/unknown'),
                       lambda a: a['sources'][0]['containers'][0].update(download_result='../authenticated.json'),
                       lambda a: a['sources'][0]['containers'][0].update(final_identity=[1, True, 10, 4, 5])):
            bad = copy.deepcopy(closure)
            mutate(bad)
            with self.assertRaises(ValueError):
                audit.source_closure(bad, frozen)

    def test_closed_source_download_receipt_mismatch_rejects(self):
        closure, frozen = self.closure_fixture()
        path = self.root / closure['sources'][0]['containers'][0]['download_result']
        raw = audit.loads(path.read_bytes())
        raw['payload_authenticated'] = False
        path.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            audit.source_closure(closure, frozen)

    def runtime_fixture(self):
        source = self.file('runtime-source.py', b'# captured source\n')
        native = self.file('runtime-native.dylib', b'captured native')
        files = {str(path): dict(bytes=path.stat().st_size, sha256=digest(path.read_bytes()))
                 for path in (source, native)}
        raw = json.dumps(dict(format='toklet-granite-quant-runtime-v1', versions=audit.VERSIONS, files=files)).encode()
        path = self.file('runtime.json', raw)
        imports = dict(numpy=str(source), **{'mlx.core': str(native), 'mlx.nn': str(source)})
        origins = dict(modules=imports, non_system_native_images={str(native): files[str(native)]},
                       system_native_images=['/usr/lib/libSystem.B.dylib'])
        return dict(runtime=dict(path=str(path), bytes=len(raw), sha256=digest(raw)),
                    admitted_runtime=dict(manifest_sha256=digest(raw), files=len(files), versions=dict(audit.VERSIONS),
                        streaming_hash_chunk_limit_bytes=audit.HASH_BLOCK, pycache_prefix='/fresh/cache',
                        imports=imports, initial_origins=origins), endpoint_origins=copy.deepcopy(origins))

    def test_runtime_saved_origins_and_inventory_positive(self):
        manifest = self.runtime_fixture()
        observed = {}
        self.assertEqual(audit.runtime_receipts(manifest, observed), 2)
        self.assertEqual(len(observed), 2)
        self.assertEqual(audit.runtime_receipts(manifest, observed), 2)

    def test_runtime_missing_or_unadmitted_origins_reject(self):
        manifest = self.runtime_fixture()
        for mutate in (lambda m: m['admitted_runtime'].update(files=True),
                       lambda m: m['admitted_runtime']['imports'].pop('numpy'),
                       lambda m: m['endpoint_origins']['modules'].update(numpy='/unknown'),
                       lambda m: m['endpoint_origins']['non_system_native_images'].update({'/unknown': {}}),
                       lambda m: m['endpoint_origins'].update(system_native_images=['/private/foreign.dylib'])):
            bad = copy.deepcopy(manifest)
            mutate(bad)
            with self.assertRaises(ValueError):
                audit.runtime_receipts(bad)

    def test_runtime_bytes_and_endpoint_identity_mutation_reject(self):
        manifest = self.runtime_fixture()
        observed = {}
        audit.runtime_receipts(manifest, observed)
        path = Path(manifest['admitted_runtime']['imports']['numpy'])
        replacement = self.file('replacement-runtime.py', path.read_bytes())
        replacement.replace(path)
        with self.assertRaises(ValueError):
            audit.runtime_receipts(manifest, observed)
        path.write_bytes(b'mutation')
        with self.assertRaises(ValueError):
            audit.runtime_receipts(manifest)

    def rich_metadata_fixture(self, ancillary=False):
        raw = b'{"dtype":"bfloat16","rope_parameters":{"rope_type":"yarn"}}\n'
        path = self.file('generation_config.json' if ancillary else 'config.json', raw)
        actual_identity = audit.identity(path.lstat())
        descriptor = dict(path=str(path), bytes=len(raw), sha256=digest(raw),
                          before_identity=actual_identity, after_identity=actual_identity)
        extra = None
        if ancillary:
            extra = dict(file=path.name, source_path=str(self.root / 'source' / path.name))
            descriptor.update(extra)
        return path, raw, descriptor, extra

    def test_real_producer_identity_bearing_metadata_shapes_pass(self):
        for ancillary in (False, True):
            path, raw, descriptor, extra = self.rich_metadata_fixture(ancillary)
            self.assertEqual(audit.metadata_descriptor(descriptor, path, audit.HASH_BLOCK,
                                                       digest(raw), extra=extra), raw)

    def test_rich_metadata_missing_unknown_and_wrong_lineage_fields_reject(self):
        for ancillary in (False, True):
            path, raw, descriptor, extra = self.rich_metadata_fixture(ancillary)
            for field in descriptor:
                missing = copy.deepcopy(descriptor)
                missing.pop(field)
                with self.subTest(ancillary=ancillary, missing=field), self.assertRaises(ValueError):
                    audit.metadata_descriptor(missing, path, audit.HASH_BLOCK, digest(raw), extra=extra)
            for changed in (dict(unknown='unexpected'), dict(path=str(self.root / 'wrong')),
                            dict(bytes=True), dict(bytes=len(raw) + 1), dict(sha256='b' * 64)):
                with self.subTest(ancillary=ancillary, changed=changed), self.assertRaises(ValueError):
                    audit.metadata_descriptor(dict(descriptor, **changed), path, audit.HASH_BLOCK,
                                              digest(raw), extra=extra)
            if ancillary:
                for changed in (dict(file='wrong.json'), dict(source_path='/wrong/source.json')):
                    with self.subTest(changed=changed), self.assertRaises(ValueError):
                        audit.metadata_descriptor(dict(descriptor, **changed), path, audit.HASH_BLOCK,
                                                  digest(raw), extra=extra)
            else:
                with self.assertRaises(ValueError):
                    audit.metadata_descriptor(dict(descriptor, file='config.json'), path,
                                              audit.HASH_BLOCK, digest(raw))

    def test_rich_metadata_both_actual_endpoint_identities_required(self):
        path, raw, descriptor, _ = self.rich_metadata_fixture()
        for field in ('before_identity', 'after_identity'):
            for index in range(5):
                changed = copy.deepcopy(descriptor)
                changed[field][index] += 1
                with self.subTest(field=field, index=index), self.assertRaises(ValueError):
                    audit.metadata_descriptor(changed, path, audit.HASH_BLOCK, digest(raw))
            for invalid in ([1, 2], [True] * 5, 'wrong'):
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    audit.metadata_descriptor(dict(descriptor, **{field: invalid}), path,
                                              audit.HASH_BLOCK, digest(raw))

    def test_rich_metadata_missing_file_rejects(self):
        for ancillary in (False, True):
            path, raw, descriptor, extra = self.rich_metadata_fixture(ancillary)
            path.unlink()
            with self.subTest(ancillary=ancillary), self.assertRaises(FileNotFoundError):
                audit.metadata_descriptor(descriptor, path, audit.HASH_BLOCK, digest(raw), extra=extra)

    def test_rich_metadata_same_bytes_path_replacement_rejects(self):
        for ancillary in (False, True):
            path, raw, descriptor, extra = self.rich_metadata_fixture(ancillary)
            replacement = self.file('replacement-rich-metadata', raw)
            replacement.replace(path)
            self.assertEqual(path.read_bytes(), raw)
            with self.subTest(ancillary=ancillary), self.assertRaises(ValueError):
                audit.metadata_descriptor(descriptor, path, audit.HASH_BLOCK, digest(raw), extra=extra)

    def test_rich_metadata_replacement_during_capture_rejects(self):
        path, raw, descriptor, _ = self.rich_metadata_fixture()
        def replace_during_capture(selected, maximum, expected):
            captured = audit.capture(selected, maximum, expected)
            replacement = self.file('replacement-during-rich-capture', raw)
            replacement.replace(path)
            return captured
        with self.assertRaises(ValueError):
            audit.metadata_descriptor(descriptor, path, audit.HASH_BLOCK, digest(raw),
                                      capture_file=replace_during_capture)

    def test_cli_requires_fresh_output_before_any_capture(self):
        existing = self.root / 'existing'
        existing.mkdir()
        argv = ['--contract', '/missing', '--contract-sha256', SHA,
                '--conversion-manifest', '/missing', '--conversion-manifest-sha256', SHA,
                '--fresh-output', str(existing)]
        with self.assertRaises(ValueError):
            audit.main(argv)
        link = self.root / 'link'
        link.symlink_to(self.root / 'absent')
        argv[-1] = str(link)
        with self.assertRaises(ValueError):
            audit.main(argv)

    def test_layout_rejects_duplicate_bad_packing_and_boolean_shapes(self):
        matrices = [dict(module='language_model.model.x', shape=[3, 64], bits=4, group_size=64, mode='affine')]
        good = audit.output_layout(matrices, [])
        self.assertEqual(good['body_bytes'], 108)
        for changed in (dict(shape=[True, 64]), dict(shape=[3, 65]), dict(bits=True), dict(bits=6),
                        dict(group_size=32), dict(mode='mxfp4')):
            with self.assertRaises(ValueError):
                audit.output_layout([dict(matrices[0], **changed)], [])
        with self.assertRaises(ValueError):
            audit.output_layout(matrices * 2, [])


if __name__ == '__main__':
    unittest.main()
