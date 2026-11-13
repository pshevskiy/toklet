"""Cold selected-config, authentication, descriptor and publication controls."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import ministral3_convert as convert
from test_ministral3_convert_layout import fixture, FIXTURES, tiny_recipe


def config(label='ministral_modified'):
    return json.loads(fixture(label + '-config.json'))


def contract():
    matrices, norms = convert.layout.selected_recipe()
    inventory = convert.layout.selected_inventory()
    manifest = json.loads((FIXTURES / 'manifest.json').read_bytes())
    parents = copy.deepcopy(manifest['sources'])
    for parent in parents:
        parent['directory'] = '/synthetic/' + parent['label']
    return dict(format='toklet-ministral3-heretic-mixed-affine-v1',
                recipe=dict(device='gpu', input_dtype='BF16', metadata_dtype='BF16',
                            maximum_rows_per_tile=224, maximum_columns=9216, group_size=64,
                            quantization_runtime=convert.runtime.VERSIONS,
                            matrices=matrices, norms=norms),
                expected_source_inventory=inventory,
                source_exclusions=[dict(name=name, **item) for name, item in inventory.items()
                                   if not name.startswith('language_model.model.')],
                expected_output=convert.layout.EXPECTED_OUTPUT,
                engine_limit_bytes=6 * 1024**3, audit_limit_bytes=512 * 1024**2,
                helper_reserve_bytes=32 * 1024**2, ordinary_Metal_context_cap=512,
                source_authentication_receipt_sha256='0' * 64,
                original_source=parents[0], source=parents[1])


def authentication(selected):
    rows = []
    for source in (selected['original_source'], selected['source']):
        row = {key: source[key] for key in ('label', 'repository', 'revision', 'directory')}
        row.update(tensor_payloads_authenticated=True, containers=[], metadata=[])
        for pin in source['containers']:
            row['containers'].append(dict(file=pin['container'], bytes=pin['expected_bytes'],
                                          sha256=pin['expected_sha256'], payload_authenticated=True,
                                          pinned_header_ranges_match=True))
        for name, pin in [('config.json', source['config_sha256']),
                          ('model.safetensors.index.json', source['index_sha256'])]:
            raw = fixture(source['label'] + '-' + name)
            row['metadata'].append(dict(file=name, bytes=len(raw), sha256=pin))
        rows.append(row)
    return dict(passed=True, remaining_expected_bytes=0, sources=rows)


class Producer(unittest.TestCase):
    def test_both_genuine_wrappers_preserve_every_field_except_new_quantization(self):
        matrices, _ = convert.layout.selected_recipe()
        for label in ('ministral_original', 'ministral_modified'):
            original = config(label)
            before = copy.deepcopy(original)
            child = convert.converted_config(original, matrices)
            self.assertEqual(original, before)
            recipe = child.pop('quantization')
            self.assertEqual(child, original)
            self.assertEqual((recipe['group_size'], recipe['bits'], recipe['mode']), (64, 4, 'affine'))
            overrides = {name: value for name, value in recipe.items()
                         if name not in ('group_size', 'bits', 'mode')}
            self.assertEqual(len(overrides), 53)
            self.assertTrue(all(value == dict(group_size=64, bits=8, mode='affine')
                                for value in overrides.values()))
            self.assertNotIn('language_model.lm_head', overrides)

    def test_ambiguous_wrapper_dtype_and_existing_quantization_reject(self):
        matrices, _ = convert.layout.selected_recipe()
        for changes in [dict(model_type='ministral3'), dict(architectures=['Other']),
                        dict(dtype='float16'), dict(torch_dtype='float32'),
                        dict(tie_word_embeddings=False), dict(quantization={}), dict(quantization_config={})]:
            source = config()
            source.update(changes)
            with self.assertRaises(ValueError):
                convert.converted_config(source, matrices)
        for changes in [dict(dtype='float16'), dict(torch_dtype='float32'), dict(hidden_size=4096),
                        dict(num_attention_heads=True), dict(head_dim=96), dict(quantization={}),
                        dict(sliding_window=4096), dict(rope_scaling={'type': 'yarn'}),
                        dict(rope_traditional=True), dict(rope_interleaved=True), dict(rope_theta=1e4)]:
            source = config()
            source['text_config'].update(changes)
            with self.assertRaises(ValueError):
                convert.converted_config(source, matrices)

    def test_yarn_alias_changed_value_unknown_or_type_and_recipe_changes_reject(self):
        matrices, _ = convert.layout.selected_recipe()
        for name, value in [('type', 'default'), ('rope_type', 'llama3'), ('factor', 8.0),
                            ('factor', True), ('factor', 16), ('mscale_all_dim', 0.0),
                            ('llama_4_scaling_beta', 0.2), ('original_max_position_embeddings', 8192),
                            ('new_parameter', 1)]:
            source = config()
            source['text_config']['rope_parameters'][name] = value
            with self.assertRaises(ValueError):
                convert.converted_config(source, matrices)
        for name in ('type', 'rope_type'):
            source = config()
            del source['text_config']['rope_parameters'][name]
            with self.assertRaises(ValueError):
                convert.converted_config(source, matrices)
        for altered in [matrices[:-1], matrices + [matrices[0]],
                        [dict(matrices[0], bits=4)] + matrices[1:]]:
            with self.assertRaises(ValueError):
                convert.converted_config(config(), altered)

    def test_complete_fixed_contract_and_draft_authentication_admission(self):
        original = contract()
        recipe, inventory = convert.validate_contract(original)
        self.assertEqual(len(recipe['matrices']), 183)
        self.assertEqual(len(inventory), 458)
        for key, value in [('source_authentication_receipt_sha256', None),
                           ('engine_limit_bytes', 7 * 1024**3), ('ordinary_Metal_context_cap', 1024)]:
            broken = copy.deepcopy(original)
            broken[key] = value
            with self.assertRaises(ValueError):
                convert.validate_contract(broken)
        for key, value in [('maximum_rows_per_tile', 256), ('maximum_columns', 8192),
                           ('group_size', True), ('metadata_dtype', 'F16'), ('device', 'cpu')]:
            broken = copy.deepcopy(original)
            broken['recipe'][key] = value
            with self.assertRaises(ValueError):
                convert.validate_contract(broken)
        broken = copy.deepcopy(original)
        del broken['expected_source_inventory']['vision_tower.patch_conv.weight']
        with self.assertRaises(ValueError):
            convert.validate_contract(broken)

    def test_authentication_requires_both_sources_allobjects_and_pinned_metadata(self):
        selected = contract()
        auth = authentication(selected)
        self.assertEqual(len(convert.validate_authentication(auth, selected)), 2)
        variants = []
        for key, value in [('passed', False), ('remaining_expected_bytes', True),
                           ('remaining_expected_bytes', 1), ('sources', auth['sources'][:1])]:
            variants.append(dict(auth, **{key: value}))
        for field, value in [('payload_authenticated', False), ('pinned_header_ranges_match', False),
                             ('bytes', True), ('sha256', '0' * 64)]:
            broken = copy.deepcopy(auth)
            broken['sources'][1]['containers'][0][field] = value
            variants.append(broken)
        broken = copy.deepcopy(auth)
        broken['sources'][1]['metadata'][0]['sha256'] = '0' * 64
        variants.append(broken)
        broken = copy.deepcopy(auth)
        broken['sources'][1]['metadata'].append(copy.deepcopy(broken['sources'][1]['metadata'][0]))
        variants.append(broken)
        for broken in variants:
            with self.assertRaises(ValueError):
                convert.validate_authentication(broken, selected)

    def test_exact_reads_and_hashes_handle_short_reads_and_reject_truncation(self):
        raw = b'0123456789' * 1000
        with tempfile.TemporaryDirectory() as folder:
            with (Path(folder) / 'source').open('w+b') as file:
                file.write(raw)
                file.flush()
                pread = os.pread
                with patch.object(convert.os, 'pread', side_effect=lambda fd, n, off: pread(fd, min(n, 317), off)):
                    self.assertEqual(convert.read_exact(file.fileno(), len(raw), 0), raw)
                    self.assertEqual(convert.hash_range(file.fileno(), len(raw)), hashlib.sha256(raw).hexdigest())
                for count, offset in [(True, 0), (convert.BLOCK + 1, 0), (1, -1), (1, False)]:
                    with self.assertRaises(ValueError):
                        convert.read_exact(file.fileno(), count, offset)
                with self.assertRaises(ValueError):
                    convert.hash_range(file.fileno(), len(raw) + 1)

    def test_named_source_replacement_and_metadata_capture_extent_or_digest_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source'
            raw = b'original'
            path.write_bytes(raw)
            sha = hashlib.sha256(raw).hexdigest()
            captured, identity = convert.capture_named(path, 16, sha)
            self.assertEqual(captured, raw)
            self.assertEqual(identity, convert.runtime.identity(path.stat()))
            for maximum, pin in [(7, sha), (16, '0' * 64)]:
                with self.assertRaises(ValueError):
                    convert.capture_named(path, maximum, pin)
            with path.open('rb') as file:
                identity = convert.verify_named(file.fileno(), path)
                path.rename(Path(folder) / 'old')
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    convert.verify_named(file.fileno(), path, identity)
            link = Path(folder) / 'link'
            link.symlink_to(path)
            with self.assertRaises(OSError):
                convert.capture_named(link, 16, sha)

    def test_exclusive_metadata_roundtrip_never_overwrites_existing_path(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'metadata'
            pin = convert.write_exclusive(path, b'bounded bytes')
            self.assertEqual(pin['bytes'], 13)
            self.assertEqual(pin['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                convert.write_exclusive(path, b'changed')
            self.assertEqual(path.read_bytes(), b'bounded bytes')

    def test_atomic_promotion_requires_owned_partial_and_exclusive_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            partial, final = root / 'model.partial', root / 'model'
            partial.write_bytes(b'complete saved child')
            with partial.open('r+b') as file:
                identity = convert.verify_named(file.fileno(), partial)
                after = convert.promote_partial(file.fileno(), partial, final, identity)
                self.assertFalse(partial.exists())
                self.assertEqual(final.read_bytes(), b'complete saved child')
                self.assertEqual(after, convert.verify_named(file.fileno(), final))
            partial.write_bytes(b'failed candidate')
            with partial.open('r+b') as file:
                identity = convert.verify_named(file.fileno(), partial)
                with self.assertRaises(FileExistsError):
                    convert.promote_partial(file.fileno(), partial, final, identity)
            self.assertEqual(partial.read_bytes(), b'failed candidate')
            self.assertEqual(final.read_bytes(), b'complete saved child')

    def test_replaced_partial_descriptor_cannot_publish_a_peer_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            partial, final = root / 'partial', root / 'final'
            partial.write_bytes(b'old')
            with partial.open('r+b') as file:
                identity = convert.verify_named(file.fileno(), partial)
                partial.rename(root / 'owned')
                partial.write_bytes(b'peer')
                with self.assertRaises(ValueError):
                    convert.promote_partial(file.fileno(), partial, final, identity)
                self.assertFalse(final.exists())

    def test_failed_finalization_restores_only_the_owned_partial_without_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            partial, final = root / 'partial', root / 'final'
            partial.write_bytes(b'candidate')
            with partial.open('r+b') as file:
                identity = convert.verify_named(file.fileno(), partial)
                convert.promote_partial(file.fileno(), partial, final, identity)
                self.assertTrue(convert.preserve_failed_partial(file.fileno(), partial, final))
                self.assertEqual(partial.read_bytes(), b'candidate')
                self.assertFalse(convert.preserve_failed_partial(file.fileno(), partial, final))
                partial.unlink()
                final.rename(root / 'owned')
                final.write_bytes(b'peer')
                self.assertFalse(convert.preserve_failed_partial(file.fileno(), partial, final))
                self.assertFalse(partial.exists())

    def test_saved_metadata_same_bytes_at_a_replaced_path_fail_endpoint_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            pin = convert.write_exclusive(path, b'{"saved":true}\n')
            convert.verify_metadata_records([pin])
            raw = path.read_bytes()
            path.rename(Path(folder) / 'old-config')
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                convert.verify_metadata_records([pin])

    def test_final_saved_tensor_hash_detects_late_byte_changes(self):
        out = convert.layout.output_layout(*tiny_recipe())
        name = next(iter(out['matrices']))
        plan = out['matrices'][name]
        with tempfile.TemporaryDirectory() as folder:
            with (Path(folder) / 'child').open('w+b') as file:
                file.truncate(out['file_bytes'])
                buffers = {key: bytes(3 * item['row_stride']) for key, item in plan['fields'].items()}
                convert.layout.write_matrix_tile(file.fileno(), out, name, 0, 3, buffers)
                fields = {key: convert.component_record(out, item['name'], 0, 3, buffers[key])
                          for key, item in plan['fields'].items()}
                convert.verify_saved_fields(file.fileno(), [dict(fields=fields)], [])
                first = next(iter(fields.values()))
                os.pwrite(file.fileno(), b'\xff', first['offset'])
                with self.assertRaises(ValueError):
                    convert.verify_saved_fields(file.fileno(), [dict(fields=fields)], [])

    def test_fresh_directories_reject_existing_symlink_alias_or_nesting_before_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            occupied = root / 'occupied'
            occupied.write_bytes(b'keep')
            link = root / 'dangling'
            link.symlink_to(root / 'absent')
            for existing in (occupied, root, link):
                peer = root / 'new-peer'
                with self.assertRaises(ValueError):
                    convert.reserve_output(existing, peer)
                self.assertFalse(peer.exists())
                with self.assertRaises(ValueError):
                    convert.reserve_output(peer, existing)
                self.assertFalse(peer.exists())
            for output, evidence in [(root / 'same', root / 'same'),
                                     (root / 'parent', root / 'parent/child')]:
                with self.assertRaises(ValueError):
                    convert.reserve_output(output, evidence)
                self.assertFalse(output.exists())

    def test_existing_cli_output_rejects_before_contract_or_runtime_access(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            occupied = root / 'occupied'
            occupied.write_bytes(b'keep')
            command = [sys.executable, '-B', str(Path(convert.__file__)),
                       '--environment-root', 'missing-env', '--runtime-manifest', 'missing-runtime',
                       '--runtime-sha256', '0' * 64, '--contract', 'missing-contract',
                       '--contract-sha256', '0' * 64, '--source-authentication', 'missing-auth',
                       '--fresh-output', str(occupied), '--fresh-evidence', str(root / 'new-evidence')]
            result = subprocess.run(command, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'Never replace', result.stderr)
            self.assertNotIn(b'FileNotFoundError', result.stderr)
            self.assertFalse((root / 'new-evidence').exists())

    def test_pending_authentication_contract_rejects_before_any_numerical_import(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pending = contract()
            pending['source_authentication_receipt_sha256'] = None
            raw = json.dumps(pending).encode()
            path = root / 'contract.json'
            path.write_bytes(raw)
            argv = ['convert', '--environment-root', 'missing-env', '--runtime-manifest', 'missing-runtime',
                    '--runtime-sha256', '0' * 64, '--contract', str(path),
                    '--contract-sha256', hashlib.sha256(raw).hexdigest(),
                    '--source-authentication', 'missing-auth', '--fresh-output', str(root / 'child'),
                    '--fresh-evidence', str(root / 'evidence')]
            with patch.object(sys, 'argv', argv), patch.object(convert.runtime, 'bind_environment') as bind:
                with self.assertRaisesRegex(ValueError, 'SHA256 identity'):
                    convert.main()
                bind.assert_not_called()
            self.assertFalse((root / 'child/model.safetensors.partial').exists())
            self.assertFalse(any(name == 'numpy' or name.startswith('numpy.') or name == 'mlx'
                                 or name.startswith('mlx.') for name in sys.modules))

    def test_source_closure_contains_new_and_every_imported_generic_helper_and_tests(self):
        names = {path.name for path in convert.sources()}
        required = {'ministral3_convert.py', 'test_ministral3_convert.py',
                    'ministral3_convert_layout.py', 'test_ministral3_convert_layout.py',
                    'ministral3_quantize.py', 'test_ministral3_quantize.py', 'ministral3_quant_probe.py',
                    'granite_convert.py', 'granite_convert_layout.py', 'granite_quantize.py',
                    'granite_quant_probe.py', 'granite_quant_runtime.py', 'minicpm5_parent.py',
                    'minicpm5_quantize.py', 'minicpm5_quant_runtime.py', 'reference_runtime.py',
                    'text_reference_runtime.py'}
        self.assertTrue(required <= names)
        self.assertEqual(len(convert.sources()), len(names))
        self.assertTrue(all(path.is_file() for path in convert.sources()))
        self.assertFalse(any(name == 'numpy' or name.startswith('numpy.') or name == 'mlx'
                             or name.startswith('mlx.') for name in sys.modules))


if __name__ == '__main__':
    unittest.main()
