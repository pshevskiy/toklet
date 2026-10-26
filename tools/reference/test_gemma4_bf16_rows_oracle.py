"""Offline oracle controls and explicit coherent captured-data rejections."""
from pathlib import Path
import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
import shutil
import signal
import struct
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import gemma4_bf16_rows_oracle as oracle
import check_gemma4_bf16_rows_oracle as checker

ROOT = Path(__file__).resolve().parents[2]


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def finish_sources(sources, primary):
    errors = []
    for source in sources:
        if source is not None:
            file, path, original = source
            try:
                checker.same(checker.identity(os.fstat(file.fileno())), original, 'control source endpoint changed')
                checker.same(checker.identity(path.lstat()), original, 'control source path changed')
            except BaseException as error: errors.append(str(error))
            try: file.close()
            except BaseException as error: errors.append('control source close: ' + str(error))
    if errors:
        if primary is not None: primary.add_note('control source cleanup: ' + '; '.join(errors))
        else: raise ValueError('control source cleanup: ' + '; '.join(errors))


class Offline(unittest.TestCase):
    def test_no_numerical_or_target_imports(self):
        for name in ('numpy', 'torch', 'bf16_source'):
            self.assertNotIn(name, sys.modules)

    def test_expansion_matches_independent_signed_zero_and_subnormal_fixture(self):
        fixture = ROOT / 'tests/fixtures/bf16-source-v1'
        raw = (fixture / 'source.bin').read_bytes()[8:32]
        expected = (fixture / 'finite.f32le').read_bytes()
        self.assertEqual(oracle.expand(raw), expected)
        self.assertEqual(checker.expanded(raw), expected)

    def test_each_nonfinite_word_and_incomplete_words_reject(self):
        for word in (0x7f80, 0xff80, 0x7fc1):
            for convert in (oracle.expand, checker.expanded):
                with self.subTest(word=word, convert=convert.__name__), self.assertRaises(ValueError):
                    convert(struct.pack('<H', word))
        for convert in (oracle.expand, checker.expanded):
            with self.assertRaises(ValueError): convert(b'x')

    def test_complete_geometry_and_consumer_exclusions(self):
        header = oracle.safe.document((ROOT / 'tests/fixtures/gemma-unquantized-header-v1/header.json').read_bytes())
        inventory = oracle.safe.document((ROOT / 'tests/fixtures/gemma-resident-storage-reference-v1/inventory.json').read_bytes())
        selected, unused = oracle.selected_geometry(header, inventory)
        self.assertEqual(len(selected), 540); self.assertEqual(len(unused), 60)
        self.assertEqual(sum(len(value['shape']) == 1 for value in selected.values()), 262)
        self.assertEqual(selected['model.language_model.embed_tokens_per_layer.weight']['shape'], [262144, 8960])

    def test_selected_and_excluded_shape_corruption_rejects(self):
        header = oracle.safe.document((ROOT / 'tests/fixtures/gemma-unquantized-header-v1/header.json').read_bytes())
        inventory = oracle.safe.document((ROOT / 'tests/fixtures/gemma-resident-storage-reference-v1/inventory.json').read_bytes())
        for name in ['model.language_model.embed_tokens.weight', 'model.language_model.layers.15.self_attn.k_proj.weight']:
            mutated = copy.deepcopy(header); mutated[name]['shape'][0] += 1
            with self.assertRaises(ValueError): oracle.selected_geometry(mutated, inventory)

    def test_duplicate_or_missing_prior_inventory_rejects(self):
        header = oracle.safe.document((ROOT / 'tests/fixtures/gemma-unquantized-header-v1/header.json').read_bytes())
        inventory = oracle.safe.document((ROOT / 'tests/fixtures/gemma-resident-storage-reference-v1/inventory.json').read_bytes())
        for mode in ('duplicate', 'missing'):
            value = copy.deepcopy(inventory)
            if mode == 'duplicate': value['matrices'].append(value['matrices'][0])
            else: value['matrices'].pop()
            with self.assertRaises(ValueError): oracle.selected_geometry(header, value)

    def test_retained_descriptor_authentication_bounds_and_path_change(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source'; path.write_bytes(b'abcd'); digest = hashlib.sha256(b'abcd').hexdigest()
            file, original = oracle.authenticate(path, 4, digest, time.monotonic() + 10)
            try:
                self.assertEqual(oracle.read_exact(file, path, original, 1, 2, time.monotonic() + 10), b'bc')
                for offset, count in [(True, 1), (-1, 1), (0, 0), (0, oracle.MAX_READ + 1), (3, 2)]:
                    with self.assertRaises(ValueError): oracle.read_exact(file, path, original, offset, count, time.monotonic() + 10)
                path.unlink(); path.write_bytes(b'abcd')
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    oracle.read_exact(file, path, original, 0, 1, time.monotonic() + 10)
            finally: file.close()

    def test_wrong_source_hash_and_deadline_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source'; path.write_bytes(b'abcd')
            with self.assertRaisesRegex(ValueError, 'SHA differs'):
                oracle.authenticate(path, 4, '0' * 64, time.monotonic() + 10)
            with self.assertRaisesRegex(ValueError, 'deadline'):
                oracle.authenticate(path, 4, hashlib.sha256(b'abcd').hexdigest(), time.monotonic() - 1)

    def test_existing_output_never_starts_source_access(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(oracle, 'authenticate') as source:
            with self.assertRaises(FileExistsError): oracle.prepare(Path(folder))
            source.assert_not_called()

    def test_source_failure_preserves_failed_manifest(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(oracle, 'authenticate', side_effect=OSError('synthetic source')):
            output = Path(folder) / 'capture'
            with contextlib.redirect_stdout(io.StringIO()): self.assertEqual(oracle.prepare(output), 1)
            value = json.loads((output / 'manifest.json').read_bytes())
            self.assertFalse(value['passed']); self.assertIn('synthetic source', value['primary_error'])
            self.assertFalse(value['official_graph_executed'])

    def test_failed_first_endpoint_still_closes_both_sources_and_retains_primary(self):
        with tempfile.TemporaryDirectory() as folder:
            handles = []
            for name in ('first', 'second'):
                path = Path(folder) / name; path.write_bytes(b'abcd'); file = path.open('rb')
                handles.append((file, path, checker.identity(os.fstat(file.fileno()))))
            handles[0][1].write_bytes(b'changed!'); primary = RuntimeError('original control error')
            finish_sources(handles, primary)
            self.assertTrue(all(file.closed for file, _, _ in handles))
            self.assertEqual(str(primary), 'original control error')
            self.assertIn('cleanup', primary.__notes__[0])


def coherent_controls(directory, manifest_sha, completed=None, failure_root=None):
    original, original_assets = checker.load(directory, manifest_sha)
    official = checker.open_source(checker.MODEL, checker.MODEL_ID)
    candidate = None; primary = None
    if completed is None: completed = []
    def retain_failure(root, mode, record):
        if failure_root is not None:
            try:
                failure_root.mkdir(exist_ok=True); shutil.copytree(root, failure_root / mode)
            except BaseException as error:
                record['capture_error'] = type(error).__name__ + ': ' + str(error)
    controls = [('tensor', 'frozen tensor selection differs'), ('rows', 'frozen source row order differs'),
                ('shape', 'frozen tensor shape differs'), ('offset', 'official source span differs'),
                ('raw_and_f32', 'official source bytes differ'), ('f32', 'frozen F32 words differ'),
                ('omit', 'frozen tensor selection differs'), ('duplicate', 'frozen tensor selection differs'),
                ('source_identity', 'official source identity differs'), ('header_identity', 'official header identity differs'),
                ('inventory_identity', 'geometry inventory differs'), ('projection_input', 'projection input descriptor differs'),
                ('projection_scalar', 'scalar projection oracle differs')]
    try:
        candidate = checker.open_source(checker.CANDIDATE, checker.CANDIDATE_ID)
        for mode, expected_error in controls:
            m = copy.deepcopy(original); assets = dict(original_assets); first = m['cases'][0]
            stride = first['shape'][1] * 4; raw_stride = stride // 2
            if mode == 'tensor': m['cases'][0], m['cases'][1] = m['cases'][1], m['cases'][0]
            elif mode == 'rows':
                first['rows'][0], first['rows'][1] = first['rows'][1], first['rows'][0]
                for key, width in [('arrays', stride), ('raw', raw_stride)]:
                    body = bytearray(assets[key]); body[:2 * width] = body[width:2 * width] + body[:width]; assets[key] = bytes(body)
                first['sources'][0]['file_offset'], first['sources'][1]['file_offset'] = first['sources'][1]['file_offset'], first['sources'][0]['file_offset']
            elif mode == 'shape':
                old_array_bytes = first['bytes']; old_raw_bytes = sum(source['bytes'] for source in first['sources'])
                chunks = [assets['raw'][source['raw_offset']:source['raw_offset'] + source['bytes']] + b'\0\0' for source in first['sources']]
                new_raw = b''.join(chunks); new_array = b''.join(checker.expanded(raw) for raw in chunks)
                assets['raw'] = new_raw + assets['raw'][old_raw_bytes:]
                assets['arrays'] = new_array + assets['arrays'][old_array_bytes:]
                first['tensor_shape'][1] += 1; first['shape'][1] += 1
                raw_delta = len(new_raw) - old_raw_bytes; array_delta = len(new_array) - old_array_bytes
                first['bytes'] = len(new_array); first['sha256'] = checker.digest(new_array)
                base = first['sources'][0]['file_offset']; cursor = 0
                for row, source, raw in zip(first['rows'], first['sources'], chunks, strict=True):
                    source.update(file_offset=base + row * len(raw), bytes=len(raw), raw_offset=cursor, sha256=checker.digest(raw)); cursor += len(raw)
                for case in m['cases'][1:]:
                    case['offset'] += array_delta
                    for source in case['sources']: source['raw_offset'] += raw_delta
                for projection in m['projections']:
                    projection['raw_offset'] += raw_delta; projection['input']['offset'] += array_delta
                    for reference in projection['references'].values(): reference['offset'] += array_delta
                for relation in m['candidate_vectors']: relation['candidate_raw_offset'] += raw_delta
            elif mode == 'offset':
                first['sources'][0]['file_offset'] += 2
                raw = checker.source_bytes(official, first['sources'][0]['file_offset'], raw_stride)
                assets['raw'] = raw + assets['raw'][raw_stride:]
                assets['arrays'] = checker.expanded(raw) + assets['arrays'][stride:]
            elif mode == 'raw_and_f32':
                body = bytearray(assets['raw']); body[0] ^= 1; assets['raw'] = bytes(body)
                assets['arrays'] = checker.expanded(assets['raw'][:2]) + assets['arrays'][4:]
            elif mode == 'f32':
                body = bytearray(assets['arrays']); body[0] ^= 1; assets['arrays'] = bytes(body)
            elif mode == 'omit': m['cases'].pop(0)
            elif mode == 'duplicate': m['cases'].insert(1, copy.deepcopy(first))
            elif mode == 'source_identity': m['model']['sha256'] = '0' * 64
            elif mode == 'header_identity': m['header']['sha256'] = '0' * 64
            elif mode == 'inventory_identity': m['inventory_sha256'] = '0' * 64
            elif mode == 'projection_input':
                projection = m['projections'][0]; body = bytearray(assets['arrays']); body[projection['input']['offset']] ^= 1; assets['arrays'] = bytes(body)
                start = projection['input']['offset']; end = start + projection['input']['bytes']; projection['input']['sha256'] = checker.digest(assets['arrays'][start:end])
            else:
                descriptor = m['projections'][0]['references']['fsum']; body = bytearray(assets['arrays']); body[descriptor['offset']] ^= 1; assets['arrays'] = bytes(body)
                start = descriptor['offset']; end = start + descriptor['bytes']; descriptor['sha256'] = checker.digest(assets['arrays'][start:end])
            if mode in ('rows', 'offset', 'raw_and_f32', 'f32'):
                first['sha256'] = checker.digest(assets['arrays'][first['offset']:first['offset'] + first['bytes']])
                for source in first['sources']:
                    start = source['raw_offset']; source['sha256'] = checker.digest(assets['raw'][start:start + source['bytes']])
            for key in ('arrays', 'raw'):
                m[key]['bytes'] = len(assets[key]); m[key]['sha256'] = checker.digest(assets[key])
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                for key in ('arrays', 'raw'): (root / m[key]['file']).write_bytes(assets[key])
                raw = encoded(m); (root / 'manifest.json').write_bytes(raw); changed_sha = checker.digest(raw)
                admitted = False
                try:
                    checker.load(root, changed_sha); admitted = True
                    checker.verify(root, changed_sha, official, candidate)
                except ValueError as error:
                    if str(error) != expected_error:
                        record = dict(control=mode, rejected=False, expected_error=expected_error, actual_error=str(error), coherent_manifest_sha256=changed_sha)
                        completed.append(record); retain_failure(root, mode, record)
                        raise ValueError(mode + ': unintended control error: ' + str(error)) from error
                    completed.append(dict(control=mode, expected_error=expected_error, rejected=True,
                                          coherent_manifest_sha256=changed_sha, binary_integrity_admitted=admitted))
                else:
                    record = dict(control=mode, rejected=False, expected_error=expected_error, actual_error=None, coherent_manifest_sha256=changed_sha)
                    completed.append(record); retain_failure(root, mode, record)
                    raise ValueError('coherent control accepted: ' + mode)
    except BaseException as error:
        primary = error
        raise
    finally:
        finish_sources([official, candidate], primary)
    return completed


if __name__ == '__main__':
    if '--capture' not in sys.argv:
        unittest.main()
    else:
        parser = argparse.ArgumentParser(); parser.add_argument('--capture', type=Path, required=True); parser.add_argument('--manifest-sha', required=True); parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(); report = dict(passed=False, primary_error=None, controls=[])
        def stop(number, _):
            for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, signal.SIG_IGN)
            raise InterruptedError(str(number))
        old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            coherent_controls(args.capture, args.manifest_sha, report['controls'], args.output.parent / (args.output.stem + '-failures'))
            report['passed'] = True
        except BaseException as error:
            report['primary_error'] = type(error).__name__ + ': ' + str(error)
            report['cleanup_notes'] = getattr(error, '__notes__', [])
        finally:
            for sig in old: signal.signal(sig, signal.SIG_IGN)
            try:
                with args.output.open('x') as file: json.dump(report, file, indent=2); file.write('\n')
            finally:
                for sig, handler in old.items(): signal.signal(sig, handler)
        print(json.dumps(report)); raise SystemExit(0 if report['passed'] else 1)
