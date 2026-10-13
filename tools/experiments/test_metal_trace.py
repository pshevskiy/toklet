"""Collector protocol and artifact checks without Apple hardware or downloads."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import metal_trace as trace


class Collector(unittest.TestCase):
    def exercise(self, script, error, timeout=False):
        real_launch = subprocess.Popen
        children = []
        def launch(*args, **kwargs):
            self.assertTrue(kwargs['start_new_session'])
            child = real_launch([sys.executable, '-c', script], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True)
            children.append(child)
            return child
        def digest(path):
            return ('fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
                    if path.name == 'model.safetensors' else 'test-identity')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'summary.json'
            raw = Path(directory)/'raw.jsonl'
            with patch.object(trace, 'digest', side_effect=digest), \
                 patch.object(trace.subprocess, 'Popen', side_effect=launch), \
                 patch.object(trace.subprocess, 'check_output', return_value='test'), \
                 patch.object(trace.native_usage, 'usage', return_value={'lifetime_max_phys_footprint': 1}):
                if timeout:
                    with patch.object(trace.metal_engine.Transport, 'read', side_effect=TimeoutError('timed out')):
                        with self.assertRaisesRegex(TimeoutError, error):
                            trace.run(Path('/tmp/test-binary'), 'qwen', output, raw, Path(directory)/'capture')
                else:
                    with self.assertRaisesRegex(ValueError, error):
                        trace.run(Path('/tmp/test-binary'), 'qwen', output, raw, Path(directory)/'capture')
            self.assertIsNotNone(children[0].poll())
            record = json.loads(output.read_text())
            self.assertIn('error', record)
            self.assertTrue(record['identities_unchanged'])
            self.assertTrue(raw.exists())
            self.assertLessEqual(output.with_suffix('.stderr').stat().st_size, 1024*1024)

    def ready(self, **changes):
        ready = dict(event='ready', model='qwen', runs=14, events_per_operation=918,
                     combined_admitted_bytes=100, capture_memory_allowance_bytes=200,
                     capture_admitted_bytes=300, debug_assertions=False,
                     baseline_memory={}, profile_memory={})
        ready.update(changes)
        return 'import json; print(' + repr(json.dumps(ready)) + ',flush=True); '

    def test_bad_admission(self):
        for changes in ({'debug_assertions': True}, {'capture_admitted_bytes': 301},
                        {'capture_admitted_bytes': 6*1024**3+1}, {'events_per_operation': 82}):
            with self.subTest(changes=changes):
                self.exercise(self.ready(**changes)+'import time; time.sleep(60)', 'admission/protocol')

    def test_profile_header_order(self):
        self.exercise(self.ready()+"input(); print('{\"event\":\"profile\",\"run\":1}',flush=True)", 'profile header')

    def test_eof(self): self.exercise('pass', 'ended unexpectedly')
    def test_timeout(self): self.exercise('import time; time.sleep(60)', 'timed out', True)
    def test_bounded_line(self): self.exercise('print("x"*65537,flush=True)', 'exceeds bounded')
    def test_stderr_flood(self):
        self.exercise('import os,time; os.write(2,b"x"*(2*1024*1024)); time.sleep(60)', 'stderr exceeds')

    def test_preserves_all_existing_destinations(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            output, raw, capture = base/'summary.json', base/'raw.jsonl', base/'capture'
            for destination in (output, raw, output.with_suffix('.stderr'), capture):
                destination.write_text('retained')
                with patch.object(trace.subprocess, 'Popen') as launch:
                    with self.assertRaisesRegex(ValueError, 'destination exists'):
                        trace.run(base/'binary', 'tiny', output, raw, capture)
                    launch.assert_not_called()
                self.assertEqual(destination.read_text(), 'retained')
                destination.unlink()

    def test_tiny_artifact_preserves_original_float_payload(self):
        manifest = json.loads((trace.ROOT/'tests/fixtures/tiny-v1/manifest.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)/'tiny'
            trace.tiny(model)
            artifact = (model/'model.safetensors').read_bytes()
            header_bytes = struct.unpack('<Q', artifact[:8])[0]
            self.assertEqual(header_bytes % 8, 0)
            header = json.loads(artifact[8:8+header_bytes])
            payload = artifact[8+header_bytes:]
            expected = sorted((item['sha256'], item['bytes'], item['shape'])
                              for name,item in manifest['arrays'].items() if name.startswith('weights.'))
            actual = []
            end = 0
            for item in header.values():
                begin, finish = item['data_offsets']
                self.assertEqual(begin, end)
                self.assertEqual(item['dtype'], 'F32')
                actual.append((hashlib.sha256(payload[begin:finish]).hexdigest(), finish-begin, item['shape']))
                end = finish
            self.assertEqual(end, len(payload))
            self.assertEqual(sorted(actual), expected)
            self.assertEqual(json.loads((model/'config.json').read_text())['hidden_size'], 32)
            with self.assertRaises(FileExistsError): trace.tiny(model)


if __name__ == '__main__': unittest.main()
