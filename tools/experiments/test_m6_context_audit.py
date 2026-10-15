"""Saved-data replay rejects altered/incomplete evidence without a model/GPU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import m6_context_audit as audit


class Replay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = audit.ROOT
        stem = root/'benchmarks/results/2026-10-15-m6-long-budget-small-probe-v1'
        cls.record = json.loads(stem.with_suffix('.json').read_text())
        cls.raw = [json.loads(line) for line in stem.with_suffix('.raw.jsonl').read_text().splitlines()]
        cls.manifest, cls.arrays = audit.base.shared.contract()
        _, long_arrays, _, _, _ = audit.base.inputs('small')
        cls.ids = long_arrays[4096]

    def replay(self, record=None, raw=None):
        return audit.replay(record if record is not None else self.record,
                            raw if raw is not None else self.raw, self.manifest, self.arrays, self.ids)

    def test_published_probe_replays_without_native_execution(self):
        with patch.object(audit.base.subprocess, 'Popen') as launch:
            row = self.replay()
        launch.assert_not_called()
        self.assertEqual(row['raw_events'], 3)
        self.assertEqual(row['phases'], 2)
        self.assertEqual(row['max_joint_native_bytes'], 1274775496)

    def test_raw_coverage_and_values_are_exact(self):
        for raw in (self.raw[:-1], self.raw + [self.raw[-1]], list(reversed(self.raw))):
            with self.assertRaisesRegex(ValueError, 'Raw event'): self.replay(raw=raw)

    def test_extra_event_in_both_summary_and_raw_is_rejected(self):
        for event in ({'event': 'unexpected'}, self.record['worker']['phases'][0]['events'][0]):
            record = copy.deepcopy(self.record)
            record['worker']['phases'][0]['events'].insert(0, event)
            raw = [record['worker']['ready'], *(e for p in record['worker']['phases'] for e in p['events'])]
            with self.assertRaisesRegex(ValueError, 'exactly one event'): self.replay(record, raw)

    def test_incomplete_failed_and_cleanup_records_are_rejected(self):
        for key, value in [('passed', False), ('collection_completed', False), ('identities_unchanged', False), ('endpoint_errors', ['changed'])]:
            record = copy.deepcopy(self.record); record[key] = value
            with self.assertRaises(ValueError): self.replay(record)
        for key, value in [('passed', False), ('exit_code', True), ('exit_code', 1), ('cleanup_errors', ['reap'])]:
            record = copy.deepcopy(self.record); record['worker'][key] = value
            with self.assertRaises(ValueError): self.replay(record)

    def test_request_metrics_native_charge_and_order_mutations_rejected(self):
        for change in ('request', 'metric', 'charge', 'owner', 'coverage', 'order', 'capacity', 'profile'):
            record = copy.deepcopy(self.record)
            if change == 'request': record['worker']['phases'][0]['request']['tokens'][0] += 1
            if change == 'metric': record['workloads'][0]['summary']['online']['total_seconds']['median'] += 1
            if change == 'charge': record['worker']['usage'][0]['joint_native_lifetime_peak_bytes'] += 1
            if change == 'owner': record['worker']['usage'][0]['driver']['lifetime_max_phys_footprint'] = True
            if change == 'coverage': record['worker']['usage'].pop()
            if change == 'order': record['worker']['usage'][0]['phase'] = 'phase_1'
            if change == 'capacity': record['capacity'] = 4096
            if change == 'profile': record['profiling'] = 0
            with self.subTest(change=change), self.assertRaises(ValueError): self.replay(record)

    def test_nonfinite_and_oversized_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'evidence'
            path.write_text('{"x":NaN}')
            with self.assertRaisesRegex(ValueError, 'Nonfinite'): audit.read_json(path)
            path.write_bytes(b' ' * 5)
            with patch.object(audit, 'BOUND', 4), self.assertRaisesRegex(ValueError, 'bound'):
                audit.read_json(path)

    def test_parsed_summary_and_raw_hashes_bind_exact_buffer_before_replacement(self):
        for lines, before, after in ((False, b'{"old":1}', b'{"new":2}'),
                                     (True, b'{"old":1}\n', b'{"new":2}\n')):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); path = root/'evidence'; path.write_bytes(before)
                parsed, sha = audit.read_json(path, lines=lines)
                self.assertEqual(parsed, [{'old': 1}] if lines else {'old': 1})
                self.assertEqual(sha, audit.hashlib.sha256(before).hexdigest())
                path.write_bytes(after)
                with patch.object(audit, 'ROOT', root), self.assertRaisesRegex(ValueError, 'Consumed bytes changed'):
                    audit.verify_bindings({'evidence': sha})


if __name__ == '__main__': unittest.main()
