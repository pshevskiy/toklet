"""Portable witness/resource controls, with no model payload or downloads."""
import copy
import unittest

import lfm3_source as source
import check_lfm3_source as reader
import test_lfm3_source as fixtures


def tensors():
    plans = {}
    for label, parent in [('parent', True), ('child', False)]:
        header, payload = fixtures.header_fixture(parent)
        plans[label] = dict(origin=32376, rows=fixtures.admit(header, payload, parent))
    children = {row['name']: row for row in plans['child']['rows']}
    items = []
    for row in plans['parent']['rows']:
        child = children[row['name']]
        n = row['end']-row['start']
        start = 32376+row['start']
        child_start = 32376+child['start']
        items.append(dict(name=row['name'], dtype='BF16', parent_shape=row['shape'],
            child_shape=child['shape'], parent_offset=start, child_offset=child_start,
            bytes=n, sha256='1'*64, blocks=[dict(parent_offset=start+i,
                child_offset=child_start+i, bytes=min(source.BLOCK, n-i), sha256='1'*64)
                for i in range(0, n, source.BLOCK)]))
    return plans, dict(bytes=source.SIZE, sha256=source.PARENT_SHA, tensors=items, exact_tensor_count=286)


def resources():
    command = ['fixture']
    owner = dict(pid=11, controller_pid=10, started_ns=100, command=command)
    parent = dict(pid=12, started_ns=200, completed_ns=800, final_transfer_wait4_peak_rss_bytes=1000)
    samples = []
    for ns, pids in [(150, [10, 11]), (300, [10, 11, 12]), (900, [10, 11])]:
        rows = [dict(pid=p, role='controller' if p == 10 else 'owned_producer_or_curl',
            resident_size=100, phys_footprint=80, lifetime_max_phys_footprint=90) for p in pids]
        samples.append(dict(observed_ns=ns, processes=rows))
    owned = dict(passed=True, owned_group_absent=True, exit_code=0, failure=None,
        sample_errors=[], source_inputs_unchanged=True, complete_parent_authenticated=True,
        final_physical_peak_available=False, model_execution_qualified=False,
        quality_qualified=False, model_adoption_qualified=False, parent_payload_retained_bytes=0,
        guard_bytes=reader.GUARD, minimum_observed_free_bytes=3*1024**3,
        started_ns=100, completed_ns=1000, sampled_observations=3,
        max_sampled_aggregate_rss_bytes=300, max_sampled_aggregate_physical_bytes=240,
        max_sampled_aggregate_lifetime_physical_bytes=270,
        final_producer_wait4_peak_rss_bytes=2000, final_curl_wait4_peak_rss_bytes=1000,
        final_controller_peak_rss_bytes=3000, conservative_final_peak_rss_sum_bytes=6000)
    return owned, owner, dict(command=command), samples, parent


class Witnesses(unittest.TestCase):
    def test_complete_fixed_schema_witness_structure(self):
        plans, result = tensors()
        reader.validate_tensors(result, plans)
        self.assertEqual(len(result['tensors']), 286)

    def test_typed_identity_shape_transform_and_complete_coverage_rejects(self):
        plans, original = tensors()
        changes = [(['bytes'], True), (['sha256'], '0'*64), (['exact_tensor_count'], 285),
            (['tensors', 0, 'name'], 'wrong'), (['tensors', 0, 'bytes'], 0),
            (['tensors', 0, 'parent_offset'], True), (['tensors', 0, 'dtype'], 'F16'),
            (['tensors', 0, 'child_shape'], [128000., 2048])]
        for path, value in changes:
            with self.subTest(path=path):
                damaged = copy.deepcopy(original)
                node = damaged
                for part in path[:-1]:
                    node = node[part]
                node[path[-1]] = value
                with self.assertRaises(ValueError):
                    reader.validate_tensors(damaged, plans)
        for action in ['missing', 'duplicate', 'extra']:
            damaged = copy.deepcopy(original)
            if action == 'missing':
                damaged['tensors'].pop()
            elif action == 'duplicate':
                damaged['tensors'][1] = damaged['tensors'][0]
            else:
                damaged['unreviewed_claim'] = True
            with self.assertRaises(ValueError):
                reader.validate_tensors(damaged, plans)

    def test_block_tail_count_offsets_and_digest_syntax_rejects(self):
        plans, original = tensors()
        for field, value in [('parent_offset', True), ('child_offset', 0), ('bytes', source.BLOCK-1),
                             ('bytes', float(source.BLOCK)), ('sha256', 'z'*64)]:
            damaged = copy.deepcopy(original)
            damaged['tensors'][0]['blocks'][0][field] = value
            with self.assertRaises(ValueError):
                reader.validate_tensors(damaged, plans)
        damaged = copy.deepcopy(original)
        damaged['tensors'][0]['blocks'].pop()
        with self.assertRaises(ValueError):
            reader.validate_tensors(damaged, plans)

    def test_closed_owned_aggregate_resource_ledger(self):
        reader.validate_resources(*resources())

    def test_ledger_process_ownership_clock_and_complete_curl_evidence_rejects(self):
        for action in ['wrong_pid', 'duplicate_pid', 'wrong_role', 'outside', 'repeated', 'missing_curl', 'zero', 'sum']:
            values = resources()
            samples = values[3]
            if action == 'wrong_pid':
                samples[1]['processes'][2]['pid'] = 99
            elif action == 'duplicate_pid':
                samples[1]['processes'][2]['pid'] = 11
            elif action == 'wrong_role':
                samples[1]['processes'][0]['role'] = 'owned_producer_or_curl'
            elif action == 'outside':
                samples[1]['observed_ns'] = 801
            elif action == 'repeated':
                samples[1]['observed_ns'] = 150
            elif action == 'missing_curl':
                samples[1]['processes'].pop()
                values[0]['max_sampled_aggregate_rss_bytes'] = 200
                values[0]['max_sampled_aggregate_physical_bytes'] = 160
                values[0]['max_sampled_aggregate_lifetime_physical_bytes'] = 180
            elif action == 'zero':
                samples[1]['processes'][2]['resident_size'] = 0
            else:
                samples[1]['processes'][2]['resident_size'] = reader.GUARD
            with self.subTest(action=action), self.assertRaises(ValueError):
                reader.validate_resources(*values)

    def test_unclosed_inflated_resource_or_qualified_outcome_rejects(self):
        for field, value in [('passed', 1), ('owned_group_absent', False), ('exit_code', False),
            ('source_inputs_unchanged', False), ('guard_bytes', reader.GUARD+1),
            ('minimum_observed_free_bytes', 2*1024**3-1), ('completed_ns', 700),
            ('sampled_observations', 2), ('max_sampled_aggregate_rss_bytes', 0),
            ('final_curl_wait4_peak_rss_bytes', 999), ('conservative_final_peak_rss_sum_bytes', 6001),
            ('final_physical_peak_available', True), ('model_adoption_qualified', True)]:
            values = resources()
            values[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                reader.validate_resources(*values)

    def test_bounded_strict_witness_json(self):
        for raw in ['{"x":1,"x":2}', '{"x":NaN}']:
            with self.assertRaises(ValueError):
                reader.parsed(raw)

    def test_each_final_rss_peak_must_cover_its_own_sampled_resident_size(self):
        fields = ['final_producer_wait4_peak_rss_bytes', 'final_curl_wait4_peak_rss_bytes',
                  'final_controller_peak_rss_bytes']
        for selected in [[key] for key in fields]+[fields]:
            values = resources()
            owned, _, _, _, parent = values
            for key in selected:
                owned[key] = 1
            if 'final_curl_wait4_peak_rss_bytes' in selected:
                parent['final_transfer_wait4_peak_rss_bytes'] = 1
            owned['conservative_final_peak_rss_sum_bytes'] = sum(owned[key] for key in fields)
            with self.subTest(selected=selected), self.assertRaisesRegex(ValueError, 'Final process RSS peaks'):
                reader.validate_resources(*values)


if __name__ == '__main__':
    unittest.main()
