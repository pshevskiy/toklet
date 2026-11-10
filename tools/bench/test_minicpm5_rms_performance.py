"""Portable version/implementation/setup and common-request invariant controls."""
import copy
import unittest

import minicpm5_rms_performance as reader
import test_minicpm5_performance as fixtures

BASE=fixtures.FULL
PARALLEL=dict(BASE,metadata_allowance_bytes=BASE['metadata_allowance_bytes']+67108864,
              admitted_bytes=BASE['admitted_bytes']+67108864)
CONTRACT=dict(teacher_tokens=fixtures.IDS,source_manifest_sha256=fixtures.SHA,
              expected_session_memory=dict(legacy=BASE,serial=BASE,**{'parallel-store':PARALLEL}))


def owner():
    return dict(fixtures.owned(),completed_ns=3000000001)


def report(variant):
    raw=fixtures.fixture()
    if variant!='legacy':
        raw['format']='toklet-minicpm5-rms-request-bench-v1'
        raw['normalization_implementation']='serial' if variant=='serial' else 'serial-reduction-parallel-store'
        raw['session_setup_seconds']=.002
        raw['session_memory']=copy.deepcopy(BASE if variant=='serial' else PARALLEL)
    return raw


class Requests(unittest.TestCase):
    def test_legacy_and_both_explicit_implementations_keep_same_work(self):
        for variant in ['legacy','serial','parallel-store']:
            result=reader.validate(report(variant),owner(),CONTRACT,variant)
            self.assertEqual(result['measured_requests'],10)
            self.assertEqual(result['sampled_tokens'],52)
            self.assertEqual(result['normalization_implementation'],variant)
            self.assertIs(result['session_setup_available'],variant!='legacy')
            self.assertFalse(result['model_quality_adoption_qualified'])

    def test_version_selection_and_new_owner_are_not_inferred_or_relabelled(self):
        for variant,expected in [('serial','parallel-store'),('parallel-store','serial'),('legacy','serial'),('serial','legacy')]:
            with self.assertRaises(ValueError):reader.validate(report(variant),owner(),CONTRACT,expected)
        raw=report('parallel-store');raw['session_memory']=copy.deepcopy(BASE)
        with self.assertRaises(ValueError):reader.validate(raw,owner(),CONTRACT,'parallel-store')
        raw=report('parallel-store');raw['extra']='speedup'
        with self.assertRaises(ValueError):reader.validate(raw,owner(),CONTRACT,'parallel-store')

    def test_setup_types_and_disjoint_setup_load_request_sum_are_bounded(self):
        for value in [True,-1,float('nan'),float('inf'),2**20+1]:
            raw=report('serial');raw['session_setup_seconds']=value
            with self.assertRaises(ValueError):reader.validate(raw,owner(),CONTRACT,'serial')
        raw=report('serial');raw['session_setup_seconds']=2.99
        with self.assertRaisesRegex(ValueError,'load/setup/requests'):reader.validate(raw,owner(),CONTRACT,'serial')

    def test_gpu_series_plus_setup_and_load_fit_owned_interval(self):
        raw=report('serial');raw['session_setup_seconds']=2.
        with self.assertRaisesRegex(ValueError,'load/setup/GPU series'):reader.validate(raw,owner(),CONTRACT,'serial')
        raw=report('serial');timing=raw['samples'][-1]['steps'][-1]['completed_timing']
        timing['gpu_start_seconds']+=3600;timing['gpu_end_seconds']+=3600
        with self.assertRaisesRegex(ValueError,'GPU series'):reader.validate(raw,owner(),CONTRACT,'serial')

    def test_gpu_commands_cannot_overlap_or_reorder(self):
        raw=report('serial');first=raw['samples'][0]['steps'][0]['completed_timing'];second=raw['samples'][0]['steps'][1]['completed_timing']
        second['gpu_start_seconds']=first['gpu_start_seconds'];second['gpu_end_seconds']=first['gpu_end_seconds']
        with self.assertRaisesRegex(ValueError,'overlap or reorder'):reader.validate(raw,owner(),CONTRACT,'serial')


if __name__=='__main__':unittest.main()
