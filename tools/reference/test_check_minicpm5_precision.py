"""Portable independent IEEE boundary tests; no NumPy, model or GPU imports."""
import copy
import hashlib
import math
from pathlib import Path
import sys
import tempfile
import unittest

import check_minicpm5_precision as audit


def fixture():
    root = Path(audit.__file__).resolve().parents[2]
    return dict(format='toklet-minicpm5-native-representability-contract-v1',
        prospective_before_representability_outcomes=True, gates_recomputed_or_relaxed=False,
        native_precision_changed=False, native_execution=False,
        original_contract_sha256=audit.producer.CONTRACT_SHA, counts=[128, 512, 1025],
        native_output_domains=dict(non_logit_stages='<f2', logits='<f4', half_stage_count=50,
            float_stage_count=1, signed_zero_numeric_equal=True), paired_coordinates=442863360,
        stages=51, source_arrays=306, expected_native_dtype_coordinates={'<f2':225480960,'<f4':217382400},
        maximum_block_coordinates=65536, audit_guard_bytes=512*1024**2,
        engine_working_set_limit_bytes=6*1024**3, disk_floor_bytes=2*1024**3, deadline_seconds=1800,
        full_quality_targets_unchanged=10000, full_quality_tasks_unchanged=100,
        quality_scored=False, model_adoption_qualified=False, performance_or_overhead_qualified=False,
        native_schema_source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest()
            for p in ['src/metal/diagnostic.rs','src/metal/mod.rs','tools/bench/minicpm5_depth.rs']})


class Precision(unittest.TestCase):
    def test_existing_files_directories_and_dangling_links_reject_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            audit.fresh_report(root/'absent')
            existing=root/'file';existing.write_bytes(b'unchanged')
            dangling=root/'dangling';dangling.symlink_to(root/'missing')
            for path in [root,existing,dangling]:
                with self.assertRaisesRegex(ValueError,'Fresh absent'):
                    audit.fresh_report(path)
            self.assertEqual(existing.read_bytes(),b'unchanged')

    def test_cold_import_and_complete_source_closure(self):
        for name in ['numpy', 'mlx.core', 'torch']:
            self.assertNotIn(name, sys.modules)
        self.assertEqual(len(audit.sources()), 14)
        self.assertEqual(len(set(audit.sources())), 14)

    def test_touching_continuous_intervals_need_not_contain_a_native_value(self):
        for dtype, exponent in [('<f2', -10), ('<f4', -23)]:
            a, b = 1., 1.+3*2.**exponent
            gate = dict(atol=1.5*2.**exponent, rtol=0.)
            lower = max(a-gate['atol'], b-gate['atol'])
            upper = min(a+gate['atol'], b+gate['atol'])
            self.assertEqual(lower, upper)
            value, feasible = audit.scalar_first(a, b, gate, dtype)
            self.assertFalse(feasible)
            self.assertEqual(value, 1.+2*2.**exponent)
            self.assertGreater(value, upper)

    def test_one_step_wider_literal_intersection_is_feasible(self):
        for dtype, exponent in [('<f2', -10), ('<f4', -23)]:
            value, feasible = audit.scalar_first(1., 1.+3*2.**exponent,
                                                dict(atol=2*2.**exponent, rtol=0.), dtype)
            self.assertTrue(feasible)
            self.assertEqual(value, 1.+2.**exponent)

    def test_scalar_search_agrees_with_exhaustive_finite_half_membership(self):
        cases = [(1., 1.+3*2.**-10, 1.5*2.**-10),
                 (1., 1.+3*2.**-10, 2*2.**-10), (0., -0., 1.e-30),
                 (-2.**-14, 2.**-14, 2.**-13)]
        finite = [audit.decode(bits, '<f2') for bits in range(65536)]
        finite = [x for x in finite if math.isfinite(x)]
        self.assertEqual(len(finite), 63488)
        for a, b, tolerance in cases:
            members = [x for x in finite if abs(x-a) <= tolerance and abs(x-b) <= tolerance]
            value, feasible = audit.scalar_first(a, b, dict(atol=tolerance, rtol=0.), '<f2')
            self.assertEqual(feasible, bool(members))
            if members:
                self.assertEqual(value, min(members))

    def test_subnormals_signed_zeros_binades_and_finite_extremes(self):
        for dtype, minimum, maximum in [('<f2', 2.**-24, 65504.),
                                       ('<f4', 2.**-149, audit.decode(0x7f7fffff, '<f4'))]:
            for a in [0., -0., minimum, -minimum, 1., -1., maximum, -maximum]:
                value, feasible = audit.scalar_first(a, a, dict(atol=minimum/4, rtol=0.), dtype)
                self.assertTrue(feasible)
                self.assertEqual(value, a)

    def test_nonfinite_untyped_and_invalid_policy_reject(self):
        for a, b in [(math.inf, 1.), (0., math.nan), (True, 0.), (0., '1')]:
            with self.assertRaises(ValueError):
                audit.scalar_first(a, b, dict(atol=.1, rtol=.001), '<f2')
        for gate in [dict(atol=0., rtol=0.), dict(atol=.1, rtol=-1.),
                     dict(atol=True, rtol=0.), dict(atol=math.inf, rtol=0.),
                     dict(atol=.1, rtol=math.nan)]:
            with self.assertRaises(ValueError):
                audit.scalar_first(0., 0., gate, '<f4')

    def test_schema_precision_coverage_workloads_and_resource_policy_are_fixed(self):
        value = fixture()
        audit.validate_contract(value)
        changes = [('prospective_before_representability_outcomes', 1),
                   ('gates_recomputed_or_relaxed', True), ('native_precision_changed', True),
                   ('native_execution', True), ('counts', [128, 512]), ('stages', 50),
                   ('source_arrays', 305), ('paired_coordinates', True),
                   ('expected_native_dtype_coordinates', {'<f4':442863360}),
                   ('audit_guard_bytes', 6*1024**3), ('full_quality_tasks_unchanged', 10),
                   ('model_adoption_qualified', True), ('native_schema_source_sha256', {})]
        for key, got in changes:
            with self.assertRaises(ValueError, msg=key):
                audit.validate_contract(dict(value, **{key:got}))
        changed = copy.deepcopy(value)
        changed['native_output_domains']['non_logit_stages'] = '<f4'
        with self.assertRaises(ValueError):
            audit.validate_contract(changed)
        changed = copy.deepcopy(value)
        changed['native_schema_source_sha256']['src/metal/diagnostic.rs'] = 'a'*64
        with self.assertRaises(ValueError):
            audit.validate_contract(changed)


if __name__ == '__main__':
    unittest.main()
