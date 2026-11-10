"""Portable calibrated-profile damage controls; no model, NumPy or GPU."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import minicpm5_profile as reader

CONTRACT = json.loads((Path(__file__).resolve().parents[2] / 'docs/minicpm5-profile-contract.json').read_text())


def owner():
    return dict(started_ns=1, completed_ns=1000000001, passed=True, exit_code=0,
                failure=None, owned_group_absent=True)


def clock():
    fields = {key: .000001 for key in reader.TIMING_FIELDS if not key.startswith('gpu_')}
    fields.update(host_encode_seconds=.0008, gpu_start_seconds=100., gpu_end_seconds=100.002, command_count=1)
    return fields


def fixture(first='baseline-first'):
    header = dict(event='ready', format='toklet-minicpm5-profile-v1', repository=CONTRACT['repository'],
        revision=CONTRACT['revision'], source_manifest_sha256=CONTRACT['source_manifest_sha256'],
        device='Apple M1 Pro', profile=CONTRACT['profile'], capacity=512, prefill_chunk=32,
        prompt=[0, 7, 13, 3], teacher_tokens=[6, 978, 44004, 130073], first_order=first,
        warmups=3, measured_samples=10, events_per_phase=718,
        baseline_memory=copy.deepcopy(CONTRACT['expected_baseline_memory']),
        profile_memory=copy.deepcopy(CONTRACT['expected_profile_memory']), shared_weights_bytes=2674218150,
        application_allowance_bytes=33554432, combined_admitted_bytes=2998202878,
        load_timing=dict(load_seconds=.001, conversion_seconds=.0006, compilation_seconds=.0002),
        ordinary_request_clock_available=False,
        clock_scope='completed evaluate_next_chunk only; excludes comparison, selection accessor, counter resolve and JSON')
    records = [header]
    for sample in range(13):
        for step in range(4):
            stages = []
            ordinal = sample * 4 + step
            gpu_origin = 10000000 + ordinal * 100000
            cpu_origin = 1000000000 + ordinal * 10000000
            profile_first = (first == 'profile-first') ^ ((sample + step) % 2 == 1)
            base_clock, profile_clock = clock(), clock()
            base_clock['gpu_start_seconds'] = 100. + ordinal * .01 + (.003 if profile_first else 0)
            profile_clock['gpu_start_seconds'] = 100. + ordinal * .01 + (0 if profile_first else .003)
            for value in [base_clock, profile_clock]:value['gpu_end_seconds'] = value['gpu_start_seconds'] + .002
            for i, (layer, role, kernel) in enumerate(reader.INVENTORY):
                begin = gpu_origin + 100 + i * 30
                end = begin + 20
                event = dict(name=kernel, host_begin_seconds=(1000 + i * 1000) / 1e9,
                    host_end_seconds=(1100 + i * 1000) / 1e9, gpu_begin_ticks=begin, gpu_end_ticks=end,
                    gpu_begin_seconds=(begin - gpu_origin) / 30000 * 3000000 / 1e9,
                    gpu_end_seconds=(end - gpu_origin) / 30000 * 3000000 / 1e9)
                stages.append(dict(index=i, layer=layer, role=role, sample=event))
            records.append(dict(event='phase', sample=sample, warmup=sample < 3, step=step,
                rows=4 if step == 0 else 1, profile_first=(first == 'profile-first') ^ ((sample + step) % 2 == 1),
                baseline=dict(wall_nanoseconds=2000000, completed_timing=base_clock),
                instrumented=dict(wall_nanoseconds=3000000, completed_timing=profile_clock),
                committed_after=4 + step, pending_after=0, history_after=4 + step,
                selected=CONTRACT['teacher_tokens'][step], all_logits_exact=True,
                logits_sha256=CONTRACT['expected_logits_sha256'][step],
                clock_start=dict(cpu_nanoseconds=cpu_origin, gpu_ticks=gpu_origin),
                clock_end=dict(cpu_nanoseconds=cpu_origin + 3000000, gpu_ticks=gpu_origin + 30000),
                host_clock_sample_bracket_seconds=.0000001, stages=stages))
    records.append(dict(event='complete', phase_pairs=52, compared_logits_values=6789120,
        ordinary_telemetry_overhead_qualified=False, model_quality_adoption_qualified=False,
        engine_optimization_adopted=False))
    return records


def damage(records, index, path, value):
    result = list(records)
    result[index] = copy.deepcopy(result[index])
    node = result[index]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return result


class Profiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = fixture()

    def validate(self, raw, receipt=None, first='baseline-first'):
        return reader.validate(raw, owner() if receipt is None else receipt, CONTRACT, first)

    def test_both_orders_exact_coverage_work_and_calibrated_units(self):
        for first in ['baseline-first', 'profile-first']:
            result = self.validate(self.raw if first == 'baseline-first' else fixture(first), first=first)
            self.assertEqual(result['events'], 37336)
            self.assertEqual(result['compared_logits_values'], 6789120)
            self.assertEqual(result['measured_phase_pairs'], 40)
            self.assertEqual(result['completed_disjoint_wall_ns'], 260000000)
            self.assertAlmostEqual(result['all_samples'][0]['sampled_stage_gpu_seconds'], .001436)
            self.assertAlmostEqual(result['all_samples'][0]['role_gpu_seconds']['head'], .000002)
            self.assertEqual(result['depths'][0]['diagnostic_phase_inflation_percent'], 50.)
            self.assertFalse(result['one_percent_telemetry_overhead_qualified'])

    def test_coverage_identity_state_and_prior_capture_parity_are_binding(self):
        for path, value in [(['sample'], True), (['step'], 1), (['warmup'], False), (['rows'], 1),
                            (['pending_after'], 1), (['selected'], 1), (['all_logits_exact'], False),
                            (['logits_sha256'], '0' * 64), (['stages', 0, 'index'], True),
                            (['stages', 1, 'layer'], True), (['stages', 0, 'role'], 'head'),
                            (['stages', 0, 'sample', 'name'], 'mlx_product')]:
            with self.assertRaises(ValueError):
                self.validate(damage(self.raw, 1, path, value))
        with self.assertRaises(ValueError):
            self.validate(self.raw[:-1])
        with self.assertRaises(ValueError):
            self.validate([self.raw[0], self.raw[2], self.raw[1], *self.raw[3:]])
        with self.assertRaises(ValueError):
            self.validate(damage(self.raw, -1, ['engine_optimization_adopted'], True))

    def test_disjoint_load_and_all_phase_pairs_share_owned_lifetime(self):
        with self.assertRaisesRegex(ValueError, 'Disjoint load'):
            self.validate(damage(self.raw, 0, ['load_timing', 'load_seconds'], .99))
        receipt = owner();receipt['completed_ns'] = 260000001
        with self.assertRaisesRegex(ValueError, 'Disjoint load'):
            self.validate(self.raw, receipt)
        for field in ['baseline', 'instrumented']:
            row = copy.deepcopy(self.raw[1][field])
            for key in row['completed_timing']:
                if key.endswith('_seconds') and not key.startswith('gpu_'):
                    row['completed_timing'][key] = .001
            with self.assertRaisesRegex(ValueError, 'host/GPU interval'):
                reader.timing(row)

    def test_calibration_raw_ticks_and_exported_seconds_cannot_be_relabelled(self):
        for path, value in [(['clock_start', 'cpu_nanoseconds'], True),
                            (['clock_end', 'cpu_nanoseconds'], 999999999),
                            (['clock_end', 'cpu_nanoseconds'], 1010000000),
                            (['stages', 0, 'sample', 'gpu_begin_ticks'], True),
                            (['stages', 0, 'sample', 'gpu_begin_ticks'], 9999999),
                            (['stages', 0, 'sample', 'gpu_end_ticks'], 10040000),
                            (['stages', 0, 'sample', 'gpu_begin_seconds'], .01),
                            (['stages', 0, 'sample', 'gpu_end_seconds'], float('nan')),
                            (['stages', 1, 'sample', 'gpu_begin_ticks'], 10000100),
                            (['stages', 1, 'sample', 'host_begin_seconds'], .000001),
                            (['host_clock_sample_bracket_seconds'], .1)]:
            row = damage(self.raw, 1, path, value)[1]
            with self.assertRaises(ValueError):
                reader.counters(row, 3000000)

    def test_serial_origins_and_encode_calibration_command_scopes_are_binding(self):
        repeated=list(self.raw)
        for index in range(1,53):
            repeated[index]=dict(repeated[index])
            for key in ['clock_start','clock_end','host_clock_sample_bracket_seconds','stages']:
                repeated[index][key]=self.raw[1][key]
        with self.assertRaisesRegex(ValueError,'calibration origins'):self.validate(repeated)
        repeated=list(self.raw)
        repeated[2]=copy.deepcopy(repeated[2])
        for execution in ['baseline','instrumented']:
            repeated[2][execution]['completed_timing']=self.raw[1][execution]['completed_timing']
        with self.assertRaisesRegex(ValueError,'command timestamps'):self.validate(repeated)
        row=copy.deepcopy(self.raw[1])
        for index,stage in enumerate(row['stages']):
            stage['sample']['host_begin_seconds']=(1000+index*2000)/1e9
            stage['sample']['host_end_seconds']=(2500+index*2000)/1e9
        with self.assertRaisesRegex(ValueError,'completed encode'):reader.counters(row,3000000)
        row=copy.deepcopy(self.raw[1])
        for index,stage in enumerate(row['stages']):
            stage['sample']['host_begin_seconds']=(1000+index*3000)/1e9
            stage['sample']['host_end_seconds']=(1100+index*3000)/1e9
        with self.assertRaisesRegex(ValueError,'span and gaps'):reader.counters(row,3000000)
        row=copy.deepcopy(self.raw[1]);row['instrumented']['completed_timing']['clock_calibration_seconds']=0
        with self.assertRaisesRegex(ValueError,'sample bracket'):reader.counters(row,3000000)
        row=copy.deepcopy(self.raw[1]);timing=row['instrumented']['completed_timing']
        timing['gpu_end_seconds']=timing['gpu_start_seconds']+.0005
        with self.assertRaisesRegex(ValueError,'completed command interval'):reader.counters(row,3000000)

    def test_whole_serial_cpu_and_gpu_spans_plus_prior_load_fit_owned_lifetime(self):
        changed=list(self.raw);changed[-2]=copy.deepcopy(changed[-2])
        for key in ['clock_start','clock_end']:changed[-2][key]['cpu_nanoseconds']+=10**9
        with self.assertRaisesRegex(ValueError,'CPU calibration series'):self.validate(changed)
        changed=list(self.raw);changed[-2]=copy.deepcopy(changed[-2])
        for execution in ['baseline','instrumented']:
            for key in ['gpu_start_seconds','gpu_end_seconds']:
                changed[-2][execution]['completed_timing'][key]+=1
        with self.assertRaisesRegex(ValueError,'GPU command series'):self.validate(changed)
        receipt=owner();receipt['completed_ns']=600000001
        changed=damage(self.raw,0,['load_timing','load_seconds'],.1)
        with self.assertRaisesRegex(ValueError,'CPU calibration series'):self.validate(changed,receipt)
        receipt=owner();receipt['completed_ns']=535000001
        changed=damage(self.raw,0,['load_timing','load_seconds'],.0205)
        with self.assertRaisesRegex(ValueError,'GPU command series'):self.validate(changed,receipt)

    def test_metal_error_sentinel_and_unbounded_command_epoch_grain_reject(self):
        row=copy.deepcopy(self.raw[1]);row['clock_end']['gpu_ticks']=2**64-1
        row['stages'][-1]['sample']['gpu_end_ticks']=2**64-1
        begin=row['clock_start']['gpu_ticks'];span=row['clock_end']['gpu_ticks']-begin
        for stage in row['stages']:
            for ticks,seconds in [('gpu_begin_ticks','gpu_begin_seconds'),('gpu_end_ticks','gpu_end_seconds')]:
                stage['sample'][seconds]=(stage['sample'][ticks]-begin)/span*3000000/1e9
        with self.assertRaisesRegex(ValueError,'counter-error sentinel'):reader.counters(row,3000000)
        row=copy.deepcopy(self.raw[1]);timing=row['instrumented']['completed_timing']
        timing['gpu_start_seconds']=float(2**40);timing['gpu_end_seconds']=float(2**40)+.002
        with self.assertRaisesRegex(ValueError,'timestamp grain'):reader.counters(row,3000000)

    def test_exact_owners_shared_weights_and_unknown_metrics_stay_explicit(self):
        for path, value in [(['profile_memory', 'metadata_allowance_bytes'], 0),
                            (['combined_admitted_bytes'], 5672421028),
                            (['shared_weights_bytes'], True), (['capacity'], 512.),
                            (['ordinary_request_clock_available'], True), (['profile'], 'Original')]:
            with self.assertRaises(ValueError):
                self.validate(damage(self.raw, 0, path, value))
        raw = damage(self.raw, 0, ['baseline_memory', 'scratch_bytes'], 0)
        raw[0]['baseline_memory']['admitted_bytes'] -= CONTRACT['expected_baseline_memory']['scratch_bytes']
        raw[0]['combined_admitted_bytes'] -= CONTRACT['expected_baseline_memory']['scratch_bytes']
        with self.assertRaises(ValueError):self.validate(raw)

    def test_modified_contract_cannot_relabel_fixed_work_owners_or_source(self):
        for key, value in [('repository', 'other/model'), ('source_manifest_sha256', '0'*64),
                           ('teacher_tokens', [6,978,44004,1]), ('capacity', 1024),
                           ('written_before_first_native_profile', 1), ('combined_admitted_bytes', 2674218150)]:
            contract=copy.deepcopy(CONTRACT);contract[key]=value
            with self.assertRaises(ValueError):reader.validate_contract(contract)
        contract=copy.deepcopy(CONTRACT);contract['expected_baseline_memory']['scratch_bytes']=0
        with self.assertRaises(ValueError):reader.validate_contract(contract)

    def test_completed_owned_receipts_reject_invalid_types_and_liveness(self):
        for field, value in [('passed', False), ('exit_code', True), ('started_ns', 0),
                             ('completed_ns', 1.), ('failure', 'cancelled'), ('owned_group_absent', False)]:
            receipt = owner();receipt[field] = value
            with self.assertRaises(ValueError):self.validate(self.raw, receipt)

    def test_jsonl_bytes_bind_receipt_and_reject_duplicates_partial_or_excess_records(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'profile.jsonl'
            for raw, accepted in [(b'{"x":0}\n', True), (b'{"x":0,"x":1}\n', False),
                                  (b'{}', False), (b'{}\n' * 55, False), (b'x' * (2 * 1024 ** 2 + 1), False)]:
                path.write_bytes(raw);receipt = dict(owner(), profile_report_sha256=hashlib.sha256(raw).hexdigest())
                if accepted:
                    self.assertEqual(reader.read(path, receipt), [{'x': 0}])
                else:
                    with self.assertRaises(ValueError):reader.read(path, receipt)
            path.write_bytes(b'{}\n')
            with self.assertRaises(ValueError):reader.read(path, dict(owner(), profile_report_sha256='0' * 64))


if __name__ == '__main__':unittest.main()
