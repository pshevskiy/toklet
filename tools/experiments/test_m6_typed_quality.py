"""Typed live/replay regression; portable saved fixtures, no model or GPU."""
import argparse
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import m6_fp32_context_audit as mixed
import m6_typed_quality as collector
import m6_typed_quality_audit as auditor


class TypedQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stem = collector.ROOT/'benchmarks/results/2026-10-15-m6-context-small-quality-4096-v1'
        cls.record = json.loads(stem.with_suffix('.json').read_text())
        cls.manifest, cls.arrays = collector.base.shared.contract()
        _, arrays, _, _, _ = collector.base.inputs('small')
        cls.ids = arrays[4096]

    def fixture(self):
        record = copy.deepcopy(self.record)
        record.update(format=collector.FORMAT, collector='m6_typed_quality', fp32_policy=collector.POLICY)
        events = record['worker']['phases'][0]['events']
        # Real larger32K row92 shortest-decimal failure transplanted only for
        # this synthetic type regression; not presented as a native collection.
        events[93].update(reference_greedy=24, online_greedy=24,
            reference_winner_logits=[30.187262,30.160664], online_winner_logits=[30.187262,30.160664],
            reference_margin=12.044475555419922, max_absolute_logit_delta=.026597976684570312,
            rms_logit_delta=.004416703839638852)
        record['quality'], record['fp32_reconstruction'] = collector.comparison(events, self.ids,
            [record['worker']['ready'][f'{m}_memory'] for m in collector.base.MODES], 'small', 4096)
        return record

    @staticmethod
    def raw(record):
        return [record['worker']['ready'], *(e for p in record['worker']['phases'] for e in p['events'])]

    def test_real_row92_precision_failure_recovers_without_raw_or_f64_mutation(self):
        record = self.fixture(); events = record['worker']['phases'][0]['events']
        raw = json.dumps(events, sort_keys=True).encode(); before = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(ValueError,'Winner exceeds vocabulary error'):
            collector.base.comparison(events, self.ids, [record['worker']['ready'][f'{m}_memory'] for m in collector.base.MODES])
        quality, proof = collector.comparison(events, self.ids,
            [record['worker']['ready'][f'{m}_memory'] for m in collector.base.MODES], 'small',4096)
        self.assertTrue(quality['gate_passed']); self.assertEqual(proof['restored_winner_values'],512)
        self.assertEqual(hashlib.sha256(json.dumps(events,sort_keys=True).encode()).hexdigest(),before)
        normalized,_ = collector.values.restore([record])
        for original, typed in zip(events[1:-1],normalized[0]['worker']['phases'][0]['events'][1:-1],strict=True):
            for key in ('reference_nll','online_nll','reference_margin','max_absolute_logit_delta','rms_logit_delta'):
                self.assertEqual(original[key],typed[key])

    def test_true_error_and_unchanged_nll_gate_still_reject(self):
        for failure in ('error','nll'):
            record=self.fixture(); events=record['worker']['phases'][0]['events']
            if failure=='error': events[93]['reference_winner_logits'][1]=30.
            else:
                for event in events[1:-1]: event['online_nll']=event['reference_nll']+.051
            with self.subTest(failure=failure),self.assertRaises(ValueError):
                collector.comparison(events,self.ids,[record['worker']['ready'][f'{m}_memory'] for m in collector.base.MODES],'small',4096)

    def test_separate_replay_keeps_original_bytes_and_requires_all_native_guards(self):
        record=self.fixture(); original=copy.deepcopy(record)
        with patch.object(collector.base.subprocess,'Popen') as launch:
            result=auditor.replay(record,self.raw(record),self.manifest,self.arrays,self.ids)
        launch.assert_not_called(); self.assertEqual(record,original)
        self.assertTrue(result['original_raw_verified_before_restoration']);self.assertEqual(result['raw_events'],131)
        mutations=[lambda r:r.update(format='toklet-m6-context-collection-v1'),
            lambda r:r.update(passed=False),lambda r:r.update(collection_completed=False),
            lambda r:r.update(identities_unchanged=False),lambda r:r.update(endpoint_errors=['changed']),
            lambda r:r['worker'].update(passed=False),lambda r:r['worker'].pop('exit_code'),
            lambda r:r['worker'].update(exit_code=1),lambda r:r['worker'].update(exit_code=True),
            lambda r:r['worker'].update(cleanup_errors=['reap']),
            lambda r:r['worker']['phases'][0]['request']['tokens'].__setitem__(0,1),
            lambda r:r['worker']['phases'][0]['events'].pop(1),
            lambda r:r['fp32_reconstruction'].update(restored_winner_values=511)]
        for mutate in mutations:
            changed=copy.deepcopy(record); mutate(changed)
            with self.assertRaises((ValueError,KeyError)):
                auditor.replay(changed,self.raw(changed),self.manifest,self.arrays,self.ids)
        with self.assertRaisesRegex(ValueError,'raw coverage'):
            auditor.replay(record,self.raw(record)[:-1],self.manifest,self.arrays,self.ids)

    def test_mixed_dispatch_never_relabels_original_or_ignores_conflicts(self):
        paths=[collector.ROOT/f'artifacts/synthetic-route-{i}.json' for i in range(2)]
        loaded=[(paths[0],{'format':'toklet-m6-context-collection-v1'},'old'),(paths[1],{'format':collector.FORMAT},'new')]
        a=dict(passed=True,records=[{'route':'old'}],bindings={'shared':'same'})
        b=dict(passed=True,records=[{'route':'typed'}],bindings={'shared':'same'})
        with patch.object(mixed.audit,'audit',return_value=a) as old,patch.object(mixed.typed_audit,'audit',return_value=b) as new:
            result=mixed.replay(paths,loaded)
        old.assert_called_once_with(paths[:1]);new.assert_called_once_with(paths[1:]);self.assertEqual(len(result['records']),2)
        b['bindings']['shared']='different'
        with patch.object(mixed.audit,'audit',return_value=a),patch.object(mixed.typed_audit,'audit',return_value=b),self.assertRaisesRegex(ValueError,'Conflicting'):
            mixed.replay(paths,loaded)
        for bad_paths,bad_loaded in (([paths[0],paths[0]],loaded), (paths,[(paths[0],{'format':'unknown'},'old')])):
            with self.assertRaises(ValueError):mixed.replay(bad_paths,bad_loaded)

    def test_live_orchestration_preserves_raw_and_failed_evidence(self):
        original=self.fixture()
        for failure in (None,'comparison','exit','endpoint'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as temporary:
                output=Path(temporary)/'record.json'; record=copy.deepcopy(original); exited=[False]
                class FakeWorker:
                    def __init__(self,*args,**kwargs):
                        self.record=record['worker'];self.admission=self.record['ready']
                        self.record.update(passed=False);self.record.pop('exit_code',None)
                    def __enter__(self):return self
                    def call(self,request,final):
                        self.assert_request=request
                        return self.record['phases'][0]['events']
                    def __exit__(self,kind,*args):
                        exited[0]=True
                        if kind is None:
                            if failure=='exit':raise ValueError('injected EOF failure')
                            self.record.update(passed=True,exit_code=0)
                compiled_calls=[0]
                def check():
                    compiled_calls[0]+=1
                    if failure=='endpoint':raise ValueError('injected endpoint failure')
                def compare(*args):
                    self.assertFalse(exited[0])
                    if failure=='comparison':raise ValueError('injected comparison failure')
                    return original['quality'],original['fp32_reconstruction']
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(collector.base,'compiled_admission',return_value=(Path('binary'),{},check)))
                    stack.enter_context(patch.object(collector.base,'inputs',return_value=({}, {4096:self.ids},Path('source'),None,{})))
                    stack.enter_context(patch.object(collector.base.shared,'contract',return_value=({},{})))
                    stack.enter_context(patch.object(collector.base,'controlled_pins',return_value={}))
                    stack.enter_context(patch.object(collector.base,'capture_identities',return_value={}))
                    stack.enter_context(patch.object(collector.base.shared.metal_continue,'observation',return_value='observed'))
                    stack.enter_context(patch.object(collector.base.shared.metal_continue,'endpoints',return_value={}))
                    stack.enter_context(patch.object(collector.wait,'Worker',FakeWorker))
                    stack.enter_context(patch.object(collector,'comparison',side_effect=compare))
                    if failure is None:
                        result=collector.run(output,'small',4096)
                        self.assertTrue(result['passed']);self.assertTrue(exited[0])
                    else:
                        with self.assertRaisesRegex(ValueError,'failed evidence retained'):collector.run(output,'small',4096)
                        result=json.loads(output.read_text());self.assertFalse(result['passed']);self.assertFalse(result['collection_completed'])
                        self.assertTrue(exited[0])
                self.assertTrue(output.with_suffix('.raw.jsonl').exists());self.assertTrue(output.with_suffix('.stderr').exists())
                self.assertEqual(record['worker']['phases'],original['worker']['phases'])


if __name__=='__main__':unittest.main()
