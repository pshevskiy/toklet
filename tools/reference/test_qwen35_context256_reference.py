import unittest
from unittest import mock
import json
from pathlib import Path
import tempfile
import qwen35_context256_reference as reference


class Plan(unittest.TestCase):
    def test_source_inputs_cycle_exact_prior_ids_and_keep_zero_absolute_offset(self):
        tokens,offset,values=reference.plan()
        self.assertEqual((len(tokens),offset),(256,0))
        self.assertEqual(tokens[:17],[1,7,13,3,9,5,11,2,17,4,6,10,19,8,21,23,29])
        self.assertEqual(tokens[-1],1)
        tokens[0]=99
        self.assertEqual(reference.plan()[0][0],1)

    def test_full_state_and_all_kv_extents_at_both_checkpoints_are_bounded(self):
        _,_,values=reference.plan()
        self.assertEqual(values,{128:18180608,256:20310528})
        self.assertEqual(sum(values.values())*12,461893632)
        self.assertLess(sum(values.values())*12,reference.MAX_CAPTURE)
        self.assertLessEqual(sum(reference.LEDGER.values()),6*1024**3)

    def test_only_final_token_of_each_full_checkpoint_is_retained(self):
        self.assertEqual([reference.checkpoint(i) for i in [0,126,127,128,254,255]],
                         [None,None,128,None,None,256])
        self.assertEqual(sum(reference.checkpoint(i) is not None for i in range(256)),2)

    def test_typed_checkpoint_coordinates_reject_boolean_and_out_of_bounds(self):
        for value in [True,False,-1,256,1.0,'127',None]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):reference.checkpoint(value)

    def test_prior_graph_and_point_workloads_remain_unchanged(self):
        self.assertEqual(reference.prior.plan('one'),([1],0,16067328))
        self.assertEqual(reference.prior.plan('context17')[1:],(37,16333568))
        self.assertEqual(reference.prior.graph.GATE,dict(atol=1e-5,rtol=1e-4))
        self.assertEqual(reference.DEADLINE_SECONDS,21600)


class FailureClosure(unittest.TestCase):
    def exercise(self,rows,config_error=False):
        class Owner:
            closed=False
            def __enter__(self):return self
            def close(self):self.closed=True
            def __exit__(self,*args):self.close()
            def audit(self):return dict(unit_owner=True,closed=self.closed)
        calls=[];owner=Owner()
        def forward(*args,**kwargs):
            for row in rows:kwargs['sink'](row.copy())
            return []
        reference.ROOT.joinpath('artifacts').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=reference.ROOT/'artifacts',prefix='unit-context256-') as folder:
            output=Path(folder)/'capture'
            with mock.patch.object(reference.prior.runtime_owner.runtime,'captured',side_effect=ValueError('unit config') if config_error else None), \
                 mock.patch.object(reference.prior.bf16_source,'Source',return_value=owner), \
                 mock.patch.object(reference.prior.weights,'weights',return_value=({},{})), \
                 mock.patch.object(reference.prior.graph,'forward',side_effect=forward):
                result=reference.collect(output,Path('unused-unit-source'),{},lambda:calls.append('endpoint') or {})
            self.assertFalse(result['completed'])
            self.assertIsNotNone(result['primary_error'])
            self.assertEqual(calls,['endpoint'])
            self.assertEqual(json.loads((output/'result.json').read_text()),result)
            if not config_error:self.assertTrue(owner.closed)
            return result

    def test_preopen_failure_still_closes_runtime_endpoint_without_success(self):
        result=self.exercise([],True)
        self.assertEqual(result['primary_error']['message'],'unit config')
        self.assertIsNone(result['source'])

    def test_duplicate_checkpoint_stage_cannot_publish_completed_reference(self):
        row=dict(token=127,layer=None,stage='embedding',values=[1.0])
        result=self.exercise([row,row])
        self.assertIn('Unique complete finite',result['primary_error']['message'])

    def test_nonfinite_checkpoint_array_cannot_publish_completed_reference(self):
        result=self.exercise([dict(token=255,layer=None,stage='embedding',values=[float('nan')])])
        self.assertIn('Unique complete finite',result['primary_error']['message'])


if __name__=='__main__':unittest.main()
