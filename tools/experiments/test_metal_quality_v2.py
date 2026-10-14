"""Frozen v2 mode, guard ordering and evidence retention; no models/GPU."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import metal_quality_v2 as v2
from test_metal_quality import fixture


class V2Loss(unittest.TestCase):
    def test_both_real_modes_require_exact_storage_geometry_and_memory(self):
        for label,spec in v2.SPECS.items():
            tokens=[1,2,3];event,*_,sha=fixture(tokens)
            event.update(weight_dtype='Q4 v2 + FP16 exclusions',q4_format='V2',
                         q4_quantizer='toklet-q4-affine-f32-source-half-metadata-rne-g8-v2',
                         hidden_size=spec['hidden'],num_layers=spec['layers'],vocab_size=151936,
                         projection_mode='Q4CompensatedTiled',normalization_mode='Compensated',
                         q4_compensated_execution_widths=[32,32])
            event['memory']['weights_bytes']=spec['weights']
            event['memory']['admitted_bytes']=sum(v for k,v in event['memory'].items() if k!='admitted_bytes')
            event['memory_admitted_bytes']=event['memory']['admitted_bytes']+64*1024**2
            v2.ready(event,tokens,sha,True,128,label=label)
            for key,value in [('weight_dtype','Q4 v1 + FP16 exclusions'),('q4_format','V1'),('q4_quantizer','other'),
                              ('hidden_size',True),('num_layers',23),('projection_mode','Q4Compensated'),
                              ('normalization_mode','Original'),('q4_compensated_execution_widths',[32,64]),
                              ('q4_compensated_execution_widths',[True,True]),('q4_compensated_execution_widths',[0,0]),
                              ('memory_admitted_bytes',v2.shared.LIMIT+1)]:
                bad=copy.deepcopy(event);bad[key]=value
                with self.subTest(label=label,key=key),self.assertRaises(ValueError):v2.ready(bad,tokens,sha,True,128,label=label)
            bad=copy.deepcopy(event);bad['memory']['weights_bytes']-=1;bad['memory']['admitted_bytes']-=1;bad['memory_admitted_bytes']-=1
            with self.assertRaisesRegex(ValueError,'residency'):v2.ready(bad,tokens,sha,True,128,label=label)
            with self.assertRaises(ValueError):v2.ready(event,tokens,sha,False,128,label=label)

    def collect(self,records,full=True,label='small',result=None):
        probes=[dict(name=n,tokens=t,corpus_sha256=sha,path='/probe',source_nll=[1.,1.,1.],
                     source_logit_path='/source',source_logit_sha256='pin') for n,t,sha in v2.probe_specs()]
        baseline={'probes':copy.deepcopy(probes),'runs':[{'losses':[{'negative_log_likelihood':2.}]*3} for _ in range(4)]}
        with patch.object(v2.shared.quality,'probe_inputs',return_value=probes),patch.object(v2.shared,'child',side_effect=records) as worker:
            out=v2.collect(Path('/binary'),Path('/source'),Path('/packed'),Path('/corpus'),[1]*10001,baseline,label,None,None,full,result=result)
            return out,worker.call_args_list

    def test_failed_worker_stops_full_and_retains_completed_rows(self):
        records=[{'passed':True,'losses':[{'negative_log_likelihood':1.}]*3},
                 {'passed':False,'error':'native failed','losses':[]}]
        out,calls=self.collect(records)
        self.assertEqual(len(calls),2);self.assertEqual(out['runs'],records)
        self.assertFalse(out['probes_completed']);self.assertFalse(out['collection_completed']);self.assertFalse(out['passed'])

    def test_guard_failure_retains_probe_and_never_runs_full(self):
        result={}
        good={'passed':True,'losses':[{'negative_log_likelihood':1.}]*3}
        bad={'passed':True,'losses':[{'negative_log_likelihood':1.01}]*3}
        with self.assertRaisesRegex(ValueError,'gate failed'):self.collect([good,bad],result=result)
        self.assertEqual(len(result['runs']),2);self.assertEqual(len(result['probes']),2)
        self.assertFalse(result['probes_completed']);self.assertFalse(result['quality_completed'])

    def test_full_loss_runs_only_after_four_guards_and_quality_is_separate(self):
        good={'passed':True,'losses':[{'negative_log_likelihood':1.}]*3}
        for delta in (0.,.051):
            out,calls=self.collect([good]*4+[{'passed':True,'mean_negative_log_likelihood':v2.SPECS['small']['mean']+delta}])
            self.assertEqual([c.args[6] for c in calls],[1,128,1,128,128])
            self.assertTrue(out['probes_completed']);self.assertTrue(out['quality_completed']);self.assertTrue(out['collection_completed'])
            self.assertEqual(out['passed'],delta==0.);self.assertEqual(out['quality_gate']['maximum_increase_nats_per_token'],.05)
            self.assertEqual(calls[-1].kwargs['extra_args'],('small',v2.MODE))
        out,calls=self.collect([good]*4,full=False)
        self.assertTrue(out['passed']);self.assertFalse(out['quality_completed']);self.assertEqual(len(calls),4)

    def test_larger_uses_matched_larger_baseline_not_small_informational_loss(self):
        good={'passed':True,'losses':[{'negative_log_likelihood':2.}]*3}
        out,calls=self.collect([good]*4,full=False,label='larger')
        self.assertEqual(out['probes'][0]['matched_fp32_nll'],[2.]*3)
        self.assertNotIn('source_nll',out['probes'][0]);self.assertEqual(calls[0].kwargs['extra_args'],('larger',v2.MODE))
        self.assertTrue(out['probes_completed'])

    def test_changed_probe_metadata_never_launches_either_model(self):
        admitted=v2.probe_specs()
        probes=[dict(name=n,tokens=t,corpus_sha256=sha,path='/probe',source_nll=[1.]*3,
                     source_logit_path='/source',source_logit_sha256='pin') for n,t,sha in admitted]
        for label in v2.SPECS:
            for field in ('name','tokens','corpus_sha256','order'):
                bad=copy.deepcopy(probes)
                if field=='order':bad.reverse()
                elif field=='tokens':bad[0][field][0]+=1
                else:bad[0][field]='replacement'
                result={}
                with self.subTest(label=label,field=field),patch.object(v2.shared.quality,'probe_inputs',return_value=bad),patch.object(v2.shared,'child') as worker:
                    with self.assertRaisesRegex(ValueError,'frozen admission'):
                        v2.collect(Path('/binary'),Path('/source'),Path('/packed'),Path('/corpus'),[1]*10001,{},label,None,None,True,result=result)
                    worker.assert_not_called();self.assertEqual(result['runs'],[]);self.assertFalse(result['probes_completed'])

    def test_existing_destination_rejects_before_source_or_worker_access(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'evidence.json';p.write_text('immutable')
            with patch.object(v2.shared,'digest') as digest,patch.object(v2.shared,'child') as worker:
                with self.assertRaisesRegex(ValueError,'destination exists'):v2.run(Path('/missing'),p,'small',True)
                digest.assert_not_called();worker.assert_not_called()
            self.assertEqual(p.read_text(),'immutable')


if __name__=='__main__':unittest.main()
