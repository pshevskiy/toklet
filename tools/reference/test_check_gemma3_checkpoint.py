"""Synthetic saved-byte corruption controls; no checkpoint, producer or GPU."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('gemma_checker',HERE/'check_gemma3_checkpoint.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class Controls(unittest.TestCase):
    def test_main_only_gate_and_exact_boundary(self):
        main=np.array([[0.,1.]],dtype=np.float32);wide=np.array([[1.,1.]],dtype=np.float32)
        group={'envelope':1.,'gate':{'atol':2.,'rtol':.001,'fatal_reference':'main'},'gate_basis':c.GATE_BASIS,'envelope_scope':'successor MAIN/WIDE diagnostic only; not a limit input'}
        legacy=dict(group['gate']);c.frozen_gate(main,wide,group,legacy)
        native=np.array([[-2.,1.]],dtype=np.float32)
        result=c.compare(native,main,wide,group['gate'])
        self.assertEqual(result['failures'],0);self.assertEqual(result['wide_exceedances'],1)
        native[0,0]=np.nextafter(np.float32(-2.),np.float32(-np.inf))
        self.assertEqual(c.compare(native,main,wide,group['gate'])['failures'],1)
        for field,value in [('envelope',.9),('gate',{'atol':2.1,'rtol':.001,'fatal_reference':'main'})]:
            bad=copy.deepcopy(group);bad[field]=value
            with self.assertRaises(c.Failure):c.frozen_gate(main,wide,bad,legacy)
        native[0,0]=np.nan
        with self.assertRaises(c.Failure):c.compare(native,main,wide,group['gate'])

    def test_changed_envelope_cannot_increase_legacy_limits_or_make_wide_fatal(self):
        main=np.array([[0.]],dtype=np.float32);wide=np.array([[100.]],dtype=np.float32)
        legacy=dict(atol=.0001,rtol=.001,fatal_reference='main')
        group=dict(envelope=100.,gate=dict(legacy),gate_basis=c.GATE_BASIS,envelope_scope='successor MAIN/WIDE diagnostic only; not a limit input')
        c.frozen_gate(main,wide,group,legacy)
        self.assertEqual(c.compare(main,main,wide,legacy)['failures'],0)
        self.assertEqual(c.compare(main,main,wide,legacy)['wide_exceedances'],1)
        changed=copy.deepcopy(group);changed['gate']['atol']=200.
        with self.assertRaisesRegex(c.Failure,'literal legacy'):c.frozen_gate(main,wide,changed,legacy)
        changed=copy.deepcopy(group);changed['gate_basis']='recomputed_envelope'
        with self.assertRaises(c.Failure):c.frozen_gate(main,wide,changed,legacy)
        desc=dict(shape=[1,1],dtype='u16le',bytes=2,sha256='1'*64)
        old=dict(main=dict(desc),wide=dict(desc));now=copy.deepcopy(old)
        now['main']['sha256']='2'*64;c.wide_identity(now,old)
        now['wide']['sha256']='2'*64
        with self.assertRaises(c.Failure):c.wide_identity(now,old)
        now=copy.deepcopy(old);now['main']['shape']=[1,2]
        with self.assertRaises(c.Failure):c.wide_identity(now,old)

    def test_full_source_profile_and_basis_metadata(self):
        cases=[dict(name='synthetic')];input_={'cases':cases}
        counts=dict(graph_stage_groups=3200,teacher_stage_groups=32,total_arrays=6464)
        legacy=dict(format='toklet-mlx-gemma3-real-reference-v2',protocol_sha256=c.LEGACY_PROTOCOL_SHA,cases=cases,counts=counts)
        limits=dict(format='toklet-gemma3-legacy-numeric-limits-v1',source={'sha256':c.LEGACY_SOURCE_SHA},group_count=3232)
        good=dict(format='toklet-mlx-gemma3-real-reference-v3',source_reference_executed=True,target_gpu_executed=False,
                  profile=copy.deepcopy(c.PROFILE_CONTRACT),policy=copy.deepcopy(c.POLICY_CONTRACT),gate_basis=copy.deepcopy(c.GATE_BASIS_CONTRACT),
                  protocol_sha256=c.PROTOCOL_SHA,input_sha256=c.INPUT_SHA,cases=cases,counts=counts)
        c.source_metadata(good,input_,legacy,limits)
        for keys,value in [
                (('profile','main'),'FP32 BLAS'),(('profile','rms'),'FP64 RMS'),
                (('profile','projection_certificate','stripe_outputs'),2048),
                (('profile','attention'),'FP32 BLAS'),(('policy','fatal_reference'),'wide'),
                (('policy','limits_sha256'),'0'*64),(('gate_basis','file_pin','sha256'),'0'*64),
                (('counts','teacher_stage_groups'),31),(('source_reference_executed',),False),
                (('target_gpu_executed',),True),(('protocol_sha256',),c.LEGACY_PROTOCOL_SHA)]:
            bad=copy.deepcopy(good);at=bad
            for key in keys[:-1]:at=at[key]
            at[keys[-1]]=value
            with self.assertRaises(c.Failure):c.source_metadata(bad,input_,legacy,limits)
        bad=copy.deepcopy(good);bad['policy']['atol_floor']=.5
        with self.assertRaises(c.Failure):c.source_metadata(bad,input_,legacy,limits)
        bad=copy.deepcopy(limits);bad['group_count']=3231
        with self.assertRaises(c.Failure):c.source_metadata(good,input_,legacy,bad)

    def test_posthoc_mode_never_qualifies_or_accepts_wrong_capture_source(self):
        new=Path('/synthetic/new/manifest.json');old=Path('/synthetic/old/manifest.json');model=Path('/synthetic/model')
        synthetic_sha='a'*64
        historical=dict(format='toklet-gemma3-real-native-v1',source_manifest_sha256=c.LEGACY_SOURCE_SHA,
                        input_sha256=c.INPUT_SHA,protocol_sha256=c.LEGACY_PROTOCOL_SHA,source_manifest=str(old),model=str(model))
        result=c.capture_identity(historical,'historical-posthoc',new,old,model,synthetic_sha)
        self.assertTrue(result['retrospective_diagnostic']);self.assertFalse(result['numerical_qualification'])
        self.assertFalse(result['numerical_qualification_eligible'])
        with self.assertRaises(c.Failure):c.capture_identity(historical,'fresh-preregistered',new,old,model,synthetic_sha)
        current=dict(format='toklet-gemma3-real-native-v2',source_manifest_sha256=synthetic_sha,
                     input_sha256=c.INPUT_SHA,protocol_sha256=c.PROTOCOL_SHA,source_manifest=str(new),model=str(model),
                     source_profile=c.PROFILE_CONTRACT['name'],source_gate_basis=c.GATE_BASIS,
                     legacy_limits_sha256=c.LEGACY_LIMITS_SHA,legacy_source_manifest_sha256=c.LEGACY_SOURCE_SHA)
        result=c.capture_identity(current,'fresh-preregistered',new,old,model,synthetic_sha)
        self.assertFalse(result['retrospective_diagnostic']);self.assertFalse(result['numerical_qualification'])
        self.assertFalse(result['numerical_qualification_eligible'])
        with self.assertRaises(c.Failure):c.capture_identity(current,'historical-posthoc',new,old,model,synthetic_sha)
        for key,value in [('source_manifest_sha256',c.LEGACY_SOURCE_SHA),('source_manifest',str(old)),
                          ('protocol_sha256',c.LEGACY_PROTOCOL_SHA),('legacy_limits_sha256','b'*64),
                          ('source_profile','old'),('source_gate_basis','recomputed')]:
            bad=dict(current);bad[key]=value
            with self.assertRaises(c.Failure):c.capture_identity(bad,'fresh-preregistered',new,old,model,synthetic_sha)

    def test_final_endpoint_failure_revokes_qualification_eligibility(self):
        for mode in ('fresh-preregistered','historical-posthoc'):
            for passed,error,endpoint in [(True,None,None),(False,None,None),
                    (True,'read failure',None),(True,None,'changed authenticated input'),
                    (False,'exception','endpoint failure')]:
                result=dict(capture_mode=mode,passed=passed,error=error,endpoint_error=endpoint,
                            numerical_qualification=True,numerical_qualification_eligible=True)
                c.finalize_qualification(result)
                self.assertFalse(result['numerical_qualification'])
                self.assertEqual(result['native_conformance_passed'],passed and not error and not endpoint)
                self.assertEqual(result['numerical_qualification_eligible'],
                    mode=='fresh-preregistered' and passed and not error and not endpoint)
        incomplete={'capture_mode':'fresh-preregistered'}
        c.finalize_qualification(incomplete)
        self.assertFalse(incomplete['numerical_qualification_eligible'])

    def test_absolute_materialized_raw_score_columns(self):
        source=np.arange(4*1*544,dtype=np.float32).reshape(4,1,544)
        record=dict(row=513,absolute_key_start=1,shape=[4,1,543])
        sliced=c.source_slice(source,'layers.0.attention_scores',513,512,32,record)
        np.testing.assert_array_equal(sliced,source[:,:,1:544])
        # This includes raw dot columns514..543, not zero-fabricated future scores.
        self.assertGreater(sliced[0,0,-1],0)
        global_record=dict(row=513,absolute_key_start=0,shape=[4,1,544])
        np.testing.assert_array_equal(c.source_slice(source,'layers.5.attention_scores',513,512,32,global_record),source)
        for key,value in [('row',512),('absolute_key_start',2),('shape',[4,1,542])]:
            bad=dict(record);bad[key]=value
            with self.assertRaises(c.Failure):c.source_slice(source,'layers.0.attention_scores',513,512,32,bad)
        probs=np.zeros((4,1,543),dtype=np.float32);probs[:,:,1:513]=1/512
        self.assertEqual(c.probability_mask(probs,0,513,1),4*31)
        probs[0,0,-1]=np.nextafter(np.float32(0),np.float32(1))
        with self.assertRaises(c.Failure):c.probability_mask(probs,0,513,1)

    def test_descriptors_hash_extents_overlap_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'raw';raw=np.array([1.,-0.,2.,3.],dtype='<f4').tobytes();path.write_bytes(raw)
            desc=dict(offset=0,bytes=16,shape=[2,2],dtype='f32le',sha256=c.sha(raw))
            owner=c.Bytes(path)
            try:
                np.testing.assert_array_equal(owner.array(desc),[[1.,-0.],[2.,3.]])
                for key,value in [('offset',4),('bytes',12),('shape',[1,4,1,1]),('sha256','0'*64),('dtype','f64le')]:
                    bad=dict(desc);bad[key]=value
                    with self.assertRaises(c.Failure):owner.array(bad)
            finally:owner.close()
            c.cover([dict(offset=0,bytes=8),dict(offset=8,bytes=8)],16)
            for descriptors in [[dict(offset=0,bytes=12),dict(offset=8,bytes=8)],[dict(offset=0,bytes=8),dict(offset=12,bytes=4)]]:
                with self.assertRaises(c.Failure):c.cover(descriptors,16)
            for value in [np.nan,np.inf,-np.inf]:
                raw=np.array([value],dtype='<f4').tobytes();path.write_bytes(raw);owner=c.Bytes(path)
                try:
                    with self.assertRaises(c.Failure):owner.array(dict(offset=0,bytes=4,shape=[1],dtype='f32le',sha256=c.sha(raw)))
                finally:owner.close()

    def test_native_capture_case_row_key_shape_and_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=np.array([1.,2.],dtype='<f4').tobytes();(root/'raw').write_bytes(raw)
            fixed=dict(label='plain_france.whole',capture_kind='source_stage',row=0,stage='embedding',shape=[1,2],absolute_key_start=0,offset=0,bytes=8)
            start=dict(kind='capture_started',**fixed);end=dict(kind='capture',dtype='f32le',finite=True,sha256=c.sha(raw),**fixed)
            for changed in [None,('label','wrong'),('row',1),('absolute_key_start',1),('shape',[2,1]),('offset',4)]:
                pair=[copy.deepcopy(start),copy.deepcopy(end)]
                if changed:
                    for x in pair:x[changed[0]]=changed[1]
                (root/'journal').write_text(''.join(json.dumps(x)+'\n' for x in pair));owner=c.Bytes(root/'raw')
                try:
                    audit=c.Audit(c.Reader(),{}, {},None,owner,c.Events(root/'journal'),None)
                    if changed:
                        with self.assertRaises(c.Failure):audit.capture('plain_france.whole','source_stage',0,'embedding',[1,2],0)
                    else:
                        values,_=audit.capture('plain_france.whole','source_stage',0,'embedding',[1,2],0)
                        np.testing.assert_array_equal(values,[[1.,2.]])
                finally:owner.close()
            self.assertEqual((root/'raw').read_bytes(),raw)

    def test_exact_budget_minus_one_and_ledger(self):
        fields=('packed_weights','effective_norms','zero_bias','cache','scratch','logits','tokens_and_history','diagnostic_gpu','diagnostic_readback_reserved','metadata','shared_framework_allowance','external_resident')
        model={key:0 for key in fields};model.update(packed_weights=100,effective_norms=16,zero_bias=2*(c.V+7),metadata=64,shared_framework_allowance=64*1024**2)
        model['admitted']=sum(model.values());model['cold_peak']=model['admitted']+1000;c.ledger(model)
        session=dict(model);session.update(cache=1000,scratch=2*(4*(32*640+7)+2*(32*1024+7)+(32*256+7)+3*(32*2048+7))+8*(4*32*544+7),logits=8*(c.V+7),tokens_and_history=4*(32+7+544),external_resident=c.APP)
        session['admitted']=sum(session[key] for key in fields);session['cold_peak']=max(model['cold_peak'],session['admitted']);c.ledger(session,model,False)
        required=session['admitted'];records=[dict(kind='session_plan',diagnostics=False,memory=session),dict(kind='session_budget_minus_one',diagnostics=False,required=required,limit=required-1,error=f'Metal admission requires {required} bytes, budget {required-1}'),dict(kind='session_exact_budget',diagnostics=False,memory=session)]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'events'
            for corrupt in (False,True):
                current=copy.deepcopy(records)
                if corrupt:current[1]['limit']=required
                path.write_text(''.join(json.dumps(x)+'\n' for x in current));a=c.Audit(c.Reader(),{}, {},None,None,c.Events(path),None)
                if corrupt:
                    with self.assertRaises(c.Failure):a.budgets(model,False)
                else:a.budgets(model,False)
        bad=dict(session);bad['admitted']-=1
        with self.assertRaises(c.Failure):c.ledger(bad,model,False)
        bad=dict(session);bad['cold_peak']=c.LIMIT+1
        with self.assertRaises(c.Failure):c.ledger(bad,model,False)

    def test_numeric_hash_lowest_tie_and_strict_json(self):
        values=np.zeros((1,c.V),dtype=np.float32);expected=c.numeric_hash(values);values[0,7]=-0.
        self.assertEqual(c.numeric_hash(values),expected)
        values[0,5]=1;values[0,8]=1;self.assertEqual(c.lowest_id(values),5)
        values[0,0]=np.inf
        with self.assertRaises(c.Failure):c.numeric_hash(values)
        for raw in ['{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}']:
            with self.assertRaises(c.Failure):c.strict_json(raw)

class ConformanceIntegrationControls(unittest.TestCase):
    def test_legacy_metrics_remain_rejected_when_coupled_subset_passes(self):
        a=c.Audit(c.Reader(),{}, {},None,None,None,None)
        main=np.zeros((1,1),dtype=np.float32);native=np.ones_like(main)
        group={'gate':{'atol':.0001,'rtol':.001,'fatal_reference':'main'},'main':{},'wide':{}}
        a.numerical_row(native,group,main,main,'local_drift',{})
        good=a.numerical_row(main,group,main,main,'strict_head',{})
        a.fatal_coupled(good)
        self.assertEqual(a.numerical['failures'],1)
        self.assertEqual(a.coupled['failures'],0)
        self.assertEqual(a.numerical['values'],2)
        self.assertEqual(a.coupled['values'],1)
        bad=a.numerical_row(native,group,main,main,'strict_head',{})
        a.fatal_coupled(bad)
        self.assertEqual(a.coupled['failures'],1)
    def test_policy_pin_and_exhaustive_routes(self):
        self.assertEqual(len(c.local.POLICY_SHA),64)
        self.assertEqual(tuple(c.STAGES),c.local.STAGES)
        self.assertEqual(sum(c.local.route(s)=='coupled'for s in c.STAGES),37)
        self.assertEqual(sum(c.local.route(s)=='local'for s in c.STAGES),363)
        with self.assertRaises(c.local.m.ContractError):c.local.route('unknown')

if __name__=='__main__':unittest.main()
