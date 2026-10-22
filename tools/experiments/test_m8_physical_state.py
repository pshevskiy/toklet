"""Authored exact-state mutations; no model, device, GPU or downloads."""
import copy
import json
import unittest

import m8_physical_state as p


def fixture(model='qwen8'):
    c=p.jobs.job(model,'state-512')[1];prompt=c['prompt'];primer=prompt[:16];target=primer[-1]
    pending=[47]*3;append=primer+pending+prompt[:2];changed=primer.copy();changed[0]=1 if changed[0]==0 else 0
    full=[prompt[i%len(prompt)] for i in range(c['capacity'])];rows=[]
    memory=dict(physicalBytes=100,lifetimePeakPhysicalBytes=200,returnedNaturalCount=93)
    def state(tokens,tail=0,ids=None,reason=1):
        return dict(busy=False,closed=False,workerFinished=False,hasModel=True,hasSession=True,status=0,
            history=len(tokens),evaluated=len(tokens)-tail,pending=tail,reason=reason,savedHistory=len(tokens),
            savedPrefix=list(tokens),ids=[] if ids is None else list(ids))
    for i,name in enumerate(p.state.names(True)):
        if name=='loaded' or name.endswith('-reset'):s=state([],reason=0)
        elif name in ['primer','primer-score','fully-cached','cached-score']:s=state(primer)
        elif name=='pending-primer':s=state(primer+pending,1,pending)
        elif name in ['append-pending','append-reference']:s=state(append+[47]*2,1,[47]*2)
        elif name in ['changed-prefix','changed-reference','usable-fresh-reference']:s=state(changed+[47]*3,1,[47]*3)
        elif name=='prefill-cancel-trigger':s=state([],reason=0);s.update(busy=True,savedPrefix=None)
        elif name=='prefill-cancelled':s=state(prompt,len(prompt)-128,reason=3)
        elif name=='prefill-completed':s=state(prompt)
        elif name in ['prefill-resumed','prefill-reference']:s=state(prompt+[47]*3,1,[47]*3)
        elif name=='closed':s=state([],reason=0);s.update(closed=True,workerFinished=True,hasModel=False,hasSession=False,savedPrefix=None)
        else:s=state(full,reason=1 if name in ['capacity-committed','overflow-continue-before-score'] else 0)
        if name in ['overflow-continue','overflow-append','overflow-prompt']:s['status']=1
        score=dict(target=target,greedy=47,vocabulary=c['vocabulary'],evaluated=s['evaluated'],targetLogit=1.,greedyLogit=2.,negativeLogLikelihood=3.) if 'score' in name else None
        rows.append(dict(name=name,state=s,scalarScore=score,callbackCount=len(s['ids']),admittedBytes=0 if name=='closed' else 1000000,
            completedUptimeNanoseconds=i+1,footprint=copy.deepcopy(memory),
            environment=dict(applicationState=0,lowPowerMode=False,thermalState=0,batteryState=2,batteryLevel=1.)))
    result=dict(passed=True,backend='metal-mlx',prefillWitnessRequired=True,prefillWitness=True,cleanupErrors=[],endpointErrors=[],closed=True,released=True,
        physicalDeviceQualified=False,m7Complete=False,filesBefore=copy.deepcopy(c['files']),filesAfter=copy.deepcopy(c['files']),rows=rows,
        trace=dict(events=[],detachedAfterClose=True),checkpoints=[dict(phase=n,footprint=copy.deepcopy(memory)) for n in ['before-artifact-proof','after-artifact-proof','loaded','closed','after-final-artifact-proof']])
    signed={'TokletHarness':'b'*64}
    proofs=[dict(phase=n,simulator=False,files=copy.deepcopy(signed)) for n in ['before','after']]
    return c,signed,result,proofs


def console(result,proofs):
    return '\n'.join(['TOKLET_INSTALLATION_JSON='+json.dumps(proof) for proof in proofs]+['TOKLET_STATE_JSON='+json.dumps(result)])


def row(result,name):return next(r for r in result['rows'] if r['name']==name)


class State(unittest.TestCase):
    def reject(self,mutate):
        c,s,r,proofs=fixture();mutate(c,r,proofs)
        with self.assertRaises((ValueError,KeyError,TypeError)):p.validate_console(console(r,proofs),c,s)

    def test_both_explicit_real_configs_and_complete_physical_witness(self):
        for model in ['qwen','qwen8']:
            c,s,r,proofs=fixture(model);self.assertEqual(p.admit_config(c),model)
            self.assertEqual(p.validate_console(console(r,proofs),c,s),dict(result=r,installed_proofs=proofs))
            self.assertEqual(len(r['rows']),33)

    def test_canonical_config_rejects_semantic_shape_and_type_substitutions(self):
        for field,value in [('version',2.),('mlx',1),('metal',1),('capacity',1153),('chunk',64),('maximum',127),('vocabulary',64),('packedFolder','Packed'),('modelFolder','Other'),('prompt',[1]*512),('extra',1)]:
            c,_,_,_=fixture();c[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):p.admit_config(c)
        with self.assertRaises(ValueError):p.admit_config(None)

    def test_backend_prefill_and_qualification_claims_are_explicit(self):
        for field,value in [('backend','metal'),('prefillWitnessRequired',False),('prefillWitness',False),('physicalDeviceQualified',True),('m7Complete',True)]:
            with self.subTest(field=field):self.reject(lambda c,r,proofs:r.update({field:value}))

    def test_partial_commit_must_be_actual_and_resume_exact(self):
        for evaluated in [0,512]:
            self.reject(lambda c,r,proofs:row(r,'prefill-cancelled')['state'].update(evaluated=evaluated,pending=512-evaluated))
        self.reject(lambda c,r,proofs:row(r,'prefill-cancelled')['state'].update(reason=1))
        self.reject(lambda c,r,proofs:row(r,'prefill-cancel-trigger')['state'].update(busy=False))
        self.reject(lambda c,r,proofs:row(r,'prefill-reference')['state']['ids'].__setitem__(0,46))

    def test_pending_append_and_changed_prefix_match_fresh_reference(self):
        self.reject(lambda c,r,proofs:row(r,'append-reference')['state']['savedPrefix'].__setitem__(16,46))
        self.reject(lambda c,r,proofs:row(r,'changed-reference')['state']['ids'].__setitem__(0,46))
        self.reject(lambda c,r,proofs:row(r,'append-pending')['state'].update(pending=0,evaluated=23))

    def test_each_capacity_rejection_preserves_exact_prefix_and_scalar_score(self):
        for name in ['overflow-continue','overflow-append','overflow-prompt']:
            self.reject(lambda c,r,proofs:row(r,name)['state'].update(evaluated=1151,pending=1))
            self.reject(lambda c,r,proofs:row(r,name)['state']['savedPrefix'].__setitem__(0,0))
            self.reject(lambda c,r,proofs:row(r,name+'-after-score')['scalarScore'].update(negativeLogLikelihood=3.1))
            self.reject(lambda c,r,proofs:row(r,name).update(callbackCount=1))

    def test_cached_score_is_finite_and_identical(self):
        self.reject(lambda c,r,proofs:row(r,'cached-score')['scalarScore'].update(targetLogit=.9))
        self.reject(lambda c,r,proofs:row(r,'cached-score')['scalarScore'].update(targetLogit=float('nan')))

    def test_physical_environment_does_not_inherit_mac_defaults(self):
        for field,value in [('applicationState',None),('applicationState',True),('applicationState',1),('thermalState',True),('thermalState',4),('lowPowerMode',0),('batteryState',True),('batteryState',0),('batteryLevel',True),('batteryLevel',1.1),('batteryLevel',float('nan'))]:
            with self.subTest(field=field,value=value):self.reject(lambda c,r,proofs:row(r,'primer')['environment'].update({field:value}))

    def test_marker_order_bundle_and_platform_proofs_reject(self):
        c,s,r,proofs=fixture();text=console(r,proofs);lines=text.splitlines()
        for bad in [console(r,proofs[:1]),text+'\n'+lines[-1],text+'\nTOKLET_OTHER_JSON={}', '\n'.join([lines[-1],*lines[:-1]])]:
            with self.assertRaises(ValueError):p.validate_console(bad,c,s)
        for field,value in [('phase','other'),('simulator',True),('files',{})]:
            with self.subTest(field=field):self.reject(lambda c,r,proofs:proofs[0].update({field:value}))

    def test_memory_clock_state_types_and_close_are_bounded(self):
        self.reject(lambda c,r,proofs:row(r,'primer')['footprint'].update(lifetimePeakPhysicalBytes=c['processBudget']+1))
        self.reject(lambda c,r,proofs:row(r,'primer').update(completedUptimeNanoseconds=1))
        self.reject(lambda c,r,proofs:row(r,'capacity-committed')['state'].update(history=True))
        self.reject(lambda c,r,proofs:row(r,'closed')['state'].update(status=1))
        self.reject(lambda c,r,proofs:r.update(released=False))
        self.reject(lambda c,r,proofs:r.update(filesAfter={}))
        self.reject(lambda c,r,proofs:r['rows'].pop())


if __name__=='__main__':unittest.main()
