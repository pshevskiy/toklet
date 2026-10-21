import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parent))
import m7_native_quality_check as check
ORACLE=[4.389499721216301,5.483734151464388,5.5483128169001885]


def fixture():
    footprint=dict(physicalBytes=1000,lifetimePeakPhysicalBytes=2000,returnedNaturalCount=112)
    samples=[];results=[]
    for backend in ('cpu','metal'):
        summary=dict(targets=3,blocks=1,blockSize=3,policy=check.teacher.POLICY,negativeLogLikelihoodSum=sum(ORACLE),meanNegativeLogLikelihood=sum(ORACLE)/3)
        result=dict(backend=backend,passed=True,cleanupErrors=[],endpointErrors=[],filesBefore=check.EXPECTED_FILES,filesAfter=check.EXPECTED_FILES,
                    tokensBeforeSHA256=check.TOKEN_SHA,tokensAfterSHA256=check.TOKEN_SHA,checkpoints=[dict(phase=p,footprint=copy.deepcopy(footprint)) for p in ['before-artifact-proof','after-artifact-proof','loaded','block-0','unloaded','closed','after-final-artifact-proof']],
                    interruptionCount=0,loadAndSessionSeconds=0.1,samplesCompleted=3,summary=summary,closed=True,released=True,
                    fullQualityQualified=False,physicalDeviceQualified=False,m7Complete=False)
        results.append(result)
        for i,target in enumerate((7,13,3)):
            samples.append(dict(target=dict(targetIndex=i,block=0,position=i+1,committedHistory=i+2,score=dict(target=target,greedy=47,vocabulary=64,evaluated=i+1,targetLogit=1.0,greedyLogit=2.0,negativeLogLikelihood=ORACLE[i])),
                                footprint=copy.deepcopy(footprint),environment=dict(thermalState=0,lowPowerMode=False)))
    failures=[]
    for label in ['overflow-targets','overflow-block','output','model-endpoint','corpus-endpoint']:
        result=copy.deepcopy(results[0]);result['passed']=False
        if label.startswith('overflow'):result.update(error='invalid policy',samplesCompleted=0,summary=None,closed=False,filesBefore={},filesAfter={},tokensBeforeSHA256=None,tokensAfterSHA256=None,loadAndSessionSeconds=0,checkpoints=[dict(phase='closed',footprint=copy.deepcopy(footprint))])
        elif label=='output':result.update(error='output failed',samplesCompleted=1,summary=None)
        else:result.update(endpointErrors=['changed'],tokensAfterSHA256=None,filesAfter={} if label=='model-endpoint' else check.EXPECTED_FILES)
        failures.append(dict(label=label,result=result))
    return dict(samples=samples,results=results,failures=failures,summary=dict(passed=True,cases=check.CASES,physicalDeviceQualified=False))


def encode(data):
    rows=[]
    for index in range(2):
        rows.extend('QUALITY_CHECK_SAMPLE_JSON='+json.dumps(v) for v in data['samples'][index*3:(index+1)*3])
        if index<len(data['results']):rows.append('QUALITY_CHECK_RESULT_JSON='+json.dumps(data['results'][index]))
    rows.extend('QUALITY_CHECK_FAILURE_JSON='+json.dumps(v) for v in data['failures'])
    rows.append('QUALITY_CHECKS_JSON='+json.dumps(data['summary']))
    return '\n'.join(rows)


class QualityOutputChecks(unittest.TestCase):
    def rejects(self,mutate):
        data=fixture();mutate(data)
        with self.assertRaises((ValueError,KeyError,TypeError)):check.validate_output(encode(data),ORACLE)
    def test_complete_native_case(self):self.assertEqual(check.validate_output(encode(fixture()),ORACLE)['sample_count'],6)
    def test_missing_stream_and_failure_records_reject(self):
        self.rejects(lambda d:d['samples'].pop())
        self.rejects(lambda d:d['results'].pop())
        self.rejects(lambda d:d['failures'].pop())
        self.rejects(lambda d:d['summary'].update(cases=[]))
    def test_artifact_corpus_and_cleanup_reject(self):
        for key,value in [('filesBefore',{}),('filesAfter',{}),('tokensAfterSHA256','bad'),('interruptionCount',1),('closed',False),('released',False),('fullQualityQualified',True)]:
            with self.subTest(key=key):self.rejects(lambda d:d['results'][0].update({key:value}))
    def test_peak_and_checkpoint_phases_reject(self):
        self.rejects(lambda d:d['samples'][0]['footprint'].update(lifetimePeakPhysicalBytes=193*1024**2))
        self.rejects(lambda d:d['samples'][0]['footprint'].update(physicalBytes=True))
        self.rejects(lambda d:d['results'][0]['checkpoints'].reverse())
    def test_prefix_identity_and_integer_counts_reject(self):
        self.rejects(lambda d:d['samples'][0]['target'].update(committedHistory=1))
        self.rejects(lambda d:d['samples'][0]['target'].update(targetIndex=False))
        self.rejects(lambda d:d['samples'][0]['target']['score'].update(target=13))
        self.rejects(lambda d:d['results'][0]['summary'].update(blocks=True))
    def test_losses_and_aggregates_reject(self):
        self.rejects(lambda d:d['samples'][0]['target']['score'].update(negativeLogLikelihood=float('nan')))
        self.rejects(lambda d:d['results'][0]['summary'].update(negativeLogLikelihoodSum=0))
        def wrong_independent(d):
            d['samples'][0]['target']['score']['negativeLogLikelihood']+=0.1
            total=sum(s['target']['score']['negativeLogLikelihood'] for s in d['samples'][:3]);d['results'][0]['summary'].update(negativeLogLikelihoodSum=total,meanNegativeLogLikelihood=total/3)
        self.rejects(wrong_independent)
    def test_invalid_oracles_reject(self):
        for oracle in ([],[1],[1,2,float('nan')]):
            with self.assertRaises(ValueError):check.validate_output(encode(fixture()),oracle)
    def test_failure_results_cannot_claim_success_or_full_output(self):
        self.rejects(lambda d:d['failures'][0]['result'].update(closed=True))
        self.rejects(lambda d:d['failures'][2]['result'].update(summary=d['results'][0]['summary']))
        self.rejects(lambda d:d['failures'][3]['result'].update(endpointErrors=[]))
        self.rejects(lambda d:d['failures'][4]['result'].update(passed=True))
        self.rejects(lambda d:d['failures'][0]['result'].update(filesBefore=check.EXPECTED_FILES))
        self.rejects(lambda d:d['failures'][3]['result'].update(tokensAfterSHA256=check.TOKEN_SHA))
    def test_marker_order_and_unexpected_output_reject(self):
        lines=encode(fixture()).splitlines()
        reordered=[lines[3],*lines[:3],*lines[4:]]
        for text in ('\n'.join(reordered),encode(fixture())+'\nQUALITY_CHECK_ERROR_JSON={}',encode(fixture())+'\nerror: incomplete'):
            with self.assertRaises(ValueError):check.validate_output(text,ORACLE)
    def test_load_bracket_requires_finite_number(self):
        for value in (True,-1,float('nan'),float('inf')):
            with self.subTest(value=value):self.rejects(lambda d:d['results'][0].update(loadAndSessionSeconds=value))
    def test_mac_environment_cannot_be_device_or_nonfinite(self):
        self.rejects(lambda d:d['samples'][0]['environment'].update(applicationState=0))
        self.rejects(lambda d:d['samples'][0]['environment'].update(thermalState=4))
        self.rejects(lambda d:d['samples'][0]['environment'].update(lowPowerMode=0))


if __name__=='__main__':unittest.main()
