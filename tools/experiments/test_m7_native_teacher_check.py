import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parent))
import m7_native_teacher_check as check

ORACLE=[4.389499721216301,5.483734151464388,5.5483128169001885]

def fixture():
    rows=[]
    for i,target in enumerate((7,13,3)):
        rows.append(dict(targetIndex=i,block=0,position=i+1,committedHistory=i+2,
                         score=dict(target=target,greedy=47,vocabulary=64,evaluated=i+1,
                                    targetLogit=1.0,greedyLogit=2.0,negativeLogLikelihood=ORACLE[i])))
    total=sum(ORACLE)
    return dict(passed=True,backend='cpu',simulated=check.SIMULATED,rows=rows,
                summary=dict(targets=3,blocks=1,blockSize=3,policy=check.POLICY,negativeLogLikelihoodSum=total,meanNegativeLogLikelihood=total/3),
                closed=True,released=True,physicalDeviceQualified=False)


class TeacherOutputChecks(unittest.TestCase):
    def validate(self,data):return check.validate_output(json.dumps(data),'cpu',ORACLE)
    def rejects(self,change):
        data=fixture();change(data)
        with self.assertRaises((ValueError,KeyError,TypeError)):self.validate(data)
    def test_complete_tiny_output(self):self.assertTrue(self.validate(fixture())['passed'])
    def test_policy_and_coverage_reject(self):
        self.rejects(lambda d:d['rows'].pop())
        self.rejects(lambda d:d['rows'].reverse())
        for key,value in [('targets',4),('targets',True),('blocks',2),('blockSize',1024),('policy','changed')]:
            with self.subTest(key=key):self.rejects(lambda d:d['summary'].update({key:value}))
    def test_row_and_committed_prefix_reject(self):
        for key,value in [('targetIndex',1),('block',1),('position',2),('committedHistory',1),('targetIndex',False)]:
            with self.subTest(key=key):self.rejects(lambda d:d['rows'][0].update({key:value}))
        for key,value in [('target',13),('evaluated',2),('vocabulary',65),('greedy',64),('targetLogit',float('nan')),('negativeLogLikelihood',True),('negativeLogLikelihood',-1)]:
            with self.subTest(key=key):self.rejects(lambda d:d['rows'][0]['score'].update({key:value}))
    def test_independent_losses_and_aggregation_reject(self):
        self.rejects(lambda d:d['summary'].update(negativeLogLikelihoodSum=0))
        self.rejects(lambda d:d['summary'].update(meanNegativeLogLikelihood=float('inf')))
        def consistent_wrong_loss(d):
            d['rows'][0]['score']['negativeLogLikelihood']+=0.1
            total=sum(r['score']['negativeLogLikelihood'] for r in d['rows']);d['summary'].update(negativeLogLikelihoodSum=total,meanNegativeLogLikelihood=total/3)
        self.rejects(consistent_wrong_loss)
    def test_lifetime_and_simulation_evidence_reject(self):
        for key,value in [('passed',False),('backend','fallback'),('closed',False),('released',False),('physicalDeviceQualified',True),('simulated',[])]:
            with self.subTest(key=key):self.rejects(lambda d:d.update({key:value}))
    def test_complete_independent_oracle_required(self):
        for oracle in ([],[1],[1,2,float('nan')]):
            with self.assertRaises(ValueError):check.validate_output(json.dumps(fixture()),'cpu',oracle)
    def test_independent_fixture_hash_precedes_reduction(self):
        with self.assertRaises(ValueError):check.independent_losses(bytes(1024))
        with self.assertRaises(ValueError):check.independent_losses(bytes(1025))


if __name__=='__main__':unittest.main()
