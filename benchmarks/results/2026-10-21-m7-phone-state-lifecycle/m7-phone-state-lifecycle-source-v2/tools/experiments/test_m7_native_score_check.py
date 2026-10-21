import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parent))
import m7_native_score_check as check


def fixture(backend='cpu'):
    rows=[]
    for name,status,history,evaluated,pending,saved in check.STATES:
        row=dict(name=name,status=status,history=history,evaluated=evaluated,pending=pending,saved=saved)
        if name in check.SCORES:
            target,count=check.SCORES[name];target_logit=2.0 if target==47 else 1.0
            row['score']=dict(target=target,evaluated=count,greedy=47,vocabulary=64,targetLogit=target_logit,
                              greedyLogit=2.0,negativeLogLikelihood=1.6877391136919981+2.0-target_logit)
        rows.append(row)
    return dict(passed=True,backend=backend,checks=rows,closed=True,released=True,physicalDeviceQualified=False,publicationRaces=['interrupt-unload','close'])


class ScoreOutputTests(unittest.TestCase):
    def validate(self,data,backend='cpu'):return check.validate_output(json.dumps(data),backend)
    def rejects(self,change):
        data=fixture();change(data)
        with self.assertRaises((ValueError,KeyError,TypeError)):self.validate(data)
    def test_exact_cpu_and_metal_sequence(self):
        for backend in ('cpu','metal'):self.assertEqual(self.validate(fixture(backend),backend)['backend'],backend)
    def test_missing_reordered_and_boolean_states_reject(self):
        self.rejects(lambda d:d['checks'].pop())
        self.rejects(lambda d:d['checks'].reverse())
        self.rejects(lambda d:d['checks'][0].update(status=True))
        self.rejects(lambda d:d['checks'][11].update(pending=0))
    def test_stale_and_missing_scores_reject(self):
        self.rejects(lambda d:d['checks'][7].update(score=d['checks'][4]['score']))
        self.rejects(lambda d:d['checks'][4].pop('score'))
    def test_score_identity_and_values_reject(self):
        for field,value in [('target',3),('evaluated',5),('greedy',2),('vocabulary',65),('target',True),
                            ('targetLogit',float('nan')),('greedyLogit',float('inf')),
                            ('negativeLogLikelihood',-1),('negativeLogLikelihood',True)]:
            with self.subTest(field=field,value=value):
                self.rejects(lambda d:d['checks'][4]['score'].update({field:value}))
    def test_repeat_recovery_and_loss_identities_reject(self):
        self.rejects(lambda d:d['checks'][5]['score'].update(negativeLogLikelihood=2.0))
        self.rejects(lambda d:d['checks'][8]['score'].update(negativeLogLikelihood=2.0))
        self.rejects(lambda d:d['checks'][6]['score'].update(negativeLogLikelihood=2.0))
        def wrong_independent_loss(data):
            for row in data['checks']:
                if row['name'] in ('target','repeat','recover'):row['score']['negativeLogLikelihood']+=0.1
            data['checks'][6]['score']['negativeLogLikelihood']+=0.1
        self.rejects(wrong_independent_loss)
    def test_failed_backend_or_lifetime_reject(self):
        for field,value in [('passed',False),('backend','cpu-fallback'),('closed',False),('released',False),('physicalDeviceQualified',True),('publicationRaces',[])]:
            with self.subTest(field=field):self.rejects(lambda d:d.update({field:value}))
    def test_legacy_sequence_not_replaceable_by_score_results(self):
        with self.assertRaises((ValueError,KeyError)):check.validate_legacy(json.dumps(fixture()),'cpu')


if __name__=='__main__':unittest.main()
