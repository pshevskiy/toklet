import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from gemma_regression import compare, DRIVERS
import test_request_cost_report as cost_fixture


class GemmaRegressionTests(unittest.TestCase):
    def make(self, path, scale=1, second=None, jitter=None):
        self.sequence = getattr(self,'sequence',0)+1
        collection = self.sequence
        offset = collection*100_000_000_000_000
        path.mkdir()
        cost_fixture.CostReportTests().write_bundle(path)
        result_path = path/'result.json'
        result = json.loads(result_path.read_text())
        result.update(os='same test OS',power="Now drawing from 'AC Power'\nBattery detail",overhead_qualified=False,
                      source_sha256={p:'f'*64 for p in DRIVERS},test_fixture_variant=scale)
        result_path.write_text(json.dumps(result))
        for batch in (1,2):
            multiplier = second if batch == 2 and second is not None else scale
            raw_path = path/f'samples-{batch}.json'
            samples = json.loads(raw_path.read_text())
            for row in samples:
                multiplier = jitter[row['trial']] if jitter else second if batch == 2 and second is not None else scale
                row['total_ms'] *= multiplier
                row['submitted_uptime_ns'] += offset
                row['request_id'] = f'collection-{collection}:'+row['request_id']
                row['ttft_ms'] = row['ttft_ms']*multiplier if row['mode'] == 'disabled' else 210*multiplier+.00001
                row['completed_uptime_ns'] = row['submitted_uptime_ns'] + round(row['total_ms']*1e6)
            raw_path.write_text(json.dumps(samples))
            summary_path = path/f'samples-{batch}.json.summaries.jsonl'
            rows = [json.loads(line) for line in summary_path.read_text().splitlines()]
            rows[0].update(profile='google/gemma-4-E2B-it BF16 managed Mac G2',
                revision='70af34e20bd4b7a91f0de6b22675850c43922a03',
                modelAndCodecManifestSHA256='654dc6dfd66fd893a24f0faa39355b71d5e7ffc3f53c2dac394dfcfd9ed95298')
            for row in rows[1:]:
                trial = int(row['requestID'].split('-')[1])
                multiplier = jitter[trial] if jitter else second if batch == 2 and second is not None else scale
                row['requestID'] = f'collection-{collection}:'+row['requestID']
                for key in ('submittedUptimeNanoseconds','startedUptimeNanoseconds','firstCallbackNanoseconds','completedUptimeNanoseconds'):
                    row[key] += offset
                start = row['submittedUptimeNanoseconds']
                row['firstCallbackNanoseconds'] = start+round(210*multiplier*1e6)
                row['callbackTTFTMilliseconds'] = 210*multiplier
                row['completedUptimeNanoseconds'] = start-20+round(1010*multiplier*1e6)
                row['requestMilliseconds'] = (row['completedUptimeNanoseconds']-start)/1e6
                for key in ('encodingNanoseconds','evaluationNanoseconds','decodingNanoseconds'):
                    row[key] = round(row[key]*multiplier)
                for key in ('prefillSeconds','decodeSeconds','samplingSeconds','totalSeconds','firstTokenSeconds'):
                    row['nativeMetrics'][key] *= multiplier
                row['nativeMetrics']['prefillTokensPerSecond'] /= multiplier
            summary_path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')

    def result(self, left, right):
        return compare(left,right,'1'*40,'2'*40)

    def test_confirmed_speed_is_only_a_signal(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b = Path(folder)/'a',Path(folder)/'b'
            self.make(a);self.make(b,.9)
            report = self.result(a,b)
            self.assertEqual(report['status'],'pass')
            self.assertTrue(report['repeatable_primary_speed_signal'])
            self.assertFalse(report['accepted_optimization'])
            self.assertAlmostEqual(report['batches'][0]['metrics']['total_ms']['change_pct'],-10)

    def test_regression_and_disagreeing_confirmation(self):
        for scale,second,state in ((1.1,None,'regression'),(.9,1,'noisy')):
            with tempfile.TemporaryDirectory() as folder:
                a,b = Path(folder)/'a',Path(folder)/'b'
                self.make(a);self.make(b,scale,second)
                self.assertEqual(self.result(a,b)['status'],state)

    def test_wide_noise_is_not_a_clean_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b = Path(folder)/'a',Path(folder)/'b'
            self.make(a);self.make(b,jitter=[1,1,1]+[.5,1.5]*5)
            report=self.result(a,b)
            self.assertEqual(report['status'],'noisy')
            self.assertAlmostEqual(report['batches'][0]['metrics']['total_ms']['change_pct'],0)

    def test_ttft_regression_blocks_primary_speed_signal(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b = Path(folder)/'a',Path(folder)/'b'
            self.make(a);self.make(b,.9)
            for batch in (1,2):
                p=b/f'samples-{batch}.json';rows=json.loads(p.read_text())
                for row in rows:
                    if row['mode']=='disabled':row['ttft_ms']=300
                p.write_text(json.dumps(rows))
            report=self.result(a,b)
            self.assertTrue(report['repeatable_primary_speed_signal'])
            self.assertEqual(report['status'],'regression')

    def test_self_comparison_and_scope_mismatch_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b = Path(folder)/'a',Path(folder)/'b'
            self.make(a);self.make(b,.99)
            with self.assertRaises(ValueError):self.result(a,a)
            for field,value in (('hardware','other'),('os','other'),('power',"Now drawing from 'Battery Power'")):
                p=b/'result.json';original=p.read_text();result=json.loads(original)
                result[field]=value;p.write_text(json.dumps(result))
                with self.assertRaises(ValueError):self.result(a,b)
                p.write_text(original)

    def test_metadata_formatting_and_relabeling_cannot_reuse_a_collection(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b = Path(folder)/'a',Path(folder)/'b'
            self.make(a);shutil.copytree(a,b)
            p=b/'result.json';result=json.loads(p.read_text());result['irrelevant']='changed'
            p.write_text(json.dumps(result,indent=4))
            with self.assertRaises(ValueError):self.result(a,b)
            # Changing request IDs still cannot turn the same clock/work samples
            # into another independently collected cohort.
            for batch in (1,2):
                p=b/f'samples-{batch}.json';rows=json.loads(p.read_text())
                for row in rows:row['request_id']='relabeled:'+row['request_id']
                p.write_text(json.dumps(rows,indent=2))
                p=b/f'samples-{batch}.json.summaries.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()]
                for row in rows[1:]:row['requestID']='relabeled:'+row['requestID']
                p.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
            with self.assertRaises(ValueError):self.result(a,b)

    def test_malformed_metadata_cli_writes_failure_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b = Path(folder)/'a',Path(folder)/'b'
            self.make(a);self.make(b,.99)
            p=b/'result.json';original=p.read_text()
            for index,(field,value) in enumerate((('power',''),('source_sha256',[]),('source_sha256',None),('hardware',None))):
                result=json.loads(original);result[field]=value;p.write_text(json.dumps(result))
                out=Path(folder)/f'failure-{index}'
                command=[sys.executable,str(Path(__file__).with_name('gemma_regression.py')),str(a),str(b),str(out),
                         '--baseline-commit','1'*40,'--candidate-commit','2'*40]
                completed=subprocess.run(command,capture_output=True,text=True,timeout=20)
                self.assertEqual(completed.returncode,1)
                self.assertEqual(json.loads((out/'failure.json').read_text())['status'],'incomparable')
                self.assertNotIn('Traceback',completed.stderr)
            p.write_text(original)
            raw=b/'samples-1.json';rows=json.loads(raw.read_text())
            rows[0]['submitted_uptime_ns']=10**400;raw.write_text(json.dumps(rows))
            for name in ('overflow','depth'):
                if name=='depth':p.write_text('['*1200+'0'+']'*1200)
                out=Path(folder)/name
                command=[sys.executable,str(Path(__file__).with_name('gemma_regression.py')),str(a),str(b),str(out),
                         '--baseline-commit','1'*40,'--candidate-commit','2'*40]
                completed=subprocess.run(command,capture_output=True,text=True,timeout=20)
                self.assertEqual(completed.returncode,1)
                self.assertEqual(json.loads((out/'failure.json').read_text())['status'],'incomparable')
                self.assertNotIn('Traceback',completed.stderr)

    def test_missing_driver_changed_boundaries_and_revision_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b = Path(folder)/'a',Path(folder)/'b'
            self.make(a);self.make(b,.99)
            p=b/'result.json';original=p.read_text()
            for change in ('missing','changed'):
                result=json.loads(original)
                if change=='missing':result['source_sha256'].pop(DRIVERS[0])
                else:result['source_sha256'][DRIVERS[0]]='e'*64
                p.write_text(json.dumps(result))
                with self.assertRaises(ValueError):self.result(a,b)
            p.write_text(original)
            with self.assertRaises(ValueError):compare(a,b,'HEAD','2'*40)

    def test_incomplete_corrupt_and_different_work_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b = Path(folder)/'a',Path(folder)/'b'
            self.make(a);self.make(b,.99)
            p=b/'samples-1.json';original=p.read_text()
            rows=json.loads(original);rows.pop();p.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):self.result(a,b)
            rows=json.loads(original)
            for row in rows:row['tokens'][0]=1
            p.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):self.result(a,b)
            p.write_text(original)
            p=b/'result.json';p.write_text('{"passed":true,"passed":false}')
            with self.assertRaises(ValueError):self.result(a,b)


if __name__ == '__main__':unittest.main()
