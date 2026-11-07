import json
from pathlib import Path
import tempfile
import unittest
from request_cost_report import report_bundle, parse


class CostReportTests(unittest.TestCase):
    def write_bundle(self, root):
        (root/'result.json').write_text(json.dumps(dict(format='toklet-o1-gemma-summary-v1',passed=True,
            build_id='b'*64,native_archive_sha256='a'*64,hardware='independent fixture')))
        for batch in (1,2):
            samples, summaries = [], []
            for trial in range(13):
                for mode in ('disabled','summary'):
                    identity = f'{batch}-{trial}-{mode}'
                    start = (trial+1)*10000000000
                    total = 1000 if mode=='disabled' else 1010
                    samples.append(dict(trial=trial,warmup=trial<3,mode=mode,request_id=identity,
                        tokens=[818,5279,529,7001,563,5213,50429,84750],history=24,committed=23,pending=1,
                        ttft_ms=200 if mode=='disabled' else 210.00001,total_ms=total,
                        submitted_uptime_ns=start,completed_uptime_ns=start+total*1000000))
                    if mode=='summary':
                        summaries.append(dict(schemaVersion=2,requestID=identity,outcome='stopToken',status=0,
                            submittedUptimeNanoseconds=start+10,startedUptimeNanoseconds=start+20,
                            firstCallbackNanoseconds=start+10+210000000,completedUptimeNanoseconds=start+total*1000000-10,
                            requestMilliseconds=(total*1000000-20)/1e6,callbackTTFTMilliseconds=210,
                            encodingNanoseconds=1000000,evaluationNanoseconds=1000000000,decodingNanoseconds=1000,
                            nativeMetricStatus='completed',nativeMetrics=dict(sessionInstance=2,
                                reusedTokens=0,prefillTokens=15,sampledTokens=9,emittedTokens=8,
                                prefillSeconds=.1,decodeSeconds=.6,samplingSeconds=.05,totalSeconds=1,
                                firstTokenSeconds=.2,prefillTokensPerSecond=150),
                            after=dict(sessionInstance=2,history=24,committed=23,pending=1),
                            promptTokens=15,callbackTokens=8,nativeEmittedTokens=8,capacity=512,chunk=32,requestedOutputTokens=12))
            manifest=dict(kind='request_summary_manifest',schemaVersion=2,buildIdentity='b'*64,
                droppedRecords=0,budgetSkippedRecords=0,exportErrors=0,profile='fixture',revision='c'*40,
                modelAndCodecManifestSHA256='d'*64,clock='uptime',metricScope='worker',unavailable=['render not measured'])
            (root/f'samples-{batch}.json').write_text(json.dumps(samples))
            (root/f'samples-{batch}.json.summaries.jsonl').write_text('\n'.join(json.dumps(r) for r in [manifest,*summaries])+'\n')

    def test_independent_phase_cost_and_missing_observations(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); self.write_bundle(root); report=report_bundle(root)
            self.assertEqual(report['dominant_observed_native_phase'],'decodeSeconds')
            for row in report['cohorts']:
                self.assertAlmostEqual(row['native_remainder_median_seconds'],.25)
                self.assertEqual(row['prefill_tokens_per_second_median'],150)
                self.assertAlmostEqual(row['paired_overhead_median_pct']['total_ms'],1)
                self.assertEqual(row['missing_observations'],['render not measured'])

    def test_oversized_input_is_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'result.json').write_bytes(b'x'*(1024*1024+1))
            with self.assertRaises(ValueError):report_bundle(root)

    def test_duplicate_fields_and_nonfinite_json_rejected(self):
        for value in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(ValueError):parse(value)

    def test_mismatched_work_scope_missing_records_and_phase_overlap_rejected(self):
        mutations=[lambda rows: rows[1].update(tokens=[1]),
                   lambda rows: rows[-1].update(pending=0)]
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);self.write_bundle(root);p=root/'samples-1.json';rows=json.loads(p.read_text());mutate(rows);p.write_text(json.dumps(rows))
                with self.assertRaises(ValueError):report_bundle(root)
        for change in ('identity','missing','phase','owner','rate','dropped'):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);self.write_bundle(root);p=root/'samples-1.json.summaries.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()]
                if change=='identity':rows[0]['buildIdentity']='other'
                if change=='missing':rows.pop()
                if change=='phase':rows[1]['nativeMetrics']['decodeSeconds']=2
                if change=='owner':rows[1]['nativeMetrics']['sessionInstance']=3
                if change=='rate':rows[1]['nativeMetrics']['prefillTokensPerSecond']=1
                if change=='dropped':rows[0]['droppedRecords']=1
                p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
                with self.assertRaises(ValueError):report_bundle(root)

    def test_duration_and_enclosure_corruption_is_rejected(self):
        for change in ('request','ttft','native','wrapper','callback'):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);self.write_bundle(root);p=root/'samples-1.json.summaries.jsonl'
                rows=[json.loads(x) for x in p.read_text().splitlines()]
                if change=='request':rows[1]['requestMilliseconds']=999999
                if change=='ttft':rows[1]['callbackTTFTMilliseconds']=999
                if change=='native':
                    rows[1]['nativeMetrics']['decodeSeconds']=1000
                    rows[1]['nativeMetrics']['totalSeconds']=1001
                if change=='wrapper':rows[1]['evaluationNanoseconds']=1
                if change=='callback':rows[1]['firstCallbackNanoseconds']+=100000000
                p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
                with self.assertRaises(ValueError):report_bundle(root)
        for mode in ('disabled','summary'):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);self.write_bundle(root);p=root/'samples-1.json'
                rows=json.loads(p.read_text());next(r for r in rows if r['mode']==mode)['total_ms']=999999
                p.write_text(json.dumps(rows))
                with self.assertRaises(ValueError):report_bundle(root)


if __name__=='__main__':unittest.main()
