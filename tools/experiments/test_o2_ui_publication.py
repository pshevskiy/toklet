import json
from pathlib import Path
import tempfile
import unittest

from o2_ui_publication import analyse_ui_batch, TEXT_SHA256
import test_request_cost_report as cost_fixture


class UIPublicationReports(unittest.TestCase):
    def fixture(self, root):
        cost_fixture.CostReportTests().write_bundle(root)
        raw=json.loads((root/'samples-1.json').read_text())
        summaries=[json.loads(line) for line in (root/'samples-1.json.summaries.jsonl').read_text().splitlines()]
        summaries[0].update(uiUpdateErrors=0,profile='google/gemma-4-E2B-it BF16 managed Mac G2',
            revision='70af34e20bd4b7a91f0de6b22675850c43922a03',
            modelAndCodecManifestSHA256='654dc6dfd66fd893a24f0faa39355b71d5e7ffc3f53c2dac394dfcfd9ed95298')
        for sample in raw:
            sample.update(text_sha256=TEXT_SHA256,footprint_current_bytes=100,process_lifetime_peak_bytes=200,thermal_state=0)
        for row in summaries[1:]:
            row['requestedOutputTokens']=32
            raw_row=next(sample for sample in raw if sample['request_id']==row['requestID'])
            start=raw_row['submitted_uptime_ns']+5
            publication=raw_row['completed_uptime_ns']-1
            total=(publication-start)/1e6
            row['ui']=dict(submittedUptimeNanoseconds=start,handledUptimeNanoseconds=publication,
                publicationUptimeNanoseconds=publication,preparationMilliseconds=(row['submittedUptimeNanoseconds']-start)/1e6,
                callbackTTFTMilliseconds=(row['firstCallbackNanoseconds']-start)/1e6,
                terminalHandlingMilliseconds=total,publicationMilliseconds=total,
                workerToHandlingMilliseconds=(publication-row['completedUptimeNanoseconds'])/1e6,
                firstNonemptyPublicationMilliseconds=total,firstAnswerPublicationMilliseconds=total,
                selectedConversation=True,nonemptyText=True,outcome='published')
        return raw,summaries

    def write(self, root, raw, summaries):
        (root/'samples.json').write_text(json.dumps(raw))
        (root/'summaries.jsonl').write_text('\n'.join(json.dumps(row) for row in summaries)+'\n')

    def test_independent_clock_formula_and_cost(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw,rows=self.fixture(root);self.write(root,raw,rows)
            result=analyse_ui_batch(root,'b'*64)
            self.assertEqual(result['measured_per_mode'],10)
            self.assertAlmostEqual(result['comparison']['total_ms']['median_ratio_overhead_pct'],1)
            self.assertAlmostEqual(result['publication_median_ms']['preparationMilliseconds'],.000005)
            self.assertAlmostEqual(result['publication_median_ms']['firstAnswerPublicationMilliseconds'],1009.999994)
            self.assertFalse(result['render_measured'])

    def test_ui_clock_and_availability_corruption_rejected(self):
        for mutate in (
            lambda row:row['ui'].update(publicationUptimeNanoseconds=1),
            lambda row:row['ui'].update(preparationMilliseconds=999),
            lambda row:row['ui'].update(firstAnswerPublicationMilliseconds=1),
            lambda row:row['ui'].update(selectedConversation=False),
            lambda row:row['ui'].update(outcome='workerRejected'),
            lambda row:row.pop('ui'),
        ):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);raw,rows=self.fixture(root);mutate(rows[-1]);self.write(root,raw,rows)
                with self.assertRaises((ValueError,KeyError)):analyse_ui_batch(root,'b'*64)

    def test_lost_or_wrong_identity_is_not_complete(self):
        for field,value in (('uiUpdateErrors',1),('uiUpdateErrors',False),('profile','other')):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);raw,rows=self.fixture(root);rows[0][field]=value;self.write(root,raw,rows)
                with self.assertRaises(ValueError):analyse_ui_batch(root,'b'*64)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw,rows=self.fixture(root);raw[-1]['request_id']='missing';self.write(root,raw,rows)
            with self.assertRaises(ValueError):analyse_ui_batch(root,'b'*64)

    def test_raw_text_work_memory_and_duration_checked(self):
        for mutate in (
            lambda row:row.update(text_sha256='e'*64),
            lambda row:row.update(tokens=[1]),
            lambda row:row.update(process_lifetime_peak_bytes=7*1024**3),
            lambda row:row.update(total_ms=999999),
            lambda row:row.update(warmup=True),
            lambda row:row.update(thermal_state=9),
            lambda row:row.update(pending=True),
        ):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);raw,rows=self.fixture(root);mutate(raw[-1]);self.write(root,raw,rows)
                with self.assertRaises(ValueError):analyse_ui_batch(root,'b'*64)

    def test_bounded_and_nonfinite_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw,rows=self.fixture(root);self.write(root,raw,rows)
            (root/'samples.json').write_bytes(b'x'*(1024*1024+1))
            with self.assertRaises(ValueError):analyse_ui_batch(root,'b'*64)
            self.write(root,raw,rows)
            rows[-1]['ui']['terminalHandlingMilliseconds']=float('nan');self.write(root,raw,rows)
            with self.assertRaises(ValueError):analyse_ui_batch(root,'b'*64)

    def test_joined_summary_state_and_fixed_work_checked(self):
        def prompt_change(row):
            row['promptTokens']=16
            row['nativeMetrics'].update(prefillTokens=16,prefillTokensPerSecond=160)
        def emission_change(row):
            row.update(callbackTokens=9,nativeEmittedTokens=9)
            row['nativeMetrics'].update(emittedTokens=9,sampledTokens=10)
        for mutate in (
            lambda row:row['after'].update(history=100,committed=99,pending=1),
            lambda row:row['after'].update(pending=True),
            prompt_change,
            emission_change,
        ):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);raw,rows=self.fixture(root);mutate(rows[-1]);self.write(root,raw,rows)
                with self.assertRaises(ValueError):analyse_ui_batch(root,'b'*64)


if __name__=='__main__':unittest.main()
