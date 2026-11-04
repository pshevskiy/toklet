"""Cold boundary capture and unchanged numerical-gate controls."""
from pathlib import Path
import hashlib
import json
import math
import tempfile
import unittest
import check_gemma_hybrid32 as check


class BoundaryControls(unittest.TestCase):
    def test_existing_gate_finite_and_coordinate_policy(self):
        report=check.fractions([0.,0.001],[0.,0.])
        self.assertEqual(report['failed_values'],1);self.assertEqual(report['examples'][0]['index'],1)
        self.assertGreater(report['maximum_gate_ratio'],1)
        self.assertEqual(check.fractions([1.00001],[1.])['failed_values'],0)
        for a,b in [([math.nan],[0.]),([0.],[math.inf]),([],[0.])]:
            with self.assertRaises(ValueError):check.fractions(a,b)

    def test_frozen_corpus_prefix_has32_ids(self):
        self.assertEqual(len(check.contract.inputs()),32)
        self.assertEqual(check.contract.inputs()[:4],[578,9877,151936,589])

    def test_complete_replay_and_rejected_row_state_mutations(self):
        block=b'\0'*1048576;native=block*24;reference=block*12
        specs=[];offset=0
        for contract in ['math_fp64','staged_fp32']:
            size=2097152 if contract=='math_fp64' else 1048576
            for position in check.contract.POSITIONS:
                specs.append(dict(contract=contract,position=position,offset=offset,bytes=size,sha256=hashlib.sha256(reference[offset:offset+size]).hexdigest()));offset+=size
        events=[];number=0
        def event(name,value):events.append('GEMMA_HYBRID_'+name+' '+json.dumps(value))
        for schedule in [[32],[31,1],[1,15,16]]:
            event('REAL_BOUNDARY_SCHEDULE',schedule)
            for name in ['boundary.logits','boundary.reset_logits']:
                for position in check.contract.POSITIONS:
                    row=dict(record=number,name=name,offset=number*1048576,bytes=1048576,before=position,count=1,values=262144)
                    event('ARRAY_ATTEMPT',row);event('ARRAY',dict(row,sha256=hashlib.sha256(block).hexdigest()));number+=1
            event('REAL_BOUNDARY_STATE',dict(schedule=schedule,positions=32,overflow_preserved=True,reset_reuse=True,allocated_bytes=1))
        event('CAPTURE',dict(records=24,bytes=len(native),sha256=hashlib.sha256(native).hexdigest(),whole_e2b_qualified=False))
        event('REAL_BOUNDARY_COMPLETE',dict(state_passed=True,positions=32,selected_rows=check.contract.POSITIONS,numerical_decision='separate reference replay',prior_precision_failures_waived=False,phone_qualified=False,app_qualified=False))
        text='running 1 test\n'+'\n'.join(events)+'\nok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 700 filtered out; finished in 1.0s\n'
        with tempfile.TemporaryDirectory()as name:
            path=Path(name);(path/'rows.bin').write_bytes(reference)
            record=dict(completed=True,tokens=check.contract.inputs(),selected_positions=check.contract.POSITIONS,gate=dict(atol=1e-5,rtol=1e-4),source_pins=check.contract.reference.source.PINS,arrays=specs,rows_sha256=hashlib.sha256(reference).hexdigest())
            (path/'result.json').write_text(json.dumps(record))
            result=check.replay(text.encode(),native,path)
            self.assertTrue(result['state_passed']);self.assertTrue(result['numerical_passed'])
            for changed in [text.replace('"before": 15','"before": 16',1),text.replace('"overflow_preserved": true','"overflow_preserved": false',1),text.replace('"records": 24','"records": 23',1),text.replace('0 failed','1 failed')]:
                with self.assertRaises(ValueError):check.replay(changed.encode(),native,path)
            with self.assertRaises(ValueError):check.replay(text.encode(),native[:-4],path)


if __name__=='__main__':unittest.main()
