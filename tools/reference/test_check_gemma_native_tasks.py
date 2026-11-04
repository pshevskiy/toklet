"""Native task records from known stops/caps; no models or GPU."""
import copy
import json
import unittest
import check_gemma_native_tasks as target


def fixture(pilot):
    cases=[dict(id=str(i),category='synthetic',prompt_tokens=[2,7],max_output_tokens=2,stop_ids=[1,106,50])for i in range(100)]
    rows=[]
    for index in range(100):
        if ((index in [0,40,70])!=pilot):continue
        visible=[7]if index%2==0 else[7,8]
        reason=dict(kind='stop_token',token_id=106)if index%2==0 else dict(kind='output_limit')
        report=dict(reason=reason,reused_tokens=0,prefill_tokens=2,sampled_tokens=2,emitted_tokens=len(visible),history_tokens=4,evaluated_tokens=3,pending_tokens=1,
            first_token_seconds=1.,prefill_seconds=1.,decode_seconds=1.,sampling_seconds=0.,total_seconds=2.)
        rows.append(dict(index=index,id=str(index),category='synthetic',prompt_tokens=2,output_reserve=2,visible_token_ids=visible,stop_reason=reason,generation=report,memory=dict(admitted_bytes=1000),full_quality_qualified=False))
    def encode(rows=rows):
        summary=dict(cases=len(rows),pilot=pilot,capacity=567,chunk_capacity=32,reset_checked=True,quality_qualified=False,admitted_bytes_including_diagnostic=1000+128*1024**2)
        lines=[b'test result: ok.']
        lines.extend(b'GEMMA_HYBRID_REAL_TASK_RESULT '+json.dumps(r).encode()for r in rows)
        lines.append(b'GEMMA_HYBRID_REAL_TASK_SUMMARY '+json.dumps(summary).encode())
        return b'\n'.join(lines)
    return cases,rows,encode


class Controls(unittest.TestCase):
    def test_pilot_remaining_stops_caps_and_exact_coverage(self):
        for pilot in [True,False]:
            cases,rows,encode=fixture(pilot)
            result=target.check(encode(),cases,pilot=pilot)
            self.assertTrue(result['passed'])
            self.assertEqual(result['cases'],3 if pilot else 97)

    def test_hidden_stop_truncation_duplicate_and_committed_state_reject(self):
        cases,rows,encode=fixture(True)
        def oversample(r):
            r[0]['visible_token_ids']=[7,8]
            r[0]['generation'].update(sampled_tokens=3,emitted_tokens=2,history_tokens=5,evaluated_tokens=4)
        for change in [oversample,lambda r:r.pop(),lambda r:r.__setitem__(1,r[0]),
            lambda r:r[0]['visible_token_ids'].append(106),
            lambda r:r[0]['generation'].update(pending_tokens=0),
            lambda r:r[0].update(output_reserve=1),
            lambda r:r[0]['generation'].update(total_seconds=float('nan'))]:
            bad=copy.deepcopy(rows);change(bad)
            with self.assertRaises(ValueError):target.check(encode(bad),cases,pilot=True)


if __name__=='__main__':unittest.main()
