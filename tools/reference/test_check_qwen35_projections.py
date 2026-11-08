import copy
import hashlib
from pathlib import Path
import struct
import unittest
import check_qwen35_projections as check


def sample_rows():
    return [dict(cohort=c,iteration=i,warmup=i<3,
        engine='metal_fp32' if (s+i+c)%2 else 'scalar_cpu',
        elapsed_ns=100 if (s+i+c)%2 else 1000,projections=249)
        for c in range(2) for i in range(13) for s in range(2)]


class EvidenceTests(unittest.TestCase):
    def ledger(self):
        descriptors = [dict(name=n,rows=s[0],columns=s[1]) for n,(kind,s) in sorted(check.producer.graph.specs(check.producer.weights.GEOMETRY).items()) if kind == 'matrix']
        owners = [dict(gpu_weight_bytes=d['rows']*d['columns']*9//16,
            scratch_bytes=4*(17*d['columns']+34*min(d['rows'],16384)+17*d['rows']+22)+240) for d in descriptors]
        report = dict(caller_bytes=155035648,scratch_bytes=311301896,
            packed_gpu_bytes=sum(o['gpu_weight_bytes'] for o in owners))
        report['admitted_bytes'] = 2436280320 + 64*1024**2 + 256*1024**2 + sum(report.values()) + 4*1024**2
        return report,owners,descriptors

    def test_frozen_recipe_admission_reconciles_every_owner(self):
        report,owners,descriptors = self.ledger()
        result = check.memory_ledger(report,owners,descriptors)
        self.assertEqual(result['cpu_load_bytes'],2436280320)
        self.assertEqual(result['caller_bytes'],155035648)
        self.assertEqual(result['scratch_bytes'],311301896)
        self.assertEqual(result['raw_payload_bytes'],3598336)
        self.assertEqual(result['raw_capacity_bounded_residual_bytes'],4*1024**2)

    def test_underreported_or_inconsistent_admission_rejected(self):
        original,owners,descriptors = self.ledger()
        for change in [dict(admitted_bytes=1),dict(caller_bytes=1),dict(admitted_bytes=1,caller_bytes=1),dict(caller_bytes=True),dict(scratch_bytes=1),dict(packed_gpu_bytes=1),dict(admitted_bytes=original['admitted_bytes']-4*1024**2),dict(admitted_bytes=original['admitted_bytes']+8*1024**2)]:
            report={**original,**change}
            with self.subTest(change=change),self.assertRaises(ValueError):check.memory_ledger(report,owners,descriptors)
        for key,value in [('scratch_bytes',1),('gpu_weight_bytes',1),('gpu_weight_bytes',6*1024**3)]:
            damaged=copy.deepcopy(owners);damaged[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):check.memory_ledger(original,damaged,descriptors)

    def test_complete_alternating_samples(self):
        result = check.samples(sample_rows())
        self.assertEqual(len(result),2)
        for cohort in result:
            self.assertAlmostEqual(cohort['improvement_fraction'],0.9)
            self.assertEqual(cohort['engines']['scalar_cpu']['mad_ns'],0)

    def test_missing_duplicate_reordered_or_mistyped_samples_rejected(self):
        original = sample_rows()
        cases = [original[:-1], original[:-1]+[original[0]], list(reversed(original))]
        for key,value in [('cohort',True),('iteration',True),('elapsed_ns',0),('elapsed_ns',True),('warmup',1),('projections',True)]:
            rows = copy.deepcopy(original); rows[0][key] = value; cases.append(rows)
        for rows in cases:
            with self.subTest(rows=rows[:1]),self.assertRaises(ValueError): check.samples(rows)

    def arrays(self,width,native):
        descriptors = [dict(name=str(i),rows=1,columns=64) for i in range(249)]
        body = bytearray(); items=[]
        for batch in [1,17]:
            for desc in descriptors:
                raw = struct.pack('<' + ('f' if width == 4 else 'd')*batch,*([0.5]*batch))
                item = dict(**desc,batch=batch,offset=len(body),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
                if native: item.update(warmed_rust_allocations=0,preflight_transactional=True)
                items.append(item); body.extend(raw)
        return descriptors,items,bytes(body)

    def test_full_arrays_cover_every_output(self):
        for width,native in [(4,True),(8,False)]:
            descriptors,items,body = self.arrays(width,native)
            result = check.arrays(items,body,width,descriptors,native)
            self.assertEqual(len(result),498)
            self.assertEqual(sum(v.size for v in result.values()),249*18)

    def test_partial_duplicate_nonfinite_or_resealed_arrays_rejected(self):
        descriptors,items,body = self.arrays(4,True)
        cases=[]
        for key,value in [('batch',True),('offset',1),('bytes',True),('name','other'),('warmed_rust_allocations',True),('preflight_transactional',1)]:
            rows=copy.deepcopy(items); rows[0][key]=value; cases.append((rows,body))
        cases += [(items[:-1],body),(items,body+b'xxxx'),(items,body[:-1])]
        rows=copy.deepcopy(items); rows[-1]=rows[0]; cases.append((rows,body))
        raw=struct.pack('<f',float('nan'))+body[4:]; rows=copy.deepcopy(items)
        rows[0]['sha256']=hashlib.sha256(raw[:4]).hexdigest(); cases.append((rows,raw))
        for rows,raw in cases:
            with self.subTest(rows=rows[:1]),self.assertRaises(ValueError): check.arrays(rows,raw,4,descriptors,True)

    def test_failed_or_unmeasured_collection_rejected(self):
        receipt=dict(passed=True,exit_code=0,error=None,group_absent=True,changed_inputs=[],lifetime_peak_physical_bytes=100,files_sha256={})
        from unittest.mock import patch
        for key,value in [('passed',False),('exit_code',True),('error','failed'),('group_absent',False),('changed_inputs',['changed']),('lifetime_peak_physical_bytes',None),('lifetime_peak_physical_bytes',True),('lifetime_peak_physical_bytes',6*1024**3+1)]:
            bad=copy.deepcopy(receipt); bad[key]=value
            with self.subTest(key=key),patch.object(check,'document',return_value=bad),self.assertRaises(ValueError): check.closed(Path('/unused'),False)

    def test_duplicate_and_nonfinite_json_rejected(self):
        for raw in [b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":1e999}']:
            with self.assertRaises(ValueError): check.prior.decode(raw)


if __name__ == '__main__': unittest.main()
