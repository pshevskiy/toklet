import copy
import hashlib
import json
from pathlib import Path
import struct
import unittest
from unittest.mock import patch
import check_qwen35_metal_context as check


class LedgerTests(unittest.TestCase):
    def report(self):
        return json.loads((Path(__file__).parent/'fixtures/qwen35-metal-accounting-v1.json').read_text())

    def test_qualified_source_geometry_reconciles_every_owner(self):
        report=self.report();memory=check.memory(report)
        self.assertEqual(memory['admitted_bytes'],2861361892)
        self.assertEqual(report['load_report']['admitted_load_policy_bytes'],2884435192)
        self.assertEqual(report['execution_owned_bytes'],35196772)
        matrices,load,cpu=check.owners()
        self.assertEqual(len(matrices),249);self.assertEqual(cpu,2436280320)
        self.assertEqual(load['retained_cpu_matrix_bytes'],0)

    def test_resealed_positive_but_underreported_ownership_rejected(self):
        original=self.report()
        paths=[('memory','admitted_bytes'),('memory','scratch_bytes'),('memory','metadata_allowance_bytes'),
               ('load_report','retained_cpu_matrix_bytes'),('load_report','packed_gpu_bytes'),
               ('load_report','admitted_load_policy_bytes'),('layout','projection')]
        for parent,key in paths:
            bad=copy.deepcopy(original);bad[parent][key]=1
            with self.subTest(parent=parent,key=key),self.assertRaises(ValueError):check.memory(bad)
        for key in ['execution_owned_bytes','model_owned_bytes','model_learned_bytes']:
            bad=copy.deepcopy(original);bad[key]=True
            with self.subTest(key=key),self.assertRaises(ValueError):check.memory(bad)

    def test_missing_duplicate_or_extra_owner_fields_rejected(self):
        for parent,key in [('memory','kv_bytes'),('load_report','maximum_upload_source_capacity_bytes')]:
            for change in ['omit','extra']:
                bad=self.report()
                if change=='omit':del bad[parent][key]
                else:bad[parent]['unknown_owner']=1
                with self.subTest(parent=parent,change=change),self.assertRaises(ValueError):check.memory(bad)

    def test_peak_kind_exit_and_process_closure_fail_closed(self):
        receipt=dict(passed=True,exit_code=0,error=None,group_absent=True,changed_inputs=[],kind='capture',lifetime_peak_physical_bytes=1,files_sha256={})
        for key,value in [('passed',False),('exit_code',True),('group_absent',False),('changed_inputs',['changed']),('kind','benchmark'),('lifetime_peak_physical_bytes',True),('lifetime_peak_physical_bytes',None),('lifetime_peak_physical_bytes',check.LIMIT+1)]:
            bad={**receipt,key:value}
            with self.subTest(key=key),patch.object(check,'document',return_value=bad),self.assertRaises(ValueError):check.closed(Path('/unused'))


class ArrayTests(unittest.TestCase):
    def data(self):
        body=bytearray();rows=[];expected={}
        for layer,stage,head,values in [(None,'embedding',None,[.5,-.25]),(3,'attention_k_cache',1,[.5,.25]),(None,'logits',None,[0.,1.,-.5])]:
            raw=struct.pack('<'+'f'*len(values),*values)
            row=dict(token=0,position=0,layer=layer,stage=stage,head=head,count=len(values),offset=len(body),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            rows.append(row);body.extend(raw);expected[check.prior.native_check.key(row)]=len(values)
        return rows,bytes(body),expected

    def test_complete_finite_arrays_are_read_without_missing_outputs(self):
        rows,body,expected=self.data();actual=check.load_arrays(rows,body,[1],0,expected)
        self.assertEqual(set(actual),set(expected));self.assertEqual(sum(v.size for v in actual.values()),7)

    def test_nonzero_context_coordinates_use_the_same_stage_keys(self):
        rows,body,expected=self.data()
        for row in rows:row.update(token=16,position=53)
        actual=check.load_arrays(rows,body,[1]*17,37,expected)
        self.assertEqual(set(actual),set(expected))

    def test_resealed_types_coordinates_and_duplicate_arrays_rejected(self):
        rows,body,expected=self.data();cases=[]
        for index,key,value in [(0,'token',True),(0,'position',True),(1,'layer',True),(1,'head',True),(0,'count',True),(0,'bytes',True),(0,'offset',1),(0,'stage','unknown')]:
            bad=copy.deepcopy(rows);bad[index][key]=value;cases.append((bad,body))
        cases.extend([(rows[:-1],body),(rows+[rows[-1]],body),(rows,body+b'xxxx'),(rows,body[:-1]),([rows[0],rows[0],rows[2]],body)])
        for bad,raw in cases:
            with self.subTest(first=bad[:1]),self.assertRaises(ValueError):check.load_arrays(bad,raw,[1],0,expected)

    def test_resealed_nonfinite_output_rejected(self):
        rows,body,expected=self.data();body=struct.pack('<f',float('nan'))+body[4:]
        rows[0]['sha256']=hashlib.sha256(body[:rows[0]['bytes']]).hexdigest()
        with self.assertRaisesRegex(ValueError,'Finite'):check.load_arrays(rows,body,[1],0,expected)

    def test_duplicate_and_nonfinite_metadata_rejected(self):
        for raw in [b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":1e999}']:
            with self.assertRaises(ValueError):check.prior.decode(raw)


if __name__=='__main__':unittest.main()
