"""Cold sealed-capture controls; no model download or GPU execution."""
from pathlib import Path
import copy
import hashlib
import json
import struct
import tempfile
import unittest
import check_gemma_hybrid512 as check


class Boundary512Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body=b'\0'*109051904;cls.events=[];offset=0;number=0
        def event(name,value):cls.events.append((name,value))
        def array(name,before,count):
            nonlocal offset,number
            value=dict(record=number,name=name,offset=offset,bytes=count*1048576,before=before,count=count,values=count*262144)
            event('ARRAY_ATTEMPT',value);event('ARRAY',dict(value,sha256=hashlib.sha256(memoryview(cls.body)[offset:offset+value['bytes']]).hexdigest()));offset+=value['bytes'];number+=1
        for index,partition in enumerate(check.PARTITIONS):
            event('REAL_CONTEXT512_SCHEDULE',partition)
            if index==0:array('context512.initial_logits',0,32)
            for position in check.POSITIONS:array('context512.logits',position,1)
            array('context512.reset_logits',0,32)
            event('REAL_CONTEXT512_STATE',dict(schedule=partition,committed_before_rejection=512,rejected_absolute_position=512,overflow_preserved=True,reset_prefix_tokens=32,reset_all32_rows_exact=True,all15_valid_kv_prefixes_exact=True,allocated_bytes=263761128))
        event('CAPTURE',dict(records=11,bytes=len(cls.body),sha256=hashlib.sha256(cls.body).hexdigest(),whole_e2b_qualified=False))
        event('REAL_CONTEXT512_COMPLETE',dict(state_passed=True,positions=512,selected_rows=check.POSITIONS,prior_precision_failures_waived=False,app_qualified=False,phone_qualified=False))
        assert offset==len(cls.body)and number==11
    def text(self,events=None):
        values=self.events if events is None else events
        return ('running 1 test\n'+'\n'.join('GEMMA_HYBRID_'+name+' '+json.dumps(value)for name,value in values)+'\nok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 900 filtered out; finished in 1.0s\n').encode()
    def references(self,folder,phase,reject=False):
        folder.mkdir();ids,positions=check.oracle.inputs(phase);raw=bytearray(len(positions)*262144*12);specs=[];offset=0
        for kind in ['math_fp64','staged_fp32']:
            size=262144*(8 if kind=='math_fp64'else 4)
            for position in positions:
                if reject and position==511:struct.pack_into('<d'if kind=='math_fp64'else'<f',raw,offset,0.1)
                specs.append(dict(contract=kind,position=position,offset=offset,bytes=size,sha256=hashlib.sha256(raw[offset:offset+size]).hexdigest()));offset+=size
        (folder/'rows.bin').write_bytes(raw)
        (folder/'result.json').write_text(json.dumps(dict(completed=True,phase=phase,tokens=ids,selected_positions=positions,source_pins=check.oracle.source_graph.source.PINS,arrays=specs,rows_sha256=hashlib.sha256(raw).hexdigest())))
    def test_complete_capture_and_separate_numerical_rejection(self):
        with tempfile.TemporaryDirectory()as name:
            root=Path(name);self.references(root/'prefix','compat32');self.references(root/'full','full512')
            result=check.replay(self.text(),self.body,root/'prefix',root/'full')
            self.assertTrue(result['state_passed']);self.assertTrue(result['numerical_passed'])
            self.assertEqual(result['numerical']['math_fp64']['compared_values'],2097152)
            self.references(root/'rejected','full512',reject=True)
            rejected=check.replay(self.text(),self.body,root/'prefix',root/'rejected')
            self.assertTrue(rejected['state_passed']);self.assertFalse(rejected['numerical_passed'])
            self.assertEqual(rejected['numerical']['math_fp64']['failed_values'],1)
            self.assertEqual(rejected['numerical']['staged_fp32']['failed_values'],1)
    def test_structural_and_reset_mismatches_rejected_before_numeric_acceptance(self):
        for kind in ['row','overflow','wrapper','truncated']:
            events=copy.deepcopy(self.events);body=self.body
            if kind=='row':next(v for n,v in events if n=='ARRAY_ATTEMPT'and v['name']=='context512.logits')['before']=1
            elif kind=='overflow':next(v for n,v in events if n=='REAL_CONTEXT512_STATE')['overflow_preserved']=False
            elif kind=='truncated':body=body[:-4]
            text=self.text(events)
            if kind=='wrapper':text=text.replace(b'0 failed',b'1 failed')
            with self.assertRaises(ValueError):check.replay(text,body,Path('/unused'),Path('/unused'))
        changed=bytearray(self.body);reset_offset=36*1048576;struct.pack_into('<f',changed,reset_offset,1.)
        events=copy.deepcopy(self.events)
        for name,value in events:
            if name=='ARRAY'and value['record']==5:value['sha256']=hashlib.sha256(changed[reset_offset:reset_offset+32*1048576]).hexdigest()
            if name=='CAPTURE':value['sha256']=hashlib.sha256(changed).hexdigest()
        with self.assertRaisesRegex(ValueError,'Reset prefix bytes differ'):check.replay(self.text(events),changed,Path('/unused'),Path('/unused'))


if __name__=='__main__':unittest.main()
