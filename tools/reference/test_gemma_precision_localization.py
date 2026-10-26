"""Mechanism controls and explicit corruption replay; no native qualification."""
from pathlib import Path
import copy
import hashlib
import io
import json
import struct
import sys
import tempfile
import unittest
import gemma_precision_localization as producer
import check_gemma_precision_localization as reader

class Mechanisms(unittest.TestCase):
    def test_execution_order_not_shape_dictionary_order(self):
        names=producer.order();self.assertEqual(len(names),713);self.assertEqual(len(set(names)),713)
        sequence=['scaled_embedding','layer.14.post_feedforward_layernorm','layer.15.post_feedforward_layernorm','layer.26.self_attn.q_proj','head','logits']
        self.assertEqual([names.index(n) for n in sequence],sorted(names.index(n) for n in sequence))
    def test_f32_rounding_and_exact_endpoint_gate(self):
        self.assertEqual(producer.f32(1+2**-24),1.0)
        self.assertEqual(struct.pack('<f',producer.f32(-0.0)),bytes.fromhex('00000080'))
        producer.assert_endpoints((1.,1.),1.,1.)
        with self.assertRaises(ValueError):producer.assert_endpoints((1.,1+2**-23),1.,1.)
    def test_original_packing_and_independent_byte_decode(self):
        codes=bytes.fromhex('1032547698badcfe')*96;scales=struct.pack('<H',0x3f80)*24;biases=struct.pack('<H',0xc100)*24
        class Input:
            inventory={'matrices':[dict(name='matrix.weight',rows=1,columns=1536)]}
            def component(self,name,dtype,shape,selections):
                return [scales if name.endswith('scales') else biases if name.endswith('biases') else codes]
        selected=producer.packed_rows(Input(),'matrix.weight',[0])[0]
        wanted=[float(i%16-8) for i in range(1536)];self.assertEqual(selected['weights'],wanted)
        self.assertEqual(reader.weights({k:v for k,v in selected.items() if k!='weights'}),wanted)
    def source(self,raw,expected=None):
        source=object.__new__(producer.Source);source.file=io.BytesIO(bytes(318228)+raw);source.journal=io.StringIO();source.consumed=[]
        complete=raw if expected is None else expected
        source.inventory={'source_components':{'tensor':dict(offset=318228,bytes=len(complete),sha256=hashlib.sha256(complete).hexdigest())}}
        source.header={'tensor':dict(dtype='U32',shape=[1,8],data_offsets=[0,len(complete)])};return source
    def test_same_consumption_complete_source_excerpts(self):
        raw=bytes(range(32));source=self.source(raw)
        self.assertEqual(source.component('tensor','U32',[1,8],[(1,3),(28,4)]),[raw[1:4],raw[28:]])
        attempt=json.loads(source.journal.getvalue());self.assertFalse(attempt['value']['completed']);self.assertTrue(source.consumed[0]['completed'])
    def test_partial_source_capture_precedes_failure(self):
        source=self.source(b'1234',bytes(range(32)))
        with self.assertRaisesRegex(ValueError,'complete component read'):source.component('tensor','U32',[1,8],[(0,8)])
        attempt=json.loads(source.journal.getvalue())['value'];self.assertEqual(attempt['bytes'],4);self.assertEqual(attempt['selections'][0]['raw'],b'1234'.hex());self.assertFalse(attempt['completed'])
    def test_source_hash_tamper_and_regular_file_admission(self):
        source=self.source(bytes(range(32)));source.inventory['source_components']['tensor']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'component hash'):source.component('tensor','U32',[1,8],[(0,8)])
        self.assertTrue(source.journal.getvalue())
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'small';path.write_bytes(b'small');link=Path(directory)/'link';link.symlink_to(path)
            with self.assertRaises(OSError):producer.Source(link,{},io.StringIO())
            owner=producer.Source(path,{},io.StringIO())
            with self.assertRaisesRegex(ValueError,'bounded file identity'):owner.authenticate()
            with self.assertRaises(ValueError):owner.finish()
            self.assertTrue(owner.file.closed)

class CapturedReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        observation,reference,inventory,model=map(Path,sys.argv[-4:]);cls.frozen=reader.Frozen(reference);cls.inventory=reader.safe.document(reader.safe.read(inventory,1024**2));cls.model=model
        cls.report=reader.safe.document(reader.safe.read(observation/'observation.json',4*1024**2));cls.journal=[reader.safe.document(v) for v in reader.safe.read(observation/'capture.jsonl',2*1024**2).splitlines()]
    def replay(self,report=None,journal=None):
        return reader.replay(self.report if report is None else report,self.journal if journal is None else journal,self.frozen,self.inventory,self.model)
    def rejects(self,report=None,journal=None):
        with self.assertRaises((ValueError,KeyError,TypeError,IndexError)):self.replay(report,journal)
    def test_complete_original_counterfactual_replay(self):
        result=self.replay();self.assertEqual(result['states'],713);self.assertEqual(result['failed_values_retained'],57);self.assertEqual(result['localized_cases'],31)
    def test_state_order_counts_and_scope(self):
        for key,value in [('passed',1),('native_engine_executed',True),('original_failures_waived',True)]:
            report=copy.deepcopy(self.report);report[key]=value;self.rejects(report)
        report=copy.deepcopy(self.report);report['analysis']['states'][0],report['analysis']['states'][1]=report['analysis']['states'][1],report['analysis']['states'][0];self.rejects(report)
    def test_missing_extra_and_wrong_counterfactual(self):
        report=copy.deepcopy(self.report);report['analysis']['cases'].pop();self.rejects(report)
        report=copy.deepcopy(self.report);report['analysis']['cases'][0]['index']=False;self.rejects(report)
    def test_rms_numerator_denominator_and_epsilon(self):
        for key,value in [('epsilon',1e-5),('staged_input_denominator',1.)]:
            report=copy.deepcopy(self.report);report['analysis']['cases'][1][key]=value;self.rejects(report)
        report=copy.deepcopy(self.report);report['analysis']['cases'][1]['decomposition']['parts']['denominator_input']=0.;self.rejects(report)
    def test_packed_weights_or_signed_inherited_contribution(self):
        report=copy.deepcopy(self.report);report['analysis']['cases'][3]['packed']['codes']='00'*768;self.rejects(report)
        report=copy.deepcopy(self.report);report['analysis']['cases'][4]['decomposition']['parts']['inherited_input']=0.;self.rejects(report)
    def test_softcap_each_staged_boundary(self):
        for key in ['inherited_head','division_storage','tanh_storage','final_storage']:
            report=copy.deepcopy(self.report);report['analysis']['cases'][4]['softcap']['parts'][key]+=1e-7;self.rejects(report)
    def test_partial_journal_and_source_digest(self):
        self.rejects(journal=self.journal[:-1])
        report=copy.deepcopy(self.report);report['consumed_components'][0]['sha256']='0'*64;self.rejects(report)
    def test_same_input_endpoint_claim_or_unknown_field(self):
        report=copy.deepcopy(self.report);report['analysis']['cases'][4]['replayed']['staged']+=1e-3;self.rejects(report)
        report=copy.deepcopy(self.report);report['analysis']['extra']=True;self.rejects(report)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Mechanisms if sys.argv[1]=='--mechanisms' else CapturedReplay)
    result=unittest.TextTestRunner().run(suite);sys.exit(0 if result.wasSuccessful() else 1)
