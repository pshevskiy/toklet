"""Cold inventory controls plus synthetic authenticated raw-reader coverage."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import time
import unittest
from unittest.mock import patch
import torch
import bf16_source
import gemma4_upstream_stream as stream
import gemma4_upstream_weights as target
import test_gemma4_upstream_stream as tiny


class OfficialControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_complete_persistent_inventory_and_wrong_shapes_reject(self):
        config,raw,inventory = target.admit()
        self.assertEqual(inventory['selected'],540)
        self.assertEqual(inventory['excluded'],60)
        selected = json.loads(raw)['selected_text']
        self.assertEqual(sum('layer_scalar' in n for n in selected),35)
        original = stream.upstream.Gemma4ForCausalLM
        def wrong_shape(config):
            model = original(config)
            model.model.layers[0].layer_scalar = torch.empty(2,device='meta')
            return model
        def unexpected_consumer(config):
            model = original(config)
            model.model.layers[15].self_attn.k_proj = torch.nn.Linear(1,1,bias=False,device='meta')
            return model
        for construct in (wrong_shape,unexpected_consumer):
            with patch.object(stream.upstream,'Gemma4ForCausalLM',side_effect=construct):
                with self.assertRaisesRegex(ValueError,'inventory'):
                    target.admit()
        with patch.object(target,'official_name',side_effect=lambda n:'wrong.embedding' if n=='lm_head.weight' else 'model.language_model.'+n[6:]):
            with self.assertRaisesRegex(ValueError,'inventory'):
                target.admit()

    def test_prospective_request_order_matches_stock_tiny_forward(self):
        config,reader,contract = tiny.fixture()
        journal = []
        class Traced:
            shape = reader.shape
            def tensor(self,name):
                journal.append(('tensor',name))
                return reader.tensor(name)
            def gather(self,name,ids):
                journal.append(('gather',name,ids))
                return reader.gather(name,ids)
            def head(self,start,end):
                journal.append(('head',stream.EMBEDDING,start,end))
                return reader.head(start,end)
        planned = target.planned_requests(config,contract['tokens'],4)
        model = stream.Streamed(config,Traced(),4)
        model(torch.tensor([contract['tokens']],dtype=torch.int64))
        self.assertEqual(journal,planned)

    def test_raw_bits_gather_tail_scalars_and_independent_spans(self):
        # Signed zero, positive/negative subnormals, normal values: no conversion.
        bits = [0,32768,1,32769,16256,49024]*260
        raw = struct.pack('<'+'H'*len(bits),*bits)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'weights'; path.write_bytes(raw)
            deadline = time.monotonic()+30
            with bf16_source.Source(path,len(raw),hashlib.sha256(raw).hexdigest(),deadline=deadline) as source:
                reader = target.OfficialWeights.__new__(target.OfficialWeights)
                reader.source,reader.deadline,reader.requests = source,deadline,[]
                reader.entries = {stream.EMBEDDING:dict(shape=[130,3],file_offset=0),
                    stream.PLE:dict(shape=[130,3],file_offset=780),
                    'model.norm.weight':dict(shape=[3],file_offset=1560),
                    'model.layers.0.layer_scalar':dict(shape=[1],file_offset=1566),
                    'model.layers.0.mlp.down_proj.weight':dict(shape=[130,3],file_offset=1572)}
                reader.rows = {n:bf16_source.Rows(source,(e['shape'][0] if len(e['shape'])==2 else 1),
                    e['shape'][-1],e['file_offset'],deadline) for n,e in reader.entries.items()}
                for name in (stream.EMBEDDING,stream.PLE):
                    value = reader.gather(name,[129,0,129,2])
                    offset = reader.entries[name]['file_offset']
                    expected = b''.join(raw[offset+i*6:offset+(i+1)*6] for i in [129,0,129,2])
                    self.assertEqual(stream.tensor_bytes(value),expected)
                    with self.assertRaises(ValueError): reader.tensor(name)
                self.assertEqual(stream.tensor_bytes(reader.head(128,130)),raw[768:780])
                for name in ('model.norm.weight','model.layers.0.layer_scalar','model.layers.0.mlp.down_proj.weight'):
                    offset = reader.entries[name]['file_offset']
                    self.assertEqual(stream.tensor_bytes(reader.tensor(name)),raw[offset:offset+2*torch.tensor(reader.shape(name)).prod().item()])
                reads = source.audit()['reads']
                for ids in ([],[130],[True],[0]*33):
                    with self.assertRaises(ValueError): reader.gather(stream.PLE,ids)
                with self.assertRaises(ValueError): reader.head(0,129)
                self.assertEqual(source.audit()['reads'],reads)
                selected = {target.official_name(n):e for n,e in reader.entries.items()}
                chain = hashlib.sha256(b'toklet-bf16-source-reads-v1\0')
                total = count = 0
                for count,(offset,size) in enumerate(target.request_spans(reader.requests,selected),1):
                    total += size
                    chain.update(struct.pack('<QQQ',count,offset,size))
                    chain.update(hashlib.sha256(raw[offset:offset+size]).digest())
                self.assertEqual((count,total,chain.hexdigest()),(reads,source.audit()['read_bytes'],source.audit()['read_chain_sha256']))
                with patch('bf16_source.os.pread',return_value=b''):
                    with self.assertRaises(bf16_source.SourceChanged): reader.head(129,130)
                self.assertTrue(source.poisoned)


if __name__ == '__main__':
    unittest.main()
