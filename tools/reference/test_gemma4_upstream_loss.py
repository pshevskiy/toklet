"""Cold scorer alignment/storage controls; no model download or Apple GPU."""
import math
import unittest
import torch
import gemma4_upstream_stream as stream
import gemma4_upstream_loss as target
import test_gemma4_upstream_stream as tiny


class LossControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_stock_rows_nll_alignment_and_position_tail(self):
        config,reader,contract = tiny.fixture()
        reader.config = config
        reference = target.LossReference(reader,positions=34)
        #32+1 scoring rows exercises position batching and the final partial batch.
        ids = torch.tensor([(contract['tokens']*4)[:34]],dtype=torch.int64)
        saved = []
        result = reference.score(ids,sink=lambda start,end,v:saved.append((start,end,v.clone())))
        self.assertEqual([(a,b) for a,b,_ in saved],[(0,32),(32,33)])
        with torch.inference_mode():
            expected = tiny.resident(config,reader)(input_ids=ids,
                attention_mask=torch.ones_like(ids),use_cache=False).logits[:,:33]
        self.assertEqual(stream.tensor_bytes(torch.cat([v for _,_,v in saved],dim=1)),stream.tensor_bytes(expected))
        for position,row in enumerate(expected[0].float().tolist()):
            maximum=max(row)
            nll=maximum+math.log(math.fsum(math.exp(v-maximum)for v in row))-row[ids[0,position+1]]
            self.assertAlmostEqual(nll,result['nlls'][position],places=12)
        with self.assertRaises(ValueError):reference.stock_logits(ids)
        short=ids[:,:33]
        saved.clear();result=reference.score(short,sink=lambda a,b,v:saved.append(v.clone()))
        self.assertEqual(stream.tensor_bytes(saved[0]),stream.tensor_bytes(reference.stock_logits(short)))
        self.assertEqual(result['targets'],32)
        self.assertTrue(all(layer.active is None for layer in reference.streamed.model.model.layers))

    def test_more_than32_unique_ids_use_bounded_gathers(self):
        config,_,_ = tiny.fixture()
        config.vocab_size = config.vocab_size_per_layer_input = 41
        torch.manual_seed(170319)
        with stream.bf16_parameters(),torch.device('cpu'):
            resident=stream.upstream.Gemma4ForCausalLM(config).eval().requires_grad_(False)
        manifest,body=stream.freeze(resident)
        reader=stream.TinyWeights(manifest,body);reader.config=config
        reference=target.LossReference(reader,positions=33)
        ids=torch.arange(33,dtype=torch.int64).reshape(1,33)
        saved=[]
        reference.score(ids,sink=lambda a,b,v:saved.append(v.clone()))
        with torch.inference_mode():
            expected=resident(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[:,:32]
        self.assertEqual(stream.tensor_bytes(saved[0]),stream.tensor_bytes(expected))
        for name,width in [(stream.EMBEDDING,64),(stream.PLE,256)]:
            self.assertEqual([r[2]//(2*width) for r in reader.requests if r[0]=='gather' and r[1]==name],[32,1])

    def test_invalid_inputs_and_memory_before_materialization(self):
        config,reader,_ = tiny.fixture();reader.config=config
        with self.assertRaises(ValueError):target.LossReference(reader,positions=1026)
        self.assertEqual(reader.requests,[])
        original=config.vocab_size;config.vocab_size=10**9
        with self.assertRaisesRegex(ValueError,'memory admission'):
            target.LossReference(reader,positions=33)
        config.vocab_size=original
        self.assertEqual(reader.requests,[])
        reference=target.LossReference(reader,positions=33)
        count=len(reader.requests)
        for ids in [torch.tensor([[1]]),torch.zeros((2,2),dtype=torch.int64),
                    torch.zeros((1,34),dtype=torch.int64),torch.tensor([[1,17]]),
                    torch.tensor([[1.,2.]])]:
            with self.assertRaises(ValueError):reference.score(ids)
        self.assertEqual(len(reader.requests),count)


if __name__=='__main__':
    unittest.main()
