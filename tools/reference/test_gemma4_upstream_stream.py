"""Explicit CPU-only upstream equivalence; no downloaded model or Apple GPU."""
import json
from pathlib import Path
import unittest
import torch
import gemma4_upstream_stream as target

FIXTURE = target.ROOT/'benchmarks/gemma-upstream-tiny-v1'


def fixture():
    manifest = json.loads((FIXTURE/'weights.json').read_text())
    reader = target.TinyWeights(manifest,(FIXTURE/'weights.bf16le').read_bytes())
    config = target.Gemma4TextConfig.from_json_file(FIXTURE/'config.json')
    config._attn_implementation = 'eager'
    return config, reader, json.loads((FIXTURE/'manifest.json').read_text())


def resident(config, reader):
    with target.bf16_parameters(),torch.device('cpu'):
        model = target.upstream.Gemma4ForCausalLM(config)
    state = {}
    for name,item in reader.manifest['tensors'].items():
        raw = reader.body[item['offset']:item['offset']+item['bytes']]
        state[name] = torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).reshape(item['shape'])
    for alias,name in reader.manifest['aliases'].items():
        state[alias] = state[name]
    model.load_state_dict(state,strict=True,assign=True)
    return model.eval().requires_grad_(False)


class StreamControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target.require_runtime()
        torch.set_num_threads(1)

    def test_resident_stream_layers_logits_causality_and_head_tail(self):
        config,reader,contract = fixture()
        reference = resident(config,reader)
        streamed = target.Streamed(config,reader,head_tile=contract['head_tile'])
        for name,source in [(target.EMBEDDING,reference.model.embed_tokens),
                            (target.PLE,reference.model.embed_tokens_per_layer)]:
            scale = streamed.scaled_embeddings[name].embed_scale
            self.assertEqual(scale.dtype,source.embed_scale.dtype)
            self.assertEqual(target.tensor_bytes(scale),target.tensor_bytes(source.embed_scale))
        expected = {}
        hooks = [layer.register_forward_hook(lambda module,args,value,index=index:
                    expected.update({index:value.detach().clone()}))
                 for index,layer in enumerate(reference.model.layers)]
        left = dict(reference.model.rotary_emb.named_buffers())
        right = dict(streamed.model.model.rotary_emb.named_buffers())
        self.assertEqual(set(left),set(right))
        for name in left:
            self.assertEqual(left[name].dtype,torch.float32)
            self.assertEqual(left[name].dtype,right[name].dtype)
            self.assertTrue(torch.equal(left[name].view(torch.uint32),right[name].view(torch.uint32)),name)
        original = None
        for case in ['tokens','changed_future_tokens']:
            ids = torch.tensor([contract[case]],dtype=torch.int64,device='cpu')
            with torch.inference_mode():
                actual_reference = reference(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
                actual_streamed = streamed(ids)
            self.assertEqual(actual_reference.shape,(1,9,17))
            self.assertEqual(target.tensor_bytes(actual_reference),target.tensor_bytes(actual_streamed))
            self.assertTrue(torch.equal(actual_reference.argmax(-1),actual_streamed.argmax(-1)))
            self.assertEqual(set(streamed.observer['outputs']),set(range(4)))
            for index in range(4):
                self.assertEqual(target.tensor_bytes(expected[index]),target.tensor_bytes(streamed.observer['outputs'][index]))
            self.assertEqual(streamed.observer['consumers'],[(2,0),(3,1)])
            self.assertEqual((streamed.observer['live'],streamed.observer['peak']),(0,1))
            self.assertTrue(all(layer.active is None for layer in streamed.model.model.layers))
            self.assertEqual(streamed.model.model.embed_tokens.weight.device.type,'meta')
            self.assertEqual(streamed.model.model.embed_tokens_per_layer.weight.device.type,'meta')
            if original is None:
                original = actual_reference.detach().clone()
            else:
                self.assertEqual(target.tensor_bytes(original[:,:8]),target.tensor_bytes(actual_reference[:,:8]))
        for hook in hooks:
            hook.remove()
        head_requests = [r for r in reader.requests if r[0]=='head']
        self.assertEqual(head_requests[-1],('head',target.EMBEDDING,64*2))
        self.assertFalse(any(r[0]=='tensor' and r[1] in (target.EMBEDDING,target.PLE) for r in reader.requests))
        self.assertEqual(torch.zeros(1,1,17,dtype=torch.bfloat16).argmax(-1).item(),0)

    def test_failed_layer_releases_lease_and_recovers(self):
        config,reader,contract = fixture()
        model = target.Streamed(config,reader)
        ids = torch.tensor([contract['tokens']],dtype=torch.int64)
        reader.fail_tensor = 'model.layers.1.mlp.down_proj.weight'
        retained_error = None
        try:
            model(ids)
        except RuntimeError as error:
            self.assertIn('injected',str(error))
            retained_error = error
        self.assertIsNotNone(retained_error)
        self.assertEqual(model.observer['live'],0)
        self.assertTrue(all(layer.active is None for layer in model.model.model.layers))
        self.assertTrue(any(r[0]=='tensor' and r[1].startswith('model.layers.1.') for r in reader.requests))
        reader.fail_tensor = None
        with torch.inference_mode():
            actual = model(ids)
            expected = resident(config,reader)(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
        self.assertEqual(target.tensor_bytes(actual),target.tensor_bytes(expected))


if __name__=='__main__':
    unittest.main()
