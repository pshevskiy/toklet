"""Explicit optional PyTorch oracle parameter-lifetime checks; no downloads."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
import qwen_streamed as qs


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        embedding=nn.Embedding(7,4,dtype=torch.bfloat16)
        layer=nn.Linear(4,4,dtype=torch.bfloat16)
        norm=nn.Linear(4,4,bias=False,dtype=torch.bfloat16)
        self.model=nn.Module()
        self.model.embed_tokens=embedding
        self.model.layers=nn.ModuleList([layer])
        self.model.norm=norm
        self.model.rotary_emb=SimpleNamespace(inv_freq=torch.ones(2))
        self.lm_head=nn.Linear(4,7,bias=False,dtype=torch.bfloat16)
        self.lm_head.weight=embedding.weight
    def forward(self,ids):
        x=self.model.embed_tokens(ids)
        x=self.model.layers[0](x)
        x=self.model.norm(x)
        return self.lm_head(x)


class Streamed(unittest.TestCase):
    def setUp(self):
        self.native=patch.object(qs.native_usage,"usage",return_value={"lifetime_max_phys_footprint":1234})
        self.native.start();self.addCleanup(self.native.stop)
    def test_tied_parameter_identity_storage_and_fp32_output(self):
        model=Tiny();ids=torch.tensor([[1,3]])
        original=[p.data for p in model.parameters()]
        parameter=model.lm_head.weight
        expected=nn.functional.embedding(ids,original[0].float())
        expected=nn.functional.linear(expected,original[1].float(),original[2].float())
        expected=nn.functional.linear(expected,original[3].float())
        expected=nn.functional.linear(expected,original[0].float())
        scopes=qs.Parameters(model,torch)
        self.addCleanup(scopes.close)
        with torch.inference_mode():actual=model(ids)
        self.assertEqual(actual.dtype,torch.float32)
        self.assertTrue(torch.equal(actual,expected))
        scopes.assert_restored()
        self.assertIs(model.model.embed_tokens.weight,parameter)
        self.assertIs(model.lm_head.weight,parameter)
        self.assertEqual([p.data_ptr() for p in model.parameters()],[p.data_ptr() for p in original])
        self.assertLess(scopes.report()["admitted_bytes"],qs.LIMIT)
    def test_failed_forward_restores_exact_original_data(self):
        model=Tiny();scopes=qs.Parameters(model,torch);self.addCleanup(scopes.close)
        with self.assertRaises(RuntimeError),torch.inference_mode():model.model.layers[0](torch.ones(2,5))
        scopes.assert_restored()
        self.assertEqual(scopes.active,{})
    def test_failed_native_scope_and_nested_scope_restore(self):
        model=Tiny();scopes=qs.Parameters(model,torch);self.addCleanup(scopes.close)
        with patch.object(qs.native_usage,"usage",side_effect=OSError("native failure")):
            with self.assertRaises(OSError),torch.inference_mode():model.model.embed_tokens(torch.tensor([1]))
        scopes.assert_restored()
        scopes.enter(model.model.layers[0],())
        with self.assertRaisesRegex(ValueError,"Nested"):scopes.enter(model.model.norm,())
        scopes.restore();scopes.assert_restored()
    def test_non_bf16_or_non_fp32_rotary_and_budget_rejected(self):
        model=Tiny().float()
        with self.assertRaisesRegex(ValueError,"BF16"):qs.Parameters(model,torch)
        model=Tiny();model.model.rotary_emb.inv_freq=torch.ones(2,dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError,"rotary"):qs.Parameters(model,torch)
        with patch.object(qs,"LIMIT",1),self.assertRaisesRegex(ValueError,"admission"):qs.Parameters(Tiny(),torch)


if __name__=="__main__":unittest.main()
