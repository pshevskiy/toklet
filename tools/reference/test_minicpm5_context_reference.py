"""Portable exact graph identity and fixed full-prefix/limit planning controls."""
import copy
import json
from pathlib import Path
import sys
import unittest

import minicpm5_context_reference as reference


def fixture():
    root=Path(__file__).resolve().parents[2]
    quality=json.loads((root/'benchmarks/minicpm5-quality-v1/manifest.json').read_bytes())
    tokens=(root/'benchmarks/minicpm5-quality-v1/held_out_loss.u32le').read_bytes()
    fields=dict(input_norm=2048,q_projection=2048,k_projection=256,v_projection=256,q_rotary=2048,
        k_rotary=256,attention_context=2048,attention_projection=2048,attention_residual=2048,
        post_attention_norm=2048,gate_projection=6144,up_projection=6144,gate_silu=6144,
        gated=6144,down_projection=2048,output=2048)
    columns=dict(embedding=2048,final_norm=2048,logits=130560)
    columns.update({f'layers.{layer}.{name}':n for layer in [0,1,41] for name,n in fields.items()})
    gates={name:dict(atol=1e-4,rtol=1e-3) for name in columns}
    old=dict(gates=gates)
    contract=dict(format='toklet-minicpm5-context1025-contract-v1',prospective_before_deeper_cpu_native_outcomes=True,
        cases=[dict(name=f'prefix{n}',tokens=n,position_offset=0) for n in [128,512,1025]],
        gates=copy.deepcopy(gates),columns=columns,model=dict(repository=reference.model.REPOSITORY,
        revision=reference.model.REVISION,sha256=reference.model.PINS['model.safetensors']),
        quality_inputs_manifest_sha256=reference.QUALITY_SHA,quality_token_sha256=reference.TOKENS_SHA,
        original_reference_manifest_sha256=reference.OLD_SHA,
        native=dict(chunks=[1,32],normalization='Compensated',attention='Reference',projection='MlxAffine',
                    rms_store='serial baseline',capacity=1025,capture_limit_bytes=4*1024**3),
        memory_ceiling_bytes=6*1024**3,disk_floor_bytes=2*1024**3,unchanged_ordinary_capacity=512)
    return contract,old,quality,tokens


class ContextReference(unittest.TestCase):
    def test_import_does_not_load_numerical_or_gpu_packages(self):
        for name in ['numpy','mlx.core','torch']:self.assertNotIn(name,sys.modules)

    def test_bound_is_the_only_original_arithmetic_ast_difference(self):
        original=Path(reference.model.__file__).read_bytes()
        extended=Path(reference.graph.__file__).read_bytes()
        self.assertTrue(reference.graph_identity(original,extended)['arithmetic_ast_equal_after_restoring_guard'])
        for changed in [extended.replace(b'wide * sigmoid',b'wide + sigmoid'),
                        extended.replace(b'offset + len(tokens)',b'len(tokens)'),
                        extended.replace(b'<= 1025',b'<= 2048')]:
            with self.assertRaises(ValueError):reference.graph_identity(original,changed)

    def test_complete_frozen_quality_prefixes_and_capture_admission(self):
        contract,old,quality,tokens=fixture()
        ids,arrays,size=reference.plan(contract,old,quality,tokens)
        self.assertEqual(len(ids),10001);self.assertEqual(len(arrays),153)
        self.assertEqual(size,2640983040)
        self.assertEqual(arrays['prefix1025.logits']['shape'],[1025,130560])
        self.assertEqual(arrays['prefix1025.layers.41.k_rotary']['shape'],[1025,256])
        self.assertLessEqual(reference.admission()['admitted_bytes'],6*1024**3)

    def test_inherited_gates_cannot_be_recomputed_or_relaxed(self):
        contract,old,quality,tokens=fixture()
        for field,value in [('atol',1.0),('rtol',0.1)]:
            changed=copy.deepcopy(contract);changed['gates']['logits'][field]=value
            with self.assertRaises(ValueError):reference.plan(changed,old,quality,tokens)
        changed=copy.deepcopy(contract);del changed['gates']['layers.41.output']
        with self.assertRaises(ValueError):reference.plan(changed,old,quality,tokens)

    def test_cases_offsets_modes_and_limits_are_not_silently_expanded(self):
        contract,old,quality,tokens=fixture()
        changes=[dict(memory_ceiling_bytes=7*1024**3),dict(disk_floor_bytes=0),dict(unchanged_ordinary_capacity=1025)]
        for change in changes:
            with self.assertRaises(ValueError):reference.plan(dict(contract,**change),old,quality,tokens)
        for field,value in [('capacity',512),('chunks',[32]),('rms_store','parallel')]:
            changed=copy.deepcopy(contract);changed['native'][field]=value
            with self.assertRaises(ValueError):reference.plan(changed,old,quality,tokens)
        changed=copy.deepcopy(contract);changed['cases'][0]['position_offset']=True
        with self.assertRaises(ValueError):reference.plan(changed,old,quality,tokens)

    def test_full_quality_identity_rejects_truncation_or_wrong_ids(self):
        contract,old,quality,tokens=fixture()
        for raw in [tokens[:-4],b'\0'*len(tokens)]:
            with self.assertRaises(ValueError):reference.plan(contract,old,quality,raw)
        with self.assertRaises(ValueError):reference.plan(contract,old,dict(quality,quality_scored=True),tokens)


if __name__=='__main__':unittest.main()
