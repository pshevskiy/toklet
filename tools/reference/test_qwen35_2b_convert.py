"""Cold conversion gate, tiny saved-byte and failed-witness controls."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import qwen35_2b_convert as convert
from test_qwen35_2b_layout import tiny_recipe


class Convert(unittest.TestCase):
    def test_cold_runtime_closure_covers_imported_project_helpers_without_numerics(self):
        folder=Path(convert.__file__).resolve().parent;admitted=set(convert.sources())
        self.assertEqual(convert.sources(),convert.lineage.sources())
        for module in list(sys.modules.values()):
            origin=getattr(module,'__file__',None)
            if origin and Path(origin).resolve().parent==folder:
                self.assertIn(Path(origin).resolve(),admitted)
        self.assertFalse(any(name=='numpy' or name.startswith('numpy.') or name=='mlx' or name.startswith('mlx.') for name in sys.modules))

    def test_complete_selected_protocol_and_exact_type_policy(self):
        value=convert.contract('/synthetic/model','1'*64,'2'*64,'3'*64)
        self.assertEqual(convert.validate_contract(value),value)
        for key,changed in [('source_bytes',True),('helper_reserve_bytes',0),
                            ('source_setup_receipt_sha256','0'*64),('other',0)]:
            bad=copy.deepcopy(value);bad[key]=changed
            with self.assertRaises(ValueError):convert.validate_contract(bad)
        bad=copy.deepcopy(value);bad['recipe']['maximum_rows_per_tile']=True
        with self.assertRaises(ValueError):convert.validate_contract(bad)

    def test_tiny_completed_matrix_has_saved_fields_and_both_exact_F32_witnesses(self):
        matrices,small=tiny_recipe();plan=convert.layout.output_layout(matrices,small)
        name=matrices[0]['module']+'.weight';source_raw=struct.pack('<H',0x3f80)*192
        span=dict(name=name,shape=[3,64],start=0,end=len(source_raw))
        buffers=dict(weight=bytes(96),scales=struct.pack('<H',0x3f80)*3,biases=bytes(6))
        witness=bytes(3*64*4);events=[]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'source').write_bytes(source_raw)
            with (root/'source').open('rb') as source,(root/'child').open('w+b') as child:
                child.truncate(plan['file_bytes'])
                with patch.object(convert.quant,'quantize',return_value=buffers), \
                        patch.object(convert.quant,'reconstruct_f32',return_value=witness), \
                        patch.object(convert.probe,'upstream_f32',return_value=witness):
                    result=convert.convert_matrix(source.fileno(),child.fileno(),0,span,plan,None,None,events.append)
                self.assertEqual(result['source_sha256'],hashlib.sha256(source_raw).hexdigest())
                self.assertEqual(result['reconstructed_values'],192)
                self.assertEqual(result['independent_F32_sha256'],result['upstream_F32_sha256'])
                self.assertTrue(events[0]['component_readback'])
                for key,field in result['fields'].items():
                    child.seek(field['offset']);self.assertEqual(child.read(field['bytes']),buffers[key])

    def test_failed_FP32_witness_is_journaled_before_rejection_and_never_written(self):
        matrices,small=tiny_recipe();plan=convert.layout.output_layout(matrices,small)
        raw=bytes(384);span=dict(name=matrices[0]['module']+'.weight',shape=[3,64],start=0,end=384)
        buffers=dict(weight=bytes(96),scales=bytes(6),biases=bytes(6));events=[]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'source').write_bytes(raw)
            with (root/'source').open('rb') as source,(root/'child').open('w+b') as child, \
                    patch.object(convert.quant,'quantize',return_value=buffers), \
                    patch.object(convert.quant,'reconstruct_f32',return_value=b'a'), \
                    patch.object(convert.probe,'upstream_f32',return_value=b'b'), \
                    patch.object(convert.layout.writes,'write_matrix_tile') as write:
                with self.assertRaises(ValueError):
                    convert.convert_matrix(source.fileno(),child.fileno(),0,span,plan,None,None,events.append)
                write.assert_not_called()
            self.assertEqual(len(events),1)
            self.assertFalse(events[0]['reconstruction_equal'])
            self.assertFalse(events[0]['component_readback'])

    def test_missing_component_write_rejects_complete_saved_field_hash(self):
        matrices,small=tiny_recipe();plan=convert.layout.output_layout(matrices,small)
        span=dict(name=matrices[0]['module']+'.weight',shape=[3,64],start=0,end=384)
        buffers=dict(weight=bytes(96),scales=struct.pack('<H',0x3f80)*3,biases=bytes(6))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'source').write_bytes(bytes(384))
            with (root/'source').open('rb') as source,(root/'child').open('w+b') as child, \
                    patch.object(convert.quant,'quantize',return_value=buffers), \
                    patch.object(convert.quant,'reconstruct_f32',return_value=bytes(768)), \
                    patch.object(convert.probe,'upstream_f32',return_value=bytes(768)), \
                    patch.object(convert.layout.writes,'write_matrix_tile'):
                child.truncate(plan['file_bytes'])
                with self.assertRaises(ValueError):
                    convert.convert_matrix(source.fileno(),child.fileno(),0,span,plan,None,None,lambda _:None)

    def test_actual_cli_invalid_contract_publishes_failure_and_stays_cold(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);protocol=root/'protocol';protocol.write_bytes(b'{}')
            args=['--environment-root',str(root),'--runtime-manifest',str(root/'runtime'),'--runtime-sha256','1'*64,
                  '--contract',str(protocol),'--contract-sha256',hashlib.sha256(b'{}').hexdigest(),
                  '--source-setup-receipt',str(root/'setup'),'--lineage-certificate',str(root/'lineage'),
                  '--fresh-output',str(root/'model'),'--fresh-evidence',str(root/'evidence')]
            with patch.object(convert.runtime,'admit',side_effect=AssertionError('Numerical runtime must stay cold')):
                self.assertEqual(convert.main(args),1)
            report=json.loads((root/'evidence/manifest.json').read_bytes())
            self.assertFalse(report['passed'])
            self.assertFalse((root/'model/model.safetensors').exists())
            self.assertFalse((root/'model/child-manifest.json').exists())


if __name__=='__main__':unittest.main()
