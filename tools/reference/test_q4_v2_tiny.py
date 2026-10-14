"""Offline independent v2 reconstruction and pre-GPU frozen graph replay."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import q4_v2_tiny as reference
import fp16
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'quantization'))
import q4v2


def make_packed(packed):
    root=reference.ROOT;source_path=root/'tests/fixtures/tiny-v1';source=json.loads((source_path/'manifest.json').read_text())
    manifest=json.loads((root/'benchmarks/results/2026-10-13-q4-artifact.json').read_text())['packed_artifacts']['tiny']['manifest']
    manifest['format']='toklet-q4-model-v2';manifest['quantizer']=q4v2.POLICY;packed.mkdir()
    for entry in manifest['matrices']:
        parts=entry['name'].split('.');key=f"weights.layers.{parts[2]}.{parts[4]}.weight"
        values=fp16.read(source_path,source,key);data=q4v2.pack_matrix(values.tolist(),values.shape[1])
        (packed/entry['file']).write_bytes(data);entry['packed_bytes']=len(data);entry['packed_sha256']=hashlib.sha256(data).hexdigest()
    manifest['packed_matrix_bytes']=sum(e['packed_bytes'] for e in manifest['matrices'])
    (packed/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


class V2Oracle(unittest.TestCase):
    def test_manual_all_nibbles_tail_values(self):
        root=reference.ROOT/'tests/fixtures/q4-v2';manifest=json.loads((root/'manifest.json').read_text())
        self.assertEqual(reference.reconstruct((root/'rows.q4').read_bytes(),2,17).tolist(),manifest['reconstructed_rows'])

    def test_malformed_version_padding_metadata_and_constant_reject(self):
        data=(reference.ROOT/'tests/fixtures/q4-v2/rows.q4').read_bytes()
        for offset in (0,8,12,32,40,48,76,85,88):
            corrupt=bytearray(data);corrupt[offset]^=1
            with self.assertRaises(ValueError):reference.reconstruct(corrupt,2,17)
        for offset,bits in ((64,0x8000),(64,0x7c00),(66,0x7e00)):
            corrupt=bytearray(data);corrupt[offset:offset+2]=bits.to_bytes(2,'little')
            with self.assertRaises(ValueError):reference.reconstruct(corrupt,2,17)
        for corrupt in (data[:63],data[:-1],data+b'\0'):
            with self.assertRaises(ValueError):reference.reconstruct(corrupt,2,17)
        with self.assertRaises(ValueError):reference.reconstruct(data,True,17)

    def test_exact_graph_regeneration_uses_frozen_tiny_inputs_without_models(self):
        root=reference.ROOT;source_path=root/'tests/fixtures/tiny-v1';source=json.loads((source_path/'manifest.json').read_text())
        old=json.loads((root/'benchmarks/results/2026-10-13-q4-artifact.json').read_text())['packed_artifacts']['tiny']['manifest']
        old['format']='toklet-q4-model-v2';old['quantizer']=q4v2.POLICY
        with tempfile.TemporaryDirectory() as tmp:
            packed=Path(tmp)/'packed';packed.mkdir();output=Path(tmp)/'oracle'
            for entry in old['matrices']:
                parts=entry['name'].split('.');key=f"weights.layers.{parts[2]}.{parts[4]}.weight"
                values=fp16.read(source_path,source,key); data=q4v2.pack_matrix(values.tolist(),values.shape[1])
                (packed/entry['file']).write_bytes(data);entry['packed_bytes']=len(data);entry['packed_sha256']=hashlib.sha256(data).hexdigest()
            old['packed_matrix_bytes']=sum(e['packed_bytes'] for e in old['matrices'])
            (packed/'manifest.json').write_text(json.dumps(old,indent=2)+'\n')
            reference.export(packed,output);frozen=root/'tests/fixtures/q4-v2-tiny-prefill-v1'
            self.assertEqual(sorted(p.name for p in output.iterdir()),sorted(p.name for p in frozen.iterdir()))
            for p in output.iterdir():self.assertEqual(p.read_bytes(),(frozen/p.name).read_bytes(),p.name)
            with self.assertRaises(ValueError):reference.export(packed,output)

    def test_duplicate_missing_or_remapped_inventory_rejects_before_forward(self):
        for mutation in ('duplicate','missing','file','excluded','source','config'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                packed=Path(tmp)/'packed';manifest=make_packed(packed);output=Path(tmp)/'oracle'
                if mutation=='duplicate':manifest['matrices'][-1]=dict(manifest['matrices'][0])
                if mutation=='missing':manifest['matrices'].pop()
                if mutation=='file':manifest['matrices'][0]['file']='../bad.q4'
                if mutation=='excluded':manifest['excluded_dtype']='F32'
                if mutation=='source':manifest['sources'][0]['sha256']='0'*64
                if mutation=='config':manifest['config_sha256']='0'*64
                (packed/'manifest.json').write_text(json.dumps(manifest))
                with patch.object(reference.fp16,'forward',side_effect=AssertionError('forward before admission')),self.assertRaises(ValueError):
                    reference.export(packed,output)
                self.assertFalse((output/'manifest.json').exists())

    def test_consumed_bytes_and_post_consumption_replacement_reject(self):
        for phase in ('before_read','after_read','manifest_after_read'):
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as tmp:
                packed=Path(tmp)/'packed';make_packed(packed);output=Path(tmp)/'oracle'
                manifest=json.loads((packed/'manifest.json').read_text());target=packed/manifest['matrices'][0]['file'];original=reference.bounded;changed=False
                def replace(path,limit,expected=None):
                    nonlocal changed
                    if path==target and phase=='before_read' and not changed:
                        data=bytearray(path.read_bytes());data[68]^=1;path.write_bytes(data);changed=True
                    result=original(path,limit,expected)
                    if path==target and phase=='after_read' and not changed:
                        data=bytearray(path.read_bytes());data[68]^=1;path.write_bytes(data);changed=True
                    if path==packed/'manifest.json' and phase=='manifest_after_read' and not changed:
                        with path.open('a') as file:file.write(' ')
                        changed=True
                    return result
                with patch.object(reference,'bounded',replace),self.assertRaises(ValueError):reference.export(packed,output)
                self.assertTrue(changed);self.assertFalse((output/'manifest.json').exists())


if __name__=='__main__':unittest.main()
