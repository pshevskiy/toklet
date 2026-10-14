"""Independent byte/reconstruction checks for the frozen group-8 candidate."""
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import model
import q4
import q4v2
from test_model import source


def reconstruct(data):
    magic, version, group, rows, columns, stride, payload = struct.unpack('<8sIIQQQQ16x',data[:64])
    assert (magic,version,group,payload)==(b'TOKQ4V2\0',2,8,rows*stride)
    result=[]
    for row in range(rows):
        values=[]
        for col in range(columns):
            start=64+row*stride+col//8*8
            scale,bias=struct.unpack_from('<ee',data,start)
            code=(data[start+4+col%8//2]>>(col%2*4))&15
            values.append(struct.unpack('<f',struct.pack('<f',scale*code+bias))[0])
        result.append(values)
    return result


class GroupEight(unittest.TestCase):
    def test_independent_manual_artifact_bytes_and_values(self):
        directory=Path(__file__).resolve().parents[2]/'tests/fixtures/q4-v2'
        frozen=json.loads((directory/'manifest.json').read_text()); data=(directory/'rows.q4').read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),frozen['packed_sha256'])
        self.assertEqual(q4v2.pack_matrix(frozen['source_rows'],17),data)
        self.assertEqual(reconstruct(data),frozen['reconstructed_rows'])
        self.assertNotEqual(data[:16],q4.pack_matrix(frozen['source_rows'],17)[:16])

    def test_group_and_row_boundaries_have_canonical_padding(self):
        for columns in (1,7,8,9,15,16,17,63,64,65):
            row=[float((i*7)%19)-8 for i in range(columns)]
            data=q4v2.pack_matrix([row],columns); stride,payload=q4v2.layout(1,columns)
            self.assertEqual(len(data),64+payload)
            records=math.ceil(columns/8)
            self.assertEqual(data[64+records*8:64+stride],bytes(stride-records*8))
            for lane in range(columns%8 or 8,8):
                byte=data[64+(records-1)*8+4+lane//2]
                self.assertEqual(byte>>(4*(lane%2))&15,0)
            # Rounded metadata error adds to half a quantization step.
            for start in range(0,columns,8):
                group=row[start:start+8]; scale,bias=struct.unpack_from('<ee',data,64+start)
                for original,actual in zip(group,reconstruct(data)[0][start:start+8]):
                    self.assertLessEqual(abs(actual-original),scale/2+abs(min(group)-bias)+abs(max(group)-(bias+15*scale))+1e-6)

    def test_constants_subnormal_ties_extrema_and_rejections(self):
        for value in (0.,-0.,1.00048828125,65504.,-65504.,2**-24):
            data=q4v2.pack_matrix([[value]*9],9)
            self.assertEqual(reconstruct(data),[[q4.half(value)]*9])
            self.assertEqual(data[64:66],bytes(2)); self.assertEqual(data[68:72],bytes(4))
        self.assertEqual(reconstruct(q4v2.pack_matrix([[0,.5,1.5,2.5,14.5,15]],6)),[[0,0,2,2,14,15]])
        self.assertEqual(reconstruct(q4v2.pack_matrix([[0,2**-25]],2)),[[0,0]])
        self.assertEqual(reconstruct(q4v2.pack_matrix([[-65504,65504]],2)),[[-65504,65536]])
        for values in ([float('nan')],[float('inf')],[70000],[1e40]):
            with self.assertRaises(ValueError): q4v2.pack_matrix([values],len(values))
        for rows,columns in ((0,1),(1,0),(True,1),(1,2**64),(2**64,1)):
            with self.assertRaises(ValueError): q4v2.header(rows,columns)
        with self.assertRaises(ValueError): q4v2.pack_matrix([[1]],2)
        with self.assertRaises(ValueError): q4v2.pack_matrix([[1]],1,limit=127)

    def test_file_identity_exclusive_ownership_and_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'source'; dst=Path(tmp)/'packed'; raw=struct.pack('<2f',1,2);src.write_bytes(raw)
            record=q4v2.pack_file(src,dst,1,2)
            self.assertEqual(record['format'],'toklet-q4-matrix-v2')
            self.assertEqual(record['source_sha256'],hashlib.sha256(raw).hexdigest())
            self.assertEqual(record['packed_sha256'],hashlib.sha256(dst.read_bytes()).hexdigest())
            saved=dst.read_bytes()
            with self.assertRaises(FileExistsError): q4v2.pack_file(src,dst,1,2)
            self.assertEqual(dst.read_bytes(),saved); dst.unlink();src.write_bytes(struct.pack('<f',float('nan')))
            with self.assertRaises(ValueError): q4v2.pack_file(src,dst,1,1)
            self.assertFalse(dst.exists())

    def test_model_format_source_pins_and_version_selected_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'source'; source(src,dtype='BF16',split=True);dst=Path(tmp)/'packed'
            with patch('sys.stderr'): manifest=model.pack_model(src,dst,version='v2')
            self.assertEqual(manifest['format'],'toklet-q4-model-v2'); self.assertEqual(manifest['quantizer'],q4v2.POLICY)
            self.assertEqual(manifest['packed_matrix_bytes'],sum(e['packed_bytes'] for e in manifest['matrices']))
            for entry in manifest['matrices']:
                data=(dst/entry['file']).read_bytes(); self.assertEqual(hashlib.sha256(data).hexdigest(),entry['packed_sha256'])
                self.assertEqual(data[:16],q4v2.header(entry['rows'],entry['columns'])[:16])
            with self.assertRaises(FileExistsError): model.pack_model(src,dst,'v2')
            with self.assertRaises(ValueError): model.pack_model(src,Path(tmp)/'invalid','v3')
            self.assertFalse((Path(tmp)/'invalid').exists())
            original=q4v2.pack_row
            def changed(values):
                with (src/'config.json').open('a') as file: file.write(' ')
                return original(values)
            failed=Path(tmp)/'failed'
            with patch.object(q4v2,'pack_row',changed),patch('sys.stderr'),self.assertRaisesRegex(ValueError,'identities changed'):
                model.pack_model(src,failed,'v2')
            self.assertFalse((failed/'manifest.json').exists())


if __name__=='__main__': unittest.main()
