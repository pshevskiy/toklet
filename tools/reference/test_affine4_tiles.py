"""Offline NumPy checks; no model download, target import, MLX or GPU required."""
from dataclasses import replace
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import tempfile
from unittest import mock
import unittest

import numpy as np
import affine4_tiles as tiles

ROOT = Path(__file__).resolve().parents[2]


def authored(rows=131, columns=64, scale=0x3b80, bias=0xbc80):
    # Independent byte construction; all 16 nibbles, every group and row vary.
    weight = bytes(((2*i + row) % 16) | (((2*i + row + 1) % 16) << 4)
                   for row in range(rows) for i in range(columns // 2))
    scales = b''.join(struct.pack('<H', scale) for _ in range(rows * columns // 64))
    biases = b''.join(struct.pack('<H', bias) for _ in range(rows * columns // 64))
    raw = b'padding' + weight + scales + biases
    return tiles.MatrixRows(io.BytesIO(raw), len(raw), rows, columns,
                            7, 7 + len(weight), 7 + len(weight) + len(scales))


def scalar(matrix, row):
    result = []
    for column in range(matrix.columns):
        matrix.stream.seek(matrix.weight_offset + row * matrix.columns // 2 + column // 2)
        byte = matrix.stream.read(1)[0]
        code = (byte // (16 ** (column % 2))) % 16
        group = row * (matrix.columns // 64) + column // 64
        matrix.stream.seek(matrix.scale_offset + group * 2)
        scale = struct.unpack('<f', struct.pack('<I', int.from_bytes(matrix.stream.read(2), 'little') << 16))[0]
        matrix.stream.seek(matrix.bias_offset + group * 2)
        bias = struct.unpack('<f', struct.pack('<I', int.from_bytes(matrix.stream.read(2), 'little') << 16))[0]
        result.append(struct.unpack('<f', struct.pack('<f', float(code) * scale + bias))[0])
    return np.array(result, dtype=np.float32)


class AffineTiles(unittest.TestCase):
    def test_frozen_small_and_actual_embedding_width_contracts(self):
        receipt = ROOT/'benchmarks/results/2026-10-25-gemma-packed-reference-v1.json'
        self.assertEqual(hashlib.sha256(receipt.read_bytes()).hexdigest(),
                         'a8349f8cf8ba1f30aac6d8480ff014160b68135b16c2574a8fd9ff7bafea5b19')
        accepted = json.loads(receipt.read_bytes())
        path = ROOT/'tests/fixtures/gemma-packed-v1/math.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), accepted['fixture_manifest_sha256'])
        manifest = json.loads(path.read_bytes())
        raw = (path.parent/'math.bin').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest['binary']['sha256'])
        for case in manifest['cases']:
            arrays = case['arrays']; width = case['width']
            def array(key):
                item = arrays[key]; source = raw[item['offset']:item['offset'] + item['bytes']]
                self.assertEqual(hashlib.sha256(source).hexdigest(), item['sha256'])
                return np.frombuffer(source, dtype={'F32':'<f4','F64':'<f8'}[item['dtype']]).reshape(item['shape'])
            matrix = tiles.MatrixRows(io.BytesIO(raw), len(raw), 3, width,
                                      *[arrays[k]['offset'] for k in ('weight','scales','biases')])
            self.assertEqual(matrix.read_rows(0,3).tobytes(), array('reconstructed_f32').tobytes())
            self.assertEqual(tiles.gather_rows(matrix,manifest['ids']).tobytes(), array('gather_f32').tobytes())
            for dtype, contract in [(np.float32,'projection_ordered_f32'),(np.float64,'projection_ideal')]:
                actual = tiles.project_rows(matrix,array('input_f32'),dtype)
                expected = array(contract)
                self.assertTrue(np.all(np.abs(actual-expected) <= 1e-5 + 1e-4*np.abs(expected)))

    def test_fp64_accumulation_uses_fp32_reconstructed_not_exact_affine_weights(self):
        matrix = authored(1,64,0x3380,0x3f80) # scale2^-24, bias1
        matrix.stream.seek(matrix.weight_offset)
        matrix.stream.write(bytes([1])+bytes(31)) # codes1,0 at the first pair
        x = np.zeros((1,64),dtype=np.float64);x[0,:2]=[1,-1]
        self.assertEqual(tiles.project_rows(matrix,x,np.float64)[0,0],0.)
        exact_affine = (1.+2.**-24)-1.
        self.assertEqual(exact_affine,2.**-24)
        self.assertNotEqual(tiles.project_rows(matrix,x,np.float64)[0,0],exact_affine)

    def test_tiled_projection_tail_repeated_gather_and_independent_scalar_sums(self):
        matrix = authored()
        ids = [130,0,128,130]
        gathered = tiles.gather_rows(matrix,ids)
        for index, row in enumerate(ids):
            self.assertEqual(gathered[index].tobytes(),scalar(matrix,row).tobytes())
        x = np.array([[(-1)**i * (i % 7) / 8 for i in range(64)],
                      [(i % 3 - 1) / 16 for i in range(64)]],dtype=np.float32)
        expected = np.array([[math.fsum(float(a)*float(b) for a,b in zip(row,scalar(matrix,i)))
                              for i in range(matrix.rows)] for row in x])
        for dtype in (np.float32,np.float64):
            np.testing.assert_array_equal(tiles.project_rows(matrix,x,dtype),expected.astype(dtype))
        gathered[0].fill(99)
        self.assertEqual(tiles.gather_rows(matrix,[130])[0].tobytes(),scalar(matrix,130).tobytes())

    def test_signed_zeros_negative_scales_subnormals_and_widest_rows(self):
        for scale,bias in [(0x8000,0x8000),(0,0x8000),(0xbb80,0x3c80),(1,0x8001)]:
            matrix = authored(2,tiles.MAX_COLUMNS,scale,bias)
            actual = matrix.read_rows(0,2)
            for row in range(2):self.assertEqual(actual[row].tobytes(),scalar(matrix,row).tobytes())
        matrix = authored(1,64,0x8000,0x8000)
        self.assertTrue(np.signbit(matrix.read_rows(0,1)).all())

    def test_extents_types_overlap_and_row_allocation_bounds(self):
        matrix = authored()
        for values in [dict(rows=0),dict(rows=True),dict(rows=tiles.MAX_ROWS+1),
                       dict(columns=65),dict(columns=tiles.MAX_COLUMNS+64),dict(file_bytes=2**63),
                       dict(weight_offset=-1),dict(scale_offset=matrix.weight_offset),
                       dict(bias_offset=matrix.file_bytes)]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):replace(matrix,**values)
        for a,b in [(-1,1),(0,129),(0,0),(130,132),(True,2),(0,1.0)]:
            with self.assertRaises(ValueError):matrix.read_rows(a,b)
        for ids in [[],[True],[-1],[131],[0]*33,(0,1)]:
            with self.assertRaises(ValueError):tiles.gather_rows(matrix,ids)
        for value,dtype in [(np.zeros((33,64)),np.float32),(np.zeros((1,63)),np.float32),
                            (np.zeros((1,64),dtype=np.int32),np.float32),
                            (np.full((1,64),np.nan),np.float32),(np.zeros((1,64)),np.float16),
                            (np.zeros((1,64)),'float32')]:
            with self.assertRaises(ValueError):tiles.project_rows(matrix,value,dtype)

    def test_nonfinite_unused_codes_and_projection_overflow_are_rejected(self):
        for scale,bias in [(0x7f80,0),(0,0x7fc1),(0x7f7f,0)]:
            matrix = authored(1,64,scale,bias)
            # Every stored nibble is zero; an unused code15 overflow still rejects.
            matrix.stream.seek(matrix.weight_offset);matrix.stream.write(bytes(32))
            with self.assertRaises(ValueError):matrix.read_rows(0,1)
        matrix = authored(1,64,0,0x3f80)
        x = np.full((1,64),np.finfo(np.float32).max,dtype=np.float32)
        with self.assertRaises(ValueError):tiles.project_rows(matrix,x,np.float32)
        self.assertTrue(np.isfinite(tiles.project_rows(matrix,x,np.float64)).all())

    def test_read_failure_oversize_and_no_descriptor_ownership_transfer(self):
        class Fault(io.BytesIO):
            def __init__(self,raw,mode):super().__init__(raw);self.mode=mode;self.calls=0;self.sizes=[]
            def read(self,n):
                self.calls+=1;self.sizes.append(n)
                if self.calls==2:
                    if self.mode=='error':raise OSError('injected read error')
                    return super().read(n-1 if self.mode=='short' else n)+ (b'x' if self.mode=='long' else b'')
                return super().read(n)
        matrix=authored(129)
        allowance=tiles.planned_buffer_bytes(matrix,1,np.float64)
        class NoRead(io.BytesIO):
            def read(self,n):raise AssertionError('Admission must precede source reads')
        blocked=replace(matrix,stream=NoRead(matrix.stream.getvalue()))
        with self.assertRaises(ValueError):
            tiles.project_rows(blocked,np.zeros((1,64)),np.float64,buffer_limit_bytes=allowance-1)
        largest=tiles.MatrixRows(None,tiles.MAX_FILE_BYTES,tiles.MAX_ROWS,tiles.MAX_COLUMNS,
                                0,tiles.MAX_ROWS*tiles.MAX_COLUMNS//2,
                                tiles.MAX_ROWS*tiles.MAX_COLUMNS//2+tiles.MAX_ROWS*tiles.MAX_COLUMNS//64*2)
        self.assertEqual(tiles.planned_buffer_bytes(largest,32,np.float64),128*1024**2)
        raw=matrix.stream.getvalue()
        for mode in ('short','long','error'):
            stream=Fault(raw,mode);broken=replace(matrix,stream=stream)
            with self.assertRaises((ValueError,OSError)):tiles.project_rows(broken,np.zeros((1,64)),np.float64)
            self.assertFalse(stream.closed)
            self.assertLessEqual(max(stream.sizes),tiles.TILE_ROWS*matrix.columns//2)
        matrix.read_rows(128,129)
        self.assertFalse(matrix.stream.closed)

    def test_real_driver_failure_interrupt_and_endpoint_rejection_retain_raw_arrays(self):
        import gemma4_affine_reference as driver
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            root=Path(directory);model=root/'model';model.mkdir()
            sample=authored(131);raw=sample.stream.getvalue()
            weight=model/'model.safetensors';weight.write_bytes(raw)
            pins={'model.safetensors':hashlib.sha256(raw).hexdigest()}
            for label,error in [('error',OSError('injected projection failure')),
                                ('interrupt',KeyboardInterrupt('injected interruption')),
                                ('endpoint',ValueError('original primary'))]:
                output=root/label
                with mock.patch.multiple(driver,MODEL=model,PINS=pins,PROJECTIONS={'sample':(131,64)}), \
                     mock.patch.object(driver,'matrices',side_effect=lambda stream:{'sample':replace(sample,stream=stream)}), \
                     mock.patch.object(tiles,'project_rows',side_effect=error), \
                     mock.patch.object(driver,'descriptor_sha',return_value='bad' if label=='endpoint' else pins['model.safetensors']):
                    self.assertEqual(driver.run(output),1)
                report=json.loads((output/'observation.json').read_bytes())
                self.assertFalse(report['passed'])
                self.assertIn(str(error),report['primary_error'])
                self.assertEqual(len(report['arrays']),3)
                self.assertTrue(all(a['completed'] for a in report['arrays']))
                captured=(output/'arrays.bin').read_bytes()
                self.assertEqual(report['binary']['sha256'],hashlib.sha256(captured).hexdigest())
                if label=='endpoint':self.assertIn('Consumed descriptor changed',report['endpoint_errors'])
                for item in report['arrays']:
                    body=captured[item['offset']:item['offset']+item['bytes']]
                    self.assertEqual(hashlib.sha256(body).hexdigest(),item['sha256'])



if __name__=='__main__':unittest.main()
