"""Portable wide-row/source/whole-graph controls; no model download or GPU."""
import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import time
import unittest
import numpy as np
import affine4_large_rows as large
import affine4_tiles as tiles
import bf16_source
import gemma4_affine_reference as scalar
import qwen35_9b_weights as weights
import qwen35_cpu_graph as graph
import qwen35_real_weights as old

ROOT = Path(__file__).resolve().parents[2]


def cells(rows=131):
    codes = bytes(((row+column*2)%16) | (((row+column*2+1)%16)<<4)
                  for row in range(rows) for column in range(32))
    scales = b''.join(struct.pack('<H', [0x3b80, 0xbb80][row%2]) for row in range(rows))
    biases = b''.join(struct.pack('<H', 0xbc80) for _ in range(rows))
    return codes, scales, biases


def wire(manifest):
    result = {}
    for name, item in manifest['weights'].items():
        shape = item['shape']
        if item['kind'] == 'matrix':
            rows, columns = shape
            for key, field, dtype, dims in [(name, 'codes', 'U32', [rows, columns//8]),
                (name.removesuffix('.weight')+'.scales', 'scales', 'BF16', [rows, columns//64]),
                (name.removesuffix('.weight')+'.biases', 'biases', 'BF16', [rows, columns//64])]:
                p = item[field]
                result[key] = dict(dtype=dtype, shape=dims, data_offsets=[p['offset'], p['offset']+p['bytes']])
        else:
            p = item['data']
            result[name] = dict(dtype='BF16', shape=shape, data_offsets=[p['offset'], p['offset']+p['bytes']])
    return result


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = ROOT/'tests/fixtures/qwen35-untied-graph-v1'
        cls.manifest = json.loads((cls.folder/'manifest.json').read_text())
        cls.raw = (cls.folder/'weights.bin').read_bytes()

    def test_sparse_rows_cross_four_gib_and_match_independent_scalar(self):
        parts = cells(); offsets = [4*1024**3-17, 4*1024**3+10001, 4*1024**3+11003]
        size = offsets[-1]+len(parts[-1])
        with tempfile.TemporaryFile() as f:
            for offset, part in zip(offsets, parts, strict=True): f.seek(offset); f.write(part)
            f.flush()
            matrix = large.LargeMatrixRows(f, size, 131, 64, *offsets)
            with self.assertRaises(ValueError): tiles.MatrixRows(f, size, 131, 64, *offsets)
            for row in [0, 128, 130]:
                self.assertEqual(matrix.read_rows(row,row+1).tobytes(), struct.pack('<64f', *scalar.scalar_row(matrix,row)))
            ids = [130, 0, 128, 130]
            for got, row in zip(tiles.gather_rows(matrix,ids),ids,strict=True):
                self.assertEqual(got.tobytes(),struct.pack('<64f',*scalar.scalar_row(matrix,row)))
            x = np.array([[(j%7-3)/8 for j in range(64)]],dtype=np.float64)
            want = np.array([[math.fsum(a*b for a,b in zip(x[0],scalar.scalar_row(matrix,row),strict=True))
                             for row in range(131)]])
            for dtype in [np.float32,np.float64]:
                np.testing.assert_array_equal(tiles.project_rows(matrix,x,dtype),want.astype(dtype))

    def test_small_row_arithmetic_is_exactly_the_original_reader(self):
        parts = cells();offsets=[1,1+len(parts[0]),1+len(parts[0])+len(parts[1])]
        raw=b'X'+b''.join(parts)
        with tempfile.TemporaryFile() as f:
            f.write(raw);f.flush()
            a=tiles.MatrixRows(f,len(raw),131,64,*offsets);b=large.LargeMatrixRows(f,len(raw),131,64,*offsets)
            for start,end in [(0,128),(128,131)]:self.assertEqual(a.read_rows(start,end).tobytes(),b.read_rows(start,end).tobytes())

    def test_strict_large_bounds_overlap_and_truncation_reject(self):
        parts=cells(1);offsets=[4*1024**3+1,4*1024**3+65,4*1024**3+69];size=offsets[-1]+2
        with tempfile.TemporaryFile() as f:
            for offset,part in zip(offsets,parts,strict=True):f.seek(offset);f.write(part)
            f.flush();matrix=large.LargeMatrixRows(f,size,1,64,*offsets)
            for fields in [dict(file_bytes=True),dict(file_bytes=large.MAX_FILE_BYTES+1),dict(rows=True),
                dict(rows=tiles.MAX_ROWS+1),dict(columns=63),dict(columns=tiles.MAX_COLUMNS+64),
                dict(weight_offset=True),dict(weight_offset=2**65),dict(scale_offset=offsets[0]),
                dict(bias_offset=size),dict(file_bytes=size-1)]:
                with self.assertRaises(ValueError):replace(matrix,**fields)
            for start,end in [(True,1),(0,True),(-1,1),(0,2)]:
                with self.assertRaises(ValueError):matrix.read_rows(start,end)
            f.truncate(size-1)
            with self.assertRaises(ValueError):matrix.read_rows(0,1)

    def test_nonfinite_and_overflowing_affine_coefficients_reject(self):
        parts=cells(1)
        for scale,bias in [(0x7f80,0),(0,0x7fc0),(0x7f7f,0x7f7f)]:
            raw=parts[0]+struct.pack('<HH',scale,bias)
            with tempfile.TemporaryFile() as f:
                f.write(raw);f.flush();matrix=large.LargeMatrixRows(f,len(raw),1,64,0,32,34)
                with self.assertRaises(ValueError):matrix.read_rows(0,1)

    def test_full_two_table_graph_matches_both_frozen_profiles(self):
        reference=(self.folder/'reference.bin').read_bytes();m=self.manifest
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'weights';raw=b'X'+self.raw;path.write_bytes(raw);deadline=time.monotonic()+120
            with bf16_source.Source(path,len(raw),hashlib.sha256(raw).hexdigest(),deadline=deadline) as owner:
                matrices,vectors=weights.bind(owner,wire(m),1,m['geometry'],deadline)
                self.assertIsNot(matrices[weights.EMBED],matrices[weights.HEAD])
                for case in m['cases']:
                    for profile,staged in [('ideal_fp64',False),('staged_fp32',True)]:
                        got=graph.forward(matrices,vectors,case['tokens'],case['position_offset'],staged,
                            geometry=m['geometry'],math_type=weights.math_type(matrices))
                        wanted=case['profiles'][profile]
                        self.assertEqual(len(got),len(wanted))
                        for a,b in zip(got,wanted,strict=True):
                            self.assertEqual((a['token'],a['layer'],a['stage']),(b['token'],b['layer'],b['stage']))
                            expected=np.frombuffer(reference[b['offset']:b['offset']+b['bytes']],dtype='<f4' if staged else '<f8')
                            actual=np.asarray(a['values']);self.assertTrue(np.isfinite(actual).all())
                            self.assertTrue(np.all(np.abs(actual-expected)<=1e-5+1e-4*np.abs(expected)))
                # Tying the same-shape head changes independent output values.
                case=m['cases'][0];right=graph.forward(matrices,vectors,case['tokens'],0,True,geometry=m['geometry'],math_type=weights.math_type(matrices))
                wrong=graph.forward(matrices,vectors,case['tokens'],0,True,geometry=m['geometry'],math_type=old.Math)
                a=next(r['values'] for r in right if r['stage']=='logits');b=next(r['values'] for r in wrong if r['stage']=='logits')
                self.assertTrue(any(abs(x-y)>1e-5+1e-4*abs(x) for x,y in zip(a,b,strict=True)))

    def test_missing_extra_bad_descriptors_and_public_cross_source_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'weights';raw=b'X'+self.raw;path.write_bytes(raw);deadline=time.monotonic()+120
            with bf16_source.Source(path,len(raw),hashlib.sha256(raw).hexdigest(),deadline=deadline) as owner:
                table=wire(self.manifest)
                for bad in ['missing','extra','head_shape','head_dtype','head_offset']:
                    broken=copy.deepcopy(table)
                    if bad=='missing':del broken[weights.HEAD]
                    elif bad=='extra':broken['extra']={}
                    elif bad=='head_shape':broken[weights.HEAD]['shape']=[True,8]
                    elif bad=='head_dtype':broken[weights.HEAD]['dtype']='F32'
                    else:broken[weights.HEAD]['data_offsets']=[0,1]
                    with self.assertRaises(ValueError):weights.bind(owner,broken,1,self.manifest['geometry'],deadline)
                for adapter in [weights,old]:
                    with self.assertRaises(ValueError):adapter.weights(owner,b'',deadline)

    def test_source_mutation_poisoned_before_any_future_large_row_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'weights';raw=b'X'+self.raw;path.write_bytes(raw);deadline=time.monotonic()+120
            with bf16_source.Source(path,len(raw),hashlib.sha256(raw).hexdigest(),deadline=deadline) as owner:
                matrices,_=weights.bind(owner,wire(self.manifest),1,self.manifest['geometry'],deadline)
                path.write_bytes(b'Y'+self.raw)
                with self.assertRaises(bf16_source.SourceChanged):matrices[weights.HEAD][0]
                self.assertTrue(owner.poisoned)

    def test_frozen_source_geometry_inventory_gate_and_old_reader(self):
        specs=graph.specs(weights.GEOMETRY)
        self.assertEqual(sum(kind=='matrix' for kind,shape in specs.values())+1,250)
        self.assertEqual(len(specs)+1,427)
        self.assertEqual(weights.ORIGIN%4,1)
        self.assertEqual(weights.pin.FILE_BYTES,5038160953)
        self.assertEqual(graph.GATE,dict(atol=1e-5,rtol=1e-4))
        wanted=weights.expected_native();self.assertEqual(len(wanted),1051)
        self.assertEqual(sum(wanted.values()),16660224)
        self.assertEqual(hashlib.sha256((ROOT/'tools/reference/affine4_tiles.py').read_bytes()).hexdigest(),
                         '16cf33497da87b857494268b275c4521e0945442548a0e66c9e21a8a30007220')


if __name__=='__main__':unittest.main()
