"""Portable independent packed witness checks; no MLX/NumPy or downloads."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from types import SimpleNamespace

import gemma_packed as witness


class PackedWitness(unittest.TestCase):
 def test_bfloat_rounding_ties_signs_and_nonfinite_admission(self):
  self.assertEqual(witness.bf16(1+2**-8),1.)
  self.assertEqual(witness.bf16(1+3*2**-8),1+2**-6)
  self.assertEqual(witness.bf16(-1-2**-8),-1.)
  self.assertEqual(witness.bf16(.1),.10009765625)
  for value in (float('nan'),float('inf'),float('-inf'),3.4028234663852886e38):
   with self.assertRaises(ValueError):witness.bf16(value)

 def test_literal_packed_patterns_independently_match_every_code_and_boundary(self):
  for width in witness.WIDTHS:
   packed,scales,biases,_=witness.operands(width)
   self.assertEqual(len(packed),3*width//2)
   self.assertEqual(len(scales),3*width//64)
   self.assertEqual(len(biases),len(scales))
   for row in range(3):
    actual=[]
    for column in range(width):
     byte=packed[(row*width+column)//2]
     actual.append((byte>>(4*(column%2)))&15)
    self.assertEqual(actual,[witness.code(row,c) for c in range(width)])
    self.assertEqual(set(actual),set(range(16)))
   self.assertNotEqual(scales[:width//64],scales[width//64:2*width//64])

 def test_shapes_repeated_lookup_and_distinct_rounding_contracts(self):
  manifest,raw=witness.export()
  self.assertEqual(len(manifest['cases']),4)
  self.assertLess(len(raw),witness.MAX_RAW)
  for case in manifest['cases']:
   width=case['width'];arrays=case['arrays']
   for entry in arrays.values():
    values=witness.load(raw,entry)
    self.assertEqual(len(values),__import__('math').prod(entry['shape']))
   gather=witness.load(raw,arrays['gather_f32']);table=witness.load(raw,arrays['reconstructed_f32'])
   for i,row in enumerate(witness.IDS):self.assertEqual(gather[i*width:(i+1)*width],table[row*width:(row+1)*width])
   self.assertNotEqual(gather,witness.load(raw,arrays['gather_bf16_once']))
   self.assertNotEqual(witness.load(raw,arrays['gather_bf16_once']),witness.load(raw,arrays['gather_bf16_separate']))
   self.assertNotEqual(witness.load(raw,arrays['input_f32']),witness.load(raw,arrays['input_bf16']))

 def test_complete_deterministic_replay_rejects_changed_buffer_ledger_and_duplicate(self):
  manifest,raw=witness.export()
  self.assertEqual((manifest,raw),witness.export())
  with tempfile.TemporaryDirectory() as temporary:
   root=Path(temporary);path=root/'math.json';binary=root/'math.bin';binary.write_bytes(raw)
   def verify(value):
    encoded=witness.encoded(value);path.write_bytes(encoded)
    return witness.verify(path,witness.sha(encoded))
   self.assertEqual(verify(manifest),(manifest,raw))
   for key,value in (('rows',4),('bf16_checkpoint_qualified',True),('ids',[0,1,2,2])):
    edited=copy.deepcopy(manifest);edited[key]=value
    with self.assertRaises(ValueError):verify(edited)
   binary.write_bytes(raw[:-1])
   with self.assertRaises(ValueError):verify(manifest)
   binary.write_bytes(raw);encoded=witness.encoded(manifest).replace(b'"rows": 3,',b'"rows": 3, "rows": 3,')
   path.write_bytes(encoded)
   with self.assertRaises(ValueError):witness.verify(path,witness.sha(encoded))

 def test_width_and_array_identities_are_bounded(self):
  for width in (0,-64,65,True,64.,2**63):
   with self.assertRaises(ValueError):witness.operands(width)
  manifest,raw=witness.export();entry=manifest['cases'][0]['arrays']['weight']
  with self.assertRaises(ValueError):witness.load(bytes([raw[0]^1])+raw[1:],entry)
  with self.assertRaises(ValueError):witness.load(raw[:4],entry)
  for dtype,values in (('BF16',[.1,-.3]),('F32',[1.25,-2.5]),('F64',[1/3]),('U32',[0,2**32-1]),('I32',[-1,2])):
   wire=witness.pack(dtype,values);decoded=witness.unpack(dtype,wire)
   self.assertEqual(witness.pack(dtype,decoded),wire)

 def test_later_failures_and_rejected_current_arrays_preserve_raw_observations(self):
  # Fake module surface exercises actual sequential execute_cases/observe and
  # compare finalization. No fake output is an accepted numerical observation.
  class Array:
   def __init__(self,values,dtype,shape=None,fail=False):
    self.wire=witness.pack(dtype,values);self.dtype=dtype;self.shape=shape or [len(values)];self.fail=fail
   def reshape(self,*shape):
    if shape==(-1,):return self
    size={'F32':4,'BF16':2,'F16':2,'U32':4,'I32':4}[self.dtype]
    self.shape=[len(self.wire)//size//shape[0] if x==-1 else x for x in shape];return self
   def view(self,_):return SimpleNamespace(reshape=lambda _:SimpleNamespace(tolist=lambda:list(self.wire)))
  def evaluate(*values):
   if any(x.fail for x in values):raise ValueError('injected evaluation failure')
  mx=SimpleNamespace(float32='F32',float16='F16',bfloat16='BF16',uint32='U32',int32='I32',uint8='U8',eval=evaluate,array=lambda values,dtype:Array(values,dtype))
  manifest,raw=witness.export();case=manifest['cases'][0];width=case['width'];arrays=case['arrays']
  values={k:witness.load(raw,v) for k,v in arrays.items()}
  with tempfile.TemporaryDirectory() as temporary:
   root=Path(temporary);reference=root/'math.json';reference.write_bytes(witness.encoded(manifest));(root/'math.bin').write_bytes(raw)
   for failure in ('construction','evaluation','shape','dtype','nonfinite','interrupt'):
    class Module:
     def __init__(self,kind):self.kind=kind
     def parameters(self):return {k:getattr(self,k) for k in ('weight','scales','biases')}
     def __call__(self,x):
      if self.kind=='linear':
       if failure=='construction':raise ValueError('injected construction failure')
       if failure=='interrupt':raise KeyboardInterrupt('injected interruption')
       return Array(values['projection_ordered_f32'],'F32',[2,3],failure=='evaluation')
      actual=list(values['gather_f32'])
      if failure=='nonfinite':actual[0]=float('nan')
      return Array(actual,'F16' if failure=='dtype' else 'F32',[width,4] if failure=='shape' else [4,width])
    nn=SimpleNamespace(QuantizedEmbedding=lambda *a,**kw:Module('embedding'),QuantizedLinear=lambda *a,**kw:Module('linear'))
    def run(data,source,observed,cases,comparisons):
     data=dict(data,cases=[data['cases'][0]])
     witness.execute_cases(mx,nn,data,source,observed,cases,comparisons)
    output=root/failure;output.mkdir()
    self.assertEqual(witness.compare(reference,witness.sha(reference.read_bytes()),output,'cpu',{},run=run),1)
    report=json.loads((output/'observation.json').read_bytes());captured=(output/'observed.bin').read_bytes()
    self.assertFalse(report['capture_complete']);self.assertTrue(report['primary_error']);self.assertFalse(report['cleanup_errors'])
    self.assertEqual(witness.sha(captured),report['binary']['sha256']);self.assertTrue(captured)
    attempts=report['cases'][0]['precisions'][0]['attempts']
    if failure in ('construction','evaluation','interrupt'):
     self.assertEqual(len(attempts),2);self.assertTrue(attempts[0]['completed']);self.assertIsNone(attempts[1]['array'])
     self.assertEqual(captured,witness.pack('F32',values['gather_f32']))
    else:
     self.assertEqual(len(attempts),1);self.assertFalse(attempts[0]['completed']);self.assertIsNotNone(attempts[0]['array'])
     self.assertEqual(attempts[0]['array']['sha256'],witness.sha(captured))

 def test_diagnostic_alternatives_have_separate_ledger_labels(self):
  comparisons=[]
  witness.check(comparisons,64,'required',[1.],[1.],True)
  witness.check(comparisons,64,'alternative',[1.],[2.],True,True)
  self.assertTrue(comparisons[0]['passed']);self.assertFalse(comparisons[0]['diagnostic_alternative'])
  self.assertFalse(comparisons[1]['passed']);self.assertTrue(comparisons[1]['diagnostic_alternative'])


if __name__=='__main__':unittest.main()
