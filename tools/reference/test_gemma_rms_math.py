"""Portable independent Decimal replay and boundary tests; no engine/GPU imports."""
from decimal import Decimal,localcontext
from pathlib import Path
import copy,importlib.util,json,math,struct,tempfile,unittest
P=Path(__file__).with_name('gemma_rms_math.py');spec=importlib.util.spec_from_file_location('gemma_rms_math',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Reference(unittest.TestCase):
 def test_decimal80_replays_every_math_and_rounded_value(self):
  compared=0
  with localcontext() as c:
   c.prec=80;eps=Decimal.from_float(m.EPSILON)
   for case in m.cases():
    width=case['width'];weights=case['weights']
    for row in range(case['rows']):
     values=case['input'][row*width:(row+1)*width];squares=sum((Decimal.from_float(v)**2 for v in values),Decimal(0));denominator=(squares/width+eps).sqrt()
     for col,value in enumerate(values):
      expected=float(Decimal.from_float(value)/denominator*(Decimal(1) if weights is None else Decimal.from_float(weights[col])));index=row*width+col
      self.assertLessEqual(abs(expected-case['math_fp64'][index]),1e-13+1e-13*abs(expected))
      self.assertEqual(struct.pack('<f',expected),struct.pack('<f',case['rounded_fp32'][index]));compared+=1
  self.assertEqual(compared,sum(len(c['input']) for c in m.cases()))
 def test_direct_scale_has_no_offset_and_closed_form(self):
  case=m.cases()[0];self.assertAlmostEqual(case['math_fp64'][0],2/math.sqrt(1+m.EPSILON),places=14)
  case=m.cases()[1];self.assertEqual(case['math_fp64'][0],0);self.assertLess(case['math_fp64'][1],1);self.assertGreater(case['math_fp64'][2],0)
 def test_actual_and_tail_shapes_present(self):
  cases=m.cases();self.assertEqual(len(cases),16);self.assertTrue({1,4,7,256,512,1536}<={c['width'] for c in cases});self.assertTrue(all(c['rows']<=256 for c in cases))
 def test_extreme_finite_inputs_do_not_overflow_reference(self):
  case=next(c for c in m.cases() if c['name']=='finite-extremes');self.assertTrue(all(math.isfinite(v) for v in case['math_fp64']));self.assertEqual(case['rounded_fp32'][:8],[1.]*8)
 def test_mixed_magnitudes_have_normal_amplified_outputs(self):
  case=next(c for c in m.cases() if c['name']=='mixed-magnitude-amplified');self.assertLess(case['input'][1]/case['input'][0],2.**-126);self.assertEqual(case['input'][3],2.**-149);self.assertGreater(case['math_fp64'][1],1.4);self.assertGreater(case['math_fp64'][3],1e-5)
 def test_exact_epsilon_and_weighted_ones_equivalence(self):
  cases=m.cases();self.assertEqual(struct.pack('<f',m.EPSILON),bytes.fromhex('bd378635'))
  a=next(c for c in cases if c['name']=='global-v');b=next(c for c in cases if c['name']=='global-v-ones');self.assertEqual(a['rounded_fp32'],b['rounded_fp32'])
 def test_weighted_output_overflow_is_not_relaxed(self):
  case=m.document()['overflow_case'];self.assertGreater(case['math_fp64'][0],m.MAXIMUM)
  with self.assertRaises(OverflowError):m.evaluate(case['input'],case['weights'],1,2)
 def test_invalid_inputs_and_shapes_reject(self):
  for values,weights,rows,width in [([],None,0,1),([1.],None,1,2),([1.],[],1,1),([math.inf],None,1,1),([1.],[math.nan],1,1),([1.1],None,1,1),([1.],None,True,1),([True],None,1,1),([1.],[False],1,1)]:
   with self.assertRaises((ValueError,OverflowError)):m.evaluate(values,weights,rows,width)
 def test_output_is_deterministic_and_never_replaced(self):
  with tempfile.TemporaryDirectory(prefix='toklet-rms-reference-') as temporary:
   root=Path(temporary);a=root/'a';b=root/'b';first=m.write(a);self.assertEqual(first,m.write(b));self.assertEqual((a/'math.json').read_bytes(),(b/'math.json').read_bytes())
   with self.assertRaises(FileExistsError):m.write(a)
   self.assertEqual(json.loads((a/'manifest.json').read_bytes()),first)
if __name__=='__main__':unittest.main(verbosity=2)
