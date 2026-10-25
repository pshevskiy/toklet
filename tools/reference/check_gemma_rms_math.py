#!/usr/bin/env python3
"""Independent Decimal80 replay from captured RMS bytes, without producer imports."""
from decimal import Decimal,localcontext
from pathlib import Path
import hashlib,json,math,struct,sys
EPSILON=struct.unpack('<f',bytes.fromhex('bd378635'))[0]
def read(path,limit):
 with path.open('rb') as f:body=f.read(limit+1)
 if len(body)>limit:raise ValueError('RMS witness exceeds bound')
 return body
def verify(math_bytes,manifest_bytes):
 data=json.loads(math_bytes);manifest=json.loads(manifest_bytes)
 assert data['format']=='toklet-gemma-metal-rms-math-v1' and manifest['format']=='toklet-gemma-metal-rms-manifest-v1'
 assert manifest['math_sha256']==hashlib.sha256(math_bytes).hexdigest() and manifest['math_bytes']==len(math_bytes)
 assert manifest['producer_sha256']==data['producer_sha256'] and len(data['producer_sha256'])==64
 assert data['epsilon']==EPSILON and data['gate']==dict(absolute=1e-5,relative=1e-4)
 cases=data['cases'];assert len(cases)==manifest['cases']==16 and len({c['name'] for c in cases})==16
 compared=0;max_error=0.;max_fraction=0.
 with localcontext() as context:
  context.prec=80;epsilon=Decimal.from_float(EPSILON)
  for case in cases:
   rows,width=case['rows'],case['width'];assert type(rows) is int and type(width) is int and 1<=rows<=256 and 1<=width<=1536 and case['epsilon']==EPSILON
   inputs,weights=case['input'],case['weights'];assert len(inputs)==len(case['math_fp64'])==len(case['rounded_fp32'])==rows*width and len(case['denominators_fp64'])==rows
   assert type(case['learned']) is bool and case['learned']==(weights is not None)
   assert weights is None or len(weights)==width
   for v in inputs+([] if weights is None else weights):assert type(v) in (int,float) and math.isfinite(v) and struct.unpack('<f',struct.pack('<f',v))[0]==v
   for row in range(rows):
    values=[Decimal.from_float(float(v)) for v in inputs[row*width:(row+1)*width]]
    denominator=(sum((v*v for v in values),Decimal(0))/width+epsilon).sqrt()
    recorded=case['denominators_fp64'][row];assert math.isfinite(recorded) and abs(float(denominator)-recorded)<=1e-13+1e-13*abs(float(denominator))
    for col,v in enumerate(values):
     expected=float(v/denominator*(Decimal(1) if weights is None else Decimal.from_float(float(weights[col]))));index=row*width+col
     actual=case['math_fp64'][index];assert type(actual) in (int,float) and math.isfinite(actual)
     error=abs(expected-actual);bound=1e-13+1e-13*abs(expected);assert error<=bound
     assert struct.pack('<f',expected)==struct.pack('<f',case['rounded_fp32'][index])
     compared+=1;max_error=max(max_error,error);max_fraction=max(max_fraction,error/bound)
  overflow=data['overflow_case']
  assert overflow['expected_error']=='nonfinite FP32 output'
  assert type(overflow['rows']) is int and overflow['rows']==1 and type(overflow['width']) is int and overflow['width']==2
  inputs,weights,outputs=overflow['input'],overflow['weights'],overflow['math_fp64']
  assert len(inputs)==len(weights)==len(outputs)==2
  maximum=struct.unpack('<f',bytes.fromhex('ffff7f7f'))[0]
  for value in inputs+weights:
   assert type(value) in (int,float) and math.isfinite(value) and struct.unpack('<f',struct.pack('<f',value))[0]==value
  assert inputs==[1.,0.] and weights==[maximum,1.]
  values=[Decimal.from_float(float(v)) for v in inputs]
  denominator=(sum((v*v for v in values),Decimal(0))/2+epsilon).sqrt()
  for value,weight,actual in zip(values,weights,outputs):
   expected=float(value/denominator*Decimal.from_float(float(weight)))
   assert type(actual) in (int,float) and math.isfinite(actual) and abs(expected-actual)<=1e-13+1e-13*abs(expected)
  assert outputs[0]>maximum and outputs[1]==0.
 assert compared==manifest['values']
 assert {1,4,7,256,512,1536}<={c['width'] for c in cases}
 mixed=next(c for c in cases if c['name']=='mixed-magnitude-amplified');assert mixed['math_fp64'][1]>1.4 and mixed['math_fp64'][3]>1e-5
 a=next(c for c in cases if c['name']=='global-v');b=next(c for c in cases if c['name']=='global-v-ones');assert a['input']==b['input'] and a['rounded_fp32']==b['rounded_fp32']
 assert data['source_or_device_qualified'] is False and data['whole_graph_qualified'] is False and data['performance_qualified'] is False and manifest['native_target_executed'] is False
 return dict(passed=True,cases=len(cases),values=compared,max_decimal_fp64_absolute_error=max_error,max_decimal_replay_bound_fraction=max_fraction,rounded_fp32_bits_match=True,exact_fp32_epsilon=EPSILON,source_or_device_qualified=False)
if __name__=='__main__':
 if len(sys.argv)!=2:raise SystemExit('usage: check_gemma_rms_math.py CAPTURE_DIRECTORY')
 directory=Path(sys.argv[1]);print(json.dumps(verify(read(directory/'math.json',8*1024**2),read(directory/'manifest.json',65536)),allow_nan=False))
