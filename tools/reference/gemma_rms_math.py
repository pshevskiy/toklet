#!/usr/bin/env python3
"""Independent pre-target FP32-storage/FP64 RMS witnesses; no engine/model imports."""
from pathlib import Path
import hashlib,json,math,struct,sys
EPSILON=struct.unpack('<f',struct.pack('<f',1e-6))[0]
MAXIMUM=struct.unpack('<f',bytes.fromhex('ffff7f7f'))[0]
def f32(value):
 value=struct.unpack('<f',struct.pack('<f',value))[0]
 if not math.isfinite(value):raise ValueError('nonfinite FP32 value')
 return value
def evaluate(values,weights,rows,width):
 if type(rows) is not int or type(width) is not int or rows<1 or width<1 or rows*width>1024**2 or len(values)!=rows*width or (weights is not None and len(weights)!=width):raise ValueError('RMS reference shape/bound')
 if any(type(v) not in (int,float) or not math.isfinite(v) or f32(v)!=v for v in values) or (weights is not None and any(type(w) not in (int,float) or not math.isfinite(w) or f32(w)!=w for w in weights)):raise ValueError('RMS requires finite FP32 storage')
 math_values=[];rounded=[];denominators=[]
 for row in range(rows):
  source=values[row*width:(row+1)*width]
  denominator=math.sqrt(math.fsum(v*v for v in source)/width+EPSILON);denominators.append(denominator)
  for i,v in enumerate(source):
   value=v/denominator*(1.0 if weights is None else weights[i]);math_values.append(value);rounded.append(f32(value))
 return dict(math_fp64=math_values,rounded_fp32=rounded,denominators_fp64=denominators)
def cases():
 result=[]
 def add(name,rows,width,values,weights):
  result.append(dict(name=name,rows=rows,width=width,learned=weights is not None,epsilon=EPSILON,input=values,weights=weights,**evaluate(values,weights,rows,width)))
 add('hand-unlearned',2,4,[2.,0.,0.,0.,0.,0.,-3.,0.],None)
 add('hand-direct-scales',2,4,[-1.,2.,-4.,8.,2.,-3.,4.,-5.],[0.,1.,-2.,.5])
 add('width-one',4,1,[0.,-0.,f32(3e38),f32(1e-30)],[-1.5])
 for name,rows,width,learned in [('odd-tail',3,7,True),('local-v',24,256,False),('global-v',24,512,False),('local-qk',24,256,True),('global-qk',24,512,True),('hidden',3,1536,True),('per-layer',3,256,True)]:
  values=[f32((((i*37+11)%257)-128)/64*2.**((i//width)%7-3)) for i in range(rows*width)]
  weights=[f32((((i*13)%19)-9)/8) for i in range(width)] if learned else None
  add(name,rows,width,values,weights)
 add('finite-extremes',4,8,[MAXIMUM]*8+[MAXIMUM]+[0.]*7+[f32(1e-30*(-1 if i%2 else 1)) for i in range(8)]+[0.,-0.]*4,None)
 add('positive-reduction-scales',2,1536,[f32((1. if i%97==0 else 2.**(-12-(i%7)))*(-1 if i%2 else 1)) for i in range(2*1536)],[f32((i%7-3)/4) for i in range(1536)])
 # Exact FP32 epsilon stays in the oracle; a target's rounded sqrt is not
 # squared back into a replacement epsilon. Probe both sides of its boundary.
 root=math.sqrt(EPSILON)
 add('epsilon-neighborhood',4,1,[f32(root*v) for v in (.5,1-2.**-12,1+2.**-12,2.)],[MAXIMUM])
 minimum=f32(2.**-149)
 add('mixed-magnitude-amplified',2,2,[MAXIMUM,1.,0.,minimum],[0.,MAXIMUM])
 add('unequal-large-scales',1,4,[MAXIMUM,1024.,1.,f32(2.**-100)],[0.,MAXIMUM,MAXIMUM,MAXIMUM])
 unlearned=next(c for c in result if c['name']=='global-v')
 add('global-v-ones',unlearned['rows'],unlearned['width'],list(unlearned['input']),[1.]*unlearned['width'])
 return result
def document():
 overflow_input=[1.,0.];overflow_weights=[MAXIMUM,1.]
 denominator=math.sqrt(.5+EPSILON)
 return dict(format='toklet-gemma-metal-rms-math-v1',producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),precision='finite FP32 storage; independent FP64 fsum/sqrt/direct scales; once-rounded FP32 outputs',scale_convention='direct learned weight, no implicit +1; unlearned weight=1',epsilon=EPSILON,gate=dict(absolute=1e-5,relative=1e-4),cases=cases(),overflow_case=dict(rows=1,width=2,input=overflow_input,weights=overflow_weights,math_fp64=[MAXIMUM/denominator,0.],expected_error='nonfinite FP32 output'),source_or_device_qualified=False,whole_graph_qualified=False,performance_qualified=False)
def write(directory):
 data=document();body=(json.dumps(data,indent=2,allow_nan=False)+'\n').encode();assert len(body)<=8*1024**2
 directory.mkdir() # New dedicated output directory; never overwrite a capture.
 with (directory/'math.json').open('xb') as f:f.write(body)
 manifest=dict(format='toklet-gemma-metal-rms-manifest-v1',producer_sha256=data['producer_sha256'],math_sha256=hashlib.sha256(body).hexdigest(),math_bytes=len(body),cases=len(data['cases']),values=sum(len(c['input']) for c in data['cases']),native_target_executed=False)
 with (directory/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
 return manifest
if __name__=='__main__':
 if len(sys.argv)!=2:raise SystemExit('usage: gemma_rms_math.py NEW_OUTPUT_DIRECTORY')
 print(json.dumps(write(Path(sys.argv[1]))))
