"""Independent four-bit/BF16 table and projection witness, before composition.

This freezes distinct FP32-reconstruction and BF16-rounding contracts. Actual
MLX module outputs are observations, including numerical rejections; completing
the capture never grants BF16 checkpoint, graph, performance or device support.
The exporter imports no numerical package. Explicit comparisons admit the full
installed reference runtime before importing MLX.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import reference_runtime as runtime

WIDTHS=(64,128,1536,8960)
IDS=(2,0,2,1)
ROWS,GROUP=3,64
MAX_JSON,MAX_RAW=1024*1024,4*1024*1024
ATOL,RTOL=1e-5,1e-4
PATTERNS=(bytes.fromhex('1032547698badcfe'),bytes.fromhex('efcdab8967452301'),bytes.fromhex('547698badcfe1032'))


def encoded(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def f32(value):return struct.unpack('<f',struct.pack('<f',value))[0]
def bf16_bits(value):
 bits=struct.unpack('<I',struct.pack('<f',value))[0]
 if bits&0x7f800000==0x7f800000:raise ValueError('Finite BF16 operand required')
 result=(bits+0x7fff+((bits>>16)&1))>>16
 if result&0x7f80==0x7f80:raise ValueError('BF16 rounding overflow')
 return result
def bf16(value):return struct.unpack('<f',struct.pack('<I',bf16_bits(value)<<16))[0]
def code(row,column):return (column%16,15-column%16,(column+4)%16)[row]
def operands(width):
 if type(width) is not int or width not in WIDTHS:raise ValueError('Frozen bounded table width required')
 groups=width//GROUP
 scales=[bf16(((-1 if (r+g)%3==0 else 1)*(1+(g*3+r)%11))/173) if (g+r)%7 else 0.0 for r in range(ROWS) for g in range(groups)]
 biases=[bf16((.1,-.3,.0078125,.51,-.0625)[(g*2+r)%5]) for r in range(ROWS) for g in range(groups)]
 packed=b''.join(pattern*(width//16) for pattern in PATTERNS)
 x=[f32(((i*5+3)%17-8)/37) for i in range(2*width)]
 return packed,scales,biases,x


def contracts(width):
 packed,scales,biases,x=operands(width)
 # Semantic code sequence is independent of the literal packed-byte patterns.
 exact=[float(scales[(r*width+c)//GROUP])*code(r,c)+float(biases[(r*width+c)//GROUP]) for r in range(ROWS) for c in range(width)]
 reconstructed=list(map(f32,exact));rounded=list(map(bf16,exact))
 separate=[bf16(bf16(scales[(r*width+c)//GROUP]*code(r,c))+biases[(r*width+c)//GROUP]) for r in range(ROWS) for c in range(width)]
 ideal=[math.fsum(exact[r*width+c]*x[t*width+c] for c in range(width)) for t in range(2) for r in range(ROWS)]
 ordered=[];bf_ordered=[]
 for t in range(2):
  for r in range(ROWS):
   total=0.;other=0.
   for c in range(width):
    total+=float(reconstructed[r*width+c])*float(x[t*width+c])
    other+=float(rounded[r*width+c])*float(bf16(x[t*width+c]))
   ordered.append(f32(total));bf_ordered.append(bf16(other))
 def gathered(values):return [values[r*width+c] for r in IDS for c in range(width)]
 return dict(weight=('U32',[ROWS,width//8],list(struct.unpack('<'+'I'*(len(packed)//4),packed))),
  scales=('BF16',[ROWS,width//GROUP],scales),biases=('BF16',[ROWS,width//GROUP],biases),
  input_f32=('F32',[2,width],x),input_bf16=('BF16',[2,width],list(map(bf16,x))),ids=('I32',[len(IDS)],list(IDS)),
  reconstructed_f32=('F32',[ROWS,width],reconstructed),gather_f32=('F32',[len(IDS),width],gathered(reconstructed)),
  gather_bf16_once=('BF16',[len(IDS),width],gathered(rounded)),gather_bf16_separate=('BF16',[len(IDS),width],gathered(separate)),
  projection_ideal=('F64',[2,ROWS],ideal),projection_ordered_f32=('F32',[2,ROWS],ordered),projection_bf16_rounded_weights=('BF16',[2,ROWS],bf_ordered))


def pack(dtype,values):
 if dtype=='BF16':return struct.pack('<'+'H'*len(values),*(bf16_bits(x) for x in values))
 fmt={'F16':'e','F32':'f','F64':'d','U32':'I','I32':'i'}[dtype]
 return struct.pack('<'+fmt*len(values),*values)
def unpack(dtype,raw):
 if dtype=='BF16':return [struct.unpack('<f',struct.pack('<I',x<<16))[0] for x in struct.unpack('<'+'H'*(len(raw)//2),raw)]
 fmt={'F16':'e','F32':'f','F64':'d','U32':'I','I32':'i'}[dtype]
 return list(struct.unpack('<'+fmt*(len(raw)//struct.calcsize(fmt)),raw))
def store(raw,dtype,shape,values):
 data=pack(dtype,values);assert math.prod(shape)==len(values)
 entry=dict(dtype=dtype,shape=shape,offset=len(raw),bytes=len(data),sha256=sha(data));raw.extend(data);return entry


def export():
 raw=bytearray();cases=[]
 for width in WIDTHS:
  cases.append(dict(width=width,arrays={name:store(raw,*value) for name,value in contracts(width).items()}))
 return dict(format='toklet-gemma-packed-math-v1',producer_sha256=sha(Path(__file__).read_bytes()),
  bits=4,group_size=GROUP,rows=ROWS,ids=list(IDS),cases=cases,binary=dict(path='math.bin',bytes=len(raw),sha256=sha(raw)),
  native_composition_implemented=False,bf16_checkpoint_qualified=False,device_qualified=False),bytes(raw)


def verify(path,expected):
 data=json.loads(runtime.captured(path,MAX_JSON,runtime.sha(expected)),object_pairs_hook=runtime.json_unique,parse_constant=runtime.bad_json_number)
 raw=runtime.captured(path.parent/'math.bin',MAX_RAW,runtime.sha(data['binary']['sha256']))
 # Entire deterministic ledger and bytes must replay. The external owner binds
 # producer identity, so editing producer+reference cannot admit a prior capture.
 wanted,values=export()
 if data!=wanted or raw!=values:raise ValueError('Packed semantic contract differs')
 return data,raw


def load(raw,entry):
 part=raw[entry['offset']:entry['offset']+entry['bytes']]
 if len(part)!=entry['bytes'] or sha(part)!=entry['sha256']:raise ValueError('Packed array identity differs')
 return unpack(entry['dtype'],part)


def observe(value,mx,raw,attempt,shape,dtype):
 # Preserve completed supported raw arrays before semantic rejection. A failed
 # evaluation retains a pending attempt; no raw bytes are invented for it.
 mx.eval(value)
 names={mx.float32:'F32',mx.float16:'F16',mx.bfloat16:'BF16',mx.uint32:'U32',mx.int32:'I32'}
 attempt['actual_shape']=list(value.shape);attempt['actual_dtype']=str(value.dtype)
 if value.dtype not in names:raise ValueError('Unsupported observation dtype; metadata retained')
 actual_dtype=names[value.dtype]
 size={'F16':2,'BF16':2,'F32':4,'I32':4,'U32':4}[actual_dtype]
 if any(type(x) is not int or x<0 for x in value.shape) or math.prod(value.shape)*size>MAX_RAW-len(raw):raise ValueError('Bounded observation storage required')
 wire=bytes(value.view(mx.uint8).reshape(-1).tolist())
 if len(wire)!=math.prod(value.shape)*size:raise ValueError('Observation wire length differs')
 entry=dict(dtype=actual_dtype,shape=list(value.shape),offset=len(raw),bytes=len(wire),sha256=sha(wire))
 raw.extend(wire);attempt['array']=entry
 if actual_dtype!=dtype or list(value.shape)!=shape:raise ValueError('Exact observation dtype and shape required')
 vals=unpack(actual_dtype,wire)
 if not all(math.isfinite(x) for x in vals):raise ValueError('Nonfinite module observation; raw bytes retained')
 attempt['completed']=True
 return vals


def check(comparisons,width,name,actual,wanted,exact=False,diagnostic_alternative=False):
 if len(actual)!=len(wanted):raise ValueError('Observation shape differs')
 errors=[abs(a-b) for a,b in zip(actual,wanted)]
 failed=[i for i,(a,b) in enumerate(zip(actual,wanted)) if (a!=b if exact else abs(a-b)>ATOL+RTOL*abs(b))]
 comparisons.append(dict(width=width,name=name,values=len(actual),failed_values=len(failed),passed=not failed,
  max_absolute_error=max(errors,default=0.),max_gate_ratio=max((e/(ATOL+RTOL*abs(b)) for e,b in zip(errors,wanted)),default=0.),
  exact=exact,diagnostic_alternative=diagnostic_alternative,first_failed_indices=failed[:16]))


def execute_cases(mx,nn,data,raw,observed,cases,comparisons):
 for case in data['cases']:
  width=case['width'];arrays=case['arrays'];values={k:load(raw,v) for k,v in arrays.items()}
  ids=mx.array(values['ids'],dtype=mx.int32);case_row=dict(width=width,precisions=[]);cases.append(case_row)
  for precision,dtype in (('f32',mx.float32),('bf16',mx.bfloat16)):
   precision_row=dict(precision=precision,attempts=[],immutable_parameters=None);case_row['precisions'].append(precision_row)
   embedding=nn.QuantizedEmbedding(ROWS,width,group_size=GROUP,bits=4)
   linear=nn.QuantizedLinear(width,ROWS,bias=False,group_size=GROUP,bits=4)
   weight=mx.array(values['weight'],dtype=mx.uint32).reshape(ROWS,-1)
   scales=mx.array(values['scales'],dtype=dtype).reshape(ROWS,-1)
   biases=mx.array(values['biases'],dtype=dtype).reshape(ROWS,-1)
   for module in (embedding,linear):
    module.weight=weight;module.scales=scales;module.biases=biases
    if set(module.parameters())!={'weight','scales','biases'}:raise ValueError('Exact module parameter inventory required')
   # Direct wire comparisons bind the actual assigned tensors, not only names.
   expected=[pack('U32',values['weight']),pack('F32' if precision=='f32' else 'BF16',values['scales']),pack('F32' if precision=='f32' else 'BF16',values['biases'])]
   mx.eval(weight,scales,biases)
   before=[bytes(x.view(mx.uint8).reshape(-1).tolist()) for x in (weight,scales,biases)]
   if before!=expected:raise ValueError('Assigned parameter bytes differ')
   x=mx.array(values['input_'+precision],dtype=dtype).reshape(2,width)
   output_values={}
   for name,operation in (('gather',lambda:embedding(ids)),('linear',lambda:linear(x)),('tied',lambda:embedding.as_linear(x))):
    attempt=dict(name=name,completed=False,array=None);precision_row['attempts'].append(attempt)
    actual=observe(operation(),mx,observed,attempt,[len(IDS),width] if name=='gather' else [2,ROWS],'F32' if precision=='f32' else 'BF16')
    output_values[name]=actual
    if name=='gather':
     if precision=='f32':check(comparisons,width,'gather_f32',actual,values['gather_f32'],True)
     else:
      for contract in ('once','separate'):check(comparisons,width,'gather_bf16_'+contract,actual,values['gather_bf16_'+contract],True,True)
    else:
     check(comparisons,width,name+'_'+precision+'_vs_ideal',actual,values['projection_ideal'],diagnostic_alternative=precision=='bf16')
     check(comparisons,width,name+'_'+precision+'_vs_ordered_f32',actual,values['projection_ordered_f32'],diagnostic_alternative=precision=='bf16')
     if precision=='bf16':check(comparisons,width,name+'_bf16_vs_rounded_weights',actual,values['projection_bf16_rounded_weights'],diagnostic_alternative=True)
   check(comparisons,width,'tied_linear_identity_'+precision,output_values['linear'],output_values['tied'],True)
   mx.eval(weight,scales,biases)
   after=[bytes(x.view(mx.uint8).reshape(-1).tolist()) for x in (weight,scales,biases)]
   precision_row['immutable_parameters']=before==after
   if before!=after:raise ValueError('Immutable packed parameters changed')


def compare(reference,expected,output,device,admitted,run=None):
 import signal
 observed=bytearray();cases=[];comparisons=[];primary=None;cleanup=[];previous={};complete=False
 def interrupted(number,_):
  for value in (signal.SIGTERM,signal.SIGINT):signal.signal(value,signal.SIG_IGN)
  raise InterruptedError('Received signal '+str(number))
 try:
  previous={value:signal.signal(value,interrupted) for value in (signal.SIGTERM,signal.SIGINT)}
  data,raw=verify(reference,expected)
  if run is not None:run(data,raw,observed,cases,comparisons)
  else:
   import mlx.core as mx
   import mlx.nn as nn
   if (str(Path(mx.__file__).resolve())!=admitted['imports']['mlx.core'] or str(Path(nn.__file__).resolve())!=admitted['imports']['mlx.nn']):raise ValueError('Actual imported MLX origins differ')
   requested={'cpu':mx.cpu,'gpu':mx.gpu}[device]
   with mx.stream(requested):execute_cases(mx,nn,data,raw,observed,cases,comparisons)
  complete=True
 except (Exception,KeyboardInterrupt) as error:primary=type(error).__name__+': '+str(error)
 finally:
  for value in previous:signal.signal(value,signal.SIG_IGN)
  report=dict(format='toklet-gemma-packed-observation-v1',capture_complete=complete,requested_device=device,
   mathematical_reference_sha256=expected,admitted_runtime=admitted,cases=cases,comparisons=comparisons,
   required_fp32_and_tied_comparisons_passed=complete and all(x['passed'] for x in comparisons if not x['diagnostic_alternative']),
   diagnostic_alternatives_all_passed=complete and all(x['passed'] for x in comparisons if x['diagnostic_alternative']),
   immutable_parameters=complete and all(p['immutable_parameters'] is True for c in cases for p in c['precisions']),
   numerical_policy=dict(atol=ATOL,rtol=RTOL),binary=dict(path='observed.bin',bytes=len(observed),sha256=sha(observed)),
   primary_error=primary,cleanup_errors=cleanup,native_composition_implemented=False,bf16_checkpoint_qualified=False,device_qualified=False)
  try:
   with (output/'observed.bin').open('xb') as stream:stream.write(observed)
  except (Exception,KeyboardInterrupt) as error:cleanup.append(type(error).__name__+': '+str(error));report['capture_complete']=False
  try:
   with (output/'observation.json').open('xb') as stream:stream.write(encoded(report))
  finally:
   for value,handler in previous.items():signal.signal(value,handler)
 print(json.dumps(dict(capture_complete=report['capture_complete'],device=device,comparisons=len(comparisons),failed=sum(not x['passed'] for x in comparisons),primary_error=primary)))
 return 0 if report['capture_complete'] and not cleanup else 1


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('export','compare'))
 parser.add_argument('--output',type=Path,required=True);parser.add_argument('--reference',type=Path)
 parser.add_argument('--expected-reference-sha256');parser.add_argument('--device',choices=('cpu','gpu'))
 parser.add_argument('--runtime-manifest',type=Path);parser.add_argument('--expected-runtime-sha256')
 args=parser.parse_args()
 if args.mode=='export':
  manifest,raw=export()
  with (args.output.parent/'math.bin').open('xb') as stream:stream.write(raw)
  with args.output.open('xb') as stream:stream.write(encoded(manifest))
  print(json.dumps(dict(exported=True,bytes=len(raw),sha256=sha(raw))))
 else:
  admitted=runtime.admit_runtime(args.runtime_manifest,args.expected_runtime_sha256,[Path(__file__).resolve()])
  raise SystemExit(compare(args.reference,args.expected_reference_sha256,args.output,args.device,admitted))
