"""Independent Half-input router fixture using the frozen scalar sorting oracle."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import moe_router_fixture as scalar

def export():
    cases=[]
    def add(name,rows,k):
        raw=[[struct.pack('<e',x)for x in row]for row in rows]
        values=[[struct.unpack('<e',x)[0]for x in row]for row in raw]
        assert all(math.isfinite(x)for row in values for x in row)
        case=scalar.case(name,values,k)
        del case['score_bits'],case['wide_gates']
        case['half_words']=[struct.unpack('<H',x)[0]for row in raw for x in row]
        cases.append(case)
    for c in scalar.export()['cases']:
        if c['name']=='extreme-finite':continue  # This FP32-only domain cannot be widened from Half.
        x=[struct.unpack('<f',struct.pack('<I',v))[0]for v in c['score_bits']]
        e=c['experts'];add(c['name'],[x[i:i+e]for i in range(0,len(x),e)],c['selected'])
    add('half-rounding-tie',[[1.0001,1.0002,1.,0.]],2)
    add('mixed-cutoff',[[2.]+[1.]*9+[-1.]*22],8)
    add('mass-order',[[0.]+[-17.]*31],8)
    add('half-extremes',[[65504.,-65504.,0.,65504.]],4)
    add('full-capacity',[[((e*13+t*7)%32-16)/4 for e in range(32)]for t in range(32)],8)
    return dict(format='toklet-moe-router-half-v1',producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scalar_sha256=hashlib.sha256(Path(scalar.__file__).read_bytes()).hexdigest(),gate_atol=2e-7,gate_rtol=2e-6,cases=cases,scope='Half inputs, exact IDs, ordered FP32 gates; standalone router only')
if __name__=='__main__':
    with Path(sys.argv[1]).open('x')as file:
        json.dump(export(),file,indent=2,allow_nan=False);file.write('\n')
