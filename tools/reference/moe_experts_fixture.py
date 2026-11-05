"""Independent nested expert-axis oracle; explicit FP32 storage, no engine import."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]

def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]

def flatten(x):
    if isinstance(x, list):
        return [y for v in x for y in flatten(v)]
    return [x]

def dot(a, b):
    total = 0.
    for x, y in zip(a, b, strict=True):
        total = f32(total + f32(x*y))
    return total

def silu(x):
    exp = f32(math.exp(-abs(x)))
    denominator = f32(1.+exp)
    sigmoid = f32(1./denominator) if x >= 0. else f32(exp/denominator)
    return f32(x*sigmoid)

def case(name, tokens, experts=4, selected=2, hidden=3, intermediate=5, permutation=False, contaminate=False):
    x = [[f32(((t*3+j*5)%13-6)/8) for j in range(hidden)]for t in range(tokens)]
    gate_up = [[[[f32(((e*19+b*11+i*7+j*3)%29-14)/16) for j in range(hidden)]for i in range(intermediate)]for b in range(2)]for e in range(experts)]
    down = [[[f32(((e*13+j*5+i*3)%23-11)/16) for i in range(intermediate)]for j in range(hidden)]for e in range(experts)]
    ids = [[(experts-1-t-k)%experts for k in range(selected)]for t in range(tokens)]
    gates = [[f32(0. if (t+k)%5==0 else (k+1)/(selected+2)) for k in range(selected)]for t in range(tokens)]
    if contaminate:
        used = {e for row in ids for e in row}
        for e in range(experts):
            if e not in used:
                gate_up[e] = [[[f32(100.+e) for _ in range(hidden)]for _ in range(intermediate)]for _ in range(2)]
                down[e] = [[f32(-100.-e) for _ in range(intermediate)]for _ in range(hidden)]
    if permutation:
        order = list(reversed(range(experts)))
        gate_up = [gate_up[e]for e in order];down = [down[e]for e in order]
        ids = [[order.index(e)for e in row]for row in ids]
    activated, up_values, values, reduced = [], [], [], []
    for token, routing, weights in zip(x, ids, gates, strict=True):
        stage, ups, outputs = [], [], []
        for e in routing:
            gate = [dot(token,row)for row in gate_up[e][0]]
            up = [dot(token,row)for row in gate_up[e][1]]
            glu = [f32(silu(g)*u)for g,u in zip(gate,up,strict=True)]
            stage.append(glu);ups.append(up);outputs.append([dot(glu,row)for row in down[e]])
        reduced.append([dot([v[j]for v in outputs],weights)for j in range(hidden)])
        activated.append(stage);up_values.append(ups);values.append(outputs)
    arrays = dict(input=x, gate_up=gate_up, down=down, gates=gates,
                  activated=activated, up=up_values, expert_output=values, output=reduced)
    return dict(name=name,tokens=tokens,experts=experts,selected=selected,hidden=hidden,intermediate=intermediate,
                ids=flatten(ids),arrays={k:[bits(v)for v in flatten(a)]for k,a in arrays.items()})

def export():
    cases=[case('base',1),case('permuted-experts',1,permutation=True),case('unselected-perturbed',1,contaminate=True)]
    cases += [case('batch-'+str(n),n)for n in [7,8,9]]
    cases.append(case('real-expert-count',2,32,8,5,3))
    assert cases[0]['arrays']['output']==cases[1]['arrays']['output']==cases[2]['arrays']['output']
    controls=[dict(name='ranked-cancellation',values=[16777216.,1.,-16777216.],gates=[1.,1.,1.],expected=0.),
              dict(name='changed-slot-order',values=[16777216.,-16777216.,1.],gates=[1.,1.,1.],expected=1.),
              dict(name='separate-product-store',values=[-1.,1.+2**-23],gates=[1.,1.-2**-23],expected=0.),
              dict(name='zero-gates',values=[4.,-7.],gates=[0.,0.],expected=0.)]
    for c in controls:
        assert dot(c['values'],c['gates'])==c['expected']
    path=Path(__file__).resolve()
    return dict(format='toklet-moe-experts-fp32-v1',producer_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                policy='Scalar FP32 ascending projection sums, stable scalar SiLU, slot-ordered separate FP32 weighted products/sums',
                atol=2e-6,rtol=2e-5,cases=cases,reduction_controls=controls,
                scope='Scalar FP32 leaves only; no Half, Metal, import, full model or quality qualification')

if __name__=='__main__':
    with Path(sys.argv[1]).open('x')as file:
        json.dump(export(),file,indent=2,allow_nan=False);file.write('\n')
