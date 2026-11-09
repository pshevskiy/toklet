"""Independent two-table hybrid graph fixture; existing tied producer unchanged.

Original graph state/attention arithmetic and its two frozen profiles are reused.
Only the final matrix operation receives an independently generated output table;
ordinary embedding lookup still reads the original input table.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import qwen35_cpu_graph as graph

HEAD = 'language_model.lm_head.weight'
EMBED = graph.P+'embed_tokens.weight'


def generate(destination):
    destination.mkdir()
    original,table,matrices,vectors,backing=graph.artifact()
    original_fixture=Path(__file__).resolve().parents[2]/'tests/fixtures/qwen35-cpu-graph-v1'
    if original!=(original_fixture/'weights.bin').read_bytes():raise ValueError('Original tied fixture bytes changed')
    raw=bytearray(original);codes=bytearray();scales=bytearray();biases=bytearray();head=[]
    for row in range(graph.GEOMETRY['vocab']):
        scale=[.03125,-.015625,.0078125][row%3];bias=-6.0*scale+(row%2)*.00390625
        sb,scale=graph.bf16(scale);bb,bias=graph.bf16(bias)
        cells=[(row*5+column*11+(row+column//8)*3+9)%16 for column in range(64)]
        for first in range(0,64,8):codes.extend(struct.pack('<I',sum(cells[first+lane]<<(lane*4) for lane in range(8))))
        scales.extend(struct.pack('<H',sb));biases.extend(struct.pack('<H',bb))
        head.append([graph.f32(scale*code+bias) for code in cells])
    def save(data):
        offset=len(raw);raw.extend(data);return dict(offset=offset,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    table[HEAD]=dict(kind='matrix',shape=[32,64],codes=save(codes),scales=save(scales),biases=save(biases))
    embedding=matrices[EMBED];matrices[HEAD]=head
    if head==embedding:raise ValueError('Independent output head aliases input table values')
    class OutputMath(graph.Math):
        def linear(self,values,weights):
            return super().linear(values,head if weights is embedding else weights)
    body=bytearray();cases=[]
    for name,tokens,offset in graph.CASES:
        profiles={}
        for profile,staged in [('ideal_fp64',False),('staged_fp32',True)]:
            arrays=[]
            for row in graph.forward(matrices,vectors,tokens,offset,staged,math_type=OutputMath):
                values=row.pop('values');encoded=struct.pack('<'+('f' if staged else 'd')*len(values),*values)
                arrays.append(dict(**row,offset=len(body),bytes=len(encoded),count=len(values),sha256=hashlib.sha256(encoded).hexdigest()));body.extend(encoded)
            profiles[profile]=arrays
        cases.append(dict(name=name,tokens=list(tokens),position_offset=offset,profiles=profiles))
    if len(body)>16*1024**2:raise ValueError('Bounded graph capture exceeded')
    (destination/'weights.bin').write_bytes(raw);(destination/'reference.bin').write_bytes(body)
    manifest=dict(format='toklet-qwen35-untied-graph-v1',geometry=graph.GEOMETRY,gate=graph.GATE,weights=table,
        embedding_name=EMBED,output_head_name=HEAD,backing_bytes=backing+len(codes)+len(scales)+len(biases),
        weights_bytes=len(raw),weights_sha256=hashlib.sha256(raw).hexdigest(),reference_bytes=len(body),reference_sha256=hashlib.sha256(body).hexdigest(),
        original_tied_weight_bytes=len(original),original_tied_weight_sha256=hashlib.sha256(original).hexdigest(),
        producer_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(graph.__file__),Path(__file__)]},
        cases=cases,fixture_only=True,real_model_qualified=False)
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    (destination/'README.md').write_text('''# Independent untied-output hybrid fixture

The original four-layer tied fixture weights are retained as an exact prefix.
An independent32x64affine4/G64/BF16 output matrix is appended, with different
values from the same-vocabulary input table. The original independent graph and
its idealFP64/stagedFP32 arithmetic stay unchanged. A reference-only final-matrix
selector uses the new head while direct token lookup uses the original embedding.

Every intermediate, committed state and complete logit row is captured for the
original one/context17/offset/future-token cases. These toy fixtures establish no
real9B, GPU, quality, context, memory-fit or managed-route qualification.

[Producer](../../../tools/reference/qwen35_untied_graph.py).
''')
    return dict(weights_bytes=len(raw),reference_bytes=len(body),manifest_sha256=hashlib.sha256((destination/'manifest.json').read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('destination',type=Path);args=parser.parse_args()
    print(json.dumps(generate(args.destination)))
