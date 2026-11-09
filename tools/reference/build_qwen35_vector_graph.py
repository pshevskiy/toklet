"""Generate a separate graph with exactly one recurrence block replacement."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
IMPORT='from qwen35_recurrent_vector import step as vector_step\n'
START='                old=states[layer];new=[0.]*len(old);out=[]\n'
END='                states[layer]=new;save('
REPLACEMENT="                new,out=vector_step(qn,kn,v,decay,beta,states[layer],(g['key_heads'],g['value_heads'],g['key_dim'],g['value_dim']),staged)\n"


def generated(source):
    if source.count(START)!=1 or source.count(END)!=1 or source.count('\nP = ')!=1:
        raise ValueError('Exact original scalar recurrence block required')
    first=source.index(START);last=source.index(END,first);old=source[first:last]
    result=(source[:first]+REPLACEMENT+source[last:]).replace('\nP = ','\n'+IMPORT+'\nP = ',1)
    if result.replace(IMPORT+'\n','',1).replace(REPLACEMENT,old,1)!=source:
        raise ValueError('Inverse vector patch must restore original scalar source exactly')
    ast.parse(result)
    return result


def validate():
    original=(ROOT/'qwen35_cpu_graph.py').read_bytes();variant=(ROOT/'qwen35_vector_graph.py').read_bytes()
    if variant!=generated(original.decode()).encode():raise ValueError('Generated vector graph differs from exact patch')
    return dict(original_sha256=hashlib.sha256(original).hexdigest(),vector_sha256=hashlib.sha256(variant).hexdigest(),
        helper_sha256=hashlib.sha256((ROOT/'qwen35_recurrent_vector.py').read_bytes()).hexdigest(),
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inverse_patch_exact=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');a=parser.parse_args()
    if not a.check:(ROOT/'qwen35_vector_graph.py').write_text(generated((ROOT/'qwen35_cpu_graph.py').read_text()))
    print(json.dumps(validate()))
