"""Frozen larger v2 paired 512/1 tile refinement; same weights and full logits."""
import argparse
import json
from pathlib import Path
import metal_prefill as shared
import metal_quality_v2 as loss

ROOT=shared.ROOT


def ready(event,accurate,larger):
    if accurate is not True or larger is not True:raise ValueError('Fixed larger compensated v2 modes required')
    shared.ready(event,True,True)
    expected={'weight_dtype':'Q4 v2 + FP16 exclusions','packed_format_version':2,'group_size':8,
              'quantizer':'toklet-q4-affine-f32-source-half-metadata-rne-g8-v2'}
    if any(type(event.get(k)) is not type(v) or event[k]!=v for k,v in expected.items()):raise ValueError('Actual prefill v2 representation differs')
    if any(event[k]['weights_bytes']!=loss.SPECS['larger']['weights'] for k in ('original_memory','tiled_memory')):
        raise ValueError('Actual prefill v2 residency differs')


def run(binary,output):
    if any(p.exists() for p in (output,output.with_suffix('.raw.jsonl'),output.with_suffix('.stderr'))):raise ValueError('Exclusive v2 prefill evidence exists')
    geometry_path=ROOT/loss.GEOMETRY_REL
    geometry=json.loads(loss.shared.frozen_bytes(geometry_path,1024**2,loss.GEOMETRY_SHA))
    if geometry.get('passed') is not True or geometry.get('review',{}).get('accepted') is not True:raise ValueError('Reviewed v2 geometry required')
    pins={ROOT/p:v['sha256'] for p,v in geometry['bindings'].items() if p.startswith('artifacts/m5-larger-q4-v2/')}
    if pins.get(ROOT/'artifacts/m5-larger-q4-v2/manifest.json')!=loss.SPECS['larger']['packed']:raise ValueError('Frozen v2 prefill artifact differs')
    pins[geometry_path]=loss.GEOMETRY_SHA
    modules=(loss,loss.shared,loss.large,loss.large.quality_rows)
    return shared.run(binary,output,True,True,artifact_path=ROOT/'artifacts/m5-larger-q4-v2',artifact_pins=pins,
                      admission_check=ready,extra_identity_paths=(Path(__file__).resolve(),geometry_path,
                      *[Path(m.__file__).resolve() for m in modules],*[p for p in pins if p!=geometry_path]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--binary',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.binary.resolve(),args.output.resolve())
