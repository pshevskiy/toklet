"""Frozen larger Q4 v2 task responses; literal grading remains separate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import metal_tasks as shared
import metal_cli_v2 as cli
import metal_quality_v2 as loss

ROOT=shared.ROOT
QUAL_REL='benchmarks/results/2026-10-14-q4-v2-cli-qualification-v1.json'
QUAL_SHA='6b95a3a0f502b7cde4539d569237971e45a3a42679a2b3429c7f0798892ee0fc'
LOSS_REL='benchmarks/results/2026-10-14-q4-v2-full-loss-qualification-v1.json'
LOSS_SHA='a8ac111bf5f4aef168986c6980284dcf22b78fb58340623fd57673f20d89b411'
BINARY_REL='artifacts/m5-q4-v2-cli-native-v1/toklet'
MODEL='larger'


def qualified_cli():
    path=ROOT/QUAL_REL
    q=json.loads(loss.shared.frozen_bytes(path,2*1024**2,QUAL_SHA))
    binary=ROOT/BINARY_REL
    if (q.get('format')!='toklet-q4-v2-cli-qualification-v1' or q.get('passed') is not True
            or q.get('review',{}).get('accepted') is not True or q.get('native_runs')!=8
            or q.get('raw_event_replay_count')!=109
            or shared.metal_engine.digest(binary)!=q['bindings'][BINARY_REL]['sha256']):
        raise ValueError('Qualified v2 CLI differs')
    prefix='artifacts/m5-q4-v2-cli-source-v1/'
    for name,value in q['bindings'].items():
        if name.startswith(prefix):
            if shared.metal_engine.digest(ROOT/name)!=value['sha256']:raise ValueError('Compiled v2 source snapshot differs')
            if name.startswith(prefix+'src/') and shared.metal_engine.digest(ROOT/name[len(prefix):])!=value['sha256']:
                raise ValueError('Current v2 numerical/CLI source differs from qualified binary')
    return binary,q


def compiled_pins(q):
    pins={ROOT/BINARY_REL:q['bindings'][BINARY_REL]['sha256']}
    prefix='artifacts/m5-q4-v2-cli-source-v1/'
    for name,value in q['bindings'].items():
        if not name.startswith(prefix):continue
        path=name[len(prefix):]
        if path.startswith('src/') or path in ('Cargo.toml','Cargo.lock'):
            pins[ROOT/path]=value['sha256']
    return pins


def check_initial_pins(identities,pins):
    if any(identities.get(str(p))!=sha for p,sha in pins.items()):
        raise ValueError('Initial frozen v2 task inputs differ')


def ready(event,model):
    if model!=MODEL:raise ValueError('Only the preselected larger loss-qualified v2 release candidate is admitted')
    return cli.ready(event,cli.MODE,2048,128,weights=loss.SPECS[model]['weights'],external=247772624)


def loss_prerequisite():
    q=json.loads(loss.shared.frozen_bytes(ROOT/LOSS_REL,2*1024**2,LOSS_SHA))
    if q.get('passed') is not True or q.get('review',{}).get('accepted') is not True:
        raise ValueError('Reviewed full-loss prerequisite required')
    models=[m for m in q['models'] if m['model']==MODEL]
    if len(models)!=1 or models[0]['loss_gate_passed'] is not True or models[0]['scored_targets']!=10000:
        raise ValueError('Larger v2 full-loss gate has not passed')
    return q


def run(output,check=False):
    paths=[output,output.with_suffix('.raw.jsonl'),output.with_suffix('.stderr')]
    if any(p.exists() for p in paths):raise ValueError('Exclusive v2 task evidence exists')
    binary,q=qualified_cli();loss_prerequisite();manifest,_=shared.controlled.contract()
    cases=shared.tasks.load_cases();spec=shared.controlled.MODELS[MODEL];source=ROOT/'models'/spec[0]
    packed=ROOT/'artifacts/m5-larger-q4-v2'
    artifact_pins={p:v['sha256'] for p,v in q['bindings'].items() if p.startswith('artifacts/m5-larger-q4-v2/')}
    if artifact_pins.get('artifacts/m5-larger-q4-v2/manifest.json')!=loss.SPECS[MODEL]['packed']:
        raise ValueError('Frozen v2 task artifact differs')
    modules=(shared,cli,cli.shared,loss,loss.shared,loss.large,loss.large.quality_rows,shared.controlled,
             shared.reference,shared.tasks,shared.tasks.prepare,shared.controlled.frozen,shared.controlled.common,
             shared.controlled.metal_prefill,shared.controlled.metal_continue,shared.metal_engine,shared.native_usage)
    files=[binary,Path(__file__),ROOT/QUAL_REL,ROOT/LOSS_REL,ROOT/'Cargo.toml',ROOT/'Cargo.lock',
           ROOT/manifest['quality']['tasks_file'],ROOT/'benchmarks/sources.json',ROOT/'benchmarks/result.schema.json',
           ROOT/'tools/bench/requirements.lock',shared.controlled.prepare.CONTRACT/'manifest.json']
    files += [Path(m.__file__) for m in modules]+list((ROOT/'src').rglob('*.rs'))+list((ROOT/'src').rglob('*.metal'))
    files += [shared.controlled.prepare.CONTRACT/v['file'] for v in manifest['arrays'].values()]
    files += [source/name for name in ('model.safetensors','config.json',*cli.shared.ASSET_PINS)]+[ROOT/p for p in artifact_pins]
    identities={str(p):shared.metal_engine.digest(p) for p in files}
    shared.controlled.initial_pins(identities,manifest,source,spec)
    pins={ROOT/QUAL_REL:QUAL_SHA,ROOT/LOSS_REL:LOSS_SHA,ROOT/manifest['quality']['tasks_file']:shared.tasks.TASK_SHA}
    pins.update(compiled_pins(q))
    pins.update({ROOT/p:sha for p,sha in artifact_pins.items()})
    pins.update({source/name:sha for name,sha in cli.shared.ASSET_PINS.items()})
    check_initial_pins(identities,pins)
    record={'format':'toklet-q4-v2-task-collection-v1','passed':False,'check_only':check,'collection_completed':False,
            'quality_completed':False,'model':MODEL,'backend':'metal','projection_mode':cli.MODE,
            'normalization_mode':'Compensated','readback_mode':'FullLogits','weight_dtype':'Q4 v2 + FP16 exclusions',
            'packed_format_version':2,'group_size':8,'task_ids':[v['id'] for v in cases],
            'task_payload_sha256':shared.tasks.TASK_SHA,'identities':identities,
            'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'scope':'One frozen larger greedy v2 response collection; literal grading and latency separate'}
    with paths[0].open('x') as summary,paths[1].open('x') as raw,paths[2].open('xb') as stderr:
        try:
            record['run']=shared.child(binary,source,MODEL,cases,raw,stderr,check,admission_check=ready,
                                       projection_argument=cli.ARG,extra_args=('--q4-weights',str(packed)))
            if not record['run']['passed']:raise ValueError('V2 task worker failed')
            record.update(passed=True,collection_completed=not check)
        except BaseException as error:record['error']=f'{type(error).__name__}: {error}'
        finally:
            record['endpoint_errors']=[]
            try:record['identities_unchanged']=all(shared.metal_engine.digest(Path(p))==sha for p,sha in identities.items())
            except BaseException as error:record['identities_unchanged']=False;record['endpoint_errors'].append(str(error))
            if not record['identities_unchanged'] or record['endpoint_errors']:record.update(passed=False,collection_completed=False)
            json.dump(record,summary,indent=2,allow_nan=False);summary.write('\n')
    if not record['passed']:raise ValueError('V2 task collection failed; see retained evidence')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--check',action='store_true');args=parser.parse_args();run(args.output.resolve(),args.check)
