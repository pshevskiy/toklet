"""Exact larger v2 full-response admission; no generated code execution."""
import hashlib
from pathlib import Path
import task_grade_admission as legacy
import metal_tasks_v2 as producer

ROOT=legacy.ROOT
evidence=legacy.evidence


def validate_records(record,cases,events):
    expected={'format':'toklet-q4-v2-task-collection-v1','passed':True,'collection_completed':True,
              'check_only':False,'quality_completed':False,'identities_unchanged':True,'endpoint_errors':[],
              'task_payload_sha256':producer.shared.tasks.TASK_SHA,'model':'larger','backend':'metal',
              'projection_mode':producer.cli.MODE,'normalization_mode':'Compensated','readback_mode':'FullLogits',
              'weight_dtype':'Q4 v2 + FP16 exclusions','packed_format_version':2,'group_size':8}
    if len(cases)!=100 or any(type(record.get(k)) is not type(v) or record[k]!=v for k,v in expected.items()) or record.get('task_ids')!=[c['id'] for c in cases]:
        raise ValueError('Complete frozen larger v2 collection required')
    run=record.get('run',{})
    if (run.get('passed') is not True or run.get('model_exited') is not True or type(run.get('exit_code')) is not int
            or run['exit_code']!=0 or run.get('cleanup_errors')!=[] or len(run.get('cases',[]))!=100
            or type(run.get('worker_pid')) is not int or run['worker_pid']<=1):
        raise ValueError('Naturally exited owned v2 model required')
    producer.ready(run['ready'],'larger');responses=[];expected_events=[run['ready']]
    for case,item in zip(cases,run['cases'],strict=True):
        if type(item.get('max_output_tokens')) is not int or item['max_output_tokens']!=case['max_output_tokens']:
            raise ValueError('Original full task cap differs')
        response=producer.shared.collect_response(case,item['native_events'],run['ready'])
        if response!=item['response']:raise ValueError('Saved v2 response differs from raw events')
        expected_events+=item['native_events'];responses.append(response)
    if events!=expected_events or [u.get('phase') for u in run.get('idle_usage',[])]!=['ready']+[f'case_{i}' for i in range(1,101)]:raise ValueError('Complete ordered raw/idle event coverage differs')
    for usage in run['idle_usage']:
        a,b=(usage[k].get('lifetime_max_phys_footprint') for k in ('worker','driver'))
        if (type(a) is not int or type(b) is not int or a<=0 or b<=0
                or type(usage.get('joint_native_lifetime_peak_bytes')) is not int
                or usage['joint_native_lifetime_peak_bytes']!=a+b or a+b>producer.shared.LIMIT):
            raise ValueError('V2 model-phase native charge differs')
    identity={k:record[k] for k in ('backend','model','projection_mode','normalization_mode','readback_mode','weight_dtype','packed_format_version','group_size')}
    return dict(identity=identity,responses=responses,worker_pid=run['worker_pid'],record=record)


def inventory(binary,q,source,manifest):
    modules=(producer.shared,producer.cli,producer.cli.shared,producer.loss,producer.loss.shared,producer.loss.large,
             producer.loss.large.quality_rows,producer.shared.controlled,producer.shared.reference,producer.shared.tasks,
             producer.shared.tasks.prepare,producer.shared.controlled.frozen,producer.shared.controlled.common,
             producer.shared.controlled.metal_prefill,producer.shared.controlled.metal_continue,
             producer.shared.metal_engine,producer.shared.native_usage)
    files=[binary,Path(producer.__file__),ROOT/producer.QUAL_REL,ROOT/producer.LOSS_REL,ROOT/'Cargo.toml',ROOT/'Cargo.lock',
           ROOT/manifest['quality']['tasks_file'],ROOT/'benchmarks/sources.json',ROOT/'benchmarks/result.schema.json',
           ROOT/'tools/bench/requirements.lock',producer.shared.controlled.prepare.CONTRACT/'manifest.json']
    files += [Path(m.__file__) for m in modules]+list((ROOT/'src').rglob('*.rs'))+list((ROOT/'src').rglob('*.metal'))
    files += [producer.shared.controlled.prepare.CONTRACT/v['file'] for v in manifest['arrays'].values()]
    files += [source/name for name in ('model.safetensors','config.json',*producer.cli.shared.ASSET_PINS)]
    files += [ROOT/p for p in q['bindings'] if p.startswith('artifacts/m5-larger-q4-v2/')]
    return {str(p) for p in files}


def validate_collection(path):
    path=Path(path).resolve();payload=legacy.bounded(path,legacy.MAX_SUMMARY);record=evidence.decode_event(payload.decode())
    raw=legacy.bounded(path.with_suffix('.raw.jsonl'),legacy.MAX_RAW)
    events=[evidence.decode_event(line) for line in raw.decode().splitlines()];cases=producer.shared.tasks.load_cases()
    valid=validate_records(record,cases,events);binary,q=producer.qualified_cli();producer.loss_prerequisite()
    source=ROOT/'models'/producer.loss.SPECS['larger']['name'];packed=ROOT/'artifacts/m5-larger-q4-v2'
    command=[str(binary),'--model',str(source),'--backend','metal','--metal-projection',producer.cli.ARG,
             '--interactive','--json','--context-capacity','2048','--prefill-chunk','128','--q4-weights',str(packed)]
    if record['run']['command']!=command:raise ValueError('Exact v2 task command differs')
    manifest=legacy.frozen_manifest();identities=record.get('identities')
    if type(identities) is not dict or set(identities)!=inventory(binary,q,source,manifest):raise ValueError('Complete v2 provenance inventory required')
    if any(not Path(p).resolve().is_relative_to(ROOT) or type(sha) is not str or len(sha)!=64 for p,sha in identities.items()):
        raise ValueError('V2 inventory escapes root or lacks SHA')
    producer.shared.controlled.initial_pins(identities,manifest,source,producer.shared.controlled.MODELS['larger'])
    pins=producer.compiled_pins(q)
    pins.update({ROOT/producer.QUAL_REL:producer.QUAL_SHA,ROOT/producer.LOSS_REL:producer.LOSS_SHA,
                 ROOT/manifest['quality']['tasks_file']:producer.shared.tasks.TASK_SHA})
    pins.update({source/name:sha for name,sha in producer.cli.shared.ASSET_PINS.items()})
    pins.update({ROOT/p:v['sha256'] for p,v in q['bindings'].items() if p.startswith('artifacts/m5-larger-q4-v2/')})
    producer.check_initial_pins(identities,pins)
    if any(producer.shared.metal_engine.digest(Path(p))!=sha for p,sha in identities.items()):raise ValueError('V2 collection endpoint differs')
    producer.shared.finalized(valid['responses'],source/'tokenizer.json',model_exited=True)
    valid.update(collection_path=str(path),collection_sha256=hashlib.sha256(payload).hexdigest(),raw_sha256=hashlib.sha256(raw).hexdigest(),cases=cases)
    return valid


def admit_idle(path,selected):
    payload=legacy.bounded(path,65536);proof=evidence.decode_event(payload.decode())
    if proof.get('format')!='toklet-q4-v2-task-idle-v1' or proof.get('collections')!=[str(Path(selected).resolve())]:
        raise ValueError('Exactly the completed v2 collection must be idle')
    accepted=validate_collection(selected)
    binding=dict(proof_path=str(Path(path).resolve()),proof_sha256=hashlib.sha256(payload).hexdigest(),
                 collections=[dict(path=accepted['collection_path'],sha256=accepted['collection_sha256'],
                                   raw_sha256=accepted['raw_sha256'],identity=accepted['identity'],worker_pid=accepted['worker_pid'])],
                 protected_binaries={'target/release/toklet':'5250694e814ea112148a52cfdeaf5bc6286ff2804b3acfd8f89e80bc73122a59',
                                     'target/release/examples/cpu_bench':'e3debabbae7db9e5801bc9bfba177275444ebe1d9849eb3ad0f2dff487ac8dea'})
    # Actual own-process/group absence must precede VM startup; use the same
    # qualified window's generic idle verification after constructing this binding.
    return binding
