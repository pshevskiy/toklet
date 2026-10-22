"""Sealed frozen task-ID jobs and physical raw response admission; no grading."""
from pathlib import Path
import json
import sys
import m7_physical_measurement as measurement
ROOT=measurement.ROOT
sys.path.insert(0,str(ROOT/'tools/bench'))
import task_cases
NATIVE_SHA='06016a4e6eeea98c9c9c0f898cc4fd107c85415ed26cd3faa8de221386002971'
RUNNER_PROOF='benchmarks/results/2026-10-21-m7-task-id-compat-qualification-v1.json'
RUNNER_PROOF_SHA='08e784ebe4274ec9cf8c643b46382eb4a659dbdbb4c4188f89ce4faba83089dc'
TASK_FILE='artifacts/quality-v1/tasks.jsonl'
READY_FORMAT='toklet-m7-physical-tasks-ready-v1'
RESULT_FORMAT='toklet-m7-physical-tasks-v1'


def frozen_tasks():
    proof=measurement.integration.metadata(ROOT/measurement.FROZEN,measurement.FROZEN_SHA)
    name='benchmarks/contract-v1/manifest.json';expected=proof['bindings'][name]
    manifest=measurement.integration.metadata(ROOT/name,expected)
    with (ROOT/TASK_FILE).open('rb') as stream:raw=stream.read(task_cases.MAX_PAYLOAD_BYTES+1)
    if len(raw)>task_cases.MAX_PAYLOAD_BYTES or measurement.package.hashlib.sha256(raw).hexdigest()!=task_cases.TASK_SHA:
        raise ValueError('Frozen100-task payload differs')
    cases=[json.loads(line) for line in raw.decode().splitlines()]
    task_cases.validate_cases(cases,manifest['quality'])
    pins={measurement.FROZEN:measurement.FROZEN_SHA,name:expected,TASK_FILE:task_cases.TASK_SHA}
    for module in (task_cases,task_cases.prepare):
        path=Path(module.__file__).resolve();pins[str(path.relative_to(ROOT))]=measurement.package.sha(path)
    return cases,pins


def task_payload(model,workload):
    if model not in ('tiny-cpu','tiny-metal','small','larger') or workload not in ('short','full'):
        raise ValueError('Unsupported physical task job')
    if model.startswith('tiny'):
        if workload!='short':raise ValueError('Tiny is component scope only')
        cases=[dict(id='first',promptTokens=[1,7,13],maximum=3),dict(id='second',promptTokens=[1,7,13,3],maximum=2)]
        data=dict(version=1,vocabulary=64,stops=[62,63],cases=cases);pins={}
    else:
        original,pins=frozen_tasks();selected=original if workload=='full' else original[:1]
        cases=[dict(id=row['id'],promptTokens=row['prompt_tokens'],maximum=row['max_output_tokens'] if workload=='full' else min(8,row['max_output_tokens'])) for row in selected]
        data=dict(version=1,vocabulary=task_cases.VOCAB,stops=list(task_cases.EOS_IDS),cases=cases)
    raw=(json.dumps(data,sort_keys=True,separators=(',',':'))+'\n').encode()
    if len(raw)>512*1024:raise ValueError('Bounded derived task set exceeded')
    return raw,data,pins


def job(model,workload,verify_sources=True):
    base,host,pins=measurement.job(model,'short',verify_sources=verify_sources)
    proof=measurement.integration.metadata(ROOT/RUNNER_PROOF,RUNNER_PROOF_SHA)
    if (proof['passed'] is not True or proof['app_integration'] is not True or proof['physical_device_qualified'] is not False
        or proof['evidence_pins']['artifacts/m7-apple-package-v5/build.json']!=NATIVE_SHA):raise ValueError('Reviewed sealed task runner required')
    measurement.verify_inputs(proof["runner_code_pins"])
    measurement.integration.merge(pins,proof["runner_code_pins"])
    pins[RUNNER_PROOF]=RUNNER_PROOF_SHA
    raw,data,extra=task_payload(model,workload);measurement.integration.merge(pins,extra)
    digest=measurement.package.hashlib.sha256(raw).hexdigest()
    config={k:base[k] for k in ('version','modelFolder','packedFolder','files','metal','numericalBudget','processBudget')}
    config.update(capacity=6 if model.startswith('tiny') else 562,chunk=3 if model.startswith('tiny') else 128,
                  tasksFile='M7Tasks/'+digest+'/tasks.json',tasksSHA256=digest,caseCount=len(data['cases']))
    return config,host,pins


def stage(config,host,output,model,workload):
    base,name,digest=measurement.stage(config,host,output)
    raw,data,_=task_payload(model,workload)
    if measurement.package.hashlib.sha256(raw).hexdigest()!=config['tasksSHA256'] or len(data['cases'])!=config['caseCount']:
        raise ValueError('Staged frozen task set differs')
    path=base/config['tasksFile'];path.parent.mkdir(parents=True)
    with path.open('xb') as stream:stream.write(raw)
    return base,name,digest


def validate_task_listing(data,device,config):
    result=data['result'];parts=Path(config['tasksFile']).parts
    if (len(parts)!=3 or parts!=('M7Tasks',config['tasksSHA256'],'tasks.json')
        or data['info']['outcome']!='success' or result['deviceIdentifier']!=device or result['domain']!='appDataContainer'
        or result['domainIdentifier']!=measurement.physical.BUNDLE or len(result['files'])!=1):raise ValueError('Copied task listing identity differs')
    model=config['modelFolder'].split('/')[1];workload='full' if config['caseCount']==100 else 'short'
    raw,_,_=task_payload(model,workload);row=result['files'][0]
    if (row['name']!='tasks.json' or row['relativePath']!='tasks.json' or row['resources']['isSymbolicLink'] is not False
        or row['resources']['isDirectory'] is not False or type(row['metadata']['size']) is not int or row['metadata']['size']!=len(raw)
        or measurement.package.hashlib.sha256(raw).hexdigest()!=config['tasksSHA256']):raise ValueError('Copied task inventory/bytes differ')


def check_console(text,config,signed,task_set,*,require_installation=True):
    if type(require_installation) is not bool:raise ValueError('Explicit installation-proof scope required')
    if config.get('version')==2:
        if config.get('mlx') is not True or config.get('metal') is not True or config.get('packedFolder')!='':
            raise ValueError('Explicit MLX backend contract differs')
        backend='metal-mlx'
    else:
        if config.get('version')!=1 or 'mlx' in config:raise ValueError('Legacy backend contract differs')
        backend='metal' if config['metal'] else 'cpu'
    uint=measurement.uint;number=measurement.nonnegative;cases=task_set['cases'];vocabulary=task_set['vocabulary'];stops=task_set['stops']
    if (not uint(config['caseCount'],32) or not 1<=config['caseCount']<=100 or len(cases)!=config['caseCount']
        or set(task_set)!={'version','vocabulary','stops','cases'} or not uint(task_set['version'],32) or task_set['version']!=1 or not uint(vocabulary,32) or vocabulary==0 or not 1<=len(stops)<=2
        or len(set(stops))!=len(stops) or any(not uint(t,32) or t>=vocabulary for t in stops)
        or not uint(config['capacity'],32) or not 1<=config['capacity']<=562 or not uint(config['chunk'],32) or not 1<=config['chunk']<=min(128,config['capacity'])):
        raise ValueError('Bounded task reference required')
    raw=(json.dumps(task_set,sort_keys=True,separators=(',',':'))+'\n').encode()
    if measurement.package.hashlib.sha256(raw).hexdigest()!=config['tasksSHA256']:raise ValueError('Exact task reference SHA differs')
    if len({c['id'] for c in cases})!=len(cases):raise ValueError('Unique ordered task IDs required')
    for case in cases:
        if (set(case)!={'id','promptTokens','maximum'} or not isinstance(case['id'],str) or not case['id'] or len(case['id'].encode())>80
            or not case['id'].isascii() or any(not (c.isalnum() or c in './_-') for c in case['id'])
            or not uint(case['maximum'],32) or not 1<=case['maximum']<=256 or not 1<=len(case['promptTokens'])<=306
            or len(case['promptTokens'])+case['maximum']>config['capacity'] or any(not uint(t,32) or t>=vocabulary for t in case['promptTokens'])):
            raise ValueError('Task prompt/admission reference differs')
    prefixes={'TOKLET_INSTALLATION_JSON=':'proof','TOKLET_TASK_CHUNK_JSON=':'chunk','TOKLET_TASK_CASE_JSON=':'case','TOKLET_TASKS_JSON=':'result'}
    events=[]
    for line in text.splitlines():
        matched=False
        for prefix,kind in prefixes.items():
            if line.startswith(prefix):events.append((kind,json.loads(line[len(prefix):])));matched=True;break
        if line.startswith('TOKLET_') and not matched:raise ValueError('Unexpected task/error marker')
    if require_installation:
        if len(events)<4 or events[0][0]!='proof' or [e[0] for e in events[-2:]]!=['proof','result']:raise ValueError('Incomplete task collection')
        proofs=[events[0][1],events[-2][1]];cursor=1;limit=len(events)-2
        if ([p['phase'] for p in proofs]!=['before','after'] or any(p['simulator'] is not False or p['files']!=signed for p in proofs)):
            raise ValueError('Task installed proofs differ')
    else:
        # Mac command-line callers have a separately admitted linked binary,
        # no signed iOS bundle and no UIKit foreground/battery observations.
        if signed is not None or len(events)<3 or any(kind=='proof' for kind,_ in events) or events[-1][0]!='result':
            raise ValueError('Unexpected Mac installation proof scope')
        proofs=[];cursor=0;limit=len(events)-1
    result=events[-1][1]
    if (result['passed'] is not True or result.get('error') is not None or result['cleanupErrors'] or result['endpointErrors']
        or result['backend']!=backend or result['closed'] is not True or result['released'] is not True
        or result['fullQualityQualified'] is not False or result['physicalDeviceQualified'] is not False or result['m7Complete'] is not False
        or not uint(result['interruptionCount']) or result['interruptionCount']!=0 or not number(result['loadAndSessionSeconds'])
        or not uint(result['casesCompleted'],32) or result['casesCompleted']!=len(cases) or not uint(result['chunksCompleted'],32)
        or not uint(result['loadedAdmittedBytes']) or not 0<result['loadedAdmittedBytes']<=config['numericalBudget']
        or result['filesBefore']!=config['files'] or result['filesAfter']!=config['files']
        or result.get('tasksBeforeSHA256')!=config['tasksSHA256'] or result.get('tasksAfterSHA256')!=config['tasksSHA256']):raise ValueError('Task result/proofs/admission differ')
    checkpoints=result['checkpoints'];phases=['before-artifact-proof','after-artifact-proof','loaded']+['case-'+str(i) for i in range(len(cases))]+['unloaded','closed','after-final-artifact-proof']
    if [c['phase'] for c in checkpoints]!=phases:raise ValueError('Task memory phases differ')
    ordered=[c['footprint'] for c in checkpoints[:3]];chunks=0;responses=[];previous_completed=0
    def observation(sample,index,case):
        if (not uint(sample['caseIndex'],32) or sample['caseIndex']!=index or sample['caseID']!=case['id']
            or not uint(sample['admittedBytes']) or sample['admittedBytes']!=result['loadedAdmittedBytes']):raise ValueError('Task sample identity/admission differs')
        env=sample['environment']
        if type(env['lowPowerMode']) is not bool or not uint(env['thermalState'],32) or env['thermalState'] not in range(4):
            raise ValueError('Task power/thermal scope differs')
        if require_installation:
            if (not uint(env['applicationState'],32) or env['applicationState']!=0 or not uint(env['batteryState'],32)
                or env['batteryState'] not in range(1,4) or not number(env['batteryLevel']) or env['batteryLevel']>1):
                raise ValueError('Task foreground/battery scope differs')
        elif any(env.get(key) is not None for key in ['applicationState','batteryState','batteryLevel']):
            raise ValueError('Unexpected Mac UIKit observation')
        ordered.append(sample['footprint'])
    for index,case in enumerate(cases):
        ids=[];sampled=0;chunk_index=0;reason=None;hidden=None;history=None;evaluated=None
        while sampled<case['maximum']:
            if cursor>=limit or events[cursor][0]!='chunk':raise ValueError('Task chunk missing or reordered')
            sample=events[cursor][1];cursor+=1;observation(sample,index,case);row=sample['chunk'];requested=min(128,case['maximum']-sampled)
            if (not uint(row['index'],32) or row['index']!=chunk_index or not uint(row['requested'],32) or row['requested']!=requested
                or not uint(row['reason'],32) or row['reason'] not in (1,2) or len(row['ids'])>requested or len(row['ids'])!=len(row['tokenTimings'])
                or any(not uint(t,32) or t>=vocabulary or t in stops for t in row['ids'])):raise ValueError('Task chunk policy/visible output differs')
            reason=row['reason'];hidden=row.get('stopToken')
            if reason==1:
                if len(row['ids'])!=requested or hidden is not None:raise ValueError('Task output limit differs')
            elif not uint(hidden,32) or hidden not in stops or len(row['ids'])>=requested:raise ValueError('Task hidden stop differs')
            sampled+=len(row['ids'])+(1 if hidden is not None else 0);ids.extend(row['ids']);history=len(case['promptTokens'])+sampled;evaluated=history-1
            if (not all(uint(row[k],32) for k in ('history','evaluated','pending')) or (row['history'],row['evaluated'],row['pending'])!=(history,evaluated,1)):raise ValueError('Task chunk committed prefix differs')
            start=sample['submittedUptimeNanoseconds'];end=sample['completedUptimeNanoseconds']
            if not uint(start) or start==0 or not uint(end) or not previous_completed<=start<end:raise ValueError('Task command brackets differ')
            elapsed=0;callback=start
            for offset,event in enumerate(row['tokenTimings']):
                value=event['nativeElapsedSeconds'];stamp=event['callbackUptimeNanoseconds']
                if (not uint(event['index'],32) or event['index']!=offset or not number(value) or value<elapsed
                    or not uint(stamp) or not callback<=stamp<=end or value>(stamp-start)/1e9):raise ValueError('Task callback order/clocks differ')
                elapsed=value;callback=stamp
            previous_completed=end;chunk_index+=1;chunks+=1
            if hidden is not None:break
        if cursor>=limit or events[cursor][0]!='case':raise ValueError('Reset-acknowledged task response missing')
        sample=events[cursor][1];cursor+=1;observation(sample,index,case);response=sample['response']
        if (response['ids']!=ids or any(not uint(t,32) for t in response['ids'])
            or not all(uint(response[k],32) for k in ('sampled','reason','history','evaluated','pending','chunks'))
            or tuple(response[k] for k in ('sampled','reason','history','evaluated','pending','chunks'))!=(sampled,reason,history,evaluated,1,chunk_index)
            or response.get('stopToken')!=hidden or hidden is not None and not uint(response['stopToken'],32)
            or not all(uint(sample[k],32) and sample[k]==0 for k in ('resetHistory','resetEvaluated','resetPending'))):raise ValueError('Exact task response/reset differs')
        responses.append(sample);ordered.append(checkpoints[3+index]['footprint'])
    if cursor!=limit or chunks!=result['chunksCompleted']:raise ValueError('Task stream has extra/incomplete chunks')
    ordered.extend(c['footprint'] for c in checkpoints[-3:]);last_peak=0
    for value in ordered:
        if (not uint(value['physicalBytes']) or not uint(value['lifetimePeakPhysicalBytes']) or not 0<value['physicalBytes']<=value['lifetimePeakPhysicalBytes']<=config['processBudget']
            or value['lifetimePeakPhysicalBytes']<last_peak or not uint(value['returnedNaturalCount'],32) or value['returnedNaturalCount']==0):raise ValueError('Task whole-process observation differs')
        last_peak=value['lifetimePeakPhysicalBytes']
    return dict(result=result,case_count=len(cases),chunk_count=chunks,responses=responses,installed_proofs=proofs)


def validate_console(text,config,signed):
    parts=config['modelFolder'].split('/')
    if len(parts)!=3 or parts[0]!='M7Inputs' or parts[2]!='Model':raise ValueError('Canonical task model namespace required')
    model=parts[1];workload='full' if config['caseCount']==100 else 'short'
    expected,host,_=job(model,workload,verify_sources=False)
    if config!=expected:raise ValueError('Frozen physical task config differs')
    _,data,_=task_payload(model,workload)
    path=ROOT/host[config['modelFolder']+'/config.json']
    with path.open('rb') as stream:raw=stream.read(512*1024+1)
    if len(raw)>512*1024 or measurement.package.hashlib.sha256(raw).hexdigest()!=config['files'][config['modelFolder']+'/config.json'] or json.loads(raw)['vocab_size']!=data['vocabulary']:
        raise ValueError('Frozen task/model vocabulary differs')
    checked=check_console(text,config,signed,data)
    if model.startswith('tiny'):
        responses=checked['responses']
        if (len(responses)!=2 or any(row['response']['ids']!=[47]*maximum
            or row['response']['sampled']!=maximum or row['response']['reason']!=1
            or row['response'].get('stopToken') is not None for row,maximum in zip(responses,(3,2)))):
            raise ValueError('Independent tiny task response differs')
    return checked


if __name__=='__main__':
    if len(sys.argv)<9:raise SystemExit('Expected native private private_sha prepared prepared_sha model workload output and options')
    options=measurement.options(sys.argv[9:])
    sys.exit(measurement.main(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],Path(sys.argv[4]),sys.argv[5],sys.argv[6],sys.argv[7],Path(sys.argv[8]),tasks=True,native_sha=NATIVE_SHA,**options))
