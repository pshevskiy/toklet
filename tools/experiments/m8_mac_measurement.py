"""Strict completed Mac measurement schema; no inference or physical proof claim.

The immutable M7 physical validator remains unchanged. This Mac counterpart
keeps its serial clocks, memory, state and first-eligible-duration policy while
requiring absent UIKit fields and never synthesizing installation proofs.
"""
import json
import math
from m7_physical_measurement import uint, nonnegative, job_backend


def validate_console(text,config,vocabulary=151936):
    if job_backend(config)!='metal-mlx' or type(vocabulary) is not int or not 0<vocabulary<=2**32:
        raise ValueError('Explicit native MLX Mac job and vocabulary required')
    def rows(prefix):return [json.loads(line[len(prefix):]) for line in text.splitlines() if line.startswith(prefix)]
    markers=('TOKLET_MEASUREMENT_SAMPLE_JSON=','TOKLET_MEASUREMENT_JSON=')
    if any(line.strip() and not line.startswith(markers) for line in text.splitlines()):
        raise ValueError('Unexpected Mac measurement marker or output')
    summaries=rows('TOKLET_MEASUREMENT_JSON=');samples=rows('TOKLET_MEASUREMENT_SAMPLE_JSON=')
    if len(summaries)!=1 or not 13<=len(samples)<=2048:raise ValueError('Incomplete bounded measurement output')
    order=[]
    for line in text.splitlines():
        for prefix,kind in [('TOKLET_MEASUREMENT_SAMPLE_JSON=','sample'),('TOKLET_MEASUREMENT_JSON=','result')]:
            if line.startswith(prefix):order.append(kind)
    if order!=['sample']*len(samples)+['result']:
        raise ValueError('App evidence marker order differs')
    result=summaries[0]
    filesystem=result.get('filesystemBefore')
    if (type(filesystem) is not dict or set(filesystem)!={'freeBytes','readStartedUptimeNanoseconds','readCompletedUptimeNanoseconds','instrument','scope'}
        or not uint(filesystem['freeBytes']) or filesystem['freeBytes']==0
        or filesystem['instrument']!='Foundation.FileManager.attributesOfFileSystem.systemFreeSize'
        or filesystem['scope']!='trusted-root-filesystem'
        or not uint(filesystem['readStartedUptimeNanoseconds']) or filesystem['readStartedUptimeNanoseconds']==0
        or not uint(filesystem['readCompletedUptimeNanoseconds'])
        or filesystem['readCompletedUptimeNanoseconds']<filesystem['readStartedUptimeNanoseconds']
        or not uint(samples[0]['submittedUptimeNanoseconds'])
        or filesystem['readCompletedUptimeNanoseconds']>samples[0]['submittedUptimeNanoseconds']):
        raise ValueError('Filesystem observation scope/window differs')
    if (result['passed'] is not True or result.get('error') is not None or result['cleanupErrors'] or result['endpointErrors']
        or result['backend']!=job_backend(config) or not uint(result['interruptionCount']) or result['interruptionCount']!=0
        or not nonnegative(result['loadAndSessionSeconds']) or not nonnegative(result['sumNativeLastEventSeconds'])
        or not uint(result['warmupsCompleted'],32) or not uint(result['measuredCompleted'],32)
        or result['closed'] is not True or result['released'] is not True or result['physicalDeviceQualified'] is not False or result['m7Complete'] is not False
        or result['filesBefore']!=config['files'] or result['filesAfter']!=config['files']):raise ValueError('Measurement identity/result/cleanup differs')
    checkpoints=result['checkpoints']
    if [r['phase'] for r in checkpoints]!=['before-artifact-proof','after-artifact-proof','loaded','unloaded','closed','after-final-artifact-proof']:
        raise ValueError('Memory checkpoint scope differs')
    def memory(m):
        if not (uint(m['physicalBytes']) and uint(m['lifetimePeakPhysicalBytes']) and 0<m['physicalBytes']<=m['lifetimePeakPhysicalBytes']<=config['processBudget']) or not uint(m['returnedNaturalCount'],32) or m['returnedNaturalCount']<=0:
            raise ValueError('Whole-process memory observation differs')
    for c in checkpoints:memory(c['footprint'])
    measured=[];total=0;ordered=[c['footprint'] for c in checkpoints[:3]]
    previous_completed=0
    for index,sample in enumerate(samples):
        if index>=13 and total>=config['minimumNativeSeconds']:
            raise ValueError('Sequence continued past its first eligible stopping point')
        if (not all(uint(sample[k],32) for k in ['index','status','history','evaluated','pending','reason'])
            or not uint(sample['admittedBytes']) or sample['admittedBytes']==0
            or not uint(sample['submittedUptimeNanoseconds']) or not uint(sample['completedUptimeNanoseconds'])
            or sample['submittedUptimeNanoseconds']<previous_completed
            or not all(uint(token,32) and token<vocabulary for token in sample['tokens'])
            or sample['index']!=index or sample['role']!=('warmup' if index<3 else 'measured') or sample['status']!=0
            or len(sample['tokens'])!=config['maximum'] or len(sample['events'])!=config['maximum']
            or sample['tokens'][:len(config['expectedPrefix'])]!=config['expectedPrefix']
            or (sample['history'],sample['evaluated'],sample['pending'],sample['reason'])!=(len(config['prompt'])+config['maximum'],len(config['prompt'])+config['maximum']-1,1,1)
            or sample['admittedBytes']>config['numericalBudget']):raise ValueError('Sample oracle/state/admission differs')
        if sample['tokens']!=samples[0]['tokens']:
            raise ValueError('Repeated greedy response IDs differ')
        memory(sample['footprint']);ordered.append(sample['footprint']);environment=sample['environment']
        if (environment.get('applicationState') is not None or environment.get('batteryState') is not None
            or environment.get('batteryLevel') is not None or type(environment['lowPowerMode']) is not bool
            or not uint(environment['thermalState'],32) or environment['thermalState'] not in (0,1,2,3)):
            raise ValueError('Mac power/thermal environment differs')
        first=None;previous=0;callback=sample['submittedUptimeNanoseconds']
        for offset,event in enumerate(sample['events']):
            native=event['nativeElapsedSeconds'];stamp=event['callbackUptimeNanoseconds']
            if (not uint(event['index'],32) or event['index']!=offset or not nonnegative(native) or native<previous
                or not uint(stamp) or not callback<=stamp<=sample['completedUptimeNanoseconds']
                or native>(stamp-sample['submittedUptimeNanoseconds'])/1e9):raise ValueError('Sample clock/index differs')
            if first is None:first=native
            previous=native;callback=stamp
        previous_completed=sample['completedUptimeNanoseconds']
        elapsed=(sample['completedUptimeNanoseconds']-sample['submittedUptimeNanoseconds'])/1e9
        if previous>elapsed or elapsed<=0:raise ValueError('Native event outside completed caller bracket')
        if index>=3:
            total+=previous
            decode=(config['maximum']-1)/(previous-first) if config['maximum']>1 and previous>first else None
            if config['maximum']>1 and (decode is None or not math.isfinite(decode) or decode<=0):raise ValueError('Decode timing interval unavailable')
            measured.append(dict(native_ttft_seconds=first,caller_ttft_seconds=(sample['events'][0]['callbackUptimeNanoseconds']-sample['submittedUptimeNanoseconds'])/1e9,
                decode_tokens_per_second=decode,caller_completed_seconds=elapsed))
    ordered.extend(c['footprint'] for c in checkpoints[3:])
    if any(b['lifetimePeakPhysicalBytes']<a['lifetimePeakPhysicalBytes'] for a,b in zip(ordered,ordered[1:])):raise ValueError('Lifetime peak moved backwards')
    if result['warmupsCompleted']!=3 or result['measuredCompleted']!=len(measured) or len(measured)<10 or total!=result['sumNativeLastEventSeconds'] or total<config['minimumNativeSeconds']:
        raise ValueError('Completed duration/repetition scope differs')
    return dict(result=result,sample_count=len(samples),measured=measured,scope='mac-native-app-coordinator',installed_proofs=[])

