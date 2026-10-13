"""Replay clock conversion/topology and conservatively attribute GPU intervals.

MPS enclosed intervals include its internal work/scheduling. Custom counters
measure encoder processing. Transitions identify time with no attributed encoder
or MPS work in this command, not proof of global physical GPU idleness.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def topology(layers):
    projection=['bias','mps_before','mps_after','store']
    layer=['rms']+projection*3+['rotary','rotary','cache','attention']+projection+['residual','rms']+projection+['silu']+projection+['gating']+projection+['residual']
    return ['embedding']+layer*layers+['rms','bias','mps_before','mps_after','validate_float']


def merged(intervals):
    result=[]
    for begin,end in sorted(intervals):
        if end<begin:raise ValueError('negative interval')
        if result and begin<=result[-1][1]:result[-1]=(result[-1][0],max(end,result[-1][1]))
        else:result.append((begin,end))
    return result


def length(intervals):return math.fsum(end-begin for begin,end in merged(intervals))


def measure(header,events,layers):
    if [event['sample']['name'] for event in events]!=topology(layers):raise ValueError('counter topology differs')
    start,end=header['start'],header['end']
    cpu=end['cpu_nanoseconds']-start['cpu_nanoseconds'];gpu=end['gpu_ticks']-start['gpu_ticks']
    if cpu<=0 or gpu<=0:raise ValueError('invalid paired-clock calibration')
    intervals={'custom':[],'marker':[],'mps':[]};groups={}
    for i,event in enumerate(events):
        sample=event['sample'];begin,finish=sample['gpu_begin_ticks'],sample['gpu_end_ticks']
        if not start['gpu_ticks']<=begin<=finish<=end['gpu_ticks'] or begin<=0 or finish==(1<<64)-1:raise ValueError('raw timestamp invalid')
        for key,value in [('gpu_begin_seconds',begin),('gpu_end_seconds',finish)]:
            expected=(value-start['gpu_ticks'])/gpu*cpu/1e9
            if not math.isclose(sample[key],expected,rel_tol=1e-13,abs_tol=1e-12):raise ValueError('GPU clock conversion differs')
        if not 0<=sample['host_begin_seconds']<=sample['host_end_seconds'] or not all(math.isfinite(sample[key]) for key in ('gpu_begin_seconds','gpu_end_seconds','host_begin_seconds','host_end_seconds')):raise ValueError('nonfinite/invalid event')
        interval=(sample['gpu_begin_seconds'],sample['gpu_end_seconds'])
        marker=sample['name'].startswith('mps_')
        intervals['marker' if marker else 'custom'].append(interval)
        groups.setdefault(sample['name'],[]).append(interval[1]-interval[0])
        if sample['name']=='mps_before':
            following=events[i+1]['sample']
            if following['name']!='mps_after' or following['gpu_begin_seconds']<interval[1]:raise ValueError('MPS enclosure differs')
            intervals['mps'].append((interval[1],following['gpu_begin_seconds']))
    for mode in ('baseline_timing','profile_timing'):
        t=header[mode]
        if t['command_count']!=1 or not 0<t['gpu_start_seconds']<t['gpu_end_seconds'] or any(not math.isfinite(value) or value<0 for value in t.values()):raise ValueError('whole command timing invalid')
    if header['baseline_timing']['clock_calibration_seconds']!=0 or header['profile_timing']['clock_calibration_seconds']<=0:raise ValueError('profile calibration attribution differs')
    first=events[0]['sample']['gpu_begin_seconds'];last=events[-1]['sample']['gpu_end_seconds']
    all_intervals=sum(intervals.values(),[])
    union=length(all_intervals);span=last-first
    if union>span+1e-10:raise ValueError('interval union outside observed span')
    native=header['profile_timing']['gpu_end_seconds']-header['profile_timing']['gpu_start_seconds']
    if span>native+1e-6:raise ValueError('calibrated stage span exceeds completed command')
    return {'run':header['run'],'operation':header['operation'],'logits_sha256':header['logits_sha256'],'greedy_id':header['greedy_id'],
            'events':len(events),'paired_cpu_gpu_span_seconds':cpu/1e9,'observed_gpu_span_seconds':span,
            'custom_encoder_union_seconds':length(intervals['custom']),'mps_enclosed_union_seconds':length(intervals['mps']),
            'marker_union_seconds':length(intervals['marker']),'attributed_union_seconds':union,
            'unattributed_transition_seconds':max(0.,span-union),
            'overlapping_attribution_seconds':math.fsum(length(v) for v in intervals.values())-union,
            'kernel_groups_seconds':{name:math.fsum(values) for name,values in groups.items()},
            'baseline_timing':header['baseline_timing'],'profile_timing':header['profile_timing'],
            'host_clock_sample_bracket_seconds':header['host_clock_sample_bracket_seconds']}


def analyze(report):
    record=json.loads(report.read_text());raw=Path(record['raw_path'])
    with raw.open('rb') as source:
        if hashlib.file_digest(source,'sha256').hexdigest()!=record['raw_sha256']:raise ValueError('raw identity differs')
    if record.get('error') or record.get('identities_unchanged') is not True:raise ValueError('collector incomplete/identities changed')
    ready=record['ready'];layers=2 if record['kind']=='tiny' else 24
    comparisons=[]
    with raw.open() as source:
        def read():
            line=source.readline(65537)
            if not line or len(line)>65536 or not line.endswith('\n'):raise ValueError('bounded raw event invalid')
            return json.loads(line)
        if read()!=ready:raise ValueError('ready raw differs')
        for run in range(14):
            for operation in ('prefill','decode'):
                header=read()
                if header.get('event')!='profile' or header.get('run')!=run or header.get('operation')!=operation or header.get('event_count')!=ready['events_per_operation']:raise ValueError('profile order differs')
                events=[read() for _ in range(header['event_count'])]
                for i,event in enumerate(events):
                    if event.get('event')!='stage' or event.get('run')!=run or event.get('operation')!=operation or event.get('index')!=i:raise ValueError('event coverage differs')
                comparisons.append(measure(header,events,layers))
            idle=read()
            if idle.get('event')!='idle' or idle.get('run')!=run or idle['baseline_memory']!=ready['baseline_memory'] or idle['profile_memory']!=ready['profile_memory']:raise ValueError('resident allocation records differ')
        if read()!= {'event':'done'} or source.read(1):raise ValueError('raw completion differs')
    for operation in ('prefill','decode'):
        rows=[r for r in comparisons if r['operation']==operation]
        if len({r['logits_sha256'] for r in rows})!=1 or len({r['greedy_id'] for r in rows})!=1:raise ValueError('repeated profiled/reference outputs differ')
    summary={}
    for operation in ('prefill','decode'):
        rows=[r for r in comparisons if r['operation']==operation and 3<=r['run']<13]
        keys=['observed_gpu_span_seconds','custom_encoder_union_seconds','mps_enclosed_union_seconds','marker_union_seconds','unattributed_transition_seconds','overlapping_attribution_seconds']
        summary[operation]={key:statistics.median(r[key] for r in rows) for key in keys}
        for mode in ('baseline_timing','profile_timing'):
            summary[operation][mode]={key:statistics.median(r[mode][key] for r in rows) for key in rows[0][mode] if key not in ('gpu_start_seconds','gpu_end_seconds')}
            summary[operation][mode]['completed_gpu_seconds']=statistics.median(r[mode]['gpu_end_seconds']-r[mode]['gpu_start_seconds'] for r in rows)
    return {'kind':record['kind'],'raw_sha256':record['raw_sha256'],'policy':'three warmups, ten uncaptured samples, one separate captured run; profiling overhead is explicit; interval unions avoid double counting',
            'mps_caveat':'enclosed MPS intervals include internal kernels/scheduling; interencoder transitions do not establish global GPU physical idleness',
            'comparisons':comparisons,'summary':summary,'ready':ready,'idle_usage':record['idle_usage'],'capture_bytes':sum(v['bytes'] for v in record['capture_files'].values())}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.report)
    with args.output.open('x') as out:out.write(json.dumps(result,indent=2)+'\n')
