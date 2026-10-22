"""Owned selected SmolLM Mac app checks; fixed raw-ID probes, no broad quality claim."""
from pathlib import Path
import argparse,json,sys,signal,struct,math,statistics,os
import m7_apple_package as p
import m7_apple_integration as integration
import m8_mlx_jobs as jobs
import m8_mac_measurement as measurement
import m7_physical_quality as quality
import m7_physical_tasks as tasks
import m7_native_state_check as state
R=p.ROOT
BUILD='artifacts/m8-smollm-mac-runner-v1'
BUILD_SHA='5b7b1e503ea77745721f3c3068c31a37759b42949fdc0327d03bc22f9a2d57ce'
NATIVE_SHA='10f6256e1a4aefa90c266b621ed9bfd84d51cf7f1f4277854b0ecec92fdc8a87'
ARRAYS='artifacts/m8-smollm-reference-v1/arrays.bin'
ARRAY_SHA='4f471fe2f15c257c8fbf4169a88b99b7ba28fa1e9bf881eaa91f77a7f8cdcca2'
PROTECTED={'target/release/toklet':'5250694e814ea112148a52cfdeaf5bc6286ff2804b3acfd8f89e80bc73122a59','target/release/examples/cpu_bench':'e3debabbae7db9e5801bc9bfba177275444ebe1d9849eb3ad0f2dff487ac8dea'}
WORKLOADS=['measurement-short','quality-short','tasks-short','state-component']


def capture(path,pin,maximum):
    with path.open('rb') as f:raw=f.read(maximum+1)
    if len(raw)>maximum or p.hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('Consumed input differs: '+str(path))
    return raw


def merge(pins,extra):integration.merge(pins,extra)


def admit_build(raw):
    build=json.loads(raw)
    if (build['format']!='toklet-smollm-mac-runner-build-v1' or build['passed'] is not True
        or build['primary_error'] or build['endpoint_errors'] or build['native_package_sha256']!=NATIVE_SHA
        or len(build['commands'])!=4 or any(row['exit_code']!=0 or row['primary_error'] or row['cleanup_errors'] for row in build['commands'])):
        raise ValueError('Successful exact SmolLM Mac build required')
    pins={};merge(pins,build['bindings']);merge(pins,build['binary_pins']);return build,pins


def preflight(output):
    if os.path.lexists(output):raise FileExistsError('Never overwrite an app trial')
    output=output.resolve()
    if not output.is_relative_to(R/'artifacts'):raise ValueError('Repository output required')
    # Exclusive creation of the complete root excludes all child collisions.
    output.mkdir();return output


def diagnostics(row,maximum=32*1024**2):
    buffers={};captures=[];errors=[]
    for key in ['stdout','stderr']:
        try:
            with (R/row[key]).open('rb') as f:raw=f.read(maximum+1)
            if len(raw)>maximum:raise ValueError('Diagnostic exceeds bound')
            captures.append(dict(path=row[key],bytes=len(raw),sha256=p.hashlib.sha256(raw).hexdigest()));buffers[key]=raw
        except BaseException as error:errors.append(dict(path=row[key],error=type(error).__name__+': '+str(error)))
    return buffers,captures,errors


def strict_state(text):
    lines=[line for line in text.splitlines() if line.strip()]
    if len(lines)!=1 or not lines[0].startswith('TOKLET_STATE_JSON='):raise ValueError('One exact state record required')
    return json.loads(lines[0][len('TOKLET_STATE_JSON='):])


def rows(text,prefix):return [json.loads(line[len(prefix):]) for line in text.splitlines() if line.startswith(prefix)]
def finite(value):return type(value) in [int,float] and math.isfinite(value)
def lse(values):
 maximum=max(values);return maximum+math.log(math.fsum(math.exp(v-maximum) for v in values))
def qualify_score(manifest,arrays,index,score):
 name='scoring.prefix_'+str(index+1)+'.logits';item=manifest['arrays'][name];raw=arrays[item['offset']:item['offset']+item['bytes']]
 assert p.hashlib.sha256(raw).hexdigest()==item['sha256'] and item['dtype']=='<f4'
 logits=struct.unpack('<49152f',raw);gate=item['gate'];target=manifest['greedy']['prompt'][index+1]
 assert score['target']==target and score['vocabulary']==49152 and score['evaluated']==index+1
 greedy=max(range(len(logits)),key=logits.__getitem__);assert score['greedy']==greedy
 for field,token in [('targetLogit',target),('greedyLogit',greedy)]:
  assert finite(score[field]) and abs(score[field]-logits[token])<=gate['atol']+gate['rtol']*abs(logits[token])
 radii=[gate['atol']+gate['rtol']*abs(v) for v in logits]
 lower=[v-r for v,r in zip(logits,radii)];upper=[v+r for v,r in zip(logits,radii)]
 low=lse(lower)-upper[target];high=lse(upper)-lower[target]
 assert finite(score['negativeLogLikelihood']) and low<=score['negativeLogLikelihood']<=high
 return dict(target=target,greedy=greedy,reference_nll=lse(logits)-logits[target],actual_nll=score['negativeLogLikelihood'],derived_frozen_logit_nll_interval=[low,high])
def footprint(result):
 values=[]
 def visit(value):
  if isinstance(value,dict):
   if 'lifetimePeakPhysicalBytes' in value:values.append(value['lifetimePeakPhysicalBytes'])
   for child in value.values():visit(child)
  elif isinstance(value,list):
   for child in value:visit(child)
 visit(result);assert values;return max(values)


def main(output):
    O=preflight(output);commands=[];captured=[];capture_errors=[];checked={};stages={};pins={};primary=None;endpoints=[]
    env,_=p.build_environment();previous=signal.signal(signal.SIGTERM,p.interrupted)
    def guard():
        for name,pin in pins.items():
            if p.sha(R/name)!=pin:raise ValueError('App input changed: '+name)
    try:
        build_raw=capture(R/BUILD/'build.json',BUILD_SHA,16*1024**2);build,admitted=admit_build(build_raw)
        merge(pins,admitted);merge(pins,{BUILD+'/build.json':BUILD_SHA});merge(pins,PROTECTED)
        paths=[Path(__file__).resolve(),*sorted((R/'tools/experiments').glob('*.py'))]
        merge(pins,{str(f.relative_to(R)):p.sha(f) for f in paths})
        manifest_raw=capture(R/jobs.SMOLLM_REFERENCE,jobs.SMOLLM_REFERENCE_SHA,256*1024)
        arrays=capture(R/ARRAYS,ARRAY_SHA,13_945_632)
        manifest=json.loads(manifest_raw);merge(pins,{jobs.SMOLLM_REFERENCE:jobs.SMOLLM_REFERENCE_SHA,ARRAYS:ARRAY_SHA})
        for workload in WORKLOADS:
            guard();folder=O/workload;record=jobs.stage('smollm',workload,folder)
            stage_raw=capture(folder/'staged.json',p.sha(folder/'staged.json'),1024*1024)
            if json.loads(stage_raw)!=record:raise ValueError('Published stage differs')
            stages[workload]=dict(path=str((folder/'staged.json').relative_to(R)),sha256=p.hashlib.sha256(stage_raw).hexdigest())
            merge(pins,record['bindings']);merge(pins,{stages[workload]['path']:stages[workload]['sha256']})
            base=folder/'transfer';job=record['job'];config=record['config'];mode=record['mode']
            if record['execution']!=jobs.SMOLLM_EXECUTION:raise ValueError('Reference execution identity differs')
            merge(pins,{str((base/name).relative_to(R)):item['sha256'] for name,item in record['staged_files'].items()})
            guard();command=[str(R/BUILD/'mlx-real'),mode,str(base),job['relative_path'],job['sha256']]
            row=p.owned(command,O/(workload+'.stdout'),O/(workload+'.stderr'),env,900);commands.append(row)
            buffers,logs,errors=diagnostics(row);captured.extend(logs);capture_errors.extend(errors)
            merge(pins,{item['path']:item['sha256'] for item in logs})
            if row['primary_error'] or row['cleanup_errors']:raise RuntimeError('Owned app failure: '+str(row['primary_error'])+' '+repr(row['cleanup_errors']))
            if errors:raise ValueError('App diagnostic capture failed')
            guard()
            if buffers['stderr'].strip():raise ValueError('Unexpected app stderr')
            text=buffers['stdout'].decode();extra={}
            if mode=='measurement':
                result=measurement.validate_console(text,config,49152)['result'];samples=rows(text,'TOKLET_MEASUREMENT_SAMPLE_JSON=')[3:]
                ttft=[r['events'][0]['nativeElapsedSeconds'] for r in samples]
                decode=[(r['events'][-1]['nativeElapsedSeconds']-r['events'][0]['nativeElapsedSeconds'])/3 for r in samples]
                extra=dict(measured_samples=10,median_native_first_token_seconds=statistics.median(ttft),median_native_decode_token_seconds=statistics.median(decode),raw_native_first_token_seconds=ttft,raw_native_decode_token_seconds=decode)
            elif mode=='quality':
                raw=capture(base/config['tokensFile'],config['tokensSHA256'],16);tokens=list(struct.unpack('<4I',raw))
                result=quality.check_console(text,config,None,tokens,49152,require_installation=False)['result']
                extra['independent_scoring']=[qualify_score(manifest,arrays,i,sample['target']['score']) for i,sample in enumerate(rows(text,'TOKLET_QUALITY_SAMPLE_JSON='))]
            elif mode=='tasks':
                data=json.loads(capture(base/config['tasksFile'],config['tasksSHA256'],512*1024))
                admitted=tasks.check_console(text,config,None,data,require_installation=False);result=admitted['result'];expected=manifest['greedy']['fp64_tokens']
                for actual,source in zip(admitted['responses'],data['cases']):
                    if actual['response']['ids']!=expected[:source['maximum']]:raise ValueError('Frozen task probe IDs differ')
                extra['responses']=admitted['responses']
            else:
                result=strict_state(text);state.validate_result(result,'metal-mlx','component',config)
            peak=footprint(result)
            if peak>config['processBudget']:raise ValueError('App process cap exceeded')
            checked[workload]=dict(result=result,peak_physical_bytes=peak,**extra)
            print(json.dumps(dict(completed=workload,peak_physical_bytes=peak)),flush=True)
    except BaseException as error:primary=type(error).__name__+': '+str(error)
    finally:
        signal.signal(signal.SIGTERM,previous)
        try:guard()
        except BaseException as error:endpoints.append(str(error))
    record=dict(format='toklet-smollm-mac-app-v1',passed=primary is None and not endpoints and not capture_errors and len(checked)==4,
        bindings=pins,protected_binary_pins=PROTECTED,commands=commands,captured_diagnostics=captured,capture_errors=capture_errors,stages=stages,checked=checked,primary_error=primary,endpoint_errors=endpoints,
        full_quality_qualified=False,physical_device_qualified=False,comparative_performance_qualified=False)
    with (O/'execution.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=record['passed'],completed=list(checked),primary_error=primary,endpoint_errors=endpoints)),flush=True)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    sys.exit(0 if main(args.output)['passed'] else 1)
