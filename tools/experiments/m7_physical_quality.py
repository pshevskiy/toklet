"""Frozen teacher-forced corpus jobs and physical stream admission.

Reuses the exact signed-install/owned-process/storage transport; no signing,
model conversion, tokenization or task grading occurs in this controller.
"""
import json
import math
from pathlib import Path
import struct
import sys
import m7_physical_measurement as measurement
ROOT=measurement.ROOT
NATIVE_SHA='06016a4e6eeea98c9c9c0f898cc4fd107c85415ed26cd3faa8de221386002971'
SCORE_PROOF='benchmarks/results/2026-10-21-m7-score-api-qualification-v1.json'
SCORE_PROOF_SHA='80fc7982f4bc9acfb65da6b3547aa9e41c0e966cd8ab8b589548d3044ecdd8c4'
CORPUS='benchmarks/contract-v1/held_out_loss.u32le'
CORPUS_SHA='f53c5b0c446959164c4d2265166040123f95be5b6d9eec784895fb45a8215beb'
POLICY='independent_blocks_of_up_to_1025_tokens_one_token_overlap_score_every_target_once'
READY_FORMAT='toklet-m7-physical-quality-ready-v1'
RESULT_FORMAT='toklet-m7-physical-quality-v1'


def corpus(model,workload):
    if model not in ('tiny-cpu','tiny-metal','small','larger') or workload not in ('short','full'):
        raise ValueError('Unsupported quality job')
    if model.startswith('tiny'):
        if workload!='short':raise ValueError('Tiny is component scope only')
        return struct.pack('<4I',1,7,13,3),{}
    with (ROOT/CORPUS).open('rb') as stream:raw=stream.read(40005)
    if len(raw)!=40004 or measurement.package.hashlib.sha256(raw).hexdigest()!=CORPUS_SHA:raise ValueError('Frozen 10001-token corpus differs')
    return (raw[:16] if workload=='short' else raw),{CORPUS:CORPUS_SHA}


def job(model,workload,verify_sources=True):
    base,host,pins=measurement.job(model,'short',verify_sources=verify_sources)
    proof=measurement.integration.metadata(ROOT/SCORE_PROOF,SCORE_PROOF_SHA)
    if proof['passed'] is not True or proof['package_build_sha256']!=NATIVE_SHA or proof['physical_device_qualified'] is not False:
        raise ValueError('Reviewed additive score package required')
    pins[SCORE_PROOF]=SCORE_PROOF_SHA
    raw,extra=corpus(model,workload);measurement.integration.merge(pins,extra)
    digest=measurement.package.hashlib.sha256(raw).hexdigest();targets=len(raw)//4-1;block=3 if workload=='short' else 1024
    config={k:base[k] for k in ('version','modelFolder','packedFolder','files','metal','numericalBudget','processBudget')}
    config.update(capacity=block+1,chunk=1,tokensFile='M7Corpora/'+digest+'/tokens.u32le',tokensSHA256=digest,targets=targets,blockSize=block)
    return config,host,pins


def stage(config,host,output,model,workload):
    base,name,digest=measurement.stage(config,host,output)
    raw,_=corpus(model,workload)
    if measurement.package.hashlib.sha256(raw).hexdigest()!=config['tokensSHA256'] or len(raw)!=(config['targets']+1)*4:
        raise ValueError('Staged corpus differs from job')
    path=base/config['tokensFile'];path.parent.mkdir(parents=True)
    with path.open('xb') as stream:stream.write(raw)
    return base,name,digest


def validate_corpus_listing(data,device,config):
    result=data['result'];parent=str(Path(config['tokensFile']).parent)
    if (data['info']['outcome']!='success' or result['deviceIdentifier']!=device or result['domain']!='appDataContainer'
        or result['domainIdentifier']!=measurement.physical.BUNDLE or len(result['files'])!=1):raise ValueError('Copied corpus listing identity differs')
    row=result['files'][0]
    if (row['name']!='tokens.u32le' or row['relativePath']!='tokens.u32le' or row['resources']['isSymbolicLink'] is not False
        or row['resources']['isDirectory'] is not False or type(row['metadata']['size']) is not int
        or row['metadata']['size']!=(config['targets']+1)*4 or parent+'/tokens.u32le'!=config['tokensFile']):
        raise ValueError('Copied corpus inventory/size differs')


def check_console(text,config,signed,tokens,vocabulary):
    n=config['targets'];block=config['blockSize'];uint=measurement.uint;number=measurement.nonnegative
    if (not uint(n,32) or not 1<=n<=10000 or not uint(block,32) or not 1<=block<=1024
        or config['capacity']!=block+1 or config['chunk']!=1 or len(tokens)!=n+1 or not uint(vocabulary,32) or vocabulary==0):
        raise ValueError('Bounded exact quality reference required')
    if any(not uint(t,32) or t>=vocabulary for t in tokens):raise ValueError('Reference token outside vocabulary')
    raw=struct.pack('<'+str(len(tokens))+'I',*tokens)
    if measurement.package.hashlib.sha256(raw).hexdigest()!=config['tokensSHA256']:raise ValueError('Exact quality reference SHA differs')
    markers=[('TOKLET_INSTALLATION_JSON=','proof'),('TOKLET_QUALITY_SAMPLE_JSON=','sample'),('TOKLET_QUALITY_JSON=','result')]
    ordered=[];proofs=[];samples=[];results=[]
    for line in text.splitlines():
        matched=False
        for prefix,kind in markers:
            if line.startswith(prefix):
                matched=True;ordered.append(kind);value=json.loads(line[len(prefix):]);{'proof':proofs,'sample':samples,'result':results}[kind].append(value);break
        if line.startswith('TOKLET_') and not matched:raise ValueError('Unexpected diagnostic/error marker')
    if ordered!=['proof']+['sample']*n+['proof','result'] or len(proofs)!=2 or len(results)!=1:
        raise ValueError('Exact physical quality stream order required')
    result=results[0]
    if ([p['phase'] for p in proofs]!=['before','after'] or any(p['simulator'] is not False or p['files']!=signed for p in proofs)
        or result['passed'] is not True or result.get('error') is not None or result['cleanupErrors'] or result['endpointErrors']
        or result['backend']!=('metal' if config['metal'] else 'cpu') or result['closed'] is not True or result['released'] is not True
        or result['fullQualityQualified'] is not False or result['physicalDeviceQualified'] is not False or result['m7Complete'] is not False
        or not uint(result['interruptionCount']) or result['interruptionCount']!=0
        or not uint(result['samplesCompleted']) or result['samplesCompleted']!=n or not number(result['loadAndSessionSeconds'])
        or result['filesBefore']!=config['files'] or result['filesAfter']!=config['files']
        or result.get('tokensBeforeSHA256')!=config['tokensSHA256'] or result.get('tokensAfterSHA256')!=config['tokensSHA256']):
        raise ValueError('Physical quality identity/proofs/cleanup differ')
    blocks=(n+block-1)//block;checkpoints=result['checkpoints']
    phases=['before-artifact-proof','after-artifact-proof','loaded']+['block-'+str(i) for i in range(blocks)]+['unloaded','closed','after-final-artifact-proof']
    if [c['phase'] for c in checkpoints]!=phases:raise ValueError('Quality memory checkpoint scope differs')
    def memory(value):
        if (not uint(value['physicalBytes']) or not uint(value['lifetimePeakPhysicalBytes'])
            or not 0<value['physicalBytes']<=value['lifetimePeakPhysicalBytes']<=config['processBudget']
            or not uint(value['returnedNaturalCount'],32) or value['returnedNaturalCount']==0):raise ValueError('Quality whole-process observation differs')
    for row in checkpoints:memory(row['footprint'])
    losses=[];memory_order=[c['footprint'] for c in checkpoints[:3]]
    for index,sample in enumerate(samples):
        row=sample['target'];score=row['score'];position=index%block+1
        if (not all(uint(row[k],32) for k in ('targetIndex','block','position','committedHistory'))
            or tuple(row[k] for k in ('targetIndex','block','position','committedHistory'))!=(index,index//block,position,position+1)
            or not all(uint(score[k],32) for k in ('target','greedy','vocabulary','evaluated'))
            or (score['target'],score['vocabulary'],score['evaluated'])!=(tokens[index+1],vocabulary,position) or score['greedy']>=vocabulary):
            raise ValueError('Teacher target/prefix/vocabulary differs')
        for key in ('targetLogit','greedyLogit'):
            value=score[key]
            if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Nonfinite quality logit')
            try:widened=struct.unpack('<f',struct.pack('<f',value))[0]
            except (OverflowError,struct.error) as error:raise ValueError('Logit outside FP32') from error
            if widened!=value:raise ValueError('Logit was not exactly widened from FP32')
        if score['greedyLogit']<score['targetLogit'] or not number(score['negativeLogLikelihood']):raise ValueError('Invalid quality loss')
        memory(sample['footprint']);memory_order.append(sample['footprint'])
        env=sample['environment']
        if (not uint(env['applicationState'],32) or env['applicationState']!=0 or type(env['lowPowerMode']) is not bool
            or not uint(env['thermalState'],32) or env['thermalState'] not in range(4)
            or not uint(env['batteryState'],32) or env['batteryState'] not in range(1,4)
            or not number(env['batteryLevel']) or env['batteryLevel']>1):raise ValueError('Foreground/power/thermal quality scope differs')
        losses.append(score['negativeLogLikelihood'])
        if position==block or index==n-1:memory_order.append(checkpoints[3+index//block]['footprint'])
    memory_order.extend(c['footprint'] for c in checkpoints[-3:])
    if any(b['lifetimePeakPhysicalBytes']<a['lifetimePeakPhysicalBytes'] for a,b in zip(memory_order,memory_order[1:])):
        raise ValueError('Quality lifetime peak moved backwards')
    summary=result['summary'];total=math.fsum(losses);mean=total/n
    if (not all(uint(summary[k],32) for k in ('targets','blocks','blockSize'))
        or (summary['targets'],summary['blocks'],summary['blockSize'],summary['policy'])!=(n,blocks,block,POLICY)
        or not number(summary['negativeLogLikelihoodSum']) or not number(summary['meanNegativeLogLikelihood'])
        or abs(summary['negativeLogLikelihoodSum']-total)>2*math.ulp(total)
        or abs(summary['meanNegativeLogLikelihood']-mean)>2*math.ulp(mean)):
        raise ValueError('Quality aggregate/policy differs')
    return dict(result=result,sample_count=n,computed_sum_nll=total,computed_mean_nll=mean,installed_proofs=proofs)


def validate_console(text,config,signed):
    parts=config['modelFolder'].split('/')
    if len(parts)!=3 or parts[0]!='M7Inputs' or parts[2]!='Model':raise ValueError('Canonical model namespace required')
    model=parts[1];workload='full' if config['targets']==10000 else 'short'
    expected,host,_=job(model,workload,verify_sources=False)
    if config!=expected:raise ValueError('Frozen physical quality config differs')
    raw,_=corpus(model,workload);tokens=list(struct.unpack('<'+str(len(raw)//4)+'I',raw))
    source=ROOT/host[config['modelFolder']+'/config.json']
    with source.open('rb') as stream:data=stream.read(512*1024+1)
    if len(data)>512*1024 or measurement.package.hashlib.sha256(data).hexdigest()!=config['files'][config['modelFolder']+'/config.json']:
        raise ValueError('Model vocabulary config differs')
    vocabulary=json.loads(data)['vocab_size']
    return check_console(text,config,signed,tokens,vocabulary)


if __name__=='__main__':
    if len(sys.argv)<9:raise SystemExit('Expected native private private_sha prepared prepared_sha model workload output and options')
    options=measurement.options(sys.argv[9:])
    sys.exit(measurement.main(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],Path(sys.argv[4]),sys.argv[5],sys.argv[6],sys.argv[7],Path(sys.argv[8]),quality=True,native_sha=NATIVE_SHA,**options))
