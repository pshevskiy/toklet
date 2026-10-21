"""Exact signed-install reuse and raw-ID measurement on the admitted iPhone.

Only frozen local artifacts are transferred to this diagnostic app's namespace.
No downloads, signing, model conversion, tokenization or other app is involved.
"""
import json
import math
import os
from pathlib import Path
import plistlib
import struct
import sys
import time
import m7_apple_package as package
import m7_apple_integration as integration
import m7_physical_integration as physical
import m7_physical_storage as storage

ROOT=package.ROOT
NATIVE_SHA='1185314043649692b7e41aff9b90e37355201eef71852965c20ef69513e6f13c'
FROZEN='benchmarks/results/2026-10-19-m6-context-reviewed-closure-v1.json'
FROZEN_SHA='9873fe34fb887a6dac2936f5ea505ee57bc2bbe7ae0da742b6e1cfc52ddb9320'
REFERENCE='benchmarks/results/2026-10-19-m7-metal-abi-qualification-v1.json'
REFERENCE_SHA='7fa4bc0b15869fa5ff445d4860932faf9f812a67204344c27e1b653ef9bbb631'


def verify_inputs(pins):
    for name,pin in pins.items():
        if package.sha(ROOT/name)!=pin:raise ValueError('Changed admitted input: '+name)


def read_prompt(path,count,expected):
    # Hash and unpack the same bounded bytes, rather than reopening after a
    # separate hash (which could admit a replaced file).
    with path.open('rb') as stream:raw=stream.read(count*4+1)
    if len(raw)!=count*4 or package.hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError('Frozen prompt bytes differ')
    return list(struct.unpack('<'+str(count)+'I',raw))


def job(model,workload,verify_sources=True):
    if model not in ('tiny-cpu','tiny-metal','small','larger') or workload not in ('short','128','512','1024','sustained'):
        raise ValueError('Unsupported explicit job')
    if model.startswith('tiny') and workload!='short':raise ValueError('Tiny is only an integration job')
    folder='M7Inputs/'+model;files={};pins={};host={};packed=''
    if model.startswith('tiny'):
        sources=dict(integration.FIXTURE);maximum=2;prompt=[1,7,13,3];expected=[47,47];numerical=128*1024**2;process=256*1024**2
    else:
        proof=integration.metadata(ROOT/FROZEN,FROZEN_SHA);reference=integration.metadata(ROOT/REFERENCE,REFERENCE_SHA)
        if proof.get('passed') is not True or reference.get('passed') is not True:raise ValueError('Reviewed prerequisites did not pass')
        pins.update({FROZEN:FROZEN_SHA,REFERENCE:REFERENCE_SHA})
        name='Qwen2.5-'+('0.5B' if model=='small' else '1.5B')+'-Instruct'
        wanted={f'models/{name}/{n}' for n in ('config.json','model.safetensors')}
        if model=='larger':
            packed='artifacts/m5-larger-q4-v2/'
            wanted.update(n for n in proof['bindings'] if n.startswith(packed))
            if len(wanted)!=199:raise ValueError('Exact larger source/packed inventory differs')
            actual={str(p.relative_to(ROOT)) for p in (ROOT/packed).iterdir() if p.is_file()}
            if actual!={n for n in wanted if n.startswith(packed)}:raise ValueError('Packed source has unexpected files')
        sources={n:proof['bindings'][n] for n in wanted}
        maximum=3 if workload=='short' else 128
        expected=([1410,1492,752] if model=='small' else [614,264,3405]) if workload=='short' else []
        prompt=[9707,11,358,498]
        if workload!='short':
            count=512 if workload=='sustained' else int(workload)
            manifest_name='benchmarks/contract-v1/manifest.json';manifest_pin=proof['bindings'][manifest_name]
            manifest=integration.metadata(ROOT/manifest_name,manifest_pin);item=manifest['arrays']['prompt_'+str(count)]
            token_name='benchmarks/contract-v1/'+item['file'];token_pin=proof['bindings'][token_name]
            if token_pin!=item['sha256'] or item['count']!=count or item['dtype']!='u32le':raise ValueError('Frozen prompt identity/shape differs')
            prompt=read_prompt(ROOT/token_name,count,token_pin);pins.update({manifest_name:manifest_pin,token_name:token_pin})
        numerical=(1536 if model=='small' else 2048)*1024**2;process=2560*1024**2
    for name,pin in sources.items():
        source=ROOT/name
        if source.is_symlink() or (verify_sources and package.sha(source)!=pin):raise ValueError('Frozen source differs: '+name)
        target=folder+('/Packed/' if packed and name.startswith(packed) else '/Model/')+source.name
        files[target]=pin;host[target]=name;pins[name]=pin
    cap=32 if workload=='short' else 1152
    config=dict(version=1,modelFolder=folder+'/Model',packedFolder=folder+'/Packed' if packed else '',files=files,
        metal=model!='tiny-cpu',capacity=cap,chunk=3 if workload=='short' else 128,
        numericalBudget=numerical,processBudget=process,prompt=prompt,maximum=maximum,expectedPrefix=expected,
        warmups=3,repetitions=10,minimumNativeSeconds=600 if workload=='sustained' else 0)
    return config,host,pins


def stage(config,host,output):
    base=output/'transfer';base.mkdir()
    for target,source in host.items():
        original=ROOT/source;destination=base/target;destination.parent.mkdir(parents=True,exist_ok=True)
        # Immutable same-filesystem hard links avoid a second multi-GiB host
        # copy. Neither this tool nor devicectl writes the source. Recheck both.
        os.link(original,destination)
        if package.sha(destination)!=config['files'][target]:raise ValueError('Staged source differs')
    raw=(json.dumps(config,sort_keys=True,separators=(',',':'))+'\n').encode()
    digest=package.hashlib.sha256(raw).hexdigest();name='M7Jobs/'+digest+'.json'
    path=base/name;path.parent.mkdir();path.write_bytes(raw)
    return base,name,digest


def validate_listing(data,device,config,base):
    r=data['result']
    if data['info']['outcome']!='success' or r['deviceIdentifier']!=device or r['domain']!='appDataContainer' or r['domainIdentifier']!=physical.BUNDLE:
        raise ValueError('Device listing identity differs')
    observed={}
    for row in r['files']:
        name=row['relativePath'];resources=row['resources']
        if row['name']!=name or resources['isSymbolicLink'] is not False or '..' in name.split('/') or name.startswith('/'):
            raise ValueError('Unexpected remote file entry')
        if resources['isDirectory'] is True:continue
        if resources['isDirectory'] is not False or base+'/'+name in observed:raise ValueError('Duplicate/special remote file')
        observed[base+'/'+name]=row['metadata']['size']
    expected={name:(ROOT/config['host'][name]).stat().st_size for name in config['files']}
    if observed!=expected:raise ValueError('Copied remote inventory/sizes differ')


def uint(value,bits=64):
    return type(value) is int and 0<=value<(1<<bits)


def nonnegative(value):
    return type(value) in (int,float) and math.isfinite(value) and value>=0


def validate_console(text,config,signed):
    def rows(prefix):return [json.loads(line[len(prefix):]) for line in text.splitlines() if line.startswith(prefix)]
    if any(line.startswith('TOKLET_MEASUREMENT_ERROR=') for line in text.splitlines()):raise ValueError('App reported measurement error')
    summaries=rows('TOKLET_MEASUREMENT_JSON=');proofs=rows('TOKLET_INSTALLATION_JSON=');samples=rows('TOKLET_MEASUREMENT_SAMPLE_JSON=')
    if len(summaries)!=1 or len(proofs)!=2 or not 13<=len(samples)<=2048:raise ValueError('Incomplete bounded measurement output')
    order=[]
    for line in text.splitlines():
        for prefix,kind in [('TOKLET_INSTALLATION_JSON=','proof'),('TOKLET_MEASUREMENT_SAMPLE_JSON=','sample'),('TOKLET_MEASUREMENT_JSON=','result')]:
            if line.startswith(prefix):order.append(kind)
    if order!=['proof']+['sample']*len(samples)+['proof','result']:
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
    if ([p['phase'] for p in proofs]!=['before','after'] or any(p['simulator'] is not False or p['files']!=signed for p in proofs)
        or result['passed'] is not True or result.get('error') is not None or result['cleanupErrors'] or result['endpointErrors']
        or result['backend']!=('metal' if config['metal'] else 'cpu') or not uint(result['interruptionCount']) or result['interruptionCount']!=0
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
            or not all(uint(token,32) for token in sample['tokens'])
            or sample['index']!=index or sample['role']!=('warmup' if index<3 else 'measured') or sample['status']!=0
            or len(sample['tokens'])!=config['maximum'] or len(sample['events'])!=config['maximum']
            or sample['tokens'][:len(config['expectedPrefix'])]!=config['expectedPrefix']
            or (sample['history'],sample['evaluated'],sample['pending'],sample['reason'])!=(len(config['prompt'])+config['maximum'],len(config['prompt'])+config['maximum']-1,1,1)
            or sample['admittedBytes']>config['numericalBudget']):raise ValueError('Sample oracle/state/admission differs')
        memory(sample['footprint']);ordered.append(sample['footprint']);environment=sample['environment']
        if not uint(environment['applicationState'],32) or environment['applicationState']!=0 or type(environment['lowPowerMode']) is not bool or not uint(environment['thermalState'],32) or environment['thermalState'] not in (0,1,2,3):raise ValueError('Foreground/device environment differs')
        if 'batteryState' in environment and (not uint(environment['batteryState'],32) or environment['batteryState']>3):raise ValueError('Battery state differs')
        if 'batteryLevel' in environment and (not nonnegative(environment['batteryLevel']) or environment['batteryLevel']>1):raise ValueError('Battery level differs')
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
    return dict(result=result,sample_count=len(samples),measured=measured,installed_proofs=proofs)


def admit_ready(folder,expected,config,inputs,prepared,prepared_sha,signed):
    record=integration.metadata(folder/'measurement.json',expected)
    if (record['format']!='toklet-m7-physical-measurement-ready-v1' or record['passed'] is not True
        or record['checked'] is not None or record['config']!=config or record['signed_bundle']!=signed
        or record['prepared_receipt']!=dict(path=str((prepared/'integration.json').relative_to(ROOT)),sha256=prepared_sha)
        or record['primary_error'] is not None or record['endpoint_errors'] or record['cleanup_errors']
        or len(record['remote_cleanup'])!=2 or any(not r['absent'] or r['errors'] for r in record['remote_cleanup'])):
        raise ValueError('Exact successful transfer-only receipt required')
    for name,pin in inputs.items():
        if record['bindings'].get(name)!=pin:raise ValueError('Ready frozen input differs')
    required={str((folder/'private-logs'/name).relative_to(ROOT)) for name in ['profile.stdout','entitlements.stdout','signature-details.stderr']}
    if not required.issubset(record['private_evidence_pins']):raise ValueError('Cached signing observations missing')
    verify_inputs(record['bindings']);verify_inputs(record['private_evidence_pins'])
    receipt=record['job_receipt'];digest=receipt['sha256']
    if (type(digest) is not str or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest)
        or receipt['relative_path']!='M7Jobs/'+digest+'.json'):
        raise ValueError('Ready job identity differs')
    path=folder/'transfer'/receipt['relative_path']
    if integration.metadata(path,digest)!=config:raise ValueError('Ready job bytes differ')
    pins=dict(record['bindings']);integration.merge(pins,record['private_evidence_pins'])
    integration.merge(pins,{str((folder/'measurement.json').relative_to(ROOT)):expected})
    return pins,receipt['relative_path'],digest


def main(native,private,private_sha,prepared,prepared_sha,model,workload,output,prepare_only=False,transfer=None,transfer_sha=None,storage_proof=None,storage_sha=None):
    native=native.resolve();private=private.resolve();prepared=prepared.resolve();output=output.resolve()
    transfer=transfer.resolve() if transfer else None
    storage_proof=storage_proof.resolve() if storage_proof else None
    if prepare_only and transfer:raise ValueError('Preparation and transfer reuse are separate')
    if storage_proof and (transfer or model.startswith('tiny')):raise ValueError('Storage proof is only for new real-model transfers')
    if any(not p.is_relative_to(ROOT/'artifacts') for p in (native,private,prepared,output,*([transfer] if transfer else []),*([storage_proof] if storage_proof else []))):raise ValueError('Repository artifact paths required')
    output.mkdir();logs=output/'private-logs';logs.mkdir();env,_=package.build_environment();env['TMPDIR']=str(output)
    pins={};commands=[];config={};remote=None;app=None;signed={};checked=None;primary=None;endpoints=[];cleanup=[];cleanup_errors=[];counter=0
    selected={};device=None;profile=None;remote_admitted=False;device_observed=False;name=None;digest=None;transfer_verified=False
    host_boot=None;host_start=None;storage_admission=None;observed_storage=None
    previous=package.signal.signal(package.signal.SIGTERM,package.interrupted)
    def execute(args,label,timeout=90,admit=True):
        if admit:verify_inputs(pins)
        # This must be the last admission work before the actual copy child.
        # In particular, no subsequent full-input hashing can age the proof.
        if label.startswith('copy-') and storage_admission is not None:
            storage_admission['checks'].append(storage.fresh(storage_admission,host_boot,time.time_ns(),time.monotonic_ns(),label))
        row=package.owned(args,logs/(label+'.stdout'),logs/(label+'.stderr'),env,timeout);commands.append(row)
        if row['primary_error'] or row['cleanup_errors']:raise RuntimeError(label+': '+str(row['primary_error'])+' '+repr(row['cleanup_errors']))
        return ROOT/row['stdout']
    def device_command(parts,label,timeout=90,admit=True):
        return execute([physical.DEVICECTL,'device',*parts,'--device',selected['device_identifier'],'--timeout',str(timeout),'--quiet','--json-output','-'],label,timeout+20,admit)
    def query():
        nonlocal counter
        counter+=1;return json.loads(package.small(device_command(['info','processes','--search','TokletHarness'],'process-'+str(counter),admit=False)))
    def signal(pid,sig):
        nonlocal counter
        counter+=1;device_command(['process','signal','--pid',str(pid),'--signal',sig],'signal-'+str(counter),admit=False)
    def current():return json.loads(package.small(device_command(['info','apps','--bundle-id',physical.BUNDLE],'current-install-'+str(len(commands)))))
    def clean():
        r=physical.stop_remote(query,signal,remote);cleanup.append(r);cleanup_errors.extend(r['errors'])
        if not r['absent']:cleanup_errors.append('Remote process absence unverified')
    try:
        selected=physical.selection(private,private_sha)
        integration.merge(pins,integration.admit_build(native,NATIVE_SHA))
        app,reused,remote,signed=physical.admit_reuse(prepared,prepared_sha,selected,NATIVE_SHA);integration.merge(pins,reused)
        modules=[package,integration,physical,storage]
        paths=[Path(m.__file__).resolve() for m in modules]+[Path(__file__).resolve(),Path(__file__).with_name('test_m7_physical_measurement.py'),Path(__file__).with_name('test_m7_physical_storage.py')]
        integration.merge(pins,{str(p.relative_to(ROOT)):package.sha(p) for p in paths})
        integration.merge(pins,{str(private.relative_to(ROOT)):private_sha})
        common_pins=dict(pins)
        config,host,inputs=job(model,workload,verify_sources=not transfer);integration.merge(pins,inputs)
        real=model in ('small','larger')
        if real and not transfer and not storage_proof:raise ValueError('Real-model copy requires a fresh same-device storage proof')
        context=None
        if real:
            tiny,_,tiny_inputs=job('tiny-cpu','short',verify_sources=False)
            required=dict(common_pins);integration.merge(required,tiny_inputs)
            context=dict(tiny_config=tiny,prepared_receipt=dict(path=str((prepared/'integration.json').relative_to(ROOT)),sha256=prepared_sha),
                         signed_bundle=signed,selection_path=str(private.relative_to(ROOT)),selection_sha=private_sha,selected=selected,remote=remote,required_bindings=required)
        if transfer:
            ready,name,digest=admit_ready(transfer,transfer_sha,config,inputs,prepared,prepared_sha,signed)
            integration.merge(pins,ready);transfer_verified=True
            if real:
                record=integration.metadata(transfer/'measurement.json',transfer_sha)
                storage_admission,proof_pins=storage.admit_ready(record,transfer/'transfer',config,name,digest,context,validate_console)
                integration.merge(pins,proof_pins)
        else:
            base,name,digest=stage(config,host,output)
            staged={str(p.relative_to(ROOT)):package.sha(p) for p in base.rglob('*') if p.is_file()};integration.merge(pins,staged)
            if real:
                observed_storage,proof_pins=storage.admit_proof(storage_proof,storage_sha,context,validate_console)
                integration.merge(pins,proof_pins)
                payload,total=storage.inventory(base,config,name,digest)
                storage_admission=storage.decision(observed_storage,storage_proof,storage_sha,payload,total)
        if not prepare_only or real:
            host_boot=storage.boot_token(package.small(execute(['/usr/sbin/sysctl','-n','kern.bootsessionuuid'],'host-boot')))
        if real and not transfer:
            storage.fresh(storage_admission,host_boot,time.time_ns(),time.monotonic_ns(),'initial-admission')
        device=physical.admit_device(json.loads(package.small(device_command(['info','details'],'device-before'))),selected)
        device_observed=True
        if observed_storage and device!=observed_storage['device']:raise ValueError('Storage observation device details differ')
        if transfer:
            # Immutable signed files and all cached signing observations are
            # hash-checked above. Re-evaluate team/device/app/expiry now without
            # repeating signing tool launches after the owner's unlock.
            cms=transfer/'private-logs/profile.stdout';ent=transfer/'private-logs/entitlements.stdout'
            signature=package.small(transfer/'private-logs/signature-details.stderr')
            profile=physical.admit_profile(plistlib.loads(package.small(cms).encode()),plistlib.loads(package.small(ent).encode()),signature,selected)
        else:
            execute(['codesign','--verify','--strict',str(app)],'signature-verify')
            cms=execute(['security','cms','-D','-i',str(app/'embedded.mobileprovision')],'profile')
            ent=execute(['codesign','-d','--entitlements',':-',str(app)],'entitlements')
            execute(['codesign','-dvv',str(app)],'signature-details')
            profile=physical.admit_profile(plistlib.loads(package.small(cms).encode()),plistlib.loads(package.small(ent).encode()),package.small(logs/'signature-details.stderr'),selected)
        physical.launch_verified(current,lambda:True,remote,selected)
        remote_admitted=True
        # The user's M7 task authorizes resetting this exact verified diagnostic
        # installation. Never stop another path or a stale remembered PID.
        clean()
        if cleanup_errors:raise ValueError('Preflight diagnostic cleanup failed')
        if not transfer:
            parent='M7Inputs/'+model
            for kind in ('Model','Packed') if config['packedFolder'] else ('Model',):
                physical.launch_verified(current,lambda:device_command(['copy','to','--source',str(base/parent/kind),'--destination','Documents/'+parent+'/'+kind,
                    '--domain-type','appDataContainer','--domain-identifier',physical.BUNDLE],'copy-'+kind,600),remote,selected)
            listed=json.loads(package.small(device_command(['info','files','--subdirectory','Documents/'+parent,'--recurse','--domain-type','appDataContainer','--domain-identifier',physical.BUNDLE],'copied-inventory')))
            validate_listing(listed,selected['device_identifier'],dict(config,host=host),parent)
            physical.launch_verified(current,lambda:device_command(['copy','to','--source',str(base/'M7Jobs'),'--destination','Documents/M7Jobs','--domain-type','appDataContainer','--domain-identifier',physical.BUNDLE],'copy-job',90),remote,selected)
            transfer_verified=True
        if not prepare_only:
            metadata=logs/'launch.json'
            args=[physical.DEVICECTL,'device','process','launch','--device',selected['device_identifier'],'--terminate-existing','--console','--timeout','2400','--json-output',str(metadata),physical.BUNDLE,'--','--measure',name,digest]
            host_start=dict(wallUnixNanoseconds=time.time_ns(),monotonicNanoseconds=time.monotonic_ns())
            raw=physical.launch_verified(current,lambda:execute(args,'measurement',2430),remote,selected)
            launch=physical.check_launch(json.loads(package.small(metadata)),remote,selected)
            # The admitted representative jobs are well below this bounded capture.
            with raw.open('rb') as f:stdout=f.read(64*1024**2+1)
            with (logs/'measurement.stderr').open('rb') as f:stderr=f.read(64*1024**2+1)
            if len(stdout)>64*1024**2 or len(stderr)>64*1024**2:raise ValueError('Measurement console exceeds capture limit')
            checked=validate_console(stdout.decode()+'\n'+stderr.decode(),config,signed);checked['launch']=launch
    except BaseException as error:primary=type(error).__name__+': '+str(error)
    finally:
        if remote and remote_admitted:
            try:clean()
            except BaseException as error:cleanup_errors.append(type(error).__name__+': '+str(error))
        if selected and device_observed:
            try:physical.admit_device(json.loads(package.small(device_command(['info','details'],'device-after',admit=False))),selected)
            except BaseException as error:endpoints.append('Device endpoint: '+str(error))
        try:verify_inputs(pins)
        except BaseException as error:endpoints.append(str(error))
        if app:
            try:
                if physical.inventory(app)!=signed:endpoints.append('Signed source endpoint differs')
            except BaseException as error:endpoints.append(str(error))
        package.signal.signal(package.signal.SIGTERM,previous)
    raw_record=dict(commands=commands,primary_error=primary,endpoint_errors=endpoints,cleanup_errors=cleanup_errors)
    private_record=output/'private-execution.json';private_record.write_text(json.dumps(raw_record,indent=2)+'\n')
    private_pins={str(p.relative_to(ROOT)):package.sha(p) for p in [private_record,*sorted(logs.iterdir())] if p.is_file()}
    complete=(transfer_verified and checked is None and profile is not None) if prepare_only else checked is not None
    passed=primary is None and not endpoints and not cleanup_errors and complete and len(cleanup)==2
    record=dict(format='toklet-m7-physical-measurement-ready-v1' if prepare_only else 'toklet-m7-physical-measurement-v1',passed=passed,bindings=pins,device=device,signing=profile,config=config,
        transfer_receipt=dict(path=str((transfer/'measurement.json').relative_to(ROOT)),sha256=transfer_sha) if transfer else None,
        prepared_receipt=dict(path=str((prepared/'integration.json').relative_to(ROOT)),sha256=prepared_sha),signed_bundle=signed,
        host_boot_token=host_boot,measurement_host_start=host_start,storage_admission=storage_admission,
        job_receipt=dict(relative_path=name,sha256=digest),checked=checked,commands=commands,controlled_environment=env,remote_cleanup=cleanup,private_evidence_pins=private_pins,
        primary_error=primary,endpoint_errors=endpoints,cleanup_errors=cleanup_errors,physical_device_qualified=False,m7_complete=False)
    record=physical.redact_record(record,selected,remote);(output/'measurement.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(passed=passed,primary_error=record['primary_error'],endpoint_errors=record['endpoint_errors'],cleanup_errors=record['cleanup_errors'])))
    return 0 if passed else 1


def options(extra):
    result={};seen=set()
    while extra:
        flag=extra[0];extra=extra[1:]
        if flag in seen:raise ValueError('Duplicate diagnostic option')
        seen.add(flag)
        if flag=='--prepare-only':result['prepare_only']=True
        elif flag in ('--reuse-transfer','--storage-proof'):
            if len(extra)<2:raise ValueError('Expected evidence folder and SHA')
            name='transfer' if flag=='--reuse-transfer' else 'storage_proof'
            result[name]=Path(extra[0]);result['transfer_sha' if name=='transfer' else 'storage_sha']=extra[1];extra=extra[2:]
        else:raise ValueError('Unknown diagnostic option')
    if result.get('prepare_only') and result.get('transfer'):raise ValueError('Preparation and reuse conflict')
    if result.get('transfer') and result.get('storage_proof'):raise ValueError('Reuse does not copy new files')
    return result


if __name__=='__main__':
    sys.exit(main(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],Path(sys.argv[4]),sys.argv[5],sys.argv[6],sys.argv[7],Path(sys.argv[8]),
                  **options(sys.argv[9:])))
