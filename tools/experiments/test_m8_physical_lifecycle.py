"""Mocked native MLX scene controller; never launches a process or device."""
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import plistlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import m8_physical_lifecycle as p
from test_m7_physical_lifecycle import fixture as legacy_fixture,text


def fixture(returned=True):
    old,r=legacy_fixture(returned);c=p.phone.jobs.job('qwen8','lifecycle-512')[1]
    delta=len(c['prompt'])-len(old['prompt'])
    r['backend']='metal-mlx';r['filesBefore']=copy.deepcopy(c['files']);r['filesAfter']=copy.deepcopy(c['files'])
    r['interruptedPrefix']=c['prompt']+r['rebuiltIDs']
    for event in r['trace']['events']:
        for state in [event['before'],event['state']]:
            saved=state.get('savedPrefix')
            if saved:
                if saved[:len(old['prompt'])]!=old['prompt']:raise AssertionError('Authored fixture prefix differs')
                state['savedPrefix']=c['prompt']+saved[len(old['prompt']):];state['savedHistory']=len(state['savedPrefix'])
                if state['history']:state['history']+=delta;state['evaluated']+=delta
    fields=[('inactive',2),('background',3),('active',5 if returned else 4)]
    if returned:fields.append(('returnInactive',4))
    for name,index in fields:r['roundTrip'][name]=copy.deepcopy(r['trace']['events'][index])
    return c,r


class Validation(unittest.TestCase):
    def test_real_mlx_schema_and_same_physical_event_contract(self):
        for returned in [False,True]:
            c,r=fixture(returned);self.assertEqual(p.validate_console(text(r),c,{})['result'],r)
    def test_mlx_cannot_claim_legacy_backend_or_config(self):
        for field,value in [('version',1),('mlx',1),('capacity',32),('prompt',[1]*512)]:
            c,r=fixture();c[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):p.validate_console(text(r),c,{})
        c,r=fixture();r['backend']='metal'
        with self.assertRaises(ValueError):p.validate_console(text(r),c,{})
    def test_scene_origin_replay_prefix_and_close_fail_closed(self):
        for mutation in ['origin','prefix','reference','close','claim']:
            c,r=fixture()
            if mutation=='origin':r['trace']['events'][2]['origin']='syntheticCheck';r['roundTrip']['inactive']=copy.deepcopy(r['trace']['events'][2])
            elif mutation=='prefix':r['interruptedPrefix'][0]=0
            elif mutation=='reference':r['referenceIDs']=[46]*3
            elif mutation=='close':r['trace']['events'][-1]['state']['workerFinished']=False
            else:r['actualOSLifecycleQualified']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):p.validate_console(text(r),c,{})


class Ready(unittest.TestCase):
    def invoke(self,mutation=None):
        with tempfile.TemporaryDirectory(dir=p.ROOT/'artifacts') as directory:
            root=Path(directory);folder=root/'ready';folder.mkdir();staged=root/'stage';staged.mkdir();proof_folder=root/'proof';proof_folder.mkdir()
            c,_=fixture();signed={'TokletHarness':'b'*64};prepared=dict(path='artifacts/authored/integration.json',sha256='c'*64)
            context=dict(prepared_receipt=prepared,signed_bundle=signed,required_bindings={'authored-input':'d'*64},remote='/installed/TokletHarness.app',selected={})
            proof=dict(device={'product_type':'iPhone15,2'},checked=dict(result=dict(filesystemBefore=dict(freeBytes=p.phone.storage.RESERVE+100))),
                measurement_host_start=dict(wallUnixNanoseconds=100,monotonicNanoseconds=100),host_boot_token='a'*64)
            payload={'authored':dict(bytes=100,sha256='e'*64)};admission=p.phone.storage.decision(proof,proof_folder,'f'*64,payload,100)
            admission['checks']=[p.phone.storage.fresh(admission,'a'*64,200,200,label) for label in ['copy-Model','copy-job']]
            natural=dict(absent=True,observations=[[]],signals=[],errors=[])
            saved=dict(format=p.phone.READY_FORMAT,passed=True,model='qwen8',workload='lifecycle-512',mode='lifecycle',checked=None,config=c,
                native_package_sha256=p.phone.NATIVE_SHA,prepared_receipt=prepared,signed_bundle=signed,
                staged_receipt=dict(path=str((staged/'staged.json').relative_to(p.ROOT)),sha256='b'*64),primary_error=None,endpoint_errors=[],cleanup_errors=[],
                full_quality_qualified=False,physical_device_qualified=False,performance_qualified=False,remote_cleanup=[natural,natural],
                bindings=dict(context['required_bindings']),private_evidence_pins={},device=proof['device'],storage_admission=admission)
            boot='a'*64
            if mutation=='mode':saved['mode']='state'
            elif mutation=='typedConfig':saved['config']=dict(c,version=2.)
            elif mutation=='context':saved['bindings']={}
            elif mutation=='proofPath':saved['storage_admission']['proof']['path']='/outside/measurement.json'
            elif mutation=='stale':saved['storage_admission']['checks'][0]['wallUnixNanoseconds']=p.phone.storage.MAX_AGE_NS+100
            elif mutation=='label':saved['storage_admission']['checks'][0]['label']='other'
            elif mutation=='checks':saved['storage_admission']['checks'].pop()
            elif mutation=='boot':boot='f'*64
            elif mutation=='device':saved['device']={'product_type':'other'}
            elif mutation=='errorType':saved['primary_error']=0
            elif mutation=='absenceType':saved['remote_cleanup']=[dict(natural,absent=1),natural]
            path=folder/'measurement.json';path.write_text(json.dumps(saved));pin=hashlib.sha256(path.read_bytes()).hexdigest()
            record=dict(config=c)
            with patch.object(p.phone.measurement,'verify_inputs'),patch.object(p.phone,'admit_storage',return_value=(proof,{})):
                actual,pins,observed=p.admit_ready(folder,pin,staged,'b'*64,record,payload,100,context,boot)
            self.assertEqual(actual,saved);self.assertEqual(observed,admission)
            self.assertEqual(pins[str(path.relative_to(p.ROOT))],pin)
    def test_exact_ready_and_original_copy_freshness(self):self.invoke()
    def test_invalid_ready_identity_types_lineage_and_storage_reject(self):
        for mutation in ['mode','typedConfig','context','proofPath','stale','label','checks','boot','device','errorType','absenceType']:
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.invoke(mutation)


class Composition(unittest.TestCase):
    def test_standalone_generic_preparation_is_admitted_by_later_lifecycle_owner(self):
        # Run the real generic producer and storage replay with mocked transport
        # and artifact admission. Import the roundtrip owner only AFTER producing
        # the tiny proof and lifecycle READY, as independent CLI calls do.
        from test_m8_physical_mlx import Transport
        transport=Transport();transport.setUp()
        try:
            removed={name:sys.modules.pop(name) for name in ['m8_physical_lifecycle','m7_physical_lifecycle']}
            try:
                _,tiny_pin=transport.staged();tiny=transport.stage
                ready=transport.root/'tiny-ready'
                self.assertTrue(p.phone.run(tiny,tiny_pin,ready,prepare_only=True)['passed'])
                proof=transport.root/'tiny-proof'
                observed=p.phone.run(tiny,tiny_pin,proof,ready=ready,ready_sha=p.phone.package.sha(ready/'measurement.json'))
                self.assertTrue(observed['passed'])
                c,_=fixture();stage=transport.root/'lifecycle-stage';base=stage/'transfer';base.mkdir(parents=True)
                host={};payload={}
                for name in c['files']:
                    path=base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'authored-artifact-boundary')
                    host[name]=str(path.relative_to(p.ROOT));payload[name]=dict(bytes=path.stat().st_size,sha256=p.phone.package.sha(path),source=host[name])
                raw=p.phone.jobs.encoded(c);job_pin=hashlib.sha256(raw).hexdigest();job='M7Jobs/'+job_pin+'.json'
                path=base/job;path.parent.mkdir();path.write_bytes(raw)
                payload[job]=dict(bytes=len(raw),sha256=job_pin,source=str(path.relative_to(p.ROOT)))
                record=dict(model='qwen8',workload='lifecycle-512',mode='lifecycle',config=c,host=host,job=dict(relative_path=job,sha256=job_pin))
                (stage/'staged.json').write_text(json.dumps(record));stage_pin=p.phone.package.sha(stage/'staged.json')
                total=sum(item['bytes'] for item in payload.values());transport.stage=stage;transport.config=c
                ready=transport.root/'lifecycle-ready'
                with patch.object(p.phone,'admitted_stage',return_value=(record,{},payload,total)):
                    saved=p.phone.run(stage,stage_pin,ready,prepare_only=True,proof=proof,proof_sha=p.phone.package.sha(proof/'measurement.json'))
                self.assertTrue(saved['passed'],saved['primary_error'])
            finally:sys.modules.update(removed)
            caller_paths={'tools/experiments/'+name+'.py' for name in removed}
            self.assertTrue(caller_paths.isdisjoint(saved['bindings']))
            common={str(transport.selection.relative_to(p.ROOT)):p.phone.SELECTION_SHA,
                    str((transport.prepared/'integration.json').relative_to(p.ROOT)):transport.prepared_sha,
                    **json.loads((transport.prepared/'integration.json').read_text())['private_evidence_pins'],
                    **p.phone.shared_helper_bindings()}
            self.assertTrue(caller_paths.isdisjoint(common))
            context=dict(prepared_receipt=saved['prepared_receipt'],signed_bundle=transport.signed,required_bindings=common,remote=transport.remote,selected=transport.selected)
            actual,pins,admission=p.admit_ready(ready,p.phone.package.sha(ready/'measurement.json'),stage,stage_pin,record,payload,total,context,observed['host_boot_token'])
            self.assertEqual(actual,saved);self.assertEqual(admission,saved['storage_admission'])
            self.assertIn(str((proof/'private-logs/measurement.stdout').relative_to(p.ROOT)),pins)
            self.assertEqual([row['label'] for row in admission['checks']],['copy-Model','copy-job'])
        finally:transport.doCleanups()


class Controller(unittest.TestCase):
    def invoke(self,fail=None):
        with tempfile.TemporaryDirectory(dir=p.ROOT/'artifacts') as directory,ExitStack() as stack:
            root=Path(directory);staged=root/'stage';staged.mkdir();ready=root/'ready';ready.mkdir();prepared=root/'prepared';logs=prepared/'private-logs';logs.mkdir(parents=True)
            for name in ['provisioning.stdout','entitlements.stdout']:(logs/name).write_bytes(plistlib.dumps({}))
            (logs/'signature-details.stderr').write_text('authored signature')
            prepared_record=dict(private_evidence_pins={str(path.relative_to(p.ROOT)):p.phone.package.sha(path) for path in logs.iterdir()})
            (prepared/'integration.json').write_text(json.dumps(prepared_record));prepared_sha=p.phone.package.sha(prepared/'integration.json')
            selection=root/'selection.json';selection.write_text('{}');selected=dict(device_identifier='fixture-device-private',device_udid='fixture-udid-private',development_team='fixture-team-private')
            remote='/var/containers/Bundle/Application/fixture-private/TokletHarness.app';signed={'TokletHarness':'b'*64};device={'product_type':'iPhone15,2'}
            c,r=fixture();record=dict(model='qwen8',workload='lifecycle-512',mode='lifecycle',config=c,job=dict(relative_path='M7Jobs/'+'a'*64+'.json',sha256='a'*64))
            calls=[];state=dict(alive=fail=='preexisting',waits=0,drained=False,wait_failed=False,capture_failed=False)
            def response(kind,**values):return dict(info=dict(outcome='success',commandType=kind),result=dict(deviceIdentifier=selected['device_identifier'],**values))
            def owned(args,out,err,env,timeout):
                label=out.stem;calls.append((label,args))
                if label=='host-boot':out.write_text('00000000-0000-4000-8000-000000000001\n')
                else:
                    if label.startswith('current-'):
                        actual='/other/TokletHarness.app' if fail=='installation' else remote
                        data=response('devicectl.device.info.apps',matchingBundleIdentifier=p.phone.physical.BUNDLE,apps=[dict(bundleIdentifier=p.phone.physical.BUNDLE,builtByDeveloper=True,url='file://'+actual)])
                    elif 'processes' in args:
                        data=response('devicectl.device.info.processes',runningProcesses=[dict(processIdentifier=42,executable='file://'+remote+'/TokletHarness')] if state['alive'] else [])
                    elif 'signal' in args:
                        state['alive']=False;sig=args[args.index('--signal')+1]
                        data=response('devicectl.device.process.signal',process=dict(processIdentifier=43 if fail=='signalResult' else 42,executable=remote+'/TokletHarness'),signal=dict(name=sig,value={'SIGTERM':15,'SIGKILL':9}[sig]))
                    elif 'launch' in args:
                        bundle=args[-1]
                        if bundle not in ['com.apple.Preferences',p.phone.physical.BUNDLE] or args.index('--device')>args.index(bundle):raise AssertionError('Actual devicectl positional boundary differs')
                        system=bundle=='com.apple.Preferences';pid=43 if fail=='pid' and not system else 42
                        data=response('devicectl.device.process.launch',process=dict(processIdentifier=pid,executable='/Applications/Preferences.app/Preferences' if system else remote+'/TokletHarness'))
                    else:
                        data=response('devicectl.device.info.details')
                        if fail==label+'-outcome':data['info']['outcome']='failure'
                        if fail==label+'-kind':data['info']['commandType']='unrelated'
                    out.write_text(json.dumps(data))
                err.write_text('')
                return dict(command=args,exit_code=0,primary_error=None,cleanup_errors=[],pid=99999,stdout=str(out.relative_to(p.ROOT)),stderr=str(err.relative_to(p.ROOT)))
            class Process:
                pid=88888
                def __init__(self,args,**kwargs):
                    calls.append(('launcher',args));state['alive']=fail not in ['launchRace','signalResult']
                    kwargs['stdout'].write(text(r,signed).encode());kwargs['stdout'].flush()
                    data=response('devicectl.device.process.launch',process=dict(processIdentifier=42,executable=remote+'/TokletHarness'),terminationResult=dict(exitCode=0,wasCoreDumpCreated=False))
                    Path(args[args.index('--json-output')+1]).write_text(json.dumps(data))
                def poll(self):return None if state['alive'] else 0
                def wait(self,timeout):
                    state['waits']+=1
                    if fail in ['timeout','drain'] and not state['wait_failed']:
                        state['wait_failed']=True;raise subprocess.TimeoutExpired(['devicectl',*selected.values(),remote],timeout)
                    if fail not in ['launchRace','signalResult']:state['alive']=False
                    return 0
            def drain(process,errors,timeout):
                state['drained']=True
                if fail in ['launchRace','signalResult']:state['alive']=True
                if fail=='drain':raise PermissionError('authored drain failure')
            original_open=Path.open
            class LateReplacement:
                def __init__(self,handle):self.handle=handle
                def __enter__(self):return self.handle
                def __exit__(self,*args):
                    self.handle.close();path=root/'output/private-logs/device-before.stdout'
                    state['original_log_pin']=hashlib.sha256(path.read_bytes()).hexdigest()
                    path.write_bytes(b'late replacement after original ledger')
            class CloseFailure:
                def __init__(self,handle):self.handle=handle
                def write(self,value):return self.handle.write(value)
                def flush(self):return self.handle.flush()
                def close(self):self.handle.close();raise OSError('authored capture close')
            def opening(path,*args,**kwargs):
                if fail=='capture' and path.name=='device-before.stdout' and args and args[0]=='rb' and not state['capture_failed']:
                    state['capture_failed']=True;raise OSError('authored read failure')
                handle=original_open(path,*args,**kwargs)
                if fail=='lateReplacement' and path.name=='private-execution.json' and args and args[0]=='xb':return LateReplacement(handle)
                return CloseFailure(handle) if fail=='close' and path.name=='lifecycle.stdout' and args and args[0]=='xb' else handle
            for target,name,value in [(p.phone,'SELECTION',selection),(p.phone,'SELECTION_SHA',p.phone.package.sha(selection)),(p.phone,'PREPARED',prepared),(p.phone,'PREPARED_SHA',prepared_sha)]:stack.enter_context(patch.object(target,name,value))
            stack.enter_context(patch.object(p.phone.package,'owned',side_effect=owned));stack.enter_context(patch.object(p.phone.package,'drain_group',side_effect=drain))
            stack.enter_context(patch.object(p.phone.integration,'admit_build',return_value={}))
            stack.enter_context(patch.object(p.phone.physical,'selection',return_value=selected))
            stack.enter_context(patch.object(p.phone.physical,'admit_reuse',return_value=(prepared,{},remote,signed)))
            stack.enter_context(patch.object(p.phone.physical,'admit_profile',return_value={'fixture':True}))
            stack.enter_context(patch.object(p.phone.physical,'admit_device',return_value=device));stack.enter_context(patch.object(p.phone.physical,'inventory',return_value=signed))
            stack.enter_context(patch.object(p.phone,'admitted_stage',return_value=(record,{}, {},0)))
            stack.enter_context(patch.object(p,'admit_ready',return_value=(dict(device=device),{},{})))
            stack.enter_context(patch.object(p.subprocess,'Popen',Process));stack.enter_context(patch.object(p.time,'sleep'));stack.enter_context(patch.object(Path,'open',opening));stack.enter_context(patch('builtins.print'))
            result=p.collect(staged,'a'*64,ready,'b'*64,root/'output')
            private=json.loads((root/'output/private-execution.json').read_text());public=json.dumps(result)
            state['final_guard']=json.loads((root/'output/private-final-guard.json').read_text())
            if fail=='lateReplacement':
                path=root/'output/private-logs/device-before.stdout';state['log_name']=str(path.relative_to(p.ROOT));state['actual_log_pin']=hashlib.sha256(path.read_bytes()).hexdigest()
            for secret in [*selected.values(),remote]:self.assertNotIn(secret,public)
            self.assertTrue(all('--terminate-existing' not in args for _,args in calls))
            if any(label=='launcher' for label,_ in calls):self.assertTrue(state['drained']);self.assertGreaterEqual(state['waits'],1)
            return result,private,calls,state
    def test_same_pid_roundtrip_and_natural_completion(self):
        r,private,calls,state=self.invoke();self.assertTrue(r['passed']);self.assertEqual(r['checked']['launch']['pid'],42)
        self.assertEqual([v['phase'] for v in r['phase_observations']],['ready-consumed','ready-process','settings-activated','background-process','harness-activated'])
        self.assertEqual(len(r['remote_cleanup']),2);self.assertTrue(all(not v['signals'] for v in r['remote_cleanup']))
        self.assertTrue(all(row['exit_code']==0 for row in private['commands']))
    def test_wrong_foreground_pid_and_existing_app_refuse_without_forced_relaunch(self):
        for fail in ['pid','preexisting','installation']:
            with self.subTest(fail=fail):
                r,_,calls,_=self.invoke(fail);self.assertFalse(r['passed'])
                if fail in ['preexisting','installation']:self.assertFalse(any(label=='launcher' for label,_ in calls))
    def test_timeout_retains_original_private_cause_and_cleans(self):
        r,private,_,_=self.invoke('timeout');self.assertFalse(r['passed']);self.assertIn('fixture-device-private',private['primary_error'])
        self.assertIn('<private-device_identifier>',r['primary_error']);self.assertTrue(r['remote_cleanup'][-1]['absent'])
    def test_launcher_is_drained_before_late_remote_cleanup(self):
        r,_,_,_=self.invoke('launchRace');self.assertFalse(r['passed']);self.assertTrue(r['remote_cleanup'][-1]['absent']);self.assertTrue(r['remote_cleanup'][-1]['signals'])
    def test_signal_response_failure_is_not_erased_by_later_absence(self):
        r,_,_,_=self.invoke('signalResult');self.assertFalse(r['passed']);self.assertTrue(r['remote_cleanup'][-1]['absent']);self.assertTrue(r['remote_cleanup'][-1]['errors'])
    def test_drain_failure_still_reaps_and_retains_capture_close_failure(self):
        for fail in ['drain','close']:
            with self.subTest(fail=fail):
                r,private,_,state=self.invoke(fail);self.assertFalse(r['passed']);self.assertTrue(r['remote_cleanup'][-1]['absent'])
                if fail=='drain':self.assertGreaterEqual(state['waits'],2);self.assertTrue(any(row['cleanup_errors'] for row in private['commands']))
                else:self.assertTrue(r['cleanup_errors'])
    def test_capture_error_keeps_original_owned_row_and_private_ledger(self):
        r,private,calls,_=self.invoke('capture');self.assertFalse(r['passed']);self.assertFalse(any(label=='launcher' for label,_ in calls))
        row=next(row for row in private['commands'] if Path(row['stdout']).stem=='device-before')
        self.assertEqual(row['exit_code'],0);self.assertIsNone(row['primary_error']);self.assertIn('authored read failure',row['capture_errors'][0])
    def test_late_replacement_fails_without_rebinding_consumed_log_or_rewriting_ledger(self):
        r,private,_,state=self.invoke('lateReplacement');self.assertFalse(r['passed'])
        self.assertEqual(private['endpoint_errors'],[])
        self.assertTrue(any('Final closure:' in error for error in state['final_guard']['endpoint_errors']))
        name=state['log_name'];original=state['original_log_pin']
        self.assertEqual(r['bindings'][name],original);self.assertEqual(r['private_evidence_pins'][name],original)
        self.assertNotEqual(state['actual_log_pin'],original)
    def test_device_details_outcome_and_kind_reject_at_both_endpoints(self):
        for label in ['device-before','device-after']:
            for field in ['outcome','kind']:
                with self.subTest(label=label,field=field):
                    r,_,calls,_=self.invoke(label+'-'+field);self.assertFalse(r['passed'])
                    if label=='device-before':self.assertFalse(any(name=='launcher' for name,_ in calls))
                    else:self.assertTrue(r['endpoint_errors'])
    def test_output_preflights_reject_existing_and_dangling_paths(self):
        with tempfile.TemporaryDirectory(dir=p.ROOT/'artifacts') as directory:
            root=Path(directory);output=root/'output';output.mkdir()
            with self.assertRaises(FileExistsError):p.collect(root,'a'*64,root,'b'*64,output)
            dangling=root/'dangling';dangling.symlink_to(root/'missing')
            with self.assertRaises(FileExistsError):p.collect(root,'a'*64,root,'b'*64,dangling)


if __name__=='__main__':unittest.main()
