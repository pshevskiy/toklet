import copy
import json
from pathlib import Path
import plistlib
import tempfile
import unittest
from unittest.mock import patch
import m7_physical_tasks as t
m=t.measurement


def fixture(count=1,stop_after=None):
    data=dict(version=1,vocabulary=64,stops=[62,63],cases=[dict(id='HumanEval/'+str(i),promptTokens=[1,7,13],maximum=256) for i in range(count)])
    raw=(json.dumps(data,sort_keys=True,separators=(',',':'))+'\n').encode();sha=m.package.hashlib.sha256(raw).hexdigest()
    config=dict(version=1,modelFolder='M7Inputs/tiny-cpu/Model',packedFolder='',files={'M7Inputs/tiny-cpu/Model/config.json':'a'*64},metal=False,
                capacity=259,chunk=128,numericalBudget=128*1024**2,processBudget=192*1024**2,tasksFile='M7Tasks/'+sha+'/tasks.json',tasksSHA256=sha,caseCount=count)
    signed={'TokletHarness':'b'*64};fp=dict(physicalBytes=100,lifetimePeakPhysicalBytes=200,returnedNaturalCount=93)
    environment=dict(applicationState=0,lowPowerMode=False,thermalState=0,batteryState=3,batteryLevel=1.0)
    events=[('TOKLET_INSTALLATION_JSON=',dict(phase='before',simulator=False,files=signed))];total=0
    for index,case in enumerate(data['cases']):
        sampled=0;ids=[];chunk_index=0
        while sampled<256:
            requested=min(128,256-sampled);remaining=(stop_after-sampled) if stop_after is not None else requested+1
            hidden=remaining<=requested;visible=remaining-1 if hidden else requested;new=[47]*visible
            sampled+=visible+int(hidden);ids+=new;start=(total+1)*5_000_000_000;end=start+2_000_000_000
            row=dict(index=chunk_index,requested=requested,ids=new,tokenTimings=[dict(index=i,nativeElapsedSeconds=(i+1)/100,callbackUptimeNanoseconds=start+(i+1)*10_000_000) for i in range(visible)],
                     reason=2 if hidden else 1,history=3+sampled,evaluated=2+sampled,pending=1)
            if hidden:row['stopToken']=62
            common=dict(caseIndex=index,caseID=case['id'],admittedBytes=100,footprint=copy.deepcopy(fp),environment=copy.deepcopy(environment))
            events.append(('TOKLET_TASK_CHUNK_JSON=',dict(**common,chunk=row,submittedUptimeNanoseconds=start,completedUptimeNanoseconds=end)))
            chunk_index+=1;total+=1
            if hidden:break
        response=dict(ids=ids,sampled=sampled,reason=2 if hidden else 1,history=3+sampled,evaluated=2+sampled,pending=1,chunks=chunk_index)
        if hidden:response['stopToken']=62
        events.append(('TOKLET_TASK_CASE_JSON=',dict(**common,response=response,resetHistory=0,resetEvaluated=0,resetPending=0)))
    result=dict(backend='cpu',passed=True,cleanupErrors=[],endpointErrors=[],closed=True,released=True,interruptionCount=0,loadAndSessionSeconds=1,
                fullQualityQualified=False,physicalDeviceQualified=False,m7Complete=False,casesCompleted=count,chunksCompleted=total,loadedAdmittedBytes=100,
                filesBefore=config['files'],filesAfter=config['files'],tasksBeforeSHA256=sha,tasksAfterSHA256=sha,
                checkpoints=[dict(phase=p,footprint=copy.deepcopy(fp)) for p in ['before-artifact-proof','after-artifact-proof','loaded']+['case-'+str(i) for i in range(count)]+['unloaded','closed','after-final-artifact-proof']])
    events.extend([('TOKLET_INSTALLATION_JSON=',dict(phase='after',simulator=False,files=signed)),('TOKLET_TASKS_JSON=',result)])
    return config,signed,data,events


def console(events):return '\n'.join(marker+json.dumps(value) for marker,value in events)


class PhysicalTasksChecks(unittest.TestCase):
    def reject(self,mutate):
        c,s,d,events=fixture();mutate(c,d,events)
        with self.assertRaises((ValueError,KeyError,TypeError)):t.check_console(console(events),c,s,d)
    def test_full_bounded_stream_and_hidden_stops(self):
        for count,stop in [(100,None),(1,1),(1,128),(1,129),(1,256)]:
            c,s,d,events=fixture(count,stop);out=t.check_console(console(events),c,s,d)
            self.assertEqual(out['case_count'],count);self.assertEqual(out['chunk_count'],count*(2 if stop is None else (stop+127)//128))
            self.assertEqual(out['responses'][0]['response']['sampled'],256 if stop is None else stop)
    def test_chunk_identity_callbacks_and_budget(self):
        for field,value in [('index',1),('requested',127),('ids',[True]*128),('reason',3),('history',1),('pending',0),('stopToken',62)]:
            with self.subTest(field=field):self.reject(lambda c,d,e:e[1][1]['chunk'].update({field:value}))
        self.reject(lambda c,d,e:e[1][1]['chunk']['tokenTimings'].pop())
        self.reject(lambda c,d,e:e[1][1]['chunk']['tokenTimings'][0].update(nativeElapsedSeconds=float('nan')))
        self.reject(lambda c,d,e:e[1][1]['chunk']['tokenTimings'][0].update(nativeElapsedSeconds=2))
        self.reject(lambda c,d,e:e[2][1].update(submittedUptimeNanoseconds=e[1][1]['completedUptimeNanoseconds']-1))
    def test_hidden_stop_cannot_be_exposed_or_changed(self):
        c,s,d,events=fixture(1,129)
        for mutate in (lambda e:e[2][1]['chunk'].update(stopToken=61),lambda e:e[2][1]['chunk'].update(ids=[62]),lambda e:e[3][1]['response'].update(stopToken=63)):
            changed=copy.deepcopy(events);mutate(changed)
            with self.assertRaises((ValueError,KeyError,TypeError)):t.check_console(console(changed),c,s,d)
    def test_case_response_reset_and_order(self):
        for field,value in [('caseID','other'),('caseIndex',True),('resetHistory',1),('admittedBytes',101)]:
            with self.subTest(field=field):self.reject(lambda c,d,e:e[3][1].update({field:value}))
        self.reject(lambda c,d,e:e[3][1]['response'].update(ids=[47]))
        self.reject(lambda c,d,e:e.pop(2))
        self.reject(lambda c,d,e:e.reverse())
    def test_installation_result_and_unknown_markers(self):
        self.reject(lambda c,d,e:e[0][1].update(simulator=True))
        self.reject(lambda c,d,e:e[-2][1].update(files={}))
        for field,value in [('passed',False),('closed',False),('released',False),('casesCompleted',True),('chunksCompleted',1),('tasksAfterSHA256','b'*64),('loadAndSessionSeconds',True),('fullQualityQualified',True)]:
            with self.subTest(field=field):self.reject(lambda c,d,e:e[-1][1].update({field:value}))
        c,s,d,e=fixture()
        with self.assertRaises(ValueError):t.check_console(console(e)+'\nTOKLET_TASKS_ERROR=bad',c,s,d)
    def test_memory_foreground_and_power(self):
        self.reject(lambda c,d,e:e[1][1]['footprint'].update(lifetimePeakPhysicalBytes=193*1024**2))
        self.reject(lambda c,d,e:e[1][1]['footprint'].update(lifetimePeakPhysicalBytes=201))
        for field,value in [('applicationState',1),('lowPowerMode',0),('thermalState',4),('batteryState',0),('batteryLevel',float('nan'))]:
            with self.subTest(field=field):self.reject(lambda c,d,e:e[1][1]['environment'].update({field:value}))
    def test_exact_reference_schema_and_sha(self):
        self.reject(lambda c,d,e:d.update(version=True))
        self.reject(lambda c,d,e:d.update(extra=True))
        self.reject(lambda c,d,e:d['cases'][0].update(extra=True))
        self.reject(lambda c,d,e:c.update(tasksSHA256='b'*64))
        self.reject(lambda c,d,e:c.update(capacity=258))
    def test_frozen_derivative_keeps_all_original_cases_and_caps(self):
        # Unit coverage needs no downloaded task artifact or model. The actual
        # frozen payload is checked explicitly in this component's evidence.
        original=[dict(id='HumanEval/'+str(i),prompt_tokens=[1,7,13],max_output_tokens=256 if i<40 else 32 if i<70 else 16) for i in range(100)]
        with patch.object(t,'frozen_tasks',return_value=(original,{'frozen':'a'*64})):
            raw,data,pins=t.task_payload('small','full')
            self.assertEqual(len(data['cases']),100);self.assertEqual(data['stops'],list(t.task_cases.EOS_IDS));self.assertEqual(pins,{'frozen':'a'*64})
            self.assertEqual([(r['id'],r['promptTokens'],r['maximum']) for r in data['cases']],[(r['id'],r['prompt_tokens'],r['max_output_tokens']) for r in original])
            self.assertLess(len(raw),512*1024)
            _,short,_=t.task_payload('larger','short');self.assertEqual(len(short['cases']),1);self.assertEqual(short['cases'][0]['maximum'],8)
        with self.assertRaises(ValueError):t.task_payload('tiny-cpu','full')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();path=root/t.TASK_FILE;path.parent.mkdir(parents=True);path.write_bytes(b'changed')
            with patch.object(t,'ROOT',root),patch.object(m.integration,'metadata',side_effect=[dict(bindings={'benchmarks/contract-v1/manifest.json':'a'*64}),dict(quality={})]),self.assertRaises(ValueError):t.frozen_tasks()
    def test_task_storage_inventory_and_copy_clock(self):
        c,_,data,_=fixture();raw=(json.dumps(data,sort_keys=True,separators=(',',':'))+'\n').encode()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();base=root/'artifacts/staged';base.mkdir(parents=True);model=base/next(iter(c['files']));model.parent.mkdir(parents=True);model.write_bytes(b'model');c['files']={str(model.relative_to(base)):m.package.sha(model)}
            payload=base/c['tasksFile'];payload.parent.mkdir(parents=True);payload.write_bytes(raw)
            job=b'{}\n';sha=m.package.hashlib.sha256(job).hexdigest();name='M7Jobs/'+sha+'.json';path=base/name;path.parent.mkdir();path.write_bytes(job)
            with patch.object(m.storage,'ROOT',root):
                inventory,total=m.storage.inventory(base,c,name,sha);self.assertEqual(total,len(b'model')+len(raw)+len(job));self.assertIn(c['tasksFile'],inventory)
                saved=dict(proof=dict(path='artifacts/proof/measurement.json',sha256='a'*64),checks=[dict(label=label,wallUnixNanoseconds=1,monotonicNanoseconds=1) for label in ['copy-Model','copy-tasks','copy-job']])
                with patch.object(m.storage,'admit_proof',return_value=({'host_boot_token':'b'*64},{})),patch.object(m.storage,'inventory',return_value=(inventory,total)),patch.object(m.storage,'decision',side_effect=lambda *a:copy.deepcopy(saved)),patch.object(m.storage,'fresh',side_effect=lambda a,b,w,n,l:dict(label=l,wallUnixNanoseconds=w,monotonicNanoseconds=n)):
                    self.assertEqual(m.storage.admit_ready(dict(storage_admission=saved),base,c,name,sha,{},None)[0],saved)
                    bad=copy.deepcopy(saved);bad['checks'].pop(1)
                    with self.assertRaises(ValueError):m.storage.admit_ready(dict(storage_admission=bad),base,c,name,sha,{},None)
    def test_task_prepare_transport_selects_payload_package_and_format(self):
        (m.ROOT/'artifacts').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=m.ROOT/'artifacts') as temp:
            folder=Path(temp).resolve();output=folder/'run';app=folder/'app';app.mkdir()
            c,_,data,_=fixture();raw=(json.dumps(data,sort_keys=True,separators=(',',':'))+'\n').encode()
            selected=dict(device_identifier='device',device_udid='udid',development_team='team');signed={'TokletHarness':'b'*64};remote={'path':'owned'};labels=[]
            listed=dict(info=dict(outcome='success'),result=dict(deviceIdentifier='device',domain='appDataContainer',domainIdentifier=m.physical.BUNDLE,
                files=[dict(name='tasks.json',relativePath='tasks.json',resources=dict(isSymbolicLink=False,isDirectory=False),metadata=dict(size=len(raw)))]))
            def stage(config,host,destination,*unused):
                base=destination/'transfer';base.mkdir();name='M7Jobs/'+('a'*64)+'.json';path=base/name;path.parent.mkdir();path.write_text('{}')
                (base/'M7Inputs/tiny-cpu/Model').mkdir(parents=True);path=base/config['tasksFile'];path.parent.mkdir(parents=True);path.write_bytes(raw)
                return base,name,'a'*64
            def owned(args,stdout,stderr,env,timeout):
                label=stdout.stem;labels.append(label)
                value=plistlib.dumps({}).decode() if label in ('profile','entitlements') else json.dumps(listed) if label=='copied-tasks-inventory' else '{}'
                stdout.write_text(value);stderr.write_text('')
                return dict(command=args,exit_code=0,primary_error=None,cleanup_errors=[],pid=1,stdout=str(stdout.relative_to(m.ROOT)),stderr=str(stderr.relative_to(m.ROOT)))
            with patch.object(m.physical,'selection',return_value=selected),patch.object(m.integration,'admit_build',return_value={}) as package_admission,\
                 patch.object(m.physical,'admit_reuse',return_value=(app,{},remote,signed)),patch.object(m,'verify_inputs'),\
                 patch.object(t,'job',return_value=(c,{},{})),patch.object(t,'stage',side_effect=stage),patch.object(t,'task_payload',return_value=(raw,data,{})),\
                 patch.object(m.package,'owned',side_effect=owned),patch.object(m.physical,'admit_device',return_value={'productType':'test'}),\
                 patch.object(m.physical,'admit_profile',return_value={}),patch.object(m.physical,'launch_verified',side_effect=lambda query,action,*a:action()),\
                 patch.object(m.physical,'stop_remote',return_value=dict(absent=True,observations=[[]],signals=[],errors=[])),\
                 patch.object(m,'validate_listing'),patch.object(m.physical,'inventory',return_value=signed),patch.object(m.physical,'redact_record',side_effect=lambda record,*a:record):
                code=m.main(folder/'native',folder/'private.json','c'*64,folder/'prepared','d'*64,'tiny-cpu','short',output,prepare_only=True,native_sha=t.NATIVE_SHA,tasks=True)
            self.assertEqual(code,0);package_admission.assert_called_once_with((folder/'native').resolve(),t.NATIVE_SHA)
            record=json.loads((output/'measurement.json').read_text());self.assertEqual(record['format'],t.READY_FORMAT);self.assertFalse(record['full_quality_qualified'])
            self.assertEqual([label for label in labels if label.startswith('copy-')],['copy-Model','copy-tasks','copy-job']);self.assertIn('copied-tasks-inventory',labels)
            self.assertEqual(len(record['remote_cleanup']),2)

    def test_wrong_package_and_mode_conflict_prevent_children(self):
        with patch.object(m.package,'owned') as owned:
            for flags in [dict(tasks=True,native_sha=m.NATIVE_SHA),dict(tasks=True,quality=True,native_sha=t.NATIVE_SHA)]:
                with self.assertRaises(ValueError):m.main(Path('no'),Path('no'),'a'*64,Path('no'),'b'*64,'small','full',Path('no'),**flags)
            owned.assert_not_called()


if __name__=='__main__':unittest.main()
