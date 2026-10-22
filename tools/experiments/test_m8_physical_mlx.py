"""Portable transport orchestration and consumed-evidence checks; no device/model download."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import plistlib
import tempfile
import unittest
from unittest.mock import patch

import m8_physical_mlx as p


class Transport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):(p.ROOT/'artifacts').mkdir(exist_ok=True)

    def setUp(self):
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.root=Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=p.ROOT/'artifacts')))
        self.selected=dict(device_identifier='test-selected-device',device_udid='test-selected-udid',development_team='ABCDE12345',product_type='iPhone15,2',os_version='27.0.1')
        self.selection=self.root/'private.json';self.selection.write_text(json.dumps(self.selected))
        self.prepared=self.root/'prepared';logs=self.prepared/'private-logs';logs.mkdir(parents=True)
        for name in ['provisioning.stdout','entitlements.stdout']:(logs/name).write_bytes(plistlib.dumps({}))
        (logs/'signature-details.stderr').write_text('test-signature')
        receipt=dict(private_evidence_pins={str(f.relative_to(p.ROOT)):p.package.sha(f) for f in logs.iterdir()})
        (self.prepared/'integration.json').write_text(json.dumps(receipt));self.prepared_sha=p.package.sha(self.prepared/'integration.json')
        self.signed={'TokletHarness':'b'*64};self.remote='/var/containers/Bundle/Application/test/TokletHarness.app'
        self.running=False;self.labels=[];self.mode='normal';self.config={};self.stage=None
        for name,value in [('SELECTION',self.selection),('SELECTION_SHA',p.package.sha(self.selection)),('PREPARED',self.prepared),('PREPARED_SHA',self.prepared_sha)]:
            self.stack.enter_context(patch.object(p,name,value))
        self.stack.enter_context(patch.object(p.integration,'admit_build',return_value={}))
        self.stack.enter_context(patch.object(p.physical,'admit_reuse',return_value=(self.prepared,{str((self.prepared/'integration.json').relative_to(p.ROOT)):self.prepared_sha},self.remote,self.signed)))
        self.stack.enter_context(patch.object(p.physical,'admit_profile',return_value={'test':True}))
        self.stack.enter_context(patch.object(p.physical,'inventory',return_value=self.signed))
        self.stack.enter_context(patch.object(p.package,'owned',side_effect=self.owned))
        self.stack.enter_context(patch('builtins.print'))

    def staged(self,workload='measurement-short'):
        self.stage=self.root/('stage-'+workload);record=p.jobs.stage('tiny',workload,self.stage);self.config=record['config']
        return record,p.package.sha(self.stage/'staged.json')

    def measurement(self):
        c=self.config;fp=dict(physicalBytes=100,lifetimePeakPhysicalBytes=200,returnedNaturalCount=93)
        environment=dict(applicationState=0,lowPowerMode=False,thermalState=0,batteryState=3,batteryLevel=1.0)
        samples=[];total=0.0
        for index in range(13):
            start=(index+1)*10**9;events=[dict(index=i,nativeElapsedSeconds=(i+1)/10,callbackUptimeNanoseconds=start+(i+1)*10**8) for i in range(4)]
            samples.append(dict(index=index,role='warmup' if index<3 else 'measured',status=0,tokens=[47]*4,history=8,evaluated=7,pending=1,reason=1,
                admittedBytes=4096,footprint=fp,environment=environment,submittedUptimeNanoseconds=start,completedUptimeNanoseconds=start+5*10**8,events=events))
            if index>=3:total+=events[-1]['nativeElapsedSeconds']
        result=dict(passed=True,backend='metal-mlx',interruptionCount=0,cleanupErrors=[],endpointErrors=[],closed=True,released=True,
            physicalDeviceQualified=False,m7Complete=False,filesBefore=c['files'],filesAfter=c['files'],
            filesystemBefore=dict(freeBytes=10*1024**3,readStartedUptimeNanoseconds=1,readCompletedUptimeNanoseconds=2,
                instrument='Foundation.FileManager.attributesOfFileSystem.systemFreeSize',scope='trusted-root-filesystem'),
            checkpoints=[dict(phase=name,footprint=fp) for name in ['before-artifact-proof','after-artifact-proof','loaded','unloaded','closed','after-final-artifact-proof']],
            warmupsCompleted=3,measuredCompleted=10,sumNativeLastEventSeconds=total,loadAndSessionSeconds=1)
        return '\n'.join(['TOKLET_INSTALLATION_JSON='+json.dumps(dict(phase='before',simulator=False,files=self.signed))]
            +['TOKLET_MEASUREMENT_SAMPLE_JSON='+json.dumps(sample) for sample in samples]
            +['TOKLET_INSTALLATION_JSON='+json.dumps(dict(phase='after',simulator=False,files=self.signed)),'TOKLET_MEASUREMENT_JSON='+json.dumps(result)])

    def owned(self,args,stdout,stderr,env,timeout):
        label=stdout.stem;self.labels.append(label)
        data=dict(info=dict(outcome='success'),result=dict(deviceIdentifier=self.selected['device_identifier']))
        r=data['result']
        if label=='host-boot':stdout.write_text('00000000-0000-4000-8000-000000000001\n')
        elif label=='measurement':
            stdout.write_text(self.measurement())
            launch=dict(info=dict(outcome='success'),result=dict(deviceIdentifier=self.selected['device_identifier'],
                process=dict(processIdentifier=123,executable=self.remote+'/TokletHarness'),terminationResult=dict(exitCode=0,wasCoreDumpCreated=False)))
            Path(args[args.index('--json-output')+1]).write_text(json.dumps(launch))
            if self.mode=='mutation':self.selection.write_text('{}');self.running=True
        else:
            if label.startswith('device-'):
                r.update(hardwareProperties=dict(udid=self.selected['device_udid'],productType='iPhone15,2',marketingName='iPhone 14 Pro',reality='physical'),
                    deviceProperties=dict(osVersionNumber='27.0.1',developerModeStatus='enabled',ddiServicesAvailable=True),
                    connectionProperties=dict(pairingState='paired',transportType='wired',tunnelState='connected'))
            elif label.startswith('current-'):
                r.update(matchingBundleIdentifier=p.physical.BUNDLE,apps=[dict(bundleIdentifier=p.physical.BUNDLE,builtByDeveloper=True,url='file://'+self.remote)])
            elif label.startswith('process-'):
                r['runningProcesses']=[dict(executable=self.remote+'/TokletHarness',processIdentifier=123)] if self.running else []
                if self.mode=='wrong-query':r['deviceIdentifier']='wrong-device'
                if self.mode=='failed-query':data['info']['outcome']='failure'
            elif label.startswith('signal-'):
                self.running=False
                if self.mode=='wrong-signal':r['deviceIdentifier']='wrong-device'
            elif label=='model-listing':
                parent=str(Path(self.config['modelFolder']).parent)
                r.update(domain='appDataContainer',domainIdentifier=p.physical.BUNDLE,files=[dict(name=name.removeprefix(parent+'/'),relativePath=name.removeprefix(parent+'/'),
                    resources=dict(isDirectory=False,isSymbolicLink=False),metadata=dict(size=(self.stage/'transfer'/name).stat().st_size)) for name in self.config['files']])
            elif label=='job-listing':
                path=next((self.stage/'transfer/M7Jobs').glob('*.json'));name=path.name
                r.update(domain='appDataContainer',domainIdentifier=p.physical.BUNDLE,files=[] if self.mode=='bad-job-listing' else [dict(name=name,relativePath=name,
                    resources=dict(isDirectory=False,isSymbolicLink=False),metadata=dict(size=path.stat().st_size))])
            elif label in ['corpus-listing','tasks-listing']:
                field='tokensFile' if label=='corpus-listing' else 'tasksFile';name=Path(self.config[field]).name
                r.update(domain='appDataContainer',domainIdentifier=p.physical.BUNDLE,files=[dict(name=name,relativePath=name,
                    resources=dict(isDirectory=False,isSymbolicLink=False),metadata=dict(size=(self.stage/'transfer'/self.config[field]).stat().st_size+(1 if self.mode=='bad-listing' else 0)))])
            stdout.write_text(json.dumps(data))
        stderr.write_text('')
        return dict(command=args,exit_code=0,primary_error=None,cleanup_errors=[],pid=1,stdout=str(stdout.relative_to(p.ROOT)),stderr=str(stderr.relative_to(p.ROOT)))

    def test_prepare_and_fast_reuse_have_exact_separate_actions(self):
        _,pin=self.staged();ready=self.root/'ready'
        prepared=p.run(self.stage,pin,ready,prepare_only=True)
        self.assertTrue(prepared['passed']);self.assertIsNone(prepared['checked']);self.assertNotIn('measurement',self.labels)
        self.assertEqual([s for s in self.labels if s.startswith('copy-')],['copy-Model','copy-job'])
        self.labels=[]
        result=p.run(self.stage,pin,self.root/'collect',ready=ready,ready_sha=p.package.sha(ready/'measurement.json'))
        self.assertTrue(result['passed']);self.assertEqual(result['checked']['sample_count'],13)
        self.assertFalse(any(s.startswith('copy-') for s in self.labels));self.assertEqual(self.labels.count('measurement'),1)
        self.assertEqual(len(result['remote_cleanup']),2);self.assertTrue(all(r['absent'] and not r['errors'] for r in result['remote_cleanup']))

    def test_state_route_uses_explicit_app_argument_and_state_validator(self):
        # Isolate dispatch from artifact/state semantics, covered by the strict
        # recipe and state fixtures. Existing real-copy tests still forbid bypass.
        _,pin=self.staged();record,extra,payload,total=p.admitted_stage(self.stage,pin)
        record=dict(record,mode='state')
        with patch.object(p,'admitted_stage',return_value=(record,extra,payload,total)),patch.object(p.state,'validate_console',return_value={'state_witness':'checked'}) as validate:
            result=p.run(self.stage,pin,self.root/'state-route')
        self.assertTrue(result['passed']);validate.assert_called_once()
        self.assertEqual(validate.call_args.args[1:],(self.config,self.signed))
        self.assertEqual(result['checked']['state_witness'],'checked')
        launch=next(row for row in result['commands'] if Path(row['stdout']).stem=='measurement')
        self.assertEqual(launch['command'][-3],'--state')
        self.assertNotIn('--measure',launch['command']);self.assertFalse(result['physical_device_qualified'])

    def test_rejected_state_console_retains_failure_and_remote_cleanup(self):
        _,pin=self.staged();record,extra,payload,total=p.admitted_stage(self.stage,pin)
        record=dict(record,mode='state')
        with patch.object(p,'admitted_stage',return_value=(record,extra,payload,total)),patch.object(p.state,'validate_console',side_effect=ValueError('state witness rejected')):
            result=p.run(self.stage,pin,self.root/'state-rejection')
        self.assertFalse(result['passed']);self.assertIn('state witness rejected',result['primary_error'])
        self.assertIsNone(result['checked']);self.assertTrue(result['remote_cleanup'][-1]['absent'])

    def test_source_mutation_cannot_block_exact_owned_cleanup(self):
        _,pin=self.staged();self.mode='mutation'
        result=p.run(self.stage,pin,self.root/'mutation')
        self.assertFalse(result['passed']);self.assertTrue(result['endpoint_errors']);self.assertFalse(self.running)
        self.assertEqual(result['remote_cleanup'][-1]['signals'],[dict(pid=123,signal='SIGTERM')])
        self.assertTrue(result['remote_cleanup'][-1]['absent']);self.assertIsNone(result['checked'])

    def test_wrong_device_and_failed_queries_never_prove_absence(self):
        _,pin=self.staged()
        for mode in ['wrong-query','failed-query']:
            self.mode=mode;self.labels=[]
            result=p.run(self.stage,pin,self.root/mode,prepare_only=True)
            self.assertFalse(result['passed']);self.assertTrue(result['cleanup_errors'])
            self.assertFalse(any(row['absent'] for row in result['remote_cleanup']))
            self.assertFalse(any(label.startswith('signal-') or label.startswith('copy-') for label in self.labels))

    def test_wrong_signal_response_keeps_failure_even_after_absence(self):
        _,pin=self.staged();self.mode='wrong-signal';self.running=True
        result=p.run(self.stage,pin,self.root/'signal',prepare_only=True)
        self.assertFalse(result['passed']);self.assertTrue(result['cleanup_errors'])
        self.assertTrue(result['remote_cleanup'][0]['absent']);self.assertTrue(result['remote_cleanup'][0]['errors'])

    def test_auxiliary_transfer_inventory_is_required_for_ready(self):
        _,pin=self.staged('quality-short');self.mode='bad-listing'
        result=p.run(self.stage,pin,self.root/'auxiliary',prepare_only=True)
        self.assertFalse(result['passed']);self.assertIn('corpus-listing',self.labels)
        self.assertNotIn('copy-job',self.labels);self.assertNotIn('measurement',self.labels)

    def test_copied_job_inventory_must_exist_before_ready(self):
        _,pin=self.staged();self.mode='bad-job-listing'
        result=p.run(self.stage,pin,self.root/'bad-job',prepare_only=True)
        self.assertFalse(result['passed']);self.assertIn('job-listing',self.labels);self.assertNotIn('measurement',self.labels)

    def test_invalid_ready_receipt_never_copies_or_launches(self):
        _,pin=self.staged();ready=self.root/'ready'
        result=p.run(self.stage,pin,ready,prepare_only=True);self.assertTrue(result['passed'])
        path=ready/'measurement.json';value=json.loads(path.read_text());value['signed_bundle']={};path.write_text(json.dumps(value));self.labels=[]
        result=p.run(self.stage,pin,self.root/'bad-ready',ready=ready,ready_sha=p.package.sha(path))
        self.assertFalse(result['passed']);self.assertFalse(any(label.startswith('copy-') or label=='measurement' for label in self.labels))

    def test_real_copy_missing_proof_rejects_before_any_child(self):
        _,pin=self.staged();record,extra,payload,total=p.admitted_stage(self.stage,pin)
        for model in ['qwen','qwen8']:
            record['model']=model;self.labels=[]
            with patch.object(p,'admitted_stage',return_value=(record,extra,payload,total)):
                result=p.run(self.stage,pin,self.root/('no-proof-'+model),prepare_only=True)
            self.assertFalse(result['passed']);self.assertIn('fresh same-app storage proof',result['primary_error']);self.assertEqual(self.labels,[])

    def test_real_ready_cannot_bypass_storage_replay(self):
        _,pin=self.staged();ready=self.root/'ready'
        prepared=p.run(self.stage,pin,ready,prepare_only=True);self.assertTrue(prepared['passed'])
        record,extra,payload,total=p.admitted_stage(self.stage,pin)
        for model in ['qwen','qwen8']:
            record['model']=model;prepared['model']=model
            prepared['storage_admission']=dict(proof=dict(path=str((self.root/'proof/measurement.json').relative_to(p.ROOT)),sha256='a'*64))
            path=ready/'measurement.json';path.write_text(json.dumps(prepared));self.labels=[]
            with patch.object(p,'admitted_stage',return_value=(record,extra,payload,total)),patch.object(p,'admit_storage',side_effect=ValueError('storage replay sentinel')) as admission:
                result=p.run(self.stage,pin,self.root/('ready-'+model),ready=ready,ready_sha=p.package.sha(path))
            self.assertFalse(result['passed']);self.assertIn('storage replay sentinel',result['primary_error']);admission.assert_called_once()
            self.assertFalse(any(label.startswith('copy-') or label=='measurement' for label in self.labels))

    def test_staged_inventory_and_recipe_reject_replacements(self):
        _,pin=self.staged();p.admitted_stage(self.stage,pin)
        path=self.stage/'staged.json';original=json.loads(path.read_text())
        for field,value in [('capacity',31),('version',2.0),('metal',1)]:
            altered=copy.deepcopy(original);altered['config'][field]=value;path.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):p.admitted_stage(self.stage,p.package.sha(path))

    def test_consumed_buffers_reject_swapped_bytes_and_restore_valid_ones(self):
        path=self.root/'evidence';path.write_bytes(b'{"value":1}');pin=p.package.sha(path)
        self.assertEqual(p.metadata(path,pin),{'value':1})
        replacement=self.root/'replacement';replacement.write_bytes(b'{"value":2}');replacement.replace(path)
        with self.assertRaises(ValueError):p.metadata(path,pin)
        with self.assertRaises(ValueError):p.capture(path,p.package.sha(path),1)
        # Capture remains the same admitted data even if the pathname is replaced later.
        raw=p.capture(path,p.package.sha(path));path.write_bytes(b'{"value":3}');self.assertEqual(json.loads(raw),{'value':2})

    def test_storage_replay_checks_bytes_after_prior_hash_admission(self):
        _,pin=self.staged();folder=self.root/'proof'
        result=p.run(self.stage,pin,folder);self.assertTrue(result['passed'])
        context=dict(prepared_receipt=result['prepared_receipt'],signed_bundle=self.signed,required_bindings={},remote=self.remote,selected=self.selected)
        proof_pin=p.package.sha(folder/'measurement.json')
        observed,_=p.admit_storage(folder,proof_pin,context);self.assertEqual(observed['checked']['sample_count'],13)
        original_verify=p.measurement.verify_inputs
        for name in ['measurement.stdout','measurement.stderr','launch.json','host-boot.stdout']:
            path=folder/'private-logs'/name;original=path.read_bytes();calls=0
            def verify(pins):
                nonlocal calls
                original_verify(pins);calls+=1
                if calls==2:path.write_bytes(b'changed after prior hash')
            try:
                with patch.object(p.measurement,'verify_inputs',side_effect=verify):
                    with self.assertRaisesRegex(ValueError,'Consumed metadata differs'):p.admit_storage(folder,proof_pin,context)
                self.assertEqual(calls,2)
            finally:path.write_bytes(original)

    def test_changed_cached_profile_rejects_before_any_device_child(self):
        _,pin=self.staged();(self.prepared/'private-logs/provisioning.stdout').write_bytes(b'changed')
        with patch.object(p.physical,'admit_profile') as admission:
            result=p.run(self.stage,pin,self.root/'profile')
        self.assertFalse(result['passed']);self.assertEqual(self.labels,[]);admission.assert_not_called()

    def test_output_and_option_preflights_are_exclusive(self):
        _,pin=self.staged();existing=self.root/'existing';existing.mkdir()
        with self.assertRaises(FileExistsError):p.run(self.stage,pin,existing)
        dangling=self.root/'dangling';dangling.symlink_to(self.root/'missing')
        with self.assertRaises(FileExistsError):p.run(self.stage,pin,dangling)
        with self.assertRaises(ValueError):p.run(self.stage,pin,self.root/'options',ready=self.root/'ready')
        self.assertEqual(self.labels,[])


if __name__=='__main__':unittest.main()
