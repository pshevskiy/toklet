import contextlib
import datetime
import io
import json
import plistlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import m7_physical_integration as m


class PhysicalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.config=dict(device_identifier='device-private',device_udid='udid-private',development_team='0123456789',product_type='iPhone15,2',os_version='27.0.1')
        self.device=dict(result=dict(hardwareProperties=dict(udid='udid-private',productType='iPhone15,2',reality='physical',marketingName='iPhone 14 Pro'),
            deviceProperties=dict(osVersionNumber='27.0.1',developerModeStatus='enabled',ddiServicesAvailable=True),
            connectionProperties=dict(pairingState='paired',transportType='wired',tunnelState='connected')))
        team=self.config['development_team']
        self.entitlements={'application-identifier':team+'.'+m.BUNDLE,'com.apple.developer.team-identifier':team,'get-task-allow':True}
        self.profile=dict(TeamIdentifier=[team],ApplicationIdentifierPrefix=[team],ProvisionedDevices=['udid-private'],
            ExpirationDate=datetime.datetime(2030,1,1),Entitlements=dict(self.entitlements))
        self.signature='Identifier='+m.BUNDLE+'\nTeamIdentifier='+team+'\n'
        self.now=datetime.datetime(2026,10,19,tzinfo=datetime.timezone.utc)
        self.app='/var/containers/Bundle/Application/owned/TokletHarness.app'

    def process(self,pid=123,path=None):
        return {'result':{'runningProcesses':[dict(executable=path or self.app+'/TokletHarness',processIdentifier=pid)]}}

    def test_device_rejects_simulator_wireless_replacement_and_disabled_services(self):
        self.assertEqual(m.admit_device(self.device,self.config)['reality'],'physical')
        for section,key,value in [('hardwareProperties','udid','another'),('hardwareProperties','reality','simulated'),
            ('connectionProperties','transportType','localNetwork'),('connectionProperties','tunnelState','disconnected'),
            ('deviceProperties','developerModeStatus','disabled'),('deviceProperties','ddiServicesAvailable',False)]:
            original=self.device['result'][section][key];self.device['result'][section][key]=value
            with self.assertRaisesRegex(ValueError,'Physical device'):m.admit_device(self.device,self.config)
            self.device['result'][section][key]=original

    def test_profile_requires_current_device_team_app_and_development_scope(self):
        self.assertTrue(m.admit_profile(self.profile,self.entitlements,self.signature,self.config,self.now)['device_in_profile'])
        for key,value in [('ProvisionedDevices',['another']),('TeamIdentifier',['another']),('ExpirationDate',datetime.datetime(2020,1,1))]:
            original=self.profile[key];self.profile[key]=value
            with self.assertRaisesRegex(ValueError,'provisioning'):m.admit_profile(self.profile,self.entitlements,self.signature,self.config,self.now)
            self.profile[key]=original
        self.profile['Entitlements']['get-task-allow']=False
        with self.assertRaisesRegex(ValueError,'provisioning'):m.admit_profile(self.profile,self.entitlements,self.signature,self.config,self.now)

    def test_valid_profile_does_not_substitute_for_signature(self):
        self.entitlements['application-identifier']='another.app'
        with self.assertRaisesRegex(ValueError,'Signed application'):m.admit_profile(self.profile,self.entitlements,self.signature,self.config,self.now)
        self.entitlements['application-identifier']=self.config['development_team']+'.'+m.BUNDLE
        with self.assertRaisesRegex(ValueError,'Signed application'):m.admit_profile(self.profile,self.entitlements,self.signature.replace('TeamIdentifier=','Other='),self.config,self.now)

    def test_cleanup_requeries_and_does_not_signal_reused_unowned_pid(self):
        query=Mock(side_effect=[self.process(),self.process(path='/var/other.app/Other')]);signal=Mock()
        result=m.stop_remote(query,signal,self.app)
        self.assertFalse(result['absent']);self.assertTrue(any('Unowned' in s for s in result['errors']))
        self.assertEqual(signal.call_args_list,[unittest.mock.call(123,'SIGTERM')])

    def test_cleanup_requires_independent_absence_and_escalates(self):
        query=Mock(side_effect=[self.process(),self.process(),dict(result=dict(runningProcesses=[]))]);signal=Mock()
        result=m.stop_remote(query,signal,self.app)
        self.assertTrue(result['absent']);self.assertEqual(query.call_count,3)
        self.assertEqual(signal.call_args_list,[unittest.mock.call(123,'SIGTERM'),unittest.mock.call(123,'SIGKILL')])
        result=m.stop_remote(Mock(return_value=self.process()),Mock(),self.app)
        self.assertFalse(result['absent']);self.assertTrue(any('remains' in s for s in result['errors']))

    def test_term_failure_still_kills_and_preserves_original_error(self):
        query=Mock(side_effect=[self.process(),self.process(),dict(result=dict(runningProcesses=[]))])
        signal=Mock(side_effect=[RuntimeError('TERM command failed'),None])
        result=m.stop_remote(query,signal,self.app)
        self.assertTrue(result['absent']);self.assertEqual(query.call_count,3)
        self.assertEqual(signal.call_args_list,[unittest.mock.call(123,'SIGTERM'),unittest.mock.call(123,'SIGKILL')])
        self.assertEqual(result['errors'],['SIGTERM: RuntimeError: TERM command failed'])

    def test_cleanup_inventory_failure_is_not_absence(self):
        signal=Mock()
        result=m.stop_remote(Mock(side_effect=RuntimeError('connection lost')),signal,self.app)
        self.assertFalse(result['absent']);self.assertEqual(len(result['errors']),3)
        self.assertTrue(all('connection lost' in s for s in result['errors']))
        signal.assert_not_called()

    def test_nested_cleanup_errors_cannot_publish_private_identifiers(self):
        message='query --device '+self.config['device_identifier']+' udid '+self.config['device_udid']+' DEVELOPMENT_TEAM='+self.config['development_team']
        private=dict(remote_cleanup=[dict(errors=[message],nested={'command':message})],commands=[[message]])
        public=m.redact_record(private,self.config,self.app)
        text=json.dumps(public)
        for key in ('device_identifier','device_udid','development_team'):
            self.assertNotIn(self.config[key],text)
            self.assertIn('<private-'+key+'>',text)
        self.assertIn(self.config['device_identifier'],json.dumps(private))

    def test_remote_launch_requires_matching_executable_device_and_clean_exit(self):
        data=dict(info=dict(outcome='success'),result=dict(deviceIdentifier='device-private',
            process=dict(processIdentifier=123,executable='file:///private'+self.app+'/TokletHarness'),
            terminationResult=dict(exitCode=0,wasCoreDumpCreated=False)))
        self.assertEqual(m.check_launch(data,self.app,self.config)['pid'],123)
        for termination in ({'exitCode':1,'wasCoreDumpCreated':False},{'signal':15,'wasCoreDumpCreated':False},
                            {'exitCode':0,'wasCoreDumpCreated':True},{'exitCode':0},
                            {'exitCode':0,'wasCoreDumpCreated':False,'signal':15}):
            data['result']['terminationResult']=termination
            with self.assertRaises(ValueError):m.check_launch(data,self.app,self.config)
        del data['result']['terminationResult']
        with self.assertRaises(KeyError):m.check_launch(data,self.app,self.config)

    def test_replaced_or_missing_install_is_zero_launch(self):
        row=dict(bundleIdentifier=m.BUNDLE,builtByDeveloper=True,url='file:///private'+self.app+'/')
        data=dict(info=dict(outcome='success'),result=dict(deviceIdentifier='device-private',matchingBundleIdentifier=m.BUNDLE,apps=[row]))
        launch=Mock(return_value='launched')
        self.assertEqual(m.launch_verified(lambda:data,launch,self.app,self.config),'launched')
        launch.reset_mock()
        for rows in ([],[row,row],[dict(row,url='file:///var/containers/Bundle/Application/replaced/TokletHarness.app/')]):
            data['result']['apps']=rows
            with self.assertRaisesRegex(ValueError,'refuse launch'):m.launch_verified(lambda:data,launch,self.app,self.config)
            launch.assert_not_called()

    def test_installed_path_cannot_redirect_to_another_app(self):
        data=dict(info=dict(outcome='success'),result=dict(deviceIdentifier='device-private',installedApplications=[dict(bundleID=m.BUNDLE,installationURL='file:///private'+self.app+'/')]))
        self.assertEqual(m.installed_path(data,self.config),self.app)
        data['result']['installedApplications'][0]['installationURL']='file:///var/other.app'
        with self.assertRaisesRegex(ValueError,'location'):m.installed_path(data,self.config)
        data['result']['deviceIdentifier']='another-device'
        with self.assertRaisesRegex(ValueError,'device differs'):m.installed_path(data,self.config)

    def console(self,backend='metal',simulator=False,files=None):
        files=files or {'TokletHarness':'hash'}
        states=[dict(zip(('name','history','evaluated','pending','saved','reason'),s)) for s in m.STATES]
        result=dict(passed=True,backend=backend,states=states,tokens=[47]*4,closed=True,released=True,physicalDeviceQualified=False)
        proofs=[dict(phase=p,simulator=simulator,files=files) for p in ('before','after')]
        return '\n'.join(['TOKLET_INSTALLATION_JSON='+json.dumps(p) for p in proofs]+['TOKLET_SELFTEST_JSON='+json.dumps(result)])

    def test_installed_proofs_require_exact_source_backend_and_physical_execution(self):
        files={'TokletHarness':'hash'};text=self.console()
        self.assertEqual(m.check_console(text,'metal',files)['result']['tokens'],[47]*4)
        for changed in (self.console('cpu'),self.console(simulator=True),text.replace('hash','changed'),text.replace('"pending": 1','"pending": 0')):
            with self.assertRaises(ValueError):m.check_console(changed,'metal',files)
        with self.assertRaises(ValueError):m.check_console(text+'\n'+text,'metal',files)
        with self.assertRaisesRegex(ValueError,'reported an error'):
            m.check_console(text+'\nTOKLET_SELFTEST_ERROR=failed','metal',files)

    def test_signed_inventory_rejects_extra_and_symlink_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            app=Path(tmp)
            names=['TokletHarness','Info.plist','PkgInfo','Tiny/config.json','Tiny/model.safetensors','embedded.mobileprovision','_CodeSignature/CodeResources']
            for name in names:
                p=app/name;p.parent.mkdir(exist_ok=True);p.write_bytes(b'data')
            self.assertEqual(len(m.inventory(app)),7)
            extra=app/'extra';extra.write_bytes(b'foreign')
            with self.assertRaisesRegex(ValueError,'inventory'):m.inventory(app)
            extra.unlink();extra.symlink_to(app/'TokletHarness')
            with self.assertRaisesRegex(ValueError,'symlink'):m.inventory(app)

    def test_changed_private_selection_is_zero_launch_and_writes_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();art=root/'artifacts';art.mkdir();config=art/'private-selection.json';config.write_text(json.dumps(self.config))
            with patch.object(m,'ROOT',root),patch.object(m.package,'ROOT',root),patch.object(m.package,'build_environment',return_value=({},[])),patch.object(m.package,'owned') as launch,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(m.main(art/'native','unused',config,'incorrect',art/'output'),1)
            launch.assert_not_called()
            receipt=json.loads((art/'output/integration.json').read_text())
            self.assertFalse(receipt['passed']);self.assertIn('Metadata SHA differs',receipt['primary_error'])

    def test_reuse_binds_historical_controller_and_unchanged_signed_app(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();folder=root/'artifacts/original';folder.mkdir(parents=True)
            app=folder/'derived/Build/Products/Release-iphoneos/TokletHarness.app';app.mkdir(parents=True)
            for name in ('TokletHarness','Info.plist','PkgInfo','Tiny/config.json','Tiny/model.safetensors','embedded.mobileprovision','_CodeSignature/CodeResources'):
                p=app/name;p.parent.mkdir(exist_ok=True);p.write_bytes(b'original')
            controller='tools/experiments/m7_physical_integration.py';snapshot=folder/'source-snapshot'/controller
            snapshot.parent.mkdir(parents=True);snapshot.write_bytes(b'original controller')
            current=root/controller;current.parent.mkdir(parents=True);current.write_bytes(b'updated controller')
            source=root/'apple/Harness/HarnessApp.swift';source.parent.mkdir(parents=True);source.write_bytes(b'original app')
            install=folder/'private-logs/install.stdout';install.parent.mkdir()
            install.write_text(json.dumps(dict(info=dict(outcome='success'),result=dict(deviceIdentifier='device-private',
                installedApplications=[dict(bundleID=m.BUNDLE,installationURL='file:///private'+self.app+'/')]))))
            record=dict(format='toklet-m7-physical-tiny-integration-v1',reviewed_native_package_sha256='native',signing={'verified':True},
                bindings={controller:m.package.sha(snapshot),str(source.relative_to(root)):m.package.sha(source)},
                private_evidence_pins={str(install.relative_to(root)):m.package.sha(install)},signed_bundle=m.inventory(app))
            receipt=folder/'integration.json';receipt.write_text(json.dumps(record));expected=m.package.sha(receipt)
            with patch.object(m,'ROOT',root):
                actual,pins,remote,_=m.admit_reuse(folder,expected,self.config,'native')
                self.assertEqual(actual,app);self.assertEqual(remote,self.app)
                self.assertNotIn(controller,pins);self.assertIn(str(snapshot.relative_to(root)),pins)
                record['format']='toklet-m7-physical-prepared-v1';receipt.write_text(json.dumps(record));expected=m.package.sha(receipt)
                self.assertEqual(m.admit_reuse(folder,expected,self.config,'native')[0],app)
                with self.assertRaisesRegex(ValueError,'SHA differs'):m.admit_reuse(folder,'wrong',self.config,'native')
                source.write_bytes(b'changed app')
                with self.assertRaisesRegex(ValueError,'Reused input'):m.admit_reuse(folder,expected,self.config,'native')
                source.write_bytes(b'original app');(app/'TokletHarness').write_bytes(b'changed signed executable')
                with self.assertRaisesRegex(ValueError,'signed app changed'):m.admit_reuse(folder,expected,self.config,'native')

    def test_prepare_only_never_launches_and_retains_cleanup_or_endpoint_failure(self):
        for failure in (None,'cleanup','endpoint'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack:
                root=Path(tmp).resolve();art=root/'artifacts';art.mkdir();output=art/'prepared'
                source=root/'tools/experiments/controller.py';source.parent.mkdir(parents=True);source.write_text('controller')
                test=source.with_name('test_m7_physical_integration.py');test.write_text('tests')
                b=source.with_name('builder.py');b.write_text('builder')
                i=source.with_name('integration.py');i.write_text('integration')
                config=art/'private-selection.json';config.write_text(json.dumps(self.config))
                files={'TokletHarness':'digest'}
                def owned(command,stdout,stderr,env,timeout):
                    label=stdout.stem
                    if label.startswith('device-'):
                        text=json.dumps(self.device)
                        if label=='device-after' and failure=='endpoint':source.write_text('changed')
                    elif label.startswith('processes-'):text=json.dumps(dict(result=dict(runningProcesses=[])))
                    elif label=='install':text=json.dumps(dict(info=dict(outcome='success'),result=dict(deviceIdentifier='device-private',
                        installedApplications=[dict(bundleID=m.BUNDLE,installationURL='file:///private'+self.app+'/')])))
                    elif label=='architecture':text='arm64'
                    elif label in ('provisioning','entitlements'):text=plistlib.dumps({}).decode()
                    else:text=''
                    stdout.write_text(text);stderr.write_text('')
                    return dict(command=command,exit_code=0,primary_error=None,cleanup_errors=[],stdout=str(stdout.relative_to(root)),stderr=str(stderr.relative_to(root)))
                for p in (patch.object(m,'ROOT',root),patch.object(m.package,'ROOT',root),patch.object(m,'__file__',str(source)),
                    patch.object(m.builder,'__file__',str(b)),patch.object(m.integration,'__file__',str(i)),
                    patch.object(m.package,'build_environment',return_value=({},[])),patch.object(m.integration,'admit_build',return_value={}),
                    patch.object(m.integration,'FIXTURE',{}),patch.object(m.builder,'prepare',return_value={}),
                    patch.object(m.builder,'bind_library_link',return_value={}),patch.object(m.builder,'verify_library_link'),
                    patch.object(m.builder,'inspect_app',side_effect=lambda app,platform:app/'TokletHarness'),patch.object(m.package,'check_load_commands'),
                    patch.object(m,'admit_profile',return_value={'verified':True}),patch.object(m,'inventory',return_value=files),
                    patch.object(m.package,'owned',side_effect=owned),patch.object(m,'stop_remote',return_value=dict(absent=True,observations=[[]],signals=[],errors=['cleanup failed'] if failure=='cleanup' else []))):
                    stack.enter_context(p)
                with contextlib.redirect_stdout(io.StringIO()):
                    code=m.main(art/'native','unused',config,m.package.sha(config),output,prepare_only=True)
                receipt=json.loads((output/'integration.json').read_text())
                self.assertEqual(code,0 if failure is None else 1)
                self.assertEqual(receipt['format'],'toklet-m7-physical-prepared-v1');self.assertEqual(receipt['results'],{})
                self.assertFalse(any('launch' in row['command'] for row in receipt['commands']))
                self.assertFalse(receipt['physical_device_qualified'])
                if failure=='cleanup':self.assertIn('cleanup failed',receipt['cleanup_errors'])
                if failure=='endpoint':self.assertTrue(any('Changed endpoint' in s for s in receipt['endpoint_errors']))


if __name__=='__main__':unittest.main()
