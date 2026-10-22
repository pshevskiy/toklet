"""Mocked installed-identity reset; no device commands."""
import copy
import unittest
import m8_physical_reset as p

class Reset(unittest.TestCase):
    def setUp(self):
        self.selected={'device_identifier':'selected'};self.remote='/var/containers/Bundle/Application/test/TokletHarness.app'
        self.apps=dict(info=dict(outcome='success',commandType='devicectl.device.info.apps'),result=dict(deviceIdentifier='selected',matchingBundleIdentifier=p.physical.BUNDLE,
            apps=[dict(bundleIdentifier=p.physical.BUNDLE,builtByDeveloper=True,url='file://'+self.remote)]))
        self.running=True;self.sent=[]
    def current(self):return copy.deepcopy(self.apps)
    def query(self):
        return dict(info=dict(outcome='success',commandType='devicectl.device.info.processes'),result=dict(deviceIdentifier='selected',runningProcesses=[dict(executable=self.remote+'/TokletHarness',processIdentifier=123)] if self.running else []))
    def send(self,pid,sig):
        self.sent.append((pid,sig));self.running=False
        return dict(info=dict(outcome='success',commandType='devicectl.device.process.signal'),result=dict(deviceIdentifier='selected',process=dict(processIdentifier=pid,executable=self.remote+'/TokletHarness'),signal=dict(name=sig,value=15 if sig=='SIGTERM' else 9)))
    def test_exact_identity_reset_and_natural_absence(self):
        result,remote=p.reset(self.current,self.query,self.send,self.selected)
        self.assertEqual(remote,self.remote);self.assertTrue(result['absent']);self.assertFalse(result['errors'])
        self.assertEqual(self.sent,[(123,'SIGTERM')]);self.sent=[]
        result,_=p.reset(self.current,self.query,self.send,self.selected);self.assertTrue(result['absent']);self.assertFalse(self.sent)
    def test_wrong_install_device_outcome_or_development_flag_is_zero_signal(self):
        for section,key,value in [('info','outcome','failure'),('info','commandType','devicectl.device.info.processes'),('result','deviceIdentifier','other'),('result','matchingBundleIdentifier','other')]:
            original=self.apps[section][key];self.apps[section][key]=value
            with self.assertRaises(ValueError):p.reset(self.current,self.query,self.send,self.selected)
            self.assertFalse(self.sent);self.apps[section][key]=original
        self.apps['result']['apps'][0]['builtByDeveloper']=False
        with self.assertRaises(ValueError):p.reset(self.current,self.query,self.send,self.selected)
        self.assertFalse(self.sent)
    def test_replaced_app_or_reused_pid_never_signals_other_process(self):
        def query():
            data=self.query();data['result']['runningProcesses'][0]['executable']='/other/TokletHarness';return data
        result,_=p.reset(self.current,query,self.send,self.selected)
        self.assertFalse(result['absent']);self.assertTrue(result['errors']);self.assertFalse(self.sent)
        calls=0
        def current():
            nonlocal calls
            calls+=1;data=self.current()
            if calls>1:data['result']['apps'][0]['url']='file:///var/containers/Bundle/Application/replacement/TokletHarness.app'
            return data
        with self.assertRaises(ValueError):p.reset(current,self.query,self.send,self.selected)
        self.assertFalse(self.sent)
    def test_failed_signal_remains_failure_even_after_natural_exit(self):
        def send(pid,sig):
            data=self.send(pid,sig);data['info']['outcome']='failure';return data
        result,_=p.reset(self.current,self.query,send,self.selected)
        self.assertTrue(result['absent']);self.assertTrue(result['errors'])
    def test_wrong_signal_target_and_signal_fields_remain_failures_after_absence(self):
        for mutation in [lambda d:d['info'].update(commandType='devicectl.device.info.apps'),
                         lambda d:d['result']['process'].update(processIdentifier=124),
                         lambda d:d['result']['process'].update(executable='/other/TokletHarness'),
                         lambda d:d['result']['signal'].update(name='SIGKILL'),
                         lambda d:d['result']['signal'].update(value=9),
                         lambda d:d['result']['signal'].update(value=15.0),
                         lambda d:d['result']['signal'].update(extra=1)]:
            self.running=True
            def send(pid,sig):
                d=self.send(pid,sig);mutation(d);return d
            result,_=p.reset(self.current,self.query,send,self.selected)
            self.assertTrue(result['absent']);self.assertTrue(result['errors'])
    def test_details_schema_requires_its_own_identifier_and_command(self):
        d=dict(info=dict(outcome='success',commandType='devicectl.device.info.details'),result=dict(identifier='selected'))
        self.assertEqual(p.response(d,self.selected,'info.details'),d)
        d['result']['identifier']='other'
        with self.assertRaises(ValueError):p.response(d,self.selected,'info.details')
    def test_dangling_output_symlink_rejects_before_child_or_publication(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=p.ROOT/'artifacts') as tmp:
            root=Path(tmp);out=root/'out';out.symlink_to(root/'missing')
            with patch.object(p.package,'owned') as owned,self.assertRaises(FileExistsError):
                p.run(root/'selection','0'*64,out)
            owned.assert_not_called();self.assertFalse((root/'missing').exists())
    def test_pid_rechecked_immediately_before_signal(self):
        calls=0
        def query():
            nonlocal calls
            calls+=1
            if calls>1:self.running=False
            return self.query()
        result,_=p.reset(self.current,query,self.send,self.selected)
        self.assertTrue(result['absent']);self.assertTrue(result['errors']);self.assertFalse(self.sent)

class ResetRun(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        self.patch=patch;self.tmp=tempfile.TemporaryDirectory(dir=p.ROOT/'artifacts');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.selection=self.root/'selection.json';self.selection.write_text('{}')
        self.selected=dict(device_identifier='selected');self.mode='normal';self.output=self.root/'result';self.calls=[]
    def owned(self,args,stdout,stderr,env,timeout):
        self.calls.append(args);command='.'.join(args[2:4])
        result=dict(deviceIdentifier='selected');data=dict(info=dict(outcome='success',commandType='devicectl.device.'+command),result=result)
        if command=='info.details':result=dict(identifier='selected');data['result']=result
        elif command=='info.apps':
            result.update(matchingBundleIdentifier=p.physical.BUNDLE,apps=[dict(bundleIdentifier=p.physical.BUNDLE,builtByDeveloper=True,url='file:///var/containers/Bundle/Application/test/TokletHarness.app')])
        elif command=='info.processes':result['runningProcesses']=[]
        if self.mode!='missing':stdout.write_text(p.json.dumps(data))
        stderr.write_text('diagnostic retained')
        return dict(command=args,exit_code=None if self.mode=='missing' else 0,
            primary_error='TimeoutExpired: owned deadline' if self.mode=='missing' else None,cleanup_errors=['owned cleanup retained'] if self.mode=='missing' else [],
            stdout=str(stdout.relative_to(p.ROOT)),stderr=str(stderr.relative_to(p.ROOT)),pid=1)
    def run_reset(self):
        with self.patch.object(p.physical,'selection',return_value=self.selected),self.patch.object(p.physical,'admit_device',return_value={}),self.patch.object(p.package,'owned',side_effect=self.owned):
            return p.run(self.selection,p.package.sha(self.selection),self.output)
    def test_selection_replacement_after_admission_is_zero_child_and_retains_original_pin(self):
        original=p.package.sha(self.selection)
        def select(path,pin):
            self.selection.write_text('{"replaced":true}');return self.selected
        with self.patch.object(p.physical,'selection',side_effect=select),self.patch.object(p.package,'owned') as owned:
            result=p.run(self.selection,original,self.output)
        self.assertFalse(result['passed']);owned.assert_not_called()
        self.assertEqual(result['bindings'][str(self.selection.relative_to(p.ROOT))],original)

    def test_success_has_consumed_byte_pins_and_private_original_ledger(self):
        result=self.run_reset();self.assertTrue(result['passed'])
        for name,pin in result['bindings'].items():self.assertEqual(p.package.sha(p.ROOT/name),pin)
        private=p.json.loads((self.output/'private-execution.json').read_text());self.assertFalse(private['capture_errors'])
        self.assertEqual(len(private['captures']),2*len(private['commands']))
    def test_owned_timeout_missing_stdout_preserves_stderr_primary_and_cleanup(self):
        self.mode='missing';result=self.run_reset();self.assertFalse(result['passed'])
        private=p.json.loads((self.output/'private-execution.json').read_text())
        self.assertIn('TimeoutExpired',private['primary_error']);self.assertEqual(private['commands'][0]['cleanup_errors'],['owned cleanup retained'])
        self.assertEqual(len(private['capture_errors']),1);self.assertTrue(any(c['path'].endswith('.stderr') for c in private['captures']))
    def test_replacement_after_capture_never_blesses_new_bytes(self):
        original=p.response;changed=False
        def response(data,selected,command):
            nonlocal changed
            result=original(data,selected,command)
            if not changed:
                changed=True;next((self.output/'private-logs').glob('*.stdout')).write_text('{}')
            return result
        with self.patch.object(p,'response',side_effect=response):result=self.run_reset()
        self.assertFalse(result['passed']);self.assertTrue(result['endpoint_errors'])
        private=p.json.loads((self.output/'private-execution.json').read_text());pin=private['captures'][0]['sha256']
        self.assertNotEqual(p.package.sha(p.ROOT/private['captures'][0]['path']),pin)
    def test_late_mutation_is_a_failed_return_with_immutable_late_cause_journal(self):
        original=p.physical.redact_record
        def redact(record,selected,remote):
            result=original(record,selected,remote);self.selection.write_text('{"changed":true}');return result
        with self.patch.object(p.physical,'redact_record',side_effect=redact):result=self.run_reset()
        self.assertFalse(result['passed']);self.assertTrue(result['endpoint_errors'])
        journal=p.json.loads((self.output/'private-late-failure.json').read_text())
        self.assertTrue(journal['late_errors']);self.assertIsNone(journal['primary_error'])
        self.assertIsNone(p.json.loads((self.output/'private-execution.json').read_text())['primary_error'])

if __name__=='__main__':unittest.main()
