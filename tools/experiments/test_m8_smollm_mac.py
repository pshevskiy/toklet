"""Collector boundary mutations; no model download, compiler or GPU."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import m8_smollm_mac as p

class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=p.R/'artifacts');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def build(self):
        return dict(format='toklet-smollm-mac-runner-build-v1',passed=True,primary_error=None,endpoint_errors=[],native_package_sha256=p.NATIVE_SHA,
                    commands=[dict(exit_code=0,primary_error=None,cleanup_errors=[]) for _ in range(4)],bindings={},binary_pins={})
    def test_capture_hashes_and_returns_same_bounded_bytes(self):
        path=self.root/'input';path.write_bytes(b'original');pin=hashlib.sha256(b'original').hexdigest()
        self.assertEqual(p.capture(path,pin,8),b'original')
        with self.assertRaises(ValueError):p.capture(path,pin,7)
        path.write_bytes(b'changed!')
        with self.assertRaises(ValueError):p.capture(path,pin,8)
    def test_conflicting_compiled_and_current_bindings_cannot_overwrite(self):
        pins={'compiled':'a'*64}
        with self.assertRaises(ValueError):p.merge(pins,{'compiled':'b'*64})
        self.assertEqual(pins,{'compiled':'a'*64})
        build=self.build();build['binary_pins']={'same':'b'*64};build['bindings']={'same':'a'*64}
        with self.assertRaises(ValueError):p.admit_build(json.dumps(build))
    def test_build_admission_requires_success_exact_native_and_all_commands(self):
        self.assertEqual(p.admit_build(json.dumps(self.build()))[1],{})
        for key,value in [('format','other'),('passed',False),('primary_error','bad'),('endpoint_errors',['bad']),('native_package_sha256','0'*64),('commands',[])]:
            build=self.build();build[key]=value
            with self.assertRaises(ValueError):p.admit_build(json.dumps(build))
        for key,value in [('exit_code',1),('primary_error','timeout'),('cleanup_errors',['remaining child'])]:
            build=self.build();build['commands'][0][key]=value
            with self.assertRaises(ValueError):p.admit_build(json.dumps(build))
    def test_complete_root_preflight_rejects_child_collision_and_dangling_link(self):
        root=self.root/'old';root.mkdir();(root/'state-component').mkdir()
        with self.assertRaises(FileExistsError):p.preflight(root)
        link=self.root/'link';link.symlink_to(self.root/'missing')
        with self.assertRaises(FileExistsError):p.preflight(link)
        self.assertFalse((self.root/'missing').exists())
    def test_diagnostic_capture_keeps_stderr_when_stdout_is_missing(self):
        stdout=self.root/'missing.stdout';stderr=self.root/'stderr';stderr.write_bytes(b'original stderr')
        row=dict(stdout=str(stdout.relative_to(p.R)),stderr=str(stderr.relative_to(p.R)),primary_error='owned timeout',cleanup_errors=['owned child error'])
        original=copy.deepcopy(row);buffers,captures,errors=p.diagnostics(row)
        self.assertEqual(row,original);self.assertEqual(buffers,{'stderr':b'original stderr'})
        self.assertEqual(captures[0]['sha256'],hashlib.sha256(b'original stderr').hexdigest());self.assertEqual(len(errors),1)
    def test_state_framing_rejects_any_second_line_or_marker(self):
        self.assertEqual(p.strict_state('TOKLET_STATE_JSON={"x":1}\n'),{'x':1})
        for text in ['noise\nTOKLET_STATE_JSON={}','TOKLET_STATE_JSON={}\nTOKLET_OTHER={}','TOKLET_STATE_JSON={}\nTOKLET_STATE_JSON={}','']:
            with self.assertRaises(ValueError):p.strict_state(text)
    def test_main_timeout_retains_owned_error_missing_stdout_and_stderr_pin(self):
        build=self.build();original=p.capture
        def capture(path,pin,maximum):
            if path==p.R/p.BUILD/'build.json':return json.dumps(build).encode()
            if path==p.R/p.jobs.SMOLLM_REFERENCE:return b'{}'
            if path==p.R/p.ARRAYS:return b''
            return original(path,pin,maximum)
        def stage(model,workload,folder):
            folder.mkdir();(folder/'transfer').mkdir()
            record=dict(bindings={},job=dict(relative_path='job',sha256='0'*64),config={},mode='measurement',execution=p.jobs.SMOLLM_EXECUTION,staged_files={})
            (folder/'staged.json').write_text(json.dumps(record));return record
        def owned(args,stdout,stderr,env,timeout):
            stderr.write_text('original timeout diagnostic')
            return dict(command=args,exit_code=None,primary_error='owned timeout',cleanup_errors=['original cleanup'],stdout=str(stdout.relative_to(p.R)),stderr=str(stderr.relative_to(p.R)),pid=1)
        with patch.object(p,'capture',side_effect=capture),patch.object(p.jobs,'stage',side_effect=stage),patch.object(p.p,'owned',side_effect=owned):
            record=p.main(self.root/'run')
        self.assertFalse(record['passed']);self.assertIn('owned timeout',record['primary_error']);self.assertEqual(record['commands'][0]['cleanup_errors'],['original cleanup'])
        self.assertEqual(len(record['capture_errors']),1);self.assertTrue(record['captured_diagnostics'][0]['path'].endswith('.stderr'))

if __name__=='__main__':unittest.main()
