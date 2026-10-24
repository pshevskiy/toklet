"""Owned routing regressions with mocked children."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import gemma_shared_kv_reference as reference


class OwnedRouting(unittest.TestCase):
    def test_bad_source_and_existing_destination_run_no_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();(root/'artifacts').mkdir();source=root/'source';source.write_bytes(b'bad');out=root/'artifacts'/'out'
            with mock.patch.object(reference,'ROOT',root),mock.patch.object(reference.owner,'owned') as child:
                self.assertEqual(reference.collect(source,out),1);child.assert_not_called()
                with self.assertRaises(FileExistsError):reference.collect(source,out)
                with self.assertRaises(ValueError):reference.collect(source,root/'escape')
            result=json.loads((out/'capture.json').read_bytes());self.assertFalse(result['passed']);self.assertEqual(result['commands'],[])

    def test_observation_replacement_after_admission_rejects_before_parse(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'observation.json';path.write_bytes(b'{"passed":false}')
            admitted=reference.pin(path)
            self.assertEqual(reference.admitted_observation(path,admitted),dict(passed=False))
            path.write_bytes(b'{"passed":true}')
            with self.assertRaises(ValueError):reference.admitted_observation(path,admitted)
            path.write_bytes(b'{"passed":false}')
            with self.assertRaisesRegex(ValueError,'size differs'):reference.admitted_observation(path,dict(admitted,bytes=1))

    def test_substitution_after_source_buffer_capture_rejects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();(root/'artifacts').mkdir();source=root/'source';source.write_bytes(b'captured')
            def replace(path,*args):raw=path.read_bytes();path.write_bytes(b'replaced');return raw
            with mock.patch.object(reference,'ROOT',root),mock.patch.object(reference.routing,'SOURCE_SHA',hashlib.sha256(b'captured').hexdigest()),mock.patch.object(reference.routing,'captured',side_effect=replace),mock.patch.object(reference.owner,'owned') as child:
                self.assertEqual(reference.collect(source,root/'artifacts'/'out'),1);child.assert_not_called()
            result=json.loads((root/'artifacts/out/capture.json').read_bytes());self.assertIn('Source buffer changed',result['primary_error'])
            self.assertEqual((root/'artifacts/out/gemma4_text.py').read_bytes(),b'captured')

    def test_failed_or_raised_child_retains_logs_and_endpoint_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();(root/'artifacts').mkdir();source=root/'source';source.write_bytes(b'captured')
            names=['tools/reference/gemma_shared_kv.py','tools/reference/test_gemma_shared_kv.py','tools/experiments/gemma_shared_kv_reference.py','tools/experiments/test_gemma_shared_kv_reference.py','target/release/toklet','target/release/examples/cpu_bench','tools/experiments/m7_apple_package.py']
            for name in names:path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic')
            def failed(command,stdout,stderr,environment,timeout):
                self.assertEqual(list(Path(environment['PYTHONPYCACHEPREFIX']).iterdir()),[])
                stdout.write_bytes(b'partial stdout');stderr.write_bytes(b'partial stderr');source.write_bytes(b'changed')
                return dict(exit_code=1,primary_error='synthetic failure',cleanup_errors=[])
            def raised(*args,**kwargs):failed(*args,**kwargs);raise OSError('synthetic helper failure')
            with mock.patch.object(reference,'ROOT',root),mock.patch.object(reference.owner,'__file__',str(root/'tools/experiments/m7_apple_package.py')),mock.patch.object(reference.routing,'SOURCE_SHA',hashlib.sha256(b'captured').hexdigest()):
                for name,operation in [('failed',failed),('raised',raised)]:
                    source.write_bytes(b'captured');out=root/'artifacts'/name
                    with mock.patch.object(reference.owner,'owned',side_effect=operation):self.assertEqual(reference.collect(source,out),1)
                    result=json.loads((out/'capture.json').read_bytes());self.assertTrue(result['endpoint_errors']);self.assertFalse(result['passed'])
                    self.assertEqual(set(result['outputs']),{'logs/ownership.stdout','logs/ownership.stderr'});self.assertEqual((out/'logs/ownership.stdout').read_bytes(),b'partial stdout')
                    self.assertEqual(len(result['commands']),1)


if __name__=='__main__':unittest.main()
