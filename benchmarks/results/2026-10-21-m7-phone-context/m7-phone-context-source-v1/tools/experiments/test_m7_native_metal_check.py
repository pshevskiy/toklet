import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

spec=importlib.util.spec_from_file_location('m7check',Path(__file__).with_name('m7_native_metal_check.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name);log=root/'logs';log.mkdir()
        for name,value in [('ROOT',root),('LOG',log),('OUTPUT',log/'result.json')]:
            p=patch.object(m,name,value);p.start();self.addCleanup(p.stop)

    def test_real_timeout_is_reaped_and_primary_preserved(self):
        result=m.run([sys.executable,'-c','import time; time.sleep(30)'],'timeout',timeout=.05)
        self.assertIn('TimeoutExpired',result['primary_error'])
        self.assertIsNotNone(result['exit_code'])
        self.assertEqual(result['cleanup_errors'],[])
        self.assertTrue((m.ROOT/result['stdout']).exists())

    def test_natural_exit_race_does_not_skip_bounded_reap(self):
        p=Mock(pid=12345);p.poll.return_value=None
        p.wait.side_effect=[subprocess.TimeoutExpired('original',1),0]
        with patch.object(m.subprocess,'Popen',return_value=p),patch.object(m.os,'killpg',side_effect=ProcessLookupError):
            r=m.run(['fixture'],'race',timeout=1)
        self.assertIn('original',r['primary_error']);self.assertEqual(r['exit_code'],0)
        self.assertEqual(r['cleanup_errors'],[]);self.assertEqual(p.wait.call_args_list[-1].kwargs,{'timeout':15})

    def test_final_reap_timeout_keeps_original_and_records_cleanup(self):
        p=Mock(pid=12345);p.poll.return_value=None
        p.wait.side_effect=[subprocess.TimeoutExpired('primary',1),subprocess.TimeoutExpired('term',15),subprocess.TimeoutExpired('kill',15)]
        with patch.object(m.subprocess,'Popen',return_value=p),patch.object(m.os,'killpg') as kill:
            r=m.run(['fixture'],'reap',timeout=1)
        self.assertIn('primary',r['primary_error']);self.assertEqual(len(r['cleanup_errors']),1)
        self.assertEqual([x.args[1] for x in kill.call_args_list],[signal.SIGTERM,signal.SIGKILL])
        self.assertEqual([x.kwargs['timeout'] for x in p.wait.call_args_list],[1,15,15])

    def test_changed_frozen_payload_is_rejected_before_any_launch(self):
        path=m.ROOT/'model';path.write_bytes(b'old')
        expected={'model':hashlib.sha256(b'old').hexdigest()};path.write_bytes(b'changed')
        with patch.object(m,'run') as launch,patch.object(m,'preflight',side_effect=lambda: m.verify_frozen([path],expected)),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main(),1)
        launch.assert_not_called();d=json.loads(m.OUTPUT.read_text())
        self.assertFalse(d['passed']);self.assertIn('Frozen payload differs',d['primary_error'])

    def test_endpoint_failure_is_recorded_even_after_worker_failure(self):
        path=m.ROOT/'source';path.write_bytes(b'before');pins={'source':m.sha(path)}
        def fail(*args):
            path.write_bytes(b'changed')
            return dict(primary_error='TimeoutExpired: original',cleanup_errors=[],exit_code=-15)
        with patch.object(m,'preflight',return_value=(pins,Path('/fixture'))),patch.object(m,'run',side_effect=fail),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main(),1)
        d=json.loads(m.OUTPUT.read_text());self.assertFalse(d['passed'])
        self.assertIn('TimeoutExpired',d['workers'][0]['primary_error'])
        self.assertEqual(d['endpoint_errors'],['Endpoint changed: source'])

    def test_metadata_hash_and_parser_use_same_bounded_buffer(self):
        p=m.ROOT/'meta';p.write_bytes(b'{"original":true}')
        expected=hashlib.sha256(p.read_bytes()).hexdigest()
        self.assertEqual(m.metadata(p,expected)[0],{'original':True})
        p.write_bytes(b'{"replacement":true}')
        with self.assertRaises(ValueError):m.metadata(p,expected)
        stream=Mock();stream.read.return_value=b'x'*(16*1024**2+1)
        stream.__enter__=Mock(return_value=stream);stream.__exit__=Mock(return_value=False)
        with patch.object(Path,'open',return_value=stream),self.assertRaises(ValueError):m.metadata(p)
        stream.read.assert_called_once_with(16*1024**2+1)


if __name__=='__main__':unittest.main()
