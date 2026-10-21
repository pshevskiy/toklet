import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import m7_apple_integration as m


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.folder=self.root/'artifacts/build';self.folder.mkdir(parents=True)
        (self.folder/'logs').mkdir()
        self.source=self.root/'integration.py';self.source.write_text('fixture')
        self.payload=self.root/'payload';self.payload.write_bytes(b'frozen')
        rows=[]
        for spec in m.package.SPECS:
            executable=self.folder/(spec['target']+'-link-smoke');executable.write_bytes(b'executable')
            library=self.folder/'target'/spec['target']/'release/libtoklet.a';library.parent.mkdir(parents=True);library.write_bytes(b'library')
            rows.append(dict(spec,link_executable=str(executable.relative_to(self.root)),link_sha256=m.package.sha(executable),library=str(library.relative_to(self.root)),library_sha256=m.package.sha(library)))
        self.build=dict(passed=True,format='toklet-m7-apple-package-build-v1',source_pins={},package_pins={},slices=rows)
        self.write_build()
        for p in (patch.object(m.package,'ROOT',self.root),patch.object(m,'__file__',str(self.source)),patch.object(m,'FIXTURE',{'payload':m.package.sha(self.payload)}),patch.object(m,'admit_simulator',return_value={})):
            p.start();self.addCleanup(p.stop)

    def write_build(self):
        path=self.folder/'build.json';path.write_text(json.dumps(self.build));self.expected=m.package.sha(path)

    def run_main(self):
        with contextlib.redirect_stdout(io.StringIO()):return m.main(self.folder,'explicit-simulator',self.expected)

    def test_mutated_frozen_input_prevents_launch(self):
        self.payload.write_bytes(b'changed')
        with patch.object(m.package,'owned') as launch:self.assertEqual(self.run_main(),1)
        launch.assert_not_called()
        self.assertIn('Preflight changed',json.loads((self.folder/'integration-v3.json').read_text())['primary_error'])

    def test_unreviewed_build_metadata_prevents_launch(self):
        (self.folder/'build.json').write_text('{"passed":true}')
        with patch.object(m.package,'owned') as launch:self.assertEqual(self.run_main(),1)
        launch.assert_not_called()

    def test_missing_or_remapped_executable_prevents_launch(self):
        for altered in ([],[dict(r,link_executable='different') for r in self.build['slices']]):
            self.build['slices']=altered;self.write_build()
            with patch.object(m.package,'owned') as launch:self.assertEqual(self.run_main(),1)
            launch.assert_not_called();(self.folder/'integration-v3.json').unlink()

    def test_conflicting_pins_are_rejected(self):
        self.build['source_pins']['same']='first';self.build['package_pins']['same']='second';self.write_build()
        with self.assertRaisesRegex(ValueError,'Conflicting binding'):m.admit_build(self.folder,self.expected)

    def test_failed_worker_preserves_failure_and_endpoint_change(self):
        def fail(*args):
            self.payload.write_bytes(b'changed')
            return dict(primary_error='original timeout',cleanup_errors=[],exit_code=-15)
        with patch.object(m.package,'build_environment',return_value=({},[])),patch.object(m.package,'owned',side_effect=fail),patch.object(m,'shutdown_simulator',return_value=dict(errors=[],shutdown_verified=True)) as cleanup:
            self.assertEqual(self.run_main(),1)
        result=json.loads((self.folder/'integration-v3.json').read_text())
        cleanup.assert_called_once()
        self.assertIn('original timeout',result['primary_error']);self.assertEqual(result['endpoint_errors'],['Changed endpoint: payload'])

    def test_simulator_failure_always_cleans_and_preserves_both_errors(self):
        def worker(command,stdout,*args):
            if command[:3]==['xcrun','simctl','spawn']:return dict(primary_error='simulator timeout',cleanup_errors=[],exit_code=-15)
            stdout.write_text(json.dumps(dict(ids=[47]*4,abi=1,passed=True)))
            return dict(primary_error=None,cleanup_errors=[],exit_code=0,stdout=str(stdout.relative_to(self.root)))
        with patch.object(m.package,'build_environment',return_value=({},[])),patch.object(m.package,'owned',side_effect=worker),patch.object(m,'shutdown_simulator',return_value=dict(errors=['shutdown timeout'],shutdown_verified=False)) as cleanup:
            self.assertEqual(self.run_main(),1)
        cleanup.assert_called_once()
        result=json.loads((self.folder/'integration-v3.json').read_text())
        self.assertIn('simulator timeout',result['primary_error']);self.assertEqual(result['simulator_cleanup']['errors'],['shutdown timeout'])

    def test_shutdown_requires_actual_shutdown_state(self):
        def worker(command,stdout,*args):
            stdout.write_text(json.dumps(dict(devices={'runtime':[dict(udid='explicit-simulator',state='Booted')]})))
            return dict(primary_error=None,cleanup_errors=[],exit_code=0,stdout=str(stdout.relative_to(self.root)))
        with patch.object(m.package,'owned',side_effect=worker):result=m.shutdown_simulator(self.folder,'explicit-simulator',{})
        self.assertFalse(result['shutdown_verified']);self.assertIn('Owned simulator shutdown not verified',result['errors'])


if __name__=='__main__':unittest.main()
