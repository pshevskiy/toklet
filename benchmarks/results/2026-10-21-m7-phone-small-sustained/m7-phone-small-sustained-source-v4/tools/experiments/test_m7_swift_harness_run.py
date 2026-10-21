import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import m7_swift_harness_run as m


class HarnessRunTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.folder=self.root/'artifacts/build';self.folder.mkdir(parents=True)
        self.app=self.folder/'derived-iphonesimulator/Build/Products/Release-iphonesimulator/TokletHarness.app';self.app.mkdir(parents=True)
        self.binary=self.app/'TokletHarness';self.binary.write_bytes(b'code')
        self.source=self.root/'runner.py';self.source.write_text('runner');self.source.with_name('test_m7_swift_harness_run.py').write_text('tests')
        executable=str(self.binary.relative_to(self.root))
        self.build=dict(passed=True,format='toklet-m7-swift-harness-build-v1',bindings={},bundle_pins={executable:m.package.sha(self.binary)},
                        library_link={},apps=[dict(platform='iPhoneSimulator',app=str(self.app.relative_to(self.root)),executable=executable,executable_sha256=m.package.sha(self.binary),signed='ad-hoc'),dict(platform='iPhoneOS')])
        self.write()
        for p in (patch.object(m,'ROOT',self.root),patch.object(m.package,'ROOT',self.root),patch.object(m,'__file__',str(self.source)),patch.object(m.builder,'verify_library_link',return_value=None)):
            p.start();self.addCleanup(p.stop)

    def write(self):
        path=self.folder/'build.json';path.write_text(json.dumps(self.build));self.expected=m.package.sha(path)

    def test_omitted_or_remapped_app_is_rejected(self):
        self.build['apps'][0]['executable']='foreign';self.write()
        with self.assertRaisesRegex(ValueError,'Remapped'):m.admit(self.folder,self.expected)
        self.build['apps']=[];self.write()
        with self.assertRaisesRegex(ValueError,'inventory'):m.admit(self.folder,self.expected)

    def test_conflicting_executable_binding_is_rejected(self):
        self.build['apps'][0]['executable_sha256']='different';self.write()
        with self.assertRaisesRegex(ValueError,'Conflicting'):m.admit(self.folder,self.expected)

    def test_installed_bundle_copy_is_verified(self):
        installed=self.root/'installed';installed.mkdir();(installed/'TokletHarness').write_bytes(b'code')
        _,pins,_=m.admit(self.folder,self.expected)
        self.assertEqual(m.installed_copy(self.app,installed,pins),{str(installed/'TokletHarness'):m.package.sha(self.binary)})
        (installed/'TokletHarness').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Installed app differs'):m.installed_copy(self.app,installed,pins)

    def test_changed_build_is_zero_launch(self):
        (self.folder/'build.json').write_text('{"changed":true}')
        with patch.object(m.package,'build_environment',return_value=({},[])),patch.object(m.package,'owned') as launch,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main(self.folder,self.expected,'simulator',self.root/'artifacts/run'),1)
        launch.assert_not_called()

    def test_boot_failure_cleans_owned_simulator_and_preserves_errors(self):
        with patch.object(m.package,'build_environment',return_value=({},[])),patch.object(m.integration,'admit_simulator',return_value={}),patch.object(m.package,'owned',return_value=dict(primary_error='original boot timeout',cleanup_errors=[],exit_code=-15)),patch.object(m.integration,'shutdown_simulator',return_value=dict(errors=['cleanup timeout'],shutdown_verified=False)) as cleanup,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main(self.folder,self.expected,'simulator',self.root/'artifacts/run'),1)
        cleanup.assert_called_once();result=json.loads((self.root/'artifacts/run/run.json').read_text())
        self.assertIn('original boot timeout',result['primary_error']);self.assertEqual(result['simulator_cleanup']['errors'],['cleanup timeout'])


if __name__=='__main__':unittest.main()
