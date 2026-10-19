import contextlib
import io
import json
from pathlib import Path
import plistlib
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import m7_swift_harness_build as m


class HarnessBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();(self.root/'apple').mkdir();(self.root/'artifacts').mkdir()
        self.package=self.root/'artifacts/package';self.package.mkdir();(self.package/'Toklet.xcframework').mkdir()
        self.fixture=self.root/'artifacts/model.safetensors';self.fixture.write_bytes(b'fixture')
        for p in (patch.object(m,'ROOT',self.root),patch.object(m.package,'ROOT',self.root),patch.object(m.integration,'FIXTURE',{'artifacts/model.safetensors':m.package.sha(self.fixture)})):
            p.start();self.addCleanup(p.stop)

    def test_foreign_library_is_never_replaced(self):
        folder=self.root/'apple/Binaries';folder.mkdir();foreign=folder/'Toklet.xcframework';foreign.write_bytes(b'foreign')
        with self.assertRaisesRegex(ValueError,'foreign library'):m.prepare(self.package)
        self.assertEqual(foreign.read_bytes(),b'foreign')

    def test_library_link_redirection_is_detected(self):
        m.prepare(self.package);binding=m.bind_library_link()
        m.verify_library_link(binding)
        link=self.root/binding['path'];link.unlink();link.symlink_to(self.root/'artifacts/foreign')
        with self.assertRaisesRegex(ValueError,'link changed'):m.verify_library_link(binding)

    def test_fixture_mutation_prevents_copy(self):
        self.fixture.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Frozen tiny fixture'):m.prepare(self.package)
        self.assertFalse((self.root/'apple/Generated/Tiny/model.safetensors').exists())

    def test_generated_symlink_cannot_redirect_fixture(self):
        generated=self.root/'apple/Generated';generated.mkdir();(generated/'Tiny').symlink_to(self.root/'artifacts',target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'fixture directory link'):m.prepare(self.package)
        self.assertEqual(self.fixture.read_bytes(),b'fixture')

    def test_binary_plist_identity_and_executable_escape_are_checked(self):
        app=self.root/'app';app.mkdir();(app/'Tiny').mkdir();(app/'Tiny/model.safetensors').write_bytes(b'fixture')
        info=dict(CFBundleExecutable='Harness',CFBundleIdentifier='dev.toklet.harness',MinimumOSVersion='18.0',CFBundleSupportedPlatforms=['iPhoneSimulator'])
        def write(): (app/'Info.plist').write_bytes(plistlib.dumps(info,fmt=plistlib.FMT_BINARY))
        write();self.assertEqual(m.inspect_app(app,'iPhoneSimulator'),app/'Harness')
        for key,value in [('CFBundleExecutable','../foreign'),('MinimumOSVersion','17.0'),('CFBundleSupportedPlatforms',['iPhoneOS'])]:
            previous=info[key];info[key]=value;write()
            with self.assertRaises(ValueError):m.inspect_app(app,'iPhoneSimulator')
            info[key]=previous

    def test_oversize_metadata_uses_one_bounded_read(self):
        stream=Mock();stream.read.return_value=b'x'*(1024**2+1)
        stream.__enter__=Mock(return_value=stream);stream.__exit__=Mock(return_value=False)
        with patch.object(Path,'open',return_value=stream),self.assertRaises(ValueError):m.read_plist(self.fixture)
        stream.read.assert_called_once_with(1024**2+1)

    def test_unreviewed_build_is_zero_launch_and_has_failed_receipt(self):
        output=self.root/'artifacts/output'
        with patch.object(m.package,'build_environment',return_value=({},[])),patch.object(m.integration,'admit_build',side_effect=ValueError('SHA differs')),patch.object(m.package,'owned') as launch,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main(self.package,'not-reviewed',output),1)
        launch.assert_not_called();result=json.loads((output/'build.json').read_text())
        self.assertFalse(result['passed']);self.assertIn('SHA differs',result['primary_error'])


if __name__=='__main__':unittest.main()
