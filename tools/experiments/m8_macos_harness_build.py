#!/usr/bin/env python3
"""Build a local, ad-hoc signed native Mac window using an admitted Rust package."""
from pathlib import Path
import hashlib
import json
import os
import plistlib
import shutil
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import m7_apple_package as package

TINY = {
    'config.json': '84b6e6078f35006f49ae2e2a8663093cdc8d2813e2b7f28ebfa6142113e9c1d9',
    'model.safetensors': 'b01a114bf2d44f0c6cb438035e85535b658e803c76691729facd78b15eefbce2',
}


def main(folder, expected, destination):
    if os.path.lexists(destination):
        raise FileExistsError('Fresh build output required')
    folder = folder.resolve()
    destination = destination.resolve()
    if not folder.is_relative_to(ROOT / 'artifacts') or not destination.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Expected repository artifact folders')
    if os.path.lexists(destination):
        raise FileExistsError('Fresh build output required')
    with (folder / 'build.json').open('rb') as stream:
        raw = stream.read(16 * 1024**2 + 1)
    if len(raw) > 16 * 1024**2 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Reviewed native package differs')
    native = json.loads(raw)
    if not native['passed'] or native['primary_error'] or native['endpoint_errors']:
        raise ValueError('Successful native package required')
    archive = folder / 'Toklet.xcframework/macos-arm64/libtoklet.a'
    archive_key = str(archive.relative_to(ROOT))
    if package.sha(archive) != native['package_pins'][archive_key]:
        raise ValueError('Native Mac archive differs')
    sources = sorted(x for x in (ROOT / 'apple/Harness').glob('*.swift') if x.name != 'HarnessApp.swift')
    sources += sorted((ROOT / 'apple/Mac').glob('*.swift'))
    inputs = [*sources, ROOT / 'apple/include/toklet.h', ROOT / 'apple/include/module.modulemap',
              Path(__file__).resolve(), ROOT / 'tools/experiments/m7_apple_package.py', folder / 'build.json', archive]
    pins = {str(x.relative_to(ROOT)): package.sha(x) for x in inputs}
    # The consumed receipt buffer remains authoritative, even if its path is
    # replaced between parse and later endpoint checks.
    pins[str((folder / 'build.json').relative_to(ROOT))] = expected
    for name in ['toklet.h', 'module.modulemap']:
        header = archive.parent / 'Headers' / name
        key = str(header.relative_to(ROOT))
        pin = native['package_pins'][key]
        if package.sha(header) != pin or package.sha(ROOT / 'apple/include' / name) != pin:
            raise ValueError('Compiled header differs from admitted native package')
        pins[key] = pin
    tiny_data = {}
    for name, pin in TINY.items():
        source = ROOT / 'artifacts/m7-native-tiny-v2' / name
        with source.open('rb') as stream:
            data = stream.read(1024**2 + 1)
        if len(data) > 1024**2 or hashlib.sha256(data).hexdigest() != pin:
            raise ValueError('Frozen tiny fixture differs')
        pins[str(source.relative_to(ROOT))] = pin
        tiny_data[name] = data
    env, _ = package.build_environment()
    destination.mkdir()
    app = destination / 'Toklet.app'
    contents = app / 'Contents'
    binary = contents / 'MacOS/Toklet'
    resources = contents / 'Resources/Tiny'
    copies = destination / 'source-snapshot'
    copy_pins = {}
    commands = []
    diagnostics = {}
    primary = None
    endpoints = []
    previous = signal.signal(signal.SIGTERM, package.interrupted)

    def guard():
        for name, pin in {**pins, **copy_pins, **diagnostics}.items():
            if package.sha(ROOT / name) != pin:
                raise ValueError('Bound input differs: ' + name)

    def execute(label, command):
        guard()
        record = package.owned(command, destination / (label + '.stdout'), destination / (label + '.stderr'), env, 600)
        commands.append(record)
        capture_errors = []
        for field in ['stdout', 'stderr']:
            try:
                path = ROOT / record[field]
                with path.open('rb') as stream:
                    data = stream.read(8 * 1024**2 + 1)
                if len(data) > 8 * 1024**2:
                    raise ValueError('Diagnostic limit exceeded')
                diagnostics[record[field]] = hashlib.sha256(data).hexdigest()
            except BaseException as error:
                capture_errors.append(field + ': ' + type(error).__name__ + ': ' + str(error))
        record['capture_errors'] = capture_errors
        if record['primary_error'] or record['cleanup_errors'] or capture_errors:
            raise RuntimeError('Owned ' + label + ' failed; command and capture causes retained')
        guard()

    try:
        (contents / 'MacOS').mkdir(parents=True)
        resources.mkdir(parents=True)
        for name, data in tiny_data.items():
            (resources / name).write_bytes(data)
        info = dict(CFBundleIdentifier='dev.toklet.harness.mac', CFBundleExecutable='Toklet',
                    CFBundleName='Toklet', CFBundleDisplayName='Toklet', CFBundlePackageType='APPL',
                    CFBundleVersion='1', CFBundleShortVersionString='0.1', LSMinimumSystemVersion='15.0',
                    NSHighResolutionCapable=True, NSPrincipalClass='NSApplication')
        (contents / 'Info.plist').write_bytes(plistlib.dumps(info))
        for name, pin in pins.items():
            if name.endswith(('.swift', '.py', '.h', '.modulemap')):
                with (ROOT / name).open('rb') as stream:
                    data = stream.read(1024**2 + 1)
                if len(data) > 1024**2 or hashlib.sha256(data).hexdigest() != pin:
                    raise ValueError('Source changed before copy: ' + name)
                target = copies / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(data)
                copy_key = str(target.relative_to(ROOT))
                if package.sha(target) != pin:
                    raise ValueError('Source copy differs: ' + name)
                copy_pins[copy_key] = pin
        command = ['xcrun', 'swiftc', '-sdk', '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk',
                   '-target', 'arm64-apple-macos15.0', '-parse-as-library', '-swift-version', '6', '-strict-concurrency=complete',
                   '-warnings-as-errors', '-O', '-I', str(copies / 'apple/include'), '-module-cache-path', str(destination / 'ModuleCache'),
                   *[str(copies / x.relative_to(ROOT)) for x in sources], str(archive), '-framework', 'Metal', '-framework', 'MetalPerformanceShaders',
                   '-framework', 'Foundation', '-framework', 'Security', '-lobjc', '-liconv', '-lresolv', '-lc++', '-o', str(binary)]
        execute('compile', command)
        execute('sign', ['codesign', '--force', '--sign', '-', '--timestamp=none', str(app)])
        execute('verify', ['codesign', '--verify', '--strict', str(app)])
    except BaseException as error:
        primary = type(error).__name__ + ': ' + str(error)
    finally:
        signal.signal(signal.SIGTERM, previous)
        for name, pin in {**pins, **copy_pins, **diagnostics}.items():
            try:
                if package.sha(ROOT / name) != pin:
                    endpoints.append(name + ': hash differs')
            except BaseException as error:
                endpoints.append(name + ': ' + type(error).__name__ + ': ' + str(error))
    app_pins = {}
    try:
        for path in sorted(app.rglob('*')):
            if path.is_symlink():
                raise ValueError('Unexpected app bundle symlink')
            if path.is_file():
                app_pins[str(path.relative_to(ROOT))] = package.sha(path)
    except BaseException as error:
        endpoints.append('App inventory: ' + type(error).__name__ + ': ' + str(error))
    result = dict(format='toklet-macos-harness-build-v1', passed=primary is None and not endpoints and len(commands) == 3,
                  reviewed_package_sha256=expected, source_pins=pins, source_copy_pins=copy_pins, diagnostic_pins=diagnostics,
                  app_pins=app_pins, commands=commands,
                  primary_error=primary, changed_endpoints=endpoints, iphone_contacted=False, runtime_qualified=False)
    with (destination / 'build.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(passed=result['passed'], app=str(app.relative_to(ROOT)), primary_error=primary)))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])))
