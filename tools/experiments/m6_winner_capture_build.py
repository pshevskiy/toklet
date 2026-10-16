"""Build the isolated M6 winner capture from a verified copy; root engine stays unchanged.

Run only with the native measurement queue idle. The build report is evidence,
not review acceptance. Collection requires its explicitly reviewed SHA.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE_RECEIPT = 'benchmarks/results/2026-10-15-m6-context-worker-qualification-v1.json'
BASE_SHA = '3d34b4516a7794c6599ed7c5f2e566434c5910fd07355c0719ee7dac206cd02c'
ASSETS = 'tools/experiments/m6_winner_capture'
CONTRACT = 'benchmarks/m6-context-v1/manifest.json'
COPIES = {'metal_context_capture.rs': 'examples/metal_context_capture.rs'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relative(path):
    path = Path(path)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Capture path must be repository relative')
    full = ROOT/path
    if full.resolve() != full or full.is_symlink():
        raise ValueError('Capture paths cannot resolve through symlinks')
    return full


def source_files(root):
    paths = [root/'Cargo.toml', root/'Cargo.lock']
    for name in ('src', 'examples', 'tests'):
        paths.extend(p for p in (root/name).rglob('*') if p.is_file())
    if any(p.is_symlink() or p.resolve() != p for p in paths):
        raise ValueError('Source symlink not admitted')
    return sorted(paths)


def inventory(root):
    return {str(p.relative_to(root)): digest(p) for p in source_files(root)}


def tool_paths():
    return [ROOT/CONTRACT, ROOT/BASE_RECEIPT, Path(__file__).resolve(),
            ROOT/'tools/experiments/m6_winner_capture.py', ROOT/'tools/experiments/m6_winner_analysis.py',
            ROOT/'tools/experiments/m6_fp32_context_audit.py', ROOT/'tools/experiments/m6_fp32_values.py',
            ROOT/'tools/experiments/m6_typed_quality.py', ROOT/'tools/experiments/m6_typed_quality_audit.py',
            ROOT/'tools/experiments/metal_engine.py',
            ROOT/'tools/experiments/native_usage.py',
            *(ROOT/ASSETS/name for name in COPIES)]


def baseline():
    with (ROOT/BASE_RECEIPT).open('rb') as stream:
        data = stream.read(2*1024**2+1)
    if len(data) > 2*1024**2 or hashlib.sha256(data).hexdigest() != BASE_SHA:
        raise ValueError('Reviewed original worker receipt differs')
    receipt = json.loads(data)
    if receipt.get('passed') is not True or receipt.get('review', {}).get('accepted') is not True:
        raise ValueError('Original worker review missing')
    expected = {str(p.relative_to(ROOT)) for p in [ROOT/'Cargo.toml', ROOT/'Cargo.lock',
                ROOT/'examples/metal_context.rs', *(ROOT/'src').rglob('*.rs'), *(ROOT/'src').rglob('*.metal')]}
    if set(receipt['source_pins']) != expected:
        raise ValueError('Original runtime source closure differs')
    if any(digest(relative(p)) != sha for p, sha in receipt['source_pins'].items()):
        raise ValueError('Original runtime differs from reviewed FP16 baseline')


def expected_snapshot(original):
    result = dict(original)
    for name, destination in COPIES.items():
        if destination in result:
            raise ValueError('Candidate overwrites original source')
        result[destination] = digest(ROOT/ASSETS/name)
    return result


def verify_record(record):
    baseline()
    if record.get('format') != 'toklet-m6-winner-capture-build-v1' or record.get('passed') is not True:
        raise ValueError('Completed capture build required')
    snapshot = relative(record['snapshot'])
    if snapshot.parent != ROOT/'artifacts':
        raise ValueError('Isolated snapshot must be directly under artifacts')
    original = inventory(ROOT)
    copied = inventory(snapshot)
    expected = expected_snapshot(original)
    if record['root_sources'] != original or record['snapshot_sources'] != expected or copied != expected:
        raise ValueError('Original or isolated source closure changed')
    tools = {str(p.relative_to(ROOT)): digest(p) for p in tool_paths()}
    if record['tools_and_contract'] != tools:
        raise ValueError('Capture tooling/contract changed')
    binary = relative(record['binary']['path'])
    if digest(binary) != record['binary']['sha256']:
        raise ValueError('Capture executable changed')
    pins = {ROOT/p: sha for p, sha in original.items()}
    pins.update({snapshot/p: sha for p, sha in copied.items()})
    pins.update({ROOT/p: sha for p, sha in tools.items()})
    pins[binary] = record['binary']['sha256']
    return pins


def reject_ambient_configuration(snapshot, cargo_home):
    """Inspect Cargo's actual cwd ancestry and reject implicit toolchain changes."""
    for ancestor in (snapshot, *snapshot.parents):
        paths = [ancestor/'.cargo'/name for name in ('config', 'config.toml')]
        paths += [ancestor/name for name in ('rust-toolchain', 'rust-toolchain.toml')]
        if any(p.exists() for p in paths):
            raise ValueError('Ambient Cargo config/toolchain requires separate qualification')
    if any((cargo_home/name).exists() for name in ('config', 'config.toml')):
        raise ValueError('Ambient Cargo home config requires separate qualification')


def build(args):
    baseline()
    snapshot, target, output = (relative(v) for v in (args.snapshot, args.target, args.output))
    if snapshot.parent != ROOT/'artifacts' or target.parent != ROOT/'artifacts' or output.parent != ROOT/'artifacts':
        raise ValueError('Build destinations must be directly under artifacts')
    paths = [snapshot, target, output, output.with_suffix('.log'), output.with_suffix('.binary')]
    if len(set(paths)) != len(paths) or any(p.exists() for p in paths):
        raise ValueError('Build destinations must be distinct and absent')
    original = inventory(ROOT)
    tools = {str(p.relative_to(ROOT)): digest(p) for p in tool_paths()}
    expected = expected_snapshot(original)
    snapshot.mkdir()
    for rel in original:
        destination = snapshot/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, destination)
    for name, rel in COPIES.items():
        shutil.copyfile(ROOT/ASSETS/name, snapshot/rel)
    if inventory(snapshot) != expected:
        raise ValueError('Snapshot preparation differs')
    # The copied source has no .cargo config. Preserve ordinary toolchain lookup,
    # remove inherited compiler overrides/flags; offline locked deps are required.
    env = {key: os.environ[key] for key in ('HOME', 'PATH', 'TMPDIR') if key in os.environ}
    env.update(CARGO_TARGET_DIR=str(target), CARGO_INCREMENTAL='0', RUSTFLAGS='', CARGO_ENCODED_RUSTFLAGS='')
    # Reject ambient Cargo configuration instead of silently treating it as built
    # into the reviewed flags. This includes ancestor and user config locations.
    reject_ambient_configuration(snapshot, Path(env['HOME'])/'.cargo')
    command = ['cargo', 'build', '--offline', '--locked', '--release', '--features', 'metal', '--example', 'metal_context_capture']
    result = dict(format='toklet-m6-winner-capture-build-v1', passed=False,
                  engine_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  snapshot=str(snapshot.relative_to(ROOT)), root_sources=original, snapshot_sources=expected,
                  tools_and_contract=tools, command=command,
                  environment={k: v for k, v in env.items() if k not in ('HOME', 'PATH', 'TMPDIR')},
                  rustc=subprocess.check_output(['rustc', '-Vv'], cwd=snapshot, env=env, text=True),
                  cargo=subprocess.check_output(['cargo', '-V'], cwd=snapshot, env=env, text=True), m6_complete=False)
    with output.open('x') as report, output.with_suffix('.log').open('xb') as log:
        try:
            subprocess.run(command, cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            reject_ambient_configuration(snapshot, Path(env['HOME'])/'.cargo')
            binary = output.with_suffix('.binary')
            shutil.copyfile(target/'release/examples/metal_context_capture', binary)
            binary.chmod(0o755)
            result['binary'] = dict(path=str(binary.relative_to(ROOT)), sha256=digest(binary))
            result['passed'] = True
            verify_record(result)
        except BaseException as error:
            result.update(passed=False, error=f'{type(error).__name__}: {error}')
            raise
        finally:
            report.write(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--output', required=True)
    build(parser.parse_args())
